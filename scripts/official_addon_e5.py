# E5 config arm: identical to official_addon_e3.py except FL_ARM="e5" and max_output_tokens 16384 -> 8192
# in the copied submission YAML (so prompts can use ~24K of the 32K window instead of 16K).
# ===================== FaultLines add-on for the official starter notebook =====================
# Run as the LAST cell of a copy of "Getting Started - Gemma 4 Developer Agent" (ryanholbrook),
# either pasted in, or via: exec(open(glob.glob('/kaggle/input/**/faultlines_official_addon.py', recursive=True)[0]).read())
# Then: Save Version -> Save & Run All (Commit).
#
# It reuses the starter's own objects (EvalConfig, Evaluator, run_sync, the running vLLM server),
# runs a stratified sample of dev tasks with the official harness + gemma-4-31b-it-qat-w4a16-ct,
# then labels every run with FaultLines. Everything lands in /kaggle/working (download from Output).

FL_N_PER_REPO = 5            # tasks per repo (4 repos -> 20 runs)
FL_TASK_IDS = ["fastapi_11194","fastapi_13537","fastapi_13920","fastapi_14258","fastapi_14301","fastapi_14306","fastapi_14372","fastapi_14419","fastapi_14448","fastapi_14458","fastapi_14463","fastapi_14479","fastapi_14487","fastapi_14492","fastapi_14616","fastapi_14786","fastapi_14794","fastapi_14873","fastapi_14986","fastapi_15280","fastapi_15588","fastapi_15589","fastapi_15661","fastapi_15800","fastapi_5077","fastapi_5624","fastapi_9425","fastapi_9555","fastapi_9753","rich_2725","rich_2943","rich_3006","rich_3043","rich_3052","rich_3061","rich_3063","rich_3067","rich_3105","rich_3180","rich_3278","rich_3454","rich_3468","rich_3470","rich_3471","rich_3472","rich_3480","rich_3506","rich_3518","rich_3521","rich_3535","rich_3675","rich_3676","rich_3718","rich_3772","rich_3777","rich_3882","rich_3894","rich_3905","rich_3930","rich_3934","rich_3935","rich_3938","rich_3942","rich_3944","rich_3953","rich_4006","rich_4075","rich_4076","rich_4077","rich_4079"]   # E3: the 70 dev tasks whose reference patch passes and empty patch fails (results/judgeable_tasks.json)
FL_TIME_MINUTES = 20         # per-task agent budget (official default 60; starter demo 5)
FL_MAX_TOOL_CALLS = 100      # sample eval_config.yaml uses 10 for the demo; 100 is the harness default
FL_SEED = 0
FL_DEADLINE_HOURS = 11.0     # do not START a new task after this much session time (Kaggle kills at 12 h)
FL_ARM = "e5"          # "control" = sample submission as-is; "treatment" = prompt addendum below
FL_OUT = f"/kaggle/working/fl_{FL_ARM}"
FL_LABEL = f"gemma-4-31b-it-qat-w4a16/{FL_ARM}"
# RQ4: configuration-only fixes for the failures seen in the pilot (appended to the submission's system prompt).
FL_PROMPT_ADDENDUM = """

## Tool-Use Rules (read carefully)
1. **Multi-line edits:** in `edit_file` and `write_file`, put real line breaks inside `old_string`, `new_string` and `content`.
2. **After "old_string not found":** call `read_file` on those exact lines, copy them verbatim (same indentation), and retry once with the copied text. Never resend an edit that already failed.
3. **Never repeat an identical tool call.** If two calls gave the same result, change your approach.
4. **`grep` exit code 1 means "no matches"**, not an error. Do not filter with `grep -v "..."` (a dot matches any character).
5. **Fix the library source.** A reproduction script alone is not a fix. Delete any repro or debug scripts you created before calling `submit_patch`, because they end up in the patch.
6. **Budget:** you have at most 50 turns and 20 minutes. Keep your thinking short. Once the fix is in place, or by turn 35 at the latest, call `submit_patch`.
"""
FL_RELABEL_GLOB = "/kaggle/input/**/fl_official"   # optional: earlier run attached as input, re-labeled with this version
FL_PKG = "__PAYLOAD__"   # FaultLines package (zip, base64) - do not edit

from pathlib import Path
import base64, concurrent.futures, copy, dataclasses, glob, inspect, io, json, os, random, re, shutil, subprocess, sys, time, traceback, zipfile
FL_T0 = time.time()
try:
    FL_SESSION_T0 = os.stat("/proc/1").st_ctime            # container start ~ Kaggle session start
except OSError:
    FL_SESSION_T0 = FL_T0
os.makedirs(FL_OUT, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(FL_PKG))).extractall("/kaggle/working/faultlines_pkg")
sys.path.insert(0, "/kaggle/working/faultlines_pkg/src")
for _m in [m for m in sys.modules if m == "faultlines" or m.startswith("faultlines.")]:
    del sys.modules[_m]
from faultlines.atlas import label_run, read_runs, write_runs
from faultlines.atlas.adapters import load_swegemma_results, trace_shape
from faultlines.atlas.report import replay as fl_replay, summarize as fl_summarize
from faultlines.toolguard import ToolRegistry


def fl_log(*a):
    print(f"[FL {time.strftime('%H:%M:%S')}]", *a, flush=True)


def fl_fields(c):
    if dataclasses.is_dataclass(c):
        return {f.name: getattr(c, f.name) for f in dataclasses.fields(c)}
    if hasattr(type(c), "model_fields"):
        return {k: getattr(c, k) for k in type(c).model_fields}
    return {k: v for k, v in vars(c).items() if not k.startswith("_")}


def fl_replace(c, upd):
    if not upd:
        return c
    if dataclasses.is_dataclass(c):
        return dataclasses.replace(c, **upd)
    if hasattr(c, "model_copy"):
        return c.model_copy(update=upd)
    c2 = copy.copy(c)
    for k, v in upd.items():
        setattr(c2, k, v)
    return c2


_NOT_BUDGET = re.compile(r"command|cmd|test|pytest|verif|request|http|connect|server|startup|tool_timeout|read|install|setup|sandbox_timeout", re.I)


def fl_budget_updates(c, depth=0):
    """Pick the per-task agent budget / tool-call cap / output dir fields by name (one nested level too)."""
    upd = {}
    for k, v in fl_fields(c).items():
        if depth < 2 and (dataclasses.is_dataclass(v) or hasattr(type(v), "model_fields")) and not isinstance(v, type):
            sub = fl_budget_updates(v, depth + 1)
            if sub:
                upd[k] = fl_replace(v, sub)
                fl_log(f"  nested {k}:", sub)
        elif isinstance(v, (str, os.PathLike)) or v is None:
            if depth == 0 and re.search(r"(output|results?|out|save)_?(dir|path|root)$", k, re.I):
                upd[k] = FL_OUT if isinstance(v, str) or v is None else type(v)(FL_OUT)
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        elif re.search(r"tool_calls?", k, re.I) and re.search(r"max|limit|budget", k, re.I):
            if FL_MAX_TOOL_CALLS is not None:
                upd[k] = FL_MAX_TOOL_CALLS
        elif re.search(r"time|minute|timeout|budget|deadline|duration", k, re.I) and not _NOT_BUDGET.search(k):
            secs = "sec" in k.lower() or ("minute" not in k.lower() and v >= 120)
            val = FL_TIME_MINUTES * 60 if secs else FL_TIME_MINUTES
            upd[k] = int(val) if isinstance(v, int) else float(val)
    return upd


def fl_jsonable(x, depth=0):
    if depth > 14:
        return str(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): fl_jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [fl_jsonable(v, depth + 1) for v in x]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return fl_jsonable({f.name: getattr(x, f.name) for f in dataclasses.fields(x)}, depth + 1)
    if hasattr(x, "model_dump"):
        try:
            return fl_jsonable(x.model_dump(), depth + 1)
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        return fl_jsonable({k: v for k, v in vars(x).items() if not k.startswith("_")}, depth + 1)
    return str(x)


def fl_task_id(t):
    if isinstance(t, dict):
        return t.get("instance_id") or t.get("task_id") or t.get("id")
    return getattr(t, "instance_id", None) or getattr(t, "task_id", None) or getattr(t, "id", None)


def fl_find_proto_task(ns):
    """A task object the starter already built (to copy its type)."""
    for k, v in list(ns.items()):
        if k.startswith("_") or k.startswith("fl_") or k.startswith("FL_"):
            continue
        cands = v if isinstance(v, (list, tuple)) else [v]
        for c in cands[:3]:
            if not isinstance(c, (str, bytes, type)) and fl_task_id(c) and (isinstance(c, dict) and "problem_statement" in c
                                                                            or hasattr(c, "problem_statement")):
                return c
    return None


def fl_to_task(d, proto):
    if proto is None or isinstance(proto, dict):
        return d
    T = type(proto)
    for meth in ("model_validate", "from_dict", "parse_obj"):
        if hasattr(T, meth):
            try:
                return getattr(T, meth)(d)
            except Exception:
                pass
    try:
        return T(**d)
    except TypeError:
        names = set(fl_fields(proto))
        return T(**{k: v for k, v in d.items() if k in names})


def fl_session_hours():
    return (time.time() - FL_SESSION_T0) / 3600


# ---------------------------------------------------------------- 1. the starter's objects
from swegemma.evaluate import Evaluator
try:
    from swegemma.config import EvalConfig
except ImportError:
    from swegemma.evaluate import EvalConfig
_ns = globals()
fl_ev = next((v for k, v in _ns.items() if isinstance(v, Evaluator) and not k.startswith("fl_")), None)
fl_cfg0 = next((v for k, v in _ns.items() if isinstance(v, EvalConfig) and not k.startswith("fl_")), None)
if fl_cfg0 is None and fl_ev is not None:
    fl_cfg0 = next((v for v in vars(fl_ev).values() if isinstance(v, EvalConfig)), None)
assert fl_cfg0 is not None, "Could not find the starter's EvalConfig. Run all starter cells first (Save & Run All)."
fl_log("starter EvalConfig fields:")
for _k, _v in fl_fields(fl_cfg0).items():
    print(f"    {_k} = {str(_v)[:120]!s}")
_known = fl_fields(fl_cfg0)
if "max_time_minutes" in _known:          # the starter's EvalConfig (swegemma.config)
    fl_upd = {"max_time_minutes": float(FL_TIME_MINUTES)}
    if FL_MAX_TOOL_CALLS is not None and "max_tool_calls" in _known:
        fl_upd["max_tool_calls"] = int(FL_MAX_TOOL_CALLS)
    if "results_dir" in _known:
        _rd = _known["results_dir"]
        fl_upd["results_dir"] = FL_OUT if isinstance(_rd, str) or _rd is None else type(_rd)(FL_OUT)
else:                                     # unknown layout: guess by field name
    fl_upd = fl_budget_updates(fl_cfg0)
if FL_ARM != "control" and FL_PROMPT_ADDENDUM and "submission_dir" in _known:
    _src = Path(str(_known["submission_dir"]))
    _dst = Path(f"/kaggle/working/submission_{FL_ARM}")
    shutil.rmtree(_dst, ignore_errors=True)
    shutil.copytree(_src, _dst)
    _agent_yaml = (_dst / "agent.yaml").read_text() if (_dst / "agent.yaml").exists() else ""
    _m = re.search(r"instruction:\s*!include\s+(\S+)", _agent_yaml)
    _prompt = _dst / (_m.group(1) if _m else "prompts/system.md")
    _prompt.write_text(_prompt.read_text() + FL_PROMPT_ADDENDUM)
    fl_upd["submission_dir"] = _dst if not isinstance(_known["submission_dir"], str) else str(_dst)
    fl_log(f"arm={FL_ARM}: prompt addendum appended to {_prompt} ({len(FL_PROMPT_ADDENDUM)} chars)")
    # E5: shrink the per-call output reserve so the prompt can use more of the 32K window.
    FL_MAX_OUTPUT_TOKENS = 8192
    _keys = r"(max_output_tokens)"
    _hits = []
    for _y in sorted(list(_dst.rglob("*.yaml")) + list(_dst.rglob("*.yml"))):
        _t = _y.read_text()
        _new, _n = re.subn(rf"(?m)^(\s*){_keys}(\s*:\s*)(\d+)", lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{FL_MAX_OUTPUT_TOKENS}", _t)
        if _n:
            _y.write_text(_new)
            _hits.append(f"{_y.relative_to(_dst)} x{_n}")
    if not _hits:
        for _y in sorted(_dst.rglob("*.y*ml")):
            print(f"----- {_y.relative_to(_dst)}\n{_y.read_text()[:3000]}")
        raise RuntimeError("E5: no output-token limit found in the submission YAML (printed above)")
    fl_log(f"arm={FL_ARM}: max output tokens set to {FL_MAX_OUTPUT_TOKENS} in {_hits}")
    for _y in sorted(_dst.rglob("*.y*ml")):
        print(f"----- {_y.relative_to(_dst)}\n{_y.read_text()[:1500]}")
fl_log("overriding:", fl_upd or "nothing matched (budget stays as in the starter; see field list above)")
fl_cfg = fl_replace(fl_cfg0, fl_upd)

try:
    _init = [p for p in inspect.signature(Evaluator.__init__).parameters.values() if p.name != "self"]
    _req = [p for p in _init if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if len(_req) <= 1:
        fl_evaluator = Evaluator(fl_cfg) if _req else Evaluator(config=fl_cfg)
    else:
        _kw = {}
        for p in _req:
            if isinstance(getattr(fl_ev, p.name, None), EvalConfig) or p.name in ("config", "cfg", "eval_config"):
                _kw[p.name] = fl_cfg
            else:
                _kw[p.name] = getattr(fl_ev, p.name, None) or getattr(fl_ev, "_" + p.name)
        fl_evaluator = Evaluator(**_kw)
except Exception as e:
    fl_log("could not build a new Evaluator (", repr(e), ") -> patching the starter's evaluator in place")
    assert fl_ev is not None, "no Evaluator instance found in the starter"
    fl_evaluator = fl_ev
    for k, v in list(vars(fl_ev).items()):
        if v is fl_cfg0:
            setattr(fl_ev, k, fl_cfg)

fl_run_sync = _ns.get("run_sync")
if not callable(fl_run_sync):
    def fl_run_sync(fn, *a, **kw):
        import asyncio
        async def _go():
            r = fn(*a, **kw) if callable(fn) else fn
            return await r if inspect.isawaitable(r) else r
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            return ex.submit(asyncio.run, _go()).result()

# vLLM server: restart it if the starter stopped it
for _k, _v in list(_ns.items()):
    if "vllm" in type(_v).__name__.lower() and "server" in type(_v).__name__.lower():
        for _probe in ("is_running", "is_alive", "healthy", "is_healthy", "running"):
            if hasattr(_v, _probe):
                _ok = getattr(_v, _probe)
                _ok = _ok() if callable(_ok) else _ok
                fl_log(f"vLLM server {_k}.{_probe} ->", _ok)
                if _ok is False and hasattr(_v, "start"):
                    fl_log("restarting vLLM server ..."); _v.start()
                break

# ---------------------------------------------------------------- 2. task sample
_tasks_path = next(iter(sorted(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True), key=len)), None)
assert _tasks_path, "tasks.jsonl not found under /kaggle/input (is the competition attached?)"
fl_all = [json.loads(l) for l in open(_tasks_path) if l.strip()]
if FL_TASK_IDS:
    fl_pick = [t for t in fl_all if t["instance_id"] in set(FL_TASK_IDS)]
else:
    _by_repo = {}
    for t in fl_all:
        _by_repo.setdefault(t.get("repo") or t["instance_id"].rsplit("_", 1)[0], []).append(t)
    _rng = random.Random(FL_SEED)
    _groups = [sorted(_rng.sample(v, min(FL_N_PER_REPO, len(v))), key=lambda t: t["instance_id"])
               for _, v in sorted(_by_repo.items())]
    fl_pick = [g[i] for i in range(max(map(len, _groups))) for g in _groups if i < len(g)]   # round-robin by repo
fl_log(f"{len(fl_pick)} tasks from {len(fl_all)} ({_tasks_path}):", [t["instance_id"] for t in fl_pick])
_proto = fl_find_proto_task(_ns)
fl_log("task type:", type(_proto).__name__ if _proto is not None else "dict (no starter task object found)")
_sig = inspect.signature(fl_evaluator.evaluate_task).parameters

# ---------------------------------------------------------------- 3. run
_done = set()
_res_file = os.path.join(FL_OUT, "fl_task_results.jsonl")
if os.path.exists(_res_file):
    _done = {json.loads(l)["instance_id"] for l in open(_res_file) if l.strip()}
try:
    for i, t in enumerate(fl_pick):
        tid = t["instance_id"]
        if tid in _done:
            continue
        if fl_session_hours() > FL_DEADLINE_HOURS:
            fl_log(f"deadline reached ({fl_session_hours():.1f} h) - stopping before {tid}")
            break
        kw = {}
        _task = fl_to_task(t, _proto)
        for name in _sig:
            if name in ("task", "instance", "task_instance"):
                kw[name] = _task
            elif name in ("task_index", "index", "idx"):
                kw[name] = i
            elif name in ("total_tasks", "total", "n_tasks"):
                kw[name] = len(fl_pick)
        if not any(n in kw for n in ("task", "instance", "task_instance")):
            args, kw = (_task,), kw
        else:
            args = ()
        fl_log(f"[{i + 1}/{len(fl_pick)}] {tid} (session {fl_session_hours():.2f} h)")
        t0 = time.time()
        err = None
        try:
            if args:
                out = fl_evaluator.evaluate_task(*args, **kw)
                out = fl_run_sync(out) if inspect.isawaitable(out) else out
            else:   # same call shape as the starter: run_sync(evaluator.evaluate_task, task=..., ...)
                out = fl_run_sync(fl_evaluator.evaluate_task, **kw)
                out = fl_run_sync(out) if inspect.isawaitable(out) else out
        except Exception as e:
            out, err = None, "".join(traceback.format_exception_only(type(e), e)).strip()
            traceback.print_exc()
        rec = {"instance_id": tid, "wall_s": round(time.time() - t0, 1), "error": err, "result": fl_jsonable(out)}
        if isinstance(rec["result"], dict):
            rec["resolved"] = rec["result"].get("resolved")
        with open(_res_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        fl_log(f"    -> resolved={rec.get('resolved')} wall={rec['wall_s']:.0f}s err={err}")
finally:
    # ------------------------------------------------------------ 4. analysis (always runs)
    _cands = [FL_OUT] + [os.path.dirname(p) for p in glob.glob("/kaggle/working/**/traces", recursive=True)] \
        + [os.path.dirname(p) for p in glob.glob("/tmp/**/traces", recursive=True)]
    fl_results = next((d for d in _cands if os.path.isdir(os.path.join(d, "traces"))
                       and os.path.getmtime(os.path.join(d, "traces")) >= FL_T0 - 60), FL_OUT)
    if fl_results != FL_OUT:
        fl_log("harness wrote results to", fl_results, "- copying into", FL_OUT)
        shutil.copytree(fl_results, FL_OUT, dirs_exist_ok=True)
    _traces = sorted(glob.glob(os.path.join(FL_OUT, "traces", "*.json")))
    fl_log(f"{len(_traces)} traces in {FL_OUT}/traces; files:", sorted(os.listdir(FL_OUT)))
    if os.path.exists(os.path.join(FL_OUT, "summary.json")):
        print(open(os.path.join(FL_OUT, "summary.json")).read()[:2000])
    for _f in ("task_results.jsonl", "fl_task_results.jsonl"):
        _p = os.path.join(FL_OUT, _f)
        if os.path.exists(_p) and open(_p).read().strip():
            fl_log(f"shape of first line of {_f}:"); print(trace_shape(json.loads(open(_p).readline())))
    if _traces:
        _tr = json.load(open(_traces[0]))
        fl_log("shape of", os.path.basename(_traces[0])); print(trace_shape(_tr, max_depth=5))
        _raw = sum(open(p).read().count("<|tool_call>") for p in _traces)
        _ids = sum(open(p).read().count("completion_token_ids") for p in _traces)
        fl_log(f"raw '<|tool_call>' strings in traces: {_raw} | steps with completion_token_ids: {_ids}")
    _model_dir = next(iter(sorted((os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
                                   if "31b" in p.lower()), key=len)), None)
    _decode = None
    if _model_dir:
        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(_model_dir)
            _decode = lambda ids: _tok.decode(ids, skip_special_tokens=False)
        except Exception as e:
            fl_log("tokenizer not loaded:", e)
    try:
        fl_runs = [label_run(r, ToolRegistry.default_harness())
                   for r in load_swegemma_results(FL_OUT, _tasks_path, FL_LABEL, context_limit=32768,
                                                  decode=_decode, harness="swegemma-official")]
        _wall = {json.loads(l)["instance_id"]: json.loads(l) for l in open(_res_file)} if os.path.exists(_res_file) else {}
        for r in fl_runs:
            w = _wall.get(r.task_id, {})
            r.timing["wall_s"] = w.get("wall_s", 0.0)
            if r.resolved is None and w.get("resolved") is not None:
                r.resolved = w["resolved"]
            if w.get("error"):
                r.notes.append("evaluate_task_error: " + w["error"][:300])
        write_runs(fl_runs, os.path.join(FL_OUT, "labeled.jsonl"))
        print(fl_summarize(fl_runs)); print(); print(fl_replay(fl_runs))
        print(f"\n{'task':26s} {'resolved':>8s} {'label':>22s} {'turns':>5s} {'wall_min':>8s}  evidence")
        for r in fl_runs:
            print(f"{r.task_id:26s} {str(r.resolved):>8s} {str(r.primary_failure):>22s} {len(r.steps):5d} "
                  f"{r.timing.get('wall_s', 0) / 60:8.1f}  {'; '.join(r.evidence)[:90]}")
        _ws = [r.timing.get("wall_s", 0) for r in fl_runs if r.timing.get("wall_s")]
        if _ws:
            _m = sum(_ws) / len(_ws)
            fl_log(f"mean wall per task {_m / 60:.1f} min -> ~{int(11 * 3600 / _m)} tasks per 11 h session")
    except Exception:
        traceback.print_exc()
    for _d in sorted(glob.glob(FL_RELABEL_GLOB, recursive=True)):      # re-label an earlier run with this version
        if os.path.abspath(_d) == os.path.abspath(FL_OUT) or not os.path.isdir(os.path.join(_d, "traces")):
            continue
        try:
            _prev = [label_run(r, ToolRegistry.default_harness())
                     for r in load_swegemma_results(_d, _tasks_path, "gemma-4-31b-it-qat-w4a16/official",
                                                    context_limit=32768, harness="swegemma-official")]
            write_runs(_prev, os.path.join(FL_OUT, "relabeled_" + os.path.basename(_d) + ".jsonl"))
            fl_log("re-labeled", _d); print(fl_summarize(_prev))
            _a = {r.task_id: r.primary_failure == "RESOLVED" for r in _prev}
            _b = {r.task_id: r.primary_failure == "RESOLVED" for r in fl_runs}
            _shared = sorted(set(_a) & set(_b))
            fl_log(f"paired resolved on {len(_shared)} shared tasks: before {sum(_a[t] for t in _shared)} "
                   f"after {sum(_b[t] for t in _shared)} | only before {[t for t in _shared if _a[t] and not _b[t]]} "
                   f"| only after {[t for t in _shared if _b[t] and not _a[t]]}")
        except Exception:
            traceback.print_exc()
    shutil.make_archive(f"/kaggle/working/fl_{FL_ARM}_results", "zip", FL_OUT)
    fl_log(f"wrote /kaggle/working/fl_{FL_ARM}_results.zip - download it from the notebook's Output tab")
