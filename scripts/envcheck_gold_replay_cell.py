# FaultLines environment check, step 2: replay every task's reference (gold) patch through the
# official verifier (swegemma.harness.verification.verify_task), exactly as the harness scores an
# agent patch, in the same subprocess sandbox. CPU only, no model.
#   gold   : reference patch applied  -> should resolve (exit 0)
#   empty  : no patch                 -> should NOT resolve (tests are meant to fail first)
#   gold300: gold again with a 300 s command timeout, only for gold runs that hit a timeout
import asyncio, concurrent.futures, json, re, time, traceback
from pathlib import Path

from adk_eval_core.tracing import SessionTrace
from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks
from swegemma.sandbox import SubprocessManager

DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
OUT = Path("/kaggle/working/fl_envcheck"); OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "gold_replay.jsonl"
DEADLINE_H = 11.0
CMD_TIMEOUT = 60          # same as the pilot / RQ4 runs (sample submission eval_config.yaml)
LONG_TIMEOUT = 300
T0 = time.time()


def run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def make_cfg():
    return EvalConfig(
        tasks_path=DATA_DIR / "tasks.jsonl",
        snapshots_dir=DATA_DIR / "snapshots",
        results_dir=OUT / "verify_artifacts",
        submission_dir=Path("/kaggle/working/sample_submission"),
        models=ModelRegistry(),
        sandbox="subprocess",
        wheels_dir=DATA_DIR / "wheels",
        graph_dir=str(DATA_DIR / "graphs"),
        embeddings_dir=str(DATA_DIR / "embeddings"),
        timeout_seconds=CMD_TIMEOUT,
    )


CFG = make_cfg()
SANDBOX = {CMD_TIMEOUT: SubprocessManager(timeout_seconds=CMD_TIMEOUT),
           LONG_TIMEOUT: SubprocessManager(timeout_seconds=LONG_TIMEOUT)}


def classify(res):
    out = (res.get("test_output") or "") + "\n" + (res.get("error") or "")
    if res.get("resolved"):
        return "resolved"
    if res.get("test_exit_code") == 124 or "timed out" in out.lower():
        return "timeout"
    m = re.search(r"No module named '([\w\.]+)'", out)
    if m:
        return "missing_module:" + m.group(1)
    if "Failed to apply" in out or "patch does not apply" in out:
        return "patch_apply_failed"
    if "error during collection" in out or "errors during collection" in out:
        return "collection_error"
    if res.get("test_exit_code") == 1:
        return "tests_failed"
    return f"other_exit_{res.get('test_exit_code')}"


def verify(task, patch, timeout):
    snap, base, pch = resolve_task_snapshot_paths(CFG.snapshots_dir, task.instance_id, task.repo)
    t = time.time()
    try:
        r = run_sync(verify_task(SANDBOX[timeout], CFG, task, snap, base_snapshot_path=base, patch_path=pch,
                                 agent_patch=patch, trace=SessionTrace(), start_time=time.perf_counter()))
        res = {"resolved": bool(r.resolved), "test_exit_code": r.test_exit_code,
               "error": r.error, "test_output": (r.test_output or "")[-3000:]}
    except Exception:
        res = {"resolved": False, "test_exit_code": None, "error": traceback.format_exc()[-2000:], "test_output": ""}
    res["seconds"] = round(time.time() - t, 1)
    res["class"] = classify(res)
    return res


tasks = load_tasks(DATA_DIR / "tasks.jsonl")
done = {}
if RESULTS.exists():
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            d = json.loads(line); done[d["instance_id"]] = d
print(f"[FL gold] {len(tasks)} tasks, {len(done)} already done, cmd timeout {CMD_TIMEOUT}s", flush=True)

for i, task in enumerate(tasks, 1):
    if task.instance_id in done:
        continue
    if (time.time() - T0) / 3600 > DEADLINE_H:
        print("[FL gold] deadline reached, stopping", flush=True); break
    rec = {"instance_id": task.instance_id, "repo": task.repo}
    rec["gold"] = verify(task, task.patch, CMD_TIMEOUT)
    rec["empty"] = verify(task, "", CMD_TIMEOUT)
    if rec["gold"]["class"] == "timeout":
        rec["gold300"] = verify(task, task.patch, LONG_TIMEOUT)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    done[task.instance_id] = rec
    print(f"[FL gold] [{i}/{len(tasks)}] {task.instance_id}: gold={rec['gold']['class']} ({rec['gold']['seconds']}s) "
          f"empty={rec['empty']['class']}" + (f" gold300={rec['gold300']['class']}" if 'gold300' in rec else ""), flush=True)

# Summary
from collections import Counter, defaultdict
recs = list(done.values())
summ = {"n": len(recs), "cmd_timeout_s": CMD_TIMEOUT,
        "gold": Counter(r["gold"]["class"] for r in recs),
        "empty": Counter(r["empty"]["class"] for r in recs),
        "gold300": Counter(r["gold300"]["class"] for r in recs if "gold300" in r)}
valid = [r["instance_id"] for r in recs if r["gold"]["class"] == "resolved" and r["empty"]["class"] != "resolved"]
valid300 = [r["instance_id"] for r in recs if (r["gold"]["class"] == "resolved" or r.get("gold300", {}).get("class") == "resolved")
            and r["empty"]["class"] != "resolved"]
by_repo = defaultdict(lambda: {"n": 0, "gold_resolved": 0, "valid": 0})
for r in recs:
    b = by_repo[r["repo"]]; b["n"] += 1
    b["gold_resolved"] += r["gold"]["class"] == "resolved"
    b["valid"] += r["instance_id"] in valid
summ["valid_at_60s"] = len(valid); summ["valid_with_300s"] = len(valid300); summ["by_repo"] = by_repo
summ["valid_task_ids_60s"] = sorted(valid); summ["valid_task_ids_300s"] = sorted(valid300)
(OUT / "gold_replay_summary.json").write_text(json.dumps(summ, indent=1, default=dict))
print(json.dumps({k: v for k, v in summ.items() if not k.startswith("valid_task_ids")}, indent=1, default=dict))
print(f"[FL gold] done in {(time.time()-T0)/3600:.2f} h")
