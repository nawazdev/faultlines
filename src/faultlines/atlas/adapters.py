"""Convert harness output into FaultLines `Run` records.

`swegemma eval` writes (per the third-party copy of the harness spec):
    results/summary.json          aggregate resolution rate
    results/task_results.jsonl    per-task metrics
    results/patches/              extracted diffs
    results/test_outputs/         pytest logs
    results/traces/               ATIF-compatible SessionTrace JSON

Trace fields follow the published ATIF v1.7 spec (harbor-framework RFC 0001).
task_results.jsonl keys are still tolerant guesses: print one real line with
`trace_shape` and tighten the `_pick(...)` lists if anything comes back empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

from .autolabel import patch_files
from .trace import Run, Step


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):  # list of content parts
        return "\n".join(_as_text(p.get("text", p) if isinstance(p, dict) else p) for p in x)
    if isinstance(x, dict):
        return _as_text(_pick(x, "text", "content", "output", default=json.dumps(x)))
    return str(x)


def _args(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, dict) else {"_raw": x}
        except json.JSONDecodeError:
            return {"_raw": x}
    return {}


NUDGE_MARKERS = (   # substrings of the three exact harness nudges (HARNESS_README.md)
    ("before the tool call finished closing", "nudge_unclosed_call"),
    ("reached the token limit while thinking", "nudge_max_tokens"),
    ("please continue your work using the available tools", "nudge_continue"),
)
COMPACTION_MARKERS = ("context_management", "compaction", "summary of the conversation", "conversation summary")


def _results_by_id(obs: Any) -> tuple[dict[str, str], list[str]]:
    """ATIF observation -> ({source_call_id: text}, [texts in order])."""
    if obs is None:
        return {}, []
    results = _pick(obs, "results", default=obs) if isinstance(obs, dict) else obs
    if not isinstance(results, list):
        results = [results]
    by_id, ordered = {}, []
    for r in results:
        text = _as_text(_pick(r, "content", "output", "text") if isinstance(r, dict) else r)
        ordered.append(text)
        cid = _pick(r, "source_call_id", "tool_call_id") if isinstance(r, dict) else None
        if cid:
            by_id[str(cid)] = text
    return by_id, ordered


def atif_to_steps(trace: dict[str, Any], decode: Callable[[list[int]], str] | None = None) -> list[Step]:
    """Map an ATIF (v1.x) trajectory to agent turns.

    ATIF v1.7: steps[] with source in {system,user,agent}; agent steps carry message,
    reasoning_content, tool_calls[{tool_call_id,function_name,arguments}],
    observation.results[{source_call_id,content}], metrics{prompt_tokens,completion_tokens,
    completion_token_ids}. If `decode` is given and token ids are logged, the raw model text
    (special tokens included) is recovered from them, so malformed calls stay visible.
    """
    raw_steps = _pick(trace, "steps", "events", "turns", default=[])
    steps: list[Step] = []
    pending: Step | None = None
    for st in raw_steps:
        source = str(_pick(st, "source", "role", "author", default="")).lower()
        extra = _pick(st, "extra", default={}) or {}
        if source in ("agent", "assistant", "model"):
            text = _as_text(_pick(st, "raw_output", "message", "content", "text"))
            reasoning = _as_text(_pick(st, "reasoning_content", "thinking"))
            calls = _pick(st, "tool_calls", "function_calls", default=[]) or []
            metrics = _pick(st, "metrics", "usage", default={}) or {}
            raw = (reasoning + "\n" + text).strip() if reasoning else text
            ids = _pick(metrics, "completion_token_ids")
            if decode and ids:
                try:
                    raw = decode(list(ids))
                except Exception:  # keep the reconstructed text
                    pass
            by_id, ordered = _results_by_id(_pick(st, "observation", "tool_results"))
            parsed = []
            for i, c in enumerate(calls):
                cid = str(_pick(c, "tool_call_id", "id", default=""))
                parsed.append({"name": _pick(c, "function_name", "name", "tool_name"),
                               "args": _args(_pick(c, "arguments", "args", "parameters")),
                               "result": by_id.get(cid, ordered[i] if i < len(ordered) else "")})
            first = parsed[0] if parsed else None
            step = Step(index=len(steps), raw_output=raw,
                        tool_name=first["name"] if first else None,
                        tool_args=first["args"] if first else None,
                        tool_result=first["result"] if first else "",
                        finish_reason=_pick(st, "finish_reason", "stop_reason", default=_pick(extra, "finish_reason")),
                        extra_calls=parsed[1:])
            step.prompt_tokens = _pick(metrics, "prompt_tokens", "input_tokens")
            step.completion_tokens = _pick(metrics, "completion_tokens", "output_tokens")
            if any(m in json.dumps(extra).lower() for m in COMPACTION_MARKERS[:2]):
                step.events.append("compaction")
            steps.append(step)
            pending = step
        elif source in ("tool", "function", "observation") and pending is not None:
            pending.tool_result = _as_text(_pick(st, "content", "message", "output", "text"))
        elif source in ("system", "user", "harness"):
            text = _as_text(_pick(st, "message", "content", "text")).lower()
            if pending is None:
                continue
            for marker, name in NUDGE_MARKERS:
                if marker in text:
                    pending.events += ["nudge", name]
            blob = text + json.dumps(extra).lower()
            if any(m in blob for m in COMPACTION_MARKERS) and "compaction" not in pending.events:
                pending.events.append("compaction")
    return steps


def trace_shape(obj: Any, depth: int = 0, max_depth: int = 4) -> str:
    """Compact key/type tree of a JSON object (for calibrating this adapter on a real trace)."""
    pad = "  " * depth
    if depth >= max_depth:
        return f"{pad}...\n"
    out = ""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kind = type(v).__name__ + (f"[{len(v)}]" if isinstance(v, (list, dict, str)) else "")
            out += f"{pad}{k}: {kind}\n"
            if isinstance(v, (dict, list)):
                out += trace_shape(v, depth + 1, max_depth)
    elif isinstance(obj, list) and obj:
        kinds = sorted({str(_pick(x, "source", default="")) for x in obj if isinstance(x, dict)} - {""})
        richest = max(obj, key=lambda x: len(x) if isinstance(x, dict) else 0)
        out += f"{pad}[richest of {len(obj)}]" + (f" (sources: {kinds})" if kinds else "") + "\n"
        out += trace_shape(richest, depth + 1, max_depth)
    return out


def load_swegemma_results(results_dir: str | Path, tasks_path: str | Path | None = None,
                          model: str = "", config_hash: str = "", context_limit: int | None = 32768,
                          decode: Callable[[list[int]], str] | None = None, harness: str = "swegemma") -> Iterator[Run]:
    """Join task_results.jsonl + traces/ + patches/ + test_outputs/ (+ tasks.jsonl for gold files)."""
    rd = Path(results_dir)
    gold: dict[str, list[str]] = {}
    if tasks_path:
        for line in Path(tasks_path).read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                tid = str(_pick(t, "instance_id", "task_id", "id"))
                gold[tid] = patch_files(_pick(t, "patch", "gold_patch", default=""))

    lines: list[str] = []
    # our wrapper records first, official ones after, so official fields win (later lines override)
    for name in ("fl_task_results.jsonl", "task_results.jsonl"):
        if (rd / name).exists():
            lines += (rd / name).read_text().splitlines()
    if not lines and (rd / "traces").exists():   # no per-task results: fall back to one run per trace
        lines = [json.dumps({"instance_id": f.stem.removeprefix("trace_")}) for f in sorted((rd / "traces").glob("*.json"))]
    latest: dict[str, dict[str, Any]] = {}   # last line per task wins (re-runs append)
    for line in lines:
        if not line.strip():
            continue
        tr = json.loads(line)
        if isinstance(tr.get("result"), dict):   # {"instance_id":..., "result": {...}} wrapper
            tr = {**tr["result"], **{k: v for k, v in tr.items() if k != "result"}}
        tid = str(_pick(tr, "instance_id", "task_id", "id"))
        latest[tid] = {**latest.get(tid, {}), **{k: v for k, v in tr.items() if v is not None}}
    for tid, tr in latest.items():
        run = Run(
            task_id=tid,
            model=model or str(_pick(tr, "model", default="")),
            config_hash=config_hash,
            resolved=_pick(tr, "resolved", "success"),
            exit_reason=str(_pick(tr, "exit_reason", "termination_reason", "stop_reason",
                                  default="error" if tr.get("error") else "")),
            patch_applied=_pick(tr, "patch_applied", "applied"),
            gold_files=gold.get(tid, []),
            context_limit=context_limit,
            harness=harness,
        )
        run.agent_patch = str(_pick(tr, "agent_patch", "patch", default="") or "")
        run.test_output = str(_pick(tr, "test_output", default="") or "")[-20000:]
        ec = _pick(tr, "test_exit_code", "exit_code")
        run.test_exit_code = int(ec) if isinstance(ec, (int, float)) else None
        for cand in (rd / "patches" / f"{tid}.diff", rd / "patches" / f"{tid}.patch"):
            if cand.exists():
                run.agent_patch = cand.read_text()
        for cand in (rd / "test_outputs" / f"{tid}.log", rd / "test_outputs" / f"{tid}.txt"):
            if cand.exists():
                run.test_output = cand.read_text()[-20000:]
        tfile = None
        if (rd / "traces").exists():
            exact = [rd / "traces" / f"trace_{tid}.json", rd / "traces" / f"{tid}.json"]
            tfile = next((f for f in exact if f.exists()), None) or next(
                (f for f in sorted((rd / "traces").glob(f"*{tid}*.json"))
                 if not f.stem.split(tid, 1)[1][:1].isdigit()), None)
        if tfile:
            run.steps = atif_to_steps(json.loads(tfile.read_text()), decode)
        else:   # trace embedded in the result record
            inline = next((v for k, v in tr.items() if isinstance(v, dict) and isinstance(v.get("steps"), list)
                           and k in ("trace", "session_trace", "trajectory", "atif")), None)
            if inline:
                run.steps = atif_to_steps(inline, decode)
        if not run.exit_reason or run.exit_reason in ("SUCCESS", "unknown"):
            run.exit_reason = "submitted" if any(s.tool_name == "submit_patch" for s in run.steps) else "not_submitted"
        yield run
