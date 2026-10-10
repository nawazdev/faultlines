"""Heuristic labeler: fills step labels and one primary label per run.

Thresholds are deliberately simple and named so the paper can state them and
the kappa check (vs. hand labels) can tune them. Tune on real traces in week 3.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..toolguard import ToolGuard, ToolRegistry, find_loops
from ..toolguard.gemma_syntax import CALL_OPEN
from ..toolguard.repair import ESCAPABLE_ARGS, ToolCall, is_over_escaped
from .taxonomy import Primary
from .trace import Run

CONTEXT_ERROR = re.compile(r"ContextWindowExceeded|maximum context length", re.I)

GRAPH_TOOLS = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}

RUNTIME_PATTERNS = [
    ("EDIT_NO_MATCH", re.compile(r"(no match|not found in file|did not match|could not find .*(string|text)|"
                                 r"old_str .*not found|0 occurrences)", re.I)),
    ("FILE_NOT_FOUND", re.compile(r"(no such file|filenotfounderror|does not exist|file not found)", re.I)),
    ("TIMEOUT", re.compile(r"(timed out|timeout expired|command timeout)", re.I)),
    ("PERMISSION", re.compile(r"(permission denied|path traversal|outside (the )?workspace)", re.I)),
    ("TOOL_ERROR", re.compile(r"^\s*(error|exception|traceback)\b", re.I)),
]

BROKEN_TEST_PATTERNS = re.compile(
    r"(error collecting|errors? during collection|SyntaxError|IndentationError|ImportError while importing|"
    r"ModuleNotFoundError|patch does not apply|failed to apply)", re.I)


ENV_MISSING_MODULE = re.compile(r"No module named '([\w.]+)'")
VERIFY_TIMEOUT_EXIT = 124


def env_unverifiable(run: Run) -> str | None:
    """Why the verifier could not judge this patch, or None. (Observed in the Kaggle subprocess sandbox.)"""
    if run.test_exit_code == VERIFY_TIMEOUT_EXIT:
        return "test run timed out (exit 124)"
    patched = " ".join(patch_files(run.agent_patch))
    for m in ENV_MISSING_MODULE.finditer(run.test_output or ""):
        mod = m.group(1).split(".")[0]
        if mod not in patched:
            return f"missing test dependency '{mod}'"
    return None


@dataclass
class Thresholds:
    tail_window: int = 5          # INTERFACE: look at the last N turns
    tail_bad: int = 3             # INTERFACE: ... at least this many bad turns
    high_invalid_rate: float = 0.2
    context_ratio: float = 0.9
    loop_tail_fraction: float = 1 / 3
    runtime_errors_flag: int = 3
    escaped_edits: int = 3        # INTERFACE (argument encoding): this many over-escaped edit calls


def patch_files(patch: str) -> list[str]:
    files = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)", patch or "", re.M):
        files.append(m.group(2))
    if not files:
        files = [m.group(1) for m in re.finditer(r"^\+\+\+ b/(\S+)", patch or "", re.M)]
    return sorted(set(files))


# swegemma tools reply with JSON: {"status": "error", "error_type": ..., "error_message": ...}
ERROR_TYPE_MAP = {
    "FileEditError": "EDIT_FAILED",          # old_string not found / not unique / empty file
    "TimeoutExceeded": "TIMEOUT",
    "ValidationError": "BAD_PATH",           # e.g. '..' traversal
    "CommandError": "COMMAND_FAILED",        # non-zero exit; often expected (failing repro test)
    "BudgetExceeded": "BUDGET",
}
NOT_A_RUNTIME_ERROR = {"COMMAND_FAILED"}   # a failing test/grep is normal agent behaviour


def classify_runtime(tool_result: str) -> str | None:
    text = (tool_result or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            if obj.get("status") != "error":
                return None
            et = str(obj.get("error_type", "TOOL_ERROR"))
            msg = str(obj.get("error_message", "")).lower()
            if et == "FileEditError" and ("not found" in msg or "no match" in msg):
                return "EDIT_NO_MATCH"
            if "no such file" in msg or "does not exist" in msg or "not exist" in msg:
                return "FILE_NOT_FOUND"
            return ERROR_TYPE_MAP.get(et, f"TOOL_{et}")
    head = text[:600]
    for code, pat in RUNTIME_PATTERNS:
        if pat.search(head):
            return code
    return None


def label_steps(run: Run, registry: ToolRegistry) -> None:
    guard = ToolGuard(registry, detect_loops=False)
    for s in run.steps:
        text = s.raw_output or ""
        if CALL_OPEN in text:
            res = guard.check(text)
        elif s.tool_name:
            # Trace kept only the parsed call: judge it at schema level.
            res = guard.check_structured(s.tool_name, s.tool_args or {})
        else:
            res = guard.check("")  # no call at all
        s.call_status = res.status
        s.call_issues = sorted({i.code.value for i in res.issues})
        s.runtime_error = classify_runtime(s.tool_result) if s.tool_name else None
        if s.tool_name and any(is_over_escaped((s.tool_args or {}).get(k)) for k in ESCAPABLE_ARGS):
            s.call_issues = sorted(set(s.call_issues) | {"ESCAPED_STRING"})
    # A closing summary after submit_patch is not a missing call.
    submitted_at = next((s.index for s in run.steps if s.tool_name == "submit_patch"), None)
    for s in run.steps:
        if submitted_at is not None and s.index > submitted_at and s.call_status == "no_call":
            s.call_status = "final"
            s.call_issues = []


def _executed_key(s) -> str | None:
    if not s.tool_name:
        return None
    return ToolCall(s.tool_name, s.tool_args or {}).key()


def label_run(run: Run, registry: ToolRegistry, th: Thresholds | None = None) -> Run:
    th = th or Thresholds()
    label_steps(run, registry)
    steps = run.steps
    n = len(steps)
    ev: list[str] = []
    sec: list[str] = []

    bad = [s.call_status in ("rejected", "no_call") for s in steps]
    if any("ESCAPED_STRING" in s.call_issues for s in steps):
        sec.append("ESCAPED_STRINGS")
    tail = bad[-th.tail_window:]
    last_truncated = bool(steps) and "TRUNCATED_CALL" in steps[-1].call_issues and steps[-1].call_status == "rejected"
    interface_breakdown = sum(tail) >= th.tail_bad or last_truncated
    invalid_rate = (sum(bad) / n) if n else 0.0
    if invalid_rate >= th.high_invalid_rate:
        sec.append("HIGH_INVALID_RATE")
    if any("TRUNCATED_CALL" in s.call_issues for s in steps):
        sec.append("HAD_TRUNCATION")

    loops = find_loops([_executed_key(s) for s in steps])
    if loops:
        sec.append("HAD_LOOP")
    loop_late = any(i >= n * (1 - th.loop_tail_fraction) for i, _ in loops)

    max_prompt = max((s.prompt_tokens or 0 for s in steps), default=0)
    compactions = sum("compaction" in e for s in steps for e in s.events)
    # A crash because the prompt no longer fits is CONTEXT; compaction alone is routine (63 of 70 E3 runs,
    # including every resolved one), so it is only a secondary flag.
    context_crash = bool(CONTEXT_ERROR.search(getattr(run, "harness_error", "") or ""))
    near_limit = bool(run.context_limit and max_prompt >= th.context_ratio * run.context_limit)
    context_pressure = context_crash or near_limit
    if context_pressure or compactions:
        sec.append("CONTEXT_PRESSURE")
    if compactions:
        sec.append("COMPACTED")

    rt = sum(1 for s in steps if s.runtime_error and s.runtime_error not in NOT_A_RUNTIME_ERROR)
    if rt >= th.runtime_errors_flag:
        sec.append("RUNTIME_ERRORS")
    if any(s.tool_name in GRAPH_TOOLS for s in steps):
        sec.append("USED_CODE_GRAPH")
    if any("nudge" in e for s in steps for e in s.events):
        sec.append("NUDGED")

    ev.append(f"turns={n} invalid_rate={invalid_rate:.2f} tail_bad={sum(tail)}/{len(tail)} "
              f"loops={len(loops)} max_prompt={max_prompt} compactions={compactions} runtime_errors={rt}")

    submitted = run.exit_reason == "submitted"
    if run.exit_reason == "error":
        primary = Primary.ENV_ERROR
        ev.append("exit_reason=error")
    elif run.resolved is True:
        primary = Primary.RESOLVED
    elif run.resolved is None:
        primary = Primary.UNKNOWN_OUTCOME
    elif submitted and (why := env_unverifiable(run)):
        primary = Primary.ENV_UNVERIFIABLE
        ev.append(why)
    elif not submitted:
        escaped = [s for s in steps if "ESCAPED_STRING" in s.call_issues]
        escaped_failed = [s for s in escaped if s.runtime_error in ("EDIT_NO_MATCH", "EDIT_FAILED")]
        if context_crash:
            primary = Primary.CONTEXT
            ev.append("harness error: prompt exceeded the context window left after the output reserve")
        elif len(escaped_failed) >= th.escaped_edits:
            primary = Primary.INTERFACE
            ev.append(f"{len(escaped_failed)} over-escaped edit calls failed (literal \\n in string args)")
        elif interface_breakdown:
            primary = Primary.INTERFACE
            ev.append("last turn truncated" if last_truncated else f"{sum(tail)} bad calls in last {len(tail)} turns")
        elif loop_late:
            primary = Primary.LOOP
            i, sig = loops[-1]
            ev.append(f"{sig.kind} loop (period {sig.period}) at turn {i}")
        elif context_pressure:
            primary = Primary.CONTEXT
            ev.append(f"max_prompt={max_prompt} limit={run.context_limit} compactions={compactions}")
        else:
            primary = Primary.BUDGET
            ev.append(f"exit_reason={run.exit_reason or 'unknown'}")
    else:
        pf = patch_files(run.agent_patch)
        if not run.agent_patch.strip() or not pf:
            primary = Primary.NO_PATCH
        elif run.patch_applied is False or BROKEN_TEST_PATTERNS.search(run.test_output or ""):
            primary = Primary.BROKEN_PATCH
            m = BROKEN_TEST_PATTERNS.search(run.test_output or "")
            ev.append(f"test output: {m.group(0)}" if m else "patch_applied=False")
        elif run.gold_files:
            overlap = set(pf) & set(run.gold_files)
            if overlap:
                primary = Primary.WRONG_FIX
                ev.append(f"touched gold file(s) {sorted(overlap)}")
            else:
                primary = Primary.WRONG_LOCATION
                ev.append(f"patched {pf}, gold {run.gold_files}")
        else:
            primary = Primary.UNRESOLVED_OTHER

    run.primary_failure = primary.value
    run.secondary = sec
    run.evidence = ev
    return run
