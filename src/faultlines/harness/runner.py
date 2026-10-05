"""Agent loop modeled on swegemma's agent_runner (HARNESS_README.md sections 5 and 7).

mode="strict": a tool call counts only if Gemma syntax parses exactly (like the
               harness's vLLM gemma4 parser); anything else is plain text.
mode="guard":  model text goes through toolguard; safe repairs are executed and
               rejected calls get specific feedback instead of a generic nudge.

Output is a FaultLines `Run` with the raw model text for every turn.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..atlas.autolabel import patch_files
from ..atlas.trace import Run, Step
from ..toolguard import ToolGuard, ToolRegistry, extract_calls, parse_body
from ..toolguard.gemma_syntax import CALL_OPEN, GemmaSyntaxError
from ..toolguard.repair import ToolCall
from .graph import CodeGraph
from .model import ChatModel
from .prompt import (DEFAULT_SYSTEM, NUDGE_CONTINUE, NUDGE_MAX_TOKENS, NUDGE_UNCLOSED_CALL, build_agent_prompt,
                     workspace_layout)
from .sandbox import SubprocessSandbox
from .tasks import DataLayout, Task
from .tools import Limits, ToolContext
from .verify import verify_task

HARNESS_VERSION = "faultlines-mini-0.1"


@dataclass
class RunConfig:
    limits: Limits = field(default_factory=Limits)
    mode: str = "strict"                 # strict | guard
    max_nudges: int = 3
    system_prompt: str = DEFAULT_SYSTEM
    compaction_tokens: int | None = 24000   # compact history when a prompt exceeds this (None = never)
    retention_messages: int = 10            # messages kept after compaction (besides system + task)
    python: str = "python3"
    allow_online_fallback: bool = False
    system_site_packages: bool = False      # tests only
    context_limit: int = 32768
    seed: int | None = 0
    label: str = ""                         # free-text config label for the paper

    def hash(self) -> str:
        d = asdict(self)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def strict_calls(text: str) -> list[ToolCall]:
    calls = []
    for raw in extract_calls(text):
        if not raw.closed:
            continue
        try:
            pb = parse_body(raw.body)
        except GemmaSyntaxError:
            continue
        calls.append(ToolCall(pb.name, pb.args))
    return calls


def _visible(text: str) -> str:
    i = text.find(CALL_OPEN)
    return (text if i < 0 else text[:i]).strip()


def _compact(messages: list[dict], keep: int) -> tuple[list[dict], int]:
    head, tail = messages[:2], messages[2:]
    if len(tail) <= keep:
        return messages, 0
    kept = tail[-keep:]
    while kept and kept[0].get("role") == "tool":   # never start on an orphan tool result
        kept = kept[1:]
    removed = len(tail) - len(kept)
    note = {"role": "user", "content": f"[Context compacted: {removed} earlier messages were removed. "
                                       "Re-read files if you need their contents again.]"}
    return head + [note] + kept, removed


def run_task(task: Task, layout: DataLayout, model: ChatModel, cfg: RunConfig, keep_sandbox: bool = False) -> Run:
    run = Run(task_id=task.instance_id, model=model.name, config_hash=cfg.hash(), seed=cfg.seed,
              gold_files=patch_files(task.patch), context_limit=cfg.context_limit,
              harness=f"{HARNESS_VERSION}/{cfg.mode}")
    venv_dir = Path(tempfile.mkdtemp(prefix="flvenv_"))
    sb = SubprocessSandbox(layout.snapshot(task.instance_id), layout.wheels if layout.wheels.exists() else None,
                           python=cfg.python, allow_online_fallback=cfg.allow_online_fallback,
                           venv_path=venv_dir / "venv", keep=keep_sandbox,
                           system_site_packages=cfg.system_site_packages)
    try:
        run.timing["setup_s"] = round(sb.setup(), 1)
        run.notes += sb.notes
        graph = CodeGraph(layout.graph(task.instance_id), layout.embeddings(task.instance_id))
        ctx = ToolContext(sandbox=sb, limits=cfg.limits, graph=graph)
        specs = ToolRegistry.default_harness()
        available = ctx.registry()
        tools = [specs.get(n).to_declaration() for n in available if specs.get(n)]
        guard = ToolGuard(ToolRegistry([specs.get(n) for n in available if specs.get(n)]), detect_loops=False)
        prompt = build_agent_prompt(task, cfg.limits, graph.available, workspace_layout(sb))
        messages: list[dict] = [{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": prompt}]

        ctx.agent_start = time.time()
        nudges = turns = over_budget_turns = 0
        exit_reason = ""
        while True:
            if ctx.remaining_seconds() <= 0:
                exit_reason = "timeout"; break
            if turns >= cfg.limits.turns:
                exit_reason = "max_turns"; break
            turns += 1
            try:
                t = model.generate(messages, tools)
            except Exception as e:  # server down, OOM, ...
                run.evidence.append(f"model_error: {type(e).__name__}: {str(e)[:300]}")
                exit_reason = "error"; break
            step = Step(index=len(run.steps), raw_output=t.text, prompt_tokens=t.prompt_tokens,
                        completion_tokens=t.completion_tokens, finish_reason=t.finish_reason)
            if t.finish_reason == "context_overflow":
                step.events.append("context_overflow")

            feedback = ""
            if cfg.mode == "guard":
                g = guard.check(t.text)
                calls = g.calls if g.status in ("valid", "repaired") else []
                if g.status == "repaired":
                    step.events.append("guard_repaired:" + ",".join(a for o in g.outcomes for a in o.applied))
                if g.status == "rejected":
                    feedback = g.feedback
                    step.events.append("guard_rejected")
            else:
                calls = strict_calls(t.text)

            if calls:
                nudges = 0
                messages.append({"role": "assistant", "content": _visible(t.text),
                                 "tool_calls": [{"type": "function", "function": {"name": c.name, "arguments": c.args}}
                                                for c in calls]})
                for j, c in enumerate(calls):
                    result = ctx.call(c.name, c.args)
                    messages.append({"role": "tool", "name": c.name, "content": result})
                    if j == 0:
                        step.tool_name, step.tool_args, step.tool_result = c.name, c.args, result[:20000]
                    else:
                        step.extra_calls.append({"name": c.name, "args": c.args, "result": result[:5000]})
                run.steps.append(step)
                if ctx.patch_submitted:
                    exit_reason = "submitted"; break
                rem = ctx.remaining_calls()
                if rem is not None and rem <= 0:
                    over_budget_turns += 1
                    if over_budget_turns > 1:
                        exit_reason = "max_tool_calls"; break
            else:
                nudges += 1
                messages.append({"role": "assistant", "content": _visible(t.text) or "(no tool call)"})
                if nudges > cfg.max_nudges:
                    run.steps.append(step)
                    exit_reason = "max_nudges"; break
                if feedback:
                    msg = feedback
                elif CALL_OPEN in t.text:
                    msg = NUDGE_UNCLOSED_CALL
                elif (t.finish_reason or "").lower() in ("length", "max_tokens"):
                    msg = NUDGE_MAX_TOKENS
                else:
                    msg = NUDGE_CONTINUE
                messages.append({"role": "user", "content": msg})
                step.events.append("nudge")
                run.steps.append(step)

            if cfg.compaction_tokens and (t.prompt_tokens or 0) > cfg.compaction_tokens:
                messages, removed = _compact(messages, cfg.retention_messages)
                if removed:
                    step.events.append(f"compaction:{removed}")

        run.timing["agent_s"] = round(time.time() - ctx.agent_start, 1)
        run.exit_reason = exit_reason
        run.agent_patch = ctx.submitted_patch if ctx.patch_submitted else sb.diff()
    except Exception as e:
        run.exit_reason = "error"
        run.evidence.append("harness_error: " + "".join(traceback.format_exception_only(type(e), e)).strip()[:500])
        return run
    finally:
        sb.cleanup()

    try:
        if run.agent_patch.strip():
            v = verify_task(task, layout, run.agent_patch, venv_dir / "venv", cfg.python, cfg.allow_online_fallback)
            run.resolved, run.patch_applied, run.test_output = v.resolved, v.patch_applied, v.output
            run.timing["verify_s"] = round(v.seconds, 1)
        else:
            run.resolved, run.patch_applied = False, None
    except Exception as e:
        run.evidence.append("verify_error: " + str(e)[:300])
        run.exit_reason = run.exit_reason or "error"
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)
    return run
