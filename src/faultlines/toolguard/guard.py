"""One entry point: check (and optionally repair) a model turn.

    guard = ToolGuard(ToolRegistry.default_harness())
    result = guard.check(model_output_text)
    if result.status in ("valid", "repaired"):
        execute(result.calls)
    else:
        send_back(result.feedback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gemma_syntax import extract_calls
from .loops import LoopDetector, LoopSignal
from .repair import (RepairOutcome, ToolCall, feedback_message, parse_structured_args,
                     repair_parsed, repair_raw)
from .schemas import ToolRegistry
from .validate import CallError, Issue


@dataclass
class GuardResult:
    status: str                                   # valid | repaired | rejected | no_call
    calls: list[ToolCall] = field(default_factory=list)
    outcomes: list[RepairOutcome] = field(default_factory=list)
    loop: LoopSignal | None = None
    feedback: str = ""

    @property
    def issues(self) -> list[Issue]:
        """Problems present in the model's ORIGINAL output (before repair)."""
        out = [i for o in self.outcomes for i in o.initial]
        if not self.outcomes and self.status == "no_call":
            out.append(Issue(CallError.NO_CALL, "no tool call in turn"))
        if self.loop is not None:
            out.append(Issue(CallError.REPEATED_CALL, f"{self.loop.kind} x{self.loop.count}"))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "calls": [c.to_dict() for c in self.calls],
            "issues": [i.to_dict() for i in self.issues],
            "repairs": [a for o in self.outcomes for a in o.applied],
            "lossy": any(o.lossy for o in self.outcomes),
            "loop": None if self.loop is None else self.loop.__dict__,
            "feedback": self.feedback,
        }


class ToolGuard:
    def __init__(self, registry: ToolRegistry, allow_lossy: bool = False, detect_loops: bool = True, **loop_kw):
        self.registry = registry
        self.allow_lossy = allow_lossy
        self.loops = LoopDetector(**loop_kw) if detect_loops else None

    def check(self, text: str) -> GuardResult:
        raws = extract_calls(text or "")
        return self._finish([repair_raw(r, self.registry, self.allow_lossy) for r in raws])

    def check_structured(self, name: str, arguments: Any) -> GuardResult:
        """For servers that already parsed the call (OpenAI-style `tool_calls`)."""
        args, applied = parse_structured_args(arguments)
        if args is None:
            bad = Issue(CallError.MALFORMED_SYNTAX, "arguments are not a JSON object")
            return self._finish([RepairOutcome(call=None, initial=[bad], remaining=[bad])])
        syntax = [Issue(CallError.MALFORMED_SYNTAX, "invalid JSON arguments")] if applied else []
        return self._finish([repair_parsed(ToolCall(name, args), self.registry, self.allow_lossy,
                                           applied=applied, syntax_issues=syntax)])

    def _finish(self, outcomes: list[RepairOutcome]) -> GuardResult:
        if not outcomes:
            return GuardResult(status="no_call",
                               feedback="No tool call found. Call exactly one tool, or submit_patch if done.")
        if any(not o.ok for o in outcomes):
            bad = next(o for o in outcomes if not o.ok)
            return GuardResult(status="rejected", outcomes=outcomes, feedback=feedback_message(bad, self.registry))
        calls = [o.call for o in outcomes if o.call is not None]
        loop = None
        if self.loops is not None:
            for c in calls:
                loop = self.loops.observe(c.key()) or loop
        status = "repaired" if any(o.applied for o in outcomes) else "valid"
        fb = ""
        if loop is not None:
            fb = ("You have repeated the same action several times without progress. "
                  "Try a different approach: re-read the error, check the file content, or change strategy.")
        return GuardResult(status=status, calls=calls, outcomes=outcomes, loop=loop, feedback=fb)
