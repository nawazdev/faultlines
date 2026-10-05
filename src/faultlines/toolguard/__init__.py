"""toolguard: parse, validate, and safely repair Gemma 4 tool calls."""

from .gemma_syntax import extract_calls, parse_body, render_call, GemmaSyntaxError, RawCall
from .guard import GuardResult, ToolGuard
from .loops import LoopDetector, find_loops
from .repair import RepairOutcome, ToolCall, repair_raw, feedback_message
from .schemas import ToolRegistry, ToolSpec, infer_registry
from .validate import CallError, Issue, validate_call

__all__ = [
    "extract_calls", "parse_body", "render_call", "GemmaSyntaxError", "RawCall",
    "GuardResult", "ToolGuard", "LoopDetector", "find_loops",
    "RepairOutcome", "ToolCall", "repair_raw", "feedback_message",
    "ToolRegistry", "ToolSpec", "infer_registry",
    "CallError", "Issue", "validate_call",
]
