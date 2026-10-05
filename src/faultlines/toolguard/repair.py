"""Safe, deterministic repair of broken tool calls.

Design rule: repair only what the model clearly meant; never invent content.

* A call cut off inside a string (e.g. half of a write_file body) is REFUSED,
  because completing it would write a partial file.
* Renames (tool or argument) happen only through an alias table or a very close
  spelling match, and only to a name that exists and is not already used.
* Dropping an argument loses information, so it is off unless allow_lossy=True.

Every repair is recorded by name so the paper can report exactly what was done.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .gemma_syntax import GemmaSyntaxError, RawCall, parse_body, render_value, CALL_OPEN, CALL_CLOSE
from .schemas import ToolRegistry
from .validate import CallError, Issue, validate_call, type_ok

TOOL_ALIASES = {
    "bash": "run_command", "shell": "run_command", "execute": "run_command", "execute_bash": "run_command",
    "run": "run_command", "cmd": "run_command", "terminal": "run_command", "run_bash": "run_command",
    "cat": "read_file", "view": "read_file", "open_file": "read_file", "view_file": "read_file",
    "read": "read_file", "open": "read_file",
    "str_replace": "edit_file", "str_replace_editor": "edit_file", "replace": "edit_file",
    "edit": "edit_file", "apply_edit": "edit_file", "replace_in_file": "edit_file",
    "create_file": "write_file", "create": "write_file", "write": "write_file", "save_file": "write_file",
    "submit": "submit_patch", "finish": "submit_patch", "done": "submit_patch", "submit_solution": "submit_patch",
    "status": "get_status",
}

ARG_ALIASES = {
    # read_file / edit_file / write_file take `filepath` in the swegemma harness
    "path": "filepath", "file": "filepath", "filename": "filepath", "file_path": "filepath",
    "path_to_file": "filepath",
    "cmd": "command", "bash": "command", "script": "command",
    "old": "old_string", "old_str": "old_string", "old_text": "old_string", "search": "old_string",
    "find": "old_string", "original": "old_string",
    "new": "new_string", "new_str": "new_string", "new_text": "new_string", "replace": "new_string",
    "replacement": "new_string",
    "replace_all": "allow_multiple", "multiple": "allow_multiple",
    "text": "content", "contents": "content", "file_text": "content", "data": "content", "body": "content",
    "start": "start_line", "line_start": "start_line", "from_line": "start_line",
    "end": "end_line", "line_end": "end_line", "to_line": "end_line",
    "top_k": "k", "limit": "k", "n": "k", "max_results": "k", "q": "query",
    "symbol": "node", "name": "node", "limit_neighbors": "max_neighbors",
}

_TRUNCATION_CODES = {"UNTERMINATED_STRING", "UNEXPECTED_END", "UNTERMINATED_KEY"}
_INT_RE = re.compile(r"^-?\d+$")


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]

    def key(self) -> str:
        """Canonical string used for loop detection."""
        return self.name + json.dumps(self.args, sort_keys=True, default=str)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args}


@dataclass
class RepairOutcome:
    call: ToolCall | None                    # best-effort call (None if nothing usable)
    initial: list[Issue] = field(default_factory=list)    # problems found before repair
    remaining: list[Issue] = field(default_factory=list)  # problems left after repair
    applied: list[str] = field(default_factory=list)      # names of repairs applied
    lossy: bool = False

    @property
    def ok(self) -> bool:
        return self.call is not None and not self.remaining

    @property
    def status(self) -> str:
        if not self.ok:
            return "rejected"
        return "repaired" if self.applied else "valid"


# ---------------------------------------------------------------- parsing layer

def _parse_with_fallbacks(raw: RawCall) -> tuple[ToolCall | None, list[str], Issue | None]:
    """Return (call, repairs_applied, fatal_issue)."""
    applied: list[str] = []
    body = raw.body

    if not raw.closed:
        try:
            pb = parse_body(body, allow_trailing=True)
        except GemmaSyntaxError as e:
            if e.code in _TRUNCATION_CODES:
                return None, [], Issue(CallError.TRUNCATED_CALL, f"cut off: {e.code}")
            return None, [], Issue(CallError.MALFORMED_SYNTAX, f"{e.code}: {e.detail}")
        if _looks_like_json(pb.warnings):
            json_call = _json_args(body.strip())
            if json_call is not None:
                return json_call, ["CLOSE_TAG", "JSON_ARGS"], None
        applied.append("CLOSE_TAG")
        applied += [f"LENIENT:{w.split(':')[0]}" for w in pb.warnings]
        return ToolCall(pb.name, pb.args), applied, None

    try:
        pb = parse_body(body)
        if _looks_like_json(pb.warnings):
            # e.g. call:read_file{"path": "a.py"} parses leniently but with quotes kept
            json_call = _json_args(body.strip())
            if json_call is not None:
                return json_call, ["JSON_ARGS"], None
        applied += [f"LENIENT:{w.split(':')[0]}" for w in pb.warnings]
        return ToolCall(pb.name, pb.args), applied, None
    except GemmaSyntaxError as e:
        first_error = e

    s = body.strip()
    # (a) whole body is a JSON / python dict: {"name": ..., "arguments": {...}}
    obj = _loads_any(s)
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if isinstance(args, str):
            args = _loads_any(args)
        if isinstance(args, dict):
            return ToolCall(obj["name"], args), ["JSON_CALL"], None

    # (c) call:NAME followed by JSON or python-literal arguments
    json_call = _json_args(s)
    if json_call is not None:
        return json_call, ["JSON_ARGS"], None

    # (b) missing 'call:' prefix but otherwise Gemma syntax: NAME{...}
    if not s.startswith("call:") and re.match(r"^[A-Za-z_][\w.\-]*\s*\{", s):
        try:
            pb = parse_body("call:" + s)
            return ToolCall(pb.name, pb.args), ["ADD_CALL_PREFIX"], None
        except GemmaSyntaxError:
            pass

    return None, [], Issue(CallError.MALFORMED_SYNTAX, f"{first_error.code}: {first_error.detail}")


def _looks_like_json(warnings: list[str]) -> bool:
    return any(w.startswith(("QUOTED_KEY", "BARE_STRING")) for w in warnings)


def _json_args(s: str) -> ToolCall | None:
    m = re.match(r"^(?:call:)?\s*([A-Za-z_][\w.\-]*)\s*(\{.*\})\s*$", s, re.S)
    if not m:
        return None
    args = _loads_any(m.group(2))
    return ToolCall(m.group(1), args) if isinstance(args, dict) else None


def _loads_any(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        # json with trailing commas
        return json.loads(re.sub(r",\s*([}\]])", r"\1", s))
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(s)  # safe: literals only
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


# ---------------------------------------------------------------- schema layer

def _coerce(value: Any, expected: str | list[str]) -> tuple[bool, Any]:
    kinds = expected if isinstance(expected, list) else [expected]
    for t in kinds:
        if t == "integer" and isinstance(value, str) and _INT_RE.match(value.strip()):
            return True, int(value.strip())
        if t == "integer" and isinstance(value, float) and value.is_integer():
            return True, int(value)
        if t == "number" and isinstance(value, str):
            try:
                return True, float(value.strip())
            except ValueError:
                pass
        if t == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return True, value.strip().lower() == "true"
        if t == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, str(value)
        if t in ("array", "object") and isinstance(value, str):
            parsed = _loads_any(value)
            if type_ok(parsed, t):
                return True, parsed
        if t == "array" and not isinstance(value, (list, dict)) and value is not None:
            return True, [value]
    return False, value


def repair_call(call: ToolCall, registry: ToolRegistry, allow_lossy: bool = False) -> tuple[ToolCall, list[str], bool]:
    applied: list[str] = []
    lossy = False
    name = call.name
    args = dict(call.args)

    if name not in registry:
        target = TOOL_ALIASES.get(name.lower())
        if target in registry:
            applied.append(f"TOOL_ALIAS:{name}->{target}")
            name = target
        else:
            close = difflib.get_close_matches(name, registry.names, n=1, cutoff=0.8)
            if close:
                applied.append(f"TOOL_TYPO:{name}->{close[0]}")
                name = close[0]
    spec = registry.get(name)
    if spec is None:
        return ToolCall(name, args), applied, lossy

    for k in list(args):
        if k in spec.params:
            continue
        free = [p for p in spec.params if p not in args]
        target = ARG_ALIASES.get(k.lower())
        if target not in free:
            close = difflib.get_close_matches(k, free, n=1, cutoff=0.8)
            target = close[0] if close else None
            kind = "ARG_TYPO"
        else:
            kind = "ARG_ALIAS"
        if target:
            args[target] = args.pop(k)
            applied.append(f"{kind}:{k}->{target}")

    for k, v in list(args.items()):
        p = spec.params.get(k)
        if p is not None and not type_ok(v, p.type):
            ok, nv = _coerce(v, p.type)
            if ok:
                args[k] = nv
                applied.append(f"COERCE:{k}")

    if allow_lossy:
        for k in [k for k in args if k not in spec.params]:
            args.pop(k)
            applied.append(f"DROP_ARG:{k}")
            lossy = True

    return ToolCall(name, args), applied, lossy


def repair_raw(raw: RawCall, registry: ToolRegistry, allow_lossy: bool = False) -> RepairOutcome:
    call, applied, fatal = _parse_with_fallbacks(raw)
    if fatal is not None:
        return RepairOutcome(call=None, initial=[fatal], remaining=[fatal])
    assert call is not None
    syntax: list[Issue] = []
    if not raw.closed:
        syntax.append(Issue(CallError.TRUNCATED_CALL, "close tag missing; arguments complete"))
    elif any(a in applied for a in ("JSON_CALL", "ADD_CALL_PREFIX", "JSON_ARGS")):
        syntax.append(Issue(CallError.MALFORMED_SYNTAX, "non-Gemma argument syntax"))
    return repair_parsed(call, registry, allow_lossy, applied=applied, syntax_issues=syntax)


def repair_parsed(call: ToolCall, registry: ToolRegistry, allow_lossy: bool = False,
                  applied: list[str] | None = None, syntax_issues: list[Issue] | None = None) -> RepairOutcome:
    """Schema-level validation and repair of an already-parsed call."""
    initial = list(syntax_issues or []) + validate_call(call.name, call.args, registry)
    fixed, more, lossy = repair_call(call, registry, allow_lossy)
    remaining = validate_call(fixed.name, fixed.args, registry)
    return RepairOutcome(call=fixed, initial=initial, remaining=remaining,
                         applied=list(applied or []) + more, lossy=lossy)


def parse_structured_args(arguments: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Arguments from an OpenAI-style tool_call: a dict, or a JSON string (maybe broken)."""
    if isinstance(arguments, dict):
        return arguments, []
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return {}, []
    try:
        val = json.loads(arguments)
        return (val, []) if isinstance(val, dict) else (None, [])
    except (json.JSONDecodeError, TypeError):
        val = _loads_any(arguments)
        return (val, ["LENIENT_JSON"]) if isinstance(val, dict) else (None, [])


# ---------------------------------------------------------------- feedback

def example_call(tool: str, registry: ToolRegistry) -> str:
    spec = registry.get(tool)
    if spec is None:
        return ""
    placeholder = {"string": "...", "integer": 1, "number": 1.0, "boolean": True, "array": [], "object": {}}
    args = {}
    for p in spec.params.values():
        if p.required:
            t = p.type[0] if isinstance(p.type, list) else p.type
            args[p.name] = placeholder.get(t, "...")
    return f"{CALL_OPEN}call:{tool}{render_value(args)}{CALL_CLOSE}"


def feedback_message(outcome: RepairOutcome, registry: ToolRegistry) -> str:
    """Short, specific error text to send back to the model (for your own scaffold)."""
    if outcome.ok:
        return ""
    lines = ["Your last tool call was not executed."]
    for iss in outcome.remaining:
        lines.append(f"- {iss.code.value}: {iss.detail}")
    codes = {i.code for i in outcome.remaining}
    if CallError.TRUNCATED_CALL in codes:
        lines.append("Your output was cut off. Make the call shorter: edit a smaller region, "
                     "or write the file in parts.")
    if CallError.UNKNOWN_TOOL in codes:
        lines.append(f"Available tools: {', '.join(registry.names)}.")
    if outcome.call is not None and outcome.call.name in registry:
        lines.append("Correct form: " + example_call(outcome.call.name, registry))
    return "\n".join(lines)


# --- Over-escaped string arguments (observed in the official-harness run) -----------------
# Gemma 4 31B sometimes writes edit_file/write_file strings with literal backslash escapes
# ("a\\nb" instead of a real newline). Every such edit failed with "old_string not found".
ESCAPABLE_ARGS = ("old_string", "new_string", "content")
_ESCAPES = {"\\n": "\n", "\\t": "\t", '\\"': '"', "\\'": "'"}


def is_over_escaped(value: str) -> bool:
    """A multi-line looking string that contains literal \\n escapes but no real newline."""
    return isinstance(value, str) and "\\n" in value and "\n" not in value


def unescape_string(value: str) -> str:
    for k, v in _ESCAPES.items():
        value = value.replace(k, v)
    return value


def unescape_args(tool: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Candidate repair: decode literal escapes in string args of edit tools.

    Safe only when the caller then checks the decoded old_string matches the file exactly once
    (harness.editing does this). Returns (args, names of changed keys)."""
    if tool not in ("edit_file", "write_file"):
        return args, []
    out, changed = dict(args), []
    for k in ESCAPABLE_ARGS:
        if is_over_escaped(out.get(k)):
            out[k] = unescape_string(out[k])
            changed.append(k)
    return out, changed
