"""Classify what is wrong with a tool call. One stable code per problem.

Call-level error codes (these become the step labels in FaultLines-Traces):

    NO_CALL            the turn contains no tool call at all
    TRUNCATED_CALL     <|tool_call> opened but never closed (usually MAX_TOKENS)
    MALFORMED_SYNTAX   the call body does not parse as Gemma syntax
    UNKNOWN_TOOL       tool name not in the registry
    MISSING_ARG        a required argument is absent
    UNEXPECTED_ARG     an argument the tool does not declare
    WRONG_TYPE         argument value has the wrong JSON type
    BAD_ENUM           value not in the declared enum
    PATH_ESCAPE        a path argument contains '..' (the harness blocks it)
    REPEATED_CALL      identical call repeated (set by loops.py, not here)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .schemas import ToolRegistry


class CallError(str, Enum):
    NO_CALL = "NO_CALL"
    TRUNCATED_CALL = "TRUNCATED_CALL"
    MALFORMED_SYNTAX = "MALFORMED_SYNTAX"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    MISSING_ARG = "MISSING_ARG"
    UNEXPECTED_ARG = "UNEXPECTED_ARG"
    WRONG_TYPE = "WRONG_TYPE"
    BAD_ENUM = "BAD_ENUM"
    PATH_ESCAPE = "PATH_ESCAPE"
    REPEATED_CALL = "REPEATED_CALL"


@dataclass
class Issue:
    code: CallError
    detail: str = ""
    arg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "detail": self.detail, "arg": self.arg}


PATH_ARG_NAMES = {"filepath", "path", "file_path", "filename", "file"}


def type_ok(value: Any, expected: str | list[str]) -> bool:
    kinds = expected if isinstance(expected, list) else [expected]
    for t in kinds:
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "null" and value is None:
            return True
    return False


def validate_call(name: str, args: dict[str, Any], registry: ToolRegistry) -> list[Issue]:
    spec = registry.get(name)
    if spec is None:
        return [Issue(CallError.UNKNOWN_TOOL, f"'{name}' not in {registry.names}")]
    issues: list[Issue] = []
    for req in spec.required:
        if req not in args:
            issues.append(Issue(CallError.MISSING_ARG, f"'{req}' is required", req))
    for k, v in args.items():
        p = spec.params.get(k)
        if p is None:
            issues.append(Issue(CallError.UNEXPECTED_ARG, f"'{k}' not declared by {name}", k))
            continue
        if not type_ok(v, p.type):
            issues.append(Issue(CallError.WRONG_TYPE, f"'{k}' should be {p.type}, got {type(v).__name__}", k))
            continue
        if p.enum is not None and v not in p.enum:
            issues.append(Issue(CallError.BAD_ENUM, f"'{k}'={v!r} not in {p.enum}", k))
        if k in PATH_ARG_NAMES and isinstance(v, str) and ".." in v.replace("\\", "/").split("/"):
            issues.append(Issue(CallError.PATH_ESCAPE, f"'{k}' contains '..'", k))
    return issues
