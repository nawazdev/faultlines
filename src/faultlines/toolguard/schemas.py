"""Tool specifications: load, look up, and infer from observed calls.

The spec format is the common function-declaration shape used by Gemma, OpenAI
and ADK:

    {"name": "read_file",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

JSON_TYPES = ("string", "integer", "number", "boolean", "array", "object", "null")


@dataclass
class ParamSpec:
    name: str
    type: str | list[str] = "string"
    required: bool = False
    enum: list[Any] | None = None


@dataclass
class ToolSpec:
    name: str
    params: dict[str, ParamSpec] = field(default_factory=dict)
    description: str = ""

    @property
    def required(self) -> list[str]:
        return [p.name for p in self.params.values() if p.required]

    @classmethod
    def from_declaration(cls, decl: dict[str, Any]) -> "ToolSpec":
        params_obj = decl.get("parameters") or decl.get("input_schema") or {}
        props = params_obj.get("properties") or {}
        req = set(params_obj.get("required") or [])
        params = {
            k: ParamSpec(name=k, type=v.get("type", "string"), required=k in req, enum=v.get("enum"))
            for k, v in props.items()
        }
        return cls(name=decl["name"], params=params, description=decl.get("description", ""))

    def to_declaration(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for p in self.params.values():
            d: dict[str, Any] = {"type": p.type}
            if p.enum is not None:
                d["enum"] = p.enum
            props[p.name] = d
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": props, "required": self.required},
        }


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec]):
        self.specs: dict[str, ToolSpec] = {s.name: s for s in specs}

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def get(self, name: str) -> ToolSpec | None:
        return self.specs.get(name)

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    @classmethod
    def from_file(cls, path: str | Path) -> "ToolRegistry":
        data = json.loads(Path(path).read_text())
        decls = data["tools"] if isinstance(data, dict) else data
        return cls(ToolSpec.from_declaration(d) for d in decls)

    @classmethod
    def default_harness(cls) -> "ToolRegistry":
        """The 9 harness tools with GUESSED argument names. Replace after week 1."""
        return cls.from_file(Path(__file__).with_name("harness_tools.json"))

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"tools": [s.to_declaration() for s in self.specs.values()]}, indent=2))


def _json_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "null"


def infer_registry(calls: Iterable[tuple[str, dict[str, Any]]], min_support: float = 0.95) -> ToolRegistry:
    """Infer tool specs from calls the harness ACCEPTED (i.e. that got a normal result).

    An argument is marked required when it appears in >= `min_support` of that
    tool's calls. Use this when the official declarations are not available.
    """
    per_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, args in calls:
        per_tool[name].append(args)
    specs = []
    for name, arg_list in per_tool.items():
        n = len(arg_list)
        seen: Counter[str] = Counter()
        types: dict[str, Counter[str]] = defaultdict(Counter)
        for args in arg_list:
            for k, v in args.items():
                seen[k] += 1
                types[k][_json_type(v)] += 1
        params = {
            k: ParamSpec(name=k, type=types[k].most_common(1)[0][0], required=seen[k] / n >= min_support)
            for k in seen
        }
        specs.append(ToolSpec(name=name, params=params, description=f"inferred from {n} calls"))
    return ToolRegistry(specs)
