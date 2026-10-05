"""Stress set for RQ3: damage valid calls in known ways, check what the guard does.

Each case says what SHOULD happen:
    "repair" -> guard must return exactly the original call
    "refuse" -> guard must reject (repairing would mean inventing or dropping content)
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from .toolguard import ToolGuard, ToolRegistry, render_call, validate_call
from .toolguard.gemma_syntax import CALL_OPEN, CALL_CLOSE, STR, render_value
from .toolguard.repair import ARG_ALIASES, TOOL_ALIASES


@dataclass
class Case:
    kind: str
    text: str
    expect: str                   # "repair" | "refuse"
    name: str
    args: dict[str, Any]


def _drop_close_tag(name, args, reg, rng):
    return render_call(name, args)[: -len(CALL_CLOSE)], "repair"


def _truncate_in_string(name, args, reg, rng):
    keys = [k for k, v in args.items() if isinstance(v, str) and len(v) >= 8]
    if not keys:
        return None
    k = rng.choice(keys)
    full = render_call(name, args)
    marker = f"{k}:{STR}"
    start = full.index(marker) + len(marker)
    cut = start + rng.randint(1, len(args[k]) - 1)
    return full[:cut], "refuse"


def _json_args(name, args, reg, rng):
    return f"{CALL_OPEN}call:{name}{json.dumps(args)}{CALL_CLOSE}", "repair"


def _json_call(name, args, reg, rng):
    return f"{CALL_OPEN}{json.dumps({'name': name, 'arguments': args})}{CALL_CLOSE}", "repair"


def _tool_alias(name, args, reg, rng):
    aliases = [a for a, t in TOOL_ALIASES.items() if t == name]
    if not aliases:
        return None
    return render_call(rng.choice(aliases), args), "repair"


def _tool_typo(name, args, reg, rng):
    if len(name) < 6:
        return None
    i = rng.randint(1, len(name) - 2)
    typo = name[:i] + name[i + 1] + name[i] + name[i + 2:]
    if typo == name or typo in reg:
        return None
    return render_call(typo, args), "repair"


def _arg_alias(name, args, reg, rng):
    options = [(k, a) for k in args for a, t in ARG_ALIASES.items() if t == k and a not in args]
    if not options:
        return None
    k, a = rng.choice(options)
    bad = {(a if kk == k else kk): v for kk, v in args.items()}
    return render_call(name, bad), "repair"


def _int_as_string(name, args, reg, rng):
    keys = [k for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool)]
    if not keys:
        return None
    k = rng.choice(keys)
    bad = dict(args)
    bad[k] = str(args[k])
    return render_call(name, bad), "repair"


def _extra_arg(name, args, reg, rng):
    bad = dict(args)
    bad["verbose"] = True
    return render_call(name, bad), "refuse"


def _missing_required(name, args, reg, rng):
    spec = reg.get(name)
    req = [k for k in (spec.required if spec else []) if k in args]
    if not req:
        return None
    drop = rng.choice(req)
    bad = {k: v for k, v in args.items() if k != drop}
    return render_call(name, bad), "refuse"


CORRUPTIONS: dict[str, Callable] = {
    "drop_close_tag": _drop_close_tag,
    "truncate_in_string": _truncate_in_string,
    "json_args": _json_args,
    "json_call": _json_call,
    "tool_alias": _tool_alias,
    "tool_typo": _tool_typo,
    "arg_alias": _arg_alias,
    "int_as_string": _int_as_string,
    "extra_arg": _extra_arg,
    "missing_required": _missing_required,
}


def sample_calls() -> list[tuple[str, dict[str, Any]]]:
    """Small built-in set for demos and tests. Use real calls from traces for the paper."""
    return [
        ("read_file", {"filepath": "src/utils/dates.py", "start_line": 40, "end_line": 120}),
        ("read_file", {"filepath": "tests/test_parser.py"}),
        ("run_command", {"command": "python -m pytest tests/test_parser.py -x -q"}),
        ("run_command", {"command": "grep -rn 'def parse_date' src/"}),
        ("edit_file", {"filepath": "src/utils/dates.py", "old_string": "if tz is None:\n    return dt",
                       "new_string": "if tz is None:\n    return dt.replace(tzinfo=UTC)"}),
        ("write_file", {"filepath": "/tmp/repro.py", "content": "from utils.dates import parse_date\nprint(parse_date('2020-01-01'))\n"}),
        ("search_similar_code", {"query": "parse_date", "k": 5}),
        ("get_code_neighbors", {"node": "utils.dates.parse_date"}),
    ]


def build_cases(calls: list[tuple[str, dict[str, Any]]], registry: ToolRegistry, seed: int = 0) -> list[Case]:
    rng = random.Random(seed)
    cases = []
    for name, args in calls:
        if validate_call(name, args, registry):
            continue  # only start from calls that are valid under this registry
        for kind, fn in CORRUPTIONS.items():
            res = fn(name, args, registry, rng)
            if res is None:
                continue
            text, expect = res
            cases.append(Case(kind, text, expect, name, args))
    return cases


def evaluate(cases: list[Case], registry: ToolRegistry, allow_lossy: bool = False) -> dict[str, Any]:
    per_kind: dict[str, Counter] = defaultdict(Counter)
    tot = Counter()
    for c in cases:
        g = ToolGuard(registry, allow_lossy=allow_lossy, detect_loops=False)
        r = g.check(c.text)
        repaired = r.status in ("valid", "repaired")
        exact = repaired and len(r.calls) == 1 and r.calls[0].name == c.name and r.calls[0].args == c.args
        correct = exact if c.expect == "repair" else not repaired
        per_kind[c.kind]["n"] += 1
        per_kind[c.kind]["correct"] += correct
        tot[f"expect_{c.expect}"] += 1
        tot[f"expect_{c.expect}_correct"] += correct
        tot["repaired_outputs"] += repaired
        tot["repaired_exact"] += exact
    return {
        "per_kind": {k: dict(v) for k, v in per_kind.items()},
        "coverage": tot["expect_repair_correct"] / max(1, tot["expect_repair"]),
        "precision": tot["repaired_exact"] / max(1, tot["repaired_outputs"]),
        "safe_refusal": tot["expect_refuse_correct"] / max(1, tot["expect_refuse"]),
        "n": len(cases),
    }


def report(result: dict[str, Any]) -> str:
    lines = ["| Corruption | Cases | Handled correctly |", "| --- | --- | --- |"]
    for k, v in sorted(result["per_kind"].items()):
        lines.append(f"| {k} | {v['n']} | {100 * v['correct'] / v['n']:.1f}% |")
    lines.append("")
    lines.append(f"Cases: {result['n']}  |  repair coverage {100 * result['coverage']:.1f}%  |  "
                 f"repair precision {100 * result['precision']:.1f}%  |  safe refusal {100 * result['safe_refusal']:.1f}%")
    return "\n".join(lines)
