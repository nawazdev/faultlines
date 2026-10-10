"""The FaultLines trace record: one agent run on one task.

This is the released dataset's schema. Adapters (adapters.py) convert harness
output into it; the labeler (autolabel.py) fills in the label fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "faultlines-trace-v0.1"


@dataclass
class Step:
    index: int
    raw_output: str = ""                        # the model's full text for this turn (may include thinking)
    tool_name: str | None = None                # as executed by the harness (None if nothing ran)
    tool_args: dict[str, Any] | None = None
    tool_result: str = ""                       # observation text returned to the model
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None            # e.g. "stop", "length"/"MAX_TOKENS"
    events: list[str] = field(default_factory=list)   # harness events, e.g. "nudge", "compaction"
    extra_calls: list[dict] = field(default_factory=list)  # further calls in the same turn (name, args, result)
    # filled by the labeler
    call_status: str | None = None              # valid | repaired | rejected | no_call
    call_issues: list[str] = field(default_factory=list)
    runtime_error: str | None = None            # e.g. EDIT_NO_MATCH, FILE_NOT_FOUND, TIMEOUT


@dataclass
class Run:
    task_id: str
    model: str = ""
    config_hash: str = ""
    seed: int | None = None
    resolved: bool | None = None
    exit_reason: str = ""                       # submitted | timeout | max_turns | max_tool_calls | error | ...
    agent_patch: str = ""
    patch_applied: bool | None = None
    test_output: str = ""
    test_exit_code: int | None = None
    gold_files: list[str] = field(default_factory=list)
    context_limit: int | None = None
    steps: list[Step] = field(default_factory=list)
    synthetic: bool = False                     # True for hand-made examples; never mix into results
    harness: str = ""                           # e.g. "faultlines-mini-0.1/strict"
    timing: dict[str, float] = field(default_factory=dict)   # setup_s, agent_s, verify_s
    notes: list[str] = field(default_factory=list)            # environment notes (e.g. online_pip_fallback)
    harness_error: str = ""                     # error message the harness recorded for the run, if any
    # filled by the labeler / annotator
    primary_failure: str | None = None
    secondary: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    human_label: str | None = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Run":
        d = dict(d)
        steps = [Step(**s) for s in d.pop("steps", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(steps=steps, **{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_runs(path: str | Path) -> Iterator[Run]:
    """Read .jsonl (one run per line), a .json list, a single .json run, or a directory of them."""
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix in (".json", ".jsonl"):
                yield from read_runs(f)
        return
    text = p.read_text()
    if p.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                yield Run.from_dict(json.loads(line))
        return
    data = json.loads(text)
    for d in data if isinstance(data, list) else [data]:
        yield Run.from_dict(d)


def write_runs(runs: Iterable[Run], path: str | Path) -> int:
    n = 0
    with open(path, "w") as f:
        for r in runs:
            f.write(json.dumps(r.to_dict()) + "\n")
            n += 1
    return n
