"""FaultLines failure taxonomy (v0 — freeze as v1 at the end of week 2).

Every run gets exactly ONE primary label, chosen by the precedence rule in
autolabel.py, plus any number of secondary flags. Groups answer RQ2
("interface vs reasoning").
"""

from __future__ import annotations

from enum import Enum


class Primary(str, Enum):
    RESOLVED = "RESOLVED"
    ENV_ERROR = "ENV_ERROR"              # harness / infra failure; excluded from rates
    ENV_UNVERIFIABLE = "ENV_UNVERIFIABLE"  # submitted, but verification broke for environment reasons
    # --- process failures: the run never produced a considered answer
    INTERFACE = "INTERFACE"              # tool calls kept breaking (truncated, malformed, no call)
    LOOP = "LOOP"                        # repeated the same action(s) without progress
    CONTEXT = "CONTEXT"                  # ran into the context limit / lost state after compaction
    BUDGET = "BUDGET"                    # out of time/turns/calls while otherwise working normally
    # --- solution failures: the run submitted, but the answer was wrong
    NO_PATCH = "NO_PATCH"                # submitted an empty diff
    BROKEN_PATCH = "BROKEN_PATCH"        # patch did not apply, or tests failed to even collect/import
    WRONG_LOCATION = "WRONG_LOCATION"    # edited none of the files the reference fix touches
    WRONG_FIX = "WRONG_FIX"              # edited the right file(s) but tests still fail
    UNRESOLVED_OTHER = "UNRESOLVED_OTHER"  # failed, but no gold files to tell location from fix
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"  # resolution result missing


GROUPS = {
    "interface": {Primary.INTERFACE, Primary.LOOP, Primary.CONTEXT},
    "budget": {Primary.BUDGET},
    "reasoning": {Primary.NO_PATCH, Primary.BROKEN_PATCH, Primary.WRONG_LOCATION,
                  Primary.WRONG_FIX, Primary.UNRESOLVED_OTHER},
    "success": {Primary.RESOLVED},
    "excluded": {Primary.ENV_ERROR, Primary.ENV_UNVERIFIABLE, Primary.UNKNOWN_OUTCOME},
}


def group_of(label: str) -> str:
    for g, members in GROUPS.items():
        if label in {m.value for m in members}:
            return g
    return "unknown"


DEFINITIONS = {
    Primary.RESOLVED: "Hidden tests pass (harness Phase 2 exit code 0).",
    Primary.ENV_ERROR: "Run ended by an infrastructure error, not by the agent.",
    Primary.ENV_UNVERIFIABLE: "Submitted, but the tests could not judge the patch: a test dependency was missing "
                              "(ModuleNotFoundError for a module the patch does not define) or the test run timed out "
                              "(exit code 124).",
    Primary.INTERFACE: "Did not submit; >=3 of the last 5 turns had a rejected or missing tool call, "
                       "or the last turn was a truncated call.",
    Primary.LOOP: "Did not submit; a repeat/cycle loop was detected in the final third of the run.",
    Primary.CONTEXT: "Did not submit; prompt size reached >=90% of the context limit, "
                     "or a compaction event occurred.",
    Primary.BUDGET: "Did not submit and none of the above: ran out of time, turns or tool calls.",
    Primary.NO_PATCH: "Submitted with an empty diff.",
    Primary.BROKEN_PATCH: "Patch failed to apply, or the test run failed at collection/import.",
    Primary.WRONG_LOCATION: "Patch touches none of the files in the reference patch.",
    Primary.WRONG_FIX: "Patch touches at least one reference file; tests still fail.",
    Primary.UNRESOLVED_OTHER: "Failed after submitting; no reference files available.",
    Primary.UNKNOWN_OUTCOME: "No resolution result recorded.",
}

SECONDARY_FLAGS = {
    "HIGH_INVALID_RATE": ">=20% of turns had a rejected or missing tool call",
    "HAD_TRUNCATION": "at least one truncated tool call",
    "HAD_LOOP": "loop detected anywhere in the run",
    "CONTEXT_PRESSURE": "prompt reached >=90% of context limit or compaction happened",
    "RUNTIME_ERRORS": ">=3 tool results were errors (edit no-match, file not found, timeout)",
    "USED_CODE_GRAPH": "called a code-intelligence tool at least once",
    "NUDGED": "harness injected a continuation nudge at least once",
}
