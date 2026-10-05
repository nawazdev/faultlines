import math
from pathlib import Path

from faultlines.atlas import label_run, read_runs
from faultlines.atlas.report import cohen_kappa, mcnemar_exact, wilson
from faultlines.stress import build_cases, evaluate, sample_calls
from faultlines.toolguard import ToolRegistry, find_loops

REG = ToolRegistry.default_harness()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "traces" / "synthetic_runs.jsonl"
EXPECTED = {
    "synthetic-resolved": "RESOLVED", "synthetic-interface": "INTERFACE", "synthetic-loop": "LOOP",
    "synthetic-context": "CONTEXT", "synthetic-budget": "BUDGET", "synthetic-wrong-location": "WRONG_LOCATION",
    "synthetic-wrong-fix": "WRONG_FIX", "synthetic-broken-patch": "BROKEN_PATCH", "synthetic-no-patch": "NO_PATCH",
    "synthetic-repairable": "RESOLVED",
}


def test_autolabel_examples():
    got = {r.task_id: label_run(r, REG).primary_failure for r in read_runs(EXAMPLES)}
    assert got == EXPECTED


def test_cycle_loop():
    hits = find_loops(["a", "b", "a", "b", "a", "b"])
    assert hits and hits[-1][1].kind == "cycle" and hits[-1][1].period == 2


def test_wilson_known_value():
    lo, hi = wilson(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-3) and math.isclose(hi, 0.7634, abs_tol=1e-3)


def test_kappa_and_mcnemar():
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert math.isclose(cohen_kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]), 0.0)
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(0, 6), 2 / 64)


def test_stress_rules_hold():
    res = evaluate(build_cases(sample_calls(), REG), REG)
    assert res["n"] > 50 and res["precision"] == 1.0 and res["safe_refusal"] == 1.0
