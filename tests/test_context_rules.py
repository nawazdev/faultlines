"""Regression tests from E3: context-window crashes and inferred compactions (labeler v0.3)."""
from faultlines.atlas import label_run
from faultlines.atlas.adapters import infer_compactions
from faultlines.atlas.trace import Run, Step
from faultlines.toolguard import ToolRegistry

REG = ToolRegistry.default_harness()
CRASH = ("Sandbox execution error: litellm.ContextWindowExceededError: litellm.BadRequestError: "
         "This model's maximum context length is 32768 tokens. However, you requested 16384 output tokens "
         "and your prompt contains at least 16385 input tokens")


def _run(prompts, harness_error=""):
    steps = [Step(index=i, tool_name="read_file", tool_args={"filepath": f"src/f{i}.py"},
                  tool_result='{"status": "ok", "content": "x"}', prompt_tokens=p, completion_tokens=50)
             for i, p in enumerate(prompts)]
    return Run(task_id="t", resolved=False, exit_reason="not_submitted", steps=steps,
               context_limit=32768, harness_error=harness_error)


def test_context_crash_is_context():
    r = label_run(_run([4000, 8000, 12000, 14000], CRASH), REG)
    assert r.primary_failure == "CONTEXT"
    assert "CONTEXT_PRESSURE" in r.secondary


def test_compaction_alone_is_not_context():
    r = _run([4000, 9000, 14600, 5300, 9000, 12000])
    assert infer_compactions(r.steps) == 1
    assert r.steps[3].events == ["compaction:inferred"]
    r = label_run(r, REG)
    assert r.primary_failure == "BUDGET"
    assert "COMPACTED" in r.secondary


def test_infer_compactions_ignores_small_prompts_and_is_idempotent():
    r = _run([3000, 1000, 9000, 8800])
    assert infer_compactions(r.steps) == 0
    r = _run([10000, 4000])
    assert infer_compactions(r.steps) == 1 and infer_compactions(r.steps) == 0
