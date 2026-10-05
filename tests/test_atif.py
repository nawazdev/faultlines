"""ATIF v1.7 adapter on a spec-shaped trajectory (harbor RFC 0001 field names)."""

import json

from faultlines.atlas import label_run
from faultlines.atlas.adapters import atif_to_steps, load_swegemma_results, trace_shape
from faultlines.toolguard import ToolRegistry

TRACE = {
    "schema_version": "ATIF-v1.7", "session_id": "s1", "agent": {"name": "swegemma", "version": "1", "model_name": "gemma"},
    "steps": [
        {"step_id": 1, "source": "system", "message": "You are an engineer."},
        {"step_id": 2, "source": "user", "message": "Fix the bug."},
        {"step_id": 3, "source": "agent", "message": "", "reasoning_content": "look first",
         "tool_calls": [{"tool_call_id": "c1", "function_name": "read_file", "arguments": {"filepath": "a.py"}},
                        {"tool_call_id": "c2", "function_name": "get_status", "arguments": {}}],
         "observation": {"results": [{"source_call_id": "c2", "content": "{\"status\": \"ok\"}"},
                                     {"source_call_id": "c1", "content": "{\"status\": \"ok\", \"content\": \"x\"}"}]},
         "metrics": {"prompt_tokens": 900, "completion_tokens": 40, "completion_token_ids": [1, 2, 3]}},
        {"step_id": 4, "source": "agent", "message": [{"type": "text", "text": "editing"}],
         "tool_calls": [{"tool_call_id": "c3", "function_name": "edit_file",
                         "arguments": {"filepath": "a.py", "old_string": "zz", "new_string": "y"}}],
         "observation": {"results": [{"source_call_id": "c3", "content":
                         "{\"status\": \"error\", \"error_type\": \"FileEditError\", \"error_message\": \"old_string not found\"}"}]}},
        {"step_id": 5, "source": "agent", "message": "I think I'm done."},
        {"step_id": 6, "source": "user", "message": "Please continue your work using the available tools, or call "
                                                    "submit_patch when you have completed and verified your changes."},
        {"step_id": 7, "source": "agent", "message": "", "extra": {"context_management": {"type": "compaction"}},
         "tool_calls": [{"tool_call_id": "c4", "function_name": "submit_patch", "arguments": {}}],
         "observation": {"results": [{"source_call_id": "c4", "content": "{\"status\": \"ok\"}"}]}},
    ],
}


def test_atif_mapping():
    steps = atif_to_steps(TRACE)
    assert [s.tool_name for s in steps] == ["read_file", "edit_file", None, "submit_patch"]
    assert json.loads(steps[0].tool_result)["content"] == "x"          # matched by source_call_id, not order
    assert steps[0].extra_calls[0]["name"] == "get_status"
    assert steps[0].prompt_tokens == 900 and steps[0].raw_output.startswith("look first")
    assert steps[1].raw_output == "editing"
    assert "nudge" in steps[2].events and "nudge_continue" in steps[2].events
    assert "compaction" in steps[3].events
    assert atif_to_steps(TRACE, decode=lambda ids: "<decoded>")[0].raw_output == "<decoded>"
    assert "tool_calls: list[2]" in trace_shape(TRACE)


def test_results_dir_without_task_results(tmp_path):
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "trace_demo_1.json").write_text(json.dumps(TRACE))
    (tmp_path / "traces" / "trace_demo_11.json").write_text(json.dumps({"steps": []}))
    runs = {r.task_id: r for r in load_swegemma_results(tmp_path)}
    assert len(runs["demo_1"].steps) == 4 and runs["demo_11"].steps == []
    r = label_run(runs["demo_1"], ToolRegistry.default_harness())
    assert r.steps[1].runtime_error == "EDIT_NO_MATCH"
    assert r.steps[2].call_status == "no_call"
