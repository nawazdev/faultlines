"""Failure patterns seen in the official-harness run (Gemma 4 31B, 16 dev tasks), as regression tests."""

import json

from faultlines.atlas import label_run
from faultlines.atlas.trace import Run, Step
from faultlines.harness.editing import apply_replacement
from faultlines.toolguard import ToolRegistry
from faultlines.toolguard.repair import is_over_escaped, unescape_args

REG = ToolRegistry.default_harness()
FAIL = json.dumps({"status": "error", "error_type": "FileEditError",
                   "error_message": "Failed to replace: old_string not found."})
OK = json.dumps({"status": "ok"})


def test_over_escaped_edit_is_detected_and_unescaped():
    args = {"filepath": "a.py", "old_string": "import a\\nimport b", "new_string": "import a\\nimport c"}
    assert is_over_escaped(args["old_string"]) and not is_over_escaped("import a\nimport b")
    fixed, changed = unescape_args("edit_file", args)
    assert fixed["old_string"] == "import a\nimport b" and changed == ["old_string", "new_string"]
    res = apply_replacement("import a\nimport b\n", args["old_string"], args["new_string"])
    assert res.strategy == "unescape" and res.content == "import a\nimport c\n"


def test_repeated_escaped_edits_label_interface_not_loop():
    bad = {"filepath": "u.py", "old_string": "x = 1\\ny = 2", "new_string": "x = 1\\ny = 3"}
    steps = [Step(index=i, tool_name="edit_file", tool_args=bad, tool_result=FAIL) for i in range(6)]
    r = label_run(Run(task_id="t", resolved=False, exit_reason="not_submitted", steps=steps), REG)
    assert r.primary_failure == "INTERFACE" and "ESCAPED_STRINGS" in r.secondary


def test_missing_test_dependency_is_env_unverifiable():
    patch = "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1 +1 @@\n-a\n+b\n"
    steps = [Step(index=0, tool_name="edit_file", tool_args={"filepath": "pkg/core.py", "old_string": "a",
                                                                "new_string": "b"}, tool_result=OK),
             Step(index=1, tool_name="submit_patch", tool_args={}, tool_result=OK),
             Step(index=2, raw_output="The issue was fixed by ...")]
    r = Run(task_id="t", resolved=False, exit_reason="submitted", agent_patch=patch, test_exit_code=2,
            test_output="E   ModuleNotFoundError: No module named 'inline_snapshot'", steps=steps)
    r = label_run(r, REG)
    assert r.primary_failure == "ENV_UNVERIFIABLE", r.evidence
    assert r.steps[2].call_status == "final"          # closing summary is not a NO_CALL
    r2 = label_run(Run(task_id="t2", resolved=False, exit_reason="submitted", agent_patch=patch,
                       test_exit_code=124, steps=steps[:2]), REG)
    assert r2.primary_failure == "ENV_UNVERIFIABLE"
