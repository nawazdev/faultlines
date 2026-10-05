"""Build the hand-made example runs in examples/traces/ (synthetic=True).

They exist only to demo and test the pipeline. They are NOT data.
"""
import json
from pathlib import Path

from faultlines.toolguard import render_call

Q = '<|"|>'
def call(name, **args):
    return render_call(name, args)

def step(i, text, name=None, args=None, result="", prompt=2000, events=None):
    return {"index": i, "raw_output": text, "tool_name": name, "tool_args": args, "tool_result": result,
            "prompt_tokens": prompt, "completion_tokens": 200, "events": events or []}

GOLD = "diff --git a/src/dates.py b/src/dates.py\n--- a/src/dates.py\n+++ b/src/dates.py\n@@ -1 +1 @@\n-x\n+y\n"
WRONG = "diff --git a/src/other.py b/src/other.py\n--- a/src/other.py\n+++ b/src/other.py\n@@ -1 +1 @@\n-x\n+y\n"
base = dict(model="synthetic", synthetic=True, context_limit=32768, gold_files=["src/dates.py"])
read = lambda i, p="src/dates.py": step(i, "Let me look. " + call("read_file", filepath=p), "read_file", {"filepath": p}, "1: def parse(...)")
submit = lambda i: step(i, call("submit_patch"), "submit_patch", {}, "submitted")
edit = lambda i, p="src/dates.py": step(i, call("edit_file", filepath=p, old_string="x", new_string="y"), "edit_file",
                                        {"filepath": p, "old_string": "x", "new_string": "y"}, "edited")

runs = [
  dict(task_id="synthetic-resolved", resolved=True, exit_reason="submitted", agent_patch=GOLD,
       steps=[read(0), edit(1), submit(2)]),
  dict(task_id="synthetic-interface", resolved=False, exit_reason="max_turns", agent_patch="",
       steps=[read(0),
              step(1, "I'll rewrite the file. <|tool_call>call:write_file{filepath:" + Q + "src/dates.py" + Q + ",content:" + Q + "import datetime\ndef par"),
              step(2, "Now I will fix it."),
              step(3, "<|tool_call>call:write_file{filepath:" + Q + "src/dates.py" + Q + ",content:" + Q + "import date"),
              step(4, "<|tool_call>call:write_file{filepath:" + Q + "src/dates.py" + Q + ",content:" + Q + "imp")]),
  dict(task_id="synthetic-loop", resolved=False, exit_reason="timeout", agent_patch="",
       steps=[read(0), read(1, "src/a.py")] + [
           step(i, call("edit_file", filepath="src/dates.py", old_string="foo", new_string="bar"), "edit_file",
                {"filepath": "src/dates.py", "old_string": "foo", "new_string": "bar"},
                '{"status": "error", "error_type": "FileEditError", "error_message": "old_string not found in src/dates.py"}')
           for i in range(2, 8)]),
  dict(task_id="synthetic-context", resolved=False, exit_reason="max_turns", agent_patch="",
       steps=[read(i, f"src/m{i}.py") for i in range(4)] + [step(4, call("read_file", filepath="src/big.py"), "read_file",
             {"filepath": "src/big.py"}, "...", prompt=31000, events=["compaction"])]),
  dict(task_id="synthetic-budget", resolved=False, exit_reason="timeout", agent_patch="",
       steps=[read(i, f"src/m{i}.py") for i in range(6)]),
  dict(task_id="synthetic-wrong-location", resolved=False, exit_reason="submitted", agent_patch=WRONG,
       steps=[read(0, "src/other.py"), edit(1, "src/other.py"), submit(2)]),
  dict(task_id="synthetic-wrong-fix", resolved=False, exit_reason="submitted", agent_patch=GOLD,
       test_output="FAILED tests/test_dates.py::test_tz - AssertionError", steps=[read(0), edit(1), submit(2)]),
  dict(task_id="synthetic-broken-patch", resolved=False, exit_reason="submitted", agent_patch=GOLD,
       test_output="E   SyntaxError: invalid syntax\nERROR collecting tests/test_dates.py", steps=[read(0), edit(1), submit(2)]),
  dict(task_id="synthetic-no-patch", resolved=False, exit_reason="submitted", agent_patch="",
       steps=[read(0), submit(1)]),
  dict(task_id="synthetic-repairable", resolved=True, exit_reason="submitted", agent_patch=GOLD,
       steps=[step(0, "<|tool_call>call:cat{file:" + Q + "src/dates.py" + Q + "}<tool_call|>", "read_file", {"filepath": "src/dates.py"}, "1: ..."),
              edit(1), submit(2)]),
]
out = Path(__file__).resolve().parents[1] / "examples" / "traces" / "synthetic_runs.jsonl"
with open(out, "w") as f:
    for r in runs:
        f.write(json.dumps({**base, **r}) + "\n")
print(f"wrote {len(runs)} synthetic runs to {out}")
