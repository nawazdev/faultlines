"""End-to-end harness tests on a tiny fake task (real git, venv, pytest; scripted model)."""

import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from faultlines.atlas import label_run
from faultlines.harness import DataLayout, Limits, RunConfig, ScriptedModel, load_tasks, run_task
from faultlines.harness.editing import EditError, apply_replacement
from faultlines.toolguard import ToolRegistry, render_call

TEST_PATCH = """diff --git a/tests/test_calc.py b/tests/test_calc.py
new file mode 100644
--- /dev/null
+++ b/tests/test_calc.py
@@ -0,0 +1,5 @@
+from calc import add
+
+
+def test_add():
+    assert add(2, 3) == 5
"""
GOLD_PATCH = """diff --git a/calc/__init__.py b/calc/__init__.py
--- a/calc/__init__.py
+++ b/calc/__init__.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    repo = tmp_path_factory.mktemp("repo")
    (repo / "calc").mkdir()
    (repo / "calc" / "__init__.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n'
                                         '[project]\nname = "calc"\nversion = "0.1"\n'
                                         '[tool.setuptools]\npackages = ["calc"]\n')
    g = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run([*g, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*g, "commit", "-qm", "init"], cwd=repo, check=True)
    (root / "snapshots").mkdir()
    with tarfile.open(root / "snapshots" / "calc_1.tgz", "w:gz") as tf:
        tf.add(repo, arcname=".")
    (root / "wheels").mkdir()
    (root / "graphs").mkdir(); (root / "embeddings").mkdir()
    graph = {"directed": True, "multigraph": True, "graph": {},
             "nodes": [{"id": "calc", "name": "calc", "text": ""},
                       {"id": "calc.add", "name": "calc.add", "text": "def add(a, b):\n    return a - b\n"},
                       {"id": "calc.sub", "name": "calc.sub", "text": "def sub(a, b): ..."}],
             "edges": [{"source": "calc", "target": "calc.add", "type": "DEFINES", "key": 0}]}
    (root / "graphs" / "calc_1.json").write_text(json.dumps(graph))
    np.savez(root / "embeddings" / "calc_1.npz", **{"calc.add": np.ones(256, "float32"),
                                                    "calc.sub": np.ones(256, "float32") * 0.5,
                                                    "calc": np.arange(256, dtype="float32")})
    task = {"instance_id": "calc_1", "repo": "demo/calc", "base_commit": "x", "problem_statement":
            "add(2, 3) returns -1 instead of 5.", "hints_text": "", "patch": GOLD_PATCH, "test_patch": TEST_PATCH}
    (root / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    return DataLayout.find(root)


def cfg(mode="strict", **kw):
    return RunConfig(limits=Limits(time_minutes=5, tool_calls=20), mode=mode, system_site_packages=True,
                     compaction_tokens=None, **kw)


def test_resolves_with_real_verification(dataset):
    task = load_tasks(dataset.tasks)[0]
    model = ScriptedModel([
        "Let me look. " + render_call("read_file", {"filepath": "calc/__init__.py"}),
        render_call("search_similar_code", {"query": "add", "k": 2}),
        render_call("edit_file", {"filepath": "calc/__init__.py", "old_string": "return a - b", "new_string": "return a + b"}),
        render_call("run_command", {"command": "cd /workspace && python3 -c 'from calc import add; assert add(2,3)==5'"}),
        render_call("submit_patch", {}),
    ])
    run = run_task(task, dataset, model, cfg())
    assert run.exit_reason == "submitted", run.evidence
    assert run.resolved is True, run.test_output
    assert "return a + b" in run.agent_patch
    assert json.loads(run.steps[0].tool_result)["status"] == "ok"
    sim = json.loads(run.steps[1].tool_result)
    assert sim["status"] == "ok" and sim["results"][0]["node_name"] == "calc.sub"
    assert json.loads(run.steps[3].tool_result)["status"] == "ok"
    labeled = label_run(run, ToolRegistry.default_harness())
    assert labeled.primary_failure == "RESOLVED" and "USED_CODE_GRAPH" in labeled.secondary


def test_truncated_calls_end_in_nudges_and_interface_label(dataset):
    task = load_tasks(dataset.tasks)[0]
    cut = '<|tool_call>call:write_file{filepath:<|"|>calc/__init__.py<|"|>,content:<|"|>def add(a, b):\n    ret'
    model = ScriptedModel([cut, cut, cut, cut, cut])
    run = run_task(task, dataset, model, cfg())
    assert run.exit_reason == "max_nudges"
    assert run.resolved is False and run.agent_patch == ""
    assert sum("nudge" in s.events for s in run.steps) == 3
    labeled = label_run(run, ToolRegistry.default_harness())
    assert labeled.primary_failure == "INTERFACE"


def test_guard_mode_executes_repairable_call_strict_does_not(dataset):
    task = load_tasks(dataset.tasks)[0]
    aliased = '<|tool_call>call:bash{cmd:<|"|>ls<|"|>}<tool_call|>'
    strict = run_task(task, dataset, ScriptedModel([aliased, render_call("submit_patch", {})]), cfg("strict"))
    assert strict.steps[0].tool_name == "bash"          # parsed, but unknown to the harness
    assert json.loads(strict.steps[0].tool_result)["error_type"] == "ToolNotFound"
    guarded = run_task(task, dataset, ScriptedModel([aliased, render_call("submit_patch", {})]), cfg("guard"))
    assert guarded.steps[0].tool_name == "run_command"
    assert json.loads(guarded.steps[0].tool_result)["status"] == "ok"
    assert any(e.startswith("guard_repaired") for e in guarded.steps[0].events)


def test_wrong_fix_is_verified_as_unresolved(dataset):
    task = load_tasks(dataset.tasks)[0]
    model = ScriptedModel([
        render_call("edit_file", {"filepath": "calc/__init__.py", "old_string": "return a - b", "new_string": "return a * b"}),
        render_call("submit_patch", {}),
    ])
    run = run_task(task, dataset, model, cfg())
    assert run.resolved is False and run.patch_applied is True
    assert label_run(run, ToolRegistry.default_harness()).primary_failure == "WRONG_FIX"


def test_edit_tiers():
    src = "def f():\n    if x:\n        return 1\n"
    assert apply_replacement(src, "if x:\n    return 1", "if x:\n    return 2").strategy == "flexible"
    assert "return 2" in apply_replacement(src, "if x:\n    return 1", "if x:\n    return 2").content
    assert apply_replacement("f(a,b)", "f( a , b )", "g()").strategy == "regex"
    with pytest.raises(EditError):
        apply_replacement("a a", "a", "b")
