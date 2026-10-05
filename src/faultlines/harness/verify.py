"""Phase 2 verification (HARNESS_README.md section 8).

Fresh workspace from the snapshot -> apply agent patch (fallback passes) ->
reset the files test_patch touches -> apply test_patch -> run pytest on those
files with the documented flags. resolved = pytest exit code 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .sandbox import SubprocessSandbox
from .tasks import DataLayout, Task, patch_targets

APPLY_PASSES = [
    "git apply --unsafe-paths {p}",
    "git apply -3 {p}",
    "git apply --unsafe-paths --ignore-space-change --ignore-whitespace {p}",
    "git apply --unsafe-paths --recount {p}",
    "git apply --unsafe-paths -p0 {p}",
    "patch -p1 --batch --forward --dry-run < {p} && patch -p1 --batch --forward < {p}",
    "patch -p0 --batch --forward --dry-run < {p} && patch -p0 --batch --forward < {p}",
]

PYTEST = ('PYTHONSAFEPATH=1 python3 -m pytest {targets} -p no:anyio -o timeout=0 '
          '-o norecursedirs=".* build dist venv" -o python_classes="Test* *Test" -q')


@dataclass
class VerifyResult:
    resolved: bool
    patch_applied: bool
    output: str
    seconds: float
    apply_pass: str = ""


def verify_task(task: Task, layout: DataLayout, patch: str, venv_path: Path, python: str = "python3",
                allow_online_fallback: bool = False, pytest_timeout: int = 1200) -> VerifyResult:
    t0 = time.time()
    sb = SubprocessSandbox(layout.snapshot(task.instance_id), layout.wheels if layout.wheels.exists() else None,
                           python=python, allow_online_fallback=allow_online_fallback, venv_path=venv_path)
    try:
        sb.setup(reuse_venv=venv_path)
        pf = sb.root / "agent.patch"
        pf.write_text(patch if patch.endswith("\n") else patch + "\n")
        used = ""
        for cmd in APPLY_PASSES:
            r = sb.run(cmd.format(p=pf), timeout=120, rewrite=False)
            if r.exit_code == 0:
                used = cmd.split(" {p}")[0].split(" <")[0]
                break
            sb._git("checkout", "--", ".", check=False)
        if not used:
            return VerifyResult(False, False, "Failed to apply agent_patch with all passes", time.time() - t0)
        targets = patch_targets(task.test_patch)
        if targets:
            sb._git("checkout", "HEAD", "--", *targets, check=False)
            sb._git("clean", "-f", "--", *targets, check=False)
        tp = sb.root / "test.patch"
        tp.write_text(task.test_patch if task.test_patch.endswith("\n") else task.test_patch + "\n")
        r = sb.run(f"git apply --unsafe-paths {tp} || patch -p1 --batch --forward < {tp}", timeout=120, rewrite=False)
        if r.exit_code != 0:
            return VerifyResult(False, True, "Failed to apply test_patch: " + r.stderr[-2000:], time.time() - t0, used)
        r = sb.run(PYTEST.format(targets=" ".join(targets) or "."), timeout=pytest_timeout, rewrite=False)
        out = (r.stdout + "\n" + r.stderr)[-20000:]
        if r.timed_out:
            out = "PYTEST TIMEOUT\n" + out
        return VerifyResult(r.exit_code == 0, True, out, time.time() - t0, used)
    finally:
        sb.cleanup()
