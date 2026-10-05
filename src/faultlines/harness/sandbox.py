"""Subprocess sandbox modeled on swegemma's `--sandbox subprocess` backend.

Per task: <tmp>/workspace (repo snapshot), <tmp>/tmp, and a venv with the repo
installed in editable mode from the offline wheels. Commands run in bash with
/workspace and /tmp rewritten to the sandbox paths, in their own process group,
killed as a group on timeout.

Documented behaviour reproduced: snapshot extraction, .git/info/exclude
patterns, offline editable install, baseline commit, git add -N . && git diff HEAD.
Not reproduced: Docker isolation, 4 GiB / 2 vCPU limits, the harness's own
pytest.ini/conftest.py (we pass the documented pytest flags on the command line).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

GIT_EXCLUDE = ["__pycache__/", "*.pyc", ".pytest_cache/", "*.egg-info/", "build/", "dist/", ".coverage"]
BUILD_BACKENDS = ["setuptools", "wheel", "hatchling", "hatch-vcs", "flit_core", "pdm-backend", "poetry-core",
                  "setuptools_scm", "pytest"]
GIT = ["git", "-c", "user.email=faultlines@local", "-c", "user.name=faultlines", "-c", "commit.gpgsign=false"]


@dataclass
class CmdResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    seconds: float = 0.0


@dataclass
class SubprocessSandbox:
    snapshot: Path
    wheels: Path | None = None
    python: str = "python3"               # interpreter used to build the venv (e.g. a uv-managed 3.12)
    allow_online_fallback: bool = False   # retry pip online if offline install fails (recorded in notes)
    system_site_packages: bool = False    # for tests only
    keep: bool = False
    venv_path: Path | None = None         # where to build the venv (shared by the agent and verify phases)
    root: Path = field(init=False)
    venv: Path | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="flsbx_"))
        (self.root / "tmp").mkdir()

    # ------------------------------------------------------------------ paths
    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    def _env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith("VIRTUAL_ENV")}
        env["TMPDIR"] = str(self.root / "tmp")
        env["TEST_TMPDIR"] = str(self.root / "tmp")
        env["PIP_NO_INPUT"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"
        if self.venv:
            env["VIRTUAL_ENV"] = str(self.venv)
            env["PATH"] = f"{self.venv / 'bin'}:{env.get('PATH', '')}"
        return env

    def rewrite(self, command: str) -> str:
        """Map the agent's absolute paths onto the sandbox, as the official subprocess backend does."""
        # Placeholders first, so a sandbox path that itself lives under /tmp is not rewritten twice.
        s = (command.replace("/workspace", "\x00WS\x00").replace("/tmp/", "\x00TMP\x00")
                    .replace("/wheels", "\x00WH\x00"))
        return (s.replace("\x00WS\x00", str(self.workspace))
                 .replace("\x00TMP\x00", str(self.root / "tmp") + "/")
                 .replace("\x00WH\x00", str(self.wheels or "/wheels")))

    # ---------------------------------------------------------------- running
    def run(self, command: str, timeout: float = 300, cwd: Path | None = None, rewrite: bool = True) -> CmdResult:
        cmd = self.rewrite(command) if rewrite else command
        t0 = time.time()
        proc = subprocess.Popen(["/bin/bash", "-c", cmd], cwd=str(cwd or self.workspace), env=self._env(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
            return CmdResult(proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"),
                             False, time.time() - t0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
            return CmdResult(-9, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), True,
                             time.time() - t0)

    def _git(self, *args: str, cwd: Path | None = None, check: bool = True) -> CmdResult:
        p = subprocess.run([*GIT, *args], cwd=str(cwd or self.workspace), capture_output=True, text=True)
        if check and p.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr[-500:]}")
        return CmdResult(p.returncode, p.stdout, p.stderr)

    # ------------------------------------------------------------------ setup
    def extract(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir()
        with tarfile.open(self.snapshot) as tf:
            try:
                tf.extractall(self.workspace, filter="tar")
            except TypeError:  # Python < 3.12
                tf.extractall(self.workspace)
        # Archives may wrap the repo in one top-level folder: flatten it.
        entries = [p for p in self.workspace.iterdir()]
        if len(entries) == 1 and entries[0].is_dir() and (entries[0] / ".git").exists():
            inner = entries[0]
            for p in inner.iterdir():
                shutil.move(str(p), str(self.workspace / p.name))
            inner.rmdir()
        if not (self.workspace / ".git").exists():
            self._git("init", "-q")
        exclude = self.workspace / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with open(exclude, "a") as f:
            f.write("\n" + "\n".join(GIT_EXCLUDE) + "\n")

    def _pip(self, args: str, online: bool = False) -> CmdResult:
        find = "" if online or not self.wheels else f"--no-index --find-links {self.wheels}"
        return self.run(f"python -m pip install -q {find} {args}", timeout=900, rewrite=False)

    def create_venv(self) -> None:
        self.venv = Path(self.venv_path) if self.venv_path else self.root / "venv"
        flag = "--system-site-packages" if self.system_site_packages else ""
        r = subprocess.run(f"{self.python} -m venv {flag} {self.venv}", shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"venv creation failed: {r.stderr[-500:]}")
        for pkg in BUILD_BACKENDS:  # best effort; not every backend is in the wheels dir
            self._pip(pkg)

    def install_repo(self) -> None:
        """Offline editable install with dependencies, then any requirements*.txt, best effort."""
        r = self._pip("-e .")
        if r.exit_code != 0:
            r = self._pip("--no-build-isolation -e .")
        if r.exit_code != 0 and self.allow_online_fallback:
            self.notes.append("online_pip_fallback")
            r = self._pip("-e .", online=True)
        if r.exit_code != 0:
            self.notes.append("editable_install_failed")
        for req in sorted(self.workspace.glob("requirements*.txt")) + sorted(self.workspace.glob("requirements/*.txt")):
            rr = self._pip(f"-r {req}")
            if rr.exit_code != 0 and self.allow_online_fallback:
                rr = self._pip(f"-r {req}", online=True)
            if rr.exit_code != 0:
                self.notes.append(f"requirements_partial:{req.name}")

    def baseline_commit(self, message: str = "baseline") -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "--no-verify", "-m", message)

    def setup(self, reuse_venv: Path | None = None) -> float:
        """Full bootstrap. Returns seconds taken (excluded from the agent's time budget)."""
        t0 = time.time()
        self.extract()
        if reuse_venv is not None and Path(reuse_venv, "bin", "python").exists():
            self.venv = Path(reuse_venv)
        else:
            self.create_venv()
        self.install_repo()
        self.baseline_commit()
        return time.time() - t0

    # ------------------------------------------------------------------ patch
    def diff(self) -> str:
        self._git("add", "-N", ".", check=False)
        return self._git("diff", "HEAD", check=False).stdout

    def cleanup(self) -> None:
        if not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)
