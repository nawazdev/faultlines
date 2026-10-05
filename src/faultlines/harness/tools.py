"""The 9 swegemma tools, reproduced from HARNESS_README.md section 6.

Every tool returns a JSON string: {"status": "ok", ...} or
{"status": "error", "error_type": ..., "error_message": ..., "details": {...}}.
get_status returns a raw object and, like submit_patch, does not count as a tool call.
"""

from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .editing import EditError, apply_replacement
from .graph import CodeGraph
from .sandbox import SubprocessSandbox


@dataclass
class Limits:
    time_minutes: float = 60.0
    tool_calls: int | None = 50
    turns: int = 500
    command_timeout: int = 300
    max_stdout_chars: int = 5000
    max_file_lines: int = 150
    max_file_chars: int = 10000


def _err(error_type: str, message: str, details: dict | None = None) -> dict:
    d = {"status": "error", "error_type": error_type, "error_message": message}
    if details:
        d["details"] = details
    return d


@dataclass
class ToolContext:
    sandbox: SubprocessSandbox
    limits: Limits
    graph: CodeGraph | None = None
    tool_calls_used: int = 0
    patch_submitted: bool = False
    submitted_patch: str = ""
    agent_start: float = field(default_factory=time.time)

    # ------------------------------------------------------------ budgeting
    def remaining_seconds(self) -> float:
        return self.limits.time_minutes * 60 - (time.time() - self.agent_start)

    def remaining_calls(self) -> int | None:
        return None if self.limits.tool_calls is None else self.limits.tool_calls - self.tool_calls_used

    def _gate(self, count: bool) -> dict | None:
        if self.remaining_seconds() <= 0:
            return _err("BudgetExceeded", "Session time budget exhausted")
        if count:
            rem = self.remaining_calls()
            if rem is not None and rem <= 0:
                return _err("BudgetExceeded", "Tool call budget exhausted; call submit_patch")
            self.tool_calls_used += 1
        return None

    def _finish(self, d: dict) -> str:
        total, rem = self.limits.tool_calls, self.remaining_calls()
        if total is not None and total >= 20 and rem is not None and rem <= 10:
            d["budget_warning"] = (f"Only {rem} tool call(s) remaining ({self.tool_calls_used}/{total} used). "
                                   "Finalize your edits and call submit_patch soon.")
        return json.dumps(d)

    def _path(self, filepath: str) -> Path:
        p = str(filepath or "").strip()
        for prefix in ("/workspace/", "/workspace"):
            if p.startswith(prefix):
                p = p[len(prefix):]
        p = p.lstrip("/")
        if ".." in Path(p).parts:
            raise ValueError("path traversal ('..') is not allowed")
        return self.sandbox.workspace / p

    # ---------------------------------------------------------------- tools
    def run_command(self, command: str) -> str:
        g = self._gate(True)
        if g:
            return json.dumps(g)
        timeout = min(self.limits.command_timeout, max(5, int(self.remaining_seconds())))
        r = self.sandbox.run(command, timeout=timeout)
        cap = self.limits.max_stdout_chars
        out, err = r.stdout[:cap], r.stderr[:cap]
        if r.timed_out:
            return self._finish(_err("TimeoutExceeded", f"Command exceeded {timeout}s timeout",
                                     {"stdout": out, "stderr": err}))
        if r.exit_code == 0:
            return self._finish({"status": "ok", "stdout": out + (("\n" + err) if err else ""), "exit_code": 0})
        return self._finish(_err("CommandError", f"Command failed with exit code {r.exit_code}",
                                 {"stdout": out, "stderr": err, "exit_code": r.exit_code}))

    def submit_patch(self) -> str:
        g = self._gate(False)
        if g:
            return json.dumps(g)
        patch = self.sandbox.diff()
        self.submitted_patch, self.patch_submitted = patch, True
        files = len({l[6:] for l in patch.splitlines() if l.startswith("+++ b/")})
        return json.dumps({"status": "ok", "patch_size": len(patch), "files_changed": files})

    def get_status(self) -> str:
        return json.dumps({
            "tool_calls_used": self.tool_calls_used, "patch_submitted": self.patch_submitted,
            "patch_size": len(self.submitted_patch), "tool_calls_remaining": self.remaining_calls(),
            "max_tool_calls": self.limits.tool_calls,
            "time_seconds_remaining": round(self.remaining_seconds(), 1),
            "max_time_minutes": self.limits.time_minutes,
            "agent_elapsed_seconds": round(time.time() - self.agent_start, 1),
            "max_turns": self.limits.turns, "command_timeout_seconds": self.limits.command_timeout,
        })

    def read_file(self, filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
        g = self._gate(True)
        if g:
            return json.dumps(g)
        try:
            p = self._path(filepath)
        except ValueError as e:
            return self._finish(_err("ValidationError", str(e)))
        if not p.is_file():
            return self._finish(_err("FileNotFoundError", f"File does not exist: {filepath}"))
        lines = p.read_text(errors="replace").split("\n")
        total = len(lines)
        s = max(1, int(start_line or 1))
        e = min(total, int(end_line) if end_line else total)
        e = min(e, s + self.limits.max_file_lines - 1)
        chunk, used, last = [], 0, s - 1
        for i in range(s - 1, e):
            add = len(lines[i]) + 1
            if used + add > self.limits.max_file_chars:
                break
            chunk.append(lines[i]); used += add; last = i + 1
        truncated = last < (min(total, int(end_line)) if end_line else total)
        return self._finish({"status": "ok", "filepath": str(filepath), "content": "\n".join(chunk),
                             "start_line": s, "end_line": last, "total_lines": total, "is_truncated": truncated})

    def edit_file(self, filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str:
        g = self._gate(True)
        if g:
            return json.dumps(g)
        try:
            p = self._path(filepath)
        except ValueError as e:
            return self._finish(_err("ValidationError", str(e)))
        if not p.is_file():
            return self._finish(_err("FileEditError", f"File does not exist: {filepath}"))
        before = p.read_text(errors="replace")
        if not before:
            return self._finish(_err("FileEditError", f"File is empty: {filepath}"))
        try:
            res = apply_replacement(before, old_string, new_string, bool(allow_multiple))
        except EditError as e:
            return self._finish(_err("FileEditError", f"{e} in {filepath}"))
        p.write_text(res.content)
        diff = "".join(difflib.unified_diff(before.replace("\r\n", "\n").splitlines(True), res.content.splitlines(True),
                                            f"a/{filepath}", f"b/{filepath}"))
        cap = self.limits.max_stdout_chars
        return self._finish({"status": "ok", "filepath": str(filepath), "occurrences": res.occurrences,
                             "strategy": res.strategy, "diff": diff[:cap], "is_truncated": len(diff) > cap})

    def write_file(self, filepath: str, content: str) -> str:
        g = self._gate(True)
        if g:
            return json.dumps(g)
        try:
            p = self._path(filepath)
        except ValueError as e:
            return self._finish(_err("ValidationError", str(e)))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if isinstance(content, str) else json.dumps(content))
        return self._finish({"status": "ok", "filepath": str(filepath), "size": p.stat().st_size})

    def get_code_neighbors(self, node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str:
        g = self._gate(True)
        if g:
            return json.dumps(g)
        if not self.graph or not self.graph.nodes:
            return self._finish(_err("NoGraph", "No code graph available for this repository"))
        return self._finish(self.graph.neighbors(node, edge_type, max_neighbors))

    def search_similar_code(self, query: str, k: int = 10) -> str:
        g = self._gate(True)
        if g:
            return json.dumps(g)
        if not self.graph:
            return self._finish(_err("NoEmbeddings", "No embeddings available for this repository"))
        return self._finish(self.graph.similar(query, k))

    def get_code_subgraph(self, nodes: list[str]) -> str:
        g = self._gate(True)
        if g:
            return json.dumps(g)
        if not self.graph or not self.graph.nodes:
            return self._finish(_err("NoGraph", "No code graph available for this repository"))
        if isinstance(nodes, str):
            nodes = [nodes]
        return self._finish(self.graph.subgraph(nodes))

    # ------------------------------------------------------------- dispatch
    def registry(self) -> dict[str, Callable[..., str]]:
        names = ["run_command", "submit_patch", "get_status", "read_file", "edit_file", "write_file"]
        if self.graph is not None and self.graph.available:
            names += ["get_code_neighbors", "search_similar_code", "get_code_subgraph"]
        return {n: getattr(self, n) for n in names}

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Execute like ADK would: unknown tool or bad arguments come back as an error result."""
        fn = self.registry().get(name)
        if fn is None:
            return json.dumps(_err("ToolNotFound", f"Function {name} is not found in the tools_dict."))
        try:
            return fn(**args)
        except TypeError as e:
            return json.dumps(_err("InvalidArguments", f"{name}: {e}"))
        except Exception as e:  # tool bug should not kill the session
            return json.dumps(_err(type(e).__name__, str(e)[:500]))
