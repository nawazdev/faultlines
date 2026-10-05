"""Competition tasks (tasks.jsonl) and dataset layout.

Dataset (Kaggle: gemma-4-developer-agent, "Data" tab):
    tasks.jsonl                 129 dev tasks (instance_id, repo, base_commit, problem_statement,
                                hints_text, patch, test_patch, created_at)
    snapshots/<id>.tgz          git repo frozen at base_commit, no future history
    graphs/<id>.json            NetworkX node-link call/dependency graph
    embeddings/<id>.npz         256-d vector per graph node
    wheels/                     offline wheels for repo + test dependencies
    sample_submission/          baseline ADK agent config
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    instance_id: str
    repo: str
    base_commit: str = ""
    problem_statement: str = ""
    hints_text: str = ""
    patch: str = ""
    test_patch: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        known = cls.__dataclass_fields__
        return cls(**{k: (v if v is not None else "") for k, v in d.items() if k in known})


def load_tasks(path: str | Path, ids: list[str] | None = None, limit: int | None = None) -> list[Task]:
    tasks = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            tasks.append(Task.from_dict(json.loads(line)))
    if ids:
        wanted = set(ids)
        tasks = [t for t in tasks if t.instance_id in wanted]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def patch_targets(patch: str) -> list[str]:
    """Files a unified diff writes to (+++ b/<path>), excluding /dev/null."""
    out = []
    for m in re.finditer(r"^\+\+\+ (?:b/)?(\S+)", patch or "", re.M):
        if m.group(1) != "/dev/null":
            out.append(m.group(1))
    return sorted(set(out))


@dataclass
class DataLayout:
    root: Path

    @classmethod
    def find(cls, root: str | Path) -> "DataLayout":
        """Accept the dataset root or any parent (Kaggle mounts under /kaggle/input/<slug>/)."""
        root = Path(root)
        for cand in [root, *sorted(root.glob("*")), *sorted(root.glob("*/*"))]:
            if (cand / "tasks.jsonl").exists() and (cand / "snapshots").exists():
                return cls(cand)
        raise FileNotFoundError(f"no tasks.jsonl + snapshots/ under {root}")

    @property
    def tasks(self) -> Path:
        return self.root / "tasks.jsonl"

    def snapshot(self, task_id: str) -> Path:
        return self.root / "snapshots" / f"{task_id}.tgz"

    def graph(self, task_id: str) -> Path:
        return self.root / "graphs" / f"{task_id}.json"

    def embeddings(self, task_id: str) -> Path:
        return self.root / "embeddings" / f"{task_id}.npz"

    @property
    def wheels(self) -> Path:
        return self.root / "wheels"

    @property
    def sample_submission(self) -> Path:
        return self.root / "sample_submission"
