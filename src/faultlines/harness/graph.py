"""Code-intelligence tools over the competition's graphs/ and embeddings/ files.

Graph: NetworkX node-link JSON (nodes[*].id/name/text, edges[*].source/target/type).
Embeddings: .npz with one 256-d float32 vector per node id.
search_similar_code resolves the query to a node key (no live embedding server,
as in the official harness) and ranks all nodes by cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CodeGraph:
    def __init__(self, graph_path: Path | None, emb_path: Path | None):
        self.nodes: dict[str, dict] = {}
        self.out_edges: dict[str, list[tuple[str, str]]] = {}
        self.in_edges: dict[str, list[tuple[str, str]]] = {}
        self.vectors = None
        self.vec_keys: list[str] = []
        if graph_path and Path(graph_path).exists() and Path(graph_path).stat().st_size > 100:
            g = json.loads(Path(graph_path).read_text())
            for n in g.get("nodes", []):
                self.nodes[str(n["id"])] = n
            for e in g.get("edges", g.get("links", [])):
                s, t, ty = str(e["source"]), str(e["target"]), str(e.get("type", "RELATED"))
                self.out_edges.setdefault(s, []).append((t, ty))
                self.in_edges.setdefault(t, []).append((s, ty))
        if emb_path and Path(emb_path).exists() and Path(emb_path).stat().st_size > 100:
            import numpy as np
            z = np.load(emb_path)
            self.vec_keys = [k[:-4] if k.endswith(".npy") else k for k in z.files]
            mat = np.stack([z[k] for k in z.files]).astype("float32")
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self.vectors = mat / np.maximum(norms, 1e-8)

    @property
    def available(self) -> bool:
        return bool(self.nodes) or self.vectors is not None

    # 4-tier resolution: exact -> suffix (after . or /) -> case-insensitive -> substring
    @staticmethod
    def _resolve(name: str, keys: list[str]) -> str | None:
        if not name:
            return None
        if name in keys:
            return name
        suf = [k for k in keys if k.endswith("." + name) or k.endswith("/" + name)]
        if suf:
            return min(suf, key=len)
        low = name.lower()
        ci = [k for k in keys if k.lower() == low]
        if ci:
            return ci[0]
        sub = [k for k in keys if low in k.lower()]
        return min(sub, key=len) if sub else None

    def resolve(self, name: str) -> str | None:
        return self._resolve(name, list(self.nodes))

    def neighbors(self, node: str, edge_type: str | None = None, max_neighbors: int = 50) -> dict[str, Any]:
        rid = self.resolve(node)
        if rid is None:
            return {"status": "error", "error_type": "NodeNotFound", "error_message": f"No graph node matches '{node}'"}
        et = edge_type.lower() if edge_type else None
        items = [{"node": t, "direction": "out", "type": ty} for t, ty in self.out_edges.get(rid, [])]
        items += [{"node": s, "direction": "in", "type": ty} for s, ty in self.in_edges.get(rid, [])]
        if et:
            items = [i for i in items if i["type"].lower() == et]
        items = items[: max(1, int(max_neighbors))]
        return {"status": "ok", "node": rid, "neighbors": items, "count": len(items)}

    def similar(self, query: str, k: int = 10) -> dict[str, Any]:
        if self.vectors is None:
            return {"status": "error", "error_type": "NoEmbeddings", "error_message": "No embeddings for this repo"}
        key = self._resolve(query, self.vec_keys)
        if key is None:
            return {"status": "error", "error_type": "NodeNotFound",
                    "error_message": f"'{query}' matches no embedded symbol; pass a class/function name"}
        i = self.vec_keys.index(key)
        sims = self.vectors @ self.vectors[i]
        order = [j for j in sims.argsort()[::-1] if j != i][: max(1, int(k))]
        res = [{"node_name": self.vec_keys[j], "code": str(self.nodes.get(self.vec_keys[j], {}).get("text", ""))[:1500],
                "similarity": round(float(sims[j]), 4)} for j in order]
        return {"status": "ok", "query": key, "results": res, "count": len(res)}

    def subgraph(self, nodes: list[str]) -> dict[str, Any]:
        ids = [r for r in (self.resolve(str(n)) for n in (nodes or [])) if r]
        idset = set(ids)
        edges = [{"from": s, "to": t, "type": ty} for s in ids for t, ty in self.out_edges.get(s, []) if t in idset]
        return {"status": "ok", "nodes": ids, "edges": edges, "node_count": len(ids), "edge_count": len(edges)}
