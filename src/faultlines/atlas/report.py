"""Statistics and markdown tables for the paper (RQ1, RQ2, RQ4, label agreement)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from .taxonomy import group_of
from .trace import Run


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(pairs: Iterable[tuple[str, str]]) -> float:
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = only A solved, c = only B solved."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def summarize(runs: list[Run]) -> str:
    by_model: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_model[r.model or "?"].append(r)
    out = []
    if any(r.synthetic for r in runs):
        out.append("> WARNING: includes synthetic example runs. Do not report these numbers.\n")

    out.append("### Resolution rate (RQ4 baseline)\n")
    out.append("| Model | Runs (scored) | Resolved | Rate | 95% CI |")
    out.append("| --- | --- | --- | --- | --- |")
    for m, rs in sorted(by_model.items()):
        scored = [r for r in rs if group_of(r.primary_failure or "") != "excluded"]
        k = sum(r.primary_failure == "RESOLVED" for r in scored)
        lo, hi = wilson(k, len(scored))
        rate = k / len(scored) if scored else 0
        out.append(f"| {m} | {len(scored)} | {k} | {_pct(rate)} | {_pct(lo)}–{_pct(hi)} |")

    out.append("\n### Primary failure cause among failed runs (RQ2)\n")
    labels = sorted({r.primary_failure for r in runs if r.primary_failure})
    out.append("| Model | " + " | ".join(labels) + " | interface share |")
    out.append("| --- | " + " | ".join("---" for _ in labels) + " | --- |")
    for m, rs in sorted(by_model.items()):
        failed = [r for r in rs if group_of(r.primary_failure or "") in ("interface", "budget", "reasoning")]
        c = Counter(r.primary_failure for r in rs)
        inter = sum(group_of(r.primary_failure) == "interface" for r in failed)
        lo, hi = wilson(inter, len(failed))
        share = f"{_pct(inter / len(failed))} ({_pct(lo)}–{_pct(hi)})" if failed else "–"
        out.append(f"| {m} | " + " | ".join(str(c.get(l, 0)) for l in labels) + f" | {share} |")

    out.append("\n### Tool-call problems per 100 turns (RQ1)\n")
    codes = sorted({i for r in runs for s in r.steps for i in s.call_issues})
    mid = (" | ".join(codes) + " | ") if codes else ""
    out.append("| Model | Turns | " + mid + "any problem |")
    out.append("| --- | --- | " + ("".join("--- | " for _ in codes)) + "--- |")
    for m, rs in sorted(by_model.items()):
        steps = [s for r in rs for s in r.steps]
        n = len(steps) or 1
        c = Counter(i for s in steps for i in s.call_issues)
        anyp = sum(bool(s.call_issues) for s in steps)
        vals = "".join(f"{100 * c.get(k, 0) / n:.1f} | " for k in codes)
        out.append(f"| {m} | {len(steps)} | " + vals + f"{100 * anyp / n:.1f} |")
    return "\n".join(out)


def replay(runs: list[Run]) -> str:
    """RQ3 on real data: what the guard would have done to each original turn."""
    status = Counter(s.call_status for r in runs for s in r.steps)
    total = sum(status.values()) or 1
    lines = ["| Guard verdict on original turn | Turns | Share |", "| --- | --- | --- |"]
    for k in ("valid", "repaired", "rejected", "no_call"):
        lines.append(f"| {k} | {status.get(k, 0)} | {_pct(status.get(k, 0) / total)} |")
    broken = status.get("repaired", 0) + status.get("rejected", 0)
    if broken:
        lines.append(f"\nRepairable share of broken calls: {_pct(status.get('repaired', 0) / broken)} "
                     f"({status.get('repaired', 0)}/{broken}). No-call turns are not repairable.")
    return "\n".join(lines)


def agreement(runs: list[Run]) -> str:
    pairs = [(r.primary_failure, r.human_label) for r in runs if r.human_label and r.primary_failure]
    if not pairs:
        return "No hand labels yet (set `human_label` on runs you check)."
    k = cohen_kappa(pairs)
    acc = sum(a == b for a, b in pairs) / len(pairs)
    confusions = Counter((a, b) for a, b in pairs if a != b).most_common(5)
    lines = [f"Hand-labeled runs: {len(pairs)}  |  raw agreement {_pct(acc)}  |  Cohen's kappa {k:.3f}"]
    if confusions:
        lines.append("Top disagreements (auto -> human): " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in confusions))
    return "\n".join(lines)


def compare(a: list[Run], b: list[Run], name_a: str = "A", name_b: str = "B") -> str:
    """Paired before/after on shared task ids (RQ4)."""
    ra = {r.task_id: r.primary_failure == "RESOLVED" for r in a if group_of(r.primary_failure or "") != "excluded"}
    rb = {r.task_id: r.primary_failure == "RESOLVED" for r in b if group_of(r.primary_failure or "") != "excluded"}
    shared = sorted(set(ra) & set(rb))
    n = len(shared)
    ka = sum(ra[t] for t in shared)
    kb = sum(rb[t] for t in shared)
    only_a = sum(ra[t] and not rb[t] for t in shared)
    only_b = sum(rb[t] and not ra[t] for t in shared)
    p = mcnemar_exact(only_a, only_b)
    la, ha = wilson(ka, n)
    lb, hb = wilson(kb, n)
    return (f"| Config | Resolved / {n} shared tasks | 95% CI |\n| --- | --- | --- |\n"
            f"| {name_a} | {ka} ({_pct(ka / n if n else 0)}) | {_pct(la)}–{_pct(ha)} |\n"
            f"| {name_b} | {kb} ({_pct(kb / n if n else 0)}) | {_pct(lb)}–{_pct(hb)} |\n\n"
            f"Only {name_a} solved: {only_a}; only {name_b} solved: {only_b}; exact McNemar p = {p:.4f}")
