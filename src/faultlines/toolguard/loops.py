"""Detect an agent stuck repeating itself.

Two patterns, both common in small-model traces:
  * the same call N times within a short window (e.g. re-reading one file), and
  * a short cycle repeated (A B A B A B, e.g. edit fails -> read -> edit fails ...).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class LoopSignal:
    kind: str       # "repeat" or "cycle"
    period: int     # 1 for repeat, cycle length otherwise
    count: int      # how many times it repeated


class LoopDetector:
    def __init__(self, window: int = 12, repeat_threshold: int = 3, cycle_repeats: int = 3, max_period: int = 3):
        self.history: deque[str] = deque(maxlen=window)
        self.repeat_threshold = repeat_threshold
        self.cycle_repeats = cycle_repeats
        self.max_period = max_period

    def observe(self, call_key: str) -> LoopSignal | None:
        self.history.append(call_key)
        h = list(self.history)
        # consecutive identical calls
        run = 1
        for prev in reversed(h[:-1]):
            if prev != call_key:
                break
            run += 1
        if run >= self.repeat_threshold:
            return LoopSignal("repeat", 1, run)
        # cycles of period 2..max_period ending at the latest call
        for period in range(2, self.max_period + 1):
            need = period * self.cycle_repeats
            if len(h) < need:
                continue
            tail = h[-need:]
            unit = tail[:period]
            if len(set(unit)) == period and all(tail[i] == unit[i % period] for i in range(need)):
                return LoopSignal("cycle", period, self.cycle_repeats)
        return None

    def reset(self) -> None:
        self.history.clear()


def find_loops(call_keys: list[str | None], **kw) -> list[tuple[int, LoopSignal]]:
    """Run a detector over a whole trace. `None` entries (no valid call) are skipped."""
    det = LoopDetector(**kw)
    hits = []
    for i, k in enumerate(call_keys):
        if k is None:
            continue
        sig = det.observe(k)
        if sig is not None:
            hits.append((i, sig))
    return hits
