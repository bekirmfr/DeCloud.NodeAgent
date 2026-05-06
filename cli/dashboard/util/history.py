"""
util/history.py — Bounded ring buffer for sparkline history.

Stateless container; the screen owns the lifetime.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable


class Ring:
    """Fixed-capacity FIFO of floats. Newest sample is the rightmost."""

    __slots__ = ("_buf",)

    def __init__(self, capacity: int) -> None:
        self._buf: Deque[float] = deque(maxlen=capacity)

    def push(self, value: float) -> None:
        self._buf.append(float(value))

    def values(self) -> list[float]:
        return list(self._buf)

    def latest(self) -> float | None:
        return self._buf[-1] if self._buf else None

    def __len__(self) -> int:
        return len(self._buf)

    def extend(self, vals: Iterable[float]) -> None:
        for v in vals:
            self.push(v)
