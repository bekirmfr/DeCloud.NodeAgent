"""
util/format.py — Formatting helpers.

Pure functions — no I/O, no side effects, easy to test.
"""

from __future__ import annotations


def fmt_bytes(n: int | float | None) -> str:
    """Human-readable byte size. 0 → '0 B'."""
    if n is None:
        return "—"
    n = float(n)
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB", "PB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}" if n < 100 else f"{n:.0f} {unit}"
    return f"{n:.1f} EB"


def fmt_duration(seconds: int | float | None) -> str:
    """Compact duration: 12s, 4m, 3h, 2d 4h, etc."""
    if seconds is None:
        return "—"
    s = int(seconds)
    if s < 0:
        s = 0
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        h, m = divmod(s, 3600)
        m //= 60
        return f"{h}h" if m == 0 else f"{h}h {m}m"
    d, rem = divmod(s, 86400)
    h = rem // 3600
    return f"{d}d" if h == 0 else f"{d}d {h}h"


def fmt_age(seconds: int | float | None) -> str:
    """Age expressed as 'just now', '12s ago', '4m ago' …"""
    if seconds is None:
        return "never"
    s = int(seconds)
    if s < 5:
        return "just now"
    return f"{fmt_duration(s)} ago"


def fmt_pct(pct: float | None, digits: int = 1) -> str:
    if pct is None:
        return "—"
    return f"{pct:.{digits}f}%"


def truncate(s: str | None, n: int, ellipsis: str = "…") -> str:
    """Truncate s to n chars, appending ellipsis if shortened."""
    if s is None:
        return "—"
    return s if len(s) <= n else s[: n - len(ellipsis)] + ellipsis


def short_id(s: str | None, n: int = 12) -> str:
    """Short identifier (first n chars). For VM IDs, node IDs."""
    if not s:
        return "—"
    return s if len(s) <= n else s[:n]
