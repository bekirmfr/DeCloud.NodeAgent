"""
widgets/keyhints.py — Bottom hint bar showing keyboard bindings and refresh state.

Always visible at the bottom of every screen, like vim's status line.
"""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.widgets import Static

from theme import COLOR


class KeyHints(Static):
    """Bottom status line: nav hints + last refresh time."""

    DEFAULT_CSS = """
    KeyHints {
        height: 1;
        dock: bottom;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    """

    BASE_HINTS = [
        ("1-0", "switch"),
        ("r",   "refresh"),
        ("?",   "help"),
        ("q",   "quit"),
    ]

    def __init__(self, extra: list[tuple[str, str]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._extra = extra or []
        self._last_update: datetime | None = None

    def set_last_update(self, when: datetime | None = None) -> None:
        self._last_update = when or datetime.now()
        self.refresh()

    def add_extra(self, hints: list[tuple[str, str]]) -> None:
        self._extra = hints
        self.refresh()

    def render(self) -> Text:
        out = Text()
        for i, (k, lbl) in enumerate([*self.BASE_HINTS, *self._extra]):
            if i:
                out.append("   ")
            out.append(k, style=f"bold {COLOR['info']}")
            out.append(" " + lbl)
        if self._last_update:
            out.append("    ")
            out.append("· updated ", style=f"{COLOR['dim']}")
            out.append(self._last_update.strftime("%H:%M:%S"),
                       style=f"{COLOR['muted']}")
        return out
