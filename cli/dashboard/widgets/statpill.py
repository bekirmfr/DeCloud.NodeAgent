"""
widgets/statpill.py — Status pill: a coloured dot + short label.

Used in the header status strip and inside cards to express
binary or graded health at a glance.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from theme import COLOR, DOT_FILLED, DOT_OPEN, CHECK, CROSS


SEVERITY_TO_COLOR = {
    "ok":      COLOR["ok"],
    "warn":    COLOR["warn"],
    "crit":    COLOR["crit"],
    "info":    COLOR["info"],
    "unknown": COLOR["dim"],
}


def status_text(severity: str, text: str, *, glyph: str | None = None) -> Text:
    """Return a Rich Text: '● label' coloured by severity."""
    color = SEVERITY_TO_COLOR.get(severity, COLOR["dim"])
    g = glyph if glyph is not None else (DOT_FILLED if severity != "unknown" else DOT_OPEN)
    out = Text()
    out.append(g + " ", style=f"{color}")
    out.append(text, style="bold")
    return out


class StatPill(Static):
    """A static, label-bearing status pill.

    Update via .set(severity, label).
    """

    DEFAULT_CSS = """
    StatPill {
        height: 1;
        width: auto;
        padding: 0 1;
    }
    """

    def __init__(self, severity: str = "unknown", label: str = "—",
                 glyph: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sev = severity
        self._label = label
        self._glyph = glyph

    def set(self, severity: str, label: str, *, glyph: str | None = None) -> None:
        self._sev = severity
        self._label = label
        self._glyph = glyph
        self.refresh()

    def render(self) -> Text:
        return status_text(self._sev, self._label, glyph=self._glyph)
