"""
widgets/sparkline.py — Inline sparkline using Unicode block characters.

A pure-rendering widget: takes a list of values, produces colour-graded
text that fits in a single line. No internal state, no fetch logic —
the parent Screen owns history.
"""

from __future__ import annotations

from rich.text import Text
from textual.widget import Widget

from theme import SPARK_BLOCKS, COLOR, severity


def render_sparkline(values: list[float], width: int, max_val: float = 100.0) -> Text:
    """Render values as a Rich Text sparkline of the given width.

    Older samples are padded on the left; newest is on the right.
    Colour is graded by the latest value (ok/warn/crit).
    """
    if not values:
        return Text(" " * width, style=f"{COLOR['dim']}")

    n_blocks = len(SPARK_BLOCKS)
    cells: list[str] = []

    # Right-align: pad with spaces if values are fewer than width
    pad = max(0, width - len(values))
    cells.extend(" " * pad)

    # Take the last `width` values to display
    visible = values[-width:]
    for v in visible:
        ratio = max(0.0, min(1.0, v / max_val if max_val > 0 else 0.0))
        idx = min(n_blocks - 1, int(ratio * (n_blocks - 1) + 0.5))
        cells.append(SPARK_BLOCKS[idx])

    color = COLOR[severity(values[-1])]
    return Text("".join(cells), style=f"{color}")


class Sparkline(Widget):
    """Static one-line sparkline widget.

    Designed for inline embedding in dashboard cards.  The owner pushes
    new values via .update_values() — no polling logic here.
    """

    DEFAULT_CSS = """
    Sparkline {
        height: 1;
        width: 1fr;
        content-align: right middle;
    }
    """

    def __init__(self, max_val: float = 100.0, width_cells: int = 24, **kwargs) -> None:
        super().__init__(**kwargs)
        self._values: list[float] = []
        self._max = max_val
        self._w = width_cells

    def update_values(self, values: list[float]) -> None:
        self._values = list(values)
        self.refresh()

    def render(self) -> Text:
        return render_sparkline(self._values, self._w, self._max)
