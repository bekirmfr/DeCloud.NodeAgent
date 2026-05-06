"""
widgets/gauge.py — Resource gauge with label, percentage, bar, and sparkline.

Composite widget: a horizontal row showing
    LABEL    [█████████░░░░░]   78%   ▁▂▃▅▇█
The bar is rendered as Rich block-character text inside a Static.
This avoids Textual's ProgressBar which doesn't easily colour-grade.

Threshold colours come from theme.severity().
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static

from theme import COLOR, severity
from util.format import fmt_pct
from widgets.sparkline import Sparkline


def render_bar(pct: float, width: int) -> Text:
    """Solid bar made of full-block chars only.

    Earlier versions added a partial eighth-block at the tip
    (`▁▂▃▄▅▆▇`), which served two bugs simultaneously: (a) those are
    *bottom* eighths, semantically wrong for a horizontal bar (a
    horizontal bar should use *left* eighths `▏▎▍▌▋▊▉`), and (b) many
    fonts that ship with `█` lack the eighth-block range entirely,
    rendering the tip as tofu.  Rounding to the nearest full block is
    visually almost identical at the widths we use and works in every
    monospace font.
    """
    pct = max(0.0, min(100.0, pct))
    full = int(round(pct / 100.0 * width))
    bar = "█" * full + " " * (width - full)
    color = COLOR[severity(pct)]
    return Text(bar, style=f"{color}")


class Gauge(Container):
    """A single labelled resource gauge.

    Composes:
      [  LABEL  ] [bar...........] [pct%] [sparkline]

    Use .update(pct, history=[...]) to refresh from the parent screen.
    """

    DEFAULT_CSS = """
    Gauge {
        height: 1;
        width: 1fr;
    }
    Gauge .gauge-label {
        width: 9;
        color: $text-muted;
    }
    Gauge .gauge-bar {
        width: 1fr;
        margin: 0 1 0 0;
    }
    Gauge .gauge-pct {
        width: 7;
        text-align: right;
        margin-right: 1;
    }
    Gauge .gauge-spark {
        width: 24;
        text-align: right;
    }
    """

    def __init__(self, label: str, *, bar_width: int = 28, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._bar_width = bar_width
        self._pct: float = 0.0
        self._history: list[float] = []

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(self._label, classes="gauge-label")
            yield Static("", classes="gauge-bar", id="bar")
            yield Static("—", classes="gauge-pct", id="pct")
            yield Sparkline(id="spark", classes="gauge-spark")

    def update(self, pct: float | None, history: list[float] | None = None) -> None:
        """Set the current percentage and (optionally) the history series."""
        self._pct = float(pct) if pct is not None else 0.0
        if history is not None:
            self._history = list(history)
        try:
            self.query_one("#bar", Static).update(render_bar(self._pct, self._bar_width))
            color = COLOR[severity(self._pct)]
            txt = Text(fmt_pct(self._pct), style=f"bold {color}")
            self.query_one("#pct", Static).update(txt)
            self.query_one("#spark", Sparkline).update_values(self._history)
        except Exception:
            # Widget may be disposed mid-refresh; ignore.
            pass

    def set_unavailable(self, reason: str = "n/a") -> None:
        """Render the gauge in a 'not present' state (e.g. no GPU)."""
        try:
            self.query_one("#bar", Static).update(
                Text("─" * self._bar_width, style=f"{COLOR['dim']}")
            )
            self.query_one("#pct", Static).update(
                Text(reason, style=f"{COLOR['dim']}")
            )
            self.query_one("#spark", Sparkline).update_values([])
        except Exception:
            pass
