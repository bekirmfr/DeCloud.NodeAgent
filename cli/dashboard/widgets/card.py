"""
widgets/card.py — Titled card container.

Uses Textual's native border-title rendering (border_title on a Container
draws the title inside the top border). Subclass to populate via compose_body().
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container


class Card(Container):
    """Bordered container whose title is drawn in the top border.

    Override compose_body() in subclasses, or yield children via compose().
    Use .set_subtitle(text) to put a small string in the top-right of the border.
    """

    DEFAULT_CSS = """
    Card {
        border: round $panel;
        padding: 0 1;
        margin: 0 0 1 0;
        height: auto;
        /* Explicitly colour the border title — Textual's default is to
         * inherit from the border, which on dark themes gives the
         * "dark blue on dark background" effect that's nearly unreadable.
         * Cyan accent + bold makes section titles pop against the panel. */
        border-title-color: $accent;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
        border-subtitle-style: none;
    }
    Card:focus-within {
        border: round $accent;
    }
    """

    def __init__(self, title: str, *, subtitle: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = title
        if subtitle:
            self.border_subtitle = subtitle

    def compose(self) -> ComposeResult:
        yield from self.compose_body()

    def compose_body(self) -> ComposeResult:  # pragma: no cover — overridden
        return iter(())

    def set_subtitle(self, text: str) -> None:
        self.border_subtitle = text
