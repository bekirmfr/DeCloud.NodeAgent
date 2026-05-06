"""
app.py — DeCloud CLI Dashboard application.

Top-level Textual App. The app owns:

  • Global keymap (1–9 to switch screens, r to refresh, q to quit)
  • A dark theme with cyan accent (matches the DeCloud web brand)
  • Initial screen (Overview)

Screens own their own data fetching. The app does not poll directly;
each screen schedules its own set_interval. This keeps the navigation
layer free of business logic.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding

from config import cfg
from screens import get_screen
from screens.overview import OverviewScreen


class DeCloudDashboard(App):

    TITLE = "DeCloud · Node Operator Dashboard"

    # ─── Global CSS ────────────────────────────────────────────────────
    # Per-screen CSS lives next to each screen (DEFAULT_CSS). Here we
    # only define rules that apply to chrome (sidebar, identity bar,
    # status strip, key hints) shared across all screens.

    CSS = """
    /* --- Tokens (Textual theme variables we override per element) --- */
    Screen { background: $background; color: $text; }

    /* Identity bar separator */
    #identity-bar { border-bottom: solid $panel; }

    /* Card chrome — accentuate focused card without being noisy */
    Card { background: $surface; }
    Card:focus-within { background: $boost; }

    /* DataTable readability */
    DataTable {
        background: $surface;
    }
    DataTable > .datatable--header {
        background: $boost;
        color: $accent;
        text-style: bold;
    }
    DataTable > .datatable--cursor {
        background: $boost;
    }

    /* Tabs */
    TabbedContent Tabs {
        background: $surface;
    }

    /* Inputs and buttons sit on slightly elevated surface */
    Input, Select { background: $boost; }
    Button { min-width: 10; }
    Button:hover { background: $accent 30%; }
    """

    BINDINGS = [
        Binding("q",       "quit",         "Quit",    show=True),
        Binding("ctrl+c",  "quit",         "Quit",    show=False),
        Binding("r",       "refresh",      "Refresh", show=True),
        Binding("?",       "help",         "Help",    show=True),
        Binding("question_mark", "help",   "Help",    show=False),
        # 1-9 switch screens. priority=True so they fire even when an Input
        # widget has focus on the current screen (otherwise the digit would
        # be typed into the input instead of switching screens).
        Binding("1", "go_1", show=False, priority=True),
        Binding("2", "go_2", show=False, priority=True),
        Binding("3", "go_3", show=False, priority=True),
        Binding("4", "go_4", show=False, priority=True),
        Binding("5", "go_5", show=False, priority=True),
        Binding("6", "go_6", show=False, priority=True),
        Binding("7", "go_7", show=False, priority=True),
        Binding("8", "go_8", show=False, priority=True),
        Binding("9", "go_9", show=False, priority=True),
        Binding("0", "go_0", show=False, priority=True),
    ]

    # Map 1..9,0 → screen label (mirrors NAV in screens/_base.py).
    _SCREEN_BY_KEY = {
        "1": "Overview",
        "2": "Hardware",
        "3": "Obligations",
        "4": "Virtual Machines",
        "5": "Network",
        "6": "Firewall",
        "7": "Services",
        "8": "Logs",
        "9": "Diagnostics",
        "0": "Settings",
    }

    # ─── Lifecycle ─────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # Use Textual's dark theme as base; cyan accent comes from $accent.
        self.dark = True
        self.push_screen(OverviewScreen())

    # ─── Actions ───────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        """Delegate to the active screen's refresh, if it has one."""
        screen = self.screen
        if hasattr(screen, "action_refresh"):
            screen.action_refresh()

    def action_help(self) -> None:
        """Show a quick keymap toast."""
        self.notify(
            "Navigation: 1–9 switch screens · r refresh · q quit\n"
            "Per-screen actions: see hint bar at the bottom",
            title="Help", timeout=6,
        )

    # Screen-switch actions — generated programmatically via __getattr__.
    def __getattr__(self, name: str):
        if name.startswith("action_go_") and name[10:].isdigit():
            key = name[10:]
            label = self._SCREEN_BY_KEY.get(key)
            if label:
                def _go(_label=label):
                    self._nav(_label)
                return _go
        raise AttributeError(name)

    def _nav(self, label: str) -> None:
        new = get_screen(label)
        if new is not None:
            # switch_screen replaces the current screen (no stack).
            self.switch_screen(new)


def run() -> None:
    DeCloudDashboard().run()
