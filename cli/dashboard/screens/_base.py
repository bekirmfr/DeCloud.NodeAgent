"""
screens/_base.py — Sidebar + BaseScreen layout.

Every Screen in the dashboard inherits from BaseScreen, which composes:

  ┌─────────┬───────────────────────────────────────────────┐
  │         │ IdentityBar                                   │
  │         ├───────────────────────────────────────────────┤
  │ Sidebar │ StatusStrip                                   │
  │  (nav)  ├───────────────────────────────────────────────┤
  │         │                                               │
  │         │     #content   (subclass-specific body)       │
  │         │                                               │
  │         ├───────────────────────────────────────────────┤
  │         │ KeyHints (1-9 switch  r refresh  ? help …)    │
  └─────────┴───────────────────────────────────────────────┘

Subclasses define:
  ACTIVE_LABEL  — sidebar item to mark as active
  EXTRA_HINTS   — list[(key, label)] for the bottom bar
  compose_content() — yields the body widgets

This keeps every screen visually and behaviourally consistent.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Label, Static

from theme import COLOR
from widgets.header import IdentityBar, StatusStrip
from widgets.keyhints import KeyHints


# Sidebar entries, grouped by section.
NAV: list[tuple[str, str, str]] = [
    ("1", "Overview",         "Health"),
    ("2", "Hardware",         "Health"),
    ("3", "Virtual Machines", "Workloads"),
    ("4", "Network",          "Connectivity"),
    ("5", "Firewall",         "Connectivity"),
    ("6", "Services",         "System"),
    ("7", "Logs",             "System"),
    ("8", "Diagnostics",      "Tools"),
    ("9", "Settings",         "Tools"),
]


class NavItem(Container):
    """One sidebar row: keybind + label, click-to-navigate."""

    DEFAULT_CSS = """
    NavItem {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    NavItem:hover { background: $boost; color: $text; }
    NavItem.active {
        color: $accent;
        background: $boost;
        text-style: bold;
    }
    NavItem .nav-key   { width: 3; color: $text-disabled; }
    NavItem .nav-label { width: 1fr; }
    """

    def __init__(self, key: str, label: str) -> None:
        super().__init__()
        self._key = key
        self._label = label

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(self._key,   classes="nav-key")
            yield Label(self._label, classes="nav-label")

    def on_click(self) -> None:
        from screens import get_screen
        target = get_screen(self._label)
        if target is not None:
            self.app.switch_screen(target)


class Sidebar(Container):
    """Permanent left navigation column."""

    DEFAULT_CSS = """
    Sidebar {
        width: 22;
        height: 100%;
        background: $surface;
        border-right: solid $panel;
    }
    Sidebar .logo {
        padding: 1 2;
        color: $accent;
        text-style: bold;
        height: 3;
        border-bottom: solid $panel;
        margin-bottom: 1;
    }
    Sidebar .section-hdr {
        padding: 1 2 0 2;
        color: $text-disabled;
        text-style: italic;
        height: 2;
    }
    """

    def __init__(self, active: str = "") -> None:
        super().__init__()
        self._active = active

    def compose(self) -> ComposeResult:
        yield Label("◆ DECLOUD\n  node ops", classes="logo")
        last_section = ""
        for key, label, section in NAV:
            if section != last_section:
                yield Label(section.upper(), classes="section-hdr")
                last_section = section
            item = NavItem(key, label)
            if label == self._active:
                item.add_class("active")
            yield item


class BaseScreen(Screen):
    """Common layout — every screen subclasses this."""

    ACTIVE_LABEL: str = ""
    EXTRA_HINTS: list[tuple[str, str]] = []

    DEFAULT_CSS = """
    BaseScreen { layout: horizontal; }
    #main-col {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }
    #content-pane {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Sidebar(active=self.ACTIVE_LABEL)
        with Vertical(id="main-col"):
            yield IdentityBar(id="identity-bar")
            yield StatusStrip(id="status-strip")
            with Vertical(id="content-pane"):
                yield from self.compose_content()
            yield KeyHints(extra=self.EXTRA_HINTS, id="key-hints")

    def compose_content(self) -> ComposeResult:  # pragma: no cover — overridden
        yield Static("(empty screen)")

    # Convenience used by all data-fetching screens.
    def mark_updated(self) -> None:
        try:
            self.query_one("#key-hints", KeyHints).set_last_update()
        except Exception:
            pass

    def show_message(self, msg: str, severity: str = "info") -> None:
        """Replace the content pane with a single centred message.

        Used when a screen has no data source available.
        """
        col = {"ok": COLOR["ok"], "warn": COLOR["warn"], "crit": COLOR["crit"],
               "info": COLOR["info"]}.get(severity, COLOR["muted"])
        self.notify(msg, severity={"crit":"error","warn":"warning"}.get(severity,"information"))
