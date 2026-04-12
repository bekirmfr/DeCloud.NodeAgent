"""screens/_base.py — Sidebar widget and BaseScreen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Label, Static

NAV: list[tuple[str, str, str]] = [
    ("1", "Dashboard",        "Overview"),
    ("2", "Nodes",            "Compute"),
    ("3", "Virtual Machines", "Compute"),
    ("4", "System VMs",       "Compute"),
    ("5", "Networking",       "Network"),
    ("6", "Ingress Routes",   "Network"),
    ("7", "Billing",          "Finance"),
    ("8", "Live Logs",        "Diagnostics"),
    ("9", "Settings",         "Diagnostics"),
]


class NavItem(Static):
    DEFAULT_CSS = """
    NavItem { height: 2; padding: 0 2; color: $text-muted; }
    NavItem:hover { background: $boost; color: $text; }
    NavItem.active { color: $accent; background: $boost; border-left: thick $accent; }
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
        screen = get_screen(self._label)
        if screen:
            self.app.switch_screen(screen)


class Sidebar(Static):
    DEFAULT_CSS = """
    Sidebar { width: 22; height: 100%; background: $surface; border-right: solid $panel; }
    Sidebar .logo { padding: 1 2; color: $accent; text-style: bold;
                    border-bottom: solid $panel; margin-bottom: 1; height: 3; }
    Sidebar .section-hdr { padding: 1 2 0 2; color: $text-disabled;
                            text-style: italic; height: 2; }
    """

    def __init__(self, active: str = "") -> None:
        super().__init__()
        self._active = active

    def compose(self) -> ComposeResult:
        yield Label("  DECLOUD", classes="logo")
        last_section = ""
        for key, label, section in NAV:
            if section != last_section:
                yield Label(section, classes="section-hdr")
                last_section = section
            item = NavItem(key, label)
            if label == self._active:
                item.add_class("active")
            yield item


class BaseScreen(Screen):
    """Base screen with sidebar. Subclasses override compose_content()."""

    ACTIVE_LABEL: str = ""

    DEFAULT_CSS = """
    BaseScreen { layout: horizontal; }
    BaseScreen #content { width: 1fr; height: 100%; padding: 1 2; overflow: auto auto; }
    """

    def render(self) -> str:
        return ""

    def compose(self) -> ComposeResult:
        yield Sidebar(active=self.ACTIVE_LABEL)
        with Vertical(id="content"):
            yield from self.compose_content()

    def compose_content(self) -> ComposeResult:
        yield Label(self.ACTIVE_LABEL)