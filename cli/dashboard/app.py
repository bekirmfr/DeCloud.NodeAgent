"""
app.py — DeCloud CLI Dashboard application.

Uses proper Textual Screen subclasses with switch_screen() for navigation.
No ContentSwitcher, no mount/unmount — each screen is a real Screen.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding

from config import cfg
from screens import get_screen
from screens.dashboard import DashboardScreen


class DeCloudDashboard(App):
    """DeCloud CLI Dashboard."""

    TITLE     = "DeCloud Dashboard"
    SUB_TITLE = cfg.orchestrator_url or cfg.node_url or "not configured"

    CSS = """
    /* Global layout */
    BaseScreen { layout: horizontal; }
    #content   { width: 1fr; height: 100%; padding: 1 2; overflow: auto auto; }

    /* Typography */
    .section-title {
        color: $text-muted; text-style: italic;
        border-bottom: solid $panel; margin-bottom: 1; height: 2;
    }
    .card-label { color: $text-muted; height: 1; }
    .card-value { text-style: bold; height: 2; }
    .card-sub   { color: $text-muted; height: 1; }

    /* Stat cards */
    .stat-card  { border: solid $panel; padding: 1 2; width: 1fr; height: 7; }
    #stat-row   { height: 8; margin-bottom: 1; }
    #balance-row { height: 8; margin-bottom: 1; }

    /* Gauges */
    .gauge-row    { height: 2; }
    .gauge-label  { width: 12; color: $text-muted; }
    .gauge-pct    { width: 7; text-align: right; color: $text-muted; }

    /* Mid row */
    #mid-row      { height: 18; margin-bottom: 1; }
    #gauges-panel { width: 1fr; margin-right: 1; }
    #events-panel { width: 1fr; }

    /* Filter / action bars */
    #filter-bar { height: 3; margin-bottom: 1; align: left middle; }
    #filter-bar Input  { width: 28; margin-right: 1; }
    #filter-bar Button { width: auto; margin-right: 1; height: 3; }
    #filter-bar Select { width: 22; margin-right: 1; }
    #action-bar { height: 3; margin-top: 1; align: left middle; }
    #action-bar Button { width: auto; margin-right: 1; height: 3; }

    /* Settings */
    #settings-actions { height: 3; margin-top: 1; align: left middle; }
    #settings-actions Button { width: auto; margin-right: 1; height: 3; }
    #settings-status  { height: 2; color: $text-muted; margin-top: 1; }

    /* DataTable */
    DataTable { height: 1fr; }

    /* Sidebar */
    Sidebar .nav-key   { width: 3; color: $text-disabled; }
    Sidebar .nav-label { width: 1fr; }
    """

    BINDINGS = [
        Binding("q",      "quit",   "Quit",    show=True),
        Binding("ctrl+c", "quit",   "Quit",    show=False),
        Binding("r",      "refresh","Refresh", show=True),
        Binding("1", "go_1", show=False), Binding("2", "go_2", show=False),
        Binding("3", "go_3", show=False), Binding("4", "go_4", show=False),
        Binding("5", "go_5", show=False), Binding("6", "go_6", show=False),
        Binding("7", "go_7", show=False), Binding("8", "go_8", show=False),
        Binding("9", "go_9", show=False),
    ]

    _NAV_LABELS = [
        "Dashboard", "Nodes", "Virtual Machines", "System VMs",
        "Networking", "Ingress Routes", "Billing", "Live Logs", "Settings",
    ]

    def compose(self) -> ComposeResult:
        # App composes nothing — the Screen provides all layout
        return
        yield  # type: ignore[misc]

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen())

    # 1-9 navigation
    def action_go_1(self) -> None: self._nav("Dashboard")
    def action_go_2(self) -> None: self._nav("Nodes")
    def action_go_3(self) -> None: self._nav("Virtual Machines")
    def action_go_4(self) -> None: self._nav("System VMs")
    def action_go_5(self) -> None: self._nav("Networking")
    def action_go_6(self) -> None: self._nav("Ingress Routes")
    def action_go_7(self) -> None: self._nav("Billing")
    def action_go_8(self) -> None: self._nav("Live Logs")
    def action_go_9(self) -> None: self._nav("Settings")

    def _nav(self, label: str) -> None:
        screen = get_screen(label)
        if screen:
            self.switch_screen(screen)

    def action_refresh(self) -> None:
        screen = self.screen
        if hasattr(screen, "action_refresh"):
            screen.action_refresh()