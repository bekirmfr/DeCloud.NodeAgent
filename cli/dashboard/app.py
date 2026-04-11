"""
app.py — DeCloud CLI Dashboard — main Textual application.

Layout:
  ┌─────────────┬──────────────────────────────┐
  │  Sidebar    │  Main content area           │
  │  (nav)      │  (active Screen)             │
  └─────────────┴──────────────────────────────┘
  │  Status bar                                 │
  └─────────────────────────────────────────────┘

Keybindings:
  q / ctrl+c   quit
  1-9          jump to screen by number
  r            refresh active screen
  ?            show help
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, Static

from config import cfg
from screens.dashboard  import DashboardScreen
from screens.nodes      import NodesScreen
from screens.vms        import VmsScreen
from screens.system_vms import SystemVmsScreen
from screens.network    import NetworkScreen
from screens.ingress    import IngressScreen
from screens.billing    import BillingScreen
from screens.logs       import LogsScreen
from screens.settings   import SettingsScreen


# ---------------------------------------------------------------------------
# Navigation registry
# ---------------------------------------------------------------------------

NAV: list[tuple[str, str, type]] = [
    #  key   label              Screen class
    ("1", "Dashboard",         DashboardScreen),
    ("2", "Nodes",             NodesScreen),
    ("3", "Virtual Machines",  VmsScreen),
    ("4", "System VMs",        SystemVmsScreen),
    ("5", "Networking",        NetworkScreen),
    ("6", "Ingress Routes",    IngressScreen),
    ("7", "Billing",           BillingScreen),
    ("8", "Live Logs",         LogsScreen),
    ("9", "Settings",          SettingsScreen),
]

# Group labels for sidebar sections
_SECTIONS: dict[str, list[str]] = {
    "Overview":    ["Dashboard"],
    "Compute":     ["Nodes", "Virtual Machines", "System VMs"],
    "Network":     ["Networking", "Ingress Routes"],
    "Finance":     ["Billing"],
    "Diagnostics": ["Live Logs", "Settings"],
}

_LABEL_TO_KEY = {label: key for key, label, _ in NAV}


# ---------------------------------------------------------------------------
# Sidebar item widget
# ---------------------------------------------------------------------------

class NavItem(Static):
    """A single sidebar navigation entry."""

    DEFAULT_CSS = """
    NavItem {
        height: 2;
        padding: 0 2;
        color: $text-muted;
    }
    NavItem:hover { background: $boost; color: $text; }
    NavItem.active {
        color: $accent;
        background: $boost;
        border-left: thick $accent;
    }
    NavItem .nav-key  { color: $text-disabled; width: 3; }
    NavItem .nav-label { width: 1fr; }
    """

    def __init__(self, key: str, label: str) -> None:
        super().__init__()
        self._key = key
        self._label = label

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(self._key, classes="nav-key")
            yield Label(self._label, classes="nav-label")

    def on_click(self) -> None:
        self.app.switch_to(self._label)  # type: ignore[attr-defined]


class Sidebar(Static):
    """Left navigation panel."""

    DEFAULT_CSS = """
    Sidebar {
        width: 22;
        background: $surface;
        border-right: solid $panel;
        height: 100%;
        padding: 0 0 1 0;
    }
    Sidebar .logo {
        padding: 1 2;
        color: $accent;
        text-style: bold;
        border-bottom: solid $panel;
        margin-bottom: 1;
    }
    Sidebar .section-header {
        padding: 1 2 0 2;
        color: $text-disabled;
        text-style: italic;
        height: 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("  DECLOUD", classes="logo")
        rendered: set[str] = set()
        for section, labels in _SECTIONS.items():
            yield Label(section, classes="section-header")
            for label in labels:
                key = _LABEL_TO_KEY.get(label, "")
                yield NavItem(key, label)
                rendered.add(label)


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

class StatusBar(Static):
    """Bottom status bar."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        border-top: solid $panel;
        layout: horizontal;
        padding: 0 1;
        color: $text-muted;
    }
    StatusBar .sb-item { width: auto; margin-right: 3; }
    StatusBar .sb-right { dock: right; color: $text-disabled; }
    """

    orch_status: reactive[str] = reactive("[dim]connecting…[/]")

    def compose(self) -> ComposeResult:
        yield Label("", id="sb-orch", classes="sb-item", markup=True)
        yield Label("", id="sb-node", classes="sb-item", markup=True)
        yield Label("q:quit  ?:help  r:refresh  1-9:nav", classes="sb-right")

    def set_orch(self, msg: str) -> None:
        self.query_one("#sb-orch", Label).update(msg)

    def set_node(self, msg: str) -> None:
        self.query_one("#sb-node", Label).update(msg)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class DeCloudDashboard(App):
    """DeCloud CLI Dashboard — Textual TUI."""

    TITLE = "DeCloud Dashboard"
    SUB_TITLE = cfg.orchestrator_url or cfg.node_url or "not configured"

    CSS = """
    /* ── Global layout ── */
    Screen {
        layout: horizontal;
    }
    #main-area {
        width: 1fr;
        height: 100%;
        overflow: auto auto;
        padding: 1 2;
    }

    /* ── Typography helpers ── */
    .section-title {
        color: $text-muted;
        text-style: italic;
        border-bottom: solid $panel;
        margin-bottom: 1;
        height: 2;
        padding-bottom: 0;
    }
    .card-label { color: $text-muted; height: 1; }
    .card-value { text-style: bold; height: 2; }
    .card-sub   { color: $text-muted; height: 1; }

    /* ── Filter / action bars ── */
    #filter-bar {
        height: 3;
        margin-bottom: 1;
        align: left middle;
    }
    #filter-bar Input  { width: 28; margin-right: 1; }
    #filter-bar Button { width: auto; margin-right: 1; height: 3; }
    #filter-bar Select { width: 22; margin-right: 1; }

    #action-bar {
        height: 3;
        margin-top: 1;
        align: left middle;
    }
    #action-bar Button { width: auto; margin-right: 1; height: 3; }

    /* ── Stat / balance rows ── */
    #stat-row, #balance-row {
        height: 8;
        margin-bottom: 1;
    }

    /* ── Mid-section two-column ── */
    #mid-row {
        height: 18;
        margin-bottom: 1;
    }
    #gauges-panel  { width: 1fr; margin-right: 1; }
    #events-panel  { width: 1fr; }

    /* ── Settings ── */
    #settings-actions {
        height: 3;
        margin-top: 1;
        align: left middle;
    }
    #settings-actions Button { width: auto; margin-right: 1; height: 3; }
    #settings-status { height: 2; color: $text-muted; margin-top: 1; }

    /* ── DataTable tweaks ── */
    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("q",       "quit",           "Quit",    show=True),
        Binding("ctrl+c",  "quit",           "Quit",    show=False),
        Binding("r",       "refresh_screen", "Refresh", show=True),
        Binding("question_mark", "help",     "Help",    show=True),
        *[Binding(key, f"go_{key}", label, show=False) for key, label, _ in NAV],
    ]

    _active_label: reactive[str] = reactive("Dashboard")
    _screen_cache: dict[str, object] = {}

    def compose(self) -> ComposeResult:
        yield Sidebar()
        yield Vertical(id="main-area")
        yield StatusBar()

    def on_mount(self) -> None:
        self.switch_to("Dashboard")
        self.set_interval(10, self._poll_status)
        self.run_worker(self._poll_status(), exclusive=True)

    # ── Screen navigation ──────────────────────────────────────────────

    def switch_to(self, label: str) -> None:
        """Mount the chosen widget into #main-area, unmounting the previous."""
        entry = next((e for e in NAV if e[1] == label), None)
        if not entry:
            return
        self._active_label = label
        self._update_nav_highlight(label)
        self.run_worker(self._mount_screen(label), exclusive=True, name="switch")

    async def _mount_screen(self, label: str) -> None:
        entry = next((e for e in NAV if e[1] == label), None)
        if not entry:
            return
        _, _, widget_cls = entry
        area = self.query_one("#main-area", Vertical)
        await area.remove_children()
        await area.mount(widget_cls())

    def _update_nav_highlight(self, active_label: str) -> None:
        for item in self.query(NavItem):
            if item._label == active_label:
                item.add_class("active")
            else:
                item.remove_class("active")

    # ── Keybinding actions for 1-9 ────────────────────────────────────

    def action_go_1(self) -> None: self.switch_to("Dashboard")
    def action_go_2(self) -> None: self.switch_to("Nodes")
    def action_go_3(self) -> None: self.switch_to("Virtual Machines")
    def action_go_4(self) -> None: self.switch_to("System VMs")
    def action_go_5(self) -> None: self.switch_to("Networking")
    def action_go_6(self) -> None: self.switch_to("Ingress Routes")
    def action_go_7(self) -> None: self.switch_to("Billing")
    def action_go_8(self) -> None: self.switch_to("Live Logs")
    def action_go_9(self) -> None: self.switch_to("Settings")

    def action_refresh_screen(self) -> None:
        """Delegate r to the active child screen."""
        area = self.query_one("#main-area", Vertical)
        for child in area.children:
            if hasattr(child, "action_refresh"):
                child.action_refresh()  # type: ignore[attr-defined]

    # ── Status bar polling ─────────────────────────────────────────────

    async def _poll_status(self) -> None:
        sb = self.query_one(StatusBar)
        if cfg.has_orchestrator:
            from api.orchestrator import OrchestratorClient, ApiError
            client = OrchestratorClient(cfg.orchestrator_url, cfg.token)
            try:
                stats = await client.get_stats()
                online = stats.get("onlineNodes", "?")
                total = stats.get("totalNodes", "?")
                sb.set_orch(f"[green]●[/] orchestrator  {online}/{total} nodes")
            except Exception:
                sb.set_orch("[red]●[/] orchestrator unreachable")
            finally:
                await client.close()

        if cfg.has_node_agent:
            from api.node_agent import NodeAgentClient
            client = NodeAgentClient(cfg.node_url)
            try:
                await client.get_summary()
                sb.set_node("[green]●[/] node agent")
            except Exception:
                sb.set_node("[red]●[/] node agent unreachable")
            finally:
                await client.close()
