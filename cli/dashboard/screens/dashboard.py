"""
screens/dashboard.py — Cluster overview screen.

Shows: stat cards (nodes/VMs/revenue/network), resource gauges,
recent event feed, and a node fleet snapshot table.
Auto-refreshes every cfg.refresh_interval seconds.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    DataTable,
    Label,
    Log,
    ProgressBar,
    Static,
)
from textual.timer import Timer

from config import cfg
from api.orchestrator import OrchestratorClient, ApiError
from api.node_agent import NodeAgentClient


_STATUS_STYLE = {
    "Online": "bold green",
    "Degraded": "bold yellow",
    "Offline": "bold red",
}


class StatCard(Static):
    """A single metric card: label + big value + sub-line."""

    DEFAULT_CSS = """
    StatCard {
        border: solid $panel;
        padding: 1 2;
        width: 1fr;
        height: 7;
    }
    StatCard .card-label { color: $text-muted; }
    StatCard .card-value { color: $text; text-style: bold; }
    StatCard .card-sub   { color: $text-muted; }
    """

    def __init__(self, label: str, value: str, sub: str = "", sub_style: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._sub = sub
        self._sub_style = sub_style

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="card-label")
        yield Label(self._value, classes="card-value")
        yield Label(self._sub, classes="card-sub")

    def update_value(self, value: str, sub: str = "") -> None:
        self.query_one(".card-value", Label).update(value)
        if sub:
            try:
                self.query_one(".card-sub", Label).update(sub)
            except Exception:
                pass


class GaugeRow(Static):
    """Label + progress bar + percentage."""

    DEFAULT_CSS = """
    GaugeRow {
        height: 2;
        layout: horizontal;
    }
    GaugeRow .gauge-label { width: 12; color: $text-muted; }
    GaugeRow ProgressBar   { width: 1fr; }
    GaugeRow .gauge-pct   { width: 6; text-align: right; color: $text-muted; }
    """

    def __init__(self, label: str, pct: float) -> None:
        super().__init__()
        self._label = label
        self._pct = pct

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="gauge-label")
        yield ProgressBar(total=100, show_eta=False, show_percentage=False)
        yield Label(f"{self._pct:.0f}%", classes="gauge-pct")

    def on_mount(self) -> None:
        self.query_one(ProgressBar).advance(self._pct)


class DashboardScreen(Widget):
    _is_mounted: bool = False

    """Main overview — stat cards, resource gauges, event log, node table."""

    BINDINGS = [("r", "refresh", "Refresh")]

    _timer: Timer | None = None
    nodes: reactive[list] = reactive([])
    stats: reactive[dict] = reactive({})

    def compose(self) -> ComposeResult:
        with Horizontal(id="stat-row"):
            yield StatCard("Nodes Online", "—", "loading…", id="card-nodes")
            yield StatCard("Running VMs", "—", id="card-vms")
            yield StatCard("Revenue / 24h", "—", id="card-rev")
            yield StatCard("Network Out", "—", id="card-net")

        with Horizontal(id="mid-row"):
            with Vertical(id="gauges-panel"):
                yield Label("Cluster Resources", classes="section-title")
                yield GaugeRow("CPU", 0)
                yield GaugeRow("Memory", 0)
                yield GaugeRow("Storage", 0)
                yield GaugeRow("Network", 0)

            with Vertical(id="events-panel"):
                yield Label("Recent Events", classes="section-title")
                yield Log(id="event-log", max_lines=cfg.log_lines)

        yield Label("Node Fleet", classes="section-title")
        yield DataTable(id="node-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#node-table", DataTable)
        table.add_columns("Node", "Region", "VMs", "CPU", "Mem", "Status", "Last Seen")
        self._timer = self.set_interval(cfg.refresh_interval, self._refresh_data)
        self.run_worker(self._refresh_data(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._refresh_data(), exclusive=True)

    async def _refresh_data(self) -> None:
        if not cfg.has_orchestrator:
            return
        client = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            stats, nodes = await _fetch_overview(client)
            self._apply_stats(stats)
            self._apply_nodes(nodes)
        except ApiError as e:
            self._log_event(f"[red]API error: {e.message}[/]")
        finally:
            await client.close()

    def _apply_stats(self, s: dict) -> None:
        online = s.get("onlineNodes", "—")
        total = s.get("totalNodes", "—")
        self.query_one("#card-nodes", StatCard).update_value(
            f"{online} / {total}", f"{s.get('offlineNodes', 0)} offline"
        )
        self.query_one("#card-vms", StatCard).update_value(
            str(s.get("runningVms", "—")), f"{s.get('totalVms', 0)} total"
        )
        self.query_one("#card-net", StatCard).update_value(
            f"{s.get('totalBandwidthMbps', 0):.0f} Mbps"
        )

    def _apply_nodes(self, nodes: list) -> None:
        table = self.query_one("#node-table", DataTable)
        table.clear()
        for n in nodes:
            res = n.get("availableResources") or n.get("totalResources") or {}
            status = n.get("status", "Unknown")
            style = _STATUS_STYLE.get(status, "")
            table.add_row(
                n.get("id", "")[:12],
                n.get("region", "—"),
                str(n.get("runningVmCount", "—")),
                f"{res.get('cpuUsagePct', 0):.0f}%",
                f"{res.get('memUsagePct', 0):.0f}%",
                f"[{style}]{status}[/]" if style else status,
                n.get("lastSeenAgo", "—"),
            )
        self._log_event(f"[dim]Refreshed — {len(nodes)} nodes[/]")

    def _log_event(self, msg: str) -> None:
        self.query_one("#event-log", Log).write_line(msg)


async def _fetch_overview(client: OrchestratorClient):
    import asyncio
    stats, nodes = await asyncio.gather(
        client.get_stats(),
        client.list_nodes(),
        return_exceptions=True,
    )
    if isinstance(stats, Exception):
        stats = {}
    if isinstance(nodes, Exception):
        nodes = []
    return stats, nodes