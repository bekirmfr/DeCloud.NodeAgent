"""
screens/nodes.py — Node fleet management screen.

Filterable list of nodes with per-node resource bars,
system VM obligation status, and agent version info.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, ProgressBar, Static
from textual.reactive import reactive

from config import cfg
from api.orchestrator import OrchestratorClient, ApiError


_STATUS_COLOR = {
    "Online": "green",
    "Degraded": "yellow",
    "Offline": "red",
}

_FILTER_LABELS = ["All", "Online", "Degraded", "Offline"]


class NodeCard(Static):
    """Compact card showing one node's health."""

    DEFAULT_CSS = """
    NodeCard {
        border: solid $panel;
        padding: 1 2;
        margin-bottom: 1;
    }
    NodeCard .node-name  { text-style: bold; color: $accent; }
    NodeCard .node-meta  { color: $text-muted; }
    NodeCard .metric-row { height: 2; layout: horizontal; margin-top: 1; }
    NodeCard .mlabel     { width: 10; color: $text-muted; }
    NodeCard .mpct       { width: 6; text-align: right; color: $text-muted; }
    """

    def __init__(self, node: dict) -> None:
        super().__init__()
        self._node = node

    def compose(self) -> ComposeResult:
        n = self._node
        res = n.get("totalResources") or n.get("availableResources") or {}
        status = n.get("status", "Unknown")
        color = _STATUS_COLOR.get(status, "white")

        yield Label(
            f"[bold cyan]{n.get('name', n.get('id', '?'))}[/]  "
            f"[{color}]{status}[/]",
            markup=True,
            classes="node-name",
        )
        yield Label(
            f"{n.get('publicIp', '?')} · {n.get('region', '?')} · "
            f"{n.get('architecture', 'x86_64')} · v{n.get('agentVersion', '?')}",
            classes="node-meta",
        )

        for label, key in [("CPU", "cpuUsagePct"), ("Memory", "memUsagePct"), ("Storage", "storageUsagePct")]:
            pct = float(res.get(key, 0))
            with Horizontal(classes="metric-row"):
                yield Label(label, classes="mlabel")
                bar = ProgressBar(total=100, show_eta=False, show_percentage=False)
                yield bar
                yield Label(f"{pct:.0f}%", classes="mpct")
            bar.advance(pct)

        # Obligation badges
        obligations = n.get("systemVmObligations") or []
        if obligations:
            parts = []
            for o in obligations:
                role = o.get("roleName", "?")
                st = o.get("statusName", "?")
                c = "green" if st == "Active" else ("yellow" if st == "Deploying" else "red")
                parts.append(f"[{c}]{role}:{st}[/]")
            yield Label("  ".join(parts), markup=True, classes="node-meta")


class NodesScreen(Widget):
    _is_mounted: bool = False
    _running: bool = False

    """Filterable node fleet view."""

    BINDINGS = [("r", "refresh", "Refresh")]

    _nodes: reactive[list] = reactive([])
    _filter_status: reactive[str] = reactive("All")
    _filter_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Input(placeholder="Filter nodes…", id="search-input")
            for label in _FILTER_LABELS:
                yield Button(label, id=f"btn-{label.lower()}", variant="default")

        yield Label("", id="node-count", classes="section-title")
        yield ScrollableContainer(id="node-list")

    def on_mount(self) -> None:
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_orchestrator:
            return
        client = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            self._nodes = await client.list_nodes()
        except ApiError:
            pass
        finally:
            await client.close()
        self._render()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value.lower()
        self._render()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        for label in _FILTER_LABELS:
            btn_id = f"btn-{label.lower()}"
            self.query_one(f"#{btn_id}", Button).variant = "default"
        event.button.variant = "primary"
        self._filter_status = event.button.label  # type: ignore[arg-type]
        self._render()

    def _render(self) -> None:
        filtered = [
            n for n in self._nodes
            if self._matches(n)
        ]
        container = self.query_one("#node-list", ScrollableContainer)
        container.remove_children()
        for node in filtered:
            container.mount(NodeCard(node))
        self.query_one("#node-count", Label).update(
            f"Showing {len(filtered)} / {len(self._nodes)} nodes"
        )

    def _matches(self, node: dict) -> bool:
        status = node.get("status", "")
        if self._filter_status != "All" and status != self._filter_status:
            return False
        if self._filter_text:
            searchable = " ".join([
                node.get("name", ""),
                node.get("id", ""),
                node.get("region", ""),
                node.get("publicIp", ""),
            ]).lower()
            if self._filter_text not in searchable:
                return False
        return True