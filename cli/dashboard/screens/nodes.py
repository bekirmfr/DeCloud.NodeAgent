"""
screens/nodes.py — Nodes owned by the same wallet as this node.

Data: node agent summary (for wallet) + orchestrator node list (filtered by wallet).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Button, Input, Label, ProgressBar, Static

from config import cfg
from api.node_agent import NodeAgentClient
from api.orchestrator import OrchestratorClient

_STATUS_COLOR = {"Online": "green", "Degraded": "yellow", "Offline": "red"}
_FILTERS = ["All", "Online", "Degraded", "Offline"]


class NodeCard(Static):
    _is_mounted: bool = False
    _running: bool = False

    DEFAULT_CSS = """
    NodeCard {
        border: solid $panel;
        padding: 1 2;
        margin-bottom: 1;
    }
    NodeCard .nc-head  { text-style: bold; color: $accent; }
    NodeCard .nc-meta  { color: $text-muted; }
    NodeCard .nc-row   { height: 2; layout: horizontal; margin-top: 1; }
    NodeCard .nc-lbl   { width: 10; color: $text-muted; }
    NodeCard .nc-pct   { width: 6; text-align: right; color: $text-muted; }
    """

    def __init__(self, node: dict, is_current: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._node = node
        self._is_current = is_current

    def compose(self) -> ComposeResult:
        n      = self._node
        status = n.get("status", "?")
        color  = _STATUS_COLOR.get(status, "white")
        marker = "  (this node)" if self._is_current else ""
        res    = n.get("totalResources") or n.get("availableResources") or {}

        yield Label(
            f"[bold]{(n.get('name') or n.get('id') or '?')[:32]}[/]  [{color}]{status}[/]{marker}",
            classes="nc-head", markup=True,
        )
        yield Label(
            f"{n.get('publicIp','?')}  ·  {n.get('region','?')}  ·  "
            f"{n.get('architecture','x86_64')}  ·  v{n.get('agentVersion','?')}",
            classes="nc-meta",
        )

        for label, key in [("CPU", "cpuUsagePct"), ("Memory", "memUsagePct"), ("Storage", "storageUsagePct")]:
            pct = float(res.get(key, 0))
            with Horizontal(classes="nc-row"):
                yield Label(label, classes="nc-lbl")
                bar = ProgressBar(total=100, show_eta=False, show_percentage=False)
                yield bar
                yield Label(f"{pct:.0f}%", classes="nc-pct")
            bar.advance(pct)

        vms = n.get("runningVmCount", 0)
        yield Label(f"VMs running: {vms}", classes="nc-meta")


class NodesScreen(Container):
    _is_mounted: bool = False
    _running: bool = False

    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._node_data: list = []
        self._my_wallet: str = ""
        self._my_node_id: str = ""
        self._filter_status: str = "All"
        self._filter_text: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Input(placeholder="Filter nodes…", id="search-input")
            for label in _FILTERS:
                yield Button(label, id=f"btn-{label.lower()}", variant="default")

        yield Label("", id="node-count", classes="section-title")
        yield ScrollableContainer(id="node-list")

    def on_mount(self) -> None:
        self.query_one("#btn-all", Button).variant = "primary"
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        # Get wallet + node ID from local node agent
        if cfg.has_node_agent:
            na = NodeAgentClient(cfg.node_url)
            try:
                summary = await na.get_summary()
                self._my_wallet  = (summary.get("walletAddress") or "").lower()
                self._my_node_id = (summary.get("nodeId") or "").lower()
            except Exception:
                pass
            finally:
                await na.close()

        # Get fleet from orchestrator, filtered to same wallet
        if cfg.has_orchestrator:
            oc = OrchestratorClient(cfg.orchestrator_url, cfg.token)
            try:
                all_nodes = await oc.list_nodes()
                if self._my_wallet:
                    self._node_data = [
                        n for n in all_nodes
                        if (n.get("walletAddress") or "").lower() == self._my_wallet
                    ]
                else:
                    self._node_data = all_nodes
            except Exception:
                pass
            finally:
                await oc.close()

        self._render()

    def _render(self) -> None:
        filtered = [n for n in self._node_data if self._matches(n)]
        container = self.query_one("#node-list", ScrollableContainer)
        container.remove_children()
        for node in filtered:
            is_current = (node.get("id") or "").lower() == self._my_node_id
            container.mount(NodeCard(node, is_current=is_current))
        self.query_one("#node-count", Label).update(
            f"Showing {len(filtered)} / {len(self._node_data)} nodes"
        )

    def _matches(self, node: dict) -> bool:
        if self._filter_status != "All" and node.get("status") != self._filter_status:
            return False
        if self._filter_text:
            haystack = " ".join([
                node.get("name", ""), node.get("id", ""),
                node.get("region", ""), node.get("publicIp", ""),
            ]).lower()
            if self._filter_text not in haystack:
                return False
        return True

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value.lower()
        self._render()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if not bid.startswith("btn-"):
            return
        for label in _FILTERS:
            self.query_one(f"#btn-{label.lower()}", Button).variant = "default"
        event.button.variant = "primary"
        self._filter_status = str(event.button.label)
        self._render()