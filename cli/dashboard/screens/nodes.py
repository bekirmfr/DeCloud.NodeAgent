"""screens/nodes.py — Nodes owned by same wallet."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Input, Label, ProgressBar, Static

from config import cfg
from api.node_agent import NodeAgentClient
from api.orchestrator import OrchestratorClient
from screens._base import BaseScreen

_STATUS_COLOR = {"Online": "green", "Degraded": "yellow", "Offline": "red"}
_FILTERS = ["All", "Online", "Degraded", "Offline"]


class NodeCard(Static):
    DEFAULT_CSS = """
    NodeCard { border: solid $panel; padding: 1 2; margin-bottom: 1; }
    NodeCard .nc-head { text-style: bold; color: $accent; }
    NodeCard .nc-meta { color: $text-muted; }
    """

    def __init__(self, node: dict, is_current: bool = False) -> None:
        super().__init__()
        self._node = node
        self._is_current = is_current

    def compose(self) -> ComposeResult:
        n      = self._node
        status = n.get("status", "?")
        color  = _STATUS_COLOR.get(status, "white")
        marker = "  ← this node" if self._is_current else ""
        res    = n.get("totalResources") or n.get("availableResources") or {}
        yield Label(
            f"[bold]{(n.get('name') or n.get('id') or '?')[:32]}[/]  [{color}]{status}[/]{marker}",
            classes="nc-head", markup=True,
        )
        yield Label(
            f"{n.get('publicIp','?')}  ·  {n.get('region','?')}  ·  v{n.get('agentVersion','?')}",
            classes="nc-meta",
        )
        for lbl, key in [("CPU", "cpuUsagePct"), ("Mem", "memUsagePct")]:
            pct = float(res.get(key, 0))
            with Horizontal():
                yield Label(f"{lbl} {pct:.0f}%", classes="nc-meta")
        yield Label(f"VMs: {n.get('runningVmCount', 0)}", classes="nc-meta")


class NodesScreen(BaseScreen):
    ACTIVE_LABEL = "Nodes"
    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._node_data: list = []
        self._my_wallet: str = ""
        self._my_node_id: str = ""
        self._filter_status: str = "All"
        self._filter_text: str = ""

    def compose_content(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Input(placeholder="Filter nodes…", id="search-input")
            for lbl in _FILTERS:
                yield Button(lbl, id=f"btn-{lbl.lower()}", variant="default")
        yield Label("", id="node-count", classes="section-title")
        yield ScrollableContainer(id="node-list")

    def on_mount(self) -> None:
        self.query_one("#btn-all", Button).variant = "primary"
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if cfg.has_node_agent:
            na = NodeAgentClient(cfg.node_url)
            try:
                s = await na.get_summary()
                self._my_wallet  = (s.get("walletAddress") or "").lower()
                self._my_node_id = (s.get("nodeId") or "").lower()
            except Exception:
                pass
            finally:
                await na.close()

        if cfg.has_orchestrator:
            oc = OrchestratorClient(cfg.orchestrator_url, cfg.token)
            try:
                all_nodes = await oc.list_nodes()
                self._node_data = [
                    n for n in all_nodes
                    if not self._my_wallet or (n.get("walletAddress") or "").lower() == self._my_wallet
                ]
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
        try:
            self.query_one("#node-count", Label).update(
                f"Showing {len(filtered)} / {len(self._node_data)} nodes"
            )
        except Exception:
            pass

    def _matches(self, node: dict) -> bool:
        if self._filter_status != "All" and node.get("status") != self._filter_status:
            return False
        if self._filter_text:
            hay = " ".join([node.get("name",""), node.get("id",""),
                            node.get("region",""), node.get("publicIp","")]).lower()
            if self._filter_text not in hay:
                return False
        return True

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value.lower()
        self._render()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if not bid.startswith("btn-"):
            return
        for lbl in _FILTERS:
            self.query_one(f"#btn-{lbl.lower()}", Button).variant = "default"
        event.button.variant = "primary"
        self._filter_status = str(event.button.label)
        self._render()