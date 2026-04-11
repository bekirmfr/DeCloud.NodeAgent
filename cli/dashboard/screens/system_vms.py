"""
screens/system_vms.py — System VM obligation status screen.

Shows DHT, Relay, and BlockStore obligations for each node, sourced
from GET /api/nodes/me/obligations (node agent) or embedded in node
records from the orchestrator.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static

from config import cfg
from api.orchestrator import OrchestratorClient, ApiError
from api.node_agent import NodeAgentClient


_ROLE_ICON = {"Dht": "⬡  DHT Node", "Relay": "⇄  Relay VM", "BlockStore": "◫  BlockStore"}
_STATUS_COLOR = {
    "Active": "green",
    "Deploying": "yellow",
    "Pending": "cyan",
    "Failed": "red",
}


class ObligationCard(Static):
    """Displays one system VM obligation."""

    DEFAULT_CSS = """
    ObligationCard {
        border: solid $panel;
        padding: 1 2;
        margin-bottom: 1;
        width: 1fr;
    }
    ObligationCard .card-title { text-style: bold; }
    ObligationCard .card-meta  { color: $text-muted; }
    ObligationCard .card-err   { color: $error; }
    """

    def __init__(self, obligation: dict) -> None:
        super().__init__()
        self._o = obligation

    def compose(self) -> ComposeResult:
        o = self._o
        role = o.get("roleName", "Unknown")
        status = o.get("statusName", "Unknown")
        color = _STATUS_COLOR.get(status, "white")
        icon = _ROLE_ICON.get(role, role)

        yield Label(
            f"{icon}  [{color}]{status}[/]",
            markup=True,
            classes="card-title",
        )
        yield Label(
            f"VM: {o.get('vmId', 'not deployed')}",
            classes="card-meta",
        )
        yield Label(
            f"Failures: {o.get('failureCount', 0)}  "
            f"Deployed: {o.get('deployedAt', '—')}  "
            f"Active: {o.get('activeAt', '—')}",
            classes="card-meta",
        )
        if err := o.get("lastError"):
            yield Label(f"Error: {err}", classes="card-err")


class NodeObligationsPanel(Static):
    """Obligations for one node."""

    DEFAULT_CSS = """
    NodeObligationsPanel {
        border: solid $panel;
        padding: 1 2;
        margin-bottom: 1;
    }
    NodeObligationsPanel .node-header { color: $accent; text-style: bold; }
    """

    def __init__(self, node_name: str, obligations: list) -> None:
        super().__init__()
        self._name = node_name
        self._obligations = obligations

    def compose(self) -> ComposeResult:
        yield Label(self._name, classes="node-header")
        if not self._obligations:
            yield Label("  No obligations", classes="card-meta")
        for o in self._obligations:
            yield ObligationCard(o)


class SystemVmsScreen(Widget):
    _is_mounted: bool = False
    _running: bool = False

    """System VM (DHT / Relay / BlockStore) obligation overview."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label("System VM Obligations", classes="section-title")
        yield ScrollableContainer(id="oblig-list")

    def on_mount(self) -> None:
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        container = self.query_one("#oblig-list", ScrollableContainer)
        container.remove_children()

        # Node-only mode: fetch from node agent directly
        if cfg.node_only and cfg.has_node_agent:
            await self._load_from_node_agent(container)
            return

        # Orchestrator mode: iterate all nodes
        if not cfg.has_orchestrator:
            container.mount(Label("No orchestrator configured.", classes="card-meta"))
            return

        client = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            nodes = await client.list_nodes()
        except ApiError as e:
            container.mount(Label(f"[red]Error: {e.message}[/]", markup=True))
            await client.close()
            return
        await client.close()

        for node in nodes:
            obligations = node.get("systemVmObligations") or []
            container.mount(
                NodeObligationsPanel(
                    node.get("name", node.get("id", "?")),
                    obligations,
                )
            )

    async def _load_from_node_agent(self, container: ScrollableContainer) -> None:
        client = NodeAgentClient(cfg.node_url)
        try:
            obligations = await client.get_obligations()
            container.mount(NodeObligationsPanel("This Node", obligations))
        except Exception as e:
            container.mount(Label(f"[red]Node Agent error: {e}[/]", markup=True))
        finally:
            await client.close()