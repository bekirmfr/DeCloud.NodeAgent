"""
screens/system_vms.py — System VM obligations for this node only.

Data: node agent GET /api/nodes/me/obligations
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static

from config import cfg
from api.node_agent import NodeAgentClient


_ROLE_LABEL = {"Dht": "DHT Node", "Relay": "Relay VM", "BlockStore": "BlockStore VM"}
_STATUS_COLOR = {"Active": "green", "Deploying": "yellow", "Pending": "cyan", "Failed": "red"}


class ObligationCard(Vertical):
    _is_mounted: bool = False
    _running: bool = False

    DEFAULT_CSS = """
    ObligationCard {
        border: solid $panel;
        padding: 1 2;
        margin-bottom: 1;
    }
    ObligationCard .oc-title { text-style: bold; }
    ObligationCard .oc-meta  { color: $text-muted; }
    ObligationCard .oc-err   { color: $error; }
    """

    def __init__(self, obligation: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._o = obligation

    def compose(self) -> ComposeResult:
        o      = self._o
        role   = o.get("roleName", "?")
        status = o.get("statusName", "?")
        color  = _STATUS_COLOR.get(status, "white")
        label  = _ROLE_LABEL.get(role, role)

        yield Label(
            f"[bold]{label}[/]   [{color}]{status}[/]",
            classes="oc-title", markup=True,
        )
        yield Label(f"VM ID: {o.get('vmId') or 'not yet deployed'}", classes="oc-meta")
        yield Label(
            f"Failures: {o.get('failureCount', 0)}   "
            f"Deployed: {o.get('deployedAt','—')}   "
            f"Active: {o.get('activeAt','—')}",
            classes="oc-meta",
        )
        if err := o.get("lastError"):
            yield Label(f"Error: {err}", classes="oc-err")


class SystemVmsScreen(Vertical):
    _is_mounted: bool = False
    _running: bool = False

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label("System VM Obligations — this node", classes="section-title")
        yield Label("", id="sysvm-status")
        yield ScrollableContainer(id="oblig-list")

    def on_mount(self) -> None:
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        container = self.query_one("#oblig-list", ScrollableContainer)
        container.remove_children()

        if not cfg.has_node_agent:
            container.mount(Label("Node agent URL not configured.", classes="oc-meta"))
            return

        na = NodeAgentClient(cfg.node_url)
        try:
            obligations = await na.get_obligations()
        except Exception as e:
            container.mount(Label(f"Error: {e}", classes="oc-err"))
            return
        finally:
            await na.close()

        if not obligations:
            container.mount(Label("No obligations assigned to this node.", classes="oc-meta"))
            return

        for o in obligations:
            container.mount(ObligationCard(o))

        active = sum(1 for o in obligations if o.get("statusName") == "Active")
        try:
            self.query_one("#sysvm-status", Label).update(
                f"{active} / {len(obligations)} obligations active"
            )
        except Exception:
            pass