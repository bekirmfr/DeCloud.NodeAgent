"""screens/vms.py — VMs on this node (from node agent summary)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Input, Label, Select

from config import cfg
from api.node_agent import NodeAgentClient
from api.orchestrator import OrchestratorClient, ApiError
from screens._base import BaseScreen

_STATE_COLOR = {
    "Running": "green", "running": "green",
    "Stopped": "red",   "stopped": "red",
    "Failed":  "red",   "Starting": "yellow", "Stopping": "yellow",
}
_VM_TYPES = {0:"General",1:"Compute",2:"Memory",3:"Storage",
             4:"GPU",5:"Relay",6:"DHT",7:"Inference",8:"BlockStore"}
_STATUS_OPTS = [("All",""),("Running","running"),("Stopped","stopped")]


def _state_str(vm: dict) -> str:
    s = vm.get("state", vm.get("status", "?"))
    if isinstance(s, int):
        return {0:"Stopped",1:"Running",2:"Starting",3:"Stopping",
                4:"Failed",5:"Deleted"}.get(s, str(s))
    return str(s)


def _fmt_bytes(b) -> str:
    try:
        b = int(b)
    except (TypeError, ValueError):
        return "—"
    for u in ("B","KB","MB","GB","TB"):
        if b < 1024: return f"{b:.0f}{u}"
        b //= 1024
    return f"{b}PB"


class VmsScreen(BaseScreen):
    ACTIVE_LABEL = "Virtual Machines"
    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._vms: list = []
        self._selected_id: str | None = None
        self._filter_text: str = ""
        self._filter_status: str = ""

    def compose_content(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Input(placeholder="Search VMs…", id="search-input")
            yield Select(options=_STATUS_OPTS, value="", id="status-select", allow_blank=False)
            yield Button("Refresh", id="btn-refresh")

        yield DataTable(id="vm-table", zebra_stripes=True, cursor_type="row")
        yield Label("", id="selected-label", classes="section-title")

        with Horizontal(id="action-bar"):
            yield Button("Start",   id="action-start")
            yield Button("Stop",    id="action-stop")
            yield Button("Restart", id="action-restart")

    def on_mount(self) -> None:
        self.query_one("#vm-table", DataTable).add_columns(
            "ID", "Name", "Type", "CPU", "RAM", "IP", "State"
        )
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            s = await na.get_summary()
            self._vms = s.get("vms") or []
        except Exception as e:
            self.notify(str(e), severity="error")
        finally:
            await na.close()
        self._render_table()

    def _render_table(self) -> None:
        table = self.query_one("#vm-table", DataTable)
        table.clear()
        for vm in self._filtered():
            state = _state_str(vm)
            color = _STATE_COLOR.get(state, "white")
            spec  = vm.get("spec") or {}
            vm_id = vm.get("vmId") or vm.get("id") or "?"
            table.add_row(
                vm_id[:10],
                (vm.get("name") or spec.get("name") or "?")[:20],
                _VM_TYPES.get(int(spec.get("vmType", 0)), "?"),
                str(spec.get("virtualCpuCores", "—")),
                _fmt_bytes(spec.get("memoryBytes")),
                vm.get("ipAddress") or spec.get("ipAddress") or "—",
                f"[{color}]{state}[/]",
                key=vm_id,
            )

    def _filtered(self) -> list:
        r = self._vms
        if self._filter_status:
            r = [v for v in r if _state_str(v).lower() == self._filter_status]
        if self._filter_text:
            r = [v for v in r if self._filter_text in str(v).lower()]
        return r

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._selected_id = str(event.row_key.value) if event.row_key else None
        try:
            self.query_one("#selected-label", Label).update(
                f"Selected: {self._selected_id}"
            )
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "btn-refresh":
            self.action_refresh()
        elif bid.startswith("action-") and self._selected_id:
            self.run_worker(self._do_action(bid.removeprefix("action-")), exclusive=False)

    async def _do_action(self, action: str) -> None:
        if not cfg.has_orchestrator:
            self.notify("Orchestrator not configured", severity="warning")
            return
        oc = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            await oc.vm_action(self._selected_id, action)
            self.notify(f"'{action}' sent")
        except ApiError as e:
            self.notify(f"Failed: {e.message}", severity="error")
        finally:
            await oc.close()
        await self._load()

    def on_input_changed(self, e: Input.Changed) -> None:
        self._filter_text = e.value.lower()
        self._render_table()

    def on_select_changed(self, e: Select.Changed) -> None:
        self._filter_status = str(e.value) if e.value else ""
        self._render_table()