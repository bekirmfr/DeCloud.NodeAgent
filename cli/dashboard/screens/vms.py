"""
screens/vms.py — Virtual machines running on this node.

Data source: node agent GET /api/dashboard/summary → vms[]
Actions (start/stop/restart) proxied via orchestrator if available.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Select

from config import cfg
from api.node_agent import NodeAgentClient
from api.orchestrator import OrchestratorClient, ApiError


_STATE_COLOR = {
    "Running": "green", "running": "green",
    "Stopped": "red",   "stopped": "red",
    "Failed":  "red",   "failed":  "red",
    "Starting": "yellow", "Stopping": "yellow",
    "Pending":  "cyan",
}

_STATUS_OPTS = [("All", ""), ("Running", "running"), ("Stopped", "stopped")]

_VM_TYPE_NAMES = {0: "General", 1: "Compute", 2: "Memory", 3: "Storage",
                  4: "GPU", 5: "Relay", 6: "DHT", 7: "Inference", 8: "BlockStore"}


def _state_str(vm: dict) -> str:
    s = vm.get("state", vm.get("status", "?"))
    if isinstance(s, int):
        return {0: "Stopped", 1: "Running", 2: "Starting", 3: "Stopping",
                4: "Failed", 5: "Deleted"}.get(s, str(s))
    return str(s)


def _fmt_bytes(b) -> str:
    try:
        b = int(b)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.0f}{unit}"
        b //= 1024
    return f"{b}PB"


class VmsScreen(Vertical):
    _is_mounted: bool = False
    _running: bool = False

    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._vms: list = []
        self._selected_id: str | None = None
        self._filter_text: str = ""
        self._filter_status: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Input(placeholder="Search VMs…", id="search-input")
            yield Select(options=_STATUS_OPTS, value="", id="status-select", allow_blank=False)
            yield Button("Refresh", id="btn-refresh", variant="default")

        yield DataTable(id="vm-table", zebra_stripes=True, cursor_type="row")
        yield Label("", id="selected-label", classes="section-title")

        with Horizontal(id="action-bar"):
            yield Button("Start",   id="action-start",   variant="default")
            yield Button("Stop",    id="action-stop",    variant="default")
            yield Button("Restart", id="action-restart", variant="default")

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
            summary = await na.get_summary()
            self._vms = summary.get("vms") or []
        except Exception as e:
            self.notify(f"Node agent error: {e}", severity="error")
        finally:
            await na.close()
        self._render_table()

    def _render_table(self) -> None:
        table = self.query_one("#vm-table", DataTable)
        table.clear()
        for vm in self._filtered():
            state  = _state_str(vm)
            color  = _STATE_COLOR.get(state, "white")
            spec   = vm.get("spec") or {}
            vt     = int(spec.get("vmType", 0))
            vm_id  = vm.get("vmId") or vm.get("id") or "?"
            table.add_row(
                vm_id[:10],
                (vm.get("name") or spec.get("name") or "?")[:20],
                _VM_TYPE_NAMES.get(vt, str(vt)),
                str(spec.get("virtualCpuCores", "—")),
                _fmt_bytes(spec.get("memoryBytes")),
                vm.get("ipAddress") or spec.get("ipAddress") or "—",
                f"[{color}]{state}[/]",
                key=vm_id,
            )

    def _filtered(self) -> list:
        result = self._vms
        if self._filter_status:
            result = [v for v in result
                      if _state_str(v).lower() == self._filter_status.lower()]
        if self._filter_text:
            result = [v for v in result
                      if self._filter_text in str(v).lower()]
        return result

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._selected_id = str(event.row_key.value) if event.row_key else None
        vm = next((v for v in self._vms if (v.get("vmId") or v.get("id")) == self._selected_id), None)
        if vm:
            self.query_one("#selected-label", Label).update(
                f"Selected: {self._selected_id}  {vm.get('name','')}"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "btn-refresh":
            self.action_refresh()
            return
        if bid.startswith("action-") and self._selected_id:
            action = bid.removeprefix("action-")
            self.run_worker(self._do_action(action), exclusive=False)

    async def _do_action(self, action: str) -> None:
        if not self._selected_id or not cfg.has_orchestrator:
            if not cfg.has_orchestrator:
                self.notify("Orchestrator not configured — cannot perform VM actions", severity="warning")
            return
        oc = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            await oc.vm_action(self._selected_id, action)
            self.notify(f"'{action}' sent to {self._selected_id}")
        except ApiError as e:
            self.notify(f"Failed: {e.message}", severity="error")
        finally:
            await oc.close()
        await self._load()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value.lower()
        self._render_table()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._filter_status = str(event.value) if event.value else ""
        self._render_table()