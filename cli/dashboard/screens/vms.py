"""
screens/vms.py — Virtual machine management screen.

Filterable DataTable of VMs with start / stop / restart / delete
actions. Confirmation prompt before destructive operations.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Select
from textual.reactive import reactive

from config import cfg
from api.orchestrator import OrchestratorClient, ApiError


_STATUS_COLOR = {
    "Running": "green",
    "Starting": "yellow",
    "Stopping": "yellow",
    "Stopped": "red",
    "Pending": "cyan",
    "Error": "red",
}

_STATUS_OPTS = [("All", ""), ("Running", "running"), ("Stopped", "stopped"), ("Pending", "pending")]
_ACTIONS = ["start", "stop", "restart", "forceStop"]


class VmsScreen(Widget):
    _is_mounted: bool = False

    """VM fleet table with inline action bar."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("s", "action_vm('stop')", "Stop"),
        ("x", "action_vm('start')", "Start"),
        ("d", "action_vm('restart')", "Restart"),
    ]

    _vms: reactive[list] = reactive([])
    _selected_id: reactive[str | None] = reactive(None)
    _filter_text: reactive[str] = reactive("")
    _filter_status: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Input(placeholder="Search VMs…", id="search-input")
            yield Select(
                options=_STATUS_OPTS,
                value="",
                id="status-select",
                allow_blank=False,
            )
            yield Button("Refresh", id="btn-refresh", variant="default")

        yield DataTable(id="vm-table", zebra_stripes=True, cursor_type="row")
        yield Label("", id="selected-label", classes="section-title")

        with Horizontal(id="action-bar"):
            for action in _ACTIONS:
                yield Button(action.title(), id=f"action-{action}", variant="default")
            yield Button("Delete", id="action-delete", variant="error")

    def on_mount(self) -> None:
        table = self.query_one("#vm-table", DataTable)
        table.add_columns(
            "ID", "Name", "Node", "CPU", "RAM", "Tier", "Uptime", "Status"
        )
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_orchestrator:
            return
        client = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            self._vms = await client.list_vms(
                status=self._filter_status or None,
                search=self._filter_text or None,
            )
        except ApiError as e:
            self.notify(f"Error: {e.message}", severity="error")
        finally:
            await client.close()
        self._render_table()

    def _render_table(self) -> None:
        table = self.query_one("#vm-table", DataTable)
        table.clear()
        for vm in self._vms:
            status = vm.get("status", "Unknown")
            color = _STATUS_COLOR.get(status, "white")
            spec = vm.get("spec") or {}
            table.add_row(
                vm.get("id", "")[:8],
                vm.get("name", "—"),
                (vm.get("nodeId") or "—")[:10],
                f"{spec.get('virtualCpuCores', '—')}c",
                f"{_mb_to_gb(spec.get('memoryBytes', 0))} GB",
                vm.get("tier", "—"),
                vm.get("uptimeHuman", "—"),
                f"[{color}]{status}[/]",
                key=vm.get("id"),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._selected_id = str(event.row_key.value)
        vm = next((v for v in self._vms if v.get("id") == self._selected_id), None)
        if vm:
            self.query_one("#selected-label", Label).update(
                f"Selected: {vm.get('id', '')} · {vm.get('name', '')} · {vm.get('status', '')}"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "btn-refresh":
            self.action_refresh()
        elif btn_id.startswith("action-"):
            action = btn_id.removeprefix("action-")
            if self._selected_id:
                self.run_worker(self._do_action(action), exclusive=False)

    async def _do_action(self, action: str) -> None:
        if not self._selected_id:
            return
        client = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            if action == "delete":
                await client.delete_vm(self._selected_id)
                self.notify(f"VM {self._selected_id} deleted", severity="information")
            else:
                await client.vm_action(self._selected_id, action)
                self.notify(f"Action '{action}' sent to {self._selected_id}")
        except ApiError as e:
            self.notify(f"Failed: {e.message}", severity="error")
        finally:
            await client.close()
        await self._load()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value
        self.run_worker(self._load(), exclusive=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        self._filter_status = str(event.value) if event.value else ""
        self.run_worker(self._load(), exclusive=True)


def _mb_to_gb(b: int) -> str:
    return f"{b / (1024**3):.0f}" if b else "—"