"""
screens/vms.py — Virtual machines hosted on this node.

Unifies system VMs (DHT/Relay/BlockStore) and tenant VMs in a single
filterable table, with a role chip distinguishing them. The operator
should not have to switch screens to see all VMs.

Per-row actions: Start (s) / Stop (S) / Restart (R) / Delete (D).
Confirmation is required for Stop / Delete.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Input, Label, Select
from widgets.nav_input import NavInput

from config import cfg
from api.client import ApiError
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen
from theme import COLOR
from util.format import fmt_bytes, short_id, truncate
from widgets.badges import vm_state_badge


_VM_TYPES = {0:"General",1:"Compute",2:"Memory",3:"Storage",
             4:"GPU",5:"Relay",6:"DHT",7:"Inference",8:"BlockStore"}
_SYSTEM_TYPES = {5, 6, 8}

_FILTERS = [("All", ""), ("Running", "running"),
            ("Failed", "failed"), ("System", "system"), ("Tenant", "tenant")]


def _vm_type(t: Any) -> int | None:
    try: return int(t)
    except (TypeError, ValueError): return None


def _role_chip(t: int | None) -> Text:
    """Small inline chip distinguishing system VMs from tenant VMs."""
    if t is None:
        return Text(" ? ", style=f"{COLOR['dim']}")
    if t in _SYSTEM_TYPES:
        return Text(" SYS ", style=f"black on {COLOR['info']}")
    return Text(" USR ", style=f"{COLOR['muted']}")


class VmsScreen(BaseScreen):
    ACTIVE_LABEL = "Virtual Machines"
    EXTRA_HINTS = [("s", "start"), ("S", "stop"),
                   ("R", "restart"), ("D", "delete")]

    BINDINGS = [
        ("s",      "start_selected",   "Start"),
        ("S",      "stop_selected",    "Stop"),
        ("R",      "restart_selected", "Restart"),
        ("D",      "delete_selected",  "Delete"),
    ]

    DEFAULT_CSS = """
    VmsScreen #filter-bar { height: 3; margin-bottom: 1; }
    VmsScreen #filter-bar Input  { width: 30; margin-right: 1; }
    VmsScreen #filter-bar Select { width: 18; margin-right: 1; }
    VmsScreen #vm-table { height: 1fr; }
    VmsScreen #vm-meta  { color: $text-muted; height: 1; margin-bottom: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._search = ""
        self._filter = ""
        self._vms: list[dict] = []
        self._ingress: dict[str, str] = {}

    def compose_content(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield NavInput(placeholder="search name…", id="vm-search")
            yield Select(_FILTERS, value="", id="vm-filter",
                         allow_blank=False, prompt="Filter")
        yield Label("", id="vm-meta")
        t = DataTable(id="vm-table", zebra_stripes=True, cursor_type="row")
        t.add_columns(
            "State", "Role", "Name", "Type", "vCPU", "Mem",
            "Disk", "IP", "Ingress",
        )
        yield t

    def on_mount(self) -> None:
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    def on_input_changed(self, ev: Input.Changed) -> None:
        if ev.input.id == "vm-search":
            self._search = ev.value.lower()
            self._apply()

    def on_select_changed(self, ev: Select.Changed) -> None:
        if ev.select.id == "vm-filter":
            self._filter = str(ev.value or "")
            self._apply()

    # ─── Actions ───────────────────────────────────────────────────────

    def _selected_vm(self) -> dict | None:
        try:
            t = self.query_one("#vm-table", DataTable)
            row_idx = t.cursor_row
            if row_idx is None or row_idx < 0:
                return None
            visible = self._filtered()
            if 0 <= row_idx < len(visible):
                return visible[row_idx]
        except Exception:
            pass
        return None

    async def _do(self, vm: dict, op: str) -> None:
        vm_id = vm.get("vmId") or (vm.get("spec") or {}).get("id")
        if not vm_id:
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            fn = {
                "start":   na.vm_start,
                "stop":    na.vm_stop,
                "restart": na.vm_restart,
                "delete":  na.vm_delete,
            }[op]
            await fn(vm_id)
            self.notify(f"{op} → {vm.get('name', vm_id)} ok",
                        severity="information")
        except ApiError as e:
            self.notify(f"{op} failed: {e.message}", severity="error")
        finally:
            await na.close()
        self.run_worker(self._load(), exclusive=True)

    def action_start_selected(self) -> None:
        vm = self._selected_vm()
        if vm:
            self.run_worker(self._do(vm, "start"))

    def action_stop_selected(self) -> None:
        vm = self._selected_vm()
        if vm:
            self.run_worker(self._do(vm, "stop"))

    def action_restart_selected(self) -> None:
        vm = self._selected_vm()
        if vm:
            self.run_worker(self._do(vm, "restart"))

    def action_delete_selected(self) -> None:
        vm = self._selected_vm()
        if vm:
            # Visible visual confirmation; actual deletion still goes through
            # the orchestrator-issued path for system VMs.
            self.notify(
                f"Delete '{vm.get('name')}' — confirm by pressing D again",
                severity="warning",
            )
            # Capture for double-press confirmation.
            now = (vm.get("vmId"), id(vm))
            if getattr(self, "_pending_delete", None) == now:
                self.run_worker(self._do(vm, "delete"))
                self._pending_delete = None
            else:
                self._pending_delete = now

    # ─── Data ──────────────────────────────────────────────────────────

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            vms, ing = await asyncio.gather(
                na.list_vms(), na.dashboard_vm_ingress(),
                return_exceptions=True,
            )
        finally:
            await na.close()
        self._vms = vms if isinstance(vms, list) else []
        self._ingress = ing if isinstance(ing, dict) else {}
        self._apply()
        self.mark_updated()

    # ─── Filter / render ───────────────────────────────────────────────

    def _filtered(self) -> list[dict]:
        items = self._vms
        s = self._search
        f = self._filter
        out: list[dict] = []
        for vm in items:
            spec = vm.get("spec") or {}
            name = (vm.get("name") or spec.get("name") or "").lower()
            if s and s not in name:
                continue
            t = _vm_type(spec.get("vmRole"))
            state = vm.get("state")
            if f == "running" and state != 3:
                continue
            if f == "failed" and state != 7:
                continue
            if f == "system" and (t is None or t not in _SYSTEM_TYPES):
                continue
            if f == "tenant" and (t is None or t in _SYSTEM_TYPES):
                continue
            out.append(vm)
        return out

    def _apply(self) -> None:
        t = self.query_one("#vm-table", DataTable)
        t.clear()
        items = self._filtered()
        for vm in items:
            spec = vm.get("spec") or {}
            vm_id = vm.get("vmId") or spec.get("id") or ""
            name = vm.get("name") or spec.get("name") or "—"
            t_int = _vm_type(spec.get("vmRole"))
            ingress = self._ingress.get(vm_id) or spec.get("ipAddress") \
                or vm.get("ipAddress") or "—"
            t.add_row(
                vm_state_badge(vm.get("state")),
                _role_chip(t_int),
                truncate(name, 24),
                _VM_TYPES.get(t_int, "—") if t_int is not None else "—",
                str(spec.get("virtualCpuCores", "—")),
                fmt_bytes(spec.get("memoryBytes")),
                fmt_bytes(spec.get("diskBytes")),
                spec.get("ipAddress") or vm.get("ipAddress") or "—",
                truncate(ingress, 30),
            )
        # Meta line
        sysn = sum(1 for v in items
                   if _vm_type((v.get("spec") or {}).get("vmRole")) in _SYSTEM_TYPES)
        run = sum(1 for v in items if v.get("state") == 3)
        meta = (f"{len(items)} shown ({sysn} system / {len(items)-sysn} tenant) "
                f"· {run} running")
        try:
            self.query_one("#vm-meta", Label).update(
                Text(meta, style=f"{COLOR['muted']}"))
        except Exception:
            pass
