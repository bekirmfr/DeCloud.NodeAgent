"""screens/dashboard.py — Node-centric overview screen."""

from __future__ import annotations

import asyncio
import re
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Label, Log, ProgressBar, Static

from config import cfg
from api.node_agent import NodeAgentClient
from api.orchestrator import OrchestratorClient
from screens._base import BaseScreen

_SYS_VM_TYPES = {5, 6, 8}


def _is_running(vm: dict) -> bool:
    s = vm.get("state", vm.get("status", ""))
    return s in (1, "Running", "running")


def _is_system(vm: dict) -> bool:
    try:
        return int((vm.get("spec") or {}).get("vmRole", -1)) in _SYS_VM_TYPES
    except (TypeError, ValueError):
        return False


def _pct(used: float, total: float) -> float:
    return (used / total * 100) if total > 0 else 0.0


class DashboardScreen(BaseScreen):
    ACTIVE_LABEL = "Dashboard"
    BINDINGS = [("r", "refresh", "Refresh")]

    def compose_content(self) -> ComposeResult:
        # Identity bar
        yield Label("", id="id-bar")

        # Stat cards
        with Horizontal(id="stat-row"):
            for cid, lbl in [
                ("c-sys", "System VMs"),
                ("c-vms", "Tenant VMs"),
                ("c-rev", "Revenue / 24h"),
                ("c-net", "Network"),
            ]:
                with Vertical(classes="stat-card", id=cid):
                    yield Label(lbl, classes="card-label")
                    yield Label("—", classes="card-value")
                    yield Label("", classes="card-sub")

        # Mid row
        with Horizontal(id="mid-row"):
            with Vertical(id="gauges-panel"):
                yield Label("Resources", classes="section-title")
                for gid, glbl in [
                    ("g-cpu", "CPU"), ("g-gpu", "GPU"),
                    ("g-stor", "Storage"), ("g-net", "Network"),
                ]:
                    with Horizontal(id=gid, classes="gauge-row"):
                        yield Label(glbl, classes="gauge-label")
                        yield ProgressBar(total=100, show_eta=False, show_percentage=False,
                                          id=f"{gid}-bar")
                        yield Label("0%", classes="gauge-pct", id=f"{gid}-pct")

            with Vertical(id="events-panel"):
                yield Label("Recent Events", classes="section-title")
                yield Log(id="event-log", max_lines=cfg.log_lines)

        yield Label("Node Fleet  (your nodes)", classes="section-title")
        yield DataTable(id="fleet-table", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one("#fleet-table", DataTable).add_columns(
            "Node", "Region", "VMs", "CPU%", "Mem%", "Status", "Last Seen"
        )
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        na = NodeAgentClient(cfg.node_url) if cfg.has_node_agent else None
        oc = OrchestratorClient(cfg.orchestrator_url, cfg.token) if cfg.has_orchestrator else None
        summary, obligations, balance, nodes = {}, [], {}, []
        try:
            coros = [
                na.get_summary()     if na else _noop({}),
                na.get_obligations() if na else _noop([]),
                oc.get_balance()     if oc else _noop({}),
                oc.list_nodes()      if oc else _noop([]),
            ]
            r = await asyncio.gather(*coros, return_exceptions=True)
            summary, obligations, balance, nodes = [
                v if not isinstance(v, Exception) else d
                for v, d in zip(r, [{}, [], {}, []])
            ]
        finally:
            if na: await na.close()
            if oc: await oc.close()

        self._apply_identity(summary)
        self._apply_stats(summary, obligations, balance)
        self._apply_gauges(summary)
        self._apply_fleet(summary, nodes)

    def _apply_identity(self, s: dict) -> None:
        nid  = s.get("nodeId", "—")
        wal  = s.get("walletAddress", "—")
        orch = s.get("orchestrator", {})
        conn = orch.get("connected", False) if isinstance(orch, dict) else False
        short_nid = (nid[:14] + "…") if len(nid) > 16 else nid
        short_wal = (wal[:8] + "…" + wal[-4:]) if len(wal) > 14 else wal
        orch_str  = "Orch: connected" if conn else "Orch: offline"
        try:
            self.query_one("#id-bar", Label).update(
                f"Node: {short_nid}   Wallet: {short_wal}   {orch_str}"
            )
        except Exception:
            pass

    def _apply_stats(self, s: dict, obligations: list, balance: dict) -> None:
        snap    = s.get("snapshot") or s
        all_vms = s.get("vms") or []

        sys_run   = sum(1 for v in all_vms if _is_system(v) and _is_running(v))
        sys_total = len(obligations) or sum(1 for v in all_vms if _is_system(v))
        self._card("c-sys", f"{sys_run} / {sys_total}", "running / obligations")

        tenant  = [v for v in all_vms if not _is_system(v)]
        ten_run = sum(1 for v in tenant if _is_running(v))
        self._card("c-vms", f"{ten_run} / {len(tenant)}", "running / scheduled")

        try:
            pending = float(balance.get("pendingBalance", balance.get("pending24h", 0)))
            self._card("c-rev", f"${pending:.2f}", "USDC pending")
        except (TypeError, ValueError):
            self._card("c-rev", "—", "USDC pending")

        net = s.get("network", {})
        if isinstance(net, dict):
            rx = net.get("totalRxMbps", net.get("rxMbps", 0))
            tx = net.get("totalTxMbps", net.get("txMbps", 0))
            self._card("c-net", f"↑{tx:.0f} / ↓{rx:.0f}", "Mbps out / in")
        else:
            self._card("c-net", "—", "Mbps out / in")

    def _apply_gauges(self, s: dict) -> None:
        snap = s.get("snapshot") or s
        cpu  = float(snap.get("virtualCpuUsagePercent", 0))
        mem_pct  = _pct(snap.get("usedMemoryBytes", 0), snap.get("totalMemoryBytes", 1))
        stor_pct = _pct(snap.get("usedStorageBytes", 0), snap.get("totalStorageBytes", 1))

        gpu_pct = 0.0
        gpus = snap.get("gpuUsage") or []
        if gpus:
            used  = sum(float(g.get("memoryAllocated", 0)) for g in gpus)
            quota = sum(float(g.get("memoryQuota", 1)) for g in gpus)
            gpu_pct = _pct(used, max(quota, 1))

        net = s.get("network", {})
        net_pct = min(float(net.get("txUtilPct", 0)), 100.0) if isinstance(net, dict) else 0.0

        for gid, pct in [
            ("g-cpu", cpu), ("g-gpu", gpu_pct),
            ("g-stor", stor_pct), ("g-net", net_pct),
        ]:
            self._gauge(gid, pct)

        try:
            self.query_one("#event-log", Log).write_line(
                f"cpu={cpu:.1f}%  mem={mem_pct:.1f}%  stor={stor_pct:.1f}%"
            )
        except Exception:
            pass

    def _apply_fleet(self, s: dict, nodes: list) -> None:
        my_wallet = (s.get("walletAddress") or "").lower()
        fleet = [n for n in nodes if (n.get("walletAddress") or "").lower() == my_wallet] \
                if my_wallet and nodes else nodes[:20]
        table = self.query_one("#fleet-table", DataTable)
        table.clear()
        for n in fleet:
            status = n.get("status", "?")
            color  = {"Online": "green", "Degraded": "yellow", "Offline": "red"}.get(status, "white")
            res    = n.get("availableResources") or n.get("totalResources") or {}
            table.add_row(
                (n.get("name") or n.get("id") or "?")[:18],
                n.get("region", "—"),
                str(n.get("runningVmCount", "—")),
                f"{res.get('cpuUsagePct', 0):.0f}%",
                f"{res.get('memUsagePct', 0):.0f}%",
                f"[{color}]{status}[/]",
                n.get("lastSeenAgo", "—"),
            )

    def _card(self, cid: str, value: str, sub: str = "") -> None:
        try:
            self.query_one(f"#{cid} .card-value", Label).update(value)
            self.query_one(f"#{cid} .card-sub",   Label).update(sub)
        except Exception:
            pass

    def _gauge(self, gid: str, pct: float) -> None:
        pct = min(max(pct, 0.0), 100.0)
        try:
            self.query_one(f"#{gid}-bar", ProgressBar).progress = pct
            self.query_one(f"#{gid}-pct", Label).update(f"{pct:.0f}%")
        except Exception:
            pass


async def _noop(val):
    return val