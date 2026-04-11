"""
screens/dashboard.py — Node-centric overview.

Header stats  : System VMs (running/obligations), Tenant VMs (running/scheduled),
                Revenue/24h (pending USDC), Network In/Out
Resources     : CPU%, GPU%, Storage%, Network%
Fleet         : Nodes owned by same wallet (orchestrator)
Login bar     : node ID, wallet, orchestrator status
"""

from __future__ import annotations

import asyncio
import re
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import DataTable, Label, ProgressBar, Static, Log

from config import cfg
from api.node_agent import NodeAgentClient
from api.orchestrator import OrchestratorClient


_SYS_VM_TYPES = {5, 6, 8}   # Relay=5, Dht=6, BlockStore=8


def _is_running(vm: dict) -> bool:
    s = vm.get("state", vm.get("status", ""))
    return s in (1, "Running", "running")


def _is_system(vm: dict) -> bool:
    try:
        return int((vm.get("spec") or {}).get("vmType", -1)) in _SYS_VM_TYPES
    except (TypeError, ValueError):
        return False


def _pct(used: float, total: float) -> float:
    return (used / total * 100) if total > 0 else 0.0


def _strip_markup(s: str) -> str:
    return re.sub(r"\[/?[\w ]+\]", "", s)


class StatCard(Static):
    _is_mounted: bool = False
    _running: bool = False

    DEFAULT_CSS = """
    StatCard { border: solid $panel; padding: 1 2; width: 1fr; height: 7; }
    StatCard .card-label { color: $text-muted; height: 1; }
    StatCard .card-value { text-style: bold; height: 2; }
    StatCard .card-sub   { color: $text-muted; height: 1; }
    """

    def __init__(self, label: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="card-label")
        yield Label("—", classes="card-value")
        yield Label("", classes="card-sub")

    def set_value(self, value: str, sub: str = "") -> None:
        try:
            self.query_one(".card-value", Label).update(value)
            self.query_one(".card-sub", Label).update(sub)
        except Exception:
            pass


class GaugeRow(Static):
    _is_mounted: bool = False
    _running: bool = False

    DEFAULT_CSS = """
    GaugeRow { height: 2; layout: horizontal; }
    GaugeRow .gl { width: 12; color: $text-muted; }
    GaugeRow ProgressBar { width: 1fr; }
    GaugeRow .gp { width: 7; text-align: right; color: $text-muted; }
    """

    def __init__(self, label: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="gl")
        yield ProgressBar(total=100, show_eta=False, show_percentage=False)
        yield Label("0%", classes="gp")

    def set_pct(self, pct: float) -> None:
        pct = min(max(pct, 0.0), 100.0)
        try:
            self.query_one(ProgressBar).progress = pct
            self.query_one(".gp", Label).update(f"{pct:.0f}%")
        except Exception:
            pass


class LoginBar(Static):
    _is_mounted: bool = False
    _running: bool = False

    DEFAULT_CSS = """
    LoginBar {
        height: 2;
        background: $surface;
        border-bottom: solid $panel;
        layout: horizontal;
        align: left middle;
        padding: 0 2;
    }
    LoginBar Label { margin-right: 3; color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        yield Label("", id="lb-node")
        yield Label("", id="lb-wallet")
        yield Label("", id="lb-orch")

    def update(self, node_id: str, wallet: str, connected: bool) -> None:
        nid = (node_id[:14] + "…") if len(node_id) > 16 else node_id
        wal = (wallet[:8] + "…" + wallet[-4:]) if len(wallet) > 14 else wallet
        orch = "Orch: connected" if connected else "Orch: offline"
        try:
            self.query_one("#lb-node",   Label).update(f"Node: {nid}")
            self.query_one("#lb-wallet", Label).update(f"Wallet: {wal}")
            self.query_one("#lb-orch",   Label).update(orch)
        except Exception:
            pass


class DashboardScreen(Container):
    _is_mounted: bool = False
    _running: bool = False

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield LoginBar(id="login-bar")

        with Horizontal(id="stat-row"):
            yield StatCard("System VMs",    id="c-sys")
            yield StatCard("Tenant VMs",    id="c-vms")
            yield StatCard("Revenue / 24h", id="c-rev")
            yield StatCard("Network",       id="c-net")

        with Horizontal(id="mid-row"):
            with Vertical(id="gauges-panel"):
                yield Label("Resources", classes="section-title")
                yield GaugeRow("CPU",     id="g-cpu")
                yield GaugeRow("GPU",     id="g-gpu")
                yield GaugeRow("Storage", id="g-stor")
                yield GaugeRow("Network", id="g-net")

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
                na.get_summary()    if na else self._empty({}),
                na.get_obligations() if na else self._empty([]),
                oc.get_balance()    if oc else self._empty({}),
                oc.list_nodes()     if oc else self._empty([]),
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)
            summary     = results[0] if not isinstance(results[0], Exception) else {}
            obligations = results[1] if not isinstance(results[1], Exception) else []
            balance     = results[2] if not isinstance(results[2], Exception) else {}
            nodes       = results[3] if not isinstance(results[3], Exception) else []
        finally:
            if na: await na.close()
            if oc: await oc.close()

        self._render_identity(summary)
        self._render_stats(summary, obligations, balance)
        self._render_gauges(summary)
        self._render_fleet(summary, nodes)

    async def _empty(self, val):
        return val

    def _render_identity(self, s: dict) -> None:
        nid  = s.get("nodeId", "—")
        wal  = s.get("walletAddress", "—")
        orch = s.get("orchestrator", {})
        conn = orch.get("connected", False) if isinstance(orch, dict) else False
        try:
            self.query_one("#login-bar", LoginBar).update(nid, wal, conn)
        except Exception:
            pass

    def _render_stats(self, s: dict, obligations: list, balance: dict) -> None:
        snap    = s.get("snapshot") or s
        all_vms = s.get("vms") or []

        sys_run   = sum(1 for v in all_vms if _is_system(v) and _is_running(v))
        sys_total = len(obligations) or sum(1 for v in all_vms if _is_system(v))
        self._set("c-sys", f"{sys_run} / {sys_total}", "running / obligations")

        tenant     = [v for v in all_vms if not _is_system(v)]
        ten_run    = sum(1 for v in tenant if _is_running(v))
        self._set("c-vms", f"{ten_run} / {len(tenant)}", "running / scheduled")

        try:
            pending = float(balance.get("pendingBalance", balance.get("pending24h", 0)))
            self._set("c-rev", f"${pending:.2f}", "USDC pending")
        except (TypeError, ValueError):
            self._set("c-rev", "—", "USDC pending")

        net = s.get("network", {})
        if isinstance(net, dict):
            rx = net.get("totalRxMbps", net.get("rxMbps", 0))
            tx = net.get("totalTxMbps", net.get("txMbps", 0))
            self._set("c-net", f"↑{tx:.0f} / ↓{rx:.0f}", "Mbps out / in")
        else:
            self._set("c-net", "—", "Mbps out / in")

    def _render_gauges(self, s: dict) -> None:
        snap = s.get("snapshot") or s

        cpu  = float(snap.get("virtualCpuUsagePercent", 0))
        mem_used  = float(snap.get("usedMemoryBytes", 0))
        mem_total = float(snap.get("totalMemoryBytes", 1))
        sto_used  = float(snap.get("usedStorageBytes", 0))
        sto_total = float(snap.get("totalStorageBytes", 1))

        gpu_pct = 0.0
        gpus = snap.get("gpuUsage") or []
        if gpus:
            used  = sum(float(g.get("memoryAllocated", 0)) for g in gpus)
            quota = sum(float(g.get("memoryQuota", 1))     for g in gpus)
            gpu_pct = _pct(used, max(quota, 1))

        net = s.get("network", {})
        net_pct = min(float(net.get("txUtilPct", 0)), 100.0) if isinstance(net, dict) else 0.0

        try:
            self.query_one("#g-cpu",  GaugeRow).set_pct(cpu)
            self.query_one("#g-gpu",  GaugeRow).set_pct(gpu_pct)
            self.query_one("#g-stor", GaugeRow).set_pct(_pct(sto_used, sto_total))
            self.query_one("#g-net",  GaugeRow).set_pct(net_pct)
        except Exception:
            pass

        try:
            self.query_one("#event-log", Log).write_line(
                f"cpu={cpu:.1f}%  mem={_pct(mem_used, mem_total):.1f}%  "
                f"stor={_pct(sto_used, sto_total):.1f}%"
            )
        except Exception:
            pass

    def _render_fleet(self, s: dict, nodes: list) -> None:
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

    def _set(self, cid: str, value: str, sub: str = "") -> None:
        try:
            self.query_one(f"#{cid}", StatCard).set_value(value, sub)
        except Exception:
            pass