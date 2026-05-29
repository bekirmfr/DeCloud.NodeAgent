"""
screens/overview.py — Single pane of glass.

Layout (top → bottom):

  ┌─ Resources ──────────────────────────┐ ┌─ System Obligations ──────────┐
  │ CPU   ████░░░░  32 %    ▁▂▃▅▇        │ │ ● DHT          Active   v1.2  │
  │ MEM   █████░░░  41 %    ▁▂▃▄▆        │ │ ● Block Store  Active   v0.4  │
  │ STOR  ██░░░░░░  12 %    ▁▁▂▂▃        │ │ ● Relay        n/a            │
  │ GPU   ▓▓░░░░░░  17 %    ▁▂▂▃▄  L1    │ └───────────────────────────────┘
  │ NET   ██░░░░░░  ↑3.2MB ↓18MB         │
  └──────────────────────────────────────┘ ┌─ Earnings (24 h / 30 d) ──────┐
                                           │ $ 0.42  $ 9.81                │
  ┌─ This Node's VMs (8) ────────────────┐ │  6 active · 2 stopped         │
  │ ▶ Running   blockstore-eu-…  …       │ └───────────────────────────────┘
  │ ▶ Running   dht-eu-…         …       │
  │ ▶ Running   ml-trainer       …       │ ┌─ Recent Events ───────────────┐
  │ ■ Stopped   web-prod         …       │ │ 14:02:03 INF VM ml-trainer up │
  └──────────────────────────────────────┘ │ 14:01:15 WRN Heartbeat 35s ago│
                                           │ 13:58:40 INF Obligation Active│
                                           └───────────────────────────────┘

The Overview is the sole place that polls the IdentityBar /
StatusStrip data — other screens consume cfg.identity, written here.

This screen also drives the StatusStrip values for any sibling screen
the user navigates to (single source of truth for top-bar state).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Static

from config import cfg
from api.node_agent import NodeAgentClient
from api.orchestrator import OrchestratorClient
from screens._base import BaseScreen
from theme import COLOR, HISTORY_LEN, severity
from util.format import (
    fmt_age, fmt_bytes, fmt_pct, short_id, truncate
)
from util.history import Ring
from widgets.card import Card
from widgets.gauge import Gauge
from widgets.statpill import status_text
from widgets.badges import vm_state_badge, obligation_badge


# Interface bytes → MB/s sparkline
def _bytes_to_mbs(bytes_per_s: float) -> float:
    return max(0.0, bytes_per_s / (1024.0 * 1024.0))


class OverviewScreen(BaseScreen):
    ACTIVE_LABEL = "Overview"
    EXTRA_HINTS = [("v", "VMs"), ("d", "Diag")]

    BINDINGS = [
        ("v", "go_vms",  "VMs"),
        ("d", "go_diag", "Diagnostics"),
    ]

    DEFAULT_CSS = """
    OverviewScreen .row {
        height: auto;
        layout: horizontal;
    }
    OverviewScreen #col-left  { width: 2fr; height: auto; }
    OverviewScreen #col-right { width: 1fr; height: auto; margin-left: 1; }
    OverviewScreen Card { height: auto; }
    OverviewScreen #res-card { min-height: 9; }
    OverviewScreen #obl-card { min-height: 9; }
    OverviewScreen #vms-card DataTable { height: auto; max-height: 12; }
    OverviewScreen #ev-card  { min-height: 8; }
    OverviewScreen #earn-card { min-height: 5; }
    OverviewScreen .earn-row {
        height: 3;
        layout: horizontal;
        content-align: center middle;
    }
    OverviewScreen .earn-tile {
        width: 1fr;
        text-align: center;
    }
    OverviewScreen .earn-tile-val {
        text-style: bold;
        color: $accent;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._h_cpu  = Ring(HISTORY_LEN)
        self._h_mem  = Ring(HISTORY_LEN)
        self._h_stor = Ring(HISTORY_LEN)
        self._h_gpu  = Ring(HISTORY_LEN)
        self._h_net  = Ring(HISTORY_LEN)
        self._prev_net_total: int | None = None
        self._prev_net_at: datetime | None = None

    # ─── Layout ────────────────────────────────────────────────────────

    def compose_content(self) -> ComposeResult:
        with Horizontal(classes="row"):
            with Vertical(id="col-left"):
                yield self._build_resources_card()
                yield self._build_vms_card()
            with Vertical(id="col-right"):
                yield self._build_obligations_card()
                yield self._build_earnings_card()
                yield self._build_events_card()

    def _build_resources_card(self) -> Card:
        c = Card("Resources", id="res-card")
        c.compose_body = lambda: self._compose_resources()  # type: ignore[assignment]
        return c

    def _compose_resources(self) -> ComposeResult:
        yield Gauge("CPU",     id="g-cpu")
        yield Gauge("Memory",  id="g-mem")
        yield Gauge("Storage", id="g-stor")
        yield Gauge("GPU",     id="g-gpu")
        yield Static("", id="net-line")  # custom row for ↑↓ throughput

    def _build_obligations_card(self) -> Card:
        c = Card("System Obligations", id="obl-card")
        c.compose_body = lambda: self._compose_obligations()  # type: ignore[assignment]
        return c

    def _compose_obligations(self) -> ComposeResult:
        # Pre-render three rows (DHT / BlockStore / Relay).
        yield Static("loading…", id="obl-list")

    def _build_earnings_card(self) -> Card:
        c = Card("Earnings", subtitle="24 h / 30 d", id="earn-card")
        c.compose_body = lambda: self._compose_earnings()  # type: ignore[assignment]
        return c

    def _compose_earnings(self) -> ComposeResult:
        with Horizontal(classes="earn-row"):
            with Vertical(classes="earn-tile"):
                yield Static("24 hours", classes="earn-tile-lbl")
                yield Static("—", id="earn-24h", classes="earn-tile-val")
            with Vertical(classes="earn-tile"):
                yield Static("30 days", classes="earn-tile-lbl")
                yield Static("—", id="earn-30d", classes="earn-tile-val")
            with Vertical(classes="earn-tile"):
                yield Static("Active VMs", classes="earn-tile-lbl")
                yield Static("—", id="earn-vms", classes="earn-tile-val")

    def _build_vms_card(self) -> Card:
        c = Card("Virtual Machines", id="vms-card")
        c.compose_body = lambda: self._compose_vms()  # type: ignore[assignment]
        return c

    def _compose_vms(self) -> ComposeResult:
        t = DataTable(id="vm-table", zebra_stripes=True, cursor_type="row")
        t.add_columns("State", "Name", "Type", "vCPU", "Mem", "IP")
        yield t

    def _build_events_card(self) -> Card:
        c = Card("Recent Events", id="ev-card")
        c.compose_body = lambda: self._compose_events()  # type: ignore[assignment]
        return c

    def _compose_events(self) -> ComposeResult:
        yield Static("loading…", id="event-list")

    # ─── Lifecycle ─────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    def action_go_vms(self) -> None:
        from screens import get_screen
        s = get_screen("Virtual Machines")
        if s: self.app.switch_screen(s)

    def action_go_diag(self) -> None:
        from screens import get_screen
        s = get_screen("Diagnostics")
        if s: self.app.switch_screen(s)

    # ─── Data fetch (async, gathered in parallel) ──────────────────────

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            return
        na = NodeAgentClient(cfg.node_url)
        oc = OrchestratorClient(cfg.orchestrator_url, cfg.token) \
            if cfg.has_orchestrator else None

        coros = [
            na.dashboard_summary(),
            na.node_snapshot(),
            na.list_vms(),
            na.dashboard_obligations(),
            na.dashboard_network(),
            na.dashboard_logs(lines=10),
        ]
        if oc is not None:
            coros.append(oc.get_balance())

        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
        finally:
            await na.close()
            if oc is not None:
                await oc.close()

        summary, snap, vms, obligations, network, logs = results[:6]
        balance = results[6] if len(results) > 6 else None

        # Each section is independent — exceptions in one don't drop others.
        self._render_summary(summary)
        self._render_resources(snap, network)
        self._render_obligations(obligations)
        self._render_vms(vms)
        self._render_earnings(balance, vms)
        self._render_events(logs)
        self._update_status_strip(summary, snap, vms)
        self.mark_updated()

    # ─── Renderers ─────────────────────────────────────────────────────

    def _render_summary(self, summary: Any) -> None:
        if not isinstance(summary, dict):
            return
        # Persist identity for IdentityBar (lives in app chrome).
        cfg.identity = summary
        try:
            from widgets.header import IdentityBar
            self.query_one("#identity-bar", IdentityBar).refresh_identity()
        except Exception:
            pass

    def _render_resources(self, snap: Any, network: Any) -> None:
        if not isinstance(snap, dict):
            return

        cpu_pct  = float(snap.get("virtualCpuUsagePercent", 0) or 0)
        total_m  = float(snap.get("totalMemoryBytes", 0) or 0)
        used_m   = float(snap.get("usedMemoryBytes", 0) or 0)
        total_s  = float(snap.get("totalStorageBytes", 0) or 0)
        used_s   = float(snap.get("usedStorageBytes", 0) or 0)
        gpu_pct  = float(snap.get("gpuUtilizationPercent", 0) or 0) \
            if "gpuUtilizationPercent" in snap else None

        mem_pct  = (used_m / total_m * 100) if total_m else 0.0
        stor_pct = (used_s / total_s * 100) if total_s else 0.0

        self._h_cpu.push(cpu_pct)
        self._h_mem.push(mem_pct)
        self._h_stor.push(stor_pct)

        self.query_one("#g-cpu",  Gauge).update(cpu_pct,  self._h_cpu.values())
        self.query_one("#g-mem",  Gauge).update(mem_pct,  self._h_mem.values())
        self.query_one("#g-stor", Gauge).update(stor_pct, self._h_stor.values())

        if gpu_pct is not None:
            self._h_gpu.push(gpu_pct)
            self.query_one("#g-gpu", Gauge).update(gpu_pct, self._h_gpu.values())
        else:
            self.query_one("#g-gpu", Gauge).set_unavailable("no GPU")

        # Network throughput row — derive rate from /proc/net/dev counters.
        rate_mbs = self._compute_net_rate(network)
        self._h_net.push(rate_mbs)
        self._render_net_line(rate_mbs, network)

    def _compute_net_rate(self, network: Any) -> float:
        if not isinstance(network, dict):
            return 0.0
        ifaces = network.get("interfaces", []) or []
        # Sum non-loopback rx+tx
        total = 0
        for i in ifaces:
            if i.get("name") == "lo":
                continue
            total += int(i.get("rxBytes", 0) or 0)
            total += int(i.get("txBytes", 0) or 0)
        now = datetime.now()
        prev_total = self._prev_net_total
        prev_at = self._prev_net_at
        self._prev_net_total = total
        self._prev_net_at = now
        if prev_total is None or prev_at is None:
            return 0.0
        dt = (now - prev_at).total_seconds()
        if dt <= 0:
            return 0.0
        return _bytes_to_mbs(max(0, total - prev_total) / dt)

    def _render_net_line(self, rate_mbs: float, network: Any) -> None:
        # Show top non-loopback interface name, plus a bar+rate.
        try:
            label = "—"
            if isinstance(network, dict):
                ifs = [i for i in network.get("interfaces", [])
                       if i.get("name") != "lo" and i.get("isUp")]
                if ifs:
                    label = ifs[0].get("name", "—")
            sev = severity(min(100.0, rate_mbs * 5))  # arbitrary 20 MB/s ≈ 100 %
            color = COLOR[sev] if rate_mbs > 0 else COLOR["dim"]
            txt = Text()
            txt.append("Network  ", style=f"{COLOR['muted']}")
            txt.append(f"{label:<12}", style="bold")
            txt.append(f"  {rate_mbs:6.2f} MB/s ", style=f"{color}")
            self.query_one("#net-line", Static).update(txt)
        except Exception:
            pass

    def _render_obligations(self, obligations: Any) -> None:
        target = self.query_one("#obl-list", Static)
        if not isinstance(obligations, list) or not obligations:
            target.update(Text("(orchestrator did not return any obligations)",
                               style=f"{COLOR['dim']}"))
            return

        out = Text()
        for i, o in enumerate(obligations):
            if i:
                out.append("\n")
            role_name = (o.get("roleName") or _role_name(o.get("role"))).ljust(13)
            status   = o.get("statusName") or o.get("status") or "Unknown"
            ver      = o.get("runningBinaryVersion") or "—"
            out.append(obligation_badge(status))
            out.append(f"  {role_name} ", style="bold")
            out.append(f"v{ver}", style=f"{COLOR['muted']}")
            err = o.get("lastError")
            if err and status not in ("Active", "Healthy"):
                out.append(f"  · {truncate(err, 32)}", style=f"{COLOR['crit']}")
        target.update(out)

    def _render_vms(self, vms: Any) -> None:
        t = self.query_one("#vm-table", DataTable)
        t.clear()
        if not isinstance(vms, list):
            return
        # Order: running first, then by name.
        def sort_key(v: dict) -> tuple:
            state = v.get("state")
            return (0 if state == 3 else 1, (v.get("name") or "").lower())
        vms_sorted = sorted(vms, key=sort_key)
        for vm in vms_sorted[:8]:  # keep concise on Overview
            spec = vm.get("spec") or {}
            t.add_row(
                vm_state_badge(vm.get("state")),
                truncate(vm.get("name") or spec.get("name") or "—", 24),
                _vm_type_name(spec.get("vmRole")),
                str(spec.get("virtualCpuCores", "—")),
                fmt_bytes(spec.get("memoryBytes")),
                spec.get("ipAddress") or vm.get("ipAddress") or "—",
            )
        sub = f"{len(vms)} total"
        try:
            self.query_one("#vms-card", Card).set_subtitle(sub)
        except Exception:
            pass

    def _render_earnings(self, balance: Any, vms: Any) -> None:
        # If there's no orchestrator, show a clear muted state — never errors.
        if not isinstance(balance, dict):
            try:
                self.query_one("#earn-24h", Static).update(
                    Text("—", style=f"{COLOR['dim']}"))
                self.query_one("#earn-30d", Static).update(
                    Text("—", style=f"{COLOR['dim']}"))
                self.query_one("#earn-vms", Static).update(
                    Text("—", style=f"{COLOR['dim']}"))
                self.query_one("#earn-card", Card).set_subtitle(
                    "configure orchestrator")
            except Exception:
                pass
            return
        try:
            self.query_one("#earn-24h", Static).update(
                Text(f"${float(balance.get('earned24h', 0)):.2f}",
                     style=f"bold {COLOR['ok']}"))
            self.query_one("#earn-30d", Static).update(
                Text(f"${float(balance.get('earned30d', 0)):.2f}",
                     style=f"bold {COLOR['info']}"))
            running = sum(1 for v in (vms or []) if v.get("state") == 3) \
                if isinstance(vms, list) else 0
            self.query_one("#earn-vms", Static).update(
                Text(str(running), style=f"bold {COLOR['info']}"))
        except Exception:
            pass

    def _render_events(self, logs: Any) -> None:
        target = self.query_one("#event-list", Static)
        if not isinstance(logs, list) or not logs:
            target.update(Text("no recent events", style=f"{COLOR['dim']}"))
            return

        # The dashboard_logs endpoint returns raw strings — same shape as
        # journalctl or the .NET logger writes them. Two common patterns:
        #   "info: SourceName[101]"           ← header
        #   "      End processing HTTP req…"  ← indented continuation
        # We classify by the first non-space token and skip pure
        # continuation lines so the card shows distinct events, not a
        # stack-traced waterfall.
        events: list[tuple[str, str]] = []   # (level3, message)
        for raw in logs:
            line = raw if isinstance(raw, str) else str(raw)
            stripped = line.lstrip()
            if not stripped:
                continue
            # Continuation lines (4+ leading spaces) — attach to last event
            if line.startswith("    ") and events:
                last_lvl, last_msg = events[-1]
                # Append a separator + the continuation, truncated soon enough
                events[-1] = (last_lvl, f"{last_msg} {stripped}")
                continue
            low = stripped.lower()
            if low.startswith(("err", "fail", "fatal", "crit")) or "error" in low[:30]:
                lvl = "ERR"
            elif low.startswith(("warn", "wrn")):
                lvl = "WRN"
            elif low.startswith(("dbug", "debug", "trce", "trace")):
                lvl = "DBG"
            else:
                lvl = "INF"
            events.append((lvl, stripped))

        if not events:
            target.update(Text("no recent events", style=f"{COLOR['dim']}"))
            return

        out = Text()
        for i, (lvl, msg) in enumerate(events[-8:]):
            if i:
                out.append("\n")
            color = {
                "ERR": COLOR["crit"], "WRN": COLOR["warn"],
                "INF": COLOR["info"], "DBG": COLOR["dim"],
            }.get(lvl, COLOR["muted"])
            out.append(f"{lvl} ", style=f"bold {color}")
            out.append(truncate(msg, 70))
        target.update(out)

    def _update_status_strip(self, summary, snap, vms) -> None:
        try:
            strip = self.query_one("#status-strip")
            # Orchestrator pill
            if isinstance(summary, dict):
                orch = summary.get("orchestrator") or {}
                if orch.get("connected"):
                    secs = orch.get("secondsAgo")
                    age = fmt_age(secs) if secs is not None else "—"
                    sev = "ok" if (secs or 0) < 30 else "warn" if (secs or 0) < 90 else "crit"
                    strip.set_orch(sev, age)
                else:
                    strip.set_orch("crit", "offline")
            # CPU/RAM/STOR
            if isinstance(snap, dict):
                cpu = float(snap.get("virtualCpuUsagePercent", 0) or 0)
                tm  = float(snap.get("totalMemoryBytes", 0) or 0)
                um  = float(snap.get("usedMemoryBytes", 0) or 0)
                ts  = float(snap.get("totalStorageBytes", 0) or 0)
                us  = float(snap.get("usedStorageBytes", 0) or 0)
                strip.set_metric("cpu",  "CPU",  fmt_pct(cpu, 0),
                                 severity=severity(cpu))
                mp = um/tm*100 if tm else 0.0
                sp = us/ts*100 if ts else 0.0
                strip.set_metric("ram",  "RAM",  fmt_pct(mp, 0),
                                 severity=severity(mp))
                strip.set_metric("stor", "STOR", fmt_pct(sp, 0),
                                 severity=severity(sp))
            # VMs / errors
            if isinstance(vms, list):
                running = sum(1 for v in vms if v.get("state") == 3)
                failed  = sum(1 for v in vms if v.get("state") == 7)
                strip.set_metric("vms",  "VMs",  f"{running}/{len(vms)}",
                                 severity="ok" if failed == 0 else "warn")
                strip.set_metric("err",  "ERR",  str(failed),
                                 severity="ok" if failed == 0 else "crit")
        except Exception:
            pass


# ─── Pure helpers ───────────────────────────────────────────────────────

_VM_TYPES = {0:"General",1:"Compute",2:"Memory",3:"Storage",
             4:"GPU",5:"Relay",6:"DHT",7:"Inference",8:"BlockStore"}


def _vm_type_name(t: Any) -> str:
    try:
        return _VM_TYPES.get(int(t), "—")
    except (TypeError, ValueError):
        return "—"


def _role_name(role: Any) -> str:
    try:
        return {0:"DHT", 1:"Relay", 2:"BlockStore", 3:"Ingress"}.get(int(role), "—")
    except (TypeError, ValueError):
        return "—"
