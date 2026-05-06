"""
screens/diagnostics.py — Health checks + snapshot export for support.

Two sections:

  ┌─ Health Checks ─────────────────────────────────────────┐
  │ ✓ Node agent reachable                                  │
  │ ✓ KVM available                                         │
  │ ✓ Heartbeat ≤ 30 s                                      │
  │ ✗ Critical service decloud-node-agent active            │
  │ ✓ All system obligations Active                         │
  └─────────────────────────────────────────────────────────┘

  ┌─ Snapshot Export ───────────────────────────────────────┐
  │ Collect full diagnostic JSON from all endpoints         │
  │ for inclusion in support tickets / bug reports.         │
  │                                                         │
  │ [ Collect & Save ]                                      │
  │                                                         │
  │ Last saved: ~/decloud-snapshot-2026-05-06_141532.json   │
  └─────────────────────────────────────────────────────────┘

Mirrors the EXPORT_ENDPOINTS pattern from wwwroot/dashboard.js.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from config import cfg
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen
from theme import COLOR, CHECK, CROSS, DOT_OPEN
from util.format import fmt_age, truncate
from widgets.card import Card


# Endpoints we collect for the support snapshot. Mirrors
# wwwroot/dashboard.js EXPORT_ENDPOINTS so support has parity with the web UI.
_EXPORT = [
    ("summary",     "/api/dashboard/summary"),
    ("snapshot",    "/api/node/snapshot"),
    ("resources",   "/api/node/resources"),
    ("vms",         "/api/vms"),
    ("obligations", "/api/dashboard/obligations"),
    ("network",     "/api/dashboard/network"),
    ("ports",       "/api/dashboard/ports"),
    ("firewall",    "/api/dashboard/firewall"),
    ("services",    "/api/dashboard/services"),
    ("logs",        "/api/dashboard/logs?lines=500"),
    ("database",    "/api/dashboard/database"),
    ("vm_ingress",  "/api/dashboard/vm-ingress"),
]


class DiagnosticsScreen(BaseScreen):
    ACTIVE_LABEL = "Diagnostics"
    EXTRA_HINTS = [("e", "export")]

    BINDINGS = [("e", "export_snapshot", "Export snapshot")]

    DEFAULT_CSS = """
    DiagnosticsScreen .row { height: auto; layout: horizontal; }
    DiagnosticsScreen #col-l { width: 1fr; height: auto; }
    DiagnosticsScreen #col-r { width: 1fr; height: auto; margin-left: 1; }
    DiagnosticsScreen #checks-list { height: auto; min-height: 12; }
    DiagnosticsScreen #export-card { min-height: 12; }
    DiagnosticsScreen #export-meta { color: $text-muted; height: auto; }
    DiagnosticsScreen #export-status { color: $accent; height: 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_export: Path | None = None

    def compose_content(self) -> ComposeResult:
        with Horizontal(classes="row"):
            with Vertical(id="col-l"):
                c = Card("Health Checks", id="checks-card")
                c.compose_body = lambda: iter(  # type: ignore[assignment]
                    [Static("running…", id="checks-list")])
                yield c
            with Vertical(id="col-r"):
                c = Card("Snapshot Export", subtitle="for support tickets",
                         id="export-card")
                c.compose_body = lambda: self._compose_export()  # type: ignore[assignment]
                yield c

    def _compose_export(self) -> ComposeResult:
        yield Static(
            "Collect a JSON snapshot of every node-agent endpoint.\n"
            "The file is written to ~/.decloud/snapshots/ with mode 0600\n"
            "and contains no secrets (the local node-agent does not return them).\n",
            id="export-meta",
        )
        with Horizontal():
            yield Button("Collect & Save", id="btn-export", variant="primary")
        yield Static("", id="export-status")

    def on_mount(self) -> None:
        self.set_interval(cfg.refresh_interval * 2, self._load_checks)
        self.run_worker(self._load_checks(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load_checks(), exclusive=True)

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "btn-export":
            self.action_export_snapshot()

    def action_export_snapshot(self) -> None:
        self.run_worker(self._do_export(), exclusive=True)

    # ─── Health checks ─────────────────────────────────────────────────

    async def _load_checks(self) -> None:
        if not cfg.has_node_agent:
            self._render_checks([("crit", "Node agent URL not configured")])
            return

        na = NodeAgentClient(cfg.node_url)
        results: list[tuple[str, str]] = []  # (severity, message)

        # Reachability + summary
        summary, snap, services, obligations = None, None, None, None
        try:
            summary, snap, services, obligations = await asyncio.gather(
                na.dashboard_summary(),
                na.node_snapshot(),
                na.dashboard_services(),
                na.dashboard_obligations(),
                return_exceptions=True,
            )
        finally:
            await na.close()

        # Check 1 — Reachability
        if isinstance(summary, dict) and summary:
            results.append(("ok", "Node agent reachable"))
        else:
            results.append(("crit", "Node agent NOT reachable"))
            self._render_checks(results)
            self.mark_updated()
            return

        # Check 2 — KVM availability
        kvm = (snap or {}).get("kvmAvailable") if isinstance(snap, dict) else None
        if kvm is True:
            results.append(("ok",   "KVM available (/dev/kvm present)"))
        elif kvm is False:
            results.append(("crit", "KVM unavailable — VMs cannot start"))
        else:
            results.append(("warn", "KVM status unknown"))

        # Check 3 — Heartbeat freshness
        orch = summary.get("orchestrator") or {}
        if orch.get("connected"):
            secs = orch.get("secondsAgo")
            if secs is None:
                results.append(("warn", "Heartbeat connected but age unknown"))
            elif secs < 30:
                results.append(("ok",   f"Orchestrator heartbeat fresh ({fmt_age(secs)})"))
            elif secs < 90:
                results.append(("warn", f"Orchestrator heartbeat stale ({fmt_age(secs)})"))
            else:
                results.append(("crit", f"Orchestrator heartbeat too old ({fmt_age(secs)})"))
        else:
            results.append(("crit", "Orchestrator not connected"))

        # Check 4 — Critical services
        crit_units = {"decloud-node-agent", "libvirtd"}
        if isinstance(services, list):
            down = [s.get("name") for s in services
                    if s.get("name") in crit_units and not s.get("isActive")]
            if not down:
                results.append(("ok", "All critical services active"))
            else:
                results.append(
                    ("crit", f"Critical service(s) inactive: {', '.join(down)}"))
        else:
            results.append(("warn", "Could not query services"))

        # Check 5 — Obligations
        if isinstance(obligations, list) and obligations:
            inactive = [o.get("roleName", "?") for o in obligations
                        if (o.get("statusName") or o.get("status"))
                            not in ("Active", "Healthy")]
            if not inactive:
                results.append(
                    ("ok", f"All {len(obligations)} system obligations Active"))
            else:
                results.append(
                    ("crit", f"Obligations not Active: {', '.join(inactive)}"))
        elif isinstance(obligations, list):
            results.append(("info", "No system obligations assigned"))
        else:
            results.append(("warn", "Could not query obligations"))

        # Check 6 — Storage headroom
        if isinstance(snap, dict):
            ts = float(snap.get("totalStorageBytes", 0) or 0)
            us = float(snap.get("usedStorageBytes", 0) or 0)
            pct = (us / ts * 100) if ts else 0.0
            if pct < 85:
                results.append(("ok",   f"Storage headroom OK ({pct:.0f}% used)"))
            elif pct < 95:
                results.append(("warn", f"Storage low ({pct:.0f}% used)"))
            else:
                results.append(("crit", f"Storage critical ({pct:.0f}% used)"))

        self._render_checks(results)
        self.mark_updated()

    def _render_checks(self, items: list[tuple[str, str]]) -> None:
        out = Text()
        for i, (sev, msg) in enumerate(items):
            if i:
                out.append("\n")
            color = {"ok": COLOR["ok"], "warn": COLOR["warn"],
                     "crit": COLOR["crit"], "info": COLOR["info"]
                     }.get(sev, COLOR["dim"])
            glyph = {"ok": CHECK, "warn": "▲",
                     "crit": CROSS, "info": DOT_OPEN}.get(sev, "·")
            out.append(f"{glyph} ", style=f"bold {color}")
            out.append(msg)
        try:
            self.query_one("#checks-list", Static).update(out)
        except Exception:
            pass

    # ─── Snapshot export ───────────────────────────────────────────────

    async def _do_export(self) -> None:
        if not cfg.has_node_agent:
            self._set_export_status("Node agent URL not configured", "crit")
            return

        self._set_export_status("Collecting…", "info")
        bundle: dict = {
            "_meta": {
                "collectedAt": datetime.utcnow().isoformat() + "Z",
                "nodeUrl":     cfg.node_url,
                "tool":        "decloud-cli-dashboard",
            },
        }

        na = NodeAgentClient(cfg.node_url)
        try:
            for key, path in _EXPORT:
                try:
                    bundle[key] = await na.get(path)
                except Exception as exc:
                    bundle[key] = {"_error": str(exc)}
        finally:
            await na.close()

        # Write to ~/.decloud/snapshots/decloud-snapshot-<ts>.json (mode 0600).
        out_dir = Path.home() / ".decloud" / "snapshots"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            target = out_dir / f"decloud-snapshot-{ts}.json"
            target.write_text(json.dumps(bundle, indent=2, default=str),
                              encoding="utf-8")
            os.chmod(target, 0o600)
            self._last_export = target
            size = target.stat().st_size
            self._set_export_status(
                f"Saved {target}  ({size//1024} KB)", "ok")
        except OSError as exc:
            self._set_export_status(f"Write failed: {exc}", "crit")

    def _set_export_status(self, msg: str, sev: str) -> None:
        color = {"ok": COLOR["ok"], "warn": COLOR["warn"],
                 "crit": COLOR["crit"], "info": COLOR["info"]
                 }.get(sev, COLOR["muted"])
        try:
            self.query_one("#export-status", Static).update(
                Text(msg, style=f"bold {color}"))
        except Exception:
            pass
