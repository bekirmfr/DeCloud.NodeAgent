"""
screens/obligations.py — Detailed view of system VM obligations.

Shows one card per obligation (DHT, Relay, BlockStore) with:
  • Obligation status, failure count, deployment timing
  • Identity state key-value pairs (stateData from the enriched endpoint)
  • Related VM details (name, IP, vCPU, RAM, disk, VNC, services, etc.)

Data sources:
  GET /api/dashboard/obligations — enriched obligations with stateData
  GET /api/vms                   — full VM list (cross-referenced by vmId)
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static

from api.node_agent import NodeAgentClient
from config import cfg
from screens._base import BaseScreen
from theme import COLOR, DOT_FILLED, DOT_OPEN, DASH
from util.format import fmt_bytes, fmt_duration, fmt_age, truncate
from widgets.card import Card


# ── Status styling ────────────────────────────────────────────────────────

_STATUS_STYLE: dict[str, tuple[str, str]] = {
    # statusName → (glyph, color_key)
    "Active":    (DOT_FILLED, "ok"),
    "Deploying": (DOT_FILLED, "warn"),
    "Pending":   (DOT_OPEN,   "warn"),
    "Failed":    (DOT_FILLED, "crit"),
}


def _status_pill(status_name: str) -> Text:
    glyph, color_key = _STATUS_STYLE.get(status_name, (DOT_OPEN, "muted"))
    color = COLOR[color_key]
    return Text.assemble(
        (f"{glyph} ", f"bold {color}"),
        (status_name, f"bold {color}"),
    )


# ── Key-value renderers ──────────────────────────────────────────────────

def _kv_line(key: str, val: Any, *, sensitive: bool = False) -> Text:
    """One key-value row for the state data section."""
    out = Text()
    out.append(f"  {key + ':':<24} ", style=f"{COLOR['muted']}")
    display = str(val) if val is not None else DASH
    if sensitive and len(display) > 20:
        # Truncate sensitive values (keys, secrets) with ellipsis
        display = display[:16] + "…" + display[-4:]
    out.append(display, style=f"{COLOR['title']}")
    return out


# Keys in stateData that are sensitive (private keys, seeds).
# Show them truncated so the operator knows they exist but can't
# accidentally expose them in a screenshot.
_SENSITIVE_KEYS = frozenset({
    "privateKey", "ed25519PrivateKey", "ed25519Seed",
    "wireguardPrivateKey", "nodePrivateKey", "secret",
})


def _render_state_data(data: dict[str, Any] | None) -> Text:
    """Render the stateData dict as key-value lines."""
    if not data:
        return Text("  (no state data)", style=f"{COLOR['dim']}")
    out = Text()
    for i, (k, v) in enumerate(data.items()):
        if i:
            out.append("\n")
        is_sensitive = any(s in k.lower() for s in ("private", "secret", "seed"))
        out.append_text(_kv_line(k, v, sensitive=is_sensitive))
    return out


def _render_vm_details(vm: dict[str, Any]) -> Text:
    """Render the related VM's details as key-value lines."""
    out = Text()
    rows: list[tuple[str, str]] = [
        ("ID",        vm.get("vmId") or DASH),
        ("State",     vm.get("state") or DASH),
        ("Type",      vm.get("vmType") or DASH),
        ("IP",        vm.get("ipAddress") or DASH),
        ("vCPU",      str(vm.get("virtualCpuCores", DASH))),
        ("RAM",       fmt_bytes(vm.get("memoryBytes"))),
        ("Disk",      fmt_bytes(vm.get("diskBytes"))),
        ("CPU usage",  f'{vm.get("virtualCpuUsagePercent", 0):.1f}%'),
        ("VNC port",  str(vm.get("vncPort") or DASH)),
        ("MAC",       vm.get("macAddress") or DASH),
        ("Started",   fmt_age(None)  # filled below
                      if not vm.get("startedAt") else
                      str(vm.get("startedAt", DASH))),
    ]
    for i, (label, val) in enumerate(rows):
        if i:
            out.append("\n")
        out.append_text(_kv_line(label, val))

    # Services sub-section
    services = vm.get("services") or []
    if services:
        out.append("\n\n")
        out.append("  Services", style=f"bold {COLOR['info']}")
        for svc in services:
            out.append("\n")
            name = svc.get("name") or "?"
            port = svc.get("port") or "?"
            proto = svc.get("protocol") or "tcp"
            status = svc.get("status") or "Unknown"
            status_msg = svc.get("statusMessage") or ""
            sev = "ok" if status.lower() in ("ready", "healthy") else "warn"
            out.append(f"    {name}  ", style=f"{COLOR['title']}")
            out.append(f":{port}/{proto}  ", style=f"{COLOR['muted']}")
            out.append(f"{status}", style=f"bold {COLOR[sev]}")
            if status_msg:
                out.append(f"  {truncate(status_msg, 40)}", style=f"{COLOR['dim']}")
    return out


# ── Screen ────────────────────────────────────────────────────────────────

class ObligationsScreen(BaseScreen):
    """Detailed obligation view — one expandable card per system VM role."""

    ACTIVE_LABEL = "Obligations"

    DEFAULT_CSS = """
    ObligationsScreen #obl-scroll {
        height: 1fr;
    }
    ObligationsScreen .obl-card {
        margin: 0 0 1 0;
    }
    ObligationsScreen .obl-header {
        height: 1;
        padding: 0 1;
    }
    ObligationsScreen .obl-section-title {
        padding: 1 1 0 1;
        text-style: bold;
    }
    ObligationsScreen .obl-body {
        padding: 0 1;
    }
    ObligationsScreen .kv {
        height: 1;
    }
    ObligationsScreen .kv-key {
        width: 26;
        color: $text-muted;
    }
    ObligationsScreen .kv-val {
        width: 1fr;
    }
    """

    def compose_content(self) -> ComposeResult:
        yield Static("Loading obligations…", id="obl-empty",
                      classes="obl-body")
        yield VerticalScroll(id="obl-scroll")

    def on_mount(self) -> None:
        self.set_interval(max(cfg.refresh_interval, 15), self._load)
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            obligations, vms = await asyncio.gather(
                na.dashboard_obligations(),
                na.list_vms(),
                return_exceptions=True,
            )
        finally:
            await na.close()

        if not isinstance(obligations, list):
            obligations = []
        if not isinstance(vms, list):
            vms = []

        # Build a vmId → vm dict for cross-referencing
        vm_by_id: dict[str, dict] = {}
        for v in vms:
            vid = v.get("vmId") or v.get("id")
            if vid:
                vm_by_id[vid] = v

        self._render_obligations(obligations, vm_by_id)
        self.mark_updated()

    def _render_obligations(self, obligations: list[dict],
                            vm_by_id: dict[str, dict]) -> None:
        scroll = self.query_one("#obl-scroll", VerticalScroll)
        empty_label = self.query_one("#obl-empty", Static)

        if not obligations:
            empty_label.update(Text("No obligations assigned to this node.",
                                    style=f"{COLOR['dim']}"))
            empty_label.display = True
            scroll.display = False
            return

        empty_label.display = False
        scroll.display = True

        # Remove old cards
        scroll.remove_children()

        for obl in obligations:
            role_name = obl.get("roleName") or f"Role {obl.get('role', '?')}"
            status_name = obl.get("statusName") or "Unknown"
            vm_id = obl.get("vmId")
            state_data = obl.get("stateData")
            failure_count = obl.get("failureCount") or 0
            last_error = obl.get("lastError")
            deployed_at = obl.get("deployedAt")
            active_at = obl.get("activeAt")
            running_ver = obl.get("runningBinaryVersion")
            current_ver = obl.get("currentBinaryVersion")
            state_version = obl.get("stateVersion", 0)

            # ── Build card content as a single Static ─────────────
            body = Text()

            # Status line
            body.append_text(_status_pill(status_name))
            if failure_count:
                body.append(f"   failures: {failure_count}",
                            style=f"bold {COLOR['crit']}")
            body.append("\n")

            # Timing
            timing_parts: list[str] = []
            if deployed_at:
                timing_parts.append(f"deployed {deployed_at}")
            if active_at:
                timing_parts.append(f"active since {active_at}")
            if timing_parts:
                body.append("  " + "  ·  ".join(timing_parts) + "\n",
                            style=f"{COLOR['muted']}")

            # Version info
            if running_ver or current_ver:
                body.append("\n")
                if running_ver:
                    body.append_text(_kv_line("Running version",
                                              truncate(running_ver, 20)))
                    body.append("\n")
                if current_ver:
                    body.append_text(_kv_line("Current version",
                                              truncate(current_ver, 20)))
                    body.append("\n")
                if state_version:
                    body.append_text(_kv_line("State version",
                                              str(state_version)))
                    body.append("\n")

            # Last error
            if last_error:
                body.append("\n")
                body.append("  Last error: ", style=f"bold {COLOR['crit']}")
                body.append(truncate(str(last_error), 80) + "\n",
                            style=f"{COLOR['crit']}")

            # State data section
            if state_data and isinstance(state_data, dict):
                body.append("\n")
                body.append("  Identity State\n",
                            style=f"bold {COLOR['info']}")
                body.append_text(_render_state_data(state_data))
                body.append("\n")

            # Related VM section
            vm = vm_by_id.get(vm_id) if vm_id else None
            if vm:
                body.append("\n")
                vm_name = vm.get("name") or vm_id or DASH
                body.append(f"  VM: {vm_name}\n",
                            style=f"bold {COLOR['info']}")
                body.append_text(_render_vm_details(vm))
            elif vm_id:
                body.append("\n")
                body.append(f"  VM: {vm_id}\n",
                            style=f"bold {COLOR['info']}")
                body.append("  (VM details not available — "
                            "VM may still be starting)\n",
                            style=f"{COLOR['dim']}")

            # Create the card and add the body as a Static
            card = Card(f"{role_name} Obligation",
                        subtitle=status_name,
                        id=f"obl-{role_name.lower()}")
            content = Static(id=f"obl-{role_name.lower()}-body")
            card._body_static = content
            scroll.mount(card)
            card.mount(content)
            content.update(body)
