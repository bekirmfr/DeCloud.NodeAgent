"""
screens/network.py — Network connectivity for this node.

Tabs:
  • Interfaces     — physical & virtual NICs with rx/tx and state
  • WireGuard      — interfaces and peers, with handshake age & traffic
  • Bridges & VMs  — virbr0 / docker0 with each tap-port mapped to its VM
  • Routes         — kernel routing table (top entries)

Data: /api/dashboard/network from the Node Agent.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import DataTable, Label, TabbedContent, TabPane

from config import cfg
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen
from theme import COLOR
from util.format import fmt_age, fmt_bytes, truncate


def _state_text(up: bool | None) -> Text:
    if up is True:
        return Text("UP", style=f"bold {COLOR['ok']}")
    if up is False:
        return Text("DOWN", style=f"bold {COLOR['crit']}")
    return Text("?", style=f"{COLOR['dim']}")


def _handshake_text(secs_ago: int | float | None) -> Text:
    if secs_ago is None:
        return Text("never", style=f"{COLOR['crit']}")
    if secs_ago > 180:
        return Text(fmt_age(secs_ago), style=f"{COLOR['warn']}")
    return Text(fmt_age(secs_ago), style=f"{COLOR['ok']}")


class NetworkScreen(BaseScreen):
    ACTIVE_LABEL = "Network"

    DEFAULT_CSS = """
    NetworkScreen DataTable { height: 1fr; }
    """

    def compose_content(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Interfaces", id="t-iface"):
                yield DataTable(id="iface-tbl", zebra_stripes=True)
            with TabPane("WireGuard", id="t-wg"):
                yield Label("", id="wg-meta", classes="section-title")
                yield DataTable(id="wg-tbl", zebra_stripes=True)
            with TabPane("Bridges & VMs", id="t-br"):
                yield DataTable(id="br-tbl", zebra_stripes=True)
            with TabPane("Routes", id="t-rt"):
                yield DataTable(id="rt-tbl", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one("#iface-tbl", DataTable).add_columns(
            "Name", "Type", "Address", "RX", "TX", "State")
        self.query_one("#wg-tbl", DataTable).add_columns(
            "Interface", "Peer (pubkey)", "Endpoint",
            "Allowed IPs", "Last handshake", "RX / TX")
        self.query_one("#br-tbl", DataTable).add_columns(
            "Bridge", "Address", "Port", "VM")
        self.query_one("#rt-tbl", DataTable).add_columns(
            "Destination", "Gateway", "Interface", "Metric")
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            net = await na.dashboard_network()
        except Exception as exc:
            self.notify(str(exc), severity="error")
            net = {}
        finally:
            await na.close()

        self._render_ifaces(net.get("interfaces", []) or [])
        self._render_wg(net.get("wireguard", []) or net.get("wireguardPeers", []) or [])
        self._render_bridges(net.get("bridges", []) or [])
        self._render_routes(net.get("routes", []) or [])
        self.mark_updated()

    def _render_ifaces(self, ifaces: list[dict]) -> None:
        t = self.query_one("#iface-tbl", DataTable)
        t.clear()
        for i in ifaces:
            ips = ", ".join(i.get("ipAddresses") or [])
            t.add_row(
                i.get("name", "—"),
                i.get("type", "—"),
                truncate(ips, 32),
                fmt_bytes(i.get("rxBytes")),
                fmt_bytes(i.get("txBytes")),
                _state_text(i.get("isUp")),
            )

    def _render_wg(self, wg_ifaces: list[dict]) -> None:
        meta_lines = []
        t = self.query_one("#wg-tbl", DataTable)
        t.clear()
        peer_count = 0
        for wg in wg_ifaces:
            ifname = wg.get("name", "—")
            port = wg.get("listenPort", "—")
            pub = wg.get("publicKey", "—")
            meta_lines.append(f"{ifname}  port={port}  pubkey={pub[:16]}…")
            for p in wg.get("peers", []) or []:
                peer_count += 1
                t.add_row(
                    ifname,
                    truncate(p.get("publicKey", "—"), 18),
                    p.get("endpoint") or "—",
                    truncate(", ".join(p.get("allowedIps", [])
                                       if isinstance(p.get("allowedIps"), list)
                                       else [str(p.get("allowedIps", ""))]),
                             24),
                    _handshake_text(p.get("handshakeSecondsAgo")),
                    f'{fmt_bytes(p.get("rxBytes"))} / {fmt_bytes(p.get("txBytes"))}',
                )
        try:
            self.query_one("#wg-meta", Label).update(
                Text(" · ".join(meta_lines) + f"   ({peer_count} peers)",
                     style=f"{COLOR['muted']}"))
        except Exception:
            pass

    def _render_bridges(self, bridges: list[dict]) -> None:
        t = self.query_one("#br-tbl", DataTable)
        t.clear()
        for b in bridges:
            addr = ", ".join(b.get("ipAddresses") or [])
            ports = b.get("ports", []) or []
            if not ports:
                t.add_row(b.get("name", "—"), addr, "—", "—")
                continue
            for p in ports:
                vm = (f'{p.get("vmName")} ({p.get("vmId","")[:8]})'
                      if p.get("vmName") else "host / unassigned")
                t.add_row(b.get("name", "—"), addr,
                          p.get("interface", "—"), vm)

    def _render_routes(self, routes: list[dict]) -> None:
        t = self.query_one("#rt-tbl", DataTable)
        t.clear()
        for r in routes[:80]:
            t.add_row(
                r.get("destination", "—") or "default",
                r.get("gateway") or "direct",
                r.get("interface", "—"),
                str(r.get("metric", "—")),
            )
