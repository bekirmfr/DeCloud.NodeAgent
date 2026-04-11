"""screens/network.py — Networking for this node."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import DataTable, Label, TabbedContent, TabPane

from config import cfg
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen


class NetworkScreen(BaseScreen):
    ACTIVE_LABEL = "Networking"
    BINDINGS = [("r", "refresh", "Refresh")]

    def compose_content(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Interfaces", id="tab-ifaces"):
                yield DataTable(id="iface-table", zebra_stripes=True)
            with TabPane("WireGuard Peers", id="tab-wg"):
                yield DataTable(id="wg-table", zebra_stripes=True)
            with TabPane("Port Rules", id="tab-ports"):
                yield DataTable(id="port-table", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one("#iface-table", DataTable).add_columns(
            "Interface", "Address", "Type", "↑ TX", "↓ RX", "State"
        )
        self.query_one("#wg-table", DataTable).add_columns(
            "Peer", "Endpoint", "Allowed IPs", "Last Handshake", "RX / TX"
        )
        self.query_one("#port-table", DataTable).add_columns(
            "Proto", "Host Port", "VM Port", "Target", "State"
        )
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            self.notify("Set DECLOUD_NODE_URL", severity="warning")
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            network = await na.get_network()
            ports   = await na.get_ports()
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        finally:
            await na.close()
        self._render_ifaces(network.get("interfaces", []))
        self._render_wg(network.get("wireguardPeers", network.get("wireguard", [])))
        self._render_ports(ports)

    def _render_ifaces(self, ifaces: list) -> None:
        t = self.query_one("#iface-table", DataTable)
        t.clear()
        for i in ifaces:
            state = i.get("state", "UP" if i.get("isUp") else "DOWN")
            color = "green" if state == "UP" else "red"
            addrs = i.get("address") or ", ".join(i.get("ipAddresses") or []) or "—"
            t.add_row(i.get("name","—"), addrs, i.get("type","—"),
                      i.get("txHuman","—"), i.get("rxHuman","—"),
                      f"[{color}]{state}[/]")

    def _render_wg(self, peers: list) -> None:
        t = self.query_one("#wg-table", DataTable)
        t.clear()
        for peer in peers:
            pk = peer.get("publicKey","")
            short = f"{pk[:6]}…{pk[-4:]}" if len(pk) > 12 else pk
            last_hs = peer.get("lastHandshakeAgo", peer.get("handshakeSecondsAgo","—"))
            if isinstance(last_hs, (int, float)):
                last_hs = f"{int(last_hs)}s ago"
            t.add_row(short, peer.get("endpoint","—"), peer.get("allowedIps","—"),
                      last_hs,
                      f"{peer.get('rxHuman','—')} / {peer.get('txHuman','—')}")

    def _render_ports(self, rules: list) -> None:
        t = self.query_one("#port-table", DataTable)
        t.clear()
        for r in rules:
            t.add_row(r.get("protocol","tcp"), str(r.get("hostPort","—")),
                      str(r.get("vmPort","—")), r.get("target","—"), r.get("state","—"))