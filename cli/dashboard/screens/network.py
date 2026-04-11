"""
screens/network.py — Network interfaces and WireGuard peer table.

Data sourced from node agent GET /api/dashboard/network.
Falls back gracefully when node agent is not configured.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable, Label, TabbedContent, TabPane

from config import cfg
from api.node_agent import NodeAgentClient


class NetworkScreen(Vertical):
    _is_mounted: bool = False

    """Interfaces + WireGuard peers + port forwarding tabs."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label("Networking", classes="section-title")
        with TabbedContent():
            with TabPane("Interfaces", id="tab-ifaces"):
                yield DataTable(id="iface-table", zebra_stripes=True)
            with TabPane("WireGuard Peers", id="tab-wg"):
                yield DataTable(id="wg-table", zebra_stripes=True)
            with TabPane("Port Rules", id="tab-ports"):
                yield DataTable(id="port-table", zebra_stripes=True)

    def on_mount(self) -> None:
        self._setup_tables()
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    def _setup_tables(self) -> None:
        self.query_one("#iface-table", DataTable).add_columns(
            "Interface", "Address", "Type", "↑ TX", "↓ RX", "State"
        )
        self.query_one("#wg-table", DataTable).add_columns(
            "Peer (short)", "Endpoint", "Allowed IPs", "Last Handshake", "RX / TX"
        )
        self.query_one("#port-table", DataTable).add_columns(
            "Proto", "Host Port", "VM Port", "Target", "State"
        )

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            self.notify("Node agent URL not configured — set DECLOUD_NODE_URL", severity="warning")
            return

        client = NodeAgentClient(cfg.node_url)
        try:
            network = await client.get_network()
            ports = await client.get_ports()
        except Exception as e:
            self.notify(f"Node agent error: {e}", severity="error")
            await client.close()
            return
        finally:
            await client.close()

        self._render_interfaces(network.get("interfaces", []))
        self._render_wg(network.get("wireguardPeers", []))
        self._render_ports(ports)

    def _render_interfaces(self, ifaces: list) -> None:
        table = self.query_one("#iface-table", DataTable)
        table.clear()
        for iface in ifaces:
            state = iface.get("state", "UP")
            color = "green" if state == "UP" else "red"
            table.add_row(
                iface.get("name", "—"),
                iface.get("address", "—"),
                iface.get("type", "—"),
                iface.get("txHuman", "—"),
                iface.get("rxHuman", "—"),
                f"[{color}]{state}[/]",
            )

    def _render_wg(self, peers: list) -> None:
        table = self.query_one("#wg-table", DataTable)
        table.clear()
        for peer in peers:
            public_key = peer.get("publicKey", "")
            short_key = f"{public_key[:6]}…{public_key[-4:]}" if len(public_key) > 12 else public_key
            last_hs = peer.get("lastHandshakeAgo", "—")
            color = "green" if "s ago" in last_hs else "yellow"
            table.add_row(
                short_key,
                peer.get("endpoint", "—"),
                peer.get("allowedIps", "—"),
                f"[{color}]{last_hs}[/]",
                f"{peer.get('rxHuman', '—')} / {peer.get('txHuman', '—')}",
            )

    def _render_ports(self, rules: list) -> None:
        table = self.query_one("#port-table", DataTable)
        table.clear()
        for rule in rules:
            table.add_row(
                rule.get("protocol", "tcp"),
                str(rule.get("hostPort", "—")),
                str(rule.get("vmPort", "—")),
                rule.get("target", "—"),
                rule.get("state", "—"),
            )