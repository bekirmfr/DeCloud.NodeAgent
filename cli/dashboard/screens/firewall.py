"""
screens/firewall.py — Security posture: ports + UFW + iptables.

Combines what the web dashboard splits across separate sections.
The operator's question this answers: 'What is reachable from outside,
and what is firewalled?'

Tabs:
  • Listening Ports — TCP/UDP with process names (from /api/dashboard/ports)
  • UFW             — status, defaults, rules
  • iptables        — INPUT, FORWARD, NAT POSTROUTING chains
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import DataTable, Label, TabbedContent, TabPane

from config import cfg
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen
from theme import COLOR
from util.format import truncate


_INFRASTRUCTURE_PORTS = {22, 53, 80, 443, 5100, 5060, 8080, 51820, 51821}


def _proc_text(name: str) -> Text:
    """Highlight known infrastructure processes for fast scanning."""
    if not name or name == "—":
        return Text("—", style=f"{COLOR['dim']}")
    if any(k in name.lower() for k in ("decloud", "libvirt", "qemu",
                                         "wireguard", "wg-")):
        return Text(name, style=f"{COLOR['info']}")
    return Text(name)


def _action_color(action: str) -> str:
    a = action.upper()
    if a in ("ACCEPT", "ALLOW"):
        return COLOR["ok"]
    if a in ("DROP", "REJECT", "DENY"):
        return COLOR["crit"]
    return COLOR["muted"]


class FirewallScreen(BaseScreen):
    ACTIVE_LABEL = "Firewall"

    DEFAULT_CSS = """
    FirewallScreen DataTable { height: 1fr; }
    FirewallScreen Label.section-title {
        height: 1; color: $accent; text-style: bold; margin-bottom: 1;
    }
    """

    def compose_content(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Listening Ports", id="t-ports"):
                yield Label("", id="ports-meta")
                yield DataTable(id="ports-tbl", zebra_stripes=True)
            with TabPane("UFW", id="t-ufw"):
                yield Label("", id="ufw-meta", classes="section-title")
                yield DataTable(id="ufw-tbl", zebra_stripes=True)
            with TabPane("iptables", id="t-ipt"):
                yield Label("INPUT", id="ipt-input-hdr",
                            classes="section-title")
                yield DataTable(id="ipt-input", zebra_stripes=True)
                yield Label("FORWARD", id="ipt-forward-hdr",
                            classes="section-title")
                yield DataTable(id="ipt-forward", zebra_stripes=True)
                yield Label("NAT POSTROUTING", id="ipt-nat-hdr",
                            classes="section-title")
                yield DataTable(id="ipt-nat", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one("#ports-tbl", DataTable).add_columns(
            "Proto", "Port", "Local Address", "Process")
        self.query_one("#ufw-tbl", DataTable).add_columns(
            "#", "To", "Action", "Direction", "From")
        for ident in ("ipt-input", "ipt-forward", "ipt-nat"):
            self.query_one(f"#{ident}", DataTable).add_columns(
                "#", "Target", "Proto", "In", "Out", "Source", "Dest", "Opts")
        self.set_interval(cfg.refresh_interval * 2, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            ports = await na.dashboard_ports()
            fw    = await na.dashboard_firewall()
        except Exception as exc:
            self.notify(str(exc), severity="error")
            ports, fw = {}, {}
        finally:
            await na.close()

        self._render_ports(ports)
        self._render_ufw(fw.get("ufw") or {})
        ipt = fw.get("iptables") or {}
        self._render_chain("ipt-input",   ipt.get("input"))
        self._render_chain("ipt-forward", ipt.get("forward"))
        self._render_chain("ipt-nat",     ipt.get("natPostrouting"))
        self.mark_updated()

    def _render_ports(self, ports: dict) -> None:
        t = self.query_one("#ports-tbl", DataTable)
        t.clear()
        rows = []
        for p in (ports.get("tcp") or []):
            rows.append(("TCP", p))
        for p in (ports.get("udp") or []):
            rows.append(("UDP", p))
        rows.sort(key=lambda x: int(x[1].get("port", 0)))
        for proto, p in rows:
            port = int(p.get("port", 0))
            port_text = (Text(str(port), style=f"bold {COLOR['info']}")
                         if port in _INFRASTRUCTURE_PORTS
                         else Text(str(port)))
            t.add_row(
                proto,
                port_text,
                p.get("localAddress") or "*",
                _proc_text(p.get("process", "—")),
            )
        try:
            self.query_one("#ports-meta", Label).update(
                Text(f"{len(rows)} listening sockets",
                     style=f"{COLOR['muted']}"))
        except Exception:
            pass

    def _render_ufw(self, ufw: dict) -> None:
        active = ufw.get("active")
        meta_t = Text()
        if active is True:
            meta_t.append("ACTIVE  ", style=f"bold {COLOR['ok']}")
        elif active is False:
            meta_t.append("INACTIVE  ", style=f"bold {COLOR['crit']}")
        else:
            meta_t.append("UNKNOWN  ", style=f"{COLOR['dim']}")
        meta_t.append(
            f'in={ufw.get("defaultIncoming","—")}  '
            f'out={ufw.get("defaultOutgoing","—")}  '
            f'fwd={ufw.get("defaultForward","—")}',
            style=f"{COLOR['muted']}",
        )
        try:
            self.query_one("#ufw-meta", Label).update(meta_t)
        except Exception:
            pass

        t = self.query_one("#ufw-tbl", DataTable)
        t.clear()
        for r in ufw.get("rules") or []:
            action = r.get("action", "—")
            t.add_row(
                str(r.get("number", "—")),
                r.get("to", "—"),
                Text(action, style=f"{_action_color(action)}"),
                r.get("direction", "—"),
                r.get("from", "—"),
            )

    def _render_chain(self, ident: str, chain: dict | None) -> None:
        t = self.query_one(f"#{ident}", DataTable)
        t.clear()
        if not chain:
            t.add_row("—", "(not available)", "", "", "", "", "", "")
            return
        for r in chain.get("rules") or []:
            target = r.get("target", "—")
            t.add_row(
                str(r.get("lineNumber", "—")),
                Text(target, style=f"{_action_color(target)}"),
                r.get("protocol", "—"),
                r.get("in", "—"), r.get("out", "—"),
                truncate(r.get("source", "—"), 18),
                truncate(r.get("destination", "—"), 18),
                truncate(r.get("options", ""), 30),
            )
