"""
screens/services.py — systemd unit health.

The Node Agent reports a fixed set plus any wg-quick@ units found
dynamically.  Their colour-graded status answers the operator's
question: 'Are the things this node depends on actually running?'
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import DataTable, Label

from config import cfg
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen
from theme import COLOR
from util.format import truncate


def _state_text(active: bool | None, sub_state: str | None = None) -> Text:
    if active is True:
        out = Text("active", style=f"bold {COLOR['ok']}")
    elif active is False:
        out = Text("inactive", style=f"bold {COLOR['crit']}")
    else:
        out = Text("?", style=f"{COLOR['dim']}")
    if sub_state:
        out.append(f" / {sub_state}", style=f"{COLOR['muted']}")
    return out


# Names that are critical for node operation — flag when not active.
_CRITICAL = {"decloud-node-agent", "libvirtd"}


class ServicesScreen(BaseScreen):
    ACTIVE_LABEL = "Services"

    DEFAULT_CSS = """
    ServicesScreen DataTable { height: 1fr; }
    ServicesScreen #svc-meta { height: 1; color: $text-muted; margin-bottom: 1; }
    """

    def compose_content(self) -> ComposeResult:
        yield Label("", id="svc-meta")
        t = DataTable(id="svc-tbl", zebra_stripes=True)
        t.add_columns("Unit", "State", "Description", "Loaded")
        yield t

    def on_mount(self) -> None:
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            services = await na.dashboard_services()
        except Exception as exc:
            self.notify(str(exc), severity="error")
            services = []
        finally:
            await na.close()

        t = self.query_one("#svc-tbl", DataTable)
        t.clear()
        # Order: critical first (failing first), then alphabetical.
        def sk(s: dict) -> tuple:
            name = s.get("name", "")
            crit = name in _CRITICAL
            active = bool(s.get("isActive"))
            return (0 if (crit and not active) else 1 if crit else 2, name)
        services.sort(key=sk)

        crit_down = 0
        for s in services:
            name = s.get("name", "—")
            active = s.get("isActive")
            if name in _CRITICAL and not active:
                crit_down += 1
            t.add_row(
                Text(name, style="bold" if name in _CRITICAL else ""),
                _state_text(active, s.get("subState")),
                truncate(s.get("description", ""), 50),
                s.get("loadState", "—"),
            )

        sev_color = COLOR["crit"] if crit_down else COLOR["muted"]
        meta = Text()
        meta.append(f"{len(services)} units · ", style=f"{COLOR['muted']}")
        if crit_down:
            meta.append(f"{crit_down} CRITICAL down",
                        style=f"bold {sev_color}")
        else:
            meta.append("all critical units active",
                        style=f"{COLOR['ok']}")
        try:
            self.query_one("#svc-meta", Label).update(meta)
        except Exception:
            pass
        self.mark_updated()
