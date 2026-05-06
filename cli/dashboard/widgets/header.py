"""
widgets/header.py — Top identity bar and live status strip.

Two stacked widgets, always visible above the content area:

  IdentityBar     │ DECLOUD <node-id>   <hostname> · <os> · uptime · v<agent>
  StatusStrip     │ ●Orch 8s ago   CPU 32%   RAM 41%   STOR 12%   VMs 6   ERR 0

Owned by the App, fed by polling on the Overview screen which centralises
the heaviest fetches. Other screens may also push updates.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

from theme import COLOR
from widgets.statpill import status_text
from util.format import fmt_duration, short_id


class IdentityBar(Static):
    """Top line: brand mark + node identity."""

    DEFAULT_CSS = """
    IdentityBar {
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $text;
    }
    """

    def render(self) -> Text:
        from config import cfg
        n = cfg.identity
        out = Text()
        out.append("DECLOUD ", style=f"bold {COLOR['info']}")
        out.append(short_id(n.get("nodeId"), 16), style="bold")
        out.append("   ", style="")
        for label, val in [
            ("host",    n.get("hostname", "—")),
            ("os",      n.get("os", "—")),
            ("uptime",  fmt_duration(n.get("uptimeSeconds"))),
            ("agent",   n.get("agentVersion", "—") or "—"),
        ]:
            out.append(f"{label} ", style=f"{COLOR['muted']}")
            out.append(f"{val}   ")
        wallet = n.get("walletAddress")
        if wallet:
            out.append("wallet ", style=f"{COLOR['muted']}")
            out.append(short_id(wallet, 10) + "…" + wallet[-4:], style="dim")
        return out


class StatusStrip(Widget):
    """Six small live tiles. Each is a Static updated externally."""

    DEFAULT_CSS = """
    StatusStrip {
        height: 1;
        layout: horizontal;
        background: $surface;
        color: $text;
    }
    StatusStrip > Static {
        width: 1fr;
        padding: 0 1;
        content-align: left middle;
    }
    StatusStrip > .strip-sep {
        width: 1;
        color: $text-disabled;
    }
    """

    TILE_IDS = ("orch", "cpu", "ram", "stor", "vms", "err")

    def compose(self) -> ComposeResult:
        for i, tid in enumerate(self.TILE_IDS):
            if i:
                yield Static("│", classes="strip-sep")
            yield Static("—", id=f"strip-{tid}")

    def set_orch(self, severity: str, age: str) -> None:
        self._set("orch", status_text(severity, f"Orch {age}"))

    def set_metric(self, key: str, label: str, value: str, severity: str = "info") -> None:
        color = {
            "ok": COLOR["ok"], "warn": COLOR["warn"], "crit": COLOR["crit"],
            "info": COLOR["info"], "unknown": COLOR["dim"],
        }.get(severity, COLOR["info"])
        out = Text()
        out.append(f"{label} ", style=f"{COLOR['muted']}")
        out.append(value, style=f"bold {color}")
        self._set(key, out)

    def _set(self, key: str, text: Text) -> None:
        try:
            self.query_one(f"#strip-{key}", Static).update(text)
        except Exception:
            pass
