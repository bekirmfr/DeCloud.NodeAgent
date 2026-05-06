"""
widgets/header.py — Top identity bar and live status strip.

Two stacked widgets, always visible above the content area:

  IdentityBar     │ ● connected  nd_…  host MSI  os Ubuntu  uptime 4d  agent 1.3.2  wallet 0x7E…a2C1
  StatusStrip     │ ●Orch 8s ago   CPU 32%   RAM 41%   STOR 12%   VMs 6   ERR 0

The IdentityBar makes the auth/connection state the first thing the
operator sees. If we have orchestrator credentials and an active
heartbeat, that's "● connected" in green; if we're running node-only
without a token, that's "○ anonymous" in dim grey; if we have a token
but the orchestrator isn't reachable, that's "● auth lost" in amber.

Owned by the App, fed by polling on the Overview screen which
centralises the heaviest fetches. Other screens may also push updates.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from theme import COLOR, DOT_FILLED, DOT_OPEN
from widgets.statpill import status_text
from util.format import fmt_duration, short_id


class IdentityBar(Static):
    """Top line: auth state + node identity.

    Built via Static.update() rather than overriding render() so we
    fully control the visual cache (an earlier render-override version
    appeared blank on some terminals because the cached visual from
    the empty-identity first paint was being re-served even after
    refresh()).
    """

    DEFAULT_CSS = """
    IdentityBar {
        height: 1;
        padding: 0 1;
        /* No tinted background — keeps the foreground text readable
         * on terminals with non-truecolor or aggressive contrast. */
        background: transparent;
        color: $text;
    }
    """

    def on_mount(self) -> None:
        # Render an initial state immediately so the bar is visible even
        # before the first /api/dashboard/summary fetch lands.
        self.refresh_identity()

    def refresh_identity(self) -> None:
        """Public hook — rebuild the bar from cfg.identity.

        Call this after cfg.identity has been updated. Internally calls
        Static.update(), which is the canonical way to change a Static's
        content (refresh() alone wouldn't invalidate the visual cache).
        """
        from config import cfg
        n = cfg.identity or {}

        out = Text(no_wrap=True, overflow="ellipsis")

        # ── Auth / connection state — first, prominent ────────────
        orch = (n.get("orchestrator") or {}) if isinstance(n, dict) else {}
        connected = bool(orch.get("connected"))
        has_token = bool(getattr(cfg, "token", None))

        if connected:
            out.append(DOT_FILLED + " ", style=f"bold {COLOR['ok']}")
            out.append("connected", style=f"bold {COLOR['ok']}")
        elif has_token:
            out.append(DOT_FILLED + " ", style=f"bold {COLOR['warn']}")
            out.append("auth lost", style=f"bold {COLOR['warn']}")
        else:
            out.append(DOT_OPEN + " ", style=f"{COLOR['muted']}")
            out.append("anonymous", style=f"{COLOR['muted']}")

        out.append("   ")

        # ── Node identity ─────────────────────────────────────────
        node_id = n.get("nodeId") if isinstance(n, dict) else None
        out.append(short_id(node_id, 16), style=f"bold {COLOR['info']}")
        out.append("   ")

        for label, val in [
            ("host",   n.get("hostname") if isinstance(n, dict) else None),
            ("os",     n.get("os") if isinstance(n, dict) else None),
            ("uptime", fmt_duration(n.get("uptimeSeconds") if isinstance(n, dict) else None)),
            ("agent",  n.get("agentVersion") if isinstance(n, dict) else None),
        ]:
            out.append(f"{label} ", style=f"{COLOR['muted']}")
            out.append(f"{val or '—'}   ", style=f"{COLOR['title']}")

        # ── Wallet (visible, not "dim") ───────────────────────────
        wallet = n.get("walletAddress") if isinstance(n, dict) else None
        if wallet and len(wallet) > 14:
            short_wallet = wallet[:8] + "…" + wallet[-4:]
            out.append("wallet ", style=f"{COLOR['muted']}")
            out.append(short_wallet, style=f"bold {COLOR['info']}")
        elif wallet:
            out.append("wallet ", style=f"{COLOR['muted']}")
            out.append(wallet, style=f"bold {COLOR['info']}")

        self.update(out)


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
        color: $text-muted;
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

