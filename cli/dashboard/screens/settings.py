"""
screens/settings.py — Connection configuration.

Edits ~/.decloud/config (chmod 0600). Token is stored as-is
in the file, but the on-screen field is a password input so
the value isn't shown over-the-shoulder.

Rule: pressing Save re-validates the config and persists; on
success the changes take effect on the next refresh tick (no
restart needed for refresh-interval / URL changes — only for
auth state, which existing requests will start using on retry).
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, Select
from widgets.nav_input import NavInput

from config import cfg
from screens._base import BaseScreen
from theme import COLOR


_REFRESH_OPTS = [
    ("3 seconds",  "3"),
    ("5 seconds",  "5"),
    ("10 seconds", "10"),
    ("30 seconds", "30"),
    ("60 seconds", "60"),
]


class SettingsScreen(BaseScreen):
    ACTIVE_LABEL = "Settings"
    EXTRA_HINTS = [("ctrl+s", "save")]

    BINDINGS = [("ctrl+s", "save", "Save")]

    DEFAULT_CSS = """
    SettingsScreen .group {
        border: round $panel;
        padding: 1 2;
        margin-bottom: 1;
        height: auto;
    }
    SettingsScreen .group-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        height: 1;
    }
    SettingsScreen .row { layout: horizontal; height: 3; }
    SettingsScreen .row Label  { width: 22; padding: 1 0 0 0; }
    SettingsScreen .row Input  { width: 1fr; }
    SettingsScreen .row Select { width: 1fr; }
    SettingsScreen #status { color: $text-muted; height: 2; margin-top: 1; }
    SettingsScreen #actions Button { margin-right: 1; }
    """

    def compose_content(self) -> ComposeResult:
        with Vertical(classes="group"):
            yield Label("Connection", classes="group-title")
            with Horizontal(classes="row"):
                yield Label("Orchestrator URL")
                yield NavInput(value=cfg.orchestrator_url,
                            placeholder="https://orchestrator.example.com",
                            id="inp-orch")
            with Horizontal(classes="row"):
                yield Label("API Token")
                yield NavInput(value=cfg.token,
                            placeholder="paste node JWT here",
                            password=True, id="inp-token")
            with Horizontal(classes="row"):
                yield Label("Node Agent URL")
                yield NavInput(value=cfg.node_url,
                            placeholder="http://localhost:5100",
                            id="inp-node")

        with Vertical(classes="group"):
            yield Label("Display", classes="group-title")
            with Horizontal(classes="row"):
                yield Label("Refresh interval")
                yield Select(_REFRESH_OPTS,
                             value=str(cfg.refresh_interval),
                             allow_blank=False, id="inp-refresh")

        with Horizontal(id="actions"):
            yield Button("Save", id="btn-save", variant="primary")
            yield Button("Reload from file", id="btn-reload")

        yield Label("", id="status")

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "btn-save":
            self.action_save()
        elif ev.button.id == "btn-reload":
            cfg.load()
            self._refill()
            self._set_status(f"Reloaded from {Path.home()/'.decloud'/'config'}",
                             "info")

    def action_save(self) -> None:
        cfg.orchestrator_url = self.query_one("#inp-orch",  Input).value.strip().rstrip("/")
        cfg.token            = self.query_one("#inp-token", Input).value.strip()
        cfg.node_url         = self.query_one("#inp-node",  Input).value.strip().rstrip("/")
        try:
            cfg.refresh_interval = max(
                1, int(self.query_one("#inp-refresh", Select).value))
        except (ValueError, TypeError):
            cfg.refresh_interval = 5

        issues = cfg.save()
        if issues:
            self._set_status("Save failed: " + "; ".join(issues), "crit")
        else:
            self._set_status(
                f"Saved (mode 0600) to {Path.home()/'.decloud'/'config'}", "ok")

    def _refill(self) -> None:
        self.query_one("#inp-orch",  Input).value = cfg.orchestrator_url
        self.query_one("#inp-token", Input).value = cfg.token
        self.query_one("#inp-node",  Input).value = cfg.node_url
        try:
            self.query_one("#inp-refresh", Select).value = str(cfg.refresh_interval)
        except Exception:
            pass

    def _set_status(self, msg: str, sev: str) -> None:
        color = {"ok": COLOR["ok"], "warn": COLOR["warn"],
                 "crit": COLOR["crit"], "info": COLOR["info"]
                 }.get(sev, COLOR["muted"])
        try:
            self.query_one("#status", Label).update(
                Text(msg, style=f"bold {color}"))
        except Exception:
            pass
