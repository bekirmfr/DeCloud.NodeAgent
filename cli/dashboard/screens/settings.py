"""
screens/settings.py — Connection and display settings screen.

Allows runtime override of orchestrator URL, token, node agent URL,
refresh interval. Changes are applied immediately in-process and
optionally written to ~/.decloud/config (chmod 600).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static, Switch

from config import cfg

_USER_CONFIG = Path.home() / ".decloud" / "config"

_REFRESH_OPTS = [
    ("3 seconds", "3"),
    ("5 seconds", "5"),
    ("10 seconds", "10"),
    ("30 seconds", "30"),
    ("60 seconds", "60"),
]


class SettingsGroup(Static):
    """Titled group of settings rows."""

    DEFAULT_CSS = """
    SettingsGroup {
        border: solid $panel;
        padding: 1 2;
        margin-bottom: 1;
    }
    SettingsGroup .group-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    SettingsGroup .setting-row {
        height: 3;
        layout: horizontal;
        align: left middle;
        margin-bottom: 1;
    }
    SettingsGroup .setting-label {
        width: 22;
        color: $text-muted;
    }
    SettingsGroup .setting-control {
        width: 1fr;
    }
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="group-title")
        yield from self._rows()

    def _rows(self) -> ComposeResult:
        return
        yield  # type: ignore[misc]


class ConnectionSettings(SettingsGroup):
    def __init__(self) -> None:
        super().__init__("Connection")

    def _rows(self) -> ComposeResult:
        with Horizontal(classes="setting-row"):
            yield Label("Orchestrator URL", classes="setting-label")
            yield Input(
                value=cfg.orchestrator_url,
                placeholder="https://orchestrator.decloud.io",
                id="inp-orch-url",
                classes="setting-control",
            )

        with Horizontal(classes="setting-row"):
            yield Label("API Token", classes="setting-label")
            yield Input(
                value="●●●●●●●●" if cfg.token else "",
                placeholder="paste JWT token",
                id="inp-token",
                password=True,
                classes="setting-control",
            )

        with Horizontal(classes="setting-row"):
            yield Label("Node Agent URL", classes="setting-label")
            yield Input(
                value=cfg.node_url,
                placeholder="http://localhost:5100",
                id="inp-node-url",
                classes="setting-control",
            )


class DisplaySettings(SettingsGroup):
    def __init__(self) -> None:
        super().__init__("Display")

    def _rows(self) -> ComposeResult:
        with Horizontal(classes="setting-row"):
            yield Label("Refresh interval", classes="setting-label")
            yield Select(
                options=_REFRESH_OPTS,
                value=str(cfg.refresh_interval),
                id="sel-refresh",
                classes="setting-control",
                allow_blank=False,
            )

        with Horizontal(classes="setting-row"):
            yield Label("Log tail lines", classes="setting-label")
            yield Input(
                value=str(cfg.log_lines),
                id="inp-log-lines",
                classes="setting-control",
            )

        with Horizontal(classes="setting-row"):
            yield Label("Node-only mode", classes="setting-label")
            yield Switch(value=cfg.node_only, id="sw-node-only")


class SettingsScreen(Container):
    _is_mounted: bool = False
    _running: bool = False

    """Connection and display settings."""

    BINDINGS = [("ctrl+s", "save", "Save to disk")]

    def compose(self) -> ComposeResult:
        yield Label("Settings", classes="section-title")
        yield ConnectionSettings()
        yield DisplaySettings()

        with Horizontal(id="settings-actions"):
            yield Button("Test Connection", id="btn-test", variant="default")
            yield Button("Apply (this session)", id="btn-apply", variant="primary")
            yield Button("Save to ~/.decloud/config", id="btn-save", variant="default")

        yield Label("", id="settings-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        if btn == "btn-test":
            self.run_worker(self._test_connection(), exclusive=True)
        elif btn == "btn-apply":
            self._apply()
        elif btn == "btn-save":
            self._apply()
            self._save_to_disk()

    def action_save(self) -> None:
        self._apply()
        self._save_to_disk()

    def _apply(self) -> None:
        """Apply values in-process without writing to disk."""
        orch_url = self.query_one("#inp-orch-url", Input).value.strip().rstrip("/")
        token_raw = self.query_one("#inp-token", Input).value.strip()
        node_url = self.query_one("#inp-node-url", Input).value.strip().rstrip("/")
        log_lines = self.query_one("#inp-log-lines", Input).value.strip()
        refresh = self.query_one("#sel-refresh", Select).value
        node_only = self.query_one("#sw-node-only", Switch).value

        if orch_url:
            cfg.orchestrator_url = orch_url
        # Only update token if the user actually changed it (not the placeholder)
        if token_raw and token_raw != "●●●●●●●●":
            cfg.token = token_raw
        if node_url:
            cfg.node_url = node_url
        if log_lines.isdigit():
            cfg.log_lines = int(log_lines)
        if refresh:
            cfg.refresh_interval = int(str(refresh))
        cfg.node_only = node_only

        self._status("Settings applied for this session.", "green")

    def _save_to_disk(self) -> None:
        """Write current cfg to ~/.decloud/config (chmod 600)."""
        _USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"DECLOUD_URL={cfg.orchestrator_url}",
            f"DECLOUD_TOKEN={cfg.token}",
            f"DECLOUD_NODE_URL={cfg.node_url}",
            f"DECLOUD_REFRESH_INTERVAL={cfg.refresh_interval}",
            f"DECLOUD_LOG_LINES={cfg.log_lines}",
        ]
        _USER_CONFIG.write_text("\n".join(lines) + "\n")
        _USER_CONFIG.chmod(0o600)
        self._status(f"Saved to {_USER_CONFIG} (chmod 600)", "green")

    async def _test_connection(self) -> None:
        self._status("Testing…", "yellow")
        from api.orchestrator import OrchestratorClient, ApiError
        client = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            stats = await client.get_stats()
            nodes = stats.get("totalNodes", "?")
            self._status(f"Orchestrator OK — {nodes} nodes registered", "green")
        except ApiError as e:
            self._status(f"Orchestrator error {e.status}: {e.message}", "red")
        except Exception as e:
            self._status(f"Connection failed: {e}", "red")
        finally:
            await client.close()

    def _status(self, msg: str, color: str = "white") -> None:
        self.query_one("#settings-status", Label).update(
            f"[{color}]{msg}[/]"
        )