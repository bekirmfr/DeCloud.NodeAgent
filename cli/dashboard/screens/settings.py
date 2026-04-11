"""screens/settings.py — Connection and display settings."""

from __future__ import annotations

import stat
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, Select, Switch

from config import cfg
from screens._base import BaseScreen

_USER_CONFIG = Path.home() / ".decloud" / "config"
_REFRESH_OPTS = [("3 seconds","3"),("5 seconds","5"),("10 seconds","10"),
                 ("30 seconds","30"),("60 seconds","60")]


class SettingsScreen(BaseScreen):
    ACTIVE_LABEL = "Settings"
    BINDINGS = [("ctrl+s", "save", "Save")]

    def compose_content(self) -> ComposeResult:
        yield Label("Settings", classes="section-title")

        with Vertical(classes="settings-group"):
            yield Label("Connection", classes="settings-group-title")
            with Horizontal(classes="setting-row"):
                yield Label("Orchestrator URL", classes="setting-label")
                yield Input(value=cfg.orchestrator_url, id="inp-orch-url",
                            placeholder="https://orchestrator.example.com")
            with Horizontal(classes="setting-row"):
                yield Label("API Token", classes="setting-label")
                yield Input(value="●●●●●●●●" if cfg.token else "",
                            id="inp-token", password=True)
            with Horizontal(classes="setting-row"):
                yield Label("Node Agent URL", classes="setting-label")
                yield Input(value=cfg.node_url, id="inp-node-url",
                            placeholder="http://localhost:5100")

        with Vertical(classes="settings-group"):
            yield Label("Display", classes="settings-group-title")
            with Horizontal(classes="setting-row"):
                yield Label("Refresh interval", classes="setting-label")
                yield Select(options=_REFRESH_OPTS, value=str(cfg.refresh_interval),
                             id="sel-refresh", allow_blank=False)
            with Horizontal(classes="setting-row"):
                yield Label("Log tail lines", classes="setting-label")
                yield Input(value=str(cfg.log_lines), id="inp-log-lines")
            with Horizontal(classes="setting-row"):
                yield Label("Node-only mode", classes="setting-label")
                yield Switch(value=cfg.node_only, id="sw-node-only")

        with Horizontal(id="settings-actions"):
            yield Button("Test Connection",          id="btn-test")
            yield Button("Apply (this session)",     id="btn-apply", variant="primary")
            yield Button("Save to ~/.decloud/config", id="btn-save")

        yield Label("", id="settings-status")

    DEFAULT_CSS = """
    SettingsScreen .settings-group {
        border: solid $panel;
        padding: 1 2;
        margin-bottom: 1;
    }
    SettingsScreen .settings-group-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    SettingsScreen .setting-row {
        height: 4;
        align: left middle;
        margin-bottom: 1;
    }
    SettingsScreen .setting-label { width: 22; color: $text-muted; }
    SettingsScreen Input  { width: 1fr; }
    SettingsScreen Select { width: 1fr; }
    """

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
        url   = self.query_one("#inp-orch-url", Input).value.strip().rstrip("/")
        token = self.query_one("#inp-token", Input).value.strip()
        node  = self.query_one("#inp-node-url", Input).value.strip().rstrip("/")
        lines = self.query_one("#inp-log-lines", Input).value.strip()
        ref   = self.query_one("#sel-refresh", Select).value
        nonly = self.query_one("#sw-node-only", Switch).value

        if url: cfg.orchestrator_url = url
        if token and token != "●●●●●●●●": cfg.token = token
        if node: cfg.node_url = node
        if lines.isdigit(): cfg.log_lines = int(lines)
        if ref: cfg.refresh_interval = int(str(ref))
        cfg.node_only = nonly
        self._status("Applied for this session.", "green")

    def _save_to_disk(self) -> None:
        _USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        _USER_CONFIG.write_text(
            "\n".join([
                f"DECLOUD_URL={cfg.orchestrator_url}",
                f"DECLOUD_TOKEN={cfg.token}",
                f"DECLOUD_NODE_URL={cfg.node_url}",
                f"DECLOUD_REFRESH_INTERVAL={cfg.refresh_interval}",
                f"DECLOUD_LOG_LINES={cfg.log_lines}",
            ]) + "\n"
        )
        _USER_CONFIG.chmod(0o600)
        self._status(f"Saved to {_USER_CONFIG} (chmod 600)", "green")

    async def _test_connection(self) -> None:
        self._status("Testing…", "yellow")
        from api.orchestrator import OrchestratorClient, ApiError
        oc = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            stats = await oc.get_stats()
            self._status(f"OK — {stats.get('totalNodes','?')} nodes", "green")
        except ApiError as e:
            self._status(f"Error {e.status}: {e.message}", "red")
        except Exception as e:
            self._status(f"Failed: {e}", "red")
        finally:
            await oc.close()

    def _status(self, msg: str, color: str = "white") -> None:
        try:
            self.query_one("#settings-status", Label).update(f"[{color}]{msg}[/]")
        except Exception:
            pass