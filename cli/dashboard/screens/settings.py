"""
screens/settings.py — Connection configuration + login/logout.

Edits ~/.decloud/config (chmod 0600). Token is stored as-is
in the file, but the on-screen field is a password input so
the value isn't shown over-the-shoulder.

Authentication:
  Login  — suspends the TUI, runs `decloud login` interactively
           (QR code for wallet scan), then resumes.
  Logout — runs `decloud logout` in background, clears credentials.
  Both require root (the dashboard is typically run as root via
  `sudo decloud dashboard`).

Rule: pressing Save re-validates the config and persists; on
success the changes take effect on the next refresh tick (no
restart needed for refresh-interval / URL changes — only for
auth state, which existing requests will start using on retry).
"""

from __future__ import annotations

import subprocess
import shutil
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
        border-title-color: $accent;
        border-title-style: bold;
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
    SettingsScreen #auth-status { height: 2; margin-bottom: 1; }
    SettingsScreen #auth-actions Button { margin-right: 1; }
    """

    def compose_content(self) -> ComposeResult:
        # ── Authentication group ──────────────────────────────────
        with Vertical(classes="group"):
            yield Label("Authentication", classes="group-title")
            yield Label("", id="auth-status")
            with Horizontal(id="auth-actions"):
                yield Button("Login with wallet", id="btn-login",
                             variant="primary")
                yield Button("Logout", id="btn-logout", variant="error")

        # ── Connection group ──────────────────────────────────────
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

    def on_mount(self) -> None:
        self._refresh_auth_status()

    def _refresh_auth_status(self) -> None:
        """Update the auth status label from cfg.identity."""
        identity = cfg.identity or {}
        node_id = identity.get("nodeId") if isinstance(identity, dict) else None
        wallet = identity.get("walletAddress") if isinstance(identity, dict) else None
        orch = (identity.get("orchestrator") or {}) if isinstance(identity, dict) else {}
        connected = bool(orch.get("connected"))

        out = Text()
        if connected and wallet:
            out.append("● ", style=f"bold {COLOR['ok']}")
            out.append("Registered", style=f"bold {COLOR['ok']}")
            out.append(f"  node: {node_id or '—'}  wallet: {wallet}",
                       style=f"{COLOR['muted']}")
        elif wallet:
            out.append("● ", style=f"bold {COLOR['warn']}")
            out.append("Wallet set, orchestrator disconnected",
                       style=f"bold {COLOR['warn']}")
            out.append(f"  wallet: {wallet}", style=f"{COLOR['muted']}")
        elif node_id and node_id != "unregistered":
            out.append("● ", style=f"bold {COLOR['warn']}")
            out.append("Partially registered", style=f"bold {COLOR['warn']}")
            out.append(f"  node: {node_id}", style=f"{COLOR['muted']}")
        else:
            out.append("○ ", style=f"{COLOR['dim']}")
            out.append("Not authenticated", style=f"{COLOR['dim']}")
            out.append("  — run Login to authenticate with your wallet",
                       style=f"{COLOR['muted']}")

        try:
            self.query_one("#auth-status", Label).update(out)
        except Exception:
            pass

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "btn-save":
            self.action_save()
        elif ev.button.id == "btn-reload":
            cfg.load()
            self._refill()
            self._set_status(f"Reloaded from {Path.home()/'.decloud'/'config'}",
                             "info")
        elif ev.button.id == "btn-login":
            self._do_login()
        elif ev.button.id == "btn-logout":
            self._do_logout()

    def _do_login(self) -> None:
        """Suspend the TUI and run `decloud login` interactively."""
        decloud_bin = shutil.which("decloud")
        if not decloud_bin:
            self._set_status(
                "Cannot find `decloud` command — is it installed in PATH?",
                "crit")
            return

        self._set_status("Suspending dashboard for login…", "info")

        try:
            with self.app.suspend():
                # The terminal is now under `decloud login`'s control.
                # It will show the QR code and wait for wallet scan.
                result = subprocess.run(
                    [decloud_bin, "login"],
                    check=False,
                )
            # TUI resumes here
            if result.returncode == 0:
                self._set_status("Login completed — refreshing state…", "ok")
            else:
                self._set_status(
                    f"Login exited with code {result.returncode}", "warn")
        except Exception as exc:
            self._set_status(f"Login failed: {exc}", "crit")

        # Refresh auth display and force the Overview to re-fetch summary
        self._refresh_auth_status()

    def _do_logout(self) -> None:
        """Run `decloud logout` in the background."""
        decloud_bin = shutil.which("decloud")
        if not decloud_bin:
            self._set_status(
                "Cannot find `decloud` command — is it installed in PATH?",
                "crit")
            return

        try:
            result = subprocess.run(
                [decloud_bin, "logout"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                cfg.identity = {}
                self._set_status("Logged out — credentials removed.", "ok")
            else:
                err = (result.stderr or result.stdout or "").strip()
                self._set_status(
                    f"Logout failed (code {result.returncode}): {err}", "crit")
        except subprocess.TimeoutExpired:
            self._set_status("Logout timed out", "crit")
        except Exception as exc:
            self._set_status(f"Logout failed: {exc}", "crit")

        self._refresh_auth_status()

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
