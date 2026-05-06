"""
screens/logs.py — Filterable log tail.

Pulls /api/dashboard/logs?lines=N. Filtering by level and free text is
applied client-side because the endpoint does not accept filter params.

The Log widget is replaced with one fresh write per refresh — keeps the
viewport scroll position predictable for filtering.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Input, Label, Log
from widgets.nav_input import NavInput

from config import cfg
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen
from theme import COLOR


_LEVELS = ["ALL", "INFO", "WARN", "ERR"]
_LEVEL_COLOR = {
    "ERR": COLOR["crit"],  "ERROR": COLOR["crit"],
    "WRN": COLOR["warn"],  "WARN":  COLOR["warn"],
    "INF": COLOR["info"],  "INFO":  COLOR["info"],
    "DBG": COLOR["dim"],   "DEBUG": COLOR["dim"],
}


class LogsScreen(BaseScreen):
    ACTIVE_LABEL = "Logs"
    EXTRA_HINTS = [("c", "clear")]

    BINDINGS = [("c", "clear_log", "Clear")]

    DEFAULT_CSS = """
    LogsScreen #log-bar { height: 3; margin-bottom: 1; }
    LogsScreen #log-bar Input  { width: 32; margin-right: 1; }
    LogsScreen #log-bar Button { width: auto; margin-right: 1; height: 3; }
    LogsScreen #log-out { height: 1fr; }
    LogsScreen #log-meta { height: 1; color: $text-muted; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._level = "ALL"
        self._search = ""

    def compose_content(self) -> ComposeResult:
        with Horizontal(id="log-bar"):
            yield NavInput(placeholder="filter text…", id="log-search")
            for lvl in _LEVELS:
                yield Button(lvl, id=f"lvl-{lvl}", variant="default")
            yield Button("Clear", id="btn-clear")
        yield Log(id="log-out", max_lines=cfg.log_lines)
        yield Label("", id="log-meta")

    def on_mount(self) -> None:
        self.query_one("#lvl-ALL", Button).variant = "primary"
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    def action_clear_log(self) -> None:
        try:
            self.query_one("#log-out", Log).clear()
        except Exception:
            pass

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        bid = ev.button.id or ""
        if bid.startswith("lvl-"):
            self._level = bid.removeprefix("lvl-")
            for lvl in _LEVELS:
                self.query_one(f"#lvl-{lvl}", Button).variant = (
                    "primary" if lvl == self._level else "default")
            self.run_worker(self._load(), exclusive=True)
        elif bid == "btn-clear":
            self.action_clear_log()

    def on_input_changed(self, ev: Input.Changed) -> None:
        if ev.input.id == "log-search":
            self._search = ev.value.lower()
            self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            try:
                self.query_one("#log-meta", Label).update(
                    Text("DECLOUD_NODE_URL not set — no logs available",
                         style=f"{COLOR['warn']}"))
            except Exception:
                pass
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            entries = await na.dashboard_logs(lines=cfg.log_lines)
        except Exception as exc:
            self.notify(str(exc), severity="error")
            entries = []
        finally:
            await na.close()

        log = self.query_one("#log-out", Log)
        log.clear()
        shown = 0
        for entry in entries:
            level = (entry.get("level") or "INF").upper()[:5].rstrip()
            level_short = level[:3]
            if self._level != "ALL":
                want = {"INFO": "INF", "WARN": "WRN", "ERR": "ERR"}[self._level]
                if level_short != want:
                    continue
            msg = entry.get("message") or entry.get("msg") or ""
            ts  = entry.get("timestamp") or entry.get("time") or ""
            line_text = f"{ts} {level} {msg}"
            if self._search and self._search not in line_text.lower():
                continue
            color = _LEVEL_COLOR.get(level_short, COLOR["muted"])
            line = Text()
            line.append(ts[-12:] if ts else "------------", style=f"{COLOR['dim']}")
            line.append(f"  {level_short:<3} ", style=f"bold {color}")
            line.append(msg)
            log.write_line(line)
            shown += 1

        try:
            self.query_one("#log-meta", Label).update(
                Text(f"{shown} / {len(entries)} lines · level={self._level}",
                     style=f"{COLOR['muted']}"))
        except Exception:
            pass
        self.mark_updated()
