"""screens/logs.py — Live log tail for this node."""

from __future__ import annotations

import re as _re
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Input, Label, Log

from config import cfg
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen

_LEVEL_COLOR = {"INF":"blue","WRN":"yellow","ERR":"red","OK":"green","DBG":"dim"}
_LEVELS = ["ALL","INFO","WARN","ERR"]


class LogsScreen(BaseScreen):
    ACTIVE_LABEL = "Live Logs"
    BINDINGS = [("r", "refresh", "Refresh"), ("c", "clear_log", "Clear")]

    def __init__(self) -> None:
        super().__init__()
        self._filter_level: str = "ALL"
        self._filter_text:  str = ""

    def compose_content(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Input(placeholder="Filter logs…", id="log-search")
            for lvl in _LEVELS:
                yield Button(lvl, id=f"lvl-{lvl}", variant="default")
            yield Button("Clear", id="btn-clear")
        yield Log(id="log-out", max_lines=cfg.log_lines)
        yield Label("", id="log-status")

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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("lvl-"):
            self._filter_level = bid.removeprefix("lvl-")
            for lvl in _LEVELS:
                self.query_one(f"#lvl-{lvl}", Button).variant = (
                    "primary" if lvl == self._filter_level else "default"
                )
            self.run_worker(self._load(), exclusive=True)
        elif bid == "btn-clear":
            self.action_clear_log()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value.lower()
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            try:
                self.query_one("#log-status", Label).update(
                    "Set DECLOUD_NODE_URL to enable logs"
                )
            except Exception:
                pass
            return

        na = NodeAgentClient(cfg.node_url)
        try:
            entries = await na.get_logs(lines=cfg.log_lines)
        except Exception as e:
            try:
                self.query_one("#log-status", Label).update(f"Error: {e}")
            except Exception:
                pass
            return
        finally:
            await na.close()

        log_widget = self.query_one("#log-out", Log)
        log_widget.clear()
        for entry in entries:
            line = self._format(entry)
            if line:
                log_widget.write_line(line)

        try:
            self.query_one("#log-status", Label).update(
                f"● {len(entries)} entries"
            )
        except Exception:
            pass

    def _format(self, entry: dict) -> str | None:
        ts    = entry.get("timestamp", entry.get("time",""))[:19]
        level = (entry.get("level") or entry.get("lvl") or "INF").upper()[:3]
        msg   = entry.get("message") or entry.get("msg") or str(entry)

        if self._filter_level != "ALL":
            map_ = {"INFO":"INF","WARN":"WRN","ERR":"ERR"}
            if level != map_.get(self._filter_level, self._filter_level):
                return None
        if self._filter_text and self._filter_text not in msg.lower():
            return None

        color = _LEVEL_COLOR.get(level, "white")
        # strip any markup tags before displaying
        msg = _re.sub(r"\[/?[\w ]+\]", "", msg)
        return f"{ts} [{color}]{level}[/{color}] {msg}"