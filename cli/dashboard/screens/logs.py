"""
screens/logs.py — Live log tail screen.

Polls node agent GET /api/dashboard/logs on refresh_interval.
Filter by level (ALL / INFO / WARN / ERR) and keyword search.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Log

from config import cfg
from api.node_agent import NodeAgentClient


_LEVEL_COLOR = {
    "INF": "blue",
    "WRN": "yellow",
    "ERR": "red",
    "OK": "green",
    "DBG": "dim",
}

_LEVELS = ["ALL", "INFO", "WARN", "ERR"]


class LogsScreen(Vertical):
    _is_mounted: bool = False

    """Live log tail with level filter and keyword search."""

    BINDINGS = [("r", "refresh", "Refresh"), ("c", "clear", "Clear")]

    _filter_level: str = "ALL"
    _filter_text: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Input(placeholder="Filter logs…", id="log-search")
            for lvl in _LEVELS:
                yield Button(lvl, id=f"lvl-{lvl}", variant="default")
            yield Button("Clear", id="btn-clear", variant="default")

        yield Log(id="log-out", max_lines=cfg.log_lines, markup=True)
        yield Label("● live poll", id="log-status")

    def on_mount(self) -> None:
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)
        # Highlight the active level button
        self.query_one("#lvl-ALL", Button).variant = "primary"

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    def action_clear(self) -> None:
        self.query_one("#log-out", Log).clear()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("lvl-"):
            self._filter_level = btn_id.removeprefix("lvl-")
            for lvl in _LEVELS:
                self.query_one(f"#lvl-{lvl}", Button).variant = (
                    "primary" if lvl == self._filter_level else "default"
                )
            self.run_worker(self._load(), exclusive=True)
        elif btn_id == "btn-clear":
            self.action_clear()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value.lower()
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            self.query_one("#log-status", Label).update(
                "[yellow]Node agent not configured — set DECLOUD_NODE_URL[/]"
            )
            return

        client = NodeAgentClient(cfg.node_url)
        try:
            entries = await client.get_logs(lines=cfg.log_lines)
        except Exception as e:
            self.query_one("#log-status", Label).update(f"[red]Error: {e}[/]")
            return
        finally:
            await client.close()

        log_widget = self.query_one("#log-out", Log)
        log_widget.clear()
        for entry in entries:
            line = self._format(entry)
            if line:
                log_widget.write_line(line)

        self.query_one("#log-status", Label).update(
            f"[dim]● {len(entries)} entries — refreshed[/]"
        )

    def _format(self, entry: dict) -> str | None:
        ts = entry.get("timestamp", entry.get("time", ""))[:19]
        level = (entry.get("level") or entry.get("lvl") or "INF").upper()[:3]
        msg = entry.get("message") or entry.get("msg") or str(entry)

        # Level filter
        if self._filter_level != "ALL":
            map_ = {"INFO": "INF", "WARN": "WRN", "ERR": "ERR"}
            if level != map_.get(self._filter_level, self._filter_level):
                return None

        # Text filter
        if self._filter_text and self._filter_text not in msg.lower():
            return None

        color = _LEVEL_COLOR.get(level, "white")
        return f"[dim]{ts}[/] [[{color}]{level}[/]] {msg}"