"""
screens/ingress.py — Ingress route table.

Shows all active subdomain → VM mappings.
Source: orchestrator GET /api/ingress + /api/central-ingress/status.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Label

from config import cfg
from api.orchestrator import OrchestratorClient, ApiError


_STATUS_COLOR = {"active": "green", "pending": "yellow", "inactive": "red"}


class IngressScreen(Screen):
    """Ingress route viewer."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label("Ingress Routes", classes="section-title")
        yield Label("", id="ingress-meta")
        yield DataTable(id="ingress-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#ingress-table", DataTable)
        table.add_columns("VM ID", "VM Name", "Public URL", "Node", "Port", "Status")
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_orchestrator:
            return

        client = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            import asyncio
            status, routes = await asyncio.gather(
                client.get_ingress_status(),
                client.list_ingress_routes(),
                return_exceptions=True,
            )
        finally:
            await client.close()

        if isinstance(status, dict):
            base = status.get("baseDomain", "—")
            enabled = status.get("isEnabled", False)
            self.query_one("#ingress-meta", Label).update(
                f"Base domain: [cyan]{base}[/]  TLS: "
                f"{'[green]enabled[/]' if enabled else '[red]disabled[/]'}",
            )

        if isinstance(routes, list):
            self._render_routes(routes)

    def _render_routes(self, routes: list) -> None:
        table = self.query_one("#ingress-table", DataTable)
        table.clear()
        for r in routes:
            status = (r.get("status") or "").lower()
            color = _STATUS_COLOR.get(status, "white")
            table.add_row(
                r.get("vmId", "—")[:12],
                r.get("vmName", "—"),
                r.get("publicUrl") or r.get("url", "—"),
                r.get("nodeId", "—")[:12],
                str(r.get("targetPort", "—")),
                f"[{color}]{status or '—'}[/]",
            )
