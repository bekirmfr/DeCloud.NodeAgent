"""screens/ingress.py — Ingress routes."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import DataTable, Label

from config import cfg
from api.orchestrator import OrchestratorClient
from screens._base import BaseScreen

_STATUS_COLOR = {"active":"green","pending":"yellow","inactive":"red"}


class IngressScreen(BaseScreen):
    ACTIVE_LABEL = "Ingress Routes"
    BINDINGS = [("r", "refresh", "Refresh")]

    def compose_content(self) -> ComposeResult:
        yield Label("Ingress Routes", classes="section-title")
        yield Label("", id="ingress-meta")
        yield DataTable(id="ingress-table", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one("#ingress-table", DataTable).add_columns(
            "VM ID", "VM Name", "Public URL", "Node", "Port", "Status"
        )
        self.set_interval(cfg.refresh_interval, self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_orchestrator:
            return
        import asyncio
        oc = OrchestratorClient(cfg.orchestrator_url, cfg.token)
        try:
            status, routes = await asyncio.gather(
                oc.get_ingress_status(), oc.list_ingress_routes(),
                return_exceptions=True,
            )
        finally:
            await oc.close()

        if isinstance(status, dict):
            base    = status.get("baseDomain","—")
            enabled = status.get("isEnabled", False)
            try:
                self.query_one("#ingress-meta", Label).update(
                    f"Base domain: [cyan]{base}[/]  TLS: "
                    f"{'[green]enabled[/]' if enabled else '[red]disabled[/]'}"
                )
            except Exception:
                pass

        if isinstance(routes, list):
            t = self.query_one("#ingress-table", DataTable)
            t.clear()
            for r in routes:
                status_str = (r.get("status") or "").lower()
                color = _STATUS_COLOR.get(status_str, "white")
                t.add_row(
                    r.get("vmId","—")[:12], r.get("vmName","—"),
                    r.get("publicUrl") or r.get("url","—"),
                    r.get("nodeId","—")[:12], str(r.get("targetPort","—")),
                    f"[{color}]{status_str or '—'}[/]",
                )