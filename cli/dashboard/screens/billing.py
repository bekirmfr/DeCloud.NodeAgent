"""screens/billing.py — Billing and transactions."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Label

from config import cfg
from api.orchestrator import OrchestratorClient
from screens._base import BaseScreen


class BillingScreen(BaseScreen):
    ACTIVE_LABEL = "Billing"
    BINDINGS = [("r", "refresh", "Refresh")]

    def compose_content(self) -> ComposeResult:
        yield Label("Billing", classes="section-title")
        with Horizontal(id="balance-row"):
            for cid, lbl in [
                ("bal-available","Balance"), ("bal-earned","Earned (30d)"),
                ("bal-pending","Pending"),   ("bal-vms","Active VMs"),
            ]:
                with Vertical(classes="stat-card", id=cid):
                    yield Label(lbl, classes="card-label")
                    yield Label("—", classes="card-value")

        yield Label("Transaction History", classes="section-title")
        yield DataTable(id="tx-table", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one("#tx-table", DataTable).add_columns(
            "VM / Description", "Duration", "Amount (USDC)", "Trigger", "Status"
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
            balance, txs = await asyncio.gather(
                oc.get_balance(), oc.list_transactions(),
                return_exceptions=True,
            )
        finally:
            await oc.close()

        if isinstance(balance, dict):
            self._set("bal-available", f"${float(balance.get('availableBalance',0)):.2f}")
            self._set("bal-earned",    f"${float(balance.get('earned30d',0)):.2f}")
            self._set("bal-pending",   f"${float(balance.get('pendingBalance',0)):.2f}")
            self._set("bal-vms",       str(balance.get("billedVmCount",balance.get("activeVms","—"))))

        if isinstance(txs, list):
            t = self.query_one("#tx-table", DataTable)
            t.clear()
            for tx in txs:
                amount = tx.get("amount", tx.get("amountUsdc", 0))
                status = tx.get("status","—")
                color  = "green" if status.lower() == "settled" else "yellow"
                t.add_row(
                    f"{tx.get('vmId','—')[:10]}  {tx.get('description','')}",
                    tx.get("duration", tx.get("durationHuman","—")),
                    f"[green]+${float(amount):.2f}[/]",
                    tx.get("trigger","—"),
                    f"[{color}]{status}[/]",
                )

    def _set(self, cid: str, value: str) -> None:
        try:
            self.query_one(f"#{cid} .card-value", Label).update(value)
        except Exception:
            pass