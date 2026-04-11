"""
screens/billing.py — Billing and payment screen.

Shows USDC balance on Polygon and the recent transaction log.
Source: orchestrator GET /api/user/balance + /api/user/transactions.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Label, Static

from config import cfg
from api.orchestrator import OrchestratorClient, ApiError


class _BalanceCard(Static):
    DEFAULT_CSS = """
    _BalanceCard {
        border: solid $panel;
        padding: 1 2;
        width: 1fr;
        height: 7;
    }
    """

    def __init__(self, label: str, amount_id: str) -> None:
        super().__init__()
        self._label = label
        self._amount_id = amount_id

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="card-label")
        yield Label("—", id=self._amount_id, classes="card-value")

    def set_value(self, value: str) -> None:
        self.query_one(f"#{self._amount_id}", Label).update(value)


class BillingScreen(Screen):
    """Billing overview — balance + transaction history."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label("Billing", classes="section-title")
        with Horizontal(id="balance-row"):
            yield _BalanceCard("Available Balance", "bal-available")
            yield _BalanceCard("Earned (30d)", "bal-earned")
            yield _BalanceCard("Pending Settlement", "bal-pending")
            yield _BalanceCard("Active VMs Billed", "bal-vms")

        yield Label("Transaction History", classes="section-title")
        yield DataTable(id="tx-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#tx-table", DataTable)
        table.add_columns("VM / Description", "Duration", "Amount (USDC)", "Trigger", "Status")
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
            balance, txs = await asyncio.gather(
                client.get_balance(),
                client.list_transactions(),
                return_exceptions=True,
            )
        finally:
            await client.close()

        if isinstance(balance, dict):
            self._apply_balance(balance)

        if isinstance(txs, list):
            self._render_txs(txs)

    def _apply_balance(self, b: dict) -> None:
        self.query_one("#bal-available", Label).update(
            f"[bold cyan]${b.get('availableBalance', 0):.2f}[/]"
        )
        self.query_one("#bal-earned", Label).update(
            f"[green]${b.get('earned30d', 0):.2f}[/]"
        )
        self.query_one("#bal-pending", Label).update(
            f"${b.get('pendingBalance', b.get('pendingSettlement', 0)):.2f}"
        )
        self.query_one("#bal-vms", Label).update(
            str(b.get("billedVmCount", b.get("activeVms", "—")))
        )

    def _render_txs(self, txs: list) -> None:
        table = self.query_one("#tx-table", DataTable)
        table.clear()
        for tx in txs:
            amount = tx.get("amount", tx.get("amountUsdc", 0))
            status = tx.get("status", "—")
            color = "green" if status.lower() == "settled" else "yellow"
            table.add_row(
                f"{tx.get('vmId', '—')[:10]}  {tx.get('description', '')}",
                tx.get("duration", tx.get("durationHuman", "—")),
                f"[green]+${float(amount):.2f}[/]",
                tx.get("trigger", "—"),
                f"[{color}]{status}[/]",
            )
