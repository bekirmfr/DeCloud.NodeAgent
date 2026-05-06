"""
api/orchestrator.py — Orchestrator API client (optional).

Kept deliberately minimal because the redesigned dashboard is node-first.
We use the orchestrator only for what the local Node Agent cannot tell us:

  • Earnings & balance (user wallet scope)
  • A glance at the fleet (other nodes the same wallet owns)
  • Central ingress base domain (also surfaced via node-agent summary)

If no token is configured, none of these screens render — and the
dashboard remains fully usable on local data only.
"""

from __future__ import annotations

from typing import Any

from .client import BaseClient, ApiError  # noqa: F401 — re-exported


class OrchestratorClient(BaseClient):

    def __init__(self, base_url: str, token: str) -> None:
        super().__init__(base_url, headers={"Authorization": f"Bearer {token}"})

    # ─── Earnings ───────────────────────────────────────────────────────

    async def get_balance(self) -> dict[str, Any]:
        data = await self.get("/api/user/balance") or {}
        return _unwrap(data)

    async def list_transactions(self, limit: int = 25) -> list[dict[str, Any]]:
        data = await self.get("/api/user/transactions",
                              params={"limit": limit}) or []
        result = _unwrap(data)
        return result if isinstance(result, list) else result.get("items", [])

    # ─── Fleet glance ───────────────────────────────────────────────────

    async def list_nodes(self) -> list[dict[str, Any]]:
        data = await self.get("/api/nodes") or []
        result = _unwrap(data)
        if isinstance(result, list):
            return result
        return result.get("items", result.get("nodes", []))

    async def get_node(self, node_id: str) -> dict[str, Any]:
        data = await self.get(f"/api/nodes/{node_id}") or {}
        return _unwrap(data)

    # ─── Central ingress ────────────────────────────────────────────────

    async def get_ingress_status(self) -> dict[str, Any]:
        data = await self.get("/api/central-ingress/status") or {}
        return _unwrap(data)


def _unwrap(payload: Any) -> Any:
    """Unwrap {success, data: …} envelopes used by some orchestrator routes."""
    if isinstance(payload, dict) and "data" in payload and "success" in payload:
        return payload["data"]
    return payload
