"""
api/orchestrator.py — Orchestrator API client.

Endpoints used:
  GET  /api/system/stats
  GET  /api/nodes
  GET  /api/nodes/{nodeId}
  GET  /api/vms              (query: page, pageSize, status, search)
  GET  /api/vms/{vmId}
  POST /api/vms/{vmId}/action
  GET  /api/central-ingress/status
  GET  /api/user/balance
  GET  /api/user/transactions
  GET  /api/admin/nodes      (admin — full node list with internals)
"""

from __future__ import annotations

from typing import Any

from .client import BaseClient, ApiError  # noqa: F401 (re-exported for callers)


class OrchestratorClient(BaseClient):

    def __init__(self, base_url: str, token: str) -> None:
        super().__init__(base_url, headers={"Authorization": f"Bearer {token}"})

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """GET /api/system/stats — cluster-wide counters."""
        data = await self.get("/api/system/stats")
        return _unwrap(data)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def list_nodes(self) -> list[dict[str, Any]]:
        """GET /api/nodes — all registered nodes."""
        data = await self.get("/api/nodes")
        result = _unwrap(data)
        if isinstance(result, list):
            return result
        # Some versions wrap in {items: [...]}
        return result.get("items", result.get("nodes", []))

    async def get_node(self, node_id: str) -> dict[str, Any]:
        """GET /api/nodes/{nodeId}."""
        data = await self.get(f"/api/nodes/{node_id}")
        return _unwrap(data)

    # ------------------------------------------------------------------
    # Virtual Machines
    # ------------------------------------------------------------------

    async def list_vms(
        self,
        status: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """GET /api/vms — paginated VM list."""
        params: dict[str, Any] = {"page": page, "pageSize": page_size}
        if status:
            params["status"] = status
        if search:
            params["search"] = search
        data = await self.get("/api/vms", params=params)
        result = _unwrap(data)
        # PagedResult shape: {items: [...], totalCount: N}
        if isinstance(result, dict):
            return result.get("items", [])
        return result

    async def get_vm(self, vm_id: str) -> dict[str, Any]:
        """GET /api/vms/{vmId}."""
        data = await self.get(f"/api/vms/{vm_id}")
        return _unwrap(data)

    async def vm_action(self, vm_id: str, action: str) -> bool:
        """POST /api/vms/{vmId}/action — start | stop | restart | forceStop."""
        await self.post(f"/api/vms/{vm_id}/action", json={"action": action})
        return True

    async def delete_vm(self, vm_id: str) -> bool:
        """DELETE /api/vms/{vmId}."""
        await self.delete(f"/api/vms/{vm_id}")
        return True

    # ------------------------------------------------------------------
    # Ingress
    # ------------------------------------------------------------------

    async def get_ingress_status(self) -> dict[str, Any]:
        """GET /api/central-ingress/status."""
        data = await self.get("/api/central-ingress/status")
        return _unwrap(data)

    async def list_ingress_routes(self) -> list[dict[str, Any]]:
        """GET /api/ingress — all ingress rules for the authenticated user."""
        data = await self.get("/api/ingress")
        result = _unwrap(data)
        if isinstance(result, dict):
            return result.get("rules", result.get("items", []))
        return result

    # ------------------------------------------------------------------
    # Billing
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict[str, Any]:
        """GET /api/user/balance — USDC balance on Polygon."""
        data = await self.get("/api/user/balance")
        return _unwrap(data)

    async def list_transactions(self, limit: int = 50) -> list[dict[str, Any]]:
        """GET /api/user/transactions."""
        data = await self.get("/api/user/transactions", params={"limit": limit})
        result = _unwrap(data)
        if isinstance(result, list):
            return result
        return result.get("items", result.get("transactions", []))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(data: Any) -> Any:
    """Strip the ApiResponse envelope {success, data, error} if present."""
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data
