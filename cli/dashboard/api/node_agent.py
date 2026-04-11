"""
api/node_agent.py — Node Agent API client (port 5100).

The Node Agent uses a node-scoped JWT (different from the orchestrator user JWT).
When running the dashboard from within a node host, no token is required for
the dashboard endpoints (they are unauthenticated on port 5100 by design).

Endpoints used:
  GET /api/dashboard/summary     — host-level snapshot
  GET /api/dashboard/network     — interfaces + WireGuard
  GET /api/dashboard/ports       — port forwarding rules
  GET /api/dashboard/firewall    — iptables NAT rules
  GET /api/dashboard/services    — systemd service status
  GET /api/dashboard/logs        — recent log lines
  GET /api/dashboard/vm-ingress  — vmId → publicUrl map (cached 30s)
  GET /api/nodes/me/obligations  — system VM obligation status
"""

from __future__ import annotations

from typing import Any

from .client import BaseClient


class NodeAgentClient(BaseClient):

    def __init__(self, base_url: str, token: str | None = None) -> None:
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        super().__init__(base_url, headers=headers)

    # ------------------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------------------

    async def get_summary(self) -> dict[str, Any]:
        """GET /api/dashboard/summary — CPU, memory, storage, VM list."""
        return await self.get("/api/dashboard/summary")

    async def get_network(self) -> dict[str, Any]:
        """GET /api/dashboard/network — interfaces and WireGuard peers."""
        return await self.get("/api/dashboard/network")

    async def get_ports(self) -> list[dict[str, Any]]:
        """GET /api/dashboard/ports — port forwarding rules."""
        data = await self.get("/api/dashboard/ports")
        return data if isinstance(data, list) else data.get("rules", [])

    async def get_firewall(self) -> list[dict[str, Any]]:
        """GET /api/dashboard/firewall — iptables NAT rules."""
        data = await self.get("/api/dashboard/firewall")
        return data if isinstance(data, list) else data.get("rules", [])

    async def get_services(self) -> list[dict[str, Any]]:
        """GET /api/dashboard/services — systemd service status."""
        data = await self.get("/api/dashboard/services")
        return data if isinstance(data, list) else data.get("services", [])

    async def get_logs(self, lines: int = 200) -> list[dict[str, Any]]:
        """GET /api/dashboard/logs — recent structured log entries."""
        data = await self.get("/api/dashboard/logs", params={"lines": lines})
        return data if isinstance(data, list) else data.get("logs", data.get("entries", []))

    async def get_vm_ingress(self) -> dict[str, str]:
        """GET /api/dashboard/vm-ingress — vmId → publicUrl."""
        data = await self.get("/api/dashboard/vm-ingress")
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Orchestrator-proxied node info
    # ------------------------------------------------------------------

    async def get_obligations(self) -> list[dict[str, Any]]:
        """GET /api/nodes/me/obligations — system VM obligation status."""
        data = await self.get("/api/nodes/me/obligations")
        if isinstance(data, dict):
            return data.get("obligations", [])
        return data
