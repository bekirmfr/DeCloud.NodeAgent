"""
api/node_agent.py — Node Agent API client (port 5100).

Covers the full read-only surface the Node Agent exposes for dashboards.
Mutating VM operations (start/stop/restart/delete) are also included
because the operator needs them on this very host.

Authentication:
  The Node Agent dashboard endpoints are unauthenticated when accessed
  on localhost (the default). For remote operation, a node-scoped JWT
  is supported via the `token` parameter and sent in the Authorization
  header (never in URLs). Mutating endpoints may require it.

All methods normalise common response shapes (top-level list vs.
{items|rules|services|...: [...]} wrappers) so callers see a uniform
return type.
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

    # ─── /api/dashboard/* ──────────────────────────────────────────────

    async def dashboard_summary(self) -> dict[str, Any]:
        """Node identity, hostname, OS, uptime, agent version, orch heartbeat."""
        return await self.get("/api/dashboard/summary") or {}

    async def dashboard_network(self) -> dict[str, Any]:
        """Interfaces, WireGuard peers, bridges (with VM mapping), routes."""
        return await self.get("/api/dashboard/network") or {}

    async def dashboard_ports(self) -> dict[str, Any]:
        """Listening TCP / UDP with process names."""
        return await self.get("/api/dashboard/ports") or {}

    async def dashboard_firewall(self) -> dict[str, Any]:
        """UFW status + iptables INPUT / FORWARD / NAT POSTROUTING."""
        return await self.get("/api/dashboard/firewall") or {}

    async def dashboard_services(self) -> list[dict[str, Any]]:
        """systemd unit status (decloud-node-agent, libvirtd, etc.)."""
        data = await self.get("/api/dashboard/services") or []
        return data if isinstance(data, list) else data.get("services", [])

    async def dashboard_logs(self, lines: int = 200) -> list[str]:
        """Recent log lines from the node-agent.

        Returns a list of strings — the upstream endpoint emits raw lines
        (one per log entry, source = file or journald). Callers are
        responsible for parsing level/timestamp from the line text.
        """
        data = await self.get("/api/dashboard/logs", params={"lines": lines}) or {}
        if isinstance(data, list):
            return [str(x) for x in data]
        # Server shape: {source, logFile, logLines: [string], count, collectedAt}
        for key in ("logLines", "logs", "entries"):
            v = data.get(key) if isinstance(data, dict) else None
            if isinstance(v, list):
                return [str(x) for x in v]
        return []

    async def dashboard_obligations(self) -> list[dict[str, Any]]:
        """SystemVm obligations (DHT/Relay/BlockStore) with state data."""
        data = await self.get("/api/dashboard/obligations") or {}
        return data.get("obligations", data) if isinstance(data, dict) else data

    async def dashboard_database(self) -> dict[str, Any]:
        """Local SQLite snapshot — VmRecords + PortMappings + schema meta."""
        return await self.get("/api/dashboard/database") or {}

    async def dashboard_vm_ingress(self) -> dict[str, str]:
        """vmId → publicUrl map (cached server-side for 30 s)."""
        data = await self.get("/api/dashboard/vm-ingress") or {}
        return data if isinstance(data, dict) else {}

    # ─── /api/node/* ───────────────────────────────────────────────────

    async def node_snapshot(self) -> dict[str, Any]:
        """Fast resource snapshot (CPU%, mem bytes, storage bytes, KVM)."""
        return await self.get("/api/node/snapshot") or {}

    async def node_resources(self) -> dict[str, Any]:
        """Full HardwareInventory (CPU benchmark, GPU detail, IOMMU, runtimes)."""
        return await self.get("/api/node/resources") or {}

    async def node_gpu(self) -> Any:
        """GPU inventory."""
        return await self.get("/api/node/gpu")

    # ─── /api/vms/* ────────────────────────────────────────────────────

    async def list_vms(self) -> list[dict[str, Any]]:
        data = await self.get("/api/vms") or []
        return data if isinstance(data, list) else data.get("items", [])

    async def get_vm(self, vm_id: str) -> dict[str, Any]:
        return await self.get(f"/api/vms/{vm_id}") or {}

    async def vm_start(self, vm_id: str) -> Any:
        return await self.post(f"/api/vms/{vm_id}/start")

    async def vm_stop(self, vm_id: str) -> Any:
        return await self.post(f"/api/vms/{vm_id}/stop")

    async def vm_restart(self, vm_id: str) -> Any:
        return await self.post(f"/api/vms/{vm_id}/restart")

    async def vm_delete(self, vm_id: str) -> Any:
        return await self.delete(f"/api/vms/{vm_id}")

    # ─── Convenience: orchestrator-proxied node info ────────────────────

    async def my_obligations(self) -> list[dict[str, Any]]:
        """GET /api/nodes/me/obligations — system VM obligations the orch sees.

        Falls back to the cached dashboard endpoint when the proxied call is
        not exposed (e.g. unauthenticated mode).
        """
        try:
            data = await self.get("/api/nodes/me/obligations")
            if isinstance(data, dict):
                return data.get("obligations", [])
            return data or []
        except Exception:
            return await self.dashboard_obligations()
