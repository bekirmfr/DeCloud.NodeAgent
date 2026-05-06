"""
screens/hardware.py — Detailed hardware inventory.

Surfaces the rich data from /api/node/resources that the current dashboard
hides: per-CPU benchmark, IOMMU state, NVIDIA driver, container runtimes,
GPU passthrough/proxy capability, per-volume storage breakdown.

Operator question this screen answers: 'What is this machine actually?'
"""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Static

from config import cfg
from api.node_agent import NodeAgentClient
from screens._base import BaseScreen
from theme import COLOR
from util.format import fmt_bytes, fmt_pct, truncate
from widgets.card import Card
from widgets.statpill import status_text


def _yn(b: object) -> Text:
    if b is None:
        return Text("—", style=f"{COLOR['dim']}")
    if bool(b):
        return Text("yes", style=f"{COLOR['ok']}")
    return Text("no",  style=f"{COLOR['crit']}")


class HardwareScreen(BaseScreen):
    ACTIVE_LABEL = "Hardware"

    DEFAULT_CSS = """
    HardwareScreen .row { height: auto; layout: horizontal; }
    HardwareScreen #col-l { width: 1fr; height: auto; }
    HardwareScreen #col-r { width: 1fr; height: auto; margin-left: 1; }
    HardwareScreen DataTable { height: auto; max-height: 14; }
    HardwareScreen .kv { height: 1; layout: horizontal; }
    HardwareScreen .kv-key   { width: 18; color: $text-muted; }
    HardwareScreen .kv-val   { width: 1fr; }
    """

    def compose_content(self) -> ComposeResult:
        with Horizontal(classes="row"):
            with Vertical(id="col-l"):
                yield self._make_card("CPU", "cpu", self._compose_cpu)
                yield self._make_card("Memory", "mem", self._compose_mem)
                yield self._make_card("Storage Volumes", "stor", self._compose_stor)
            with Vertical(id="col-r"):
                yield self._make_card("GPU", "gpu", self._compose_gpu)
                yield self._make_card("Virtualization & Runtimes",
                                      "virt", self._compose_virt)
                yield self._make_card("Network Interfaces", "nics",
                                      self._compose_nics)

    def _make_card(self, title, ident, body_fn) -> Card:
        c = Card(title, id=f"hw-{ident}-card")
        c.compose_body = body_fn  # type: ignore[assignment]
        return c

    # ─── Card bodies ───────────────────────────────────────────────────

    def _compose_cpu(self) -> ComposeResult:
        for k in ("model", "cores", "freq", "arch", "bench", "flags"):
            with Horizontal(classes="kv"):
                yield Static({
                    "model": "Model", "cores": "Cores",
                    "freq":  "Freq",  "arch":  "Architecture",
                    "bench": "Benchmark", "flags": "Features",
                }[k], classes="kv-key")
                yield Static("—", classes="kv-val", id=f"cpu-{k}")

    def _compose_mem(self) -> ComposeResult:
        for k in ("total", "available", "swap"):
            with Horizontal(classes="kv"):
                yield Static({
                    "total": "Total", "available": "Available", "swap": "Swap",
                }[k], classes="kv-key")
                yield Static("—", classes="kv-val", id=f"mem-{k}")

    def _compose_stor(self) -> ComposeResult:
        t = DataTable(id="stor-table", zebra_stripes=True)
        t.add_columns("Mount", "Type", "Used", "Total", "Free")
        yield t

    def _compose_gpu(self) -> ComposeResult:
        for k in ("count", "model", "memory", "driver", "iommu",
                  "passthrough", "proxy"):
            with Horizontal(classes="kv"):
                yield Static({
                    "count":       "GPUs",
                    "model":       "Model",
                    "memory":      "VRAM",
                    "driver":      "Driver",
                    "iommu":       "IOMMU",
                    "passthrough": "Passthrough",
                    "proxy":       "Proxy capable",
                }[k], classes="kv-key")
                yield Static("—", classes="kv-val", id=f"gpu-{k}")

    def _compose_virt(self) -> ComposeResult:
        for k in ("kvm", "wsl", "runtimes", "gpu_containers"):
            with Horizontal(classes="kv"):
                yield Static({
                    "kvm":            "KVM",
                    "wsl":            "WSL2",
                    "runtimes":       "Container runtimes",
                    "gpu_containers": "GPU containers",
                }[k], classes="kv-key")
                yield Static("—", classes="kv-val", id=f"virt-{k}")

    def _compose_nics(self) -> ComposeResult:
        t = DataTable(id="nics-table", zebra_stripes=True)
        t.add_columns("Name", "MAC", "Speed", "State")
        yield t

    # ─── Lifecycle ─────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.set_interval(max(cfg.refresh_interval, 30), self._load)
        self.run_worker(self._load(), exclusive=True)

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        if not cfg.has_node_agent:
            self.notify("DECLOUD_NODE_URL not configured", severity="warning")
            return
        na = NodeAgentClient(cfg.node_url)
        try:
            inv, snap = await asyncio.gather(
                na.node_resources(), na.node_snapshot(),
                return_exceptions=True,
            )
        finally:
            await na.close()
        if isinstance(inv, dict):
            self._apply(inv, snap if isinstance(snap, dict) else {})
        self.mark_updated()

    # ─── Rendering ─────────────────────────────────────────────────────

    def _apply(self, inv: dict, snap: dict) -> None:
        cpu  = inv.get("cpu") or {}
        mem  = inv.get("memory") or {}
        gpus = inv.get("gpus") or []
        runtimes = inv.get("containerRuntimes") or []

        # CPU
        self._set("cpu-model", truncate(cpu.get("model", "—"), 50))
        cores = f'{cpu.get("logicalCores","—")} logical / {cpu.get("physicalCores","—")} physical'
        self._set("cpu-cores", cores)
        self._set("cpu-freq",  f'{cpu.get("frequencyMhz", "—")} MHz')
        self._set("cpu-arch",  cpu.get("architecture", "—"))
        bench = cpu.get("benchmarkScore")
        bench_t = (Text(f"{bench:.0f}", style=f"bold {COLOR['info']}")
                   if isinstance(bench, (int, float)) and bench > 0
                   else Text("not run", style=f"{COLOR['dim']}"))
        self._set_text("cpu-bench", bench_t)
        flags = cpu.get("flags") or []
        self._set("cpu-flags", truncate(", ".join(flags), 60))

        # Memory
        self._set("mem-total",     fmt_bytes(mem.get("totalBytes")))
        self._set("mem-available", fmt_bytes(mem.get("availableBytes")))
        self._set("mem-swap",      fmt_bytes(mem.get("swapBytes")))

        # Storage
        t = self.query_one("#stor-table", DataTable)
        t.clear()
        for s in inv.get("storage", []) or []:
            total = int(s.get("totalBytes", 0) or 0)
            used  = int(s.get("usedBytes",  0) or 0)
            free  = max(0, total - used)
            t.add_row(
                s.get("mountPoint") or s.get("path") or "—",
                s.get("filesystem", "—"),
                fmt_bytes(used),
                fmt_bytes(total),
                fmt_bytes(free),
            )

        # GPU
        self._set("gpu-count", str(len(gpus)) if gpus else "0")
        if gpus:
            g = gpus[0]
            self._set("gpu-model",  truncate(g.get("name", "—"), 40))
            self._set("gpu-memory", fmt_bytes(g.get("memoryBytes")))
            self._set("gpu-driver", g.get("driverVersion", "—") or "—")
        self._set_text("gpu-iommu",       _yn(inv.get("hasIommuCapableGpu")))
        self._set_text("gpu-passthrough", _yn(inv.get("hasPassthroughCapableGpu")))
        self._set_text("gpu-proxy",       _yn(inv.get("supportsGpuProxy")))

        # Virtualisation
        kvm = snap.get("kvmAvailable")
        if kvm is False:
            self._set_text("virt-kvm", Text("not available",
                                            style=f"{COLOR['crit']}"))
        elif kvm is True:
            self._set_text("virt-kvm", Text("available",
                                            style=f"{COLOR['ok']}"))
        else:
            self._set_text("virt-kvm", Text("unknown",
                                            style=f"{COLOR['dim']}"))
        self._set_text("virt-wsl", _yn(inv.get("isWsl2")))
        self._set("virt-runtimes",
                  ", ".join(r.get("name","?") + " v" + r.get("version","?")
                            for r in runtimes) or "none")
        self._set_text("virt-gpu_containers",
                       _yn(inv.get("supportsGpuContainers")))

        # NICs
        nt = self.query_one("#nics-table", DataTable)
        nt.clear()
        for n in inv.get("network", []) or []:
            up = n.get("isUp")
            state = (Text("UP",   style=f"{COLOR['ok']}") if up
                     else Text("DOWN", style=f"{COLOR['crit']}"))
            nt.add_row(
                n.get("name", "—"),
                n.get("macAddress", "—") or "—",
                f'{n.get("speedMbps","—")} Mbps' if n.get("speedMbps") else "—",
                state,
            )

    def _set(self, ident: str, value: str) -> None:
        try:
            self.query_one(f"#{ident}", Static).update(value)
        except Exception:
            pass

    def _set_text(self, ident: str, value: Text) -> None:
        try:
            self.query_one(f"#{ident}", Static).update(value)
        except Exception:
            pass
