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
                yield self._make_card("Network", "net",
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
        for k in ("total", "available", "used", "reserved"):
            with Horizontal(classes="kv"):
                yield Static({
                    "total": "Total", "available": "Available",
                    "used":  "Used",  "reserved":  "Reserved (host)",
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
        # /api/node/resources returns NetworkInfo as a single dict (publicIp,
        # privateIp, wireGuardIp, wireGuardPort, bandwidthBitsPerSecond) —
        # NOT a list of interfaces. Per-interface detail lives on the
        # Network screen (which uses /api/dashboard/network).
        for k in ("public", "private", "wg_ip", "wg_port", "bandwidth"):
            with Horizontal(classes="kv"):
                yield Static({
                    "public":    "Public IP",
                    "private":   "Private IP",
                    "wg_ip":     "WireGuard IP",
                    "wg_port":   "WireGuard port",
                    "bandwidth": "Bandwidth",
                }[k], classes="kv-key")
                yield Static("—", classes="kv-val", id=f"net-{k}")

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
            # Fetch snapshot (always fast) + cached inventory in parallel.
            # GET /api/node/resources?cached=true returns the HardwareInventory
            # from the in-memory cache (populated by the first heartbeat after
            # agent start) without running a CPU benchmark.  Falls back to the
            # flat snapshot if the cache isn't warm yet (404).
            inv_r, snap_r = await asyncio.gather(
                na.get("/api/node/resources", params={"cached": "true"}),
                na.node_snapshot(),
                return_exceptions=True,
            )
        finally:
            await na.close()

        snap = snap_r if isinstance(snap_r, dict) else {}
        inv  = inv_r  if isinstance(inv_r,  dict) else None

        if inv:
            self._apply(inv, snap)
        elif snap:
            self._apply_from_snapshot(snap)

        self.mark_updated()

    def _apply_from_snapshot(self, snap: dict) -> None:
        """Populate Hardware screen from the flat ResourceSnapshot.

        ResourceSnapshot fields (camelCase after serialisation):
          totalPhysicalCores, totalVirtualCpuCores, usedVirtualCpuCores,
          availableVirtualCpuCores, virtualCpuUsagePercent,
          totalComputePoints, usedComputePoints, availableComputePoints,
          computePointUsagePercent,
          totalMemoryBytes, usedMemoryBytes, availableMemoryBytes,
          totalStorageBytes, usedStorageBytes, availableStorageBytes,
          totalGpus, usedGpus, availableGpus
        """
        # ── CPU card ──────────────────────────────────────────────
        phys = snap.get("totalPhysicalCores")
        virt = snap.get("totalVirtualCpuCores")
        used_v = snap.get("usedVirtualCpuCores")
        cpu_pct = snap.get("virtualCpuUsagePercent")

        if phys or virt:
            self._set("cpu-cores",
                       f"{virt or '—'} logical / {phys or '—'} physical")
        if cpu_pct is not None:
            self._set("cpu-model",
                       f"Usage: {cpu_pct:.1f}%"
                       + (f"  ({used_v}/{virt} vCPUs allocated)"
                          if used_v is not None and virt else ""))

        bench = snap.get("benchmarkScore")
        if isinstance(bench, (int, float)) and bench > 0:
            self._set_text("cpu-bench",
                           Text(f"{bench:.0f}", style=f"bold {COLOR['info']}"))

        # Compute points
        total_cp = snap.get("totalComputePoints", 0)
        used_cp = snap.get("usedComputePoints", 0)
        if total_cp:
            self._set("cpu-freq",
                       f"{used_cp}/{total_cp} compute points allocated")

        # ── Memory card ───────────────────────────────────────────
        total_m = snap.get("totalMemoryBytes")
        used_m = snap.get("usedMemoryBytes")
        avail_m = snap.get("availableMemoryBytes")

        if total_m:
            self._set("mem-total", fmt_bytes(total_m))
        if used_m is not None:
            self._set("mem-used", fmt_bytes(used_m))
        if avail_m is not None:
            self._set("mem-available", fmt_bytes(avail_m))
        elif total_m and used_m is not None:
            self._set("mem-available", fmt_bytes(max(0, total_m - used_m)))

        # ── Storage card (aggregate row) ──────────────────────────
        total_s = snap.get("totalStorageBytes")
        used_s = snap.get("usedStorageBytes")
        avail_s = snap.get("availableStorageBytes")

        if total_s:
            try:
                t = self.query_one("#stor-table", DataTable)
                t.clear()
                t.add_row(
                    "(all)",               # mount
                    "—",                   # filesystem
                    fmt_bytes(used_s or 0),
                    fmt_bytes(total_s),
                    fmt_bytes(avail_s if avail_s is not None
                              else max(0, total_s - (used_s or 0))),
                )
            except Exception:
                pass

        # ── GPU card ──────────────────────────────────────────────
        total_g = snap.get("totalGpus", 0)
        used_g = snap.get("usedGpus", 0)
        self._set("gpu-count", str(total_g))
        if total_g:
            self._set("gpu-model",
                       f"{used_g} used / {total_g - used_g} available")
        else:
            try:
                gauge = self.query_one("#hw-gpu-gauge")
                if hasattr(gauge, "set_unavailable"):
                    gauge.set_unavailable("no GPU")
            except Exception:
                pass

        # ── Virtualisation card ───────────────────────────────────
        kvm = snap.get("kvmAvailable")
        if kvm is True:
            self._set_text("virt-kvm",
                           Text("available", style=f"{COLOR['ok']}"))
        elif kvm is False:
            self._set_text("virt-kvm",
                           Text("not available", style=f"{COLOR['crit']}"))

        wsl = snap.get("isWsl2")
        if wsl is not None:
            self._set_text("virt-wsl", _yn(wsl))

        runtimes = snap.get("containerRuntimes")
        if isinstance(runtimes, list):
            self._set("virt-runtimes",
                       ", ".join(str(r) for r in runtimes) or "none")

        gpu_cont = snap.get("supportsGpuContainers")
        if gpu_cont is not None:
            self._set_text("virt-gpu_containers", _yn(gpu_cont))

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
        self._set("cpu-arch",  cpu.get("architecture", "—") or "—")
        bench = cpu.get("benchmarkScore")
        bench_t = (Text(f"{bench:.0f}", style=f"bold {COLOR['info']}")
                   if isinstance(bench, (int, float)) and bench > 0
                   else Text("not run", style=f"{COLOR['dim']}"))
        self._set_text("cpu-bench", bench_t)
        flags = cpu.get("flags") or []
        self._set("cpu-flags", truncate(", ".join(flags), 60))

        # Memory — MemoryInfo has no swapBytes; show used + reserved instead.
        self._set("mem-total",     fmt_bytes(mem.get("totalBytes")))
        self._set("mem-available", fmt_bytes(mem.get("availableBytes")))
        self._set("mem-used",      fmt_bytes(mem.get("usedBytes")))
        self._set("mem-reserved",  fmt_bytes(mem.get("reservedBytes")))

        # Storage — per-volume rows. Note: filesystem field is `fileSystem`
        # (capital S) after camelCase serialisation.  Use AvailableBytes
        # directly rather than computing it (it's authoritative on the host).
        t = self.query_one("#stor-table", DataTable)
        t.clear()
        for s in inv.get("storage", []) or []:
            total = int(s.get("totalBytes", 0) or 0)
            used  = int(s.get("usedBytes",  0) or 0)
            free  = int(s.get("availableBytes", max(0, total - used)) or 0)
            t.add_row(
                s.get("mountPoint") or s.get("devicePath") or "—",
                s.get("fileSystem") or "—",
                fmt_bytes(used),
                fmt_bytes(total),
                fmt_bytes(free),
            )

        # GPU — GpuInfo uses `model` and `vendor`; IOMMU/passthrough flags
        # are PER-GPU, not on HardwareInventory.  Derive aggregate booleans.
        self._set("gpu-count", str(len(gpus)) if gpus else "0")
        if gpus:
            g = gpus[0]
            vendor = g.get("vendor") or ""
            model  = g.get("model")  or "—"
            display = f"{vendor} {model}".strip()
            self._set("gpu-model",  truncate(display, 40))
            self._set("gpu-memory", fmt_bytes(g.get("memoryBytes")))
            self._set("gpu-driver", g.get("driverVersion", "—") or "—")
        else:
            self._set("gpu-model",  "—")
            self._set("gpu-memory", "—")
            self._set("gpu-driver", "—")

        any_iommu = any(g.get("isIommuEnabled") for g in gpus)
        any_pt    = any(g.get("isAvailableForPassthrough") for g in gpus)
        self._set_text("gpu-iommu",       _yn(any_iommu if gpus else None))
        self._set_text("gpu-passthrough", _yn(any_pt    if gpus else None))
        self._set_text("gpu-proxy",       _yn(inv.get("supportsGpuProxy")))

        # Virtualisation
        kvm = snap.get("kvmAvailable")
        if kvm is None:
            kvm = inv.get("kvmAvailable")
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
        # ContainerRuntimes is a list of strings (e.g. ["docker", "podman"]).
        self._set("virt-runtimes", ", ".join(str(r) for r in runtimes) or "none")
        self._set_text("virt-gpu_containers",
                       _yn(inv.get("supportsGpuContainers")))

        # Network — NetworkInfo is a single object, not a list.
        net = inv.get("network") or {}
        self._set("net-public",    net.get("publicIp")  or "—")
        self._set("net-private",   net.get("privateIp") or "—")
        self._set("net-wg_ip",     net.get("wireGuardIp") or "—")
        self._set("net-wg_port",   str(net.get("wireGuardPort") or "—"))
        bw = net.get("bandwidthBitsPerSecond")
        self._set("net-bandwidth",
                  f"{bw/1e9:.1f} Gbps" if bw else "—")

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