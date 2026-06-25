# DeCloud GPU Proxy — Architecture Review

**Date Range:** 2026-02-27 through 2026-06-25
**Authors:** BMA + Claude AI assistant
**Status:** Production-ready with known limitations — Ollama 3B GPU confirmed; 7B+ blocked by Bug 23a; Ollama pinned to 0.7.0 (Bug 23). SEC-1 root fix (fork-per-VM workers) **shipped and hardware-validated** (2026-06-25): per-VM CUDA-context isolation, worker/supervisor lifecycle, and per-tenant VRAM quota all confirmed on hardware. Cross-tenant *denial* (different-owner probe) and the vsock quota path remain untested — see §5.

---

## Executive Summary

Over ten debugging sessions spanning three weeks, we built a complete CUDA virtualization layer that allows VMs without physical GPUs to run GPU-accelerated inference through a TCP/vsock RPC proxy. The system intercepts CUDA Runtime API, Driver API, and NVML calls in the VM, forwards them to a daemon on the host with the real GPU, and returns results transparently.

**Final performance (llama3.2:1b on RTX 4060 Laptop GPU):**
- Prompt eval: **436 tok/s** (warm), **188 tok/s** (cold)
- Generation: **13-21 tok/s**
- 100% GPU offload, zero manual configuration

**PyTorch performance (GPT-2, RTX 4060 Laptop GPU, confirmed 2026-03-13):**
- Full fine-tuning (AdamW, 125M params): **1,252 tok/s**, 409ms/step
- LoRA fine-tuning (PEFT r=8, 811K params): **1,038 tok/s**, 493ms/step, **1,360MB peak VRAM**
- Inference, backward pass, optimizer step, and LoRA all confirmed working end-to-end

The proxy is now **fully generic** — no hardcoded application (Ollama/ggml) or vendor (NVIDIA RTX 4060) dependencies in the C code. Application-specific configuration is driven by template environment variables written to `/etc/decloud/gpu-proxy.env`.

> 🟡 **Security (SEC-1, updated 2026-06-25):** The original finding — all VM connections shared the daemon's single primary CUDA context, allowing cross-tenant VRAM access — is **addressed by the fork-per-VM worker model, now shipped and hardware-validated**. Each VM connection is served by a separate forked worker process with its own CUDA context (GPU-MMU-enforced separation). Worker/supervisor lifecycle (context reaping on disconnect, killability, clean shutdown) and per-tenant VRAM quota are confirmed on hardware. **Still open:** the cross-tenant *denial* property has not been directly tested with two different-owner tenants (mechanism is proven; the security boundary itself is not yet probed), and the vsock quota path is code-verified only (not testable on the WSL2 dev node). See §5.

---

## 1. Architecture

```
┌─────────────────────────────────────────┐
│  VM (no physical GPU)                   │
│                                         │
│  Application (Ollama, PyTorch, etc.)    │
│      ↓ cuda*() calls                    │
│  libcudart.so.12  (Runtime API shim)    │
│      ↓ cu*() calls                      │
│  libcuda.so.1     (Driver API shim)     │
│      ↓ RPC over TCP/vsock               │
├─────────────────────────────────────────┤
│  Host (real GPU)                        │
│                                         │
│  gpu-proxy-daemon                       │
│      ↓ real CUDA calls                  │
│  Real NVIDIA Driver + GPU               │
└─────────────────────────────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| CUDA Runtime Shim | `shim/cuda_shim.c` | Intercepts `cuda*()` API calls, forwards via RPC |
| CUDA Driver Shim | `shim/cuda_driver_shim.c` | Intercepts `cu*()` API calls, device enumeration, `cuGetProcAddress` dispatch |
| Transport Layer | `shim/transport.{c,h}` | TCP/vsock connection, TCP_NODELAY + TCP_QUICKACK |
| NVML Shim | `shim/nvml_shim.c` | Fakes GPU management/monitoring info |
| GPU Proxy Daemon | `daemon/gpu_proxy_daemon.c` | Host-side daemon executing real CUDA operations |
| Protocol | `proto/gpu_proxy_proto.h` | Wire protocol structs and 35+ command IDs |
| cuBLAS Stub | `stubs/cublas_stub.c` | Separate library with `@@libcublas.so.12` version tags |
| cuBLAS Lt Stub | `stubs/cublasLt_stub.c` | DT_NEEDED placeholder for cublasLt |

### Wire Protocol

- Magic: `0x44435544` ("DCUD"), Version: 2
- Transport: TCP (port 9999) or vsock (CID=2, port=9999)
- Auth: Token-based (per-VM tokens, SIGHUP reload)
- Header: 16 bytes (magic, version, cmd, flags, payload_len, status)
- Max payload: 2GB (streaming for large fatbins)
- Critical: `TCP_NODELAY` + `TCP_QUICKACK` (eliminates 40ms delayed ACK)

### Command ID Map

| Range | Group | Commands |
|-------|-------|----------|
| 0x01-0x05 | Device mgmt | GetDeviceCount, GetProperties, SetDevice, DriverVersion, UUID |
| 0x10-0x13 | Memory | Malloc, Free, Memcpy, Memset |
| 0x20-0x24 | Execution | LaunchKernel, DeviceSync, CtxCreate, MemGetInfo, CtxDestroy |
| 0x30-0x32 | Streams | Create, Destroy, Synchronize |
| 0x40-0x44 | Events | Create, Destroy, Record, Synchronize, ElapsedTime |
| 0x50-0x55 | Modules | RegisterModule/Function/Var, FuncGetAttributes, OccupancyMaxBlocks |
| 0x56-0x57 | cuBLAS | GemmBatched, GemmStrided |
| 0x60-0x61 | Resource mgmt | SetMemoryQuota, GetUsageStats |
| 0x70-0x77 | Virtual memory | VmemCreate/Release/Reserve/Free/Map/Unmap/SetAccess/GetGranularity |
| 0xF0-0xF1 | Lifecycle | Hello, Goodbye |

---

## 2. Generic Proxy Design

### Configuration-Driven Architecture

All application-specific behavior is controlled by `/etc/decloud/gpu-proxy.env`, written during cloud-init from template `DefaultEnvironmentVariables`. The shim constructor reads this file and propagates non-transport vars via `setenv()`.

**Proxy-level flags:**

| Flag | Default | Effect |
|------|---------|--------|
| `DECLOUD_GPU_VMEM_PROXY` | 0 | Virtual memory APIs proxy to daemon (1) or return `NOT_SUPPORTED` (0) |
| `DECLOUD_GPU_DEBUG` | unset | Enable debug logging from shim constructor |

> **Note:** `DECLOUD_GPU_GRAPH_NOOP` was removed in Session 15 (Mar 17). CUDA graphs now always return `cudaErrorNotSupported` / `CUDA_ERROR_NOT_SUPPORTED`, forcing applications to use direct kernel execution. See [Debugging Journal, Session 15](#) for rationale.

**Config flow:** Orchestrator template → `DefaultEnvironmentVariables` → `EnsureGpuProxyShim` extracts `GGML_*`/`CUDA_*`/`DECLOUD_GPU_*` → writes to `/etc/decloud/gpu-proxy.env` → shim constructor reads and applies.

### Key Design Decisions

**TCP_QUICKACK (Session 9-10):** The biggest performance fix. TCP delayed ACK imposed 40ms wait per small RPC. Must be re-armed before every `read()` (Linux resets per-operation). Applied to shims, transport, and daemon.

**Deferred + Eager Module Upload:** Fat binaries stored locally at `__cudaRegisterFatBinary`, uploaded to daemon only when needed. `cudaFuncGetAttributes` triggers eager upload because ggml queries attributes before first launch.

**Streaming Module Upload:** 1.56GB fatbin from libggml-cuda.so written directly from mmap'd memory — zero copy, zero malloc.

**cuBLAS GEMM Proxy:** cuBLAS init requires `cuGetExportTable` (private NVIDIA internals, cannot be proxied). Solution: stub init, use ggml's MMQ path, proxy only `cublasGemmBatchedEx`/`cublasGemmStridedBatchedEx` for GQA attention.

**CUDA Graph Pass-Through:** CUDA graphs cannot be faithfully proxied — capture records host-side API calls but execution is remote. Both the runtime and driver shims return "not supported" for graph capture (`cudaStreamBeginCapture`, `cuStreamBeginCapture`), forcing applications to fall back to direct kernel execution. This is both simpler and more correct than the earlier no-op or capture/replay approaches. Note: libcudart resolves `cudaStreamBeginCapture` through the driver API (`cuGetProcAddress`), so both shims must return consistent results.

> ⚠️ **Ollama Version Compatibility (May 2026):** Ollama 0.24.0 introduced `ggml_backend_cuda_graph_reserve()` in context init, which calls `cudaStreamBeginCapture` wrapped in `CUDA_CHECK()` with no fallback — any non-zero return causes `ggml_abort()`. The `cudaErrorNotSupported` pass-through approach (Session 15) breaks with this version. **Ollama must be pinned to 0.7.0** in templates until the shim is updated to return `cudaSuccess` on the first `cudaStreamBeginCapture` call per-process (satisfying the reserve) while still rejecting inference-time captures. See Debugging Journal Session 16.

**Real GPU Attributes:** `cu_func_get_attribute` queries daemon via RPC, caches per-function, falls back to safe defaults. Eliminates hardcoded sm_89 vendor dependency.

> ⚠️ **7B+ Model Compatibility (May 2026 — Bug 23a):** Models with n_embd≥4096, n_ff≥14336, or 32+ layers produce Ġ token (U+0120) corruption in GPU inference starting at ~token 200-250, cascading to complete output failure. The 3B model (n_embd=3072, 28 layers) is clean. GEMM proxy is ruled out (MMQ path confirmed). Root cause is a geometry-dependent error in kernel launch parameter handling or KV cache stride calculations for larger tensor dimensions. See Debugging Journal Session 17.

---

## 3. Deployment Pipeline

### Build
```bash
sudo bash install.sh  # Builds shims (Docker compat), daemon, deploys to 9p share
```

install.sh handles: Docker compat build (glibc 2.31), native daemon build, stale daemon kill + restart with captured args, libcudart.so.12 sync, 9p share freshness verification.

### VM Provisioning (cloud-init, fully automatic)
1. Mount 9p share → copy shims → replace Ollama's bundled CUDA libs
2. Install NVML/libcuda shims to system paths
3. Write `/etc/decloud/gpu-proxy.env` (transport + app vars)
4. Create systemd override for application
5. Restart application

### Daemon Lifecycle
- `GpuProxyService.cs` starts daemon when first GPU VM boots
- Per-VM token auth, SIGHUP reload
- Per-connection resource tracking (memory quota, kernel stats)

---

## 4. Current Status

| Capability | Status | Notes |
|------------|--------|-------|
| GPU detection + model loading | ✅ | Automatic via cloud-init |
| Kernel launch via RPC | ✅ | 13-21 tok/s generation (3B models confirmed) |
| Ollama 7B+ model inference | ❌ Blocked | Bug 23a — Ġ token corruption in extended generation; tensor geometry exceeds proxy's correct range |
| Prompt evaluation | ✅ | 188-436 tok/s |
| 1.56GB streaming module upload | ✅ | Zero-copy from mmap |
| cuBLAS GEMM proxy | ✅ | GQA attention via RPC |
| cublasLtMatmul proxy | ✅ | PyTorch linear/attention GEMMs |
| Real kernel attributes + occupancy | ✅ | Cached, with fallback |
| CUDA graph pass-through | ✅ Working (pinned) | Returns not-supported, forces direct kernel execution. **Ollama pinned to 0.7.0** — 0.24.0+ incompatible (see Session 16) |
| TCP_NODELAY + TCP_QUICKACK | ✅ | Sub-ms RPC latency |
| Template-driven env vars | ✅ | Generic proxy |
| Zero-touch VM deployment | ✅ | Cloud-init automated |
| Multi-tenant GPU sharing | 🟡 Mechanism shipped | SEC-1 root fix (fork-per-VM workers) shipped + hardware-validated; per-tenant quota enforced (TCP). Cross-tenant *denial* probe and vsock quota path still untested — see §5 |
| PyTorch inference (forward pass) | ✅ | Confirmed 2026-03-13 |
| PyTorch training (backward + optimizer) | ✅ | Confirmed 2026-03-13 |
| LoRA fine-tuning via PEFT | ✅ | Confirmed 2026-03-13; 1,360MB VRAM |
| JupyterLab kernel GPU access | ✅ | Via systemd EnvironmentFile |
| CUDA 12.1 ABI offsets (SM8.9) | ✅ | Verified offset map committed |
| Virtual memory (cuMem*) | ✅ Enabled | `DECLOUD_GPU_VMEM_PROXY=1` for PyTorch |
| cublasLt debug gate | ✅ Fixed | `DECLOUD_GPU_DEBUG` gate (rebuild pending deploy) |
| Stable Diffusion WebUI Forge | ✅ Confirmed | Image generation 1.61 it/s (Mar 15) |
| cuDNN/cuFFT/cuSPARSE proxy | Deferred | For scientific computing |

---

## 5. Security Model & Multi-Tenancy (SEC-1)

**Status:** 🟡 Root fix shipped and hardware-validated (2026-06-25). Isolation mechanism, lifecycle, and per-tenant quota proven on hardware; cross-tenant denial test and vsock quota path still open.

### The Finding (original, 2026-06-11)

The daemon adopts the device's **primary CUDA context** (`cuDevicePrimaryCtxRetain`, Session 7) — the correct fix for runtime/driver API interop, with an unexamined consequence: the primary context is per-device per-process, and the daemon was one process serving all VM connections. Every tenant therefore shared **one GPU address space**. Within a CUDA context there is no isolation: any connection could read (`Memcpy` D2H), corrupt (`Memcpy` H2D / `Memset`), or free another tenant's allocations by naming their device addresses — and could launch its own kernels whose pointer arguments reference foreign buffers.

Per-VM tokens authenticate connections, not addresses. Memory quotas limit allocation amounts, not addressable ranges. Per-connection function tables cover modules, not memory. The attacker is simply another paying tenant.

### Why Pointer Validation Cannot Close It

A per-connection allocation map can validate Memcpy/Memset/Free — but `LaunchKernel` parameters are an **opaque byte blob** with no signature information. The daemon cannot distinguish pointers from integers, so kernel-mediated access to foreign memory is unvalidatable in principle. Only context separation closes the hole.

### Fix Plan & Status

| Layer | Measure | Status |
|-------|---------|--------|
| Interim | Scheduler invariant: at most one wallet's GPU VMs per physical GPU | Superseded by the root fix for the isolation property; still useful as defense-in-depth |
| Root fix | Fork-per-VM daemon workers — per-process primary contexts, GPU MMU-enforced isolation | ✅ **Shipped + hardware-validated (2026-06-25).** Supervisor stays CUDA-free and forks one worker per connection; each worker holds its own context. Confirmed on RTX 4060/WSL2: two same-owner VMs ran concurrently, each in a separate forked worker with its own context |
| Lifecycle | Worker context reaping on disconnect; killability; clean supervisor shutdown | ✅ **Shipped + hardware-validated.** Half-open detection (TCP keepalive / vsock recv-timeout), worker SIGTERM-default (killable, no orphan-on-restart), deterministic supervisor exit (close-listen-FD + `_exit`). Confirmed: leaked workers were the cause of a VRAM leak (5719 MiB → 13 MiB on reap); all three teardown paths verified |
| Hardening — zero-on-free | `cudaMemset(0)` on Free + teardown (recycled VRAM not scrubbed) | ✅ Present in `handle_free` (scrub before `cudaFree`) |
| Per-tenant quota | VRAM quota enforced across ALL of a VM's connections, not per-connection | ✅ **TCP path shipped + hardware-validated;** 🟡 **vsock path code-verified only.** Shared-memory slot ledger summed by `vm_id`, self-healing via PID-liveness. Confirmed: a VM running two models was held to its quota (second model DENIED) instead of getting 2× quota |
| Hardening — compute watchdog | Per-launch timeout → kill worker → context released | Partial — kernel timeout (`-t`) exists; per-launch worker-kill watchdog not yet wired |
| Hardening — transport | TCP restricted to WireGuard overlay / vsock only | Planned — token still travels plaintext on the TCP path; do not bind a routable interface |
| Out of scope | GPU microarchitectural side channels; no MIG on consumer cards → confidential workloads use the dedicated-GPU tier | Documented threat-model boundary |

### What is proven vs. still open (be precise)

**Proven on hardware (2026-06-25):**
- Fork-per-VM workers create separate CUDA contexts (mechanism).
- Worker/supervisor lifecycle is correct: contexts are reaped on disconnect, workers are killable by SIGTERM, the supervisor exits cleanly on agent restart with no orphan and VRAM returning to 0.
- Per-tenant VRAM quota is enforced across a VM's connections on the TCP path.

**Still open (do not claim closed):**
- **Cross-tenant denial.** The isolation *mechanism* is proven, but the security *property* — that tenant A literally cannot read/write tenant B's VRAM — has not been directly tested with two **different-owner** tenants and a deliberate cross-context probe (`sec1_probe`). Same-owner concurrency was tested; the trust boundary itself was not. Until that probe runs (ideally with the `GGML_CUDA_NO_PEER_COPY` flag's role resolved so a DENIED result is attributable to the fork boundary), the denial guarantee is *expected* but *unverified*.
- **vsock quota enforcement.** The per-tenant quota and its vsock (CID-based) identity path are code-verified but not hardware-tested — vsock does not function under WSL2. First real exercise is on a bare-metal node. Note that prior to this work, quota was enforced only on the TCP path at all; bare-metal/vsock nodes had no VRAM quota.
- **Adopt-on-pgrep.** `GpuProxyService.HealthCheckAsync` still adopts any running `gpu-proxy-daemon` found via `pgrep` rather than verifying it is the current binary — the same permissive pattern that caused the original stale-daemon incident, one layer up. Closing it needs a version/capability probe.

**Cost of the root fix (as built):** ~400 MB VRAM per CUDA context on consumer GPUs (deducted per-connection from the tenant's quota via `GPU_CONTEXT_OVERHEAD_BYTES`). Tenant density is scheduler-visible via the existing `GpuVramBytes` accounting.

**Full analysis:** Debugging Journal, Sessions 18–22.

---

## 6. Files Reference

### GPU Proxy Source (`src/gpu-proxy/`)

| File | Purpose |
|------|---------|
| `shim/cuda_shim.c` | Runtime API shim (~1900 lines) |
| `shim/cuda_driver_shim.c` | Driver API shim (~1800 lines) |
| `shim/transport.c` / `transport.h` | TCP/vsock transport layer |
| `shim/nvml_shim.c` | NVML fake GPU info |
| `daemon/gpu_proxy_daemon.c` | Host daemon (~2100 lines) |
| `proto/gpu_proxy_proto.h` | Wire protocol definitions |
| `stubs/cublas_stub.c` / `cublasLt_stub.c` | cuBLAS stubs with version tags |

### Orchestrator / NodeAgent

| File | Purpose |
|------|---------|
| `LibvirtVmManager.cs` | `EnsureGpuProxyShim` — injects GPU config into cloud-init |
| `TemplateSeederService.cs` | Template definitions with GPU env vars |
| `GpuProxyService.cs` | Daemon lifecycle management |
| `install.sh` | Full build + deploy pipeline |