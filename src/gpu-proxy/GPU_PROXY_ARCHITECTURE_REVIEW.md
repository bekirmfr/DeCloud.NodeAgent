# DeCloud GPU Proxy — Architecture Review

**Date Range:** 2026-02-27 through 2026-03-06
**Authors:** BMA + Claude AI assistant
**Status:** Production-ready — Generic proxy with template-driven configuration

---

## Executive Summary

Over ten debugging sessions spanning three weeks, we built a complete CUDA virtualization layer that allows VMs without physical GPUs to run GPU-accelerated inference through a TCP/vsock RPC proxy. The system intercepts CUDA Runtime API, Driver API, and NVML calls in the VM, forwards them to a daemon on the host with the real GPU, and returns results transparently.

**Final performance (llama3.2:1b on RTX 4060 Laptop GPU):**
- Prompt eval: **436 tok/s** (warm), **188 tok/s** (cold)
- Generation: **13-21 tok/s**
- 100% GPU offload, zero manual configuration

The proxy is now **fully generic** — no hardcoded application (Ollama/ggml) or vendor (NVIDIA RTX 4060) dependencies in the C code. Application-specific configuration is driven by template environment variables written to `/etc/decloud/gpu-proxy.env`.

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
| `DECLOUD_GPU_GRAPH_NOOP` | 1 | Graph stubs return `cudaSuccess` (1) or `cudaErrorNotSupported` (0) |
| `DECLOUD_GPU_VMEM_PROXY` | 0 | Virtual memory APIs proxy to daemon (1) or return `NOT_SUPPORTED` (0) |
| `DECLOUD_GPU_DEBUG` | unset | Enable debug logging from shim constructor |

**Config flow:** Orchestrator template → `DefaultEnvironmentVariables` → `EnsureGpuProxyShim` extracts `GGML_*`/`CUDA_*`/`DECLOUD_GPU_*` → writes to `/etc/decloud/gpu-proxy.env` → shim constructor reads and applies.

### Key Design Decisions

**TCP_QUICKACK (Session 9-10):** The biggest performance fix. TCP delayed ACK imposed 40ms wait per small RPC. Must be re-armed before every `read()` (Linux resets per-operation). Applied to shims, transport, and daemon.

**Deferred + Eager Module Upload:** Fat binaries stored locally at `__cudaRegisterFatBinary`, uploaded to daemon only when needed. `cudaFuncGetAttributes` triggers eager upload because ggml queries attributes before first launch.

**Streaming Module Upload:** 1.56GB fatbin from libggml-cuda.so written directly from mmap'd memory — zero copy, zero malloc.

**cuBLAS GEMM Proxy:** cuBLAS init requires `cuGetExportTable` (private NVIDIA internals, cannot be proxied). Solution: stub init, use ggml's MMQ path, proxy only `cublasGemmBatchedEx`/`cublasGemmStridedBatchedEx` for GQA attention.

**Graph Stubs via cuGetProcAddress:** libcudart resolves `cudaStreamBeginCapture` through the driver API (`cuGetProcAddress`), bypassing the runtime shim. Both shims must return consistent results. Driver shim reads `DECLOUD_GPU_GRAPH_NOOP` in a `__attribute__((constructor))` before any CUDA call.

**Real GPU Attributes:** `cu_func_get_attribute` queries daemon via RPC, caches per-function, falls back to safe defaults. Eliminates hardcoded sm_89 vendor dependency.

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
| Kernel launch via RPC | ✅ | 13-21 tok/s generation |
| Prompt evaluation | ✅ | 188-436 tok/s |
| 1.56GB streaming module upload | ✅ | Zero-copy from mmap |
| cuBLAS GEMM proxy | ✅ | GQA attention via RPC |
| Real kernel attributes + occupancy | ✅ | Cached, with fallback |
| Configurable graph stubs | ✅ | `DECLOUD_GPU_GRAPH_NOOP` |
| TCP_NODELAY + TCP_QUICKACK | ✅ | Sub-ms RPC latency |
| Template-driven env vars | ✅ | Generic proxy |
| Zero-touch VM deployment | ✅ | Cloud-init automated |
| Virtual memory (cuMem*) | Scaffolded | Off by default, for PyTorch/vLLM |
| cuDNN/cuFFT/cuSPARSE proxy | Deferred | For scientific computing |

---

## 5. Files Reference

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