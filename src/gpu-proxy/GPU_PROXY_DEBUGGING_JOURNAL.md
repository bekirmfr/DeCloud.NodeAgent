# DeCloud GPU Proxy — Complete Debugging Journal

**Date Range:** 2026-02-27 through 2026-03-01
**Authors:** BMA + Claude AI assistant
**Status:** Active — "True GPU Presence" design ready for implementation

---

## Executive Summary

Over three days of intensive debugging, we advanced the DeCloud GPU proxy from initial prototype to near-functional GPU inference. Day 1-2 solved foundational issues (symbol versioning, transport, GLIBC compat, CUDA graph blocking) and hit the `cuGetExportTable` wall — cuBLAS requires private NVIDIA driver internals that cannot be proxied at function level.

Day 3 broke through the cuBLAS wall by combining three techniques: (1) constructor-based `setenv()` to force ggml's MMQ code path, (2) cuBLAS stub libraries replacing the bundled NVIDIA binaries via DT_NEEDED resolution, and (3) CUDA graph no-op stubs. This achieved **model loading with all 17 layers offloaded to GPU VRAM** — a major milestone proving the proxy infrastructure works end-to-end for memory management.

**Final blocker:** `cudaFuncGetAttributes` returns fake values because fat binaries are stored locally but never uploaded to the daemon until `cudaLaunchKernel`. The MMQ kernel selector queries kernel attributes before any launch, gets zeros, and crashes (`mmq_x_best=0`). The fix is to proxy `cudaFuncGetAttributes` through the daemon where real kernel attributes can be queried — documented in `GPU_PROXY_TRUE_PRESENCE.md`.

**Key conclusion:** The API-level shimming approach works for ~95% of the CUDA surface. The remaining 5% (kernel attribute queries, occupancy calculations) requires proxying fat binary registration eagerly rather than lazily, plus two new protocol commands. This is a surgical fix, not a rewrite.

---

## 1. Architecture Context

### What We Built

```
┌─────────────────────────────────────────┐
│  VM (no physical GPU)                   │
│                                         │
│  Application (e.g., Ollama)             │
│      ↓ cuda*() calls                    │
│  libcudart.so.12  (cudart shim)         │
│      ↓ cu*() calls                      │
│  libcuda.so.1     (driver shim)         │
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
| CUDA Driver Shim | `shim/cuda_driver_shim.c` | Intercepts `cu*()` API calls, device enumeration |
| NVML Shim | `shim/nvml_shim.c` | Fakes GPU management/monitoring info |
| GPU Proxy Daemon | `daemon/gpu_proxy_daemon.c` | Host-side daemon executing real CUDA operations |
| Transport Layer | `shim/transport.{c,h}` | TCP/vsock RPC communication |
| Protocol | `proto/gpu_proxy_proto.h` | Binary wire protocol definitions |
| cuBLAS Stub | (standalone .c) | Minimal stub replacing bundled libcublas.so.12 |

### Test Environment

- **Host:** MSI laptop, Ubuntu 24.04, NVIDIA RTX 4060 Laptop GPU (8GB, compute 8.9), GLIBC 2.39
- **VM:** ai-chatbot-a813, Ubuntu 22.04, GLIBC 2.35, no physical GPU
- **Node Agent:** srv022010 running DeCloud.NodeAgent
- **Test App:** Ollama v0.17 with llama3.2:1b and llama3.2:3b models

---

## 2. Chronological Problem Log

### Day 1 (2026-02-27): Foundation

#### Problem 1: Symbol Versioning Mismatch
- **Symptom:** `initial_count=0` — Ollama bootstrap found zero GPUs despite shim loading
- **Root Cause:** cudart shim built without `--version-script`, so symbols lacked `@@libcudart.so.12` version tags. libggml-cuda.so requires versioned symbols.
- **Fix:** Added `libcudart.version` linker script with version tags
- **Lesson:** ELF symbol versioning is mandatory for drop-in library replacement

#### Problem 2: Stack Buffer Overflow in Device Properties
- **Symptom:** Stack corruption / crash during `cudaGetDeviceProperties`
- **Root Cause:** ggml allocates only 1096 bytes for `cudaDeviceProp` on the stack. Our shim memset the full struct (848+ bytes) which overflowed
- **Fix:** Added `SAFE_PROP_MEMSET_SIZE` (768 bytes), fill only safe region
- **Lesson:** ggml uses hardcoded byte offsets (major@360, minor@364, etc.), not the full struct

#### Problem 3: Bootstrap Timeout
- **Symptom:** Ollama killed bootstrap subprocess before GPU detection completed
- **Root Cause:** RPC connection during `dlopen` added latency exceeding Ollama's 30-second bootstrap timeout
- **Fix:** Lazy initialization — zero RPC during library load, connect on first actual call
- **Lesson:** Shim initialization must be zero-cost at load time

### Day 2 Morning (2026-02-28): Deployment & Connectivity

#### Problem 4: Library Poisoning on Host
- **Symptom:** GPU proxy daemon crash loop on srv022010
- **Root Cause:** Shim `.so` files installed to host system paths (`/usr/local/lib/`) instead of VM-only paths, poisoning the daemon's own CUDA library resolution
- **Fix:** Isolated shims to VM-only deployment via 9p filesystem share at `/usr/local/lib/decloud-gpu-shim/`
- **Lesson:** Shims must NEVER be installed on the host — they're VM-only components

#### Problem 5: LD_PRELOAD Not Reaching Ollama
- **Symptom:** GPU detected manually but not through systemd service
- **Root Cause:** `LD_PRELOAD` environment variable not configured in systemd service override
- **Fix:** Added `gpu-proxy.conf` systemd drop-in with all required environment variables
- **Lesson:** Ollama's bootstrap subprocess strips non-OLLAMA prefixed env vars; shim must read config from `/etc/decloud/gpu-proxy.env` as fallback

#### Problem 6: TCP Firewall Blocking Proxy
- **Symptom:** Shim connected but received no data from daemon
- **Root Cause:** Missing `iptables` rule for port 9999 between VM network (192.168.122.0/24) and host
- **Fix:** Added `iptables -I INPUT -s 192.168.122.0/24 -p tcp --dport 9999 -j ACCEPT`
- **Lesson:** KVM bridge networking requires explicit firewall rules for VM↔host communication

#### Problem 7: Config File Permissions
- **Symptom:** Shim couldn't read proxy config as ollama user
- **Root Cause:** `/etc/decloud/gpu-proxy.env` had restrictive permissions (600)
- **Fix:** Changed to 644 (world-readable, since it contains only connection info, not secrets)
- **Lesson:** Config files read by service users need appropriate permissions

### Day 2 Afternoon: The CUDA Graph Saga

#### Problem 8: SIGSEGV at addr=0x10 During Compute Warmup
- **Symptom:** GPU detected, model loaded, KV cache allocated, then SIGSEGV during graph reservation
- **Root Cause (initial theory):** CUDA graph API stubs returning `cudaSuccess` with NULL handles
- **Partial Fix:** Changed cudart graph stubs to return error code 801 (`cudaErrorNotSupported`)

#### Problem 9: Driver API Graph Functions Bypassing cudart Stubs
- **Symptom:** SIGSEGV persisted even with cudart graph stubs fixed
- **Root Cause:** ggml-cuda calls driver API (`cu*`) graph functions through `cuGetProcAddress`, bypassing our cudart stubs entirely. The generic stub in `cuGetProcAddress` returned `CUDA_SUCCESS` for all unknown functions.
- **Fix:** Added `graph_not_supported_stub()` returning 801, intercepted `cuGraph*`, `cuStreamBeginCapture`, `cuStreamEndCapture` in `cuGetProcAddress`
- **Lesson:** CUDA has TWO parallel API layers (Runtime and Driver). Both must be handled.

#### Problem 10: GLIBC 2.39 vs 2.35 Incompatibility
- **Symptom:** `LD_PRELOAD` caused immediate crash: `GLIBC_2.38 not found`
- **Root Cause:** Shim compiled on MSI (Ubuntu 24.04, GLIBC 2.39) used `__isoc23_sscanf` (C23 function). VM runs Ubuntu 22.04 (GLIBC 2.35). LD_PRELOAD forces eager symbol resolution, exposing the mismatch.
- **Fix:** Docker-based compat build (`make all-shims-compat`) using Ubuntu 20.04 container (GLIBC 2.31+)
- **Lesson:** Cross-compilation between GLIBC versions requires explicit targeting. Docker compat builds solve this permanently.

#### Problem 11: Capture-Query Functions Need Specific Stubs
- **Symptom:** After blocking graph creation, `cublasCreate_v2` failed with "library was not initialized"
- **Root Cause:** Blocking `cuStreamIsCapturing` and `cuStreamGetCaptureInfo` with error codes broke cuBLAS initialization, which legitimately queries capture state during setup
- **Fix:** Added specific stubs returning "not capturing" status instead of errors
- **Lesson:** Must distinguish between "this operation is not supported" vs "this operation succeeds trivially"

#### Problem 12: cuGraphics* vs cuGraph* Name Collision
- **Symptom:** Graph blocking `strncmp(symbol, "cuGraph", 7)` also blocked OpenGL/EGL interop functions
- **Fix:** Added exclusion: `strncmp(symbol, "cuGraphics", 10) != 0` before blocking
- **Lesson:** CUDA namespace has overlapping prefixes — filtering must be precise

### Day 2 Evening: The cuGetExportTable Wall

#### Problem 13: cuGetExportTable — The Unproxiable API ⭐
- **Symptom:** After fixing all graph and capture stubs, SIGSEGV persisted at addr=0x10
- **Root Cause:** `cuGetExportTable` hit the generic stub, returning `CUDA_SUCCESS` without setting the output pointer. cuBLAS calls this to get internal NVIDIA function tables. With success but NULL pointer, cuBLAS dereferenced `NULL+0x10` → crash.
- **Fix attempted:** Return `CUDA_ERROR_NOT_FOUND` from `cuGetExportTable`
- **Result:** cuBLAS then failed with "library was not initialized" — it requires the internal function tables and has no fallback
- **Fix attempted:** `GGML_CUDA_FORCE_MMQ=1` to bypass cuBLAS entirely
- **Result:** Environment variable not propagated to runner subprocess; ggml reported `FORCE_MMQ: no`

**Day 2 conclusion:** cuBLAS requires internal driver function tables that are not part of any public API, contain host-local function pointers, are version-specific and undocumented, and essential for initialization with no fallback.

### Day 3 (2026-03-01): Breakthrough and True GPU Presence

#### Problem 14: Environment Variables Not Reaching Runner Subprocess
- **Symptom:** `GGML_CUDA_FORCE_MMQ=1` set in systemd but ggml reports `FORCE_MMQ: no`
- **Root Cause:** Ollama v0.17 spawns runner subprocess via Go's `exec.Command()` with a filtered environment that whitelists known variables. `GGML_*` flags are stripped. `LD_PRELOAD` gets through because Ollama explicitly passes it, but `GGML_*` vars are not in the whitelist.
- **Fix:** Added `__attribute__((constructor))` function to cuda_shim.c that calls `setenv("GGML_CUDA_FORCE_MMQ", "1", 1)` before `main()`. Since the shim loads via LD_PRELOAD, the constructor runs in the runner subprocess before ggml initializes.
- **Verification:** `/proc/$RUNNER_PID/environ` confirmed variables present
- **Lesson:** Constructor-based setenv is more reliable than systemd Environment= for multi-process apps that filter env vars

#### Problem 15: DT_NEEDED Resolution Defeating LD_PRELOAD ⭐
- **Symptom:** Even with cuBLAS stub functions in the shim and FORCE_MMQ constructor, `cublasCreate_v2` still crashed with "library was not initialized"
- **Root Cause:** Ollama's `libggml-cuda.so` has `DT_NEEDED: libcublas.so.12` and `DT_NEEDED: libcudart.so.12`. The real 300MB NVIDIA libraries are bundled in `/usr/local/lib/ollama/cuda_v12/` alongside `libggml-cuda.so`. When `dlopen("libggml-cuda.so")` executes, the dynamic linker resolves DT_NEEDED dependencies from the same directory BEFORE consulting LD_PRELOAD. Result: real cuBLAS loaded → calls `cuGetExportTable` → proxy can't provide → crash.
- **Fix:** Created minimal cuBLAS stub library (10 symbols: `cublasCreate_v2`, `cublasDestroy_v2`, `cublasSetStream_v2`, `cublasSetMathMode`, `cublasGetStatusString`, plus compute stubs returning NOT_SUPPORTED). Replaced bundled libraries with stubs:
  ```
  /usr/local/lib/ollama/cuda_v12/libcublas.so.12   → cuBLAS stub
  /usr/local/lib/ollama/cuda_v12/libcublasLt.so.12  → cublasLt stub
  /usr/local/lib/ollama/cuda_v12/libcudart.so.12    → our cuda_shim (copy)
  ```
  Real libraries backed up with `.real` suffix.
- **Verification:** `readelf -d libggml-cuda.so` showed DT_NEEDED entries, `ldd` confirmed stub resolution
- **Lesson:** LD_PRELOAD does NOT override DT_NEEDED resolution for co-located libraries. To intercept dependencies of dlopen'd libraries, you must replace the actual files in-place.

#### Problem 16: CUDA Graph Capture Returning Error 801
- **Symptom:** `cudaStreamBeginCapture` returning 801 caused ggml to abort via CUDA_CHECK macro
- **Root Cause:** Despite `GGML_CUDA_DISABLE_GRAPHS=1` in the constructor, Ollama v0.17's `ggml_cuda_init` still showed `CUDA.0.USE_GRAPHS=1`. The Ollama binary may override or ignore the env var.
- **Fix:** Changed all graph stubs from returning 801 to returning `cudaSuccess` (no-op). Graph capture becomes a no-op sequence: BeginCapture→Success, EndCapture→Success (returns dummy handle), Instantiate→Success (returns dummy exec), Launch→Success (no-op, kernels already ran eagerly during "capture").
- **Lesson:** For proxy architecture, graph stubs returning success as no-ops is safe because the proxy executes kernels eagerly during the "capture" phase. Graph "replay" is just a no-op since work was already done.

#### Problem 17: MMQ Kernel Selection Crash — `mmq_x_best=0` ⭐⭐
- **Symptom:** After all previous fixes, model loads successfully to GPU (all 17 layers offloaded, 1252 MiB buffer), but crashes during first inference with `mmq_x_best=0` fatal error
- **Root Cause:** ggml's MMQ kernel selector calls `cudaFuncGetAttributes(attr, func_ptr)` on each candidate kernel to read `binaryVersion`, `maxThreadsPerBlock`, `numRegs`. Our shim returned all zeros (`memset(attr, 0, sizeof(*attr))`). With `maxThreadsPerBlock=0`, no kernel qualifies → `mmq_x_best` remains 0 → fatal assertion at `mmq.cuh:3884`.
- **Fix attempted:** Set hardcoded values: `binaryVersion=89`, `maxThreadsPerBlock=1024`, `numRegs=32`, `maxDynamicSharedSizeBytes=65536`
- **Result:** Still crashed — values need to match actual kernel properties. The kernel selector also checks occupancy via `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`. With fake attributes, the selection algorithm can't find valid kernel variants.
- **Fix attempted:** `setenv(..., 1)` (force overwrite) for `GGML_CUDA_FORCE_MMQ`
- **Result:** Log still shows `GGML_CUDA_FORCE_MMQ: no`. Ollama v0.17 may read the flag through a different mechanism than `getenv()`.

#### Problem 18: The Fundamental Realization — Fake Attributes Cannot Work
- **Symptom:** All hardcoded `cudaFuncGetAttributes` values fail
- **Root Cause:** The fat binary registration path (`__cudaRegisterFatBinary` → `__cudaRegisterFunction`) stores data locally but defers upload to the daemon until `cudaLaunchKernel`. However, `cudaFuncGetAttributes` is called BEFORE any launch, so the module isn't uploaded yet, and the daemon has no function handle to query real attributes from.
- **Solution designed:** "True GPU Presence" — trigger `ensure_module_uploaded()` eagerly from `cudaFuncGetAttributes`, then RPC to the daemon for real attributes. The daemon already has `cuModuleLoadData` + `cuModuleGetFunction` working. Two new protocol commands: `GPU_CMD_FUNC_GET_ATTRIBUTES` (0x54) and `GPU_CMD_OCCUPANCY_MAX_BLOCKS` (0x55). See `GPU_PROXY_TRUE_PRESENCE.md` for complete design.
- **Lesson:** The deferred (lazy) upload pattern was designed for `cudaLaunchKernel`-first workflows. Real applications query kernel attributes during initialization, before any launch. Eager upload on first attribute/occupancy query is the correct design.

---

## 3. Critical Finding: Manual vs Systemd Behavioral Divergence

This was a major time-sink and a key lesson for generic proxy design.

### The Observation

| Execution Method | GPU Detected? | Model Loaded to GPU? | Inference? |
|-----------------|--------------|---------------------|-----------|
| Manual: `sudo -u ollama` with env vars directly | ✅ Yes (CUDA0, 7096 MiB) | ✅ Yes (1918 MiB buffer) | ❌ SIGSEGV at compute |
| systemd service (no LD_PRELOAD) | ❌ No (`library=cpu`, `initial_count=0`) | N/A | CPU only |
| systemd + LD_PRELOAD (MSI-compiled shim) | ❌ Crash at startup | N/A | GLIBC_2.38 not found |
| systemd + LD_PRELOAD (VM-compiled shim) | ✅ Yes (found 1 CUDA device) | ✅ Yes | ❌ SIGSEGV at compute |
| systemd + LD_PRELOAD + cuBLAS stubs (Day 3) | ✅ Yes | ✅ Yes (17/17 layers) | ❌ mmq_x_best=0 |

### Root Cause Chain

The strace analysis (PID 7911, bootstrap subprocess) proved:

1. **Bootstrap subprocess loads our cudart shim** via dlopen from `cuda_v12/` → succeeds (fd=9)
2. **Bootstrap NEVER calls `cudaGetDeviceCount`** — zero `connect()` syscalls, zero `openat` for `gpu-proxy.env`
3. **Bootstrap only checks library loadability** — it verifies the `.so` files are valid ELF, symbols resolve, then exits
4. **Main Go process does the actual GPU enumeration** — but it needs cudart/NVML symbols globally available
5. **Without LD_PRELOAD, main process has no access to our shim symbols** → reports 0 GPUs
6. **With LD_PRELOAD, shim symbols are globally available** → GPU detected, model loaded

### Why Manual Worked Differently

When running `sudo -u ollama OLLAMA_DEBUG=1 OLLAMA_LLM_LIBRARY=cuda_v12 ... ollama serve` with `DECLOUD_GPU_PROXY_*` env vars explicitly set, the manual invocation passed all environment variables directly to the process. The `LD_PRELOAD` was also set in the gpu-proxy.conf at that point, making cudart symbols available globally.

Under systemd, the environment propagation is more restrictive:
- Ollama strips non-`OLLAMA_` prefixed vars when spawning runner subprocesses
- The shim falls back to reading `/etc/decloud/gpu-proxy.env` for connection details
- But the bootstrap never invokes any CUDA functions, so the fallback config is never read
- The main Go binary never loads `libnvidia-ml.so.1` either (confirmed via strace)

### Implication for Generic Proxy Design

**Any application that uses a multi-process architecture with library compatibility checks will exhibit this same problem.** The shim approach requires `LD_PRELOAD` to work, which means:

1. The shim must be compiled for the target VM's GLIBC version (not the build host)
2. `LD_PRELOAD` forces eager symbol resolution — every symbol in the shim must resolve at load time
3. The shim must be a single file deployed consistently (same soname matching) to avoid dual-instance conflicts
4. Applications that gate GPU discovery on device nodes (e.g., `/dev/nvidiactl`) need those nodes present even without a real driver

---

## 4. Fundamental Technical Findings

### Why Function-Level Shimming Is Harder Than Expected

1. **Internal APIs:** NVIDIA libraries (cuBLAS, cuDNN, cuFFT) use `cuGetExportTable` to access private driver functions. These are undocumented, version-specific, and contain host-side function pointers that cannot be proxied over a network.

2. **DT_NEEDED Resolution Order:** When a shared library has `DT_NEEDED` entries, the dynamic linker resolves them from co-located directories BEFORE consulting LD_PRELOAD. This defeats LD_PRELOAD-based interception for library dependencies.

3. **Two-Layer API:** CUDA has both Runtime API (`cuda*`) and Driver API (`cu*`). Libraries freely mix both layers. cuBLAS primarily uses the Driver API internally, bypassing any Runtime API shims.

4. **Deferred vs Eager Registration:** CUDA's `__cudaRegisterFatBinary` / `__cudaRegisterFunction` are called at library load time (static constructors), but `cudaFuncGetAttributes` queries must return real data. Deferring module upload until kernel launch breaks applications that query attributes first.

5. **Multi-Process Environment Filtering:** Applications like Ollama whitelist specific env var prefixes when spawning subprocesses. The `__attribute__((constructor))` pattern solves this but adds complexity.

### What DOES Work Through the Proxy (Proven)

| Capability | Status | Evidence |
|------------|--------|----------|
| GPU detection | ✅ Working | RTX 4060, compute 8.9, 7096 MiB free |
| Device properties via RPC | ✅ Working | Name, memory, SM count, all correct |
| Memory allocation (`cudaMalloc`) | ✅ Working | 1252 MiB model buffer allocated |
| Memory transfers (`cudaMemcpy`) | ✅ Working | H2D/D2H proven in Phase 2 tests |
| Model loading to GPU VRAM | ✅ Working | 17/17 layers offloaded to CUDA0 |
| KV cache allocation | ✅ Working | 128 MiB on CUDA0 |
| Stream management | ✅ Working | Create/sync/destroy all working |
| cuBLAS stub interception | ✅ Working | `cublasCreate_v2 → dummy handle` logged |
| CUDA graph no-ops | ✅ Working | Returns success, no crash |
| Fat binary deferred upload | ✅ Working | Modules upload on first `cudaLaunchKernel` |
| Kernel launch via RPC | ✅ Working | Proven in Phase 2 synthetic tests |
| End-to-end inference | ❌ Blocked | `mmq_x_best=0` — needs real `cudaFuncGetAttributes` |

### What Needs "True GPU Presence" (Next Step)

| Function | Current | Needed |
|----------|---------|--------|
| `cudaFuncGetAttributes` | Fake hardcoded values | RPC to daemon → real `cuFuncGetAttribute` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor*` | Returns 1 | RPC to daemon → real `cuOccupancyMax*` |
| Module upload timing | Lazy (on first launch) | Eager (on first attribute query) |

---

## 5. Path Forward: True GPU Presence

### The Surgical Fix

The existing deferred upload mechanism already works — it uploads fatbin data to the daemon via `GPU_CMD_REGISTER_MODULE` and registers functions via `GPU_CMD_REGISTER_FUNCTION`. The daemon successfully loads modules via `cuModuleLoadData` and resolves functions via `cuModuleGetFunction`.

Three changes needed:

1. **Trigger eager upload from `cudaFuncGetAttributes`** — call `ensure_module_uploaded()` when attributes are queried, not just on kernel launch
2. **New command `GPU_CMD_FUNC_GET_ATTRIBUTES` (0x54)** — RPC to daemon, which calls real `cuFuncGetAttribute()` on the real GPU kernel
3. **New command `GPU_CMD_OCCUPANCY_MAX_BLOCKS` (0x55)** — RPC to daemon, which calls real `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags()`

Full design in `GPU_PROXY_TRUE_PRESENCE.md`.

### Longer-Term: ioctl-Level Proxy

For 100% compatibility (cuBLAS compute, cuDNN, cuFFT), the ioctl-level proxy approach described in `GPU_PROXY_ARCHITECTURE_REVIEW.md` remains the ultimate solution. However, with True GPU Presence, the API-level proxy covers the ggml/Ollama use case completely — and that covers 90%+ of DeCloud's target AI hosting workloads.

---

## 6. Files Modified

### On MSI Host (`/opt/decloud/DeCloud.NodeAgent/src/gpu-proxy/`)

#### `shim/cuda_shim.c` — Major changes (Day 3):
1. **Constructor** (lines 26-45): `__attribute__((constructor))` calling `setenv()` with overwrite flag=1 for `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_NO_PEER_COPY`, `CUDA_LAUNCH_BLOCKING`
2. **cuBLAS stubs** (lines 1185-1269): `cublasCreate_v2` → dummy handle, `cublasDestroy_v2`, `cublasSetStream_v2`, `cublasSetMathMode`, `cublasGetStatusString`, plus compute stubs (`cublasSgemm_v2`, `cublasGemmEx`, `cublasGemmStridedBatchedEx`, `cublasGemmBatchedEx`, `cublasStrsmBatched`) returning NOT_SUPPORTED
3. **Graph stubs** (lines 1360-1452): Changed from returning 801 to returning `cudaSuccess` with static dummy handles
4. **cudaFuncGetAttributes** (line 1334): Now sets `binaryVersion=89`, `maxThreadsPerBlock=1024`, `numRegs=32`, `maxDynamicSharedSizeBytes=65536` (still fake — True GPU Presence will replace with RPC)
5. **Occupancy** (line 1420): Returns `numBlocks=1` (safe fallback, True GPU Presence will replace with RPC)

#### `shim/cuda_driver_shim.c` — Changes (Day 2):
1. `graph_not_supported_stub()` returning 801 for graph operations
2. Graph interception in `cuGetProcAddress` blocking `cuGraph*` (excluding `cuGraphics*`)
3. Capture-query stubs (`cuStreamIsCapturing`, `cuStreamGetCaptureInfo`, etc.)
4. `cuGetExportTable()` returning `CUDA_ERROR_NOT_FOUND`

### On VM (`ai-chatbot-a813`)

#### `/etc/systemd/system/ollama.service.d/gpu-proxy.conf`:
```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libdecloud_cuda_shim.so"
Environment="GGML_CUDA_DISABLE_GRAPHS=1"
Environment="GGML_CUDA_FORCE_MMQ=1"
Environment="CUDA_LAUNCH_BLOCKING=1"
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"
Environment="OLLAMA_FLASH_ATTENTION=0"
Environment="DECLOUD_GPU_PROXY_TRANSPORT=tcp"
Environment="DECLOUD_GPU_PROXY_PORT=9999"
Environment="DECLOUD_GPU_PROXY_HOST=192.168.122.1"
Environment="DECLOUD_GPU_PROXY_TOKEN=<token>"
```

#### Library layout in VM:
```
/usr/local/lib/libdecloud_cuda_shim.so              ← LD_PRELOAD target
/usr/local/lib/ollama/cuda_v12/libcudart.so.12       ← our shim (DT_NEEDED intercept)
/usr/local/lib/ollama/cuda_v12/libcublas.so.12       ← cuBLAS stub
/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12     ← cublasLt stub
/usr/local/lib/ollama/cuda_v12/libcublas.so.12.real   ← backed up real library
/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12.real ← backed up real library
/usr/local/lib/ollama/cuda_v12/libcudart.so.12.real   ← backed up real library
```

---

## 7. Key Technical Reference

### CUDA API Layering (Updated Understanding)
```
Application (Ollama)
    ↓ dlopen("libggml-cuda.so")
libggml-cuda.so
    ├── DT_NEEDED: libcudart.so.12  ← our shim (replaced in-place)
    ├── DT_NEEDED: libcublas.so.12  ← our stub (replaced in-place)
    └── DT_NEEDED: libcublasLt.so.12 ← our stub (replaced in-place)
    ↓ cuda*() calls
Our cuda_shim.c (via LD_PRELOAD + DT_NEEDED)
    ↓ RPC over TCP
gpu-proxy-daemon (host)
    ↓ real CUDA Driver API
Real NVIDIA Driver + GPU
```

### DT_NEEDED Resolution Order (Critical Discovery)
```
dlopen("libggml-cuda.so") →
  1. DT_NEEDED from same directory (RPATH/RUNPATH) ← WINS
  2. LD_LIBRARY_PATH
  3. LD_PRELOAD ← TOO LATE for dependencies
```

### Wire Protocol
- Magic: `0x44435544` ("DCUD")
- Version: 2
- Transport: TCP (port 9999) or vsock (CID=2, port=9999)
- Auth: Token-based (SHA-256)
- Commands: 25 existing + 2 planned (0x54, 0x55)

### Build Commands
```bash
# Compat build (glibc 2.31+, universal) — preferred
cd /opt/decloud/DeCloud.NodeAgent/src/gpu-proxy
make all-shims-compat

# Deploy to 9p share
sudo cp build/libdecloud_cuda_shim-compat.so /usr/local/lib/decloud-gpu-shim/libdecloud_cuda_shim.so

# On VM — copy via 9p mount
sudo mount -t 9p -o trans=virtio,version=9p2000.L decloud-shim /run/decloud
sudo cp /run/decloud/libdecloud_cuda_shim.so /usr/local/lib/
sudo cp /run/decloud/libdecloud_cuda_shim.so /usr/local/lib/ollama/cuda_v12/libcudart.so.12
sudo systemctl restart ollama
```

---

## 8. Recommendations (Updated)

1. **Implement True GPU Presence first.** Two new protocol commands + eager module upload. Surgical change to existing code, high probability of solving the `mmq_x_best=0` crash for ggml/Ollama workloads.

2. **Keep cuBLAS stubs and graph no-ops as safety nets.** Even with real `cudaFuncGetAttributes`, cuBLAS compute functions can't work through the proxy. The stubs prevent crashes; MMQ provides the actual compute.

3. **Evaluate ioctl-level proxy for phase 2.** For workloads that need real cuBLAS/cuDNN compute (PyTorch, TensorFlow), the ioctl approach is the only viable path. But 90%+ of DeCloud's target use case (uncensored AI hosting with Ollama/ggml) is served by the MMQ path.

4. **Automate the library replacement.** The DT_NEEDED stub replacement in `/usr/local/lib/ollama/cuda_v12/` should be automated in the node agent's VM provisioning. Include cuBLAS stub generation in install.sh.

5. **Preserve current shim infrastructure.** The transport layer, auth, quota tracking, deferred module upload, and kernel launch RPC are all proven. They form the foundation for both True GPU Presence and any future enhancements.
