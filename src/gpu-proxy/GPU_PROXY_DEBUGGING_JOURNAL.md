# DeCloud GPU Proxy — Complete Debugging Journal

**Date Range:** 2026-02-27 through 2026-03-01
**Authors:** BMA + Claude AI assistant
**Status:** Active — "True GPU Presence" implementation is the final remaining step

---

## Executive Summary

Over four days of intensive debugging, we advanced the DeCloud GPU proxy from initial prototype to near-functional GPU inference. Days 1-2 solved foundational issues (symbol versioning, transport, GLIBC compat, CUDA graph blocking) and hit the `cuGetExportTable` wall — cuBLAS requires private NVIDIA driver internals that cannot be proxied at function level.

Day 3 broke through the cuBLAS wall by combining three techniques: (1) constructor-based `setenv()` to force ggml's MMQ code path, (2) cuBLAS stub libraries replacing the bundled NVIDIA binaries via DT_NEEDED resolution, and (3) CUDA graph no-op stubs. This achieved **model loading with all 17 layers offloaded to GPU VRAM** — a major milestone proving the proxy infrastructure works end-to-end for memory management.

**Day 4 (Session 2, 2026-03-01)** resolved a critical silent failure: the CUDA backend wasn't loading in Ollama's runner subprocess because the cuBLAS stub had wrong ELF version tags (`@@libcudart.so.12` instead of `@@libcublas.so.12`). Building a separate `cublas_stub.c` with proper version script and soname fixed this. GPU discovery now works fully through systemd — Ollama detects "NVIDIA GeForce RTX 4060 Laptop GPU" with 6.9 GiB available at every restart. End-to-end inference remains blocked by `mmq_x_best=0` — the final blocker requiring True GPU Presence.

**Final blocker:** `cudaFuncGetAttributes` returns fake values because fat binaries are stored locally but never uploaded to the daemon until `cudaLaunchKernel`. The MMQ kernel selector queries kernel attributes before any launch, gets identical fake values for all variants, and crashes (`mmq_x_best=0`). The fix is to proxy `cudaFuncGetAttributes` through the daemon where real kernel attributes can be queried — documented in `GPU_PROXY_TRUE_PRESENCE.md`.

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
| Transport Layer | `shim/transport.{c,h}` | vsock/TCP connection management |
| Protocol | `proto/gpu_proxy_proto.h` | Wire protocol structs and command IDs |
| cuBLAS Stub | `stubs/cublas_stub.c` | Separate library with correct `@@libcublas.so.12` version tags |

---

## 2. Problem Log

### Day 1-2 (2026-02-27 through 2026-02-28): Foundation

#### Problem 1-13: [See previous journal entries]
Issues covering: transport initialization, vsock configuration, GLIBC 2.38 compatibility, symbol versioning, LD_PRELOAD mechanics, CUDA graph API conflicts, cuGetExportTable wall.

### Day 3 (2026-03-01, Session 1): Breakthrough and True GPU Presence

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
- **Fix:** Changed all graph stubs from returning 801 to returning `cudaSuccess` (no-op). Graph capture becomes a silent no-op since kernels execute eagerly via RPC anyway.
- **Lesson:** Never return error codes from graph APIs when the caller wraps them in CUDA_CHECK. No-op success is always safer.

### Day 4 (2026-03-01, Session 2): cuBLAS Version Tag Fix ⭐⭐

#### Problem 17: cuBLAS Stub Has Wrong ELF Version Tags (Silent CUDA Backend Load Failure)
- **Symptom:** Ollama's runner subprocess loads CPU backend but silently skips CUDA backend. `/info` endpoint returns `[]` (empty device list). No error messages — completely silent failure.
- **Discovery Method:** Manual runner subprocess test (`ollama runner --ollama-engine --port 12345`) showed CPU backend loaded but no "loaded CUDA backend" message. strace would have revealed `dlopen(libggml-cuda.so)` failing, but symbol analysis with `nm` was more targeted.
- **Root Cause:** The cloud-init template copies the same `libdecloud_cuda_shim.so` binary to three locations:
  ```
  cuda_v12/libcudart.so.12   → cuda_shim (38KB, correct @@libcudart.so.12 tags)
  cuda_v12/libcublas.so.12   → cuda_shim (38KB, WRONG @@libcudart.so.12 tags)
  cuda_v12/libcublasLt.so.12 → cuda_shim (38KB, WRONG @@libcudart.so.12 tags)
  ```
  All three inherit `libcudart.so.12` version script. When `dlopen(libggml-cuda.so)` resolves DT_NEEDED, the dynamic linker looks for `cublasCreate_v2@libcublas.so.12` but only finds `cublasCreate_v2@@libcudart.so.12` — version tag mismatch causes symbol resolution failure → `dlopen` returns NULL → CUDA backend silently not loaded.
- **Symbol Evidence:**
  ```
  # What libggml-cuda.so needs:
  U cublasCreate_v2@libcublas.so.12        ← requires libcublas.so.12 version tag
  U cublasGemmBatchedEx@libcublas.so.12
  U cublasGetStatusString@libcublas.so.12
  U cublasStrsmBatched@libcublas.so.12

  # What our broken stub exported:
  T cublasCreate_v2@@libcudart.so.12       ← WRONG version tag
  ```
- **Fix:** Built a separate `cublas_stub.c` with its own version script:
  ```bash
  # Version script (libcublas.version):
  libcublas.so.12 {
      global: cublasCreate_v2; cublasDestroy_v2; cublasSetStream_v2;
              cublasSetMathMode; cublasGetMathMode; cublasGetStatusString;
              cublasSgemm_v2; cublasGemmEx; cublasGemmStridedBatchedEx;
              cublasGemmBatchedEx; cublasStrsmBatched;
      local: *;
  };

  # Build:
  gcc -shared -fPIC -o libcublas_stub.so cublas_stub.c \
      -Wl,-soname,libcublas.so.12 \
      -Wl,--version-script=libcublas.version

  # Deploy:
  cp libcublas_stub.so /usr/local/lib/ollama/cuda_v12/libcublas.so.12
  ```
- **Verification:**
  ```bash
  # Symbol tags now correct:
  nm -D --defined-only libcublas_stub.so | grep cublas
  # T cublasCreate_v2@@libcublas.so.12     ← CORRECT

  # libggml-cuda.so loads:
  LD_LIBRARY_PATH=... python3 -c "ctypes.CDLL('libggml-cuda.so')"
  # SUCCESS: libggml-cuda.so loaded ✅
  ```
- **Lesson:** ELF version tags (`@@soname`) are part of symbol identity. `cublasCreate_v2@libcublas.so.12` ≠ `cublasCreate_v2@@libcudart.so.12` even though the function name is identical. You CANNOT reuse the same binary for libraries with different sonames if the consumer requires versioned symbols. Each library needs its own version script and soname.

#### Problem 18: GPU Discovery Now Works Through Systemd ✅
- **Symptom (before fix):** `systemctl restart ollama` → `library=cpu`, `initial_count=0`
- **Fix Applied:** Problem 17 fix (correct cuBLAS version tags)
- **Result After Fix:** Full GPU discovery through systemd service:
  ```
  inference compute: id=GPU-0000b8ff-0100-0000-00c0-000000000000
    library=CUDA compute=8.9 name=CUDA0
    description="NVIDIA GeForce RTX 4060 Laptop GPU"
    driver=12.8 type=discrete total="8.0 GiB" available="6.9 GiB"
  ```
- **Confirmed:** Ollama detects the GPU at every `systemctl restart ollama` — this is now reliable and repeatable.
- **Lesson:** The GPU discovery path requires `dlopen(libggml-cuda.so)` to succeed, which requires ALL DT_NEEDED dependencies to resolve with correct version tags. One wrong tag = total silent failure.

#### Problem 19: End-to-End Inference Still Crashes (`mmq_x_best=0`) — Confirmed Blocker
- **Symptom:** `ollama run llama3.2:1b "Say hello" → Error: 500 Internal Server Error: llama runner process has terminated: exit status 2`
- **Crash Trace:**
  ```
  llama_kv_cache: CUDA0 KV buffer size = 128.00 MiB
  mmq_x_best=0
  //ml/backend/ggml/ggml/src/ggml-cuda/template-instances/../mmq.cuh:3884: fatal error
  SIGABRT: abort
  signal arrived during cgo execution
  ```
- **Root Cause:** Confirmed same as documented — `cudaFuncGetAttributes` returns fake hardcoded values. The MMQ kernel selector gets identical attributes for all variants, can't distinguish them, rejects all → `mmq_x_best=0`.
- **Key Insight:** This is NOT a library loading issue anymore. The runner:
  1. ✅ Loads CUDA backend successfully
  2. ✅ Detects GPU with correct properties
  3. ✅ Allocates model buffer (1252 MiB) on GPU
  4. ✅ Allocates KV cache (128 MiB) on GPU
  5. ❌ Crashes during first inference when MMQ kernel selector calls `cudaFuncGetAttributes`
- **Fix Required:** True GPU Presence — see `GPU_PROXY_TRUE_PRESENCE.md`

---

## 3. Critical Finding: Manual vs Systemd Behavioral Divergence

This was a major time-sink and a key lesson for generic proxy design.

### The Observation (Updated Day 4)

| Execution Method | CUDA Backend Loaded? | GPU Detected? | Model Loaded to GPU? | Inference? |
|-----------------|---------------------|--------------|---------------------|-----------|
| Manual runner with env vars (before cuBLAS fix) | ❌ Silent failure | ❌ Empty [] | N/A | N/A |
| Manual runner with env vars (after cuBLAS fix) | ✅ Yes | ✅ Yes (CUDA0, 7440 MiB free) | ✅ Yes | ❌ mmq_x_best=0 |
| systemd (before cuBLAS fix) | ❌ Silent failure | ❌ `library=cpu` | N/A | CPU only |
| systemd (after cuBLAS fix) | ✅ Yes | ✅ Yes (CUDA0, 6.9 GiB) | ✅ Yes | ❌ mmq_x_best=0 |

### Root Cause Chain (Complete)

The full chain from library deployment to crash:

1. Cloud-init installs `libdecloud_cuda_shim.so` as all three CUDA libraries
2. **Day 3 issue (fixed):** DT_NEEDED loaded real cuBLAS → cuGetExportTable crash. Fix: replace with stubs.
3. **Day 4 issue (fixed):** Stubs had wrong version tags → silent dlopen failure. Fix: separate cuBLAS stub with correct version script.
4. **Current blocker:** Kernel attribute query returns fake values → MMQ selector fails → crash.

### Key Lesson: ELF Dynamic Linking Resolution Priority
```
  1. DT_NEEDED from same directory (RPATH/RUNPATH) ← WINS
  2. LD_LIBRARY_PATH
  3. LD_PRELOAD ← TOO LATE for dependencies

  Version tag matching is REQUIRED for versioned symbol references.
  Unversioned exports CANNOT satisfy versioned import requirements.
```

---

## 4. Current Status Matrix (Updated Day 4)

### What DOES Work Through the Proxy (Proven)

| Capability | Status | Evidence |
|------------|--------|----------|
| GPU detection via systemd | ✅ Working | RTX 4060, compute 8.9, 6.9 GiB free — every restart |
| Device properties via RPC | ✅ Working | Name, memory, SM count, all correct |
| Memory allocation (`cudaMalloc`) | ✅ Working | 1252 MiB model buffer allocated |
| Memory transfers (`cudaMemcpy`) | ✅ Working | H2D/D2H proven in Phase 2 tests |
| Model loading to GPU VRAM | ✅ Working | 17/17 layers offloaded to CUDA0 |
| KV cache allocation | ✅ Working | 128 MiB on CUDA0 |
| Stream management | ✅ Working | Create/sync/destroy all working |
| cuBLAS stub interception | ✅ Working | `cublasCreate_v2 → dummy handle` logged |
| cuBLAS version tags | ✅ Fixed (Day 4) | Separate stub with `@@libcublas.so.12` |
| CUDA graph no-ops | ✅ Working | Returns success, no crash |
| Fat binary deferred upload | ✅ Working | Modules upload on first `cudaLaunchKernel` |
| Kernel launch via RPC | ✅ Working | Proven in Phase 2 synthetic tests |
| libggml-cuda.so loading | ✅ Fixed (Day 4) | `dlopen()` succeeds, CUDA backend registered |
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
4. **cudaFuncGetAttributes**: Now has eager upload + RPC call structure ready for True GPU Presence (currently falls back to fake values until daemon commands implemented)
5. **Occupancy**: Has eager upload + RPC structure ready (currently returns safe fallback of 2)

#### `shim/cuda_driver_shim.c` — Changes (Day 2):
1. `graph_not_supported_stub()` returning 801 for graph operations
2. Graph interception in `cuGetProcAddress` blocking `cuGraph*` (excluding `cuGraphics*`)
3. Capture-query stubs (`cuStreamIsCapturing`, `cuStreamGetCaptureInfo`, etc.)
4. `cuGetExportTable` returning CUDA_ERROR_NOT_FOUND (blocks cuBLAS private internals)
5. `/proc/driver/nvidia/version` creation (later proven unnecessary with OLLAMA_LLM_LIBRARY)

#### `stubs/cublas_stub.c` — NEW (Day 4):
Separate cuBLAS stub library with correct `@@libcublas.so.12` version tags. Contains 11 symbols: `cublasCreate_v2`, `cublasDestroy_v2`, `cublasSetStream_v2`, `cublasSetMathMode`, `cublasGetMathMode`, `cublasGetStatusString`, `cublasSgemm_v2`, `cublasGemmEx`, `cublasGemmStridedBatchedEx`, `cublasGemmBatchedEx`, `cublasStrsmBatched`. Init/handle functions return SUCCESS, compute functions return NOT_SUPPORTED.

#### `stubs/libcublas.version` — NEW (Day 4):
Version script for cuBLAS stub that exports symbols under `libcublas.so.12` version namespace.

### On VM (`/usr/local/lib/ollama/cuda_v12/`)

| File | Source | Size | Purpose |
|------|--------|------|---------|
| `libcudart.so.12` | `libdecloud_cuda_shim.so` (copy) | 38KB | CUDA Runtime shim |
| `libcublas.so.12` | `cublas_stub.so` (separate build) | 17KB | cuBLAS stub with correct version tags |
| `libcublasLt.so.12` | `libdecloud_cuda_shim.so` (copy) | 38KB | cublasLt placeholder (no versioned symbols needed) |

---

## 7. Quick Reference

### Key Environment Variables
```bash
OLLAMA_LLM_LIBRARY=cuda_v12       # Bypasses /proc/driver/nvidia/version check
OLLAMA_DEBUG=INFO                   # Enable debug logging
GGML_CUDA_FORCE_MMQ=1             # Set by constructor (bypasses cuBLAS)
GGML_CUDA_DISABLE_GRAPHS=1        # Set by constructor (safety net)
GGML_CUDA_NO_PEER_COPY=1          # Set by constructor (prevent multi-GPU attempts)
```

### Key Library Paths
```
/usr/local/lib/libcuda.so.1                          → driver shim
/usr/local/lib/libnvidia-ml.so.1                     → NVML shim
/usr/local/lib/ollama/cuda_v12/libcudart.so.12       → cudart shim (copy)
/usr/local/lib/ollama/cuda_v12/libcublas.so.12       → cuBLAS stub (SEPARATE BUILD)
/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12     → placeholder shim (copy)
/usr/local/lib/ollama/cuda_v12/libggml-cuda.so       → Ollama's CUDA backend (original)
/usr/local/lib/ollama/libggml-base.so.0              → Ollama's base backend (original)
```

### DT_NEEDED Resolution Priority
```
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

# Build cuBLAS stub separately
make cublas-stub

# Deploy to 9p share
sudo cp build/libdecloud_cuda_shim-compat.so /usr/local/lib/decloud-gpu-shim/libdecloud_cuda_shim.so
sudo cp build/libcublas_stub.so /usr/local/lib/decloud-gpu-shim/libcublas_stub.so

# On VM — copy via 9p mount
sudo mount -t 9p -o trans=virtio,version=9p2000.L decloud-shim /run/decloud
sudo cp /run/decloud/libdecloud_cuda_shim.so /usr/local/lib/
sudo cp /run/decloud/libdecloud_cuda_shim.so /usr/local/lib/ollama/cuda_v12/libcudart.so.12
sudo cp /run/decloud/libcublas_stub.so /usr/local/lib/ollama/cuda_v12/libcublas.so.12
sudo systemctl restart ollama
```

---

## 8. Recommendations (Updated Day 4)

1. **Implement True GPU Presence first.** Two new protocol commands + eager module upload. Surgical change to existing code, high probability of solving the `mmq_x_best=0` crash for ggml/Ollama workloads.

2. **Keep cuBLAS stubs and graph no-ops as safety nets.** Even with real `cudaFuncGetAttributes`, cuBLAS compute functions can't work through the proxy. The stubs prevent crashes; MMQ provides the actual compute.

3. **Update cloud-init template.** The TemplateSeederService.cs must deploy the separate cuBLAS stub (with correct version tags) instead of copying the same shim binary for all three libraries. The cuBLAS stub is only ~17KB and can be embedded as base64 or distributed via 9p share.

4. **Update Makefile.** Add `cublas-stub` and `cublaslt-stub` build targets that compile stubs with their respective version scripts and sonames.

5. **Evaluate ioctl-level proxy for phase 2.** For workloads that need real cuBLAS/cuDNN compute (PyTorch, TensorFlow), the ioctl approach is the only viable path. But 90%+ of DeCloud's target use case (uncensored AI hosting with Ollama/ggml) is served by the MMQ path.

6. **Preserve current shim infrastructure.** The transport layer, auth, quota tracking, deferred module upload, and kernel launch RPC are all proven. They form the foundation for both True GPU Presence and any future enhancements.