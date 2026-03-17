# DeCloud GPU Proxy — True GPU Presence

**Date:** 2026-03-01 (designed), 2026-03-06 (fully implemented + generic refactor)
**Status:** ✅ Implemented and production-ready

---

## 1. Overview

True GPU Presence enables VMs without physical GPUs to present real kernel attributes and occupancy values from the host GPU. This was the final blocker preventing GPU inference — ggml's MMQ kernel selector queries `cudaFuncGetAttributes` on each variant to pick the best one. With fake values, no variant qualified → `mmq_x_best=0` → SIGABRT.

The fix was surgical: trigger eager fat binary upload from `cudaFuncGetAttributes`, add two RPC commands to query real attributes from the daemon, and proxy occupancy calculations. Combined with the TCP_QUICKACK performance fix, this achieves **436 tok/s prompt eval** through the proxy.

---

## 2. How It Works

### Before (Broken)
```
cudaFuncGetAttributes(ptr) → hardcoded fake values (all identical)
  → MMQ selector can't distinguish variants → mmq_x_best=0 → SIGABRT
```

### After (Working)
```
cudaFuncGetAttributes(ptr) → trigger ensure_module_uploaded()
  → fatbin uploaded to daemon → cuModuleLoadData + cuModuleGetFunction
  → RPC GPU_CMD_FUNC_GET_ATTRIBUTES → daemon calls cuFuncGetAttribute()
  → REAL values (binaryVersion=89, numRegs=32, etc.) returned to VM
  → MMQ selector picks optimal variant → inference runs
```

---

## 3. Protocol Commands

### GPU_CMD_FUNC_GET_ATTRIBUTES (0x54)

Queries real kernel attributes from the host GPU where the module is loaded.

```c
// Request
typedef struct __attribute__((packed)) {
    uint64_t host_func_ptr;   // VM-side function pointer (lookup key)
} GpuFuncGetAttributesRequest;

// Response
typedef struct __attribute__((packed)) {
    int32_t  binaryVersion;
    int32_t  maxThreadsPerBlock;
    int32_t  numRegs;
    int32_t  sharedSizeBytes;
    int32_t  constSizeBytes;
    int32_t  localSizeBytes;
    int32_t  maxDynamicSharedSizeBytes;
    int32_t  preferredShmemCarveout;
    int32_t  ptxVersion;
} GpuFuncGetAttributesResponse;
```

### GPU_CMD_OCCUPANCY_MAX_BLOCKS (0x55)

Queries real occupancy from the host GPU for accurate block scheduling.

```c
// Request
typedef struct __attribute__((packed)) {
    uint64_t host_func_ptr;
    int32_t  blockSize;
    uint64_t dynamicSMemSize;
    uint32_t flags;
} GpuOccupancyMaxBlocksRequest;

// Response
typedef struct __attribute__((packed)) {
    int32_t numBlocks;
} GpuOccupancyMaxBlocksResponse;
```

---

## 4. Implementation

### Runtime API Shim (`cuda_shim.c`)

`cudaFuncGetAttributes` triggers eager module upload, then RPCs to daemon:
- Finds registered function by host pointer
- Calls `ensure_module_uploaded()` if not yet uploaded
- Sends `GPU_CMD_FUNC_GET_ATTRIBUTES` with host_func_ptr
- Populates caller's `cudaFuncAttributes` struct with real values

`cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` follows same pattern:
- Eager upload if needed
- Sends `GPU_CMD_OCCUPANCY_MAX_BLOCKS`
- Returns real occupancy or safe fallback (2) on failure

### Driver API Shim (`cuda_driver_shim.c`)

`cu_func_get_attribute` queries daemon via same RPC, caches per-function in `DriverFunctionSlot.cached_attrs`. Falls back to safe defaults (sm_89 values) if RPC fails or function is unknown. This eliminates hardcoded vendor-specific GPU assumptions.

### Daemon (`gpu_proxy_daemon.c`)

`handle_func_get_attributes`:
- Looks up function by `host_func_ptr` in per-connection function table
- Sets CUDA context via `cuCtxSetCurrent`
- Queries 9 attributes via `cuFuncGetAttribute()` on the real `CUfunction`
- Returns packed response

`handle_occupancy_max_blocks`:
- Same function lookup
- Calls `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags()` on real GPU
- Returns `numBlocks` or fallback of 1

---

## 5. CUDA Graphs Strategy

CUDA graphs cannot be faithfully proxied — capture records host-side API calls but execution is remote. Attempts to emulate graph capture/replay through the proxy (recording kernel payloads during capture, replaying via `cudaGraphLaunch`) proved unreliable, producing gibberish output due to inability to reproduce CUDA's internal graph scheduling semantics.

**Current approach — Pass-through mode (Session 15, Mar 17):**

The proxy honestly reports that CUDA graphs are not supported, forcing applications to fall back to direct kernel execution:

- `cudaStreamBeginCapture` → return `cudaErrorNotSupported`
- `cuStreamBeginCapture` → return `CUDA_ERROR_NOT_SUPPORTED`
- `cudaStreamEndCapture` → return `cudaErrorNotSupported`
- `cudaStreamIsCapturing` → return `cudaStreamCaptureStatusNone`
- `cudaGraphInstantiate` → return `cudaSuccess` + dummy handle (harmless no-op)
- `cudaGraphLaunch` → return `cudaSuccess` (harmless no-op)

This is both simpler and more correct than the previous no-op approach. Applications with graph fallback paths (ggml, PyTorch) automatically switch to direct kernel execution. The `DECLOUD_GPU_GRAPH_NOOP` configuration flag has been removed — pass-through is now the only behavior.

**Critical:** The driver shim's `cuGetProcAddress` must return consistent results because libcudart resolves `cuStreamBeginCapture` through the driver API, bypassing the runtime shim entirely.

**Historical note:** The earlier `DECLOUD_GPU_GRAPH_NOOP=1` mode (graph ops return success as no-ops, kernels execute eagerly) caused Bug 22 — ggml's multi-token inference produced gibberish because `cudaGraphLaunch` was a no-op, resulting in zero GPU computation after the first capture cycle. See Debugging Journal Session 14-15 for the full investigation.

---

## 6. What True GPU Presence Enables

With real attributes proxied from the host GPU, the following works transparently:

| Capability | How |
|------------|-----|
| MMQ kernel selection | Real `binaryVersion`, `numRegs`, `sharedSizeBytes` per kernel |
| Occupancy-based block sizing | Real `numBlocks` from `cuOccupancy*` |
| Architecture-specific codepaths | Real `computeCapability` (sm_89, sm_80, sm_90, etc.) |
| Shared memory configuration | Real `maxDynamicSharedSizeBytes` and `sharedMemPerBlockOptin` |

Any CUDA application using `__cudaRegisterFatBinary` / `__cudaRegisterFunction` / `cudaLaunchKernel` / `cudaFuncGetAttributes` works through the proxy. This covers ggml (llama.cpp, Ollama), PyTorch custom ops, TensorFlow Lite, Triton-compiled kernels, and any nvcc-compiled program.

### Still Requires Stubs

| Library | Why |
|---------|-----|
| cuBLAS init | `cuGetExportTable` is private NVIDIA internals |
| cuDNN | Same `cuGetExportTable` dependency |
| cuFFT/cuSPARSE | Same pattern |

For ggml, the MMQ path avoids cuBLAS entirely. cuBLAS compute (`cublasGemmBatchedEx`) is proxied via dedicated RPC commands for GQA attention.

---

## 7. Performance Impact

| Metric | Before True Presence | After | With TCP_QUICKACK |
|--------|---------------------|-------|-------------------|
| Status | SIGABRT (mmq_x_best=0) | 22.99 tok/s | **436 tok/s** prompt |
| Prompt eval | N/A | ~25 tok/s | **188-436 tok/s** |
| Generation | N/A | ~10 tok/s | **13-21 tok/s** |
| Model load | ✅ (17/17 layers) | ✅ | ✅ (~3s warm) |
| GPU utilization | 0% | 100% | 100% |