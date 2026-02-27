# CUDA Runtime Shim — Production Fixes (2026-02-27)

## Summary

These changes transform `cuda_shim.c` from a working-but-blocking shim into a production shim that passes Ollama v0.17's GPU discovery, model loading, and memory allocation stages. The key breakthrough was implementing **lazy registration** to avoid RPC during `dlopen()` constructors.

## Changes vs Repository Version

### 1. Lazy Registration Pattern *(CRITICAL — fixes bootstrap timeout)*

**Problem:** `__cudaRegisterFatBinary` and `__cudaRegisterFunction` performed synchronous RPC during `dlopen()` constructors. With hundreds of flash_attn kernels in `libggml-cuda.so`, the cumulative RPC overhead (each fatbin is ~10MB) caused Ollama's bootstrap subprocess to exceed its ~180ms timeout and receive SIGKILL before ever calling `cudaGetDeviceCount`.

**Root Cause Evidence:**
```
strace PID 30101 (cuda_v12 bootstrap):
  - Loads libggml-cuda.so ✅
  - Loads libcudart.so.12 (our shim) ✅
  - NO connect() syscalls before SIGKILL ❌
```

**Fix — store locally during dlopen, defer RPC to first kernel launch:**

- `__cudaRegisterFatBinary` — stores fatbin pointer locally in `DeferredModule g_modules[]`, **zero RPC calls**
- `__cudaRegisterFunction` — stores function name + host pointer locally in `RegisteredFunction`, **zero RPC calls**
- `__cudaUnregisterFatBinary` — no-op (was doing RPC to unregister module)
- **New** `ensure_module_uploaded(mod_idx)` — lazily uploads fatbin + registers all functions for a module on first use
- `find_registered_function()` — now finds unregistered functions too (needed for lazy path)
- `cudaLaunchKernel` — calls `ensure_module_uploaded()` if function not yet registered

**Result:** Bootstrap completes in ~500ms (down from timeout at 30s), `initial_count=1`.

### 2. Raw Byte Offset Writes in `cudaGetDeviceProperties` *(CRITICAL — fixes compute=0.0)*

**Problem:** Our simplified `struct cudaDeviceProp` places `major` at byte offset 328, but ggml-cuda reads major at offset **360** (real NVIDIA struct v2 layout with many texture/surface fields between `clockRate` and `major`). Ollama saw `compute=0.0` and filtered the GPU out with "filtering device which didn't fully initialize".

**Discovery Method:**
```python
# Scanning for known values in the prop buffer
prop = ctypes.create_string_buffer(1096)
lib.cudaGetDeviceProperties(prop, 0)
# Found: major=8 at offset 328, minor=9 at offset 332
# ggml-cuda expects: major at 360, minor at 364
```

**Fix:** After filling struct fields normally, also write at raw binary offsets that ggml-cuda actually reads:
```c
uint8_t *raw = (uint8_t *)prop;
*(int *)(raw + 360) = resp.major;                        // compute capability major
*(int *)(raw + 364) = resp.minor;                        // compute capability minor
*(int *)(raw + 356) = resp.multi_processor_count;        // SM count
*(int *)(raw + 368) = resp.max_threads_per_multiprocessor;
```

**Result:** `compute=8.9` correctly reported, GPU passes verification.

### 3. `posix_memalign` for `cudaHostAlloc` *(fixes TENSOR_ALIGNMENT assertion)*

**Problem:** `cudaHostAlloc` used `malloc()` which returns 16-byte aligned memory. ggml asserts `(uintptr_t)ptr % TENSOR_ALIGNMENT == 0` where TENSOR_ALIGNMENT is 32+ bytes, causing:
```
GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned") failed
```

**Fix:** `posix_memalign(pHost, 4096, size)` for page-aligned allocation.

### 4. Symbol Version Script *(fixes dlopen symbol resolution)*

**Problem:** Initial version script used `CUDART_12.0` tag, but `libggml-cuda.so` requires symbols versioned as `@@libcudart.so.12`:
```bash
nm -D libggml-cuda.so | grep cudaGetDeviceProperties
# Output: cudaGetDeviceProperties@@libcudart.so.12
```

**Fix:** Version script:
```
libcudart.so.12 {
    global: cuda*; __cuda*;
    local: *;
};
```

### 5. Expanded Capacity

- `MAX_REGISTERED_FUNCTIONS`: 512 → **2048** (flash_attn has hundreds of kernels)
- `MAX_DEFERRED_MODULES`: new, set to **64**

### 6. `RegisteredFunction` struct expanded

Added fields for lazy registration:
```c
typedef struct {
    uint64_t host_func_ptr;
    uint32_t num_params;
    uint32_t param_sizes[GPU_MAX_KERNEL_PARAMS];
    int      registered;
    char     device_name[256];   // NEW — stored for deferred RPC
    int      module_index;       // NEW — index into g_modules[]
} RegisteredFunction;
```

### 7. `DeferredModule` struct added

```c
typedef struct {
    const void *fatbin_data;     // pointer into process memory
    size_t      fatbin_size;
    uint64_t    remote_handle;   // daemon module handle after upload
    int         uploaded;        // 1 = sent to daemon
} DeferredModule;
```

### 8. Windows Line Ending Fix

Server file had `\r\n` (CR+LF) line endings which silently broke all string-matching patch attempts. Fixed with `sed -i 's/\r$//'`.

## Test Results on srv022010 (Ollama v0.17 + RTX 4060 Laptop GPU)

| Metric | Result |
|---|---|
| `initial_count` | **1** ✅ |
| Compute capability | **8.9** ✅ |
| Device name | NVIDIA GeForce RTX 4060 Laptop GPU ✅ |
| Total VRAM | 8.0 GiB ✅ |
| Available VRAM | 6.9 GiB ✅ |
| Inference compute | **GPU (CUDA)** ✅ |
| Model layers to GPU | 23/23 ✅ |
| Bootstrap time | ~500ms ✅ (was timeout at 30s) |
| cudaMalloc alignment | 256-byte ✅ |
| cudaHostAlloc alignment | 4096-byte ✅ |
| Lazy registration (0 RPC in dlopen) | ✅ |

## Remaining Issue — Driver API Memory Allocation

Model loads to GPU successfully, but inference crashes with "CUDA error" because **libcublas** allocates memory through the **CUDA Driver API** (`cuMemAlloc`), not the Runtime API (`cudaMalloc`). The driver shim's `cuGetProcAddress` returns generic stubs for all `cuMem*` functions:

```
cuGetProcAddress("cuMemAlloc", v3020) → 0x7fd344b42490 [stub]
cuGetProcAddress("cuMemAllocManaged", v6000) → 0x7fd344b42490 [stub]
```

**Next step:** Implement `cuMemAlloc`, `cuMemFree`, `cuMemcpy` forwarding in `cuda_driver_shim.c` through the GPU proxy daemon, replacing the generic stubs with real RPC functions that `cuGetProcAddress` can return.

## Files

- `cuda_shim.c` — Production CUDA Runtime API shim (1340 lines)
- `libcudart.version` — Symbol version script
- `gpu_proxy_proto.h` — Wire protocol header (unchanged from repo)