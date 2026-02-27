# CUDA Runtime Shim (cuda_shim.c) — Production Fixes from 2026-02-27

## Summary
These changes transform the cuda_shim.c from a working-but-blocking shim into a
production shim that passes Ollama v0.17's GPU discovery, model loading, and
memory allocation stages.

## Changes vs Repository Version

### 1. Lazy Registration Pattern (CRITICAL — fixes bootstrap timeout)
**Problem:** `__cudaRegisterFatBinary` and `__cudaRegisterFunction` performed
synchronous RPC during `dlopen()` constructors. With hundreds of flash_attn
kernels in libggml-cuda.so, the cumulative RPC overhead caused Ollama's
bootstrap subprocess to exceed its ~180ms timeout (SIGKILL).

**Fix:**
- `__cudaRegisterFatBinary` — stores fatbin pointer locally in `DeferredModule g_modules[]`, NO RPC
- `__cudaRegisterFunction` — stores function name + host pointer locally, NO RPC
- `__cudaUnregisterFatBinary` — no-op (was doing RPC to unregister)
- New `ensure_module_uploaded()` — lazily uploads fatbin + registers all functions on first `cudaLaunchKernel` call
- `find_registered_function()` — now finds unregistered functions (needed for lazy path)
- `cudaLaunchKernel` — calls `ensure_module_uploaded()` if function not yet registered

### 2. Raw Byte Offset Writes in cudaGetDeviceProperties (CRITICAL — fixes compute=0.0)
**Problem:** Our simplified `struct cudaDeviceProp` places `major` at offset 328,
but ggml-cuda reads major at offset 360 (real NVIDIA struct v2 layout). Ollama
saw `compute=0.0` and filtered the GPU out.

**Fix:** After filling struct fields normally, also write at raw binary offsets:
```c
### 3. posix_memalign for cudaHostAlloc (fixes TENSOR_ALIGNMENT assertion)
**Problem:** `cudaHostAlloc` used `malloc()` which returns 16-byte aligned
memory. ggml asserts `(uintptr_t)ptr % TENSOR_ALIGNMENT == 0` where
TENSOR_ALIGNMENT may be 32 or 64 bytes.

**Fix:** Use `posix_memalign(pHost, 4096, size)` for page-aligned allocation.

### 4. Symbol Version Script (fixes dlopen failure)
**Problem:** Initial version script used `CUDART_12.0` tag, but libggml-cuda.so
expects symbols versioned as `@@libcudart.so.12`.

**Fix:** Version script uses `libcudart.so.12 { global: cuda*; __cuda*; local: *; };`

### 5. Expanded MAX_REGISTERED_FUNCTIONS
Changed from 512 to 2048 to handle the large number of flash_attn kernels.

### 6. RegisteredFunction struct expanded
Added `char device_name[256]` and `int module_index` for lazy registration.

### 7. DeferredModule struct added
New struct for lazy fatbin upload: `fatbin_data`, `fatbin_size`, `remote_handle`, `uploaded`.

## Test Results
- `initial_count=1` ✅
- `compute=8.9` ✅ (RTX 4060 Laptop GPU)
- `total="8.0 GiB" available="6.9 GiB"` ✅
- `inference compute: GPU (CUDA)` ✅
- Model loads 23/23 layers to GPU ✅
- cudaMalloc returns 256-byte aligned GPU pointers ✅
- cudaHostAlloc returns 4096-byte aligned host pointers ✅

## Remaining Issue
- Driver shim's `cuGetProcAddress` returns generic stubs for `cuMemAlloc` etc.
  libcublas uses driver API for memory, causing "CUDA error" at inference time.
  Next step: implement cuMemAlloc/cuMemFree forwarding in cuda_driver_shim.c.
