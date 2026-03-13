# DeCloud GPU Proxy — Debugging Journal

**Date Range:** 2026-02-27 through 2026-03-13
**Authors:** BMA + Claude AI assistant
**Status:** Active — PyTorch inference + training + LoRA confirmed production-ready

---

## Session Timeline

| Session | Date | Key Achievement |
|---------|------|-----------------|
| 1-2 | Feb 27-28 | Foundation: transport, symbol versioning, GLIBC compat, cuGetExportTable wall |
| 3 | Mar 1 | Breakthrough: constructor setenv, cuBLAS stubs, DT_NEEDED fix, graph no-ops |
| 4 | Mar 1 | cuBLAS version tag fix (`@@libcublas.so.12`), GPU discovery through systemd |
| 5 | Mar 1-2 | True GPU Presence: eager module upload, real `cudaFuncGetAttributes` via RPC |
| 6 | Mar 2 | Shared memory struct fix, warpSize SIGFPE fix, cudaDeviceProp layout mapping |
| 7 | Mar 2-3 | cuBLAS GEMM batched proxy design, 1.56GB streaming module upload, module overflow fix |
| 8 | Mar 3-4 | GEMM batched D2H pointer fix, stream sync, **first successful GPU inference (22.99 tok/s)** |
| 9 | Mar 4-5 | TCP_NODELAY in wrong branch, stale 9p binaries, install.sh sync fixes |
| 10 | Mar 5-6 | **TCP_QUICKACK fix (0.07→436 tok/s)**, generic proxy refactor, production deployment |
| 11 | Mar 6-13 | PyTorch CUDA 12 compatibility, Bug 17b/17c SIGFPE chain, **PyTorch inference confirmed** |
| 12 | Mar 13 | **PyTorch full training + LoRA fine-tuning confirmed**; cublasLt debug gate |

---

## Sessions 1-4: Foundation (Feb 27 — Mar 1)

### Problems 1-13 (Sessions 1-2)
Transport initialization, vsock configuration, GLIBC 2.38 compatibility, symbol versioning, LD_PRELOAD mechanics, CUDA graph API conflicts, cuGetExportTable wall.

### Problem 14: Env Vars Not Reaching Runner (Session 3)
Ollama's runner subprocess strips non-whitelisted env vars. Fix: `__attribute__((constructor))` in `cuda_shim.c` calls `setenv()` before `main()`.

### Problem 15: DT_NEEDED Defeating LD_PRELOAD (Session 3) ⭐
Ollama bundles real NVIDIA libs alongside `libggml-cuda.so`. DT_NEEDED resolves from same directory BEFORE LD_PRELOAD. Fix: replace bundled libs with stubs in-place.

### Problem 16: CUDA Graph Error 801 (Session 3)
`cudaStreamBeginCapture` returning NOT_SUPPORTED crashes ggml's CUDA_CHECK. Fix: return `cudaSuccess` for all graph stubs.

### Problem 17: cuBLAS Version Tag Mismatch (Session 4) ⭐⭐
Same shim binary copied as `libcublas.so.12` had `@@libcudart.so.12` tags — silent dlopen failure. Fix: separate `cublas_stub.c` with its own version script.

### Problem 18: mmq_x_best=0 Crash (Session 4)
Fake `cudaFuncGetAttributes` returns identical values for all kernel variants → MMQ selector can't choose → SIGABRT. Root cause identified, fix designed → Session 5.

---

## Session 5: True GPU Presence (Mar 1-2)

### Problem 19: Eager Module Upload
Fat binaries stored locally but never uploaded until `cudaLaunchKernel`. ggml queries attributes before any launch. Fix: trigger `ensure_module_uploaded()` from `cudaFuncGetAttributes`.

### Problem 20: Real Kernel Attributes
Added `GPU_CMD_FUNC_GET_ATTRIBUTES` (0x54) and `GPU_CMD_OCCUPANCY_MAX_BLOCKS` (0x55). Daemon queries real GPU via `cuFuncGetAttribute()` and `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags()`.

---

## Session 6: Struct Layout Fixes (Mar 2)

### Problem 21: Shared Memory Fields Zero
`cudaDeviceProp` struct missing `sharedMemPerBlockOptin` and `sharedMemPerMultiprocessor` at correct offsets. MMQ selector needs non-zero shared memory to select kernels.

### Problem 22: warpSize=0 SIGFPE
`cudaDeviceProp` fields at wrong offsets caused `warpSize=0` → divide-by-zero. Fixed by mapping all critical fields to real NVIDIA CUDA 12 struct layout via offset probing.

---

## Session 7: Streaming Upload + Module Overflow (Mar 2-3)

### Problem 23: 1.56GB Fatbin Exceeds Payload Limit
`libggml-cuda.so` embeds a 1.56GB fatbin. Original 64MB max payload caused silent failures. Fix: increased to 2GB, implemented streaming upload directly from mmap'd memory (zero copy, zero malloc).

### Problem 24: Module Slot Overflow
`MAX_DEFERRED_MODULES=64` overflowed — 6244 functions misassigned to module 63 → `cuModuleGetFunction NOT_FOUND`. Increased to 256 modules, 8192 function slots.

### Problem 25: CUDA Context Mismatch
`cuCtxCreate` creates a new context while runtime API uses primary context. Kernel launches from runtime API failed with `CUDA_ERROR_INVALID_HANDLE`. Fix: switched daemon to `cuDevicePrimaryCtxRetain`.

---

## Session 8: cuBLAS GEMM + First Inference (Mar 3-4) ⭐⭐⭐

### Problem 26: cuBLAS Stub Targeting Wrong Function
Patch script modified `cublasGemmEx` instead of `cublasGemmBatchedEx`. The SEGFAULT persisted because the right function still dereferenced VM device pointers.

### Problem 27: D2H Pointer Array Transfer
`cublasGemmBatchedEx` receives device pointer arrays (`Aarray`, `Barray`, `Carray`). The VM cannot dereference these — they're device memory addresses. Fix: daemon reads pointer arrays via `cudaMemcpy D2H` before calling real cuBLAS.

### Problem 28: Stream Sync Race
cuBLAS results not ready when shim reads response. Fix: `cudaStreamSynchronize` before D2H read in daemon's GEMM handler.

### Result: First Successful GPU Inference
```
kernels=4344, ktime=571962µs
eval rate: 22.99 tok/s
```

---

## Session 9: TCP_NODELAY + Install.sh (Mar 4-5)

### Problem 29: Performance Regression to 0.07 tok/s
After Session 8's success, performance dropped to 0.07 tok/s. One fast run (23 tok/s) occurred but never repeated.

### Problem 30: TCP_NODELAY in Error Branch ⭐
`transport.c` had `setsockopt(TCP_NODELAY)` inside the `connect() < 0` error path — set on a closed fd, dead code. Moved to success path.

### Problem 31: Stale Binaries in 9p Share
`install.sh` updated `libdecloud_cuda_shim.so` but never synced `libcudart.so.12`. VMs copied the Mar 1 binary without TCP_NODELAY. Fix: `cp libdecloud_cuda_shim.so libcudart.so.12` after every build.

### Problem 32: install.sh Daemon Lifecycle
Old daemon binaries kept running after `install.sh`. Fix: kill stale processes before replacing binary, restart with captured args.

---

## Session 10: TCP_QUICKACK + Generic Proxy (Mar 5-6) ⭐⭐⭐

### Problem 33: TCP Delayed ACK (THE Root Cause) ⭐⭐⭐
strace revealed every small RPC took exactly **44ms** — matching `ato:40` from `ss` output. `TCP_NODELAY` only disables Nagle (sender buffering). **TCP_QUICKACK** disables delayed ACK (receiver-side 40ms timer). Must be re-armed before every `read()` because Linux resets it per-operation.

**Before:** 0.07 tok/s (44ms per RPC)
**After:** 436 tok/s prompt eval, 13-21 tok/s generation

### Generic Proxy Refactor (applied incrementally, tested after each step)

**Priority 1a — Config-driven constructor:**
Constructor reads `/etc/decloud/gpu-proxy.env` instead of hardcoding `GGML_*` setenv calls. Any non-transport, non-`DECLOUD_*` keys propagated via `setenv()`.

**Priority 1b+5 — Configurable graph stubs:**
`g_graph_noop` flag (default 1) controls graph stub return values. Driver shim reads flag in `__attribute__((constructor))` BEFORE `cuGetProcAddress` is called. Key insight: defaulting to 1 (safe) prevents breakage even if config file reading fails.

**Priority 2 — Real GPU attributes via RPC:**
`cu_func_get_attribute` in driver shim queries daemon, caches per-function in `DriverFunctionSlot`, falls back to safe defaults.

**Priority 3 — Virtual memory API scaffolding:**
8 new command IDs (0x70-0x77), request/response structs, shim-side proxying gated by `DECLOUD_GPU_VMEM_PROXY=0` (default off).

**Orchestrator changes:**
`EnsureGpuProxyShim` accepts `gpuEnvVars` parameter, extracts GPU vars from template `DefaultEnvironmentVariables`, writes to env file.

---

## Key Lessons Learned

1. **TCP_QUICKACK is as important as TCP_NODELAY** for request-response protocols. Delayed ACK adds 40ms per small response.

2. **cuGetProcAddress is the real dispatch** in modern CUDA. Runtime shim functions may never be called if libcudart resolves via the driver API.

3. **Constructor init order matters.** Driver shim's `cuGetProcAddress` is called before `cuInit`. Config must be loaded in `__attribute__((constructor))`.

4. **Default to safe behavior.** `g_graph_noop=1` by default means graph stubs work even without config. Explicit opt-out (`=0`) for apps that want honest errors.

5. **ELF version tags are symbol identity.** `func@@libcublas.so.12` ≠ `func@@libcudart.so.12`. Separate builds needed for each soname.

6. **Never trust file timestamps in 9p shares.** Always verify with `objdump` or `cmp` that the binary contains expected code.

7. **Streaming upload is essential** for large fatbins. `malloc(1.5GB)` fails; `write_exact` from mmap succeeds.

8. **Test incrementally.** The generic proxy refactor broke when applied all at once but succeeded when applied one step at a time with tests after each.
---

## Session 11: PyTorch CUDA 12 Compatibility (Mar 6-13) ⭐⭐

### Bug 17b: `cuOccupancyMaxActiveBlocksPerMultiprocessor` Invisible to PLT

**Symptom:** PyTorch / GPT-2 `torch.multinomial` → SIGFPE crash in `mbtopk::get_items_per_thread`. `objdump` and `nm -D` confirmed the function existed in the dispatch table but was `static`, giving it type `t` (local) rather than `T` (global exported).

**Root cause:** `static` qualifier in `cuda_driver_shim.c` makes the function invisible to PLT symbol resolution — the dynamic linker can only bind to exported (`T`) symbols. PyTorch's `cuOccupancyMaxActiveBlocksPerMultiprocessor` call resolved to a zero-returning stub instead.

**Fix in `src/gpu-proxy/shim/cuda_driver_shim.c`:**
- Added non-static exported wrapper `cuOccupancyMaxActiveBlocksPerMultiprocessor` before the destructor block
- Verified with `nm -D build/libcuda.so.1 | grep -i occupancy` → type `T`

### Bug 17c: `maxThreadsPerMultiProcessor` at Wrong `cudaDeviceProp` Offset ⭐

**Symptom:** After Bug 17b fix, SIGFPE persisted. `mbtopk::get_items_per_thread` at `0x288(%rax)` read value 0, then performed integer divide → `idiv %rcx` with rcx=0 → SIGFPE.

**Root cause chain:**
- `maxThreadsPerMultiProcessor` was being written to `raw+648` (0x288)
- Real CUDA 12 offset for `maxThreadsPerMultiProcessor` is `raw+624` (0x270)
- Offset 0x288 is `regsPerMultiprocessor` (should be 65536), not `maxThreadsPerMultiProcessor` (1536)
- PyTorch's `mbtopk` kernel reads 0x288 expecting `regsPerMultiprocessor`, applies `/ 10240` trick → 6 (correct)
- But it was reading 0 (miswritten `maxThreadsPerMultiProcessor`) → divide by zero

**Fix in `src/gpu-proxy/shim/cuda_shim.c`:**
```c
*(int *)(raw + 624) = resp.max_threads_per_multiprocessor;  // 0x270 — correct CUDA 12 offset
*(int *)(raw + 648) = 65536;                                // 0x288 — regsPerMultiprocessor (constant SM3.0+)
```

**Confirmed CUDA 12.1 / PyTorch 2.3.1 offset map for RTX 4060 (SM8.9):**

| Offset | Field | Value |
|--------|-------|-------|
| 0x184 (388) | multiProcessorCount | 24 |
| 0x270 (624) | maxThreadsPerMultiProcessor | 1536 |
| 0x288 (648) | regsPerMultiprocessor | 65536 |
| 0x2c8 (712) | maxBlocksPerMultiProcessor | 1024 |
| 0x2d0 (720) | reservedSharedMemPerBlock | 65536 |

**Result: PyTorch inference confirmed passing all 4 test steps** including `generate(do_sample=True, max_new_tokens=20)`.

### JupyterLab Kernel GPU Access Confirmed

LD_PRELOAD injected via `/etc/decloud/gpu-proxy.env` as `EnvironmentFile=` in the JupyterLab systemd unit. Kernels inherit the full proxy environment automatically — no per-notebook configuration needed.

```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU (7GB)
GPT-2 generated text: confirmed ✅
```

---

## Session 12: PyTorch Training + LoRA Confirmed (Mar 13) ⭐⭐⭐

### PyTorch Full Fine-Tuning Confirmed

**Test:** GPT-2 small (125M params), AdamW, batch=4, seq=128, 3 training steps.

```
step 0 OK, loss: 13.0006
step 1 OK, loss: 11.8199
step 2 OK, loss: 11.2646
TRAINING OK
```

All gradient and optimizer CUDA kernels proxied correctly:
- Autograd graph construction ✅
- Backward pass through all 12 transformer layers ✅
- Adam moment accumulation kernels ✅
- Weight update kernels ✅
- Loss decreasing monotonically ✅

**No cuDNN required** — GPT-2 uses scaled dot-product attention via native CUDA kernels, not cuDNN. This means any transformer model using the same attention path works without cuDNN proxy.

### PyTorch Training Benchmark (RTX 4060 Laptop GPU)

```
10 steps in 4.09s = 409ms/step
Throughput: 1252 tokens/sec
```

**Analysis:** ~50% of bare metal (estimated 2,500-3,000 tok/s native). The gap vs inference comes from training having far more small kernel launches per step — each pays the RPC round-trip cost. In contrast, inference hot path is dominated by cublasLtMatmul RPC which is already well-optimized.

### LoRA Fine-Tuning via PEFT Confirmed ⭐

**Test:** GPT-2 + LoRA (r=8, `target_modules=["c_attn","c_proj"]`, 0.65% trainable params), AdamW, batch=4, seq=128, 20 steps.

```
Total params:     125,250,816
Trainable params: 811,008  (0.65%)

Steps:          20
Batch x Seq:    4 x 128 = 512 tokens/step
Total time:     9.86s
ms / step:      493 ms
Throughput:     1038 tokens/sec
Peak VRAM:      1360 MB
Loss (start):   12.1723
Loss (end):     11.0411
Loss delta:     1.1312  (OK decreasing)
LORA BENCHMARK COMPLETE
```

**Key findings:**
- LoRA is slightly *slower* per step than full fine-tune in proxy mode — RPC overhead dominates over fewer Adam ops
- VRAM drops dramatically: 1,360MB vs ~3,500MB for full fine-tune — the real LoRA win
- 1,360MB VRAM means Llama-3.2-1B LoRA (~2.5GB) and Llama-3.2-3B LoRA (~5GB) are feasible on 8GB nodes
- Loss decreasing confirms all gradient + optimizer paths work correctly

### Bug: cublasLt Unconditional `fprintf(stderr,...)` Logging

**Symptom:** Every inference/training step produced hundreds of lines:
```
[cublasLt-stub] cublasLtMatmul: Arows=2304 Acols=768 ...
```

**Root cause:** `STUB_LOG` macro in `stubs/cublasLt_stub.c` was an unconditional `fprintf(stderr,...)`.

**Fix (applied, rebuild pending for new VM deploy):**
```c
static int g_lt_debug = -1;
static inline int lt_debug(void) {
    if (g_lt_debug < 0) g_lt_debug = (getenv("DECLOUD_GPU_DEBUG") != NULL);
    return g_lt_debug;
}
#define STUB_LOG(fmt, ...) \
    do { if (lt_debug()) fprintf(stderr, "[cublasLt-stub] " fmt "\n", ##__VA_ARGS__); } while(0)
```

**Behavior after fix:** Silent in normal operation. `DECLOUD_GPU_DEBUG=1` re-enables all stub logging for debugging.

---

## Key Lessons Learned (Sessions 11-12)

9. **CUDA 12 lazy loading bypasses `__cudaRegisterFunction` entirely.** `CUDA_MODULE_LOADING=EAGER` is mandatory for PyTorch. Without it, module registration is deferred to `__cudaInitModule` which the proxy doesn't intercept.

10. **`static` functions in dispatch tables are PLT-invisible.** Any function that PyTorch resolves via the dynamic linker must be a non-static exported symbol (`T` in `nm -D`). This bit occupancy and will bite any future function added with `static`.

11. **`cudaDeviceProp` offsets are CUDA-version-specific.** The struct layout changed between CUDA versions. The offset map must be verified against the actual PyTorch CUDA wheel (2.3.1+cu121 = CUDA 12.1 ABI). Wrong offsets cause silent wrong values, not crashes — making them hard to find.

12. **LoRA VRAM savings are real; LoRA speed savings are not in proxy mode.** The RPC overhead dominates over fewer Adam states. Position LoRA as a VRAM efficiency tool, not a speed tool, when pitching on DeCloud.

13. **Training support follows directly from good inference support.** Once autograd backward + optimizer steps worked, training "just worked" — no new protocol commands needed. The proxy is genuinely general-purpose.

---

## Production Status (2026-03-13)

| Workload | Status | Performance |
|----------|--------|-------------|
| Ollama / ggml inference | ✅ Production | 436 tok/s prompt, 13-21 tok/s gen |
| PyTorch inference | ✅ Confirmed | All ops including sampling |
| PyTorch full fine-tuning | ✅ Confirmed | 1,252 tok/s, 409ms/step |
| PyTorch LoRA (PEFT) | ✅ Confirmed | 1,038 tok/s, 493ms/step, 1,360MB VRAM |
| JupyterLab kernel | ✅ Confirmed | Via systemd EnvironmentFile |
| Stable Diffusion Forge | ⚠️ Partial | UI+model OK; cuBLAS backward pending |

## Pending Items

| Item | Priority | Notes |
|------|----------|-------|
| cublasLt `DECLOUD_GPU_DEBUG` gate rebuild + deploy | High | Fix written, needs `make all-shims-compat` + new VM |
| Stable Diffusion WebUI Forge — cuBLAS backward | High | Next debug target after cublasLt fix deployed |
| `cuGetExportTable` stub | Medium | Unblocks cuBLAS init path, estimated 1-2 days |
| Llama-3.2-1B LoRA benchmark | Medium | Validates real-world fine-tuning target |
| `.orig` guard content-check fix (Bug 9) | Low | Prevents re-replacement on redeploy |
| GPU_PROXY_DEBUGGING_JOURNAL.md sync | ✅ Done | This update |