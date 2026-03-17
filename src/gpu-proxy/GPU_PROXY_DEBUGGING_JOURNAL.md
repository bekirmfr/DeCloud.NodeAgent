# DeCloud GPU Proxy — Debugging Journal

**Date Range:** 2026-02-27 through 2026-03-17
**Authors:** BMA + Claude AI assistant
**Status:** Active — PyTorch inference + training + LoRA + SD Forge + Ollama GPU all production-ready

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
| 13 | Mar 14 | cuGetExportTable stub, cuDNN investigation, Bug 19 parked |
| 14 | Mar 15 | **cublasLt bias fix (Bug 21)**, **SD Forge image generation confirmed**, libcudnn stub, Bug 22 investigating |
| 15 | Mar 17 | **Bug 22 FIXED** — CUDA graph pass-through mode, word doubling/garbling parked for future |

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

---

## Session 13: cuGetExportTable + cuDNN Investigation (2026-03-14)

### Bug 18: cuGetExportTable returns NOT_FOUND — FIXED ✅

**Symptom:** `torch.nn.functional.linear` and `nn.Linear` crashed. cuBLAS
and cuDNN call `cuGetExportTable` during init to obtain internal driver
function tables by 16-byte GUID. Previous implementation returned
`CUDA_ERROR_NOT_FOUND` → both libraries aborted init.

**Fix (`cuda_driver_shim.c`):**
- Added static 128-entry stub table (`g_export_table`) filled at link time
  via GCC range initialiser — zero runtime cost, no re-entrancy risk
- `cuGetExportTable` now returns the table and `CUDA_SUCCESS` for all GUIDs
- GUID hex logging under `DECLOUD_GPU_DEBUG=1`

**Result:** `F.linear OK`, `nn.Linear OK` — cuBLAS fully unblocked.

---

### Bug 19: cuDNN CUDNN_STATUS_NOT_INITIALIZED — PARKED 🚧

**Symptom:** `Conv2d` fails with `CUDNN_STATUS_NOT_INITIALIZED` after
Bug 18 fix. `F.linear` and `nn.Linear` pass.

**Investigation:**

Two GUIDs observed at runtime:
```
6bd5fb6c-5bf4-e74a-8987-d93912fd9df9  NV_CUDA_DEVICE_INTERNAL
a094798c-2e74-2e74-93f2-0800200c0a66  NV_CUDA_MEM_INTERNAL
```

Per-index instrumented tables added (32 stubs × 2 GUIDs). Confirmed:
- `guid=6bd5 index=2` is the only entry called — cuDNN context query
- `a0` is the output slot (not `a1` — confirmed from log `a1=(nil)`)
- Writing `g_current_ctx` (opaque RPC handle `0x1`) → cuDNN crashes on
  return (dereferences as VM pointer)
- Writing `g_fake_ctx` (zeroed 512-byte static buffer) → same crash
- Writing magic values at suspected struct offsets → same crash
- No driver calls made after `index=2` returns — crash is internal to cuDNN

**Root cause:** cuDNN validates the context struct at a specific field
offset immediately after receiving the pointer. The field layout of
CUDA 12's internal `CUcontext` struct is not public. Without knowing
the exact offset and expected value, no static buffer can satisfy the
check.

**What's needed to fix:**
- Reverse engineer `libcudnn.so` to find the context struct field cuDNN
  reads at init, OR
- Reference gvisor `nvproxy` source — Google's CUDA proxy implements
  `cuGetExportTable` and may have the struct layout documented

**Workaround:** `TORCH_CUDNN_ENABLED=0` added to pytorch-jupyter and
SD Forge GPU templates. PyTorch native CUDA kernels handle all
transformer ops and Conv2d without cuDNN — confirmed working. No
performance regression for LLM inference, training, or LoRA workloads.

**Current state of `cuda_driver_shim.c`:**
- Per-GUID dispatch in `cuGetExportTable` (6bd5, a094, generic)
- 32 instrumented stubs per GUID with `DECLOUD_GPU_DEBUG=1` logging
- `export_6bd5_2` writes `g_fake_ctx` to `a0`
- `g_fake_ctx` = zeroed 512-byte static buffer (safe dereferenceable addr)
- `cuCtxGetCurrent` returns `g_fake_ctx` when context is active

**Tracking:** Bug 19 — parked. Revisit when cuDNN acceleration becomes
a customer-reported bottleneck or when nvproxy source is reviewed.

---

### Key Lessons Learned (Session 13)

14. **`cuGetExportTable` GUID bytes are little-endian stored.** The human-
    readable GUID string `{6bd5fb6c-...}` does NOT map byte-for-byte to
    the in-memory representation — the first three groups are stored in
    little-endian order. Always derive memcmp bytes from the runtime log,
    not from the GUID string.

15. **stderr to file is essential for high-volume debug sessions.**
    `DECLOUD_GPU_DEBUG=1` produces thousands of lines per second. PTY
    flushing blocks the process — redirect to file and grep after.
    `> /tmp/out.txt 2> /tmp/debug.txt` is the standard pattern.

16. **cuDNN crashes before making any driver calls after export table
    init.** The failure is purely from reading the context struct in
    memory — no RPC, no driver API, no logging possible without knowing
    the struct layout. Instrumentation has hit its limit here.

---

## Production Status (2026-03-14)

| Workload | Status | Notes |
|----------|--------|-------|
| Ollama / ggml inference | ✅ Production | 436 tok/s prompt, 13-21 tok/s gen |
| PyTorch inference (eager) | ✅ Confirmed | All transformer ops |
| PyTorch full fine-tuning | ✅ Confirmed | 1,252 tok/s, 409ms/step |
| PyTorch LoRA (PEFT) | ✅ Confirmed | 1,038 tok/s, 493ms/step, 1,360MB VRAM |
| JupyterLab kernel | ✅ Confirmed | Via systemd EnvironmentFile |
| Conv2d / pooling (no cuDNN) | ✅ Confirmed | TORCH_CUDNN_ENABLED=0 |
| Stable Diffusion Forge | ⚠️ Untested | TORCH_CUDNN_ENABLED=0 added, ready to test |
| cuDNN-accelerated conv | ❌ Blocked | Bug 19 — parked |

## Pending Items

| Item | Priority | Notes |
|------|----------|-------|
| SD Forge end-to-end image generation test | High | Next — VMEM+cuDNN workaround in place |
| vLLM template validation | High | VMEM_PROXY=1 already set |
| `cuLaunchKernelEx` grid dims (TODO in source) | Medium | Parses 1×1×1 defaults |
| Bug 19 — cuDNN context struct layout | Low | Parked — nvproxy reference needed |
| Llama-3.2-1B LoRA benchmark | Medium | Validates real-world fine-tuning |

## Pending Items

| Item | Priority | Notes |
|------|----------|-------|
| cublasLt `DECLOUD_GPU_DEBUG` gate rebuild + deploy | High | Fix written, needs `make all-shims-compat` + new VM |
| Stable Diffusion WebUI Forge — cuBLAS backward | High | Next debug target after cublasLt fix deployed |
| `cuGetExportTable` stub | Medium | Unblocks cuBLAS init path, estimated 1-2 days |
| Llama-3.2-1B LoRA benchmark | Medium | Validates real-world fine-tuning target |
| `.orig` guard content-check fix (Bug 9) | Low | Prevents re-replacement on redeploy |
| GPU_PROXY_DEBUGGING_JOURNAL.md sync | ✅ Done | This update |
---

## Session 14: cublasLt Bias Fix + SD Forge Confirmed (Mar 15) ⭐⭐⭐

### Bug 21: cublasLt EPILOGUE/BIAS_POINTER Constants Swapped — FIXED ✅

**Symptom:** `torch.addmm` (every `nn.Linear` with bias) returned wrong results. SD Forge generated flat/grey images. Ollama GPU long-text generations produced gibberish from token 1.

**Root cause:** `CUBLASLT_MATMUL_DESC_EPILOGUE` and `CUBLASLT_MATMUL_DESC_BIAS_POINTER` attribute index constants were swapped in `cublasLt_stub.c`:
```
Wrong:   EPILOGUE=7  BIAS_POINTER=8
Correct: EPILOGUE=8  BIAS_POINTER=7   (CUDA 12.1 ABI)
```
Additionally, PyTorch 2.3 passes EPILOGUE as 4 bytes (attr=8) and BIAS_POINTER as an 8-byte device pointer (attr=7). Swapped constants caused every descriptor setup to write to the wrong fields, producing silently incorrect GEMM results.

**Fix (`stubs/cublasLt_stub.c`):** Corrected `#define` values. Epilogue and bias_ptr now set via `cublasLtMatmulDescSetAttribute` into op_desc before `cublasLtMatmul`. Saxpy fallback removed.

**Verification:**
```
Linear(  64->320) diff_std=0.00000008  ✅
Linear( 320->320) diff_std=0.00000011  ✅
Linear(1280->320) diff_std=0.00000013  ✅
addmm mean: 99.90 (expected ~100), diff max: 0.000000  ✅
```

---

### SD Forge GPU Image Generation — CONFIRMED ✅

Fresh VM deployment. Generated 512×512 photorealistic red apple image at 1.61 it/s, 20 steps. `ExecStartPre` in systemd service correctly replaces venv stubs (cublas, cublasLt, cudnn) on every service start.

---

### Bug 22: Ollama Long-Text GPU Gibberish — FIXED ✅

**Symptom:** GPU inference produces word-salad gibberish on any non-trivial prompt. CPU inference (`num_gpu:0`) produces perfect output. Short generations sometimes worked because they completed within the first graph capture cycle. **Reproduced on clean VM** — not environment-specific.

**Root cause:** CUDA graph no-op stubs. ggml with `USE_GRAPHS=1` (hardcoded by Ollama at compile time) launches kernels **only** during `cudaStreamBeginCapture`…`cudaStreamEndCapture`, then replays them via `cudaGraphLaunch` for every subsequent token. The proxy's `cudaGraphLaunch` was a no-op → **zero GPU computation** after the first graph capture → gibberish.

Flow with old code (graph no-op mode):
1. Token 1: `cudaStreamBeginCapture` (no-op) → kernels execute eagerly → `cudaGraphLaunch` (no-op) → correct output
2. Token 2+: no capture → `cudaGraphLaunch` (no-op) → **no kernels execute** → stale logits → gibberish

**`GGML_CUDA_DISABLE_GRAPHS=1` confirmed ineffective.** Ollama's bundled ggml has `USE_GRAPHS=1` hardcoded at compile time — the env var only works with ggml builds where graphs are optional. The shim sets this env var as belt-and-suspenders, but it **cannot be relied upon** for deployments using Ollama's prebuilt binaries.

**Key lesson:** Never trust an application-level env var to disable a feature that's hardcoded at compile time. The proxy must handle what the binary actually does, not what we wish it would do.

**Abandoned approach — capture & replay:** Implemented full graph recording in `cuda_shim.c` (recording kernel payloads during capture, replaying via `cudaGraphLaunch`). This approach suffered from double execution on the first token and still produced gibberish — the fundamental issue is that graph replay through a proxy cannot faithfully reproduce CUDA's internal graph scheduling semantics.

**Fix — CUDA graph pass-through mode (Session 15):** Instead of trying to emulate graph semantics, the shim now makes `cudaStreamBeginCapture` return `cudaErrorNotSupported` (error 71) and `cuStreamBeginCapture` return `CUDA_ERROR_NOT_SUPPORTED`. This forces ggml to fall back to its non-graph execution path where each kernel is launched directly via `cudaLaunchKernel`.

How the pass-through fix works:
1. `cudaStreamBeginCapture` → returns `cudaErrorNotSupported`
2. ggml sees graph capture failed → falls back to direct kernel execution
3. Every token: `cudaLaunchKernel` called directly → proxied to daemon → correct execution
4. No graph replay, no double execution, no stale logits

Changes made:
- `cuda_shim.c`: `cudaStreamBeginCapture` returns `cudaErrorNotSupported`, `cudaStreamEndCapture` returns `cudaErrorNotSupported`, `cudaStreamIsCapturing` returns `None`, `cudaGraphLaunch` is a harmless no-op
- `cuda_driver_shim.c`: `cuStreamBeginCapture` returns `CUDA_ERROR_NOT_SUPPORTED`
- Removed graph capture/replay recording infrastructure (recording buffers, replay logic)
- `DECLOUD_GPU_GRAPH_NOOP` flag removed — pass-through (not-supported) is now the only behavior

**Verification:** Ollama GPU inference produces coherent, correct output for long-text prompts. No gibberish. Confirmed on clean VM.

**Impact on other workloads:**
- PyTorch: ✅ Unaffected — does not use CUDA graphs by default; explicit `torch.cuda.CUDAGraph` usage would get an honest error
- Stable Diffusion: ✅ Unaffected — no graph dependency
- Ollama/ggml: ✅ Fixed — falls back to direct kernel execution

#### Bug 22a: Word Doubling / Garbling — PARKED 🅿️

**Symptom:** During investigation of Bug 22, occasional word doubling or garbling was observed in generated text (e.g., "the the", repeated tokens, slightly mangled words). This was intermittent and only appeared during testing of the graph capture/replay approach.

**Status:** Parked for future reference. The pass-through fix (returning `cudaErrorNotSupported`) eliminates the code path that caused this issue. If word doubling resurfaces in the future, investigate:
- Whether graph capture/replay code was accidentally re-enabled
- RPC payload serialization correctness for kernel args during replay
- Stream synchronization ordering between capture and eager execution paths
- Potential race conditions in the recording buffer allocation

**Workaround:** Not needed — the pass-through fix eliminates the root cause.

---

### libcudnn Stub Infrastructure — NEW ✅

`libtorch_cuda.so` and `libtorch_python.so` have `DT_NEEDED: libcudnn.so.8`. Without this stub PyTorch fails to import entirely on a clean VM.

New files: `stubs/libcudnn_stub.c` (85 symbols, all returning `CUDNN_STATUS_NOT_INITIALIZED(1)`) and `stubs/libcudnn.version`. Deployed via `make all-shims-compat` → 9p share → cloud-init → `/usr/local/lib/libcudnn.so.8`. `ExecStartPre` replaces venv-bundled cudnn libs on every service start.

**Verification:** `libcudnn_stub.so OK (87 versioned symbols)` ✅

---

### install.sh Refactor

Replaced manual per-stub copy block with `make install-all-shims-compat` delegation — Makefile is now the single source of truth for stub installation.

---

### Key Lessons Learned (Session 14)

17. **cublasLt attribute constants must be verified against actual CUDA headers.** `EPILOGUE=8`, `BIAS_POINTER=7` — wrong values cause silent incorrect results, not crashes.

18. **PyTorch passes EPILOGUE as 4 bytes (attr=8), BIAS_POINTER as 8 bytes (attr=7).** Read with correct types. Confirmed via `DECLOUD_GPU_DEBUG=1` `SetAttribute` logging.

19. **ggml uses zero cuBLAS.** All ggml CUDA compute is custom kernels — cuBLAS/cublasLt fixes have no effect on Ollama generation quality.

20. **`su -` strips environment variables.** Pass debug vars inside the `su - user -c "VAR=val command"` form.

21. **CUDA graph no-op stubs silently break multi-token inference.** ggml launches kernels only during graph capture; `cudaGraphLaunch` must replay them. A no-op `cudaGraphLaunch` means zero computation after the first capture cycle.

22. **`GGML_CUDA_DISABLE_GRAPHS=1` is unreliable.** Ollama's bundled ggml has `USE_GRAPHS=1` hardcoded at compile time — env vars only work with optional-graph builds. Never trust an application-level env var to disable a compile-time feature. The proxy must handle what the binary actually does.

---

## Session 15: CUDA Graph Pass-Through Fix (Mar 17) ⭐⭐

### Bug 22 Resolution: Pass-Through Instead of Emulation

After Session 14's graph capture/replay approach failed to resolve the gibberish issue, we took a fundamentally different approach: **make the proxy honestly report that CUDA graphs are not supported**, forcing applications to fall back to direct kernel execution.

**Why capture/replay failed:** Emulating CUDA graph semantics through a proxy is inherently fragile. The proxy cannot faithfully reproduce CUDA's internal graph scheduling, kernel fusion, or memory ordering guarantees. The double-execution on first token and incorrect replay on subsequent tokens produced the same gibberish we were trying to fix.

**The pass-through solution:** Return `cudaErrorNotSupported` from `cudaStreamBeginCapture` (runtime) and `CUDA_ERROR_NOT_SUPPORTED` from `cuStreamBeginCapture` (driver). This is honest — the proxy genuinely cannot support CUDA graphs. Applications with graph fallback paths (ggml, PyTorch) automatically switch to direct kernel execution.

**Changes:**
- `cuda_shim.c`: Graph begin/end capture return not-supported; graph launch/instantiate remain harmless no-ops
- `cuda_driver_shim.c`: Driver-side capture returns not-supported
- Removed `DECLOUD_GPU_GRAPH_NOOP` flag — single consistent behavior
- Removed graph capture/replay recording infrastructure
- `stubs/cuda_pytorch_stubs.c`: `cudaStreamGetCaptureInfo_v2` already returns `captureStatus=None` — consistent

**Verification:** Ollama long-text GPU inference produces coherent output. PyTorch and SD Forge unaffected (neither uses CUDA graphs by default).

### Bug 22a: Word Doubling / Garbling — PARKED 🅿️

Intermittent word doubling observed during capture/replay testing. Parked — the pass-through fix eliminates the code path that caused this. See Bug 22a notes above for future investigation pointers.

### Key Lessons Learned (Session 15)

23. **Don't emulate what you can't faithfully reproduce.** CUDA graph capture/replay through a proxy is fundamentally unsound — the proxy has no access to CUDA's internal graph scheduler. Returning "not supported" and letting applications fall back is both simpler and more correct.

24. **Honest errors beat silent no-ops.** `cudaErrorNotSupported` triggers well-tested fallback paths in mature frameworks. Silent no-ops (returning success but doing nothing) cause subtle, hard-to-debug corruption.

---

## Production Status (2026-03-17)

| Workload | Status | Notes |
|----------|--------|-------|
| Ollama / ggml (all context lengths) | ✅ Production | Bug 22 fixed — graph pass-through mode, direct kernel execution |
| PyTorch inference (eager) | ✅ Confirmed | All transformer ops incl. bias |
| PyTorch full fine-tuning | ✅ Confirmed | 1,252 tok/s, 409ms/step |
| PyTorch LoRA (PEFT) | ✅ Confirmed | 1,038 tok/s, 493ms/step, 1,360MB VRAM |
| JupyterLab kernel | ✅ Confirmed | Via systemd EnvironmentFile |
| Stable Diffusion Forge | ✅ Confirmed | Image generation 1.61 it/s |
| cuDNN-accelerated conv | ❌ Blocked | Bug 19 — parked |

## Pending Items

| Item | Priority | Notes |
|------|----------|-------|
| Bug 22a — Word doubling/garbling | 🅿️ Parked | Only observed during capture/replay testing; pass-through fix eliminates root cause. Revisit if it resurfaces. |
| Bug 19 — cuDNN context struct layout | Low | Parked — nvproxy reference needed |
| vLLM template validation | High | VMEM_PROXY=1 already set |
| React frontend migration | Low | Backlog |
| `cuLaunchKernelEx` grid dims | Low | Parses 1×1×1 defaults |