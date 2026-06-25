# DeCloud GPU Proxy — Debugging Journal

**Date Range:** 2026-02-27 through 2026-06-25
**Authors:** BMA + Claude AI assistant
**Status:** Active — PyTorch inference + training + LoRA + SD Forge + Ollama 3B GPU production-ready; Ollama pinned to 0.7.0; 7B+ models blocked by Bug 23a. SEC-1 root fix (fork-per-VM workers) **shipped and hardware-validated** (2026-06-25) — including worker/supervisor lifecycle, per-tenant VRAM quota, and the **cross-tenant denial property itself** (fork DENIED vs. threaded VULNERABLE, two wallets). vsock quota path code-verified only (untestable on WSL2).

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
| 16 | May 24 | **Bug 23** — Ollama 0.24.0 incompatible with `cudaErrorNotSupported`; pinned to 0.7.0 |
| 17 | May 25 | **Bug 23a** — 7B+ models corrupt at extended generation; confirmed proxy bug, not model |
| 18 | Jun 11 | **SEC-1** — shared primary context allows cross-tenant GPU memory access; interim scheduler invariant + fork-per-VM worker plan |
| 19 | Jun 25 | **SEC-1 root fix shipped** — fork-per-VM workers built + deployed via on-node daemon build pipeline; per-VM context isolation hardware-confirmed |
| 20 | Jun 25 | **Worker/supervisor lifecycle** — context/VRAM leak root-caused (half-open blocking + unkillable workers + supervisor orphan); all three fixed and hardware-validated |
| 21 | Jun 25 | **Per-tenant VRAM quota** — quota was per-connection (VM could get N× via N connections); fixed with shared-memory ledger; TCP hardware-confirmed, vsock code-complete |
| 22 | Jun 25 | **Quota validation + docs** — enforcement confirmed end-to-end (oversized model DENIED; per-tenant cap holds); docs updated |
| 23 | Jun 25 | **SEC-1 cross-tenant denial VALIDATED** — two different-owner VMs; B's read of A's VRAM DENIED under fork, VULNERABLE under threaded control. The security property itself, not just the mechanism, confirmed on hardware |

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

---

## Session 16: Ollama Version Compatibility Break (May 24, 2026)

### Bug 23: `ggml_backend_cuda_graph_reserve` Aborts on `cudaErrorNotSupported`

**Symptom:** Fresh AI Chatbot VM deployment fails on every chat request. Ollama runner crashes with `CUDA error: unknown error` at `cudaStreamBeginCapture` inside `ggml_backend_cuda_graph_reserve`, then `SIGABRT`. The shim config (systemd drop-in, LD_PRELOAD, library injection) was all correct.

**Installed version:** Ollama 0.24.0 (latest as of May 2026, installed by `curl -fsSL https://ollama.com/install.sh | sh`).

**Root cause:** Ollama 0.24.0 added `ggml_backend_cuda_graph_reserve()`, called unconditionally during `llama_init_from_model` (context creation). This function calls `cudaStreamBeginCapture` and wraps it in ggml's `CUDA_CHECK()` macro, which calls `ggml_abort()` on **any** non-zero return — including `cudaErrorNotSupported`. Previously, `cudaStreamBeginCapture` was only called during the inference path, where ggml had a graceful fallback to direct execution. The reserve call has no such fallback.

The Session 15 fix (returning `cudaErrorNotSupported`) was correct for the Ollama versions in use at the time. The assumption it relied on — that ggml always handles `NOT_SUPPORTED` from `cudaStreamBeginCapture` gracefully — was broken by the new reserve call site.

**Crash stack (simplified):**
```
llama_init_from_model
  → ggml_backend_cuda_graph_reserve          ← new in 0.24.0, no fallback
    → cudaStreamBeginCapture
      → shim returns cudaErrorNotSupported
        → CUDA_CHECK() → ggml_abort() → SIGABRT
```

**Versions investigated:**

| Version | cuda_v12 layout | Result | Reason |
|---------|----------------|--------|--------|
| 0.24.0 (latest) | `libcudart.so.12` (symlink), `libcublas.so.12` (symlink), `libcublasLt.so.12` (symlink), `libggml-cuda.so` | ❌ SIGABRT | `ggml_backend_cuda_graph_reserve` aborts on `cudaErrorNotSupported` |
| 0.5.13 | `libcudart.so.12` → dangling symlink, `libcublasLt.so.12.8.3.14` (real 737MB), `libggml-cuda.so` | ❌ CPU fallback | Different layout — dangling `libcudart` symlink, no `libcublas.so.12`; shim injection script assumptions don't match |
| 0.7.0 | `libcudart.so.12` (symlink), `libcublas.so.12` (symlink), `libcublasLt.so.12` (symlink), `libggml-cuda.so` | ✅ GPU working | Correct layout for shim injection; `cudaErrorNotSupported` from `cudaStreamBeginCapture` handled gracefully |

**Fix (immediate):** Downgrade to Ollama 0.7.0.

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.7.0 sh
# Re-inject shims after downgrade (install overwrites cuda_v12/)
CUDA_DIR=/usr/local/lib/ollama/cuda_v12
for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
  target="$CUDA_DIR/$lib"
  if [ -f "$target" ] || [ -L "$target" ]; then
    [ ! -f "$target.orig" ] && cp -P "$target" "$target.orig"
    rm -f "$target"
    case "$lib" in
      libcublas.so.12)   cp /usr/local/lib/libcublas_stub.so "$target" ;;
      libcublasLt.so.12) cp /usr/local/lib/libcublasLt_stub.so "$target" ;;
      *)                 cp /usr/local/lib/libdecloud_cuda_shim.so "$target" ;;
    esac
  fi
done
systemctl restart ollama
```

**Fix (template):** Pin `OLLAMA_VERSION=0.7.0` in cloud-init Mode 2 install line (`TemplateSeederService.cs` and the seeded template document):

```bash
# Before:
curl -fsSL https://ollama.com/install.sh | sh
# After:
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.7.0 sh
```

**Confirmed working on 0.7.0:**
- `CUDA0 model buffer size = 1918.35 MiB` ✅
- `CUDA0 KV buffer size = 448.00 MiB` ✅
- `nvidia-smi` shows 2767 MiB VRAM used ✅
- Chat response ~17s (GPU generation at ~13-21 tok/s) ✅

**Root fix needed (platform):** The shim's `cudaStreamBeginCapture` needs to return `cudaSuccess` when called from `ggml_backend_cuda_graph_reserve` (context init) while still forcing `cudaErrorNotSupported` during inference-time capture. These two call sites are indistinguishable from the shim side by function name alone. One approach: return `cudaSuccess` on the first call per-process (satisfying reserve) and `cudaErrorNotSupported` on all subsequent calls (forcing direct execution). This is a targeted `cuda_shim.c` change. Until this is implemented, Ollama must remain pinned at 0.7.0.

### Key Lessons Learned (Session 16)

25. **Version-pin application dependencies that interact with the shim.** The shim makes behavioral assumptions about how applications handle specific error codes. When the application changes that behavior (new call site without fallback), the shim breaks. Template installs must pin to a verified-compatible version.

26. **`cudaErrorNotSupported` is not universally safe.** It triggers correct fallback behavior in ggml's inference path, but the same code in a context-init path (no fallback) causes abort. Error handling contracts must be verified per call site, not per error code.

27. **Always re-inject shims after Ollama upgrades/downgrades.** The install script replaces `/usr/local/lib/ollama/` entirely, including `cuda_v12/`. The shim injection performed by cloud-init is not persistent across Ollama reinstalls. This is a known operational hazard.

28. **Ollama's `cuda_v12` layout changes between versions.** 0.5.13 uses versioned filenames with symlinks and ships real cuBLAS Lt (737MB). 0.7.0+ uses plain symlink names. The cloud-init injection script targets the symlink names (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`) — this only works for the 0.7.0+ layout.
---

## Session 17: 7B+ Model GPU Corruption (May 25, 2026)

### Bug 23a: Ġ Token Corruption in Extended Generation for Large Models

**Symptom:** `dolphin-llama3:8b` produces increasingly corrupted output during GPU inference. Clean for the first ~150-200 tokens, then standalone `G` characters appear between words, worsening until complete cascade collapse (output becomes streams of `GivesGGGivesC...`). The 3B model (`llama3.2:3b`) generates clean output at any length under the same conditions.

**Model under test:**
- `dolphin-2.9-llama3-8b`, Q4_0, 4.33 GiB, 8.03B params
- 32 layers, n_embd=4096, n_ff=14336, n_head=32, n_head_kv=8, n_gqa=4
- KV cache: 1024 MiB (512 K + 512 V), `--parallel 2`, `--no-mmap`

**Working model (3B) for comparison:**
- `llama3.2:3b`, Q4_K_M, 1.87 GiB, 3.21B params
- 28 layers, n_embd=3072, n_ff=8192, n_head=24, n_head_kv=8, n_gqa=3
- KV cache: 448 MiB, `--parallel 1`, mmap=true

**What the "G" characters actually are:** Unicode `Ġ` (U+0120) — Llama's BPE space-prefix token. Normally fused with the following word (e.g. `Ġocean` = " ocean"). When the GPU produces slightly wrong logits, the space-prefix token scores above threshold as a standalone token, inserting a visible `G` between words. The cascading collapse is self-reinforcing: corrupted tokens become corrupted KV cache entries, which corrupt subsequent attention computations.

**Diagnostic steps and findings:**

| Test | Result | Conclusion |
|------|--------|------------|
| 3 sentences (8B, GPU) | Clean | Short context within safe range |
| 10 sentences (8B, GPU) | 1 G at end | Corruption onset ~token 250 |
| 30 sentences (8B, GPU) | Clean to ~sentence 20, then cascade | Progressive accumulation |
| `OLLAMA_NUM_PARALLEL=1` (8B, GPU) | Slightly later onset, still cascades | Parallel not the root cause |
| 30 sentences (3B, GPU) | Fully clean all 30 | 3B geometry works correctly |
| `DECLOUD_GPU_DEBUG` GEMM log | Empty — no GemmStrided/GemmBatched calls | MMQ path used, not GEMM proxy |
| CPU isolation attempt | Could not achieve — NVML/libcuda shims always report GPU; Ollama ignores `OLLAMA_LLM_LIBRARY=cpu` | Cannot rule out model corruption via this path |

**Why CPU isolation failed:** The NVML shim (`libnvidia-ml.so.1`) and driver shim (`libcuda.so.1`) are installed system-wide. Ollama's GPU detection uses both and always reports the proxied GPU regardless of `OLLAMA_LLM_LIBRARY=cpu` or `OLLAMA_NUM_GPU_LAYERS=0`. Removing the NVML shim did not help because `libcuda.so.1` alone is sufficient for Ollama to discover and use the GPU. True CPU isolation on this VM requires removing all shims from system paths, which is too invasive for a production VM.

**Root cause (hypothesis):** The proxy's kernel launch parameter handling or memory write-back logic has a geometry-dependent error that accumulates over sequential token generations. The 8B model's larger tensor dimensions (n_embd=4096 vs 3072, n_ff=14336 vs 8192, 32 vs 28 layers, 1024 MiB KV vs 448 MiB) exceed a boundary where the proxy computes addresses or strides correctly. The GEMM proxy is ruled out (MMQ path confirmed). The most likely candidates are:

1. **KV cache write stride error** — the 8B's `n_embd_k_gqa=1024` per layer across 32 layers with `n_seq_max=2` may cause offset calculations that overflow a fixed-size buffer or produce wrong addresses after enough sequential writes
2. **Kernel launch parameter truncation** — `cudaLaunchKernel` payload for the 8B's larger grid/block dimensions may overflow the RPC protocol's parameter packing

**Status:** 🔴 Open — 7B+ models unusable on GPU proxy until fixed.

**Debugging path:**
```bash
# Enable full RPC logging
touch /tmp/gpu-proxy-diag
DECLOUD_GPU_DEBUG=1 ollama run dolphin-llama3:8b "write 30 sentences about the ocean" 2>&1

# After run, compare launch parameters vs 3B
cat /tmp/gpu-proxy-diag.log | grep -E "LaunchKernel|gridDim|blockDim" | head -60
# Run same for 3B and diff the grid/block size patterns
```

The divergence point will be in kernel launches that occur after token ~200, compared against the 3B's equivalent launches at that sequence position.

**Workarounds:**
- Use `llama3.2:3b` for GPU inference — clean at all output lengths
- 8B on GPU is usable for short responses (<150 tokens) only
- 8B on CPU (true CPU, via shim removal) would be correct but ~2 tok/s — impractical

### Key Lessons Learned (Session 17)

29. **Model compatibility must be verified at each major parameter boundary.** A proxy that works for 3B models may fail for 7B+ due to larger tensor dimensions, deeper layer counts, or bigger KV cache geometry. Each boundary (n_embd, n_ff, n_layers, KV size) is an independent failure surface.

30. **NVML + libcuda shims make CPU isolation impossible on a live GPU proxy VM.** Removing one shim is insufficient. This is an operational constraint that makes root-cause isolation harder. For future debugging, maintain a separate "bare metal" test VM without shims installed.

31. **Ġ token corruption is a reliable proxy bug indicator.** The BPE space-prefix token (U+0120) appearing as a standalone character means the GPU computed wrong logits — not a model file issue, not a tokenizer issue. It specifically indicates that activation values entering the final linear layer are subtly wrong, pointing to a memory write or compute ordering error in the proxy path.

32. **Corruption onset token count is a diagnostic signal.** Corruption starting at token ~200-250 and cascading is consistent with a per-token accumulated error (e.g. wrong KV cache address that gets read back every attention step). Instant corruption from token 1 would indicate a model weight loading error.


---

## Session 18: Security Review — Shared Primary Context (SEC-1) (June 11, 2026)

### SEC-1: Cross-Tenant GPU Memory Access via Shared Primary Context

**Classification:** Security finding from architecture review — not a runtime bug. No code misbehaves; the vulnerability is an emergent property of correct decisions composing.
**Severity:** 🔴 Critical for multi-tenant GPU deployments. No impact under single-tenant-per-GPU operation.

**Origin:** Session 7, Problem 25. The switch from `cuCtxCreate` to `cuDevicePrimaryCtxRetain` was the correct fix for the runtime/driver context mismatch (`CUDA_ERROR_INVALID_HANDLE`). Its unexamined consequence: the primary context is per-device **per-process**, and the daemon is one process serving every VM connection on the node. All tenants therefore share one CUDA context — one GPU address space.

**Why this matters:** A CUDA context is the GPU's equivalent of a process address space. Device pointers are virtual addresses valid context-wide, and within a context CUDA provides **zero isolation** — any code in the context can read, write, or free any allocation. CUDA's security model assumes one context = one trust domain. Per-VM tokens authenticate *who is calling*; they say nothing about *which addresses* the call may name. Memory quotas limit how much a tenant allocates, not which addresses it can reference. Per-connection function tables cover modules, not memory.

**Attack surface (per RPC verb):**

| Verb | Cross-tenant capability |
|------|------------------------|
| `GPU_CMD_MEMCPY` D2H with foreign device address | Read another tenant's VRAM — weights, KV cache, prompts. Address space is scannable (success/failure oracle; VRAM is small) |
| `GPU_CMD_MEMCPY` H2D / `GPU_CMD_MEMSET` with foreign address | Silent integrity corruption — weight poisoning, KV cache manipulation. Victim may never know |
| `GPU_CMD_FREE` on foreign pointer | Cross-tenant use-after-free / denial of service |
| `GPU_CMD_LAUNCH_KERNEL` with attacker kernel | Full read/write of foreign buffers via pointer arguments — **unvalidatable in principle** (see below) |

The attacker requires no exploit and no privilege — only a valid tenant token, which a permissionless platform hands out by design. The Session 8 design (daemon dereferences VM-supplied device pointers, e.g. D2H reads of GEMM pointer arrays) makes this a documented, normal code path.

**Why daemon-side pointer validation is insufficient:** The daemon could maintain a per-connection allocation map (it already sees every Malloc/Free) and reject memcpy/memset/free naming foreign addresses — that genuinely closes three verbs. But `cudaLaunchKernel` parameters arrive as an **opaque byte blob**: no kernel signature, no type information, no way to distinguish pointers from integers. A tenant launches its own kernel whose arguments point into foreign buffers, and the kernel — running in the shared context — accesses them freely. Validation alone can never close the hole. Per the platform design philosophy: don't wrap a systemic problem in an artificial control; align to the boundary the system provides.

**Root fix (planned): fork-per-VM daemon workers.** The CUDA primary context is per-process, so one daemon worker process per VM connection yields per-tenant contexts with **GPU MMU-enforced isolation** — cross-context access fails at the driver level, no validation code written by us. Additional benefits:
- Fault isolation: a worker crash (malformed RPC, daemon bug) kills one tenant's session, not every tenant on the node
- Enables a per-launch compute watchdog: timeout → kill worker → context released cleanly (impossible to do surgically inside a shared-context monolith)

**Cost (honest):** ~200–400MB VRAM per CUDA context on consumer GPUs (driver state + reserved heap). On an 8GB RTX 4060, 3-4 workers consume 1GB+ before tensors load. Tenant density becomes a scheduler-visible resource — feeds the existing `GpuVramBytes` accounting in `ResourceSnapshot`. At realistic model sizes, consumer-GPU density is 2-3 tenants anyway, so the overhead rarely changes the answer.

**Interim mitigation (REQUIRED before any GPU co-tenancy):** Scheduler invariant — **at most one wallet's GPU VMs per physical GPU**. A one-line constraint in orchestrator GPU scheduling that makes the entire attack class structurally impossible at zero cost at current scale.

**Residual hardening (ship with the worker refactor):**
- **Zero-on-free:** `cudaMalloc` does not zero memory and NVIDIA does not guarantee scrubbing of recycled VRAM — published "CUDA Leaks"-class research recovers prior tenants' data across processes. Daemon must `cudaMemset(0)` on every Free and on connection teardown.
- **Compute watchdog:** an adversarial non-terminating kernel hangs the shared silicon regardless of context isolation. Per-launch timeout → worker kill → context reset.
- **Transport:** the per-VM token travels plaintext on the TCP path. Restrict TCP transport to the WireGuard overlay or vsock only — never bind a routable interface.

**Out of scope at this layer (document in threat model):** GPU microarchitectural side channels (timing, cache contention) and the absence of MIG-style hardware partitioning on consumer cards. Workloads with hard confidentiality requirements get the dedicated-GPU tier — an honest product boundary, not a weakness.

### Key Lessons Learned (Session 18)

33. **A correct fix can carry an unexamined trust-model consequence.** Problem 25's primary-context switch was right for API interop and wrong for multi-tenancy. Review every shared-resource decision against the tenant boundary, not just the bug in front of you.

34. **Authentication is not authorization for addresses.** Per-VM tokens prove who is calling; they say nothing about which memory the call may name. Any RPC carrying raw device pointers crosses a trust boundary and must be treated as such.

35. **Opaque kernel launch parameters make daemon-side pointer validation impossible in principle.** Only context separation — per-process primary contexts enforced by the GPU MMU — actually closes the hole. Align to the boundary the driver provides instead of building a validation layer that cannot be complete.

36. **`cudaMalloc` does not zero memory, and freed VRAM is not scrubbed.** Cross-tenant data survives deallocation. Zero-on-free in the daemon is mandatory hygiene for any shared GPU, even after process separation.

---

## Session 19: SEC-1 Root Fix Shipped — Fork-per-VM Workers (June 25, 2026)

The fork-per-VM worker model planned in Session 18 was built, deployed, and hardware-validated. The supervisor stays CUDA-free (it probes the GPU in a short-lived forked child so it never initializes CUDA in its own address space — CUDA state cannot survive `fork()`), and forks one worker process per connection. Each worker calls `cuInit` itself, getting its own primary CUDA context → cross-context access fails at the GPU MMU. Three isolation modes (`auto`/`fork`/`threaded`), with `auto` the default and `threaded` an explicit single-tenant opt-in the orchestrator never selects.

**Deployment mechanism (separate but enabling work):** the daemon links against the host's exact CUDA version, so it cannot ship as a prebuilt binary. It ships as cosign-verified **source** in the release; `install.sh`/`decloud update` fetch, verify, and compile it on-node, rebuilding only when the source hash changes. This closed a latent incident: the installed daemon had been a stale May-7 binary with no `-i` support; the agent launched it with `-i auto`, it exited `invalid option`, and VMs silently fell back to CPU. The on-node build pipeline makes daemon fixes flow through `decloud update` like any other change — exercised five times this session.

**Hardware result:** on RTX 4060 / WSL2, `auto` mode forks correctly despite the WSL2 passthrough CUDA stack (`per-VM fork isolation active (SEC-1)` logged; VM connected; inference ran). Two same-owner VMs later ran concurrently, each in its own forked worker with its own context.

### Key Lessons

37. **A correct daemon binary is worthless if the deploy path can ship a stale one.** The fix wasn't only the fork code — it was making the build staleness-checked and automatic. A security fix that depends on manual rebuild will silently rot.

38. **Never hand-launch the daemon while the agent is running.** A manually started daemon wins the port race, serves stale tokens, and starves the agent's own daemon — diagnosed twice this session. The agent owns the lifecycle.

---

## Session 20: Worker & Supervisor Lifecycle (June 25, 2026)

Concurrent same-owner VMs surfaced a context/VRAM leak: orphaned forked workers holding CUDA contexts (5719 MiB held with VMs idle; killing them → 13 MiB, proving the leak was real contexts, not WSL2 accounting). Three distinct defects, all teardown-related, all fixed and hardware-verified.

**Defect 1 — half-open leak.** A worker blocks forever in `read_exact` when its VM connection dies without a clean close (killed runner, dropped link). No FIN/GOODBYE arrives, the blocking `recv` never returns, the existing `break → cleanup → _exit` path never fires. **Fix:** `SO_KEEPALIVE` (60/10/3 ≈ 90 s) on accepted TCP connections so a dead peer makes the read return; `SO_RCVTIMEO` ceiling for vsock. The dead-peer case now becomes observable to the already-correct cleanup path.

**Defect 2 — unkillable worker.** The forked child inherited the supervisor's flag-only `SIGTERM` handler (sets `g_running`, which a worker blocked in `recv` never re-checks). Only `SIGKILL` ended it → workers orphaned across `decloud update` restarts. **Fix:** child resets `signal(SIGTERM, SIG_DFL)` (alongside the existing `SIGCHLD` reset). Workers are now individually killable and die with the daemon.

**Defect 3 — supervisor orphan.** On WSL2 the supervisor runs TCP-only fallback: main thread parks in `sleep(1)` while `tcp_listener_thread` blocks in `accept()`. On SIGTERM the handler set `g_running=0` and `kill(0,SIGTERM)` (reaping workers — worked) but nothing closed the listen FD, so the listener thread never returned and the supervisor reparented to init (PPID 1) on restart — then got re-adopted by the next agent via the pgrep health path (same permissive "a daemon exists → trust it" antipattern as the original stale-binary incident). **Fix:** handler closes the listen FDs to unblock `accept()` then `_exit(0)` (the supervisor holds no CUDA context — VRAM→0 proved it — so immediate exit is safe); agent's `StopAsync` first kill changed to `entireProcessTree: true`.

**Hardware result:** plain `kill` (not `-9`) now ends a worker; stopping a VM's Ollama drops its worker; agent restart leaves no surviving worker or supervisor PID and VRAM returns to 0.

### Key Lessons

39. **A blocked syscall ignoring a flag-only signal handler is the same bug at every level.** Worker `recv` and supervisor `accept` both ignored `g_running`. The fix in both cases is to make the syscall *return* (close the FD / arm keepalive), not to hope the flag is noticed.

40. **The right fix for a leaked resource is to bind its lifetime to the boundary the system already has** — a worker's life = its connection's life; the supervisor's life = the agent's. Not a reaper sweep or an idle-kill timer, which only conceal the systemic problem.

---

## Session 21–22: Per-Tenant VRAM Quota (June 25, 2026)

With the lifecycle stable, the VRAM quota was tested and found enforced **per-connection, not per-tenant**. Confirmed on hardware: one VM with `OLLAMA_MAX_LOADED_MODELS=2` loaded two ~2 GB models concurrently (two runners = two connections) = 6.7 GB on GPU against a 2672 MB quota — each connection independently got the full quota. This breaks the scheduler's `GpuVramBytes` accounting (the orchestrator packs assuming each VM is bounded by its quota) and is a multi-tenant fairness/DoS vector.

A second, deeper gap surfaced while scoping the fix: quota was enforced **only on the TCP path**. `ctx->memory_quota` and `ctx->vm_id` were set exclusively inside the `if (ctx->is_tcp)` branch of `handle_hello`, from the auth token. vsock connections (preferred on bare metal) carry no token, skipped that branch, and got `vm_id=""`/`quota=0` → no enforcement at all. So on bare-metal/vsock nodes, VRAM quotas were unenforced entirely — independent of the per-tenant bug.

**Fix (two parts):**
- **Per-tenant ledger.** A shared-memory slot table (`mmap(MAP_SHARED|MAP_ANONYMOUS)` created by the supervisor before forking, so all workers inherit it), one slot per connection `{vm_id, pid, allocated}`. `handle_malloc` enforces against the **sum of live slots for the VM**, not the connection's private counter. Self-healing: dead-PID slots are reaped while summing (`kill(pid,0)==ESRCH`), so a SIGKILL'd worker can't leak budget; fail-safe (over-counts → tenant gets less, never over-allocates). Chosen over a running counter precisely because hard-killed workers happen.
- **vsock coverage.** The registry line gained a `cid` field (`<token> <vm_id> <quota> <cid>`); the daemon resolves a vsock connection's VM from the guest source CID (`peer.svm_cid`, captured at accept — unspoofable, the hypervisor assigns it). Both transports now populate `vm_id`+`quota` before `handle_malloc`.

**Hardware result (TCP path):** quota enforcement confirmed end-to-end first — an oversized single model (llama3.1:8b, ~4.3 GB) was DENIED at `cudaMalloc(4617396480)` against the 2801795072-byte quota, VM got `unable to allocate CUDA0 buffer`. (Per-tenant patch validated by re-running the two-model test: second model now DENIED, `ollama ps` shows one model, instead of 6.7 GB.)

**Still open:** the vsock path is code-complete but untestable on WSL2 (vsock unavailable) — first real exercise is on bare metal. And the cross-tenant *denial* security test (two different-owner tenants + `sec1_probe`) remains the distinct, still-unrun validation of the isolation *property* (vs. the mechanism, which is proven).

### Key Lessons

41. **A per-connection control is not a per-tenant control when one tenant can open many connections.** The quota was correct in value and wrong in scope. Test the boundary the threat model cares about (the tenant), not the one the code happens to track (the connection).

42. **Cross-process enforcement needs cross-process state.** Fork-per-VM isolation (separate address spaces) is exactly what made the per-VM counter hard — the same boundary that isolates contexts isolates counters. Shared memory keyed on identity, summed over live owners, is the aligned fix.

43. **Validate enforcement is armed before assuming it works — and after.** Quota looked unenforced (guest saw the whole card); the daemon log proved it *was* set; an oversized-model test proved it *fired*. Each step corrected a wrong assumption. "Configured" ≠ "enforced" ≠ "enforced at the right scope."

---

## Session 23: SEC-1 Cross-Tenant Denial — Validated (June 25, 2026)

The security property at the heart of SEC-1 — *a tenant cannot read another tenant's VRAM* — was directly tested for the first time. Everything prior validated the mechanism (fork-per-VM contexts) and resource fairness (quota); this tested the trust boundary itself, with **two different owner wallets**.

**Method (guest-side `sec1_probe_guest`, over the real TCP transport, each VM using its own valid token):**
- VM-A (victim): `cudaMalloc(1 MB)`, `cudaMemset` a `0xAB` sentinel, print the device pointer (`DEVPTR=0x705800000`), hold the connection open.
- VM-B (attacker, different wallet): `cudaMemcpy` device→host from A's exact address.

**Result:**

| Daemon mode | VM-B reading VM-A's `0xAB` buffer | Process topology |
|-------------|-----------------------------------|------------------|
| `auto` (fork) | `READ DENIED — isolation OK` | supervisor + per-VM forked workers |
| `threaded` (shared ctx, control) | `>>> VULNERABLE: first=0xAB` | single daemon, no forks |

The threaded control is what makes the result conclusive: it reproduces the original SEC-1 vulnerability on demand (B reads A's sentinel through the shared context), proving the probe genuinely detects cross-tenant access. The same probe, same addresses, is DENIED only when fork isolation gives each worker its own context. **DENIED-under-fork + VULNERABLE-under-threaded is the result that closes SEC-1.**

The probe uses D2H memcpy (not a peer-to-peer device copy), so `GGML_CUDA_NO_PEER_COPY` is not a factor — the denial is attributable to the context boundary.

**Scope (honest):** this covers the read (D2H) path — the canonical confidentiality test. Write (H2D/memset), free, and kernel-launch cross-tenant access are closed by the *same* per-worker context boundary (a foreign address is unmapped regardless of which verb names it), but were not each separately probed. A read DENIED is strong evidence the boundary holds for every path sharing that context. The threaded node config was reverted to `auto` immediately after the control.

### Key Lessons

44. **A mechanism test and a property test are different claims.** "Workers fork into separate contexts" (mechanism) was proven days before "tenant B cannot read tenant A" (property). Same-owner concurrency exercises the mechanism; only different-owner probing tests the boundary the threat model cares about. Don't let the first stand in for the second.

45. **A security test that cannot fail proves nothing — build the positive control.** An isolated DENIED result is indistinguishable from a broken probe. Forcing the vulnerable (threaded) configuration and confirming the probe reports VULNERABLE is what earns trust in the DENIED. The contrast is the evidence, not either result alone.

---




| Workload | Status | Notes |
|----------|--------|-------|
| Ollama llama3.2:3b GPU | ✅ Production | Clean output at all lengths; 13-21 tok/s generation |
| Ollama 7B+ models GPU | ❌ Blocked | Bug 23a — Ġ token corruption in extended generation |
| Ollama 7B+ models CPU | ⚠️ Impractical | ~2 tok/s; CPU isolation not possible on live proxy VM |
| PyTorch inference (eager) | ✅ Confirmed | All transformer ops incl. bias |
| PyTorch full fine-tuning | ✅ Confirmed | 1,252 tok/s, 409ms/step |
| PyTorch LoRA (PEFT) | ✅ Confirmed | 1,038 tok/s, 493ms/step, 1,360MB VRAM |
| JupyterLab kernel | ✅ Confirmed | Via systemd EnvironmentFile |
| Stable Diffusion Forge | ✅ Confirmed | Image generation 1.61 it/s |
| cuDNN-accelerated conv | ❌ Blocked | Bug 19 — parked |
| Multi-tenant GPU sharing | 🟢 Isolation validated | SEC-1 root fix + lifecycle + per-tenant quota + **cross-tenant denial** all hardware-validated (Sessions 19–23; fork DENIED vs. threaded VULNERABLE, two wallets). vsock quota path code-verified only |

## Pending Items

| Item | Priority | Notes |
|------|----------|-------|
| SEC-1 root fix — fork-per-VM daemon workers | ✅ Shipped + hardware-validated (Session 19) | Per-process primary contexts → GPU MMU isolation. ~400 MB VRAM/context deducted from quota. Lifecycle (Session 20) and per-tenant quota (Sessions 21–22) also shipped |
| SEC-1 — cross-tenant *denial* test | ✅ Validated (Session 23) | Two different-owner VMs: B's D2H read of A's address DENIED under fork, VULNERABLE under threaded control. Read path proven; write/free/kernel-launch closed by the same context boundary but not separately probed |
| SEC-1 — vsock quota enforcement | 🟡 Code-complete, untested | Per-tenant quota + CID-based vsock identity written but not exercisable on WSL2 (no vsock). Validate on a bare-metal node. Note: before this work, quota was TCP-only — bare-metal/vsock nodes had no VRAM cap at all |
| SEC-1 — adopt-on-pgrep | 🟡 Open | `HealthCheckAsync` adopts any running `gpu-proxy-daemon` via pgrep instead of verifying it's the current binary — same permissive pattern as the original stale-daemon incident. Needs a version/capability probe |
| SEC-1 hardening — compute watchdog / transport restriction | Medium | Per-launch worker-kill watchdog not yet wired (kernel `-t` timeout exists); restrict TCP transport to WireGuard overlay / vsock (token travels plaintext on TCP) |
| Template debt — `DECLOUD_GPU_GRAPH_NOOP` in ai-chatbot template | Low | Dead flag (removed Session 15) still emitted by `tenant-vms/ai-chatbot/cloud-init.yaml`; confirmed not read by daemon. Drop from the YAML so reseeds stop carrying it |
| Bug 23a — 7B+ model Ġ token corruption | High | Proxy kernel launch or KV stride error at large tensor geometry. Enable `DECLOUD_GPU_DEBUG`, capture RPC log, diff vs 3B launch params. |
| Bug 23 — Shim fix for `ggml_backend_cuda_graph_reserve` | High | Root fix: `cudaStreamBeginCapture` must return `cudaSuccess` on first call per-process. Until fixed, Ollama pinned to 0.7.0. |
| Bug 22a — Word doubling/garbling | 🅿️ Parked | Only observed during capture/replay testing; pass-through fix eliminates root cause. Revisit if it resurfaces. |
| Bug 19 — cuDNN context struct layout | Low | Parked — nvproxy reference needed |
| vLLM template validation | High | VMEM_PROXY=1 already set |
| React frontend migration | Low | Backlog |
| `cuLaunchKernelEx` grid dims | Low | Parses 1×1×1 defaults |
