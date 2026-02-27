# DeCloud GPU Proxy — CUDA Shim Production Debugging: Technical Handoff

## Document Purpose

This document provides complete technical context for continuing work on the DeCloud GPU proxy system's CUDA shims. It covers the problem space, all issues encountered and resolved during the 2026-02-27 debugging session, current achievement state, and the specific remaining work needed to achieve GPU-accelerated inference through the proxy.

**Server:** `srv022010` (ai-chatbot-dae9) — DeCloud node agent running Ollama v0.17
**GPU Host:** MSI laptop with NVIDIA RTX 4060 Laptop GPU (compute capability 8.9, 8GB VRAM)
**Working files on server:** `/tmp/gpu-proxy/cuda_shim.c`, `/tmp/gpu-proxy/libcudart.version`
**Repository path:** `src/gpu-proxy/shim/cuda_shim.c`
**Deployed binary:** `/usr/local/lib/ollama/libcudart.so.12` (also symlinked at `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90`)

---

## 1. Architecture Overview

### How DeCloud GPU Proxy Works

DeCloud provides GPU access to virtual machines that don't have physical GPUs. The system consists of:

1. **GPU Proxy Daemon** — runs on the host machine (MSI laptop) with the real NVIDIA GPU. Listens on TCP port 9999 (and vsock). Receives GPU commands over a binary RPC protocol and executes them using real CUDA APIs.

2. **CUDA Runtime Shim** (`libcudart.so.12` / `cuda_shim.c`) — a drop-in replacement for NVIDIA's CUDA Runtime library. Deployed inside the VM where Ollama runs. Intercepts all `cuda*` function calls and forwards them to the daemon via RPC. This is what we've been debugging.

3. **CUDA Driver Shim** (`libcuda.so.1` / `cuda_driver_shim.c`) — a drop-in replacement for NVIDIA's CUDA Driver library. Also deployed inside the VM. Intercepts `cu*` function calls. Currently has basic implementations for device discovery but returns generic stubs for memory operations.

4. **NVML Shim** (`libnvidia-ml.so.1` / `nvml_shim.c`) — a drop-in replacement for NVIDIA's management library. Provides device monitoring info.

### How Ollama Discovers GPUs

Ollama v0.17 performs GPU discovery through a **bootstrap subprocess** mechanism:

```
Parent Process (Go)
  │
  ├─ Spawns bootstrap subprocess with LD_LIBRARY_PATH pointing to cuda_v12/
  │    │
  │    ├─ dlopen("libggml-cuda.so")  ← this triggers loading of:
  │    │    ├─ libcudart.so.12       ← our Runtime API shim
  │    │    ├─ libcublas.so.12       ← real NVIDIA cuBLAS (bundled by Ollama)
  │    │    ├─ libcublasLt.so.12     ← real NVIDIA cuBLAS Light
  │    │    └─ libcuda.so.1          ← our Driver API shim
  │    │
  │    ├─ Calls cudaGetDeviceCount() → shim connects to daemon → returns 1
  │    ├─ Calls cudaGetDeviceProperties() → gets name, VRAM, compute capability
  │    └─ Reports results back to parent
  │
  ├─ Parent receives: 1 GPU, RTX 4060, compute=8.9, 8GB VRAM
  ├─ Verification: spawns SECOND bootstrap with CUDA_VISIBLE_DEVICES + GGML_CUDA_INIT=1
  │    └─ Must also return device info successfully
  │
  └─ If both pass → schedules model layers to GPU
```

**Critical constraint:** The bootstrap subprocess has a **strict timeout**. If it doesn't complete within the allotted time, the parent sends SIGKILL. For Ollama v0.17, this timeout is approximately 30 seconds for discovery, but the subprocess can also be killed early if it appears to hang.

### The RPC Wire Protocol

Communication between shim and daemon uses a custom binary protocol defined in `proto/gpu_proxy_proto.h`:

```c
typedef struct {
    uint32_t magic;        // 0x44435544 ("DUCD")
    uint8_t  version;      // Protocol version (2)
    uint8_t  cmd;          // Command ID (e.g., GPU_CMD_GET_DEVICE_COUNT)
    uint16_t reserved;
    uint32_t payload_len;  // Length of following payload
    int32_t  status;       // Response status (0 = success)
} GpuProxyHeader;
```

Key commands: `GPU_CMD_HELLO`, `GPU_CMD_GET_DEVICE_COUNT`, `GPU_CMD_GET_DEVICE_PROPERTIES`, `GPU_CMD_REGISTER_MODULE`, `GPU_CMD_REGISTER_FUNCTION`, `GPU_CMD_MALLOC`, `GPU_CMD_FREE`, `GPU_CMD_MEMCPY`, `GPU_CMD_LAUNCH_KERNEL`, `GPU_CMD_STREAM_CREATE`, etc.

---

## 2. Problems Encountered and Resolved

### Problem 1: Symbol Versioning Mismatch

**Symptom:** `initial_count=0`. Standalone Python test worked, but Ollama bootstrap subprocess returned zero devices.

**Root Cause:** `libggml-cuda.so` (Ollama's CUDA backend) was compiled against real NVIDIA cudart and expects versioned symbols like `cudaGetDeviceCount@@libcudart.so.12`. Our initial shim builds used either wrong version tags (`@@CUDART_12.0`) or no version tags at all, causing the dynamic linker to warn "no version information available" and potentially fail symbol resolution.

**Evidence:**
```bash
# What libggml-cuda.so expects:
nm -D libggml-cuda.so | grep cudaGetDeviceProperties
# Output: cudaGetDeviceProperties@@libcudart.so.12

# What real NVIDIA cudart exports:
nm -D libcudart.so.12.8.90 | grep cudaGetDeviceCount
# Output: cudaGetDeviceCount@@libcudart.so.12
```

**Fix:** Created `libcudart.version` version script with correct tag:
```
libcudart.so.12 {
    global: cuda*; __cuda*;
    local: *;
};
```
Built with: `gcc ... -Wl,-soname,libcudart.so.12 -Wl,--version-script=libcudart.version`

---

### Problem 2: Missing Symbols (18 unresolved)

**Symptom:** `dlopen("libggml-cuda.so")` failed silently. Python `ctypes.CDLL('libggml-cuda.so')` reported "undefined symbol".

**Root Cause:** `libggml-cuda.so` requires 49 CUDA symbols. Our shim initially only exported 31. The 18 missing symbols caused dlopen to fail.

**Missing symbols identified:**
```
cudaDeviceCanAccessPeer          cudaDeviceEnablePeerAccess
cudaDeviceDisablePeerAccess      cudaFuncGetAttributes
cudaGraphDestroy                 cudaGraphExecDestroy
cudaGraphExecUpdate              cudaGraphInstantiate
cudaGraphLaunch                  cudaMallocManaged
cudaMemcpy2DAsync                cudaMemcpy3DPeerAsync
cudaMemcpyPeerAsync              cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
cudaStreamBeginCapture           cudaStreamEndCapture
cudaStreamIsCapturing            cudaStreamWaitEvent
```

**Fix:** Added stub implementations for all 18 symbols that return `cudaSuccess` or `cudaErrorNotSupported` as appropriate. These stubs exist in the current `cuda_shim.c`.

---

### Problem 3: Missing Internal CUDA Runtime Symbols

**Symptom:** dlopen still failed after adding the 18 public stubs. Error: `undefined symbol: __cudaPopCallConfiguration, version libcudart.so.12`

**Root Cause:** `libggml-cuda.so` uses CUDA's `<<<grid, block>>>` kernel launch syntax, which the NVCC compiler transforms into calls to internal runtime symbols:

- `__cudaPushCallConfiguration` — stores grid/block dims in thread-local storage
- `__cudaPopCallConfiguration` — retrieves them for `cudaLaunchKernel`
- `__cudaRegisterFatBinaryEnd` — signals end of module registration

**Fix:** Implemented all three using thread-local storage:
```c
static __thread dim3 g_push_grid;
static __thread dim3 g_push_block;
static __thread size_t g_push_shared;
static __thread cudaStream_t g_push_stream;

unsigned int __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                          size_t sharedMem, cudaStream_t stream) {
    g_push_grid = gridDim; g_push_block = blockDim;
    g_push_shared = sharedMem; g_push_stream = stream;
    return 0;
}
// ... etc
```

---

### Problem 4: Bootstrap Timeout Due to RPC During dlopen (THE BIG ONE)

**Symptom:** dlopen succeeded in standalone tests, shim connected to daemon, returned 1 device. But Ollama's bootstrap subprocess was killed by SIGKILL after ~180ms with zero `connect()` syscalls observed.

**Root Cause:** When `libggml-cuda.so` is loaded via `dlopen()`, its static constructors (generated by NVCC) call `__cudaRegisterFatBinary()` and `__cudaRegisterFunction()` for every CUDA kernel in the library. `libggml-cuda.so` contains **hundreds** of flash_attn kernels. Each `__cudaRegisterFatBinary()` was uploading the fatbin (~10MB) to the daemon via RPC, and each `__cudaRegisterFunction()` was doing another RPC call. The cumulative network overhead of hundreds of RPC calls during dlopen constructors caused the bootstrap subprocess to exceed its timeout before ever reaching `cudaGetDeviceCount()`.

**Evidence from strace (before fix):**
```
PID 32107:
  write(2, "[cudart-shim] __cudaRegisterFunction(hostFun=0x7fc88b640dc0...)
  write(9, "DUCD\2R\0\0\253\0\0\0\0\0\0\0", 16) = 16  ← RPC to daemon
  write(2, "[cudart-shim]   registered function -> 21 params")
  ... hundreds more ...
  +++ killed by SIGTERM +++
```

**Fix — Lazy Registration Pattern:**

The fundamental insight: the bootstrap subprocess only needs `cudaGetDeviceCount()` and `cudaGetDeviceProperties()` — it never launches kernels. So module/function registration can be deferred until the first `cudaLaunchKernel()` call.

New architecture:

```
DURING dlopen (constructors):
  __cudaRegisterFatBinary() → store pointer locally in g_modules[] (NO RPC)
  __cudaRegisterFunction()  → store name/ptr locally in g_functions[] (NO RPC)

DURING bootstrap:
  cudaGetDeviceCount()     → small RPC to daemon (fast)
  cudaGetDeviceProperties() → small RPC to daemon (fast)
  ✅ Bootstrap completes in ~500ms

DURING inference (first kernel launch):
  cudaLaunchKernel()
    → ensure_module_uploaded(module_index)
      → Upload fatbin via RPC (once per module)
      → Register all functions for that module via RPC
    → Execute kernel via RPC
```

Key data structures added:

```c
#define MAX_REGISTERED_FUNCTIONS 2048  // was 512
#define MAX_DEFERRED_MODULES 64

typedef struct {
    uint64_t host_func_ptr;
    uint32_t num_params;
    uint32_t param_sizes[GPU_MAX_KERNEL_PARAMS];
    int      registered;       // 1 = RPC done, params filled
    char     device_name[256]; // stored for deferred RPC
    int      module_index;     // index into g_modules[]
} RegisteredFunction;

typedef struct {
    const void *fatbin_data;   // pointer into process memory
    size_t      fatbin_size;
    uint64_t    remote_handle; // daemon module handle after upload
    int         uploaded;      // 1 = sent to daemon
} DeferredModule;
```

**Verification:**
```bash
# Zero RPC calls in registration functions:
sed -n '/^void __cudaRegisterFunction/,/^}/p' cuda_shim.c | grep -c 'rpc_call'
# Output: 0 ✅
sed -n '/^void \*\*__cudaRegisterFatBinary/,/^}/p' cuda_shim.c | grep -c 'rpc_call'
# Output: 0 ✅
```

---

### Problem 5: cudaDeviceProp Struct Layout Mismatch (compute=0.0)

**Symptom:** After lazy registration fix, `initial_count=1` but Ollama logged `compute=0.0` and filtered the GPU with "filtering device which didn't fully initialize".

**Root Cause:** Our simplified `struct cudaDeviceProp` definition has fewer fields than the real NVIDIA struct. The real struct has dozens of texture, surface, and capability fields between `clockRate` and `major`. This means:

| Field | Our struct offset | Real NVIDIA offset (v2) |
|---|---|---|
| `major` | 328 | **360** |
| `minor` | 332 | **364** |
| `multiProcessorCount` | ~312 | **356** |

ggml-cuda reads at the **real** offsets (it was compiled against real CUDA headers), so it reads zeros.

**Fix:** After filling struct fields normally (which populates our struct's offsets), we also write critical values at the raw binary offsets that ggml-cuda actually reads:

```c
uint8_t *raw = (uint8_t *)prop;
*(int *)(raw + 360) = resp.major;                         // compute capability major
*(int *)(raw + 364) = resp.minor;                         // compute capability minor
*(int *)(raw + 356) = resp.multi_processor_count;         // SM count
*(int *)(raw + 368) = resp.max_threads_per_multiprocessor;
```

**Note:** The `SAFE_PROP_MEMSET_SIZE` is set to 768 bytes to avoid stack smashing — ggml-cuda allocates the struct on the stack with `sub $0x448,%rsp` (1096 bytes), but prior attempts to memset with 2048 bytes destroyed the return address.

---

### Problem 6: TENSOR_ALIGNMENT Assertion Failure

**Symptom:** After GPU was detected and model layers scheduled to GPU, the runner crashed with:
```
GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned") failed
```

**Root Cause:** `cudaHostAlloc()` was implemented using `malloc()`, which returns 16-byte aligned memory on most systems. ggml requires at least 32-byte alignment (TENSOR_ALIGNMENT).

**Fix:** Changed to `posix_memalign()` with 4096-byte (page) alignment:
```c
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    (void)flags;
    if (!pHost) return cudaErrorInvalidValue;
    if (posix_memalign(pHost, 4096, size) != 0) {
        *pHost = NULL;
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}
```

---

### Problem 7: Windows Line Endings

**Symptom:** All Python-based patching scripts failed silently — string matching, regex, and sed-based patches all failed to find their targets despite looking correct.

**Root Cause:** The `cuda_shim.c` file on the server had Windows-style `\r\n` (CR+LF) line endings. The invisible `\r` characters caused every string comparison to fail.

**Evidence:** `cat -A` showed `^M$` at end of every line.

**Fix:** `sed -i 's/\r$//' cuda_shim.c`

---

## 3. Current Achievement State

### What Works (all verified on srv022010)

| Component | Status | Evidence |
|---|---|---|
| GPU discovery | ✅ | `initial_count=1` |
| Compute capability | ✅ | `compute=8.9` |
| Device identification | ✅ | `NVIDIA GeForce RTX 4060 Laptop GPU` |
| VRAM detection | ✅ | `total="8.0 GiB" available="6.9 GiB"` |
| Inference compute target | ✅ | `library=CUDA` (not CPU) |
| Model layer scheduling | ✅ | All 23 TinyLlama layers → CUDA0 |
| Weight loading estimate | ✅ | 571.4 MiB weights + 44 MiB KV cache + 148 MiB graph |
| cudaMalloc via proxy | ✅ | Returns 256-byte aligned GPU pointers |
| cudaHostAlloc alignment | ✅ | 4096-byte aligned via posix_memalign |
| Lazy registration | ✅ | Zero RPC during dlopen, ~500ms bootstrap |
| Symbol versioning | ✅ | `@@libcudart.so.12` tags match libggml-cuda.so |
| All 49 required symbols | ✅ | dlopen succeeds without unresolved symbols |

### What Fails

| Component | Status | Error |
|---|---|---|
| GPU inference execution | ❌ | "CUDA error" during model loading/inference |

---

## 4. THE REMAINING PROBLEM — Driver API Memory Forwarding

### The Issue

When Ollama loads the model to GPU and starts inference, it doesn't just use the CUDA Runtime API (`cuda*` functions) — it also uses the CUDA Driver API (`cu*` functions) through **libcublas.so.12**. Specifically, libcublas allocates GPU memory using the Driver API function `cuMemAlloc`, not the Runtime API function `cudaMalloc`.

Our driver shim (`cuda_driver_shim.c`) has a `cuGetProcAddress` implementation that:
1. Checks if the requested symbol exists as a real export in our library (via `dlsym(RTLD_DEFAULT, symbol)`)
2. If not found, returns a pointer to `generic_not_supported_stub` — a function that returns `CUDA_ERROR_NOT_SUPPORTED`

The problem: `cuMemAlloc`, `cuMemFree`, `cuMemcpy`, `cuMemAllocManaged`, `cuMemAllocAsync`, etc. are **not** exported by our driver shim, so they resolve to the generic stub. When libcublas calls `cuMemAlloc` through the function pointer it got from `cuGetProcAddress`, it gets `CUDA_ERROR_NOT_SUPPORTED` and the inference crashes.

### Evidence from Ollama Logs

```
[cuda-driver-shim] cuGetProcAddress("cuMemAllocManaged", v6000) → 0x7fd344b42490 [stub]
[cuda-driver-shim] cuGetProcAddress("cuMemAlloc", v3020) → 0x7fd344b42490 [stub]
[cuda-driver-shim] cuGetProcAddress("cuMemAllocPitch", v3020) → 0x7fd344b42490 [stub]
[cuda-driver-shim] cuGetProcAddress("cuMemHostAlloc", v2020) → 0x7fd344b42490 [stub]
[cuda-driver-shim] cuGetProcAddress("cuMemAllocAsync", v11020) → 0x7fd344b42490 [stub]
[cuda-driver-shim] cuGetProcAddress("cuMemAllocFromPoolAsync", v11020) → 0x7fd344b42490 [stub]
```

All memory functions point to the same generic stub address (`0x7fd344b42490`).

### What Needs to Happen

The driver shim (`src/gpu-proxy/shim/cuda_driver_shim.c`) needs **real forwarding implementations** for Driver API memory functions. These should forward through the GPU proxy daemon using the same RPC protocol that the cudart shim uses. The key functions needed:

**Critical (must forward through proxy):**
- `cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)` → same as `cudaMalloc`, forward via `GPU_CMD_MALLOC`
- `cuMemFree(CUdeviceptr dptr)` → same as `cudaFree`, forward via `GPU_CMD_FREE`
- `cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t byteCount)` → forward via `GPU_CMD_MEMCPY`
- `cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t byteCount)` → forward via `GPU_CMD_MEMCPY`
- `cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t byteCount)` → forward via `GPU_CMD_MEMCPY`
- `cuMemsetD8(CUdeviceptr dptr, unsigned char uc, size_t N)` → forward via `GPU_CMD_MEMSET`
- `cuMemsetD32(CUdeviceptr dptr, unsigned int ui, size_t N)` → forward via `GPU_CMD_MEMSET`
- `cuMemGetInfo(size_t *free, size_t *total)` → forward via `GPU_CMD_MEM_GET_INFO`
- `cuMemAllocAsync` → forward or map to synchronous `cuMemAlloc`
- `cuMemFreeAsync` → forward or map to synchronous `cuMemFree`

**Can remain stubs (return success):**
- `cuMemAllocManaged` — can map to regular `cuMemAlloc`
- `cuMemHostAlloc` — can use `posix_memalign` locally
- `cuMemAllocPitch` — can map to `cuMemAlloc` with pitch = width
- GL interop functions (`cuGL*`) — stubs are fine

### Implementation Approach

The driver shim already has a shared transport layer (`shim/transport.h` + `shim/transport.c`) that provides the same TCP/vsock connection to the daemon. The `cuGetProcAddress` function should be modified to:

1. Check for known Driver API memory functions by name
2. Return pointers to real forwarding implementations (not the generic stub)
3. These implementations use the same `rpc_call` pattern as the cudart shim

Example implementation pattern:
```c
// In cuda_driver_shim.c:
static CUresult cu_mem_alloc(CUdeviceptr *dptr, size_t bytesize)
{
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    GpuMallocRequest req = { .size = (uint64_t)bytesize };
    GpuMallocResponse resp;
    int err = transport_rpc_call(GPU_CMD_MALLOC, &req, sizeof(req),
                                 &resp, sizeof(resp), NULL);
    if (err == 0) {
        *dptr = (CUdeviceptr)resp.device_ptr;
    } else {
        *dptr = 0;
    }
    return err;
}

// In cuGetProcAddress:
if (strcmp(symbol, "cuMemAlloc") == 0 || strcmp(symbol, "cuMemAlloc_v2") == 0) {
    *pfn = (void *)cu_mem_alloc;
    return CUDA_SUCCESS;
}
```

### Files to Modify

1. **`src/gpu-proxy/shim/cuda_driver_shim.c`** — add forwarding implementations for `cuMemAlloc`, `cuMemFree`, `cuMemcpy*`, `cuMemset*`, `cuMemGetInfo`, and update `cuGetProcAddress` to return them
2. Possibly **`src/gpu-proxy/proto/gpu_proxy_proto.h`** — if new command types are needed (probably not, since the existing `GPU_CMD_MALLOC`, `GPU_CMD_FREE`, `GPU_CMD_MEMCPY` commands can be reused)

### Testing Plan

1. Build updated driver shim
2. Deploy to `/usr/local/lib/libcuda.so.1` on srv022010
3. Restart Ollama
4. Run `ollama run tinyllama "Say hello"` 
5. Check logs for `cuGetProcAddress("cuMemAlloc"...)` — should show real function address, not `[stub]`
6. If model loads without "CUDA error", GPU inference is working

---

## 5. Files and Locations Reference

### On Server (srv022010)

| Path | Description |
|---|---|
| `/tmp/gpu-proxy/cuda_shim.c` | **Production cudart shim** (1340 lines, all fixes applied) |
| `/tmp/gpu-proxy/libcudart.version` | Symbol version script |
| `/tmp/gpu-proxy/proto/gpu_proxy_proto.h` | Wire protocol header |
| `/usr/local/lib/ollama/libcudart.so.12` | Deployed cudart shim binary |
| `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90` | Symlink to deployed binary |
| `/usr/local/lib/libcuda.so.1` | Deployed driver shim (NEEDS UPDATE) |
| `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` | Ollama's CUDA backend (1.6GB, read-only) |
| `/usr/local/lib/ollama/cuda_v12/libcublas.so.12` | NVIDIA cuBLAS (read-only) |

### Repository Structure

```
src/gpu-proxy/
├── proto/
│   └── gpu_proxy_proto.h       # Wire protocol definitions
├── shim/
│   ├── cuda_shim.c             # Runtime API shim ← COMMIT UPDATED VERSION
│   ├── cuda_driver_shim.c      # Driver API shim  ← NEEDS cuMem* forwarding
│   ├── nvml_shim.c             # NVML shim
│   ├── transport.h             # Shared transport header
│   ├── transport.c             # Shared transport implementation
│   └── libcudart.version       # NEW FILE — symbol version script
├── daemon/
│   └── gpu_proxy_daemon.c      # Host-side daemon
├── Makefile                    # ← COMMIT UPDATED VERSION (add version script flags)
└── build/                      # Build output directory
```

### Ollama Service Configuration

```bash
# Service
systemctl status ollama

# Environment (set in /etc/systemd/system/ollama.service.d/override.conf)
OLLAMA_LLM_LIBRARY=cuda_v12      # Force CUDA v12 library selection
OLLAMA_DEBUG=DEBUG                # Enable debug logging

# Logs
journalctl -u ollama -f           # Follow live
journalctl -u ollama --since "5 min ago"  # Recent

# Key log patterns to grep for:
grep -E 'initial_count|inference compute|discovering|filter|compute=' 
grep -E 'CUDA error|cuGetProcAddress|cuMemAlloc|stub'
```

### Build Commands

```bash
# On server (quick rebuild + deploy):
cd /tmp/gpu-proxy
gcc -shared -fPIC -o libcudart_shim.so cuda_shim.c -ldl -lpthread \
    -Wl,-soname,libcudart.so.12 \
    -Wl,--version-script=libcudart.version
systemctl stop ollama
cp libcudart_shim.so /usr/local/lib/ollama/libcudart.so.12
cp libcudart_shim.so /usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90
ldconfig
systemctl start ollama

# Repository build (via Makefile):
make shim            # builds to build/libdecloud_cuda_shim.so
make driver-shim     # builds to build/libcuda.so.1
```

---

## 6. Key Technical Details to Remember

1. **ggml-cuda stack allocation:** `cudaDeviceProp` is allocated on the stack with only 1096 bytes (`sub $0x448,%rsp`). Never memset more than `SAFE_PROP_MEMSET_SIZE` (768 bytes).

2. **Raw byte offsets for device properties:** ggml-cuda reads `major` at byte offset 360, `minor` at 364, `multiProcessorCount` at 356, `maxThreadsPerMultiProcessor` at 368. These are hardcoded offsets from the real NVIDIA struct layout.

3. **Bootstrap timeout:** Ollama kills the bootstrap subprocess if it takes too long. Lazy registration ensures zero RPC during dlopen, keeping bootstrap under 500ms.

4. **Symbol versioning is mandatory:** Without `@@libcudart.so.12` version tags, libggml-cuda.so warns "no version information available" and the bootstrap may fail.

5. **Two API layers:** The CUDA Runtime API (`cuda*`) and CUDA Driver API (`cu*`) are separate. Our cudart shim handles the runtime layer. libcublas uses the driver layer for memory. Both must forward through the proxy for inference to work.

6. **The daemon connection:** Shim connects via vsock (CID=2, port=9999) first, falls back to TCP (`192.168.122.1:9999`). The daemon on the host must be running for any GPU operations to work.

7. **`cuGetProcAddress` is the gatekeeper:** libcudart calls `cuGetProcAddress` to get function pointers for every Driver API function. Currently our driver shim returns a generic stub for unknown functions. The fix is to return real forwarding implementations for memory operations.

---

## 7. Commit Checklist

Files to commit to the repository:

- [ ] `src/gpu-proxy/shim/cuda_shim.c` — copy from server `/tmp/gpu-proxy/cuda_shim.c`
- [ ] `src/gpu-proxy/shim/libcudart.version` — copy from server `/tmp/gpu-proxy/libcudart.version`
- [ ] `src/gpu-proxy/Makefile` — use updated version with `--version-script` flags

Then continue with:

- [ ] `src/gpu-proxy/shim/cuda_driver_shim.c` — add `cuMemAlloc`/`cuMemFree`/`cuMemcpy` forwarding
- [ ] Test full GPU inference pipeline end-to-end
