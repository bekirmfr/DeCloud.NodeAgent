# DeCloud GPU Proxy — True GPU Presence via Fat Binary Proxying

**Date:** 2026-03-01  
**Status:** Design ready for implementation  
**Context:** Current stub approach fails because `cudaFuncGetAttributes` returns fake values, causing ggml's MMQ kernel selector to crash (`mmq_x_best=0`). Root cause: fat binaries are stored locally but never uploaded to the daemon until first `cudaLaunchKernel`, so kernel attribute queries return hardcoded garbage.

---

## 1. Problem Statement

### Current Flow (Broken)

```
Application starts → libggml-cuda.so loads
  ├── __cudaRegisterFatBinary(fatbin)     → stored locally, NO RPC
  ├── __cudaRegisterFunction(ptr, name)   → stored locally, NO RPC
  ├── cudaFuncGetAttributes(ptr)          → returns FAKE hardcoded values ❌
  ├── cudaOccupancy...(ptr)               → returns FAKE value (1) ❌
  ├── [kernel selection uses fake attrs]  → mmq_x_best=0, ABORT ❌
  └── cudaLaunchKernel(ptr)               → triggers lazy upload (never reached)
```

The kernel selection algorithm in ggml calls `cudaFuncGetAttributes` on each MMQ kernel variant to determine `binaryVersion`, `maxThreadsPerBlock`, `numRegs`, and `sharedSizeBytes`. It then picks the best variant. With all-zero or identical fake values, no variant qualifies → `mmq_x_best=0` → fatal error.

### Required Flow (True GPU Presence)

```
Application starts → libggml-cuda.so loads
  ├── __cudaRegisterFatBinary(fatbin)     → stored locally (deferred, same as now)
  ├── __cudaRegisterFunction(ptr, name)   → stored locally (deferred, same as now)
  ├── cudaFuncGetAttributes(ptr)          → trigger eager upload → RPC → REAL values ✅
  ├── cudaOccupancy...(ptr, blockSize)    → RPC → REAL occupancy from daemon ✅
  ├── [kernel selection uses real attrs]  → picks correct MMQ variant ✅
  └── cudaLaunchKernel(ptr)              → module already uploaded, direct launch ✅
```

---

## 2. Design: Surgical 3-Part Change

The existing deferred upload mechanism (`ensure_module_uploaded`) already works — it uploads fatbin data to the daemon via `GPU_CMD_REGISTER_MODULE` and registers functions via `GPU_CMD_REGISTER_FUNCTION`. The daemon successfully loads modules via `cuModuleLoadData` and resolves functions via `cuModuleGetFunction`.

We need exactly three changes:

### 2.1 Trigger Eager Upload from `cudaFuncGetAttributes`

Instead of waiting for `cudaLaunchKernel`, trigger `ensure_module_uploaded` when `cudaFuncGetAttributes` is called. This is a one-line change in the shim.

### 2.2 New Protocol Command: `GPU_CMD_FUNC_GET_ATTRIBUTES`

Query real kernel attributes from the daemon where the real GPU and real kernels live.

### 2.3 New Protocol Command: `GPU_CMD_OCCUPANCY_MAX_BLOCKS`

Query real occupancy from the daemon for accurate kernel selection.

---

## 3. Protocol Additions

### 3.1 `GPU_CMD_FUNC_GET_ATTRIBUTES` (0x54)

**Request:**
```c
typedef struct __attribute__((packed)) {
    uint64_t host_func_ptr;   /* VM-side function pointer (lookup key) */
} GpuFuncGetAttributesRequest;
```

**Response:**
```c
typedef struct __attribute__((packed)) {
    int32_t  binaryVersion;          /* Compute capability (e.g., 89 for sm_89) */
    int32_t  maxThreadsPerBlock;     /* Max threads per block for this kernel */
    int32_t  numRegs;                /* Registers used per thread */
    int32_t  sharedSizeBytes;        /* Static shared memory per block */
    int32_t  constSizeBytes;         /* Constant memory used */
    int32_t  localSizeBytes;         /* Local memory per thread */
    int32_t  maxDynamicSharedSizeBytes; /* Max dynamic shared memory */
    int32_t  preferredShmemCarveout; /* Preferred L1/shared split */
    int32_t  ptxVersion;             /* PTX version */
} GpuFuncGetAttributesResponse;
```

### 3.2 `GPU_CMD_OCCUPANCY_MAX_BLOCKS` (0x55)

**Request:**
```c
typedef struct __attribute__((packed)) {
    uint64_t host_func_ptr;   /* VM-side function pointer (lookup key) */
    int32_t  blockSize;       /* Block size to query */
    uint64_t dynamicSMemSize; /* Dynamic shared memory per block */
    uint32_t flags;           /* Flags (usually 0) */
} GpuOccupancyMaxBlocksRequest;
```

**Response:**
```c
typedef struct __attribute__((packed)) {
    int32_t numBlocks;        /* Max active blocks per SM */
} GpuOccupancyMaxBlocksResponse;
```

---

## 4. Implementation Changes

### 4.1 `proto/gpu_proxy_proto.h` — Add Command IDs and Structs

```c
/* Add to GpuCommand enum */
GPU_CMD_FUNC_GET_ATTRIBUTES    = 0x54,
GPU_CMD_OCCUPANCY_MAX_BLOCKS   = 0x55,

/* Add request/response structs (see Section 3) */
```

### 4.2 `shim/cuda_shim.c` — Replace Stubs with RPC Calls

#### `cudaFuncGetAttributes` — Before:
```c
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes_t *attr, const void *func)
{
    (void)func;
    if (attr) { memset(attr, 0, sizeof(*attr));
                attr->binaryVersion = 89;
                attr->maxThreadsPerBlock = 1024;
                attr->numRegs = 32;
                attr->maxDynamicSharedSizeBytes = 65536; }
    return cudaSuccess;
}
```

#### `cudaFuncGetAttributes` — After:
```c
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes_t *attr, const void *func)
{
    if (!attr) return cudaErrorInvalidValue;

    /* Find registered function */
    RegisteredFunction *rf = find_registered_function((uint64_t)(uintptr_t)func);
    if (!rf) {
        SHIM_LOG("cudaFuncGetAttributes: unknown function %p", func);
        return cudaErrorInvalidDeviceFunction;
    }

    /* Ensure module is uploaded and function registered on daemon */
    if (!rf->registered) {
        ensure_module_uploaded(rf->module_index);
        if (!rf->registered) {
            SHIM_LOG("cudaFuncGetAttributes: eager upload failed for %s",
                     rf->device_name);
            return cudaErrorInvalidDeviceFunction;
        }
    }

    /* RPC to daemon for real attributes */
    GpuFuncGetAttributesRequest req = {
        .host_func_ptr = (uint64_t)(uintptr_t)func,
    };
    GpuFuncGetAttributesResponse resp;
    memset(&resp, 0, sizeof(resp));

    int err = rpc_call(GPU_CMD_FUNC_GET_ATTRIBUTES,
                       &req, sizeof(req), &resp, sizeof(resp), NULL);
    if (err != 0) {
        SHIM_LOG("cudaFuncGetAttributes: RPC failed (err=%d)", err);
        return (cudaError_t)err;
    }

    /* Populate caller's struct */
    memset(attr, 0, sizeof(*attr));
    attr->binaryVersion          = resp.binaryVersion;
    attr->maxThreadsPerBlock     = resp.maxThreadsPerBlock;
    attr->numRegs                = resp.numRegs;
    attr->sharedSizeBytes        = resp.sharedSizeBytes;
    attr->constSizeBytes         = resp.constSizeBytes;
    attr->localSizeBytes         = resp.localSizeBytes;
    attr->maxDynamicSharedSizeBytes = resp.maxDynamicSharedSizeBytes;
    attr->preferredShmemCarveout = resp.preferredShmemCarveout;
    attr->ptxVersion             = resp.ptxVersion;

    SHIM_LOG("cudaFuncGetAttributes(%s): binary=%d maxThreads=%d regs=%d",
             rf->device_name, resp.binaryVersion,
             resp.maxThreadsPerBlock, resp.numRegs);
    return cudaSuccess;
}
```

#### `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` — After:
```c
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize,
    size_t dynamicSMemSize, unsigned int flags)
{
    if (!numBlocks) return cudaErrorInvalidValue;

    RegisteredFunction *rf = find_registered_function((uint64_t)(uintptr_t)func);
    if (!rf) {
        *numBlocks = 1; /* safe fallback */
        return cudaSuccess;
    }

    if (!rf->registered) {
        ensure_module_uploaded(rf->module_index);
        if (!rf->registered) {
            *numBlocks = 1;
            return cudaSuccess;
        }
    }

    GpuOccupancyMaxBlocksRequest req = {
        .host_func_ptr   = (uint64_t)(uintptr_t)func,
        .blockSize       = blockSize,
        .dynamicSMemSize = (uint64_t)dynamicSMemSize,
        .flags           = flags,
    };
    GpuOccupancyMaxBlocksResponse resp;
    memset(&resp, 0, sizeof(resp));

    int err = rpc_call(GPU_CMD_OCCUPANCY_MAX_BLOCKS,
                       &req, sizeof(req), &resp, sizeof(resp), NULL);
    if (err != 0) {
        *numBlocks = 1; /* safe fallback on RPC failure */
        return cudaSuccess;
    }

    *numBlocks = resp.numBlocks;
    return cudaSuccess;
}
```

### 4.3 `daemon/gpu_proxy_daemon.c` — Add Handlers

#### `handle_func_get_attributes`:
```c
static int handle_func_get_attributes(ConnectionCtx *ctx,
                                       const void *payload,
                                       uint32_t payload_len)
{
    if (payload_len < sizeof(GpuFuncGetAttributesRequest)) {
        return send_response(ctx->fd, GPU_CMD_FUNC_GET_ATTRIBUTES, -1, NULL, 0);
    }

    GpuFuncGetAttributesRequest req;
    memcpy(&req, payload, sizeof(req));

    /* Find the function by host_func_ptr */
    FunctionSlot *fs = NULL;
    for (int i = 0; i < MAX_FUNCTIONS; i++) {
        if (ctx->functions[i].in_use &&
            ctx->functions[i].host_func_ptr == req.host_func_ptr) {
            fs = &ctx->functions[i];
            break;
        }
    }

    if (!fs) {
        LOG_ERR("CID %u: func_get_attributes: unknown func_ptr 0x%lx",
                ctx->peer_cid, (unsigned long)req.host_func_ptr);
        return send_response(ctx->fd, GPU_CMD_FUNC_GET_ATTRIBUTES,
                             (int32_t)cudaErrorInvalidDeviceFunction, NULL, 0);
    }

    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);

    /* Query real attributes using Driver API */
    GpuFuncGetAttributesResponse resp;
    memset(&resp, 0, sizeof(resp));

    int val;
    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fs->cu_func);
    resp.maxThreadsPerBlock = val;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_NUM_REGS, fs->cu_func);
    resp.numRegs = val;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fs->cu_func);
    resp.sharedSizeBytes = val;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, fs->cu_func);
    resp.constSizeBytes = val;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fs->cu_func);
    resp.localSizeBytes = val;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, fs->cu_func);
    resp.maxDynamicSharedSizeBytes = val;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, fs->cu_func);
    resp.preferredShmemCarveout = val;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_BINARY_VERSION, fs->cu_func);
    resp.binaryVersion = val;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_PTX_VERSION, fs->cu_func);
    resp.ptxVersion = val;

    LOG_DBG("CID %u: func_get_attributes: binary=%d maxThreads=%d regs=%d",
            ctx->peer_cid, resp.binaryVersion,
            resp.maxThreadsPerBlock, resp.numRegs);

    return send_response(ctx->fd, GPU_CMD_FUNC_GET_ATTRIBUTES, 0,
                         &resp, sizeof(resp));
}
```

#### `handle_occupancy_max_blocks`:
```c
static int handle_occupancy_max_blocks(ConnectionCtx *ctx,
                                        const void *payload,
                                        uint32_t payload_len)
{
    if (payload_len < sizeof(GpuOccupancyMaxBlocksRequest)) {
        return send_response(ctx->fd, GPU_CMD_OCCUPANCY_MAX_BLOCKS, -1, NULL, 0);
    }

    GpuOccupancyMaxBlocksRequest req;
    memcpy(&req, payload, sizeof(req));

    FunctionSlot *fs = NULL;
    for (int i = 0; i < MAX_FUNCTIONS; i++) {
        if (ctx->functions[i].in_use &&
            ctx->functions[i].host_func_ptr == req.host_func_ptr) {
            fs = &ctx->functions[i];
            break;
        }
    }

    if (!fs) {
        GpuOccupancyMaxBlocksResponse resp = { .numBlocks = 1 };
        return send_response(ctx->fd, GPU_CMD_OCCUPANCY_MAX_BLOCKS, 0,
                             &resp, sizeof(resp));
    }

    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);

    int numBlocks = 0;
    CUresult cr = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &numBlocks, fs->cu_func, req.blockSize,
        (size_t)req.dynamicSMemSize, req.flags);

    if (cr != CUDA_SUCCESS) {
        numBlocks = 1; /* safe fallback */
    }

    GpuOccupancyMaxBlocksResponse resp = { .numBlocks = numBlocks };
    return send_response(ctx->fd, GPU_CMD_OCCUPANCY_MAX_BLOCKS, 0,
                         &resp, sizeof(resp));
}
```

#### Add to command dispatch switch:
```c
case GPU_CMD_FUNC_GET_ATTRIBUTES:
    rc = handle_func_get_attributes(ctx, buf, hdr.payload_len);
    break;
case GPU_CMD_OCCUPANCY_MAX_BLOCKS:
    rc = handle_occupancy_max_blocks(ctx, buf, hdr.payload_len);
    break;
```

---

## 5. What This Eliminates

With true GPU presence via proxied attributes:

| Hack | Status | Why |
|------|--------|-----|
| `GGML_CUDA_FORCE_MMQ=1` env var | **No longer needed** | Real cuBLAS stubs still return NOT_SUPPORTED, but kernel selection works with real attributes |
| `GGML_CUDA_DISABLE_GRAPHS=1` env var | **Still needed** | CUDA graphs can't work across a proxy — capture is local but execution is remote |
| cuBLAS stub library | **Still needed** | cuBLAS internal init requires `cuGetExportTable` which can't be proxied. Stubs prevent crash, MMQ provides the actual compute |
| Constructor `setenv()` | **Can be simplified** | Only need `GGML_CUDA_DISABLE_GRAPHS=1`, not the MMQ force |
| Fake `cudaFuncGetAttributes` values | **Eliminated** | Real values from daemon |
| Fake occupancy values | **Eliminated** | Real values from daemon |

---

## 6. CUDA Graphs Strategy

CUDA graphs capture a sequence of kernel launches and replay them as a single optimized unit. This is fundamentally incompatible with a proxy because:

1. **Capture** records host-side API calls into a graph object
2. **Instantiate** compiles the graph into an executable
3. **Launch** replays the entire sequence on the GPU

With a proxy, each kernel launch is an individual RPC. The proxy executes kernels eagerly — there's no host-side GPU state to "capture." Therefore:

- `cudaStreamBeginCapture` → return `cudaSuccess` (no-op)
- `cudaStreamEndCapture` → return `cudaSuccess` with dummy graph handle
- `cudaGraphInstantiate` → return `cudaSuccess` with dummy exec handle
- `cudaGraphLaunch` → return `cudaSuccess` (no-op — kernels already ran eagerly)
- `cudaStreamIsCapturing` → always return `cudaStreamCaptureStatusNone`
- `cudaGraphExecUpdate` → return `cudaSuccess`
- `cudaGraphDestroy` / `cudaGraphExecDestroy` → return `cudaSuccess`

This is safe because ggml checks `GGML_CUDA_DISABLE_GRAPHS` and falls back to eager execution. The `setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1)` in the constructor ensures ggml never attempts graph capture. The graph stubs returning success are a safety net.

---

## 7. Build & Test Plan

### Build
```bash
cd /opt/decloud/DeCloud.NodeAgent/src/gpu-proxy

# Edit the three files
# 1. proto/gpu_proxy_proto.h  (add command IDs + structs)
# 2. shim/cuda_shim.c         (replace stubs with RPC calls)
# 3. daemon/gpu_proxy_daemon.c (add handlers)

# Rebuild daemon (requires CUDA on host)
make daemon

# Rebuild shims (compat build via Docker)
make all-shims-compat

# Deploy daemon
sudo systemctl stop gpu-proxy-daemon
sudo cp build/gpu-proxy-daemon /usr/local/bin/
sudo systemctl start gpu-proxy-daemon

# Deploy shims to 9p share
sudo cp build/libdecloud_cuda_shim-compat.so /usr/local/lib/decloud-gpu-shim/libdecloud_cuda_shim.so
```

### Test on VM
```bash
# Remount 9p share
sudo umount /run/decloud 2>/dev/null
sudo mount -t 9p -o trans=virtio,version=9p2000.L decloud-shim /run/decloud

# Install updated shim
sudo cp /run/decloud/libdecloud_cuda_shim.so /usr/local/lib/
sudo cp /run/decloud/libdecloud_cuda_shim.so /usr/local/lib/ollama/cuda_v12/libcudart.so.12

# Restart and test
sudo systemctl restart ollama
ollama run llama3.2:1b "Say hello in exactly 5 words" --verbose 2>&1 | tail -15
```

### Expected Log Output
```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
[cublas-stub] cublasCreate_v2 → dummy handle
[cudart-shim] cudaFuncGetAttributes(kernel_name): binary=89 maxThreads=1024 regs=32
load_tensors: offloaded 17/17 layers to GPU
load_tensors: CUDA0 model buffer size = 1252.41 MiB
```

### Verification
```bash
# GPU memory should show model loaded
nvidia-smi  # On host — should show >1.2GB VRAM used

# Token rate should be GPU-class (80-150 tok/s for 1B model)
# vs CPU-class (~20 tok/s) which we saw before
```

---

## 8. Future: What Becomes Truly Generic

With this change, the proxy supports any CUDA application that uses:
- `__cudaRegisterFatBinary` / `__cudaRegisterFunction` (nvcc-compiled code)
- `cudaLaunchKernel` (kernel execution)
- `cudaFuncGetAttributes` (kernel selection)
- `cudaMalloc` / `cudaMemcpy` (memory management)
- Streams and events (async execution)

This covers: ggml (llama.cpp, Ollama), PyTorch custom ops, TensorFlow Lite, Triton-compiled kernels, Flash Attention, any nvcc-compiled CUDA program.

What still requires stubs: cuBLAS, cuDNN, cuFFT, cuSPARSE (use `cuGetExportTable` which is private NVIDIA driver internals). For ggml, the MMQ path avoids cuBLAS entirely, making this a non-issue.
