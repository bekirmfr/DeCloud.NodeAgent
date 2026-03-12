/*
 * DeCloud CUDA PyTorch Compatibility Stubs (libcuda_pytorch_stubs.so)
 *
 * Provides the CUDA Runtime symbols required by PyTorch 2.3.x / Stable Diffusion
 * Forge that are NOT exported by the main DeCloud CUDA shim (libdecloud_cuda_shim.so).
 *
 * Affected libraries (PyTorch 2.3.1+cu121, Forge f2.0.1):
 *   libtorch_cuda.so          — cudaGraphInstantiateWithFlags, cudaGraphGetNodes, ...
 *   libbitsandbytes_cuda121.so — cudaMemPrefetchAsync
 *   libc10_cuda.so            — cudaFreeAsync, cudaMallocAsync, cudaStreamCreateWithPriority,
 *                               cudaDeviceGetStreamPriorityRange, + 9 more (see below)
 *
 * Symbols are exported with @@libcudart.so.12 version tags so the dynamic
 * linker resolves them correctly. Load AFTER libdecloud_cuda_shim.so in LD_PRELOAD.
 *
 * Discovery:
 *   Initial 16 symbols: 2026-03-06 — libtorch_cuda.so / libbitsandbytes scan
 *   Additional 13 symbols: 2026-03-07 — libc10_cuda.so scan:
 *     objdump -T libc10_cuda.so | awk '/\*UND\* && /cuda/{print $NF}' | sort -u
 *     Diff against: nm -D libdecloud_cuda_shim.so libcuda_pytorch_stubs.so | grep ' T '
 *   See: GPU_PROXY_DEBUGGING_JOURNAL.md (Stable Diffusion Forge session)
 *
 * Build:
 *   make pytorch-stub          (host glibc)
 *   make pytorch-stub-compat   (Docker, glibc 2.31 — preferred)
 *
 * Deployment:
 *   /usr/local/lib/decloud-gpu-shim/libcuda_pytorch_stubs.so (9p share → VM)
 *
 * LD_PRELOAD order in /etc/decloud/gpu-proxy.env:
 *   LD_PRELOAD=libdecloud_cuda_shim.so:libcuda_pytorch_stubs.so
 *
 * CRITICAL: Our shim must come FIRST so it wins symbol resolution for
 * cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags and related calls.
 * Stubs second — they supply only symbols our shim does NOT export.
 */

#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Types ───────────────────────────────────────────────────────────── */
typedef int            cudaError_t;
typedef void          *cudaStream_t;
typedef void          *cudaEvent_t;
typedef void          *cudaGraph_t;
typedef void          *cudaGraphNode_t;
typedef unsigned long long cudaGraphExec_t;
typedef void          *cudaMemPool_t;
typedef int            cudaMemPoolAttr;

/* cudaStreamCallback_t — used by cudaStreamAddCallback */
typedef void (*cudaStreamCallback_t)(cudaStream_t stream,
                                     cudaError_t  status,
                                     void        *userData);

/* cudaStreamCaptureInfo_v2 status enum value */
#define cudaStreamCaptureStatusNone 0

/* cudaMemAccessDesc — used by cudaMemPoolSetAccess (opaque for stubs) */
typedef struct {
    struct { int type; int id; } location;
    int flags;
} cudaMemAccessDesc;

/* cudaPointerAttributes — reported type values */
typedef struct {
    int    type;           /* 2 = cudaMemoryTypeDevice */
    int    device;
    void  *devicePointer;
    void  *hostPointer;
    int    isManaged;
    unsigned allocationFlags;
} cudaPointerAttributes;

/* IPC handles (64-byte opaque blobs) */
typedef uint8_t cudaIpcEventHandle_t[64];
typedef uint8_t cudaIpcMemHandle_t[64];

/* Error codes */
#define cudaSuccess              0
#define cudaErrorInvalidValue    1
#define cudaErrorNotSupported   71

/* ══════════════════════════════════════════════════════════════════════
 * 1. __cudaInitModule
 *    Called by the CUDA module loader (CUDA 12+) during lazy init.
 *    Returning 1 signals "module already initialized".
 * ══════════════════════════════════════════════════════════════════════ */
int __cudaInitModule(void **fatCubinHandle)
{
    (void)fatCubinHandle;
    return 1;
}

/* ══════════════════════════════════════════════════════════════════════
 * 2. cudaDeviceGetPCIBusId
 *    Returns a fake PCI bus ID string. PyTorch uses this for device
 *    identification; the exact value doesn't matter for proxy mode.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    (void)device;
    if (pciBusId && len > 0)
        snprintf(pciBusId, len, "0000:00:00.0");
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 3. cudaPointerGetAttributes
 *    Reports every pointer as device memory on device 0.
 *    PyTorch uses this to decide copy direction; device→device copy
 *    paths in the proxy go through cudaMemcpy which is fully proxied.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attrs,
                                      const void *ptr)
{
    if (!attrs) return cudaErrorInvalidValue;
    memset(attrs, 0, sizeof(*attrs));
    attrs->type          = 2;   /* cudaMemoryTypeDevice */
    attrs->device        = 0;
    attrs->devicePointer = (void *)ptr;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 4. cudaHostGetDevicePointer
 *    Mapped pinned memory — not supported in proxy mode.
 *    PyTorch falls back to explicit cudaMemcpy on cudaErrorNotSupported.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                      unsigned int flags)
{
    (void)pHost; (void)flags;
    if (pDevice) *pDevice = NULL;
    return cudaErrorNotSupported;
}

/* ══════════════════════════════════════════════════════════════════════
 * 5. cudaMemPrefetchAsync
 *    Hints the driver to prefetch managed memory to a device/host.
 *    No managed memory exists in proxy mode — safe no-op.
 *    Required by: libbitsandbytes_cuda121.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                  int dstDevice, cudaStream_t stream)
{
    (void)devPtr; (void)count; (void)dstDevice; (void)stream;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 6. cudaStreamGetPriority
 *    Returns priority 0 (default, no priority differentiation in proxy).
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority)
{
    (void)stream;
    if (priority) *priority = 0;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 7. cudaStreamGetCaptureInfo_v2
 *    Graph capture not supported in proxy mode.
 *    Returns status = cudaStreamCaptureStatusNone so callers skip
 *    any capture-related paths.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream,
                                         int         *captureStatus,
                                         unsigned long long *id,
                                         cudaGraph_t *graph,
                                         const cudaGraphNode_t **deps,
                                         size_t      *numDeps)
{
    (void)stream;
    if (captureStatus) *captureStatus = cudaStreamCaptureStatusNone;
    if (id)      *id      = 0;
    if (graph)   *graph   = NULL;
    if (deps)    *deps    = NULL;
    if (numDeps) *numDeps = 0;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 8. cudaStreamAddCallback
 *    Execute callback immediately (synchronously) and return success.
 *    Proxy mode has no async stream execution to hook into.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                   cudaStreamCallback_t callback,
                                   void *userData, unsigned int flags)
{
    (void)stream; (void)flags;
    if (callback) callback(stream, cudaSuccess, userData);
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 9. cudaProfilerStart / cudaProfilerStop
 *    Profiling not available in proxy mode — safe no-ops.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
cudaError_t cudaProfilerStop(void)  { return cudaSuccess; }

/* ══════════════════════════════════════════════════════════════════════
 * 10. cudaIpcGetEventHandle / cudaIpcOpenEventHandle
 *     IPC event sharing not supported in proxy mode.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
                                   cudaEvent_t event)
{
    (void)event;
    if (handle) memset(handle, 0, sizeof(*handle));
    return cudaErrorNotSupported;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
                                    cudaIpcEventHandle_t handle)
{
    (void)handle;
    if (event) *event = NULL;
    return cudaErrorNotSupported;
}

/* ══════════════════════════════════════════════════════════════════════
 * 11. cudaIpcGetMemHandle
 *     IPC memory sharing not supported in proxy mode.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
{
    (void)devPtr;
    if (handle) memset(handle, 0, sizeof(*handle));
    return cudaErrorNotSupported;
}

/* ══════════════════════════════════════════════════════════════════════
 * 12. cudaGraphInstantiateWithFlags
 *     CUDA graphs not supported in proxy mode — return empty exec handle.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec,
                                           cudaGraph_t graph,
                                           unsigned long long flags)
{
    (void)graph; (void)flags;
    if (pGraphExec) *pGraphExec = 0;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 13. cudaGraphGetNodes
 *     Returns empty node list — no real graph in proxy mode.
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaGraphGetNodes(cudaGraph_t graph,
                               cudaGraphNode_t *nodes,
                               size_t *numNodes)
{
    (void)graph; (void)nodes;
    if (numNodes) *numNodes = 0;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 14. cudaGraphDebugDotPrint
 *     Writes a DOT-format graph description to a file.
 *     No-op in proxy mode (no real graph to dump).
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph,
                                    const char *path,
                                    unsigned int flags)
{
    (void)graph; (void)path; (void)flags;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 15. cudaIpcOpenMemHandle
 *     IPC memory handle open — not supported in proxy mode.
 *     Required by: libc10_cuda.so (PyTorch 2.3.x)
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle,
                                  unsigned int flags)
{
    (void)handle; (void)flags;
    if (devPtr) *devPtr = NULL;
    return cudaErrorNotSupported;
}

/* ══════════════════════════════════════════════════════════════════════
 * 16. cudaIpcCloseMemHandle
 *     IPC memory handle close — no-op since Open always fails.
 *     Required by: libc10_cuda.so (PyTorch 2.3.x)
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaIpcCloseMemHandle(void *devPtr)
{
    (void)devPtr;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * Symbols below discovered 2026-03-07 via libc10_cuda.so scan:
 *   objdump -T libc10_cuda.so | awk '/\*UND\* && /cuda/{print $NF}' | sort -u
 *   Missing from shim + old stubs confirmed by nm diff.
 * ══════════════════════════════════════════════════════════════════════ */

/* ══════════════════════════════════════════════════════════════════════
 * 17. cudaDeviceGetDefaultMemPool
 *     Returns a sentinel pool handle. PyTorch uses this for async alloc
 *     fallback paths; we redirect cudaMallocAsync → cudaMalloc anyway.
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *pool, int device)
{
    (void)device;
    if (pool) *pool = (cudaMemPool_t)0xdeadbeef;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 18. cudaDeviceGetStreamPriorityRange
 *     No stream priority differentiation in proxy mode — both limits 0.
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                              int *greatestPriority)
{
    if (leastPriority)    *leastPriority    = 0;
    if (greatestPriority) *greatestPriority = 0;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 19. cudaFreeAsync
 *     Async free — redirect to synchronous cudaFree since all proxy
 *     operations are already serialised over the vsock/TCP channel.
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaFreeAsync(void *ptr, cudaStream_t stream)
{
    (void)stream;
    extern cudaError_t cudaFree(void *);
    return cudaFree(ptr);
}

/* ══════════════════════════════════════════════════════════════════════
 * 20. cudaMallocAsync
 *     Async malloc — redirect to synchronous cudaMalloc (same reason
 *     as cudaFreeAsync above).
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaMallocAsync(void **ptr, size_t size, cudaStream_t stream)
{
    (void)stream;
    extern cudaError_t cudaMalloc(void **, size_t);
    return cudaMalloc(ptr, size);
}

/* ══════════════════════════════════════════════════════════════════════
 * 21. cudaMemPoolGetAttribute
 *     Memory pool attribute query — return 0 for all attributes.
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t pool,
                                     cudaMemPoolAttr attr,
                                     void *value)
{
    (void)pool; (void)attr;
    if (value) *(int *)value = 0;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 22. cudaMemPoolSetAccess
 *     Memory pool access control — no-op (single device in proxy mode).
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaMemPoolSetAccess(cudaMemPool_t pool,
                                  const cudaMemAccessDesc *descList,
                                  size_t count)
{
    (void)pool; (void)descList; (void)count;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 23. cudaMemPoolSetAttribute
 *     Memory pool attribute setter — no-op.
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                     cudaMemPoolAttr attr,
                                     void *value)
{
    (void)pool; (void)attr; (void)value;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 24. cudaMemPoolTrimTo
 *     Trim pool reserved memory — no-op in proxy mode.
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool, size_t minBytesToKeep)
{
    (void)pool; (void)minBytesToKeep;
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 25. cudaStreamCreateWithPriority
 *     Priority ignored — delegate to regular cudaStreamCreate.
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                          unsigned int flags,
                                          int priority)
{
    (void)flags; (void)priority;
    extern cudaError_t cudaStreamCreate(cudaStream_t *);
    return cudaStreamCreate(pStream);
}

/* ══════════════════════════════════════════════════════════════════════
 * 26. cudaStreamQuery
 *     Reports every stream as complete (all proxy ops are synchronous).
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    (void)stream;
    return cudaSuccess; /* cudaSuccess == stream complete */
}

/* ══════════════════════════════════════════════════════════════════════
 * 27. cudaThreadExchangeStreamCaptureMode
 *     Graph capture mode exchange — no capture in proxy, return 0.
 *     Required by: libc10_cuda.so
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode)
{
    if (mode) *mode = 0; /* cudaStreamCaptureModeGlobal */
    return cudaSuccess;
}

/* ══════════════════════════════════════════════════════════════════════
 * 28. cudaEventQuery
 *     All proxy ops are synchronous — every event is already complete.
 *     Required by: libc10_cuda.so (PyTorch 2.x)
 *     Discovered: 2026-03-08, libc10_cuda.so + libtorch_cuda.so scan
 * ══════════════════════════════════════════════════════════════════════ */
cudaError_t cudaEventQuery(cudaEvent_t event)
{
    (void)event;
    return cudaSuccess; /* cudaSuccess == event complete */
}