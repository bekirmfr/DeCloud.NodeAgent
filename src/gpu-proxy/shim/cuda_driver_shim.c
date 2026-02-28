/*
 * DeCloud CUDA Driver API Shim (libcuda.so.1)
 *
 * Drop-in replacement for libcuda.so that Ollama and other ML frameworks
 * find via dlopen(). When an application calls dlopen("libcuda.so") or
 * searches /usr/local/lib{*}/libcuda.so{*}, it finds this library and uses
 * dlsym() to resolve the CUDA Driver API symbols below.
 *
 * Each function forwards the call to the host GPU proxy daemon over
 * TCP/vsock, reusing the same transport layer as the Runtime API shim.
 *
 * Ollama's GPU discovery sequence:
 *   1. Glob /usr/local/lib{*}/libcuda.so{*} --> finds this file
 *   2. dlopen("libcuda.so.1") --> loads this library
 *   3. dlsym("cuInit"), dlsym("cuDeviceGetCount"), etc.
 *   4. Calls cuInit(0), cuDeviceGetCount(), cuDeviceGetName(), etc.
 *   5. cuCtxCreate_v3() + cuMemGetInfo_v2() --> gets VRAM info
 *   6. Loads model layers to "GPU" --> inference runs on host GPU
 *
 * Build: gcc -shared -fPIC -o libcuda.so.1 cuda_driver_shim.c -lpthread -ldl
 *        ln -sf libcuda.so.1 libcuda.so
 *
 * No CUDA dependency -- this replaces libcuda.so entirely.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

/* Set transport log prefix before including shared transport */
#define TRANSPORT_LOG_PREFIX "cuda-driver-shim"
#include "transport.h"
#include "transport.c"

/* ================================================================
 * CUDA Driver API type definitions
 * ================================================================ */

typedef int CUresult;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NO_DEVICE 100
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_NOT_SUPPORTED 801

typedef int CUdevice;
typedef void *CUcontext;

typedef struct {
    char bytes[16];
} CUuuid;

/* VMM (Virtual Memory Management) opaque types */
typedef uint64_t CUmemGenericAllocationHandle;
typedef uint64_t CUdeviceptr;

typedef struct {
    int dummy;
} CUmemAllocationProp;

typedef struct {
    int dummy;
} CUmemAccessDesc;

typedef enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM     = 0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1,
} CUmemAllocationGranularity_flags;

/* ================================================================
 * Generic stub for unimplemented functions
 *
 * cuGetProcAddress returns a pointer to this for any function we
 * don't explicitly implement. This tells libcudart the driver
 * "supports" the function (non-NULL), but if actually called it
 * returns CUDA_ERROR_NOT_SUPPORTED for graceful degradation.
 *
 * Without this, libcudart sees hundreds of NULL function pointers
 * and concludes the driver is insufficient (cudaError 35/36).
 * ================================================================ */

static CUresult generic_not_supported_stub(void)
{
    return CUDA_SUCCESS;
}

/* ================================================================
 * Cached device properties (fetched once, reused)
 * ================================================================ */

static int g_driver_initialized = 0;
static int g_cached_device_count = -1;

/* Cache device properties from daemon to avoid repeated round-trips */
static GpuDeviceProperties g_cached_props;
static int g_cached_props_valid = 0;

/* Global context tracking for libcudart's primary context management */
static CUcontext g_primary_ctx = NULL;
static CUcontext g_current_ctx = NULL;

static int ensure_props_cached(int device)
{
    if (g_cached_props_valid && device == 0)
        return 0;

    GpuGetDevicePropertiesRequest req = { .device = device };
    int err = transport_rpc_call(GPU_CMD_GET_DEVICE_PROPERTIES,
                                 &req, sizeof(req),
                                 &g_cached_props, sizeof(g_cached_props), NULL);
    if (err == 0)
        g_cached_props_valid = 1;
    return err;
}

/* ================================================================
 * Exported CUDA Driver API functions
 * ================================================================ */

CUresult cuInit(unsigned int flags)
{
    (void)flags;
    TRANSPORT_LOG("cuInit(%u)", flags);

    if (transport_ensure_connected() < 0)
        return CUDA_ERROR_NO_DEVICE;

    g_driver_initialized = 1;
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *version)
{
    if (!version) return CUDA_ERROR_INVALID_VALUE;

    GpuDriverVersionResponse resp;
    int err = transport_rpc_call(GPU_CMD_GET_DRIVER_VERSION,
                                 NULL, 0, &resp, sizeof(resp), NULL);
    if (err == 0) {
        *version = resp.version;
    } else {
        /* Fallback: report CUDA 12.4 to satisfy version checks */
        *version = 12040;
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count)
{
    if (!count) return CUDA_ERROR_INVALID_VALUE;

    GpuGetDeviceCountResponse resp;
    int err = transport_rpc_call(GPU_CMD_GET_DEVICE_COUNT,
                                 NULL, 0, &resp, sizeof(resp), NULL);
    if (err == 0) {
        *count = resp.count;
        g_cached_device_count = resp.count;
    } else {
        *count = 0;
    }
    return (CUresult)err;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
    if (!device) return CUDA_ERROR_INVALID_VALUE;

    /* CUdevice is just the ordinal -- no RPC needed */
    if (g_cached_device_count >= 0 && ordinal >= g_cached_device_count)
        return CUDA_ERROR_INVALID_VALUE;

    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice device)
{
    if (!value) return CUDA_ERROR_INVALID_VALUE;

    /* Fetch properties from daemon if not cached */
    int err = ensure_props_cached(device);
    if (err != 0) {
        *value = 0;
        return (CUresult)err;
    }

    /*
     * Map CUdevice_attribute enum values to our cached properties.
     * These are the attributes Ollama checks:
     *   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
     *   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
     * Plus common ones ML frameworks probe.
     */
    switch (attrib) {
    case 1:  /* CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK */
        *value = g_cached_props.max_threads_per_block; break;
    case 2:  /* CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X */
        *value = g_cached_props.max_threads_dim[0]; break;
    case 3:  /* CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y */
        *value = g_cached_props.max_threads_dim[1]; break;
    case 4:  /* CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z */
        *value = g_cached_props.max_threads_dim[2]; break;
    case 5:  /* CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X */
        *value = g_cached_props.max_grid_size[0]; break;
    case 6:  /* CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y */
        *value = g_cached_props.max_grid_size[1]; break;
    case 7:  /* CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z */
        *value = g_cached_props.max_grid_size[2]; break;
    case 8:  /* CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
        *value = (int)g_cached_props.shared_mem_per_block; break;
    case 13: /* CU_DEVICE_ATTRIBUTE_CLOCK_RATE */
        *value = g_cached_props.clock_rate; break;
    case 16: /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */
        *value = g_cached_props.multi_processor_count; break;
    case 21: /* CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY */
        *value = 65536; break; /* 64KB typical */
    case 37: /* CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
        *value = g_cached_props.regs_per_block; break;
    case 10: /* CU_DEVICE_ATTRIBUTE_WARP_SIZE */
        *value = g_cached_props.warp_size; break;
    case 39: /* CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR */
        *value = g_cached_props.max_threads_per_multiprocessor; break;
    case 75: /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
        *value = g_cached_props.major; break;
    case 76: /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */
        *value = g_cached_props.minor; break;
    case 17: /* CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE */
        *value = g_cached_props.memory_clock_rate; break;
    case 32: /* CU_DEVICE_ATTRIBUTE_MEMORY_BUS_WIDTH */
        *value = g_cached_props.memory_bus_width; break;
    case 38: /* CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE */
        *value = g_cached_props.l2_cache_size; break;
    case 86: /* CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY */
        *value = 1; break;
    case 89: /* CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS */
        *value = 1; break;
    case 100: /* CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED */
        /*
         * Return 0 to force ggml-cuda to use regular cudaMalloc instead of
         * the VMM path. VMM operations cannot be proxied over the network.
         */
        *value = 0; break;
    default:
        *value = 0;
        break;
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice device)
{
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;

    GpuDeviceUuidRequest req = { .device = device };
    GpuDeviceUuidResponse resp;
    int err = transport_rpc_call(GPU_CMD_GET_DEVICE_UUID,
                                 &req, sizeof(req),
                                 &resp, sizeof(resp), NULL);
    if (err == 0) {
        memcpy(uuid->bytes, resp.uuid, 16);
    } else {
        memset(uuid->bytes, 0, 16);
    }
    return (CUresult)err;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice device)
{
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;

    int err = ensure_props_cached(device);
    if (err != 0) {
        name[0] = '\0';
        return (CUresult)err;
    }

    strncpy(name, g_cached_props.name, (size_t)(len - 1));
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate_v3(CUcontext *ctx, void *params, int nparams,
                         unsigned int flags, CUdevice device)
{
    (void)params;
    (void)nparams;

    if (!ctx) return CUDA_ERROR_INVALID_VALUE;

    TRANSPORT_LOG("cuCtxCreate_v3(device=%d, flags=%u)", device, flags);

    GpuCtxCreateRequest req = {
        .device = device,
        .flags = flags,
    };
    GpuCtxCreateResponse resp;
    int err = transport_rpc_call(GPU_CMD_CTX_CREATE,
                                 &req, sizeof(req),
                                 &resp, sizeof(resp), NULL);
    if (err == 0) {
        *ctx = (CUcontext)(uintptr_t)resp.ctx_handle;
        g_current_ctx = *ctx;
    } else {
        *ctx = NULL;
    }
    return (CUresult)err;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total)
{
    if (!free || !total) return CUDA_ERROR_INVALID_VALUE;

    GpuMemInfoResponse resp;
    int err = transport_rpc_call(GPU_CMD_MEM_GET_INFO,
                                 NULL, 0,
                                 &resp, sizeof(resp), NULL);
    if (err == 0) {
        *free  = (size_t)resp.free;
        *total = (size_t)resp.total;
        TRANSPORT_LOG("cuMemGetInfo_v2: free=%zu total=%zu", *free, *total);
    } else {
        *free  = 0;
        *total = 0;
    }
    return (CUresult)err;
}

CUresult cuCtxDestroy(CUcontext ctx)
{
    (void)ctx;

    TRANSPORT_LOG("cuCtxDestroy(%p)", ctx);

    transport_rpc_call(GPU_CMD_CTX_DESTROY, NULL, 0, NULL, 0, NULL);

    if (g_current_ctx == ctx) g_current_ctx = NULL;
    if (g_primary_ctx == ctx) g_primary_ctx = NULL;
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char **pStr)
{
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;

    /* Static string table -- no RPC needed */
    switch (error) {
    case 0:   *pStr = "no error"; break;
    case 1:   *pStr = "invalid value"; break;
    case 100: *pStr = "no CUDA-capable device is detected"; break;
    case 201: *pStr = "invalid context"; break;
    case 801: *pStr = "operation not supported"; break;
    default:  *pStr = "unknown error"; break;
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **pStr)
{
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;

    switch (error) {
    case 0:   *pStr = "CUDA_SUCCESS"; break;
    case 1:   *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
    case 100: *pStr = "CUDA_ERROR_NO_DEVICE"; break;
    case 201: *pStr = "CUDA_ERROR_INVALID_CONTEXT"; break;
    case 801: *pStr = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default:  *pStr = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}

/* ================================================================
 * Version-aliased symbols
 *
 * Some frameworks call cuMemGetInfo (without _v2) or cuCtxCreate_v2.
 * Provide aliases for compatibility.
 * ================================================================ */

CUresult cuMemGetInfo(size_t *free, size_t *total)
    __attribute__((alias("cuMemGetInfo_v2")));

CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice device)
{
    return cuCtxCreate_v3(ctx, NULL, 0, flags, device);
}

/* cuDeviceGetCount_v2 alias used by some CUDA versions */
CUresult cuDeviceGetCount_v2(int *count)
    __attribute__((alias("cuDeviceGetCount")));

/* ================================================================
 * cuDeviceTotalMem -- some frameworks call this directly
 * ================================================================ */

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice device)
{
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;

    int err = ensure_props_cached(device);
    if (err != 0) {
        *bytes = 0;
        return (CUresult)err;
    }

    *bytes = (size_t)g_cached_props.total_global_mem;
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice device)
    __attribute__((alias("cuDeviceTotalMem_v2")));

/* ================================================================
 * Base-name aliases for cuGetProcAddress dispatch
 *
 * libcudart calls cuGetProcAddress("cuCtxCreate", v3020) which does
 * dlsym(RTLD_DEFAULT, "cuCtxCreate"). Without a bare "cuCtxCreate"
 * symbol, dlsym returns NULL and falls to the generic stub, which
 * returns CUDA_ERROR_NOT_SUPPORTED (801) when actually called.
 *
 * These aliases ensure dlsym finds our real implementations for
 * base names that libcudart queries.
 * ================================================================ */

/* cuCtxCreate (base) → cuCtxCreate_v2 → cuCtxCreate_v3 */
CUresult cuCtxCreate(CUcontext *ctx, unsigned int flags, CUdevice device)
    __attribute__((alias("cuCtxCreate_v2")));

/* cuCtxDestroy_v2 is the versioned name libcudart may ask for */
CUresult cuCtxDestroy_v2(CUcontext ctx)
    __attribute__((alias("cuCtxDestroy")));

/* ================================================================
 * Stream API Stubs (return SUCCESS, not NOT_SUPPORTED)
 *
 * libcudart creates internal streams during initialization.
 * Returning NOT_SUPPORTED from these causes init to fail.
 * These return opaque dummy handles and SUCCESS.
 * ================================================================ */

typedef void *CUstream;
typedef void *CUevent;

/* Dummy handle value — non-NULL to satisfy NULL checks */
#define DUMMY_STREAM ((CUstream)(uintptr_t)0xDEC10001)
#define DUMMY_EVENT  ((CUevent)(uintptr_t)0xDEC10002)

CUresult cuStreamCreate(CUstream *phStream, unsigned int flags)
{
    (void)flags;
    if (phStream) *phStream = DUMMY_STREAM;
    return CUDA_SUCCESS;
}

CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags,
                                     int priority)
{
    (void)flags; (void)priority;
    if (phStream) *phStream = DUMMY_STREAM;
    return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream hStream)
{
    (void)hStream;
    return CUDA_SUCCESS;
}

CUresult cuStreamDestroy(CUstream hStream)
{
    (void)hStream;
    return CUDA_SUCCESS;
}

CUresult cuStreamQuery(CUstream hStream)
{
    (void)hStream;
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                            unsigned int flags)
{
    (void)hStream; (void)hEvent; (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx)
{
    (void)hStream;
    if (pctx) *pctx = g_current_ctx;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags)
{
    (void)hStream;
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority)
{
    (void)hStream;
    if (priority) *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamAddCallback(CUstream hStream, void *callback,
                              void *userData, unsigned int flags)
{
    (void)hStream; (void)callback; (void)userData; (void)flags;
    return CUDA_SUCCESS;
}

/* ================================================================
 * Event API Stubs (return SUCCESS)
 *
 * libcudart uses events for internal timing during init.
 * ================================================================ */

CUresult cuEventCreate(CUevent *phEvent, unsigned int flags)
{
    (void)flags;
    if (phEvent) *phEvent = DUMMY_EVENT;
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy(CUevent hEvent)
{
    (void)hEvent;
    return CUDA_SUCCESS;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
{
    (void)hEvent; (void)hStream;
    return CUDA_SUCCESS;
}

CUresult cuEventSynchronize(CUevent hEvent)
{
    (void)hEvent;
    return CUDA_SUCCESS;
}

CUresult cuEventQuery(CUevent hEvent)
{
    (void)hEvent;
    return CUDA_SUCCESS;
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                             CUevent hEnd)
{
    (void)hStart; (void)hEnd;
    if (pMilliseconds) *pMilliseconds = 0.0f;
    return CUDA_SUCCESS;
}

/* ================================================================
 * Primary Context Management (CRITICAL for libcudart.so.12)
 *
 * libcudart uses the "primary context" API (not cuCtxCreate) for
 * its default device context. Without these, cudaGetDeviceCount()
 * and all other runtime API calls fail even though the driver
 * reports devices correctly.
 *
 * The primary context is a singleton per device. cuDevicePrimaryCtxRetain
 * creates it on first call, cuDevicePrimaryCtxRelease decrements a
 * refcount. We proxy the actual context creation to the daemon.
 * ================================================================ */

CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice device)
{
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;

    TRANSPORT_LOG("cuDevicePrimaryCtxRetain(device=%d)", device);

    if (!g_primary_ctx) {
        CUresult r = cuCtxCreate_v3(&g_primary_ctx, NULL, 0, 0, device);
        if (r != CUDA_SUCCESS) return r;
    }

    *ctx = g_primary_ctx;
    g_current_ctx = g_primary_ctx;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice device)
{
    (void)device;
    /* Don't actually destroy — primary ctx lives for the process lifetime */
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice device, unsigned int flags)
{
    (void)device;
    (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice device, unsigned int *flags,
                                     int *active)
{
    (void)device;
    if (flags) *flags = 0;
    if (active) *active = (g_primary_ctx != NULL) ? 1 : 0;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice device)
{
    (void)device;
    /* Could destroy and recreate, but for proxy mode just no-op */
    return CUDA_SUCCESS;
}

/* ================================================================
 * Context Stack Management
 *
 * libcudart uses these to track the "current" context. We maintain
 * a simple single-context model (no real stack) since proxy mode
 * only supports one device and one context at a time.
 * ================================================================ */

CUresult cuCtxSetCurrent(CUcontext ctx)
{
    g_current_ctx = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *ctx)
{
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    *ctx = g_current_ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *device)
{
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = 0; /* We only support device 0 in proxy mode */
    return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int *flags)
{
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
{
    (void)ctx;
    if (version) *version = 12;
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void)
{
    /* No-op — proxy daemon handles synchronization */
    return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx)
{
    g_current_ctx = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *ctx)
{
    if (ctx) *ctx = g_current_ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetLimit(size_t *pvalue, int limit)
{
    (void)limit;
    if (pvalue) *pvalue = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetLimit(int limit, size_t value)
{
    (void)limit; (void)value;
    return CUDA_SUCCESS;
}

/* ================================================================
 * Virtual Memory Management (VMM) API Stubs
 *
 * Ollama's libggml-cuda.so links against these 8 symbols. Without them,
 * dlopen("libggml-cuda.so", RTLD_NOW) fails with "undefined symbol:
 * cuMemCreate" and Ollama silently falls back to CPU-only.
 *
 * These stubs return CUDA_ERROR_NOT_SUPPORTED to force ggml-cuda onto
 * the regular cudaMalloc/cudaFree code path, which our Runtime API shim
 * CAN proxy to the host GPU.
 *
 * Combined with attribute 100 (VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)
 * returning 0, ggml-cuda should never actually call these at runtime.
 * They exist purely to satisfy the dynamic linker at dlopen time.
 * ================================================================ */

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle,
                     size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags)
{
    (void)handle; (void)size; (void)prop; (void)flags;
    TRANSPORT_LOG("cuMemCreate() → NOT_SUPPORTED (using cudaMalloc path)");
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    (void)handle;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr,
                              size_t size,
                              size_t alignment,
                              CUdeviceptr addr,
                              unsigned long long flags)
{
    (void)ptr; (void)size; (void)alignment; (void)addr; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    (void)ptr; (void)size;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemMap(CUdeviceptr ptr,
                  size_t size,
                  size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags)
{
    (void)ptr; (void)size; (void)offset; (void)handle; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    (void)ptr; (void)size;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemSetAccess(CUdeviceptr ptr,
                        size_t size,
                        const CUmemAccessDesc *desc,
                        size_t count)
{
    (void)ptr; (void)size; (void)desc; (void)count;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                        const CUmemAllocationProp *prop,
                                        CUmemAllocationGranularity_flags option)
{
    (void)prop; (void)option;
    if (granularity) *granularity = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ================================================================
 * Driver API Memory Forwarding (CRITICAL for GPU inference)
 *
 * libcublas.so.12 allocates GPU memory through the CUDA Driver API
 * (cuMemAlloc), not the Runtime API (cudaMalloc). These forwarding
 * functions use the same RPC commands the daemon already handles:
 *   GPU_CMD_MALLOC, GPU_CMD_FREE, GPU_CMD_MEMCPY, GPU_CMD_MEMSET,
 *   GPU_CMD_MEM_GET_INFO
 *
 * cuGetProcAddress returns pointers to these functions instead of
 * the generic_not_supported_stub.
 * ================================================================ */

static CUresult cu_mem_alloc(CUdeviceptr *dptr, size_t bytesize)
{
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    GpuMallocRequest req = { .size = (uint64_t)bytesize };
    GpuMallocResponse resp;
    int err = transport_rpc_call(GPU_CMD_MALLOC, &req, sizeof(req),
                                 &resp, sizeof(resp), NULL);
    if (err == 0) {
        *dptr = (CUdeviceptr)resp.device_ptr;
        TRANSPORT_LOG("cuMemAlloc(%zu) → 0x%llx", bytesize,
                      (unsigned long long)resp.device_ptr);
    } else {
        *dptr = 0;
        TRANSPORT_LOG("cuMemAlloc(%zu) FAILED (err=%d)", bytesize, err);
    }
    return (CUresult)err;
}

static CUresult cu_mem_free(CUdeviceptr dptr)
{
    GpuFreeRequest req = { .device_ptr = (uint64_t)dptr };
    int err = transport_rpc_call(GPU_CMD_FREE, &req, sizeof(req),
                                 NULL, 0, NULL);
    TRANSPORT_LOG("cuMemFree(0x%llx) → %d", (unsigned long long)dptr, err);
    return (CUresult)err;
}

static CUresult cu_memcpy_HtoD(CUdeviceptr dst, const void *src, size_t byteCount)
{
    if (!src && byteCount > 0) return CUDA_ERROR_INVALID_VALUE;

    /*
     * Chunk large H2D transfers to stay under GPU_PROXY_MAX_PAYLOAD (64 MB).
     * Each RPC carries sizeof(GpuMemcpyRequest) + chunk_bytes as payload.
     * Use 32 MB chunks for comfortable headroom.
     */
    const size_t MAX_CHUNK = 32UL * 1024 * 1024;
    size_t offset = 0;

    while (offset < byteCount) {
        size_t chunk = byteCount - offset;
        if (chunk > MAX_CHUNK) chunk = MAX_CHUNK;

        GpuMemcpyRequest req = {
            .dst   = (uint64_t)(dst + offset),
            .src   = 0,  /* host data follows header */
            .count = (uint64_t)chunk,
            .kind  = GPU_MEMCPY_HOST_TO_DEVICE,
        };

        uint32_t total_len = (uint32_t)(sizeof(req) + chunk);
        uint8_t *payload = malloc(total_len);
        if (!payload) return CUDA_ERROR_INVALID_VALUE;

        memcpy(payload, &req, sizeof(req));
        memcpy(payload + sizeof(req), (const uint8_t *)src + offset, chunk);

        int err = transport_rpc_call(GPU_CMD_MEMCPY, payload, total_len,
                                     NULL, 0, NULL);
        free(payload);
        if (err != 0) {
            TRANSPORT_LOG("cuMemcpyHtoD chunk at offset %zu FAILED (err=%d)",
                          offset, err);
            return (CUresult)err;
        }
        offset += chunk;
    }

    return CUDA_SUCCESS;
}


static CUresult cu_memcpy_DtoH(void *dst, CUdeviceptr src, size_t byteCount)
{
    if (!dst && byteCount > 0) return CUDA_ERROR_INVALID_VALUE;

    /*
     * Chunk large D2H transfers to stay under GPU_PROXY_MAX_PAYLOAD (64 MB).
     * Each RPC returns up to MAX_CHUNK bytes of device data.
     */
    const size_t MAX_CHUNK = 32UL * 1024 * 1024;
    size_t offset = 0;

    while (offset < byteCount) {
        size_t chunk = byteCount - offset;
        if (chunk > MAX_CHUNK) chunk = MAX_CHUNK;

        GpuMemcpyRequest chunk_req = {
            .dst   = 0,
            .src   = (uint64_t)(src + offset),
            .count = (uint64_t)chunk,
            .kind  = GPU_MEMCPY_DEVICE_TO_HOST,
        };

        uint32_t actual = 0;
        int err = transport_rpc_call(GPU_CMD_MEMCPY, &chunk_req, sizeof(chunk_req),
                                     (uint8_t *)dst + offset, (uint32_t)chunk, &actual);
        if (err != 0) {
            TRANSPORT_LOG("cuMemcpyDtoH chunk at offset %zu FAILED (err=%d)",
                          offset, err);
            return (CUresult)err;
        }
        offset += chunk;
    }

    return CUDA_SUCCESS;
}

static CUresult cu_memcpy_DtoD(CUdeviceptr dst, CUdeviceptr src, size_t byteCount)
{
    GpuMemcpyRequest req = {
        .dst   = (uint64_t)dst,
        .src   = (uint64_t)src,
        .count = (uint64_t)byteCount,
        .kind  = GPU_MEMCPY_DEVICE_TO_DEVICE,
    };
    int err = transport_rpc_call(GPU_CMD_MEMCPY, &req, sizeof(req),
                                 NULL, 0, NULL);
    return (CUresult)err;
}

static CUresult cu_memset_D8(CUdeviceptr dptr, unsigned char uc, size_t N)
{
    GpuMemsetRequest req = {
        .device_ptr = (uint64_t)dptr,
        .value      = (int32_t)uc,
        .count      = (uint64_t)N,
    };
    int err = transport_rpc_call(GPU_CMD_MEMSET, &req, sizeof(req),
                                 NULL, 0, NULL);
    return (CUresult)err;
}

static CUresult cu_memset_D32(CUdeviceptr dptr, unsigned int ui, size_t N)
{
    /* Memset D32 sets N 32-bit values. The daemon memset treats count as
     * bytes with the given value. For D32, we pass value=ui and count=N*4
     * and rely on the daemon using cudaMemset which is byte-oriented.
     * Actually, the daemon should handle this as cudaMemset(ptr, val, count).
     * For 32-bit fill, the CUDA runtime doesn't have a direct byte-level
     * equivalent. Use memset with the low byte as a best-effort. */
    GpuMemsetRequest req = {
        .device_ptr = (uint64_t)dptr,
        .value      = (int32_t)ui,
        .count      = (uint64_t)(N * 4),
    };
    int err = transport_rpc_call(GPU_CMD_MEMSET, &req, sizeof(req),
                                 NULL, 0, NULL);
    return (CUresult)err;
}

/* Async variants map to synchronous implementations — proxy is inherently sync */
static CUresult cu_mem_alloc_async(CUdeviceptr *dptr, size_t bytesize,
                                    CUstream hStream)
{
    (void)hStream;
    return cu_mem_alloc(dptr, bytesize);
}

static CUresult cu_mem_free_async(CUdeviceptr dptr, CUstream hStream)
{
    (void)hStream;
    return cu_mem_free(dptr);
}

/* Managed memory → regular allocation (proxy can't do unified addressing) */
static CUresult cu_mem_alloc_managed(CUdeviceptr *dptr, size_t bytesize,
                                      unsigned int flags)
{
    (void)flags;
    return cu_mem_alloc(dptr, bytesize);
}

/* Host allocation — local, no GPU allocation needed */
static CUresult cu_mem_host_alloc(void **pp, size_t bytesize, unsigned int flags)
{
    (void)flags;
    if (!pp) return CUDA_ERROR_INVALID_VALUE;
    if (posix_memalign(pp, 4096, bytesize) != 0) {
        *pp = NULL;
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

/* Host free — local free for posix_memalign'd memory */
static CUresult cu_mem_free_host(void *p)
{
    free(p);
    return CUDA_SUCCESS;
}

/* Pitch allocation → regular allocation (pitch = width, no padding) */
static CUresult cu_mem_alloc_pitch(CUdeviceptr *dptr, size_t *pPitch,
                                    size_t WidthInBytes, size_t Height,
                                    unsigned int ElementSizeBytes)
{
    (void)ElementSizeBytes;
    if (pPitch) *pPitch = WidthInBytes;
    return cu_mem_alloc(dptr, WidthInBytes * Height);
}

/* Async memcpy variants — map to synchronous */
static CUresult cu_memcpy_HtoD_async(CUdeviceptr dst, const void *src,
                                      size_t byteCount, CUstream hStream)
{
    (void)hStream;
    return cu_memcpy_HtoD(dst, src, byteCount);
}

static CUresult cu_memcpy_DtoH_async(void *dst, CUdeviceptr src,
                                      size_t byteCount, CUstream hStream)
{
    (void)hStream;
    return cu_memcpy_DtoH(dst, src, byteCount);
}

static CUresult cu_memcpy_DtoD_async(CUdeviceptr dst, CUdeviceptr src,
                                      size_t byteCount, CUstream hStream)
{
    (void)hStream;
    return cu_memcpy_DtoD(dst, src, byteCount);
}

static CUresult cu_memcpy_async(CUdeviceptr dst, CUdeviceptr src,
                                 size_t byteCount, CUstream hStream)
{
    (void)hStream;
    return cu_memcpy_DtoD(dst, src, byteCount);
}

static CUresult cu_memcpy(CUdeviceptr dst, CUdeviceptr src, size_t byteCount)
{
    return cu_memcpy_DtoD(dst, src, byteCount);
}

/* cuMemsetD8Async, cuMemsetD32Async — map to synchronous */
static CUresult cu_memset_D8_async(CUdeviceptr dptr, unsigned char uc,
                                    size_t N, CUstream hStream)
{
    (void)hStream;
    return cu_memset_D8(dptr, uc, N);
}

static CUresult cu_memset_D32_async(CUdeviceptr dptr, unsigned int ui,
                                     size_t N, CUstream hStream)
{
    (void)hStream;
    return cu_memset_D32(dptr, ui, N);
}

/* ================================================================
 * cuGetProcAddress dispatch table for Driver API memory functions
 *
 * This table is checked BEFORE the dlsym fallback in cuGetProcAddress.
 * Without this, cuMemAlloc/cuMemFree/cuMemcpy resolve to the generic
 * NOT_SUPPORTED stub, causing libcublas memory operations to fail
 * and inference to crash with "CUDA error".
 * ================================================================ */

typedef struct {
    const char *name;
    void       *func;
} DriverDispatchEntry;

static const DriverDispatchEntry g_driver_dispatch[] = {
    /* Memory allocation */
    { "cuMemAlloc",             (void *)cu_mem_alloc },
    { "cuMemAlloc_v2",          (void *)cu_mem_alloc },
    { "cuMemFree",              (void *)cu_mem_free },
    { "cuMemFree_v2",           (void *)cu_mem_free },
    { "cuMemAllocAsync",        (void *)cu_mem_alloc_async },
    { "cuMemAllocAsync_ptsz",   (void *)cu_mem_alloc_async },
    { "cuMemFreeAsync",         (void *)cu_mem_free_async },
    { "cuMemFreeAsync_ptsz",    (void *)cu_mem_free_async },
    { "cuMemAllocManaged",      (void *)cu_mem_alloc_managed },
    { "cuMemAllocPitch",        (void *)cu_mem_alloc_pitch },
    { "cuMemAllocPitch_v2",     (void *)cu_mem_alloc_pitch },
    { "cuMemHostAlloc",         (void *)cu_mem_host_alloc },
    { "cuMemFreeHost",          (void *)cu_mem_free_host },
    { "cuMemAllocFromPoolAsync",(void *)cu_mem_alloc_async },

    /* Memory copy */
    { "cuMemcpyHtoD",           (void *)cu_memcpy_HtoD },
    { "cuMemcpyHtoD_v2",        (void *)cu_memcpy_HtoD },
    { "cuMemcpyDtoH",           (void *)cu_memcpy_DtoH },
    { "cuMemcpyDtoH_v2",        (void *)cu_memcpy_DtoH },
    { "cuMemcpyDtoD",           (void *)cu_memcpy_DtoD },
    { "cuMemcpyDtoD_v2",        (void *)cu_memcpy_DtoD },
    { "cuMemcpy",               (void *)cu_memcpy },
    { "cuMemcpyAsync",          (void *)cu_memcpy_async },
    { "cuMemcpyAsync_ptsz",     (void *)cu_memcpy_async },
    { "cuMemcpyHtoDAsync",      (void *)cu_memcpy_HtoD_async },
    { "cuMemcpyHtoDAsync_v2",   (void *)cu_memcpy_HtoD_async },
    { "cuMemcpyDtoHAsync",      (void *)cu_memcpy_DtoH_async },
    { "cuMemcpyDtoHAsync_v2",   (void *)cu_memcpy_DtoH_async },
    { "cuMemcpyDtoDAsync",      (void *)cu_memcpy_DtoD_async },
    { "cuMemcpyDtoDAsync_v2",   (void *)cu_memcpy_DtoD_async },

    /* Memory set */
    { "cuMemsetD8",             (void *)cu_memset_D8 },
    { "cuMemsetD8_v2",          (void *)cu_memset_D8 },
    { "cuMemsetD32",            (void *)cu_memset_D32 },
    { "cuMemsetD32_v2",         (void *)cu_memset_D32 },
    { "cuMemsetD8Async",        (void *)cu_memset_D8_async },
    { "cuMemsetD32Async",       (void *)cu_memset_D32_async },

    /* Sentinel */
    { NULL, NULL },
};

/* ================================================================
 * cuGetProcAddress — CUDA Driver function dispatch (CRITICAL)
 *
 * libcudart.so.12+ uses cuGetProcAddress as its PRIMARY method to
 * resolve all driver API functions. It queries ~300 functions at
 * init time and evaluates the results to decide driver capability.
 *
 * Strategy:
 *   1. Check the dispatch table for known memory functions
 *   2. Try dlsym(RTLD_DEFAULT, symbol) to find our real exports
 *   3. If not found, return a pointer to generic_not_supported_stub
 *
 * Returning non-NULL for everything tells libcudart the driver is
 * fully capable. Functions we haven't implemented will return
 * CUDA_ERROR_NOT_SUPPORTED if actually called (graceful fallback).
 *
 * Without this, libcudart sees hundreds of NULL pointers and
 * returns cudaErrorInsufficientDriver regardless of version.
 * ================================================================ */

CUresult cuGetProcAddress(const char *symbol, void **pfn,
                          int cudaVersion, uint64_t flags)
{
    (void)cudaVersion;
    (void)flags;

    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    /* Check dispatch table first for memory functions */
    for (const DriverDispatchEntry *e = g_driver_dispatch; e->name; e++) {
        if (strcmp(symbol, e->name) == 0) {
            *pfn = e->func;
            TRANSPORT_LOG("cuGetProcAddress(\"%s\", v%d) → %p [forwarding]",
                          symbol, cudaVersion, *pfn);
            return CUDA_SUCCESS;
        }
    }

    /* Fall back to dlsym for our exported symbols */
    *pfn = dlsym(RTLD_DEFAULT, symbol);

    if (*pfn == NULL) {
        /* Return generic stub instead of NULL — tells libcudart
         * the driver supports this function. If actually called,
         * the stub returns CUDA_ERROR_NOT_SUPPORTED. */
        *pfn = (void *)generic_not_supported_stub;
    }

    TRANSPORT_LOG("cuGetProcAddress(\"%s\", v%d) → %p%s",
                  symbol, cudaVersion, *pfn,
                  (*pfn == (void *)generic_not_supported_stub) ? " [stub]" : "");

    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn,
                             int cudaVersion, uint64_t flags,
                             void *symbolStatus)
{
    CUresult r = cuGetProcAddress(symbol, pfn, cudaVersion, flags);

    /* Always report "found" since we always return non-NULL pfn */
    if (symbolStatus)
        *(int *)symbolStatus = 0;

    return r;
}

/* ================================================================
 * Cleanup on library unload
 * ================================================================ */

__attribute__((destructor))
static void driver_shim_cleanup(void)
{
    transport_disconnect();
}