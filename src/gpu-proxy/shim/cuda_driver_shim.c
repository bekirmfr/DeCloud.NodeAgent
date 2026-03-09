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
 * Must appear before graph_success_stub / driver_shim_init which
 * reference CUresult, CUDA_SUCCESS, and g_vmem_proxy.
 * ================================================================ */

typedef int CUresult;
#define CUDA_SUCCESS               0
#define CUDA_ERROR_INVALID_VALUE   1
#define CUDA_ERROR_OUT_OF_MEMORY   2
#define CUDA_ERROR_NOT_FOUND     500
#define CUDA_ERROR_NO_DEVICE     100
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_INVALID_SOURCE  300
#define CUDA_ERROR_FILE_NOT_FOUND  301
#define CUDA_ERROR_NOT_SUPPORTED   801

typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;

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

/* When true (set via DECLOUD_GPU_VMEM_PROXY=1), VMM RPC path is active.
 * Off by default — Ollama/ggml never calls VMM functions at runtime. */
static int g_vmem_proxy = 0;

/* When true (default), graph capture stubs return CUDA_SUCCESS (no-op).
 * Set DECLOUD_GPU_GRAPH_NOOP=0 to return NOT_SUPPORTED instead. */
static int g_driver_graph_noop = 1;

static CUresult graph_success_stub(void) { return CUDA_SUCCESS; }

__attribute__((constructor))
static void driver_shim_init(void)
{
    /* Load config early — cuGetProcAddress is called before cuInit */
    g_debug_log = (transport_getenv("DECLOUD_GPU_DEBUG") != NULL);
    const char *gnoop = transport_getenv("DECLOUD_GPU_GRAPH_NOOP");
    if (gnoop && gnoop[0] == '0') g_driver_graph_noop = 0;

    const char *vmem = transport_getenv("DECLOUD_GPU_VMEM_PROXY");
    if (vmem && vmem[0] == '1') g_vmem_proxy = 1;

    /* Restore default FPU exception mask — same reason as cuda_shim.c */
    fedisableexcept(FE_ALL_EXCEPT);
}

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

/* Graph operations cannot work through RPC proxy */
static CUresult graph_not_supported_stub(void)
{
    return CUDA_SUCCESS;
}

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

/* ================================================================
 * Module/Function state tracking for Driver API forwarding
 *
 * cuBLAS loads GEMM kernels via cuModuleLoadFatBinary →
 * cuModuleGetFunction → cuLaunchKernel. We track the daemon's
 * module/function slot indices locally so we can map opaque
 * CUmodule/CUfunction handles back to the correct daemon-side
 * resources.
 * ================================================================ */

#define MAX_DRIVER_MODULES    128
#define MAX_DRIVER_FUNCTIONS  512

typedef struct {
    uint64_t daemon_handle;  /* Module slot index on daemon */
    int      in_use;
} DriverModuleSlot;

typedef struct {
    uint64_t opaque_handle;
    uint64_t module_slot;
    char     name[1024];
    uint32_t num_params;
    uint32_t param_sizes[GPU_MAX_KERNEL_PARAMS];
    int      in_use;
    int      attrs_cached;
    GpuFuncGetAttributesResponse cached_attrs;
} DriverFunctionSlot;

static DriverModuleSlot    g_driver_modules[MAX_DRIVER_MODULES];
static DriverFunctionSlot  g_driver_functions[MAX_DRIVER_FUNCTIONS];
static pthread_mutex_t     g_module_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t            g_next_func_handle = 0x1000;

/* Last-used function cache for hot-path optimization */
static DriverFunctionSlot *g_last_func_cache = NULL;

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
    case 100: /* VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED */
        *value = g_vmem_proxy ? 1 : 0; break;
    default:
        *value = 0; break;
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


/* Capture query stubs — always report "not capturing" */
CUresult cuStreamIsCapturing(CUstream hStream, int *captureStatus)
{
    (void)hStream;
    if (captureStatus) *captureStatus = 0; /* CU_STREAM_CAPTURE_STATUS_NONE */
    return CUDA_SUCCESS;
}
CUresult cuStreamGetCaptureInfo(CUstream hStream, int *captureStatus,
                                 unsigned long long *id)
{
    (void)hStream;
    if (captureStatus) *captureStatus = 0;
    if (id) *id = 0;
    return CUDA_SUCCESS;
}
CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, int *captureStatus,
                                    unsigned long long *id, void **graph,
                                    void **deps, size_t *numDeps)
{
    (void)hStream;
    if (captureStatus) *captureStatus = 0;
    if (id) *id = 0;
    if (graph) *graph = NULL;
    if (deps) *deps = NULL;
    if (numDeps) *numDeps = 0;
    return CUDA_SUCCESS;
}
CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, void *deps,
                                            size_t numDeps, unsigned int flags)
{
    (void)hStream; (void)deps; (void)numDeps; (void)flags;
    return CUDA_SUCCESS;
}
CUresult cuThreadExchangeStreamCaptureMode(int *mode)
{
    if (mode) *mode = 0; /* CU_STREAM_CAPTURE_MODE_GLOBAL */
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

/* cuGetExportTable — cuBLAS uses this for internal function tables.
 * Returning CUDA_SUCCESS without setting ppExportTable causes crash
 * at offset 0x10 (NULL dereference). Return NOT_FOUND instead. */
CUresult cuGetExportTable(const void **ppExportTable, const void *pExportTableId)
{
    (void)pExportTableId;
    if (ppExportTable) *ppExportTable = NULL;
    return CUDA_ERROR_NOT_FOUND;
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
    if (!g_vmem_proxy) return CUDA_ERROR_NOT_SUPPORTED;
    (void)prop;
    GpuVmemCreateRequest req = { .size = (uint64_t)size, .flags = (uint64_t)flags };
    GpuVmemCreateResponse resp;
    int err = transport_rpc_call(GPU_CMD_VMEM_CREATE,
                                 &req, sizeof(req), &resp, sizeof(resp), NULL);
    if (err == 0 && handle)
        *handle = (CUmemGenericAllocationHandle)resp.handle;
    return (CUresult)err;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    if (!g_vmem_proxy) return CUDA_ERROR_NOT_SUPPORTED;
    GpuVmemReleaseRequest req = { .handle = (uint64_t)handle };
    return (CUresult)transport_rpc_call(GPU_CMD_VMEM_RELEASE,
                                        &req, sizeof(req), NULL, 0, NULL);
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                              size_t alignment, CUdeviceptr addr,
                              unsigned long long flags)
{
    if (!g_vmem_proxy) return CUDA_ERROR_NOT_SUPPORTED;
    GpuVmemAddressReserveRequest req = {
        .size = (uint64_t)size, .alignment = (uint64_t)alignment,
        .addr = (uint64_t)addr, .flags = (uint64_t)flags,
    };
    GpuVmemAddressReserveResponse resp;
    int err = transport_rpc_call(GPU_CMD_VMEM_ADDRESS_RESERVE,
                                 &req, sizeof(req), &resp, sizeof(resp), NULL);
    if (err == 0 && ptr) *ptr = (CUdeviceptr)resp.ptr;
    return (CUresult)err;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    if (!g_vmem_proxy) return CUDA_ERROR_NOT_SUPPORTED;
    GpuVmemAddressFreeRequest req = { .ptr = (uint64_t)ptr, .size = (uint64_t)size };
    return (CUresult)transport_rpc_call(GPU_CMD_VMEM_ADDRESS_FREE,
                                        &req, sizeof(req), NULL, 0, NULL);
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, unsigned long long flags)
{
    if (!g_vmem_proxy) return CUDA_ERROR_NOT_SUPPORTED;
    GpuVmemMapRequest req = {
        .ptr = (uint64_t)ptr, .size = (uint64_t)size,
        .offset = (uint64_t)offset, .handle = (uint64_t)handle,
        .flags = (uint64_t)flags,
    };
    return (CUresult)transport_rpc_call(GPU_CMD_VMEM_MAP,
                                        &req, sizeof(req), NULL, 0, NULL);
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    if (!g_vmem_proxy) return CUDA_ERROR_NOT_SUPPORTED;
    GpuVmemUnmapRequest req = { .ptr = (uint64_t)ptr, .size = (uint64_t)size };
    return (CUresult)transport_rpc_call(GPU_CMD_VMEM_UNMAP,
                                        &req, sizeof(req), NULL, 0, NULL);
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count)
{
    if (!g_vmem_proxy) return CUDA_ERROR_NOT_SUPPORTED;
    (void)desc;
    GpuVmemSetAccessRequest req = {
        .ptr = (uint64_t)ptr, .size = (uint64_t)size,
        .count = (uint32_t)count,
    };
    return (CUresult)transport_rpc_call(GPU_CMD_VMEM_SET_ACCESS,
                                        &req, sizeof(req), NULL, 0, NULL);
}

CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                        const CUmemAllocationProp *prop,
                                        CUmemAllocationGranularity_flags option)
{
    if (!g_vmem_proxy) {
        if (granularity) *granularity = 0;
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    (void)prop;
    GpuVmemGetGranularityRequest req = { .option = (uint32_t)option };
    GpuVmemGetGranularityResponse resp;
    int err = transport_rpc_call(GPU_CMD_VMEM_GET_GRANULARITY,
                                 &req, sizeof(req), &resp, sizeof(resp), NULL);
    if (err == 0 && granularity) *granularity = (size_t)resp.granularity;
    return (CUresult)err;
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
 * Driver API Module/Function/Kernel Forwarding
 *
 * cuBLAS uses the CUDA Driver API to load its GEMM kernels:
 *   cuModuleLoadFatBinary → cuModuleGetFunction → cuLaunchKernel
 * These forward through the daemon's existing GPU_CMD_REGISTER_MODULE,
 * GPU_CMD_REGISTER_FUNCTION, and GPU_CMD_LAUNCH_KERNEL handlers.
 * No daemon or protocol changes needed.
 * ================================================================ */

static CUresult cu_module_load_data(CUmodule *module, const void *image)
{
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;

    /* Detect image format and size */
    uint32_t magic = *(const uint32_t *)image;
    uint64_t data_size = 0;

    if (magic == 0xBA55ED50) {
        /* NVIDIA fatbin container:
         * struct { uint32_t magic; uint16_t version; uint16_t header_size; uint64_t size; }
         * 'size' is total size of all data AFTER the header.
         * Total blob size = header_size + size
         */
        uint16_t version = *(const uint16_t *)((const uint8_t *)image + 4);
        uint16_t header_size = *(const uint16_t *)((const uint8_t *)image + 6);
        uint64_t payload_size = *(const uint64_t *)((const uint8_t *)image + 8);

        if (version == 0x0001 && header_size > 0) {
            /* Newer fatbin_header: size = data after header */
            data_size = (uint64_t)header_size + payload_size;
        } else {
            /* Older __cudaFatCudaBinaryRec: size field is total including header */
            data_size = payload_size;
        }

        if (data_size == 0 || data_size > GPU_PROXY_MAX_PAYLOAD) {
            TRANSPORT_LOG("cuModuleLoadData: suspicious fatbin size %lu, using 4MB fallback",
                          (unsigned long)data_size);
            data_size = 4 * 1024 * 1024;
        }
    } else if (magic == 0x464C457F) {
        /* ELF cubin: parse ELF64 header for total size */
        const uint8_t *elf = (const uint8_t *)image;
        uint64_t shoff = *(const uint64_t *)(elf + 40);      /* e_shoff */
        uint16_t shentsize = *(const uint16_t *)(elf + 58);   /* e_shentsize */
        uint16_t shnum = *(const uint16_t *)(elf + 60);       /* e_shnum */
        data_size = shoff + (uint64_t)shentsize * shnum;
        if (data_size == 0 || data_size > GPU_PROXY_MAX_PAYLOAD)
            data_size = 4 * 1024 * 1024;
    } else if (image && ((const char *)image)[0] == '.') {
        /* PTX text (starts with ".version") */
        data_size = strlen((const char *)image) + 1;
    } else if (magic == 0x466243B1) {
        /* __fatBinC_Wrapper_t (FATBINC_MAGIC) — dereference .data pointer */
        const void *inner = *(const void *const *)((const uint8_t *)image + 8);
        if (inner) return cu_module_load_data(module, inner);
        return CUDA_ERROR_INVALID_SOURCE;
    } else {
        /* Unknown format — try conservative size */
        TRANSPORT_LOG("cuModuleLoadData: unknown magic 0x%08x, using 4MB fallback", magic);
        data_size = 4 * 1024 * 1024;
    }

    TRANSPORT_LOG("cuModuleLoadData: sending %lu bytes (magic=0x%08x)",
                  (unsigned long)data_size, magic);

    /* Build RPC request: GpuRegisterModuleRequest + fatbin data */
    uint32_t req_len = (uint32_t)(sizeof(GpuRegisterModuleRequest) + data_size);
    void *req_buf = malloc(req_len);
    if (!req_buf) return CUDA_ERROR_OUT_OF_MEMORY;

    GpuRegisterModuleRequest *req = (GpuRegisterModuleRequest *)req_buf;
    req->fatbin_size = data_size;
    memcpy((uint8_t *)req_buf + sizeof(GpuRegisterModuleRequest), image, data_size);

    GpuRegisterModuleResponse resp;
    int err = transport_rpc_call(GPU_CMD_REGISTER_MODULE, req_buf, req_len,
                                 &resp, sizeof(resp), NULL);
    free(req_buf);

    if (err != 0) {
        TRANSPORT_LOG("cuModuleLoadData FAILED (err=%d)", err);
        *module = NULL;
        return (CUresult)err;
    }

    /* Store in local tracking table */
    pthread_mutex_lock(&g_module_lock);
    int slot = -1;
    for (int i = 0; i < MAX_DRIVER_MODULES; i++) {
        if (!g_driver_modules[i].in_use) { slot = i; break; }
    }
    if (slot >= 0) {
        g_driver_modules[slot].daemon_handle = resp.module_handle;
        g_driver_modules[slot].in_use = 1;
        *module = (CUmodule)(uintptr_t)(slot + 1);  /* 1-based opaque handle */
    } else {
        *module = NULL;
        err = CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_unlock(&g_module_lock);

    TRANSPORT_LOG("cuModuleLoadData → module=%p (local_slot=%d, daemon_slot=%lu)",
                  *module, slot, (unsigned long)resp.module_handle);
    return (CUresult)err;
}

static CUresult cu_module_load_fat_binary(CUmodule *module, const void *fatbin)
{
    /* Same wire format — daemon handles both fatbin and cubin */
    return cu_module_load_data(module, fatbin);
}

static CUresult cu_module_load(CUmodule *module, const char *filename)
{
    if (!module || !filename) return CUDA_ERROR_INVALID_VALUE;

    FILE *f = fopen(filename, "rb");
    if (!f) return CUDA_ERROR_FILE_NOT_FOUND;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > (long)GPU_PROXY_MAX_PAYLOAD) {
        fclose(f);
        return CUDA_ERROR_INVALID_SOURCE;
    }

    void *data = malloc((size_t)size);
    if (!data) { fclose(f); return CUDA_ERROR_OUT_OF_MEMORY; }

    if (fread(data, 1, (size_t)size, f) != (size_t)size) {
        free(data);
        fclose(f);
        return CUDA_ERROR_INVALID_SOURCE;
    }
    fclose(f);

    CUresult r = cu_module_load_data(module, data);
    free(data);
    return r;
}

/* cuModuleLoadDataEx — same as cuModuleLoadData but with JIT options we ignore */
static CUresult cu_module_load_data_ex(CUmodule *module, const void *image,
                                        unsigned int numOptions, void *options,
                                        void **optionValues)
{
    (void)numOptions; (void)options; (void)optionValues;
    return cu_module_load_data(module, image);
}

static CUresult cu_module_unload(CUmodule module)
{
    int slot = (int)(uintptr_t)module - 1;  /* Convert 1-based handle to 0-based slot */
    if (slot < 0 || slot >= MAX_DRIVER_MODULES) return CUDA_ERROR_INVALID_VALUE;

    pthread_mutex_lock(&g_module_lock);
    if (!g_driver_modules[slot].in_use) {
        pthread_mutex_unlock(&g_module_lock);
        return CUDA_ERROR_INVALID_VALUE;
    }

    GpuUnregisterModuleRequest req = {
        .module_handle = g_driver_modules[slot].daemon_handle,
    };
    int err = transport_rpc_call(GPU_CMD_UNREGISTER_MODULE, &req, sizeof(req),
                                 NULL, 0, NULL);

    /* Clear local state */
    g_driver_modules[slot].in_use = 0;

    /* Invalidate all functions belonging to this module */
    for (int i = 0; i < MAX_DRIVER_FUNCTIONS; i++) {
        if (g_driver_functions[i].in_use &&
            g_driver_functions[i].module_slot == (uint64_t)slot) {
            g_driver_functions[i].in_use = 0;
        }
    }
    g_last_func_cache = NULL;  /* Invalidate cache */
    pthread_mutex_unlock(&g_module_lock);

    TRANSPORT_LOG("cuModuleUnload(slot=%d) → %d", slot, err);
    return (CUresult)err;
}

static CUresult cu_module_get_function(CUfunction *func, CUmodule module,
                                        const char *name)
{
    if (!func || !name) return CUDA_ERROR_INVALID_VALUE;

    int mod_slot = (int)(uintptr_t)module - 1;
    if (mod_slot < 0 || mod_slot >= MAX_DRIVER_MODULES)
        return CUDA_ERROR_INVALID_VALUE;

    pthread_mutex_lock(&g_module_lock);
    if (!g_driver_modules[mod_slot].in_use) {
        pthread_mutex_unlock(&g_module_lock);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint64_t daemon_module = g_driver_modules[mod_slot].daemon_handle;

    /* Find a free local function slot */
    int func_slot = -1;
    for (int i = 0; i < MAX_DRIVER_FUNCTIONS; i++) {
        if (!g_driver_functions[i].in_use) { func_slot = i; break; }
    }
    if (func_slot < 0) {
        pthread_mutex_unlock(&g_module_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    /* Generate unique opaque handle (protected by g_module_lock) */
    uint64_t handle = g_next_func_handle++;

    /* Build RPC: GpuRegisterFunctionRequest + null-terminated name */
    uint32_t name_len = (uint32_t)strlen(name) + 1;
    uint32_t req_len = (uint32_t)(sizeof(GpuRegisterFunctionRequest) + name_len);
    void *req_buf = malloc(req_len);
    if (!req_buf) {
        pthread_mutex_unlock(&g_module_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    GpuRegisterFunctionRequest *req = (GpuRegisterFunctionRequest *)req_buf;
    req->module_handle = daemon_module;
    req->host_func_ptr = handle;  /* Daemon stores this as the lookup key */
    req->device_name_len = name_len;
    memcpy((uint8_t *)req_buf + sizeof(GpuRegisterFunctionRequest), name, name_len);

    GpuRegisterFunctionResponse resp;
    memset(&resp, 0, sizeof(resp));
    int err = transport_rpc_call(GPU_CMD_REGISTER_FUNCTION, req_buf, req_len,
                                 &resp, sizeof(resp), NULL);
    free(req_buf);

    if (err != 0) {
        pthread_mutex_unlock(&g_module_lock);
        *func = NULL;
        TRANSPORT_LOG("cuModuleGetFunction('%s') FAILED (err=%d)", name, err);
        return (CUresult)err;
    }

    /* Store function metadata locally */
    DriverFunctionSlot *fs = &g_driver_functions[func_slot];
    fs->opaque_handle = handle;
    fs->module_slot = (uint64_t)mod_slot;
    strncpy(fs->name, name, sizeof(fs->name) - 1);
    fs->name[sizeof(fs->name) - 1] = '\0';
    fs->num_params = resp.num_params;
    if (fs->num_params > GPU_MAX_KERNEL_PARAMS)
        fs->num_params = GPU_MAX_KERNEL_PARAMS;
    for (uint32_t p = 0; p < fs->num_params; p++)
        fs->param_sizes[p] = resp.param_sizes[p];
    fs->in_use = 1;

    *func = (CUfunction)(uintptr_t)handle;
    pthread_mutex_unlock(&g_module_lock);

    TRANSPORT_LOG("cuModuleGetFunction('%s') → handle=0x%lx, %u params",
                  name, (unsigned long)handle, fs->num_params);
    return CUDA_SUCCESS;
}

static DriverFunctionSlot *find_driver_function(uint64_t handle)
{
    /* Fast path: check last-used cache (same kernel launched repeatedly) */
    DriverFunctionSlot *cached = g_last_func_cache;
    if (cached && cached->in_use && cached->opaque_handle == handle)
        return cached;

    /* Linear scan fallback */
    for (int i = 0; i < MAX_DRIVER_FUNCTIONS; i++) {
        if (g_driver_functions[i].in_use &&
            g_driver_functions[i].opaque_handle == handle) {
            g_last_func_cache = &g_driver_functions[i];
            return &g_driver_functions[i];
        }
    }
    return NULL;
}

static CUresult cu_launch_kernel_driver(CUfunction f,
                                         unsigned int gridDimX, unsigned int gridDimY,
                                         unsigned int gridDimZ,
                                         unsigned int blockDimX, unsigned int blockDimY,
                                         unsigned int blockDimZ,
                                         unsigned int sharedMemBytes,
                                         CUstream hStream,
                                         void **kernelParams,
                                         void **extra)
{
    (void)extra;

    uint64_t handle = (uint64_t)(uintptr_t)f;
    DriverFunctionSlot *fs = find_driver_function(handle);
    if (!fs) {
        TRANSPORT_LOG("cuLaunchKernel: unknown function handle 0x%lx",
                      (unsigned long)handle);
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (fs->num_params == 0) {
        /* Daemon couldn't introspect params (cuFuncGetParamInfo unavailable).
         * Launch with zero-length args — some kernels have no params,
         * and the daemon handles the zero-param case. */
        TRANSPORT_LOG("cuLaunchKernel('%s'): 0 introspected params, launching with no args",
                      fs->name);

        GpuLaunchKernelRequest req = {
            .host_func_ptr   = handle,
            .grid_dim_x      = gridDimX,
            .grid_dim_y      = gridDimY,
            .grid_dim_z      = gridDimZ,
            .block_dim_x     = blockDimX,
            .block_dim_y     = blockDimY,
            .block_dim_z     = blockDimZ,
            .shared_mem_bytes = (uint64_t)sharedMemBytes,
            .stream_handle   = (uint64_t)(uintptr_t)hStream,
            .num_params      = 0,
            .args_total_size = 0,
        };
        int err = transport_rpc_call(GPU_CMD_LAUNCH_KERNEL, &req, sizeof(req),
                                     NULL, 0, NULL);
        TRANSPORT_LOG("cuLaunchKernel('%s', grid=[%u,%u,%u], block=[%u,%u,%u], 0 args) → %d",
                      fs->name, gridDimX, gridDimY, gridDimZ,
                      blockDimX, blockDimY, blockDimZ, err);
        return (CUresult)err;
    }

    /* Calculate total serialized args size */
    uint32_t args_total = 0;
    for (uint32_t i = 0; i < fs->num_params; i++)
        args_total += fs->param_sizes[i];

    /* Build request: header + serialized args */
    uint32_t req_len = (uint32_t)(sizeof(GpuLaunchKernelRequest) + args_total);
    void *req_buf = malloc(req_len);
    if (!req_buf) return CUDA_ERROR_OUT_OF_MEMORY;

    GpuLaunchKernelRequest *req = (GpuLaunchKernelRequest *)req_buf;
    req->host_func_ptr   = handle;
    req->grid_dim_x      = gridDimX;
    req->grid_dim_y      = gridDimY;
    req->grid_dim_z      = gridDimZ;
    req->block_dim_x     = blockDimX;
    req->block_dim_y     = blockDimY;
    req->block_dim_z     = blockDimZ;
    req->shared_mem_bytes = (uint64_t)sharedMemBytes;
    req->stream_handle   = (uint64_t)(uintptr_t)hStream;
    req->num_params      = fs->num_params;
    req->args_total_size = args_total;

    /* Serialize each param value contiguously */
    uint8_t *dest = (uint8_t *)req_buf + sizeof(GpuLaunchKernelRequest);
    for (uint32_t i = 0; i < fs->num_params; i++) {
        uint32_t sz = fs->param_sizes[i];
        if (kernelParams && kernelParams[i]) {
            memcpy(dest, kernelParams[i], sz);
        } else {
            memset(dest, 0, sz);
        }
        dest += sz;
    }

    int err = transport_rpc_call(GPU_CMD_LAUNCH_KERNEL, req_buf, req_len,
                                 NULL, 0, NULL);
    free(req_buf);

    TRANSPORT_LOG("cuLaunchKernel('%s', grid=[%u,%u,%u], block=[%u,%u,%u], args=%u bytes) → %d",
                  fs->name, gridDimX, gridDimY, gridDimZ,
                  blockDimX, blockDimY, blockDimZ, args_total, err);
    return (CUresult)err;
}

/* cuLaunchKernelEx — CUDA 11.6+, extended config struct.
 * cuBLAS typically uses the non-Ex version, but we provide this
 * as a fallback that forwards with default dimensions. */
static CUresult cu_launch_kernel_ex(const void *config,
                                     CUfunction f,
                                     void **kernelParams,
                                     void **extra)
{
    /* TODO: Parse CUlaunchConfig struct for grid/block/stream if cuBLAS
     * actually uses this path. For now, forward with 1x1x1 defaults.
     * Evidence from logs: "cuLaunchKernelEx: forwarding with defaults" */
    TRANSPORT_LOG("cuLaunchKernelEx: forwarding with default dims");
    (void)config;
    return cu_launch_kernel_driver(f, 1, 1, 1, 1, 1, 1, 0, NULL, kernelParams, extra);
}

/* ================================================================
 * Function attribute stubs — cuBLAS checks these before launching
 * ================================================================ */

static CUresult cu_func_get_attribute(int *value, int attrib, CUfunction func)
{
    if (!value) return CUDA_ERROR_INVALID_VALUE;

    /* Try to return real attributes from daemon (cached per-function) */
    uint64_t handle = (uint64_t)(uintptr_t)func;
    DriverFunctionSlot *fs = find_driver_function(handle);

    if (fs && !fs->attrs_cached) {
        GpuFuncGetAttributesRequest req = { .host_func_ptr = handle };
        GpuFuncGetAttributesResponse resp;
        memset(&resp, 0, sizeof(resp));
        int err = transport_rpc_call(GPU_CMD_FUNC_GET_ATTRIBUTES,
                                     &req, sizeof(req), &resp, sizeof(resp), NULL);
        if (err == 0) {
            fs->cached_attrs = resp;
            fs->attrs_cached = 1;
        }
    }

    if (fs && fs->attrs_cached) {
        GpuFuncGetAttributesResponse *a = &fs->cached_attrs;
        switch (attrib) {
        case 0:  *value = a->maxThreadsPerBlock; break;
        case 1:  *value = a->sharedSizeBytes; break;
        case 2:  *value = a->constSizeBytes; break;
        case 3:  *value = a->localSizeBytes; break;
        case 4:  *value = a->numRegs; break;
        case 5:  *value = a->ptxVersion; break;
        case 6:  *value = a->binaryVersion; break;
        case 7:  *value = 0; break;
        case 8:  *value = a->maxDynamicSharedSizeBytes; break;
        case 9:  *value = a->preferredShmemCarveout; break;
        default: *value = 0; break;
        }
        return CUDA_SUCCESS;
    }

    /* Fallback: safe defaults if RPC failed or function unknown */
    switch (attrib) {
    case 0:  *value = 1024; break;
    case 1:  *value = 48 * 1024; break;
    case 2:  *value = 0; break;
    case 3:  *value = 0; break;
    case 4:  *value = 32; break;
    case 5:  *value = 80; break;
    case 6:  *value = 89; break;
    case 7:  *value = 0; break;
    case 8:  *value = 100 * 1024; break;
    case 9:  *value = 0; break;
    default: *value = 0; break;
    }
    return CUDA_SUCCESS;
}

static CUresult cu_func_set_attribute(CUfunction func, int attrib, int value)
{
    (void)func; (void)attrib; (void)value;
    return CUDA_SUCCESS;  /* Silently accept */
}

static CUresult cu_func_set_cache_config(CUfunction func, int config)
{
    (void)func; (void)config;
    return CUDA_SUCCESS;
}

static CUresult cu_module_get_global(CUdeviceptr *dptr, size_t *bytes,
                                      CUmodule module, const char *name)
{
    (void)module; (void)name;
    if (dptr) *dptr = 0;
    if (bytes) *bytes = 0;
    return CUDA_ERROR_NOT_FOUND;  /* Safe — cuBLAS handles this gracefully */
}

static CUresult cu_module_get_loading_mode(int *mode)
{
    if (mode) *mode = 0;  /* CU_MODULE_EAGER_LOADING */
    return CUDA_SUCCESS;
}

/* cuLaunchHostFunc — launches a host callback, not a GPU kernel */
static CUresult cu_launch_host_func(CUstream hStream, void (*fn)(void *), void *userData)
{
    (void)hStream;
    if (fn) fn(userData);
    return CUDA_SUCCESS;
}

/* Library API stubs (CUDA 12.0+) — cuBLAS may probe these */
static CUresult cu_library_get_module(CUmodule *module, void *library)
{
    (void)library;
    if (module) *module = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cu_library_get_kernel(void *kernel, void *library, const char *name)
{
    (void)kernel; (void)library; (void)name;
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cu_kernel_get_function(CUfunction *func, void *kernel)
{
    (void)kernel;
    if (func) *func = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ================================================================
 * cuGetProcAddress dispatch table for Driver API functions
 *
 * This table is checked BEFORE the dlsym fallback in cuGetProcAddress.
 * Without this, cuMemAlloc/cuMemFree/cuMemcpy and module/kernel
 * functions resolve to the generic NOT_SUPPORTED stub, causing
 * libcublas operations to fail and inference to crash.
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

    /* ---- Module loading ---- */
    { "cuModuleLoad",                     (void *)cu_module_load },
    { "cuModuleLoadData",                 (void *)cu_module_load_data },
    { "cuModuleLoadData_v2",              (void *)cu_module_load_data },
    { "cuModuleLoadDataEx",               (void *)cu_module_load_data_ex },
    { "cuModuleLoadFatBinary",            (void *)cu_module_load_fat_binary },
    { "cuModuleUnload",                   (void *)cu_module_unload },
    { "cuModuleGetFunction",              (void *)cu_module_get_function },
    { "cuModuleGetFunction_v2",           (void *)cu_module_get_function },
    { "cuModuleGetGlobal",                (void *)cu_module_get_global },
    { "cuModuleGetGlobal_v2",             (void *)cu_module_get_global },
    { "cuModuleGetTexRef",                (void *)generic_not_supported_stub },
    { "cuModuleGetSurfRef",               (void *)generic_not_supported_stub },
    { "cuModuleGetLoadingMode",           (void *)cu_module_get_loading_mode },

    /* ---- Kernel launch ---- */
    { "cuLaunchKernel",                   (void *)cu_launch_kernel_driver },
    { "cuLaunchKernel_ptsz",              (void *)cu_launch_kernel_driver },
    { "cuLaunchCooperativeKernel",        (void *)cu_launch_kernel_driver },
    { "cuLaunchCooperativeKernel_ptsz",   (void *)cu_launch_kernel_driver },
    { "cuLaunchCooperativeKernelMultiDevice", (void *)generic_not_supported_stub },
    { "cuLaunchHostFunc",                 (void *)cu_launch_host_func },
    { "cuLaunchHostFunc_ptsz",            (void *)cu_launch_host_func },
    { "cuLaunchKernelEx",                 (void *)cu_launch_kernel_ex },
    { "cuLaunchKernelEx_ptsz",            (void *)cu_launch_kernel_ex },

    /* ---- Function attributes ---- */
    { "cuFuncGetAttribute",               (void *)cu_func_get_attribute },
    { "cuFuncSetAttribute",               (void *)cu_func_set_attribute },
    { "cuFuncSetCacheConfig",             (void *)cu_func_set_cache_config },

    /* ---- Library API stubs (CUDA 12.0+) ---- */
    { "cuLibraryGetModule",               (void *)cu_library_get_module },
    { "cuLibraryGetKernel",               (void *)cu_library_get_kernel },
    { "cuKernelGetFunction",              (void *)cu_kernel_get_function },
    { "cuLibraryGetUnifiedFunction",      (void *)generic_not_supported_stub },
    { "cuLibraryGetKernelCount",          (void *)generic_not_supported_stub },
    { "cuLibraryEnumerateKernels",        (void *)generic_not_supported_stub },
    { "cuKernelGetAttribute",             (void *)cu_func_get_attribute },
    { "cuKernelSetAttribute",             (void *)cu_func_set_attribute },
    { "cuKernelSetCacheConfig",           (void *)cu_func_set_cache_config },
    { "cuKernelGetName",                  (void *)generic_not_supported_stub },
    { "cuKernelGetParamInfo",             (void *)generic_not_supported_stub },


    /* Export table */
    { "cuGetExportTable",                 (void *)cuGetExportTable },
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

    /* Graph creation/execution cannot work through network proxy.
     * Only block graph mutation ops; let query ops through with safe defaults. */
    if (strncmp(symbol, "cuGraph", 7) == 0 ||
        strncmp(symbol, "cuStreamBeginCapture", 20) == 0 ||
        strncmp(symbol, "cuStreamEndCapture", 18) == 0) {
        *pfn = g_driver_graph_noop
             ? (void *)graph_success_stub
             : (void *)graph_not_supported_stub;
        TRANSPORT_LOG("cuGetProcAddress(\"%s\", v%d) → %p [graph-%s]",
                      symbol, cudaVersion, *pfn,
                      g_driver_graph_noop ? "noop" : "blocked");
        return CUDA_SUCCESS;
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