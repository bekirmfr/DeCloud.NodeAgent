/*
 * DeCloud CUDA Driver API Shim (libcuda.so.1)
 *
 * Drop-in replacement for libcuda.so that Ollama and other ML frameworks
 * find via dlopen(). When an application calls dlopen("libcuda.so") or
 * searches /usr/local/lib[arch]/libcuda.so, it finds this library and
 * uses dlsym() to resolve the CUDA Driver API symbols below.
 *
 * Each function forwards the call to the host GPU proxy daemon over
 * TCP/vsock, reusing the same transport layer as the Runtime API shim.
 *
 * Ollama GPU discovery sequence:
 *   1. Glob /usr/local/lib[arch]/libcuda.so -- finds this file
 *   2. dlopen("libcuda.so.1") -- loads this library
 *   3. dlsym("cuInit"), dlsym("cuDeviceGetCount"), etc.
 *   4. Calls cuInit(0), cuDeviceGetCount(), cuDeviceGetName(), etc.
 *   5. cuCtxCreate_v3() + cuMemGetInfo_v2() -- gets VRAM info
 *   6. Loads model layers to "GPU" -- inference runs on host GPU
 *
 * Build: gcc -shared -fPIC -o libcuda.so.1 cuda_driver_shim.c -lpthread
 *        ln -sf libcuda.so.1 libcuda.so
 *
 * No CUDA dependency -- this replaces libcuda.so entirely.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

typedef int CUdevice;
typedef void *CUcontext;

typedef struct {
    char bytes[16];
} CUuuid;

/* ================================================================
 * Cached device properties (fetched once, reused)
 * ================================================================ */

static int g_driver_initialized = 0;
static int g_cached_device_count = -1;

/* Cache device properties from daemon to avoid repeated round-trips */
static GpuDeviceProperties g_cached_props;
static int g_cached_props_valid = 0;

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
    default:  *pStr = "unknown error"; break;
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
 * Cleanup on library unload
 * ================================================================ */

__attribute__((destructor))
static void driver_shim_cleanup(void)
{
    transport_disconnect();
}
