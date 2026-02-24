/*
 * DeCloud NVML Shim (libnvidia-ml.so.1)
 *
 * Drop-in replacement for libnvidia-ml.so that Ollama and other ML
 * frameworks find via dlopen(). NVML is NVIDIA's Management Library,
 * used for runtime VRAM monitoring and device enumeration.
 *
 * Ollama's NVML discovery:
 *   1. dlopen("libnvidia-ml.so.1")
 *   2. dlsym("nvmlInit_v2"), dlsym("nvmlDeviceGetHandleByIndex_v2"), etc.
 *   3. nvmlInit_v2() → nvmlDeviceGetHandleByIndex_v2() → nvmlDeviceGetMemoryInfo()
 *   4. Uses VRAM info to decide how many model layers to offload to GPU
 *
 * Each function forwards to the same GPU proxy daemon via TCP/vsock,
 * reusing the shared transport layer.
 *
 * Build: gcc -shared -fPIC -o libnvidia-ml.so.1 nvml_shim.c -lpthread
 *        ln -sf libnvidia-ml.so.1 libnvidia-ml.so
 *
 * No CUDA or NVML dependency — this replaces libnvidia-ml.so entirely.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Set transport log prefix before including shared transport */
#define TRANSPORT_LOG_PREFIX "nvml-shim"
#include "transport.h"
#include "transport.c"

/* ================================================================
 * NVML type definitions
 * ================================================================ */

typedef int nvmlReturn_t;
#define NVML_SUCCESS 0
#define NVML_ERROR_INVALID_ARGUMENT 2
#define NVML_ERROR_NOT_FOUND 6
#define NVML_ERROR_UNINITIALIZED 1

/* Device handle — we use (void *)(uintptr_t)(index + 1) as a non-NULL opaque */
typedef void *nvmlDevice_t;

typedef struct {
    unsigned long long total;  /* Total VRAM in bytes */
    unsigned long long free;   /* Free VRAM in bytes */
    unsigned long long used;   /* Used VRAM in bytes */
} nvmlMemory_t;

/* ================================================================
 * State
 * ================================================================ */

static int g_nvml_initialized = 0;
static int g_nvml_device_count = -1;

/* Cached device properties */
static GpuDeviceProperties g_nvml_cached_props;
static int g_nvml_props_valid = 0;

static int nvml_ensure_props(void)
{
    if (g_nvml_props_valid)
        return 0;

    GpuGetDevicePropertiesRequest req = { .device = 0 };
    int err = transport_rpc_call(GPU_CMD_GET_DEVICE_PROPERTIES,
                                 &req, sizeof(req),
                                 &g_nvml_cached_props, sizeof(g_nvml_cached_props),
                                 NULL);
    if (err == 0)
        g_nvml_props_valid = 1;
    return err;
}

/* ================================================================
 * Exported NVML functions
 * ================================================================ */

nvmlReturn_t nvmlInit_v2(void)
{
    TRANSPORT_LOG("nvmlInit_v2()");

    if (transport_ensure_connected() < 0)
        return NVML_ERROR_NOT_FOUND;

    g_nvml_initialized = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlInit(void)
{
    return nvmlInit_v2();
}

nvmlReturn_t nvmlShutdown(void)
{
    TRANSPORT_LOG("nvmlShutdown()");
    g_nvml_initialized = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count)
{
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;

    GpuGetDeviceCountResponse resp;
    int err = transport_rpc_call(GPU_CMD_GET_DEVICE_COUNT,
                                 NULL, 0, &resp, sizeof(resp), NULL);
    if (err == 0) {
        *count = (unsigned int)resp.count;
        g_nvml_device_count = resp.count;
    } else {
        *count = 0;
    }
    return (err == 0) ? NVML_SUCCESS : NVML_ERROR_NOT_FOUND;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device)
{
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;

    /* Fetch device count if not cached */
    if (g_nvml_device_count < 0) {
        unsigned int count;
        nvmlReturn_t rc = nvmlDeviceGetCount_v2(&count);
        if (rc != NVML_SUCCESS) return rc;
    }

    if ((int)index >= g_nvml_device_count)
        return NVML_ERROR_NOT_FOUND;

    /* Encode index as opaque handle (non-NULL) */
    *device = (nvmlDevice_t)(uintptr_t)(index + 1);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)
{
    if (!memory) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;

    GpuMemInfoResponse resp;
    int err = transport_rpc_call(GPU_CMD_MEM_GET_INFO,
                                 NULL, 0,
                                 &resp, sizeof(resp), NULL);
    if (err == 0) {
        memory->total = (unsigned long long)resp.total;
        memory->free  = (unsigned long long)resp.free;
        memory->used  = memory->total - memory->free;
        TRANSPORT_LOG("nvmlDeviceGetMemoryInfo: total=%llu free=%llu used=%llu",
                      memory->total, memory->free, memory->used);
    } else {
        memset(memory, 0, sizeof(*memory));
    }
    return (err == 0) ? NVML_SUCCESS : NVML_ERROR_NOT_FOUND;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int len)
{
    if (!name || len == 0) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;

    int err = nvml_ensure_props();
    if (err != 0) {
        name[0] = '\0';
        return NVML_ERROR_NOT_FOUND;
    }

    strncpy(name, g_nvml_cached_props.name, (size_t)(len - 1));
    name[len - 1] = '\0';
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int len)
{
    if (!uuid || len == 0) return NVML_ERROR_INVALID_ARGUMENT;

    /* Decode device index from handle */
    int index = (int)((uintptr_t)device - 1);

    GpuDeviceUuidRequest req = { .device = index };
    GpuDeviceUuidResponse resp;
    int err = transport_rpc_call(GPU_CMD_GET_DEVICE_UUID,
                                 &req, sizeof(req),
                                 &resp, sizeof(resp), NULL);
    if (err == 0) {
        /* Format as GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx */
        snprintf(uuid, len,
                 "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                 resp.uuid[0],  resp.uuid[1],  resp.uuid[2],  resp.uuid[3],
                 resp.uuid[4],  resp.uuid[5],  resp.uuid[6],  resp.uuid[7],
                 resp.uuid[8],  resp.uuid[9],  resp.uuid[10], resp.uuid[11],
                 resp.uuid[12], resp.uuid[13], resp.uuid[14], resp.uuid[15]);
    } else {
        snprintf(uuid, len, "GPU-00000000-0000-0000-0000-000000000000");
    }
    return (err == 0) ? NVML_SUCCESS : NVML_ERROR_NOT_FOUND;
}

/* ================================================================
 * Additional stubs for NVML functions some frameworks check
 * ================================================================ */

nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)
{
    if (!version || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "535.129.03");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length)
{
    if (!version || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "12.535.129.03");
    return NVML_SUCCESS;
}

/* ================================================================
 * Cleanup on library unload
 * ================================================================ */

__attribute__((destructor))
static void nvml_shim_cleanup(void)
{
    transport_disconnect();
}
