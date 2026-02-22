/*
 * DeCloud CUDA Shim (LD_PRELOAD)
 *
 * Intercepts CUDA Runtime API calls inside a guest VM and forwards them
 * to the host GPU proxy daemon over virtio-vsock.
 *
 * Usage (inside VM):
 *   export LD_PRELOAD=/usr/local/lib/libdecloud_cuda_shim.so
 *   export DECLOUD_GPU_PROXY_CID=2     # CID 2 = host
 *   export DECLOUD_GPU_PROXY_PORT=9999
 *   python -c "import torch; print(torch.cuda.is_available())"
 *
 * Build: gcc -shared -fPIC -o libdecloud_cuda_shim.so cuda_shim.c -lpthread
 *        (No CUDA dependency — this replaces libcudart)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>

#include "../proto/gpu_proxy_proto.h"

/* ================================================================
 * CUDA type definitions (we don't link against CUDA, so define them)
 * ================================================================ */

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorNoDevice 100

typedef enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
} cudaMemcpyKind_t;

/* Subset of cudaDeviceProp that matches what frameworks check */
struct cudaDeviceProp {
    char     name[256];
    size_t   totalGlobalMem;
    size_t   sharedMemPerBlock;
    int      regsPerBlock;
    int      warpSize;
    size_t   memPitch;
    int      maxThreadsPerBlock;
    int      maxThreadsDim[3];
    int      maxGridSize[3];
    int      clockRate;
    size_t   totalConstMem;
    int      major;
    int      minor;
    size_t   textureAlignment;
    size_t   texturePitchAlignment;
    int      deviceOverlap;
    int      multiProcessorCount;
    int      kernelExecTimeoutEnabled;
    int      integrated;
    int      canMapHostMemory;
    int      computeMode;
    int      maxTexture1D;
    int      maxTexture1DMipmap;
    int      maxTexture1DLinear;
    int      maxTexture2D[2];
    int      maxTexture2DMipmap[2];
    int      maxTexture2DLinear[3];
    int      maxTexture2DGather[2];
    int      maxTexture3D[3];
    int      maxTexture3DAlt[3];
    int      maxTextureCubemap;
    int      maxTexture1DLayered[2];
    int      maxTexture2DLayered[3];
    int      maxTextureCubemapLayered[2];
    int      maxSurface1D;
    int      maxSurface2D[2];
    int      maxSurface3D[3];
    int      maxSurface1DLayered[2];
    int      maxSurface2DLayered[3];
    int      maxSurfaceCubemap;
    int      maxSurfaceCubemapLayered[2];
    size_t   surfaceAlignment;
    int      concurrentKernels;
    int      ECCEnabled;
    int      pciBusID;
    int      pciDeviceID;
    int      pciDomainID;
    int      tccDriver;
    int      asyncEngineCount;
    int      unifiedAddressing;
    int      memoryClockRate;
    int      memoryBusWidth;
    int      l2CacheSize;
    int      persistingL2CacheMaxSize;
    int      maxThreadsPerMultiProcessor;
    int      streamPrioritiesSupported;
    int      globalL1CacheSupported;
    int      localL1CacheSupported;
    size_t   sharedMemPerMultiprocessor;
    int      regsPerMultiprocessor;
    int      managedMemory;
    int      isMultiGpuBoard;
    int      multiGpuBoardGroupID;
    /* ... more fields exist but are rarely checked ... */
    char     _padding[512]; /* Safety padding for struct size mismatches */
};

/* ================================================================
 * Connection management
 * ================================================================ */

static pthread_mutex_t g_conn_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_conn_fd = -1;
static int g_initialized = 0;

#define SHIM_LOG(fmt, ...) \
    fprintf(stderr, "[cuda-shim] " fmt "\n", ##__VA_ARGS__)

static int get_env_int(const char *name, int def)
{
    const char *val = getenv(name);
    return val ? atoi(val) : def;
}

static int ensure_connected(void)
{
    pthread_mutex_lock(&g_conn_lock);
    if (g_conn_fd >= 0) {
        pthread_mutex_unlock(&g_conn_lock);
        return 0;
    }

    int cid = get_env_int("DECLOUD_GPU_PROXY_CID", VMADDR_CID_HOST);
    int port = get_env_int("DECLOUD_GPU_PROXY_PORT", GPU_PROXY_PORT);

    int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (fd < 0) {
        SHIM_LOG("socket(AF_VSOCK) failed: %s", strerror(errno));
        pthread_mutex_unlock(&g_conn_lock);
        return -1;
    }

    struct sockaddr_vm addr = {
        .svm_family = AF_VSOCK,
        .svm_cid    = (unsigned int)cid,
        .svm_port   = (unsigned int)port,
    };

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        SHIM_LOG("connect(vsock CID=%d port=%d) failed: %s",
                 cid, port, strerror(errno));
        close(fd);
        pthread_mutex_unlock(&g_conn_lock);
        return -1;
    }

    g_conn_fd = fd;

    /* Send HELLO */
    GpuHelloRequest hello = {
        .shim_version = GPU_PROXY_VERSION,
        .pid = (uint32_t)getpid(),
    };
    GpuProxyHeader hdr = {
        .magic = GPU_PROXY_MAGIC,
        .version = GPU_PROXY_VERSION,
        .cmd = GPU_CMD_HELLO,
        .payload_len = sizeof(hello),
        .status = 0,
    };

    /* Write header + payload; if it fails, clean up */
    ssize_t n = write(fd, &hdr, sizeof(hdr));
    if (n == (ssize_t)sizeof(hdr)) {
        write(fd, &hello, sizeof(hello));
    }

    /* Read response (best-effort, non-blocking would be overkill here) */
    GpuProxyHeader resp_hdr;
    if (read(fd, &resp_hdr, sizeof(resp_hdr)) == sizeof(resp_hdr) &&
        resp_hdr.payload_len >= sizeof(GpuHelloResponse)) {
        GpuHelloResponse resp;
        read(fd, &resp, sizeof(resp));
        SHIM_LOG("Connected to GPU proxy (v%u, %u devices)",
                 resp.daemon_version, resp.device_count);
    }

    g_initialized = 1;
    pthread_mutex_unlock(&g_conn_lock);
    return 0;
}

/* ================================================================
 * I/O helpers
 * ================================================================ */

static int read_exact(int fd, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, (char *)buf + done, len - done);
        if (n <= 0) {
            if (n == 0) return -1;
            if (errno == EINTR) continue;
            return -1;
        }
        done += n;
    }
    return 0;
}

static int write_exact(int fd, const void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, (const char *)buf + done, len - done);
        if (n <= 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        done += n;
    }
    return 0;
}

/* Send request, receive response. Caller must hold g_conn_lock or
 * ensure single-threaded access. Returns cudaError_t from daemon.
 * Response payload (if any) is written to resp_buf. */
static int rpc_call(uint8_t cmd,
                    const void *req_payload, uint32_t req_len,
                    void *resp_buf, uint32_t resp_buf_size,
                    uint32_t *resp_actual_len)
{
    if (ensure_connected() < 0)
        return cudaErrorNoDevice;

    pthread_mutex_lock(&g_conn_lock);

    GpuProxyHeader hdr = {
        .magic = GPU_PROXY_MAGIC,
        .version = GPU_PROXY_VERSION,
        .cmd = cmd,
        .payload_len = req_len,
        .status = 0,
    };

    if (write_exact(g_conn_fd, &hdr, sizeof(hdr)) < 0) goto err;
    if (req_len > 0 && req_payload) {
        if (write_exact(g_conn_fd, req_payload, req_len) < 0) goto err;
    }

    /* Read response header */
    GpuProxyHeader resp_hdr;
    if (read_exact(g_conn_fd, &resp_hdr, sizeof(resp_hdr)) < 0) goto err;

    if (resp_hdr.magic != GPU_PROXY_MAGIC) {
        SHIM_LOG("bad response magic 0x%08x", resp_hdr.magic);
        goto err;
    }

    /* Read response payload */
    if (resp_hdr.payload_len > 0) {
        if (resp_buf && resp_hdr.payload_len <= resp_buf_size) {
            if (read_exact(g_conn_fd, resp_buf, resp_hdr.payload_len) < 0)
                goto err;
        } else {
            /* Drain excess data */
            char drain[4096];
            uint32_t remaining = resp_hdr.payload_len;
            while (remaining > 0) {
                uint32_t chunk = remaining < sizeof(drain) ? remaining : sizeof(drain);
                if (read_exact(g_conn_fd, drain, chunk) < 0) goto err;
                remaining -= chunk;
            }
        }
    }

    if (resp_actual_len) *resp_actual_len = resp_hdr.payload_len;
    pthread_mutex_unlock(&g_conn_lock);
    return resp_hdr.status;

err:
    /* Connection broken — reset */
    close(g_conn_fd);
    g_conn_fd = -1;
    g_initialized = 0;
    pthread_mutex_unlock(&g_conn_lock);
    return cudaErrorNoDevice;
}

/* ================================================================
 * Intercepted CUDA Runtime API functions
 *
 * These are the symbols that LD_PRELOAD overrides. Application code
 * (PyTorch, TensorFlow, etc.) calls these thinking they're libcudart.
 * ================================================================ */

cudaError_t cudaGetDeviceCount(int *count)
{
    GpuGetDeviceCountResponse resp;
    int err = rpc_call(GPU_CMD_GET_DEVICE_COUNT, NULL, 0,
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess && count) {
        *count = resp.count;
    } else if (count) {
        *count = 0;
    }
    return err;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    if (!prop) return 1;
    memset(prop, 0, sizeof(*prop));

    GpuGetDevicePropertiesRequest req = { .device = device };
    GpuDeviceProperties resp;
    int err = rpc_call(GPU_CMD_GET_DEVICE_PROPERTIES,
                       &req, sizeof(req), &resp, sizeof(resp), NULL);

    if (err == cudaSuccess) {
        strncpy(prop->name, resp.name, sizeof(prop->name) - 1);
        prop->totalGlobalMem               = resp.total_global_mem;
        prop->sharedMemPerBlock             = resp.shared_mem_per_block;
        prop->regsPerBlock                  = resp.regs_per_block;
        prop->warpSize                      = resp.warp_size;
        prop->maxThreadsPerBlock            = resp.max_threads_per_block;
        prop->maxThreadsDim[0]              = resp.max_threads_dim[0];
        prop->maxThreadsDim[1]              = resp.max_threads_dim[1];
        prop->maxThreadsDim[2]              = resp.max_threads_dim[2];
        prop->maxGridSize[0]                = resp.max_grid_size[0];
        prop->maxGridSize[1]                = resp.max_grid_size[1];
        prop->maxGridSize[2]                = resp.max_grid_size[2];
        prop->clockRate                     = resp.clock_rate;
        prop->multiProcessorCount           = resp.multi_processor_count;
        prop->major                         = resp.major;
        prop->minor                         = resp.minor;
        prop->maxThreadsPerMultiProcessor   = resp.max_threads_per_multiprocessor;
        prop->memoryClockRate               = resp.memory_clock_rate;
        prop->memoryBusWidth                = resp.memory_bus_width;
        prop->l2CacheSize                   = resp.l2_cache_size;
    }
    return err;
}

cudaError_t cudaSetDevice(int device)
{
    GpuSetDeviceRequest req = { .device = device };
    return rpc_call(GPU_CMD_SET_DEVICE, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    if (!devPtr) return 1;

    GpuMallocRequest req = { .size = (uint64_t)size };
    GpuMallocResponse resp;
    int err = rpc_call(GPU_CMD_MALLOC, &req, sizeof(req),
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *devPtr = (void *)(uintptr_t)resp.device_ptr;
    } else {
        *devPtr = NULL;
    }
    return err;
}

cudaError_t cudaFree(void *devPtr)
{
    GpuFreeRequest req = { .device_ptr = (uint64_t)(uintptr_t)devPtr };
    return rpc_call(GPU_CMD_FREE, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind_t kind)
{
    GpuMemcpyRequest req = {
        .dst   = (uint64_t)(uintptr_t)dst,
        .src   = (uint64_t)(uintptr_t)src,
        .count = (uint64_t)count,
        .kind  = (int32_t)kind,
    };

    switch (kind) {
    case cudaMemcpyHostToDevice: {
        /* Send request struct + host data in one payload */
        size_t total = sizeof(req) + count;
        void *buf = malloc(total);
        if (!buf) return 1;
        memcpy(buf, &req, sizeof(req));
        memcpy((char *)buf + sizeof(req), src, count);

        int err = rpc_call(GPU_CMD_MEMCPY, buf, (uint32_t)total,
                           NULL, 0, NULL);
        free(buf);
        return err;
    }

    case cudaMemcpyDeviceToHost: {
        /* Response carries the data */
        uint32_t actual = 0;
        int err = rpc_call(GPU_CMD_MEMCPY, &req, sizeof(req),
                           dst, (uint32_t)count, &actual);
        return err;
    }

    case cudaMemcpyDeviceToDevice: {
        return rpc_call(GPU_CMD_MEMCPY, &req, sizeof(req), NULL, 0, NULL);
    }

    case cudaMemcpyHostToHost:
        memcpy(dst, src, count);
        return cudaSuccess;

    default:
        return 1;
    }
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    GpuMemsetRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)devPtr,
        .value = value,
        .count = (uint64_t)count,
    };
    return rpc_call(GPU_CMD_MEMSET, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaDeviceSynchronize(void)
{
    return rpc_call(GPU_CMD_DEVICE_SYNCHRONIZE, NULL, 0, NULL, 0, NULL);
}

/* ================================================================
 * Stubs for functions that frameworks probe but don't need GPU for
 * ================================================================ */

cudaError_t cudaGetDevice(int *device)
{
    if (device) *device = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device)
{
    /* Return 0 for unknown attributes — most frameworks have fallbacks */
    if (value) *value = 0;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
    if (runtimeVersion) *runtimeVersion = 12000; /* Pretend CUDA 12.0 */
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *driverVersion)
{
    if (driverVersion) *driverVersion = 12000;
    return cudaSuccess;
}

const char *cudaGetErrorString(cudaError_t error)
{
    switch (error) {
    case 0:   return "no error";
    case 100: return "no CUDA-capable device is detected (proxy unavailable)";
    default:  return "unknown error";
    }
}

const char *cudaGetErrorName(cudaError_t error)
{
    switch (error) {
    case 0:   return "cudaSuccess";
    case 100: return "cudaErrorNoDevice";
    default:  return "cudaErrorUnknown";
    }
}

/* Cleanup on unload */
__attribute__((destructor))
static void shim_cleanup(void)
{
    pthread_mutex_lock(&g_conn_lock);
    if (g_conn_fd >= 0) {
        /* Best-effort GOODBYE */
        GpuProxyHeader hdr = {
            .magic = GPU_PROXY_MAGIC,
            .version = GPU_PROXY_VERSION,
            .cmd = GPU_CMD_GOODBYE,
            .payload_len = 0,
            .status = 0,
        };
        write(g_conn_fd, &hdr, sizeof(hdr));
        close(g_conn_fd);
        g_conn_fd = -1;
    }
    pthread_mutex_unlock(&g_conn_lock);
}
