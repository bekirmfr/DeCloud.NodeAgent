/*
 * DeCloud CUDA Shim (LD_PRELOAD)
 *
 * Intercepts CUDA Runtime API calls inside a guest VM and forwards them
 * to the host GPU proxy daemon over virtio-vsock.
 *
 * Also intercepts the CUDA registration hooks (__cudaRegisterFatBinary,
 * __cudaRegisterFunction) that the NVIDIA compiler toolchain emits into
 * every CUDA application. This is required for kernel launch: the fat
 * binary (PTX/cubin) must be sent to the daemon so it can load the
 * module via the Driver API, and function registrations map host-side
 * stub pointers to device function names.
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
#define cudaErrorInvalidValue 1

typedef enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
} cudaMemcpyKind_t;

/* Stream/event are opaque pointers in the real CUDA runtime */
typedef void *cudaStream_t;
typedef void *cudaEvent_t;

/* dim3 — used by cudaLaunchKernel */
typedef struct {
    unsigned int x, y, z;
} dim3;

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
 * CUDA fat binary structures (from NVIDIA's internal headers)
 *
 * When nvcc compiles a .cu file, it generates:
 *   __cudaRegisterFatBinary(fatCubin) at static init time
 *   __cudaRegisterFunction(handle, hostFun, deviceName, ...) per kernel
 *   __cudaUnregisterFatBinary(handle) at static destroy time
 *
 * The fatCubin pointer points to a __fatBinC_Wrapper_t which
 * contains the actual fat binary data (PTX + cubin for various archs).
 * ================================================================ */

#define FATBINC_MAGIC   0x466243B1  /* __fatBinC_Wrapper_t.magic */
#define FATBIN_MAGIC    0xBA55ED50  /* __cudaFatMAGIC2 */

typedef struct {
    int   magic;
    int   version;
    const unsigned long long *data;  /* Points to the actual fatbin */
    void *filename_or_fatbins;
} __fatBinC_Wrapper_t;

/* ================================================================
 * Per-function metadata cache (returned by daemon after register)
 * ================================================================ */

#define MAX_REGISTERED_FUNCTIONS 512

typedef struct {
    uint64_t host_func_ptr;                      /* Key: the host stub address */
    uint64_t module_handle;                      /* Daemon-side module slot */
    uint32_t num_params;
    uint32_t param_sizes[GPU_MAX_KERNEL_PARAMS];
    int      registered;                         /* 1 if valid */
} RegisteredFunction;

static RegisteredFunction g_functions[MAX_REGISTERED_FUNCTIONS];
static int g_function_count = 0;

/* Module handle from the most recent __cudaRegisterFatBinary */
static uint64_t g_current_module_handle = 0;
static int g_module_registered = 0;

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

    ssize_t n = write(fd, &hdr, sizeof(hdr));
    if (n == (ssize_t)sizeof(hdr)) {
        write(fd, &hello, sizeof(hello));
    }

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

/* Send request, receive response. Returns cudaError_t from daemon. */
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
    close(g_conn_fd);
    g_conn_fd = -1;
    g_initialized = 0;
    pthread_mutex_unlock(&g_conn_lock);
    return cudaErrorNoDevice;
}

/* ================================================================
 * CUDA registration hooks — intercepted at static init time
 *
 * These are called by the CUDA runtime's internal __cudaRegisterAll()
 * function, which nvcc emits into every compiled .cu file's static
 * initializers.
 * ================================================================ */

/*
 * Determine the size of the fat binary data from the __fatBinC_Wrapper_t.
 * The fat binary starts with a header that includes the total size.
 */
static size_t get_fatbin_size(const void *fatbin_data)
{
    /* The fat binary header: magic(4) + version(2) + headerSize(2) + totalSize(8) */
    const uint8_t *p = (const uint8_t *)fatbin_data;
    uint32_t magic;
    memcpy(&magic, p, 4);
    if (magic != FATBIN_MAGIC) {
        /* Not a recognized fat binary — try treating the first 8 bytes
         * after a 8-byte header as the size */
        uint64_t size;
        memcpy(&size, p + 8, sizeof(size));
        return (size_t)size;
    }
    /* Standard fatbin: total size at offset 8 */
    uint64_t total_size;
    memcpy(&total_size, p + 8, sizeof(total_size));
    return (size_t)total_size;
}

void **__cudaRegisterFatBinary(void *fatCubin)
{
    SHIM_LOG("__cudaRegisterFatBinary(%p)", fatCubin);

    __fatBinC_Wrapper_t *wrapper = (__fatBinC_Wrapper_t *)fatCubin;
    if (!wrapper || wrapper->magic != FATBINC_MAGIC) {
        SHIM_LOG("  bad fatbin wrapper magic");
        /* Return a dummy handle — we can't fail here without crashing the app */
        static void *dummy_handle = NULL;
        return &dummy_handle;
    }

    const void *fatbin_data = (const void *)wrapper->data;
    size_t fatbin_size = get_fatbin_size(fatbin_data);

    SHIM_LOG("  fatbin data=%p size=%zu", fatbin_data, fatbin_size);

    if (fatbin_size == 0 || fatbin_size > GPU_PROXY_MAX_PAYLOAD) {
        SHIM_LOG("  fatbin size invalid or too large, skipping module registration");
        static void *dummy_handle = NULL;
        return &dummy_handle;
    }

    /* Build request: [GpuRegisterModuleRequest][fatbin data] */
    uint32_t req_len = sizeof(GpuRegisterModuleRequest) + (uint32_t)fatbin_size;
    void *req_buf = malloc(req_len);
    if (!req_buf) {
        static void *dummy_handle = NULL;
        return &dummy_handle;
    }

    GpuRegisterModuleRequest *req = (GpuRegisterModuleRequest *)req_buf;
    req->fatbin_size = (uint64_t)fatbin_size;
    memcpy((uint8_t *)req_buf + sizeof(GpuRegisterModuleRequest), fatbin_data, fatbin_size);

    GpuRegisterModuleResponse resp;
    int err = rpc_call(GPU_CMD_REGISTER_MODULE, req_buf, req_len,
                       &resp, sizeof(resp), NULL);
    free(req_buf);

    if (err == 0) {
        g_current_module_handle = resp.module_handle;
        g_module_registered = 1;
        SHIM_LOG("  module registered -> handle %lu",
                 (unsigned long)g_current_module_handle);
    } else {
        SHIM_LOG("  module registration failed (err=%d)", err);
        g_module_registered = 0;
    }

    /*
     * Return a pointer-to-pointer that the runtime passes to
     * __cudaRegisterFunction as the first arg (fatCubinHandle).
     * We store our module handle in a static so the register function
     * calls can find it. The returned value itself isn't dereferenced
     * by guest code — it's just an opaque cookie.
     */
    static void *handle_ptr = (void *)1; /* non-NULL to satisfy runtime checks */
    return &handle_ptr;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    SHIM_LOG("__cudaUnregisterFatBinary(%p)", fatCubinHandle);

    if (g_module_registered) {
        GpuUnregisterModuleRequest req = {
            .module_handle = g_current_module_handle,
        };
        rpc_call(GPU_CMD_UNREGISTER_MODULE, &req, sizeof(req), NULL, 0, NULL);
        g_module_registered = 0;
    }
}

void __cudaRegisterFunction(
    void   **fatCubinHandle,
    const char *hostFun,
    char       *deviceFun,
    const char *deviceName,
    int         thread_limit,
    void       *tid,
    void       *bid,
    void       *bDim,
    void       *gDim,
    int        *wSize)
{
    /* deviceFun is the mangled device function name (e.g. "_Z9my_kernelPfS_i") */
    const char *name = deviceFun ? deviceFun : deviceName;
    SHIM_LOG("__cudaRegisterFunction(hostFun=%p, device='%s')",
             hostFun, name ? name : "(null)");

    if (!g_module_registered || !name) {
        SHIM_LOG("  skipping (no module or no name)");
        return;
    }

    if (g_function_count >= MAX_REGISTERED_FUNCTIONS) {
        SHIM_LOG("  too many registered functions (%d)", g_function_count);
        return;
    }

    uint32_t name_len = (uint32_t)strlen(name) + 1; /* include null terminator */

    /* Build request: [GpuRegisterFunctionRequest][device_name] */
    uint32_t req_len = sizeof(GpuRegisterFunctionRequest) + name_len;
    void *req_buf = malloc(req_len);
    if (!req_buf) return;

    GpuRegisterFunctionRequest *req = (GpuRegisterFunctionRequest *)req_buf;
    req->module_handle = g_current_module_handle;
    req->host_func_ptr = (uint64_t)(uintptr_t)hostFun;
    req->device_name_len = name_len;
    memcpy((uint8_t *)req_buf + sizeof(GpuRegisterFunctionRequest), name, name_len);

    GpuRegisterFunctionResponse resp;
    memset(&resp, 0, sizeof(resp));
    int err = rpc_call(GPU_CMD_REGISTER_FUNCTION, req_buf, req_len,
                       &resp, sizeof(resp), NULL);
    free(req_buf);

    if (err == 0) {
        RegisteredFunction *rf = &g_functions[g_function_count++];
        rf->host_func_ptr = (uint64_t)(uintptr_t)hostFun;
        rf->module_handle = g_current_module_handle;
        rf->num_params = resp.num_params;
        memcpy(rf->param_sizes, resp.param_sizes,
               resp.num_params * sizeof(uint32_t));
        rf->registered = 1;
        SHIM_LOG("  function registered: %u params", resp.num_params);
    } else {
        SHIM_LOG("  function registration failed (err=%d)", err);
    }
}

/* __cudaRegisterVar — stub: device global variables (not needed for basic kernel launch) */
void __cudaRegisterVar(
    void **fatCubinHandle,
    char  *hostVar,
    char  *deviceAddress,
    const char *deviceName,
    int    ext,
    size_t size,
    int    constant,
    int    global)
{
    SHIM_LOG("__cudaRegisterVar('%s', size=%zu) — stub", deviceName, size);
    /* TODO: implement for device-side global variables if needed */
}

/* ================================================================
 * Intercepted CUDA Runtime API functions — existing
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
    if (!prop) return cudaErrorInvalidValue;
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
    if (!devPtr) return cudaErrorInvalidValue;

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
        size_t total = sizeof(req) + count;
        void *buf = malloc(total);
        if (!buf) return cudaErrorInvalidValue;
        memcpy(buf, &req, sizeof(req));
        memcpy((char *)buf + sizeof(req), src, count);

        int err = rpc_call(GPU_CMD_MEMCPY, buf, (uint32_t)total,
                           NULL, 0, NULL);
        free(buf);
        return err;
    }

    case cudaMemcpyDeviceToHost: {
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
        return cudaErrorInvalidValue;
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
 * Kernel launch
 * ================================================================ */

static RegisteredFunction *find_registered_function(uint64_t host_func_ptr)
{
    for (int i = 0; i < g_function_count; i++) {
        if (g_functions[i].registered &&
            g_functions[i].host_func_ptr == host_func_ptr)
            return &g_functions[i];
    }
    return NULL;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                              void **args, size_t sharedMem,
                              cudaStream_t stream)
{
    RegisteredFunction *rf = find_registered_function(
        (uint64_t)(uintptr_t)func);
    if (!rf) {
        SHIM_LOG("cudaLaunchKernel: unknown function %p", func);
        return cudaErrorInvalidValue;
    }

    /*
     * Serialize arguments. Each arg in args[i] is a pointer to the actual
     * value. We know the size of each from the param metadata the daemon
     * sent back during registration (or default to 8 bytes = sizeof(void*)).
     */
    uint32_t args_total = 0;
    for (uint32_t i = 0; i < rf->num_params; i++) {
        args_total += rf->param_sizes[i];
    }

    /* If daemon didn't provide sizes (CUDA < 12), assume 8 bytes per param.
     * The shim must determine param count from the launch call. Since the
     * runtime API doesn't expose this, we use the daemon-reported count.
     * If num_params is 0 (unknown), we can't launch. */
    if (rf->num_params == 0) {
        SHIM_LOG("cudaLaunchKernel: param metadata unavailable for %p", func);
        return cudaErrorInvalidValue;
    }

    uint32_t req_len = sizeof(GpuLaunchKernelRequest) + args_total;
    void *req_buf = malloc(req_len);
    if (!req_buf) return cudaErrorInvalidValue;

    GpuLaunchKernelRequest *req = (GpuLaunchKernelRequest *)req_buf;
    req->host_func_ptr = (uint64_t)(uintptr_t)func;
    req->grid_dim_x = gridDim.x;
    req->grid_dim_y = gridDim.y;
    req->grid_dim_z = gridDim.z;
    req->block_dim_x = blockDim.x;
    req->block_dim_y = blockDim.y;
    req->block_dim_z = blockDim.z;
    req->shared_mem_bytes = (uint64_t)sharedMem;
    req->stream_handle = (uint64_t)(uintptr_t)stream;
    req->num_params = rf->num_params;
    req->args_total_size = args_total;

    /* Copy each argument value contiguously after the header */
    uint8_t *dst = (uint8_t *)req_buf + sizeof(GpuLaunchKernelRequest);
    for (uint32_t i = 0; i < rf->num_params; i++) {
        memcpy(dst, args[i], rf->param_sizes[i]);
        dst += rf->param_sizes[i];
    }

    int err = rpc_call(GPU_CMD_LAUNCH_KERNEL, req_buf, req_len, NULL, 0, NULL);
    free(req_buf);
    return err;
}

/* ================================================================
 * Stream operations
 * ================================================================ */

cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
    if (!pStream) return cudaErrorInvalidValue;

    GpuStreamCreateRequest req = { .flags = 0 };
    GpuStreamCreateResponse resp;
    int err = rpc_call(GPU_CMD_STREAM_CREATE, &req, sizeof(req),
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *pStream = (cudaStream_t)(uintptr_t)resp.stream_handle;
    } else {
        *pStream = NULL;
    }
    return err;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    if (!pStream) return cudaErrorInvalidValue;

    GpuStreamCreateRequest req = { .flags = flags };
    GpuStreamCreateResponse resp;
    int err = rpc_call(GPU_CMD_STREAM_CREATE, &req, sizeof(req),
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *pStream = (cudaStream_t)(uintptr_t)resp.stream_handle;
    } else {
        *pStream = NULL;
    }
    return err;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    GpuStreamDestroyRequest req = {
        .stream_handle = (uint64_t)(uintptr_t)stream,
    };
    return rpc_call(GPU_CMD_STREAM_DESTROY, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    GpuStreamSynchronizeRequest req = {
        .stream_handle = (uint64_t)(uintptr_t)stream,
    };
    return rpc_call(GPU_CMD_STREAM_SYNCHRONIZE, &req, sizeof(req), NULL, 0, NULL);
}

/* ================================================================
 * Event operations
 * ================================================================ */

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    if (!event) return cudaErrorInvalidValue;

    GpuEventCreateRequest req = { .flags = 0 };
    GpuEventCreateResponse resp;
    int err = rpc_call(GPU_CMD_EVENT_CREATE, &req, sizeof(req),
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *event = (cudaEvent_t)(uintptr_t)resp.event_handle;
    } else {
        *event = NULL;
    }
    return err;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    if (!event) return cudaErrorInvalidValue;

    GpuEventCreateRequest req = { .flags = flags };
    GpuEventCreateResponse resp;
    int err = rpc_call(GPU_CMD_EVENT_CREATE, &req, sizeof(req),
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *event = (cudaEvent_t)(uintptr_t)resp.event_handle;
    } else {
        *event = NULL;
    }
    return err;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    GpuEventDestroyRequest req = {
        .event_handle = (uint64_t)(uintptr_t)event,
    };
    return rpc_call(GPU_CMD_EVENT_DESTROY, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    GpuEventRecordRequest req = {
        .event_handle = (uint64_t)(uintptr_t)event,
        .stream_handle = (uint64_t)(uintptr_t)stream,
    };
    return rpc_call(GPU_CMD_EVENT_RECORD, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    GpuEventSynchronizeRequest req = {
        .event_handle = (uint64_t)(uintptr_t)event,
    };
    return rpc_call(GPU_CMD_EVENT_SYNCHRONIZE, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    if (!ms) return cudaErrorInvalidValue;

    GpuEventElapsedTimeRequest req = {
        .start_event = (uint64_t)(uintptr_t)start,
        .end_event   = (uint64_t)(uintptr_t)end,
    };
    GpuEventElapsedTimeResponse resp;
    int err = rpc_call(GPU_CMD_EVENT_ELAPSED_TIME, &req, sizeof(req),
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *ms = resp.elapsed_ms;
    }
    return err;
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
    if (value) *value = 0;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
    if (runtimeVersion) *runtimeVersion = 12000;
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
    case 1:   return "invalid value";
    case 100: return "no CUDA-capable device is detected (proxy unavailable)";
    default:  return "unknown error";
    }
}

const char *cudaGetErrorName(cudaError_t error)
{
    switch (error) {
    case 0:   return "cudaSuccess";
    case 1:   return "cudaErrorInvalidValue";
    case 100: return "cudaErrorNoDevice";
    default:  return "cudaErrorUnknown";
    }
}

cudaError_t cudaGetLastError(void)
{
    return cudaSuccess;
}

cudaError_t cudaPeekAtLastError(void)
{
    return cudaSuccess;
}

/* Cleanup on unload */
__attribute__((destructor))
static void shim_cleanup(void)
{
    pthread_mutex_lock(&g_conn_lock);
    if (g_conn_fd >= 0) {
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
