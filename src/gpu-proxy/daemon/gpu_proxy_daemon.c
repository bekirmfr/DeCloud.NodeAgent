/*
 * DeCloud GPU Proxy Daemon
 *
 * Host-side service that listens on virtio-vsock and proxies CUDA Runtime
 * API calls from guest VMs to the real GPU.
 *
 * Each VM connects with its unique CID. The daemon maintains per-connection
 * state (loaded modules, registered functions, streams, events) for isolation.
 *
 * Kernel launch uses the CUDA Driver API (cuModuleLoadData, cuModuleGetFunction,
 * cuLaunchKernel) because the Runtime API's cudaLaunchKernel requires the
 * actual host-side function pointer from the compiled fat binary, which we
 * don't have — the guest sends us raw PTX/cubin instead.
 *
 * Build: gcc -o gpu-proxy-daemon gpu_proxy_daemon.c -lcuda -lcudart -lpthread
 * Run:   gpu-proxy-daemon [-p port] [-v]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../proto/gpu_proxy_proto.h"

/* ================================================================
 * Logging
 * ================================================================ */

static int g_verbose = 0;
static volatile sig_atomic_t g_running = 1;
static uint64_t g_kernel_timeout_us = GPU_PROXY_DEFAULT_KERNEL_TIMEOUT_US;

/* Monotonic clock helper — returns microseconds since an arbitrary epoch */
static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[gpu-proxy] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) \
    fprintf(stderr, "[gpu-proxy] ERROR: " fmt "\n", ##__VA_ARGS__)
#define LOG_DBG(fmt, ...) \
    do { if (g_verbose) fprintf(stdout, "[gpu-proxy] DBG: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ================================================================
 * I/O helpers — read/write exactly N bytes
 * ================================================================ */

static int read_exact(int fd, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, (char *)buf + done, len - done);
        if (n <= 0) {
            if (n == 0) return -1;   /* EOF */
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

/* ================================================================
 * Send a response header (optionally followed by payload)
 * ================================================================ */

static int send_response(int fd, uint8_t cmd, int32_t status,
                         const void *payload, uint32_t payload_len)
{
    GpuProxyHeader hdr = {
        .magic       = GPU_PROXY_MAGIC,
        .version     = GPU_PROXY_VERSION,
        .cmd         = cmd,
        .flags       = 0,
        .payload_len = payload_len,
        .status      = status,
    };
    if (write_exact(fd, &hdr, sizeof(hdr)) < 0) return -1;
    if (payload_len > 0 && payload) {
        if (write_exact(fd, payload, payload_len) < 0) return -1;
    }
    return 0;
}

/* ================================================================
 * Per-connection state: modules, functions, streams, events
 *
 * Each VM connection gets its own set of loaded CUDA modules and
 * registered functions. This provides isolation between VMs and
 * ensures cleanup when a VM disconnects.
 * ================================================================ */

#define MAX_MODULES   128
#define MAX_FUNCTIONS 512
#define MAX_STREAMS   64
#define MAX_EVENTS    128

typedef struct {
    CUmodule  module;
    int       in_use;
} ModuleSlot;

typedef struct {
    uint64_t   host_func_ptr;   /* Shim-side key */
    CUfunction cu_func;         /* Driver API function handle */
    uint32_t   num_params;
    uint32_t   param_sizes[GPU_MAX_KERNEL_PARAMS];
    uint32_t   param_offsets[GPU_MAX_KERNEL_PARAMS]; /* Byte offset for serialized args */
    int        in_use;
} FunctionSlot;

typedef struct {
    cudaStream_t stream;
    int          in_use;
} StreamSlot;

typedef struct {
    cudaEvent_t event;
    int         in_use;
} EventSlot;

/* Per-allocation tracking for memory quota enforcement */
#define MAX_ALLOCS 4096

typedef struct {
    uint64_t device_ptr;
    uint64_t size;
    int      in_use;
} AllocSlot;

typedef struct {
    int          fd;
    unsigned int peer_cid;
    uint32_t     conn_mode;       /* GpuConnectionMode: PROXY or METER */

    /* Per-connection CUDA Driver API context (proxy mode only) */
    CUcontext    cu_ctx;

    ModuleSlot   modules[MAX_MODULES];
    FunctionSlot functions[MAX_FUNCTIONS];
    StreamSlot   streams[MAX_STREAMS];
    EventSlot    events[MAX_EVENTS];

    /* Memory quota enforcement (proxy mode) */
    uint64_t  memory_quota;       /* 0 = unlimited */
    uint64_t  memory_allocated;   /* Current total allocated bytes */
    uint64_t  peak_memory;        /* High-water mark */
    uint64_t  total_alloc_bytes;  /* Cumulative allocation total */
    AllocSlot allocs[MAX_ALLOCS]; /* Track individual allocations */

    /* Metering / billing (both modes) */
    uint64_t  connect_time_us;    /* Timestamp of connection start */
    uint32_t  kernel_launches;    /* Total kernel launches */
    uint32_t  kernel_timeouts;    /* Kernels killed by timeout */
    uint64_t  kernel_time_us;     /* Cumulative kernel execution time */

    /* Last stats pushed by guest agent (meter mode only) */
    GpuReportUsageStats last_agent_stats;
} ConnectionCtx;

/* ================================================================
 * Module / function lookup helpers
 * ================================================================ */

static int alloc_module_slot(ConnectionCtx *ctx)
{
    for (int i = 0; i < MAX_MODULES; i++) {
        if (!ctx->modules[i].in_use) return i;
    }
    return -1;
}

static int alloc_function_slot(ConnectionCtx *ctx)
{
    for (int i = 0; i < MAX_FUNCTIONS; i++) {
        if (!ctx->functions[i].in_use) return i;
    }
    return -1;
}

static FunctionSlot *find_function(ConnectionCtx *ctx, uint64_t host_func_ptr)
{
    for (int i = 0; i < MAX_FUNCTIONS; i++) {
        if (ctx->functions[i].in_use &&
            ctx->functions[i].host_func_ptr == host_func_ptr)
            return &ctx->functions[i];
    }
    return NULL;
}

static int alloc_stream_slot(ConnectionCtx *ctx)
{
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (!ctx->streams[i].in_use) return i;
    }
    return -1;
}

static StreamSlot *find_stream(ConnectionCtx *ctx, uint64_t handle)
{
    if (handle == 0) return NULL; /* default stream */
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (ctx->streams[i].in_use &&
            (uint64_t)(uintptr_t)ctx->streams[i].stream == handle)
            return &ctx->streams[i];
    }
    return NULL;
}

static int alloc_event_slot(ConnectionCtx *ctx)
{
    for (int i = 0; i < MAX_EVENTS; i++) {
        if (!ctx->events[i].in_use) return i;
    }
    return -1;
}

static EventSlot *find_event(ConnectionCtx *ctx, uint64_t handle)
{
    for (int i = 0; i < MAX_EVENTS; i++) {
        if (ctx->events[i].in_use &&
            (uint64_t)(uintptr_t)ctx->events[i].event == handle)
            return &ctx->events[i];
    }
    return NULL;
}

/* Resolve a stream handle from the wire to a cudaStream_t.
 * 0 means the default stream (NULL). */
static cudaStream_t resolve_stream(ConnectionCtx *ctx, uint64_t handle)
{
    if (handle == 0) return NULL;
    StreamSlot *s = find_stream(ctx, handle);
    return s ? s->stream : NULL;
}

/* Resolve an event handle from the wire to a cudaEvent_t. */
static cudaEvent_t resolve_event(ConnectionCtx *ctx, uint64_t handle)
{
    EventSlot *e = find_event(ctx, handle);
    return e ? e->event : NULL;
}

/* Free all per-connection CUDA resources */
static void cleanup_connection(ConnectionCtx *ctx)
{
    /* Log final usage stats for billing */
    if (ctx->connect_time_us > 0) {
        uint64_t uptime = now_us() - ctx->connect_time_us;
        LOG_INFO("CID %u final stats: mem_peak=%lu alloc_total=%lu "
                 "kernels=%u timeouts=%u ktime=%lu µs uptime=%lu µs",
                 ctx->peer_cid,
                 (unsigned long)ctx->peak_memory,
                 (unsigned long)ctx->total_alloc_bytes,
                 ctx->kernel_launches,
                 ctx->kernel_timeouts,
                 (unsigned long)ctx->kernel_time_us,
                 (unsigned long)uptime);
    }

    /* Free tracked device allocations */
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (ctx->allocs[i].in_use) {
            cudaFree((void *)(uintptr_t)ctx->allocs[i].device_ptr);
            ctx->allocs[i].in_use = 0;
        }
    }

    for (int i = 0; i < MAX_EVENTS; i++) {
        if (ctx->events[i].in_use) {
            cudaEventDestroy(ctx->events[i].event);
            ctx->events[i].in_use = 0;
        }
    }
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (ctx->streams[i].in_use) {
            cudaStreamDestroy(ctx->streams[i].stream);
            ctx->streams[i].in_use = 0;
        }
    }
    for (int i = 0; i < MAX_FUNCTIONS; i++) {
        ctx->functions[i].in_use = 0;
    }
    for (int i = 0; i < MAX_MODULES; i++) {
        if (ctx->modules[i].in_use) {
            cuModuleUnload(ctx->modules[i].module);
            ctx->modules[i].in_use = 0;
        }
    }
    if (ctx->cu_ctx) {
        cuCtxDestroy(ctx->cu_ctx);
        ctx->cu_ctx = NULL;
    }
}

/* ================================================================
 * Command handlers — existing (device mgmt, memory, sync)
 * ================================================================ */

static int handle_hello(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    GpuHelloRequest req = {0};
    if (len >= 8) {
        /* Backwards compatible: old clients send 8 bytes (no conn_mode) */
        memcpy(&req, payload, len < sizeof(req) ? len : sizeof(req));
    } else {
        return send_response(ctx->fd, GPU_CMD_HELLO, -1, NULL, 0);
    }

    ctx->conn_mode = req.conn_mode;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    const char *mode_str = (req.conn_mode == GPU_CONN_METER)
                           ? "METER (passthrough)" : "PROXY";

    LOG_INFO("HELLO from guest PID %u (v%u, mode=%s), %d GPU(s) available",
             req.pid, req.shim_version, mode_str, count);

    GpuHelloResponse resp = {
        .daemon_version = GPU_PROXY_VERSION,
        .device_count   = (uint32_t)count,
    };
    return send_response(ctx->fd, GPU_CMD_HELLO, (int32_t)err,
                         &resp, sizeof(resp));
}

static int handle_get_device_count(int fd)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    GpuGetDeviceCountResponse resp = { .count = count };
    return send_response(fd, GPU_CMD_GET_DEVICE_COUNT, (int32_t)err,
                         &resp, sizeof(resp));
}

static int handle_get_device_properties(int fd, const void *payload, uint32_t len)
{
    GpuGetDevicePropertiesRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_GET_DEVICE_PROPERTIES, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    struct cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, req.device);

    GpuDeviceProperties resp;
    memset(&resp, 0, sizeof(resp));

    if (err == cudaSuccess) {
        strncpy(resp.name, props.name, sizeof(resp.name) - 1);
        resp.total_global_mem                = props.totalGlobalMem;
        resp.shared_mem_per_block            = props.sharedMemPerBlock;
        resp.regs_per_block                  = props.regsPerBlock;
        resp.warp_size                       = props.warpSize;
        resp.max_threads_per_block           = props.maxThreadsPerBlock;
        resp.max_threads_dim[0]              = props.maxThreadsDim[0];
        resp.max_threads_dim[1]              = props.maxThreadsDim[1];
        resp.max_threads_dim[2]              = props.maxThreadsDim[2];
        resp.max_grid_size[0]                = props.maxGridSize[0];
        resp.max_grid_size[1]                = props.maxGridSize[1];
        resp.max_grid_size[2]                = props.maxGridSize[2];
        resp.clock_rate                      = props.clockRate;
        resp.multi_processor_count           = props.multiProcessorCount;
        resp.major                           = props.major;
        resp.minor                           = props.minor;
        resp.max_threads_per_multiprocessor  = props.maxThreadsPerMultiprocessor;
        resp.memory_clock_rate               = props.memoryClockRate;
        resp.memory_bus_width                = props.memoryBusWidth;
        resp.l2_cache_size                   = props.l2CacheSize;
    }

    return send_response(fd, GPU_CMD_GET_DEVICE_PROPERTIES, (int32_t)err,
                         &resp, sizeof(resp));
}

static int handle_set_device(int fd, const void *payload, uint32_t len)
{
    GpuSetDeviceRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_SET_DEVICE, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    cudaError_t err = cudaSetDevice(req.device);
    return send_response(fd, GPU_CMD_SET_DEVICE, (int32_t)err, NULL, 0);
}

static int handle_malloc(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    GpuMallocRequest req;
    if (len < sizeof(req)) {
        return send_response(ctx->fd, GPU_CMD_MALLOC, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    /* Enforce memory quota */
    if (ctx->memory_quota > 0 &&
        ctx->memory_allocated + req.size > ctx->memory_quota) {
        LOG_ERR("CID %u: cudaMalloc(%lu) DENIED — would exceed quota "
                "(%lu + %lu > %lu)",
                ctx->peer_cid, (unsigned long)req.size,
                (unsigned long)ctx->memory_allocated,
                (unsigned long)req.size,
                (unsigned long)ctx->memory_quota);
        /* cudaErrorMemoryAllocation = 2 */
        return send_response(ctx->fd, GPU_CMD_MALLOC, 2, NULL, 0);
    }

    void *devptr = NULL;
    cudaError_t err = cudaMalloc(&devptr, (size_t)req.size);

    LOG_DBG("cudaMalloc(%lu) -> %p (err=%d)", (unsigned long)req.size, devptr, err);

    if (err == cudaSuccess) {
        /* Track allocation */
        ctx->memory_allocated += req.size;
        ctx->total_alloc_bytes += req.size;
        if (ctx->memory_allocated > ctx->peak_memory)
            ctx->peak_memory = ctx->memory_allocated;

        /* Record in alloc table for cleanup */
        for (int i = 0; i < MAX_ALLOCS; i++) {
            if (!ctx->allocs[i].in_use) {
                ctx->allocs[i].device_ptr = (uint64_t)(uintptr_t)devptr;
                ctx->allocs[i].size = req.size;
                ctx->allocs[i].in_use = 1;
                break;
            }
        }
    }

    GpuMallocResponse resp = {
        .device_ptr = (uint64_t)(uintptr_t)devptr,
    };
    return send_response(ctx->fd, GPU_CMD_MALLOC, (int32_t)err,
                         &resp, sizeof(resp));
}

static int handle_free(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    GpuFreeRequest req;
    if (len < sizeof(req)) {
        return send_response(ctx->fd, GPU_CMD_FREE, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    void *devptr = (void *)(uintptr_t)req.device_ptr;
    cudaError_t err = cudaFree(devptr);

    LOG_DBG("cudaFree(%p) -> err=%d", devptr, err);

    if (err == cudaSuccess) {
        /* Release from alloc tracking */
        for (int i = 0; i < MAX_ALLOCS; i++) {
            if (ctx->allocs[i].in_use &&
                ctx->allocs[i].device_ptr == req.device_ptr) {
                if (ctx->memory_allocated >= ctx->allocs[i].size)
                    ctx->memory_allocated -= ctx->allocs[i].size;
                else
                    ctx->memory_allocated = 0;
                ctx->allocs[i].in_use = 0;
                break;
            }
        }
    }

    return send_response(ctx->fd, GPU_CMD_FREE, (int32_t)err, NULL, 0);
}

static int handle_memcpy(int fd, const void *payload, uint32_t payload_len)
{
    if (payload_len < sizeof(GpuMemcpyRequest)) {
        return send_response(fd, GPU_CMD_MEMCPY, -1, NULL, 0);
    }

    GpuMemcpyRequest req;
    memcpy(&req, payload, sizeof(req));

    const uint8_t *extra_data = (const uint8_t *)payload + sizeof(req);
    uint32_t extra_len = payload_len - sizeof(req);

    cudaError_t err;

    switch (req.kind) {
    case GPU_MEMCPY_HOST_TO_DEVICE: {
        if (extra_len < req.count) {
            return send_response(fd, GPU_CMD_MEMCPY, -1, NULL, 0);
        }
        void *dst = (void *)(uintptr_t)req.dst;
        err = cudaMemcpy(dst, extra_data, (size_t)req.count,
                         cudaMemcpyHostToDevice);
        LOG_DBG("cudaMemcpy H2D %lu bytes -> %p (err=%d)",
                (unsigned long)req.count, dst, err);
        return send_response(fd, GPU_CMD_MEMCPY, (int32_t)err, NULL, 0);
    }

    case GPU_MEMCPY_DEVICE_TO_HOST: {
        void *src = (void *)(uintptr_t)req.src;
        void *buf = malloc((size_t)req.count);
        if (!buf) {
            return send_response(fd, GPU_CMD_MEMCPY, -1, NULL, 0);
        }
        err = cudaMemcpy(buf, src, (size_t)req.count,
                         cudaMemcpyDeviceToHost);
        LOG_DBG("cudaMemcpy D2H %lu bytes from %p (err=%d)",
                (unsigned long)req.count, src, err);

        int rc;
        if (err == cudaSuccess) {
            rc = send_response(fd, GPU_CMD_MEMCPY, 0,
                               buf, (uint32_t)req.count);
        } else {
            rc = send_response(fd, GPU_CMD_MEMCPY, (int32_t)err, NULL, 0);
        }
        free(buf);
        return rc;
    }

    case GPU_MEMCPY_DEVICE_TO_DEVICE: {
        void *dst = (void *)(uintptr_t)req.dst;
        void *src = (void *)(uintptr_t)req.src;
        err = cudaMemcpy(dst, src, (size_t)req.count,
                         cudaMemcpyDeviceToDevice);
        return send_response(fd, GPU_CMD_MEMCPY, (int32_t)err, NULL, 0);
    }

    default:
        return send_response(fd, GPU_CMD_MEMCPY, -1, NULL, 0);
    }
}

static int handle_memset(int fd, const void *payload, uint32_t len)
{
    GpuMemsetRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_MEMSET, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    void *devptr = (void *)(uintptr_t)req.device_ptr;
    cudaError_t err = cudaMemset(devptr, req.value, (size_t)req.count);
    return send_response(fd, GPU_CMD_MEMSET, (int32_t)err, NULL, 0);
}

static int handle_device_synchronize(int fd)
{
    cudaError_t err = cudaDeviceSynchronize();
    return send_response(fd, GPU_CMD_DEVICE_SYNCHRONIZE, (int32_t)err, NULL, 0);
}

/* ================================================================
 * Command handlers — module / function registration
 * ================================================================ */

static int handle_register_module(ConnectionCtx *ctx, const void *payload, uint32_t payload_len)
{
    if (payload_len < sizeof(GpuRegisterModuleRequest)) {
        return send_response(ctx->fd, GPU_CMD_REGISTER_MODULE, -1, NULL, 0);
    }

    GpuRegisterModuleRequest req;
    memcpy(&req, payload, sizeof(req));

    const void *fatbin_data = (const uint8_t *)payload + sizeof(req);
    uint32_t fatbin_len = payload_len - sizeof(req);

    if (fatbin_len < req.fatbin_size) {
        LOG_ERR("CID %u: register_module: fatbin truncated (%u < %lu)",
                ctx->peer_cid, fatbin_len, (unsigned long)req.fatbin_size);
        return send_response(ctx->fd, GPU_CMD_REGISTER_MODULE, -1, NULL, 0);
    }

    int slot = alloc_module_slot(ctx);
    if (slot < 0) {
        LOG_ERR("CID %u: register_module: no free module slots", ctx->peer_cid);
        return send_response(ctx->fd, GPU_CMD_REGISTER_MODULE, -1, NULL, 0);
    }

    /* Ensure driver API context is initialized for this connection */
    if (!ctx->cu_ctx) {
        CUdevice dev;
        CUresult cr = cuDeviceGet(&dev, 0);
        if (cr != CUDA_SUCCESS) {
            LOG_ERR("CID %u: cuDeviceGet failed (%d)", ctx->peer_cid, cr);
            return send_response(ctx->fd, GPU_CMD_REGISTER_MODULE, (int32_t)cr, NULL, 0);
        }
        cr = cuCtxCreate(&ctx->cu_ctx, 0, dev);
        if (cr != CUDA_SUCCESS) {
            LOG_ERR("CID %u: cuCtxCreate failed (%d)", ctx->peer_cid, cr);
            return send_response(ctx->fd, GPU_CMD_REGISTER_MODULE, (int32_t)cr, NULL, 0);
        }
    } else {
        cuCtxSetCurrent(ctx->cu_ctx);
    }

    CUmodule mod;
    CUresult cr = cuModuleLoadData(&mod, fatbin_data);
    if (cr != CUDA_SUCCESS) {
        LOG_ERR("CID %u: cuModuleLoadData failed (%d)", ctx->peer_cid, cr);
        return send_response(ctx->fd, GPU_CMD_REGISTER_MODULE, (int32_t)cr, NULL, 0);
    }

    ctx->modules[slot].module = mod;
    ctx->modules[slot].in_use = 1;

    LOG_INFO("CID %u: registered module in slot %d (%u bytes fatbin)",
             ctx->peer_cid, slot, (unsigned)req.fatbin_size);

    GpuRegisterModuleResponse resp = {
        .module_handle = (uint64_t)slot,
    };
    return send_response(ctx->fd, GPU_CMD_REGISTER_MODULE, 0,
                         &resp, sizeof(resp));
}

static int handle_unregister_module(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    if (len < sizeof(GpuUnregisterModuleRequest)) {
        return send_response(ctx->fd, GPU_CMD_UNREGISTER_MODULE, -1, NULL, 0);
    }

    GpuUnregisterModuleRequest req;
    memcpy(&req, payload, sizeof(req));

    int slot = (int)req.module_handle;
    if (slot < 0 || slot >= MAX_MODULES || !ctx->modules[slot].in_use) {
        return send_response(ctx->fd, GPU_CMD_UNREGISTER_MODULE, -1, NULL, 0);
    }

    /* Invalidate all functions from this module */
    for (int i = 0; i < MAX_FUNCTIONS; i++) {
        ctx->functions[i].in_use = 0; /* conservative: clear all */
    }

    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);
    cuModuleUnload(ctx->modules[slot].module);
    ctx->modules[slot].in_use = 0;

    LOG_DBG("CID %u: unregistered module slot %d", ctx->peer_cid, slot);

    return send_response(ctx->fd, GPU_CMD_UNREGISTER_MODULE, 0, NULL, 0);
}

static int handle_register_function(ConnectionCtx *ctx, const void *payload, uint32_t payload_len)
{
    if (payload_len < sizeof(GpuRegisterFunctionRequest)) {
        return send_response(ctx->fd, GPU_CMD_REGISTER_FUNCTION, -1, NULL, 0);
    }

    GpuRegisterFunctionRequest req;
    memcpy(&req, payload, sizeof(req));

    const char *device_name = (const char *)payload + sizeof(req);
    uint32_t name_space = payload_len - sizeof(req);

    if (req.device_name_len == 0 || req.device_name_len > name_space) {
        LOG_ERR("CID %u: register_function: bad device_name_len %u",
                ctx->peer_cid, req.device_name_len);
        return send_response(ctx->fd, GPU_CMD_REGISTER_FUNCTION, -1, NULL, 0);
    }

    int mod_slot = (int)req.module_handle;
    if (mod_slot < 0 || mod_slot >= MAX_MODULES || !ctx->modules[mod_slot].in_use) {
        LOG_ERR("CID %u: register_function: invalid module handle %d",
                ctx->peer_cid, mod_slot);
        return send_response(ctx->fd, GPU_CMD_REGISTER_FUNCTION, -1, NULL, 0);
    }

    int func_slot = alloc_function_slot(ctx);
    if (func_slot < 0) {
        LOG_ERR("CID %u: register_function: no free function slots", ctx->peer_cid);
        return send_response(ctx->fd, GPU_CMD_REGISTER_FUNCTION, -1, NULL, 0);
    }

    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);

    CUfunction func;
    CUresult cr = cuModuleGetFunction(&func, ctx->modules[mod_slot].module, device_name);
    if (cr != CUDA_SUCCESS) {
        LOG_ERR("CID %u: cuModuleGetFunction('%s') failed (%d)",
                ctx->peer_cid, device_name, cr);
        return send_response(ctx->fd, GPU_CMD_REGISTER_FUNCTION, (int32_t)cr, NULL, 0);
    }

    FunctionSlot *fs = &ctx->functions[func_slot];
    fs->host_func_ptr = req.host_func_ptr;
    fs->cu_func = func;
    fs->in_use = 1;

    /*
     * Query parameter info using cuFuncGetParamInfo (CUDA 12.0+).
     * If not available (older driver), fall back to a heuristic:
     * we can't determine param sizes server-side, so we tell the shim
     * to use sizeof(void*) for each param (works for pointer and scalar args
     * up to 8 bytes, which covers most ML kernel signatures).
     */
    fs->num_params = 0;
    memset(fs->param_sizes, 0, sizeof(fs->param_sizes));
    memset(fs->param_offsets, 0, sizeof(fs->param_offsets));

#if CUDA_VERSION >= 12000
    /* cuFuncGetParamInfo available in CUDA 12.0+ Driver API */
    for (uint32_t p = 0; p < GPU_MAX_KERNEL_PARAMS; p++) {
        size_t paramOffset, paramSize;
        cr = cuFuncGetParamInfo(func, p, &paramOffset, &paramSize);
        if (cr != CUDA_SUCCESS) break; /* no more params */
        fs->param_offsets[p] = (uint32_t)paramOffset;
        fs->param_sizes[p] = (uint32_t)paramSize;
        fs->num_params = p + 1;
    }
#else
    /*
     * Fallback for CUDA < 12: The daemon cannot introspect parameter sizes.
     * The shim must send param sizes explicitly in the launch request.
     * Set num_params=0 to signal "sizes unknown, shim must provide them".
     */
    fs->num_params = 0;
#endif

    LOG_INFO("CID %u: registered function '%s' -> slot %d, %u params",
             ctx->peer_cid, device_name, func_slot, fs->num_params);

    GpuRegisterFunctionResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.num_params = fs->num_params;
    memcpy(resp.param_sizes, fs->param_sizes,
           fs->num_params * sizeof(uint32_t));

    return send_response(ctx->fd, GPU_CMD_REGISTER_FUNCTION, 0,
                         &resp, sizeof(resp));
}

/* ================================================================
 * Command handler — kernel launch
 * ================================================================ */

static int handle_launch_kernel(ConnectionCtx *ctx, const void *payload, uint32_t payload_len)
{
    if (payload_len < sizeof(GpuLaunchKernelRequest)) {
        return send_response(ctx->fd, GPU_CMD_LAUNCH_KERNEL, -1, NULL, 0);
    }

    GpuLaunchKernelRequest req;
    memcpy(&req, payload, sizeof(req));

    const uint8_t *args_data = (const uint8_t *)payload + sizeof(req);
    uint32_t args_space = payload_len - sizeof(req);

    if (req.args_total_size > args_space) {
        LOG_ERR("CID %u: launch_kernel: args truncated (%u > %u)",
                ctx->peer_cid, req.args_total_size, args_space);
        return send_response(ctx->fd, GPU_CMD_LAUNCH_KERNEL, -1, NULL, 0);
    }

    FunctionSlot *fs = find_function(ctx, req.host_func_ptr);
    if (!fs) {
        LOG_ERR("CID %u: launch_kernel: unknown function ptr 0x%lx",
                ctx->peer_cid, (unsigned long)req.host_func_ptr);
        return send_response(ctx->fd, GPU_CMD_LAUNCH_KERNEL, -1, NULL, 0);
    }

    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);

    /*
     * Build the void* args[] array for cuLaunchKernel.
     * The shim serialized each parameter value contiguously.
     * We reconstruct pointers into that buffer.
     */
    void *kernel_params[GPU_MAX_KERNEL_PARAMS];
    uint32_t offset = 0;

    for (uint32_t i = 0; i < req.num_params && i < GPU_MAX_KERNEL_PARAMS; i++) {
        kernel_params[i] = (void *)(args_data + offset);
        offset += fs->num_params > 0 ? fs->param_sizes[i] : sizeof(uint64_t);
    }

    /* Resolve stream */
    CUstream cu_stream = (CUstream)resolve_stream(ctx, req.stream_handle);

    uint64_t t0 = now_us();

    CUresult cr = cuLaunchKernel(
        fs->cu_func,
        req.grid_dim_x, req.grid_dim_y, req.grid_dim_z,
        req.block_dim_x, req.block_dim_y, req.block_dim_z,
        (unsigned int)req.shared_mem_bytes,
        cu_stream,
        kernel_params,
        NULL  /* extra — unused when kernel_params is provided */
    );

    ctx->kernel_launches++;

    /*
     * Kernel timeout enforcement:
     * After launching, synchronize with a timeout. If the kernel exceeds the
     * configured limit, we reset the device to abort the runaway kernel.
     * This prevents a single VM from monopolizing the GPU indefinitely.
     */
    if (cr == CUDA_SUCCESS && g_kernel_timeout_us > 0) {
        /* Poll-based sync with timeout */
        CUresult sync_cr;
        uint64_t deadline = t0 + g_kernel_timeout_us;
        int timed_out = 0;

        while (1) {
            sync_cr = cuStreamQuery(cu_stream ? cu_stream : 0);
            if (sync_cr == CUDA_SUCCESS) break;
            if (sync_cr != CUDA_ERROR_NOT_READY) {
                cr = sync_cr;
                break;
            }
            if (now_us() >= deadline) {
                timed_out = 1;
                break;
            }
            usleep(1000); /* 1ms polling interval */
        }

        if (timed_out) {
            LOG_ERR("CID %u: kernel TIMEOUT after %lu µs (limit=%lu µs) — "
                    "resetting CUDA context",
                    ctx->peer_cid,
                    (unsigned long)(now_us() - t0),
                    (unsigned long)g_kernel_timeout_us);
            ctx->kernel_timeouts++;
            /* cuCtxResetPersistingL2Cache is lightweight; for a hard stop
             * we destroy and recreate the context on the next call. */
            if (ctx->cu_ctx) {
                cuCtxSetCurrent(ctx->cu_ctx);
                cudaDeviceReset();
                /* Invalidate the context so it's recreated on next module load */
                cuCtxDestroy(ctx->cu_ctx);
                ctx->cu_ctx = NULL;
            }
            cr = CUDA_ERROR_LAUNCH_TIMEOUT; /* 702 */
        }
    } else if (cr == CUDA_SUCCESS && g_kernel_timeout_us == 0) {
        /* No timeout — just sync for metering */
        cuStreamSynchronize(cu_stream ? cu_stream : 0);
    }

    uint64_t elapsed = now_us() - t0;
    ctx->kernel_time_us += elapsed;

    LOG_DBG("CID %u: cuLaunchKernel(func=0x%lx, grid=[%u,%u,%u], block=[%u,%u,%u]) "
            "-> %d (%lu µs)",
            ctx->peer_cid, (unsigned long)req.host_func_ptr,
            req.grid_dim_x, req.grid_dim_y, req.grid_dim_z,
            req.block_dim_x, req.block_dim_y, req.block_dim_z,
            cr, (unsigned long)elapsed);

    return send_response(ctx->fd, GPU_CMD_LAUNCH_KERNEL, (int32_t)cr, NULL, 0);
}

/* ================================================================
 * Command handlers — stream operations
 * ================================================================ */

static int handle_stream_create(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    GpuStreamCreateRequest req = { .flags = 0 };
    if (len >= sizeof(req))
        memcpy(&req, payload, sizeof(req));

    int slot = alloc_stream_slot(ctx);
    if (slot < 0) {
        LOG_ERR("CID %u: stream_create: no free stream slots", ctx->peer_cid);
        return send_response(ctx->fd, GPU_CMD_STREAM_CREATE, -1, NULL, 0);
    }

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, req.flags);

    if (err == cudaSuccess) {
        ctx->streams[slot].stream = stream;
        ctx->streams[slot].in_use = 1;

        LOG_DBG("CID %u: cudaStreamCreate -> slot %d, handle %p",
                ctx->peer_cid, slot, (void *)stream);

        GpuStreamCreateResponse resp = {
            .stream_handle = (uint64_t)(uintptr_t)stream,
        };
        return send_response(ctx->fd, GPU_CMD_STREAM_CREATE, 0,
                             &resp, sizeof(resp));
    }

    return send_response(ctx->fd, GPU_CMD_STREAM_CREATE, (int32_t)err, NULL, 0);
}

static int handle_stream_destroy(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    if (len < sizeof(GpuStreamDestroyRequest)) {
        return send_response(ctx->fd, GPU_CMD_STREAM_DESTROY, -1, NULL, 0);
    }

    GpuStreamDestroyRequest req;
    memcpy(&req, payload, sizeof(req));

    StreamSlot *s = find_stream(ctx, req.stream_handle);
    if (!s) {
        return send_response(ctx->fd, GPU_CMD_STREAM_DESTROY, -1, NULL, 0);
    }

    cudaError_t err = cudaStreamDestroy(s->stream);
    s->in_use = 0;

    LOG_DBG("CID %u: cudaStreamDestroy(%p) -> %d",
            ctx->peer_cid, (void *)s->stream, err);

    return send_response(ctx->fd, GPU_CMD_STREAM_DESTROY, (int32_t)err, NULL, 0);
}

static int handle_stream_synchronize(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    if (len < sizeof(GpuStreamSynchronizeRequest)) {
        return send_response(ctx->fd, GPU_CMD_STREAM_SYNCHRONIZE, -1, NULL, 0);
    }

    GpuStreamSynchronizeRequest req;
    memcpy(&req, payload, sizeof(req));

    cudaStream_t stream = resolve_stream(ctx, req.stream_handle);
    cudaError_t err = cudaStreamSynchronize(stream);

    return send_response(ctx->fd, GPU_CMD_STREAM_SYNCHRONIZE, (int32_t)err, NULL, 0);
}

/* ================================================================
 * Command handlers — event operations
 * ================================================================ */

static int handle_event_create(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    GpuEventCreateRequest req = { .flags = 0 };
    if (len >= sizeof(req))
        memcpy(&req, payload, sizeof(req));

    int slot = alloc_event_slot(ctx);
    if (slot < 0) {
        LOG_ERR("CID %u: event_create: no free event slots", ctx->peer_cid);
        return send_response(ctx->fd, GPU_CMD_EVENT_CREATE, -1, NULL, 0);
    }

    cudaEvent_t event;
    cudaError_t err = cudaEventCreateWithFlags(&event, req.flags);

    if (err == cudaSuccess) {
        ctx->events[slot].event = event;
        ctx->events[slot].in_use = 1;

        LOG_DBG("CID %u: cudaEventCreate -> slot %d, handle %p",
                ctx->peer_cid, slot, (void *)event);

        GpuEventCreateResponse resp = {
            .event_handle = (uint64_t)(uintptr_t)event,
        };
        return send_response(ctx->fd, GPU_CMD_EVENT_CREATE, 0,
                             &resp, sizeof(resp));
    }

    return send_response(ctx->fd, GPU_CMD_EVENT_CREATE, (int32_t)err, NULL, 0);
}

static int handle_event_destroy(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    if (len < sizeof(GpuEventDestroyRequest)) {
        return send_response(ctx->fd, GPU_CMD_EVENT_DESTROY, -1, NULL, 0);
    }

    GpuEventDestroyRequest req;
    memcpy(&req, payload, sizeof(req));

    EventSlot *e = find_event(ctx, req.event_handle);
    if (!e) {
        return send_response(ctx->fd, GPU_CMD_EVENT_DESTROY, -1, NULL, 0);
    }

    cudaError_t err = cudaEventDestroy(e->event);
    e->in_use = 0;

    return send_response(ctx->fd, GPU_CMD_EVENT_DESTROY, (int32_t)err, NULL, 0);
}

static int handle_event_record(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    if (len < sizeof(GpuEventRecordRequest)) {
        return send_response(ctx->fd, GPU_CMD_EVENT_RECORD, -1, NULL, 0);
    }

    GpuEventRecordRequest req;
    memcpy(&req, payload, sizeof(req));

    cudaEvent_t event = resolve_event(ctx, req.event_handle);
    if (!event) {
        return send_response(ctx->fd, GPU_CMD_EVENT_RECORD, -1, NULL, 0);
    }

    cudaStream_t stream = resolve_stream(ctx, req.stream_handle);
    cudaError_t err = cudaEventRecord(event, stream);

    return send_response(ctx->fd, GPU_CMD_EVENT_RECORD, (int32_t)err, NULL, 0);
}

static int handle_event_synchronize(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    if (len < sizeof(GpuEventSynchronizeRequest)) {
        return send_response(ctx->fd, GPU_CMD_EVENT_SYNCHRONIZE, -1, NULL, 0);
    }

    GpuEventSynchronizeRequest req;
    memcpy(&req, payload, sizeof(req));

    cudaEvent_t event = resolve_event(ctx, req.event_handle);
    if (!event) {
        return send_response(ctx->fd, GPU_CMD_EVENT_SYNCHRONIZE, -1, NULL, 0);
    }

    cudaError_t err = cudaEventSynchronize(event);
    return send_response(ctx->fd, GPU_CMD_EVENT_SYNCHRONIZE, (int32_t)err, NULL, 0);
}

static int handle_event_elapsed_time(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    if (len < sizeof(GpuEventElapsedTimeRequest)) {
        return send_response(ctx->fd, GPU_CMD_EVENT_ELAPSED_TIME, -1, NULL, 0);
    }

    GpuEventElapsedTimeRequest req;
    memcpy(&req, payload, sizeof(req));

    cudaEvent_t start = resolve_event(ctx, req.start_event);
    cudaEvent_t end   = resolve_event(ctx, req.end_event);
    if (!start || !end) {
        return send_response(ctx->fd, GPU_CMD_EVENT_ELAPSED_TIME, -1, NULL, 0);
    }

    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&ms, start, end);

    GpuEventElapsedTimeResponse resp = { .elapsed_ms = ms };
    return send_response(ctx->fd, GPU_CMD_EVENT_ELAPSED_TIME, (int32_t)err,
                         &resp, sizeof(resp));
}

/* ================================================================
 * Command handlers — resource management (quotas, metering)
 * ================================================================ */

static int handle_set_memory_quota(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    if (len < sizeof(GpuSetMemoryQuotaRequest)) {
        return send_response(ctx->fd, GPU_CMD_SET_MEMORY_QUOTA, -1, NULL, 0);
    }

    GpuSetMemoryQuotaRequest req;
    memcpy(&req, payload, sizeof(req));

    ctx->memory_quota = req.quota_bytes;

    LOG_INFO("CID %u: memory quota set to %lu bytes (%lu MB)%s",
             ctx->peer_cid,
             (unsigned long)req.quota_bytes,
             (unsigned long)(req.quota_bytes / (1024 * 1024)),
             req.quota_bytes == 0 ? " (unlimited)" : "");

    return send_response(ctx->fd, GPU_CMD_SET_MEMORY_QUOTA, 0, NULL, 0);
}

static int handle_get_usage_stats(ConnectionCtx *ctx)
{
    GpuUsageStatsResponse resp = {
        .memory_allocated = ctx->memory_allocated,
        .memory_quota     = ctx->memory_quota,
        .peak_memory      = ctx->peak_memory,
        .total_alloc_bytes = ctx->total_alloc_bytes,
        .kernel_launches  = ctx->kernel_launches,
        .kernel_timeouts  = ctx->kernel_timeouts,
        .kernel_time_us   = ctx->kernel_time_us,
        .connect_time_us  = now_us() - ctx->connect_time_us,
    };

    LOG_DBG("CID %u: usage stats — mem=%lu/%lu peak=%lu kernels=%u "
            "timeouts=%u ktime=%lu µs uptime=%lu µs",
            ctx->peer_cid,
            (unsigned long)resp.memory_allocated,
            (unsigned long)resp.memory_quota,
            (unsigned long)resp.peak_memory,
            resp.kernel_launches,
            resp.kernel_timeouts,
            (unsigned long)resp.kernel_time_us,
            (unsigned long)resp.connect_time_us);

    return send_response(ctx->fd, GPU_CMD_GET_USAGE_STATS, 0,
                         &resp, sizeof(resp));
}

static int handle_report_usage_stats(ConnectionCtx *ctx,
                                     const void *payload, uint32_t len)
{
    if (len < sizeof(GpuReportUsageStats)) {
        LOG_ERR("CID %u: REPORT_USAGE_STATS too short (%u < %zu)",
                ctx->peer_cid, len, sizeof(GpuReportUsageStats));
        return send_response(ctx->fd, GPU_CMD_REPORT_USAGE_STATS, -1, NULL, 0);
    }

    GpuReportUsageStats stats;
    memcpy(&stats, payload, sizeof(stats));

    /* Store latest agent stats */
    ctx->last_agent_stats = stats;

    /* Mirror key fields into the connection-level metering so
     * the daemon's final-stats log and GET_USAGE_STATS work
     * uniformly for both proxy and metering connections. */
    ctx->memory_allocated = stats.memory_allocated;
    if (stats.peak_memory > ctx->peak_memory)
        ctx->peak_memory = stats.peak_memory;

    LOG_DBG("CID %u: agent report — mem=%lu/%lu MiB, gpu=%u%%, "
            "temp=%d°C, power=%u/%u mW, uptime=%lu µs",
            ctx->peer_cid,
            (unsigned long)(stats.memory_allocated / (1024 * 1024)),
            (unsigned long)(stats.memory_total / (1024 * 1024)),
            stats.gpu_utilization,
            stats.temperature_c,
            stats.power_usage_mw,
            stats.power_limit_mw,
            (unsigned long)stats.uptime_us);

    return send_response(ctx->fd, GPU_CMD_REPORT_USAGE_STATS, 0, NULL, 0);
}

/* ================================================================
 * Per-connection handler (one thread per VM)
 * ================================================================ */

static void *connection_handler(void *arg)
{
    ConnectionCtx *ctx = (ConnectionCtx *)arg;
    int fd = ctx->fd;
    unsigned int cid = ctx->peer_cid;

    LOG_INFO("VM CID %u connected", cid);

    ctx->connect_time_us = now_us();

    /* Allocate a reusable payload buffer */
    size_t buf_cap = 4096;
    void *buf = malloc(buf_cap);
    if (!buf) {
        LOG_ERR("CID %u: malloc failed", cid);
        goto done;
    }

    while (g_running) {
        /* Read request header */
        GpuProxyHeader hdr;
        if (read_exact(fd, &hdr, sizeof(hdr)) < 0) {
            LOG_DBG("CID %u: connection closed", cid);
            break;
        }

        if (hdr.magic != GPU_PROXY_MAGIC) {
            LOG_ERR("CID %u: bad magic 0x%08x", cid, hdr.magic);
            break;
        }

        if (hdr.payload_len > GPU_PROXY_MAX_PAYLOAD) {
            LOG_ERR("CID %u: payload too large (%u bytes)", cid, hdr.payload_len);
            break;
        }

        /* Read payload */
        if (hdr.payload_len > 0) {
            if (hdr.payload_len > buf_cap) {
                buf_cap = hdr.payload_len;
                void *newbuf = realloc(buf, buf_cap);
                if (!newbuf) {
                    LOG_ERR("CID %u: realloc failed for %u bytes", cid, hdr.payload_len);
                    break;
                }
                buf = newbuf;
            }
            if (read_exact(fd, buf, hdr.payload_len) < 0) {
                LOG_ERR("CID %u: failed to read payload", cid);
                break;
            }
        }

        /* Dispatch */
        int rc = 0;
        switch (hdr.cmd) {
        case GPU_CMD_HELLO:
            rc = handle_hello(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_GET_DEVICE_COUNT:
            rc = handle_get_device_count(fd);
            break;
        case GPU_CMD_GET_DEVICE_PROPERTIES:
            rc = handle_get_device_properties(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_SET_DEVICE:
            rc = handle_set_device(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_MALLOC:
            rc = handle_malloc(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_FREE:
            rc = handle_free(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_MEMCPY:
            rc = handle_memcpy(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_MEMSET:
            rc = handle_memset(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_DEVICE_SYNCHRONIZE:
            rc = handle_device_synchronize(fd);
            break;

        /* Module / function registration */
        case GPU_CMD_REGISTER_MODULE:
            rc = handle_register_module(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_UNREGISTER_MODULE:
            rc = handle_unregister_module(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_REGISTER_FUNCTION:
            rc = handle_register_function(ctx, buf, hdr.payload_len);
            break;

        /* Kernel launch */
        case GPU_CMD_LAUNCH_KERNEL:
            rc = handle_launch_kernel(ctx, buf, hdr.payload_len);
            break;

        /* Stream operations */
        case GPU_CMD_STREAM_CREATE:
            rc = handle_stream_create(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_STREAM_DESTROY:
            rc = handle_stream_destroy(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_STREAM_SYNCHRONIZE:
            rc = handle_stream_synchronize(ctx, buf, hdr.payload_len);
            break;

        /* Event operations */
        case GPU_CMD_EVENT_CREATE:
            rc = handle_event_create(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_EVENT_DESTROY:
            rc = handle_event_destroy(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_EVENT_RECORD:
            rc = handle_event_record(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_EVENT_SYNCHRONIZE:
            rc = handle_event_synchronize(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_EVENT_ELAPSED_TIME:
            rc = handle_event_elapsed_time(ctx, buf, hdr.payload_len);
            break;

        /* Resource management */
        case GPU_CMD_SET_MEMORY_QUOTA:
            rc = handle_set_memory_quota(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_GET_USAGE_STATS:
            rc = handle_get_usage_stats(ctx);
            break;
        case GPU_CMD_REPORT_USAGE_STATS:
            rc = handle_report_usage_stats(ctx, buf, hdr.payload_len);
            break;

        case GPU_CMD_GOODBYE:
            LOG_INFO("CID %u: graceful disconnect (mode=%s mem=%lu kernels=%u ktime=%lu µs)",
                     cid,
                     ctx->conn_mode == GPU_CONN_METER ? "METER" : "PROXY",
                     (unsigned long)ctx->memory_allocated,
                     ctx->kernel_launches,
                     (unsigned long)ctx->kernel_time_us);
            send_response(fd, GPU_CMD_GOODBYE, 0, NULL, 0);
            goto done;
        default:
            LOG_ERR("CID %u: unknown command 0x%02x", cid, hdr.cmd);
            send_response(fd, hdr.cmd, -1, NULL, 0);
            break;
        }

        if (rc < 0) {
            LOG_ERR("CID %u: write error, disconnecting", cid);
            break;
        }
    }

done:
    free(buf);
    cleanup_connection(ctx);
    close(fd);
    LOG_INFO("VM CID %u disconnected (resources cleaned up)", cid);
    free(ctx);
    return NULL;
}

/* ================================================================
 * Main: vsock listener
 * ================================================================ */

static void sig_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [-p port] [-t timeout_sec] [-v]\n", prog);
    fprintf(stderr, "  -p port          vsock port to listen on (default: %d)\n", GPU_PROXY_PORT);
    fprintf(stderr, "  -t timeout_sec   kernel execution timeout in seconds (default: %d, 0=disable)\n",
            (int)(GPU_PROXY_DEFAULT_KERNEL_TIMEOUT_US / 1000000));
    fprintf(stderr, "  -v               verbose logging\n");
    exit(1);
}

int main(int argc, char **argv)
{
    int port = GPU_PROXY_PORT;
    int opt;

    while ((opt = getopt(argc, argv, "p:t:v")) != -1) {
        switch (opt) {
        case 'p': port = atoi(optarg); break;
        case 't': g_kernel_timeout_us = (uint64_t)atoi(optarg) * 1000000ULL; break;
        case 'v': g_verbose = 1; break;
        default: usage(argv[0]);
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    /* Initialize CUDA Driver API (required for cuModuleLoadData, cuLaunchKernel) */
    CUresult cr = cuInit(0);
    if (cr != CUDA_SUCCESS) {
        LOG_ERR("cuInit failed (%d) — is the NVIDIA driver loaded?", cr);
        return 1;
    }

    /* Verify CUDA is available */
    int dev_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&dev_count);
    if (cerr != cudaSuccess || dev_count == 0) {
        LOG_ERR("No CUDA devices found (cudaGetDeviceCount=%d, err=%s)",
                dev_count, cudaGetErrorString(cerr));
        return 1;
    }
    LOG_INFO("Found %d CUDA device(s)", dev_count);

    for (int i = 0; i < dev_count; i++) {
        struct cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        LOG_INFO("  GPU %d: %s (%ldMB, compute %d.%d)",
                 i, props.name,
                 (long)(props.totalGlobalMem / (1024 * 1024)),
                 props.major, props.minor);
    }

    /* Create vsock listener */
    int listen_fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        LOG_ERR("socket(AF_VSOCK) failed: %s", strerror(errno));
        return 1;
    }

    struct sockaddr_vm addr = {
        .svm_family = AF_VSOCK,
        .svm_cid    = VMADDR_CID_ANY,  /* Accept from any VM */
        .svm_port   = (unsigned int)port,
    };

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        LOG_ERR("bind(vsock port %d) failed: %s", port, strerror(errno));
        close(listen_fd);
        return 1;
    }

    if (listen(listen_fd, 16) < 0) {
        LOG_ERR("listen() failed: %s", strerror(errno));
        close(listen_fd);
        return 1;
    }

    LOG_INFO("Listening on vsock port %d (CID=any) — kernel launch enabled, "
             "timeout=%lu s, quotas=enabled",
             port, (unsigned long)(g_kernel_timeout_us / 1000000));

    /* Accept loop */
    while (g_running) {
        struct sockaddr_vm peer;
        socklen_t peer_len = sizeof(peer);
        int client_fd = accept(listen_fd, (struct sockaddr *)&peer, &peer_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            LOG_ERR("accept() failed: %s", strerror(errno));
            break;
        }

        ConnectionCtx *ctx = calloc(1, sizeof(ConnectionCtx));
        if (!ctx) {
            LOG_ERR("calloc failed for connection context");
            close(client_fd);
            continue;
        }
        ctx->fd = client_fd;
        ctx->peer_cid = peer.svm_cid;

        pthread_t tid;
        if (pthread_create(&tid, NULL, connection_handler, ctx) != 0) {
            LOG_ERR("pthread_create failed: %s", strerror(errno));
            close(client_fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }

    LOG_INFO("Shutting down");
    close(listen_fd);
    return 0;
}
