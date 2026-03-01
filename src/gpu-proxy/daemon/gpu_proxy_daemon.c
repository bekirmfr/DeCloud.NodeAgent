/*
 * DeCloud GPU Proxy Daemon
 *
 * Host-side service that proxies CUDA Runtime API calls from guest VMs
 * to the real GPU. Supports two transports:
 *   - virtio-vsock (bare metal, CID-authenticated)
 *   - TCP over virbr0 (WSL2 fallback, token-authenticated)
 *
 * Each VM connects with its unique CID (vsock) or auth token (TCP).
 * The daemon maintains per-connection state (loaded modules, registered
 * functions, streams, events) for isolation.
 *
 * Kernel launch uses the CUDA Driver API (cuModuleLoadData, cuModuleGetFunction,
 * cuLaunchKernel) because the Runtime API's cudaLaunchKernel requires the
 * actual host-side function pointer from the compiled fat binary, which we
 * don't have — the guest sends us raw PTX/cubin instead.
 *
 * Build: gcc -o gpu-proxy-daemon gpu_proxy_daemon.c -lcuda -lcudart -lpthread
 * Run:   gpu-proxy-daemon [-p port] [-T 192.168.122.1] [-v]
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
#include <netinet/in.h>
#include <arpa/inet.h>

#include <dlfcn.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../proto/gpu_proxy_proto.h"

/*
 * cuFuncGetParamInfo (CUDA 12.0+ Driver API) may be missing from
 * the linked libcuda.so (e.g. WSL2 stubs). We look it up via dlsym
 * at runtime so the daemon still compiles/links everywhere, and
 * falls back to "sizes unknown" when the symbol isn't available.
 */
typedef CUresult (*pfn_cuFuncGetParamInfo)(CUfunction, size_t, size_t *, size_t *);
static pfn_cuFuncGetParamInfo g_cuFuncGetParamInfo = NULL;
static int g_cuFuncGetParamInfo_resolved = 0;

static pfn_cuFuncGetParamInfo resolve_cuFuncGetParamInfo(void)
{
    if (!g_cuFuncGetParamInfo_resolved) {
        g_cuFuncGetParamInfo = (pfn_cuFuncGetParamInfo)dlsym(RTLD_DEFAULT, "cuFuncGetParamInfo");
        g_cuFuncGetParamInfo_resolved = 1;
    }
    return g_cuFuncGetParamInfo;
}

/* ================================================================
 * Globals
 * ================================================================ */

static int g_verbose = 0;
static volatile sig_atomic_t g_running = 1;
static volatile sig_atomic_t g_reload_tokens = 0;
static uint64_t g_kernel_timeout_us = GPU_DEFAULT_KERNEL_TIMEOUT_US;

/* TCP fallback transport */
static int g_tcp_enabled = 0;
static const char *g_tcp_bind = GPU_PROXY_TCP_BIND;

/* Monotonic clock helper — returns microseconds since an arbitrary epoch */
static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ================================================================
 * Logging (defined early — used by token registry and all handlers)
 * ================================================================ */

#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[gpu-proxy] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) \
    fprintf(stderr, "[gpu-proxy] ERROR: " fmt "\n", ##__VA_ARGS__)
#define LOG_DBG(fmt, ...) \
    do { if (g_verbose) fprintf(stdout, "[gpu-proxy] DBG: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ================================================================
 * Auth token registry — maps tokens to VM identifiers.
 * Tokens are loaded from a file written by NodeAgent.
 * File format: one line per VM: <hex_token> <vm_id>
 * ================================================================ */
#define MAX_TOKENS 256

typedef struct {
    uint8_t  token[GPU_PROXY_TOKEN_LEN];
    char     vm_id[64];
    int      active;
} TokenEntry;

static TokenEntry g_tokens[MAX_TOKENS];
static pthread_mutex_t g_token_lock = PTHREAD_MUTEX_INITIALIZER;

/* Path to the token registry file managed by NodeAgent */
static const char *g_token_file = "/var/lib/decloud/gpu-proxy-tokens";

static void load_tokens(void)
{
    pthread_mutex_lock(&g_token_lock);
    memset(g_tokens, 0, sizeof(g_tokens));

    FILE *f = fopen(g_token_file, "r");
    if (!f) {
        LOG_DBG("No token file at %s (vsock-only mode)", g_token_file);
        pthread_mutex_unlock(&g_token_lock);
        return;
    }

    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), f) && count < MAX_TOKENS) {
        char hex_token[65], vm_id[64];
        if (sscanf(line, "%64s %63s", hex_token, vm_id) != 2) continue;

        /* Parse hex string → bytes */
        size_t tlen = strlen(hex_token);
        if (tlen != GPU_PROXY_TOKEN_LEN * 2) continue;

        TokenEntry *e = &g_tokens[count];
        for (int i = 0; i < GPU_PROXY_TOKEN_LEN; i++) {
            unsigned int byte;
            sscanf(&hex_token[i * 2], "%02x", &byte);
            e->token[i] = (uint8_t)byte;
        }
        strncpy(e->vm_id, vm_id, sizeof(e->vm_id) - 1);
        e->active = 1;
        count++;
    }

    fclose(f);
    LOG_INFO("Loaded %d auth token(s) from %s", count, g_token_file);
    pthread_mutex_unlock(&g_token_lock);
}

static int validate_token(const uint8_t *token, char *vm_id_out, size_t vm_id_size)
{
    /* All-zero token = vsock (no auth needed, CID is authoritative) */
    int all_zero = 1;
    for (int i = 0; i < GPU_PROXY_TOKEN_LEN; i++) {
        if (token[i] != 0) { all_zero = 0; break; }
    }
    if (all_zero) return 0; /* Not a TCP auth attempt */

    pthread_mutex_lock(&g_token_lock);
    for (int i = 0; i < MAX_TOKENS; i++) {
        if (!g_tokens[i].active) continue;
        if (memcmp(g_tokens[i].token, token, GPU_PROXY_TOKEN_LEN) == 0) {
            if (vm_id_out) strncpy(vm_id_out, g_tokens[i].vm_id, vm_id_size - 1);
            pthread_mutex_unlock(&g_token_lock);
            return 1; /* Valid */
        }
    }
    pthread_mutex_unlock(&g_token_lock);
    return -1; /* Invalid token */
}

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
    int          is_tcp;
    char         vm_id[64];
    int          fd;
    unsigned int peer_cid;

    /* Per-connection CUDA Driver API context */
    CUcontext    cu_ctx;
    cublasHandle_t cublas_handle;   /* Lazy-init on first GEMM RPC */

    ModuleSlot   modules[MAX_MODULES];
    FunctionSlot functions[MAX_FUNCTIONS];
    StreamSlot   streams[MAX_STREAMS];
    EventSlot    events[MAX_EVENTS];

    /* Memory quota enforcement */
    uint64_t  memory_quota;       /* 0 = unlimited */
    uint64_t  memory_allocated;   /* Current total allocated bytes */
    uint64_t  peak_memory;        /* High-water mark */
    uint64_t  total_alloc_bytes;  /* Cumulative allocation total */
    AllocSlot allocs[MAX_ALLOCS]; /* Track individual allocations */

    /* Metering / billing */
    uint64_t  connect_time_us;    /* Timestamp of connection start */
    uint32_t  kernel_launches;    /* Total kernel launches */
    uint32_t  kernel_timeouts;    /* Kernels killed by timeout */
    uint64_t  kernel_time_us;     /* Cumulative kernel execution time */
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
        LOG_INFO("%s %s final stats: mem_peak=%lu alloc_total=%lu "
                 "kernels=%u timeouts=%u ktime=%lu µs uptime=%lu µs",
                 ctx->is_tcp ? "VM" : "CID",
                 ctx->is_tcp ? ctx->vm_id : "vsock",
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
    /* Destroy cuBLAS handle before CUDA context */
    if (ctx->cublas_handle) {
        cublasDestroy_v2(ctx->cublas_handle);
        ctx->cublas_handle = NULL;
    }
    if (ctx->cu_ctx) {
        cuCtxDestroy(ctx->cu_ctx);
        ctx->cu_ctx = NULL;
    }
}

/* ================================================================
 * Command handlers — HELLO (with TCP auth validation)
 * ================================================================ */

static int handle_hello(ConnectionCtx *ctx, const void *payload, uint32_t len)
{
    GpuHelloRequest req;
    if (len < sizeof(req)) {
        return send_response(ctx->fd, GPU_CMD_HELLO, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    /* Validate auth token for TCP connections BEFORE sending success */
    if (ctx->is_tcp) {
        char vm_id[64] = {0};
        int tok_result = validate_token(req.auth_token, vm_id, sizeof(vm_id));
        if (tok_result < 0) {
            LOG_ERR("TCP connection rejected: invalid auth token (PID %u)", req.pid);
            send_response(ctx->fd, GPU_CMD_HELLO, -1, NULL, 0);
            return -1; /* Triggers disconnect */
        }
        strncpy(ctx->vm_id, vm_id, sizeof(ctx->vm_id) - 1);
        LOG_INFO("TCP auth OK: VM %s (PID %u, shim v%u)",
                 vm_id, req.pid, req.shim_version);
    }

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    LOG_INFO("HELLO from %s PID %u (shim v%u), %d GPU(s) available",
             ctx->is_tcp ? ctx->vm_id : "vsock CID",
             req.pid, req.shim_version, count);

    GpuHelloResponse resp = {
        .daemon_version = GPU_PROXY_VERSION,
        .device_count   = (uint32_t)count,
    };

    return send_response(ctx->fd, GPU_CMD_HELLO, (int32_t)err,
                         &resp, sizeof(resp));
}

/* ================================================================
 * Command handlers — device management, memory, sync
 * ================================================================ */

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
        resp.max_threads_per_multiprocessor  = props.maxThreadsPerMultiProcessor;
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

    /*
     * Try to introspect kernel parameter sizes via cuFuncGetParamInfo
     * (CUDA 12.0+ Driver API, resolved at runtime via dlsym).
     * If the symbol isn't available (e.g. WSL2, older drivers), fall
     * back to num_params=0 which tells the shim to send sizes itself.
     */
    pfn_cuFuncGetParamInfo fnGetParamInfo = resolve_cuFuncGetParamInfo();
    if (fnGetParamInfo) {
        for (uint32_t p = 0; p < GPU_MAX_KERNEL_PARAMS; p++) {
            size_t paramOffset, paramSize;
            cr = fnGetParamInfo(func, p, &paramOffset, &paramSize);
            if (cr != CUDA_SUCCESS) break; /* no more params */
            fs->param_offsets[p] = (uint32_t)paramOffset;
            fs->param_sizes[p] = (uint32_t)paramSize;
            fs->num_params = p + 1;
        }
    }

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

/* ----------------------------------------------------------------
 * Queries real kernel attributes from the GPU using the CUDA Driver API.
 * Called when the shim's cudaFuncGetAttributes triggers an RPC after
 * eagerly uploading the fat binary module.
 * ---------------------------------------------------------------- */

static int handle_func_get_attributes(ConnectionCtx *ctx,
                                       const void *payload,
                                       uint32_t payload_len)
{
    if (payload_len < sizeof(GpuFuncGetAttributesRequest)) {
        return send_response(ctx->fd, GPU_CMD_FUNC_GET_ATTRIBUTES, -1, NULL, 0);
    }

    GpuFuncGetAttributesRequest req;
    memcpy(&req, payload, sizeof(req));

    /* Find the function by host_func_ptr (same key used in register + launch) */
    FunctionSlot *fs = find_function(ctx, req.host_func_ptr);
    if (!fs) {
        LOG_ERR("CID %u: func_get_attributes: unknown func_ptr 0x%lx",
                ctx->peer_cid, (unsigned long)req.host_func_ptr);
        return send_response(ctx->fd, GPU_CMD_FUNC_GET_ATTRIBUTES,
                             (int32_t)cudaErrorInvalidDeviceFunction, NULL, 0);
    }

    /* Ensure CUDA context is current for Driver API calls */
    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);

    /* Query each attribute individually using Driver API.
     * cuFuncGetAttribute returns a single int per attribute. */
    GpuFuncGetAttributesResponse resp;
    memset(&resp, 0, sizeof(resp));

    int val = 0;

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_BINARY_VERSION, fs->cu_func);
    resp.binaryVersion = val;

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

    cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_PTX_VERSION, fs->cu_func);
    resp.ptxVersion = val;

    LOG_DBG("CID %u: func_get_attributes(0x%lx): binary=%d maxThreads=%d regs=%d shared=%d",
            ctx->peer_cid, (unsigned long)req.host_func_ptr,
            resp.binaryVersion, resp.maxThreadsPerBlock,
            resp.numRegs, resp.sharedSizeBytes);

    return send_response(ctx->fd, GPU_CMD_FUNC_GET_ATTRIBUTES, 0,
                         &resp, sizeof(resp));
}

/* ================================================================
 * cuBLAS lazy initialization and GEMM RPC handlers
 *
 * GQA (Grouped Query Attention) requires cublasGemmBatchedEx for Q×K
 * matrix multiplication — this is NOT covered by GGML_CUDA_FORCE_MMQ.
 * The shim sends GEMM parameters + device pointers via RPC, and the
 * daemon executes on the real GPU using the real cuBLAS library.
 * ================================================================ */

/* Lazy-initialize cuBLAS handle for this connection.
 * Called on first GEMM RPC — not at connection time, because cuBLAS
 * needs an active CUDA context which only exists after module load. */
static int ensure_cublas(ConnectionCtx *ctx)
{
    if (ctx->cublas_handle) return 0;

    /* Ensure correct CUDA context is active on this thread */
    if (ctx->cu_ctx)
        cuCtxSetCurrent(ctx->cu_ctx);

    cublasStatus_t cs = cublasCreate_v2(&ctx->cublas_handle);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        LOG_ERR("%s %s: cublasCreate_v2 failed (%d)",
                ctx->is_tcp ? "VM" : "CID",
                ctx->is_tcp ? ctx->vm_id : "vsock",
                (int)cs);
        ctx->cublas_handle = NULL;
        return -1;
    }

    LOG_INFO("%s %s: cuBLAS handle initialized",
             ctx->is_tcp ? "VM" : "CID",
             ctx->is_tcp ? ctx->vm_id : "vsock");
    return 0;
}

/* Determine scalar size (alpha/beta) from CUDA compute type */
static int cublas_scalar_size(int computeType)
{
    /* CUBLAS_COMPUTE_16F=64, 32F=68, 32F_FAST_16F=74, 64F=70 */
    switch (computeType) {
        case 64: return 2;  /* half */
        case 70: return 8;  /* double */
        default: return 4;  /* float (most common) */
    }
}

/* ----------------------------------------------------------------
 * handle_cublas_gemm_batched (GPU_CMD_CUBLAS_GEMM_BATCHED = 0x56)
 *
 * Payload: GpuCublasGemmBatchedRequest header
 *        + 3 * batchCount * uint64_t device pointers (A[], B[], C[])
 * ---------------------------------------------------------------- */
static int handle_cublas_gemm_batched(ConnectionCtx *ctx,
                                       const void *payload, uint32_t payload_len)
{
    if (payload_len < sizeof(GpuCublasGemmBatchedRequest)) {
        LOG_ERR("CID %u: gemm_batched: payload too small (%u < %zu)",
                ctx->peer_cid, payload_len, sizeof(GpuCublasGemmBatchedRequest));
        return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_BATCHED, -1, NULL, 0);
    }

    GpuCublasGemmBatchedRequest req;
    memcpy(&req, payload, sizeof(req));

    int bc = req.batchCount;
    if (bc <= 0 || bc > 65536) {
        LOG_ERR("CID %u: gemm_batched: invalid batchCount=%d", ctx->peer_cid, bc);
        return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_BATCHED, -1, NULL, 0);
    }

    uint32_t ptrs_size = 3 * bc * (uint32_t)sizeof(uint64_t);
    uint32_t expected = (uint32_t)sizeof(GpuCublasGemmBatchedRequest) + ptrs_size;
    if (payload_len < expected) {
        LOG_ERR("CID %u: gemm_batched: payload truncated (%u < %u)",
                ctx->peer_cid, payload_len, expected);
        return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_BATCHED, -1, NULL, 0);
    }

    /* Lazy-init cuBLAS */
    if (ensure_cublas(ctx) < 0) {
        return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_BATCHED,
                             CUBLAS_STATUS_NOT_INITIALIZED, NULL, 0);
    }

    /* Extract device pointer arrays from payload */
    const uint64_t *ptrs = (const uint64_t *)((const uint8_t *)payload
                            + sizeof(GpuCublasGemmBatchedRequest));

    /* Convert uint64 device pointers to void* arrays for cuBLAS */
    const void **Aarray = (const void **)malloc(bc * sizeof(void *));
    const void **Barray = (const void **)malloc(bc * sizeof(void *));
    void **Carray = (void **)malloc(bc * sizeof(void *));
    if (!Aarray || !Barray || !Carray) {
        free((void *)Aarray); free((void *)Barray); free(Carray);
        return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_BATCHED, -1, NULL, 0);
    }

    for (int i = 0; i < bc; i++) {
        Aarray[i] = (const void *)(uintptr_t)ptrs[i];
        Barray[i] = (const void *)(uintptr_t)ptrs[bc + i];
        Carray[i] = (void *)(uintptr_t)ptrs[2 * bc + i];
    }

    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);

    cublasStatus_t cs = cublasGemmBatchedEx(
        ctx->cublas_handle,
        (cublasOperation_t)req.transa,
        (cublasOperation_t)req.transb,
        req.m, req.n, req.k,
        req.alpha,
        Aarray, (cudaDataType_t)req.Atype, req.lda,
        Barray, (cudaDataType_t)req.Btype, req.ldb,
        req.beta,
        Carray, (cudaDataType_t)req.Ctype, req.ldc,
        bc,
        (cublasComputeType_t)req.computeType,
        (cublasGemmAlgo_t)req.algo);

    /* Sync to ensure GEMM completes before we respond */
    cudaDeviceSynchronize();

    free((void *)Aarray);
    free((void *)Barray);
    free(Carray);

    if (cs != CUBLAS_STATUS_SUCCESS) {
        LOG_ERR("CID %u: cublasGemmBatchedEx failed: %d (m=%d n=%d k=%d bc=%d "
                "Atype=%d Ctype=%d compute=%d)",
                ctx->peer_cid, (int)cs, req.m, req.n, req.k, bc,
                req.Atype, req.Ctype, req.computeType);
    } else {
        LOG_DBG("CID %u: cublasGemmBatchedEx OK (m=%d n=%d k=%d bc=%d)",
                ctx->peer_cid, req.m, req.n, req.k, bc);
    }

    return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_BATCHED,
                         (int32_t)cs, NULL, 0);
}

/* ----------------------------------------------------------------
 * handle_cublas_gemm_strided (GPU_CMD_CUBLAS_GEMM_STRIDED = 0x57)
 *
 * Payload: GpuCublasGemmStridedRequest (fixed size, includes ptrs)
 * ---------------------------------------------------------------- */
static int handle_cublas_gemm_strided(ConnectionCtx *ctx,
                                       const void *payload, uint32_t payload_len)
{
    if (payload_len < sizeof(GpuCublasGemmStridedRequest)) {
        return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_STRIDED, -1, NULL, 0);
    }

    GpuCublasGemmStridedRequest req;
    memcpy(&req, payload, sizeof(req));

    if (req.batchCount <= 0 || req.batchCount > 65536) {
        return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_STRIDED, -1, NULL, 0);
    }

    /* Lazy-init cuBLAS */
    if (ensure_cublas(ctx) < 0) {
        return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_STRIDED,
                             CUBLAS_STATUS_NOT_INITIALIZED, NULL, 0);
    }

    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);

    cublasStatus_t cs = cublasGemmStridedBatchedEx(
        ctx->cublas_handle,
        (cublasOperation_t)req.transa,
        (cublasOperation_t)req.transb,
        req.m, req.n, req.k,
        req.alpha,
        (const void *)(uintptr_t)req.A_ptr, (cudaDataType_t)req.Atype,
            req.lda, req.strideA,
        (const void *)(uintptr_t)req.B_ptr, (cudaDataType_t)req.Btype,
            req.ldb, req.strideB,
        req.beta,
        (void *)(uintptr_t)req.C_ptr, (cudaDataType_t)req.Ctype,
            req.ldc, req.strideC,
        req.batchCount,
        (cublasComputeType_t)req.computeType,
        (cublasGemmAlgo_t)req.algo);

    /* Sync to ensure GEMM completes before we respond */
    cudaDeviceSynchronize();

    if (cs != CUBLAS_STATUS_SUCCESS) {
        LOG_ERR("CID %u: cublasGemmStridedBatchedEx failed: %d (m=%d n=%d k=%d bc=%d)",
                ctx->peer_cid, (int)cs, req.m, req.n, req.k, req.batchCount);
    } else {
        LOG_DBG("CID %u: cublasGemmStridedBatchedEx OK (m=%d n=%d k=%d bc=%d)",
                ctx->peer_cid, req.m, req.n, req.k, req.batchCount);
    }

    return send_response(ctx->fd, GPU_CMD_CUBLAS_GEMM_STRIDED,
                         (int32_t)cs, NULL, 0);
}

/* ----------------------------------------------------------------
 * Queries real occupancy from the GPU using the CUDA Driver API.
 * Returns the maximum number of active blocks per SM for a given
 * kernel with the specified block size and shared memory usage.
 * ---------------------------------------------------------------- */

static int handle_occupancy_max_blocks(ConnectionCtx *ctx,
                                        const void *payload,
                                        uint32_t payload_len)
{
    if (payload_len < sizeof(GpuOccupancyMaxBlocksRequest)) {
        return send_response(ctx->fd, GPU_CMD_OCCUPANCY_MAX_BLOCKS, -1, NULL, 0);
    }

    GpuOccupancyMaxBlocksRequest req;
    memcpy(&req, payload, sizeof(req));

    /* Find the function — fall back to safe value if not found */
    FunctionSlot *fs = find_function(ctx, req.host_func_ptr);
    if (!fs) {
        GpuOccupancyMaxBlocksResponse resp = { .numBlocks = 1 };
        return send_response(ctx->fd, GPU_CMD_OCCUPANCY_MAX_BLOCKS, 0,
                             &resp, sizeof(resp));
    }

    /* Ensure CUDA context is current */
    if (ctx->cu_ctx) cuCtxSetCurrent(ctx->cu_ctx);

    int numBlocks = 0;
    CUresult cr = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &numBlocks, fs->cu_func, req.blockSize,
        (size_t)req.dynamicSMemSize, req.flags);

    if (cr != CUDA_SUCCESS) {
        LOG_ERR("CID %u: occupancy query failed (cr=%d), using fallback=1",
                ctx->peer_cid, cr);
        numBlocks = 1; /* safe fallback */
    }

    LOG_DBG("CID %u: occupancy_max_blocks(0x%lx, blockSize=%d): numBlocks=%d",
            ctx->peer_cid, (unsigned long)req.host_func_ptr,
            req.blockSize, numBlocks);

    GpuOccupancyMaxBlocksResponse resp = { .numBlocks = numBlocks };
    return send_response(ctx->fd, GPU_CMD_OCCUPANCY_MAX_BLOCKS, 0,
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

    cudaEvent_t start = resolve_event(ctx, req.start_handle);
    cudaEvent_t end   = resolve_event(ctx, req.end_handle);
    if (!start || !end) {
        return send_response(ctx->fd, GPU_CMD_EVENT_ELAPSED_TIME, -1, NULL, 0);
    }

    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&ms, start, end);

    GpuEventElapsedTimeResponse resp = { .milliseconds = ms };
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

/* ================================================================
 * CUDA Driver API handlers (Phase 2 — Ollama / ML framework support)
 *
 * These handle requests from the Driver API shim (libcuda.so.1) which
 * is dlopen'd by Ollama, llama.cpp, vLLM, and other ML frameworks.
 * ================================================================ */

static int handle_get_driver_version(int fd)
{
    int version = 0;
    CUresult cr = cuDriverGetVersion(&version);
    GpuDriverVersionResponse resp = { .version = (int32_t)version };
    return send_response(fd, GPU_CMD_GET_DRIVER_VERSION,
                         (int32_t)cr, &resp, sizeof(resp));
}

static int handle_get_device_uuid(int fd, const void *payload, uint32_t len)
{
    GpuDeviceUuidRequest req;
    if (len < sizeof(req))
        return send_response(fd, GPU_CMD_GET_DEVICE_UUID, -1, NULL, 0);
    memcpy(&req, payload, sizeof(req));

    CUdevice dev;
    CUresult cr = cuDeviceGet(&dev, req.device);
    if (cr != CUDA_SUCCESS)
        return send_response(fd, GPU_CMD_GET_DEVICE_UUID, (int32_t)cr, NULL, 0);

    CUuuid uuid;
    cr = cuDeviceGetUuid(&uuid, dev);
    GpuDeviceUuidResponse resp;
    memcpy(resp.uuid, uuid.bytes, 16);
    return send_response(fd, GPU_CMD_GET_DEVICE_UUID,
                         (int32_t)cr, &resp, sizeof(resp));
}

static int handle_ctx_create(int fd, const void *payload, uint32_t len)
{
    GpuCtxCreateRequest req;
    if (len < sizeof(req))
        return send_response(fd, GPU_CMD_CTX_CREATE, -1, NULL, 0);
    memcpy(&req, payload, sizeof(req));

    /* Use cudaSetDevice as implicit context creation —
     * the CUDA runtime manages contexts per-device. */
    cudaError_t err = cudaSetDevice(req.device);
    GpuCtxCreateResponse resp = {
        .ctx_handle = (uint64_t)(uintptr_t)(req.device + 1) /* non-null opaque */
    };
    return send_response(fd, GPU_CMD_CTX_CREATE,
                         (int32_t)err, &resp, sizeof(resp));
}

static int handle_mem_get_info(int fd)
{
    size_t free_mem = 0, total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    GpuMemInfoResponse resp = {
        .free  = (uint64_t)free_mem,
        .total = (uint64_t)total_mem,
    };
    return send_response(fd, GPU_CMD_MEM_GET_INFO,
                         (int32_t)err, &resp, sizeof(resp));
}

static int handle_ctx_destroy(int fd)
{
    /* No-op — context cleanup happens per-connection in cleanup_connection() */
    return send_response(fd, GPU_CMD_CTX_DESTROY, 0, NULL, 0);
}

/* ================================================================
 * Per-connection handler (one thread per VM)
 * ================================================================ */

static void *connection_handler(void *arg)
{
    ConnectionCtx *ctx = (ConnectionCtx *)arg;
    int fd = ctx->fd;
    unsigned int cid = ctx->peer_cid;

    LOG_INFO("%s %u connected",
             ctx->is_tcp ? "TCP client" : "VM CID", cid);

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
        case GPU_CMD_FUNC_GET_ATTRIBUTES:
            rc = handle_func_get_attributes(ctx, buf, hdr.payload_len);
            break;
        case GPU_CMD_OCCUPANCY_MAX_BLOCKS:
            rc = handle_occupancy_max_blocks(ctx, buf, hdr.payload_len);
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

        /* CUDA Driver API (Phase 2 — Ollama / ML frameworks) */
        case GPU_CMD_GET_DRIVER_VERSION:
            rc = handle_get_driver_version(fd);
            break;
        case GPU_CMD_GET_DEVICE_UUID:
            rc = handle_get_device_uuid(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_CTX_CREATE:
            rc = handle_ctx_create(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_MEM_GET_INFO:
            rc = handle_mem_get_info(fd);
            break;
        case GPU_CMD_CTX_DESTROY:
            rc = handle_ctx_destroy(fd);
            break;

        /* cuBLAS GEMM proxy */
        case GPU_CMD_CUBLAS_GEMM_BATCHED:
            return handle_cublas_gemm_batched(ctx, payload, payload_len);
        case GPU_CMD_CUBLAS_GEMM_STRIDED:
            return handle_cublas_gemm_strided(ctx, payload, payload_len);

        case GPU_CMD_GOODBYE:
            LOG_INFO("CID %u: graceful disconnect (mem=%lu kernels=%u ktime=%lu µs)",
                     cid,
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
            LOG_ERR("CID %u: handler error, disconnecting", cid);
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
 * Signal handlers
 * ================================================================ */

static void sig_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

/* SIGHUP: set flag for main loop to reload tokens (async-signal-safe) */
static void sighup_handler(int sig)
{
    (void)sig;
    g_reload_tokens = 1;
}

/* Check and reload tokens if signaled — call from accept loops */
static void maybe_reload_tokens(void)
{
    if (g_reload_tokens) {
        g_reload_tokens = 0;
        load_tokens();
    }
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [-p port] [-t timeout_sec] [-T tcp_bind] [-v]\n", prog);
    fprintf(stderr, "  -p port          vsock/TCP port (default: %d)\n", GPU_PROXY_PORT);
    fprintf(stderr, "  -t timeout_sec   kernel timeout (default: %d, 0=disable)\n",
            (int)(GPU_DEFAULT_KERNEL_TIMEOUT_US / 1000000));
    fprintf(stderr, "  -T tcp_bind      enable TCP listener on addr (e.g. %s)\n", GPU_PROXY_TCP_BIND);
    fprintf(stderr, "  -v               verbose logging\n");
    exit(1);
}

/* ================================================================
 * TCP listener thread — runs alongside vsock listener in main()
 * ================================================================ */
static void *tcp_listener_thread(void *arg)
{
    int port = *(int *)arg;

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        LOG_ERR("TCP socket() failed: %s", strerror(errno));
        return NULL;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port   = htons((uint16_t)port),
    };
    inet_aton(g_tcp_bind, &addr.sin_addr);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        LOG_ERR("TCP bind(%s:%d) failed: %s", g_tcp_bind, port, strerror(errno));
        close(listen_fd);
        return NULL;
    }

    if (listen(listen_fd, 16) < 0) {
        LOG_ERR("TCP listen() failed: %s", strerror(errno));
        close(listen_fd);
        return NULL;
    }

    LOG_INFO("TCP listener on %s:%d (auth-token required)", g_tcp_bind, port);

    while (g_running) {
        maybe_reload_tokens();

        struct sockaddr_in peer;
        socklen_t peer_len = sizeof(peer);
        int client_fd = accept(listen_fd, (struct sockaddr *)&peer, &peer_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            LOG_ERR("TCP accept() failed: %s", strerror(errno));
            break;
        }

        ConnectionCtx *ctx = calloc(1, sizeof(ConnectionCtx));
        if (!ctx) {
            close(client_fd);
            continue;
        }
        ctx->fd = client_fd;
        ctx->peer_cid = 0;  /* No CID for TCP — identified by token */
        ctx->is_tcp = 1;

        LOG_INFO("TCP connection from %s:%d",
                 inet_ntoa(peer.sin_addr), ntohs(peer.sin_port));

        pthread_t tid;
        if (pthread_create(&tid, NULL, connection_handler, ctx) != 0) {
            LOG_ERR("pthread_create failed for TCP client: %s", strerror(errno));
            close(client_fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }

    close(listen_fd);
    return NULL;
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    int port = GPU_PROXY_PORT;
    int opt;

    while ((opt = getopt(argc, argv, "p:t:T:v")) != -1) {
        switch (opt) {
        case 'p': port = atoi(optarg); break;
        case 't': g_kernel_timeout_us = (uint64_t)atoi(optarg) * 1000000ULL; break;
        case 'T': g_tcp_enabled = 1; g_tcp_bind = optarg; break;
        case 'v': g_verbose = 1; break;
        default: usage(argv[0]);
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);
    signal(SIGHUP, sighup_handler);

    /* Load auth tokens for TCP connections */
    load_tokens();

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

    /* Create vsock listener (may fail on WSL2 — fall through to TCP-only) */
    int listen_fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        if (g_tcp_enabled) {
            LOG_INFO("vsock unavailable (%s) — running TCP-only mode", strerror(errno));

            /* Start TCP listener in foreground */
            pthread_t tcp_tid;
            if (pthread_create(&tcp_tid, NULL, tcp_listener_thread, &port) != 0) {
                LOG_ERR("Failed to start TCP listener: %s", strerror(errno));
                return 1;
            }

            /* Wait for shutdown signal */
            while (g_running) {
                maybe_reload_tokens();
                sleep(1);
            }
            LOG_INFO("Shutting down (TCP-only mode)");
            return 0;
        }
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

        /* vsock bind failed but TCP is available — fall through to TCP-only */
        if (g_tcp_enabled) {
            LOG_INFO("vsock bind failed — falling back to TCP-only mode");
            pthread_t tcp_tid;
            if (pthread_create(&tcp_tid, NULL, tcp_listener_thread, &port) != 0) {
                LOG_ERR("Failed to start TCP listener: %s", strerror(errno));
                return 1;
            }
            while (g_running) {
                maybe_reload_tokens();
                sleep(1);
            }
            LOG_INFO("Shutting down (TCP-only mode)");
            return 0;
        }
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

    /* Start TCP listener thread if enabled */
    pthread_t tcp_tid = 0;
    if (g_tcp_enabled) {
        if (pthread_create(&tcp_tid, NULL, tcp_listener_thread, &port) != 0) {
            LOG_ERR("Failed to start TCP listener thread: %s", strerror(errno));
        }
    }

    /* Accept loop (vsock) */
    while (g_running) {
        maybe_reload_tokens();

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
        ctx->is_tcp = 0;

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