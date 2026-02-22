/*
 * DeCloud GPU Proxy Protocol
 *
 * Wire format for forwarding CUDA Runtime API calls between a guest VM
 * (via the CUDA shim) and the host GPU proxy daemon over virtio-vsock.
 *
 * All multi-byte integers are little-endian.
 * All messages are request/response pairs: shim sends GpuProxyRequest,
 * daemon replies with GpuProxyResponse.
 */

#ifndef DECLOUD_GPU_PROXY_PROTO_H
#define DECLOUD_GPU_PROXY_PROTO_H

#include <stdint.h>

#define GPU_PROXY_MAGIC     0x44435544  /* "DCUD" (DeCloud CUDA) */
#define GPU_PROXY_VERSION   1
#define GPU_PROXY_PORT      9999        /* vsock port the daemon listens on */
#define GPU_PROXY_MAX_PAYLOAD (64 * 1024 * 1024)  /* 64 MB max transfer */

/* ================================================================
 * Command IDs — one per CUDA Runtime API function we intercept
 * ================================================================ */
typedef enum {
    /* Device management */
    GPU_CMD_GET_DEVICE_COUNT       = 0x01,
    GPU_CMD_GET_DEVICE_PROPERTIES  = 0x02,
    GPU_CMD_SET_DEVICE             = 0x03,

    /* Memory management */
    GPU_CMD_MALLOC                 = 0x10,
    GPU_CMD_FREE                   = 0x11,
    GPU_CMD_MEMCPY                 = 0x12,
    GPU_CMD_MEMSET                 = 0x13,

    /* Execution */
    GPU_CMD_LAUNCH_KERNEL          = 0x20,
    GPU_CMD_DEVICE_SYNCHRONIZE     = 0x21,

    /* Stream management */
    GPU_CMD_STREAM_CREATE          = 0x30,
    GPU_CMD_STREAM_DESTROY         = 0x31,
    GPU_CMD_STREAM_SYNCHRONIZE     = 0x32,

    /* Event management */
    GPU_CMD_EVENT_CREATE           = 0x40,
    GPU_CMD_EVENT_DESTROY          = 0x41,
    GPU_CMD_EVENT_RECORD           = 0x42,
    GPU_CMD_EVENT_SYNCHRONIZE      = 0x43,
    GPU_CMD_EVENT_ELAPSED_TIME     = 0x44,

    /* Module/function registration (required for kernel launch) */
    GPU_CMD_REGISTER_MODULE        = 0x50,  /* Send fat binary → load CUmodule */
    GPU_CMD_UNREGISTER_MODULE      = 0x51,  /* Unload CUmodule */
    GPU_CMD_REGISTER_FUNCTION      = 0x52,  /* Map host ptr → device function name */
    GPU_CMD_REGISTER_VAR           = 0x53,  /* Register device variable */

    /* Resource management */
    GPU_CMD_SET_MEMORY_QUOTA       = 0x60,  /* Set per-VM GPU memory quota */
    GPU_CMD_GET_USAGE_STATS        = 0x61,  /* Query cumulative GPU usage */

    /* Lifecycle */
    GPU_CMD_HELLO                  = 0xF0,  /* Handshake: shim → daemon */
    GPU_CMD_GOODBYE                = 0xF1,  /* Graceful disconnect */
} GpuProxyCmd;

/* cudaMemcpyKind equivalent */
typedef enum {
    GPU_MEMCPY_HOST_TO_HOST     = 0,
    GPU_MEMCPY_HOST_TO_DEVICE   = 1,
    GPU_MEMCPY_DEVICE_TO_HOST   = 2,
    GPU_MEMCPY_DEVICE_TO_DEVICE = 3,
} GpuMemcpyKind;

/* ================================================================
 * Wire format
 * ================================================================
 *
 * Request:  [GpuProxyHeader][payload bytes...]
 * Response: [GpuProxyHeader][payload bytes...]
 *
 * The header is fixed-size (16 bytes). Payload is variable and
 * command-specific.
 */

typedef struct __attribute__((packed)) {
    uint32_t magic;          /* GPU_PROXY_MAGIC */
    uint8_t  version;        /* GPU_PROXY_VERSION */
    uint8_t  cmd;            /* GpuProxyCmd */
    uint16_t flags;          /* Reserved, must be 0 */
    uint32_t payload_len;    /* Bytes following this header */
    int32_t  status;         /* Request: 0. Response: cudaError_t */
} GpuProxyHeader;

/* ================================================================
 * Per-command payload structures
 * ================================================================ */

/* --- GPU_CMD_HELLO (request) --- */
typedef struct __attribute__((packed)) {
    uint32_t shim_version;   /* Shim protocol version */
    uint32_t pid;            /* Guest PID (for logging) */
} GpuHelloRequest;

/* --- GPU_CMD_HELLO (response) --- */
typedef struct __attribute__((packed)) {
    uint32_t daemon_version; /* Daemon protocol version */
    uint32_t device_count;   /* Number of GPUs available */
} GpuHelloResponse;

/* --- GPU_CMD_GET_DEVICE_COUNT (response) --- */
typedef struct __attribute__((packed)) {
    int32_t count;
} GpuGetDeviceCountResponse;

/* --- GPU_CMD_GET_DEVICE_PROPERTIES (request) --- */
typedef struct __attribute__((packed)) {
    int32_t device;
} GpuGetDevicePropertiesRequest;

/* --- GPU_CMD_GET_DEVICE_PROPERTIES (response) ---
 * Subset of cudaDeviceProp that ML frameworks typically check */
typedef struct __attribute__((packed)) {
    char     name[256];
    uint64_t total_global_mem;
    uint64_t shared_mem_per_block;
    int32_t  regs_per_block;
    int32_t  warp_size;
    int32_t  max_threads_per_block;
    int32_t  max_threads_dim[3];
    int32_t  max_grid_size[3];
    int32_t  clock_rate;
    int32_t  multi_processor_count;
    int32_t  major;          /* Compute capability major */
    int32_t  minor;          /* Compute capability minor */
    int32_t  max_threads_per_multiprocessor;
    int32_t  memory_clock_rate;
    int32_t  memory_bus_width;
    int32_t  l2_cache_size;
} GpuDeviceProperties;

/* --- GPU_CMD_SET_DEVICE (request) --- */
typedef struct __attribute__((packed)) {
    int32_t device;
} GpuSetDeviceRequest;

/* --- GPU_CMD_MALLOC (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t size;
} GpuMallocRequest;

/* --- GPU_CMD_MALLOC (response) --- */
typedef struct __attribute__((packed)) {
    uint64_t device_ptr;     /* Opaque handle (daemon-side pointer) */
} GpuMallocResponse;

/* --- GPU_CMD_FREE (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t device_ptr;
} GpuFreeRequest;

/* --- GPU_CMD_MEMCPY (request) ---
 * For H2D: payload follows this struct (payload_len - sizeof = data size)
 * For D2H: no trailing data, response carries the data
 * For D2D: no trailing data */
typedef struct __attribute__((packed)) {
    uint64_t dst;            /* Device ptr (H2D, D2D) or ignored (D2H) */
    uint64_t src;            /* Device ptr (D2H, D2D) or ignored (H2D) */
    uint64_t count;          /* Bytes to copy */
    int32_t  kind;           /* GpuMemcpyKind */
} GpuMemcpyRequest;

/* --- GPU_CMD_MEMSET (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t device_ptr;
    int32_t  value;
    uint64_t count;
} GpuMemsetRequest;

/* ================================================================
 * Module / function registration (for kernel launch)
 * ================================================================
 *
 * CUDA kernel launch flow:
 * 1. __cudaRegisterFatBinary(fatCubin)  → GPU_CMD_REGISTER_MODULE
 *    Sends the fat binary (PTX/cubin) to the daemon, which loads it
 *    via cuModuleLoadData and returns a module handle.
 *
 * 2. __cudaRegisterFunction(fatHandle, hostFun, deviceName, ...)
 *    → GPU_CMD_REGISTER_FUNCTION
 *    Maps a host-side function pointer to a device function name.
 *    Daemon calls cuModuleGetFunction and returns parameter metadata
 *    (count + sizes) so the shim can serialize args at launch time.
 *
 * 3. cudaLaunchKernel(func, grid, block, args, sharedMem, stream)
 *    → GPU_CMD_LAUNCH_KERNEL
 *    Shim looks up the function's param metadata, serializes the
 *    argument values, and sends them to the daemon which calls
 *    cuLaunchKernel on the real GPU.
 */

/* --- GPU_CMD_REGISTER_MODULE (request) ---
 * Payload: [GpuRegisterModuleRequest][fat binary data...] */
typedef struct __attribute__((packed)) {
    uint64_t fatbin_size;    /* Size of fat binary data following this struct */
} GpuRegisterModuleRequest;

/* --- GPU_CMD_REGISTER_MODULE (response) --- */
typedef struct __attribute__((packed)) {
    uint64_t module_handle;  /* Opaque handle (daemon-side CUmodule index) */
} GpuRegisterModuleResponse;

/* --- GPU_CMD_UNREGISTER_MODULE (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t module_handle;
} GpuUnregisterModuleRequest;

/* --- GPU_CMD_REGISTER_FUNCTION (request) ---
 * Payload: [GpuRegisterFunctionRequest][device_name string (null-terminated)] */
typedef struct __attribute__((packed)) {
    uint64_t module_handle;       /* From GPU_CMD_REGISTER_MODULE response */
    uint64_t host_func_ptr;       /* Shim-side key for this function */
    uint32_t device_name_len;     /* Length of device function name including null */
} GpuRegisterFunctionRequest;

/* --- GPU_CMD_REGISTER_FUNCTION (response) --- */
#define GPU_MAX_KERNEL_PARAMS 64
typedef struct __attribute__((packed)) {
    uint32_t num_params;                        /* Number of kernel parameters */
    uint32_t param_sizes[GPU_MAX_KERNEL_PARAMS]; /* Size (bytes) of each param */
} GpuRegisterFunctionResponse;

/* --- GPU_CMD_LAUNCH_KERNEL (request) ---
 * Payload: [GpuLaunchKernelRequest][serialized arg data...] */
typedef struct __attribute__((packed)) {
    uint64_t host_func_ptr;       /* Identifies the function (shim-side key) */
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint64_t shared_mem_bytes;
    uint64_t stream_handle;       /* 0 = default stream */
    uint32_t num_params;          /* Number of parameters */
    uint32_t args_total_size;     /* Total bytes of serialized arg data following */
} GpuLaunchKernelRequest;

/* ================================================================
 * Stream operations
 * ================================================================ */

/* --- GPU_CMD_STREAM_CREATE (request) --- */
typedef struct __attribute__((packed)) {
    uint32_t flags;              /* cudaStreamDefault=0, cudaStreamNonBlocking=1 */
} GpuStreamCreateRequest;

/* --- GPU_CMD_STREAM_CREATE (response) --- */
typedef struct __attribute__((packed)) {
    uint64_t stream_handle;      /* Opaque handle (daemon-side cudaStream_t) */
} GpuStreamCreateResponse;

/* --- GPU_CMD_STREAM_DESTROY (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t stream_handle;
} GpuStreamDestroyRequest;

/* --- GPU_CMD_STREAM_SYNCHRONIZE (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t stream_handle;
} GpuStreamSynchronizeRequest;

/* ================================================================
 * Event operations
 * ================================================================ */

/* --- GPU_CMD_EVENT_CREATE (request) --- */
typedef struct __attribute__((packed)) {
    uint32_t flags;              /* cudaEventDefault=0, cudaEventDisableTiming=2, etc. */
} GpuEventCreateRequest;

/* --- GPU_CMD_EVENT_CREATE (response) --- */
typedef struct __attribute__((packed)) {
    uint64_t event_handle;       /* Opaque handle (daemon-side cudaEvent_t) */
} GpuEventCreateResponse;

/* --- GPU_CMD_EVENT_DESTROY (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t event_handle;
} GpuEventDestroyRequest;

/* --- GPU_CMD_EVENT_RECORD (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t event_handle;
    uint64_t stream_handle;      /* 0 = default stream */
} GpuEventRecordRequest;

/* --- GPU_CMD_EVENT_SYNCHRONIZE (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t event_handle;
} GpuEventSynchronizeRequest;

/* --- GPU_CMD_EVENT_ELAPSED_TIME (request) --- */
typedef struct __attribute__((packed)) {
    uint64_t start_event;
    uint64_t end_event;
} GpuEventElapsedTimeRequest;

/* --- GPU_CMD_EVENT_ELAPSED_TIME (response) --- */
typedef struct __attribute__((packed)) {
    float elapsed_ms;
} GpuEventElapsedTimeResponse;

/* ================================================================
 * Resource management — memory quotas & usage metering
 * ================================================================ */

/* --- GPU_CMD_SET_MEMORY_QUOTA (request) ---
 * Sent by the orchestrator (via daemon CLI or config) to cap per-VM
 * GPU memory usage. The daemon enforces this on cudaMalloc. */
typedef struct __attribute__((packed)) {
    uint64_t quota_bytes;        /* 0 = unlimited */
} GpuSetMemoryQuotaRequest;

/* --- GPU_CMD_GET_USAGE_STATS (response) ---
 * Cumulative GPU usage for a single VM connection, used for billing. */
typedef struct __attribute__((packed)) {
    uint64_t memory_allocated;   /* Current GPU memory in use (bytes) */
    uint64_t memory_quota;       /* Configured quota (0 = unlimited) */
    uint64_t peak_memory;        /* High-water mark (bytes) */
    uint64_t total_alloc_bytes;  /* Cumulative bytes allocated */
    uint32_t kernel_launches;    /* Total kernel launches */
    uint32_t kernel_timeouts;    /* Kernels killed by timeout */
    uint64_t kernel_time_us;     /* Cumulative kernel execution time (µs) */
    uint64_t connect_time_us;    /* Time since connection (µs) */
} GpuUsageStatsResponse;

/* Default kernel execution timeout (microseconds). 0 = no timeout.
 * Can be overridden per-daemon via -t flag. */
#define GPU_PROXY_DEFAULT_KERNEL_TIMEOUT_US  (30ULL * 1000000ULL)  /* 30 seconds */

/* ================================================================
 * Helpers
 * ================================================================ */

static inline uint32_t gpu_memcpy_h2d_payload_len(uint64_t data_size)
{
    return (uint32_t)(sizeof(GpuMemcpyRequest) + data_size);
}

#endif /* DECLOUD_GPU_PROXY_PROTO_H */
