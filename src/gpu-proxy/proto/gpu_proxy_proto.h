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
 * Helper: compute total request size for memcpy H2D
 * ================================================================ */
static inline uint32_t gpu_memcpy_h2d_payload_len(uint64_t data_size)
{
    return (uint32_t)(sizeof(GpuMemcpyRequest) + data_size);
}

#endif /* DECLOUD_GPU_PROXY_PROTO_H */
