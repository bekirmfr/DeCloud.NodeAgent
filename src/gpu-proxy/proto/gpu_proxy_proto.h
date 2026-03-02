/*
 * DeCloud GPU Proxy Protocol
 *
 * Wire format for forwarding CUDA Runtime API calls between a guest VM
 * (via the CUDA shim) and the host GPU proxy daemon over virtio-vsock.
 *
 * All multi-byte integers are little-endian.
 */

#ifndef DECLOUD_GPU_PROXY_PROTO_H
#define DECLOUD_GPU_PROXY_PROTO_H

#include <stdint.h>

#define GPU_PROXY_MAGIC     0x44435544  /* "DCUD" (DeCloud CUDA) */
#define GPU_PROXY_VERSION   2
#define GPU_PROXY_PORT      9999
#define GPU_PROXY_TCP_BIND  "192.168.122.1"
#define GPU_PROXY_MAX_PAYLOAD (2048UL * 1024 * 1024)  /* 2 GB — large fatbins */
#define GPU_PROXY_CHUNK_SIZE  (32UL * 1024 * 1024)    /* 32 MB memcpy chunks */
#define GPU_PROXY_TOKEN_LEN 32
#define GPU_MAX_KERNEL_PARAMS 64
#define GPU_DEFAULT_KERNEL_TIMEOUT_US 30000000  /* 30 seconds */

/* ================================================================
 * Command IDs
 * ================================================================ */
typedef enum {
    /* Device management */
    GPU_CMD_GET_DEVICE_COUNT       = 0x01,
    GPU_CMD_GET_DEVICE_PROPERTIES  = 0x02,
    GPU_CMD_SET_DEVICE             = 0x03,

    /* CUDA Driver API */
    GPU_CMD_GET_DRIVER_VERSION     = 0x04,
    GPU_CMD_GET_DEVICE_UUID        = 0x05,

    /* Memory management */
    GPU_CMD_MALLOC                 = 0x10,
    GPU_CMD_FREE                   = 0x11,
    GPU_CMD_MEMCPY                 = 0x12,
    GPU_CMD_MEMSET                 = 0x13,

    /* Execution */
    GPU_CMD_LAUNCH_KERNEL          = 0x20,
    GPU_CMD_DEVICE_SYNCHRONIZE     = 0x21,
    GPU_CMD_CTX_CREATE             = 0x22,
    GPU_CMD_MEM_GET_INFO           = 0x23,
    GPU_CMD_CTX_DESTROY            = 0x24,

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

    /* Module/function registration */
    GPU_CMD_REGISTER_MODULE        = 0x50,
    GPU_CMD_UNREGISTER_MODULE      = 0x51,
    GPU_CMD_REGISTER_FUNCTION      = 0x52,
    GPU_CMD_REGISTER_VAR           = 0x53,

    /* True GPU Presence — kernel attribute & occupancy queries */
    GPU_CMD_FUNC_GET_ATTRIBUTES    = 0x54,
    GPU_CMD_OCCUPANCY_MAX_BLOCKS   = 0x55,

    /* cuBLAS GEMM proxy (GQA attention requires real cuBLAS on host) */
    GPU_CMD_CUBLAS_GEMM_BATCHED    = 0x56,
    GPU_CMD_CUBLAS_GEMM_STRIDED    = 0x57,

    /* Resource management */
    GPU_CMD_SET_MEMORY_QUOTA       = 0x60,
    GPU_CMD_GET_USAGE_STATS        = 0x61,

    /* Lifecycle */
    GPU_CMD_HELLO                  = 0xF0,
    GPU_CMD_GOODBYE                = 0xF1,
} GpuProxyCmd;

/* cudaMemcpyKind equivalent */
typedef enum {
    GPU_MEMCPY_HOST_TO_HOST     = 0,
    GPU_MEMCPY_HOST_TO_DEVICE   = 1,
    GPU_MEMCPY_DEVICE_TO_HOST   = 2,
    GPU_MEMCPY_DEVICE_TO_DEVICE = 3,
} GpuMemcpyKind;

/* ================================================================
 * Wire format header (16 bytes)
 * ================================================================ */

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t  version;
    uint8_t  cmd;
    uint16_t flags;
    uint32_t payload_len;
    int32_t  status;
} GpuProxyHeader;

/* ================================================================
 * Per-command payload structures
 * ================================================================ */

/* --- HELLO --- */
typedef struct __attribute__((packed)) {
    uint32_t shim_version;
    uint32_t pid;
    uint8_t  auth_token[GPU_PROXY_TOKEN_LEN];
} GpuHelloRequest;

typedef struct __attribute__((packed)) {
    uint32_t daemon_version;
    uint32_t device_count;
} GpuHelloResponse;

/* --- GET_DEVICE_COUNT --- */
typedef struct __attribute__((packed)) {
    int32_t count;
} GpuGetDeviceCountResponse;

/* --- GET_DEVICE_PROPERTIES --- */
typedef struct __attribute__((packed)) {
    int32_t device;
} GpuGetDevicePropertiesRequest;

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
    int32_t  major;
    int32_t  minor;
    int32_t  max_threads_per_multiprocessor;
    int32_t  memory_clock_rate;
    int32_t  memory_bus_width;
    int32_t  l2_cache_size;
} GpuDeviceProperties;

/* --- SET_DEVICE --- */
typedef struct __attribute__((packed)) {
    int32_t device;
} GpuSetDeviceRequest;

/* --- MALLOC --- */
typedef struct __attribute__((packed)) {
    uint64_t size;
} GpuMallocRequest;

typedef struct __attribute__((packed)) {
    uint64_t device_ptr;
} GpuMallocResponse;

/* --- FREE --- */
typedef struct __attribute__((packed)) {
    uint64_t device_ptr;
} GpuFreeRequest;

/* --- MEMCPY --- */
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint64_t src;
    uint64_t count;
    int32_t  kind;
} GpuMemcpyRequest;

/* Helper: calculate total H2D payload including embedded data */
static inline uint32_t gpu_memcpy_h2d_payload_len(uint64_t count) {
    return (uint32_t)(sizeof(GpuMemcpyRequest) + count);
}

/* --- MEMSET --- */
typedef struct __attribute__((packed)) {
    uint64_t device_ptr;
    int32_t  value;
    uint64_t count;
} GpuMemsetRequest;

/* --- REGISTER_MODULE --- */
typedef struct __attribute__((packed)) {
    uint64_t fatbin_size;
} GpuRegisterModuleRequest;

typedef struct __attribute__((packed)) {
    uint64_t module_handle;
} GpuRegisterModuleResponse;

/* --- UNREGISTER_MODULE --- */
typedef struct __attribute__((packed)) {
    uint64_t module_handle;
} GpuUnregisterModuleRequest;

/* --- REGISTER_FUNCTION --- */
typedef struct __attribute__((packed)) {
    uint64_t module_handle;
    uint64_t host_func_ptr;
    uint32_t device_name_len;
} GpuRegisterFunctionRequest;

typedef struct __attribute__((packed)) {
    uint32_t num_params;
    uint32_t param_sizes[GPU_MAX_KERNEL_PARAMS];
} GpuRegisterFunctionResponse;

/* --- LAUNCH_KERNEL --- */
typedef struct __attribute__((packed)) {
    uint64_t host_func_ptr;
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint64_t shared_mem_bytes;
    uint64_t stream_handle;
    uint32_t num_params;
    uint32_t args_total_size;
} GpuLaunchKernelRequest;

/* --- STREAM --- */
typedef struct __attribute__((packed)) {
    uint32_t flags;
} GpuStreamCreateRequest;

typedef struct __attribute__((packed)) {
    uint64_t stream_handle;
} GpuStreamCreateResponse;

typedef struct __attribute__((packed)) {
    uint64_t stream_handle;
} GpuStreamDestroyRequest;

typedef struct __attribute__((packed)) {
    uint64_t stream_handle;
} GpuStreamSynchronizeRequest;

/* --- EVENTS --- */
typedef struct __attribute__((packed)) {
    uint32_t flags;
} GpuEventCreateRequest;

typedef struct __attribute__((packed)) {
    uint64_t event_handle;
} GpuEventCreateResponse;

typedef struct __attribute__((packed)) {
    uint64_t event_handle;
} GpuEventDestroyRequest;

typedef struct __attribute__((packed)) {
    uint64_t event_handle;
    uint64_t stream_handle;
} GpuEventRecordRequest;

typedef struct __attribute__((packed)) {
    uint64_t event_handle;
} GpuEventSynchronizeRequest;

typedef struct __attribute__((packed)) {
    uint64_t start_handle;
    uint64_t end_handle;
} GpuEventElapsedTimeRequest;

typedef struct __attribute__((packed)) {
    float milliseconds;
} GpuEventElapsedTimeResponse;

/* --- DRIVER API --- */
typedef struct __attribute__((packed)) {
    int32_t version;
} GpuDriverVersionResponse;

typedef struct __attribute__((packed)) {
    int32_t device;
} GpuDeviceUuidRequest;

typedef struct __attribute__((packed)) {
    uint8_t uuid[16];
} GpuDeviceUuidResponse;

typedef struct __attribute__((packed)) {
    int32_t  device;
    uint32_t flags;
} GpuCtxCreateRequest;

typedef struct __attribute__((packed)) {
    uint64_t ctx_handle;
} GpuCtxCreateResponse;

typedef struct __attribute__((packed)) {
    uint64_t free;
    uint64_t total;
} GpuMemInfoResponse;

/* --- TRUE GPU PRESENCE: FUNC_GET_ATTRIBUTES (0x54) --- */
typedef struct __attribute__((packed)) {
    uint64_t host_func_ptr;   /* VM-side function pointer (lookup key) */
} GpuFuncGetAttributesRequest;

typedef struct __attribute__((packed)) {
    int32_t  binaryVersion;
    int32_t  maxThreadsPerBlock;
    int32_t  numRegs;
    int32_t  sharedSizeBytes;
    int32_t  constSizeBytes;
    int32_t  localSizeBytes;
    int32_t  maxDynamicSharedSizeBytes;
    int32_t  preferredShmemCarveout;
    int32_t  ptxVersion;
} GpuFuncGetAttributesResponse;

/* --- TRUE GPU PRESENCE: OCCUPANCY_MAX_BLOCKS (0x55) --- */
typedef struct __attribute__((packed)) {
    uint64_t host_func_ptr;
    int32_t  blockSize;
    uint64_t dynamicSMemSize;
    uint32_t flags;
} GpuOccupancyMaxBlocksRequest;

typedef struct __attribute__((packed)) {
    int32_t numBlocks;
} GpuOccupancyMaxBlocksResponse;

/* --- CUBLAS GEMM BATCHED (0x56) ---
 *
 * Proxies cublasGemmBatchedEx for GQA attention Q×K multiplication.
 * Fixed header followed by (3 * batchCount) uint64_t device pointers:
 *   Aarray[batchCount], Barray[batchCount], Carray[batchCount]
 *
 * Device pointers come from the same connection's cudaMalloc calls.
 * Daemon calls real cublasGemmBatchedEx on the host GPU.
 */
typedef struct __attribute__((packed)) {
    int32_t  transa;         /* CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_OP_C=2 */
    int32_t  transb;
    int32_t  m;
    int32_t  n;
    int32_t  k;
    int32_t  Atype;          /* CUDA_R_16F=2, CUDA_R_32F=0, etc. */
    int32_t  lda;
    int32_t  Btype;
    int32_t  ldb;
    int32_t  Ctype;
    int32_t  ldc;
    int32_t  batchCount;
    int32_t  computeType;
    int32_t  algo;           /* CUBLAS_GEMM_DEFAULT=(-1), _TENSOR_OP=99 */
    uint8_t  alpha[16];      /* Up to 128-bit scalar (covers double complex) */
    uint8_t  beta[16];
    /* Followed by: uint64_t Aarray[batchCount]
     *              uint64_t Barray[batchCount]
     *              uint64_t Carray[batchCount]
     */
} GpuCublasGemmBatchedRequest;

/* --- CUBLAS GEMM STRIDED BATCHED (0x57) --- */
typedef struct __attribute__((packed)) {
    int32_t  transa;
    int32_t  transb;
    int32_t  m;
    int32_t  n;
    int32_t  k;
    int32_t  Atype;
    int32_t  lda;
    int64_t  strideA;
    int32_t  Btype;
    int32_t  ldb;
    int64_t  strideB;
    int32_t  Ctype;
    int32_t  ldc;
    int64_t  strideC;
    int32_t  batchCount;
    int32_t  computeType;
    int32_t  algo;
    uint8_t  alpha[16];
    uint8_t  beta[16];
    uint64_t A_ptr;          /* Single device pointer + stride */
    uint64_t B_ptr;
    uint64_t C_ptr;
} GpuCublasGemmStridedRequest;

/* --- RESOURCE MANAGEMENT --- */
typedef struct __attribute__((packed)) {
    uint64_t quota_bytes;
} GpuSetMemoryQuotaRequest;

typedef struct __attribute__((packed)) {
    uint64_t memory_allocated;
    uint64_t memory_quota;
    uint64_t peak_memory;
    uint64_t total_alloc_bytes;
    uint32_t kernel_launches;
    uint32_t kernel_timeouts;
    uint64_t kernel_time_us;
    uint64_t connect_time_us;
} GpuUsageStatsResponse;

#endif /* DECLOUD_GPU_PROXY_PROTO_H */