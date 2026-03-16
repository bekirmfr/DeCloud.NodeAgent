/*
 * DeCloud CUDA Runtime API Shim (libcudart.so.12)
 *
 * Production shim that forwards CUDA Runtime API calls to the host
 * GPU proxy daemon over TCP/vsock.
 *
 * Build: gcc -shared -fPIC -o libcudart.so.12 cuda_shim.c -ldl -lpthread \
 *        -Wl,-soname,libcudart.so.12
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <stdint.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <signal.h>
#include <ucontext.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "proto/gpu_proxy_proto.h"

/* ================================================================
 * Constructor — force GPU proxy env vars for subprocess propagation.
 *
 * Ollama (and potentially other multi-process frameworks) spawns runner
 * subprocesses with a filtered environment that strips non-OLLAMA_ vars.
 * Since this shim loads via LD_PRELOAD before main(), we can setenv()
 * the critical ggml flags here so they're present when ggml reads them.
 * ================================================================ */

/* Global flag: when true (default), CUDA graph stubs return cudaSuccess.
 * When false, they return cudaErrorNotSupported (honest).
 * Defaults to 1 for safety — set DECLOUD_GPU_GRAPH_NOOP=0 to disable. */
static int g_graph_noop = 1;
static int g_debug_log = 0;

/* ================================================================
 * Diagnostic file logging — writes to /tmp/gpu-proxy-diag.log
 * Enabled by:  touch /tmp/gpu-proxy-diag   (or DECLOUD_GPU_DIAG=1)
 * Always appends so multiple processes show interleaved logs.
 * ================================================================ */
#define DIAG_LOG_PATH "/tmp/gpu-proxy-diag.log"
#define DIAG_TRIGGER  "/tmp/gpu-proxy-diag"

static int g_diag_enabled = 0;
static FILE *g_diag_fp = NULL;
static pthread_mutex_t g_diag_lock = PTHREAD_MUTEX_INITIALIZER;

/* Atomic counters for summary stats */
static volatile int g_diag_kernel_launches = 0;
static volatile int g_diag_graph_begins = 0;
static volatile int g_diag_graph_ends = 0;
static volatile int g_diag_graph_instantiates = 0;
static volatile int g_diag_graph_launches = 0;
static volatile int g_diag_graph_updates = 0;
static volatile int g_diag_memcpy_h2d = 0;
static volatile int g_diag_memcpy_d2h = 0;
static volatile int g_diag_mallocs = 0;
static volatile int g_diag_rpc_errors = 0;

static void diag_init(void)
{
    /* Check trigger file or env var */
    struct stat st;
    if (stat(DIAG_TRIGGER, &st) == 0 || getenv("DECLOUD_GPU_DIAG")) {
        g_diag_fp = fopen(DIAG_LOG_PATH, "a");
        if (g_diag_fp) {
            g_diag_enabled = 1;
            setvbuf(g_diag_fp, NULL, _IOLBF, 0); /* line-buffered */
        }
    }
}

static void diag_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void diag_log(const char *fmt, ...)
{
    if (!g_diag_enabled) return;
    pthread_mutex_lock(&g_diag_lock);
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    fprintf(g_diag_fp, "[%02d:%02d:%02d.%03ld pid=%d] ",
            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000,
            (int)getpid());
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_diag_fp, fmt, ap);
    va_end(ap);
    fputc('\n', g_diag_fp);
    pthread_mutex_unlock(&g_diag_lock);
}

#define DIAG(fmt, ...) do { if (g_diag_enabled) diag_log(fmt, ##__VA_ARGS__); } while(0)

/* ================================================================
 * CUDA Graph Capture & Replay (Bug 22 fix)
 *
 * ggml with USE_GRAPHS=1 launches kernels ONLY during graph capture,
 * then replays via cudaGraphLaunch for subsequent tokens.  With the
 * old no-op stubs, cudaGraphLaunch did nothing → 0 computation →
 * gibberish from token 2 onward.
 *
 * Fix: record every kernel launch during capture but do NOT execute
 * them eagerly.  On cudaGraphLaunch, replay the recorded launches
 * via RPC — that is the sole execution.
 *
 * The old approach executed kernels eagerly during capture AND replayed
 * them via cudaGraphLaunch, causing double-execution.  Any kernel that
 * uses atomicAdd (flash-attention softmax denominator, reductions) or
 * writes in-place (RMSNorm, residual add) produced corrupted results
 * on the replay pass because device memory had already been mutated
 * by the eager pass.  Symptom: garbled / interleaved token output.
 * ================================================================ */

#define GRAPH_MAX_RECORDED_OPS  16384
#define GRAPH_MAX_EXEC_SLOTS    8

/* Function pointer type for RPC calls — matches both rpc_call() and
 * transport_rpc_call() signatures.  Used to route graph-replay ops
 * through the correct daemon connection (runtime shim vs driver shim). */
typedef int (*graph_rpc_fn_t)(uint8_t cmd,
                               const void *req, uint32_t req_len,
                               void *resp, uint32_t resp_len,
                               uint32_t *actual);

typedef struct {
    void    *rpc_payload;   /* deep copy of serialized RPC request (kernel or cuBLAS) */
    uint32_t rpc_len;
    uint8_t  rpc_cmd;       /* GPU_CMD_LAUNCH_KERNEL, GPU_CMD_CUBLAS_GEMM_BATCHED, etc. */
    graph_rpc_fn_t rpc_fn;  /* RPC function for replay (NULL → use default rpc_call) */
} GraphRecordedOp;

typedef struct {
    GraphRecordedOp *ops;
    int              count;
    int              capacity;
} GraphRecord;

/* Per-graph-exec slot */
typedef struct {
    GraphRecord record;
    int         in_use;
} GraphExecSlot;

static GraphExecSlot  g_graph_exec_slots[GRAPH_MAX_EXEC_SLOTS];
static pthread_mutex_t g_graph_exec_lock = PTHREAD_MUTEX_INITIALIZER;

/* Capture state (thread-local — ggml uses a single thread for CUDA ops) */
static __thread int          g_graph_capturing = 0;
static __thread GraphRecord  g_graph_capture_buf;

static void graph_record_free(GraphRecord *rec)
{
    if (!rec->ops) return;
    for (int i = 0; i < rec->count; i++)
        free(rec->ops[i].rpc_payload);
    free(rec->ops);
    rec->ops = NULL;
    rec->count = 0;
    rec->capacity = 0;
}

static void graph_record_copy(GraphRecord *dst, const GraphRecord *src)
{
    dst->count    = src->count;
    dst->capacity = src->count;
    dst->ops      = (GraphRecordedOp *)calloc(src->count, sizeof(GraphRecordedOp));
    for (int i = 0; i < src->count; i++) {
        dst->ops[i].rpc_cmd     = src->ops[i].rpc_cmd;
        dst->ops[i].rpc_len     = src->ops[i].rpc_len;
        dst->ops[i].rpc_fn      = src->ops[i].rpc_fn;
        dst->ops[i].rpc_payload = malloc(src->ops[i].rpc_len);
        memcpy(dst->ops[i].rpc_payload, src->ops[i].rpc_payload,
               src->ops[i].rpc_len);
    }
}

static int graph_exec_alloc(void)
{
    pthread_mutex_lock(&g_graph_exec_lock);
    for (int i = 0; i < GRAPH_MAX_EXEC_SLOTS; i++) {
        if (!g_graph_exec_slots[i].in_use) {
            g_graph_exec_slots[i].in_use = 1;
            memset(&g_graph_exec_slots[i].record, 0, sizeof(GraphRecord));
            pthread_mutex_unlock(&g_graph_exec_lock);
            return i;
        }
    }
    pthread_mutex_unlock(&g_graph_exec_lock);
    return -1;
}

static void graph_exec_free(int idx)
{
    if (idx < 0 || idx >= GRAPH_MAX_EXEC_SLOTS) return;
    pthread_mutex_lock(&g_graph_exec_lock);
    graph_record_free(&g_graph_exec_slots[idx].record);
    g_graph_exec_slots[idx].in_use = 0;
    pthread_mutex_unlock(&g_graph_exec_lock);
}

/* Encode slot index as an opaque pointer (offset by 0xDE000000 to avoid NULL) */
#define GRAPH_EXEC_TO_PTR(idx)  ((cudaGraphExec_t)(uintptr_t)(0xDE000000 + (idx)))
#define GRAPH_EXEC_FROM_PTR(p)  ((int)((uintptr_t)(p) - 0xDE000000))

/* ================================================================
 * Exported graph-capture helpers for cuBLAS/cublasLt stubs
 *
 * The cuBLAS stub (libcublas_stub.so) is a separate .so and cannot
 * access our thread-local g_graph_capturing directly.  These functions
 * let the stub query capture state and record cuBLAS RPC ops into the
 * capture buffer, so they participate in graph replay correctly.
 * ================================================================ */

int decloud_graph_is_capturing(void)
{
    return g_graph_capturing;
}

/* Forward declaration */
int decloud_graph_record_op_ex(uint8_t cmd, const void *payload, uint32_t len,
                                void *rpc_fn);

int decloud_graph_record_op(uint8_t cmd, const void *payload, uint32_t len)
{
    return decloud_graph_record_op_ex(cmd, payload, len, NULL);
}

/* Extended variant: records an op with a specific RPC function for replay.
 * The driver shim passes its own transport_rpc_call so that driver-API
 * kernels (flash-attention) are replayed through the driver connection
 * where those functions are registered, not the runtime shim's connection. */
int decloud_graph_record_op_ex(uint8_t cmd, const void *payload, uint32_t len,
                                void *rpc_fn)
{
    if (!g_graph_capturing) return -1;
    if (g_graph_capture_buf.count >= GRAPH_MAX_RECORDED_OPS) return -1;

    GraphRecord *cap = &g_graph_capture_buf;
    if (cap->count >= cap->capacity) {
        int new_cap = cap->capacity ? cap->capacity * 2 : 256;
        if (new_cap > GRAPH_MAX_RECORDED_OPS) new_cap = GRAPH_MAX_RECORDED_OPS;
        cap->ops = (GraphRecordedOp *)realloc(cap->ops,
                    new_cap * sizeof(GraphRecordedOp));
        cap->capacity = new_cap;
    }
    GraphRecordedOp *op = &cap->ops[cap->count++];
    op->rpc_cmd     = cmd;
    op->rpc_len     = len;
    op->rpc_fn      = (graph_rpc_fn_t)rpc_fn;
    op->rpc_payload = malloc(len);
    memcpy(op->rpc_payload, payload, len);
    return 0;
}

/* Mask all FP exceptions in SSE MXCSR and x87 FCW.
 * Replaces fedisableexcept(FE_ALL_EXCEPT) — no -lm dependency. */
static inline void mask_fpe_exceptions(void)
{
    unsigned int mxcsr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 0x1F80U;   /* mask all SSE exceptions: IM|DM|ZM|OM|UM|PM */
    mxcsr &= ~0x003FU;  /* clear pending sticky exception flags */
    __asm__ __volatile__("ldmxcsr %0" : : "m"(mxcsr));
    __asm__ __volatile__("fnclex");  /* clear x87 exception flags */
    unsigned short fcw;
    __asm__ __volatile__("fstcw %0" : "=m"(fcw));
    fcw |= 0x003FU;     /* mask all x87 exceptions */
    __asm__ __volatile__("fldcw %0" : : "m"(fcw));
}

static void sigfpe_handler(int sig, siginfo_t *si, void *ctx)
{
    (void)sig; (void)si; (void)ctx;
    /* Mask all FPE exceptions and clear pending sticky flags.
     * Both steps are required: masking prevents re-trigger,
     * clearing sticky bits prevents the re-executed instruction
     * from immediately firing another SIGFPE. */
    mask_fpe_exceptions();
}

/* Intercept feenableexcept — libtorch_cuda.so calls this to enable FPE
 * exceptions during CublasHandlePool init. By shadowing libm's version
 * via LD_PRELOAD, we prevent FPE from ever being enabled. This is the
 * correct fix: prevent the exception, not catch it after the fact.
 * Returning 0 means "no exceptions were previously enabled" (correct). */
int feenableexcept(int excepts)
{
    fprintf(stderr, "[cudart-shim] feenableexcept(%d) intercepted — suppressed\n", excepts);
    (void)excepts;
    return 0;
}

__attribute__((constructor))
static void shim_init(void)
{
    /* Init diagnostics early — before any other work */
    diag_init();

    /* Read /proc/self/exe for process identification */
    char exe_path[256] = {0};
    readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    DIAG("=== CUDART SHIM CONSTRUCTOR START ===");
    DIAG("Process: %s (pid=%d, ppid=%d)", exe_path, getpid(), getppid());

    /* Read config file and propagate app-specific env vars.
     * Cloud-init templates write application env vars here (e.g.,
     * GGML_CUDA_FORCE_MMQ=1 for Ollama, CUDA_VISIBLE_DEVICES for PyTorch).
     * We setenv() any non-transport vars so they reach the application
     * even when the parent process filters the environment. */
    FILE *f = fopen("/etc/decloud/gpu-proxy.env", "r");
    int count = 0;
    if (f) {
        DIAG("Opened /etc/decloud/gpu-proxy.env");
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            char *p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '\0' || *p == '\n' || *p == '#') continue;
            char *nl = strchr(p, '\n');
            if (nl) *nl = '\0';
            char *eq = strchr(p, '=');
            if (!eq) continue;
            *eq = '\0';
            char *key = p;
            char *val = eq + 1;

            DIAG("  env-file: %s=%s", key, strcmp(key, "DECLOUD_GPU_PROXY_TOKEN") == 0 ? "<redacted>" : val);

            /* Skip transport config (handled by transport layer) and LD_PRELOAD */
            /* Consume proxy-level flags we handle internally */
            if (strcmp(key, "DECLOUD_GPU_GRAPH_NOOP") == 0) {
                g_graph_noop = (val[0] == '1');
                DIAG("  → g_graph_noop = %d", g_graph_noop);
                continue;
            }
            if (strcmp(key, "DECLOUD_GPU_DEBUG") == 0) {
                g_debug_log = 1;
                DIAG("  → g_debug_log = 1");
                continue;
            }
            if (strncmp(key, "DECLOUD_", 8) == 0) continue;
            if (strcmp(key, "LD_PRELOAD") == 0) continue;

            setenv(key, val, 1);
            count++;
        }
        fclose(f);
    } else {
        DIAG("WARNING: cannot open /etc/decloud/gpu-proxy.env: %s", strerror(errno));
    }

    if (getenv("DECLOUD_GPU_DEBUG")) {
        g_debug_log = 1;
        fprintf(stderr, "[cudart-shim] constructor: %d app env vars loaded\n", count);
    }

    /* Belt-and-suspenders: ask ggml to disable CUDA graphs if it checks
     * this env var.  Graph replay is now implemented below, but disabling
     * graphs avoids the overhead of capture+replay when the env var works. */
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0); /* don't overwrite if user set it */

    /* Dump critical env state for diagnostics */
    DIAG("Env state after constructor:");
    DIAG("  GGML_CUDA_DISABLE_GRAPHS=%s", getenv("GGML_CUDA_DISABLE_GRAPHS") ?: "(unset)");
    DIAG("  GGML_CUDA_FORCE_MMQ=%s", getenv("GGML_CUDA_FORCE_MMQ") ?: "(unset)");
    DIAG("  CUDA_VISIBLE_DEVICES=%s", getenv("CUDA_VISIBLE_DEVICES") ?: "(unset)");
    DIAG("  LD_PRELOAD=%s", getenv("LD_PRELOAD") ?: "(unset)");
    DIAG("  g_graph_noop=%d, g_debug_log=%d", g_graph_noop, g_debug_log);
    DIAG("  app env vars loaded from file: %d", count);
    DIAG("=== CUDART SHIM CONSTRUCTOR END ===");

    mask_fpe_exceptions();
    /* NOTE: We intentionally do NOT install a SIGFPE handler here.
     * FPE_INTDIV (integer divide by zero from idiv) cannot be masked via
     * MXCSR — it always re-triggers after the handler returns, causing an
     * infinite signal loop. FPE bugs must be fixed at source (see device
     * property safety clamps in cudaGetDeviceProperties). */
}

#define SHIM_LOG(fmt, ...) \
    do { if (g_debug_log) fprintf(stderr, "[cudart-shim] " fmt "\n", ##__VA_ARGS__); } while(0)

/*
 * CRITICAL: ggml-cuda allocates cudaDeviceProp on the stack with as little
 * as 1040 bytes (sub $0x410,%rsp with struct at rsp+0x0). Our struct
 * definition must NOT cause memset to write beyond the caller's allocation.
 * We use a fixed safe size for memset instead of sizeof(*prop).
 */
#define SAFE_PROP_MEMSET_SIZE 768

/* ================================================================
 * CUDA type definitions
 * ================================================================ */

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidDeviceFunction 8
#define cudaErrorNotSupported 71
#define cudaErrorNoDevice 100

typedef void *cudaStream_t;
typedef void *cudaEvent_t;

typedef enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
} cudaMemcpyKind_t;

typedef struct { unsigned int x, y, z; } dim3;

/*
 * cudaDeviceProp subset - fields that ML frameworks check.
 * NOTE: sizeof(this struct) may differ from caller's version.
 * NEVER use memset(prop, 0, sizeof(*prop)) on a caller-provided pointer.
 */
struct cudaDeviceProp {
    char     name[256];
    /* CUDA 12 ABI: real struct has uuid(16) + luid(8) + luidDeviceNodeMask(4) + pad(4) = 32 bytes
     * between name[256] and totalGlobalMem. Without these, all raw offset writes
     * are 32 bytes early, causing fields like maxThreadsPerMultiProcessor and
     * maxBlocksPerMultiProcessor to land at wrong addresses and read as 0 → FPE_INTDIV.
     * Verified by disassembling mbtopk::get_items_per_thread in libtorch_cuda.so:
     *   0x270 = maxThreadsPerMultiProcessor, 0x288 = regsPerMultiprocessor, 0x2c8 = maxBlocksPerMultiProcessor */
    unsigned char _uuid[16];
    char          _luid[8];
    unsigned int  _luidDeviceNodeMask;
    unsigned int  _pad;
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
};

/* ================================================================
 * CUDA fat binary structures
 * ================================================================ */

#define FATBINC_MAGIC 0x466243B1
#define FATBIN_MAGIC  0xBA55ED50

typedef struct {
    int   magic;
    int   version;
    const void *data;
    void  *filename_or_fatbins;
} __fatBinC_Wrapper_t;

/* ================================================================
 * Registration tracking
 * ================================================================ */

#define MAX_REGISTERED_FUNCTIONS 65536
#define MAX_DEFERRED_MODULES 512

typedef struct {
    uint64_t host_func_ptr;
    uint32_t num_params;
    uint32_t param_sizes[GPU_MAX_KERNEL_PARAMS];
    int      registered;
    char     device_name[1024];
    int      module_index;
} RegisteredFunction;

typedef struct {
    const void *fatbin_data;
    size_t      fatbin_size;
    uint64_t    remote_handle;
    int         uploaded;
} DeferredModule;

static RegisteredFunction g_functions[MAX_REGISTERED_FUNCTIONS];
static int g_function_count = 0;
static DeferredModule g_modules[MAX_DEFERRED_MODULES];
static int g_module_count = 0;
static int g_current_module_index = -1;

/* ================================================================
 * Connection state
 * ================================================================ */

static pthread_mutex_t g_conn_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_conn_fd = -1;
static int g_initialized = 0;

/* ================================================================
 * Helpers
 * ================================================================ */

static int get_env_int(const char *name, int def)
{
    const char *val = getenv(name);
    return val ? atoi(val) : def;
}

static int parse_hex_token(const char *hex, uint8_t *out, int len)
{
    if (!hex || (int)strlen(hex) != len * 2) return -1;
    for (int i = 0; i < len; i++) {
        unsigned int byte;
        if (sscanf(&hex[i * 2], "%02x", &byte) != 1) return -1;
        out[i] = (uint8_t)byte;
    }
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

/* ================================================================
 * Transport -- vsock first, TCP fallback
 * ================================================================ */

static int try_tcp_connect(int port)
{
    const char *host = getenv("DECLOUD_GPU_PROXY_HOST");
    if (!host) host = GPU_PROXY_TCP_BIND;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port   = htons((uint16_t)port),
    };
    if (inet_aton(host, &addr.sin_addr) == 0) {
        close(fd);
        return -1;
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        SHIM_LOG("TCP connect(%s:%d) failed: %s", host, port, strerror(errno));
        close(fd);
        return -1;
    }

    /* Disable Nagle — critical for low-latency RPC */
    int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    /* Disable delayed ACK — eliminates 40ms ato delay on small RPCs */
    int quickack = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));

    SHIM_LOG("Connected via TCP to %s:%d", host, port);
    return fd;
}

static int try_vsock_connect(int port)
{
    int cid = get_env_int("DECLOUD_GPU_PROXY_CID", VMADDR_CID_HOST);

    int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_vm addr = {
        .svm_family = AF_VSOCK,
        .svm_cid    = (unsigned int)cid,
        .svm_port   = (unsigned int)port,
    };

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        SHIM_LOG("vsock connect(CID=%d port=%d) failed: %s",
                 cid, port, strerror(errno));
        close(fd);
        return -1;
    }

    SHIM_LOG("Connected via vsock (CID=%d port=%d)", cid, port);
    return fd;
}

static int ensure_connected(void)
{
    pthread_mutex_lock(&g_conn_lock);
    if (g_conn_fd >= 0) {
        pthread_mutex_unlock(&g_conn_lock);
        return 0;
    }

    int port = get_env_int("DECLOUD_GPU_PROXY_PORT", GPU_PROXY_PORT);
    int is_tcp = 0;

    /* Respect DECLOUD_GPU_PROXY_TRANSPORT — "tcp" skips vsock entirely */
    const char *transport_mode = getenv("DECLOUD_GPU_PROXY_TRANSPORT");
    int try_vsock = 1;
    if (transport_mode && strcmp(transport_mode, "tcp") == 0)
        try_vsock = 0;

    int fd = -1;
    if (try_vsock) {
        fd = try_vsock_connect(port);
    }
    if (fd < 0) {
        fd = try_tcp_connect(port);
        if (fd < 0) {
            SHIM_LOG("All transports failed — GPU proxy unavailable");
            pthread_mutex_unlock(&g_conn_lock);
            return -1;
        }
        is_tcp = 1;
    }

    g_conn_fd = fd;

    /* Send HELLO handshake */
    GpuHelloRequest hello = {
        .shim_version = GPU_PROXY_VERSION,
        .pid = (uint32_t)getpid(),
        .auth_token = {0},
    };

    if (is_tcp) {
        const char *tok_hex = getenv("DECLOUD_GPU_PROXY_TOKEN");
        if (tok_hex) {
            parse_hex_token(tok_hex, hello.auth_token, GPU_PROXY_TOKEN_LEN);
        }
    }

    GpuProxyHeader hdr = {
        .magic = GPU_PROXY_MAGIC,
        .version = GPU_PROXY_VERSION,
        .cmd = GPU_CMD_HELLO,
        .payload_len = sizeof(hello),
        .status = 0,
    };

    if (write_exact(fd, &hdr, sizeof(hdr)) < 0 ||
        write_exact(fd, &hello, sizeof(hello)) < 0) {
        SHIM_LOG("Failed to send HELLO");
        close(fd);
        g_conn_fd = -1;
        pthread_mutex_unlock(&g_conn_lock);
        return -1;
    }

    GpuProxyHeader resp_hdr;
    if (read_exact(fd, &resp_hdr, sizeof(resp_hdr)) == 0) {
        if (resp_hdr.status != 0) {
            SHIM_LOG("HELLO rejected (status=%d) — auth failed?", resp_hdr.status);
            close(fd);
            g_conn_fd = -1;
            pthread_mutex_unlock(&g_conn_lock);
            return -1;
        }
        if (resp_hdr.payload_len >= sizeof(GpuHelloResponse)) {
            GpuHelloResponse resp;
            read_exact(fd, &resp, sizeof(resp));
            SHIM_LOG("Connected to GPU proxy (v%u, %u devices, %s)",
                     resp.daemon_version, resp.device_count,
                     is_tcp ? "TCP" : "vsock");
        }
    }

    g_initialized = 1;
    pthread_mutex_unlock(&g_conn_lock);
    return 0;
}

/* ================================================================
 * RPC call -- send request, receive response
 * ================================================================ */

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

    /* Re-arm TCP_QUICKACK before reading response (Linux resets it per-operation) */
    int qa = 1;
    setsockopt(g_conn_fd, IPPROTO_TCP, TCP_QUICKACK, &qa, sizeof(qa));

    GpuProxyHeader resp_hdr;
    if (read_exact(g_conn_fd, &resp_hdr, sizeof(resp_hdr)) < 0) goto err;

    if (resp_hdr.magic != GPU_PROXY_MAGIC) {
        SHIM_LOG("bad response magic 0x%08x", resp_hdr.magic);
        goto err;
    }

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
 * Fat binary registration
 * ================================================================ */

static size_t get_fatbin_size(const void *fatbin_data)
{
    const uint8_t *p = (const uint8_t *)fatbin_data;
    uint32_t magic;
    memcpy(&magic, p, 4);
    if (magic != FATBIN_MAGIC) {
        uint64_t size;
        memcpy(&size, p + 8, sizeof(size));
        return (size_t)size;
    }
    uint64_t total_size;
    memcpy(&total_size, p + 8, sizeof(total_size));
    return (size_t)total_size;
}

void **__cudaRegisterFatBinary(void *fatCubin)
{
    __fatBinC_Wrapper_t *wrapper = (__fatBinC_Wrapper_t *)fatCubin;
    if (!wrapper || wrapper->magic != FATBINC_MAGIC) {
        static void *dummy_handle = NULL;
        return &dummy_handle;
    }
    const void *fatbin_data = (const void *)wrapper->data;
    size_t fatbin_size = get_fatbin_size(fatbin_data);
    if (fatbin_size == 0) {
        static void *dummy_handle = NULL;
        return &dummy_handle;
    }
    /* No upper size limit — fatbin_data is mmap'd, we store a pointer.
     * Streaming upload in ensure_module_uploaded() writes directly. */
     
    /* LOCAL ONLY - no RPC */
    if (g_module_count < MAX_DEFERRED_MODULES) {
        DeferredModule *dm = &g_modules[g_module_count];
        dm->fatbin_data   = fatbin_data;
        dm->fatbin_size   = fatbin_size;
        dm->remote_handle = 0;
        dm->uploaded      = 0;
        g_current_module_index = g_module_count;
        g_module_count++;
    }
    static void *handle_ptr = (void *)1;
    return &handle_ptr;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    (void)fatCubinHandle;
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
    (void)fatCubinHandle; (void)thread_limit;
    (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
    const char *name = deviceFun ? deviceFun : deviceName;
    /* LOCAL ONLY - no RPC during dlopen */
    if (name && g_function_count < MAX_REGISTERED_FUNCTIONS) {
        RegisteredFunction *rf = &g_functions[g_function_count++];
        rf->host_func_ptr = (uint64_t)(uintptr_t)hostFun;
        rf->registered    = 0;
        rf->num_params    = 0;
        rf->module_index  = g_current_module_index;
        strncpy(rf->device_name, name, sizeof(rf->device_name) - 1);
        rf->device_name[sizeof(rf->device_name) - 1] = '\0';
        SHIM_LOG("__cudaRegisterFunction('%s', hostFun=%p) — #%d, mod=%d",
            name, hostFun, g_function_count, g_current_module_index);
    }
}

void __cudaRegisterVar(
    void   **fatCubinHandle,
    char    *hostVar,
    char    *deviceAddress,
    const char *deviceName,
    int      ext,
    size_t   size,
    int      constant,
    int      global)
{
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress;
    (void)ext; (void)constant; (void)global;
    SHIM_LOG("__cudaRegisterVar('%s', size=%zu) — stub", deviceName, size);
}

/* ================================================================
 * CUDA Runtime API — Device Management
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

/*
 * cudaGetDeviceProperties — fill caller's struct with device info.
 *
 * CRITICAL SAFETY: The caller (e.g. ggml-cuda) may allocate the struct
 * on the stack with as little as 1040 bytes. Our struct definition with
 * padding could be larger. We use SAFE_PROP_MEMSET_SIZE (768 bytes) to
 * avoid stack smashing, which safely covers all fields we populate.
 */
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    if (!prop) return cudaErrorInvalidValue;

    /* Safe memset — never exceed caller's stack allocation */
    memset(prop, 0, SAFE_PROP_MEMSET_SIZE);

    GpuGetDevicePropertiesRequest req = { .device = device };
    GpuDeviceProperties resp;
    int err = rpc_call(GPU_CMD_GET_DEVICE_PROPERTIES,
                       &req, sizeof(req), &resp, sizeof(resp), NULL);

    if (err == cudaSuccess) {
        /* Fill via struct fields (covers our struct layout) */
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
        prop->major                         = resp.major;
        prop->minor                         = resp.minor;
        prop->multiProcessorCount           = resp.multi_processor_count;
        prop->maxThreadsPerMultiProcessor   = resp.max_threads_per_multiprocessor;
        prop->memoryClockRate               = resp.memory_clock_rate;
        prop->memoryBusWidth                = resp.memory_bus_width;
        prop->l2CacheSize                   = resp.l2_cache_size;

        /*
         * CRITICAL: ggml-cuda reads cudaDeviceProp at REAL NVIDIA binary
         * offsets, not our simplified struct layout. Our struct fields above
         * land at WRONG offsets. We must overwrite at the exact offsets the
         * real CUDA 12 struct uses.
         *
         * Offsets determined by dumping a real RTX 4060 cudaDeviceProp:
         *   288: totalGlobalMem (size_t)         296: sharedMemPerBlock (size_t)
         *   304: regsPerBlock (int)              308: warpSize (int)  ← 0 = SIGFPE!
         *   320: maxThreadsPerBlock (int)        324-332: maxThreadsDim[3]
         *   336-344: maxGridSize[3]              348: clockRate (int)
         *   352: totalConstMem (size_t)          360: major  364: minor
         *   368: textureAlignment (size_t)       388: multiProcessorCount (int)
         *   608: memoryClockRate (int)           612: memoryBusWidth (int)
         *   616: l2CacheSize (int)               624: maxThreadsPerMP (int)
         *   640: sharedMemPerMultiprocessor      696: sharedMemPerBlockOptin
         */
        uint8_t *raw = (uint8_t *)prop;

        /* Memory specs */
        *(size_t *)(raw + 288) = resp.total_global_mem;
        *(size_t *)(raw + 296) = resp.shared_mem_per_block;
        *(int *)  (raw + 304) = resp.regs_per_block;
        *(int *)  (raw + 308) = resp.warp_size;             /* 0 → SIGFPE */
        *(size_t *)(raw + 312) = 2147483647;                /* memPitch */

        /* Thread/block dims */
        *(int *)(raw + 320) = resp.max_threads_per_block;
        *(int *)(raw + 324) = resp.max_threads_dim[0];
        *(int *)(raw + 328) = resp.max_threads_dim[1];
        *(int *)(raw + 332) = resp.max_threads_dim[2];
        *(int *)(raw + 336) = resp.max_grid_size[0];
        *(int *)(raw + 340) = resp.max_grid_size[1];
        *(int *)(raw + 344) = resp.max_grid_size[2];

        /* Clock & constant memory */
        *(int *)  (raw + 348) = resp.clock_rate;
        *(size_t *)(raw + 352) = 65536;                     /* totalConstMem */

        /* Compute capability */
        *(int *)(raw + 360) = resp.major;
        *(int *)(raw + 364) = resp.minor;
        *(size_t *)(raw + 368) = 512;                       /* textureAlignment */
        *(size_t *)(raw + 376) = 32;                        /* texturePitchAlignment */

        /* Processor info — NOTE: real offset 388, NOT 356 */
        *(int *)(raw + 384) = 1;                            /* deviceOverlap */
        *(int *)(raw + 388) = resp.multi_processor_count;

        /* Memory bus info */
        *(int *)(raw + 608) = resp.memory_clock_rate;
        *(int *)(raw + 612) = resp.memory_bus_width;
        *(int *)(raw + 616) = resp.l2_cache_size;

        /* maxThreadsPerMultiProcessor — real CUDA 12 offset 0x270=624; regsPerMultiprocessor at 0x288=648 */
        *(int *)(raw + 624) = resp.max_threads_per_multiprocessor;   /* maxThreadsPerMultiProcessor — real CUDA 12 offset 0x270=624 */
        *(int *)(raw + 648) = 65536;                                  /* regsPerMultiprocessor — real CUDA 12 offset 0x288=648 */

        /* Shared memory limits (derived from compute capability).
         * These are per-architecture constants from NVIDIA documentation.
         * Without correct values: mmq_x_best=0 → SIGABRT.
         */
        /* sharedMemPerMultiprocessor @ offset 640 */
        if      (resp.major == 8 && resp.minor == 0) *(int *)(raw + 640) = 167936;
        else if (resp.major >= 8)                    *(int *)(raw + 640) = 102400;
        else if (resp.major == 7 && resp.minor >= 5) *(int *)(raw + 640) = 65536;
        else if (resp.major == 7)                    *(int *)(raw + 640) = 98304;
        else                                         *(int *)(raw + 640) = 49152;

        /* sharedMemPerBlockOptin @ offset 696 */
        if      (resp.major == 8 && resp.minor == 0) *(int *)(raw + 696) = 167936;
        else if (resp.major >= 8)                    *(int *)(raw + 696) = 101376;
        else if (resp.major == 7 && resp.minor >= 5) *(int *)(raw + 696) = 65536;
        else if (resp.major == 7)                    *(int *)(raw + 696) = 98304;
        else                                         *(int *)(raw + 696) = 49152;

        /* Boolean capabilities ggml-cuda may check */
        *(int *)(raw + 576) = 1;                            /* unifiedAddressing */
        *(int *)(raw + 600) = 1;                            /* managedMemory */
        *(size_t *)(raw + 720) = 65536;                     /* reservedSharedMemPerBlock — real CUDA 12 offset 720 */
        *(int *)(raw + 712) = 1024;                         /* maxBlocksPerMultiProcessor — real CUDA 12 offset 0x2c8=712 */

        /* Safety clamps: mbtopk::launch divides by (multiProcessorCount *
         * maxThreadsPerMultiProcessor). If either is 0 → integer SIGFPE.
         * Clamp to safe minimums so PyTorch can compute a valid thread count.
         * Log any zero so we can fix the root cause. */
        int *sms_ptr    = (int *)(raw + 388);
        int *mptmp_ptr  = (int *)(raw + 624);
        if (*sms_ptr == 0) {
            fprintf(stderr, "[cudart-shim] WARNING: multiProcessorCount=0 from daemon — clamping to 1\n");
            *sms_ptr = 1;
        }
        if (*mptmp_ptr == 0) {
            /* Derive from maxThreadsPerBlock (offset 320) as a conservative fallback */
            int max_tpb = *(int *)(raw + 320);
            int fallback = (max_tpb > 0) ? max_tpb : 1024;
            fprintf(stderr, "[cudart-shim] WARNING: maxThreadsPerMultiProcessor=0 from daemon — clamping to %d\n", fallback);
            *mptmp_ptr = fallback;
        }

        SHIM_LOG("DevProps: warpSize=%d maxThreadsPerBlock=%d numSMs=%d maxThreadsPerMP=%d major=%d minor=%d",
                 *(int *)(raw + 308), *(int *)(raw + 320),
                 *(int *)(raw + 388), *(int *)(raw + 624),
                 *(int *)(raw + 360), *(int *)(raw + 364));
    }
    return err;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device)
    __attribute__((alias("cudaGetDeviceProperties")));

cudaError_t cudaSetDevice(int device)
{
    GpuSetDeviceRequest req = { .device = device };
    return rpc_call(GPU_CMD_SET_DEVICE, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaGetDevice(int *device)
{
    if (device) *device = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device)
{
    (void)device;
    if (!value) return cudaErrorInvalidValue;
    /* Forward to GetDeviceProperties — daemon has real values */
    GpuGetDevicePropertiesRequest req = { .device = device };
    GpuDeviceProperties resp;
    int err = rpc_call(GPU_CMD_GET_DEVICE_PROPERTIES,
                       &req, sizeof(req), &resp, sizeof(resp), NULL);
    if (err != cudaSuccess) { *value = 0; return err; }

    switch (attr) {
        /* --- Thread/block dimensions --- */
        case  1: *value = resp.max_threads_dim[0]; break;       /* cudaDevAttrMaxBlockDimX */
        case  2: *value = resp.max_threads_dim[1]; break;       /* cudaDevAttrMaxBlockDimY */
        case  3: *value = resp.max_threads_dim[2]; break;       /* cudaDevAttrMaxBlockDimZ */
        case  4: *value = resp.max_grid_size[0]; break;         /* cudaDevAttrMaxGridDimX */
        case  5: *value = resp.max_grid_size[1]; break;         /* cudaDevAttrMaxGridDimY */
        case  6: *value = resp.max_grid_size[2]; break;         /* cudaDevAttrMaxGridDimZ */

        /* --- Shared memory (CRITICAL for MMQ kernel selection) --- */
        case  8: *value = (int)resp.shared_mem_per_block; break; /* cudaDevAttrMaxSharedMemoryPerBlock */

        /* --- Core device properties --- */
        case 10: *value = resp.warp_size; break;                 /* cudaDevAttrWarpSize */
        case 13: *value = resp.clock_rate; break;                /* cudaDevAttrClockRate */
        case 16: *value = resp.multi_processor_count; break;     /* cudaDevAttrMultiProcessorCount */
        case 21: *value = 65536; break;                          /* cudaDevAttrTotalConstantMemory (64KB typical) */
        case 24: *value = resp.max_threads_per_block; break;     /* cudaDevAttrMaxThreadsPerBlock */
        case 36: *value = resp.memory_clock_rate; break;         /* cudaDevAttrMemoryClockRate */
        case 37: *value = resp.memory_bus_width; break;          /* cudaDevAttrMemoryBusWidth */
        case 38: *value = resp.l2_cache_size; break;             /* cudaDevAttrL2CacheSize */
        case 39: *value = resp.max_threads_per_multiprocessor; break; /* cudaDevAttrMaxThreadsPerMultiProcessor */

        /* --- Registers --- */
        case 12: *value = resp.regs_per_block; break;            /* cudaDevAttrMaxRegistersPerBlock */

        /* --- Compute capability --- */
        case 75: *value = resp.major; break;                     /* cudaDevAttrComputeCapabilityMajor */
        case 76: *value = resp.minor; break;                     /* cudaDevAttrComputeCapabilityMinor */

        /* --- Shared memory attributes NOT in proto (derived from compute cap) ---
         *
         * These are critical for MMQ kernel selection. Values are per-architecture
         * constants from NVIDIA documentation. Without these, the MMQ kernel
         * selector sees 0 shared memory → mmq_x_best=0 → SIGABRT.
         *
         * sm_89 (Ada Lovelace, RTX 4060/4070/4080/4090):
         *   MaxSharedPerBlock         = 49152  (48 KB)
         *   MaxSharedPerBlockOptin    = 101376 (99 KB, with dynamic shmem)
         *   MaxSharedPerMultiprocessor = 102400 (100 KB)
         *
         * sm_86 (Ampere, RTX 3060/3070/3080/3090):
         *   MaxSharedPerBlock         = 49152  (48 KB)
         *   MaxSharedPerBlockOptin    = 101376 (99 KB)
         *   MaxSharedPerMultiprocessor = 102400 (100 KB)
         *
         * sm_80 (A100):
         *   MaxSharedPerBlock         = 49152  (48 KB)
         *   MaxSharedPerBlockOptin    = 167936 (164 KB)
         *   MaxSharedPerMultiprocessor = 167936 (164 KB)
         *
         * sm_75 (Turing, RTX 2060/2070/2080):
         *   MaxSharedPerBlock         = 49152  (48 KB)
         *   MaxSharedPerBlockOptin    = 65536  (64 KB)
         *   MaxSharedPerMultiprocessor = 65536  (64 KB)
         *
         * sm_70 (Volta, V100):
         *   MaxSharedPerBlock         = 49152  (48 KB)
         *   MaxSharedPerBlockOptin    = 98304  (96 KB)
         *   MaxSharedPerMultiprocessor = 98304  (96 KB)
         */
        case 81: /* cudaDevAttrMaxSharedMemoryPerMultiprocessor */
            if      (resp.major == 8 && resp.minor == 0) *value = 167936;
            else if (resp.major >= 8)                    *value = 102400;
            else if (resp.major == 7 && resp.minor >= 5) *value = 65536;
            else if (resp.major == 7)                    *value = 98304;
            else                                         *value = 49152;
            break;

        case 97: /* cudaDevAttrMaxSharedMemoryPerBlockOptin */
            if      (resp.major == 8 && resp.minor == 0) *value = 167936;
            else if (resp.major >= 8)                    *value = 101376;
            else if (resp.major == 7 && resp.minor >= 5) *value = 65536;
            else if (resp.major == 7)                    *value = 98304;
            else                                         *value = 49152;
            break;

        /* --- Memory management capabilities --- */
        case 83: *value = 1; break;   /* cudaDevAttrComputePreemptionSupported */
        case 84: *value = 1; break;   /* cudaDevAttrCanUseHostPointerForRegisteredMem */
        case 86: *value = 1; break;   /* cudaDevAttrManagedMemory */
        case 89: *value = 1; break;   /* cudaDevAttrConcurrentManagedAccess */

        case 100: /* cudaDevAttrVirtualMemoryManagementSupported */
            /*
             * Return 0 to force ggml-cuda to use regular cudaMalloc instead of
             * the VMM path. VMM operations cannot be proxied over the network.
             */
            *value = 0; break;

        case 101: *value = 0; break;  /* cudaDevAttrHandleTypePosixFileDescriptorSupported */
        case 102: *value = 0; break;  /* cudaDevAttrHandleTypeWin32HandleSupported */
        case 103: *value = 0; break;  /* cudaDevAttrHandleTypeWin32KMTHandleSupported */

        default: *value = 0; break;
    }
    return cudaSuccess;
}

/* ================================================================
 * Memory Management — forwarded to daemon (real GPU memory)
 * ================================================================ */

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    if (!devPtr) return cudaErrorInvalidValue;

    __sync_fetch_and_add(&g_diag_mallocs, 1);
    GpuMallocRequest req = { .size = (uint64_t)size };
    GpuMallocResponse resp;
    int err = rpc_call(GPU_CMD_MALLOC, &req, sizeof(req),
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *devPtr = (void *)(uintptr_t)resp.device_ptr;
        DIAG("cudaMalloc: %zu bytes → devPtr=0x%lx (total mallocs=%d)",
             size, (unsigned long)resp.device_ptr, g_diag_mallocs);
    } else {
        *devPtr = NULL;
        DIAG("cudaMalloc: %zu bytes FAILED err=%d (total mallocs=%d)",
             size, err, g_diag_mallocs);
    }
    return err;
}

cudaError_t cudaFree(void *devPtr)
{
    if (!devPtr) return cudaSuccess;
    GpuFreeRequest req = { .device_ptr = (uint64_t)(uintptr_t)devPtr };
    return rpc_call(GPU_CMD_FREE, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind_t kind)
{
    switch (kind) {
    case cudaMemcpyHostToDevice: {
        /*
         * Chunk large H2D transfers to stay under GPU_PROXY_MAX_PAYLOAD (64 MB).
         * Each RPC carries sizeof(GpuMemcpyRequest) + chunk_bytes as payload.
         * Use 32 MB chunks for comfortable headroom.
         */
        const size_t MAX_CHUNK = 32UL * 1024 * 1024;
        size_t offset = 0;
        while (offset < count) {
            size_t chunk = count - offset;
            if (chunk > MAX_CHUNK) chunk = MAX_CHUNK;

            GpuMemcpyRequest req = {
                .dst   = (uint64_t)((uintptr_t)dst + offset),
                .src   = 0,
                .count = (uint64_t)chunk,
                .kind  = GPU_MEMCPY_HOST_TO_DEVICE,
            };

            size_t total = sizeof(req) + chunk;
            void *buf = malloc(total);
            if (!buf) return cudaErrorInvalidValue;
            memcpy(buf, &req, sizeof(req));
            memcpy((char *)buf + sizeof(req), (const char *)src + offset, chunk);
            int err = rpc_call(GPU_CMD_MEMCPY, buf, (uint32_t)total,
                               NULL, 0, NULL);
            free(buf);
            if (err != cudaSuccess) return err;
            offset += chunk;
        }
        return cudaSuccess;
    }

    case cudaMemcpyDeviceToHost: {
        /*
         * Chunk large D2H transfers similarly — each response payload
         * must stay under GPU_PROXY_MAX_PAYLOAD.
         */
        const size_t MAX_CHUNK = 32UL * 1024 * 1024;
        size_t offset = 0;
        while (offset < count) {
            size_t chunk = count - offset;
            if (chunk > MAX_CHUNK) chunk = MAX_CHUNK;

            GpuMemcpyRequest req = {
                .dst   = 0,
                .src   = (uint64_t)((uintptr_t)src + offset),
                .count = (uint64_t)chunk,
                .kind  = GPU_MEMCPY_DEVICE_TO_HOST,
            };
            uint32_t actual = 0;
            int err = rpc_call(GPU_CMD_MEMCPY, &req, sizeof(req),
                               (char *)dst + offset, (uint32_t)chunk, &actual);
            if (err != cudaSuccess) return err;
            offset += chunk;
        }
        mask_fpe_exceptions();
        return cudaSuccess;
    }

    case cudaMemcpyDeviceToDevice: {
        GpuMemcpyRequest req = {
            .dst   = (uint64_t)(uintptr_t)dst,
            .src   = (uint64_t)(uintptr_t)src,
            .count = (uint64_t)count,
            .kind  = GPU_MEMCPY_DEVICE_TO_DEVICE,
        };

        /* During graph capture, record D2D copies — do NOT execute.
         * The source buffer may be a kernel output that hasn't been
         * computed yet (kernel only recorded).  Eager execution reads
         * stale data; recording ensures the copy runs during replay
         * after the source kernel has actually written its output. */
        if (g_graph_capturing) {
            decloud_graph_record_op(GPU_CMD_MEMCPY, &req, sizeof(req));
            return cudaSuccess;
        }

        return rpc_call(GPU_CMD_MEMCPY, &req, sizeof(req), NULL, 0, NULL);
    }

    case cudaMemcpyHostToHost:
        memcpy(dst, src, count);
        return cudaSuccess;

    default:
        return cudaErrorInvalidValue;
    }
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind_t kind, cudaStream_t stream)
{
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    GpuMemsetRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)devPtr,
        .value = value,
        .count = (uint64_t)count,
    };

    /* During graph capture, record — do NOT execute.
     * cudaMemsetAsync is captured as a memset node in real CUDA graphs.
     * Eager execution here followed by no replay causes buffers to hold
     * stale data on subsequent graph launches (e.g., flash-attention
     * accumulators not re-zeroed → corrupted softmax). */
    if (g_graph_capturing) {
        decloud_graph_record_op(GPU_CMD_MEMSET, &req, sizeof(req));
        return cudaSuccess;
    }

    return rpc_call(GPU_CMD_MEMSET, &req, sizeof(req), NULL, 0, NULL);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream)
{
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
    if (!free || !total) return cudaErrorInvalidValue;

    GpuMemInfoResponse resp;
    int err = rpc_call(GPU_CMD_MEM_GET_INFO, NULL, 0,
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *free  = (size_t)resp.free;
        *total = (size_t)resp.total;
    } else {
        *free = 0;
        *total = 0;
    }
    return (cudaError_t)err;
}

/* ================================================================
 * Synchronization
 * ================================================================ */

cudaError_t cudaDeviceSynchronize(void)
{
    cudaError_t err = rpc_call(GPU_CMD_DEVICE_SYNCHRONIZE, NULL, 0, NULL, 0, NULL);
    mask_fpe_exceptions();
    return err;
}

/* ================================================================
 * Kernel Launch
 * ================================================================ */

static int ensure_module_uploaded(int mod_idx)
{
    if (mod_idx < 0 || mod_idx >= g_module_count) return -1;
    DeferredModule *m = &g_modules[mod_idx];
    if (m->uploaded) return 0;
    ensure_connected();
    if (g_conn_fd < 0) return -1;

    /*
     * STREAMING MODULE UPLOAD — writes fatbin directly from mmap'd
     * memory instead of malloc+memcpy (which OOMs for 1.5GB+ fatbins).
     * Wire format: [header][GpuRegisterModuleRequest][fatbin bytes]
     */
    pthread_mutex_lock(&g_conn_lock);

    uint32_t payload_len = (uint32_t)(sizeof(GpuRegisterModuleRequest) + m->fatbin_size);
    SHIM_LOG("streaming module %d: %.1f MB fatbin",
            mod_idx, (double)m->fatbin_size / (1024.0 * 1024.0));

    GpuProxyHeader hdr = {
        .magic = GPU_PROXY_MAGIC, .version = GPU_PROXY_VERSION,
        .cmd = GPU_CMD_REGISTER_MODULE, .flags = 0,
        .payload_len = payload_len, .status = 0,
    };
    if (write_exact(g_conn_fd, &hdr, sizeof(hdr)) < 0) goto mod_err;

    GpuRegisterModuleRequest mreq_s = { .fatbin_size = (uint64_t)m->fatbin_size };
    if (write_exact(g_conn_fd, &mreq_s, sizeof(mreq_s)) < 0) goto mod_err;

    /* Stream fatbin directly from mmap — zero copy, zero malloc */
    if (write_exact(g_conn_fd, m->fatbin_data, m->fatbin_size) < 0) goto mod_err;

    /* Read response */
    GpuProxyHeader rhdr;
    if (read_exact(g_conn_fd, &rhdr, sizeof(rhdr)) < 0) goto mod_err;
    GpuRegisterModuleResponse mresp;
    memset(&mresp, 0, sizeof(mresp));
    if (rhdr.payload_len > 0 && rhdr.payload_len <= sizeof(mresp)) {
        if (read_exact(g_conn_fd, &mresp, rhdr.payload_len) < 0) goto mod_err;
    } else if (rhdr.payload_len > sizeof(mresp)) {
        char drain[256]; uint32_t rem = rhdr.payload_len;
        while (rem > 0) { uint32_t c = rem < 256 ? rem : 256;
            if (read_exact(g_conn_fd, drain, c) < 0) goto mod_err; rem -= c; }
    }
    int err = rhdr.status;
    pthread_mutex_unlock(&g_conn_lock);
    if (0) { mod_err:
        close(g_conn_fd); g_conn_fd = -1; g_initialized = 0;
        pthread_mutex_unlock(&g_conn_lock);
        SHIM_LOG("streaming module %d: connection lost", mod_idx);
        return -1;
    }
    if (err != 0) { SHIM_LOG("lazy upload: module %d failed", mod_idx); return -1; }
    m->remote_handle = mresp.module_handle;
    m->uploaded = 1;
    for (int i = 0; i < g_function_count; i++) {
        RegisteredFunction *rf = &g_functions[i];
        if (rf->module_index != mod_idx || rf->registered) continue;
        uint32_t name_len = (uint32_t)strlen(rf->device_name) + 1;
        uint32_t freq_len = sizeof(GpuRegisterFunctionRequest) + name_len;
        void *freq_buf = malloc(freq_len);
        if (!freq_buf) continue;
        GpuRegisterFunctionRequest *freq = (GpuRegisterFunctionRequest *)freq_buf;
        freq->module_handle = m->remote_handle;
        freq->host_func_ptr = rf->host_func_ptr;
        freq->device_name_len = name_len;
        memcpy((uint8_t *)freq_buf + sizeof(GpuRegisterFunctionRequest), rf->device_name, name_len);
        GpuRegisterFunctionResponse fresp;
        memset(&fresp, 0, sizeof(fresp));
        err = rpc_call(GPU_CMD_REGISTER_FUNCTION, freq_buf, freq_len, &fresp, sizeof(fresp), NULL);
        free(freq_buf);
        if (err == 0) {
            rf->num_params = fresp.num_params;
            if (rf->num_params > GPU_MAX_KERNEL_PARAMS) rf->num_params = GPU_MAX_KERNEL_PARAMS;
            for (uint32_t p = 0; p < rf->num_params; p++) rf->param_sizes[p] = fresp.param_sizes[p];
            rf->registered = 1;
        }
    }
    return 0;
}

static RegisteredFunction *find_registered_function(uint64_t host_func_ptr)
{
    for (int i = 0; i < g_function_count; i++) {
        if (g_functions[i].host_func_ptr == host_func_ptr)
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
    if (!rf->registered) {
        ensure_module_uploaded(rf->module_index);
        if (!rf->registered) {
            SHIM_LOG("cudaLaunchKernel: lazy reg failed for %s", rf->device_name);
            return cudaErrorInvalidValue;
        }
    }

    uint32_t args_total = 0;
    for (uint32_t i = 0; i < rf->num_params; i++) {
        args_total += rf->param_sizes[i];
    }

    if (rf->num_params == 0) {
        SHIM_LOG("cudaLaunchKernel: 0 params — can't serialize");
        return cudaErrorInvalidValue;
    }

    uint32_t req_len = sizeof(GpuLaunchKernelRequest) + args_total;
    void *req_buf = malloc(req_len);
    if (!req_buf) return cudaErrorMemoryAllocation;

    GpuLaunchKernelRequest *req = (GpuLaunchKernelRequest *)req_buf;
    req->host_func_ptr   = (uint64_t)(uintptr_t)func;
    req->grid_dim_x      = gridDim.x;
    req->grid_dim_y      = gridDim.y;
    req->grid_dim_z      = gridDim.z;
    req->block_dim_x     = blockDim.x;
    req->block_dim_y     = blockDim.y;
    req->block_dim_z     = blockDim.z;
    req->shared_mem_bytes = (uint64_t)sharedMem;
    req->stream_handle   = (uint64_t)(uintptr_t)stream;
    req->num_params      = rf->num_params;
    req->args_total_size = args_total;

    /* Serialize argument values */
    uint8_t *dest = (uint8_t *)req_buf + sizeof(GpuLaunchKernelRequest);
    for (uint32_t i = 0; i < rf->num_params; i++) {
        uint32_t sz = rf->param_sizes[i];
        if (args && args[i]) {
            memcpy(dest, args[i], sz);
        } else {
            memset(dest, 0, sz);
        }
        dest += sz;
    }

    /* Record this launch if we're inside a graph capture.
     * We save a copy of the serialized payload so cudaGraphLaunch
     * can replay it on subsequent tokens. */
    if (g_graph_capturing && g_graph_capture_buf.count < GRAPH_MAX_RECORDED_OPS) {
        GraphRecord *cap = &g_graph_capture_buf;
        if (cap->count >= cap->capacity) {
            int new_cap = cap->capacity ? cap->capacity * 2 : 256;
            if (new_cap > GRAPH_MAX_RECORDED_OPS) new_cap = GRAPH_MAX_RECORDED_OPS;
            cap->ops = (GraphRecordedOp *)realloc(cap->ops,
                        new_cap * sizeof(GraphRecordedOp));
            cap->capacity = new_cap;
        }
        GraphRecordedOp *op = &cap->ops[cap->count++];
        op->rpc_cmd     = GPU_CMD_LAUNCH_KERNEL;
        op->rpc_len     = req_len;
        op->rpc_payload = malloc(req_len);
        memcpy(op->rpc_payload, req_buf, req_len);
        DIAG("  graph-capture: recorded op #%d (func=0x%lx, grid=%ux%ux%u, block=%ux%ux%u, args=%u bytes)",
             cap->count, (unsigned long)req->host_func_ptr,
             req->grid_dim_x, req->grid_dim_y, req->grid_dim_z,
             req->block_dim_x, req->block_dim_y, req->block_dim_z,
             req->args_total_size);
    }

    /* During graph capture, only record — do NOT execute.
     * Real CUDA semantics: kernels are not executed during capture,
     * only during cudaGraphLaunch.  Eager execution here caused
     * double-execution → corrupted atomics/in-place ops → gibberish. */
    if (g_graph_capturing) {
        free(req_buf);
        return cudaSuccess;
    }

    __sync_fetch_and_add(&g_diag_kernel_launches, 1);
    int err = rpc_call(GPU_CMD_LAUNCH_KERNEL, req_buf, req_len,
                       NULL, 0, NULL);
    if (err != 0) {
        __sync_fetch_and_add(&g_diag_rpc_errors, 1);
        DIAG("cudaLaunchKernel FAILED: err=%d (func=0x%lx)", err, (unsigned long)(uintptr_t)func);
    }
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
    cudaError_t err = rpc_call(GPU_CMD_STREAM_SYNCHRONIZE, &req, sizeof(req), NULL, 0, NULL);
    mask_fpe_exceptions();
    return err;
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
        .event_handle  = (uint64_t)(uintptr_t)event,
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
        .start_handle = (uint64_t)(uintptr_t)start,
        .end_handle   = (uint64_t)(uintptr_t)end,
    };
    GpuEventElapsedTimeResponse resp;
    int err = rpc_call(GPU_CMD_EVENT_ELAPSED_TIME, &req, sizeof(req),
                       &resp, sizeof(resp), NULL);
    if (err == cudaSuccess) {
        *ms = resp.milliseconds;
    } else {
        *ms = 0.0f;
    }
    return err;
}

/* ================================================================
 * Misc stubs — return success for functions ggml-cuda probes
 * ================================================================ */

cudaError_t cudaRuntimeGetVersion(int *version)
{
    if (version) *version = 12080;  /* CUDA 12.8 */
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *version)
{
    if (version) *version = 12080;
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaPeekAtLastError(void) { return cudaSuccess; }

const char *cudaGetErrorString(cudaError_t error)
{
    switch (error) {
        case 0:   return "no error";
        case 1:   return "invalid argument";
        case 2:   return "out of memory";
        case 100: return "no CUDA-capable device";
        default:  return "unknown error";
    }
}

const char *cudaGetErrorName(cudaError_t error)
{
    switch (error) {
        case 0:   return "cudaSuccess";
        case 1:   return "cudaErrorInvalidValue";
        case 2:   return "cudaErrorMemoryAllocation";
        case 100: return "cudaErrorNoDevice";
        default:  return "cudaErrorUnknown";
    }
}

cudaError_t cudaDeviceReset(void) { return cudaSuccess; }
cudaError_t cudaSetDeviceFlags(unsigned int flags) { (void)flags; return cudaSuccess; }
cudaError_t cudaDeviceSetCacheConfig(int config) { (void)config; return cudaSuccess; }

cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value)
{
    (void)func; (void)attr; (void)value;
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                                const void *func,
                                                size_t dynamicSMemSize,
                                                int blockSizeLimit)
{
    (void)func; (void)dynamicSMemSize; (void)blockSizeLimit;
    if (minGridSize) *minGridSize = 1;
    if (blockSize) *blockSize = 256;
    return cudaSuccess;
}

/* Host memory stubs */
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    (void)flags;
    if (!pHost) return cudaErrorInvalidValue;
    /* ggml requires TENSOR_ALIGNMENT (typically 32 or 64 bytes) */
    if (posix_memalign(pHost, 4096, size) != 0) {
        *pHost = NULL;
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    return cudaHostAlloc(ptr, size, 0);
}

cudaError_t cudaFreeHost(void *ptr)
{
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
    (void)ptr; (void)size; (void)flags;
    return cudaSuccess;
}

cudaError_t cudaHostUnregister(void *ptr)
{
    (void)ptr;
    return cudaSuccess;
}

/* ================================================================
 * Exported RPC helper for cublas_stub.so
 *
 * The stub has correct @@libcublas.so.12 version tags but no
 * transport. It finds us via dlsym(RTLD_DEFAULT) at runtime.
 * ================================================================ */
int decloud_rpc_call(uint8_t cmd, const void *req, uint32_t req_len,
                     void *resp, uint32_t resp_size, uint32_t *resp_len)
{
    return rpc_call(cmd, req, req_len, resp, resp_size, resp_len);
}

/* ================================================================
 * cuBLAS stub functions
 *
 * ggml unconditionally creates cuBLAS handles during CUDA backend init,
 * even when GGML_CUDA_FORCE_MMQ=1. Without these stubs, cublasCreate_v2()
 * calls into the real cuBLAS which needs cuGetExportTable → crash.
 *
 * With MMQ mode active, cuBLAS compute functions (gemm, etc.) are never
 * called by ggml, so dummy handles are safe. For PyTorch workloads,
 * compute functions are intentionally absent here — they resolve to
 * libcublas_stub.so which proxies them via RPC to the host daemon.
 * Defining them here would shadow libcublas_stub.so due to LD_PRELOAD
 * symbol priority and break PyTorch inference.
 * ================================================================ */

typedef void *cublasHandle_t;
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS          0
#define CUBLAS_STATUS_NOT_INITIALIZED  1
#define CUBLAS_STATUS_NOT_SUPPORTED   15

static int g_cublas_dummy_handle = 0xDEC10BD;  /* DECloud cuBLAS Dummy */

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
{
    SHIM_LOG("cublasCreate_v2 → stub (dummy handle)");
    if (handle) *handle = &g_cublas_dummy_handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
{
    (void)handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream)
{
    (void)handle; (void)stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode)
{
    (void)handle; (void)mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, int *mode)
{
    (void)handle;
    if (mode) *mode = 0;  /* CUBLAS_DEFAULT_MATH */
    return CUBLAS_STATUS_SUCCESS;
}

/* ================================================================
 * Cleanup
 * ================================================================ */

__attribute__((destructor))
static void shim_cleanup(void)
{
    DIAG("=== CUDART SHIM DESTRUCTOR (pid=%d) ===", getpid());
    DIAG("  Kernel launches:       %d", g_diag_kernel_launches);
    DIAG("  Graph begins:          %d", g_diag_graph_begins);
    DIAG("  Graph ends:            %d", g_diag_graph_ends);
    DIAG("  Graph instantiates:    %d", g_diag_graph_instantiates);
    DIAG("  Graph launches:        %d", g_diag_graph_launches);
    DIAG("  Graph updates:         %d", g_diag_graph_updates);
    DIAG("  cudaMalloc calls:      %d", g_diag_mallocs);
    DIAG("  RPC errors:            %d", g_diag_rpc_errors);
    DIAG("  g_graph_noop:          %d", g_graph_noop);
    DIAG("=== END STATS ===");

    pthread_mutex_lock(&g_conn_lock);
    if (g_conn_fd >= 0) {
        GpuProxyHeader hdr = {
            .magic = GPU_PROXY_MAGIC,
            .version = GPU_PROXY_VERSION,
            .cmd = GPU_CMD_GOODBYE,
            .payload_len = 0,
            .status = 0,
        };
        write_exact(g_conn_fd, &hdr, sizeof(hdr));
        close(g_conn_fd);
        g_conn_fd = -1;
    }
    pthread_mutex_unlock(&g_conn_lock);

    if (g_diag_fp) {
        fclose(g_diag_fp);
        g_diag_fp = NULL;
    }
}
/* ================================================================
 * Additional stubs required by libggml-cuda.so
 * ================================================================ */

/* Peer access */
cudaError_t cudaDeviceCanAccessPeer(int *canAccess, int device, int peerDevice)
{
    (void)device; (void)peerDevice;
    if (canAccess) *canAccess = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    (void)peerDevice; (void)flags;
    return cudaSuccess;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
{
    (void)peerDevice;
    return cudaSuccess;
}

/* Function attributes */
typedef struct {
    int binaryVersion;
    size_t cacheModeCA;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxDynamicSharedSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int preferredShmemCarveout;
    size_t ptxVersion;
    size_t sharedSizeBytes;
} cudaFuncAttributes_t;

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes_t *attr, const void *func)
{
    if (!attr) return cudaErrorInvalidValue;

    /* Find registered function by host-side pointer */
    RegisteredFunction *rf = find_registered_function((uint64_t)(uintptr_t)func);
    if (!rf) {
        SHIM_LOG("cudaFuncGetAttributes: unknown function %p", func);
        return cudaErrorInvalidDeviceFunction;
    }

    /* EAGER upload: trigger module upload NOW, not on first launch.
     * This is the key fix — ggml queries attributes BEFORE any launch. */
    if (!rf->registered) {
        ensure_module_uploaded(rf->module_index);
        if (!rf->registered) {
            SHIM_LOG("cudaFuncGetAttributes: eager upload failed for %s",
                     rf->device_name);
            return cudaErrorInvalidDeviceFunction;
        }
    }

    /* RPC to daemon for REAL kernel attributes from the physical GPU */
    GpuFuncGetAttributesRequest req = {
        .host_func_ptr = (uint64_t)(uintptr_t)func,
    };
    GpuFuncGetAttributesResponse resp;
    memset(&resp, 0, sizeof(resp));

    int err = rpc_call(GPU_CMD_FUNC_GET_ATTRIBUTES,
                       &req, sizeof(req), &resp, sizeof(resp), NULL);
    if (err != 0) {
        SHIM_LOG("cudaFuncGetAttributes: RPC failed for %s (err=%d)",
                 rf->device_name, err);
        return (cudaError_t)err;
    }

    /* Populate caller's struct with real values from daemon */
    memset(attr, 0, sizeof(*attr));
    attr->binaryVersion          = resp.binaryVersion;
    attr->maxThreadsPerBlock     = resp.maxThreadsPerBlock;
    attr->numRegs                = resp.numRegs;
    attr->sharedSizeBytes        = resp.sharedSizeBytes;
    attr->constSizeBytes         = resp.constSizeBytes;
    attr->localSizeBytes         = resp.localSizeBytes;
    attr->maxDynamicSharedSizeBytes = resp.maxDynamicSharedSizeBytes;
    attr->preferredShmemCarveout = resp.preferredShmemCarveout;
    attr->ptxVersion             = resp.ptxVersion;

    SHIM_LOG("cudaFuncGetAttributes(%s): binary=%d maxThreads=%d regs=%d shared=%d",
             rf->device_name, resp.binaryVersion,
             resp.maxThreadsPerBlock, resp.numRegs, resp.sharedSizeBytes);
    return cudaSuccess;
}

/* ================================================================
 * CUDA Graph Capture & Replay
 *
 * ggml with USE_GRAPHS=1 launches kernels ONLY during graph capture,
 * then replays via cudaGraphLaunch for subsequent tokens.
 *
 * During capture, the stream is NOT truly in capture mode (our
 * cudaStreamBeginCapture is lightweight), so kernels execute eagerly.
 * We also record each launch.  On cudaGraphLaunch we replay the
 * recorded launches.  The first token thus gets a harmless double-
 * execution; subsequent tokens get exactly one execution (the replay).
 *
 * ggml wraps graph calls in CUDA_CHECK() which aborts on ANY non-zero
 * return, so every graph API must return cudaSuccess.
 * ================================================================ */
typedef void *cudaGraph_t;
typedef void *cudaGraphExec_t;
typedef int cudaGraphExecUpdateResult;

/* The "graph" from cudaStreamEndCapture is just a pointer to the
 * thread-local capture buffer.  It is consumed by cudaGraphInstantiate
 * or cudaGraphExecUpdate before the next capture begins. */
static int g_dummy_graph_tag = 0xDEC10001;

cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    /* Nothing to free — the capture buffer is owned by g_graph_capture_buf
     * and freed on next capture or at process exit. */
    (void)graph;
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
    if (!g_graph_noop) return cudaErrorNotSupported;
    int idx = GRAPH_EXEC_FROM_PTR(graphExec);
    graph_exec_free(idx);
    return cudaSuccess;
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                 cudaGraphExecUpdateResult *updateResult_out)
{
    __sync_fetch_and_add(&g_diag_graph_updates, 1);
    if (!g_graph_noop) {
        if (updateResult_out) *updateResult_out = 1; /* error */
        return cudaSuccess;
    }
    (void)hGraph;
    /* Replace the exec's recorded ops with the latest capture buffer */
    int idx = GRAPH_EXEC_FROM_PTR(hGraphExec);
    if (idx >= 0 && idx < GRAPH_MAX_EXEC_SLOTS && g_graph_exec_slots[idx].in_use) {
        pthread_mutex_lock(&g_graph_exec_lock);
        graph_record_free(&g_graph_exec_slots[idx].record);
        graph_record_copy(&g_graph_exec_slots[idx].record, &g_graph_capture_buf);
        pthread_mutex_unlock(&g_graph_exec_lock);
        if (updateResult_out) *updateResult_out = 0; /* success */
        DIAG("cudaGraphExecUpdate: slot %d updated with %d ops (call #%d)",
             idx, g_graph_exec_slots[idx].record.count, g_diag_graph_updates);
    } else {
        if (updateResult_out) *updateResult_out = 1; /* error → forces re-instantiate */
        DIAG("cudaGraphExecUpdate: slot %d INVALID → error result (call #%d)",
             idx, g_diag_graph_updates);
    }
    return cudaSuccess;
}

/* Shared implementation for all cudaGraphInstantiate variants */
static cudaError_t graph_instantiate_impl(cudaGraphExec_t *pGraphExec, const char *variant)
{
    if (!g_graph_noop) return cudaErrorNotSupported;

    __sync_fetch_and_add(&g_diag_graph_instantiates, 1);
    int idx = graph_exec_alloc();
    if (idx < 0) {
        /* All slots full — reuse slot 0 */
        DIAG("%s: all %d slots full, reusing slot 0", variant, GRAPH_MAX_EXEC_SLOTS);
        graph_exec_free(0);
        idx = graph_exec_alloc();
    }
    if (idx >= 0) {
        pthread_mutex_lock(&g_graph_exec_lock);
        graph_record_copy(&g_graph_exec_slots[idx].record, &g_graph_capture_buf);
        pthread_mutex_unlock(&g_graph_exec_lock);
    }
    if (pGraphExec) *pGraphExec = GRAPH_EXEC_TO_PTR(idx >= 0 ? idx : 0);
    DIAG("%s: slot=%d, ops=%d (call #%d)",
         variant, idx, idx >= 0 ? g_graph_exec_slots[idx].record.count : 0,
         g_diag_graph_instantiates);
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                  void *pErrorNode, char *pLogBuffer, size_t bufferSize)
{
    (void)graph; (void)pErrorNode; (void)pLogBuffer; (void)bufferSize;
    return graph_instantiate_impl(pGraphExec, "cudaGraphInstantiate");
}

/* CUDA 11.4+ variant used by newer ggml/llama.cpp instead of cudaGraphInstantiate.
 * Previously only in the PyTorch stubs (which returned a NULL handle → graph
 * launch became a no-op → 2000+ kernels never executed → garbled output). */
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec,
                                           cudaGraph_t graph,
                                           unsigned long long flags)
{
    (void)graph; (void)flags;
    return graph_instantiate_impl(pGraphExec, "cudaGraphInstantiateWithFlags");
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    (void)stream;
    if (!g_graph_noop) return cudaErrorNotSupported;

    __sync_fetch_and_add(&g_diag_graph_launches, 1);
    int idx = GRAPH_EXEC_FROM_PTR(graphExec);
    if (idx < 0 || idx >= GRAPH_MAX_EXEC_SLOTS || !g_graph_exec_slots[idx].in_use) {
        DIAG("cudaGraphLaunch: UNKNOWN handle %p (idx=%d) → no-op! (call #%d)",
             graphExec, idx, g_diag_graph_launches);
        return cudaSuccess; /* no-op for unknown handles */
    }

    GraphRecord *rec = &g_graph_exec_slots[idx].record;
    SHIM_LOG("cudaGraphLaunch: replaying %d recorded kernel launches", rec->count);
    DIAG("cudaGraphLaunch: replaying %d ops from slot %d (call #%d)",
         rec->count, idx, g_diag_graph_launches);

    int errors = 0;
    for (int i = 0; i < rec->count; i++) {
        /* Use the recorded RPC function if set (driver-shim ops use
         * the driver connection where their functions are registered),
         * otherwise fall back to the runtime shim's rpc_call. */
        graph_rpc_fn_t fn = rec->ops[i].rpc_fn ? rec->ops[i].rpc_fn : rpc_call;
        int err = fn(rec->ops[i].rpc_cmd,
                     rec->ops[i].rpc_payload, rec->ops[i].rpc_len,
                     NULL, 0, NULL);
        if (err != 0) errors++;
    }
    if (errors > 0) {
        DIAG("cudaGraphLaunch: %d/%d replay RPCs FAILED", errors, rec->count);
    }
    return cudaSuccess;
}

/* Managed memory */
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    (void)flags;
    if (!devPtr) return cudaErrorInvalidValue;
    /* Forward as regular device malloc */
    return cudaMalloc(devPtr, size);
}

/* 2D/3D/Peer memcpy */
typedef struct {
    void *src; size_t spitch; size_t width; size_t height;
} cudaMemcpy2DParams;

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                               size_t spitch, size_t width, size_t height,
                               cudaMemcpyKind_t kind, cudaStream_t stream)
{
    (void)dpitch; (void)spitch; (void)stream;
    return cudaMemcpy(dst, src, width * height, kind);
}

typedef struct { int dummy; } cudaMemcpy3DPeerParms;

cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms *p, cudaStream_t stream)
{
    (void)p; (void)stream;
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                 int srcDevice, size_t count, cudaStream_t stream)
{
    (void)dstDevice; (void)srcDevice; (void)stream;
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

/* Occupancy */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize,
    size_t dynamicSMemSize, unsigned int flags)
{
    if (!numBlocks) return cudaErrorInvalidValue;

    RegisteredFunction *rf = find_registered_function((uint64_t)(uintptr_t)func);
    if (!rf) {
        SHIM_LOG("occupancy: unknown function %p, returning 2", func);
        *numBlocks = 2; /* safe fallback for unknown functions */
        return cudaSuccess;
    }

    /* EAGER upload if not yet registered on daemon */
    if (!rf->registered) {
        ensure_module_uploaded(rf->module_index);
        if (!rf->registered) {
            SHIM_LOG("occupancy: eager upload failed for %s, returning 2", rf->device_name);
            *numBlocks = 2;
            return cudaSuccess;
        }
    }

    /* RPC to daemon for REAL occupancy from the physical GPU */
    GpuOccupancyMaxBlocksRequest req = {
        .host_func_ptr   = (uint64_t)(uintptr_t)func,
        .blockSize       = blockSize,
        .dynamicSMemSize = (uint64_t)dynamicSMemSize,
        .flags           = flags,
    };
    GpuOccupancyMaxBlocksResponse resp;
    memset(&resp, 0, sizeof(resp));

    int err = rpc_call(GPU_CMD_OCCUPANCY_MAX_BLOCKS,
                       &req, sizeof(req), &resp, sizeof(resp), NULL);
    if (err != 0) {
        SHIM_LOG("occupancy: RPC failed for %s (err=%d), returning 2", rf->device_name, err);
        *numBlocks = 2; /* safe fallback on RPC failure */
        return cudaSuccess;
    }

    /* Sanity: daemon should never return 0 for a valid kernel */
    if (resp.numBlocks <= 0) {
        SHIM_LOG("occupancy: daemon returned %d for %s (blockSize=%d), clamping to 2",
                 resp.numBlocks, rf->device_name, blockSize);
        *numBlocks = 2;
        return cudaSuccess;
    }

    *numBlocks = resp.numBlocks;
    return cudaSuccess;
}

/* Non-Flags variant that delegates to the Flags version */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const void *func, int blockSize,
    size_t dynamicSMemSize)
{
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, func, blockSize, dynamicSMemSize, 0);
}

/* Stream capture */
typedef enum {
    cudaStreamCaptureModeGlobal = 0,
} cudaStreamCaptureMode_t;

typedef enum {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
} cudaStreamCaptureStatus_t;

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode_t mode)
{
    (void)stream; (void)mode;
    if (!g_graph_noop) return cudaErrorNotSupported;

    __sync_fetch_and_add(&g_diag_graph_begins, 1);
    /* Free any previous capture buffer and start fresh */
    graph_record_free(&g_graph_capture_buf);
    g_graph_capturing = 1;
    SHIM_LOG("cudaStreamBeginCapture: recording started");
    DIAG("cudaStreamBeginCapture: stream=%p, mode=%d, recording started (call #%d)",
         stream, (int)mode, g_diag_graph_begins);
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    (void)stream;
    if (!g_graph_noop) return cudaErrorNotSupported;

    g_graph_capturing = 0;
    __sync_fetch_and_add(&g_diag_graph_ends, 1);
    SHIM_LOG("cudaStreamEndCapture: recorded %d kernel launches",
             g_graph_capture_buf.count);
    DIAG("cudaStreamEndCapture: recorded %d kernel launches (call #%d)",
         g_graph_capture_buf.count, g_diag_graph_ends);
    /* Return a token so ggml can pass it to cudaGraphInstantiate/Update */
    if (pGraph) *pGraph = (cudaGraph_t)&g_dummy_graph_tag;
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus_t *pCaptureStatus)
{
    (void)stream;
    if (pCaptureStatus)
        *pCaptureStatus = g_graph_capturing ? cudaStreamCaptureStatusActive
                                            : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    (void)stream; (void)event; (void)flags;
    return cudaSuccess;
}

/* ================================================================
 * Internal CUDA runtime symbols for <<<>>> launch syntax
 *
 * __cudaPushCallConfiguration / __cudaPopCallConfiguration are used by
 * nvcc-compiled code for the <<<grid, block, shared, stream>>> syntax.
 * They store launch parameters that cudaLaunchKernel then uses.
 * ================================================================ */

static __thread dim3 g_push_grid;
static __thread dim3 g_push_block;
static __thread size_t g_push_shared;
static __thread cudaStream_t g_push_stream;

unsigned int __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                          size_t sharedMem,
                                          cudaStream_t stream)
{
    g_push_grid   = gridDim;
    g_push_block  = blockDim;
    g_push_shared = sharedMem;
    g_push_stream = stream;
    return 0; /* cudaSuccess */
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                        size_t *sharedMem,
                                        cudaStream_t *stream)
{
    if (gridDim)   *gridDim   = g_push_grid;
    if (blockDim)  *blockDim  = g_push_block;
    if (sharedMem) *sharedMem = g_push_shared;
    if (stream)    *stream    = g_push_stream;
    return cudaSuccess;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
    (void)fatCubinHandle;
    /* No-op — signals end of registration for a module */
}