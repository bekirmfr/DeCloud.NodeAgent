/*
 * DeCloud cuBLAS Lt Stub Library (libcublasLt.so.12)
 *
 * Full proxy implementation for PyTorch 2.3.x.
 *
 * Problem with previous NOT_SUPPORTED stubs:
 *   PyTorch's gemm_internal_cublaslt() checks returnedResults from
 *   cublasLtMatmulAlgoGetHeuristic. When count==0 it executes:
 *     TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED)
 *   which throws — no fallback, no catch. Returning count=0 or
 *   CUBLAS_STATUS_NOT_SUPPORTED from the heuristic always throws.
 *
 * Fix:
 *   - Track descriptor state (transa/transb/computeType/scaleType, per-matrix
 *     rows/cols/ld/type/stride/batchCount) — accumulated across API calls
 *   - cublasLtMatmulAlgoGetHeuristic returns count=1 so PyTorch proceeds
 *   - cublasLtMatmul is a pure serializer: packs raw stored descriptor state
 *     into GpuCublasLtMatmulRequest and sends GPU_CMD_CUBLAS_LT_MATMUL RPC.
 *     No arithmetic. No GPU memory ops. No C/D logic.
 *   - Daemon calls the real cublasLtMatmul on the host GPU, which natively
 *     supports separate C (bias/accumulator) and D (output) pointers.
 *
 * Design principle:
 *   The stub is a serializer. It accumulates API state and encodes it into
 *   wire format. All computation — GEMM execution, dimension interpretation,
 *   C≠D fused add — happens on the host via the real cuBLAS library.
 *
 * Build:
 *   gcc -shared -fPIC -I. -ldl \
 *       -Wl,-soname,libcublasLt.so.12 \
 *       -Wl,--version-script=stubs/libcublasLt.version \
 *       -o build/libcublasLt_stub.so stubs/cublasLt_stub.c
 */

#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>

#include "../proto/gpu_proxy_proto.h"

/* ----------------------------------------------------------------
 * Status codes — must match real cuBLAS ABI exactly
 * ---------------------------------------------------------------- */
#define CUBLAS_STATUS_SUCCESS          0
#define CUBLAS_STATUS_ALLOC_FAILED     3
#define CUBLAS_STATUS_NOT_SUPPORTED   15

/* Operation types */
#define CUBLAS_OP_N  0
#define CUBLAS_OP_T  1
#define CUBLAS_OP_C  2

/* Compute types */
#define CUBLAS_COMPUTE_16F  64
#define CUBLAS_COMPUTE_32F  68
#define CUBLAS_COMPUTE_64F  70

/* CUDA data types */
#define CUDA_R_32F  0
#define CUDA_R_64F  1
#define CUDA_R_16F  2

/* cublasLtMatmulDesc attribute enum values (cuBLAS 12 ABI) */
#define CUBLASLT_MATMUL_DESC_COMPUTE_TYPE  0
#define CUBLASLT_MATMUL_DESC_SCALE_TYPE    1
#define CUBLASLT_MATMUL_DESC_TRANSA        3
#define CUBLASLT_MATMUL_DESC_TRANSB        4

/* cublasLtMatrixLayout attribute enum values (cuBLAS 12 ABI) */
#define CUBLASLT_MATRIX_LAYOUT_TYPE                  0
#define CUBLASLT_MATRIX_LAYOUT_ROWS                  2
#define CUBLASLT_MATRIX_LAYOUT_COLS                  3
#define CUBLASLT_MATRIX_LAYOUT_LD                    4
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT           5
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET  6

/* Mask all FP exceptions in SSE MXCSR and x87 FCW.
 * Replaces fedisableexcept(FE_ALL_EXCEPT) — no -lm dependency. */
static inline void mask_fpe_exceptions(void)
{
    unsigned int mxcsr;
    __asm__ volatile("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 0x1F80U;  /* bits 7-12: mask all SSE FP exceptions */
    __asm__ volatile("ldmxcsr %0" : : "m"(mxcsr));

    unsigned short fcw;
    __asm__ volatile("fstcw %0" : "=m"(fcw));
    fcw |= 0x3FU;       /* bits 0-5: mask all x87 FP exceptions */
    __asm__ volatile("fldcw %0" : : "m"(fcw));
}

/* ----------------------------------------------------------------
 * Type definitions
 * ---------------------------------------------------------------- */
typedef int    cublasStatus_t;
typedef void  *cublasLtHandle_t;
typedef void  *cublasLtMatrixLayout_t;
typedef void  *cublasLtMatmulDesc_t;
typedef void  *cublasLtMatmulPreference_t;
typedef void  *cublasLtMatrixTransformDesc_t;
typedef int    cublasComputeType_t;
typedef int    cudaDataType_t;
typedef void  *cudaStream_t;

/* cublasLtMatmulAlgo_t — 64-byte opaque struct, must match real CUDA 12 ABI.
 * Defined in cublasLt.h as: typedef struct { uint64_t data[8]; } cublasLtMatmulAlgo_t;
 * Common mistake: some docs say 128 bytes — that is WRONG for CUDA 12.
 * Using the wrong size corrupts the caller's stack in AlgoGetHeuristic. */
typedef struct { uint64_t data[8]; } cublasLtMatmulAlgo_t; /* 64 bytes */

/* cublasLtMatmulHeuristicResult_t — 96 bytes total, must match real CUDA 12 ABI:
 *   algo(64) + workspaceSize(8) + state(4) + wavesCount(4) + reserved(16) */
typedef struct {
    cublasLtMatmulAlgo_t algo;    /*  64 bytes — opaque algo descriptor */
    size_t   workspaceSize;        /*   8 bytes */
    int32_t  state;                /*   4 bytes — cublasStatus_t */
    float    wavesCount;           /*   4 bytes */
    int32_t  reserved[4];          /*  16 bytes */
} cublasLtMatmulHeuristicResult_t; /* 96 bytes total */

#define STUB_LOG(fmt, ...) \
    fprintf(stderr, "[cublasLt-stub] " fmt "\n", ##__VA_ARGS__)

/* ================================================================
 * RPC bridge — resolves decloud_rpc_call from cuda_shim.so at runtime
 * ================================================================ */
typedef int (*rpc_call_fn)(uint8_t cmd, const void *req, uint32_t req_len,
                           void *resp, uint32_t resp_size, uint32_t *resp_len);

static rpc_call_fn g_rpc_call   = NULL;
static int         g_rpc_resolved = 0;

static rpc_call_fn get_rpc(void)
{
    if (!g_rpc_resolved) {
        g_rpc_call = (rpc_call_fn)dlsym(RTLD_DEFAULT, "decloud_rpc_call");
        if (!g_rpc_call)
            STUB_LOG("WARNING: decloud_rpc_call not found — cublasLt GEMM unavailable");
        g_rpc_resolved = 1;
    }
    return g_rpc_call;
}

/* ================================================================
 * Descriptor state tables
 *
 * Handles ARE pointers into these static arrays — no heap allocation,
 * no pointer chasing, no malloc failures. Max 32 concurrent matmul
 * descs and 64 matrix layouts is sufficient for PyTorch's usage pattern
 * (one gemm call creates ~4 layouts + 1 desc, destroys them immediately).
 * ================================================================ */
#define LT_MAX_DESCS    32
#define LT_MAX_LAYOUTS  64

typedef struct {
    int      in_use;
    int      transa;       /* CUBLAS_OP_N=0, T=1, C=2 */
    int      transb;
    int      computeType;  /* cublasComputeType_t */
    int      scaleType;    /* cudaDataType_t for alpha/beta */
} LtMatmulDescEntry;

typedef struct {
    int      in_use;
    int      dataType;     /* cudaDataType_t */
    uint64_t rows;
    uint64_t cols;
    int64_t  ld;
    int32_t  batchCount;
    int64_t  batchStride;
} LtMatrixLayoutEntry;

static LtMatmulDescEntry   g_matmul_descs[LT_MAX_DESCS];
static LtMatrixLayoutEntry g_matrix_layouts[LT_MAX_LAYOUTS];

static LtMatmulDescEntry *lt_alloc_matmul_desc(void)
{
    for (int i = 0; i < LT_MAX_DESCS; i++) {
        if (!g_matmul_descs[i].in_use) {
            memset(&g_matmul_descs[i], 0, sizeof(g_matmul_descs[i]));
            g_matmul_descs[i].in_use = 1;
            g_matmul_descs[i].transa = CUBLAS_OP_N;
            g_matmul_descs[i].transb = CUBLAS_OP_N;
            return &g_matmul_descs[i];
        }
    }
    return NULL;
}

static LtMatrixLayoutEntry *lt_alloc_layout(void)
{
    for (int i = 0; i < LT_MAX_LAYOUTS; i++) {
        if (!g_matrix_layouts[i].in_use) {
            memset(&g_matrix_layouts[i], 0, sizeof(g_matrix_layouts[i]));
            g_matrix_layouts[i].in_use     = 1;
            g_matrix_layouts[i].batchCount = 1;
            return &g_matrix_layouts[i];
        }
    }
    return NULL;
}

/* ================================================================
 * Handle management
 * ================================================================ */
static int g_lt_dummy_handle = 0xDEC10B17;

void __decloud_cublasLt_stub_v1(void) {}

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle)
{
    if (lightHandle) *lightHandle = &g_lt_dummy_handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle)
{
    (void)lightHandle;
    return CUBLAS_STATUS_SUCCESS;
}

const char *cublasLtGetStatusString(cublasStatus_t status)
{
    switch (status) {
    case 0:  return "CUBLAS_STATUS_SUCCESS";
    case 3:  return "CUBLAS_STATUS_ALLOC_FAILED";
    case 15: return "CUBLAS_STATUS_NOT_SUPPORTED";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

size_t cublasLtGetVersion(void)
{
    return 120800;
}

/* ================================================================
 * Matmul descriptor
 * ================================================================ */
cublasStatus_t cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t *matmulDesc,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType)
{
    LtMatmulDescEntry *e = lt_alloc_matmul_desc();
    if (!e) {
        if (matmulDesc) *matmulDesc = NULL;
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    e->computeType = computeType;
    e->scaleType   = scaleType;
    if (matmulDesc) *matmulDesc = (cublasLtMatmulDesc_t)e;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
{
    LtMatmulDescEntry *e = (LtMatmulDescEntry *)matmulDesc;
    if (e && e->in_use) e->in_use = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmulDesc, int attr,
    const void *buf, size_t sizeInBytes)
{
    LtMatmulDescEntry *e = (LtMatmulDescEntry *)matmulDesc;
    if (!e || !e->in_use || !buf) return CUBLAS_STATUS_SUCCESS;

    switch (attr) {
    case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
        if (sizeInBytes >= sizeof(int)) e->computeType = *(const int *)buf;
        break;
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE:
        if (sizeInBytes >= sizeof(int)) e->scaleType = *(const int *)buf;
        break;
    case CUBLASLT_MATMUL_DESC_TRANSA:
        if (sizeInBytes >= sizeof(int)) e->transa = *(const int *)buf;
        break;
    case CUBLASLT_MATMUL_DESC_TRANSB:
        if (sizeInBytes >= sizeof(int)) e->transb = *(const int *)buf;
        break;
    default:
        break;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(
    cublasLtMatmulDesc_t matmulDesc, int attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten)
{
    LtMatmulDescEntry *e = (LtMatmulDescEntry *)matmulDesc;
    if (sizeWritten) *sizeWritten = 0;
    if (!e || !e->in_use || !buf) return CUBLAS_STATUS_SUCCESS;

    switch (attr) {
    case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
        if (sizeInBytes >= sizeof(int)) {
            *(int *)buf = e->computeType;
            if (sizeWritten) *sizeWritten = sizeof(int);
        }
        break;
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE:
        if (sizeInBytes >= sizeof(int)) {
            *(int *)buf = e->scaleType;
            if (sizeWritten) *sizeWritten = sizeof(int);
        }
        break;
    case CUBLASLT_MATMUL_DESC_TRANSA:
        if (sizeInBytes >= sizeof(int)) {
            *(int *)buf = e->transa;
            if (sizeWritten) *sizeWritten = sizeof(int);
        }
        break;
    case CUBLASLT_MATMUL_DESC_TRANSB:
        if (sizeInBytes >= sizeof(int)) {
            *(int *)buf = e->transb;
            if (sizeWritten) *sizeWritten = sizeof(int);
        }
        break;
    default:
        break;
    }
    return CUBLAS_STATUS_SUCCESS;
}

/* ================================================================
 * Matrix layout
 * ================================================================ */
cublasStatus_t cublasLtMatrixLayoutCreate(
    cublasLtMatrixLayout_t *matLayout,
    cudaDataType_t type,
    uint64_t rows, uint64_t cols, int64_t ld)
{
    LtMatrixLayoutEntry *e = lt_alloc_layout();
    if (!e) {
        if (matLayout) *matLayout = NULL;
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    e->dataType = type;
    e->rows     = rows;
    e->cols     = cols;
    e->ld       = ld;
    if (matLayout) *matLayout = (cublasLtMatrixLayout_t)e;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
{
    LtMatrixLayoutEntry *e = (LtMatrixLayoutEntry *)matLayout;
    if (e && e->in_use) e->in_use = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(
    cublasLtMatrixLayout_t matLayout, int attr,
    const void *buf, size_t sizeInBytes)
{
    LtMatrixLayoutEntry *e = (LtMatrixLayoutEntry *)matLayout;
    if (!e || !e->in_use || !buf) return CUBLAS_STATUS_SUCCESS;

    switch (attr) {
    case CUBLASLT_MATRIX_LAYOUT_TYPE:
        if (sizeInBytes >= sizeof(int)) e->dataType = *(const int *)buf;
        break;
    case CUBLASLT_MATRIX_LAYOUT_ROWS:
        if (sizeInBytes >= sizeof(uint64_t)) e->rows = *(const uint64_t *)buf;
        break;
    case CUBLASLT_MATRIX_LAYOUT_COLS:
        if (sizeInBytes >= sizeof(uint64_t)) e->cols = *(const uint64_t *)buf;
        break;
    case CUBLASLT_MATRIX_LAYOUT_LD:
        if (sizeInBytes >= sizeof(int64_t)) e->ld = *(const int64_t *)buf;
        break;
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
        if (sizeInBytes >= sizeof(int32_t)) e->batchCount = *(const int32_t *)buf;
        break;
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
        if (sizeInBytes >= sizeof(int64_t)) e->batchStride = *(const int64_t *)buf;
        break;
    default:
        break;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(
    cublasLtMatrixLayout_t matLayout, int attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten)
{
    (void)matLayout; (void)attr; (void)buf; (void)sizeInBytes;
    if (sizeWritten) *sizeWritten = 0;
    return CUBLAS_STATUS_SUCCESS;
}

/* ================================================================
 * Preference (opaque to us — workspace size hints, ignored)
 * ================================================================ */
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
{
    if (pref) *pref = &g_lt_dummy_handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
{
    (void)pref;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref, int attr,
    const void *buf, size_t sizeInBytes)
{
    (void)pref; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

/* ================================================================
 * Algorithm selection
 *
 * CRITICAL: returning count=0 with SUCCESS causes PyTorch to execute:
 *   TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED)
 * which throws unconditionally — there is NO fallback path.
 *
 * We must return count=1 with a valid (zero-filled, SUCCESS state)
 * heuristic result. PyTorch then calls cublasLtMatmul with the fake
 * algo, which we proxy via RPC below.
 * ================================================================ */
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference,
    int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t *heuristicResultsArray,
    int *returnAlgoCount)
{
    (void)lightHandle; (void)operationDesc;
    (void)Adesc; (void)Bdesc; (void)Cdesc; (void)Ddesc; (void)preference;

    if (returnAlgoCount) *returnAlgoCount = 0;

    if (!get_rpc()) {
        /* RPC unavailable — returning 0 causes a throw, which is the
         * correct behaviour: we cannot compute without the daemon. */
        return CUBLAS_STATUS_SUCCESS;
    }

    if (heuristicResultsArray && requestedAlgoCount > 0) {
        memset(heuristicResultsArray, 0, sizeof(*heuristicResultsArray));
        heuristicResultsArray->state        = CUBLAS_STATUS_SUCCESS; /* 0 */
        heuristicResultsArray->workspaceSize = 0;
        heuristicResultsArray->wavesCount    = 1.0f;
        if (returnAlgoCount) *returnAlgoCount = 1;
    }

    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoInit(
    cublasLtHandle_t lightHandle,
    cublasComputeType_t computeType, cudaDataType_t scaleType,
    cudaDataType_t Atype, cudaDataType_t Btype,
    cudaDataType_t Ctype, cudaDataType_t Dtype,
    int algoId, cublasLtMatmulAlgo_t *algo)
{
    (void)lightHandle; (void)computeType; (void)scaleType;
    (void)Atype; (void)Btype; (void)Ctype; (void)Dtype;
    (void)algoId; (void)algo;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulAlgoCheck(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo,
    cublasLtMatmulHeuristicResult_t *result)
{
    (void)lightHandle; (void)operationDesc;
    (void)Adesc; (void)Bdesc; (void)Cdesc; (void)Ddesc;
    (void)algo; (void)result;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(
    const cublasLtMatmulAlgo_t *algo, int attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten)
{
    (void)algo; (void)attr; (void)buf; (void)sizeInBytes;
    if (sizeWritten) *sizeWritten = 0;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(
    const cublasLtMatmulAlgo_t *algo, int attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten)
{
    (void)algo; (void)attr; (void)buf; (void)sizeInBytes;
    if (sizeWritten) *sizeWritten = 0;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(
    cublasLtMatmulAlgo_t *algo, int attr,
    const void *buf, size_t sizeInBytes)
{
    (void)algo; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ================================================================
 * cublasLtMatmul — pure serializer
 *
 * Packs raw stored descriptor state into GpuCublasLtMatmulRequest
 * and sends GPU_CMD_CUBLAS_LT_MATMUL to the daemon.
 *
 * No arithmetic. No GPU memory operations. No C/D logic.
 * The daemon calls the real cublasLtMatmul on the host GPU, which
 * natively supports C≠D (fused bias-add / addmm).
 * ================================================================ */
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void *alpha,
    const void *A, cublasLtMatrixLayout_t Adesc,
    const void *B, cublasLtMatrixLayout_t Bdesc,
    const void *beta,
    const void *C, cublasLtMatrixLayout_t Cdesc,
    void *D,       cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo,
    void *workspace, size_t workspaceSizeInBytes,
    cudaStream_t stream)
{
    (void)lightHandle;
    (void)algo; (void)workspace; (void)workspaceSizeInBytes; (void)stream;

    rpc_call_fn rpc = get_rpc();
    if (!rpc) return CUBLAS_STATUS_NOT_SUPPORTED;

    LtMatmulDescEntry   *desc = (LtMatmulDescEntry *)computeDesc;
    LtMatrixLayoutEntry *Al   = (LtMatrixLayoutEntry *)Adesc;
    LtMatrixLayoutEntry *Bl   = (LtMatrixLayoutEntry *)Bdesc;
    LtMatrixLayoutEntry *Cl   = (LtMatrixLayoutEntry *)Cdesc;
    LtMatrixLayoutEntry *Dl   = (LtMatrixLayoutEntry *)Ddesc;

    if (!desc || !Al || !Bl || !Dl ||
        !desc->in_use || !Al->in_use || !Bl->in_use || !Dl->in_use) {
        STUB_LOG("cublasLtMatmul: invalid descriptor handles");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    /* Serialize raw stored state — no interpretation, no arithmetic */
    GpuCublasLtMatmulRequest req;
    memset(&req, 0, sizeof(req));

    /* Operation descriptor — raw stored values */
    req.transa      = desc->transa;
    req.transb      = desc->transb;
    req.computeType = desc->computeType;
    req.scaleType   = desc->scaleType;

    /* Matrix layouts — raw rows/cols/ld/type/stride as stored */
    req.Atype   = Al->dataType;
    req.A_rows  = Al->rows;
    req.A_cols  = Al->cols;
    req.lda     = Al->ld;
    req.strideA = Al->batchStride;

    req.Btype   = Bl->dataType;
    req.B_rows  = Bl->rows;
    req.B_cols  = Bl->cols;
    req.ldb     = Bl->ld;
    req.strideB = Bl->batchStride;

    /* C may be NULL when beta=0 and caller omits it */
    if (Cl && Cl->in_use) {
        req.Ctype   = Cl->dataType;
        req.C_rows  = Cl->rows;
        req.C_cols  = Cl->cols;
        req.ldc     = Cl->ld;
        req.strideC = Cl->batchStride;
    } else {
        req.Ctype   = Dl->dataType; /* fallback: same type as D */
        req.C_rows  = Dl->rows;
        req.C_cols  = Dl->cols;
        req.ldc     = Dl->ld;
        req.strideC = Dl->batchStride;
    }

    req.Dtype   = Dl->dataType;
    req.D_rows  = Dl->rows;
    req.D_cols  = Dl->cols;
    req.ldd     = Dl->ld;
    req.strideD = Dl->batchStride;

    req.batchCount = Dl->batchCount;

    /* Scalars: always send full 16 bytes — daemon reads what it needs
     * based on scaleType. No size arithmetic in the stub. */
    if (alpha) memcpy(req.alpha, alpha, sizeof(req.alpha));
    if (beta)  memcpy(req.beta,  beta,  sizeof(req.beta));

    /* Device pointers — passed through verbatim */
    req.A_ptr = (uint64_t)(uintptr_t)A;
    req.B_ptr = (uint64_t)(uintptr_t)B;
    req.C_ptr = (uint64_t)(uintptr_t)C;
    req.D_ptr = (uint64_t)(uintptr_t)D;

    STUB_LOG("cublasLtMatmul: Arows=%llu Acols=%llu Brows=%llu Bcols=%llu "
             "Drows=%llu Dcols=%llu transa=%d transb=%d computeType=%d "
             "bc=%d C%sD",
             (unsigned long long)req.A_rows, (unsigned long long)req.A_cols,
             (unsigned long long)req.B_rows, (unsigned long long)req.B_cols,
             (unsigned long long)req.D_rows, (unsigned long long)req.D_cols,
             req.transa, req.transb, req.computeType,
             req.batchCount, (C == D) ? "==" : "!=");

    int err = rpc(GPU_CMD_CUBLAS_LT_MATMUL,
                  &req, sizeof(req), NULL, 0, NULL);
    if (err) {
        STUB_LOG("cublasLtMatmul: RPC failed (err=%d)", err);
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    
    /* Mask FPE on exit — PyTorch's CublasHandlePool enables MXCSR exceptions
     * during init. Masking here ensures the calling thread's MXCSR is safe
     * for CPU code (e.g. temperature sampling) that runs after our return. */
    mask_fpe_exceptions();
    return transport_result ? CUBLAS_STATUS_EXECUTION_FAILED : CUBLAS_STATUS_SUCCESS;
}

/* ================================================================
 * Matrix transform — NOT_SUPPORTED (not used by LLM inference)
 * ================================================================ */
cublasStatus_t cublasLtMatrixTransformDescCreate(
    cublasLtMatrixTransformDesc_t *transformDesc,
    cudaDataType_t scaleType)
{
    (void)scaleType;
    if (transformDesc) *transformDesc = &g_lt_dummy_handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixTransformDescDestroy(
    cublasLtMatrixTransformDesc_t transformDesc)
{
    (void)transformDesc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixTransformDescSetAttribute(
    cublasLtMatrixTransformDesc_t transformDesc, int attr,
    const void *buf, size_t sizeInBytes)
{
    (void)transformDesc; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixTransform(
    cublasLtHandle_t lightHandle,
    cublasLtMatrixTransformDesc_t transformDesc,
    const void *alpha,
    const void *A, cublasLtMatrixLayout_t Adesc,
    const void *beta,
    const void *B, cublasLtMatrixLayout_t Bdesc,
    void *C,       cublasLtMatrixLayout_t Cdesc,
    cudaStream_t stream)
{
    (void)lightHandle; (void)transformDesc;
    (void)alpha; (void)A; (void)Adesc;
    (void)beta;  (void)B; (void)Bdesc;
    (void)C; (void)Cdesc; (void)stream;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}