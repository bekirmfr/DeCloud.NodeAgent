/*
 * DeCloud cuBLAS Lt Stub Library (libcublasLt.so.12)
 *
 * Provides versioned cublasLt symbols required by PyTorch 2.3.x.
 *
 * Background:
 *   PyTorch's libtorch_cuda.so resolves cublasLt symbols with versioned
 *   lookups (e.g. cublasLtCreate@libcublasLt.so.12). The previous stub
 *   had no version script and only a placeholder symbol — dlopen succeeded
 *   for ggml (which only needs DT_NEEDED satisfied) but failed for PyTorch
 *   with: version `libcublasLt.so.12' not found
 *
 *   In DeCloud proxy mode, PyTorch's matmuls route through
 *   cublasGemmStridedBatchedEx (proxied via GPU_CMD_CUBLAS_GEMM_STRIDED),
 *   not through the cublasLt path. These stubs exist purely to satisfy
 *   dlopen — they return CUBLAS_STATUS_NOT_SUPPORTED (3) for all calls.
 *   If a caller checks the return code and falls back, that is correct
 *   behaviour. If a caller does not check, the subsequent GPU operation
 *   will fail with a clear CUDA error rather than a silent crash.
 *
 * Build:
 *   gcc -shared -fPIC \
 *       -Wl,-soname,libcublasLt.so.12 \
 *       -Wl,--version-script=stubs/libcublasLt.version \
 *       -o libcublasLt_stub.so stubs/cublasLt_stub.c
 */

#include <stddef.h>
#include <stdint.h>

/* cublasStatus_t values */
#define CUBLAS_STATUS_SUCCESS          0
#define CUBLAS_STATUS_NOT_SUPPORTED    3

typedef int cublasStatus_t;
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatmulPreference_t;
typedef void* cublasLtMatrixTransformDesc_t;
typedef int   cublasLtMatmulAlgo_t;  /* opaque in real API — int placeholder */
typedef int   cublasLtMatmulHeuristicResult_t;
typedef int   cublasComputeType_t;
typedef int   cudaDataType_t;
typedef void* cudaStream_t;

/* ----------------------------------------------------------------
 * Internal placeholder — keeps the .so non-empty even if all
 * other symbols are stripped. Required by older linkers.
 * ---------------------------------------------------------------- */
void __decloud_cublasLt_stub_v1(void) {}

/* ----------------------------------------------------------------
 * Handle management
 * ---------------------------------------------------------------- */
cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle)
{
    if (lightHandle) *lightHandle = NULL;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle)
{
    (void)lightHandle;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

const char* cublasLtGetStatusString(cublasStatus_t status)
{
    (void)status;
    return "CUBLAS_STATUS_NOT_SUPPORTED (DeCloud proxy stub)";
}

size_t cublasLtGetVersion(void)
{
    return 120800; /* CUDA 12.8 version string */
}

/* ----------------------------------------------------------------
 * Matrix layout
 * ---------------------------------------------------------------- */
cublasStatus_t cublasLtMatrixLayoutCreate(
    cublasLtMatrixLayout_t *matLayout,
    cudaDataType_t type,
    uint64_t rows, uint64_t cols, int64_t ld)
{
    (void)type; (void)rows; (void)cols; (void)ld;
    if (matLayout) *matLayout = NULL;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
{
    (void)matLayout;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(
    cublasLtMatrixLayout_t matLayout, int attr,
    const void *buf, size_t sizeInBytes)
{
    (void)matLayout; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(
    cublasLtMatrixLayout_t matLayout, int attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten)
{
    (void)matLayout; (void)attr; (void)buf; (void)sizeInBytes;
    if (sizeWritten) *sizeWritten = 0;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ----------------------------------------------------------------
 * Matmul descriptor
 * ---------------------------------------------------------------- */
cublasStatus_t cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t *matmulDesc,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType)
{
    (void)computeType; (void)scaleType;
    if (matmulDesc) *matmulDesc = NULL;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
{
    (void)matmulDesc;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmulDesc, int attr,
    const void *buf, size_t sizeInBytes)
{
    (void)matmulDesc; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(
    cublasLtMatmulDesc_t matmulDesc, int attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten)
{
    (void)matmulDesc; (void)attr; (void)buf; (void)sizeInBytes;
    if (sizeWritten) *sizeWritten = 0;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ----------------------------------------------------------------
 * Preference
 * ---------------------------------------------------------------- */
cublasStatus_t cublasLtMatmulPreferenceCreate(
    cublasLtMatmulPreference_t *pref)
{
    if (pref) *pref = NULL;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(
    cublasLtMatmulPreference_t pref)
{
    (void)pref;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref, int attr,
    const void *buf, size_t sizeInBytes)
{
    (void)pref; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ----------------------------------------------------------------
 * Algorithm selection
 * ---------------------------------------------------------------- */
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
    (void)Adesc; (void)Bdesc; (void)Cdesc; (void)Ddesc;
    (void)preference; (void)requestedAlgoCount;
    (void)heuristicResultsArray;
    if (returnAlgoCount) *returnAlgoCount = 0;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatmulAlgoInit(
    cublasLtHandle_t lightHandle,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType,
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

/* ----------------------------------------------------------------
 * Compute
 * ---------------------------------------------------------------- */
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
    (void)lightHandle; (void)computeDesc;
    (void)alpha; (void)A; (void)Adesc;
    (void)B; (void)Bdesc; (void)beta;
    (void)C; (void)Cdesc; (void)D; (void)Ddesc;
    (void)algo; (void)workspace; (void)workspaceSizeInBytes;
    (void)stream;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ----------------------------------------------------------------
 * Matrix transform
 * ---------------------------------------------------------------- */
cublasStatus_t cublasLtMatrixTransformDescCreate(
    cublasLtMatrixTransformDesc_t *transformDesc,
    cudaDataType_t scaleType)
{
    (void)scaleType;
    if (transformDesc) *transformDesc = NULL;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatrixTransformDescDestroy(
    cublasLtMatrixTransformDesc_t transformDesc)
{
    (void)transformDesc;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtMatrixTransformDescSetAttribute(
    cublasLtMatrixTransformDesc_t transformDesc, int attr,
    const void *buf, size_t sizeInBytes)
{
    (void)transformDesc; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_NOT_SUPPORTED;
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