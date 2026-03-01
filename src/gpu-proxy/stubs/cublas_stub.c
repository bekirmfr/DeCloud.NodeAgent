/*
 * DeCloud cuBLAS Stub Library (libcublas.so.12)
 *
 * Minimal stub that satisfies libggml-cuda.so's DT_NEEDED: libcublas.so.12
 * with correct ELF version tags (@@libcublas.so.12).
 *
 * Background:
 *   Ollama v0.17 bundles real NVIDIA cuBLAS (~300MB) in cuda_v12/. Real cuBLAS
 *   calls cuGetExportTable (private NVIDIA driver internals) which can't be
 *   proxied. This stub replaces the real library — init/handle functions return
 *   SUCCESS so ggml's cuBLAS init path doesn't crash, compute functions return
 *   NOT_SUPPORTED so ggml falls back to its native MMQ kernels.
 *
 * CRITICAL: Must be compiled with:
 *   gcc -shared -fPIC -o libcublas_stub.so cublas_stub.c \
 *       -Wl,-soname,libcublas.so.12 \
 *       -Wl,--version-script=libcublas.version
 *
 *   The version script exports symbols under the libcublas.so.12 version
 *   namespace. Without it, symbols get tagged @@libcudart.so.12 (wrong)
 *   and dlopen(libggml-cuda.so) fails silently.
 *
 * See: GPU_PROXY_DEBUGGING_JOURNAL.md, Problem 17 (Day 4)
 */

#include <stddef.h>

typedef void *cublasHandle_t;
typedef int cublasStatus_t;
typedef void *cudaStream_t;

#define CUBLAS_STATUS_SUCCESS          0
#define CUBLAS_STATUS_NOT_INITIALIZED  1
#define CUBLAS_STATUS_NOT_SUPPORTED   15

static int g_cublas_dummy_handle = 0xDEC10BD;

/* ================================================================
 * Init / handle management — return SUCCESS
 * ================================================================ */

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
{
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
    if (mode) *mode = 0;
    return CUBLAS_STATUS_SUCCESS;
}

const char *cublasGetStatusString(cublasStatus_t status)
{
    switch (status) {
        case 0:  return "CUBLAS_STATUS_SUCCESS";
        case 1:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case 15: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

/* ================================================================
 * Compute stubs — return NOT_SUPPORTED
 *
 * These are called when ggml bypasses MMQ and attempts cuBLAS compute.
 * Returning NOT_SUPPORTED makes ggml fall back to its native MMQ kernels.
 * ================================================================ */

cublasStatus_t cublasSgemm_v2(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const float *alpha,
    const float *A, int lda, const float *B, int ldb,
    const float *beta, float *C, int ldc)
{
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;
    (void)alpha;(void)A;(void)lda;(void)B;(void)ldb;
    (void)beta;(void)C;(void)ldc;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmEx(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta, void *C, int Ctype, int ldc,
    int computeType, int algo)
{
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;
    (void)alpha;(void)A;(void)Atype;(void)lda;
    (void)B;(void)Btype;(void)ldb;
    (void)beta;(void)C;(void)Ctype;(void)ldc;
    (void)computeType;(void)algo;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta, void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType, int algo)
{
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;
    (void)alpha;(void)A;(void)Atype;(void)lda;(void)strideA;
    (void)B;(void)Btype;(void)ldb;(void)strideB;
    (void)beta;(void)C;(void)Ctype;(void)ldc;(void)strideC;
    (void)batchCount;(void)computeType;(void)algo;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const void *alpha,
    const void *const A[], int Atype, int lda,
    const void *const B[], int Btype, int ldb,
    const void *beta, void *const C[], int Ctype, int ldc,
    int batchCount, int computeType, int algo)
{
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;
    (void)alpha;(void)A;(void)Atype;(void)lda;
    (void)B;(void)Btype;(void)ldb;
    (void)beta;(void)C;(void)Ctype;(void)ldc;
    (void)batchCount;(void)computeType;(void)algo;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t h,
    int side, int uplo, int trans, int diag,
    int m, int n, const float *alpha,
    const float *const A[], int lda,
    float *const B[], int ldb, int batchCount)
{
    (void)h;(void)side;(void)uplo;(void)trans;(void)diag;
    (void)m;(void)n;(void)alpha;(void)A;(void)lda;
    (void)B;(void)ldb;(void)batchCount;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
