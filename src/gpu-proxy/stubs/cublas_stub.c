/*
 * DeCloud cuBLAS Stub Library (libcublas.so.12)
 *
 * Provides ALL cuBLAS symbols required by PyTorch 2.3.x libtorch_cuda.so
 * with correct @@libcublas.so.12 version tags.
 *
 * Architecture:
 *   - Handle/mode/workspace management: return SUCCESS (dummy handles)
 *   - GEMM (S/D/H): delegate to cublasGemmStridedBatchedEx → RPC to daemon
 *   - GEMM (C/Z complex): return NOT_SUPPORTED (PyTorch falls back)
 *   - GEMV, dot, TRSM, GELS, GETRF, GETRS, GEQRF: return NOT_SUPPORTED
 *     (PyTorch handles these gracefully or uses cuSOLVER instead)
 *
 * Symbol list sourced from:
 *   objdump -T libtorch_cuda.so | awk '/libcublas\.so\.12/{print $NF}' | sort -u
 *
 * Build:
 *   gcc -shared -fPIC -I. \
 *       -Wl,-soname,libcublas.so.12 \
 *       -Wl,--version-script=stubs/libcublas.version \
 *       -o build/libcublas_stub.so stubs/cublas_stub.c -ldl
 */

#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>

#include "../proto/gpu_proxy_proto.h"

/* ----------------------------------------------------------------
 * Type aliases
 * ---------------------------------------------------------------- */
typedef void   *cublasHandle_t;
typedef int     cublasStatus_t;
typedef void   *cudaStream_t;
typedef struct { float  x, y; } cuComplex;
typedef struct { double x, y; } cuDoubleComplex;

#define CUBLAS_STATUS_SUCCESS           0
#define CUBLAS_STATUS_NOT_INITIALIZED   1
#define CUBLAS_STATUS_ALLOC_FAILED      3
#define CUBLAS_STATUS_EXECUTION_FAILED 13
#define CUBLAS_STATUS_NOT_SUPPORTED    15

/* CUDA data type constants */
#define CUDA_R_32F   0
#define CUDA_R_64F   1
#define CUDA_R_16F   2
#define CUDA_C_32F   4
#define CUDA_C_64F   5

/* cuBLAS compute type constants */
#define CUBLAS_COMPUTE_16F  64
#define CUBLAS_COMPUTE_32F  68
#define CUBLAS_COMPUTE_64F  70

#define STUB_LOG(fmt, ...) \
    fprintf(stderr, "[cublas-stub] " fmt "\n", ##__VA_ARGS__)

static int g_cublas_dummy_handle = 0xDEC10BD;
static int g_pointer_mode = 0; /* CUBLAS_POINTER_MODE_HOST */

/* ================================================================
 * RPC bridge — resolves decloud_rpc_call from the shim at runtime
 * ================================================================ */
typedef int (*rpc_call_fn)(uint8_t cmd, const void *req, uint32_t req_len,
                           void *resp, uint32_t resp_size, uint32_t *resp_len);

static rpc_call_fn g_rpc_call = NULL;
static int g_rpc_resolved = 0;

static rpc_call_fn get_rpc(void)
{
    if (!g_rpc_resolved) {
        g_rpc_call = (rpc_call_fn)dlsym(RTLD_DEFAULT, "decloud_rpc_call");
        if (!g_rpc_call)
            STUB_LOG("WARNING: decloud_rpc_call not found — GEMM RPC unavailable");
        g_rpc_resolved = 1;
    }
    return g_rpc_call;
}

/* ================================================================
 * Handle / stream / mode management
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
    if (mode) *mode = 0; /* CUBLAS_DEFAULT_MATH */
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, int mode)
{
    (void)handle;
    g_pointer_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, int *mode)
{
    (void)handle;
    if (mode) *mode = g_pointer_mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle,
                                      void *workspace, size_t sizeInBytes)
{
    (void)handle; (void)workspace; (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version)
{
    (void)handle;
    if (version) *version = 120800;
    return CUBLAS_STATUS_SUCCESS;
}

const char *cublasGetStatusString(cublasStatus_t status)
{
    switch (status) {
    case 0:  return "CUBLAS_STATUS_SUCCESS";
    case 1:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case 3:  return "CUBLAS_STATUS_ALLOC_FAILED";
    case 13: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case 15: return "CUBLAS_STATUS_NOT_SUPPORTED";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdout,
                                      int logToStderr, const char *logFileName)
{
    (void)logIsOn; (void)logToStdout; (void)logToStderr; (void)logFileName;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetLoggerCallback(void *callback)
{
    (void)callback;
    return CUBLAS_STATUS_SUCCESS;
}

/* ================================================================
 * Core RPC GEMM path
 *
 * All real/half GEMM variants funnel here. Complex GEMM variants
 * return NOT_SUPPORTED — PyTorch falls back to cuSOLVER or CPU.
 * ================================================================ */

/* Forward declarations needed by delegation chain */
cublasStatus_t cublasGemmEx(cublasHandle_t, int, int, int, int, int,
    const void *, const void *, int, int, const void *, int, int,
    const void *, void *, int, int, int, int);

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t, int, int,
    int, int, int, const void *, const void *, int, int, long long,
    const void *, int, int, long long, const void *, void *, int, int,
    long long, int, int, int);

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta, void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType, int algo)
{
    (void)h;
    if (batchCount <= 0) return CUBLAS_STATUS_SUCCESS;

    rpc_call_fn rpc = get_rpc();
    if (!rpc) return CUBLAS_STATUS_NOT_SUPPORTED;

    int scalar_size = 4;
    if (computeType == CUBLAS_COMPUTE_64F) scalar_size = 8;
    if (computeType == CUBLAS_COMPUTE_16F) scalar_size = 2;

    GpuCublasGemmStridedRequest req;
    memset(&req, 0, sizeof(req));
    req.transa      = ta;
    req.transb      = tb;
    req.m           = m;
    req.n           = n;
    req.k           = k;
    req.Atype       = Atype;
    req.lda         = lda;
    req.strideA     = strideA;
    req.Btype       = Btype;
    req.ldb         = ldb;
    req.strideB     = strideB;
    req.Ctype       = Ctype;
    req.ldc         = ldc;
    req.strideC     = strideC;
    req.batchCount  = batchCount;
    req.computeType = computeType;
    req.algo        = algo;
    req.A_ptr       = (uint64_t)(uintptr_t)A;
    req.B_ptr       = (uint64_t)(uintptr_t)B;
    req.C_ptr       = (uint64_t)(uintptr_t)C;

    if (alpha) memcpy(req.alpha, alpha, scalar_size);
    if (beta)  memcpy(req.beta,  beta,  scalar_size);

    int err = rpc(GPU_CMD_CUBLAS_GEMM_STRIDED,
                  &req, sizeof(req), NULL, 0, NULL);
    return err ? CUBLAS_STATUS_EXECUTION_FAILED : CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta, void *C, int Ctype, int ldc,
    int computeType, int algo)
{
    return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha,
        A, Atype, lda, 0, B, Btype, ldb, 0,
        beta, C, Ctype, ldc, 0, 1, computeType, algo);
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const void *alpha,
    const void *const A[], int Atype, int lda,
    const void *const B[], int Btype, int ldb,
    const void *beta, void *const C[], int Ctype, int ldc,
    int batchCount, int computeType, int algo)
{
    (void)h;
    if (batchCount <= 0) return CUBLAS_STATUS_SUCCESS;

    rpc_call_fn rpc = get_rpc();
    if (!rpc) return CUBLAS_STATUS_NOT_SUPPORTED;

    int scalar_size = 4;
    if (computeType == CUBLAS_COMPUTE_64F) scalar_size = 8;
    if (computeType == CUBLAS_COMPUTE_16F) scalar_size = 2;

    GpuCublasGemmBatchedRequest req;
    memset(&req, 0, sizeof(req));
    req.transa      = ta;
    req.transb      = tb;
    req.m           = m;
    req.n           = n;
    req.k           = k;
    req.Atype       = Atype;
    req.lda         = lda;
    req.Btype       = Btype;
    req.ldb         = ldb;
    req.Ctype       = Ctype;
    req.ldc         = ldc;
    req.batchCount  = batchCount;
    req.computeType = computeType;
    req.algo        = algo;
    req.A_array_dev = (uint64_t)(uintptr_t)A;
    req.B_array_dev = (uint64_t)(uintptr_t)B;
    req.C_array_dev = (uint64_t)(uintptr_t)C;

    if (alpha) memcpy(req.alpha, alpha, scalar_size);
    if (beta)  memcpy(req.beta,  beta,  scalar_size);

    int err = rpc(GPU_CMD_CUBLAS_GEMM_BATCHED,
                  &req, sizeof(req), NULL, 0, NULL);
    return err ? CUBLAS_STATUS_EXECUTION_FAILED : CUBLAS_STATUS_SUCCESS;
}

/* ================================================================
 * Real GEMM wrappers — delegate to GemmStridedBatchedEx
 * ================================================================ */

/* FP32 */
cublasStatus_t cublasSgemm_v2(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const float *alpha,
    const float *A, int lda, const float *B, int ldb,
    const float *beta, float *C, int ldc)
{
    return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha,
        A, CUDA_R_32F, lda, 0, B, CUDA_R_32F, ldb, 0,
        beta, C, CUDA_R_32F, ldc, 0, 1, CUBLAS_COMPUTE_32F, -1);
}

cublasStatus_t cublasSgemmEx(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const float *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const float *beta, void *C, int Ctype, int ldc)
{
    return cublasGemmEx(h, ta, tb, m, n, k, alpha,
        A, Atype, lda, B, Btype, ldb,
        beta, C, Ctype, ldc, CUBLAS_COMPUTE_32F, -1);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const float *alpha,
    const float *A, int lda, long long strideA,
    const float *B, int ldb, long long strideB,
    const float *beta, float *C, int ldc, long long strideC,
    int batchCount)
{
    return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha,
        A, CUDA_R_32F, lda, strideA, B, CUDA_R_32F, ldb, strideB,
        beta, C, CUDA_R_32F, ldc, strideC, batchCount, CUBLAS_COMPUTE_32F, -1);
}

/* FP64 */
cublasStatus_t cublasDgemm_v2(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const double *alpha,
    const double *A, int lda, const double *B, int ldb,
    const double *beta, double *C, int ldc)
{
    return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha,
        A, CUDA_R_64F, lda, 0, B, CUDA_R_64F, ldb, 0,
        beta, C, CUDA_R_64F, ldc, 0, 1, CUBLAS_COMPUTE_64F, -1);
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const double *alpha,
    const double *A, int lda, long long strideA,
    const double *B, int ldb, long long strideB,
    const double *beta, double *C, int ldc, long long strideC,
    int batchCount)
{
    return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha,
        A, CUDA_R_64F, lda, strideA, B, CUDA_R_64F, ldb, strideB,
        beta, C, CUDA_R_64F, ldc, strideC, batchCount, CUBLAS_COMPUTE_64F, -1);
}

/* FP16 */
cublasStatus_t cublasHgemm(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const void *alpha,
    const void *A, int lda, const void *B, int ldb,
    const void *beta, void *C, int ldc)
{
    return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha,
        A, CUDA_R_16F, lda, 0, B, CUDA_R_16F, ldb, 0,
        beta, C, CUDA_R_16F, ldc, 0, 1, CUBLAS_COMPUTE_16F, -1);
}

/* ================================================================
 * Complex GEMM — NOT_SUPPORTED
 * PyTorch detects this and falls back to cuSOLVER or CPU paths.
 * ================================================================ */

cublasStatus_t cublasCgemm_v2(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const cuComplex *alpha,
    const cuComplex *A, int lda, const cuComplex *B, int ldb,
    const cuComplex *beta, cuComplex *C, int ldc)
{
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;
    (void)alpha;(void)A;(void)lda;(void)B;(void)ldb;
    (void)beta;(void)C;(void)ldc;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const cuComplex *alpha,
    const cuComplex *A, int lda, long long strideA,
    const cuComplex *B, int ldb, long long strideB,
    const cuComplex *beta, cuComplex *C, int ldc, long long strideC,
    int batchCount)
{
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;
    (void)alpha;(void)A;(void)lda;(void)strideA;
    (void)B;(void)ldb;(void)strideB;
    (void)beta;(void)C;(void)ldc;(void)strideC;(void)batchCount;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
    const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;
    (void)alpha;(void)A;(void)lda;(void)B;(void)ldb;
    (void)beta;(void)C;(void)ldc;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda, long long strideA,
    const cuDoubleComplex *B, int ldb, long long strideB,
    const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long strideC,
    int batchCount)
{
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;
    (void)alpha;(void)A;(void)lda;(void)strideA;
    (void)B;(void)ldb;(void)strideB;
    (void)beta;(void)C;(void)ldc;(void)strideC;(void)batchCount;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ================================================================
 * GEMV (matrix-vector multiply) — NOT_SUPPORTED
 * ================================================================ */

cublasStatus_t cublasSgemv_v2(cublasHandle_t h, int trans,
    int m, int n, const float *alpha,
    const float *A, int lda, const float *x, int incx,
    const float *beta, float *y, int incy)
{
    (void)h;(void)trans;(void)m;(void)n;(void)alpha;
    (void)A;(void)lda;(void)x;(void)incx;(void)beta;(void)y;(void)incy;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t h, int trans,
    int m, int n, const double *alpha,
    const double *A, int lda, const double *x, int incx,
    const double *beta, double *y, int incy)
{
    (void)h;(void)trans;(void)m;(void)n;(void)alpha;
    (void)A;(void)lda;(void)x;(void)incx;(void)beta;(void)y;(void)incy;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgemv_v2(cublasHandle_t h, int trans,
    int m, int n, const cuComplex *alpha,
    const cuComplex *A, int lda, const cuComplex *x, int incx,
    const cuComplex *beta, cuComplex *y, int incy)
{
    (void)h;(void)trans;(void)m;(void)n;(void)alpha;
    (void)A;(void)lda;(void)x;(void)incx;(void)beta;(void)y;(void)incy;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t h, int trans,
    int m, int n, const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
    const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    (void)h;(void)trans;(void)m;(void)n;(void)alpha;
    (void)A;(void)lda;(void)x;(void)incx;(void)beta;(void)y;(void)incy;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ================================================================
 * Dot products — NOT_SUPPORTED
 * ================================================================ */

cublasStatus_t cublasSdot_v2(cublasHandle_t h, int n,
    const float *x, int incx, const float *y, int incy, float *result)
{
    (void)h;(void)n;(void)x;(void)incx;(void)y;(void)incy;
    if (result) *result = 0.0f;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t h, int n,
    const double *x, int incx, const double *y, int incy, double *result)
{
    (void)h;(void)n;(void)x;(void)incx;(void)y;(void)incy;
    if (result) *result = 0.0;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCdotc_v2(cublasHandle_t h, int n,
    const cuComplex *x, int incx, const cuComplex *y, int incy,
    cuComplex *result)
{
    (void)h;(void)n;(void)x;(void)incx;(void)y;(void)incy;
    if (result) { result->x = 0.0f; result->y = 0.0f; }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t h, int n,
    const cuComplex *x, int incx, const cuComplex *y, int incy,
    cuComplex *result)
{
    (void)h;(void)n;(void)x;(void)incx;(void)y;(void)incy;
    if (result) { result->x = 0.0f; result->y = 0.0f; }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t h, int n,
    const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy,
    cuDoubleComplex *result)
{
    (void)h;(void)n;(void)x;(void)incx;(void)y;(void)incy;
    if (result) { result->x = 0.0; result->y = 0.0; }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t h, int n,
    const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy,
    cuDoubleComplex *result)
{
    (void)h;(void)n;(void)x;(void)incx;(void)y;(void)incy;
    if (result) { result->x = 0.0; result->y = 0.0; }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDotEx(cublasHandle_t h, int n,
    const void *x, int xType, int incx,
    const void *y, int yType, int incy,
    void *result, int resultType, int executionType)
{
    (void)h;(void)n;(void)x;(void)xType;(void)incx;
    (void)y;(void)yType;(void)incy;
    (void)result;(void)resultType;(void)executionType;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ================================================================
 * TRSM (triangular solve) — NOT_SUPPORTED
 * ================================================================ */

cublasStatus_t cublasStrsm_v2(cublasHandle_t h,
    int side, int uplo, int trans, int diag,
    int m, int n, const float *alpha,
    const float *A, int lda, float *B, int ldb)
{
    (void)h;(void)side;(void)uplo;(void)trans;(void)diag;
    (void)m;(void)n;(void)alpha;(void)A;(void)lda;(void)B;(void)ldb;
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

cublasStatus_t cublasDtrsm_v2(cublasHandle_t h,
    int side, int uplo, int trans, int diag,
    int m, int n, const double *alpha,
    const double *A, int lda, double *B, int ldb)
{
    (void)h;(void)side;(void)uplo;(void)trans;(void)diag;
    (void)m;(void)n;(void)alpha;(void)A;(void)lda;(void)B;(void)ldb;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t h,
    int side, int uplo, int trans, int diag,
    int m, int n, const double *alpha,
    const double *const A[], int lda,
    double *const B[], int ldb, int batchCount)
{
    (void)h;(void)side;(void)uplo;(void)trans;(void)diag;
    (void)m;(void)n;(void)alpha;(void)A;(void)lda;
    (void)B;(void)ldb;(void)batchCount;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCtrsm_v2(cublasHandle_t h,
    int side, int uplo, int trans, int diag,
    int m, int n, const cuComplex *alpha,
    const cuComplex *A, int lda, cuComplex *B, int ldb)
{
    (void)h;(void)side;(void)uplo;(void)trans;(void)diag;
    (void)m;(void)n;(void)alpha;(void)A;(void)lda;(void)B;(void)ldb;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t h,
    int side, int uplo, int trans, int diag,
    int m, int n, const cuComplex *alpha,
    const cuComplex *const A[], int lda,
    cuComplex *const B[], int ldb, int batchCount)
{
    (void)h;(void)side;(void)uplo;(void)trans;(void)diag;
    (void)m;(void)n;(void)alpha;(void)A;(void)lda;
    (void)B;(void)ldb;(void)batchCount;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t h,
    int side, int uplo, int trans, int diag,
    int m, int n, const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb)
{
    (void)h;(void)side;(void)uplo;(void)trans;(void)diag;
    (void)m;(void)n;(void)alpha;(void)A;(void)lda;(void)B;(void)ldb;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t h,
    int side, int uplo, int trans, int diag,
    int m, int n, const cuDoubleComplex *alpha,
    const cuDoubleComplex *const A[], int lda,
    cuDoubleComplex *const B[], int ldb, int batchCount)
{
    (void)h;(void)side;(void)uplo;(void)trans;(void)diag;
    (void)m;(void)n;(void)alpha;(void)A;(void)lda;
    (void)B;(void)ldb;(void)batchCount;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ================================================================
 * Batched solvers (GELS, GETRF, GETRS, GEQRF) — NOT_SUPPORTED
 * ================================================================ */

cublasStatus_t cublasSgelsBatched(cublasHandle_t h, int trans,
    int m, int n, int nrhs, float *const A[], int lda,
    float *const C[], int ldc, int *info, int *devInfoArray, int batchSize)
{
    (void)h;(void)trans;(void)m;(void)n;(void)nrhs;
    (void)A;(void)lda;(void)C;(void)ldc;(void)info;
    (void)devInfoArray;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgelsBatched(cublasHandle_t h, int trans,
    int m, int n, int nrhs, double *const A[], int lda,
    double *const C[], int ldc, int *info, int *devInfoArray, int batchSize)
{
    (void)h;(void)trans;(void)m;(void)n;(void)nrhs;
    (void)A;(void)lda;(void)C;(void)ldc;(void)info;
    (void)devInfoArray;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgelsBatched(cublasHandle_t h, int trans,
    int m, int n, int nrhs, cuComplex *const A[], int lda,
    cuComplex *const C[], int ldc, int *info, int *devInfoArray, int batchSize)
{
    (void)h;(void)trans;(void)m;(void)n;(void)nrhs;
    (void)A;(void)lda;(void)C;(void)ldc;(void)info;
    (void)devInfoArray;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgelsBatched(cublasHandle_t h, int trans,
    int m, int n, int nrhs, cuDoubleComplex *const A[], int lda,
    cuDoubleComplex *const C[], int ldc, int *info,
    int *devInfoArray, int batchSize)
{
    (void)h;(void)trans;(void)m;(void)n;(void)nrhs;
    (void)A;(void)lda;(void)C;(void)ldc;(void)info;
    (void)devInfoArray;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t h, int n,
    float *const A[], int lda, int *P, int *info, int batchSize)
{
    (void)h;(void)n;(void)A;(void)lda;(void)P;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t h, int n,
    double *const A[], int lda, int *P, int *info, int batchSize)
{
    (void)h;(void)n;(void)A;(void)lda;(void)P;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t h, int n,
    cuComplex *const A[], int lda, int *P, int *info, int batchSize)
{
    (void)h;(void)n;(void)A;(void)lda;(void)P;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t h, int n,
    cuDoubleComplex *const A[], int lda, int *P, int *info, int batchSize)
{
    (void)h;(void)n;(void)A;(void)lda;(void)P;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t h, int trans, int n,
    int nrhs, const float *const A[], int lda, const int *devIpiv,
    float *const B[], int ldb, int *info, int batchSize)
{
    (void)h;(void)trans;(void)n;(void)nrhs;(void)A;(void)lda;
    (void)devIpiv;(void)B;(void)ldb;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t h, int trans, int n,
    int nrhs, const double *const A[], int lda, const int *devIpiv,
    double *const B[], int ldb, int *info, int batchSize)
{
    (void)h;(void)trans;(void)n;(void)nrhs;(void)A;(void)lda;
    (void)devIpiv;(void)B;(void)ldb;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t h, int trans, int n,
    int nrhs, const cuComplex *const A[], int lda, const int *devIpiv,
    cuComplex *const B[], int ldb, int *info, int batchSize)
{
    (void)h;(void)trans;(void)n;(void)nrhs;(void)A;(void)lda;
    (void)devIpiv;(void)B;(void)ldb;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t h, int trans, int n,
    int nrhs, const cuDoubleComplex *const A[], int lda, const int *devIpiv,
    cuDoubleComplex *const B[], int ldb, int *info, int batchSize)
{
    (void)h;(void)trans;(void)n;(void)nrhs;(void)A;(void)lda;
    (void)devIpiv;(void)B;(void)ldb;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgeqrfBatched(cublasHandle_t h, int m, int n,
    float *const A[], int lda, float *const TAU[], int *info, int batchSize)
{
    (void)h;(void)m;(void)n;(void)A;(void)lda;
    (void)TAU;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t h, int m, int n,
    double *const A[], int lda, double *const TAU[], int *info, int batchSize)
{
    (void)h;(void)m;(void)n;(void)A;(void)lda;
    (void)TAU;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t h, int m, int n,
    cuComplex *const A[], int lda, cuComplex *const TAU[],
    int *info, int batchSize)
{
    (void)h;(void)m;(void)n;(void)A;(void)lda;
    (void)TAU;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t h, int m, int n,
    cuDoubleComplex *const A[], int lda, cuDoubleComplex *const TAU[],
    int *info, int batchSize)
{
    (void)h;(void)m;(void)n;(void)A;(void)lda;
    (void)TAU;(void)info;(void)batchSize;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}