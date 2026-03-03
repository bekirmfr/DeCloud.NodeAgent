/*
 * DeCloud cuBLAS Stub Library (libcublas.so.12)
 *
 * Provides cublasGemmBatchedEx and cublasGemmStridedBatchedEx with
 * correct @@libcublas.so.12 version tags. GEMM calls are forwarded
 * via RPC to the daemon through the shim's exported decloud_rpc_call.
 *
 * Build:
 *   gcc -shared -fPIC -o libcublas_stub.so cublas_stub.c \
 *       -Wl,-soname,libcublas.so.12 \
 *       -Wl,--version-script=libcublas.version -ldl
 */

#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>

/* Include proto for struct definitions */
#include "../proto/gpu_proxy_proto.h"

typedef void *cublasHandle_t;
typedef int cublasStatus_t;
typedef void *cudaStream_t;

#define CUBLAS_STATUS_SUCCESS           0
#define CUBLAS_STATUS_NOT_INITIALIZED   1
#define CUBLAS_STATUS_ALLOC_FAILED      3
#define CUBLAS_STATUS_EXECUTION_FAILED 13
#define CUBLAS_STATUS_NOT_SUPPORTED    15

#define STUB_LOG(fmt, ...) \
    fprintf(stderr, "[cublas-stub] " fmt "\n", ##__VA_ARGS__)

static int g_cublas_dummy_handle = 0xDEC10BD;

/* ================================================================
 * RPC bridge to the shim's daemon connection
 *
 * The shim (libcudart.so.12, loaded via LD_PRELOAD) exports
 * decloud_rpc_call() which shares the daemon connection.
 * We resolve it lazily via dlsym(RTLD_DEFAULT).
 * ================================================================ */

typedef int (*rpc_call_fn)(uint8_t cmd, const void *req, uint32_t req_len,
                           void *resp, uint32_t resp_size, uint32_t *resp_len);

static rpc_call_fn g_rpc_call = NULL;
static int g_rpc_resolved = 0;

static rpc_call_fn get_rpc(void)
{
    if (!g_rpc_resolved) {
        g_rpc_call = (rpc_call_fn)dlsym(RTLD_DEFAULT, "decloud_rpc_call");
        if (!g_rpc_call) {
            STUB_LOG("WARNING: decloud_rpc_call not found — GEMM will fail");
        }
        g_rpc_resolved = 1;
    }
    return g_rpc_call;
}

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
        case 3:  return "CUBLAS_STATUS_ALLOC_FAILED";
        case 13: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case 15: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

/* ================================================================
 * GEMM compute functions — RPC to daemon via shim
 * ================================================================ */

/* Forward declarations for delegation chain */
cublasStatus_t cublasGemmEx(cublasHandle_t, int, int, int, int, int,
    const void *, const void *, int, int, const void *, int, int,
    const void *, void *, int, int, int, int);
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t, int, int,
    int, int, int, const void *, const void *, int, int, long long,
    const void *, int, int, long long, const void *, void *, int, int,
    long long, int, int, int);

cublasStatus_t cublasSgemm_v2(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const float *alpha,
    const float *A, int lda, const float *B, int ldb,
    const float *beta, float *C, int ldc)
{
    /* Delegate to GemmEx with float types */
    return cublasGemmEx(h, ta, tb, m, n, k, alpha,
        A, 0 /*CUDA_R_32F*/, lda,
        B, 0 /*CUDA_R_32F*/, ldb,
        beta, C, 0 /*CUDA_R_32F*/, ldc,
        68 /*CUBLAS_COMPUTE_32F*/, -1 /*CUBLAS_GEMM_DEFAULT*/);
}

cublasStatus_t cublasGemmEx(cublasHandle_t h, int ta, int tb,
    int m, int n, int k, const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta, void *C, int Ctype, int ldc,
    int computeType, int algo)
{
    /* Delegate to strided batched with batch=1, stride=0 */
    return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha,
        A, Atype, lda, 0,
        B, Btype, ldb, 0,
        beta, C, Ctype, ldc, 0,
        1, computeType, algo);
}

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
    if (!rpc) {
        STUB_LOG("cublasGemmStridedBatchedEx: no RPC — NOT_SUPPORTED");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    int scalar_size = 4;
    if (computeType == 70) scalar_size = 8;  /* CUBLAS_COMPUTE_64F */
if (computeType == 64) scalar_size = 2;

    /*
     * A, B, C are DEVICE pointers to arrays of per-batch device pointers.
     * We CANNOT dereference them — they live in GPU memory, not host RAM.
     * Attempting A[i] would SEGFAULT (the original bug).
     *
     * Instead, send the device base addresses to the daemon, which reads
     * the pointer arrays via cudaMemcpy D2H before calling real cuBLAS.
     */
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
    if (beta)  memcpy(req.beta, beta, scalar_size);

    STUB_LOG("cublasGemmBatchedEx: m=%d n=%d k=%d bc=%d A_dev=%p B_dev=%p C_dev=%p",
             m, n, k, batchCount,
             (void *)(uintptr_t)A, (void *)(uintptr_t)B, (void *)(uintptr_t)C);

    int err = rpc(GPU_CMD_CUBLAS_GEMM_BATCHED,
                  &req, sizeof(req), NULL, 0, NULL);
    if (err != 0) {
        STUB_LOG("cublasGemmStridedBatchedEx: RPC failed (err=%d)", err);
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
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
    if (!rpc) {
        STUB_LOG("cublasGemmBatchedEx: no RPC — NOT_SUPPORTED");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    int scalar_size = 4;
    if (computeType == 70) scalar_size = 8;
    if (computeType == 64) scalar_size = 2;

    uint32_t ptrs_size = 3 * batchCount * (uint32_t)sizeof(uint64_t);
    uint32_t req_len = (uint32_t)sizeof(GpuCublasGemmBatchedRequest) + ptrs_size;
    void *req_buf = malloc(req_len);
    if (!req_buf) return CUBLAS_STATUS_ALLOC_FAILED;

    GpuCublasGemmBatchedRequest *req = (GpuCublasGemmBatchedRequest *)req_buf;
    memset(req, 0, sizeof(*req));
    req->transa      = ta;
    req->transb      = tb;
    req->m           = m;
    req->n           = n;
    req->k           = k;
    req->Atype       = Atype;
    req->lda         = lda;
    req->Btype       = Btype;
    req->ldb         = ldb;
    req->Ctype       = Ctype;
    req->ldc         = ldc;
    req->batchCount  = batchCount;
    req->computeType = computeType;
    req->algo        = algo;

    if (alpha) memcpy(req->alpha, alpha, scalar_size);
    if (beta)  memcpy(req->beta, beta, scalar_size);

    uint64_t *ptrs = (uint64_t *)((uint8_t *)req_buf
                      + sizeof(GpuCublasGemmBatchedRequest));
    for (int i = 0; i < batchCount; i++) {
        ptrs[i]                  = (uint64_t)(uintptr_t)A[i];
        ptrs[batchCount + i]     = (uint64_t)(uintptr_t)B[i];
        ptrs[2 * batchCount + i] = (uint64_t)(uintptr_t)C[i];
    }

    int err = rpc(GPU_CMD_CUBLAS_GEMM_BATCHED,
                  req_buf, req_len, NULL, 0, NULL);
    free(req_buf);

    if (err != 0) {
        STUB_LOG("cublasGemmBatchedEx: RPC failed (err=%d)", err);
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
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