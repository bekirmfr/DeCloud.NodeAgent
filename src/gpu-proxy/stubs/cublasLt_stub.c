/*
 * DeCloud cuBLAS Lt Stub Library (libcublasLt.so.12)
 *
 * Minimal stub that satisfies libggml-cuda.so's DT_NEEDED: libcublasLt.so.12.
 *
 * Background:
 *   ggml-cuda links against libcublasLt.so.12 because real cuBLAS uses it
 *   internally. In DeCloud's architecture, cuBLAS GEMM calls are proxied
 *   via RPC to the daemon on the host, where the real cuBLAS + cublasLt
 *   libraries exist. The VM only needs this stub to satisfy the dynamic
 *   linker — no actual cublasLt functions are called in the VM.
 *
 * Build:
 *   gcc -shared -fPIC -Wl,-soname,libcublasLt.so.12 \
 *       -o libcublasLt_stub.so cublasLt_stub.c
 */

/* Placeholder to ensure the .so has at least one symbol */
void __decloud_cublasLt_stub_v1(void) {}
