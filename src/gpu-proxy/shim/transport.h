/*
 * DeCloud GPU Proxy — Shared Transport Layer
 *
 * Connection management, RPC call, and I/O helpers shared by all shim
 * libraries (Runtime API shim, Driver API shim, NVML shim).
 *
 * Each .so includes transport.c directly (compiled into each library)
 * since dlopen'd libraries can't have cross-.so dependencies.
 */

#ifndef DECLOUD_GPU_PROXY_TRANSPORT_H
#define DECLOUD_GPU_PROXY_TRANSPORT_H

#include <stdint.h>
#include <stddef.h>
#include "../proto/gpu_proxy_proto.h"

/* Logging prefix — each shim defines TRANSPORT_LOG_PREFIX before including */
#ifndef TRANSPORT_LOG_PREFIX
#define TRANSPORT_LOG_PREFIX "gpu-shim"
#endif

#define TRANSPORT_LOG(fmt, ...) \
    fprintf(stderr, "[" TRANSPORT_LOG_PREFIX "] " fmt "\n", ##__VA_ARGS__)

/*
 * Ensure the shared connection to the daemon is established.
 * Thread-safe (uses internal mutex). Returns 0 on success, -1 on failure.
 */
int transport_ensure_connected(void);

/*
 * Send an RPC request and receive the response.
 *
 * cmd:            GpuProxyCmd command ID
 * req_payload:    Request payload bytes (NULL if none)
 * req_len:        Request payload length
 * resp_buf:       Buffer for response payload (NULL to discard)
 * resp_buf_size:  Size of resp_buf
 * resp_actual_len: If non-NULL, receives actual response payload length
 *
 * Returns: status field from daemon's response header (0 = success).
 *          Returns 100 (cudaErrorNoDevice) on transport failure.
 */
int transport_rpc_call(uint8_t cmd,
                       const void *req_payload, uint32_t req_len,
                       void *resp_buf, uint32_t resp_buf_size,
                       uint32_t *resp_actual_len);

/*
 * Close the daemon connection and clean up.
 * Sends GPU_CMD_GOODBYE before disconnecting.
 */
void transport_disconnect(void);

/*
 * Parse hex string into byte array.
 * Returns 0 on success, -1 on invalid input.
 */
int transport_parse_hex_token(const char *hex, uint8_t *out, int len);

/*
 * Exact I/O helpers — read/write exactly len bytes.
 * Return 0 on success, -1 on failure.
 */
int transport_read_exact(int fd, void *buf, size_t len);
int transport_write_exact(int fd, const void *buf, size_t len);

#endif /* DECLOUD_GPU_PROXY_TRANSPORT_H */
