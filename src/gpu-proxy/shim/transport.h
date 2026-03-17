/*
 * DeCloud GPU Proxy -- Shared Transport Layer
 *
 * Declarations and macros shared by all shim libraries.
 * Each .so includes transport.c directly (compiled into each library).
 *
 * Two modes controlled by TRANSPORT_SHARED_RPC_ONLY:
 *   - Undefined (runtime shim): full TCP/vsock transport
 *   - Defined (driver/NVML shim): prefers shared RPC via runtime shim,
 *     falls back to direct TCP/vsock when LD_PRELOAD is stripped
 */

#ifndef DECLOUD_GPU_PROXY_TRANSPORT_H
#define DECLOUD_GPU_PROXY_TRANSPORT_H

#include <stdint.h>
#include <stddef.h>
#include "../proto/gpu_proxy_proto.h"

/* Logging prefix -- each shim defines TRANSPORT_LOG_PREFIX before including */
#ifndef TRANSPORT_LOG_PREFIX
#define TRANSPORT_LOG_PREFIX "gpu-shim"
#endif

static int g_debug_log = 0;
#define TRANSPORT_LOG(fmt, ...) \
    do { if (g_debug_log) fprintf(stderr, "[" TRANSPORT_LOG_PREFIX "] " fmt "\n", ##__VA_ARGS__); } while(0)

/* Core transport API -- provided by both modes of transport.c */
int transport_ensure_connected(void);
int transport_rpc_call(uint8_t cmd,
                       const void *req_payload, uint32_t req_len,
                       void *resp_buf, uint32_t resp_buf_size,
                       uint32_t *resp_actual_len);
void transport_disconnect(void);

/* Full-mode only -- used by runtime shim's own rpc_call */
int transport_parse_hex_token(const char *hex, uint8_t *out, int len);
int transport_read_exact(int fd, void *buf, size_t len);
int transport_write_exact(int fd, const void *buf, size_t len);

#endif /* DECLOUD_GPU_PROXY_TRANSPORT_H */
