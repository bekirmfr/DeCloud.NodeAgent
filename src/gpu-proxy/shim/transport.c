/*
 * DeCloud GPU Proxy -- Shared Transport Implementation
 *
 * TCP/vsock connection, RPC call, and I/O helpers.
 * Included directly into each shim .so via the build system.
 *
 * Environment variables:
 *   DECLOUD_GPU_PROXY_CID   -- vsock CID (default: VMADDR_CID_HOST = 2)
 *   DECLOUD_GPU_PROXY_PORT  -- port (default: 9999)
 *   DECLOUD_GPU_PROXY_HOST  -- TCP host (default: 192.168.122.1)
 *   DECLOUD_GPU_PROXY_TOKEN -- hex-encoded auth token for TCP
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "transport.h"

/* ================================================================
 * Connection state (per-process, shared across threads)
 * ================================================================ */

static pthread_mutex_t g_transport_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_transport_fd = -1;
static int g_transport_initialized = 0;

/* Error code returned on transport failures (matches cudaErrorNoDevice) */
#define TRANSPORT_ERROR_NO_DEVICE 100

/* ================================================================
 * Helpers
 * ================================================================ */

static int transport_get_env_int(const char *name, int def)
{
    const char *val = getenv(name);
    return val ? atoi(val) : def;
}

int transport_parse_hex_token(const char *hex, uint8_t *out, int len)
{
    if (!hex || strlen(hex) != (size_t)(len * 2)) return -1;
    for (int i = 0; i < len; i++) {
        unsigned int byte;
        if (sscanf(&hex[i * 2], "%02x", &byte) != 1) return -1;
        out[i] = (uint8_t)byte;
    }
    return 0;
}

/* ================================================================
 * I/O -- read/write exactly N bytes
 * ================================================================ */

int transport_read_exact(int fd, void *buf, size_t len)
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

int transport_write_exact(int fd, const void *buf, size_t len)
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
 * Transport connect -- vsock first, TCP fallback
 * ================================================================ */

static int transport_try_tcp(int port)
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
        TRANSPORT_LOG("TCP connect(%s:%d) failed: %s", host, port, strerror(errno));
        close(fd);
        return -1;
    }

    TRANSPORT_LOG("Connected via TCP to %s:%d", host, port);
    return fd;
}

static int transport_try_vsock(int port)
{
    int cid = transport_get_env_int("DECLOUD_GPU_PROXY_CID", VMADDR_CID_HOST);

    int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_vm addr = {
        .svm_family = AF_VSOCK,
        .svm_cid    = (unsigned int)cid,
        .svm_port   = (unsigned int)port,
    };

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        TRANSPORT_LOG("vsock connect(CID=%d port=%d) failed: %s",
                      cid, port, strerror(errno));
        close(fd);
        return -1;
    }

    TRANSPORT_LOG("Connected via vsock (CID=%d port=%d)", cid, port);
    return fd;
}

int transport_ensure_connected(void)
{
    pthread_mutex_lock(&g_transport_lock);
    if (g_transport_fd >= 0) {
        pthread_mutex_unlock(&g_transport_lock);
        return 0;
    }

    int port = transport_get_env_int("DECLOUD_GPU_PROXY_PORT", GPU_PROXY_PORT);
    int is_tcp = 0;

    /* Try vsock first (bare metal), then TCP fallback */
    int fd = transport_try_vsock(port);
    if (fd < 0) {
        fd = transport_try_tcp(port);
        if (fd < 0) {
            TRANSPORT_LOG("All transports failed -- GPU proxy unavailable");
            pthread_mutex_unlock(&g_transport_lock);
            return -1;
        }
        is_tcp = 1;
    }

    g_transport_fd = fd;

    /* Build HELLO with auth token (for TCP) */
    GpuHelloRequest hello = {
        .shim_version = GPU_PROXY_VERSION,
        .pid = (uint32_t)getpid(),
        .auth_token = {0},
    };

    if (is_tcp) {
        const char *tok_hex = getenv("DECLOUD_GPU_PROXY_TOKEN");
        if (tok_hex) {
            transport_parse_hex_token(tok_hex, hello.auth_token, GPU_PROXY_TOKEN_LEN);
        }
    }

    GpuProxyHeader hdr = {
        .magic = GPU_PROXY_MAGIC,
        .version = GPU_PROXY_VERSION,
        .cmd = GPU_CMD_HELLO,
        .payload_len = sizeof(hello),
        .status = 0,
    };

    if (transport_write_exact(fd, &hdr, sizeof(hdr)) < 0 ||
        transport_write_exact(fd, &hello, sizeof(hello)) < 0) {
        TRANSPORT_LOG("Failed to send HELLO");
        close(fd);
        g_transport_fd = -1;
        pthread_mutex_unlock(&g_transport_lock);
        return -1;
    }

    GpuProxyHeader resp_hdr;
    if (transport_read_exact(fd, &resp_hdr, sizeof(resp_hdr)) == 0) {
        if (resp_hdr.status != 0) {
            TRANSPORT_LOG("HELLO rejected (status=%d) -- auth failed?", resp_hdr.status);
            close(fd);
            g_transport_fd = -1;
            pthread_mutex_unlock(&g_transport_lock);
            return -1;
        }
        if (resp_hdr.payload_len >= sizeof(GpuHelloResponse)) {
            GpuHelloResponse resp;
            transport_read_exact(fd, &resp, sizeof(resp));
            TRANSPORT_LOG("Connected to GPU proxy (v%u, %u devices, %s)",
                          resp.daemon_version, resp.device_count,
                          is_tcp ? "TCP" : "vsock");
        }
    }

    g_transport_initialized = 1;
    pthread_mutex_unlock(&g_transport_lock);
    return 0;
}

/* ================================================================
 * RPC call -- send request, receive response
 * ================================================================ */

int transport_rpc_call(uint8_t cmd,
                       const void *req_payload, uint32_t req_len,
                       void *resp_buf, uint32_t resp_buf_size,
                       uint32_t *resp_actual_len)
{
    if (transport_ensure_connected() < 0)
        return TRANSPORT_ERROR_NO_DEVICE;

    pthread_mutex_lock(&g_transport_lock);

    GpuProxyHeader hdr = {
        .magic = GPU_PROXY_MAGIC,
        .version = GPU_PROXY_VERSION,
        .cmd = cmd,
        .payload_len = req_len,
        .status = 0,
    };

    if (transport_write_exact(g_transport_fd, &hdr, sizeof(hdr)) < 0) goto err;
    if (req_len > 0 && req_payload) {
        if (transport_write_exact(g_transport_fd, req_payload, req_len) < 0) goto err;
    }

    /* Read response header */
    GpuProxyHeader resp_hdr;
    if (transport_read_exact(g_transport_fd, &resp_hdr, sizeof(resp_hdr)) < 0) goto err;

    if (resp_hdr.magic != GPU_PROXY_MAGIC) {
        TRANSPORT_LOG("bad response magic 0x%08x", resp_hdr.magic);
        goto err;
    }

    /* Read response payload */
    if (resp_hdr.payload_len > 0) {
        if (resp_buf && resp_hdr.payload_len <= resp_buf_size) {
            if (transport_read_exact(g_transport_fd, resp_buf, resp_hdr.payload_len) < 0)
                goto err;
        } else {
            /* Drain excess data */
            char drain[4096];
            uint32_t remaining = resp_hdr.payload_len;
            while (remaining > 0) {
                uint32_t chunk = remaining < sizeof(drain) ? remaining : sizeof(drain);
                if (transport_read_exact(g_transport_fd, drain, chunk) < 0) goto err;
                remaining -= chunk;
            }
        }
    }

    if (resp_actual_len) *resp_actual_len = resp_hdr.payload_len;
    pthread_mutex_unlock(&g_transport_lock);
    return resp_hdr.status;

err:
    close(g_transport_fd);
    g_transport_fd = -1;
    g_transport_initialized = 0;
    pthread_mutex_unlock(&g_transport_lock);
    return TRANSPORT_ERROR_NO_DEVICE;
}

/* ================================================================
 * Disconnect -- send GOODBYE and close
 * ================================================================ */

void transport_disconnect(void)
{
    pthread_mutex_lock(&g_transport_lock);
    if (g_transport_fd >= 0) {
        GpuProxyHeader hdr = {
            .magic = GPU_PROXY_MAGIC,
            .version = GPU_PROXY_VERSION,
            .cmd = GPU_CMD_GOODBYE,
            .payload_len = 0,
            .status = 0,
        };
        write(g_transport_fd, &hdr, sizeof(hdr));
        close(g_transport_fd);
        g_transport_fd = -1;
    }
    pthread_mutex_unlock(&g_transport_lock);
}
