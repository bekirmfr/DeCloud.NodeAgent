/*
 * DeCloud GPU Proxy -- Shared Transport Implementation
 *
 * Included directly into each shim .so via #include.
 *
 * Two modes:
 *   1. Full mode (runtime shim) — owns the TCP/vsock connection to the
 *      daemon and exports decloud_shared_rpc_call() for other shims.
 *   2. Shared-preferred mode (driver shim, NVML shim) — tries to
 *      delegate RPC calls through the runtime shim's exported function
 *      (single-channel).  If the runtime shim is not loaded (e.g.
 *      Ollama strips LD_PRELOAD from runner subprocesses), falls back
 *      to opening its own direct TCP/vsock connection.
 *      Define TRANSPORT_SHARED_RPC_ONLY before including.
 *
 * Both modes provide:
 *   - transport_getenv()       config-file-aware getenv
 *   - transport_rpc_call()     send RPC and receive response
 *   - transport_ensure_connected()
 *   - transport_disconnect()
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
#include <netinet/tcp.h>
#include <arpa/inet.h>

#include "transport.h"

/* Error code returned on transport failures (matches cudaErrorNoDevice) */
#define TRANSPORT_ERROR_NO_DEVICE 100

/* ================================================================
 * Config file reading (shared by all modes)
 * ================================================================
 *
 * Ollama's runner subprocess strips non-OLLAMA_ env vars, so
 * DECLOUD_GPU_PROXY_* are lost.  This reads /etc/decloud/gpu-proxy.env
 * as a fallback.
 *
 * CRITICAL: We MUST NOT call setenv() here.  Go's runtime runs
 * multiple OS threads and setenv() modifies the global `environ`
 * array without synchronization, causing SIGSEGV in cgo context.
 */
#define GPU_PROXY_CONFIG_FILE "/etc/decloud/gpu-proxy.env"

#define MAX_CONFIG_ENTRIES 16
#define MAX_CONFIG_KEY_LEN 64
#define MAX_CONFIG_VAL_LEN 256

typedef struct {
    char key[MAX_CONFIG_KEY_LEN];
    char val[MAX_CONFIG_VAL_LEN];
} ConfigEntry;

static ConfigEntry g_config_entries[MAX_CONFIG_ENTRIES];
static int g_config_count = 0;
static int g_config_file_loaded = 0;

static void transport_load_config_file(void)
{
    if (g_config_file_loaded) return;
    g_config_file_loaded = 1;

    FILE *f = fopen(GPU_PROXY_CONFIG_FILE, "r");
    if (!f) return;

    char line[512];
    while (fgets(line, sizeof(line), f) && g_config_count < MAX_CONFIG_ENTRIES) {
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

        ConfigEntry *ce = &g_config_entries[g_config_count++];
        strncpy(ce->key, key, MAX_CONFIG_KEY_LEN - 1);
        ce->key[MAX_CONFIG_KEY_LEN - 1] = '\0';
        strncpy(ce->val, val, MAX_CONFIG_VAL_LEN - 1);
        ce->val[MAX_CONFIG_VAL_LEN - 1] = '\0';
    }
    fclose(f);
}

/* Thread-safe getenv: checks real env first, then config file.
 * NEVER calls setenv(). */
static const char *transport_getenv(const char *name)
{
    const char *val = getenv(name);
    if (val) return val;

    transport_load_config_file();
    for (int i = 0; i < g_config_count; i++) {
        if (strcmp(g_config_entries[i].key, name) == 0)
            return g_config_entries[i].val;
    }
    return NULL;
}

static int transport_get_env_int(const char *name, int def)
{
    const char *val = transport_getenv(name);
    return val ? atoi(val) : def;
}

/* ================================================================
 * Shared RPC delegation (TRANSPORT_SHARED_RPC_ONLY mode)
 *
 * When the runtime shim is loaded via LD_PRELOAD, the driver and
 * NVML shims can delegate all RPC through it to share a single
 * TCP connection.  This is the preferred path for unified stream/
 * event tables on the daemon side.
 *
 * When LD_PRELOAD is stripped (Ollama runner subprocesses), the
 * runtime shim is absent and dlsym() fails.  In that case we fall
 * through to the direct connection code below.
 * ================================================================ */
#ifdef TRANSPORT_SHARED_RPC_ONLY

#include <dlfcn.h>

typedef int (*shared_rpc_fn_t)(uint8_t cmd,
                                const void *req_payload, uint32_t req_len,
                                void *resp_buf, uint32_t resp_buf_size,
                                uint32_t *resp_actual_len);
static shared_rpc_fn_t g_transport_shared_rpc = NULL;
static int g_shared_rpc_checked = 0;

/* Returns 1 if shared RPC is available, 0 otherwise. */
static int transport_try_shared_rpc(void)
{
    if (g_transport_shared_rpc) return 1;
    if (g_shared_rpc_checked) return 0;
    g_shared_rpc_checked = 1;

    g_transport_shared_rpc = (shared_rpc_fn_t)dlsym(RTLD_DEFAULT,
                                                      "decloud_shared_rpc_call");
    if (g_transport_shared_rpc) {
        TRANSPORT_LOG("Using runtime shim's shared RPC (single-channel mode)");
        return 1;
    }
    TRANSPORT_LOG("Runtime shim not loaded — falling back to direct connection "
                  "(Ollama runner may have stripped LD_PRELOAD)");
    return 0;
}

#endif /* TRANSPORT_SHARED_RPC_ONLY */

/* ================================================================
 * Direct connection code (always compiled)
 *
 * In full mode (runtime shim): this is the only transport.
 * In shared-preferred mode: used as fallback when the runtime
 * shim is not loaded (LD_PRELOAD stripped by Ollama).
 * ================================================================ */

static pthread_mutex_t g_transport_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_transport_fd = -1;
static int g_transport_initialized = 0;

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
    const char *host = transport_getenv("DECLOUD_GPU_PROXY_HOST");
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

    int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    int quickack = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));

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
    transport_load_config_file();

#ifdef TRANSPORT_SHARED_RPC_ONLY
    /* Prefer shared RPC when runtime shim is available */
    if (transport_try_shared_rpc())
        return 0;
    /* Runtime shim not loaded — fall through to direct connection */
#endif

    pthread_mutex_lock(&g_transport_lock);
    if (g_transport_fd >= 0) {
        pthread_mutex_unlock(&g_transport_lock);
        return 0;
    }

    const char *transport_mode = transport_getenv("DECLOUD_GPU_PROXY_TRANSPORT");
    int port = transport_get_env_int("DECLOUD_GPU_PROXY_PORT", GPU_PROXY_PORT);
    int try_vsock = 1, try_tcp = 1;

    if (transport_mode) {
        if (strcmp(transport_mode, "tcp") == 0)   try_vsock = 0;
        if (strcmp(transport_mode, "vsock") == 0)  try_tcp = 0;
    }

    int fd = -1;
    int is_tcp = 0;

    if (try_tcp) {
        fd = transport_try_tcp(port);
        if (fd >= 0) is_tcp = 1;
    }
    if (fd < 0 && try_vsock) {
        fd = transport_try_vsock(port);
    }
    if (fd < 0) {
        TRANSPORT_LOG("All transports failed -- GPU proxy unavailable");
        pthread_mutex_unlock(&g_transport_lock);
        return -1;
    }

    g_transport_fd = fd;

    GpuHelloRequest hello = {
        .shim_version = GPU_PROXY_VERSION,
        .pid = (uint32_t)getpid(),
        .auth_token = {0},
    };

    if (is_tcp) {
        const char *tok_hex = transport_getenv("DECLOUD_GPU_PROXY_TOKEN");
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
#ifdef TRANSPORT_SHARED_RPC_ONLY
    /* Prefer shared RPC when runtime shim is available */
    if (g_transport_shared_rpc || transport_try_shared_rpc()) {
        return g_transport_shared_rpc(cmd, req_payload, req_len,
                                       resp_buf, resp_buf_size,
                                       resp_actual_len);
    }
    /* Runtime shim not loaded — fall through to direct connection */
#endif

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

    int qa = 1;
    setsockopt(g_transport_fd, IPPROTO_TCP, TCP_QUICKACK, &qa, sizeof(qa));

    GpuProxyHeader resp_hdr;
    if (transport_read_exact(g_transport_fd, &resp_hdr, sizeof(resp_hdr)) < 0) goto err;
    if (resp_hdr.magic != GPU_PROXY_MAGIC) {
        TRANSPORT_LOG("bad response magic 0x%08x", resp_hdr.magic);
        goto err;
    }

    if (resp_hdr.payload_len > 0) {
        if (resp_buf && resp_hdr.payload_len <= resp_buf_size) {
            if (transport_read_exact(g_transport_fd, resp_buf, resp_hdr.payload_len) < 0)
                goto err;
        } else {
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
#ifdef TRANSPORT_SHARED_RPC_ONLY
    /* If using shared RPC, the runtime shim owns the connection */
    if (g_transport_shared_rpc)
        return;
#endif

    pthread_mutex_lock(&g_transport_lock);
    if (g_transport_fd >= 0) {
        GpuProxyHeader hdr = {
            .magic = GPU_PROXY_MAGIC,
            .version = GPU_PROXY_VERSION,
            .cmd = GPU_CMD_GOODBYE,
            .payload_len = 0,
            .status = 0,
        };
        (void)!write(g_transport_fd, &hdr, sizeof(hdr));
        close(g_transport_fd);
        g_transport_fd = -1;
    }
    pthread_mutex_unlock(&g_transport_lock);
}
