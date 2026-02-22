/*
 * DeCloud GPU Proxy Daemon
 *
 * Host-side service that listens on virtio-vsock and proxies CUDA Runtime
 * API calls from guest VMs to the real GPU.
 *
 * Each VM connects with its unique CID. The daemon creates a separate CUDA
 * context per connection for isolation.
 *
 * Build: gcc -o gpu-proxy-daemon gpu_proxy_daemon.c -lcuda -lcudart -lpthread
 * Run:   gpu-proxy-daemon [-p port] [-v]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <errno.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>

#include <cuda_runtime.h>

#include "../proto/gpu_proxy_proto.h"

/* ================================================================
 * Logging
 * ================================================================ */

static int g_verbose = 0;
static volatile sig_atomic_t g_running = 1;

#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[gpu-proxy] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) \
    fprintf(stderr, "[gpu-proxy] ERROR: " fmt "\n", ##__VA_ARGS__)
#define LOG_DBG(fmt, ...) \
    do { if (g_verbose) fprintf(stdout, "[gpu-proxy] DBG: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ================================================================
 * I/O helpers — read/write exactly N bytes
 * ================================================================ */

static int read_exact(int fd, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, (char *)buf + done, len - done);
        if (n <= 0) {
            if (n == 0) return -1;   /* EOF */
            if (errno == EINTR) continue;
            return -1;
        }
        done += n;
    }
    return 0;
}

static int write_exact(int fd, const void *buf, size_t len)
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
 * Send a response header (optionally followed by payload)
 * ================================================================ */

static int send_response(int fd, uint8_t cmd, int32_t status,
                         const void *payload, uint32_t payload_len)
{
    GpuProxyHeader hdr = {
        .magic       = GPU_PROXY_MAGIC,
        .version     = GPU_PROXY_VERSION,
        .cmd         = cmd,
        .flags       = 0,
        .payload_len = payload_len,
        .status      = status,
    };
    if (write_exact(fd, &hdr, sizeof(hdr)) < 0) return -1;
    if (payload_len > 0 && payload) {
        if (write_exact(fd, payload, payload_len) < 0) return -1;
    }
    return 0;
}

/* ================================================================
 * Command handlers
 * ================================================================ */

static int handle_hello(int fd, const void *payload, uint32_t len)
{
    GpuHelloRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_HELLO, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    LOG_INFO("HELLO from guest PID %u (shim v%u), %d GPU(s) available",
             req.pid, req.shim_version, count);

    GpuHelloResponse resp = {
        .daemon_version = GPU_PROXY_VERSION,
        .device_count   = (uint32_t)count,
    };
    return send_response(fd, GPU_CMD_HELLO, (int32_t)err,
                         &resp, sizeof(resp));
}

static int handle_get_device_count(int fd)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    GpuGetDeviceCountResponse resp = { .count = count };
    return send_response(fd, GPU_CMD_GET_DEVICE_COUNT, (int32_t)err,
                         &resp, sizeof(resp));
}

static int handle_get_device_properties(int fd, const void *payload, uint32_t len)
{
    GpuGetDevicePropertiesRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_GET_DEVICE_PROPERTIES, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    struct cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, req.device);

    GpuDeviceProperties resp;
    memset(&resp, 0, sizeof(resp));

    if (err == cudaSuccess) {
        strncpy(resp.name, props.name, sizeof(resp.name) - 1);
        resp.total_global_mem                = props.totalGlobalMem;
        resp.shared_mem_per_block            = props.sharedMemPerBlock;
        resp.regs_per_block                  = props.regsPerBlock;
        resp.warp_size                       = props.warpSize;
        resp.max_threads_per_block           = props.maxThreadsPerBlock;
        resp.max_threads_dim[0]              = props.maxThreadsDim[0];
        resp.max_threads_dim[1]              = props.maxThreadsDim[1];
        resp.max_threads_dim[2]              = props.maxThreadsDim[2];
        resp.max_grid_size[0]                = props.maxGridSize[0];
        resp.max_grid_size[1]                = props.maxGridSize[1];
        resp.max_grid_size[2]                = props.maxGridSize[2];
        resp.clock_rate                      = props.clockRate;
        resp.multi_processor_count           = props.multiProcessorCount;
        resp.major                           = props.major;
        resp.minor                           = props.minor;
        resp.max_threads_per_multiprocessor  = props.maxThreadsPerMultiprocessor;
        resp.memory_clock_rate               = props.memoryClockRate;
        resp.memory_bus_width                = props.memoryBusWidth;
        resp.l2_cache_size                   = props.l2CacheSize;
    }

    return send_response(fd, GPU_CMD_GET_DEVICE_PROPERTIES, (int32_t)err,
                         &resp, sizeof(resp));
}

static int handle_set_device(int fd, const void *payload, uint32_t len)
{
    GpuSetDeviceRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_SET_DEVICE, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    cudaError_t err = cudaSetDevice(req.device);
    return send_response(fd, GPU_CMD_SET_DEVICE, (int32_t)err, NULL, 0);
}

static int handle_malloc(int fd, const void *payload, uint32_t len)
{
    GpuMallocRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_MALLOC, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    void *devptr = NULL;
    cudaError_t err = cudaMalloc(&devptr, (size_t)req.size);

    LOG_DBG("cudaMalloc(%lu) -> %p (err=%d)", (unsigned long)req.size, devptr, err);

    GpuMallocResponse resp = {
        .device_ptr = (uint64_t)(uintptr_t)devptr,
    };
    return send_response(fd, GPU_CMD_MALLOC, (int32_t)err,
                         &resp, sizeof(resp));
}

static int handle_free(int fd, const void *payload, uint32_t len)
{
    GpuFreeRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_FREE, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    void *devptr = (void *)(uintptr_t)req.device_ptr;
    cudaError_t err = cudaFree(devptr);

    LOG_DBG("cudaFree(%p) -> err=%d", devptr, err);

    return send_response(fd, GPU_CMD_FREE, (int32_t)err, NULL, 0);
}

static int handle_memcpy(int fd, const void *payload, uint32_t payload_len)
{
    if (payload_len < sizeof(GpuMemcpyRequest)) {
        return send_response(fd, GPU_CMD_MEMCPY, -1, NULL, 0);
    }

    GpuMemcpyRequest req;
    memcpy(&req, payload, sizeof(req));

    const uint8_t *extra_data = (const uint8_t *)payload + sizeof(req);
    uint32_t extra_len = payload_len - sizeof(req);

    cudaError_t err;

    switch (req.kind) {
    case GPU_MEMCPY_HOST_TO_DEVICE: {
        /* Shim sent the host data after the request struct */
        if (extra_len < req.count) {
            return send_response(fd, GPU_CMD_MEMCPY, -1, NULL, 0);
        }
        void *dst = (void *)(uintptr_t)req.dst;
        err = cudaMemcpy(dst, extra_data, (size_t)req.count,
                         cudaMemcpyHostToDevice);
        LOG_DBG("cudaMemcpy H2D %lu bytes -> %p (err=%d)",
                (unsigned long)req.count, dst, err);
        return send_response(fd, GPU_CMD_MEMCPY, (int32_t)err, NULL, 0);
    }

    case GPU_MEMCPY_DEVICE_TO_HOST: {
        /* Allocate host buffer, copy from device, send back */
        void *src = (void *)(uintptr_t)req.src;
        void *buf = malloc((size_t)req.count);
        if (!buf) {
            return send_response(fd, GPU_CMD_MEMCPY, -1, NULL, 0);
        }
        err = cudaMemcpy(buf, src, (size_t)req.count,
                         cudaMemcpyDeviceToHost);
        LOG_DBG("cudaMemcpy D2H %lu bytes from %p (err=%d)",
                (unsigned long)req.count, src, err);

        int rc;
        if (err == cudaSuccess) {
            rc = send_response(fd, GPU_CMD_MEMCPY, 0,
                               buf, (uint32_t)req.count);
        } else {
            rc = send_response(fd, GPU_CMD_MEMCPY, (int32_t)err, NULL, 0);
        }
        free(buf);
        return rc;
    }

    case GPU_MEMCPY_DEVICE_TO_DEVICE: {
        void *dst = (void *)(uintptr_t)req.dst;
        void *src = (void *)(uintptr_t)req.src;
        err = cudaMemcpy(dst, src, (size_t)req.count,
                         cudaMemcpyDeviceToDevice);
        return send_response(fd, GPU_CMD_MEMCPY, (int32_t)err, NULL, 0);
    }

    default:
        return send_response(fd, GPU_CMD_MEMCPY, -1, NULL, 0);
    }
}

static int handle_memset(int fd, const void *payload, uint32_t len)
{
    GpuMemsetRequest req;
    if (len < sizeof(req)) {
        return send_response(fd, GPU_CMD_MEMSET, -1, NULL, 0);
    }
    memcpy(&req, payload, sizeof(req));

    void *devptr = (void *)(uintptr_t)req.device_ptr;
    cudaError_t err = cudaMemset(devptr, req.value, (size_t)req.count);
    return send_response(fd, GPU_CMD_MEMSET, (int32_t)err, NULL, 0);
}

static int handle_device_synchronize(int fd)
{
    cudaError_t err = cudaDeviceSynchronize();
    return send_response(fd, GPU_CMD_DEVICE_SYNCHRONIZE, (int32_t)err, NULL, 0);
}

/* ================================================================
 * Per-connection handler (one thread per VM)
 * ================================================================ */

typedef struct {
    int fd;
    unsigned int peer_cid;
} ConnectionCtx;

static void *connection_handler(void *arg)
{
    ConnectionCtx *ctx = (ConnectionCtx *)arg;
    int fd = ctx->fd;
    unsigned int cid = ctx->peer_cid;

    LOG_INFO("VM CID %u connected", cid);

    /* Allocate a reusable payload buffer */
    size_t buf_cap = 4096;
    void *buf = malloc(buf_cap);
    if (!buf) {
        LOG_ERR("CID %u: malloc failed", cid);
        goto done;
    }

    while (g_running) {
        /* Read request header */
        GpuProxyHeader hdr;
        if (read_exact(fd, &hdr, sizeof(hdr)) < 0) {
            LOG_DBG("CID %u: connection closed", cid);
            break;
        }

        if (hdr.magic != GPU_PROXY_MAGIC) {
            LOG_ERR("CID %u: bad magic 0x%08x", cid, hdr.magic);
            break;
        }

        if (hdr.payload_len > GPU_PROXY_MAX_PAYLOAD) {
            LOG_ERR("CID %u: payload too large (%u bytes)", cid, hdr.payload_len);
            break;
        }

        /* Read payload */
        if (hdr.payload_len > 0) {
            if (hdr.payload_len > buf_cap) {
                buf_cap = hdr.payload_len;
                void *newbuf = realloc(buf, buf_cap);
                if (!newbuf) {
                    LOG_ERR("CID %u: realloc failed for %u bytes", cid, hdr.payload_len);
                    break;
                }
                buf = newbuf;
            }
            if (read_exact(fd, buf, hdr.payload_len) < 0) {
                LOG_ERR("CID %u: failed to read payload", cid);
                break;
            }
        }

        /* Dispatch */
        int rc = 0;
        switch (hdr.cmd) {
        case GPU_CMD_HELLO:
            rc = handle_hello(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_GET_DEVICE_COUNT:
            rc = handle_get_device_count(fd);
            break;
        case GPU_CMD_GET_DEVICE_PROPERTIES:
            rc = handle_get_device_properties(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_SET_DEVICE:
            rc = handle_set_device(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_MALLOC:
            rc = handle_malloc(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_FREE:
            rc = handle_free(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_MEMCPY:
            rc = handle_memcpy(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_MEMSET:
            rc = handle_memset(fd, buf, hdr.payload_len);
            break;
        case GPU_CMD_DEVICE_SYNCHRONIZE:
            rc = handle_device_synchronize(fd);
            break;
        case GPU_CMD_GOODBYE:
            LOG_INFO("CID %u: graceful disconnect", cid);
            send_response(fd, GPU_CMD_GOODBYE, 0, NULL, 0);
            goto done;
        default:
            LOG_ERR("CID %u: unknown command 0x%02x", cid, hdr.cmd);
            send_response(fd, hdr.cmd, -1, NULL, 0);
            break;
        }

        if (rc < 0) {
            LOG_ERR("CID %u: write error, disconnecting", cid);
            break;
        }
    }

done:
    free(buf);
    close(fd);
    LOG_INFO("VM CID %u disconnected", cid);
    free(ctx);
    return NULL;
}

/* ================================================================
 * Main: vsock listener
 * ================================================================ */

static void sig_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [-p port] [-v]\n", prog);
    fprintf(stderr, "  -p port   vsock port to listen on (default: %d)\n", GPU_PROXY_PORT);
    fprintf(stderr, "  -v        verbose logging\n");
    exit(1);
}

int main(int argc, char **argv)
{
    int port = GPU_PROXY_PORT;
    int opt;

    while ((opt = getopt(argc, argv, "p:v")) != -1) {
        switch (opt) {
        case 'p': port = atoi(optarg); break;
        case 'v': g_verbose = 1; break;
        default: usage(argv[0]);
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    /* Verify CUDA is available */
    int dev_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&dev_count);
    if (cerr != cudaSuccess || dev_count == 0) {
        LOG_ERR("No CUDA devices found (cudaGetDeviceCount=%d, err=%s)",
                dev_count, cudaGetErrorString(cerr));
        return 1;
    }
    LOG_INFO("Found %d CUDA device(s)", dev_count);

    for (int i = 0; i < dev_count; i++) {
        struct cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        LOG_INFO("  GPU %d: %s (%ldMB, compute %d.%d)",
                 i, props.name,
                 (long)(props.totalGlobalMem / (1024 * 1024)),
                 props.major, props.minor);
    }

    /* Create vsock listener */
    int listen_fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        LOG_ERR("socket(AF_VSOCK) failed: %s", strerror(errno));
        return 1;
    }

    struct sockaddr_vm addr = {
        .svm_family = AF_VSOCK,
        .svm_cid    = VMADDR_CID_ANY,  /* Accept from any VM */
        .svm_port   = (unsigned int)port,
    };

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        LOG_ERR("bind(vsock port %d) failed: %s", port, strerror(errno));
        close(listen_fd);
        return 1;
    }

    if (listen(listen_fd, 16) < 0) {
        LOG_ERR("listen() failed: %s", strerror(errno));
        close(listen_fd);
        return 1;
    }

    LOG_INFO("Listening on vsock port %d (CID=any)", port);

    /* Accept loop */
    while (g_running) {
        struct sockaddr_vm peer;
        socklen_t peer_len = sizeof(peer);
        int client_fd = accept(listen_fd, (struct sockaddr *)&peer, &peer_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            LOG_ERR("accept() failed: %s", strerror(errno));
            break;
        }

        ConnectionCtx *ctx = malloc(sizeof(ConnectionCtx));
        if (!ctx) {
            LOG_ERR("malloc failed for connection context");
            close(client_fd);
            continue;
        }
        ctx->fd = client_fd;
        ctx->peer_cid = peer.svm_cid;

        pthread_t tid;
        if (pthread_create(&tid, NULL, connection_handler, ctx) != 0) {
            LOG_ERR("pthread_create failed: %s", strerror(errno));
            close(client_fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }

    LOG_INFO("Shutting down");
    close(listen_fd);
    return 0;
}
