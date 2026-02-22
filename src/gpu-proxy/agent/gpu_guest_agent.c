/*
 * DeCloud GPU Guest Agent
 *
 * Lightweight metering agent that runs inside passthrough GPU VMs.
 * Periodically polls nvidia-smi for GPU utilization metrics and
 * pushes them to the host GPU proxy daemon over virtio-vsock.
 *
 * The host daemon aggregates these stats identically to proxy-mode
 * stats, providing a unified metering/billing interface regardless
 * of whether the VM uses GPU passthrough or GPU proxy mode.
 *
 * No CUDA dependency — only needs nvidia-smi (comes with the driver).
 *
 * Build:  gcc -Wall -Wextra -O2 -o decloud-gpu-agent gpu_guest_agent.c -lpthread
 * Run:    decloud-gpu-agent [-i interval_sec] [-v]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>

#include "../proto/gpu_proxy_proto.h"

/* ================================================================
 * Logging
 * ================================================================ */

static int g_verbose = 0;
static volatile sig_atomic_t g_running = 1;

#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[gpu-agent] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) \
    fprintf(stderr, "[gpu-agent] ERROR: " fmt "\n", ##__VA_ARGS__)
#define LOG_DBG(fmt, ...) \
    do { if (g_verbose) fprintf(stdout, "[gpu-agent] DBG: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ================================================================
 * I/O helpers
 * ================================================================ */

static int read_exact(int fd, void *buf, size_t len)
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
 * Send a request (header + optional payload)
 * ================================================================ */

static int send_request(int fd, uint8_t cmd,
                        const void *payload, uint32_t payload_len)
{
    GpuProxyHeader hdr = {
        .magic       = GPU_PROXY_MAGIC,
        .version     = GPU_PROXY_VERSION,
        .cmd         = cmd,
        .flags       = 0,
        .payload_len = payload_len,
        .status      = 0,
    };
    if (write_exact(fd, &hdr, sizeof(hdr)) < 0) return -1;
    if (payload_len > 0 && payload) {
        if (write_exact(fd, payload, payload_len) < 0) return -1;
    }
    return 0;
}

/* Read a response header */
static int recv_response(int fd, GpuProxyHeader *hdr)
{
    if (read_exact(fd, hdr, sizeof(*hdr)) < 0) return -1;
    if (hdr->magic != GPU_PROXY_MAGIC) {
        LOG_ERR("bad magic in response: 0x%08x", hdr->magic);
        return -1;
    }
    /* Drain any payload we don't need */
    if (hdr->payload_len > 0) {
        char drain[256];
        uint32_t left = hdr->payload_len;
        while (left > 0) {
            size_t chunk = left < sizeof(drain) ? left : sizeof(drain);
            if (read_exact(fd, drain, chunk) < 0) return -1;
            left -= chunk;
        }
    }
    return 0;
}

/* ================================================================
 * nvidia-smi polling
 *
 * Parse a single CSV line from nvidia-smi with the fields:
 *   memory.used, memory.total, utilization.gpu, utilization.memory,
 *   temperature.gpu, fan.speed, power.draw, power.limit
 * ================================================================ */

static int poll_nvidia_smi(GpuReportUsageStats *stats)
{
    FILE *fp = popen(
        "nvidia-smi --query-gpu="
        "memory.used,memory.total,"
        "utilization.gpu,utilization.memory,"
        "temperature.gpu,fan.speed,"
        "power.draw,power.limit"
        " --format=csv,noheader,nounits 2>/dev/null",
        "r");
    if (!fp) {
        LOG_ERR("failed to run nvidia-smi: %s", strerror(errno));
        return -1;
    }

    char line[512];
    int ok = 0;
    if (fgets(line, sizeof(line), fp)) {
        /* Parse CSV: "1234, 24576, 45, 23, 62, 30, 250.50, 350.00" */
        double mem_used_mib = 0, mem_total_mib = 0;
        double gpu_util = 0, mem_util = 0;
        double temp = 0, fan = 0;
        double power_draw = 0, power_limit = 0;

        int n = sscanf(line,
                       "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf",
                       &mem_used_mib, &mem_total_mib,
                       &gpu_util, &mem_util,
                       &temp, &fan,
                       &power_draw, &power_limit);
        if (n >= 4) {
            stats->memory_allocated = (uint64_t)(mem_used_mib * 1024 * 1024);
            stats->memory_total     = (uint64_t)(mem_total_mib * 1024 * 1024);
            stats->gpu_utilization  = (uint32_t)gpu_util;
            stats->mem_utilization  = (uint32_t)mem_util;
            stats->temperature_c    = (int32_t)temp;
            stats->fan_speed_pct    = (uint32_t)fan;
            stats->power_usage_mw   = (uint32_t)(power_draw * 1000.0);
            stats->power_limit_mw   = (uint32_t)(power_limit * 1000.0);

            /* Track peak */
            if (stats->memory_allocated > stats->peak_memory)
                stats->peak_memory = stats->memory_allocated;

            ok = 1;
        } else {
            LOG_ERR("nvidia-smi parse failed (got %d fields): %s", n, line);
        }
    }

    pclose(fp);
    return ok ? 0 : -1;
}

/* ================================================================
 * vsock connection
 * ================================================================ */

static int connect_vsock(unsigned int cid, unsigned int port)
{
    int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (fd < 0) {
        LOG_ERR("socket(AF_VSOCK) failed: %s", strerror(errno));
        return -1;
    }

    struct sockaddr_vm addr = {
        .svm_family = AF_VSOCK,
        .svm_cid    = cid,
        .svm_port   = port,
    };

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        LOG_ERR("connect(vsock CID=%u port=%u) failed: %s",
                cid, port, strerror(errno));
        close(fd);
        return -1;
    }

    return fd;
}

/* ================================================================
 * Handshake: send HELLO with conn_mode=METER
 * ================================================================ */

static int do_handshake(int fd)
{
    GpuHelloRequest req = {
        .shim_version = GPU_PROXY_VERSION,
        .pid          = (uint32_t)getpid(),
        .conn_mode    = GPU_CONN_METER,
    };

    if (send_request(fd, GPU_CMD_HELLO, &req, sizeof(req)) < 0) {
        LOG_ERR("failed to send HELLO");
        return -1;
    }

    GpuProxyHeader resp_hdr;
    if (recv_response(fd, &resp_hdr) < 0) {
        LOG_ERR("failed to read HELLO response");
        return -1;
    }

    if (resp_hdr.status != 0) {
        LOG_ERR("HELLO rejected: status=%d", resp_hdr.status);
        return -1;
    }

    LOG_INFO("Handshake OK (mode=METER)");
    return 0;
}

/* ================================================================
 * Signal handler
 * ================================================================ */

static void sig_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

/* Monotonic clock helper */
static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ================================================================
 * Main
 * ================================================================ */

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-c host_cid] [-p port] [-i interval_sec] [-v]\n"
            "  -c host_cid    vsock CID of the host (default: 2 = VMADDR_CID_HOST)\n"
            "  -p port        vsock port (default: %d)\n"
            "  -i interval    reporting interval in seconds (default: %d)\n"
            "  -v             verbose logging\n",
            prog, GPU_PROXY_PORT, GPU_AGENT_REPORT_INTERVAL_SEC);
    exit(1);
}

int main(int argc, char **argv)
{
    unsigned int host_cid = 2;  /* VMADDR_CID_HOST */
    unsigned int port = GPU_PROXY_PORT;
    int interval_sec = GPU_AGENT_REPORT_INTERVAL_SEC;
    int opt;

    while ((opt = getopt(argc, argv, "c:p:i:v")) != -1) {
        switch (opt) {
        case 'c': host_cid = (unsigned int)atoi(optarg); break;
        case 'p': port = (unsigned int)atoi(optarg); break;
        case 'i': interval_sec = atoi(optarg); break;
        case 'v': g_verbose = 1; break;
        default:  usage(argv[0]);
        }
    }

    if (interval_sec < 1) interval_sec = 1;

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    LOG_INFO("DeCloud GPU Guest Agent starting "
             "(host CID=%u, port=%u, interval=%ds)",
             host_cid, port, interval_sec);

    /* Verify nvidia-smi is available */
    GpuReportUsageStats probe = {0};
    if (poll_nvidia_smi(&probe) < 0) {
        LOG_ERR("nvidia-smi not available or no GPU detected — exiting");
        return 1;
    }
    LOG_INFO("GPU detected: %lu/%lu MiB used, %u%% utilization",
             (unsigned long)(probe.memory_allocated / (1024 * 1024)),
             (unsigned long)(probe.memory_total / (1024 * 1024)),
             probe.gpu_utilization);

    uint64_t start_us = now_us();

    /*
     * Main loop: connect → handshake → report in a loop.
     * If the connection drops, reconnect with exponential backoff.
     */
    int backoff_sec = 1;

    while (g_running) {
        /* Connect to host daemon */
        int fd = connect_vsock(host_cid, port);
        if (fd < 0) {
            LOG_ERR("Connection failed — retrying in %ds", backoff_sec);
            sleep(backoff_sec);
            if (backoff_sec < 30) backoff_sec *= 2;
            continue;
        }

        /* Handshake */
        if (do_handshake(fd) < 0) {
            close(fd);
            sleep(backoff_sec);
            if (backoff_sec < 30) backoff_sec *= 2;
            continue;
        }

        backoff_sec = 1;  /* Reset on successful connection */

        /* Report loop */
        while (g_running) {
            sleep(interval_sec);
            if (!g_running) break;

            GpuReportUsageStats stats = {0};
            if (poll_nvidia_smi(&stats) < 0) {
                LOG_ERR("nvidia-smi poll failed — skipping report");
                continue;
            }

            stats.uptime_us = now_us() - start_us;

            LOG_DBG("Reporting: mem=%lu/%lu MiB, gpu=%u%%, temp=%d°C, "
                    "power=%u/%u mW",
                    (unsigned long)(stats.memory_allocated / (1024 * 1024)),
                    (unsigned long)(stats.memory_total / (1024 * 1024)),
                    stats.gpu_utilization,
                    stats.temperature_c,
                    stats.power_usage_mw,
                    stats.power_limit_mw);

            if (send_request(fd, GPU_CMD_REPORT_USAGE_STATS,
                             &stats, sizeof(stats)) < 0) {
                LOG_ERR("Failed to send stats — reconnecting");
                break;
            }

            /* Read ACK */
            GpuProxyHeader ack;
            if (recv_response(fd, &ack) < 0) {
                LOG_ERR("Failed to read ACK — reconnecting");
                break;
            }

            if (ack.status != 0) {
                LOG_ERR("Host rejected stats report: status=%d", ack.status);
            }
        }

        /* Send GOODBYE if still connected */
        send_request(fd, GPU_CMD_GOODBYE, NULL, 0);
        close(fd);
        LOG_INFO("Disconnected from host");
    }

    LOG_INFO("GPU guest agent stopping");
    return 0;
}
