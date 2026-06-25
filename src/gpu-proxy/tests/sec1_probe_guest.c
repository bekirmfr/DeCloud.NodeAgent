/* sec1_probe_guest.c — SEC-1 isolation test, run FROM INSIDE a VM (WSL2/TCP).
 *
 * Difference vs the host-side sec1_probe.c:
 *   - Reads the VM's own token from DECLOUD_GPU_PROXY_TOKEN and HEX-DECODES it
 *     (the shim sends the token as raw bytes parsed from a hex string).
 *   - Defaults host/port to DECLOUD_GPU_PROXY_HOST/PORT (192.168.122.1:9999).
 *   - Adds `scan` mode: a true black-box attack that sweeps an address range
 *     looking for any readable (non-faulting, non-zero) region — what a real
 *     adversary does, since it cannot see the victim's pointers.
 *
 * This is the faithful SEC-1 attack: VM-B uses ITS OWN valid token over the
 * real transport and tries to name VM-A's VRAM.
 *
 * Build inside the guest (needs the proto header copied in or on an -I path):
 *   gcc -O2 -o sec1_probe_guest sec1_probe_guest.c
 *
 * Subcommands (token auto-read from env; override with --token-hex):
 *   hold  --size BYTES [--pattern 0xAB]
 *       VM-A: alloc + fill, print DEVPTR, hold connection (ENTER to release).
 *   read  --addr 0x.... --size N
 *       VM-B: D2H from a KNOWN addr (use when VM-A printed its DEVPTR).
 *   scan  --start 0x.... --end 0x.... --step BYTES --size N
 *       VM-B: sweep [start,end) reading --size bytes at each step; reports any
 *       address that returns data (status 0, nonzero) == cross-tenant leak.
 *   hello
 *       Just authenticate (quota-log / floor test).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "gpu_proxy_proto.h"   /* adjust path / use -I as needed */

static int write_exact(int fd, const void *buf, size_t n) {
    const uint8_t *p = buf; size_t off = 0;
    while (off < n) { ssize_t w = write(fd, p + off, n - off);
        if (w <= 0) { if (errno == EINTR) continue; return -1; } off += (size_t)w; }
    return 0;
}
static int read_exact(int fd, void *buf, size_t n) {
    uint8_t *p = buf; size_t off = 0;
    while (off < n) { ssize_t r = read(fd, p + off, n - off);
        if (r == 0) return -1; if (r < 0) { if (errno == EINTR) continue; return -1; } off += (size_t)r; }
    return 0;
}
static int send_req(int fd, uint16_t cmd, const void *payload, uint32_t len) {
    GpuProxyHeader h; memset(&h, 0, sizeof(h));
    h.magic = GPU_PROXY_MAGIC; h.version = GPU_PROXY_VERSION; h.cmd = cmd;
    h.flags = 0; h.payload_len = len; h.status = 0;
    if (write_exact(fd, &h, sizeof(h)) < 0) return -1;
    if (len && write_exact(fd, payload, len) < 0) return -1;
    return 0;
}
static long recv_resp(int fd, GpuProxyHeader *hdr, void *buf, uint32_t cap) {
    if (read_exact(fd, hdr, sizeof(*hdr)) < 0) return -1;
    uint32_t pl = hdr->payload_len; if (pl > cap) pl = cap;
    if (pl && read_exact(fd, buf, pl) < 0) return -1;
    return (long)pl;
}

/* Decode a hex string into out[], up to outlen bytes. Mirrors the shim's
 * parse_hex_token: each pair of hex chars -> one byte. */
static void hex_decode(const char *hex, uint8_t *out, size_t outlen) {
    memset(out, 0, outlen);
    size_t n = strlen(hex), b = 0;
    for (size_t i = 0; i + 1 < n && b < outlen; i += 2) {
        char hi = hex[i], lo = hex[i + 1];
        int H = isdigit(hi) ? hi - '0' : (tolower(hi) - 'a' + 10);
        int L = isdigit(lo) ? lo - '0' : (tolower(lo) - 'a' + 10);
        out[b++] = (uint8_t)((H << 4) | L);
    }
}

static int connect_tcp(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }
    struct sockaddr_in a; memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
    if (inet_aton(host, &a.sin_addr) == 0) { fprintf(stderr, "bad host %s\n", host); return -1; }
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) < 0) { perror("connect"); return -1; }
    return fd;
}

static int do_hello(int fd, const uint8_t *token_bytes) {
    GpuHelloRequest r; memset(&r, 0, sizeof(r));
    r.pid = (uint32_t)getpid(); r.shim_version = GPU_PROXY_VERSION;
    memcpy(r.auth_token, token_bytes, sizeof(r.auth_token));
    if (send_req(fd, GPU_CMD_HELLO, &r, sizeof(r)) < 0) return -1;
    GpuProxyHeader h; GpuHelloResponse resp;
    long pl = recv_resp(fd, &h, &resp, sizeof(resp));
    if (pl < 0) { fprintf(stderr, "HELLO: connection closed (rejected)\n"); return -1; }
    if (h.status != 0) { fprintf(stderr, "HELLO rejected (status=%d)\n", (int)h.status); return -1; }
    return 0;
}
static uint64_t do_malloc(int fd, uint64_t size) {
    GpuMallocRequest q; memset(&q, 0, sizeof(q)); q.size = size;
    if (send_req(fd, GPU_CMD_MALLOC, &q, sizeof(q)) < 0) return 0;
    GpuProxyHeader h; GpuMallocResponse resp;
    long pl = recv_resp(fd, &h, &resp, sizeof(resp));
    if (pl < 0 || h.status != 0) { fprintf(stderr, "MALLOC failed (status=%d)\n", (int)h.status); return 0; }
    return resp.device_ptr;
}
static int do_memset(int fd, uint64_t ptr, int val, uint64_t count) {
    GpuMemsetRequest q; memset(&q, 0, sizeof(q)); q.device_ptr = ptr; q.value = val; q.count = count;
    if (send_req(fd, GPU_CMD_MEMSET, &q, sizeof(q)) < 0) return -1;
    GpuProxyHeader h; uint8_t tmp[64];
    long pl = recv_resp(fd, &h, tmp, sizeof(tmp));
    if (pl < 0 || h.status != 0) { fprintf(stderr, "MEMSET failed (status=%d)\n", (int)h.status); return -1; }
    return 0;
}
/* Returns: 0 denied, 1 returned-zero, 2 returned-nonzero(LEAK), -1 transport err */
static int probe_read(int fd, uint64_t addr, uint64_t count, uint8_t *first_byte) {
    GpuMemcpyRequest q; memset(&q, 0, sizeof(q));
    q.dst = 0; q.src = addr; q.count = count; q.kind = GPU_MEMCPY_DEVICE_TO_HOST;
    if (send_req(fd, GPU_CMD_MEMCPY, &q, sizeof(q)) < 0) return -1;
    GpuProxyHeader h; static uint8_t buf[1 << 20];
    long pl = recv_resp(fd, &h, buf, sizeof(buf));
    if (pl < 0) return -1;
    if (h.status != 0 || pl <= 0) return 0;          /* faulted -> denied */
    int nonzero = 0; for (long i = 0; i < pl; i++) if (buf[i]) { nonzero = 1; break; }
    if (first_byte) *first_byte = buf[0];
    return nonzero ? 2 : 1;
}

static const char *opt(int c, char **v, const char *k, const char *d) {
    for (int i = 2; i < c - 1; i++) if (!strcmp(v[i], k)) return v[i + 1];
    return d;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s {hello|hold|read|scan} ...\n", argv[0]); return 1; }
    const char *cmd  = argv[1];
    const char *host = opt(argc, argv, "--host", getenv("DECLOUD_GPU_PROXY_HOST") ?: "192.168.122.1");
    int port         = atoi(opt(argc, argv, "--port", getenv("DECLOUD_GPU_PROXY_PORT") ?: "9999"));

    /* Token: --token-hex overrides; else the VM's own env token. */
    const char *tok_hex = opt(argc, argv, "--token-hex", getenv("DECLOUD_GPU_PROXY_TOKEN"));
    if (!tok_hex) { fprintf(stderr, "no token: set DECLOUD_GPU_PROXY_TOKEN or pass --token-hex\n"); return 1; }
    uint8_t token[GPU_PROXY_TOKEN_LEN];
    hex_decode(tok_hex, token, sizeof(token));

    int fd = connect_tcp(host, port);
    if (fd < 0) return 1;

    if (!strcmp(cmd, "hello")) {
        int rc = do_hello(fd, token);
        printf(rc == 0 ? "HELLO ACCEPTED\n" : "HELLO REJECTED\n");
        close(fd); return rc == 0 ? 0 : 1;
    }
    if (!strcmp(cmd, "hold")) {
        uint64_t size = strtoull(opt(argc, argv, "--size", "1048576"), NULL, 0);
        int pat = (int)strtol(opt(argc, argv, "--pattern", "0xAB"), NULL, 0);
        if (do_hello(fd, token) < 0) { close(fd); return 1; }
        uint64_t ptr = do_malloc(fd, size); if (!ptr) { close(fd); return 1; }
        if (do_memset(fd, ptr, pat, size) < 0) { close(fd); return 1; }
        printf("DEVPTR=0x%llx SIZE=%llu PATTERN=0x%02X\n",
               (unsigned long long)ptr, (unsigned long long)size, pat & 0xFF);
        printf("Holding open. ENTER to release...\n"); fflush(stdout);
        getchar(); close(fd); return 0;
    }
    if (!strcmp(cmd, "read")) {
        uint64_t addr = strtoull(opt(argc, argv, "--addr", "0"), NULL, 0);
        uint64_t size = strtoull(opt(argc, argv, "--size", "64"), NULL, 0);
        if (!addr) { fprintf(stderr, "--addr required\n"); close(fd); return 1; }
        if (do_hello(fd, token) < 0) { close(fd); return 1; }
        uint8_t fb = 0; int r = probe_read(fd, addr, size, &fb);
        if (r == 2)      printf(">>> VULNERABLE: read returned data (first=0x%02X)\n", fb);
        else if (r == 1) printf("returned zeros — inconclusive\n");
        else             printf("READ DENIED — isolation OK\n");
        close(fd); return r == 2 ? 2 : 0;
    }
    if (!strcmp(cmd, "scan")) {
        uint64_t start = strtoull(opt(argc, argv, "--start", "0"), NULL, 0);
        uint64_t end   = strtoull(opt(argc, argv, "--end",   "0"), NULL, 0);
        uint64_t step  = strtoull(opt(argc, argv, "--step",  "1048576"), NULL, 0); /* 1 MiB */
        uint64_t size  = strtoull(opt(argc, argv, "--size",  "64"), NULL, 0);
        if (!start || !end || end <= start) { fprintf(stderr, "--start/--end required (end>start)\n"); close(fd); return 1; }
        if (do_hello(fd, token) < 0) { close(fd); return 1; }
        uint64_t hits = 0, probes = 0;
        for (uint64_t a = start; a < end; a += step) {
            uint8_t fb = 0; int r = probe_read(fd, a, size, &fb);
            probes++;
            if (r == 2) { hits++; printf(">>> LEAK at 0x%llx (first=0x%02X)\n",
                                         (unsigned long long)a, fb); }
            if (r == -1) { printf("transport error at 0x%llx (worker died?) — stopping\n",
                                  (unsigned long long)a); break; }
        }
        printf("scan done: %llu probes, %llu leaks\n",
               (unsigned long long)probes, (unsigned long long)hits);
        close(fd); return hits ? 2 : 0;
    }
    fprintf(stderr, "unknown subcommand %s\n", cmd);
    close(fd); return 1;
}
