# GPU Proxy Daemon

The `gpu-proxy-daemon` is the host-side process that bridges CUDA calls from
guest VMs to the physical GPU. GPU-less VMs run CUDA workloads (Ollama, PyTorch)
by having their CUDA libraries shimmed; the shim forwards every call over
TCP/vsock to this daemon, which executes it on the real device and returns the
result.

This document explains how the daemon is built and kept current, how to build it
manually, and how to troubleshoot it.

## Why the daemon is built on the node (not shipped prebuilt)

Every other DeCloud artifact (node agent, CLI, GPU shims) ships as a prebuilt,
cosign-verified release binary. The daemon is the deliberate exception: it links
against `libcuda`/`libcudart`/`libcublas`/`libcublasLt` at the host's **exact**
CUDA version. A single prebuilt binary would be unsafe across the heterogeneous
CUDA versions real nodes run, so the daemon is always compiled locally against
the node's own CUDA toolkit.

What ships in the release is the daemon **source** — `gpu_proxy_daemon.c`,
`gpu_proxy_proto.h`, and the `Makefile` — as a verified artifact
(`decloud-gpu-daemon-src-<version>.tar.gz`). The source travels with the matching
shims in the same signed release, so the daemon a node builds always matches the
shims it serves. The binary is local; the source is versioned and trusted.

## Automated build (install and update)

`decloud install` and `decloud update` build the daemon automatically. Both run
`install.sh`, which:

1. Resolves CUDA (`install_cuda_toolkit` sets `CUDA_HOME`; installs
   `nvidia-cuda-toolkit` if no toolkit is found).
2. Fetches and cosign-verifies the daemon source artifact (same path as the
   shims).
3. Builds **only if stale** — rebuilds when the source hash changes or the binary
   is missing, and skips otherwise. A no-op `decloud update` does not recompile.
4. Installs the binary to `/usr/local/bin/gpu-proxy-daemon` and records the source
   hash at `/usr/local/bin/.gpu-proxy-daemon.sha256`.

The staleness check hashes exactly the daemon's two source inputs, so it triggers
precisely when the daemon code changes and never on unrelated shim changes. You do
not normally need to build the daemon by hand — `decloud update` keeps it current.

The node agent (`GpuProxyService`) starts the daemon when the first GPU-proxied VM
boots and stops it when none remain; you do not start it manually in normal
operation.

## Manual build

Build by hand when iterating on the daemon source locally, or to recover a node
whose daemon predates automated builds. The daemon is compiled with the project
`Makefile`, which carries the correct link flags.

```bash
cd <gpu-proxy source tree>          # contains daemon/, proto/, Makefile

# CUDA_HOME: /usr/local/cuda for a standard toolkit, or /usr when the headers
# come from the nvidia-cuda-toolkit apt package (cuda.h in /usr/include).
sudo make daemon CUDA_HOME=/usr/local/cuda

sudo install -m 755 build/gpu-proxy-daemon /usr/local/bin/gpu-proxy-daemon

# verify it understands the isolation flag (SEC-1):
/usr/local/bin/gpu-proxy-daemon --help 2>&1 | grep -- '-i'
```

Use `make daemon` (the daemon target only) — **not** `make install` or `make all`,
which also build the host-glibc runtime shim and would overwrite the
universal-compat shims in `/usr/local/lib/decloud-gpu-shim/` that VMs depend on.

The daemon needs a CUDA toolkit (headers + `libcuda`/`libcudart`/`libcublas`/
`libcublasLt`). On a node without one:

```bash
sudo apt-get install -y nvidia-cuda-toolkit   # headers in /usr/include → CUDA_HOME=/usr
```

## Running and flags

```
gpu-proxy-daemon [-p port] [-T tcp_bind] [-t timeout_sec] [-i mode] [-v]
```

- `-p` — vsock/TCP port (default `9999`).
- `-T` — enable the TCP listener on the given bind address (e.g.
  `192.168.122.1`). Required on WSL2, where vsock is owned by Hyper-V.
- `-t` — kernel execution timeout in seconds (`0` disables). Guards against a
  runaway kernel monopolizing the device.
- `-v` — verbose logging.
- `-i` — isolation mode (see below). May also be set via the
  `DECLOUD_GPU_ISOLATION` environment variable.

### Isolation modes (SEC-1)

The daemon serves each VM's CUDA calls under one of three isolation modes:

- **`auto`** (default) — forks one worker process per VM when a GPU is present.
  Each worker has its own CUDA context, giving GPU-MMU-level isolation between
  tenants. This is the safe default for multi-tenant nodes.
- **`fork`** — force per-VM fork even when the supervisor cannot probe the GPU.
- **`threaded`** — legacy single-process, shared-context mode with **no
  cross-tenant isolation**. Lower per-VM VRAM overhead (one context for the
  node). Only appropriate when the operator knows the node is single-tenant.

The orchestrator never selects `threaded` — it cannot prove single-tenancy on a
permissionless node, so isolation is the default. An operator who knows their node
is private may opt in explicitly:

```bash
# per-node, via the agent's environment:
DECLOUD_GPU_ISOLATION=threaded
```

### Per-tenant VRAM quota

Each proxied VM has a VRAM quota (`GpuVramBytes`, set by the orchestrator and
written to the token registry). The daemon enforces it at allocation time in
`handle_malloc`: an allocation that would push the VM over its quota is denied
(`cudaErrorMemoryAllocation`), and the guest sees `unable to allocate CUDA0
buffer`. The guest still *sees* the whole card via `cudaMemGetInfo` — the quota
caps what it can `cudaMalloc`, not what it can observe.

The quota is enforced **per-tenant, across all of a VM's connections** — not
per-connection. A single VM can open several proxy connections (e.g. multiple
Ollama runners, or a terminal plus a web UI), each served by its own forked
worker. Enforcement sums every live connection's allocations for the VM and
checks the total against the quota, via a shared-memory slot ledger the
supervisor creates before forking. The ledger is self-healing: a worker killed
without clean teardown (SIGKILL) leaves a slot behind, which is reaped on the
next allocation sweep by a PID-liveness check (`kill(pid,0) == ESRCH`). The
accounting is fail-safe — a not-yet-reaped slot over-counts (the tenant gets
slightly less than quota) and never under-counts.

In a forked worker, ~400 MB of the quota is reserved for the worker's own CUDA
context (`GPU_CONTEXT_OVERHEAD_BYTES`), so a 3072 MB quota yields ~2672 MB
usable. A quota too small to leave a usable floor after this reservation is
rejected at `HELLO`.

> **Transport note:** quota enforcement requires the daemon to know which VM a
> connection belongs to. On TCP (WSL2) this comes from the auth token; on vsock
> (bare metal) it comes from the guest's source CID, resolved against the
> registry's `cid` field. The vsock path is code-complete but has only been
> exercised on TCP so far — vsock is unavailable under WSL2.

### Worker and supervisor lifecycle

Each VM connection is served by a forked worker that lives exactly as long as its
connection. A worker exits — releasing its CUDA context and VRAM — when the VM
disconnects: cleanly on a normal close, or within ~90 s of an abrupt drop
(half-open connections are detected by TCP keepalive, or a recv-timeout on
vsock). Workers honor `SIGTERM` (they reset the inherited supervisor handler to
default), so they are individually killable and are not orphaned across a daemon
restart. On shutdown the supervisor signals its worker process group and exits
deterministically (it closes its listen FDs to unblock `accept()` and `_exit`s);
it holds no CUDA context, so an immediate exit is safe. A correct restart leaves
no orphaned `gpu-proxy-daemon` process and returns VRAM to baseline.

### WSL2 note

WSL2 reaches the GPU through a passthrough CUDA stack
(`/usr/lib/wsl/lib/libcuda.so.1`). Initializing CUDA in a forked child against
this passthrough can fail where a non-forked init succeeds. If the daemon in
`auto` mode logs `No CUDA devices (probe)` while `nvidia-smi` works, the fork
probe hit this limitation — pin the node to threaded mode with
`DECLOUD_GPU_ISOLATION=threaded`. (Threaded mode forfeits cross-tenant isolation,
which is acceptable on a single-tenant WSL2 dev node.)

## Verifying the daemon

```bash
# is it running?
pgrep -af gpu-proxy-daemon

# is the port reachable from the bridge?
ss -tlnp | grep 9999

# build provenance: does the installed binary support -i?
/usr/local/bin/gpu-proxy-daemon --help 2>&1 | grep -- '-i' \
  && echo "current (SEC-1)" || echo "STALE — pre-isolation build"

# from inside a proxied VM:
timeout 3 bash -c 'echo > /dev/tcp/192.168.122.1/9999' \
  && echo "proxy reachable" || echo "proxy unreachable"
```

A common stale-state signature: the agent launches the daemon with `-i` but the
installed binary predates that flag, so it exits immediately with
`invalid option -- 'i'`. The `--help | grep -- '-i'` check above detects this; the
fix is to let `decloud update` rebuild the daemon, or build it manually.

## Uninstall

`decloud uninstall` removes the daemon binary, its staleness marker, and the shim
directory, and stops any running daemon. The GPU proxy enters and leaves the
system with the rest of the node agent.

## Files

| Path | Purpose |
|---|---|
| `/usr/local/bin/gpu-proxy-daemon` | The compiled daemon binary |
| `/usr/local/bin/.gpu-proxy-daemon.sha256` | Source hash for the staleness check |
| `/usr/local/lib/decloud-gpu-shim/` | Compat shims delivered to VMs (do not overwrite with host-glibc shims) |
| `/var/lib/decloud/gpu-proxy-tokens` | Per-VM registry, one line per VM: `<token> <vm_id> [<quota_bytes>] [<cid>]`. Fields 3–4 optional (missing → 0). `quota_bytes` enforces the per-VM VRAM cap; `cid` lets the daemon map a vsock connection (which carries no token) to its VM. Daemon reloads on `SIGHUP` |
| `daemon/gpu_proxy_daemon.c`, `proto/gpu_proxy_proto.h`, `Makefile` | Source shipped in `decloud-gpu-daemon-src-<version>.tar.gz` |

## Related

- `GpuProxyService.cs` — daemon lifecycle (start/stop, isolation-mode resolution,
  per-VM token registration).
- `install.sh` `build_gpu_proxy_daemon` — automated fetch + staleness build.
- `diag-gpu-proxy.sh` — in-VM diagnostic collector for proxy issues.
