# SEC-1 Cross-Tenant Denial — Live Two-VM Runbook (WSL2)

The actual security test: **can tenant A read tenant B's VRAM** when A and B are
**different wallets** on the same GPU? Everything validated so far is mechanism +
quota (same-owner). This probes the trust boundary itself.

Uses the **guest-side** harness `sec1_probe_guest.c` (speaks the wire protocol over
TCP, reads each VM's own token from env). Runs from inside the two VMs against the
live agent-managed daemon — no hand-built token file, no hand-launched daemon.

> **Why this is trustworthy only with the control.** An "ISOLATED" result means
> nothing on its own — a broken probe also prints ISOLATED. The threaded-mode
> control (Step 4) forces the *old shared-context* behavior and must show
> VULNERABLE. Only the contrast — threaded VULNERABLE, fork DENIED — proves both
> that the probe works AND that isolation closes the hole. **Do not skip Step 4.**

> **Confound resolved:** the probe uses **D2H memcpy** (device→host read of a
> foreign address), not a peer-to-peer device copy. `GGML_CUDA_NO_PEER_COPY=1`
> governs P2P copies between two devices; it does not affect a single-device D2H.
> So a DENIED result is attributable to the fork/context boundary, not that flag.
> No flag toggling needed.

---

## Prerequisites

- Two VMs, **different owner wallets**, both GPU-proxied, both on this node,
  both confirmed reaching `library=cuda`. Call them **VM-A** (victim) and
  **VM-B** (attacker).
- Daemon currently in `auto` (default) — fork-per-VM isolation active.
- The proto header `gpu_proxy_proto.h` available to compile the probe inside each
  VM. The shims were deployed from the same release, so the header matches the
  daemon. If it's not already in the VM, copy it in (see Step 1).

---

## Step 1 — Build the probe inside BOTH VMs

The probe needs the project's proto header. Two ways to get it into the VM:

```bash
# Option A — if the gpu-proxy source/headers are reachable in the VM, point -I at it.
# Option B — copy the two files in via the node. From the NODE, for each VM:
#   the proto header lives in the daemon source tree on the node:
#   /opt/decloud/gpu-proxy/daemon-src/proto/gpu_proxy_proto.h
# Copy proto header + sec1_probe_guest.c into each VM (scp / paste / 9p share).
```

Inside **each** VM, with `sec1_probe_guest.c` and `proto/gpu_proxy_proto.h` in place:
```bash
# expects ../proto/gpu_proxy_proto.h relative to the .c, or use -I:
gcc -O2 -o sec1_probe_guest sec1_probe_guest.c -I.    # adjust -I to where proto/ is
# sanity: it reads the VM's own token from the environment
echo "token present: ${DECLOUD_GPU_PROXY_TOKEN:+yes}"
echo "proxy host:    ${DECLOUD_GPU_PROXY_HOST:-192.168.122.1}"
```

> The probe auto-reads `DECLOUD_GPU_PROXY_TOKEN` / `DECLOUD_GPU_PROXY_HOST` /
> `DECLOUD_GPU_PROXY_PORT` — the same env the shim uses. Each VM authenticates
> with **its own** valid token, which is the faithful attack: B is a legitimate
> tenant who then tries to name A's memory.

If the token isn't in the interactive shell's env, source it:
```bash
[ -f /etc/decloud/gpu-proxy.env ] && set -a && . /etc/decloud/gpu-proxy.env && set +a
```

---

## Step 2 — VM-A (victim): allocate, fill a sentinel, hold the connection open

```bash
# in VM-A:
./sec1_probe_guest hold --size 1048576 --pattern 0xAB
# -> DEVPTR=0x.............  SIZE=1048576 PATTERN=0xAB
#    Holding open. ENTER to release...
```
Leave this running (the connection — and thus A's worker + its filled allocation —
stays alive). **Note the DEVPTR hex value.**

> Stop Ollama in VM-A first (`systemctl stop ollama`) if you want a clean, quiet
> allocation, but it's not required — the probe's own MALLOC is what we read.

---

## Step 3 — VM-B (attacker): try to read VM-A's address  ★ THE GATE ★

### 3a. Targeted read (you hand B the address — the "leaked pointer" case)
```bash
# in VM-B, using the DEVPTR that VM-A printed:
./sec1_probe_guest read --addr 0x<A_DEVPTR> --size 64
```
- **PASS (isolated):** `READ DENIED — isolation OK`
  A's address is unmapped in B's separate CUDA context; the daemon's D2H faults.
- **FAIL (vulnerable):** `>>> VULNERABLE: read returned data (first=0xAB)`
  B read A's sentinel — shared address space, SEC-1 still open.

### 3b. Blind scan (the realistic attack — B does NOT know A's pointer)
A real co-tenant can't see the victim's pointer; it sweeps. VRAM is small/scannable.
```bash
# in VM-B — sweep a plausible device-address window:
# start/end around where the daemon hands out device pointers (use A's DEVPTR to
# pick a window, e.g. +/- 256 MiB around it), step 1 MiB, read 64 bytes each:
./sec1_probe_guest scan --start 0x<LOW> --end 0x<HIGH> --step 0x100000 --size 64
# -> "scan done: N probes, M leaks"
```
- **PASS:** `0 leaks` (and/or transport errors as B's worker faults on bad addrs).
- **FAIL:** any `>>> LEAK at 0x...` line, especially `first=0xAB` (A's sentinel).

Record both 3a and 3b results.

---

## Step 4 — CONTROL: prove the probe can detect a leak (threaded mode)  ★ DO NOT SKIP ★

Force the daemon into the **old shared-context** model and repeat. This must show
VULNERABLE — if it does, the probe genuinely detects cross-tenant access; if Step 3
showed DENIED and this shows VULNERABLE, isolation is what changed, and SEC-1 is
closed. If Step 4 *also* shows DENIED, the probe is broken and Step 3 proves nothing.

```bash
# on the NODE — pin the agent's daemon to threaded (shared context):
sudo systemctl edit decloud-node-agent
#   [Service]
#   Environment="DECLOUD_GPU_ISOLATION=threaded"
sudo systemctl restart decloud-node-agent
# confirm threaded:
journalctl -u decloud-node-agent --no-pager -n 20 | grep -iE "threaded|isolation|shared"
```

Re-run Step 2 (VM-A hold) and Step 3 (VM-B read/scan) unchanged.

- **Expected in threaded:** `>>> VULNERABLE: read returned data (first=0xAB)` —
  B reads A's sentinel through the shared context.

Then **revert** to isolation:
```bash
# on the NODE — remove the override:
sudo systemctl revert decloud-node-agent     # or delete the override file
sudo systemctl restart decloud-node-agent
journalctl -u decloud-node-agent --no-pager -n 20 | grep -iE "fork isolation active|isolation"
```

> ⚠️ Threaded mode forfeits cross-tenant isolation by design — only run it for this
> control, on this dev node, and revert immediately after. Never leave a
> multi-tenant node in threaded.

---

## Verdict matrix

| Step 3 (auto/fork) | Step 4 (threaded) | Conclusion |
|--------------------|-------------------|------------|
| DENIED | VULNERABLE | ✅ **SEC-1 closed.** Probe works; fork isolation denies cross-tenant access |
| DENIED | DENIED | ❓ Inconclusive — probe can't detect a leak; fix/investigate the probe before trusting any result |
| VULNERABLE | VULNERABLE | 🔴 **Isolation NOT working** — fork mode is leaking; investigate (is the daemon actually forking? `ps -ef | grep gpu-proxy-daemon` — one worker per VM?) |
| VULNERABLE | (n/a) | 🔴 SEC-1 open |

The only result that closes SEC-1 is **DENIED under fork + VULNERABLE under threaded.**

---

## Notes & honest caveats

- **What this tests:** the read (D2H) cross-tenant path. For completeness the
  same address can be tried with a write (H2D / memset) and a free — the probe
  has the primitives; extend if you want the full read/write/free matrix. The
  read is the canonical confidentiality test and the one SEC-1 centers on.
- **Kernel-launch path not covered.** SEC-1 notes that `LaunchKernel` parameters
  are opaque and unvalidatable in principle — fork isolation is *exactly* what
  closes that (a foreign pointer in A's kernel is unmapped in A's context). This
  runbook tests the memcpy path directly; the kernel path is closed by the same
  context boundary but is harder to exercise from the protocol probe. A DENIED
  memcpy result is strong evidence the context boundary holds for all paths,
  since they share the same per-worker context.
- **Scan window:** device pointers from `cudaMalloc` cluster in a range; use A's
  real DEVPTR to centre the scan window. A wide blind scan mostly produces
  transport errors (B's worker faults on unmapped addrs) — which is itself the
  isolation working. The targeted 3a read is the cleaner signal; 3b is the
  realism check.
- **Same-owner sanity:** if you also run the probe with B using a token for a VM
  owned by the SAME wallet as A, expect the same DENIED result — isolation is
  per-connection/per-context, not per-wallet. Different wallets matter for the
  *threat model* (it's a real cross-tenant boundary), not for the daemon's
  mechanism, which doesn't know about wallets.
