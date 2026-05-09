# Node Agent Lifecycle Reference

What actually happens when an operator installs, updates, or uninstalls
a DeCloud node agent. Use this document to answer "what's supposed to
happen at this step?" when debugging, or to understand the system
without reading shell scripts.

> **Companion document:** [`RELEASE-PIPELINE.md`](RELEASE-PIPELINE.md)
> describes how the release artifacts this document consumes are
> produced.

---

## Table of Contents

1. [Overview](#overview)
2. [Fresh install](#fresh-install)
3. [Update](#update)
4. [Uninstall](#uninstall)
5. [On-disk layout](#on-disk-layout)
6. [Configuration](#configuration)
7. [Trust chain at runtime](#trust-chain-at-runtime)
8. [Common operator tasks](#common-operator-tasks)
9. [Recovery scenarios](#recovery-scenarios)
10. [Migration from legacy installs](#migration-from-legacy-installs)

---

## Overview

Three lifecycle operations, each with one canonical entry point:

| Operation | Entry point | Reads from | Writes to | Verifies |
| --- | --- | --- | --- | --- |
| Install | `curl ... \| sudo bash` | GitHub Release | `/opt/decloud`, `/usr/local/bin`, `/etc/decloud` | Cosign + SHA-256 |
| Update | `sudo decloud update` | GitHub Release (or local fallback) | Same as install | Cosign + SHA-256 |
| Uninstall | `sudo decloud uninstall` | GitHub Release (or local fallback) | Removes most of the above | None (cleanup only) |

Update is "install with saved parameters" — same script, same code path,
just non-interactive. There is one installation procedure.

---

## Fresh install

The single command an operator pastes:

```bash
curl -fsSL https://github.com/bekirmfr/DeCloud.NodeAgent/releases/latest/download/install.sh \
  | sudo bash -s -- \
      --orchestrator https://decloud.stackfi.tech \
      --wallet 0xYourWalletAddress \
      --name "MyNode" \
      --region "us-east-1" \
      --zone "us-east-1-nyc-1a"
```

Wall-clock time on a fresh Ubuntu box: 3-5 minutes. On a re-run with
most dependencies cached: ~30 seconds.

### What runs, in order

1. **Bootstrap fetch** (`curl`, ~1 s)
   - TLS to `github.com`, redirect to `objects.githubusercontent.com`
   - `install.sh` body streamed into bash's stdin (never touches disk
     in this mode)

2. **Argument parse** (`install.sh`, ~5 s)
   - Reads `--orchestrator`, `--wallet`, `--name`, `--region`, `--zone`
   - Persists arguments to `/etc/decloud/install-params` (used by
     `decloud update` later)
   - **If a required flag is missing**: prompts via `/dev/tty` if a
     controlling terminal is available, otherwise errors with copy-pasteable
     correction (see [Interactive prompts](#interactive-prompts) below)

3. **Preflight checks**
   - OS detection (Ubuntu 22.04+/24.04 supported)
   - Architecture (x86_64 or aarch64)
   - Hardware virtualization support (`/proc/cpuinfo` flags)
   - Free RAM ≥ 4 GB, disk ≥ 50 GB
   - Ports 5100 (Agent API) and 51821 (WireGuard) free
   - Outbound HTTPS to orchestrator URL works

4. **Base dependencies** (~30 s first time)
   - `apt-get install` of `curl wget git jq tar xz-utils ...`
   - `install_cosign`: downloads cosign 2.4.1 from sigstore's GitHub
     Release, drops at `/usr/local/bin/cosign`. **From this point, cosign
     is the cryptographic anchor.**

5. **ASP.NET Core 8 runtime** (~60 s first time)
   - Microsoft's signed apt repo added
   - `aspnetcore-runtime-8.0` installed
   - **Runtime only**, no SDK. No compiler is installed on production
     nodes.

6. **Virtualization, GPU detection, WireGuard, Python, SSH CA**
   - Existing infrastructure setup, mostly idempotent
   - GPU mode auto-detected: `none` / `proxy` (CUDA shim + daemon) /
     `passthrough` (VFIO)

7. **Resolve release version** (~1 s)
   - If `--version vX.Y.Z` was given, uses that
   - Otherwise calls `releases/latest` API
   - On 404 (no stable release exists), falls back to
     `releases?per_page=1` with a clear pre-release warning

8. **Fetch and verify manifest** (~2 s)
   - Downloads `manifest.json`, `.sig`, `.pem`
   - `cosign verify-blob` with workflow-identity regex
   - Fail → install aborts immediately

9. **Download and verify artifacts** (~10-30 s)
   - For each tarball: download → sha256sum → compare against verified
     manifest
   - GPU shim fetched only if `GPU_MODE=proxy` and amd64

10. **Atomic version swap** (~1 s)
    - Each tarball extracted to its versioned directory
    - `/opt/decloud/publish` symlink atomically swapped to point at new
      version (legacy real directory removed if present — see
      [Migration](#migration-from-legacy-installs))

11. **Install CLI tools** (~2 s)
    - From `/opt/decloud/stage/`: copy `decloud`, `cli-decloud-node`,
      `decloud-relay-nat`, `vm-cleanup.sh` into `/usr/local/bin/`
    - Refresh `install.sh` and `uninstall.sh` at
      `/usr/local/share/decloud/` (offline fallback for future
      `decloud update`/`decloud uninstall`)

12. **Configuration & service** (~3 s)
    - Generate `appsettings.Production.json` from orchestrator URL,
      wallet, machine ID
    - Create `/etc/systemd/system/decloud-node-agent.service`
    - `systemctl enable --now decloud-node-agent`

13. **Done**
    - Summary box, next-steps prompt for `decloud login`

### Interactive prompts

When `install.sh` is run from a terminal (or via `curl | bash` with
`/dev/tty` available — most SSH sessions qualify), missing required
flags trigger interactive prompts:

```
Orchestrator URL [https://decloud.stackfi.tech]: 
Wallet address (0x...): 0x...
```

The orchestrator default is hardcoded; the wallet is required and
validated in a loop until a valid format is provided.

When run truly non-interactively (CI, no controlling terminal), missing
required flags abort with copy-pasteable usage examples.

---

## Update

```bash
sudo decloud update
```

### What runs, in order

1. **Read saved parameters** from `/etc/decloud/install-params`
2. **Hybrid install.sh resolution:**
   - **Primary**: `curl` from `releases/latest/download/install.sh`
     (30 s timeout). On success, writes to `/tmp/decloud-install-XXXX.sh`.
   - **Fallback**: if curl fails (offline, GitHub unreachable, network
     restricted), uses `/usr/local/share/decloud/install.sh` — the copy
     left by the previous successful install
   - **Both fail**: clear error explaining how to manually re-seed the
     local copy
3. **Run `install.sh` non-interactively** with saved parameters
4. **Cleanup**: remove the `/tmp/decloud-install-XXXX.sh` if curl was
   used; keep the `/usr/local/share/decloud/install.sh` for next time

The install.sh that runs handles `UPDATE_MODE` automatically: it detects
the running service on port 5100, stops it before swapping the publish
symlink, and starts it again afterward. From systemd's perspective the
working directory path doesn't change; only the version behind the
symlink does.

### Self-healing property

Because every successful install (initial or update) refreshes
`/usr/local/share/decloud/install.sh`, the offline fallback is always
the version of install.sh that came with the *currently active*
release. This means:

- A node running `vN` always has `vN`'s install.sh on disk
- That install.sh knows how to install `vN` (and forward, given backward
  compat)
- If the operator goes offline, they can still run `decloud update` and
  it will re-install `vN` cleanly (no version change, but state is
  refreshed)

### What it does not do

- Does **not** preserve appsettings hand-edits across upgrades. Each
  install regenerates `appsettings.Production.json` from
  `/etc/decloud/install-params`. To change orchestrator/wallet/etc.,
  edit that file (or pass new flags via `decloud install ...`) and
  re-run.
- Does **not** roll back automatically on failure. If the new release
  fails to start, the symlink already points at the new version. See
  [Recovery scenarios](#recovery-scenarios).

---

## Uninstall

```bash
sudo decloud uninstall
```

Optional flags (passed through to `uninstall.sh`):

- `--force` — skip confirmation prompt
- `--keep-vms` — leave VMs running (useful for migration)
- `--keep-data` — preserve `/var/lib/decloud` (databases, VM disks)
- `--keep-wg` — preserve WireGuard keys/configs in `/etc/wireguard`

### What runs, in order

1. **Hybrid uninstall.sh resolution** (same pattern as update):
   curl from latest release first, fall back to
   `/usr/local/share/decloud/uninstall.sh`
2. **Confirmation prompt** (unless `--force`): operator must type
   `REMOVE` exactly
3. **Notify orchestrator** (best-effort): if credentials exist, POST to
   `/api/nodes/<id>/deregister` with the API key. Skipped silently if
   credentials are missing or the orchestrator is unreachable.
4. **Stop and disable systemd service**, remove
   `/etc/systemd/system/decloud-node-agent.service`
5. **Destroy all VMs** (unless `--keep-vms`): `virsh destroy` +
   `virsh undefine --remove-all-storage` for every VM defined on the
   host
6. **Remove WireGuard interfaces** (unless `--keep-wg`): managed
   interfaces (`wg0`, `wg-relay`, `wg-relay-server`, `wg-hub`) are
   brought down; configs backed up with `.uninstall-<timestamp>` suffix;
   node WG keys removed
   - **Orchestrator-managed keys** (`/etc/wireguard/orchestrator-*.key`)
     are NOT removed — they belong to the orchestrator
7. **Remove SSH CA configuration** from `/etc/ssh/sshd_config`, remove
   the `decloud` user
8. **Remove firewall rules** (`ufw delete` on agent ports)
9. **Remove binaries**: `/usr/local/bin/{decloud,cli-decloud-node,decloud-relay-nat,vm-cleanup.sh,cosign}`,
   `/usr/local/lib/decloud-gpu-shim/*`, `/usr/local/share/decloud/*`,
   `/usr/local/share/doc/decloud/*`
10. **Remove application directories**: `/opt/decloud`, `/var/log/decloud`
    - `/etc/decloud/install-params` is **backed up** to
      `/tmp/decloud-install-params.bak` before deletion (so a
      re-install can recover the original parameters)
    - `/var/lib/decloud` removed unless `--keep-data`

### What is intentionally NOT removed

- `aspnetcore-runtime-8.0` (system package, may be used elsewhere)
- `libvirt`, `qemu-kvm`, `wireguard-tools` (system packages)
- `cosign` (system package)
- Orchestrator WG keys at `/etc/wireguard/orchestrator-*.key`

This makes uninstall reversible-ish: a re-install on the same machine
finds existing libvirt, dotnet runtime, and WG infrastructure intact
and reuses them.

---

## On-disk layout

After a successful install, the canonical tree:

```
/opt/decloud/
├── publish              -> publish.vX.Y.Z          (active symlink)
├── publish.vX.Y.Z/                                  (current release)
│   ├── DeCloud.NodeAgent.dll
│   ├── DeCloud.Shared.dll
│   ├── appsettings.Production.json                 (regenerated each install)
│   ├── appsettings.json
│   ├── *.dll                                        (~16 MB total)
│   └── wwwroot/                                     (dashboard static files)
├── publish.vX.Y.W/                                  (previous version, kept for rollback)
├── stage/                                           (extracted CLI bundle)
│   ├── cli/
│   ├── install.sh
│   ├── uninstall.sh
│   ├── decloud-relay-nat
│   └── vm-cleanup.sh
├── gpu-proxy/build/                                 (extracted GPU shims, if applicable)
├── releases/                                        (manifest cache + tarballs)
│   ├── manifest.vX.Y.Z.json
│   ├── manifest.vX.Y.Z.json.sig
│   ├── manifest.vX.Y.Z.json.pem
│   ├── decloud-node-agent-vX.Y.Z-linux-amd64.tar.gz
│   └── decloud-cli-vX.Y.Z.tar.gz
├── current-version                                  (text: "vX.Y.Z")
└── previous-version                                 (text: "vX.Y.W", if a prior version exists)

/usr/local/bin/
├── decloud                                          (operator CLI)
├── cli-decloud-node                                 (auth helper)
├── decloud-relay-nat
├── vm-cleanup.sh
└── cosign

/usr/local/lib/decloud-gpu-shim/                     (GPU mode = proxy only)
├── libdecloud_cuda_shim.so
├── libcudart.so.12
├── libcuda.so.1
├── libnvidia-ml.so.1
├── libcublas_stub.so
├── libcublasLt_stub.so
├── libcudnn_stub.so
└── libcuda_pytorch_stubs.so

/usr/local/share/decloud/                            (offline fallback scripts)
├── install.sh
└── uninstall.sh

/usr/local/share/doc/decloud/                        (operator docs)
├── README.md
├── QUICKREF.md
└── DESIGN.md

/etc/decloud/
├── install-params                                   (mode 600, used by 'decloud update')
├── credentials                                      (after 'decloud login')
├── pending-auth                                     (during login flow)
└── ssh-ca/                                          (SSH CA private key + cert)

/etc/systemd/system/decloud-node-agent.service       (uses /opt/decloud/publish symlink)

/var/lib/decloud/                                    (state)
├── vms/                                             (VM disk images)
└── *.db                                             (SQLite databases, if any)

/var/log/decloud/                                    (logs)
└── install.log
```

### Symlink semantics

- The `/opt/decloud/publish` symlink is the active-version pointer.
- The systemd unit's `WorkingDirectory=/opt/decloud/publish` and
  `ExecStart=/usr/bin/dotnet /opt/decloud/publish/DeCloud.NodeAgent.dll`
  resolve through it transparently — version swaps are atomic from
  systemd's perspective.
- Disk layout cleanup keeps current + 1 previous publish.* directory.
  Older versions are pruned on each successful install.

---

## Configuration

### Where settings live

| Setting | Source | Mutability |
| --- | --- | --- |
| Orchestrator URL, wallet, name, region, zone | `/etc/decloud/install-params` | Edit and re-run `decloud update` |
| `appsettings.Production.json` | Generated by `install.sh` from install-params | Regenerated on each install |
| Node ID, API key | `/etc/decloud/credentials` (after login) | Removed by `logout`, regenerated on next login |
| Machine ID | `/etc/machine-id` | OS-level, do not edit |
| Node identity | `SHA-256(machine-id + wallet)` | Derived; changes if either input changes |

### Changing parameters after install

```bash
# Edit
sudo nano /etc/decloud/install-params

# Apply (regenerates appsettings, restarts service)
sudo decloud update
```

Or, override individual flags by re-running install with new values —
they will be saved to install-params and used on subsequent updates.

---

## Trust chain at runtime

Three layered claims, each with a different cryptographic anchor.
The same chain runs on every install and every update.

```
Operator copies command from landing page
        │ TLS to github.com
        ▼
install.sh (small, auditable, ~3000 lines)
        │ cosign verify-blob with pinned identity regex
        ▼
manifest.json (signed by NodeAgent's release.yml workflow)
        │ SHA-256 lookup, byte-exact comparison
        ▼
Each .tar.gz / .so artifact downloaded
        │ extract, install
        ▼
Running NodeAgent
```

What this chain proves:

| Layer | Anchor | What it attests |
| --- | --- | --- |
| TLS | GitHub's certificate | The bytes we got are the bytes GitHub served (no MITM) |
| Cosign keyless | GitHub OIDC + Sigstore Fulcio cert | The manifest was signed by `bekirmfr/DeCloud.NodeAgent`'s `release.yml` workflow at a specific commit |
| SHA-256 | The verified manifest | Each artifact is byte-identical to what the workflow produced |

What this chain does **not** prove:

- That the source code at the signed commit is benign. That depends
  entirely on the maintainer process: signed commits, branch protection,
  code review.
- That GitHub itself is uncompromised. If `github.com` were
  compromised, the entire chain is moot — but so is everything else
  hosted there.
- That install.sh is what it appears to be. Operators who care should
  download it first and inspect before running. install.sh is small
  and human-readable.

---

## Common operator tasks

### Check status

```bash
decloud status                       # comprehensive overview
systemctl status decloud-node-agent  # service-level
journalctl -u decloud-node-agent -n 50
```

### Install a specific version

```bash
curl -fsSL https://github.com/bekirmfr/DeCloud.NodeAgent/releases/latest/download/install.sh \
  | sudo bash -s -- \
      --orchestrator https://decloud.stackfi.tech \
      --wallet 0x... \
      --version v2.1.5
```

### Update to latest

```bash
sudo decloud update
```

### Update to a specific version (downgrade or pin)

```bash
# Edit /etc/decloud/install-params to add:
#   --version
#   v2.1.5
# Then:
sudo decloud update
```

### Roll back to the previous version

Today, manual:

```bash
PREV=$(cat /opt/decloud/previous-version)
sudo systemctl stop decloud-node-agent
sudo ln -sfn "/opt/decloud/publish.${PREV}" /opt/decloud/publish.new
sudo mv -Tf /opt/decloud/publish.new /opt/decloud/publish
sudo systemctl start decloud-node-agent
```

### Re-authenticate

```bash
sudo decloud logout
sudo decloud login
```

### View what version is active

```bash
cat /opt/decloud/current-version
ls -la /opt/decloud/publish
```

---

## Recovery scenarios

### `/opt/decloud/publish` is a real directory, blocking the symlink swap

This was a bug in early versions of install.sh. Current install.sh
detects the case and removes the directory automatically before the
swap. If you encounter it on an old install.sh:

```bash
sudo rm -rf /opt/decloud/publish
sudo decloud update
```

### `decloud update` fails with "Could not fetch install.sh and no local copy"

Network is unreachable AND the local fallback is missing. Re-seed:

```bash
sudo curl -fsSL \
  https://github.com/bekirmfr/DeCloud.NodeAgent/releases/latest/download/install.sh \
  -o /usr/local/share/decloud/install.sh
sudo chmod 755 /usr/local/share/decloud/install.sh
sudo decloud update
```

### Cosign signature verification fails

Possible causes:
1. **Network MITM**: investigate, do not bypass
2. **Compromised release**: contact maintainer immediately, do not
   install
3. **Cosign version mismatch**: rare, but if cosign was upgraded
   incompatibly, install.sh's COSIGN_VERSION pin may be stale

Never bypass signature verification. There is no flag to disable it,
intentionally.

### Service fails to start after update

```bash
journalctl -u decloud-node-agent -n 100 --no-pager
```

Look for missing dependencies, permission errors, or appsettings
validation failures. If the new version is broken, manual rollback per
[Common operator tasks](#common-operator-tasks) above.

### Install completed but `decloud login` doesn't reach the wallet prompt

The Python authentication CLI (`cli-decloud-node`) needs the system
Python with `web3`, `eth-account`, `requests`, `qrcode`, `pillow`.
install.sh installs them via `pip install --user`. If the install was
interrupted or the user's Python environment is unusual:

```bash
sudo decloud update             # re-runs install, redoes pip
```

---

## Migration from legacy installs

Earlier (pre-2.2.0) installs used `git clone` of the source repo plus
`dotnet publish` on the node, leaving:

- `/opt/decloud/DeCloud.NodeAgent/` (full git checkout)
- `/opt/decloud/Decloud.Shared/` (DeCloud.Shared clone)
- `/opt/decloud/publish/` (real directory of compiled output)
- `dotnet-sdk-8.0` apt package (compiler on the production node)

When install.sh ≥ 2.2.0 runs on a node previously installed via the
legacy path, the migration block in `download_node_agent`:

1. Detects `/opt/decloud/DeCloud.NodeAgent/.git` and removes that tree
2. Detects `/opt/decloud/Decloud.Shared/.git` and removes it
3. Detects `/opt/decloud/publish` as a real directory (instead of a
   symlink) and removes it before the atomic swap

The migration is **idempotent** and **one-time**: subsequent runs find
nothing to migrate and proceed directly to the swap.

The `dotnet-sdk-8.0` package is **not** removed automatically. To free
the disk space:

```bash
sudo apt-get remove -y dotnet-sdk-8.0
sudo apt-get autoremove -y
```

This is safe — release-mode installs only need the runtime, and the
runtime is installed independently as `aspnetcore-runtime-8.0`.

---

## Files referenced in this document

| Path | Purpose |
| --- | --- |
| `install.sh` | Top-level installer; ~3000 lines, single file |
| `uninstall.sh` | Top-level uninstaller |
| `cli/decloud` | Operator CLI; commands resolve install.sh + uninstall.sh via hybrid curl + local fallback |
| `cli/cli-decloud-node` | Python wallet authentication helper |
| `decloud-relay-nat` | NAT helper for relay-mode nodes |
| `vm-cleanup.sh` | Per-VM cleanup helper |
| `/opt/decloud/publish` | Active version symlink — the systemd unit points here |
| `/etc/decloud/install-params` | Saved install arguments for `decloud update` |
| `/usr/local/share/decloud/install.sh` | Offline fallback for `decloud update` |
| `/usr/local/share/decloud/uninstall.sh` | Offline fallback for `decloud uninstall` |
