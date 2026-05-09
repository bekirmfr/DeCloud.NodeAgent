# Node Agent Lifecycle Reference

What happens from "machine has nothing on it" to "node hosting tenant
workloads" to "node is gone." This document covers two layered
perspectives:

- **Operator lifecycle** — the seven-command contract operators see,
  the states a node passes through, and the cryptographic invariants
  the system holds by construction.
- **Infrastructure mechanics** — what `install.sh`, `decloud update`,
  and `decloud uninstall` actually do on disk: the release fetch flow,
  the trust chain, the file layout, recovery procedures.

New readers should start with [Part 1](#part-1--operator-lifecycle).
Maintainers debugging install/update/uninstall behavior can jump to
[Part 3](#part-3--infrastructure-mechanics).

> **Status notice.** Part 1 describes the **target** operator
> lifecycle — agreed in design but not yet fully implemented.
> [Part 2](#part-2--implementation-status) enumerates the gap between
> target and current. Part 3 describes infrastructure mechanics that
> apply today regardless of which operator-facing model is active.

> **Companion documents:**
> - [`LOCALITY_STANDARDS.md`](LOCALITY_STANDARDS.md) — country/region/zone semantics
> - [`SCHEDULING.md`](SCHEDULING.md) — how the orchestrator places VMs
> - [`RELEASE-PIPELINE.md`](RELEASE-PIPELINE.md) — how the binaries `install.sh` consumes are produced

---

## Table of Contents

**Part 1 — Operator Lifecycle (target)**
- [1. States](#1-states)
- [2. Commands](#2-commands)
- [3. Settings model](#3-settings-model)
- [4. Cryptographic invariants](#4-cryptographic-invariants)
- [5. VM compliance handling](#5-vm-compliance-handling)
- [6. Trust model](#6-trust-model)

**Part 2 — Implementation Status**
- [7. What's deployed today](#7-whats-deployed-today)
- [8. The deferred work](#8-the-deferred-work)

**Part 3 — Infrastructure Mechanics (current)**
- [9. Overview of operations](#9-overview-of-operations)
- [10. Fresh install](#10-fresh-install)
- [11. Update](#11-update)
- [12. Uninstall](#12-uninstall)
- [13. On-disk layout](#13-on-disk-layout)
- [14. Configuration](#14-configuration)
- [15. Trust chain at runtime](#15-trust-chain-at-runtime)
- [16. Common operator tasks](#16-common-operator-tasks)
- [17. Recovery scenarios](#17-recovery-scenarios)
- [18. Migration from legacy installs](#18-migration-from-legacy-installs)

---

# Part 1 — Operator Lifecycle

## 1. States

A node passes through five states. Each transition is driven by exactly
one operator command.

```
                                           configure
                                       ┌───── (locality only;
                                       │   re-register required)
                                       │
GONE ──install──► INSTALLED ──configure──► CONFIGURED ──register──► ENROLLED ──login──► ACTIVE
 ▲                                                                       ▲    │
 │                                                                       │    │
 │                                                                  login│    │ logout
 │                                                                       └────┘
 │
 └─────────────────────── uninstall (from any state) ─────────────────────────
```

| State | What it means | What's persisted |
| --- | --- | --- |
| `GONE` | Nothing on the machine | Nothing |
| `INSTALLED` | Binaries on disk, wallet declared, orchestrator URL recorded | Minimal `/etc/decloud/settings` (wallet + orchestrator); systemd unit installed but service idle |
| `CONFIGURED` | Operational settings declared (locality, name, rates) but orchestrator hasn't seen them | Full `/etc/decloud/settings` |
| `ENROLLED` | Orchestrator knows this node, has issued credentials, has computed obligations | Above + `/etc/decloud/credentials` (JWT, refresh token, obligations) |
| `ACTIVE` | Heartbeating, with a fresh wallet signature attesting the node is running registered settings | All of the above + active heartbeat loop |

**Important:** `ENROLLED ⇄ ACTIVE` oscillates freely. A node can `logout`
and `login` repeatedly without re-running register or configure. Login
signs the *current* settings; if local settings have drifted from what
was enrolled, login fails until the operator re-registers.

---

## 2. Commands

Seven commands cover every state transition.

### `decloud install`

Installs binaries; prompts for wallet address; records orchestrator URL.

| Aspect | Detail |
| --- | --- |
| **Inputs** | `--wallet 0x...` (required), `--orchestrator <URL>` (defaults to `https://decloud.stackfi.tech`) |
| **State change** | `GONE` → `INSTALLED` |
| **Side effects** | Binaries placed at `/opt/decloud/`; CLI at `/usr/local/bin/decloud`; systemd unit installed; minimal `/etc/decloud/settings` written |
| **Network** | Fetches release artifacts from GitHub; verifies cosign + SHA-256 |
| **Failure mode** | Binaries removed on rollback; wallet not yet declared; node remains GONE |

After install the node is on disk but has no operational settings beyond
wallet/orchestrator. The agent service is not started. The operator
must `configure` next.

### `decloud configure`

Commits operational settings: locality, display name, service rates,
capacity preferences.

| Aspect | Detail |
| --- | --- |
| **Inputs** | `--country <CC>`, `--region <region>`, `--zone <zone>`, `--name <text>`, `--description <text>`, `--rate-cpu-hour <amount>`, etc. |
| **State change** | `INSTALLED` → `CONFIGURED` (or `CONFIGURED` → `CONFIGURED'` for updates) |
| **Side effects** | Updates `/etc/decloud/settings`; regenerates `appsettings.Production.json`; restarts agent service |
| **Network** | None — purely local |
| **Validation** | Country must be in `countries.json`; region in `regions.json`; zone format validated; client-side check via `--validate` |
| **Refused if** | Node is `ENROLLED` and the change includes locality fields — locality changes on enrolled nodes go through `register` (which validates VM compliance) |

The settings written are *staged*: the agent will read them on next
start, but the orchestrator has not seen them. The operator must
`register` to commit to the orchestrator.

Cosmetic-only updates (`--name`, `--description`) on enrolled nodes
are pushed to the orchestrator immediately via the existing profile
endpoint — no re-register needed.

### `decloud register`

Pushes current settings to the orchestrator; receives credentials and
computes obligations.

| Aspect | Detail |
| --- | --- |
| **Inputs** | None (reads `/etc/decloud/settings`) |
| **State change** | `CONFIGURED` → `ENROLLED` (first registration) or `ENROLLED` → `ENROLLED'` (re-registration) |
| **Side effects** | Wallet-signing flow (QR + sign.html); writes `/etc/decloud/credentials`; stamps `/etc/decloud/settings.locality` block with signature |
| **Network** | `POST /api/nodes/register` |
| **Authorization** | Wallet signature over canonical locality message; 5-minute validity window with ±2-minute orchestrator skew tolerance |
| **VM compliance** | Re-registration path: orchestrator walks running VMs; flags non-compliant ones for migration scheduler |

The orchestrator's response includes:
- JWT (used by subsequent authenticated calls)
- Obligations: system VMs to host (DHT/BlockStore/Relay), attestation cadence
- List of any non-compliant VMs (re-registration only)

The canonical locality message format:

```
DeCloud Node Locality Declaration

Country:    DE
Region:     eu-central
Machine ID: <machine-id>
Wallet:     <wallet-address>
Timestamp:  2026-05-09T11:42:30Z

This signature attests that the operator of this wallet declares
the above DeCloud node to be located in the stated country and
network region. The orchestrator will use this declaration for
jurisdictional placement of tenant workloads.
```

Note: only `country` and `region` are in the signed payload. `zone` is
operator-side organizational metadata and is not signed.

### `decloud login`

Activates the heartbeat loop. Proves to the orchestrator that the node
is currently running with the registered settings.

| Aspect | Detail |
| --- | --- |
| **Inputs** | None (reads `/etc/decloud/credentials` and `/etc/decloud/settings`) |
| **State change** | `ENROLLED` → `ACTIVE` |
| **Side effects** | Wallet-signs canonical locality message with current timestamp; agent begins heartbeat loop |
| **Network** | `POST /api/nodes/{id}/login` (with fresh signature) |
| **Failure mode** | If orchestrator computes a different canonical message than the one signed (settings drift), login is rejected with structured error pointing at the drifted field |

This is the cryptographic checkpoint of the lifecycle. The wallet signs
the live settings; the orchestrator validates against what it has
stored from register; mismatch means either the node has been tampered
with or settings were edited locally without re-registering.

The signature uses the same canonical message format as register. The
5-minute validity window applies — operator must complete the login
flow within 5 minutes of the wallet signing.

### `decloud logout`

Clears local credentials. Optional courtesy notification to orchestrator.

| Aspect | Detail |
| --- | --- |
| **Inputs** | None |
| **State change** | `ACTIVE` → `ENROLLED` (orchestrator-side state preserved) |
| **Side effects** | Removes `/etc/decloud/credentials`; stops heartbeat loop |
| **Network** | Best-effort `POST /api/nodes/{id}/logout` (graceful pause) |

The node remains enrolled at the orchestrator. JWT is technically still
valid (in the orchestrator's records) until the next `register` or
`uninstall`. To resume operation, the operator runs `login` again —
which re-signs current settings, validates against the orchestrator's
stored state, and resumes heartbeating.

If the operator skips logout and just stops the service (or the node
crashes), the orchestrator's heartbeat-timeout safety net detects
silence within ~5 minutes and marks the node Offline. The two
mechanisms coexist: explicit logout for graceful pause, timeout as
the fallback.

### `decloud vm drain` / `decloud vm migrate`

Workload movement primitives. **Deferred to Phase 2.5b** — stub
implementations in v1 print "not yet implemented; manually migrate or
destroy VMs before uninstall."

When implemented, these will be:

**`decloud vm drain`** — Mark this node as winding down. Orchestrator
stops scheduling new VMs onto it; existing VMs are flagged for
migration to other eligible nodes via the migration scheduler.

**`decloud vm drain --to <node-id>`** — Same as above, but prefer
migration to a specific operator-controlled target node. Wallet
equality between source and target is the security boundary that
distinguishes "self-pool migration" from "transfer to a different
operator." Standard scheduling filters still apply on the target
node; if it's ineligible for a given VM, that VM falls back to
general scheduling.

**`decloud vm migrate <vm-id> --to <node-id>`** — Migrate one specific
VM to a chosen target. Same wallet-equality rule.

These commands stay as stubs in v1 to establish the operator-facing
CLI surface. Operators discover them via `decloud vm --help`. When
2.5b lands the surface is unchanged; only the implementation behind
the stubs changes.

### `decloud uninstall`

Full teardown.

| Aspect | Detail |
| --- | --- |
| **Inputs** | `--force` (skip confirmation prompts; destroy running VMs) |
| **State change** | Any → `GONE` |
| **Side effects** | Notifies orchestrator (deregister); removes binaries, credentials, settings; optionally removes `decloud` user |
| **Network** | `POST /api/nodes/{id}/deregister` (best-effort); orchestrator removes node record and adds JWT to revocation list |
| **Refused if** | Running VMs detected and `--force` not provided |
| **JWT revocation** | Orchestrator-side revocation list ensures stale credentials cannot authenticate even if re-presented later |

Uninstall is the only path that produces the GONE state. There is no
separate `deregister` command — deregistration is a side effect of
uninstall, never an independent operation. (Earlier design rounds
explored a separate `decloud deregister`; we landed on uninstall-as-
the-only-path because the cases for needing deregister independently
of uninstall did not survive scrutiny.)

---

## 3. Settings model

`/etc/decloud/settings` is a JSON file (mode 600) holding the node's
operational configuration.

```json
{
  "version": 1,
  "wallet": "0x86b8fE9ad3b4596a66b2C586F988A04f03be45F9",
  "orchestrator_url": "https://decloud.stackfi.tech",
  "locality": {
    "country": "DE",
    "region": "eu-central",
    "zone": "eu-central-1",
    "signature": "0x...",
    "signed_at": "2026-05-09T11:42:30Z"
  },
  "profile": {
    "name": "MyHetznerBox",
    "description": "16-core EU node, NVMe storage"
  },
  "rates": {
    "cpu_hour_eur": 0.04,
    "ram_gb_hour_eur": 0.005,
    "storage_gb_month_eur": 0.10
  }
}
```

### Field categories and mutability

| Field | Category | Mutable while ENROLLED? | Authorization |
| --- | --- | --- | --- |
| `wallet` | identity | Never (re-install required) | Cannot change |
| `orchestrator_url` | identity | Never | Cannot change |
| `locality.country` | locality (signed) | Re-register required | Wallet signature |
| `locality.region` | locality (signed) | Re-register required | Wallet signature |
| `locality.zone` | locality (cosmetic) | Re-register required | None (not signed) |
| `profile.name` | cosmetic | Yes (no re-register) | JWT only |
| `profile.description` | cosmetic | Yes (no re-register) | JWT only |
| `rates.*` | service | Re-register required | Wallet signature |

Cosmetic updates flow through `decloud configure --name "X"` and use
the orchestrator's profile endpoint directly — no re-register needed.

Locality and rate updates require re-register: the orchestrator
receives the new signed declaration, validates VM compliance, and
returns any flagged VMs in the response.

### Why zone is not signed

Zone is operator-scoped organizational metadata. Two nodes in different
zones don't promise failure independence — they're an operator-side
labeling for "rack 1" vs "rack 2" or similar. Tenants can filter by
zone when they care, but zone has no jurisdictional or trust meaning
that would justify cryptographic binding.

Country and region do carry trust meaning: country determines legal
jurisdiction; region determines the locality bloc tenants pay for.
Both are wallet-signed. Zone follows alongside but isn't part of the
signed payload.

---

## 4. Cryptographic invariants

The system holds these invariants by construction:

**(a) Server-authoritative settings.** What the orchestrator believes
about a node's locality is what was registered. Local edits to
`/etc/decloud/settings` without re-register cannot change orchestrator-
side state.

**(b) Login proves current local state matches registered state.**
Every login signs the *current* canonical locality message. The
orchestrator computes the canonical message it expects (from
registered settings) and compares; signatures over different messages
recover different signers. A drift in country or region between local
and registered settings causes login to fail with a structured
"settings drift detected" error.

**(c) Wallet ownership required for jurisdictional changes.** A leaked
JWT cannot change locality. Country/region updates require fresh wallet
signature; the JWT alone is insufficient. (This is the entire point of
the wallet-signature-bound design.)

**(d) Revoked credentials stay revoked.** JWT revocation list survives
orchestrator restarts (MongoDB-backed). A node uninstalled today cannot
have its JWT replayed tomorrow. The check is on every authenticated
request, not just heartbeat.

**(e) Signature freshness bounded.** All wallet signatures expire 5
minutes after the embedded timestamp, with ±2 minutes orchestrator
skew tolerance. Captured signatures cannot be replayed days later.
The timestamp is part of the signed payload, so an attacker cannot
adjust it without invalidating the signature.

These five together mean: if a node is `ACTIVE` and heartbeating, the
orchestrator has cryptographic proof that the node is currently running
the locality settings the wallet authorized within the last few
minutes.

---

## 5. VM compliance handling

When `decloud register` runs on an already-enrolled node and the
locality has changed, the orchestrator:

1. Validates the new locality (country in `countries.json`, region in
   `regions.json`)
2. Verifies wallet signature over canonical message
3. Walks all running VMs on this node, checking each against the new
   locality:
   - `RequiredJurisdictionTag` — does the new country still carry it?
   - `RequiredCountry` — does the new country match?
   - `ForbiddenCountries` — is the new country in the forbidden list?
4. For each non-compliant VM:
   - Sets `VirtualMachine.NonCompliantSince` and `NonComplianceReason`
   - Migration scheduler picks them up on next cycle and moves them
     to compliant nodes
5. Returns the registration response with the list of flagged VMs

The operator sees:

```
✓ Locality updated: TR/eu-east → BR/sa-east
⚠ 2 VMs flagged for migration (no longer compliant):
    vm-abc123  required jurisdiction tag EU; node is now in BR
    vm-def456  required country DE; node is now in BR
ℹ Migration scheduler will move these to compliant nodes within ~5 minutes.
```

This is **accept and flag**, not reject. The operator has authority
over their node; the orchestrator doesn't refuse the locality change.
Tenant workload safety is preserved by the migration scheduler picking
up flagged VMs on its next cycle.

The flagging mechanism uses two new fields on `VirtualMachine`:

```csharp
public DateTime? NonCompliantSince { get; set; }
public string? NonComplianceReason { get; set; }
```

The migration scheduler treats `NonCompliantSince != null` as a
trigger to schedule replacement, the same way it treats `VmStatus.Error`
as a trigger today.

---

## 6. Trust model

Three credentials, three purposes:

| Credential | Source | Lifetime | Used for |
| --- | --- | --- | --- |
| **Wallet** | Operator's external wallet (MetaMask, hardware wallet, etc.) | Forever (operator-controlled) | Initial registration; locality changes; rate changes |
| **JWT** | Issued by orchestrator at register | Hours, refreshable | Heartbeat, profile updates, normal API calls |
| **Locality signature** | Generated at register/login from wallet | 5 minutes from `signed_at` | Validating local settings match registered settings |

The wallet is the trust root. JWTs are convenience tokens issued after
wallet authentication. Locality signatures are short-lived attestations
that "yes, the wallet currently endorses these settings."

Three-credential split lets the JWT carry routine traffic without
involving the wallet, while ensuring jurisdictionally-significant
operations (initial register, locality change, rate change, etc.)
require fresh wallet involvement.

---

# Part 2 — Implementation Status

## 7. What's deployed today

The currently-deployed lifecycle is simpler than the target. Two
operational states, four commands:

```
GONE ──install──► RUNNING ──login──► AUTHENTICATED ──logout──► RUNNING ──uninstall──► GONE
                                                       │ uninstall
                                                       └────────────────────────────────► GONE
```

| Deployed command | Behavior | Target equivalent |
| --- | --- | --- |
| `install` | Prompts for wallet, country, region, zone, name; installs binaries; saves to `install-params` | `install` + `configure` (combined) |
| `login` | Wallet signing flow → registers with orchestrator → obtains JWT → starts heartbeat | `register` + `login` (combined, no separate enrollment step) |
| `logout` | Removes credentials; stops service | `logout` |
| `uninstall` | Deregisters with orchestrator; removes binaries | `uninstall` (no JWT revocation list) |

Today's `decloud login` does what target's `register` and `login`
together do — wallet signature, registration request, JWT receipt,
heartbeat start, all in one flow. There's no distinction between
"enrolled" and "active."

The deployed model works for the basic case (install once, declare
locality at install, run forever, eventually uninstall). It does not
support:
- Settings updates without full reinstall (every change goes through
  edit-install-params + `decloud update`)
- Distinct enroll vs activate steps
- Cryptographic detection of local settings drift (no signature is
  stored against which to validate)
- VM compliance flagging on locality change (no re-register-with-
  validation flow)
- JWT revocation list (uninstall removes node record but doesn't
  blacklist the JWT)

## 8. The deferred work

Implementing the target lifecycle requires changes across CLI, node
agent runtime, and orchestrator:

### Node-side (CLI changes)

- New `decloud configure` subcommand: read/write JSON settings;
  regenerate appsettings; restart agent; refused for locality changes
  while enrolled
- Split current `decloud login` into:
  - `decloud register` — wallet signing + registration request +
    credentials capture
  - `decloud login` — sign current settings + activate heartbeat
- New `decloud vm drain` and `decloud vm migrate` stub subcommands
  (CLI surface for 2.5b)
- Settings file format migration: `install-params` (newline-separated
  args) → `settings` (JSON with locality block)
- `install.sh` slim-down: prompt for wallet only; locality moves to
  `decloud configure`

### Node-agent runtime

- Read settings from JSON `/etc/decloud/settings` instead of
  `install-params`-derived `appsettings.Production.json`
- Locality signature generation hook (invoke wallet helper with
  current canonical message)
- Heartbeat-pause flag at `/etc/decloud/logged-out` (signals graceful
  pause without removing credentials)

### Orchestrator-side

- New `POST /api/nodes/{id}/login` endpoint: validates locality
  signature against stored state; returns structured drift error
  on mismatch
- Locality signature validation in `RegisterNodeAsync`: canonical
  message construction, timestamp window check, signature verification
- `FlagNonCompliantVms` helper: walks running VMs, checks each
  against new locality, sets `NonCompliantSince` / `NonComplianceReason`
- New `JwtRevocationService`: MongoDB collection `revoked_jwts`,
  in-memory cache, middleware integration in JWT validation
- Deregister endpoint formalization: revoke JWT on success, refuse
  deregister with running tenant VMs unless `--force`

### Documentation

- Operator-facing migration guide: how existing nodes (using deployed
  model) move to the target lifecycle when it ships
- The `decloud configure` reference page
- The locality-signature semantics page (probably folded into
  `LOCALITY_STANDARDS.md` since it's a locality concern)

### Why deferred

The deployed model meets current operational needs. There is no
production-driving requirement that the simpler model fails. The
target design above is documented now to:
1. Prevent design churn next time the topic comes up
2. Give a clear contract for when the work does become priority
3. Anchor any partial work (e.g., adding JWT revocation alone)
   against an agreed end state

When operators have a real, repeated need to change settings without
reinstalling — or when a security review elevates JWT revocation to
required — this work moves from deferred to scheduled.

The current focus per project priorities is **scheduling and locality
implementation**, not the lifecycle redesign. Lifecycle work resumes
when those land and a driver materializes.

---

# Part 3 — Infrastructure Mechanics

> **Scope:** This part describes how `install.sh`, `decloud update`,
> and `decloud uninstall` work on disk today. The mechanics here apply
> regardless of which operator-facing model (deployed or target) is
> active — they're concerned with binary placement, signature
> verification, version management, and recovery, not with operator
> command semantics.

## 9. Overview of operations

Three lifecycle operations, each with one canonical entry point:

| Operation | Entry point | Reads from | Writes to | Verifies |
| --- | --- | --- | --- | --- |
| Install | `curl ... \| sudo bash` | GitHub Release | `/opt/decloud`, `/usr/local/bin`, `/etc/decloud` | Cosign + SHA-256 |
| Update | `sudo decloud update` | GitHub Release (or local fallback) | Same as install | Cosign + SHA-256 |
| Uninstall | `sudo decloud uninstall` | GitHub Release (or local fallback) | Removes most of the above | None (cleanup only) |

Update is "install with saved parameters" — same script, same code path,
just non-interactive. There is one installation procedure.

## 10. Fresh install

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

Wall-clock time on a fresh Ubuntu box: 3–5 minutes. On a re-run with
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
     controlling terminal is available, otherwise errors with copy-
     pasteable correction (see [Interactive prompts](#interactive-prompts) below)

3. **Preflight checks**
   - OS detection (Ubuntu 22.04+/24.04 supported)
   - Architecture (x86_64 or aarch64)
   - Hardware virtualization support (`/proc/cpuinfo` flags)
   - Free RAM ≥ 4 GB, disk ≥ 50 GB
   - Ports 5100 (Agent API) and 51821 (WireGuard) free
   - Outbound HTTPS to orchestrator URL works

4. **Cosign installation**
   - Pinned to the version specified in `install.sh`'s `COSIGN_VERSION`
   - Downloaded from Sigstore release; SHA-256 verified against value
     in install.sh
   - Installed to `/usr/local/bin/cosign`

5. **Release manifest fetch and verification**
   - Manifest URL: `releases/latest/download/manifest.json`
   - Manifest signature URL: `manifest.json.sig`
   - Manifest cert URL: `manifest.json.pem`
   - `cosign verify-blob` with pinned identity regex (matches
     `bekirmfr/DeCloud.NodeAgent` workflow)
   - Failure here aborts the install with no files written

6. **Artifact downloads (per manifest)**
   - Each tarball downloaded; SHA-256 compared to manifest entry
   - Architecture-specific: x86_64 box gets the amd64 tarball; ARM64
     box gets the arm64 tarball
   - GPU shim only downloaded if `detect_gpu_mode` selected proxy mode

7. **Binary placement (atomic version swap)**
   - Tarballs extracted to `/opt/decloud/publish.<version>/`
   - `appsettings.Production.json` generated from install-params
   - Active symlink swapped: `/opt/decloud/publish` → `publish.<version>`
   - Previous version preserved as `publish.<previous-version>` for
     rollback

8. **System integration**
   - SSH CA setup: ed25519 key at `/etc/decloud/ssh-ca/`
   - `decloud` system user creation (for SSH jump host)
   - sshd config additions: `TrustedUserCAKeys`, `Match User decloud`
   - WireGuard key generation (operator-side)
   - libvirt setup verified
   - `cli/decloud` and `cli/cli-decloud-node` placed in `/usr/local/bin/`
   - `install.sh` and `uninstall.sh` copies preserved at
     `/usr/local/share/decloud/` for offline `decloud update`/`decloud uninstall`

9. **systemd unit creation and start**
   - Unit file: `/etc/systemd/system/decloud-node-agent.service`
   - `WorkingDirectory=/opt/decloud/publish`
   - `ExecStart=/usr/bin/dotnet /opt/decloud/publish/DeCloud.NodeAgent.dll`
   - Service enabled and started; idle until login

10. **Done**
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

When run truly non-interactively (CI, no controlling terminal),
missing required flags abort with copy-pasteable usage examples.

## 11. Update

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
- That install.sh knows how to install `vN` (and forward, given
  backward compat)
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
  [Recovery scenarios](#17-recovery-scenarios).

## 12. Uninstall

```bash
sudo decloud uninstall
```

Same hybrid resolution as update: tries to fetch the latest
`uninstall.sh` from GitHub, falls back to
`/usr/local/share/decloud/uninstall.sh`.

### What it does

1. Deregisters with the orchestrator (best-effort — proceeds even on
   failure)
2. Stops the agent service
3. Removes binaries from `/opt/decloud/`
4. Removes the active symlink
5. Removes systemd unit file
6. Removes `/usr/local/bin/decloud`, `/usr/local/bin/cli-decloud-node`,
   and other CLI bits
7. Removes `/etc/decloud/` (credentials, install-params, SSH CA)
8. Optionally removes the `decloud` system user
9. Reverts sshd config additions

The user is prompted to confirm before destructive actions unless
`--force` is passed.

## 13. On-disk layout

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

### Future change (target operator lifecycle)

When the target operator lifecycle (Part 1) ships,
`/etc/decloud/install-params` will be replaced by `/etc/decloud/settings`
(JSON format). Existing nodes will be migrated automatically on first
install/update under the new model. Until that work lands, the file
remains `install-params` with the args-pair format described below.

## 14. Configuration

### Where settings live (current implementation)

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

> **Future:** Under the target lifecycle (Part 1), settings changes
> happen via `decloud configure --country DE`, with locality changes
> requiring `decloud register` to commit cryptographically. This
> deferred work is documented above in [Part 2](#part-2--implementation-status).

## 15. Trust chain at runtime

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

## 16. Common operator tasks

| Task | Command |
| --- | --- |
| Check service status | `sudo systemctl status decloud-node-agent` |
| Read recent logs | `sudo journalctl -u decloud-node-agent -n 100 --no-pager` |
| Tail logs live | `sudo journalctl -u decloud-node-agent -f` |
| Restart service | `sudo systemctl restart decloud-node-agent` |
| Update to latest release | `sudo decloud update` |
| Re-authenticate | `sudo decloud logout && sudo decloud login` |
| Inspect installed version | `cat /opt/decloud/current-version` |
| Manual rollback to previous version | `sudo ln -sfn /opt/decloud/publish.$(cat /opt/decloud/previous-version) /opt/decloud/publish && sudo systemctl restart decloud-node-agent` |
| Check installed binaries | `ls -la /opt/decloud/publish/` |
| View saved install params | `sudo cat /etc/decloud/install-params` |

## 17. Recovery scenarios

### Service fails to start after update

```bash
journalctl -u decloud-node-agent -n 100 --no-pager
```

Look for missing dependencies, permission errors, or appsettings
validation failures. If the new version is broken, manual rollback per
[Common operator tasks](#16-common-operator-tasks) above.

### Cosign verification fails

`install.sh` exits before any files are written. Investigate possible
causes:

1. **Network MITM**: investigate; do not bypass
2. **Compromised release**: contact maintainer immediately; do not
   install
3. **Cosign version mismatch**: rare, but if cosign was upgraded
   incompatibly, install.sh's `COSIGN_VERSION` pin may be stale

Never bypass signature verification. There is no flag to disable it,
intentionally.

### Install completed but `decloud login` doesn't reach the wallet prompt

The Python authentication CLI (`cli-decloud-node`) needs the system
Python with `web3`, `eth-account`, `requests`, `qrcode`, `pillow`.
install.sh installs them via `pip install --user`. If the install was
interrupted or the user's Python environment is unusual:

```bash
sudo decloud update             # re-runs install, redoes pip
```

### Node agent loses connection to orchestrator

This is normal at the orchestrator level — heartbeat will time out
after 5 minutes and the orchestrator marks the node Offline. When
connectivity returns, the agent's next heartbeat brings it back to
Online automatically.

If heartbeat continues to fail when connectivity is healthy, check:

```bash
# Credentials still present?
sudo ls -la /etc/decloud/credentials

# Agent reaching orchestrator?
sudo journalctl -u decloud-node-agent -n 50 | grep -i 'heartbeat\|orchestrator'
```

If credentials are missing, run `sudo decloud login` to re-authenticate.

## 18. Migration from legacy installs

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
| `/etc/decloud/install-params` | Saved install arguments for `decloud update` (current; will become `settings` JSON under target lifecycle) |
| `/etc/decloud/credentials` | JWT credentials (after login) |
| `/usr/local/share/decloud/install.sh` | Offline fallback for `decloud update` |
| `/usr/local/share/decloud/uninstall.sh` | Offline fallback for `decloud uninstall` |
