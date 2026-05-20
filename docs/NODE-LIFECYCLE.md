# Node Agent Lifecycle Reference

What happens from "machine has nothing on it" to "node hosting tenant
workloads" to "node is gone." This document covers two layered
perspectives:

- **Operator lifecycle** — the nine-command contract operators see,
  the states a node passes through, and the cryptographic invariants
  the system holds by construction.
- **Infrastructure mechanics** — what `install.sh`, `decloud update`,
  and `decloud uninstall` actually do on disk: the release fetch flow,
  the trust chain, the file layout, recovery procedures.

New readers should start with [Part 1](#part-1--operator-lifecycle).
Maintainers debugging install/update/uninstall behavior can jump to
[Part 3](#part-3--infrastructure-mechanics).

> **Status notice.** Part 1 describes the operator lifecycle as
> implemented. [Part 2](#part-2--implementation-status) covers
> implementation status. Remaining work: live migration (Phase 2.5b).
> Part 3 describes infrastructure mechanics.

> **Companion documents:**
> - [`LOCALITY_STANDARDS.md`](LOCALITY_STANDARDS.md) — country/region/zone semantics
> - [`SCHEDULING.md`](SCHEDULING.md) — how the orchestrator places VMs
> - [`RELEASE-PIPELINE.md`](RELEASE-PIPELINE.md) — how the binaries `install.sh` consumes are produced
> - [`NODE-LIFECYCLE-RESOURCE-REDESIGN.md`](NODE-LIFECYCLE-RESOURCE-REDESIGN.md) — gap analysis and implementation plan for the evaluate/allocate separation

---

## Table of Contents

**Part 1 — Operator Lifecycle**
- [1. States](#1-states)
- [2. Commands](#2-commands)
- [3. Resource evaluation and allocation](#3-resource-evaluation-and-allocation)
- [4. Settings model](#4-settings-model)
- [5. Cryptographic invariants](#5-cryptographic-invariants)
- [6. Continuous drift detection](#6-continuous-drift-detection)
- [7. VM compliance handling](#7-vm-compliance-handling)
- [8. Trust model](#8-trust-model)

**Part 2 — Implementation Status**
- [9. What's implemented](#9-whats-implemented)
- [10. Remaining work](#10-remaining-work)

**Part 3 — Infrastructure Mechanics**
- [11. Overview of operations](#11-overview-of-operations)
- [12. Fresh install](#12-fresh-install)
- [13. Update](#13-update)
- [14. Uninstall](#14-uninstall)
- [15. On-disk layout](#15-on-disk-layout)
- [16. Configuration](#16-configuration)
- [17. Trust chain at runtime](#17-trust-chain-at-runtime)
- [18. Common operator tasks](#18-common-operator-tasks)
- [19. Recovery scenarios](#19-recovery-scenarios)
- [20. Migration from legacy installs](#20-migration-from-legacy-installs)

---

# Part 1 — Operator Lifecycle

## 1. States

A node passes through four states. Each transition is driven by
exactly one operator command.

```
                                     configure
                                  ┌──── (requires logout first
                                  │      for locality/rate changes)
                                  │
GONE ──install──► CONFIGURED ──register──► ENROLLED ──login──► ACTIVE
 ▲                                            ▲    │               │
 │                                            │    │               │
 │                                       login│    │ logout        │ logout
 │                                            │    │               │
 │                                            └────┘               │
 │                                            ▲                    │
 │                                            └────────────────────┘
 │
 └─────────────────── uninstall (from any state) ──────────────────
```

Within the ENROLLED state, two additional operations enrich the node
before it goes live:

```
                       ENROLLED
           ┌───────────────────────────────┐
           │                               │
register──►│  evaluate ──► allocate        │──login──► ACTIVE
           │  (orchestrator   (operator    │
           │   benchmarks,     sets        │
           │   assigns         resource    │
           │   obligations)    percentages)│
           │                               │
           └───────────────────────────────┘
```

| State | What it means | What's persisted |
| --- | --- | --- |
| `GONE` | Nothing on the machine | Nothing |
| `CONFIGURED` | Binaries on disk, wallet declared, operational settings written, orchestrator hasn't seen them | `/opt/decloud/` binaries; `/etc/decloud/settings` (full); systemd unit installed but service idle |
| `ENROLLED` | Orchestrator knows this node, has issued credentials; node is heartbeating but **not schedulable**. Evaluation and allocation happen here. | Above + `/etc/decloud/credentials` (JWT); active heartbeat loop; optionally: cached performance evaluation, obligation list, allocation percentages |
| `ACTIVE` | Heartbeating **and schedulable** — orchestrator has computed concrete capacity from allocation percentages and will place VMs here | Same as ENROLLED + scheduling flag set + `TotalResources` materialized |

### Why four states, not six

Evaluate and allocate are operations within the ENROLLED state, not
separate lifecycle states. A node that has registered but not yet
evaluated is still ENROLLED — it heartbeats, it has credentials, the
orchestrator knows it exists. The difference is that the scheduler
will refuse to activate it (login will fail) until evaluation is
complete. This is a precondition on login, not a distinct state.

Adding EVALUATED and ALLOCATED as separate states would model
checkpoints inside a preparation sequence, not meaningful operational
postures. The node's relationship with the rest of the system does not
change between "enrolled and evaluated" and "enrolled and allocated" —
in both cases it is heartbeating and not schedulable.

Earlier design rounds also separated `INSTALLED` (binaries + wallet
only) from `CONFIGURED` (binaries + wallet + operational settings). In
practice every installation provides operational settings in the same
step — there is no useful operational posture where the node has
binaries and a wallet but no locality. Merging the two removes a state
that existed to model a checkpoint in `install.sh`, not a meaningful
node posture.

### ENROLLED vs ACTIVE: the scheduling boundary

`ENROLLED ⇄ ACTIVE` is an operational toggle, not a cryptographic
ceremony.

An `ENROLLED` node is heartbeating — the orchestrator sees its
metrics, sees its running VMs, knows it's alive. But the scheduler
will not place new VMs on it. This gives the operator a window to
evaluate, allocate, change settings, wait for migrations, perform
maintenance, or simply pause.

An `ACTIVE` node is enrolled, evaluated, allocated, and additionally
schedulable. `login` flips the flag; `logout` clears it.

The wallet is not involved in this toggle. The wallet speaks at
`register`, where it has something to say ("I attest to this
locality"). Login and logout speak at the scheduler, where they have
something to say ("I'm ready" or "I'm pausing"). Each signal goes to
the system boundary where it belongs.

### Unattended restart

Because login is a lightweight, JWT-authenticated readiness signal
(not a wallet ceremony), the node agent can auto-login on startup if
valid credentials exist on disk and evaluation has previously
completed. A node that reboots at 3 AM comes back online and
schedulable without operator intervention.

The agent startup sequence:

1. Check for `/etc/decloud/credentials` — if absent, wait for
   `decloud register`
2. Credentials present → begin heartbeating (ENROLLED)
3. Check for cached performance evaluation — if absent, remain
   ENROLLED (operator must run `decloud evaluate`)
4. Evaluation present + no `/etc/decloud/logged-out` sentinel →
   auto-call `POST /api/nodes/{id}/login` using existing JWT → ACTIVE

If the operator logged out before the reboot (the `logged-out`
sentinel file exists at `/etc/decloud/logged-out`), the agent
heartbeats but does **not** auto-login. The operator's intent to
pause is preserved across restarts.

---

## 2. Commands

Nine commands cover every state transition.

### `decloud install`

Installs binaries, records wallet address and orchestrator URL.
Operational settings (locality, name) are handled separately by
`decloud configure`.

| Aspect | Detail |
| --- | --- |
| **Inputs** | `--wallet 0x...` (required), `--orchestrator <URL>` (defaults to `https://decloud.stackfi.tech`) |
| **State change** | `GONE` → `CONFIGURED` |
| **Side effects** | Binaries placed at `/opt/decloud/`; CLI at `/usr/local/bin/decloud`; systemd unit installed; `/etc/decloud/settings.json` written with identity fields; `appsettings.Production.json` generated with defaults |
| **Network** | Fetches release artifacts from GitHub; verifies cosign + SHA-256 |
| **Failure mode** | Binaries removed on rollback; node remains GONE |
| **Validation** | Wallet format (0x + 40 hex, not null address) |

After install the node is on disk with binaries and identity but no
operational settings. The agent service is installed and starts with
defaults (hostname for name, "default" for region/zone). The operator
must `configure` and then `register` to become operational.

### `decloud configure`

Updates operational settings: locality, display name, service rates.
Resource allocation is handled separately by `decloud allocate`.

| Aspect | Detail |
| --- | --- |
| **Inputs** | `--country <CC>`, `--region <region>`, `--zone <zone>`, `--name <text>`, `--description <text>` |
| **State change** | `CONFIGURED` → `CONFIGURED'` (settings updated locally) |
| **Side effects** | Updates `/etc/decloud/settings.json`; pushes cosmetic changes (name, description) to orchestrator if enrolled |
| **Network** | None for locality/rate changes (local staging); cosmetic changes pushed via PATCH if enrolled |
| **Validation** | Country: `^[A-Z]{2}$`; region: `^[a-z]+(-[a-z]+)*$`; zone: `^<region>-[1-9][0-9]*$` (client-side format check; existence validated server-side at register) |
| **Refused if** | Node is ENROLLED or ACTIVE and the change includes locality or rate fields. Operator must `logout` first, then configure, then `register` to commit. |

The settings written are *staged*: the agent will read them on next
start, but the orchestrator has not seen them. The operator must
`register` to commit to the orchestrator.

**Cosmetic-only updates** (`--name`, `--description`) on enrolled or
active nodes are allowed without logout. These are pushed to the
orchestrator immediately via the existing profile endpoint — no
re-register needed.

**Locality and rate changes** on enrolled nodes require the full
sequence:

```bash
sudo decloud logout                      # stop scheduling, open settings window
sudo decloud configure --country BR --region sa-east
sudo decloud register                    # wallet ceremony, VM compliance check
sudo decloud evaluate                    # re-benchmark (optional, if hardware unchanged)
# ... wait for migrations if flagged VMs ...
sudo decloud login                       # resume scheduling when ready
```

### Why configure requires logout for locality/rate changes

Accepting new VMs under settings the operator is about to change is
wrong. An enrolled node declaring `country: DE` with VMs placed under
German jurisdiction requirements should not accept more VMs while
the operator is mid-change to `country: BR`. Logout removes the node
from scheduling *before* settings change, not after.

The refusal is in the CLI, not the orchestrator. `decloud configure
--country BR` while ENROLLED or ACTIVE prints:

```
✗ Cannot change locality while enrolled. Run 'decloud logout' first.

  The full sequence for locality/rate changes:
    1. sudo decloud logout          # pause scheduling
    2. sudo decloud configure ...   # change settings
    3. sudo decloud register        # wallet-sign and commit
    4. sudo decloud evaluate        # re-benchmark if needed
    5. sudo decloud login           # resume when ready
```

### `decloud register`

Establishes node identity with the orchestrator. This is the only
command that requires the wallet. Registration is lightweight — it
does not benchmark the node or compute capacity.

| Aspect | Detail |
| --- | --- |
| **Inputs** | None (reads `/etc/decloud/settings`) |
| **State change** | `CONFIGURED` → `ENROLLED` (first registration) or `ENROLLED` → `ENROLLED'` (re-registration after settings change) |
| **Side effects** | Wallet-signing flow (QR + sign.html); writes `/etc/decloud/credentials`; stamps `/etc/decloud/settings.locality` block with signature |
| **Network** | `POST /api/nodes/register` |
| **Authorization** | Wallet signature over canonical locality message; 5-minute validity window with ±2-minute orchestrator skew tolerance |
| **VM compliance** | Re-registration path: orchestrator walks running VMs; flags non-compliant ones for migration scheduler |

The orchestrator's response includes:
- Node ID (derived from `SHA256(machineId + walletAddress)`)
- JWT (used by all subsequent authenticated calls)
- Heartbeat interval
- List of any non-compliant VMs (re-registration only)

What registration does **not** include:
- Performance evaluation (moved to `decloud evaluate`)
- Obligation assignment (moved to `decloud evaluate`)
- Capacity computation (moved to `decloud login`)
- Scheduling config (returned at evaluate time)

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

After first registration, the node enters ENROLLED state:
heartbeating, visible to the orchestrator, but not yet schedulable.
The operator runs `evaluate` next to benchmark the node.

After re-registration (settings change), the node returns to ENROLLED
(paused). The operator can inspect the VM compliance results, optionally
re-evaluate, and `login` when satisfied.

### `decloud evaluate`

Triggers orchestrator-driven performance evaluation and obligation
assignment. The orchestrator benchmarks the node, computes its
performance tier, and determines which system VM roles it must host.

| Aspect | Detail |
| --- | --- |
| **Inputs** | None |
| **State change** | None (remains ENROLLED); enriches node record with evaluation data |
| **Precondition** | Node must be registered (JWT exists) |
| **Side effects** | Orchestrator runs benchmark command on node; computes `PerformanceEvaluation`; assigns obligations (DHT, Relay, BlockStore based on eligibility); generates obligation identity states and system VM templates; agent caches evaluation locally |
| **Network** | `POST /api/nodes/{id}/evaluate` (JWT-authenticated); orchestrator sends benchmark command to node, receives raw results |
| **Authorization** | Existing JWT — no wallet involvement |
| **Failure mode** | If benchmark fails, agent retries; orchestrator returns error with guidance |

The orchestrator is authoritative for evaluation. The node does not
self-evaluate and send a claimed score. The orchestrator sends the
benchmark command, receives raw output, and computes the score. This
prevents incentive misalignment — a node cannot lie about performance
to earn more compute points.

The evaluation response includes:
- Performance evaluation (benchmark score, points per core, performance multiplier, eligible tiers)
- Obligations (system VM roles assigned based on node capabilities)
- Obligation identity states (cryptographic identity for each system VM role)
- System VM template payloads (for roles that need deployment)
- Scheduling config (tier definitions, overcommit ratios, baseline benchmark)
- DHT bootstrap peers

The operator sees:

```
✓ Performance evaluation complete

  Benchmark Score:  3200 (capped: 3200)
  Points per Core:  3.20
  Total Points:     25.60 (8 cores × 3.20)
  Highest Tier:     Standard
  Eligible Tiers:   Standard, Balanced, Burstable

  Obligations assigned:
    DHT        (1 vCPU, 512 MB RAM)
    BlockStore (1 vCPU, 512 MB RAM, 25 GB storage)

  Run 'decloud allocate --show' to review resource allocation.
```

### Why evaluation precedes allocation

The operator needs to know their node's actual capacity before deciding
how much of it to offer. Without evaluation, the operator is guessing
("I think I have about 20 compute points"). After evaluation, the
operator knows ("I have 25.60 compute points, I'm eligible for
Standard tier, and system VMs will consume 2 vCPUs and 1 GB RAM").

Obligations also affect allocation decisions. A node obligated to run
DHT and BlockStore has committed resources before the operator even
sets their allocation. Seeing obligations before allocating lets the
operator account for system VM overhead.

### Why the orchestrator is authoritative for evaluation

Nodes are economically incentivized — they earn money based on compute
points offered. A node that self-evaluates could report inflated
benchmark scores to get more points and charge more.

The orchestrator controls the benchmark (which command, which duration,
which scoring formula) and validates the raw output. The node executes
the benchmark but does not interpret the results. Defense in depth:
the orchestrator can re-evaluate any node at any time via the
heartbeat command channel.

### `decloud allocate`

Sets resource allocation percentages. Determines how much of the
node's detected capacity the operator offers to the platform.

| Aspect | Detail |
| --- | --- |
| **Inputs** | `--cpu-percent <1-95>`, `--memory-percent <1-95>`, `--storage-percent <1-95>`, `--gpu-count <N>`, `--show` |
| **State change** | None (remains ENROLLED); updates allocation in settings and pushes to orchestrator |
| **Precondition** | Node must be registered (JWT exists); evaluation should be completed (warning if not) |
| **Side effects** | Stores percentages in `/etc/decloud/settings.json` under `resources`; pushes to orchestrator via `POST /api/nodes/{id}/allocate` |
| **Network** | `POST /api/nodes/{id}/allocate` (JWT-authenticated) |
| **Authorization** | Existing JWT — no wallet involvement |
| **Defaults** | 90% for CPU, memory, and storage; all detected GPUs |

Resource allocation uses **percentages only** for continuous resources
(CPU, RAM, storage). GPU count is an absolute integer (discrete
resource where percentages don't apply). Percentages survive
re-evaluation — if the benchmark produces different compute points,
the percentage maps correctly to the new total.

If `decloud allocate` is never run, platform defaults (90% for all
continuous resources, all GPUs) are applied at login time. The command
is optional for operators who accept the defaults.

**The `--show` flag** displays current allocation with evaluation context:

```
✓ Resource Allocation

  Compute Points:  25.60 total → allocating 90% = 23.04 points
  Memory:          32 GB total → allocating 80% = 25.6 GB
  Storage:         500 GB total → allocating 85% = 425 GB
  GPUs:            2 detected → allocating 2

  System VM overhead (obligations):
    DHT:           1 vCPU, 512 MB RAM
    BlockStore:    1 vCPU, 512 MB RAM, 25 GB storage

  Available for tenant VMs (after obligations):
    ~21 compute points, ~24.6 GB RAM, ~400 GB storage, 2 GPUs
```

### Why allocation is a percentage, not absolute points

On first registration, compute points don't exist yet — the node
hasn't been benchmarked. Even after evaluation, absolute point
allocation breaks on re-evaluation: if a node previously earned 12
points and the operator allocated 10, but a re-evaluation produces
only 8 points, the absolute allocation of 10 exceeds what exists.

Percentages are stable across re-evaluations. "Allocate 80% of
whatever I have" is always valid regardless of what the benchmark
returns. The orchestrator translates percentages to concrete values
at login time, using the current evaluation.

### Why allocation is JWT-authenticated, not wallet-signed

Resource allocation is an operational decision: "how much of my
hardware do I want to offer today." It has no jurisdictional or trust
meaning — the platform doesn't make placement guarantees based on
allocation percentages the way it does for country/region. The JWT
is the right credential for operational actions; the wallet is reserved
for trust-bearing assertions (locality, identity).

### `decloud login`

Signals operational readiness. The orchestrator computes concrete
capacity from the node's allocation percentages and current
performance evaluation, then begins considering this node for VM
placement.

| Aspect | Detail |
| --- | --- |
| **Inputs** | None (reads `/etc/decloud/credentials`) |
| **State change** | `ENROLLED` → `ACTIVE` |
| **Precondition** | Evaluation must be completed (`PerformanceEvaluation` exists on node record). If not, login fails with: "Node has not been evaluated. Run 'decloud evaluate' first." |
| **Side effects** | Orchestrator computes `TotalResources` from allocation percentages × evaluation; validates no over-allocation against `UsedResources`; sets scheduling-ready flag; clears `/etc/decloud/logged-out` sentinel if present |
| **Network** | `POST /api/nodes/{id}/login` (JWT-authenticated) |
| **Authorization** | Existing JWT — no wallet involvement |
| **Failure mode** | If credentials are missing or invalid, login fails with a message pointing at `register`. If evaluation is missing, login fails pointing at `evaluate`. |

Login is the moment concrete capacity is materialized. Before login,
the orchestrator knows the node has "90% of its compute points
allocated." At login, the orchestrator computes "90% of 25.60 = 23.04
points" and stores it as `TotalResources.ComputePoints`. This is when
the scheduler can place VMs.

The orchestrator validates that the node is enrolled and that the JWT
is valid. If the node's heartbeat-carried settings hash doesn't match
the orchestrator's stored state, login is refused with a structured
"settings drift detected" error pointing at the drifted field.

### `decloud logout`

Pauses scheduling. The node continues heartbeating and hosting its
existing VMs — it simply stops receiving new ones.

| Aspect | Detail |
| --- | --- |
| **Inputs** | None |
| **State change** | `ACTIVE` → `ENROLLED` (or no-op if already ENROLLED) |
| **Side effects** | Clears scheduling-ready flag at orchestrator; writes `/etc/decloud/logged-out` sentinel to prevent auto-login on restart |
| **Network** | `POST /api/nodes/{id}/logout` (JWT-authenticated) |
| **Credentials** | Preserved — credentials are NOT removed |

The node remains enrolled, heartbeating, and running its existing
VMs. Only new scheduling is paused. The operator can:

- Change settings (`configure`) and re-register
- Re-evaluate performance (`evaluate`)
- Adjust allocation (`allocate`)
- Drain existing workloads (`vm drain`)
- Perform maintenance
- Resume at any time (`login`)

If the operator skips logout and just stops the service (or the node
crashes), the orchestrator's heartbeat-timeout safety net detects
silence within ~5 minutes and marks the node Offline. The two
mechanisms serve different purposes: logout is an intentional
scheduling pause (node is healthy, heartbeating); timeout is a
failure detection (node is unresponsive). They coexist without
overlap.

### `decloud vm drain` / `decloud vm migrate`

Workload movement primitives. **Deferred to Phase 2.5b** — stub
implementations in v1 print "not yet implemented; manually migrate or
destroy VMs before uninstall."

Drain composes on logout. Conceptually:

```
drain = logout + request migration of existing VMs
```

When implemented, these will be:

**`decloud vm drain`** — Logout (if not already), then mark existing
VMs for migration. Orchestrator's migration scheduler moves them to
other eligible nodes. The operator can monitor progress and `login`
when the node is empty, or proceed to `uninstall`.

**`decloud vm drain --to <node-id>`** — Same as above, but prefer
migration to a specific operator-controlled target node. Wallet
equality between source and target is the security boundary that
distinguishes "self-pool migration" from "transfer to a different
operator." Standard scheduling filters still apply on the target
node; if it's ineligible for a given VM, that VM falls back to
general scheduling.

**`decloud vm migrate <vm-id> --to <node-id>`** — Migrate one specific
VM to a chosen target. Same wallet-equality rule. Does not imply
logout — this is a single-VM operation, not a node-wide pause.

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
| **Refused if** | Running tenant VMs detected and `--force` not provided. System VMs are destroyed unconditionally. |
| **JWT revocation** | Orchestrator-side revocation list ensures stale credentials cannot authenticate even if re-presented later |

Uninstall is the only path that produces the GONE state. There is no
separate `deregister` command — deregistration is a side effect of
uninstall, never an independent operation. (Earlier design rounds
explored a separate `decloud deregister`; we landed on uninstall-as-
the-only-path because the cases for needing deregister independently
of uninstall did not survive scrutiny.)

The recommended path for a clean departure:

```bash
sudo decloud vm drain        # migrate workloads off (when implemented)
# ... wait for migrations ...
sudo decloud uninstall       # clean teardown, no tenant VMs to destroy
```

### Command summary

| Command | Auth | Boundary | When to use |
| --- | --- | --- | --- |
| `install` | None | Binary placement | First-time setup |
| `configure` | None (local) / JWT (cosmetic push) | Settings | Locality, name, rates |
| `register` | Wallet signature | Identity | First enrollment or settings re-commit |
| `evaluate` | JWT | Performance | After register, or after hardware change |
| `allocate` | JWT | Capacity | After evaluate, or to adjust offering |
| `login` | JWT | Scheduling | Go live |
| `logout` | JWT | Scheduling | Pause |
| `vm drain/migrate` | JWT | Workload | Pre-uninstall or maintenance |
| `uninstall` | JWT (best-effort) | Teardown | Leave the platform |

### First-time operator flow

```bash
# 1. Install agent
curl -fsSL https://github.com/bekirmfr/DeCloud.NodeAgent/releases/latest/download/install.sh \
  | sudo bash -s -- --orchestrator https://decloud.stackfi.tech --wallet 0xYourWallet

# 2. Configure locality and profile
sudo decloud configure --country DE --region eu-central --name MyNode

# 3. Register with orchestrator (wallet signature)
sudo decloud register

# 4. Evaluate performance (orchestrator benchmarks the node)
sudo decloud evaluate

# 5. Allocate resources (optional — defaults to 90% for all)
sudo decloud allocate --cpu-percent 80 --memory-percent 85 --storage-percent 90

# 6. Go live
sudo decloud login
```

---

## 3. Resource evaluation and allocation

### The chicken-and-egg problem

Resource allocation requires knowing what resources exist. Compute
points — the unit of CPU capacity — are calculated by the orchestrator
based on a hardware benchmark. Until evaluation runs, no compute
points exist. Asking the operator to allocate compute points before
evaluation is asking them to allocate a quantity they cannot know.

Even on re-registration, absolute allocations are fragile. A node
that previously earned 12 compute points and allocated 10 might
re-evaluate at only 8 points — making the absolute allocation of 10
invalid. The system would either silently cap it (hiding the
discrepancy) or reject it (blocking re-registration).

### The solution: percentages and sequencing

Two design decisions eliminate the chicken-and-egg problem:

1. **Percentages, not absolutes.** CPU, memory, and storage allocation
   are expressed as percentages of detected capacity. "Allocate 80%
   of my compute points" is always valid, regardless of what the
   benchmark produces. The orchestrator resolves percentages to
   concrete values at login time.

2. **Evaluate before allocate.** The operator sees their node's actual
   performance before deciding how much to offer. Obligations (system
   VM commitments) are also visible at this point, so the operator
   can account for platform overhead.

### Resource types and allocation semantics

| Resource | Allocation unit | Default | Notes |
| --- | --- | --- | --- |
| CPU | Percentage of hardware-max compute points | 90% | Overcommit applied per-tier by scheduler |
| Memory | Percentage of physical RAM | 90% | No overcommit — memory is physical only |
| Storage | Percentage of physical storage (pre-overcommit) | 90% | Tier-specific storage overcommit applied on top |
| GPU | Absolute count | All detected | Discrete resource; 0 = operator has GPUs but does not offer them |

### When capacity is materialized

Capacity lives as percentages until login. At login, the orchestrator
resolves:

```
TotalResources.ComputePoints = hardwareMaxPoints × cpuPercent
TotalResources.MemoryBytes   = physicalRam × memoryPercent
TotalResources.StorageBytes  = physicalStorage × storagePercent × tierOvercommit
```

This is the only moment percentages become concrete values. If the
operator logs out, re-evaluates (different benchmark score), and logs
back in, the percentages produce different concrete values
automatically.

### Allocation push endpoint

`POST /api/nodes/{id}/allocate` — JWT-authenticated. Accepts:

```json
{
  "cpuPercent": 0.80,
  "memoryPercent": 0.85,
  "storagePercent": 0.90,
  "gpuCount": 2
}
```

The orchestrator validates:
- Percentages in range 0.01–0.95
- GPU count ≤ detected GPUs (or null for all)
- If node has running VMs: allocation not below currently used resources

This endpoint is separate from registration because allocation is an
operational decision (JWT), not a trust-bearing jurisdictional claim
(wallet).

---

## 4. Settings model

`/etc/decloud/settings.json` is a JSON file (mode 600) holding the node's
operational configuration.

```json
{
  "version": 2,
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
  },
  "resources": {
    "cpu": { "mode": "percent", "value": 80 },
    "memory": { "mode": "percent", "value": 85 },
    "storage": { "mode": "percent", "value": 90 },
    "gpu": { "count": 2 }
  }
}
```

### Field categories and mutability

| Field | Category | When changeable | Authorization | Command |
| --- | --- | --- | --- | --- |
| `wallet` | identity | Never (re-install required) | Cannot change | `install` |
| `orchestrator_url` | identity | Never | Cannot change | `install` |
| `locality.country` | locality (signed) | Logout → configure → register | Wallet signature | `configure` |
| `locality.region` | locality (signed) | Logout → configure → register | Wallet signature | `configure` |
| `locality.zone` | locality (cosmetic) | Logout → configure → register | None (not signed) | `configure` |
| `profile.name` | cosmetic | Any time (no logout/re-register) | JWT only | `configure` |
| `profile.description` | cosmetic | Any time (no logout/re-register) | JWT only | `configure` |
| `rates.*` | service | Logout → configure → register | Wallet signature | `configure` |
| `resources.cpu` | allocation | Any time while enrolled | JWT only | `allocate` |
| `resources.memory` | allocation | Any time while enrolled | JWT only | `allocate` |
| `resources.storage` | allocation | Any time while enrolled | JWT only | `allocate` |
| `resources.gpu` | allocation | Any time while enrolled | JWT only | `allocate` |

Cosmetic updates flow through `decloud configure --name "X"` and use
the orchestrator's profile endpoint directly — no logout or re-register
needed.

Locality and rate updates require the full sequence: logout (stop
scheduling) → configure (change settings) → register (wallet-sign and
commit, VM compliance check) → evaluate (if needed) → login (resume).

Resource allocation updates flow through `decloud allocate` and are
pushed to the orchestrator via the allocate endpoint. No logout
required — the orchestrator applies updated percentages on next login.
If the node is active, the operator must logout first, allocate, then
login to apply the new values (the orchestrator recomputes capacity
at login).

### Settings hash

A SHA-256 hash of the settings file's canonical JSON representation
(sorted keys, no whitespace) is computed by the node agent and
included in every heartbeat payload. The orchestrator compares this
hash against its stored state to detect drift continuously (see
[§6 Continuous drift detection](#6-continuous-drift-detection)).

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

## 5. Cryptographic invariants

The system holds these invariants by construction:

**(a) Server-authoritative settings.** What the orchestrator believes
about a node's locality is what was registered. Local edits to
`/etc/decloud/settings` without re-register cannot change orchestrator-
side state.

**(b) Continuous drift detection.** Every heartbeat carries a hash of
the node's local settings. The orchestrator compares this against
its stored state and detects drift within one heartbeat interval
(~30 seconds). This is strictly stronger than a one-time check at
login — it catches tampering continuously, not only at a ceremony.
See [§6](#6-continuous-drift-detection).

**(c) Wallet ownership required for jurisdictional changes.** A leaked
JWT cannot change locality. Country/region updates require fresh wallet
signature; the JWT alone is insufficient. The wallet signs at
`register`, where jurisdictional claims have trust meaning. Routine
operations (heartbeat, login, logout, allocate, profile updates) use
the JWT.

**(d) Revoked credentials stay revoked.** JWT revocation list survives
orchestrator restarts (MongoDB-backed). A node uninstalled today cannot
have its JWT replayed tomorrow. The check is on every authenticated
request, not just heartbeat.

**(e) Signature freshness bounded.** Wallet signatures at registration
expire 5 minutes after the embedded timestamp, with ±2 minutes
orchestrator skew tolerance. Captured registration signatures cannot
be replayed days later. The timestamp is part of the signed payload,
so an attacker cannot adjust it without invalidating the signature.

**(f) Orchestrator-authoritative evaluation.** A node cannot
self-declare its performance. The orchestrator drives the benchmark,
receives raw output, and computes the score. The node's economic
incentive (more points = more earnings) cannot influence the
evaluation because the node does not control the scoring.

These six together mean: if a node is `ACTIVE` and heartbeating, the
orchestrator has cryptographic proof that (a) the locality was
wallet-authorized at registration time, (b) the node's local settings
currently match what was registered, (c) the node's performance has
been independently evaluated, and (d) the node's operator has signaled
readiness for scheduling.

---

## 6. Continuous drift detection

Drift detection uses the heartbeat — the system's natural liveness
boundary — rather than a separate ceremony.

### How it works

1. The node agent computes a SHA-256 hash of the canonical JSON
   representation of `/etc/decloud/settings` (sorted keys, no
   whitespace, excluding transient fields like `signature` and
   `signed_at`).

2. Every heartbeat payload includes this hash in the
   `settingsHash` field.

3. The orchestrator computes the expected hash from its stored
   registration data and compares.

4. On mismatch:
   - The orchestrator logs a warning with the node ID
   - The node's scheduling eligibility is suspended (even if ACTIVE)
   - The heartbeat response includes a structured `settingsDrift`
     error identifying the drifted field(s)
   - The node agent logs the drift warning locally

5. To resolve: the operator must re-register (which re-commits
   settings with wallet authority) or revert the local edit.

### What this replaces

The earlier design used a wallet-signed locality message at login time
as the drift detection mechanism. This had two problems: it only
checked at login (not continuously), and it required the wallet for
an operational action (login), which prevented unattended restart.

Heartbeat-carried hash detection is strictly stronger: it checks every
~30 seconds instead of once, and it doesn't require the wallet at
login time. The wallet speaks at registration, where it has something
to attest. The heartbeat carries operational state, where drift
detection belongs.

### What drift detection does NOT do

Drift detection is an integrity check, not a security boundary. A
compromised node agent could lie about its settings hash. The security
boundary is the wallet signature at registration — the orchestrator
will not accept new locality claims without a fresh wallet signature.
Drift detection catches *accidental* divergence (operator edited the
file and forgot to re-register) and *naive* tampering (someone edited
the file without understanding the system). Sophisticated attacks
against a compromised agent require different mitigations (remote
attestation, hardware roots of trust) that are out of scope.

---

## 7. VM compliance handling

When `decloud register` runs on an already-enrolled node and the
locality has changed, the orchestrator:

1. Validates the new locality (country in `countries.json`, region in
   `regions.json`)
2. Verifies wallet signature over canonical message
3. Walks all running tenant VMs on this node. For each VM with
   `spec.Constraints`, evaluates every constraint against the node's
   new locality using the same `IConstraintEvaluator` that FILTER 10
   uses at scheduling time. This covers jurisdiction tags, country
   requirements, region filtering, and any other constraint the tenant
   specified — one evaluator, one answer.
4. For each non-compliant VM:
   - Sets `VirtualMachine.NonCompliantSince` and `NonComplianceReason`
   - Migration scheduler picks them up on next cycle and moves them
     to compliant nodes
5. Returns the registration response with the list of flagged VMs

VMs without constraints are always compliant — they placed no
locality demands at creation time. System VMs (Relay, DHT,
BlockStore) are never compliance-checked — they are
orchestrator-controlled infrastructure, not tenant workloads.

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

Because the operator must `logout` before changing locality, no new
VMs are being scheduled onto the node during this window. Existing
VMs continue running (heartbeat is active) while the migration
scheduler moves non-compliant ones. The operator can monitor progress
and `login` when the node is clean — or login immediately and let
migrations complete asynchronously.

The flagging mechanism uses two fields on `VirtualMachine`:

```csharp
public DateTime? NonCompliantSince { get; set; }
public string? NonComplianceReason { get; set; }
```

The migration scheduler treats `NonCompliantSince != null` as a
trigger to schedule replacement, the same way it treats `VmStatus.Error`
as a trigger today.

---

## 8. Trust model

Two credentials, two purposes:

| Credential | Source | Lifetime | Used for |
| --- | --- | --- | --- |
| **Wallet** | Operator's external wallet (MetaMask, hardware wallet, etc.) | Forever (operator-controlled) | Registration; locality changes; rate changes |
| **JWT** | Issued by orchestrator at register | Hours, refreshable | Heartbeat, login, logout, evaluate, allocate, profile updates, all routine API calls |

The wallet is the trust root. JWTs are convenience tokens issued after
wallet authentication.

This two-credential split keeps the wallet involved only where it has
genuine trust meaning — asserting jurisdictional claims and rate
commitments. All routine operations run on the JWT. The wallet never
needs to be present for day-to-day node operation, which is why
unattended restart works.

### Why not three credentials

An earlier design revision included a third credential: a short-lived
"locality signature" generated at login time, where the wallet re-signed
the current settings as proof of ongoing endorsement. This was removed
because:

1. It required the wallet at login, preventing unattended restart
2. Drift detection is better served continuously via heartbeat-carried
   settings hash than by a one-time check at a ceremony
3. The wallet's trust contribution happens at registration, where the
   jurisdictional claim is made — re-attesting the same claim at login
   adds ceremony without adding security

The wallet signs once at register. The heartbeat validates continuously.
Each mechanism does what it's good at.

---

# Part 2 — Implementation Status

## 9. What's implemented

The lifecycle described in Part 1 is fully implemented.
Implementation status by tier:

### Tier 1 — Foundation (Orchestrator + CLI)

- `Node.SchedulingReady` boolean field (default `false`; set to
  `true` by `LoginNodeAsync` after evaluation and capacity computation)
- `POST /api/nodes/{id}/login` and `POST /api/nodes/{id}/logout`
  endpoints on `NodesController`
- `LoginNodeAsync` / `LogoutNodeAsync` in `NodeService`
- FILTER 1.5 in `VmSchedulingService.ApplyHardFiltersAsync` —
  rejects nodes with `!SchedulingReady` ("Node not scheduling-ready
  (operator logged out)")
- Canonical signing message in `cli-decloud-node` with country,
  region, ISO 8601 timestamp
- `ValidateSignatureTimestamp` in `RegisterNodeAsync` — 5-minute
  window, ±2-minute skew, legacy format passthrough

### Tier 2 — Core lifecycle (CLI + Agent)

**Bash CLI (`cli/decloud`):**
- `cmd_register` — wallet ceremony, delegates to `cli-decloud-node
  register`, waits for `AuthenticationManager` to complete
- `cmd_login` — lightweight `curl` to orchestrator login endpoint,
  removes `/etc/decloud/logged-out` sentinel
- `cmd_logout` — non-destructive, calls logout endpoint, writes
  sentinel, credentials preserved
- `cmd_configure` — reads/writes `settings.json` via
  `_read_setting` / `_write_setting` helpers; refuses locality
  changes while logged in (credentials exist AND no sentinel);
  cosmetic changes allowed anytime and pushed to orchestrator
  via `PATCH /api/nodes/me/profile` if enrolled
- `cmd_evaluate` — primary lifecycle step; calls
  `POST /api/orchestrator/evaluate` which triggers orchestrator-driven
  benchmark, obligation assignment, and system template delivery.
  Displays evaluation results and obligation list.
- `cmd_allocate` — sets resource allocation percentages
  (`--cpu-percent`, `--memory-percent`, `--storage-percent`,
  `--gpu-count`); writes to `settings.json`, pushes to orchestrator
  via `POST /api/orchestrator/allocate`. `--show` displays current
  allocation with resolved capacity.
- `cmd_vm_drain` / `cmd_vm_migrate` — stubs printing "not yet
  implemented" with manual alternative guidance
- `get_api_key` / `get_orchestrator_url` utility helpers
- Help text: NODE LIFECYCLE section with register, evaluate, allocate,
  login, logout, configure

**Python CLI (`cli/cli-decloud-node`):**
- Single-purpose wallet ceremony tool: `register` + `version` only
- `cmd_status`, `cmd_logout`, and their parsers removed (dead paths)
- `run_wallet_signing` (renamed from `login_with_web_signing`)

**Node agent:**
- `AutoLoginIfNotLoggedOutAsync` in `AuthenticationManager` — checks
  sentinel, auto-evaluates if `PerformanceEvaluation` is null, then
  calls `LoginAsync` if evaluation succeeds. Non-fatal on failure.
- `EvaluateNodeAsync` on `OrchestratorClient` — posts hardware
  inventory to `/api/nodes/me/evaluate`, deserializes
  `EvaluateNodeResponse`, persists evaluation, scheduling config,
  obligation states, system templates, and obligation descriptors.
- `AllocateAsync` on `OrchestratorClient` — posts percentages to
  `/api/nodes/{id}/allocate`, returns resolved capacity.
- `LoginAsync` / `LogoutAsync` on `OrchestratorClient`
- `HeartbeatService` waits for full initialization
  (`WaitForInitializationAsync`) before starting heartbeat loop

### Tier 3 — Security & integrity (Shared + Agent + Orchestrator)

**Settings hash drift detection:**
- `SettingsHash.Compute()` in `DeCloud.Shared` — SHA-256 over
  canonical concatenation of wallet, country, region (zone excluded,
  consistent with non-signed status)
- `Heartbeat.SettingsHash` field computed in `HeartbeatService`,
  included in wire payload via `BuildHeartbeatPayload`
- `Node.RegisteredSettingsHash` computed in `RegisterNodeAsync`
- `ProcessHeartbeatAsync` compares hashes, returns
  `SettingsDriftInfo` on mismatch; both-null skip for backward compat
- Agent-side `ProcessHeartbeatResponseAsync` reads `settingsDrift`
  from heartbeat response and logs warning

**JWT revocation:**
- `JwtRevocationService` — MongoDB `revoked_jwts` collection +
  `ConcurrentDictionary` in-memory cache; O(1) hot-path check
- `IJwtRevocationService` interface
- `OnTokenValidated` hook in JWT bearer pipeline checks revocation
- `Node.CurrentJti` stores latest JTI; `jti` generated before JWT
  creation in `RegisterNodeAsync` and passed to
  `GenerateNodeJwtToken`
- Cache loaded from MongoDB on startup via `LoadFromStoreAsync`

**Deregister safety guard:**
- `POST /api/nodes/{nodeId}/deregister` endpoint
  (`NodesController`) — called by `uninstall.sh`
- Refuses with 409 Conflict if tenant VMs running and `!force`;
  system VMs not counted
- Revokes JWT on success via `JwtRevocationService`
- Removes node record from DataStore

### Tier 4 — VM compliance (Orchestrator)

- `VirtualMachine.NonCompliantSince` (DateTime?) and
  `NonComplianceReason` (string?) fields
- `FlagNonCompliantVmsAsync` in `NodeService` — walks running tenant
  VMs, evaluates `spec.Constraints` against new node locality using
  `IConstraintEvaluator` (same evaluator as FILTER 10; no parallel
  bespoke field checks)
- Called in `RegisterNodeAsync` after node save, only on
  re-registration with locality change
- `NodeRegistrationResponse` extended with `NonCompliantVms`
  (List\<NonCompliantVmInfo\>?)
- Migration scheduler extended: `ScanMigratingVmsAsync` scans for
  `NonCompliantSince != null + Running + General`, transitions to
  Error for existing migration pipeline
- Compliance flag cleared in `ProcessCommandAcknowledgmentAsync`
  after successful migration ack
- `TransitionContext.Compliance` and `TransitionTrigger.Compliance`
  in `VmLifecycleManager`

### Tier 5 — Evaluate/allocate lifecycle separation

**Shared library (`DeCloud.Shared`):**
- `AllocatedResources` model with v2 percentage fields (`CpuPercent`,
  `MemoryPercent`, `StoragePercent`), `SchemaVersion` discriminator,
  `ToPercentFormat()` migration helper, `Validate()`, effective
  percentage properties
- `NodeAllocateRequest` / `NodeAllocateResponse` wire DTOs
- `EvaluateNodeResponse` (orchestrator-side) / `EvaluateNodeResponse`
  (agent-side mirror) carrying performance evaluation, scheduling
  config, obligations, identity states, system templates

**Orchestrator:**
- `RegisterNodeAsync` slimmed to identity-only: wallet auth, node
  creation, JWT, settings hash, VM compliance. No evaluation, no
  obligations, no capacity computation. `SchedulingReady = false`.
- `POST /api/nodes/me/evaluate` expanded: benchmark → obligation
  seeding → CGNAT relay assignment → obligation states + system
  templates → returns `EvaluateNodeResponse`
- `GenerateObligationPayloadsAsync` on `NodeService` — reusable
  wrapper calling `GenerateAndAttachObligationStates` and
  `GenerateSystemTemplatePayloads` with empty version dictionaries
- `LoginNodeAsync` computes `TotalResources` from allocation
  percentages via `NodeCapacityCalculator` at login time. Validates
  used ≤ allocated. Precondition: evaluation must exist.
- `POST /api/nodes/{id}/allocate` endpoint on `NodesController` —
  JWT-authenticated, validates percentages, merges with existing
  allocation, returns resolved capacity
- `NodeCapacityCalculator` reads v2 percentages via
  `ResolveComputePoints` / `ResolveMemoryBytes` /
  `ResolveStorageBytes` helpers. Registered as DI singleton.

**Node agent:**
- `NodeRegistrationResponse` slimmed: NodeId, ApiKey,
  WireGuardPublicKey, HeartbeatInterval, NonCompliantVms
- `EvaluateNodeAsync` on `OrchestratorClient`: full lifecycle step
  that discovers hardware, posts to evaluate endpoint, persists all
  returned data (evaluation, config, states, templates, obligations)
- `BuildAllocatedResources` sends v2 format with percentages
- `NodeMetadataService` exposes `AllocatedMemoryPercent` and
  `AllocatedStoragePercent` for lossless percent transmission
- `cmd_allocate` CLI command with `--cpu-percent`, `--memory-percent`,
  `--storage-percent`, `--gpu-count`, `--show`
- `cmd_configure` no longer handles resource allocation
- `cmd_evaluate` uses `api_post`, displays obligations from response

## 10. Remaining work

### Live migration (Phase 2.5b)

The `vm drain` and `vm migrate` stubs exist in the CLI. The
compliance migration currently transitions VMs to Error (stops
them) before migrating. True live migration (start on target, then
stop on source) requires coordination the current pipeline does
not have.

### Reverse compliance check

If the operator changes locality back to a compliant state before
migration completes, flagged VMs still migrate unnecessarily. A
future enhancement could re-check flagged VMs on each heartbeat
cycle and clear the flag if the node is now compliant. Edge case,
low priority — the operator can avoid it by not logging in until
committed to the new locality.

---

# Part 3 — Infrastructure Mechanics

> **Scope:** This part describes how `install.sh`, `decloud update`,
> and `decloud uninstall` work on disk. These mechanics are concerned
> with binary placement, signature verification, version management,
> and recovery — not with operator command semantics (which are
> covered in Part 1).

## 11. Overview of operations

Three lifecycle operations, each with one canonical entry point:

| Operation | Entry point | Reads from | Writes to | Verifies |
| --- | --- | --- | --- | --- |
| Install | `curl ... \| sudo bash` | GitHub Release | `/opt/decloud`, `/usr/local/bin`, `/etc/decloud` | Cosign + SHA-256 |
| Update | `sudo decloud update` | GitHub Release (or local fallback) | Same as install | Cosign + SHA-256 |
| Uninstall | `sudo decloud uninstall` | GitHub Release (or local fallback) | Removes most of the above | None (cleanup only) |

Update is "install with saved parameters" — same script, same code path,
just non-interactive. There is one installation procedure.

## 12. Fresh install

The single command an operator pastes:

```bash
curl -fsSL https://github.com/bekirmfr/DeCloud.NodeAgent/releases/latest/download/install.sh \
  | sudo bash -s -- \
      --orchestrator https://decloud.stackfi.tech \
      --wallet 0xYourWalletAddress
```

After install completes, configure, register, evaluate, and go live:

```bash
sudo decloud configure --country TR --region eu-east --name MyNode
sudo decloud register
sudo decloud evaluate
sudo decloud allocate --memory-percent 80    # optional, defaults to 90%
sudo decloud login
```

Wall-clock time on a fresh Ubuntu box: 3–5 minutes for install. The
full configure → register → evaluate → login sequence adds ~2 minutes
(evaluation benchmark takes ~30 seconds).

### About the install URL

`/releases/latest/download/install.sh` is a GitHub-native redirect pattern.
GitHub resolves `latest` to the most recent release that is published as
a **stable (non-pre-release)** release, then issues a transparent redirect
to the actual download URL for the asset named `install.sh` in that release.
`curl -fsSL` follows the redirect automatically — the operator never sees
the version number.

**Pre-releases are excluded from `latest`.** A release tagged `vX.Y.Z-rc1`
and marked as a pre-release on GitHub is not returned by this URL. If the
current published release is a release candidate, the `latest` URL either
resolves to the previous stable release (if one exists) or to nothing.

To install a specific version — including a release candidate — use the
version-tagged URL directly:

```bash
# Install a specific version (e.g. during rc testing)
curl -fsSL https://github.com/bekirmfr/DeCloud.NodeAgent/releases/download/v2.2.0-rc1/install.sh \
  | sudo bash -s -- \
      --orchestrator https://decloud.stackfi.tech \
      --wallet 0xYourWalletAddress
```

**The version-agnostic URL is the stable-release install path.** Once an
rc is promoted to stable (uncheck "This is a pre-release" in the GitHub
release editor), the `latest` URL immediately resolves to it — no code
change, no new release, no workflow run required.

**Two-layer version resolution.** Once `install.sh` is running, it uses
the GitHub Releases API (`RELEASE_API_LATEST`) to independently resolve
which version of the agent binaries to download. The URL you curl
determines which `install.sh` you get; the `install.sh` itself determines
which binaries it installs (always the latest stable unless overridden).
Under normal circumstances both layers point to the same version.

### What runs, in order

1. **Bootstrap fetch** (`curl`, ~1 s)
   - TLS to `github.com`, redirect to `objects.githubusercontent.com`
   - `install.sh` body streamed into bash's stdin (never touches disk
     in this mode)

2. **Argument parse** (`install.sh`, ~5 s)
   - Reads `--orchestrator`, `--wallet`
   - Persists arguments to `/etc/decloud/settings.json` (used by
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
   - `appsettings.Production.json` generated from settings.json
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
   - Service enabled and started; idle until register

10. **Done**
    - Summary box, next-steps prompt for `decloud configure` and
      `decloud register`

### Interactive prompts

When `install.sh` is run from a terminal (or via `curl | bash` with
`/dev/tty` available — most SSH sessions qualify), missing required
flags trigger interactive prompts:

```
Orchestrator URL [https://decloud.stackfi.tech]: 
Wallet address (0x...): 0x...
Country code (ISO 3166-1 alpha-2): TR
```

The orchestrator default is hardcoded; the wallet is required and
validated in a loop until a valid format is provided.

When run truly non-interactively (CI, no controlling terminal),
missing required flags abort with copy-pasteable usage examples.

## 13. Update

```bash
sudo decloud update
```

### What runs, in order

1. **Read saved parameters** from `/etc/decloud/settings.json`
2. **Hybrid install.sh resolution:**
   - **Primary**: `curl` from `releases/latest/download/install.sh`
     (30 s timeout). On success, writes to `/tmp/decloud-install-XXXX.sh`.
     `latest` resolves to the most recent stable (non-pre-release) release.
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
  `/etc/decloud/settings.json`. To change operational settings, use
  `decloud configure` followed by `decloud register`.
- Does **not** roll back automatically on failure. If the new release
  fails to start, the symlink already points at the new version. See
  [Recovery scenarios](#19-recovery-scenarios).

## 14. Uninstall

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
7. Removes `/etc/decloud/` (credentials, settings.json, SSH CA)
8. Optionally removes the `decloud` system user
9. Reverts sshd config additions

The user is prompted to confirm before destructive actions unless
`--force` is passed.

## 15. On-disk layout

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
├── settings.json                                    (mode 600, used by 'decloud update' and 'decloud configure')
├── credentials                                      (after 'decloud register')
├── logged-out                                       (sentinel: operator paused scheduling)
├── pending-auth                                     (during register flow)
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

### Settings file format

`/etc/decloud/settings.json` stores all node settings in JSON format.
Identity fields (orchestrator URL, wallet) are written by `install.sh`.
Operational settings (country, region, zone, name) are managed by
`decloud configure`. Resource allocation (CPU, memory, storage, GPU)
is managed by `decloud allocate`. The file is merged on `decloud
update` — identity fields are refreshed while operational settings
and resource allocations are preserved.

## 16. Configuration

### Where settings live (current implementation)

| Setting | Source | Mutability |
| --- | --- | --- |
| Orchestrator URL, wallet | `/etc/decloud/settings.json` | Set at install; change requires reinstall |
| Country, region, zone, name | `/etc/decloud/settings.json` | Set via `decloud configure`; locality changes require logout → configure → register → login |
| Resource allocation (CPU%, memory%, storage%, GPU count) | `/etc/decloud/settings.json` | Set via `decloud allocate`; pushed to orchestrator; applied at next login |
| `appsettings.Production.json` | Generated by `install.sh` | Infrastructure only (URLs, ports, paths). Safe to regenerate — contains no operator data |
| Node ID, API key | `/etc/decloud/credentials` (after register) | Removed by `uninstall`, regenerated on next register |
| Machine ID | `/etc/machine-id` | OS-level, do not edit |
| Node identity | `SHA-256(machine-id + wallet)` | Derived; changes if either input changes |

### Changing parameters after install

Operational settings (locality, name):

```bash
sudo decloud configure --country DE --region eu-central --name MyNode
sudo decloud register       # commit to orchestrator with wallet signature
sudo decloud evaluate       # re-benchmark if needed
sudo decloud login           # resume scheduling
```

For locality changes on an enrolled node, logout first:

```bash
sudo decloud logout
sudo decloud configure --country BR --region sa-east
sudo decloud register
sudo decloud evaluate       # optional if hardware hasn't changed
sudo decloud login
```

Resource allocation changes:

```bash
sudo decloud logout                            # if currently active
sudo decloud allocate --cpu-percent 80 --memory-percent 70
sudo decloud login                             # re-materializes capacity
```

## 17. Trust chain at runtime

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

## 18. Common operator tasks

| Task | Command |
| --- | --- |
| Check service status | `sudo systemctl status decloud-node-agent` |
| Read recent logs | `sudo journalctl -u decloud-node-agent -n 100 --no-pager` |
| Tail logs live | `sudo journalctl -u decloud-node-agent -f` |
| Restart service | `sudo systemctl restart decloud-node-agent` |
| Update to latest stable | `sudo decloud update` |
| Install a specific version | `curl -fsSL https://github.com/bekirmfr/DeCloud.NodeAgent/releases/download/vX.Y.Z/install.sh \| sudo bash -s -- --orchestrator <URL> --wallet <addr>` |
| Pause scheduling | `sudo decloud logout` |
| Resume scheduling | `sudo decloud login` |
| Re-evaluate performance | `sudo decloud evaluate` |
| Adjust resource allocation | `sudo decloud allocate --cpu-percent 80 --memory-percent 70` |
| View current allocation | `sudo decloud allocate --show` |
| Change locality | `sudo decloud logout && sudo decloud configure --country BR --region sa-east && sudo decloud register && sudo decloud evaluate && sudo decloud login` |
| Inspect installed version | `cat /opt/decloud/current-version` |
| Manual rollback to previous version | `sudo ln -sfn /opt/decloud/publish.$(cat /opt/decloud/previous-version) /opt/decloud/publish && sudo systemctl restart decloud-node-agent` |
| Check installed binaries | `ls -la /opt/decloud/publish/` |
| View saved settings | `sudo cat /etc/decloud/settings.json` |

## 19. Recovery scenarios

### Service fails to start after update

```bash
journalctl -u decloud-node-agent -n 100 --no-pager
```

Look for missing dependencies, permission errors, or appsettings
validation failures. If the new version is broken, manual rollback per
[Common operator tasks](#18-common-operator-tasks) above.

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

### Install completed but `decloud register` doesn't reach the wallet prompt

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

If credentials are missing, run `sudo decloud register` to
re-authenticate with the wallet.

### Settings drift detected

If heartbeat responses contain `settingsDrift` errors, the node's
local settings don't match what the orchestrator has from registration.
Either:

1. Someone edited `/etc/decloud/settings` without re-registering —
   revert the edit or run the full logout → configure → register →
   evaluate → login sequence
2. A bug produced inconsistent state — inspect both sides and
   re-register to reconcile

### Evaluation missing after update

If a new agent version clears the cached evaluation, the node may
fail to auto-login on restart. Run `sudo decloud evaluate` followed
by `sudo decloud login` to restore.

## 20. Migration from legacy installs

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
| `/etc/decloud/settings.json` | Node settings (JSON); identity fields written by install, operational fields by configure, resource allocation by allocate |
| `/etc/decloud/credentials` | JWT credentials (after register) |
| `/etc/decloud/logged-out` | Sentinel file indicating operator-initiated scheduling pause; prevents auto-login on restart |
| `/usr/local/share/decloud/install.sh` | Offline fallback for `decloud update` |
| `/usr/local/share/decloud/uninstall.sh` | Offline fallback for `decloud uninstall` |
