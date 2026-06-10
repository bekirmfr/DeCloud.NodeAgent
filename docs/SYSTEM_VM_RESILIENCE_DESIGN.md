# System VM Design

**Status:** Implemented — All Phases Complete (P1–P11)
**Phase:** Complete
**Last Updated:** 2026-05-01
**Author:** BMA / DeCloud Architecture

> **Change note (2026-04-26):** Artifact delivery simplified to **external-only**.
> The platform no longer hosts artifact bytes; authors provide an HTTPS URL plus
> SHA256, and the node agent cache fetches from the author's origin. This removes
> the upload path and `IsExternalReference` flag, reduces attack surface, and
> shifts the platform's legal posture from *host* to *directory*. See §3.4 and §6.3.

> **Change note (2026-04-27):** Document reorganised. The previous "System VM
> Resilience" design and the "System VM Integration Plan v2" merged into this
> single canonical design. Three substantive design changes folded in:
>
> 1. **Lifecycle authority moves to the node** (§2 and §5). The orchestrator
>    declares intent, identity, and templates. The node decides when to Create
>    or Delete its own system VMs. The orchestrator-side reconciliation loop
>    (`SystemVmReconciliationService`) is removed entirely.
> 2. **Mesh health is a regular service** (§5.4). Each system VM exposes a
>    `/health/mesh` endpoint that the existing `VmReadinessMonitor` checks via
>    `HttpGet`. No special-case threshold logic, no `Degraded` state.
> 3. **Cloud-init folder vanishes** (§3.10, §9). System templates now live
>    entirely in MongoDB and are pushed to nodes via the obligation channel.
>    The node agent's bundled `CloudInit/Templates/` folder, the Go toolchain,
>    `GoBinaryBuildStartupService`, and all `Inject*` methods are deleted.
>
> The legacy "Implementation Plan" (Phases A–E) has been replaced by an
> eleven-phase plan that sequences these changes (§9).
>
> Companion documents kept for context, not for implementation guidance:
> `SYSTEM_VM_LIFECYCLE_FLOW_MAP.md` (gap analysis of the pre-refactor code),
> `SYSTEM_VM_RECONCILIATION_DESIGN.md` (the reasoning behind the matrix model).

> **Change note (2026-05-01):** All eleven implementation phases complete.
> The system VM resilience redesign is fully live in production.
> Summary of key architectural decisions made during implementation:
>
> 1. **Artifact pipeline unified** (P8). `TemplateArtifact` supports two source
>    schemes: HTTPS (external hosting, platform stores metadata only) and `data:`
>    URI inline (RFC 2397 base64, stored in template MongoDB document). Binary
>    artifacts always use HTTPS. Scripts, dashboards, and config use inline.
>    Per-artifact limit 5 MB inline; per-template 10 MB inline.
>
> 2. **Template identity gate replaced** (reviewed during P10). The original plan
>    used slug-as-gate to prevent community templates from claiming system roles.
>    This was replaced with stable MongoDB `_id` binding on `SystemVmObligation`.
>    The obligation channel carries `TemplateId` (`_id`), not slug. Slugs are
>    display-only. The gate is the obligation channel itself — no template reaches
>    a node's system VM reconciler unless the orchestrator put its `_id` there.
>
> 3. **All VMs are building blocks** (design clarification during P10). The
>    distinction between "system template" and "community template" is deployment
>    context, not template metadata. The same template deployed under the
>    obligation system is a system VM (persistent identity, self-healing). The
>    same template deployed by a platform user through the marketplace is a tenant
>    VM (ephemeral identity, user-managed). No reserved slugs, no substrate flags.
>
> 4. **Security layer 3 added to ObligationStateController** (review finding 2.1).
>    Caller-IP-to-role binding on `GET /api/obligations/{role}/state`. Prevents
>    tenant VMs on the same virbr0 bridge from exfiltrating system VM identity.
>    Three allow-with-warning cases handle first-boot timing windows.
>
> 5. **`CloudInitTemplateService` scope narrowed** (P11). System VM code paths
>    (`PopulateDhtVariablesAsync`, `PopulateRelayVariablesAsync`,
>    `PopulateBlockStoreVariablesAsync`, `InjectDhtExternalTemplatesAsync`,
>    `InjectRelayExternalTemplatesAsync`, `InjectBlockStoreExternalTemplatesAsync`,
>    `PopulateWireGuardMeshVariablesAsync`, `ResolveLocalRelayInfoAsync`) deleted.
>    Service now handles General and Inference VMs only.
>
> 6. **`SystemVmObligationService` scope narrowed** (P7). Background loop now only
>    runs `EnsureObligationsAsync` — backfilling roles on capability drift and
>    adopting existing VMs. All deploy/check/verify/retry logic deleted. Deployment
>    authority is exclusively on the node's `SystemVmReconciler`.
>
> 7. **`DeployDhtVmAsync`, `DeployBlockStoreVmAsync`, `DeployRelayVmAsync` deleted**
>    (P7). Interfaces retain only peer discovery and mesh management methods.
>    Heartbeat-path `Deploying → Active` promotion deleted from `NodeService`.
>
> 8. **`DeCloud.Builds` restructured** for the artifact pipeline:
>    `system-vms/{role}/src/` (Go source), `system-vms/{role}/assets/` (scripts,
>    dashboards), `system-vms/{role}/cloud-init.yaml`. `compute-artifact-constants.sh`
>    auto-discovers all `assets/` directories — no hardcoded maps.

---

## Table of contents

1. Problem statement
2. Solution overview
3. Unified enriched `VmTemplate`
4. ObligationStateService — identity & template persistence
5. Lifecycle authority — node-side reconciliation
6. Security
7. Relay revenue-share implementation
8. Resilience scenarios
9. Implementation plan
10. File change summary
11. Success criteria

---

## 1. Problem statement

### 1.1 Current pain points

System VMs (Relay, DHT, BlockStore) are stateful services whose identity —
WireGuard keypairs, libp2p Ed25519 peer IDs, auth tokens, subnet IPs — is
baked into cloud-init labels at deploy time and lives only in the VM's disk.

When a system VM crashes or is redeployed:

- The VM disk is deleted → Ed25519 key at `/var/lib/decloud-dht/identity.key` is gone
- `loadOrCreateIdentity()` generates a **new key** → new peer ID
- Every peer in the DHT/BlockStore network had the old peer ID → cascading reconnections
- Mesh-enrolled VMs have the old WireGuard public key baked into their configs → reconnection failures
- Result: temporary network fragmentation, cascading redeployments, operator confusion

Additionally, system VM deployment logic is scattered across three bespoke
services (`RelayNodeService`, `DhtNodeService`, `BlockStoreService`), each with
its own variable injection, identity generation, and label construction —
duplicating ~150 lines of near-identical logic per role. Lifecycle decisions
sit in a fourth orchestrator-side service (`SystemVmReconciliationService`,
~1500 lines) whose decisions are based on cached state that frequently
contradicts reality, producing flapping, stuck obligations, and the catalogue
of issues documented in the companion flow map.

### 1.2 Root causes

| Problem                              | Root cause                                                                                  |
|--------------------------------------|---------------------------------------------------------------------------------------------|
| Identity lost on redeploy            | Identity lives on VM disk only — no persistence layer                                        |
| Cascading mesh disruption            | Peer ID changes force all mesh peers to update routing tables                                |
| Code duplication                     | Per-role bespoke deployment services instead of unified template path                        |
| No community extensibility           | System VMs are internal platform constructs, not marketplace templates                       |
| Lifecycle flapping & stuck VMs       | Orchestrator reconciles from cached `obligation.Status` / `vm.UpdatedAt`, not from current truth |
| Self-healing fragility               | Orchestrator-side loop is the single point of failure for the mesh substrate                 |
| Cloud-init divergence                | Two cloud-init pipelines: tenant (template-driven) vs. system (folder-bundled)                |

---

## 2. Solution overview

Three complementary changes:

1. **ObligationStateService** (Node Agent) — persists system VM identity *and*
   the current system VM templates in node-local SQLite, decoupling both from
   VM lifecycle and from orchestrator availability.
2. **Unified enriched `VmTemplate`** (Orchestrator) — extends the existing
   template model with artifacts and enables system VMs to be first-class
   marketplace templates. One template-rendering pipeline for system and tenant
   VMs alike.
3. **Node-authoritative lifecycle** — the node decides when to Create or
   Delete its own system VMs, using a small reconciliation matrix that consumes
   local data only. The orchestrator declares intent, identity, and templates,
   then steps out of the lifecycle. The `SystemVmReconciliationService` on
   the orchestrator is removed.

Together these establish:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              System VM Boot — Three Clean Data Sources                       │
│                                                                              │
│   ARTIFACTS              IDENTITY               CONFIGURATION                │
│   (Template)             (Node SQLite)          (Deployment)                 │
│                                                                              │
│   Binaries               Ed25519 keys           Bootstrap peers              │
│   Dashboards             WireGuard keys         Orchestrator URL             │
│   Scripts                Subnet/tunnel IP       VM ID, hostname              │
│   Config files           Auth tokens            Region, arch                 │
│                                                                              │
│   Served by:             Served by:             Injected by:                 │
│   artifact cache         /api/obligations/      cloud-init                   │
│   ${ARTIFACT_URL:x}      {role}/state           ${DECLOUD_*}                 │
│                                                                              │
│   Nothing overlaps. Nothing is duplicated across sources.                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

And on top of this substrate:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              System VM Authority — Two Sides, One Boundary                   │
│                                                                              │
│   ORCHESTRATOR DECLARES               NODE DECIDES                           │
│   ──────────────────────              ───────────────                        │
│   Obligations                         When to Create                         │
│      (which roles run here)              (matrix sees None →                 │
│                                          intent yes → issue Create)          │
│   Identity                                                                   │
│      (Ed25519, WG keys, tokens)       When to Delete                         │
│                                          (matrix sees Unhealthy or           │
│   Templates                              intent=no → issue Delete)           │
│      (cloud-init + artifacts)                                                │
│                                       When to wait                           │
│   Pushed at registration,                (matrix sees pending command)       │
│   re-evaluation, rotation                                                    │
│                                                                              │
│   ↓                                   ↓                                      │
│   Node SQLite                         CommandQueue (same as tenant VMs)      │
│                                                                              │
│   The node reports rolled-up obligationHealth in the heartbeat.              │
│   The orchestrator uses this for scheduling — not for lifecycle.             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Design principles

- **KISS** — minimum new abstractions; extend existing patterns.
- **Security first** — identity served only on virbr0 (local VM bridge), not
  externally exposed.
- **Backward compatible** — existing user VM deployment path unchanged.
- **One code path** — `BuildVmRequestFromTemplateAsync()` works for all VMs,
  system or user. Cloud-init renders the same way regardless of who initiated
  the deploy.
- **No special cases** — system VMs are marketplace templates the platform
  happens to deploy automatically. Mesh health is a service like any other.
- **Authority follows the natural boundary** — placement is a node concern
  (the node knows what it is); cluster-level eligibility is a cluster concern
  (only the orchestrator sees the whole picture).
- **Self-healing under partition** — node-local reconciliation means the
  system VM mesh survives orchestrator outages; identity, templates, and
  obligations are all cached locally.

---

## 3. Unified enriched `VmTemplate`

### 3.1 Design goals

- System VMs (Relay, DHT, BlockStore) become regular marketplace templates.
- Templates can carry binary files, dashboards, and scripts as artifacts.
- Artifact delivery is fast (node agent local cache, served over virbr0 at ~1 Gbps).
- Community members can fork system VM templates and extend them.
- Existing template functionality (cloud-init, variables, pricing, ratings) is
  unchanged.

### 3.2 `TemplateArtifact` model

**File:** `src/Orchestrator/Models/TemplateArtifact.cs` *(new)*

```csharp
/// <summary>
/// A file artifact attached to a VmTemplate.
///
/// Single delivery model: SourceUrl is author-controlled (e.g., GitHub
/// Releases, S3, CDN). The orchestrator stores only metadata (URL, SHA256,
/// size) — never the bytes.
///
/// Delivery to VMs is through the node agent's local artifact cache, which
/// fetches once from SourceUrl, verifies SHA256, and serves over virbr0 at
/// ~1Gbps. VMs never see the upstream URL — only the local cache URL.
/// </summary>
public class TemplateArtifact
{
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Artifact name — used in cloud-init variable substitution.
    /// Cloud-init references: ${ARTIFACT_URL:name}
    /// Example: "dht-node", "dashboard-assets", "shared-scripts"
    /// </summary>
    public string Name { get; set; } = string.Empty;

    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// SHA256 hex digest. Mandatory. Immutable after template publish.
    /// Verified by node agent on download, and again inside the VM.
    /// </summary>
    public string Sha256 { get; set; } = string.Empty;

    public long SizeBytes { get; set; }

    /// <summary>
    /// Author-controlled URL where the artifact bytes can be fetched.
    /// Mandatory. HTTPS required.
    ///
    /// Verified once at publish time: orchestrator GETs the URL, computes
    /// SHA256, and rejects the publish if it does not match the declared
    /// hash. Bytes are not retained — only the (URL, SHA256) tuple is stored.
    ///
    /// Immutable for a given template Version. Author rotation requires
    /// bumping VmTemplate.Version (which triggers system VM redeployment
    /// for system templates, and a re-publish review for community ones).
    ///
    /// Examples: GitHub Releases, S3 public objects, author's CDN.
    /// </summary>
    public string SourceUrl { get; set; } = string.Empty;

/// <summary>
    /// Block Store CID once artifact is replicated (future — Phase 2).
    /// When set, node agents download from Block Store instead of SourceUrl.
    ///
    /// Phase 1 assertion: field exists in schema, value is ALWAYS NULL on
    /// write in this phase. No resolution logic references this field.
    /// ArtifactCacheService always uses SourceUrl in Phase 1.
    /// Phase 2 will populate this field once BlockStore CID replication
    /// is operational — no schema migration required.
    /// </summary>
    public string? BlockStoreCid { get; set; }

    /// <summary>
    /// Target CPU architecture. null = universal (scripts, dashboards, configs).
    /// "amd64" or "arm64" for compiled binaries.
    /// </summary>
    public string? Architecture { get; set; }

    public ArtifactType Type { get; set; }

    public DateTime RegisteredAt { get; set; } = DateTime.UtcNow;
    public string RegisteredBy { get; set; } = string.Empty; // userId
}

public enum ArtifactType
{
    Binary,     // Compiled executables (dht-node, blockstore-node, relay-node)
    Script,     // Shell scripts (wg-mesh-enroll.sh, notify-ready.sh)
    WebAsset,   // HTML/CSS/JS bundles (dashboard.tar.gz)
    Config,     // Configuration files (systemd units, nginx.conf)
    Archive     // General archives (tar.gz with mixed content)
}
```

### 3.3 `VmTemplate` extensions

**File:** `src/Orchestrator/Models/VmTemplate.cs` *(modified)*

Add to existing `VmTemplate` class:

```csharp
/// <summary>
/// Files attached to this template, referenced by author-controlled URL.
/// The node agent cache fetches each artifact once from its SourceUrl,
/// verifies SHA256, and serves it to local VMs over virbr0 at ~1Gbps.
/// The platform stores only metadata — never the bytes.
/// </summary>
public List<TemplateArtifact> Artifacts { get; set; } = new();

/// <summary>
/// Template version — bumped when cloud-init or artifacts change.
/// The node-side SystemVmReconciler's Healthy predicate compares running
/// version to template version; mismatch triggers redeploy.
/// </summary>
public int Version { get; set; } = 1;

/// <summary>
/// Slug for system templates (e.g., "system-relay", "system-dht", "system-blockstore").
/// Community templates may also have slugs for discoverability.
/// </summary>
public string? Slug { get; set; }
```

The existing `Services[]` field on `VmTemplate` already carries service
declarations (name, port, check type, timeout). System templates use it to
declare both ordinary services (`dht-api` on port 5090) and mesh-health
services (`dht-mesh` on port 5090, `HttpGet /health/mesh`). See §5.4.

### 3.4 Artifact storage model

**The platform never stores artifact bytes.** Authors host their own bytes
(GitHub Releases, S3, CDN) and provide the URL plus SHA256. Only metadata
(`Name`, `SourceUrl`, `Sha256`, `SizeBytes`, `Architecture`, `Type`) is
stored in MongoDB on the `VmTemplate.Artifacts[]` embedded array.

**Why external-only:**

| Concern          | Outcome                                                                                          |
|------------------|--------------------------------------------------------------------------------------------------|
| Legal posture    | Platform is a *directory*, not a *host* (DMCA / EU DSA / criminal-content regimes). On takedown, de-listing replaces removing-from-disk. |
| Attack surface   | No upload endpoint → no zip bombs, MIME spoofing, path-traversal in filenames, "served as trusted platform content" attacks. |
| Operational      | No disk pressure, no backup of artifact files, no orphan GC, no quota enforcement.               |
| Decentralization | Aligns with platform ethos. Block Store Phase 2 is the durability layer (replicate by CID).      |
| Performance      | Unchanged — node agent cache (§3.6) serves at ~1 Gbps regardless of origin.                       |

**Size limits:** None enforced by the platform. The node agent cache may
prune the largest least-recently-used artifacts when disk pressure rises
(see `IArtifactCacheService.PruneAsync`).

**Availability:** If an author takes down their URL, any node that has
already cached the artifact continues serving it. New nodes fail prefetch
with a clear error and the system VM stays in `Pending` until resolved.
Block Store Phase 2 (replicate by CID) closes this gap permanently — when
`BlockStoreCid` is populated, the node agent prefers Block Store over
`SourceUrl`.

### 3.5 New orchestrator endpoints

**File:** `src/Orchestrator/Controllers/MarketplaceController.cs` *(extended)*

```
POST /api/marketplace/templates/{id}/artifacts
    Register an artifact reference on a template.
    Body: { name, sourceUrl, sha256, sizeBytes, type, architecture? }
    Validation:
      - URL must be HTTPS
      - Orchestrator issues HEAD to confirm reachability
      - Artifact name must be unique within the template
    Allowed only while template is in Draft status.
    Returns: TemplateArtifact

DELETE /api/marketplace/templates/{id}/artifacts/{name}
    Remove an artifact reference. Draft status only.

POST /api/marketplace/artifacts/probe          (UX helper, optional)
    Body: { sourceUrl }
    Server fetches once, returns { sizeBytes, sha256, contentType }.
    Lets the authoring UI offer "paste URL → auto-fill hash" so authors
    don't have to compute SHA256 by hand. The author confirms before save.
    Rate-limited per user.
```

There is no `GET /artifacts/{name}` from the orchestrator. The orchestrator
never serves artifact bytes — node agents prefetch directly from `SourceUrl`,
and VMs read only from the local node agent cache.

### 3.6 Node agent artifact cache

**File:** `src/DeCloud.NodeAgent/Controllers/ArtifactCacheController.cs` *(new)*

```csharp
/// <summary>
/// Serves cached artifacts to local VMs over virbr0 (192.168.122.1:5100).
/// VMs use ${ARTIFACT_URL:name} which resolves to:
///   http://192.168.122.1:5100/api/artifacts/{sha256}
///
/// On first request after a template assignment, the cache prefetches the
/// artifact from its author-controlled SourceUrl (e.g., GitHub Releases),
/// streams the bytes through a SHA256 hasher, and stores them locally
/// keyed by hash. Subsequent VMs on this node hit the cache directly.
///
/// Security: virbr0 is accessible only from VMs on this node.
/// No authentication required — isolation is network-level.
/// SHA256 verification is the integrity mechanism.
/// </summary>
[ApiController]
[Route("api/artifacts")]
public class ArtifactCacheController : ControllerBase
{
    // GET /api/artifacts/{sha256}
    // Serve artifact bytes from local cache.
    // Returns 404 if not cached (VM should not call before prefetch succeeds).

    // GET /api/artifacts/{sha256}/sha256
    // Returns sha256 checksum as plain text. Fast integrity check.

    // POST /api/artifacts/prefetch
    // Body: { sha256: string, sourceUrl: string, expectedSizeBytes: long }
    // Downloads artifact from sourceUrl, verifies SHA256, caches locally.
    // Returns 200 immediately if already cached.
    // Returns 202 if download started (poll /status/{sha256} for completion).

    // GET /api/artifacts/prefetch/status/{sha256}
    // Returns prefetch status: Pending | Downloading | Ready | Failed
}
```

**Cache location:** `/var/lib/decloud/artifact-cache/{sha256}` (keyed by
content hash, not name).

**File:** `src/DeCloud.NodeAgent.Infrastructure/Services/ArtifactCacheService.cs` *(new)*

```csharp
public interface IArtifactCacheService
{
    Task<bool> IsCachedAsync(string sha256, CancellationToken ct = default);
    Task<Stream?> GetArtifactStreamAsync(string sha256, CancellationToken ct = default);
    Task<PrefetchResult> PrefetchAsync(string sha256, string sourceUrl, long expectedSizeBytes, CancellationToken ct = default);
    Task<PrefetchStatus> GetPrefetchStatusAsync(string sha256, CancellationToken ct = default);
    Task PruneAsync(IEnumerable<string> activeSha256s, CancellationToken ct = default);
}
```

### 3.7 Variable resolution

**File:** `src/Orchestrator/Services/TemplateService.cs` *(extended)*

Add `ResolveArtifactVariables()` to expand artifact references in cloud-init
before VM deployment:

```csharp
/// <summary>
/// Expands artifact variable references in cloud-init templates.
/// Called from BuildVmRequestFromTemplateAsync() before sending CreateVm command.
///
/// Variable prefixes:
///   ${ARTIFACT_URL:name}    → node agent cache URL (always local — VMs never see upstream URL)
///   ${ARTIFACT_SHA256:name} → SHA256 checksum for in-VM verification
///
/// Architecture selection: selects the artifact matching the target node's architecture.
/// Falls back to architecture=null (universal) if no arch-specific artifact exists.
/// </summary>
private Dictionary<string, string> ResolveArtifactVariables(
    VmTemplate template,
    string targetArchitecture)  // "amd64" or "arm64"
{
    var variables = new Dictionary<string, string>();

    foreach (var artifact in template.Artifacts)
    {
        // Skip arch-specific artifacts that don't match target
        if (artifact.Architecture != null &&
            artifact.Architecture != targetArchitecture)
            continue;

        // All artifacts resolve to the local node agent cache.
        // The cache fetches from artifact.SourceUrl on first miss.
        variables[$"ARTIFACT_URL:{artifact.Name}"] =
            $"http://192.168.122.1:5100/api/artifacts/{artifact.Sha256}";

        variables[$"ARTIFACT_SHA256:{artifact.Name}"] = artifact.Sha256;
    }

    return variables;
}
```

### 3.8 System VM templates in `TemplateSeederService`

System VMs are seeded as regular `VmTemplate` records by `TemplateSeederService`.
The only difference from community templates: `AuthorId = "system"`,
`IsCommunity = false`.

**Template slugs:**

| Role        | Template slug      | Category         |
|-------------|--------------------|------------------|
| Relay       | `system-relay`      | `infrastructure` |
| DHT         | `system-dht`        | `infrastructure` |
| BlockStore  | `system-blockstore` | `infrastructure` |

**Example system template structure (`system-dht`):**

```csharp
new VmTemplate
{
    Slug = "system-dht",
    Name = "DHT Peer Node",
    Version = 1,
    Category = "infrastructure",
    AuthorId = "system",
    IsCommunity = false,
    IsVerified = true,
    Visibility = TemplateVisibility.Public,    // Visible in marketplace
    PricingModel = TemplatePricingModel.Free,  // Free to deploy

    CloudInitTemplate = /* full YAML — read at seed time from
                          assets/dht/cloud-init.yaml in the repo */,

    Artifacts = new List<TemplateArtifact>
    {
        new() { Name = "dht-node", Type = ArtifactType.Binary, Architecture = "amd64", ... },
        new() { Name = "dht-node", Type = ArtifactType.Binary, Architecture = "arm64", ... },
        new() { Name = "dht-dashboard-assets", Type = ArtifactType.WebAsset, ... },
        new() { Name = "shared-scripts", Type = ArtifactType.Archive, ... },
    },

    // Service declarations consumed by VmReadinessMonitor on the node.
    // Includes mesh-health checks alongside ordinary service checks (see §5.4).
    Services = new List<TemplateServiceDeclaration>
    {
        new() { Name = "System",   CheckType = CheckType.CloudInitDone, TimeoutSeconds = 300 },
        new() { Name = "dht-api",  Port = 5090, CheckType = CheckType.HttpGet, HttpPath = "/health",       TimeoutSeconds = 60 },
        new() { Name = "dht-mesh", Port = 5090, CheckType = CheckType.HttpGet, HttpPath = "/health/mesh",  TimeoutSeconds = 600 },
    }
}
```

The system VM binary owns the semantics of `/health/mesh` — see §5.4.

### 3.9 Unified deployment path

Both system and tenant VMs deploy through the same template-rendering
pipeline. The difference is who initiates the deploy:

```csharp
// Tenant VM — user-initiated via marketplace:
var request = await _templateService.BuildVmRequestFromTemplateAsync(
    templateId,
    vmName: userVmName,
    customSpec: userOverrides,
    environmentVariables: userEnv);

await _vmService.CreateVmAsync(userId, request, targetNodeId: null);

// System VM — node-initiated via local SystemVmReconciler (§5):
var template = await _localTemplateStore.GetSystemTemplateAsync(role);  // from node SQLite
var request  = await _templateRenderer.BuildVmRequestFromTemplateAsync(
    template,
    vmName: $"{role.ToString().ToLower()}-{nodeId[..8]}",
    environmentVariables: deployConfig);

await _commandQueue.EnqueueCreateAsync(request);  // same internal path tenant CreateVm uses
```

`BuildSystemVmDeployConfig` provides **deployment-specific** variables only
(bootstrap peers, orchestrator URL, region) — **not** identity. Identity is
fetched by the VM from `ObligationStateService` at boot time.

There is no orchestrator-side `DeploySystemVmAsync` method. The orchestrator
does not initiate system VM deploys. The node's reconciliation matrix does
(§5).

### 3.10 What was deleted ✅

All planned deletions shipped in P7 and P11.

**Orchestrator (P7):**

```
src/Orchestrator/Services/SystemVm/SystemVmReconciliationService.cs
    DELETED entirely — renamed to SystemVmObligationService during
    implementation; all deploy/check/verify/retry logic removed;
    background loop now runs EnsureObligationsAsync only

src/Orchestrator/Services/RelayNodeService.cs
    DELETED: DeployRelayVmAsync() ✅
    KEPT:    AssignCgnatNodeToRelayAsync(), FindBestRelayForCgnatNodeAsync(),
             EnsurePeerRegisteredAsync(), FindAndAssignNewRelayAsync(),
             RemoveCgnatNodeFromRelayAsync()

src/Orchestrator/Services/DhtNodeService.cs
    DELETED: DeployDhtVmAsync() ✅
    KEPT:    GetBootstrapPeersAsync(), GetAdvertiseIp(), port constants

src/Orchestrator/Services/BlockStoreService.cs
    DELETED: DeployBlockStoreVmAsync() ✅
    KEPT:    CID indexing, manifest management, /join handlers,
             GetBootstrapPeersAsync()

src/Orchestrator/Services/NodeService.cs
    DELETED: heartbeat-path Active promotion block ✅
             (Deploying → Active via heartbeat.ActiveVms — double-writer pattern)
    KEPT:    system-VM ghost detection already excluded in prior patch
             (reportedType is VmType.Relay or VmType.Dht or VmType.BlockStore → continue)
    ADDED:   obligationHealth consumed from heartbeat; ObligationHealth
             field on Node record stamped per role (P6 work)

src/Orchestrator/Services/SystemVm/SystemVmObligationService.cs
    DELETED: CloudInitReadyTimeout, ActiveVmGracePeriod, HeartbeatSnapshotTtl,
             ProvisioningTimeout, StuckDeletingTimeout (dead constants)
    KEPT:    EnsureObligationsAsync, TryAdoptExistingVmAsync,
             TryDiscoverHealthySystemVmAsync, EnsureObligationsForNodeAsync
    UPDATED: Class doc — obligation management only; VM deployment excluded
```

**Node Agent (P11):**

```
src/DeCloud.NodeAgent/Services/GoBinaryBuildStartupService.cs
    DELETED entirely ✅

src/DeCloud.NodeAgent/CloudInit/Templates/dht-vm/     DELETED ✅
src/DeCloud.NodeAgent/CloudInit/Templates/blockstore-vm/  DELETED ✅
src/DeCloud.NodeAgent/CloudInit/Templates/relay-vm/   DELETED ✅

src/DeCloud.NodeAgent.Infrastructure/Services/CloudInitTemplateService.cs
    DELETED: PopulateDhtVariablesAsync ✅
    DELETED: PopulateRelayVariablesAsync ✅
    DELETED: PopulateBlockStoreVariablesAsync ✅
    DELETED: InjectDhtExternalTemplatesAsync ✅
    DELETED: InjectRelayExternalTemplatesAsync ✅
    DELETED: InjectBlockStoreExternalTemplatesAsync ✅
    DELETED: PopulateWireGuardMeshVariablesAsync ✅
    DELETED: ResolveLocalRelayInfoAsync ✅
    DELETED: VmType.Dht, VmType.Relay, VmType.BlockStore switch cases ✅
    KEPT:    InjectGeneralExternalTemplatesAsync, PopulateInferenceVariablesAsync,
             LoadExternalTemplateAsync, ReplaceWithIndentation,
             GetTargetArchitecture (now uses ResourceDiscoveryService.GetArchitectureNormalised())

install.sh
    DELETED: install_go() function ✅
    DELETED: build_dht_binary() function ✅
    DELETED: build_blockstore_binary() function ✅
    DELETED: cleanup_stale_system_vms() function ✅
    DELETED: binary cache preserve/restore blocks in download_node_agent() ✅
    KEPT:    build_gpu_proxy(), build_node_agent(), all non-binary install steps
```

**Node agent:**

```
src/DeCloud.NodeAgent/CloudInit/Templates/                    DELETED (entire folder)
    └── general-vm-cloudinit.yaml
    └── relay-vm-cloudinit.yaml         (and bundled relay-api.py, dashboard.*)
    └── dht-vm-cloudinit.yaml           (and bundled dht-node-src/, dashboard.*)
    └── blockstore-vm-cloudinit.yaml    (and bundled blockstore-node-src/, dashboard.*)
    └── inference-vm-cloudinit.yaml

src/DeCloud.NodeAgent/Services/GoBinaryBuildStartupService.cs DELETED entirely

src/DeCloud.NodeAgent.Infrastructure/Services/CloudInitTemplateService.cs
    DELETED: LoadTemplateAsync, GetTemplatePath, _templateBasePath, _templateCache
    DELETED: InjectGeneralExternalTemplatesAsync,
             InjectDhtExternalTemplatesAsync,
             InjectRelayExternalTemplatesAsync,
             InjectBlockStoreExternalTemplatesAsync
    DELETED: LoadDhtBinaryAsync, LoadBlockStoreBinaryAsync
    DELETED: BuildDhtBinaryFromSourceAsync, BuildBlockStoreBinaryFromSourceAsync
    DELETED: _externalTemplateCache, _templateCache, per-role variable methods
    KEPT:    BuildTemplateVariablesAsync (still substitutes runtime values)
    KEPT:    the substitution engine itself — but it now operates on the
             CreateVm payload's UserData string, not on disk-loaded files

install.sh
    REMOVE Go toolchain install step
    REMOVE dht-node-source / blockstore-node-source build invocations
    REMOVE the .sha256 marker handling for those builds
```

**Repo asset relocation:** the cloud-init YAML and bundled assets aren't
deleted from the repository — they move to a top-level `assets/` directory
and are referenced by the system templates' `Artifacts[].SourceUrl` (typically
GitHub Release URLs). The build/release pipeline uploads them to GitHub
Releases on each version bump.

```
assets/                                       (in repo root)
  ├── dht/
  │   ├── cloud-init.yaml          (read by TemplateSeederService at seed time)
  │   ├── dht-node-src/            (Go source — built by CI, uploaded to releases)
  │   └── dashboard.*              (uploaded as a tarball artifact)
  ├── blockstore/
  │   └── ... (same structure)
  └── relay/
      └── ... (same structure)
```

---


## 4. ObligationStateService — identity & template persistence

### 4.1 Design goals

- System VM identity (Ed25519 keys, WireGuard keys, auth tokens, subnet IPs)
  survives VM crashes and redeployments.
- The current system VM templates also live on the node, so the node can
  redeploy a system VM without contacting the orchestrator.
- Both are stored on the **node** (in SQLite), not in the VM's ephemeral disk.
- Node agent serves identity to VMs over virbr0 — same trust boundary as the
  artifact cache.
- Orchestrator remains the source of truth for both; node agent stores the
  authoritative local copy.
- Version-based conflict resolution: **higher version always wins**. The
  orchestrator is the only writer of authoritative state; under normal
  operation it always holds the highest version. See §4.7.
- No orchestrator contact required at VM recovery time.

### 4.2 SQLite schema

**File:** `src/DeCloud.NodeAgent.Infrastructure/Persistence/ObligationStateRepository.cs` *(new)*

Extends the existing SQLite database (same file as `VmRepository` and
`PortMappingRepository`):

```sql
-- Identity per role
CREATE TABLE IF NOT EXISTS obligation_state (
    role        TEXT PRIMARY KEY,   -- "relay" | "dht" | "blockstore"
    state_json  TEXT NOT NULL,      -- Role-specific identity blob (see §4.3)
    version     INTEGER NOT NULL,   -- Monotonic version from orchestrator
    updated_at  TEXT NOT NULL       -- ISO 8601 UTC timestamp
);

-- The set of obligations this node currently holds.
-- Pushed by the orchestrator at registration and on capability changes.
CREATE TABLE IF NOT EXISTS obligation (
    role        TEXT PRIMARY KEY,   -- "relay" | "dht" | "blockstore"
    deps_json   TEXT NOT NULL,      -- JSON array of dependency role names
    updated_at  TEXT NOT NULL
);

-- System templates per role.
-- Pushed by the orchestrator at registration and on template version bumps.
CREATE TABLE IF NOT EXISTS system_template (
    role          TEXT PRIMARY KEY, -- "relay" | "dht" | "blockstore"
    template_json TEXT NOT NULL,    -- Full VmTemplate including Artifacts[]
    version       INTEGER NOT NULL,
    updated_at    TEXT NOT NULL
);
```

Three tables, all keyed by canonical role name. The node-side
`SystemVmReconciler` (§5) consumes all three:

- `obligation` answers *intent* (does this node have to run this role?).
- `obligation_state` is the *identity* the VM gets at boot.
- `system_template` is the *deploy spec* the reconciler uses to render
  `CreateVm` requests.

### 4.3 Identity state per role

**File:** `src/DeCloud.NodeAgent.Core/Models/ObligationState.cs` *(new)*

```csharp
/// <summary>
/// Base class for all obligation identity state.
/// Stored in SQLite obligation_state table as JSON.
/// </summary>
public abstract class ObligationStateBase
{
    public int Version { get; set; }
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}

public class RelayObligationState : ObligationStateBase
{
    public string WireGuardPrivateKey { get; set; } = string.Empty;
    public string WireGuardPublicKey  { get; set; } = string.Empty;
    public string TunnelIp            { get; set; } = string.Empty;  // e.g. "10.20.0.1"
    public string RelaySubnet         { get; set; } = string.Empty;  // e.g. "10.20.0.0/24"
    public string AuthToken           { get; set; } = string.Empty;  // NEVER log
}

public class DhtObligationState : ObligationStateBase
{
    public string Ed25519PrivateKeyBase64 { get; set; } = string.Empty;  // NEVER log
    public string PeerId                  { get; set; } = string.Empty;  // safe to log
    public string WireGuardPrivateKey     { get; set; } = string.Empty;  // NEVER log
    public string WireGuardPublicKey      { get; set; } = string.Empty;  // safe to log
    public string TunnelIp                { get; set; } = string.Empty;  // e.g. "10.30.0.x"
    public string AuthToken               { get; set; } = string.Empty;  // NEVER log
}

public class BlockStoreObligationState : ObligationStateBase
{
    public string Ed25519PrivateKeyBase64 { get; set; } = string.Empty;
    public string PeerId                  { get; set; } = string.Empty;
    public string WireGuardPrivateKey     { get; set; } = string.Empty;
    public string WireGuardPublicKey      { get; set; } = string.Empty;
    public string TunnelIp                { get; set; } = string.Empty;
    public string AuthToken               { get; set; } = string.Empty;
    public long   StorageQuotaBytes       { get; set; }              // 5% of total node storage
}
```

### 4.4 Service interface

**File:** `src/DeCloud.NodeAgent.Core/Interfaces/IObligationStateService.cs` *(new)*

```csharp
public interface IObligationStateService
{
    // ── Identity ─────────────────────────────────────────────────────
    Task StoreStateAsync(string role, string stateJson, int version, CancellationToken ct = default);
    Task<string?> GetStateJsonAsync(string role, CancellationToken ct = default);
    Task<int>     GetStateVersionAsync(string role, CancellationToken ct = default);
    Task          DeleteStateAsync(string role, CancellationToken ct = default);

    // ── Obligations (the set of roles this node holds) ───────────────
    Task StoreObligationsAsync(IEnumerable<ObligationDescriptor> obligations, CancellationToken ct = default);
    Task<IReadOnlyList<ObligationDescriptor>> GetObligationsAsync(CancellationToken ct = default);

    // ── System templates ─────────────────────────────────────────────
    Task StoreSystemTemplateAsync(string role, string templateJson, int version, CancellationToken ct = default);
    Task<VmTemplate?> GetSystemTemplateAsync(string role, CancellationToken ct = default);
    Task<Dictionary<string, int>> GetSystemTemplateVersionsAsync(CancellationToken ct = default);
}
```

### 4.5 Node agent HTTP endpoint

**File:** `src/DeCloud.NodeAgent/Controllers/ObligationStateController.cs` *(new)*

```csharp
/// <summary>
/// Serves obligation identity state to system VMs over virbr0.
///
/// Security model:
///   - Bound to 192.168.122.1:5100 (virbr0 bridge only)
///   - Accessible only from VMs on this node's libvirt network
///   - No authentication: isolation is network-level (virbr0 is not routable externally)
///   - State contains private keys — virbr0 isolation is the security boundary
///
/// Note: Only identity is exposed via HTTP. Obligations and templates are
/// internal to the node agent (consumed by SystemVmReconciler) and are NOT
/// served over this endpoint — VMs have no need for them.
/// </summary>
[ApiController]
[Route("api/obligations")]
public class ObligationStateController : ControllerBase
{
    // GET /api/obligations/{role}/state
    // Returns identity state JSON for the given role.
    // Returns 404 if no state exists (first boot or state cleared).

    // GET /api/obligations/{role}/version
    // Returns current version as integer. Lightweight check.
}
```

### 4.6 Orchestrator state generation and delivery

**File:** `src/Orchestrator/Services/SystemVm/ObligationStateGenerator.cs` *(new)*

The orchestrator generates obligation state when obligations are first
created, and delivers it to the node agent via the registration response
and via a heartbeat-driven version-pull mechanism.

```csharp
public class ObligationStateGenerator
{
    /// <summary>
    /// Generate fresh identity state for a new obligation.
    /// Called once at obligation creation time.
    /// </summary>
    public ObligationStateBase GenerateState(SystemVmRole role, Node node) { ... }
}
```

### 4.7 Conflict resolution

```
Orchestrator version > node version  → node accepts update
Orchestrator version == node version → no update (preserved)
Orchestrator version <  node version → log warning, no update
                                       (node holds a higher version than orchestrator;
                                       this indicates node SQLite corruption yielding a
                                       phantom-high version, or a rollback. The node keeps
                                       its higher version — "higher version wins" is the
                                       single rule. Operator recovery: clear node SQLite
                                       obligation_state for the affected role; next heartbeat
                                       will re-push from orchestrator at current version.)
```

The node reports its current versions in each heartbeat; the orchestrator
includes any payloads where its version is higher in the heartbeat response.
This applies symmetrically to identity state and to system templates.

### 4.8 Cloud-init identity query pattern

System VM templates query the node agent for identity at boot, before
starting services:

```yaml
# In system-dht cloud-init template:
runcmd:
  # ── Artifacts (from cache — fast, local) ──
  - curl -sf ${ARTIFACT_URL:dht-node} -o /usr/local/bin/dht-node
  - echo "${ARTIFACT_SHA256:dht-node}  /usr/local/bin/dht-node" | sha256sum -c -
  - chmod +x /usr/local/bin/dht-node
  - curl -sf ${ARTIFACT_URL:dht-dashboard-assets} | tar xz -C /opt/decloud-dht/
  - curl -sf ${ARTIFACT_URL:shared-scripts} | tar xz -C /usr/local/bin/

  # ── Identity (from obligation state — persistent across redeployments) ──
  - |
    IDENTITY=$(curl -sf --retry 5 --retry-delay 2 \
      http://192.168.122.1:5100/api/obligations/dht/state)

    if [ -z "$IDENTITY" ]; then
      echo "FATAL: No obligation state available — cannot start DHT service"
      exit 1
    fi

    mkdir -p /var/lib/decloud-dht /etc/decloud-dht
    chmod 700 /var/lib/decloud-dht

    echo "$IDENTITY" | jq -r '.ed25519PrivateKeyBase64' | base64 -d \
      > /var/lib/decloud-dht/identity.key
    chmod 600 /var/lib/decloud-dht/identity.key

  # ── Start service (reads identity.key from disk — same peer ID as before) ──
  - systemctl enable --now decloud-dht
```

`loadOrCreateIdentity()` in the DHT binary reads from disk first — cloud-init
placed the key file before the binary starts, so it always finds the
existing key. **Same peer ID across every redeployment.**

---


## 5. Lifecycle authority — node-side reconciliation

### 5.1 The authority split

| Concern                                   | Authority    | Mechanism                                              |
|-------------------------------------------|--------------|--------------------------------------------------------|
| Intent (which roles a node should run)    | Orchestrator | Pushed at registration + on capability changes         |
| Identity (Ed25519, WG keys, auth tokens)  | Orchestrator | Pushed at registration + on rotation                   |
| Templates (cloud-init + artifact refs)    | Orchestrator | Pushed at registration + on version bumps              |
| **Lifecycle** (when to Create / Delete)   | **Node**     | Local matrix consuming local data                      |
| Health observation                        | Node reports, orchestrator consumes | Heartbeat carries rolled-up `obligationHealth` |
| Scheduling eligibility                    | Orchestrator | Reads `obligationHealth`, fast-paths admission         |

Why this split: system VMs are **node fixtures**, not cluster placements.
Their existence is determined by what the node *is* (its hardware
capabilities and dependency-graph position), not by what the cluster decides
to place. The node sees its own state most freshly. Cluster-level policy —
which nodes are eligible for which roles — remains an orchestrator concern,
expressed by the obligation set the orchestrator pushes.

This boundary also makes the substrate self-healing under orchestrator
outage: every node has obligations, identity, and templates locally cached.
A node that boots after a power loss restores its system VMs without
contacting the orchestrator.

### 5.2 The reconciliation matrix

A new background service on the node agent runs every 30 seconds. For each
role the node has an obligation for:

```
intent  = node has obligation(role) ∧ deps satisfied
        // pure read of local SQLite

reality = project_local(role)
        // None      — no VM of this role tracked locally
        // Healthy   — VM Running ∧ IsFullyReady
        // Unhealthy — anything else

pending = lookup_outstanding(role)
        // None | Create(issuedAt) | Delete(issuedAt)
```

Action is a pure function of the triple:

| intent | reality   | pending | action                              |
|--------|-----------|---------|-------------------------------------|
| *      | *         | Create  | wait for completion or timeout      |
| *      | *         | Delete  | wait for completion or timeout      |
| no     | None      | None    | nothing — converged                 |
| no     | *         | None    | issue Delete                        |
| yes    | None      | None    | issue Create (deps satisfied)       |
| yes    | Healthy   | None    | nothing — converged                 |
| yes    | Unhealthy | None    | issue Delete (next cycle: Create)   |

That is the entire state machine. Every scenario the previous orchestrator-
side reconciler handled — first deploy, redeploy after crash, identity
rotation, binary version drift, capability loss, racing creates, lost ack,
node reboot, node-agent restart — falls out of the matrix without
case-specific code. The companion document `SYSTEM_VM_RECONCILIATION_DESIGN.md`
walks through each scenario in detail.

### 5.3 `OutstandingCommands` map

The matrix's `pending` axis is an in-memory map on the node agent keyed by
role:

```csharp
internal sealed class OutstandingCommand
{
    public required string CommandId { get; init; }
    public required CommandKind Kind { get; init; }   // Create | Delete
    public string? VmId { get; init; }                // bound at issuance
    public required DateTime IssuedAt { get; init; }
}

public interface IOutstandingCommands
{
    bool TryGet(SystemVmRole role, out OutstandingCommand cmd);
    void Set(SystemVmRole role, OutstandingCommand cmd);
    void Clear(SystemVmRole role);
    void SweepExpired(TimeSpan timeout);   // called once per cycle
}
```

The map is in-memory because the node's command pipeline is in-process.
On node-agent restart, `LibvirtVmManager.ReconcileAllWithLibvirtAsync`
reconciles VM state from libvirt; outstanding commands from the previous
boot are inherently expired (any unacked command from the previous process
is gone with the process). The matrix re-evaluates from scratch on first
cycle — idempotent at the libvirt layer (duplicate-name guard).

### 5.4 Mesh health as a service

Mesh health is **not a special case**. It is an ordinary service declared
in the system template, checked by `VmReadinessMonitor` via existing
`HttpGet` plumbing, contributing to `IsFullyReady` like any other service.

Each system VM exposes a `/health/mesh` endpoint that returns 200 when
peers are sufficient, 503 when not. The system VM binary owns the
threshold semantics:

| Role        | `/health/mesh` semantics                                                                                       |
|-------------|---------------------------------------------------------------------------------------------------------------|
| DHT         | 200 if `connectedPeers >= MIN_PEERS && now - lastPeerSeen < STALENESS`, else 503. Cold-start: 200 for first 60 s. |
| Relay       | 200 if at least one assigned CGNAT peer has handshake age < 5 min, OR if no peers are assigned (an idle relay is healthy), else 503. |
| BlockStore  | 200 if `connectedPeers >= 1` past cold-start, else 503.                                                       |

The threshold values (`MIN_PEERS`, `STALENESS`) are owned by the system VM
binary, not the node agent. They are exposed as env vars in the cloud-init
env file so operators can tune without a binary rebuild:

| Role       | Env var                  | Default | Injected by                          |
|------------|--------------------------|---------|--------------------------------------|
| DHT        | `DHT_MESH_MIN_PEERS`     | `1`     | `dht.env` in cloud-init `write_files` |
| DHT        | `DHT_MESH_STALENESS_SEC` | `300`   | `dht.env` in cloud-init `write_files` |
| BlockStore | `BS_MESH_MIN_PEERS`      | `1`     | `blockstore.env` in cloud-init        |
| Relay      | `RELAY_HANDSHAKE_MAX_AGE`| `300`   | `relay.env` in cloud-init             |

These env vars are present in the cloud-init YAML constants in
`SystemVmTemplateSeeder.cs`. To change a threshold: update the env file
line in the relevant `*_CLOUD_INIT` constant, bump `TemplateRevision`,
and commit. The reconciler redeploys on next version drift detection.                                                     |

Cold-start grace is the service's `TimeoutSeconds` (e.g. 600 s for
`dht-mesh`). The system VM's endpoint implementation decides what to
return during the warm-up window — typically 200 unconditionally, then
flip to threshold-based once the binary has had time to dial peers.

**Why this factoring works:** the node agent has no special threshold
logic. `VmReadinessMonitor` runs the existing HTTP check, gets 200 or
503, marks the service `Ready` or `TimedOut`/`Failed`. `IsFullyReady`
naturally requires every service Ready, including mesh. The matrix sees
`Healthy` only when all services pass. Threshold tuning lives where it
belongs — inside the system VM's binary, alongside the peer-counting logic
it already has.

**Binary version drift uses the same factoring.** The system template
declares a `version` service:

```yaml
services:
  - name: dht-version
    port: 5090
    checkType: HttpGet
    httpPath: /health/version
    timeoutSeconds: 30
```

The DHT VM's `/health/version` endpoint returns 200 if its running binary's
SHA256 matches the expected hash (passed via cloud-init env var from the
template's `${ARTIFACT_SHA256:dht-node}`), 503 otherwise. When the template
version is bumped on the orchestrator, the bump propagates to nodes via
the `system_template` push. New deployments use the new artifact hash.
Existing VMs are still running the old hash — `/health/version` returns
503 — `IsFullyReady` is false — the matrix sees `Unhealthy` — Delete →
Create with the new template. **No "binary update" branch in the matrix.
No canary mechanism. The existing per-cycle backoff naturally spreads
redeploys across the fleet.**

### 5.5 Heartbeat shape

The heartbeat carries one new field — a per-role rolled-up health signal:

```csharp
public class NodeHeartbeat
{
    // ... existing fields (metrics, activeVms, etc.) unchanged ...

    /// <summary>
    /// Per-role system VM health, rolled up by the node from local matrix
    /// reality projection. Used by the orchestrator's scheduling fast-path.
    /// </summary>
    public Dictionary<string, string> ObligationHealth { get; set; } = new();
    // e.g. { "relay": "Healthy", "dht": "Healthy", "blockstore": "Unhealthy" }
}
```

The orchestrator reads this field from each heartbeat and uses it in the
scheduler hard-filter. It does **not** write `ObligationHealth` to a
parallel `Node.ObligationHealth` property — `SystemVmObligation.Status`
(on the `Node` record) remains the single orchestrator-side writer for
per-role health. Adding a parallel field would create two writers
(`SystemVmObligationService` and the heartbeat handler) fighting while
both are active — the rejected double-writer pattern (see P4 decision log).
The rolled-up signal from heartbeat is consumed in-flight for scheduling;
it is not persisted as a separate field.

> Node N is ineligible to receive new tenant VMs if any of its required
> system VM obligations report not-Healthy for longer than
> `scheduling_grace_window`.

The underlying per-VM service-readiness data continues to ride the heartbeat
as it does today. The orchestrator can verify the rolled-up signal against
the raw data on demand — but this is an audit, not a hot-path operation.

### 5.6 What the orchestrator still does

Five surfaces remain on the orchestrator. All narrow.

1. **`IObligationEligibility.ComputeObligations`** — runs at registration
   and on capability re-evaluation. The result is pushed to the node's
   `obligation` table. Same logic as today.

2. **`ObligationStateGenerator`** — runs at registration and on rotation;
   pushes identity to the node's `obligation_state` table. Same as the
   resilience design.

3. **System template push** — at registration and on template version
   bumps, the orchestrator includes the current `system-relay`,
   `system-dht`, `system-blockstore` templates in the response, scoped to
   the node's eligibility. Re-pushed when versions diverge from what the
   node reports.

4. **`MarkNodeVmsAsErrorAsync`** — for **tenant** VMs only when a node
   goes offline. System VMs are not touched (their existence is the node's
   concern; an offline node's system VMs simply stop being reachable, which
   neighbouring nodes notice via mesh health on their side).

5. **Heartbeat consumption** — orchestrator reads `obligationHealth` per
   role from each heartbeat. Used by the scheduler hard-filter.

The orchestrator does *not* run a system-VM reconciliation loop. The
~1500-line `SystemVmReconciliationService` is removed entirely.

### 5.7 What the node agent gains

| Surface                    | Status        | Description                                                                          |
|----------------------------|---------------|--------------------------------------------------------------------------------------|
| `SystemVmReconciler`        | **new**       | 30-second background service. Runs the matrix.                                       |
| `OutstandingCommands` map   | **new**       | In-memory pending-command tracking, keyed by role.                                   |
| `ObligationStateService`    | extended      | New responsibilities: store obligations table, store system templates table.        |
| `system_template` table     | **new**       | SQLite, mirrors `obligation_state` shape.                                            |
| `obligation` table          | **new**       | SQLite, holds the role list and dependency graph snapshot.                          |
| `VmReadinessMonitor`        | unchanged     | Already runs `HttpGet` checks; mesh-health and version-check are just more entries. |
| Heartbeat builder           | extended      | Computes and includes the rolled-up `ObligationHealth` field.                        |
| `CloudInitTemplateService`  | shrunk        | Substitution engine only; the template arrives via `CreateVm` payload, not from disk. |
| `SystemVmWatchdogService`   | unchanged     | Still ensures `virbr0` active, restarts stopped system VMs at startup, periodic tap re-attach. Triggers a single matrix tick after startup so the reconciler picks up restarted VMs immediately. |

### 5.8 The matrix in pseudocode

```csharp
// SystemVmReconciler.ExecuteAsync — one cycle per 30 seconds

foreach (var obligation in await _obligationStore.GetObligationsAsync(ct))
{
    var role = obligation.Role;

    // ── 1. Compute the three axes ─────────────────────────────────────
    var intent  = ComputeIntent(obligation);            // (wantDeployed, depsMet)
    var reality = ProjectReality(role);                 // None | Healthy | Unhealthy
    var pending = _outstanding.TryGet(role, out var p)
                  ? p
                  : null;

    // ── 2. Decide ─────────────────────────────────────────────────────
    var action = Decide(intent, reality, pending);

    // ── 3. Act ────────────────────────────────────────────────────────
    switch (action)
    {
        case ReconcileAction.Wait:
            break;

        case ReconcileAction.IssueCreate:
            var template = await _obligationStore.GetSystemTemplateAsync(role, ct);
            var request  = await _renderer.BuildVmRequestFromTemplateAsync(
                template, vmName: $"{role}-{_nodeId[..8]}",
                environmentVariables: BuildDeployConfig(role));
            var commandId = await _commandQueue.EnqueueCreateAsync(request, ct);
            _outstanding.Set(role, new OutstandingCommand {
                CommandId = commandId, Kind = CommandKind.Create,
                IssuedAt = DateTime.UtcNow });
            break;

        case ReconcileAction.IssueDelete:
            var vmId = LookupExistingVmIdForRole(role);
            var deleteId = await _commandQueue.EnqueueDeleteAsync(vmId, ct);
            _outstanding.Set(role, new OutstandingCommand {
                CommandId = deleteId, Kind = CommandKind.Delete,
                VmId = vmId, IssuedAt = DateTime.UtcNow });
            break;
    }
}

// Sweep expired outstanding commands at end of cycle.
// Next cycle's matrix re-evaluates from current reality.
_outstanding.SweepExpired(CommandTimeout);
```

Two timeouts govern the entire system:

- **`HeartbeatFreshness`** — used by the orchestrator for `obligationHealth`
  staleness. Default: 90 seconds.
- **`CommandTimeout`** — used by the node's `OutstandingCommands` sweeper.
  Default: 20 minutes (covers cloud-init for a slow first boot, including
  artifact downloads from cold cache).

No other system-VM-lifecycle timeouts exist. Cloud-init readiness, active
grace, provisioning timeout, stuck-deleting — all collapse into
`CommandTimeout`. If a command hasn't completed in 20 minutes, it's gone;
next cycle decides afresh.

---


## 6. Security

> **Implementation note (2026-05-01):** Caller-IP-to-role binding (Layer 3,
> review finding 2.1) was added to `ObligationStateController` during
> implementation. See §6.2 for the updated three-layer model.

### 6.1 Artifact integrity chain

```
Publish time (in orchestrator):
  Author submits { name, sourceUrl, sha256 }
  Orchestrator GETs sourceUrl once, computes SHA256 server-side
  If mismatch → reject publish (no metadata stored)
  If match    → store (sourceUrl, sha256) on template, discard bytes
  The hash is now immutable for this template Version.

Node agent prefetch (first miss on a node):
  Fetches from artifact.SourceUrl (author-controlled)
  Streams bytes through SHA256 hasher → rejects on mismatch
  Caches under /var/lib/decloud/artifact-cache/{sha256}

VM boot:
  Reads from node agent cache (http://192.168.122.1:5100)
  Re-verifies ${ARTIFACT_SHA256:name} inside cloud-init
  Refuses to execute if mismatch

Trust root:
  MongoDB (orchestrator) holds the authoritative SHA256 captured at publish.

  Compromised node agent cannot serve a poisoned artifact — the VM verifies
  the hash provided by the orchestrator (via ${ARTIFACT_SHA256:name}).

  Compromised author origin (post-publish swap) cannot poison either —
  node agent prefetch streams through a hasher and fails closed on mismatch.
  Any VM that boots from a compromised cache would also fail the in-VM
  re-check. The pre-publish hash capture binds SourceUrl bytes to that
  specific hash for the lifetime of the template Version.
```

### 6.2 ObligationStateService security

- Endpoint `/api/obligations/{role}/state` binds to `192.168.122.1:5100`
  (virbr0 only).
- virbr0 is not routable externally — only local VMs can reach it.
- Three-layer security model (updated 2026-05-01, review finding 2.1):
  - **Layer 1 — Network boundary:** virbr0 not externally routable.
  - **Layer 2 — Subnet check (all endpoints):** caller IP validated
    against `192.168.122.0/24` on every request.
  - **Layer 3 — Caller-IP-to-role binding (`/state` only):** caller IP
    must match the IP of the system VM assigned to the requested role.
    Prevents a tenant VM on the same virbr0 bridge exfiltrating system VM
    private keys. Three allow-with-warning cases handle first-boot timing
    windows where the DHCP lease is still being acquired.
  `GET /api/obligations/{role}/version` uses Layer 1+2 only — version
  integers carry no key material.
- State contains private keys — log access at DEBUG level, never at INFO.
- SQLite file permissions: `600` (root-owned, no world read).
- State JSON never logged in full (only role + version logged).
- Obligations and system templates are **not** served over this endpoint.
  They're internal node-agent state, consumed only by `SystemVmReconciler`.

### 6.3 Artifact publish security

- `SourceUrl` must be HTTPS (rejected at registration if not).
- Orchestrator HEAD-checks reachability at registration time.
- Orchestrator GETs once at publish time to verify SHA256 — bytes are not
  retained.
- SHA256 is captured server-side at publish; the client-provided hash is
  treated as a hint and always re-verified.
- After publish, the artifact's `(SourceUrl, Sha256)` tuple is immutable
  for that template `Version` — author rotation requires bumping the
  version.
- AI review (Template Review Gate, see `COMPLIANCE.md` §6) inspects
  artifact bytes during the same one-time fetch — covers Binary, Script,
  WebAsset, Config, Archive types uniformly.
- Defense in depth: even if an author swaps bytes at the URL after publish,
  every prefetch fails closed on hash mismatch (no silent compromise).
- No upload path: zip bombs, MIME spoofing, path-traversal in filenames,
  and "served as trusted platform content" attacks are eliminated by
  construction.
- Legal posture: platform is a *directory* not a *host*; takedowns
  de-list rather than remove-from-disk.

### 6.4 Node-authoritative lifecycle security

The shift to node-side lifecycle authority introduces no new attack
surface. The node already owns its libvirt domain; it already executes
cloud-init at root; the orchestrator already trusts the node's heartbeat
to be truthful (otherwise no VM management would work). The matrix is
just a different place to compute the same decisions.

What changes:

- **The orchestrator's scheduler reads node-reported `obligationHealth`.**
  A misbehaving node could lie — claim healthy when it isn't, to keep
  receiving tenant VMs; or claim unhealthy when it is, to dodge work.
  Mitigation: the heartbeat carries both the rolled-up signal *and* the
  underlying per-VM service-readiness data. The orchestrator can verify
  the former from the latter on demand. A divergence is logged and
  treated as suspicious; the scheduler can fall back to the raw data.
- **The node decides its own redeploys.** A misbehaving node could refuse
  to redeploy a system VM. Mitigation: this manifests as
  `obligationHealth: Unhealthy` on the heartbeat (because the local
  `IsFullyReady` check fails), which scheduling sees and acts on by
  filtering the node out of new placements. The mesh health check
  (§5.4) is the cross-node validation: the *other* nodes' system VMs
  notice when this node's DHT/relay/blockstore peer disappears, so the
  problem doesn't stay invisible to the cluster.

### 6.5 Secrets-at-rest trust model

Node operators are trusted with their own node's secrets. The Ed25519
private keys, WireGuard private keys, and auth tokens stored in node SQLite
(`obligation_state` table, mode 0600, root-owned) are accessible to anyone
with root on the node host. This is consistent with the platform's overall
trust model — node operators are permissioned participants, and the risk of
a malicious operator exfiltrating their own node's system VM identity is
accepted.

**What this does not protect against:** node disposal (disk not wiped),
backup theft, or forensic recovery of a decommissioned node.

**Defences that do apply:**
- Cluster-level mesh health cross-validation: peers notice misbehaviour
  (phantom peer IDs, forged routes) and the scheduler filters out the node.
- Identity rotation: the orchestrator can issue a new version with fresh
  keys, which `ObligationStateService` accepts on next heartbeat
  (higher version wins). The old keys are invalidated from the
  network's perspective on first rotation.
- SQLite 0600 permissions prevent non-root processes on the host from
  reading secrets directly.

**Out of scope for Phase 1:** hardware-rooted operator trust (TPM, sealed
storage). If this becomes a requirement, `ObligationStateRepository` is the
single write point — a TPM-sealed keystore can be substituted without
changing the interface.

### 6.6 Relay revenue-share security

- `relayNodeId` sourced from `node.CgnatInfo.AssignedRelayNodeId`
  (orchestrator-assigned, not user-provided).
- Revenue share calculation happens server-side in `BillingService` — no
  relay VM involvement.
- `PaymentConfig.RelayRevenueShareRatio` is a server-side config value
  (cannot be spoofed by relay operator).

---

## 7. Relay revenue-share implementation

### 7.1 Overview

CGNAT node operators pay a small revenue share (configurable, default 2%)
to the relay operator that enables their network participation. No billing
logic lives in the relay VM — it's purely an orchestrator-side calculation.

### 7.2 Configuration

**File:** `src/Orchestrator/Models/Payment/PaymentConfig.cs` *(extended)*

```csharp
// Added to PaymentConfig:

/// <summary>
/// Percentage of CGNAT node revenue transferred to the assigned relay node operator.
/// Applied during billing settlement for nodes with non-null CgnatInfo.AssignedRelayNodeId.
/// Range: 0.0 to 1.0 (default: 0.02 = 2%)
/// </summary>
public decimal RelayRevenueShareRatio { get; set; } = 0.02m;
```

### 7.3 Billing integration

**File:** `src/Orchestrator/Services/Balance/BillingService.cs` *(extended)*

```csharp
// In CalculateNodeEarningsAsync() or equivalent billing settlement method:

if (node.CgnatInfo?.AssignedRelayNodeId is { } relayNodeId
    && _paymentConfig.RelayRevenueShareRatio > 0)
{
    var relayShare = nodeEarnings * _paymentConfig.RelayRevenueShareRatio;
    var cgnatShare = nodeEarnings - relayShare;

    // Record two settlements
    await RecordEarningsAsync(node.Id, cgnatShare);
    await RecordEarningsAsync(relayNodeId, relayShare);

    _logger.LogInformation(
        "Relay revenue share: CGNAT node {CgnatId} earned {CgnatAmount:C}, " +
        "relay node {RelayId} earned {RelayAmount:C} ({Ratio:P0} share)",
        node.Id, cgnatShare, relayNodeId, relayShare,
        _paymentConfig.RelayRevenueShareRatio);
}
else
{
    // Public IP node or no relay — full earnings to this node
    await RecordEarningsAsync(node.Id, nodeEarnings);
}
```

---

## 8. Resilience scenarios

| Failure                              | Recovery                                                                                                                                              |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| System VM crashes                    | Node's `VmReadinessMonitor` flips a service to `Failed` → `IsFullyReady` = false → matrix sees `Unhealthy` → Delete + Create from local template + identity. Same peer ID. Mesh rejoins instantly. |
| System VM disk corrupted             | Same as above. Identity preserved in node SQLite.                                                                                                      |
| Orchestrator unreachable             | Node has everything locally: obligations, identity, templates, artifacts (cached). The matrix continues to operate. New deploys, recovery from crashes, mesh health — all function. |
| Node reboots (power failure)         | Node agent starts → SQLite intact → `SystemVmWatchdogService` restarts stopped system VMs → matrix re-evaluates → any genuinely broken VM is redeployed. Same identity. Orchestrator learns via the next heartbeat. |
| DHT redeployment (software update)   | Template version bump on orchestrator → pushed to nodes → `/health/version` returns 503 → matrix redeploys. New VM boots, reads Ed25519 key from SQLite → same peer ID → other peers reconnect without cascading updates. |
| Relay redeployment                   | WireGuard keypair preserved in SQLite → same public key → all mesh-enrolled DHT/BlockStore VMs reconnect automatically.                                |
| Orchestrator DB lost                 | Node agent's SQLite has authoritative obligations + identity + templates. Orchestrator can reconstruct its own view from heartbeats; nothing on the node needs replacement. |
| Mesh peer becomes unreachable        | The node hosting the peer notices via its own `/health/mesh` (peer count drops, handshake age stale, etc.) → marks service Unhealthy → matrix redeploys. Cross-node validation: a peer's neighbour notices missing connectivity, *its* mesh-health may also flip, triggering its own redeploy if the cause is local. The mesh self-validates. |
| Node-agent restart mid-deploy        | `OutstandingCommands` map is in-memory and is lost. After restart, `LibvirtVmManager.ReconcileAllWithLibvirtAsync` reconciles VM state from libvirt; `SystemVmReconciler` runs a fresh cycle. If the in-flight command had already created the domain, the matrix sees `Healthy` (or `Unhealthy` and acts again). Idempotent at the libvirt layer (duplicate-name guard). |
| Capability loss (e.g. disk shrinks)  | Orchestrator's `IObligationEligibility` re-evaluates and pushes an updated obligation set. The retired role is no longer in the node's `obligation` table. Matrix sees `(intent=no, reality=Healthy, pending=None)` → Delete. Drains naturally. |
| Author URL takedown                  | Already-cached node continues serving from local artifact cache. New nodes fail prefetch with a clear error; matrix retries on each cycle. Block Store CID replication (Phase 2) closes the gap permanently. |

---


## 9. Implementation plan

Eleven phases organised into four meta-stages. Each phase is independently
shippable and independently revertible. Stages I/II (node-side substrate
+ lifecycle) and Stage III (templates) can run partly in parallel — they
touch disjoint files until P10 fuses them.

```
  STAGE I — Substrate (node-side)         STAGE II — Lifecycle (node-side)
  ────────────────────────────────         ────────────────────────────────
  P1  OutstandingCommands map              P5  SystemVmReconciler in shadow mode
  P2  Reality projection                   P6  Cut over to reconciler
  P3  Intent computation                   P7  Delete orchestrator system-VM lifecycle code
  P4  Heartbeat carries obligationHealth

  STAGE III — Templates                   STAGE IV — Convergence
  ─────────────────────                    ───────────────────────
  P8  TemplateArtifact + cache             P10  System templates seeded + node-pushed
  P9  ObligationStateService               P11  Cloud-init folder vanishes
       extended (obligations + templates)
```

The longest dependency chain is **P1 → P5 → P6 → P10 → P11**. Stage III
can ship in parallel with Stages I and II.

### Stage I — Substrate (node-side)

#### P1 — `OutstandingCommands` map ✅

**What:** in-memory map on the node agent keyed by canonical role string,
holding `{ commandId, kind, vmId?, issuedAt }`. Sweeper helper for expiry.

**Why first:** every later phase reads from this. New code, no deletions,
no behaviour change.

**Files delivered:**
- `Core/Models/OutstandingCommand.cs` — `OutstandingCommand` record + `CommandKind` enum
- `Core/Interfaces/SystemVm/IOutstandingCommands.cs` — interface
- `Infrastructure/Services/SystemVm/OutstandingCommands.cs` — `ConcurrentDictionary` implementation; `TimeProvider` injection; compare-and-remove sweep protects sweep-vs-Set race

**Decisions:** keyed by canonical role string (not enum) for consistency with all other
matrix inputs. `TryRemoveIfMatch(commandId)` compare-and-remove prevents a sweep racing
with a new `Set`. In-memory by design — outstanding commands from a previous process are
inherently stale; the matrix re-evaluates from scratch on restart.

**Verification:** unit tests deferred to P11 (user decision: avoid rewriting against
intermediate shapes during rollout).

#### P2 — Reality projection (node) ✅

**What:** function `ProjectReality(role) → None | Healthy | Unhealthy`.
Reads from `IVmManager.GetAllVms()` and per-VM `Services[]`.

**Why now:** the matrix's `reality` axis. Pure read, unit-testable.

**Files delivered:**
- `Core/Models/Reality.cs` — `Reality` enum + `RealitySnapshot` record (State, VmId, VmState for diagnostics); static `RealitySnapshot.None` instance
- `Core/Interfaces/SystemVm/IRealityProjection.cs` — interface
- `Infrastructure/Services/SystemVm/RealityProjection.cs` — synchronous implementation

**Decisions:** synchronous (GetAllVms is in-memory, no wait implied). Three-state only — no
`Degraded`; mesh-health threshold lives inside the system VM's `/health/mesh` endpoint.
`NotFound`/`Deleted` VMs filtered out (tombstones, not present reality). Multi-VM-per-role
defended: prefer Healthy if any, else surface most-recently-created Unhealthy; log Warning
on invariant violation. `VmState` carried on snapshot for diagnostics only — matrix branches
on `Reality.State`.

**Verification:** unit tests deferred to P11.

#### P3 — Intent computation (node) ✅

**What:** function `ComputeIntent(obligation) → (wantDeployed, depsMet)`.
Reads from local obligation SQLite. `IObligationStateService` extended
with `SaveObligationsAsync()` + `GetObligationsAsync()`.

**Why now:** the matrix's `intent` axis.

**Files delivered (new):**
- `Core/Models/Intent.cs` — `Intent` record (`WantDeployed`, `DepsMet`); static `Intent.None`
- `Core/Models/ObligationDescriptor.cs` — `(Role, Deps[], UpdatedAt)`; deps pushed by orchestrator, not hardcoded
- `Core/Interfaces/SystemVm/IIntentComputation.cs` — interface; `Compute(role, allObligations)` synchronous
- `Infrastructure/Services/SystemVm/IntentComputation.cs` — applicable deps (those also in local obligation list); `DepsMet` ⇔ all applicable deps project to `Healthy` via `IRealityProjection`

**Files delivered (modified — drop-in replacements):**
- `Core/Interfaces/State/IObligationStateService.cs` — added `SaveObligationsAsync` + `GetObligationsAsync`; identity methods unchanged
- `Infrastructure/Services/State/ObligationStateService.cs` — implements new methods; role/dep canonicalisation + Warning logs before SQL
- `Infrastructure/Persistence/ObligationStateRepository.cs` — `obligation` table added to `InitialiseSchema` (idempotent `CREATE TABLE IF NOT EXISTS`); `ReplaceObligationsAsync` (transactional wipe-and-insert); `GetObligationsAsync`; `JsonSerializerOptions` field for deps serialisation

**Schema addition:**
```sql
CREATE TABLE IF NOT EXISTS obligation (
    role        TEXT PRIMARY KEY,   -- "relay" | "dht" | "blockstore"
    deps_json   TEXT NOT NULL,      -- JSON array of dep role names
    updated_at  TEXT NOT NULL
);
```

**Decisions:** one `IObligationStateService`, three concerns (identity now; obligations now;
system templates in P9). Whole-set obligation replacement — avoids drift from missed deletes.
Dependencies pushed with obligations rather than hardcoded on node. `Intent` split into
`(WantDeployed, DepsMet)` for diagnostic clarity; matrix's `Decide` treats them together.
`IntentComputation` is sync, takes obligation list as parameter (matrix fetches once per
cycle, passes it — no per-call SQLite hit in the compute path).

**Verification:** unit tests deferred to P11.

#### P4 — Heartbeat carries `obligationHealth` ✅

**What:** `HeartbeatService` adds `ObligationHealth: Dictionary<string, string>`
computed from `IRealityProjection` per obligation in the local SQLite table.
Field is included in the heartbeat wire payload. **Wire-format only in P4** —
the orchestrator deserialises but does not consume the field until P6.

**Why now:** the field is populated and flowing before anything depends on it.
P6's consumption logic has a fully-populated signal waiting.

**Files delivered (integration guide — surgical edits to large existing files):**
- `Orchestrator/Models/Node.cs` — `NodeHeartbeat` record: `Dictionary<string, string>? ObligationHealth = null` parameter added. **No new property on `Node` class** — `Node.SystemVmObligations` remains the single source of truth for per-role health on the orchestrator side
- `NodeAgent.Core/Models/NodeModels.cs` — `Heartbeat` domain class: `Dictionary<string, string>? ObligationHealth` field added
- `NodeAgent/Services/HeartbeatService.cs` — `IRealityProjection` + `IObligationStateService` injected; `BuildObligationHealthAsync` helper iterates obligations + projects each; wired into `Heartbeat` initialiser
- `NodeAgent/Services/OrchestratorClient.cs` — `obligationHealth = heartbeat.ObligationHealth` appended to anonymous-object payload in `BuildHeartbeatPayload` (pure serialisation, no new dependency)
- `NodeAgent/Program.cs` — `IRealityProjection → RealityProjection` registered as singleton (P2 shipped the implementation without DI registration)

**Key architectural decision:** a parallel `Node.ObligationHealth` property was
considered and rejected. The orchestrator already maintains `SystemVmObligations[role].Status`
as the per-role health signal. Adding a second field would create two writers
(`SystemVmReconciliationService` and the heartbeat handler) that fight while both
are active. P6 applies `obligationHealth` to `SystemVmObligation.Status` updates
*as part of* disabling the orchestrator-side reconciler — one authoritative writer
at a time.

**Behaviour change:** informational only. Scheduler does not yet filter on this
signal. `SystemVmReconciliationService` continues to drive deployment unaffected.

**Verification:** smoke test — insert a test row into local `obligation` SQLite,
observe `obligationHealth` in heartbeat payload. Orchestrator deserialises without
error; `Node` record and `SystemVmObligations` unchanged.

---

> **Stage I complete.** P1–P4 are shipped. All three matrix input axes
> (outstanding commands, reality, intent) are built and independently
> exercisable. The heartbeat carries the rolled-up health signal.
> Nothing acts yet — Stage II wires them into decisions.

### Stage II — Lifecycle (node-side)

#### P5 — `SystemVmReconciler` in shadow mode ✅

**What:** new `BackgroundService` on the node agent. Runs the matrix every
30 s. For each obligation, computes `(intent, reality, pending) → action`
and **logs** the decision. Does not act.

**Why now:** P1–P3 give all three inputs; P4 confirms the heartbeat
carries the rolled-up signal. Shadow mode catches discrepancies before
they affect production.

**Touches:** new `Services/SystemVm/SystemVmReconciler.cs`.

**Behaviour change:** none — actions are not taken.

**Verification:** run for at least one week on a staging cluster. For each
`(role, cycle)`, log `{node_decision, orchestrator_decision}`. Discrepancies
are bugs in `ProjectReality`, `ComputeIntent`, or `Decide` — fix until they
agree. The categories of discrepancy that *can* legitimately remain (e.g.,
orchestrator decides based on cluster-level state the node can't see) are
documented and explicitly accepted.

#### P6 — Cut over to the node-side reconciler ✅

**What:** the node's reconciler becomes authoritative. When the matrix
decides Create, the node enqueues a `CreateVm` command into its own command
pipeline. When the matrix decides Delete, likewise. The orchestrator's
`SystemVmReconciliationService` is **disabled** (DI registration commented
out) but not yet deleted.

**Why now:** P5 has proven the matrix is right.

**Touches:** `SystemVmReconciler.ExecuteAsync` — replace logging with
command issuance. Orchestrator `Program.cs` — disable
`SystemVmReconciliationService` DI registration.

**Behaviour change:** behaviour identical to P5's logged predictions. If
the shadow soak was clean, this is observably a no-op.

**Verification:** repeat the staging soak. Add chaos: kill a system VM,
push an artifact version bump (after Stage III), simulate a stale ack,
restart the node agent, restart the orchestrator. Each scenario self-heals
via the matrix.

#### P7 — Delete orchestrator system-VM lifecycle code ✅

**What:** delete `SystemVmReconciliationService` and all surfaces in §3.10
"Orchestrator". Remove `obligation.Status` etc. from MongoDB writes.
Remove the heartbeat-path Active promotion. Remove ghost detection's
system-VM branch.

**Why now:** P6 has been running for a week without the orchestrator loop
contributing decisions. The code is dead.

**Touches:** see §3.10 for the full list. Actual P7 scope was narrower
than originally planned — `SystemVmReconciliationService` was not deleted
as a file but renamed to `SystemVmObligationService` and stripped of
all deployment methods. The background loop retained `EnsureObligationsAsync`
for obligation backfilling, which is orchestrator authority (not node
lifecycle authority). The five dead timeout constants were also removed.

**Behaviour change:** none.

**Verification:** orchestrator loop simpler; deploy methods no longer
callable; heartbeat Active promotion path gone — one authoritative writer
per obligation status field.

### Stage III — Templates

#### P8 — `TemplateArtifact` model + node-agent artifact cache ✅

**What:** the existing resilience design's Phase A items 1–6, unchanged.

**Touches:** `TemplateArtifact` model, `VmTemplate.Artifacts[]`,
`MarketplaceController` extensions, `TemplateService.PublishTemplateAsync`
SHA256 verification, `ResolveArtifactVariables`, node agent
`ArtifactCacheService` + `ArtifactCacheController`.

**Behaviour change:** authors can attach external artifacts to templates.
Tenant templates can use them immediately. System VMs are still on the
bundled-folder path until P10/P11.

**Implementation addition:** two source schemes were implemented rather
than one. HTTPS (original design) for externally-hosted artifacts. `data:`
URI (added during P8) for inline artifacts — author provides RFC 2397
base64-encoded bytes stored inside the template MongoDB document (same
legal character as cloud-init YAML). Binary artifacts always use HTTPS
(size, versioning, architecture-specificity). Scripts and dashboards use
inline (immutable, small, no external dependency). Per-artifact inline
limit: 5 MB. Per-template inline budget: 10 MB.

**Verification:** register external artifact → publish template → deploy
tenant VM → verify SHA256 chain end-to-end.

#### P9 — `ObligationStateService` extended (obligations + templates) ✅

**What:** the existing resilience design's Phase B items 7–14, *plus*:

- `obligation` SQLite table populated from registration response
- `system_template` SQLite table populated from registration response
- Heartbeat-driven version-pull mechanism extended to cover templates

The schema for both new tables mirrors the existing `obligation_state`
table (role-keyed, versioned, JSON-payloaded). The
`NodeRegistrationResponse` and the heartbeat response gain a
`systemTemplates: { role → { version, ... } }` block parallel to the
existing obligation-state version reporting.

**Why combined:** the channel already exists; adding payloads is a small
extension. Keeping templates on a separate channel would duplicate the
versioning, push, and SQLite plumbing.

**Touches:** `IObligationStateService` (new methods),
`ObligationStateRepository` (new tables + CRUD),
`NodeRegistrationResponse` (extended),
heartbeat-response handler on the node side (process template updates),
`ObligationStateGenerator` on orchestrator (pick current templates per
role and include them in the payload).

**Interface stability contract (P3 → P9):** Between P3 and P9,
`IObligationStateService` is extended by **additions only** — no method
renames, no signature changes to existing methods. P3 shipped
`SaveObligationsAsync` and `GetObligationsAsync`. P9 adds
`SaveSystemTemplateAsync`, `GetSystemTemplateJsonAsync`,
`GetSystemTemplateRevisionAsync`. Any code consuming the interface at P3
remains valid at P9 without recompilation. This contract is closed at P9
— the interface is now stable.

**Behaviour change:** node SQLite holds obligations and current system
templates. Nothing reads the templates yet — `SystemVmReconciler` from P6
still deploys via the legacy bundled path. Templates are present but inert.

**Verification:** register a node, verify identity + obligations + templates
all land in SQLite. Bump a template version on the orchestrator; verify
the next heartbeat triggers a re-pull. Stop the orchestrator, restart the
node agent — everything still present from local SQLite.

### Stage IV — Convergence

#### P10 — System templates seeded and node-driven deploys use them ✅

**What:** orchestrator's `TemplateSeederService` seeds the three system
templates with full cloud-init (migrated from the bundled YAML files) and
`Artifacts[]` (referencing the binaries / dashboards / scripts at their
new URLs). The node-side `SystemVmReconciler` Create branch is updated:
instead of calling the legacy bundled path, it reads the system template
from local SQLite, builds a `CreateVmRequest` via the same template-rendering
logic tenant VMs use, and enqueues it into the local command pipeline.

**Why now:** templates exist locally (P9). Artifact cache is operational
(P8). Reconciler is authoritative (P6).

**Touches:**
- `SystemVmTemplateSeeder` (new) — seeds `system-relay`, `system-dht`,
  `system-blockstore` templates into MongoDB. Revision-aware (skips if
  stored revision ≥ seeder revision). Binary artifacts reference GitHub
  Releases in `DeCloud.Builds` (binaries/v1.0.0). Script/dashboard
  artifacts are inline `data:` URI constants generated by
  `compute-artifact-constants.sh`. Cloud-init YAML constants embedded as
  C# verbatim string literals.
- `SystemVmObligation.TemplateId` (new field) — stable MongoDB `_id`
  of the assigned template. Stamped at obligation creation. The obligation
  channel carries `TemplateId`, not slug. Slugs are display-only.
- `DeCloud.Builds` restructured: `system-vms/{role}/src/` (Go source),
  `system-vms/{role}/assets/` (scripts/dashboards), `system-vms/{role}/
  cloud-init.yaml`. `compute-artifact-constants.sh` auto-discovers all
  `assets/` directories.
- node-side `SystemVmReconciler` Create branch reads system template from
  SQLite, substitutes `${ARTIFACT_URL:name}` variables, builds `VmSpec`,
  calls `IVmManager.CreateVmAsync` directly.

**Behaviour change:** new system VM deploys go through the template path.
**Existing running system VMs are not redeployed** — the matrix sees them
as `Healthy` and takes no action. Their identity is preserved in
`ObligationStateService` and survives any future redeploy. The migration
is silent for running VMs and template-driven for the next deploy.

**Verification:** induce a system VM crash (`virsh destroy`). Matrix detects
`Unhealthy`, dispatches Delete, then Create. Template renders, identity
restored from local SQLite, peer ID unchanged, mesh reconnects without
configuration drift on neighbouring VMs.

#### P11 — Cloud-init folder vanishes

**What:** delete `src/DeCloud.NodeAgent/CloudInit/Templates/` and every
dependent code path that referenced it. The deletions are interlocking
(see §3.10 for the full list); they ship together.

**Why last:** every prior phase contributes a piece of the replacement.
P8 provides the artifact cache that replaces bundled binaries. P9 provides
the template channel that replaces bundled YAML. P10 wires them together.
P11 only removes what nothing references anymore.

**Touches:** see §3.10 "Node agent" for the exhaustive list.

**Scope clarification:** `CloudInit/Templates/` was not deleted entirely.
The folder retains `general/general-vm-cloudinit.yaml` and the attestation
agent b64 files (`decloud-agent-amd64.b64`, `decloud-agent-arm64.b64`)
which are tenant VM infrastructure, not system VM infrastructure. Only the
`dht-vm/`, `blockstore-vm/`, and `relay-vm/` subdirectories were removed.

`CloudInitTemplateService` was narrowed — the eight system VM methods were
deleted but the service itself retained for General and Inference tenant VMs.
`GetTargetArchitecture` now delegates to
`ResourceDiscoveryService.GetArchitectureNormalised()` — single source of
truth for host architecture detection across the entire node agent.

**Behaviour change:** none. The deletion is observable as a smaller install
footprint, faster installs (no Go toolchain required), and the disappearance
of `~/.cache/go-build`-style artefacts from node disks.

**Verification:**
- Fresh install on a clean Ubuntu host completes without `go` installed.
- New node registers, receives obligations + identity + templates, deploys
  all three system VMs from artifact cache — no path through deleted code.
- Existing nodes upgraded in place: running system VMs continue running;
  redeploys triggered by chaos testing render from templates.

### Effort shape

```
                  | P1 P2 P3 P4 | P5 P6 P7 | P8  P9  P10 | P11
  Code volume     |  S  S  S  S |  M  S  L |  L   M   M  |  L (mostly deletions)
  Risk            |  L  L  L  L |  L  M  L |  M   L   L  |  L
  Reverts cleanly?|  Y  Y  Y  Y |  Y  Y  Y |  Y   Y   Y  |  Y
  Independent?    |  Y  Y  Y  Y |  Y  N* N*|  Y   N†  N‡ |  N§
                  |             |          |             |
  *  P5/P6/P7 ship in order.   M = medium, L = large, S = small
  †  P9 depends on P8 (artifact references in templates need the artifact model)
  ‡  P10 depends on P6 (matrix authoritative) + P9 (templates pushed)
  §  P11 depends on P10 (template path proven for system VMs)
```

### Risks and mitigations

| Risk                                                                                       | Mitigation                                                                                          |
|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Matrix shadow run hides a category of mismatch we don't notice                              | P5 runs ≥1 week, every WARNING-level discrepancy is triaged                                          |
| Network partition: node decides to redeploy but orchestrator can't be told                  | Orchestrator doesn't need to be told. Node redeploys locally. When connectivity returns, heartbeat reports the new state. |
| Orchestrator DB lost                                                                       | Each node holds authoritative obligations + identity + templates in local SQLite. Orchestrator can rebuild its own view from heartbeats. |
| System VM template version mismatch after orchestrator restart                             | `system_template.version` in node SQLite is the source of truth on the node. Orchestrator fetches versions on next heartbeat and pushes if higher. |
| Author URL takedown breaks new node bring-up                                               | Cached nodes keep serving. New nodes fail-closed with clear error. Block Store CID replication is the eventual fix. |
| Refactor introduces a regression we don't catch in shadow mode                             | Each phase independently revertible. P1–P3 additive. P4 adds a heartbeat field. P5 doesn't act. P6 is one switch. P7–P11 delete only proven-unreachable code. |
| System template seeding races with running clusters                                        | P10 only seeds the templates; running VMs are not affected. Reverting is `db.templates.deleteMany({authorId: "system"})`. |
| Node-agent restart loses outstanding-command state                                         | By design. Outstanding commands are per-cycle ephemera. After restart, libvirt reconciliation reports current VM state, the matrix re-evaluates from scratch, and any in-flight command is treated as timed out. Idempotent at the libvirt layer (duplicate-name guard). |

---


## 10. File change summary

### Orchestrator — new files

```
src/Orchestrator/Models/TemplateArtifact.cs
src/Orchestrator/Services/SystemVm/ObligationStateGenerator.cs
```

### Orchestrator — modified files

```
src/Orchestrator/Models/VmTemplate.cs              (add Artifacts[], Version, Slug;
                                                     Services[] declarations expanded
                                                     to include mesh + version checks)
src/Orchestrator/Models/Payment/PaymentConfig.cs    (add RelayRevenueShareRatio)
src/Orchestrator/Models/SystemVmObligation.cs       (add StateJson, StateVersion;
                                                     drop Status / DeployedAt / ActiveAt /
                                                     LastError / FailureCount as persisted
                                                     decision fields)
src/Orchestrator/Models/Node.cs                     (add ObligationHealth dictionary)
src/Orchestrator/Controllers/MarketplaceController.cs
                                                    (artifact registration + probe endpoints)
src/Orchestrator/Services/TemplateService.cs        (ResolveArtifactVariables;
                                                     publish-time SHA256 verification)
src/Orchestrator/Services/TemplateSeederService.cs  (seed system templates with artifacts)
src/Orchestrator/Services/NodeService.cs            (deliver obligations + identity +
                                                     templates at registration; heartbeat
                                                     handler simplified — stamps
                                                     ObligationHealth, drops Active
                                                     promotion, drops system-VM ghost
                                                     detection)
src/Orchestrator/Services/Balance/BillingService.cs (relay revenue share)
src/Orchestrator/Program.cs                         (remove SystemVmReconciliationService
                                                     DI registration)
```

### Orchestrator — deleted files

```
src/Orchestrator/Services/SystemVm/SystemVmReconciliationService.cs   (entire file, ~1500 lines)
```

### Orchestrator — deleted logic (within retained files)

```
RelayNodeService.DeployRelayVmAsync()         → replaced by node-side template path
DhtNodeService.DeployDhtVmAsync()             → replaced by node-side template path
BlockStoreService.DeployBlockStoreVmAsync()   → replaced by node-side template path

NodeService heartbeat handler:
  - DELETE: heartbeat-path Active promotion block
  - DELETE: system-VM ghost detection branch (tenant branch survives)
  - DELETE: rogue UpdatedAt write
```

### Node agent — new files

```
src/DeCloud.NodeAgent/Controllers/ArtifactCacheController.cs
src/DeCloud.NodeAgent/Controllers/ObligationStateController.cs
  (updated post-implementation: Layer 3 caller-IP binding added)

src/Orchestrator/Services/SystemVm/SystemVmTemplateSeeder.cs  (new — P10)
  Seeds system-relay, system-dht, system-blockstore into MongoDB.
  Revision-aware. Inline data: URI + HTTPS artifact constants.

DeCloud.Builds/system-vms/compute-artifact-constants.sh  (new — P10)
  Auto-discovers all assets/ directories, generates C# constants.
  No hardcoded role maps.

DeCloud.Builds/.github/workflows/release-binaries.yml  (updated — P10)
  Paths updated from {role}-node-src/ to {role}/src/.
src/DeCloud.NodeAgent.Core/Models/ObligationState.cs
src/DeCloud.NodeAgent.Core/Interfaces/IObligationStateService.cs
src/DeCloud.NodeAgent.Core/Interfaces/IArtifactCacheService.cs
src/DeCloud.NodeAgent.Core/Interfaces/IOutstandingCommands.cs
src/DeCloud.NodeAgent.Infrastructure/Services/ArtifactCacheService.cs
src/DeCloud.NodeAgent.Infrastructure/Services/ObligationStateService.cs
src/DeCloud.NodeAgent.Infrastructure/Persistence/ObligationStateRepository.cs
src/DeCloud.NodeAgent/Services/SystemVm/SystemVmReconciler.cs
src/DeCloud.NodeAgent/Services/SystemVm/RealityProjection.cs
src/DeCloud.NodeAgent/Services/SystemVm/IntentComputation.cs
src/DeCloud.NodeAgent/Services/SystemVm/OutstandingCommands.cs
src/DeCloud.NodeAgent/Services/SystemVm/Decide.cs
```

### Node agent — deleted files

```
src/DeCloud.NodeAgent/Services/GoBinaryBuildStartupService.cs
src/DeCloud.NodeAgent/CloudInit/Templates/                   (entire folder)
```

### Node agent — modified files

```
src/DeCloud.NodeAgent.Infrastructure/Services/CloudInitTemplateService.cs
                                                    (remove Inject* / Load* / Build*;
                                                     keep substitution engine —
                                                     operates on payload-delivered
                                                     UserData string)
src/DeCloud.NodeAgent.Infrastructure/Persistence/VmRepository.cs
                                                    (add migrations for obligation_state,
                                                     obligation, system_template tables)
src/DeCloud.NodeAgent/Services/HeartbeatService.cs  (build ObligationHealth field)
src/DeCloud.NodeAgent/Services/OrchestratorClient.cs
                                                    (process system-template version-pull
                                                     in heartbeat response, parallel to
                                                     existing obligation-state pull)
src/DeCloud.NodeAgent/Program.cs                    (register new services)
install.sh                                          (remove Go toolchain install;
                                                     remove dht-node / blockstore-node
                                                     build invocations; remove .sha256
                                                     marker handling)
```

### Repository asset relocation (not deleted, moved)

```
src/DeCloud.NodeAgent/CloudInit/Templates/dht-vm/dht-node-src/
    → assets/dht/dht-node-src/                      (built by CI, uploaded to Releases)

src/DeCloud.NodeAgent/CloudInit/Templates/blockstore-vm/blockstore-node-src/
    → assets/blockstore/blockstore-node-src/         (same)

src/DeCloud.NodeAgent/CloudInit/Templates/relay-vm/dashboard.*
    → assets/relay/                                 (uploaded as tarball artifact)

src/DeCloud.NodeAgent/CloudInit/Templates/dht-vm/dashboard.*
    → assets/dht/                                   (same)

src/DeCloud.NodeAgent/CloudInit/Templates/blockstore-vm/dashboard.*
    → assets/blockstore/                            (same)

Cloud-init YAML files → embedded as VmTemplate.CloudInitTemplate strings,
                         seeded by TemplateSeederService at orchestrator startup.
```

---

## 11. Success criteria

### Identity preservation

| Criterion                                              | Verification                                                                                |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
| System VM identity survives redeployment               | Redeploy DHT VM → verify same peer ID in DHT network                                        |
| Relay redeployment does not cascade                    | Redeploy relay → DHT/BlockStore VMs reconnect without reconfiguration                        |
| Identity survives orchestrator outage                  | Stop orchestrator → kill DHT VM on a node → node redeploys from local SQLite with same peer ID |

### Artifact integrity

| Criterion                                              | Verification                                                                                |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Artifacts served at ~1 Gbps locally                     | `iperf3` or `curl` timing from VM to node agent over virbr0                                  |
| SHA256 verified end-to-end                             | Modify hash on template doc → node agent prefetch fails closed; in-VM re-check fails closed |
| Author URL takedown resilience                         | Already-cached node continues serving; new node prefetch fails with clear error             |

### Unified template pipeline

| Criterion                                              | Verification                                                                                |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
| System VMs deployable from marketplace UI              | User deploys "DHT Peer Node" from marketplace as a tenant VM — succeeds                      |
| Unified deployment path                                | `RelayNodeService.DeployRelayVmAsync()` deletion does not break relay deployment             |
| No regression on user VM deployment                    | Existing community template deployments unchanged                                            |
| Cloud-init folder is fully gone                        | Fresh install on clean Ubuntu host completes without `go` installed; no references remain    |

### Node-authoritative lifecycle

| Criterion                                              | Verification                                                                                |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Matrix is the only system-VM decision-maker             | Search reveals zero remaining call sites of `SystemVmReconciliationService`                  |
| Self-healing under orchestrator outage                  | Stop orchestrator → induce system VM crash → node redeploys without orchestrator contact     |
| Mesh-health-driven redeploy                            | Block DHT peer connectivity → `/health/mesh` flips to 503 → matrix detects Unhealthy and redeploys |
| Binary-version-driven redeploy                         | Bump system template artifact SHA256 → matrix detects mismatch via `/health/version` → redeploys with new artifact |
| Cluster-level scheduling responds to obligation health | Force a node's BlockStore obligation to Unhealthy → orchestrator stops scheduling new tenant VMs onto that node after `scheduling_grace_window` |
| Node reboot recovery                                   | Reboot a node → `SystemVmWatchdogService` restarts stopped system VMs → matrix re-evaluates → no spurious redeploys for healthy VMs |

### Revenue share

| Criterion                                              | Verification                                                                                |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Relay revenue share calculated correctly               | `BillingService` unit test: CGNAT node $100 → relay gets $2, CGNAT gets $98                  |

---

*End of Design Document*