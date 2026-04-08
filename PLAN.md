# Block Store System VM — Design & Implementation Plan

**Date:** 2026-02-16 (Updated: 2026-03-20)
**Status:** Phase A–C complete (production-verified 2026-03-20). Phase D next.

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| A — Orchestrator core | ✅ Complete | BlockStoreService, BlockStoreController, labels, eligibility |
| B — NodeAgent + Go binary | ✅ Complete | cloud-init, blockstore-node, scripts, callback controller |
| C — Integration | ✅ Production-verified 2026-03-20 | Dashboard live, binary build pipeline stable |
| D — Lazysync & Migration | 🔲 Next | Items 25–33 below |
| D-fixes — Replication reliability | ✅ Production-verified 2026-04-08 | Overlay-only fix, bitswap peer targeting, concurrency cap, port firewall, retry queue |
| E — Split-Brain Prevention | 🔲 Follows D | Items 34–41 below |
**Depends on:** DHT system VMs (production-verified 2026-02-15)
**Follows patterns from:** Relay VMs, DHT VMs

---

## 1. Vision & Purpose

The Block Store system VM creates the **distributed storage backbone** for the DeCloud network. Every node with ≥100 GB storage contributes **5% of its total storage** as a network duty — forming a collective, content-addressed storage medium across the platform.

The primary purpose is **VM resilience and live migration**. When a node goes offline, its VMs can be rescheduled to another node because VM disk state is continuously replicated across the block store network via **lazysync** — a background process that streams dirty disk chunks to the network as they change. No single node failure causes data loss.

Each Block Store VM:

1. **Stores replicated blocks** — Holds content-addressed chunks of OTHER nodes' VM disk state
2. **Announces blocks** — Publishes provider records to the DHT ("I have block X")
3. **Transfers blocks** — Serves blocks to other nodes via libp2p bitswap protocol
4. **Pulls blocks autonomously** — Nodes close to a block's CID in Kademlia XOR space pull and store it via bitswap, with no orchestrator direction
5. **Garbage collects** — Local LRU eviction within the 5% budget; orchestrator audits provider counts

This enables continuous VM disk replication, live migration on node failure, template image distribution, and eventually a full decentralized filesystem — all without centralized cloud storage. Replication is fully decentralized via Kademlia's natural scatter properties; the orchestrator acts as an auditor, not a coordinator.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Orchestrator                                    │
│  ┌──────────────────────┐  ┌───────────────────────────────────┐ │
│  │ BlockStoreService     │  │ BlockStoreController              │ │
│  │ - DeployBlockStoreVm  │  │ POST /api/blockstore/join         │ │
│  │ - ScheduleMigration() │  │ POST /api/blockstore/announce     │ │
│  │ - AuditReplication()  │  │ GET  /api/blockstore/locate/{cid} │ │
│  └──────────────────────┘  │ POST /api/blockstore/manifest      │ │
│                              │ GET  /api/blockstore/audit/{vmId}  │ │
│  ┌──────────────────────┐  │ GET  /api/blockstore/stats         │ │
│  │ LazysyncManager       │  └───────────────────────────────────┘ │
│  │ - Manifest versioning  │                                        │
│  │ - Version audit        │  Node.BlockStoreInfo:                 │
│  │ - Provider count check │  { VmId, PeerId, Capacity,           │
│  └──────────────────────┘    Used, Status }                      │
└───────────────────┬──────────────────────────────────────────────┘
                    │ CreateVm command (labels)
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Node Agent                                  │
│  ┌─────────────────────────┐  ┌────────────────────────────────┐ │
│  │ CommandProcessorService  │  │ BlockStoreCallbackController   │ │
│  │ - Renders cloud-init     │  │ POST /api/blockstore/ready     │ │
│  │ - Substitutes labels     │  │ (backup registration)          │ │
│  └─────────────────────────┘  └────────────────────────────────┘ │
└───────────────────┬──────────────────────────────────────────────┘
                    │ VM boots
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│              Block Store VM (Debian 12 minimal)                    │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  blockstore-node (Go binary)                                  │ │
│  │  - libp2p host (reuses DHT identity pattern)                  │ │
│  │  - Bitswap protocol for block exchange                        │ │
│  │  - Content-addressed FlatFS block storage                     │ │
│  │  - DAG/manifest support (Merkle DAGs over block collections)  │ │
│  │  - HTTP API on 127.0.0.1:5090                                 │ │
│  │    GET  /health                                                │ │
│  │    POST /blocks         (put block → returns CID)              │ │
│  │    GET  /blocks/{cid}   (get block)                            │ │
│  │    DELETE /blocks/{cid} (delete block)                         │ │
│  │    GET  /blocks         (list local blocks)                    │ │
│  │    POST /dag            (put manifest + blocks atomically)     │ │
│  │    GET  /dag/{cid}      (resolve DAG, return manifest)         │ │
│  │    GET  /stats          (storage usage stats)                  │ │
│  │    POST /gc             (run garbage collection, LRU eviction) │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐   │
│  │ WireGuard     │  │ Bootstrap Poll   │  │ Dashboard         │   │
│  │ wg-mesh       │  │ (orchestrator    │  │ (Python, port     │   │
│  │ (same mesh    │  │  peer discovery) │  │  8080 → Nginx 80) │   │
│  │  as DHT VM)   │  │                  │  │                   │   │
│  └──────────────┘  └──────────────────┘  └───────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### How It Connects to the DHT

The Block Store VM does **not** run its own DHT. Instead, it:

1. **Connects to the co-located DHT VM** via the WireGuard mesh
2. Uses **libp2p bitswap** for block exchange between block store peers
3. Uses the **DHT's provider records** to announce "I have CID X" (via the DHT's Kademlia routing)
4. Discovers other block store peers through the DHT network
5. **Subscribes to GossipSub** topic `decloud/blockstore/new-blocks` for near-instant notification of new blocks
6. **Pulls blocks autonomously** — evaluates XOR distance to announced CIDs, pulls via bitswap if close enough (adaptive threshold based on local capacity)
7. **Queries DHT proximity** — uses `GET /proximity/{cid}` on the co-located DHT VM for neighborhood scans and diagnostics

This keeps the DHT VM lightweight (just routing + discovery) while the Block Store VM handles heavy storage I/O. The DHT's Kademlia routing naturally scatters blocks across the network — no central coordinator decides which node stores which block.

---

## 3. System VM Lifecycle (Following Existing Patterns)

### 3.1 Obligation & Eligibility — 5% Storage Duty

Every node with sufficient resources has a **storage duty** — a mandatory contribution to the network's distributed block store. This is the same obligation pattern as relay and DHT duties.

Already stubbed in `ObligationEligibility.cs`:

```
Eligible if:
  - Total storage >= 100 GB (MinBlockStoreStorage)
  - Total RAM >= 2 GB (MinBlockStoreRam)
  - DHT obligation Active (dependency in SystemVmDependencies)
```

Storage allocated to the Block Store VM: **5% of the node's total storage**. This is a fixed duty — not configurable per node. The 5% forms a baseline storage medium across the entire platform.

```
Examples:
  Node with 100 GB storage  → 5 GB block store duty
  Node with 1 TB storage    → 50 GB block store duty
  Node with 4 TB storage    → 200 GB block store duty

  Network of 1,000 nodes averaging 1 TB each → 50 TB aggregate block store
```

The block store on each node holds **replicated overlay chunks from OTHER nodes' VMs** — not the node's own data. A node's own VM overlay blocks are pushed to its local block store by lazysync, then scatter across the network via Kademlia's natural propagation.

### 3.2 Deployment Flow

```
1. Orchestrator: ObligationEligibility computes BlockStore obligation
2. Reconciliation loop: TryDeployAsync checks DHT dependency is Active
3. BlockStoreService.DeployBlockStoreVmAsync():
   a. Calculate storage allocation: 5% of node.TotalStorage
   b. Resolve WireGuard mesh labels (same as DHT)
   c. Generate auth token (32-byte random)
   d. Collect bootstrap block store peers
   e. Create VM with VmType.BlockStore + labels
   f. Store BlockStoreInfo on node
4. NodeAgent: Receives CreateVm, renders blockstore-vm-cloudinit.yaml
5. VM boots: WireGuard mesh enrollment, binary starts, calls /api/blockstore/join
6. Orchestrator: Registers PeerId, returns bootstrap peers
```

### 3.3 Labels (Orchestrator → NodeAgent → VM)

```
role                    = "blockstore"
blockstore-listen-port  = "5001"          # libp2p bitswap port
blockstore-api-port     = "5090"          # localhost HTTP API
blockstore-storage-bytes = "53687091200"  # 5% of 1 TB node = ~50 GB
blockstore-auth-token   = "<base64>"      # HMAC secret
blockstore-advertise-ip = "10.20.1.202"   # WireGuard tunnel IP
node-region             = "us-east"
node-id                 = "node-abc123"
architecture            = "x86_64"
# WireGuard labels (same as DHT):
wg-relay-endpoint       = "..."
wg-relay-pubkey         = "..."
wg-tunnel-ip            = "..."
wg-relay-api            = "..."
```

### 3.4 Authentication (Same Pattern as DHT)

```
Orchestrator → VM:  auth token via labels → cloud-init
VM → Orchestrator:  HMAC-SHA256(authToken, nodeId:vmId) via X-BlockStore-Token header
VM → NodeAgent:     HMAC-SHA256(machineId, vmId:peerId) via X-BlockStore-Token header
```

---

## 4. Changes Required

### 4.1 Orchestrator Changes

#### NEW: `src/Orchestrator/Models/BlockStoreVmSpec.cs`

```csharp
public static class BlockStoreVmSpec
{
    /// <summary>
    /// Block Store node — lightweight VM for the 5% storage duty.
    /// libp2p + bitswap + FlatFS uses ~150-250 MB RAM at steady state.
    /// Disk is the primary resource — fixed at 5% of node's total storage.
    /// </summary>
    public const double StorageDutyFraction = 0.05;  // 5% of total node storage

    public static VmSpec Create(long nodeStorageTotalBytes) => new()
    {
        VmType = VmType.BlockStore,
        VirtualCpuCores = 1,
        MemoryBytes = 512L * 1024 * 1024,                             // 512 MB
        DiskBytes = (long)(nodeStorageTotalBytes * StorageDutyFraction), // 5% duty
        QualityTier = QualityTier.Burstable,
        ImageId = "debian-12-blockstore",
        ComputePointCost = 1,
    };

    public const long MinNodeStorageBytes = 100L * 1024 * 1024 * 1024;  // 100 GB eligibility threshold
}
```

#### NEW: `src/Orchestrator/Services/BlockStoreService.cs`

Interface & service following `IDhtNodeService` / `DhtNodeService` pattern:

```csharp
public interface IBlockStoreService
{
    // Deployment
    Task<string?> DeployBlockStoreVmAsync(Node node, IVmService vmService, CancellationToken ct = default);
    Task<List<string>> GetBootstrapPeersAsync(string? excludeNodeId = null);

    // Lazysync manifest lifecycle
    Task<ManifestRecord> RegisterManifestAsync(string vmId, string rootCid, int version,
        List<string> changedBlockCids, long totalBytes, CancellationToken ct = default);

    // Replication audit (not coordination — the DHT handles replication)
    Task<ReplicationAudit> AuditManifestReplicationAsync(string vmId, int version, CancellationToken ct = default);

    // Migration support
    Task<MigrationPlan> PlanMigrationAsync(string vmId, List<string> candidateNodeIds, CancellationToken ct = default);

    // Stats
    Task<BlockStoreStats> GetNetworkStatsAsync();
}
```

Key responsibilities:
- Calculate storage allocation: 5% of node's total storage
- Resolve WireGuard mesh labels (reuse DhtNodeService pattern)
- Generate auth token + labels, deploy VM via VmService
- Store `BlockStoreInfo` on Node model
- **Manifest lifecycle**: register evolving manifests from lazysync, track confirmed vs current version
- **Replication audit**: query the DHT for provider counts per chunk CID; advance `confirmedVersion` when all chunks in a manifest version have ≥N providers. The orchestrator does not direct replication — Kademlia handles scatter natively
- **Migration planning**: given a VM to migrate, rank candidate nodes by scheduling fit (CPU, RAM, GPU, region, affinity) and resource headroom. Block locality is not a factor — the target node fetches overlay blocks via bitswap from scattered providers

#### NEW: `src/Orchestrator/Controllers/BlockStoreController.cs`

Following `DhtController.cs` pattern:

```
POST /api/blockstore/join
  Request:  { nodeId, vmId, peerId, capacityBytes, usedBytes }
  Auth:     X-BlockStore-Token: HMAC-SHA256(authToken, nodeId:vmId)
  Response: { success, bootstrapPeers, peerIdRegistered }

POST /api/blockstore/announce
  Request:  { nodeId, vmId, cids: ["bafk..."], action: "add"|"remove" }
  Auth:     X-BlockStore-Token
  Response: { success, recorded: count }
  Purpose:  Batch announce/withdraw CIDs (orchestrator maintains secondary index;
            the DHT is the primary source of truth for provider records)

GET /api/blockstore/locate/{cid}
  Response: { cid, providers: [{ nodeId, peerId, multiaddr }],
              replication: count }
  Purpose:  Find which nodes have a specific block. Queries the DHT's
            provider records. Used for audit and diagnostics.

POST /api/blockstore/manifest
  Request:  { vmId, nodeId, rootCid, version, changedBlockCids: ["bafk..."], totalBytes }
  Auth:     Internal (nodeId must match vm.NodeId — see Section 7 for fencing)
  Response: { success, manifestVersion }
  Purpose:  Register an updated VM manifest from a lazysync cycle.
            Orchestrator records the new version. No replication plan is
            returned — blocks replicate via Kademlia scatter autonomously.

GET /api/blockstore/manifest/{vmId}
  Response: { vmId, currentVersion, confirmedVersion, confirmedRootCid,
              timestamp, replicationFactor: 3,
              replicationStatus: { providerCounts: { "bafk...": 12, ... } } }
  Purpose:  Query manifest version and replication status for a VM.
            replicationFactor = per-VM configurable N (default 3, 0 = ephemeral).
            confirmedVersion = latest version where ALL chunks have ≥N
            providers in the DHT.
            currentVersion = latest registered (may still be propagating).

GET /api/blockstore/audit/{vmId}
  Response: { vmId, version, totalChunks, chunksWithSufficientProviders,
              underReplicatedChunks: [{ cid, providerCount }] }
  Purpose:  Detailed replication health for a specific VM's manifest.
            Queries FindProviders for each chunk CID. Used by the
            orchestrator's background audit loop to advance confirmedVersion
            and detect under-replication.

GET /api/blockstore/stats
  Response: { totalNodes, totalCapacity, totalUsed, totalBlocks,
              avgReplication, manifestCount, totalReplicatedBytes }
  Purpose:  Network-wide storage metrics
```

#### MODIFY: `src/Orchestrator/Models/Node.cs`

Add `BlockStoreInfo` class (alongside existing `DhtNodeInfo`, `RelayNodeInfo`):

```csharp
public class BlockStoreInfo
{
    public string BlockStoreVmId { get; set; }
    public string? PeerId { get; set; }                 // libp2p peer ID
    public string ListenAddress { get; set; }           // IP:5001
    public int ApiPort { get; set; } = 5090;
    public long CapacityBytes { get; set; }             // Allocated storage (5% of node)
    public long UsedBytes { get; set; }                 // Currently used
    public int BlockCount { get; set; }                 // Local blocks stored (determined by Kademlia proximity)
    public BlockStoreStatus Status { get; set; } = BlockStoreStatus.Initializing;
    public DateTime? LastHealthCheck { get; set; }
}

public enum BlockStoreStatus { Initializing, Active, Degraded, Full, Offline }
```

Add `BlockStoreInfo? BlockStoreInfo` property to the Node class.

Note: Nodes no longer receive pinning commands from the orchestrator. Which blocks a node stores is determined by Kademlia XOR proximity — nodes pull and serve blocks whose CIDs are close to their peer ID in XOR space. GC is local LRU eviction within the 5% budget.

#### MODIFY: `src/Orchestrator/Models/VirtualMachine.cs`

Add `BlockStore` to VmType enum:

```csharp
public enum VmType
{
    General, Compute, Memory, Storage, Gpu, Relay, Dht, Inference,
    BlockStore    // NEW
}
```

#### MODIFY: `src/Orchestrator/Services/SystemVm/SystemVmReconciliationService.cs`

Wire up `DeployBlockStoreVmAsync`:

```csharp
private async Task<string?> DeployBlockStoreVmAsync(Node node, CancellationToken ct)
{
    var vmService = _serviceProvider.GetRequiredService<IVmService>();
    return await _blockStoreService.DeployBlockStoreVmAsync(node, vmService, ct);
}
```

#### MODIFY: `src/Orchestrator/Services/SystemVm/ObligationEligibility.cs`

Uncomment the BlockStore eligibility block.

#### MODIFY: `src/Orchestrator/Services/SystemVm/SystemVmLabelSchema.cs`

Add required labels for `VmType.BlockStore`:

```csharp
[VmType.BlockStore] = [
    "role",
    "blockstore-listen-port",
    "blockstore-api-port",
    "blockstore-storage-bytes",
    "blockstore-auth-token",
    "blockstore-advertise-ip",
    "node-region",
    "node-id",
],
```

#### MODIFY: `src/Orchestrator/Services/VmService.cs`

Add `VmType.BlockStore` to system VM detection:

```csharp
var isSystemVm = request.VmType is VmType.Relay or VmType.Dht or VmType.BlockStore;
```

#### MODIFY: `src/Orchestrator/Program.cs`

Register BlockStoreService in DI.

---

### 4.2 Node Agent Changes

#### MODIFY: `VmType` enum in `VmModels.cs`

Add `BlockStore` variant to match Orchestrator.

#### NEW: `src/DeCloud.NodeAgent/CloudInit/Templates/blockstore-vm-cloudinit.yaml`

Cloud-init template following `dht-vm-cloudinit.yaml` pattern:

**Key components:**
- `blockstore` unprivileged user
- WireGuard mesh enrollment (same as DHT VM)
- `blockstore-node` Go binary (gzip+base64 injected)
- Systemd service for the binary
- Bootstrap poll service (polls `POST /api/blockstore/join`)
- Ready callback service (backup path via NodeAgent)
- Nginx reverse proxy for dashboard
- Health check script
- qemu-guest-agent for orchestrator monitoring

**Environment file** (`/etc/decloud-blockstore/blockstore.env`):
```bash
BLOCKSTORE_LISTEN_PORT=__BLOCKSTORE_LISTEN_PORT__
BLOCKSTORE_API_PORT=__BLOCKSTORE_API_PORT__
BLOCKSTORE_ADVERTISE_IP=__WG_TUNNEL_IP__
BLOCKSTORE_STORAGE_BYTES=__BLOCKSTORE_STORAGE_BYTES__
BLOCKSTORE_DATA_DIR=/var/lib/decloud-blockstore
DECLOUD_NODE_ID=__NODE_ID__
DECLOUD_REGION=__DHT_REGION__
```

#### NEW: `src/DeCloud.NodeAgent/CloudInit/Templates/blockstore-vm/`

Directory containing:

```
blockstore-vm/
├── blockstore-node-src/
│   ├── main.go                  # Block store binary
│   ├── go.mod
│   └── go.sum
├── blockstore-bootstrap-poll.sh # Polls orchestrator for peers
├── blockstore-notify-ready.sh   # Callback to NodeAgent
├── blockstore-health-check.sh   # Health check for monitoring
└── blockstore-dashboard.py      # Dashboard server (Python)
```

#### NEW: `blockstore-node-src/main.go` — Block Store Binary

Go binary using libp2p with bitswap:

```go
// Key dependencies:
// - github.com/libp2p/go-libp2p          (networking)
// - github.com/ipfs/go-bitswap           (block exchange)
// - github.com/ipfs/go-datastore         (block storage backend)
// - github.com/ipfs/go-ds-flatfs         (flat file storage)
// - github.com/ipfs/go-cid               (content identifiers)
// - github.com/multiformats/go-multihash (hashing)

Features:
1. libp2p host with persistent Ed25519 identity
2. Bitswap client+server for block exchange
3. FlatFS backend for block storage (content-addressed flat files)
4. Storage quota enforcement (refuse writes when full)
5. Garbage collection (LRU eviction within 5% budget)
6. GossipSub subscription (`decloud/blockstore/new-blocks`) for near-instant block discovery
7. Adaptive XOR threshold pull logic (closer distance + more free space → more aggressive pulling)
8. Periodic DHT neighborhood scan (durable fallback for missed GossipSub messages)
9. Localhost HTTP API on port 5090
10. Provider record announcement via DHT connection
```

**HTTP API (localhost only, port 5090):**

```
GET  /health
  → { status, peerId, connectedPeers, blockCount, usedBytes,
      capacityBytes, usagePercent }

POST /blocks
  Body: raw bytes (Content-Type: application/octet-stream)
  → { cid: "bafk...", size: 1234, stored: true }

GET  /blocks/{cid}
  → raw bytes (Content-Type: application/octet-stream)
  Note: If not local, attempts bitswap retrieval from network

DELETE /blocks/{cid}
  → { cid, deleted: true }

GET  /blocks
  Query: ?offset=0&limit=100
  → { blocks: [{ cid, size, lastAccess, createdAt }], total: 1234 }

POST /dag
  Body: JSON { manifest: { type, vmId?, parentCid?, chunks: [{ offset, cid }] },
               blocks: { "cid": "<base64 bytes>", ... } }
  → { rootCid: "bafk...", blockCount: N, totalBytes: M, stored: true }
  Note: Stores a Merkle DAG manifest + referenced blocks atomically.
        Used for VM overlay manifests.

GET  /dag/{cid}
  → { rootCid, manifest: { type, vmId, chunks: [...] },
      localBlocks: N, totalBlocks: M, complete: bool }
  Note: Returns the manifest and reports how many of its referenced
        blocks are available locally (complete=true when all present).

GET  /stats
  → { capacityBytes, usedBytes, usagePercent, blockCount,
      connectedPeers, bitswapSent, bitswapReceived }

POST /gc
  → { freedBytes, freedBlocks, remainingBytes }
  Note: Evicts least-recently-used blocks. Local LRU eviction —
        no orchestrator coordination needed.

POST /connect
  Body: { peers: ["/ip4/.../p2p/..."] }
  → { results, connected, total }
```

#### NEW: `src/DeCloud.NodeAgent/Controllers/BlockStoreCallbackController.cs`

Following `DhtCallbackController.cs` pattern:

```
POST /api/blockstore/ready
  Body: { vmId, peerId }
  Auth: X-BlockStore-Token: HMAC-SHA256(machineId, vmId:peerId)
```

#### MODIFY: `CommandProcessorService.cs`

Add `VmType.BlockStore` handling in `HandleCreateVmAsync`:
- Load `blockstore-vm-cloudinit.yaml` template
- Substitute labels into cloud-init variables
- Same WireGuard mesh enrollment pattern as DHT

---

### 4.3 Go Binary Design Details

#### Storage Backend: FlatFS

```
/var/lib/decloud-blockstore/
├── identity.key           # Persistent libp2p identity
├── blocks/                # FlatFS content-addressed storage
│   ├── _README            # Sharding info
│   ├── BA/                # First 2 chars of CID
│   │   └── FKREIG...      # Block files named by CID
│   └── QM/
│       └── ...
├── dags/                  # DAG manifests (JSON, indexed by root CID)
├── dynamic-peers          # Runtime peer injection (same as DHT)
└── datastore/             # LevelDB for metadata (block index, stats)
```

#### Block Lifecycle

```
1. POST /blocks (raw bytes)
   → SHA-256 hash → CIDv1 (raw codec, SHA2-256)
   → Write to FlatFS
   → Announce provider record to DHT (via connected DHT node)
   → Return CID

2. GET /blocks/{cid}
   → Check local FlatFS
   → If not found: bitswap request to network
   → Timeout after 30 seconds
   → Return bytes or 404

3. DELETE /blocks/{cid}
   → Delete from FlatFS
   → Withdraw provider record from DHT

4. POST /dag (manifest + blocks)
   → Store each block in FlatFS (deduplicated by CID)
   → Store manifest in dags/ directory
   → Manifest root CID = SHA-256 of manifest JSON
   → Return root CID

5. GC (local LRU eviction)
   → Sort blocks by last access time (LRU)
   → Delete least-recently-used blocks until usage is under 85% of capacity
   → Hard refuse writes at 95% capacity
   → Withdraw provider records for deleted blocks
   → Kademlia naturally re-replicates if provider count drops —
     other close nodes discover the gap and pull from remaining providers
```

#### DAG / Manifest Structure

A DAG manifest describes the current state of a VM's **overlay disk only** — not the full virtual disk. VMs use a qcow2 backing chain: a read-only base image (shared, downloadable from the image registry) plus a per-VM writable overlay that captures all writes. The base image and cloud-init configuration are reconstructible artifacts — only the overlay carries unique state.

The manifest is a single evolving document where chunk CIDs are replaced in-place as the lazysync daemon detects dirty blocks. There is no full/delta distinction. Just one manifest per VM that evolves over time:

```json
{
  "type": "vm-overlay",
  "vmId": "vm-abc123",
  "replicationFactor": 3,
  "version": 47,
  "timestamp": "2026-02-16T12:05:00Z",
  "baseImageId": "debian-12-generic",
  "baseImageHash": "sha256:a1b2c3...",
  "overlayVirtualSizeBytes": 107374182400,
  "blockSizeBytes": 1048576,
  "blockCount": 3,
  "chunks": [
    { "offset": 0,       "cid": "bafk...aaa" },
    { "offset": 1048576, "cid": "bafk...bbb" },
    { "offset": 2097152, "cid": "bafk...ccc" }
  ]
}
```

**Key properties:**
- **`type: "vm-overlay"`** — not `vm-disk`. This manifest represents the overlay layer only.
- **`baseImageId` + `baseImageHash`** — identifies the backing image. The target node downloads this from the image registry (or from block store peers that have it cached). The orchestrator knows the URL.
- **Sparse chunks** — only offsets with allocated clusters appear. A 100 GB virtual disk with 3 GB of overlay writes has ~3,000 chunks, not 100,000. Offsets not in the manifest read through to the base image.
- Cloud-init ISO is not in the manifest — the orchestrator regenerates it from the VM's labels during migration.
- **`blockCount`** — number of chunks in this manifest version. The billing unit:
  `storage_cost = blockCount × replicationFactor × costPerMbPerHour × (blockSizeBytes / 1_048_576)`.
  Updated each lazysync cycle. The orchestrator passes `blockCount` and
  `blockSizeKb` to `settleCycle()` — the contract applies the rate formula
  on-chain so billing math is fully trustless.
- **Tail block** — the final chunk of an overlay is almost never exactly
  `blockSizeBytes`. It is stored at its real size and billed as one full block.
  Maximum over-billing per VM: one block worth of storage. This avoids
  zero-padding waste while keeping block count as a clean billing integer.

The `version` field is a monotonically increasing integer, incremented on each lazysync cycle that produces changes. When the lazysync daemon detects dirty blocks, it:
1. Reads the dirty chunks (crash-consistent via QEMU incremental backup)
2. Hashes each chunk → CIDv1
3. Replaces the corresponding offset's CID in the manifest (if it actually changed)
4. Increments version
5. Pushes only the new blocks to the block store
6. Registers the updated manifest with the orchestrator

The orchestrator tracks two versions per VM:
- **currentVersion**: latest manifest registered (may be partially replicated)
- **confirmedVersion**: latest manifest where ALL referenced blocks have ≥N providers in the DHT (verified by audit loop)

Blocks referenced by a current manifest naturally maintain high provider counts (nodes keep re-announcing them, and they get recent bitswap access which protects them from LRU eviction). Blocks no longer in any current manifest stop being re-announced, their provider records expire, and they naturally disappear via LRU GC. This eliminates delta chains, delta consolidation, and the full/delta manifest type distinction entirely.

Content addressing provides automatic deduplication: if two VMs use the same base image and write the same packages, those overlay chunks share the same CID and are stored once. Unchanged overlay chunks across lazysync cycles share the same CID and are never re-stored or re-transferred.

#### Chunk Sizes by Manifest Type

`blockSizeBytes` is a **per-manifest-type constant** declared in the manifest and
enforced by the block store binary at write time. Different content types have
different optimal chunk sizes based on their access patterns, update frequency,
and scale. The billing unit is always **one block**, with cost normalized to
`costPerMbPerHour × (blockSizeBytes / 1_048_576)` — so all manifest types pay
the same rate per MB of replicated data.

| Manifest Type | `blockSizeBytes` | Rationale |
|---|---|---|
| `vm-overlay` | **1 MB** (1,048,576) | Aligns with QEMU dirty bitmap granularity. A VM with 5 GB overlay → ~5,120 blocks. Manageable manifest size. Unchanged regions produce identical CIDs — never re-transferred. |
| `model-shard` | **64 MB** (67,108,864) | Aligns with transformer layer boundaries for pipeline-parallel inference. Llama-3 70B Q4 (~40 GB) → 640 blocks. Bitswap transfers of 64 MB amortize connection overhead efficiently. |
| `lora-adapter` | **256 KB** (262,144) | Fine-grained deduplication across fine-tuned variants of the same base model — many adapters share base layer weights. |
| `image-template` | **4 MB** (4,194,304) | Base OS images (2–4 GB) → 500–1,000 blocks. Good deduplication across Ubuntu/Debian versions that share many 4 MB regions. |

**Chunk count reference for model-shard type:**

| Model | Precision | Size | 64 MB Chunks |
|---|---|---|---|
| Llama-3 8B | FP16 | ~16 GB | 256 |
| Llama-3 70B | Q4_K_M | ~40 GB | 640 |
| Llama-3 70B | FP16 | ~140 GB | 2,240 |
| Mistral 7B | Q4 | ~4 GB | 64 |
| GPT-4 scale | Q4 (est.) | ~1.8 TB | ~28,800 |

For the distributed AI inference vision: an inference VM needing layers 16–31
of a 70B model requests the ~7–8 specific 64 MB chunk CIDs covering those
layers. The block store network fetches them via bitswap from the nearest
providers. No full model download required before inference can begin.

**Block store enforcement:**
```go
// Enforced in blockstore-node main.go
// Each manifest type declares its block size; the binary validates on write
var ManifestTypeBlockSize = map[ResourceType]int64{
    ResourceTypeVMOverlay:     1 * 1024 * 1024,    // 1 MB
    ResourceTypeModelShard:    64 * 1024 * 1024,   // 64 MB
    ResourceTypeLoRAAdapter:   256 * 1024,          // 256 KB
    ResourceTypeImageTemplate: 4 * 1024 * 1024,    // 4 MB
}

// Tail block exception: final chunk may be smaller than blockSizeBytes.
// Billed as one full block. All other chunks must be exactly blockSizeBytes.
```

#### Bitswap Integration

The Go binary connects to other Block Store peers via libp2p and uses the bitswap protocol for block exchange:

```go
// Bitswap network: blocks flow between peers automatically
// When a peer requests a block we have → serve it
// When we need a block we don't have → request from peers
bswap := bitswap.New(ctx, network, blockstore)
```

Bitswap is the **transfer mechanism**; GossipSub + Kademlia provide the **discovery mechanism**. When a new block is pushed to a block store, the source publishes the CID to the `decloud/blockstore/new-blocks` GossipSub topic. Block store nodes that are close to the CID in Kademlia XOR space (and have capacity) add it to their bitswap want list and pull it automatically. For nodes that missed the GossipSub message, periodic DHT neighborhood scans serve as the durable fallback. No orchestrator commands are needed — blocks scatter across the network autonomously.

#### DHT Provider Records

The Block Store binary announces which blocks it holds to the DHT network. This uses the standard IPFS provider record mechanism — any DHT node can answer "who has CID X?".

```go
// Announce that we provide a block
routing.Provide(ctx, cid, true)

// Find providers for a block we need
providers := routing.FindProviders(ctx, cid)
```

---

## 5. Resource Specifications

### Block Store VM Sizing

| Tier | vCPUs | RAM | Disk | Eligibility |
|------|-------|-----|------|-------------|
| Standard | 1 | 512 MB | 5% of node storage | ≥100 GB total storage, ≥2 GB RAM |

Disk allocation formula:
```
allocatedStorage = node.TotalStorageBytes * 0.05    // 5% duty, fixed
```

No min/max caps needed — the 100 GB eligibility threshold ensures the smallest allocation is 5 GB (sufficient for the VM + a meaningful number of replicated blocks), and 5% naturally scales with larger nodes.

| Node Storage | Block Store Allocation | Typical Use |
|-------------|----------------------|-------------|
| 100 GB | 5 GB | ~5,000 blocks (1 MB each) |
| 500 GB | 25 GB | ~25,000 blocks |
| 1 TB | 50 GB | ~50,000 blocks |
| 4 TB | 200 GB | ~200,000 blocks |

### Network Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 5001 | TCP (libp2p) | Bitswap block exchange |
| 5090 | TCP (HTTP) | Localhost API (internal only) |
| 8080 | TCP (HTTP) | Dashboard (proxied via Nginx port 80) |

---

## 5.5 Replication Factor

### Default: All Tenant VMs at N=3

Every non-system VM is created with `replicationFactor=3` by default. System
VMs (Relay, DHT, BlockStore) use `replicationFactor=0` — they are ephemeral by
design, reconstructed from their cloud-init template on redeploy, not from disk
state.

On Phase D launch, every tenant VM is automatically enrolled in lazysync with no
user configuration required. The platform's resilience guarantee is uniform and
unconditional by default. Users shouldn't have to opt into data protection.

### Phase D+1: User-Configurable Replication Factor

When exposed as a VM creation parameter, the supported tiers are:

| Factor | Meaning | Durability | Use case |
|--------|---------|-----------|---------|
| 0 | Ephemeral | None — data lost on node failure | Stateless workloads, batch jobs, CI runners. User explicitly accepts data loss. |
| 1 | Single replica | Survives if ≥1 provider holds the blocks | Dev environments, cost-sensitive non-critical workloads |
| 3 | Standard (default) | Survives loss of 2 provider nodes simultaneously | All production workloads |
| 5 | High availability | Survives loss of 4 provider nodes simultaneously | Databases, ML training checkpoints, critical stateful services |

Factor 2 is valid but not exposed — the meaningful resilience jump is from 1 to 3
(tolerates 2 simultaneous failures). Exposing 2 creates confusion without adding
real value.

`replicationFactor` is **immutable after VM creation**. Changing it mid-life would
require seeding or withdrawing blocks across the network (a Phase D+2 feature). For
now: set at creation, honored by lazysync, used in billing.

### Model fields
```csharp
// VirtualMachine or VmSpec:
public int ReplicationFactor { get; set; } = 3;   // 0=ephemeral, 1/3/5=replicated

// Updated each lazysync cycle (for accurate billing):
public long LastKnownOverlayBytes { get; set; }
public DateTime? LastLazysyncAt { get; set; }
```

### Pricing Impact

Replication factor drives two cost components:

**1. Storage replication cost** — N=3 means 3× the overlay bytes consumed across the
block store network. The VM owner pays for the storage duty they impose on the
network. In practice, overlay is sparse: a 100 GB virtual disk typically has 2–8 GB
of allocated overlay after a few hours of use, so N=3 costs ~6–24 GB of distributed
storage, not 300 GB.

**2. Compute cost modifier** — lazysync I/O (dirty bitmap export + chunk push) on the
source node is small enough to fold into the storage cost rather than price separately.

**Formula:**
VM hourly cost = compute_cost + storage_replication_cost
compute_cost =
vCPUs × cpu_rate + memory_gb × memory_rate + disk_gb × storage_rate
storage_replication_cost =
blockCount × replicationFactor × costPerMbPerHour × (blockSizeKb / 1024)
Where:
costPerMbPerHour = platform constant (e.g. $0.000001/MB/hour)
blockCount       = number of chunks in the current confirmed manifest
(updated by LazysyncManager each audit cycle)
blockSizeKb      = chunk size for this manifest type (1024 for vm-overlay,
65536 for model-shard, 256 for lora-adapter, 4096 for image-template)
replicationFactor = 0 / 1 / 3 / 5 (immutable after VM creation)

**Why block-based pricing is better than GB-based:**

- **Deterministic** — `blockCount` is exact, not estimated. The manifest records
  exactly how many chunks exist at the current version.
- **Trustless on-chain** — `settleCycle()` receives `blockCounts[]`,
  `blockSizeKb[]`, and `replicationFactors[]` as raw inputs. The contract
  applies `costPerMbPerHour` itself. The orchestrator cannot inflate the
  storage amount — only the block count, which is verifiable from the manifest.
- **User-legible** — "Your VM has 3,847 blocks at 1 MB each, N=3 replicas,
  $0.000001/MB/hour = $0.01155/hour for replication" is actionable.
  "Your overlay is 3.7 GB" is not.
- **Scales correctly to AI workloads** — A 70B model shard at 640 × 64 MB
  blocks bills at the same rate-per-MB as a 5,120 × 1 MB VM overlay.
  No special-casing needed.

**Example costs at $0.000001/MB/hour:**

| Workload | Blocks | Block Size | N | Storage cost/hr |
|---|---|---|---|---|
| Typical VM, light use | 512 | 1 MB | 3 | $0.001536 |
| Typical VM, active | 4,096 | 1 MB | 3 | $0.012288 |
| Llama-3 70B Q4 | 640 | 64 MB | 3 | $0.122880 |
| Llama-3 70B FP16 | 2,240 | 64 MB | 5 | $0.716800 |
| Fine-tune adapter | 200 | 256 KB | 3 | $0.000015 |

`replicationFactor=0` is priced noticeably lower — enough to make it a genuine
option for stateless workloads, but not so cheap that users accidentally choose it
for stateful workloads without understanding the tradeoff.

### Scheduler Impact

When `replicationFactor > 0`, the scheduler adds one filter condition:
candidate node must have BlockStoreInfo.Status == Active
A node without an Active BlockStore VM cannot participate in lazysync for tenant VMs
running on it — there is nowhere to push dirty blocks locally before they scatter
across the network.

This means the scheduler naturally steers replicated workloads toward nodes with
≥100 GB storage (the BlockStore eligibility threshold). Nodes below this threshold
can only host ephemeral (`replicationFactor=0`) VMs.

### Implementation touchpoints (Phase D)

| Component | Change |
|-----------|--------|
| `VmService.CreateVmAsync` | Set `ReplicationFactor = 3` as default for all non-system VMs |
| `CreateVmRequest` | Add optional `ReplicationFactor` field (defaults to 3, validated: 0/1/3/5) |
| `VmSpec` / `VirtualMachine` | Add `ReplicationFactor`, `LastKnownOverlayBytes`, `LastLazysyncAt` |
| `VmSchedulingService` | Add BlockStore Active filter when `ReplicationFactor > 0` |
| `LazysyncDaemon` (NodeAgent) | Skip VM entirely if `replicationFactor == 0`; use factor as provider count threshold |
| `LazysyncManager` (Orchestrator) | Use `replicationFactor` as N in `confirmedVersion` audit loop |
| `CalculateHourlyRate` | Add `storage_replication_cost` component |
| Source-offline alerting | `replicationFactor=0` → LOST (by design, no alert); `>0` → UNRECOVERABLE / RECOVERING / MIGRATING |
| Dashboard | Show replication factor + "Ephemeral" badge when factor=0 |

### Phase D+1: Replication-Aware GC (Priority Eviction of Confirmed Blocks)

**Context:** The local blockstore on the VM's hosting node is a transit buffer, not a
durable store. The VM overlay disk itself is the primary copy. The local blockstore
exists to stage blocks before they scatter to remote nodes via Kademlia. Once blocks
are confirmed replicated remotely (`remoteCount >= replicationFactor`), the local
copy is redundant from a durability standpoint.

**Current behavior:** Local GC is pure LRU — evicts by last access time regardless of
whether remote copies exist. This wastes the 5% duty on blocks that are already safe.

**Proposed GC priority order:**

```
1. Evict blocks confirmed remote (remoteCount >= RF, per last audit)
   → Safe: redundant copies exist on ≥N other nodes
   → Frees space for blocks still needing replication

2. Evict blocks not referenced by any current manifest
   → Orphaned chunks from old manifest versions — safe to evict anywhere

3. Evict by LRU (fallback)
   → May trigger Kademlia re-replication if provider count drops below RF
   → Current behavior
```

**Effect:** The local blockstore becomes a true **write-through cache** — it holds
blocks long enough for them to scatter, then evicts them to make room for new ones.
This maximises the useful throughput of the 5% duty rather than filling it with
already-safe data.

**Implementation touchpoints:**

| Component | Change |
|-----------|--------|
| `LazysyncManager` | After advancing `ConfirmedVersion`, write confirmed CID set to a per-VM confirmed-blocks file on the node (via orchestrator → node agent callback or node agent polling) |
| `blockstore-node main.go` | `runGC()`: before LRU pass, evict all CIDs present in the confirmed-blocks set for any VM. Withdraw provider record after eviction |
| `POST /api/blockstore/confirmed` | New NodeAgent endpoint: receives confirmed CID list from orchestrator after each audit cycle |
| Confirmed CIDs storage | Simple newline-delimited file per VM: `/var/lib/decloud-blockstore/confirmed/{vmId}.cids` — updated after each `ConfirmedVersion` advance |

**Note:** The local blockstore still serves bitswap requests for evicted blocks —
remote peers can fetch them from the remaining ≥N providers. Eviction only removes
the local copy; provider records are withdrawn so peers route requests elsewhere.

**Dependency:** Requires stable `ConfirmedVersion` audit loop (Phase D) and the
`reannounceExistingBlocks` startup fix (Phase D+1 — see item below).

### Phase D+1: Startup Re-announce (Provider Record Recovery)

**Problem:** When the blockstore binary restarts, `dht.Provide()` is only called for
newly written blocks. Existing blocks on FlatFS are loaded into the LRU cache but
never re-announced. Provider records in the DHT expire after 24 hours — a restarted
blockstore becomes invisible to `FindProvidersAsync` until new blocks are written.

**Affected scenarios:**
- Binary restart / systemctl restart
- DHT VM redeployment (fresh LevelDB, all provider records wiped)
- Node agent update triggering blockstore restart
- 24h provider record TTL expiry with no new writes

**Fix:** Add `reannounceExistingBlocks()` goroutine to `main.go`, called after
`connectBootstrapPeers()` with a 15s delay to allow the Kademlia routing table
to populate first.

```go
func (n *BlockNode) reannounceExistingBlocks(ctx context.Context) {
    time.Sleep(15 * time.Second) // routing table warmup
    ch, err := n.bstore.AllKeysChan(ctx)
    if err != nil { return }
    count := 0
    for c := range ch {
        if ctx.Err() != nil { return }
        announceCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
        _ = n.dht.Provide(announceCtx, c, true)
        cancel()
        count++
        if count%100 == 0 {
            log.Printf("reannounce: %d blocks announced", count)
        }
    }
    log.Printf("reannounce: complete — %d blocks re-announced", count)
}
```

**Insertion point in `main()`:**
```go
node.connectBootstrapPeers(ctx)
go node.reannounceExistingBlocks(ctx)  // ← add this line
```

### Phase D+1: DHT Neighborhood Scan (General Block Recovery)

**Problem:** The GossipSub fetch retry queue (item 31b) recovers blocks where the
notification arrived but the bitswap fetch failed. It cannot recover blocks where the
GossipSub message was missed entirely — node was offline, network partition, restart
during a seeding cycle, or the message was dropped by the ephemeral pub/sub layer.
In those cases, under-replicated blocks remain permanently undetected by the binary.

**Scope:** This is the general/cold recovery path. The retry queue is the hot path.
Together they eliminate both failure modes:

```
GossipSub notification received, fetch failed  → retry queue    (hot path)
GossipSub notification missed entirely         → neighborhood scan (cold path)
```

**Mechanism:** Periodically walk the local Kademlia neighborhood to discover blocks
that belong on this node (based on XOR proximity) but aren't present locally.

```
Every ScanInterval (e.g. 10 minutes):
  1. Walk the local Kademlia keyspace neighborhood
     → Use kadDHT.GetClosestPeers(ctx, myPeerId) to identify the keyspace
       range this node is responsible for
  2. For each CID in the neighborhood's provider records:
     → Call FindProviders(ctx, cid) — returns who has it
     → Check local bstore.Has(ctx, cid)
     → If not local AND within XOR threshold AND capacity allows:
         → GetBlock via bitswap from a provider
         → Store locally, announce to DHT
         → Log as neighborhood_pull
  3. Skip CIDs already in the retry queue (avoid double-fetching)
  4. Rate-limit: max N blocks per scan cycle to avoid I/O bursts
```

**Why the DHT alone doesn't do this:** The DHT is a passive lookup service. It
answers "who has block X?" but never decides "I should have block X." That decision
requires knowing the XOR proximity threshold and local capacity — context only the
block store binary has. The neighborhood scan is the bridge between the DHT's passive
records and the binary's active pull behavior.

**Implementation touchpoints (`main.go`):**
- Add `ScanInterval = 10 * time.Minute` constant
- Add `startNeighborhoodScanLoop(ctx)` goroutine launched from `main()`
- Uses `n.dht.GetClosestPeers()` + `n.dht.FindProvidersAsync()` for discovery
- Shares `n.fetchSem` with GossipSub fetches to bound total concurrency
- Logs `neighborhood_pull` and `neighborhood_scan_complete` diag events
- Skip CIDs present in `n.retryQueue` (check via a shadow `sync.Map`)

**Dependency:** Requires item 32 (DHT proximity endpoint) for the keyspace walk.

---

## 6. VM Disk Replication & Migration (Lazysync)

This is the **primary use case** for the block store — not a future feature.

### 6.1 Lazysync Model — Continuous Dirty Block Replication

Every user VM has its **overlay disk continuously replicated** via a background lazysync process. There are no "snapshot events." Dirty overlay chunks flow to the block store network as they change, and a single evolving manifest per VM tracks the current overlay state.

Lazysync replicates **only the overlay** — the qcow2 layer that captures writes on top of the read-only base image. The base image (a standard OS image like `debian-12-generic`) is a well-known artifact the orchestrator can re-fetch from the image registry during migration. The cloud-init ISO is regenerated from the VM's labels. Only the overlay carries unique, irreplaceable state.

```
Node A runs VM-1 (100 GB virtual disk):

  Disk structure:
    base image:   debian-12-generic.qcow2  (read-only, shared, ~2 GB)
    overlay:      disk.qcow2               (writable, per-VM, sparse)
    cloud-init:   cloud-init.iso           (regenerable from labels)

  Only disk.qcow2 (the overlay) is replicated via lazysync.

Background lazysync daemon (runs on the node agent, cycles every ~5 minutes):

  For each running VM on this node:

  1. QEMU incremental backup via QMP:
     → drive-backup sync=incremental
     → Exports dirty overlay blocks since last sync cycle
     → Crash-consistent: QEMU uses copy-on-write for blocks
       written during export — VM is NOT paused
     → QEMU's Changed Block Tracking (CBT) via persistent dirty
       bitmaps tracks which blocks have been written (since 4.0+)

  2. Chunk the exported dirty data into 1 MB blocks → CIDv1 per block

  3. Compare each chunk's CID against current manifest:
     → Skip any block whose CID matches (written but not actually changed)
     → Only genuinely new/changed blocks proceed

  4. Push new blocks to Node A's local block store VM:
     → POST /blocks for each new block
     → Block store announces provider records to DHT
     → Block store publishes new CIDs to GossipSub topic
       "decloud/blockstore/new-blocks" (fast notification path)

  5. Update manifest in-place:
     → manifest.chunks[offset] = newCID for each changed offset
     → manifest.version++
     → manifest.timestamp = now

  6. Register updated manifest with orchestrator:
     → POST /api/blockstore/manifest { vmId, nodeId, rootCid, version,
         changedBlockCids, totalBytes }
     → Orchestrator validates nodeId matches vm.NodeId (Section 7 fencing)
     → Orchestrator records the new version (no replication plan returned)

  7. Blocks replicate via Kademlia scatter (autonomous, no orchestrator action):
     → Block store announced provider records + GossipSub notification in step 4
     → Other block store nodes receive GossipSub message with new CIDs
     → Each receiving node computes XOR distance: distance(myPeerId, cid)
     → If close enough (adaptive threshold based on local free space) → pull via bitswap
     → Blocks scatter across many nodes (not limited to scheduling-eligible ones)
     → GossipSub is the fast path; DHT provider records are the durable fallback
       for nodes that missed the GossipSub message (offline, network partition)

  8. Orchestrator audits replication asynchronously:
     → Background audit loop queries DHT: FindProviders(chunkCid)
       for each chunk in the latest manifest
     → When all chunks have ≥N providers → advance confirmedVersion
     → Under-replicated chunks are flagged but NOT actively pushed —
       Kademlia's re-publication handles recovery naturally

  9. VMs with replicationFactor=0 (ephemeral) skip lazysync entirely — no
     overlay replication, no manifest tracking. Dashboard shows ephemeral status.
```

### 6.1.1 Initial Overlay Seeding

New VMs start with a minimal overlay (just cloud-init writes + first-boot setup). The lazysync daemon begins pushing overlay blocks on the first cycle after the VM is provisioned. The allocated overlay clusters (not the full virtual disk) are chunked and replicated:

```
New VM:
  - Virtual disk size: 100 GB
  - Overlay allocated clusters after first boot: ~500 MB
  - Initial seed: ~500 blocks (1 MB each)
  - Time to seed: ~2-5 minutes over typical node bandwidth

Sparse overlay allocation via `qemu-img map`:
  - Only allocated clusters are chunked and replicated — zero/unallocated
    regions are not included in the manifest.
```

The orchestrator tracks seeding state via its audit loop. VMs with `confirmedVersion == 0` (no chunks have ≥N providers yet) are flagged as "unprotected" in the dashboard. Seeding is rate-limited to avoid saturating the node's block store VM or network.

Compare this to full-disk seeding: a 100 GB disk would require 100,000 chunks regardless of actual usage. Overlay-only seeding typically handles 1-10% of that volume.

### 6.1.2 QEMU Integration

The lazysync daemon requires integration with the hypervisor layer via QEMU's QMP (QEMU Machine Protocol):

```
QEMU QMP commands used:

1. block-dirty-bitmap-add
   → Creates a persistent dirty bitmap on the VM's drive
   → Called once when a VM is enrolled in lazysync

2. drive-backup sync=incremental bitmap=lazysync-N
   → Exports dirty blocks to a temp file (crash-consistent)
   → VM continues running — QEMU handles CoW internally
   → Returns the dirty regions for chunking

3. block-dirty-bitmap-clear
   → Resets the bitmap after successful export
   → Called after blocks are confirmed pushed to block store
```

The node agent already manages VMs via libvirt/QEMU and has access to QMP sockets. The lazysync daemon runs as a background service on the node agent, cycling through running VMs on a configurable interval.

### 6.1.3 Manifest Versioning & Consistency

The orchestrator tracks manifest versions and audits replication via DHT provider counts:

```
Manifest v1: [chunk0=A, chunk1=B, chunk2=C]  → all chunks have ≥N providers (confirmed)
Lazysync: chunk1 changed → CID D
Manifest v2: [chunk0=A, chunk1=D, chunk2=C]  → chunk D propagating via Kademlia...
Lazysync: chunk2 changed → CID E
Manifest v3: [chunk0=A, chunk1=D, chunk2=E]  → chunk E propagating...

Orchestrator state for this VM:
  currentVersion:   3   (latest registered)
  confirmedVersion: 1   (latest where ALL chunks have ≥N providers in DHT)

Rules:
  - confirmedVersion advances when the audit loop verifies provider counts
  - Old blocks (e.g., chunk B from v1) naturally lose providers as nodes
    GC them via LRU eviction — no explicit "unpin" needed
  - If v2's chunks all reach ≥N providers before v3 is registered →
    confirmedVersion advances to 2
  - Recovery always uses confirmedVersion's manifest
```

### 6.2 Migration on Node Failure

```
Node A goes offline (shutdown / blocked / hardware failure):

For each VM on Node A:
  1. Orchestrator retrieves VM's scheduling requirements
     (CPU, RAM, GPU, region, affinity rules, etc.)

  2. Orchestrator retrieves VM's confirmedVersion manifest
     (root CID + all referenced block CIDs — a single flat manifest,
      no delta chain to walk)

  3. For each candidate node, evaluate:
     scheduling_fit:      does the node meet VM's requirements? (filter)
     available_resources: enough CPU/RAM/disk headroom? (rank)

     Block locality is NOT a factor — the target node fetches
     overlay blocks via bitswap from scattered providers.

  4. Select best candidate based on scheduling fit + resource headroom

  5. Orchestrator updates vm.NodeId from A → B (Section 7 fencing)

  6. Orchestrator sends CreateVm command to Node B with:
     - VM spec (CPU, RAM, disk, GPU mode, etc.)
     - Manifest root CID (points to confirmed overlay state)
     - Base image ID (Node B downloads or uses cached)
     - Labels (for cloud-init regeneration)

  7. Node B:
     a. Downloads base image (if not cached) from image registry
     b. Creates empty overlay qcow2 backed by base image
     c. Fetches overlay chunks via bitswap from scattered providers
        (many sources → parallelized → fast)
     d. Writes chunks into overlay at correct offsets
     e. Regenerates cloud-init ISO from labels
     f. Boots VM

  8. VM resumes from confirmed state
     (data loss = changes between confirmedVersion and currentVersion)
```

#### Why This Is Fast

- **Base image**: probably cached on Node B (it's a standard image). If not, it's a single download of a well-known artifact, not a block-by-block fetch.
- **Overlay**: only the allocated clusters — typically 1-10% of the virtual disk size. Blocks are scattered across many providers — bitswap fetches from the closest/fastest ones in parallel. More providers = faster reconstruction than pulling from 3 pre-selected replicas.
- **Cloud-init**: regenerated in milliseconds from labels. Zero network transfer.
- **No chain traversal**: the manifest IS the complete overlay state. Read chunks, write file, done.

### 6.3 DHT-Native Scatter Replication

Replication is **decoupled from placement**. These are independent concerns:

- **Replication = durability.** Scatter blocks as widely as possible across the network.
- **Scheduling = placement.** The orchestrator picks an eligible node when migration is needed.

The block store does NOT replicate to specific "target nodes." Instead, Kademlia's XOR metric naturally scatters blocks across the network. Every node with a block store (≥100 GB storage) participates in replication regardless of its scheduling capabilities.

```
Example: VM-1 requires 4 vCPUs, 8 GB RAM, us-east region

Lazysync pushes overlay blocks to Node A's local block store.
Block store announces provider records to the DHT.
Kademlia naturally scatters blocks to the K closest nodes in XOR space:

  Node B: us-east, 8 cores, 32 GB RAM     ← could host VM-1
  Node C: ap-south, 2 cores, 4 GB RAM     ← can't host VM-1, CAN store its blocks
  Node D: eu-west, 16 cores, 64 GB RAM    ← could host VM-1
  Node E: us-east, 1 core, 2 GB RAM       ← Raspberry Pi, CAN store blocks
  Node F: eu-west, 4 cores, 8 GB RAM      ← could host VM-1
  ...12 more nodes scattered across regions

Durability: blocks survive even if an entire region goes down.
Candidate pool: ALL nodes with storage contribute, not just scheduling-eligible ones.
Migration: orchestrator picks from {B, D, F, ...} based on scheduling fit.
           Target fetches overlay via bitswap from {B, C, D, E, F, ...} in parallel.
```

**Why this is better than placement-aware replication:**

1. **Larger replication pool** — A 2-core Raspberry Pi can store chunks for a 16-core GPU VM. This expands the candidate pool dramatically compared to restricting replicas to scheduling-eligible nodes.

2. **Better durability** — Kademlia's XOR metric distributes blocks across the ID space, which is statistically independent of geographic or failure domain clustering. More diverse placement = more resilient.

3. **Simpler replication logic** — No scheduling requirements evaluation in the replication path. No failure domain reasoning. The block store only asks: "Am I close to this CID in XOR space? Do I have room?"

4. **Faster migration** — With N scattered providers (potentially 10-20+ nodes), bitswap parallelizes the fetch across many sources. Placement-aware replication with 3 pre-selected replicas means 3 sources. Scatter gives you many more.

5. **No orchestrator bottleneck** — Replication happens autonomously via Kademlia. The orchestrator only audits provider counts.

### 6.4 Adaptive XOR Threshold

Block store nodes decide whether to pull a block based on two factors:

1. **XOR distance**: `distance(myPeerId, blockCid)` — closer = higher priority
2. **Local free space**: more room = more aggressive pulling

```go
// Pseudocode for pull decision
func shouldPull(cid CID) bool {
    distance := xorDistance(myPeerId, cid)
    freePercent := (capacity - used) / capacity
    
    // Base threshold: only pull if we're in the K closest
    // Adaptive: relax threshold when we have more space
    threshold := baseThreshold * (1.0 + freePercent)
    
    return distance < threshold
}
```

This naturally balances the network:
- Nodes with lots of free space pull more aggressively → fill up → slow down
- Nodes near capacity only pull blocks they're very close to in XOR space
- No central coordinator needed

### 6.5 GossipSub Fast Path

Standard Kademlia provider records are passive — they answer queries but don't push notifications. Waiting for periodic DHT scans to discover new blocks would add minutes of latency to replication.

GossipSub provides near-instant notification:

```
Source block store (Node A) pushes new block:
  1. Store block locally
  2. Announce provider record to DHT
  3. Publish CID to GossipSub topic "decloud/blockstore/new-blocks"

Receiving block stores (Nodes B, C, D, ...):
  1. Receive GossipSub message with new CID
  2. Compute XOR distance: distance(myPeerId, cid)
  3. If close enough (adaptive threshold) → add to bitswap want list
  4. Bitswap fetches block from Node A (or any other provider)
  5. Store locally, announce provider record

Time from push to replication: seconds (not minutes)
```

For nodes that missed the GossipSub message (offline, network partition), periodic DHT neighborhood scans serve as the durable fallback. No orchestrator commands are needed — blocks scatter across the network autonomously.

### 6.6 Handling "Scatter Window" Data Loss

There's an inherent window between when a block is pushed to the source node's block store and when it scatters to other providers. If the source node dies in this window, the block may be lost.

```
Timeline:
  T=0:  Lazysync pushes block X to Node A's block store
  T=1s: GossipSub notification sent
  T=3s: Nodes B, C start pulling via bitswap
  T=5s: Block X replicated to B, C (provider count = 3)
  ---
  T=2s: Node A crashes (before B, C finish pulling)
        → Block X may be lost (only provider was A)
```

**Mitigation:**

1. **confirmedVersion tracking**: The orchestrator doesn't advance `confirmedVersion` until all chunks have ≥N providers. Recovery uses `confirmedVersion`, not `currentVersion`. This means:
   - If Node A dies at T=2s, the VM migrates using `confirmedVersion` (before block X)
   - Block X (in `currentVersion` but not `confirmedVersion`) is lost
   - Data loss = changes between confirmed and current

2. **Source-offline alerting**: When a node goes offline, the orchestrator checks each VM's manifest state:

```
For each VM on the dead node:

  Case 1: confirmedVersion == 0
    → No confirmed state exists. VM cannot be recovered.
    → Alert: "VM-X has no confirmed state. Data may be lost.
       The VM cannot be migrated."
    → Dashboard status: UNRECOVERABLE

  Case 2: confirmedVersion > 0 but < currentVersion
    → Alert: "VM-X will recover from confirmed version N.
       Approximately M minutes of recent state may be lost."
    → Proceed with migration from confirmed manifest
    → Dashboard status: RECOVERING (partial data loss)

  Case 3: confirmedVersion == currentVersion (fully caught up)
    → No data loss. Proceed with migration normally.
    → Dashboard status: MIGRATING

  Case 4: replicationFactor == 0 (ephemeral)
    → VM is gone. No alert about data loss (user opted in).
    → Dashboard status: LOST (ephemeral)
```

This is the honest approach: the scatter window is small (seconds to minutes) and cannot be eliminated without synchronous replication (which would add latency to every write). The mitigation is visibility — the orchestrator tracks the gap between `currentVersion` and `confirmedVersion` and alerts when it matters.

### 6.7 DHT Proximity Endpoint

The block store binary needs to evaluate its position in Kademlia XOR space relative to a given CID. The block store has its own persistent peer ID and can compute `xor(myPeerId, cid)` locally — this is a trivial bitwise operation.

However, for diagnostics, debugging, and the periodic neighborhood scan, the **DHT VM exposes a proximity endpoint**:

```
GET /proximity/{cid}
  → {
      distance: "0x1a3f...",          // XOR distance from this DHT node's peer ID
      closerKnownPeers: 7,            // how many peers in the routing table are closer
      estimatedRank: 8,               // approximate position (1 = closest known peer)
      kValue: 20                       // Kademlia K parameter
    }

  Purpose: Allows the block store to make informed pull decisions
  beyond simple XOR distance. "Am I in the K closest?" is more useful
  than "what is my raw distance?" for deciding whether to store a block.
```

For the **normal fast path** (GossipSub announce → pull decision), the block store uses local XOR distance + adaptive threshold (Section 6.4). The DHT proximity endpoint is used for:
- Periodic neighborhood scan (durable fallback)
- Diagnostic queries ("why did/didn't this node store block X?")
- Dashboard visualization of block distribution

This endpoint is added to the DHT VM's existing localhost HTTP API (port 5080), alongside `/health`, `/peers`, `/connect`, and `/publish`.

---

## 7. Split-Brain / Zombie VM Prevention

When a node fails and a VM is migrated to a new host, the original node may come back online with the VM still present locally. Without proper fencing, this creates a **split-brain** scenario: two instances of the same VM running simultaneously.

### 7.1 The Problem

```
Timeline of a split-brain event:

T=0        T=5min      T=8min       T=15min      T=20min
 │           │           │            │            │
 ▼           ▼           ▼            ▼            ▼

Node A     Node A      Orchestrator  Node B      Node A
running    fails       declares A    boots       comes back!
VM-1       (crash,     dead,         VM-1        VM-1 resumes
           network)    migrates      from        automatically
                       to Node B     storage
                                      │            │
                                      ▼            ▼
                                 ┌─────────┐  ┌─────────┐
                                 │  VM-1   │  │  VM-1   │
                                 │ (new)   │  │ (zombie)│
                                 │ Node B  │  │ Node A  │
                                 └─────────┘  └─────────┘
                                      │            │
                                      └─────┬──────┘
                                            │
                                            ▼
                                 💥 TWO INSTANCES RUNNING
```

**Consequences of split-brain:**

| Issue | Impact |
|-------|--------|
| **IP Conflict** | Both VMs have same WireGuard IP → routing chaos |
| **State Divergence** | Node A has pre-crash state, Node B has migrated state + new writes |
| **Identity Collision** | Same VM ID, same wallet association, same credentials |
| **Data Corruption** | If both write to external services (DBs, APIs), inconsistent state |
| **Lazysync Conflict** | Both nodes attempt to push manifest updates for the same VM |

### 7.2 Solution: NodeId as Authority (KISS Approach)

The `VirtualMachine` model already has a `NodeId` field that tracks which node is authorized to run the VM. Rather than creating a separate lease service, we use this existing field as the **single source of truth**.

**Rule: Only the node whose ID matches `vm.NodeId` may run the VM.**

```
┌───────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR DB                          │
│                                                               │
│   VM-123:                                                     │
│     NodeId: "node-B"    ← Authoritative assignment           │
│     TargetNodeId: null                                        │
│     Status: Running                                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
 ┌─────────────────┐         ┌─────────────────┐
 │     Node A      │         │     Node B      │
 │                 │         │                 │
 │  "Am I node-B?" │         │  "Am I node-B?" │
 │   NO → STOP VM  │         │   YES → RUN OK  │
 │                 │         │                 │
 └─────────────────┘         └─────────────────┘
```

This approach requires **no new services, no lease renewals, no fencing tokens** — just enforcement of an existing field.

### 7.3 Node Agent Implementation

#### 7.3.1 VM Reconciliation on Startup and Heartbeat

The Node Agent reconciles local VMs against the orchestrator's authoritative state:

```csharp
// In HeartbeatService or a dedicated VmReconciliationService

private async Task ReconcileLocalVmsAsync(CancellationToken ct)
{
    // Get all VMs running locally via libvirt
    var localVms = await _libvirtManager.GetAllVmsAsync();
    
    foreach (var localVm in localVms)
    {
        // Skip system VMs (relay, DHT, blockstore) — managed separately
        if (IsSystemVm(localVm.VmId))
            continue;
            
        // Query orchestrator for authoritative VM state
        var vmInfo = await _orchestratorClient.GetVmInfoAsync(localVm.VmId, ct);
        
        if (vmInfo == null)
        {
            // VM deleted from orchestrator — destroy locally
            _logger.LogWarning(
                "VM {VmId} not found in orchestrator, destroying orphan",
                localVm.VmId);
            await DestroyAndCleanupVmAsync(localVm.VmId);
            continue;
        }
        
        if (vmInfo.NodeId != _nodeId)
        {
            // We are NOT the authorized host — stop immediately
            _logger.LogWarning(
                "VM {VmId} assigned to node {AuthorizedNode}, not us ({OurNode}). " +
                "Destroying zombie VM.",
                localVm.VmId, vmInfo.NodeId, _nodeId);
            
            await DestroyAndCleanupVmAsync(localVm.VmId);
        }
    }
}

private async Task DestroyAndCleanupVmAsync(string vmId)
{
    // Force destroy the VM
    await _libvirtManager.DestroyVmAsync(vmId);
    
    // Optionally clean up local disk to free space
    // (the authoritative copy is now on the new node or in block store)
    await _diskManager.CleanupVmDiskAsync(vmId);
    
    _logger.LogInformation("Zombie VM {VmId} destroyed and cleaned up", vmId);
}
```

#### 7.3.2 Reconciliation Triggers

Reconciliation runs at multiple points to ensure quick detection:

| Trigger | Purpose |
|---------|---------|
| **Node Agent startup** | Catch VMs that were migrated while node was offline |
| **Every heartbeat cycle** | Continuous verification (every 30-60 seconds) |
| **On network reconnection** | After recovering from network partition |

#### 7.3.3 Fail-Safe Behavior

If the Node Agent cannot reach the orchestrator to verify VM ownership:

```csharp
private async Task ReconcileLocalVmsAsync(CancellationToken ct)
{
    foreach (var localVm in localVms)
    {
        try
        {
            var vmInfo = await _orchestratorClient.GetVmInfoAsync(localVm.VmId, ct);
            // ... normal reconciliation logic
        }
        catch (HttpRequestException ex) when (IsTransientError(ex))
        {
            // Cannot reach orchestrator — what to do?
            
            // Option A: Keep running (availability over consistency)
            // Suitable if network partitions are brief
            _logger.LogWarning(
                "Cannot verify VM {VmId} ownership — keeping running. " +
                "Will retry on next heartbeat.",
                localVm.VmId);
            
            // Option B: Stop after grace period (consistency over availability)
            // if (localVm.LastVerifiedAt < DateTime.UtcNow.AddMinutes(-5))
            // {
            //     await DestroyAndCleanupVmAsync(localVm.VmId);
            // }
        }
    }
}
```

**Recommendation:** Use Option A (keep running) for most cases. Brief network partitions are more common than actual migrations. The zombie will be cleaned up when connectivity resumes.

### 7.4 Migration Sequence with NodeId Update

```
BEFORE MIGRATION:
┌─────────────────────────────────────────────────────────────┐
│  VM-123:                                                     │
│    NodeId: "node-A"        ← A is authorized                │
│    TargetNodeId: null                                        │
│    Status: Running                                           │
└─────────────────────────────────────────────────────────────┘

STEP 1: Orchestrator detects Node A failure (missed heartbeats)

STEP 2: Orchestrator sets migration target
┌─────────────────────────────────────────────────────────────┐
│  VM-123:                                                     │
│    NodeId: "node-A"                                          │
│    TargetNodeId: "node-B"  ← Migration target set           │
│    Status: Migrating                                         │
└─────────────────────────────────────────────────────────────┘

STEP 3: Orchestrator atomically transfers authority
┌─────────────────────────────────────────────────────────────┐
│  VM-123:                                                     │
│    NodeId: "node-B"        ← B is NOW authorized            │
│    TargetNodeId: null      ← Cleared                        │
│    Status: Migrating                                         │
└─────────────────────────────────────────────────────────────┘

STEP 4: Orchestrator sends CreateVm command to Node B
  - Node B downloads overlay from block store (confirmed manifest)
  - Node B regenerates cloud-init from VM labels
  - Node B boots VM

STEP 5: Node B reports VM running → Status: Running

STEP 6: Node A comes back online
  - Heartbeat triggers reconciliation
  - For each local VM: query orchestrator
  - VM-123.NodeId = "node-B", but we're "node-A"
  - Result: destroy local VM-123
```

### 7.5 Lazysync Fencing

The lazysync daemon must also respect NodeId authority. A zombie VM should not push manifest updates:

```csharp
// In LazysyncDaemon on Node Agent

private async Task SyncVmAsync(string vmId, CancellationToken ct)
{
    // Verify we're still the authoritative host before pushing
    var vmInfo = await _orchestratorClient.GetVmInfoAsync(vmId, ct);
    
    if (vmInfo?.NodeId != _nodeId)
    {
        _logger.LogWarning(
            "Skipping lazysync for VM {VmId} — we're not the authorized host",
            vmId);
        return;
    }
    
    // Proceed with dirty block export and manifest push
    // ...
}
```

The orchestrator also validates manifest updates:

```csharp
// In BlockStoreController.cs

[HttpPost("manifest")]
public async Task<IActionResult> RegisterManifest([FromBody] ManifestUpdateRequest request)
{
    var vm = await _dataStore.GetVmAsync(request.VmId);
    
    if (vm == null)
        return NotFound();
    
    // Verify the reporting node is the authorized host
    if (vm.NodeId != request.NodeId)
    {
        _logger.LogWarning(
            "Rejecting manifest update for VM {VmId} from node {ReportingNode} — " +
            "authorized node is {AuthorizedNode}",
            request.VmId, request.NodeId, vm.NodeId);
        
        return StatusCode(403, new { error = "Not authorized to update this VM's manifest" });
    }
    
    // Process manifest update
    // ...
}
```

### 7.6 Network Fencing (Defense in Depth)

As an additional layer, the orchestrator can revoke the zombie VM's network access:

```
On migration:
1. Orchestrator updates vm.NodeId from A → B (primary fencing)
2. Orchestrator notifies relay VMs: revoke WireGuard peer for Node A's VM-1 subnet
3. Even if zombie VM keeps running, it cannot reach the network

Result: Zombie VM is isolated — can run locally but cannot:
- Communicate with users
- Access external services
- Push lazysync updates
- Cause harm
```

This is implemented by extending the relay peer management:

```csharp
// In RelayNodeService.cs

public async Task RevokeVmNetworkAccessAsync(string vmId, string oldNodeId, CancellationToken ct)
{
    var vm = await _dataStore.GetVmAsync(vmId);
    if (vm == null) return;
    
    // Find the relay serving the old node
    var oldNode = await _dataStore.GetNodeAsync(oldNodeId);
    if (oldNode?.CgnatInfo?.RelayNodeId == null) return;
    
    var relay = await _dataStore.GetNodeAsync(oldNode.CgnatInfo.RelayNodeId);
    if (relay?.RelayNodeInfo?.RelayVmId == null) return;
    
    // Remove the VM's WireGuard peer from the relay
    // (The VM's tunnel IP was associated with the old node)
    await RemoveWireGuardPeerAsync(relay, vm.NetworkConfig.TunnelIp, ct);
    
    _logger.LogInformation(
        "Revoked network access for VM {VmId} on old node {OldNode}",
        vmId, oldNodeId);
}
```

### 7.7 Optional Enhancement: Assignment Version

For extra protection against race conditions (e.g., NodeId bouncing due to flapping), add a version counter:

```csharp
public class VirtualMachine
{
    public string NodeId { get; set; }
    public string? TargetNodeId { get; set; }
    
    /// <summary>
    /// Incremented on every NodeId change. Used as fencing token
    /// to detect stale assignments.
    /// </summary>
    public long NodeAssignmentVersion { get; set; } = 0;
}
```

The Node Agent stores the version when it receives a CreateVm command. On reconciliation:

```csharp
if (localAssignmentVersion < orchestratorAssignmentVersion)
{
    // Our assignment is stale — we were migrated away and back
    // but our local state is from a previous incarnation
    await DestroyAndCleanupVmAsync(vmId);
}
```

This handles edge cases where:
- VM migrates A → B → A (same NodeId, but different incarnation)
- Node has stale state from before migration

### 7.8 Orchestrator Heartbeat Response Enhancement

To reduce per-VM queries, the orchestrator can proactively inform nodes about invalid VMs:

```csharp
public class HeartbeatResponse
{
    // Existing fields...
    public List<NodeCommand>? PendingCommands { get; set; }
    public CgnatConfigResponse? CgnatConfig { get; set; }
    
    /// <summary>
    /// VM IDs that this node should NOT be running.
    /// Includes VMs that were migrated away, deleted, or never assigned to this node.
    /// Node agent should destroy any of these found locally.
    /// </summary>
    public List<string>? InvalidVmIds { get; set; }
}
```

The orchestrator populates this by comparing the node's reported running VMs against its authoritative records:

```csharp
// In HeartbeatService on Orchestrator

private List<string> GetInvalidVmsForNode(string nodeId, List<string> reportedRunningVmIds)
{
    var invalidVms = new List<string>();
    
    foreach (var vmId in reportedRunningVmIds)
    {
        var vm = _dataStore.GetVm(vmId);
        
        // VM doesn't exist or belongs to different node
        if (vm == null || vm.NodeId != nodeId)
        {
            invalidVms.Add(vmId);
        }
    }
    
    return invalidVms;
}
```

### 7.9 Summary

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| **Primary** | `NodeId` field check | Only authorized node runs VM |
| **Reconciliation** | Startup + heartbeat checks | Detect and destroy zombies |
| **Lazysync** | NodeId validation | Reject manifest updates from zombies |
| **Network** | WireGuard peer revocation | Isolate zombies from network |
| **Optional** | `NodeAssignmentVersion` | Handle edge cases with flapping |

**Key principle:** The existing `NodeId` field is the authoritative lease. No separate lease service needed — just enforcement of what already exists.

---

## 8. Implementation Order

### Phase A: Orchestrator — Core Block Store

1. **BlockStoreVmSpec.cs** — Resource specification (5% duty, 512 MB RAM)
2. **BlockStoreInfo** on Node.cs — Model for tracking state (capacity, usage, status)
3. **VmType.BlockStore** — Add to enum
4. **IBlockStoreService / BlockStoreService** — Deployment, manifest lifecycle, replication audit, migration planning
5. **BlockStoreController.cs** — `/join`, `/announce`, `/locate`, `/manifest`, `/audit`, `/stats`
6. **SystemVmLabelSchema** — Add required labels
7. **ObligationEligibility** — Uncomment BlockStore eligibility, lower RAM threshold to 2 GB
8. **SystemVmReconciliationService** — Wire up deployment method
9. **VmService** — Add BlockStore to isSystemVm check
10. **Program.cs** — Register DI

### Phase B: NodeAgent — VM Template & Binary

11. **VmType.BlockStore** — Add to enum
12. **blockstore-vm-cloudinit.yaml** — Cloud-init template
13. **blockstore-node Go binary** — Core storage engine with DAG/manifest support, GossipSub subscription for `decloud/blockstore/new-blocks`, adaptive XOR threshold pull logic
14. **blockstore-bootstrap-poll.sh** — Orchestrator peer discovery
15. **blockstore-notify-ready.sh** — Callback to NodeAgent
16. **blockstore-health-check.sh** — Health monitoring
17. **blockstore-dashboard.py** — Status dashboard
18. **BlockStoreCallbackController.cs** — NodeAgent callback endpoint
19. **CommandProcessorService** — Handle VmType.BlockStore in CreateVm

### Phase C: Basic Integration Testing

20. Build Go binary, encode as gzip+base64
21. Test deployment on single node
22. Test multi-node block exchange via bitswap
23. Test local LRU GC (eviction under capacity pressure, provider record withdrawal)
24. Test self-healing (reconciliation redeploy)

### Phase D: Lazysync & Migration

### Phase D: Lazysync & Migration

25. **ReplicationFactor on VmSpec/VirtualMachine** — add field (default 3), add to
    `CreateVmRequest` (optional, validated: 0/1/3/5), set in `VmService.CreateVmAsync`
    for non-system VMs, add `LastKnownOverlayBytes` + `LastLazysyncAt` fields
26. **Scheduler BlockStore filter** — `VmSchedulingService` rejects nodes without Active
    BlockStore when `replicationFactor > 0`; ephemeral VMs (factor=0) bypass filter
27. **Block-based storage pricing** — replace overlay_gb estimate with exact
    block-count billing: `blockCount × blockSizeKb / 1024 × replicationFactor × costPerMbPerHour`.
    Add `StoragePerMbPerHour` to `PricingConfig`. Add `BlockCount` + `BlockSizeKb` to
    `ManifestRecord`. Update `CalculateHourlyRate` to use block count once available
    (falls back to `DiskBytes × 5% / blockSizeBytes` estimate before first lazysync cycle).
    `replicationFactor=0` VMs pay zero storage cost.
28. **LazysyncManager** in orchestrator — manifest version tracking, confirmed vs current,
    provider count audit loop using `replicationFactor` as N; MongoDB persistence for
    ManifestRecord (replacing in-memory stub from Phase A); updates `BlockCount` on
    `ManifestRecord` and `CurrentManifestBlockCount` on `VirtualMachine` each cycle
    for accurate billing
29. **Lazysync daemon** on node agent — QEMU QMP integration (dirty bitmaps, incremental
    backup), overlay chunking via `qemu-img map` for sparse cluster discovery, block push,
    manifest update cycle. Skip VMs with `replicationFactor=0` entirely
30. **Initial overlay seeding** — push allocated overlay clusters on first enrollment,
    progress tracking, rate limiting
31. **GossipSub scatter propagation** — block store publishes new CIDs to
    `decloud/blockstore/new-blocks`; receiving nodes evaluate XOR distance + adaptive
    threshold; pull via bitswap if close enough
31a. **settleCycle() integration** — upgrade `BlockchainService.ExecuteSettlementAsync`
     to call `settleCycle()` instead of `batchReportUsage()`. Pass per-VM
     `blockCounts`, `blockSizeKbs`, `replicationFactors` arrays alongside compute
     amounts. Pass per-node `storageBytes` array from `node.BlockStoreInfo.UsedBytes`.
     Update `BlockchainService` ABI to include `settleCycle` signature.
     Update `SettlementService.RecordUsageAsync` to record compute and storage
     components separately for cycle aggregation.
31b. **GossipSub fetch retry queue** — block store binary maintains an in-memory
     retry queue (`chan cid.Cid`, capacity 500) fed by two failure paths:
     (a) semaphore skip (`bitswap_fetch_skip`) — CID received via GossipSub but all
     concurrent fetch slots were occupied at arrival;
     (b) fetch timeout (`bitswap_fetch_fail`) — slot acquired but `GetBlock` exceeded
     30s deadline, typically on the first cross-region DHT walk before the Kademlia
     routing table is warm.
     A `startRetryLoop` background goroutine ticks every 60 seconds (after a 30s
     startup delay for routing table warmup) and drains the queue **sequentially**
     (concurrency=1). Sequential processing is intentional: the first retry performs
     the DHT walk and caches the provider record; all subsequent retries in the same
     batch resolve via the warm cache in under 8ms each.
     Per-CID attempt tracking (sync.Map, string→int) caps retries at 3. On max
     retries exceeded the CID is logged (`retry_give_up`) and abandoned; the DHT
     neighborhood scan (item 32) handles anything that genuinely cannot be reached.
     Retry fetches use a 60s timeout (vs 30s for the initial GossipSub path) since
     the routing table is warm at retry time and the longer window handles cross-region
     high-latency links. Successfully recovered blocks are announced to the DHT with
     role `retry_pull` and are eligible for GC like any other block.
     New diag events: `retry_fetch`, `retry_fetch_fail`, `retry_give_up`,
     `retry_queue_full`.
     **Relation to DHT:** The retry queue is the narrow (failure-triggered) recovery
     path. It only recovers blocks where GossipSub delivery was confirmed but bitswap
     failed. The DHT neighborhood scan (item 32) is the general recovery path for
     blocks where GossipSub was missed entirely. The two are complementary and
     non-redundant — GossipSub → retry queue is the hot path, neighborhood scan is
     the cold path.
32. **DHT proximity endpoint** — `GET /proximity/{cid}` on DHT VM for neighborhood scan
    and diagnostics
33. **Migration planner** — PlanMigrationAsync with scheduling fit + resource headroom;
    requires candidate node to have Active BlockStore when migrating replicated VMs
34. **Source-offline alerting** — detect under-replicated VMs on node failure:
    `replicationFactor=0` → LOST (ephemeral, no alert);
    `confirmedVersion=0` → UNRECOVERABLE;
    `confirmedVersion < currentVersion` → RECOVERING (partial data loss);
    `confirmedVersion == currentVersion` → MIGRATING (no data loss)
35. **Disk reconstruction** — fetch overlay chunks from block store via bitswap + write to
    new qcow2 at correct offsets + regenerate cloud-init ISO from VM labels
36. **End-to-end migration test** — simulate node failure, verify VM rescheduling from
    confirmed manifest (base image + overlay from bitswap + cloud-init)
37. **Owner-indexed block deletion** — blockstore binary maintains a per-VM owner index
    (`/var/lib/decloud-blockstore/owners/{vmId}.cids`, append-only, newline-delimited)
    updated on every `POST /blocks?cid={cid}&owner={vmId}` write. On VM deletion,
    NodeAgent calls `DELETE /owners/{vmId}` — blockstore reads the owner file, deletes
    each referenced block from FlatFS, withdraws DHT provider records, and removes the
    owner file. Single HTTP call regardless of block count (3,420 blocks = 1 call).
    Reference counting: a block shared by multiple owners (rare in practice — VM overlays
    are unique) is only evicted from FlatFS when its last owner is deleted.
    Orchestrator hook: `VmLifecycleManager.OnVmDeletedAsync` calls
    `DataStore.DeleteManifestAsync(vmId)` to remove the MongoDB manifest record.
    NodeAgent hook: `HandleDeleteVmAsync` calls `DELETE /owners/{vmId}` on the local
    blockstore after successful VM deletion, before cleaning up the VM directory.
    Changes: `blockstore-node main.go` (owner index write + DELETE endpoint),
    `LazysyncDaemon.cs` (append `?owner={vmId}` to block push URL),
    `LibvirtVmManager.DeleteVmAsync` (call blockstore owner delete),
    `VmLifecycleManager.OnVmDeletedAsync` (call DeleteManifestAsync).

### Phase E: Split-Brain / Zombie VM Prevention

34. **VmReconciliationService** on Node Agent — startup + heartbeat reconciliation loop
35. **GetVmInfoAsync** endpoint — lightweight VM ownership query for reconciliation
36. **InvalidVmIds in HeartbeatResponse** — proactive notification of zombie VMs
37. **Lazysync NodeId validation** — reject manifest updates from non-authoritative nodes
38. **BlockStoreController NodeId check** — enforce on manifest POST endpoint
39. **Network fencing** (optional) — WireGuard peer revocation on migration
40. **NodeAssignmentVersion** (optional) — fencing token for edge cases
41. **Integration test** — simulate node failure, migration, node return, verify zombie destroyed

---

## 9. How This Connects to Future Features

### Built-In (Phase A-E)
- **Continuous VM disk replication (lazysync)** — Core feature, not optional
- **Live migration on node failure** — Core feature, the primary purpose of the block store
- **Template image distribution** — Distribute base images via block store (dedup across nodes)
- **Split-brain prevention** — Zero-config zombie VM detection and cleanup

### Near-Term Extensions
- **On-demand VM migration** — User-initiated, not just failure-driven
- **Point-in-time recovery** — Retain historical manifest versions for rollback (keep N confirmed manifests instead of GC'ing immediately)
- **Content delivery** — Serve static content from nearest block store node
- **User file storage** — Simple put/get API for users to store data persistently

### Long-Term Vision
- **Full IPFS compatibility** — Upgrade to IPFS node for interoperability
- **Erasure coding** — Reed-Solomon coding for storage efficiency (reduce replication overhead)
- **Storage marketplace** — Users pay for persistent storage, node operators earn

---

## 10. Key Design Decisions

### Decision 1: Separate VM from DHT (not embedded)
**Why:** DHT VMs are lightweight (512 MB RAM, 2 GB disk). Block storage needs different resources (more disk, similar RAM). Keeping them separate means:
- DHT stays fast and lean (just routing)
- Block store can be deployed selectively (only on nodes with ≥100 GB storage)
- Independent failure domains (block store crash doesn't break DHT)
- Different scaling profiles

### Decision 2: 5% storage duty (not configurable, not percentage of free space)
**Why:** A fixed 5% of total storage creates a predictable, bounded obligation:
- Every eligible node contributes proportionally — fair across hardware tiers
- The orchestrator can calculate aggregate network capacity deterministically
- No node sacrifices significant resources — 5% is small enough to never conflict with user VMs
- The aggregate across many nodes creates substantial capacity (1,000 nodes × 1 TB avg = 50 TB)
- Nodes hold OTHER nodes' replicated data — this is a collective duty, not local storage

### Decision 3: libp2p bitswap (not custom protocol)
**Why:** Bitswap is battle-tested by IPFS with millions of nodes. It handles:
- Want/have negotiation
- Block prioritization
- Peer reputation
- Parallel downloads from multiple peers
Using a standard protocol also opens the door to IPFS interop later.

### Decision 4: FlatFS storage backend (not LevelDB for blocks)
**Why:** FlatFS stores each block as a separate file on disk. This is:
- Simple to debug (blocks are just files)
- Easy to backup/restore (cp -r)
- Good for large blocks (no LSM tree overhead)
- Proven by IPFS Kubo for years
LevelDB is only used for metadata (access times, block index, stats).

### Decision 5: DHT-native scatter replication (not orchestrator-directed placement)
**Why:** Decoupling replication from scheduling creates a simpler, more resilient system:
- **Storage ≠ compute**: a 2-core node can store chunks for a 16-core VM. Restricting replicas to scheduling-eligible nodes wastes the storage capacity of smaller nodes and shrinks the replication pool
- **Kademlia XOR scatter is free**: the DHT's mathematical properties naturally distribute blocks across the network with high diversity. No placement logic needed
- **Larger replication pool = better durability**: blocks land on 10-20+ nodes scattered across the ID space, not 3 pre-selected "placement-aware" targets. An entire region can go offline without data loss
- **Faster migration**: bitswap fetches from many scattered providers in parallel, not 3 specific replicas
- **Simpler**: no `SelectReplicaNodesAsync`, no failure domain reasoning in the replication path, no `/replicate` commands. The block store just participates in Kademlia — blocks flow to where they belong
- The orchestrator's role reduces to **auditing** (checking provider counts) and **scheduling** (picking migration targets based on compute fit). Bitswap is both the transfer and replication mechanism

### Decision 6: Local LRU GC + orchestrator audit (not orchestrator-coordinated pinning)
**Why:** With scatter replication across many nodes, local GC is safe:
- Blocks have 10-20+ providers scattered across the network. One node evicting a block via LRU has negligible impact on durability
- Kademlia's provider record re-publication naturally recovers from provider loss — nearby nodes discover the gap and pull from remaining providers
- No orchestrator pin/unpin lifecycle needed. Nodes manage their own 5% budget via LRU eviction
- Old blocks (no longer in any current manifest) naturally disappear: nodes stop re-announcing → provider records expire (TTL) → blocks become GC candidates everywhere
- The orchestrator audits provider counts and advances `confirmedVersion` — this is a read-only check, not a coordination step
- In the rare case of persistent under-replication, the orchestrator can publish a "re-replicate" hint — but this is a fallback, not the normal path

### Decision 7: Content-addressed with CIDv1
**Why:** Using IPFS-compatible CIDs (Content IDentifiers) means:
- Deduplication is automatic (same content = same CID)
- Integrity verification is built-in
- Future IPFS interop
- Industry standard format
- Critical for lazysync efficiency: unchanged overlay blocks across cycles share the same CID and are never re-transferred

### Decision 8: DAG manifests from day one (not bolted on later)
**Why:** VM disk state is a structured collection of blocks, not a flat blob. Supporting DAG manifests in the initial design means:
- The single evolving manifest per VM is clean and native
- The block store understands "pin this DAG" (manifest + all referenced blocks)
- Future features (large file storage, directory trees) get DAG support for free
- Avoids painful migration from flat-block-only to DAG-aware later

### Decision 9: Lazysync over periodic snapshots + delta chains
**Why:** The traditional approach of periodic full snapshots with delta chains between them has inherent complexity: delta chains that need consolidation, snapshot events that create I/O bursts, two different manifest types, and recovery requiring chain traversal. Lazysync replaces this with a simpler model:
- **No snapshot events**: dirty blocks trickle out continuously in the background. A VM writing 1 GB/hour generates ~17 MB per 5-minute cycle — easily absorbed
- **No delta chains**: there's one manifest per VM. Chunk CIDs are replaced in-place. No chain to walk, no consolidation
- **Better recovery point**: ~5-7 minutes vs 30+ minutes with periodic deltas
- **Simpler GC**: blocks become unreferenced when their manifest slot is updated. Track "is this block referenced by any confirmed manifest?" — if not, GC it
- **Natural write coalescing**: if a chunk is written 100 times between cycles, only the final state is replicated
- **Simpler reconstruction**: read manifest, assemble chunks in offset order.

### Decision 10: Overlay-only replication (not full disk)
**Why:** Replicating only the overlay layer (not the full virtual disk) reduces storage and bandwidth by 90%+:
- **Base images are shared**: `debian-12-generic.qcow2` is the same for thousands of VMs. Store it once in the image registry, not replicated per-VM
- **Overlays are sparse**: a 100 GB virtual disk might have 3 GB of actual writes. Replicate 3 GB, not 100 GB
- **Cloud-init is regenerable**: the orchestrator has the VM's labels — regenerate the ISO in milliseconds
- `qemu-img map` discovers allocation without reading the full virtual disk
- **Natural fit with qcow2**: QEMU already tracks dirty blocks at the overlay level. The dirty bitmap covers overlay writes only — base image reads are invisible to CBT

### Decision 11: GossipSub fast path + DHT durable fallback (not pure Kademlia)
**Why:** Standard Kademlia provider records are passive — they answer queries but don't push notifications. Waiting for periodic DHT scans to discover new blocks would add minutes of latency to replication. GossipSub provides near-instant notification:
- Source block store publishes new CIDs to `decloud/blockstore/new-blocks` topic
- All subscribed block store nodes receive the notification in seconds
- Nodes evaluate XOR distance locally and pull if close enough
- GossipSub is ephemeral (missed messages are lost), so the DHT serves as the durable fallback: periodic neighborhood scans catch anything missed
- This dual-path approach gives both speed (GossipSub) and reliability (DHT provider records)

### Decision 12: Per-VM configurable replication factor (not fixed network-wide)
**Why:** Different workloads have different durability requirements and cost tolerances:
- N=0 (ephemeral): development VMs, CI runners — no replication cost
- N=3 (default): production workloads — standard durability
- N=5+ (premium): databases, stateful services — higher confidence
- The cost lever is primarily N=0 vs N>0 (whether lazysync runs at all). Once running, incremental cost of higher N is minimal — Kademlia scatters to many nodes regardless, N just sets the confirmation threshold
- Ties directly to pricing tiers without complicating the replication mechanism

### Decision 13: NodeId as authority (not separate lease service)
**Why:** The `VirtualMachine.NodeId` field already tracks which node is authorized to run a VM. Using this existing field as the fencing mechanism:
- Requires no new services or database tables
- Leverages existing heartbeat/reconciliation infrastructure
- Simple to understand: "if NodeId != myNodeId, stop the VM"
- Eliminates lease renewal complexity and potential lease expiration edge cases
- Already used throughout the codebase for VM-to-node association

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Block store VM crash | Medium | Low | Self-healing reconciliation redeploys; blocks scattered across many providers |
| Node storage full | Medium | Low | Quota enforcement (hard refuse at 95%); LRU GC naturally trims excess as nodes fill up |
| Source offline before scatter | Low | Medium | GossipSub + bitswap propagation completes in seconds; window is small. Orchestrator alerts on confirmedVersion=0 + node failure. Cannot be fully eliminated without synchronous replication |
| GossipSub message loss | Low | Low | GossipSub fetch retry queue recovers failed fetches within 60s (hot path); DHT neighborhood scan (Phase D+1) catches entirely missed messages within 10 minutes (cold path) |
| Ephemeral VM data loss (N=0) | N/A | N/A | By design — user explicitly chose N=0. Dashboard shows ephemeral status. No alert on node failure |
| Split-brain / zombie VM | Medium | High | NodeId authority enforcement, reconciliation on startup/heartbeat, lazysync fencing, optional network fencing |
| False positive zombie detection | Low | High | Conservative approach: keep running if orchestrator unreachable; version tracking for edge cases |
| Reconciliation delays | Low | Medium | Multiple reconciliation triggers (startup, heartbeat, reconnection); InvalidVmIds in heartbeat response for proactive notification |

---

## 12. Success Metrics

### Phase A-C (Core Block Store)
- Block Store VM deploys on all eligible nodes (≥100 GB storage, ≥2 GB RAM)
- 5% storage allocation is enforced — no node exceeds its duty
- Blocks survive node restart (persistent FlatFS)
- Cross-node block retrieval works via bitswap
- GossipSub propagation delivers new block notifications to subscribed nodes within seconds
- Adaptive XOR threshold correctly adjusts pull behavior based on local capacity
- Local LRU GC works correctly — evicts least-recently-used blocks, enforces 5% budget
- Bootstrap polling discovers peers within 60 seconds
- Self-healing redeploys on VM failure
- DHT proximity endpoint returns accurate distance and rank estimates

### Phase D (Lazysync & Migration)
- Lazysync daemon continuously replicates dirty overlay blocks for all running VMs
- Lazysync cycle completes within configured interval (~5 minutes) under normal load
- QEMU incremental backup produces crash-consistent dirty block exports without VM pause
- Confirmed manifest version stays within 2 versions of current (Kademlia scatter keeps up)
- All chunks in confirmed manifests have ≥N providers in the DHT (verified by audit loop)
- Blocks scatter across the network via Kademlia XOR proximity — no orchestrator-directed placement
- When a node goes offline, its VMs are rescheduled to a scheduling-eligible node
- Migration target selected by scheduling fit + resource headroom (bitswap fetches overlay from scattered providers)
- VM disk reconstruction from confirmed manifest completes successfully (base image + overlay via bitswap + cloud-init regeneration)
- Initial overlay seeding of new VMs completes within minutes (not hours)
- Recovery point objective: ≤10 minutes of data loss under normal conditions
- Overlay-only replication keeps per-VM storage cost at 1-10% of virtual disk size
- Self-healing: provider count recovers autonomously when nodes leave/rejoin the network
- Per-VM replication factor correctly enforced:
    N=0 → lazysync skipped entirely, VM flagged Ephemeral on dashboard
    N=1 → lazysync runs, confirmedVersion advances when ≥1 provider confirmed
    N=3 → default; confirmedVersion advances when all chunks have ≥3 providers
    N=5 → high availability; confirmedVersion advances when all chunks have ≥5 providers
- Scheduler rejects nodes without Active BlockStore for VMs with replicationFactor > 0
- Storage replication cost billed correctly: overlay_gb × replication_factor × storage_rate
- Source-offline alerting correctly classifies VMs as UNRECOVERABLE / RECOVERING /
  MIGRATING / LOST based on confirmedVersion, currentVersion, and replicationFactor
- `replicationFactor=0` VMs correctly classified as LOST on node failure with no alert
  (user opted in to ephemeral semantics at creation time)
- Block-based billing is exact: `blockCount × blockSizeKb / 1024 × replicationFactor × costPerMbPerHour`
  matches the actual manifest chunk count — no estimation
- `settleCycle()` atomically collects storage fees and distributes storage pool in one
  on-chain transaction per billing cycle
- Storage pool distribution is proportional: `reward = pool × (node.UsedBytes / Σ UsedBytes)`
  calculated on-chain from raw byte inputs — trustless
- `StorageRewarded` events emitted per-node per-cycle, fully auditable from chain
- Block size enforcement: `blockSizeBytes` matches manifest type constant at write time;
  non-conforming writes rejected by block store binary
- Model shard chunk counts are manageable: Llama-3 70B Q4 = 640 × 64 MB blocks;
  manifest fits in memory; DHT provider records scale linearly

### Phase D-fixes (Replication Reliability) — Production-verified 2026-04-08

- Lazysync daemon replicates overlay data only — base OS image excluded via
  `qemu-img map` depth=0 whitelist; initial seed ~30 MB not ~2 GB
- GossipSub-triggered bitswap fetches target correct peer via DHT session, not
  the relay hop; all successful fetches show correct `source` node ID in diag log
- Blockstores establish direct cross-region libp2p connections on boot; both nodes
  show each other's WireGuard IP in peer list within 60 seconds of startup
- `bitswapSent` counter correctly reflects blocks served via bitswap
- GossipSub fetch retry queue recovers blocks that timed out on initial fetch;
  `retry_fetch` events visible in diag log within one retry cycle (60s)
- `retry_give_up` rate is 0 under normal cross-region conditions (routing table
  warm after first retry; subsequent retries resolve in <8ms)

### Production Bug Fixes (2026-04-08)

Discovered and fixed during initial lazysync seeding tests across two nodes
(us-east-1 and tr-south) with a VM at replicationFactor=1.

**Bug 1 — LazysyncDaemon replicated full backing chain, not overlay only**
`qemu-img convert` merges the backing chain into a flat raw file. The `-S 65536`
flag skips zero/unallocated regions, not backing-file extents. The base OS image
(~2 GB of non-zero EXT4 data) was replicated every cycle alongside actual overlay
writes. Fix: run `qemu-img map --force-share --output=json` first to build a
whitelist of depth=0 chunk offsets; `ScanChunksAsync` skips any offset not in the
whitelist. File: `LazysyncDaemon.cs`.

**Bug 2 — Bitswap fetches targeted the GossipSub relay hop, not the block provider**
Blockstores were only connected to their co-located DHT VM. GossipSub messages
propagated via the DHT relay chain: source blockstore → source DHT → (WireGuard) →
target DHT → target blockstore. `msg.ReceivedFrom` in the GossipSub receive loop
was the target's local DHT VM peer ID, not the source blockstore. `sessionForPeer`
sent WANT messages to that peer — which holds no blocks — causing guaranteed 30s
timeouts on every fetch. Fix: remove the `nodeIDToPeerID` map write and always use
`bsExch.NewSession(ctx)` in `handleNewBlockAnnouncement`, which performs a DHT
FindProviders walk to locate the actual provider. File: `main.go`.

**Bug 3 — Bitswap port 5001 blocked by host firewall**
`blockstore-vm-cloudinit.yaml` runcmd had no `ufw allow` rule for port 5001.
`install.sh` only opens agent port 5100/tcp and WireGuard 51820/udp. Direct cross-
region libp2p connections were dropped at the host firewall; blockstores remained
permanently isolated from each other via bitswap. Fix: add
`ufw allow __BLOCKSTORE_LISTEN_PORT__/tcp` to cloud-init runcmd.
File: `blockstore-vm-cloudinit.yaml`.

**Bug 4 — Bootstrap poll backed off too aggressively on simultaneous boot**
`POLL_INTERVAL_CONNECTED=300s` (5 minutes). When two blockstores booted
simultaneously and one joined the orchestrator before the other was registered as
Active, the first received 0 bootstrap peers, set `INITIAL_POLL_DONE=true`, then
backed off for 5 minutes while the second blockstore became discoverable within
seconds. Fix: reduce `POLL_INTERVAL_CONNECTED` to 60s. File:
`blockstore-bootstrap-poll.sh`.

**Bug 5 — `bitswapSent` counter never incremented**
`n.bitswapSent uint64` field was declared in `BlockNode` but never written; all
three handlers reported 0 permanently. Fix: remove field, replace with
`n.bsExch.Stat().BlocksSent` read in `handleHealth`, `handleStats`, and
`handleDiagnostics`. File: `main.go`.

**Bug 6 — GossipSub fetch concurrency caused burst timeouts**
All GossipSub-triggered fetch goroutines fired simultaneously on message burst.
With 100 messages arriving in one batch and each requiring a 30s DHT walk over WAN,
concurrent walks saturated the link and all timed out. Fix: add `fetchSem chan struct{}`
(capacity `GossipSubFetchConcurrency=4`) with blocking `select` and 5-minute wait
timeout, so goroutines queue rather than drop. File: `main.go`.

### Phase E (Split-Brain Prevention)
- Zombie VMs destroyed within 60 seconds of node reconnection
- No split-brain events in production (two instances of same VM never run simultaneously)
- Lazysync manifest updates from zombie nodes rejected 100%
- Network fencing (if enabled) isolates zombies within 30 seconds of migration
- Reconciliation adds < 100ms to heartbeat cycle
- False positive rate (valid VMs incorrectly destroyed): 0%