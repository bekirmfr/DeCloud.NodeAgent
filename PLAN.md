# Block Store System VM — Design & Implementation Plan

**Date:** 2026-02-16
**Status:** Design Phase
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
│                    Node Agent                                      │
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
│  │    DELETE /blocks/{cid} (unpin block)                          │ │
│  │    GET  /blocks         (list local blocks)                    │ │
│  │    POST /dag            (put manifest + blocks atomically)     │ │
│  │    GET  /dag/{cid}      (resolve DAG, return manifest)        │ │
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
5. **Pulls blocks autonomously** — when provider records appear for CIDs close to this node in Kademlia XOR space, the block store fetches and stores them via bitswap

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
  Request:  { vmId, rootCid, version, changedBlockCids: ["bafk..."], totalBytes }
  Auth:     Internal
  Response: { success, manifestVersion }
  Purpose:  Register an updated VM manifest from a lazysync cycle.
            Orchestrator records the new version. No replication plan is
            returned — blocks replicate via Kademlia scatter autonomously.

GET /api/blockstore/manifest/{vmId}
  Response: { vmId, currentVersion, confirmedVersion, confirmedRootCid,
              timestamp, replicationStatus: { targetFactor: 3,
              providerCounts: { "bafk...": 12, ... } } }
  Purpose:  Query manifest version and replication status for a VM.
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
6. Autonomous block pulling (fetch blocks close to peer ID in Kademlia XOR space)
7. Localhost HTTP API on port 5090
8. Provider record announcement via DHT connection
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
  → { cid, unpinned: true, deleted: true }

GET  /blocks
  Query: ?offset=0&limit=100
  → { blocks: [{ cid, size, pinned, createdAt }], total: 1234 }

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
      pinnedCount, connectedPeers, bitswapSent, bitswapReceived }

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
  "version": 47,
  "timestamp": "2026-02-16T12:05:00Z",
  "baseImageId": "debian-12-generic",
  "baseImageHash": "sha256:a1b2c3...",
  "overlayVirtualSizeBytes": 107374182400,
  "blockSizeBytes": 1048576,
  "chunks": [
    { "offset": 0,       "cid": "bafk...aaa" },
    { "offset": 3145728, "cid": "bafk...ggg" },
    { "offset": 8388608, "cid": "bafk...ccc" }
  ]
}
```

**Key properties:**
- **`type: "vm-overlay"`** — not `vm-disk`. This manifest represents the overlay layer only.
- **`baseImageId` + `baseImageHash`** — identifies the backing image. The target node downloads this from the image registry (or from block store peers that have it cached). The orchestrator knows the URL.
- **Sparse chunks** — only offsets with allocated clusters appear. A 100 GB virtual disk with 3 GB of overlay writes has ~3,000 chunks, not 100,000. Offsets not in the manifest read through to the base image.
- Cloud-init ISO is not in the manifest — the orchestrator regenerates it from the VM's labels during migration.

The `version` field is a monotonically increasing integer, incremented on each lazysync cycle that produces changes. When the lazysync daemon detects dirty blocks, it:
1. Reads the dirty chunks (crash-consistent via QEMU incremental backup)
2. Hashes each chunk → CIDv1
3. Replaces the corresponding offset's CID in the manifest (if it actually changed)
4. Increments version
5. Pushes only the new blocks to the block store
6. Registers the updated manifest with the orchestrator

The orchestrator tracks two versions per VM:
- **currentVersion**: latest manifest registered (may be partially replicated)
- **confirmedVersion**: latest manifest where ALL referenced blocks are verified replicated on ≥N target nodes

Blocks referenced by a current manifest naturally maintain high provider counts (nodes keep re-announcing them, and they get recent bitswap access which protects them from LRU eviction). Blocks no longer in any current manifest stop being re-announced, their provider records expire, and they naturally disappear via LRU GC. This eliminates delta chains, delta consolidation, and the full/delta manifest type distinction entirely.

Content addressing provides automatic deduplication: if two VMs use the same base image and write the same packages, those overlay chunks share the same CID and are stored once. Unchanged overlay chunks across lazysync cycles share the same CID and are never re-stored or re-transferred.

#### Bitswap Integration

The Go binary connects to other Block Store peers via libp2p and uses the bitswap protocol for block exchange:

```go
// Bitswap network: blocks flow between peers automatically
// When a peer requests a block we have → serve it
// When we need a block we don't have → request from peers
bswap := bitswap.New(ctx, network, blockstore)
```

Bitswap is both the **transfer mechanism** and the **replication mechanism**. When a block is announced to the DHT, nodes close to the block's CID in Kademlia XOR space discover the provider record and pull it via bitswap autonomously. No orchestrator commands are needed — the DHT's natural properties scatter blocks across the network.

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

  5. Update manifest in-place:
     → manifest.chunks[offset] = newCID for each changed offset
     → manifest.version++
     → manifest.timestamp = now

  6. Register updated manifest with orchestrator:
     → POST /api/blockstore/manifest { vmId, rootCid, version,
         changedBlockCids, totalBytes }
     → Orchestrator records the new version (no replication plan returned)

  7. Blocks replicate via Kademlia scatter (autonomous, no orchestrator action):
     → Block store announced provider records in step 4
     → Nodes close to each block's CID in Kademlia XOR space discover
       the provider records and pull the blocks via bitswap
     → This happens naturally and continuously — not triggered by the orchestrator
     → Blocks scatter across many nodes (not limited to scheduling-eligible ones)

  8. Orchestrator audits replication asynchronously:
     → Background audit loop queries DHT: FindProviders(chunkCid)
       for each chunk in the latest manifest
     → When all chunks have ≥N providers → advance confirmedVersion
     → Under-replicated chunks are flagged but NOT actively pushed —
       Kademlia's re-publication handles recovery naturally

  9. Reset QEMU dirty bitmap for next cycle
```

**What this eliminates:**
- No full-disk replication — only overlay writes are tracked and replicated
- No "full snapshot" events — the first sync seeds allocated overlay clusters, subsequent syncs only push dirty blocks
- No "delta" manifests — there's one manifest per VM, chunks get replaced in-place
- No delta chains to traverse during reconstruction
- No delta consolidation — nothing to consolidate
- No snapshot-type field — just `vm-overlay` manifests with monotonically increasing version numbers

**Recovery point:** If lazysync runs every 5 minutes and replication takes ~2 minutes, the recovery point is typically 5-7 minutes behind live state. This is crash-consistent (QEMU guarantees this) — the same guarantees as traditional hypervisor crash recovery.

**Write coalescing:** If a VM writes the same chunk 100 times between lazysync cycles, only the final state is replicated. The intermediate writes are invisible — only the net change per cycle matters.

**Bandwidth:** A VM writing 1 GB/hour generates ~17 MB per 5-minute cycle. This trickles out continuously rather than bursting during snapshot events.

### 6.1.1 Initial Seeding

When a VM is first enrolled in lazysync, the **allocated clusters of its overlay** must be pushed to the block store. Because the overlay is sparse (qcow2 only allocates clusters that have been written to), and the base image is excluded, initial seeding is dramatically smaller than the virtual disk size:

```
VM with 100 GB virtual disk, freshly booted:
  Base image:      ~2 GB  (debian-12-generic — NOT replicated, downloadable)
  Overlay writes:  ~500 MB (cloud-init setup, package installs, config)
  Allocated overlay clusters: ~500 chunks (1 MB each)

  Initial seed: 500 MB → push to block store → Kademlia scatters to network
  Time: seconds to minutes, not hours

VM with 100 GB virtual disk, mature workload:
  Base image:      ~2 GB  (NOT replicated)
  Overlay writes:  ~5-15 GB (application data, logs, databases)
  Allocated overlay clusters: ~5,000-15,000 chunks

  Initial seed: 5-15 GB → push to block store → Kademlia scatters
  Time: minutes, depending on network

The lazysync daemon uses `qemu-img map` to discover which clusters
in the overlay are allocated (vs. reading through to the base image).
Only allocated clusters are chunked and replicated — zero/unallocated
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
     available_resources: enough CPU/RAM/disk headroom? (filter)
     base_image_cached:  does the node already have the base image? (bonus)

     score = resource_headroom * weight_resources
           + base_image_local * weight_image_cache

     Block locality is NOT a factor. Overlay blocks are scattered across
     many nodes via Kademlia — the target fetches them via bitswap from
     wherever they exist. With N scattered providers per block, the
     fetch fans out massively and parallelizes well.

  4. Select best candidate node (highest score that passes all filters)

  5. Reconstruct VM disk on target node:

     a) Base image — ensure available:
        The manifest contains baseImageId + baseImageHash.
        If the target node's image cache already has it → done (common case,
        many VMs share the same base image like debian-12-generic).
        If not → download from image registry (same path as normal VM creation).

     b) Overlay — assemble from block store network:
        Target node requests each chunk CID via bitswap.
        Bitswap parallelizes across all providers — blocks are scattered
        across many nodes, so the fetch is inherently distributed.
        Write overlay chunks at their manifest offsets into a new qcow2 file
        with backing file = base image path.

     c) Cloud-init ISO — regenerate:
        Orchestrator has all the VM's labels and configuration.
        Node agent renders cloud-init template from labels (same as initial
        VM creation). Generate new ISO.

     d) Assemble VM directory:
        /var/lib/decloud/vms/{vmId}/
        ├── disk.qcow2       (overlay, reconstructed from manifest blocks)
        ├── cloud-init.iso   (regenerated from labels)
        ├── domain.xml       (generated from VmSpec)
        └── metadata.json    (from orchestrator)

  6. Boot VM on new node.
     VM resumes from confirmed manifest state.
     (~5-7 minutes of state lost since last confirmed lazysync)
```

Reconstruction is fast because:
- **Base image**: likely already cached on the target node (shared across VMs of the same type). If not, it's a single download of a well-known artifact, not a block-by-block fetch.
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

5. **No coupling** — Storage and compute are independent concerns. A node's replication duty doesn't depend on what VMs it could run.

### 6.4 GC — Local LRU + Orchestrator Audit

Garbage collection is **fully local** — each node manages its own 5% budget. The orchestrator **audits** replication health but does not control what individual nodes store or evict.

```
Local GC (runs on each block store node):
  1. Nodes store blocks they're close to in Kademlia XOR space
  2. When usage approaches 85% of capacity → trigger LRU eviction
  3. Evict least-recently-accessed blocks first
  4. Hard refuse new writes at 95% capacity
  5. Withdraw provider records for evicted blocks from DHT
  6. Recently-accessed blocks (served via bitswap) naturally survive GC

Orchestrator audit (background loop):
  1. For each VM with confirmedVersion < currentVersion:
     → Query DHT FindProviders for each chunk CID in latest manifest
     → If ALL chunks have ≥N providers → advance confirmedVersion
  2. If a chunk's provider count drops below threshold:
     → Flag as under-replicated in dashboard
     → Kademlia's natural re-publication handles recovery:
       remaining providers re-announce, nearby nodes pull the block
     → In rare cases of persistent under-replication, orchestrator
       can publish a "re-replicate" hint via the DHT (fallback, not normal path)

Self-healing example:
  Chunk B has 15 providers. 3 nodes go offline → 12 providers remain.
  Kademlia provider record TTL expires for the 3 offline nodes.
  Remaining 12 providers re-announce. Nearby nodes in XOR space
  that don't yet have the block discover it and pull via bitswap.
  Provider count recovers to ~15 without any orchestrator intervention.

GC lifecycle example:
  confirmedVersion v1: chunks [A, B, C] → all have ≥N providers
  confirmedVersion v2: chunks [A, D, C] → chunk D propagates, reaches ≥N
  Chunk B is no longer in any current manifest → nodes stop re-announcing
  Provider records for B expire (TTL) → B naturally disappears
  No explicit "unpin" command needed
```

### 6.5 Storage Budget Under the 5% Duty

The 5% cap constrains how many VM manifests a node can hold replicas of. With overlay-only replication, the effective cost per VM is dramatically lower than the virtual disk size:

```
Node B has 50 GB block store (5% of 1 TB):

Currently stored (determined by Kademlia XOR proximity):
  Scattered chunks from many VMs: ~12 GB total
  (Node B doesn't know or care which VMs these chunks belong to —
   it simply stores blocks it's close to in XOR space)
  Free: 38 GB

Compare to full-disk replication:
  A single VM's full disk could consume 20-40 GB of Node B's budget.
  With overlay-only: scattered chunks from dozens of VMs fit easily.

Each node manages its own budget locally via LRU eviction.
The aggregate across all nodes provides massive redundancy:
  1,000 nodes × 50 GB average = 50 TB aggregate block store
  A VM with 3 GB overlay replicated to ~15 nodes = 45 GB total
  That's 0.09% of the network's capacity per VM.
```

Content addressing still provides deduplication on top of overlay-only savings: if two VMs install the same packages on the same base image, those overlay chunks share the same CID. And with lazysync, each cycle only adds the genuinely new blocks — unchanged chunks share the same CID and cost nothing.

---

## 7. Implementation Order

### Phase A: Orchestrator — Core Block Store

1. **BlockStoreVmSpec.cs** — Resource specification (5% duty, 512 MB RAM)
2. **BlockStoreInfo** on Node.cs — Model for tracking state (including PinnedManifestRootCids)
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
13. **blockstore-node Go binary** — Core storage engine with DAG/manifest support
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
23. Test GC with orchestrator-driven pin/unpin
24. Test self-healing (reconciliation redeploy)

### Phase D: Lazysync & Migration

25. **LazysyncManager** in orchestrator — manifest version tracking, confirmed vs current, provider count audit loop
26. **Lazysync daemon** on node agent — QEMU QMP integration (dirty bitmaps, incremental backup), overlay chunking via `qemu-img map` for sparse cluster discovery, block push, manifest update cycle
27. **Initial overlay seeding** — push allocated overlay clusters on first enrollment, progress tracking, rate limiting
28. **Kademlia scatter replication** — block store nodes autonomously pull blocks close to their peer ID in XOR space; verify propagation via provider count audit
29. **Migration planner** — PlanMigrationAsync with scheduling fit + resource headroom (no block locality scoring — target fetches via bitswap)
30. **Disk reconstruction** — ensure base image available (cache or download) + fetch overlay from scattered providers via bitswap + regenerate cloud-init ISO from labels
31. **End-to-end migration test** — simulate node failure, verify VM rescheduling from confirmed manifest (base image + overlay from bitswap + cloud-init)

---

## 8. How This Connects to Future Features

### Built-In (Phase A-D)
- **Continuous VM disk replication (lazysync)** — Core feature, not optional
- **Live migration on node failure** — Core feature, the primary purpose of the block store
- **Template image distribution** — Distribute base images via block store (dedup across nodes)

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

## 9. Key Design Decisions

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
LevelDB is only used for metadata (pin state, access times, stats).

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
- **Simpler reconstruction**: read manifest, assemble chunks in offset order. No delta patching
- **QEMU native support**: QEMU's Changed Block Tracking (persistent dirty bitmaps, since 4.0+) and incremental backup provide crash-consistent dirty block exports without pausing the VM. This is the same mechanism that powers live migration in traditional hypervisors

### Decision 10: Overlay-only replication (not full disk)
**Why:** VMs use a qcow2 backing chain: a read-only base image (shared, downloadable) plus a per-VM writable overlay. The base image and cloud-init ISO are reconstructible artifacts. Only the overlay carries unique state. Replicating the overlay only instead of the full virtual disk provides:
- **Dramatically smaller seeding**: a freshly booted 100 GB VM has ~500 MB of overlay writes, not 100 GB. Initial seed completes in seconds instead of hours
- **Lower storage cost**: overlay-only replicas are typically 1-10% of virtual disk size. A node's 5% block store budget can hold replicas for dozens of VMs instead of 2-3
- **Faster migration**: reconstruct = download base image (likely already cached) + assemble overlay (small, mostly local) + regenerate cloud-init (milliseconds). Base images are shared across VMs of the same type — a node running 10 Debian VMs downloads the base image once
- **Sparse manifest**: only allocated overlay clusters appear in the manifest. `qemu-img map` discovers allocation without reading the full virtual disk
- **Natural fit with qcow2**: QEMU already tracks dirty blocks at the overlay level. The dirty bitmap covers overlay writes only — base image reads are invisible to CBT

---

## 10. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large Go binary size | Slow VM boot | Gzip+base64 encoding, ~5-8 MB compressed |
| Storage quota enforcement | Disk full crashes | Hard refuse writes at 95%, GC trigger at 85% |
| Bitswap flood | Network saturation | Rate limiting per peer, bandwidth caps |
| Block loss (all providers offline) | VM cannot migrate | Kademlia scatter ensures 10-20+ providers per block across diverse nodes; entire region can fail without data loss |
| Stale provider records | Failed retrievals | TTL on records, periodic re-announcement; Kademlia republishes automatically |
| DHT dependency | Can't discover peers without DHT | Bootstrap poll as fallback, direct peer list |
| 5% budget exceeded | Node disk pressure | Local LRU eviction enforces budget; hard refuse at 95% |
| Overlay too large for 5% | Can't replicate heavy-write VMs | Overlay-only is typically 1-10% of virtual disk; dedup further reduces; scatter spreads cost across many nodes |
| High write-rate VM | Lazysync generates heavy replication traffic | Rate limiting per VM; adaptive cycle interval (slower for hot VMs); write coalescing reduces effective dirty block count |
| QEMU CBT compatibility | Older QEMU lacks persistent dirty bitmaps | Require QEMU ≥4.0; fallback to full re-scan if bitmap lost (rare) |
| Initial seeding slow | VM unprotected during first sync | Overlay-only seeding is typically 500 MB–15 GB (not 100 GB); completes in minutes; track progress; prioritize new VMs |
| Base image unavailable during migration | Can't reconstruct VM disk | Base images cached on most nodes; image registry as fallback; block store can distribute base images via bitswap |
| Manifest version drift | Confirmed version lags far behind current | Alert when gap exceeds threshold; orchestrator can throttle lazysync cycle until Kademlia propagation catches up |
| Orchestrator unavailable | Audit pauses, confirmedVersion stalls | Replication continues autonomously via Kademlia — blocks still scatter and providers re-announce. Lazysync continues locally, queues manifest registrations. Only confirmedVersion advancement pauses |
| Kademlia hot spots | Blocks cluster on certain nodes | XOR metric statistically distributes evenly; LRU eviction naturally sheds excess; large network dilutes any clustering |
| Over-replication waste | Blocks replicate to more nodes than needed | Benign — extra replicas improve durability and migration speed. LRU GC naturally trims excess as nodes fill up |

---

## 11. Success Metrics

### Phase A-C (Core Block Store)
- Block Store VM deploys on all eligible nodes (≥100 GB storage, ≥2 GB RAM)
- 5% storage allocation is enforced — no node exceeds its duty
- Blocks survive node restart (persistent FlatFS)
- Cross-node block retrieval works via bitswap
- GC respects pins — pinned blocks are never evicted
- Bootstrap polling discovers peers within 60 seconds
- Self-healing redeploys on VM failure

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
