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
4. **Accepts replication commands** — Orchestrator directs which blocks to fetch and pin
5. **Garbage collects** — Removes unpinned blocks under orchestrator coordination

This enables continuous VM disk replication, live migration on node failure, template image distribution, and eventually a full decentralized filesystem — all without centralized cloud storage.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Orchestrator                                    │
│  ┌──────────────────────┐  ┌───────────────────────────────────┐ │
│  │ BlockStoreService     │  │ BlockStoreController              │ │
│  │ - DeployBlockStoreVm  │  │ POST /api/blockstore/join         │ │
│  │ - PlaceReplicas()     │  │ POST /api/blockstore/announce     │ │
│  │ - ScheduleMigration() │  │ POST /api/blockstore/replicate    │ │
│  └──────────────────────┘  │ GET  /api/blockstore/locate/{cid} │ │
│                              │ POST /api/blockstore/manifest      │ │
│  ┌──────────────────────┐  │ GET  /api/blockstore/stats         │ │
│  │ LazysyncManager       │  └───────────────────────────────────┘ │
│  │ - Dirty block tracking │                                        │
│  │ - Manifest versioning  │  Node.BlockStoreInfo:                 │
│  │ - Replication tracking │  { VmId, PeerId, Capacity,           │
│  │ - Pin/unpin lifecycle  │    Used, Status, PinnedManifests }    │
│  └──────────────────────┘                                        │
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
│  │    POST /pin/{cid}      (pin block, prevent GC)               │ │
│  │    POST /gc             (run garbage collection)               │ │
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

This keeps the DHT VM lightweight (just routing + discovery) while the Block Store VM handles heavy storage I/O.

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

The block store on each node holds **replicated block chunks from OTHER nodes' VMs** — not the node's own data. A node's own VM snapshots are distributed to other nodes' block stores by the orchestrator.

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

    // Replication & placement
    Task ReplicateBlocksAsync(string targetNodeId, List<string> cids, bool pin, CancellationToken ct = default);
    Task UnpinBlocksAsync(string targetNodeId, List<string> cids, CancellationToken ct = default);
    Task<List<ReplicationTarget>> SelectReplicaNodesAsync(string manifestRootCid, int replicationFactor,
        VmSchedulingRequirements? affinityHint = null, string? excludeNodeId = null);

    // Lazysync manifest lifecycle
    Task<ManifestRecord> RegisterManifestAsync(string vmId, string rootCid, int version,
        List<string> changedBlockCids, long totalBytes, CancellationToken ct = default);
    Task ConfirmManifestReplicatedAsync(string vmId, int version, CancellationToken ct = default);

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
- **Placement-aware replication**: select target nodes considering failure domains, node capabilities (CPU, RAM, GPU, region), and VM scheduling requirements so replicas land on nodes that could host the VM if migration is needed
- **Manifest lifecycle**: register evolving manifests from lazysync, track confirmed vs current version, GC blocks unreferenced by any confirmed manifest
- **Migration planning**: given a VM to migrate, rank candidate nodes by block locality (how many blocks they already have) combined with scheduling fit

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
  Purpose:  Batch announce/withdraw CIDs (orchestrator maintains global index)

POST /api/blockstore/replicate
  Request:  { targetNodeId, cids: ["bafk..."], pin: true, priority: "normal"|"urgent" }
  Auth:     Internal (orchestrator → node agent → block store VM)
  Response: { success, queued: count, estimatedBytes }
  Purpose:  Orchestrator directs a node to fetch and pin specific blocks.
            Used for snapshot replication and migration pre-staging.

GET /api/blockstore/locate/{cid}
  Response: { cid, providers: [{ nodeId, peerId, multiaddr,
              nodeCapabilities: { vcpus, ramBytes, gpus, region } }],
              replication: count }
  Purpose:  Find which nodes have a specific block. Includes node capabilities
            so the caller (or orchestrator) can make placement-aware decisions.

POST /api/blockstore/manifest
  Request:  { vmId, rootCid, version, changedBlockCids: ["bafk..."], totalBytes }
  Auth:     Internal
  Response: { success, manifestVersion, replicationPlan: [{ nodeId, cidsToPin }] }
  Purpose:  Register an updated VM manifest from a lazysync cycle.
            Orchestrator records the new manifest version and returns a
            replication plan for the changed blocks only.

GET /api/blockstore/manifest/{vmId}
  Response: { vmId, currentVersion, confirmedVersion, confirmedRootCid,
              timestamp, replicationStatus: { targetFactor: 3,
              confirmedOnNodes: [nodeIds] } }
  Purpose:  Query manifest version and replication status for a VM.
            confirmedVersion = latest version fully replicated on ≥N target nodes.
            currentVersion = latest version (may be partially replicated).

GET /api/blockstore/stats
  Response: { totalNodes, totalCapacity, totalUsed, totalBlocks,
              avgReplication, snapshotCount, totalSnapshotBytes }
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
    public int BlockCount { get; set; }                 // Local blocks
    public int PinnedBlockCount { get; set; }           // Pinned (won't GC)
    public List<string> PinnedManifestRootCids { get; set; } = [];  // VM manifest DAGs pinned on this node
    public BlockStoreStatus Status { get; set; } = BlockStoreStatus.Initializing;
    public DateTime? LastHealthCheck { get; set; }
}

public enum BlockStoreStatus { Initializing, Active, Degraded, Full, Offline }
```

Add `BlockStoreInfo? BlockStoreInfo` property to the Node class.

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
5. Pinning system (pinned blocks survive GC)
6. Garbage collection (LRU eviction of unpinned blocks)
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
        Used for VM snapshot manifests and delta manifests.

GET  /dag/{cid}
  → { rootCid, manifest: { type, vmId, chunks: [...] },
      localBlocks: N, totalBlocks: M, complete: bool }
  Note: Returns the manifest and reports how many of its referenced
        blocks are available locally (complete=true when all present).

GET  /stats
  → { capacityBytes, usedBytes, usagePercent, blockCount,
      pinnedCount, connectedPeers, bitswapSent, bitswapReceived }

POST /pin/{cid}
  → { cid, pinned: true }
  Note: If cid is a DAG root, pins the manifest AND all referenced blocks.

DELETE /pin/{cid}
  → { cid, pinned: false }

POST /gc
  → { freedBytes, freedBlocks, remainingBytes }
  Note: Only evicts unpinned blocks. Orchestrator controls what is pinned.

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
├── pins.json              # List of pinned CIDs (including DAG roots)
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
   Note: Blocks are NOT auto-pinned. The orchestrator explicitly pins
         what it wants retained. Unpinned blocks are GC candidates.

2. GET /blocks/{cid}
   → Check local FlatFS
   → If not found: bitswap request to network
   → Timeout after 30 seconds
   → Return bytes or 404

3. DELETE /blocks/{cid}
   → Remove pin if pinned
   → Delete from FlatFS
   → Withdraw provider record from DHT

4. POST /dag (manifest + blocks)
   → Store each block in FlatFS (deduplicated by CID)
   → Store manifest in dags/ directory
   → Manifest root CID = SHA-256 of manifest JSON
   → Return root CID

5. GC (orchestrator-coordinated)
   → Find unpinned blocks (orchestrator controls pin lifecycle)
   → Sort by last access time (LRU)
   → Delete until usage is under 85% of capacity
   → Hard refuse writes at 95% capacity
   → Withdraw provider records for deleted blocks
```

#### DAG / Manifest Structure

A DAG manifest describes the complete current state of a VM's disk — a single evolving document where chunk CIDs are replaced in-place as the lazysync daemon detects dirty blocks. There is no full/delta distinction. Just one manifest per VM that evolves over time:

```json
{
  "type": "vm-disk",
  "vmId": "vm-abc123",
  "version": 47,
  "timestamp": "2026-02-16T12:05:00Z",
  "diskSizeBytes": 107374182400,
  "blockSizeBytes": 1048576,
  "chunks": [
    { "offset": 0,       "cid": "bafk...aaa" },
    { "offset": 1048576, "cid": "bafk...ggg" },
    { "offset": 2097152, "cid": "bafk...ccc" }
  ]
}
```

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

Blocks referenced by `confirmedVersion` are never GC'd. Blocks only referenced by older versions are fair game. This eliminates delta chains, delta consolidation, and the full/delta manifest type distinction entirely.

Content addressing provides automatic deduplication: if a VM has 100 GB disk but only 20 GB of unique changing data, the actual block store consumption is ~20 GB (times replication factor). Unchanged blocks share the same CID and are never re-stored or re-transferred.

#### Bitswap Integration

The Go binary connects to other Block Store peers via libp2p and uses the bitswap protocol for block exchange:

```go
// Bitswap network: blocks flow between peers automatically
// When a peer requests a block we have → serve it
// When we need a block we don't have → request from peers
bswap := bitswap.New(ctx, network, blockstore)
```

Bitswap is the **transfer mechanism** — the orchestrator decides **what** to replicate and **where**, then issues `POST /api/blockstore/replicate` commands. The target node's block store binary uses bitswap to pull the requested blocks from the network.

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

Instead of periodic snapshots with delta chains, every user VM has its disk **continuously replicated** via a background lazysync process. There are no "snapshot events." Dirty disk chunks flow to the block store network as they change, and a single evolving manifest per VM tracks the current state.

```
Node A runs VM-1 (100 GB disk):

Background lazysync daemon (runs on the node agent, cycles every ~5 minutes):

  For each running VM on this node:

  1. QEMU incremental backup via QMP:
     → drive-backup sync=incremental
     → Exports dirty blocks since last sync cycle
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
     → Orchestrator returns replication plan for changed blocks only

  7. Orchestrator replicates changed blocks:
     → POST /api/blockstore/replicate to each target node
     → Target nodes fetch new blocks via bitswap from Node A
     → Target nodes update their pinned manifest copy
     → When all targets confirm → orchestrator advances confirmedVersion

  8. Reset QEMU dirty bitmap for next cycle
```

**What this eliminates:**
- No "full snapshot" events — the first sync seeds the entire disk, subsequent syncs only push dirty blocks
- No "delta" manifests — there's one manifest per VM, chunks get replaced in-place
- No delta chains to traverse during reconstruction
- No delta consolidation — nothing to consolidate
- No snapshot-type field — just `vm-disk` manifests with monotonically increasing version numbers

**Recovery point:** If lazysync runs every 5 minutes and replication takes ~2 minutes, the recovery point is typically 5-7 minutes behind live state. This is crash-consistent (QEMU guarantees this) — the same guarantees as traditional hypervisor crash recovery.

**Write coalescing:** If a VM writes the same chunk 100 times between lazysync cycles, only the final state is replicated. The intermediate writes are invisible — only the net change per cycle matters.

**Bandwidth:** A VM writing 1 GB/hour generates ~17 MB per 5-minute cycle. This trickles out continuously rather than bursting during snapshot events.

### 6.1.1 Initial Seeding

When a VM is first created (or first enrolled in lazysync), the entire disk must be pushed to the block store. This is the one-time "full sync":

```
First lazysync cycle for a new VM:
  100 GB disk → 100,000 chunks → CIDs → push all to block store → replicate
  This takes time (minutes to hours depending on network)
  Until confirmed: VM has no migration safety net

Subsequent cycles:
  Only dirty blocks → typically small → fast replication
```

The orchestrator tracks seeding state. VMs with `confirmedVersion == 0` are flagged as "unprotected" in the dashboard. Seeding is rate-limited to avoid saturating the node's block store VM or network.

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

The orchestrator tracks manifest versions to ensure replication consistency:

```
Manifest v1: [chunk0=A, chunk1=B, chunk2=C]  → fully replicated (confirmed)
Lazysync: chunk1 changed → CID D
Manifest v2: [chunk0=A, chunk1=D, chunk2=C]  → replicating...
Lazysync: chunk2 changed → CID E
Manifest v3: [chunk0=A, chunk1=D, chunk2=E]  → pending

Orchestrator state for this VM:
  currentVersion:   3   (latest registered)
  confirmedVersion: 1   (latest where all blocks verified on ≥3 nodes)

Rules:
  - Blocks referenced by confirmedVersion are NEVER GC'd
  - Blocks only referenced by versions < confirmedVersion are GC candidates
  - If replication of v2 completes before v3 is registered → confirmedVersion advances to 2
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

  3. For each candidate node, compute migration score:
     scheduling_fit:  does the node meet VM's requirements? (filter)
     block_locality:  how many of the VM's blocks does this node
                      already have in its block store? (rank)
     available_resources: enough CPU/RAM/disk headroom? (filter)

     score = block_locality_ratio * weight_locality
           + resource_headroom * weight_resources

  4. Select best candidate node (highest score that passes all filters)

  5. Fetch missing blocks:
     Candidate node's block store already has most blocks (pinned replicas).
     Remaining blocks are fetched via bitswap from other replica nodes.
     This is fast — most data is already local.

  6. Reconstruct VM disk from manifest:
     Read manifest → assemble chunks in offset order → write disk image
     No delta chain to apply — the manifest IS the complete state.

  7. Boot VM on new node.
     VM resumes from confirmed manifest state.
     (~5-7 minutes of state lost since last confirmed lazysync)
```

Reconstruction is simpler than the snapshot + delta model: just read chunks from the manifest in order. No chain traversal, no patching deltas on top.

### 6.3 Placement-Aware Replication

The orchestrator doesn't replicate randomly. It places replicas on nodes that:

1. **Are in different failure domains** — different racks, availability zones, or regions. If all 3 replicas are on the same rack and the rack loses power, the data is gone.

2. **Could plausibly host the VM** — meet the VM's scheduling requirements (enough CPU, RAM, right GPU type, acceptable region). This way, when migration is needed, a candidate node already has the data locally. No transfer needed during the emergency.

3. **Have block store capacity available** — the 5% allocation must not be exceeded. The orchestrator tracks each node's `UsedBytes` vs `CapacityBytes`.

```
Example: VM-1 requires 4 vCPUs, 8 GB RAM, us-east region

Orchestrator selects replica nodes:
  Node B: us-east, 8 cores, 32 GB RAM, 40 GB block store free  ✓
  Node C: us-east, 16 cores, 64 GB RAM, 90 GB block store free ✓
  Node D: eu-west, 4 cores, 16 GB RAM, 25 GB block store free  ✓ (different region = failure domain diversity)

NOT selected:
  Node E: us-east, 2 cores, 4 GB RAM  ✗ (can't host VM-1, too small)
  Node F: us-east, same rack as Node B  ✗ (same failure domain)
```

### 6.4 GC and Pin Lifecycle

Garbage collection is **orchestrator-coordinated** through the pin mechanism:

```
Orchestrator owns the pin lifecycle:
  1. New manifest version registered → orchestrator pins changed blocks on target nodes
  2. New manifest version confirmed replicated → orchestrator unpins blocks from
     previous version that are no longer referenced by any confirmed manifest
  3. Unpinned blocks become local GC candidates
  4. Node's block store runs GC: evicts unpinned blocks via LRU
     - GC triggers at 85% capacity
     - Hard refuse new writes at 95% capacity

A node never decides "is this block safe to delete?" — it only follows pins.
If the orchestrator unpinned it, it's fair game for GC.

Pin lifecycle example:
  confirmedVersion v1: chunks [A, B, C] → all pinned on target nodes
  confirmedVersion v2: chunks [A, D, C] → block D pinned, block B unpinned
  Block B has no other references → GC candidate
  Block A and C unchanged → still pinned, zero cost for this transition
```

### 6.5 Storage Budget Under the 5% Duty

The 5% cap constrains how many VM manifests a node can hold replicas of. The orchestrator must balance this:

```
Node B has 50 GB block store (5% of 1 TB):

Currently pinned:
  VM-1 manifest: 8 GB  (100 GB disk, but only 8 GB unique blocks after dedup)
  VM-3 manifest: 12 GB
  VM-7 manifest: 3 GB
  Total pinned:  23 GB
  Free:          27 GB

Orchestrator knows Node B can accept ~27 GB more replica data.
When placing new manifest replicas, Node B is a candidate if it
has enough headroom and meets the VM's scheduling requirements.
```

Content addressing is critical here — deduplication means the effective cost of storing a VM's blocks is much less than the raw disk size. A 100 GB VM disk with mostly static content (OS, libraries) might only have 5-10 GB of unique blocks. And with lazysync, each cycle only adds the genuinely new blocks — unchanged chunks share the same CID and cost nothing.

---

## 7. Implementation Order

### Phase A: Orchestrator — Core Block Store

1. **BlockStoreVmSpec.cs** — Resource specification (5% duty, 512 MB RAM)
2. **BlockStoreInfo** on Node.cs — Model for tracking state (including PinnedSnapshotRootCids)
3. **VmType.BlockStore** — Add to enum
4. **IBlockStoreService / BlockStoreService** — Deployment, replication, snapshot lifecycle, migration planning
5. **BlockStoreController.cs** — `/join`, `/announce`, `/replicate`, `/locate`, `/snapshot`, `/stats`
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

25. **LazysyncManager** in orchestrator — manifest version tracking, confirmed vs current, pin lifecycle
26. **Lazysync daemon** on node agent — QEMU QMP integration (dirty bitmaps, incremental backup), chunking, block push, manifest update cycle
27. **Initial seeding** — full disk sync on first enrollment, progress tracking, rate limiting
28. **Placement-aware replication** — SelectReplicaNodesAsync with scheduling affinity
29. **Migration planner** — PlanMigrationAsync with block locality scoring
30. **Disk reconstruction** — assemble VM disk from confirmed manifest (flat chunk assembly, no delta chain)
31. **End-to-end migration test** — simulate node failure, verify VM rescheduling from confirmed manifest

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

### Decision 5: Orchestrator-driven replication (not organic gossip)
**Why:** With a constrained 5% budget per node, replication can't be random. The orchestrator:
- Knows every node's capabilities, capacity, and current load
- Knows every VM's scheduling requirements
- Can place replicas on nodes that could host the VM (migration readiness)
- Can enforce failure domain diversity
- Can track replication factor per snapshot
Bitswap is the transfer mechanism; the orchestrator is the placement brain.

### Decision 6: Orchestrator-coordinated GC via pin lifecycle
**Why:** With the 5% duty model, local GC decisions are dangerous — a node might evict blocks that are the last replica. Instead:
- The orchestrator explicitly pins blocks it wants retained
- The orchestrator unpins old snapshots only after confirming new ones are replicated
- Local GC only evicts unpinned blocks (safe by construction)
- No distributed consensus needed — the orchestrator is the single source of truth for pin state

### Decision 7: Content-addressed with CIDv1
**Why:** Using IPFS-compatible CIDs (Content IDentifiers) means:
- Deduplication is automatic (same content = same CID)
- Integrity verification is built-in
- Future IPFS interop
- Industry standard format
- Critical for snapshot efficiency: unchanged disk blocks across snapshots share the same CID

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

---

## 10. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large Go binary size | Slow VM boot | Gzip+base64 encoding, ~5-8 MB compressed |
| Storage quota enforcement | Disk full crashes | Hard refuse writes at 95%, GC trigger at 85% |
| Bitswap flood | Network saturation | Rate limiting per peer, bandwidth caps |
| Block loss (all replicas offline) | VM cannot migrate | Min replication factor 3, failure domain diversity |
| Stale provider records | Failed retrievals | TTL on records, periodic re-announcement |
| DHT dependency | Can't discover peers without DHT | Bootstrap poll as fallback, direct peer list |
| 5% budget exceeded | Node disk pressure | Orchestrator tracks capacity, hard limit on pins |
| Manifest too large for 5% | Can't replicate large VMs | Dedup reduces effective size; spread across many nodes |
| High write-rate VM | Lazysync generates heavy replication traffic | Rate limiting per VM; adaptive cycle interval (slower for hot VMs); write coalescing reduces effective dirty block count |
| QEMU CBT compatibility | Older QEMU lacks persistent dirty bitmaps | Require QEMU ≥4.0; fallback to full re-scan if bitmap lost (rare) |
| Initial seeding slow | VM unprotected during first full sync | Track seeding progress; prioritize seeding for new VMs; rate-limit to avoid saturating network |
| Manifest version drift | Confirmed version lags far behind current | Alert when gap exceeds threshold; orchestrator can throttle lazysync cycle until replication catches up |
| Orchestrator unavailable | Can't coordinate replication | Existing pins persist; GC won't evict pinned blocks; lazysync continues locally, queues manifest registrations for when orchestrator returns |

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
- Lazysync daemon continuously replicates dirty blocks for all running VMs
- Lazysync cycle completes within configured interval (~5 minutes) under normal load
- QEMU incremental backup produces crash-consistent dirty block exports without VM pause
- Confirmed manifest version stays within 2 versions of current (replication keeps up)
- Replication factor of 3 is maintained for all confirmed manifests
- Replicas are placed on scheduling-compatible nodes (migration readiness)
- When a node goes offline, its VMs are rescheduled to a new node
- Migration selects nodes with highest block locality (minimal transfer)
- VM disk reconstruction from confirmed manifest completes successfully (flat chunk assembly)
- Initial seeding of new VMs completes and reaches confirmed state
- Recovery point objective: ≤10 minutes of data loss under normal conditions
