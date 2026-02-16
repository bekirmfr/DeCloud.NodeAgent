# Block Store System VM — Design & Implementation Plan

**Date:** 2026-02-16
**Status:** Design Phase
**Depends on:** DHT system VMs (production-verified 2026-02-15)
**Follows patterns from:** Relay VMs, DHT VMs

---

## 1. Vision & Purpose

The Block Store system VM creates the **distributed storage backbone** for the DeCloud network. Every node with ≥100 GB storage contributes **5% of its total storage** as a network duty — forming a collective, content-addressed storage medium across the platform.

The primary purpose is **VM resilience and live migration**. When a node goes offline, its VMs can be rescheduled to another node because VM snapshots and deltas are already replicated across the block store network. No single node failure causes data loss.

Each Block Store VM:

1. **Stores replicated blocks** — Holds content-addressed chunks of OTHER nodes' VM snapshots and deltas
2. **Announces blocks** — Publishes provider records to the DHT ("I have block X")
3. **Transfers blocks** — Serves blocks to other nodes via libp2p bitswap protocol
4. **Accepts replication commands** — Orchestrator directs which blocks to fetch and pin
5. **Garbage collects** — Removes unpinned blocks under orchestrator coordination

This enables VM snapshot distribution, live migration on node failure, template image distribution, and eventually a full decentralized filesystem — all without centralized cloud storage.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Orchestrator                                    │
│  ┌──────────────────────┐  ┌───────────────────────────────────┐ │
│  │ BlockStoreService     │  │ BlockStoreController              │ │
│  │ - DeployBlockStoreVm  │  │ POST /api/blockstore/join         │ │
│  │ - PlaceReplicas()     │  │ POST /api/blockstore/announce     │ │
│  │ - TriggerSnapshot()   │  │ POST /api/blockstore/replicate    │ │
│  │ - ScheduleMigration() │  │ GET  /api/blockstore/locate/{cid} │ │
│  └──────────────────────┘  │ POST /api/blockstore/snapshot      │ │
│                              │ GET  /api/blockstore/stats         │ │
│  ┌──────────────────────┐  └───────────────────────────────────┘ │
│  │ SnapshotManager       │                                        │
│  │ - Snapshot chains      │  Node.BlockStoreInfo:                 │
│  │ - Delta consolidation  │  { VmId, PeerId, Capacity,           │
│  │ - Replication tracking │    Used, Status, PinnedSnapshots }    │
│  │ - Pin/unpin lifecycle  │                                        │
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
    Task<List<ReplicationTarget>> SelectReplicaNodesAsync(string snapshotRootCid, int replicationFactor,
        VmSchedulingRequirements? affinityHint = null, string? excludeNodeId = null);

    // Snapshot lifecycle
    Task<SnapshotRecord> RegisterSnapshotAsync(string vmId, string rootCid, SnapshotType type, CancellationToken ct = default);
    Task ConsolidateDeltasAsync(string vmId, CancellationToken ct = default);

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
- **Snapshot lifecycle**: register snapshots, track delta chains, trigger consolidation
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

POST /api/blockstore/snapshot
  Request:  { vmId, rootCid, type: "full"|"delta", parentCid?, blockCount, totalBytes }
  Auth:     Internal
  Response: { success, snapshotId, replicationPlan: [{ nodeId, cidsToPin }] }
  Purpose:  Register a new VM snapshot. Orchestrator records the snapshot chain
            and returns a replication plan (which nodes should pin which blocks).

GET /api/blockstore/snapshot/{vmId}
  Response: { vmId, currentSnapshot: { rootCid, timestamp, type },
              deltaChain: [{ rootCid, parentCid, timestamp, dirtyBlocks }],
              replicationStatus: { targetFactor: 3, actual: { cid: [nodeIds] } } }
  Purpose:  Query snapshot chain and replication status for a VM.

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
    public List<string> PinnedSnapshotRootCids { get; set; } = [];  // Snapshot DAGs pinned on this node
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

A DAG manifest describes a structured collection of blocks — primarily VM snapshots:

```json
{
  "type": "vm-snapshot",
  "vmId": "vm-abc123",
  "timestamp": "2026-02-16T10:30:00Z",
  "snapshotType": "full",
  "parentCid": null,
  "diskSizeBytes": 107374182400,
  "blockSizeBytes": 1048576,
  "chunks": [
    { "offset": 0,       "cid": "bafk...aaa" },
    { "offset": 1048576, "cid": "bafk...bbb" },
    { "offset": 2097152, "cid": "bafk...ccc" }
  ]
}
```

For delta snapshots, only changed chunks are included:

```json
{
  "type": "vm-snapshot",
  "vmId": "vm-abc123",
  "timestamp": "2026-02-16T12:00:00Z",
  "snapshotType": "delta",
  "parentCid": "bafk...root-of-full-snapshot",
  "dirtyChunkCount": 42,
  "chunks": [
    { "offset": 3145728,  "cid": "bafk...ddd" },
    { "offset": 17825792, "cid": "bafk...eee" }
  ]
}
```

Content addressing provides automatic deduplication: if a VM has 100 GB disk but only 20 GB of unique changing data across snapshots, the actual block store consumption is ~20 GB (times replication factor). Unchanged blocks share the same CID and are never re-stored or re-transferred.

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

## 6. VM Snapshots & Migration

This is the **primary use case** for the block store — not a future feature.

### 6.1 Snapshot Model

Every user VM has its disk continuously snapshotted and distributed across the block store network. When a node goes offline, the VM can be rescheduled to another node because its disk state is already replicated.

```
Node A runs VM-1 (100 GB disk):

1. Periodic snapshot (e.g., every 6 hours):
   VM-1 disk → chunk into 1 MB blocks → CIDv1 per block
   → DAG manifest (root CID referencing all chunk CIDs)
   → Orchestrator registers snapshot: POST /api/blockstore/snapshot

2. Between snapshots — deltas (e.g., every 30 minutes):
   Dirty blocks since last snapshot → chunk → CIDs
   → Delta manifest (references parent snapshot, only changed offsets)
   → Orchestrator registers delta

3. Orchestrator replication:
   For full snapshot + deltas:
   → Select 3 target nodes (placement-aware, see 6.3)
   → POST /api/blockstore/replicate to each target
   → Target nodes fetch blocks via bitswap from Node A's block store
   → Target nodes pin the blocks (won't GC)

4. Delta consolidation (e.g., weekly):
   When delta chain grows long → trigger new full snapshot
   → Orchestrator unpins old snapshot + deltas on target nodes
   → Old blocks become GC candidates
```

### 6.2 Migration on Node Failure

```
Node A goes offline (shutdown / blocked / hardware failure):

For each VM on Node A:
  1. Orchestrator retrieves VM's scheduling requirements
     (CPU, RAM, GPU, region, affinity rules, etc.)

  2. Orchestrator retrieves VM's latest snapshot chain
     (root CID + delta CIDs + all referenced block CIDs)

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

  6. Reconstruct VM disk from snapshot manifest:
     Read DAG manifest → assemble chunks in order → write disk image
     Apply delta chain on top (if any deltas since last full snapshot)

  7. Boot VM on new node.
     VM resumes from last snapshot + delta state.
```

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
  1. New snapshot registered → orchestrator pins blocks on target nodes
  2. New full snapshot verified replicated → orchestrator unpins old snapshot chain
  3. Unpinned blocks become local GC candidates
  4. Node's block store runs GC: evicts unpinned blocks via LRU
     - GC triggers at 85% capacity
     - Hard refuse new writes at 95% capacity

A node never decides "is this block safe to delete?" — it only follows pins.
If the orchestrator unpinned it, it's fair game for GC.
```

### 6.5 Storage Budget Under the 5% Duty

The 5% cap constrains how many VM snapshots a node can hold replicas of. The orchestrator must balance this:

```
Node B has 50 GB block store (5% of 1 TB):

Currently pinned:
  VM-1 snapshot: 8 GB  (100 GB disk, but only 8 GB unique blocks after dedup)
  VM-3 snapshot: 12 GB
  VM-7 deltas:   2 GB
  Total pinned:  22 GB
  Free:          28 GB

Orchestrator knows Node B can accept ~28 GB more replica data.
When placing new snapshot replicas, Node B is a candidate if it
has enough headroom and meets the VM's scheduling requirements.
```

Content addressing is critical here — deduplication means the effective cost of storing a VM snapshot is much less than the raw disk size. A 100 GB VM disk with mostly static content (OS, libraries) might only have 5-10 GB of unique blocks.

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

### Phase D: Snapshot & Migration

25. **SnapshotManager** in orchestrator — snapshot chain tracking, delta consolidation
26. **Snapshot trigger integration** — periodic snapshots + delta capture for user VMs
27. **Placement-aware replication** — SelectReplicaNodesAsync with scheduling affinity
28. **Migration planner** — PlanMigrationAsync with block locality scoring
29. **Disk reconstruction** — assemble VM disk from snapshot manifest + delta chain
30. **End-to-end migration test** — simulate node failure, verify VM rescheduling

---

## 8. How This Connects to Future Features

### Built-In (Phase A-D)
- **VM snapshot distribution** — Core feature, not optional
- **Live migration on node failure** — Core feature, the primary purpose of the block store
- **Template image distribution** — Distribute base images via block store (dedup across nodes)

### Near-Term Extensions
- **On-demand VM migration** — User-initiated, not just failure-driven
- **Distributed backups** — Scheduled backup policies per VM
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
**Why:** VM snapshots are structured collections of blocks, not flat blobs. Supporting DAG manifests in the initial design means:
- Snapshot/delta chain representation is clean and native
- The block store understands "pin this DAG" (manifest + all referenced blocks)
- Future features (large file storage, directory trees) get DAG support for free
- Avoids painful migration from flat-block-only to DAG-aware later

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
| Snapshot too large for 5% | Can't replicate large VMs | Dedup reduces effective size; spread across many nodes |
| Delta chain too long | Slow reconstruction | Periodic consolidation (configurable, e.g., weekly) |
| Orchestrator unavailable | Can't coordinate replication | Existing pins persist; GC won't evict pinned blocks; resume when orchestrator returns |

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

### Phase D (Snapshots & Migration)
- VM snapshots are created and distributed within configured intervals
- Replication factor of 3 is maintained for all active snapshots
- Replicas are placed on scheduling-compatible nodes (migration readiness)
- When a node goes offline, its VMs are rescheduled to a new node
- Migration selects nodes with highest block locality (minimal transfer)
- VM disk reconstruction from snapshot + delta chain completes successfully
- Delta consolidation keeps chain length bounded
