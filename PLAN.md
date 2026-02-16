# Block Store System VM — Design & Implementation Plan

**Date:** 2026-02-16
**Status:** Design Phase
**Depends on:** DHT system VMs (production-verified 2026-02-15)
**Follows patterns from:** Relay VMs, DHT VMs

---

## 1. Vision & Purpose

The Block Store system VM creates a **decentralized, content-addressed storage layer** for the DeCloud network. Every eligible node runs a Block Store VM that:

1. **Stores blocks** — Content-addressed (SHA-256) chunks of data on local disk
2. **Announces blocks** — Publishes provider records to the DHT ("I have block X")
3. **Transfers blocks** — Serves blocks to other nodes via libp2p bitswap protocol
4. **Replicates** — Ensures blocks exist on multiple nodes for durability
5. **Garbage collects** — Removes unpinned blocks when storage pressure rises

This enables user-facing features like VM snapshots, template image distribution, file storage, and eventually a full decentralized filesystem — all without centralized cloud storage.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator                                   │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │ BlockStoreService    │  │ BlockStoreController             │  │
│  │ - DeployBlockStoreVm │  │ POST /api/blockstore/join        │  │
│  │ - GetStorageNodes()  │  │ POST /api/blockstore/announce    │  │
│  │ - AllocateStorage()  │  │ GET  /api/blockstore/locate/{h}  │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
│                                                                   │
│  Node.BlockStoreInfo: { VmId, PeerId, Capacity, Used, Status }  │
└────────────────────┬────────────────────────────────────────────┘
                     │ CreateVm command (labels)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Node Agent                                     │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │ CommandProcessorService  │  │ BlockStoreCallbackController │  │
│  │ - Renders cloud-init     │  │ POST /api/blockstore/ready   │  │
│  │ - Substitutes labels     │  │ (backup registration)        │  │
│  └─────────────────────────┘  └──────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │ VM boots
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Block Store VM (Debian 12 minimal)                   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  blockstore-node (Go binary)                                 │ │
│  │  - libp2p host (reuses DHT identity pattern)                 │ │
│  │  - Bitswap protocol for block exchange                       │ │
│  │  - Content-addressed FlatFS block storage                    │ │
│  │  - HTTP API on 127.0.0.1:5090                                │ │
│  │    GET  /health                                               │ │
│  │    POST /blocks         (put block → returns CID)             │ │
│  │    GET  /blocks/{cid}   (get block)                           │ │
│  │    DELETE /blocks/{cid} (unpin block)                         │ │
│  │    GET  /blocks         (list local blocks)                   │ │
│  │    GET  /stats          (storage usage stats)                 │ │
│  │    POST /pin/{cid}      (pin block, prevent GC)              │ │
│  │    POST /gc             (run garbage collection)              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ WireGuard     │  │ Bootstrap Poll   │  │ Dashboard         │  │
│  │ wg-mesh       │  │ (orchestrator    │  │ (Python, port     │  │
│  │ (same mesh    │  │  peer discovery) │  │  8080 → Nginx 80) │  │
│  │  as DHT VM)   │  │                  │  │                   │  │
│  └──────────────┘  └──────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
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

### 3.1 Obligation & Eligibility

Already stubbed in `ObligationEligibility.cs`:

```
Eligible if:
  - Total storage >= 100 GB (MinBlockStoreStorage)
  - Total RAM >= 4 GB (MinBlockStoreRam)
  - DHT obligation Active (dependency in SystemVmDependencies)
```

Storage allocated to the Block Store VM: **~50% of free storage** (after host OS + other VMs), configurable.

### 3.2 Deployment Flow

```
1. Orchestrator: ObligationEligibility computes BlockStore obligation
2. Reconciliation loop: TryDeployAsync checks DHT dependency is Active
3. BlockStoreService.DeployBlockStoreVmAsync():
   a. Calculate storage allocation (min(50% free, 500GB))
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
blockstore-storage-bytes = "107374182400" # allocated storage
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
    /// Standard Block Store node.
    /// libp2p + bitswap + FlatFS uses ~200-300 MB RAM at steady state.
    /// Disk is the primary resource — allocated dynamically based on host capacity.
    /// </summary>
    public static VmSpec Create(long storageBytes) => new()
    {
        VmType = VmType.BlockStore,
        VirtualCpuCores = 1,
        MemoryBytes = 1L * 1024 * 1024 * 1024,      // 1 GB
        DiskBytes = storageBytes,                      // Dynamic (50% of free)
        QualityTier = QualityTier.Burstable,
        ImageId = "debian-12-blockstore",
        ComputePointCost = 1,
    };

    public const long MinStorageBytes = 20L * 1024 * 1024 * 1024;   // 20 GB minimum
    public const long MaxStorageBytes = 500L * 1024 * 1024 * 1024;  // 500 GB cap
}
```

#### NEW: `src/Orchestrator/Services/BlockStoreService.cs`

Interface & service following `IDhtNodeService` / `DhtNodeService` pattern:

```csharp
public interface IBlockStoreService
{
    Task<string?> DeployBlockStoreVmAsync(Node node, IVmService vmService, CancellationToken ct = default);
    Task<List<string>> GetBootstrapPeersAsync(string? excludeNodeId = null);
    Task<BlockStoreStats> GetNetworkStatsAsync();
}
```

Key responsibilities:
- Calculate storage allocation from node's free resources
- Resolve WireGuard mesh labels (reuse DhtNodeService pattern)
- Generate auth token + labels
- Deploy VM via VmService
- Store `BlockStoreInfo` on Node model
- Provide bootstrap peer list for joining nodes

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

GET /api/blockstore/locate/{cid}
  Response: { cid, providers: [{ nodeId, peerId, multiaddr }], replication: count }
  Purpose:  Find which nodes have a specific block (used by clients)

GET /api/blockstore/stats
  Response: { totalNodes, totalCapacity, totalUsed, totalBlocks, avgReplication }
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
    public long CapacityBytes { get; set; }             // Allocated storage
    public long UsedBytes { get; set; }                 // Currently used
    public int BlockCount { get; set; }                 // Local blocks
    public int PinnedBlockCount { get; set; }           // Pinned (won't GC)
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

GET  /stats
  → { capacityBytes, usedBytes, usagePercent, blockCount,
      pinnedCount, connectedPeers, bitswapSent, bitswapReceived }

POST /pin/{cid}
  → { cid, pinned: true }

DELETE /pin/{cid}
  → { cid, pinned: false }

POST /gc
  → { freedBytes, freedBlocks, remainingBytes }

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
├── pins.json              # List of pinned CIDs
├── dynamic-peers          # Runtime peer injection (same as DHT)
└── datastore/             # LevelDB for metadata (block index, stats)
```

#### Block Lifecycle

```
1. PUT /blocks (raw bytes)
   → SHA-256 hash → CIDv1 (raw codec, SHA2-256)
   → Write to FlatFS
   → Auto-pin (configurable)
   → Announce provider record to DHT (via connected DHT node)
   → Return CID

2. GET /blocks/{cid}
   → Check local FlatFS
   → If not found: bitswap request to network
   → Timeout after 30 seconds
   → Return bytes or 404

3. DELETE /blocks/{cid}
   → Remove pin
   → Delete from FlatFS
   → Withdraw provider record from DHT

4. GC (periodic or manual)
   → Find unpinned blocks
   → Sort by last access time (LRU)
   → Delete until under storage quota (90% of capacity)
   → Withdraw provider records for deleted blocks
```

#### Bitswap Integration

The Go binary connects to other Block Store peers via libp2p and uses the bitswap protocol for block exchange:

```go
// Bitswap network: blocks flow between peers automatically
// When a peer requests a block we have → serve it
// When we need a block we don't have → request from peers
bswap := bitswap.New(ctx, network, blockstore)
```

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
| Standard | 1 | 1 GB | Dynamic (20-500 GB) | 100 GB total storage, 4 GB RAM |

Disk allocation formula:
```
freeStorage = node.TotalStorage - node.UsedStorage - reservedForVMs
allocatedStorage = min(freeStorage * 0.5, MaxStorageBytes)
allocatedStorage = max(allocatedStorage, MinStorageBytes)
```

### Network Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 5001 | TCP (libp2p) | Bitswap block exchange |
| 5090 | TCP (HTTP) | Localhost API (internal only) |
| 8080 | TCP (HTTP) | Dashboard (proxied via Nginx port 80) |

---

## 6. Implementation Order

### Phase A: Orchestrator (implement first, deploy last)

1. **BlockStoreVmSpec.cs** — Resource specification
2. **BlockStoreInfo** on Node.cs — Model for tracking state
3. **VmType.BlockStore** — Add to enum
4. **IBlockStoreService / BlockStoreService** — Deployment & peer management
5. **BlockStoreController.cs** — `/api/blockstore/join` + `/locate` + `/stats`
6. **SystemVmLabelSchema** — Add required labels
7. **ObligationEligibility** — Uncomment BlockStore eligibility
8. **SystemVmReconciliationService** — Wire up deployment method
9. **VmService** — Add BlockStore to isSystemVm check
10. **Program.cs** — Register DI

### Phase B: NodeAgent

11. **VmType.BlockStore** — Add to enum
12. **blockstore-vm-cloudinit.yaml** — Cloud-init template
13. **blockstore-node Go binary** — Core storage engine
14. **blockstore-bootstrap-poll.sh** — Orchestrator peer discovery
15. **blockstore-notify-ready.sh** — Callback to NodeAgent
16. **blockstore-health-check.sh** — Health monitoring
17. **blockstore-dashboard.py** — Status dashboard
18. **BlockStoreCallbackController.cs** — NodeAgent callback endpoint
19. **CommandProcessorService** — Handle VmType.BlockStore in CreateVm

### Phase C: Integration & Testing

20. Build Go binary, encode as gzip+base64
21. Test deployment on single node
22. Test multi-node block exchange
23. Test GC under storage pressure
24. Test self-healing (reconciliation redeploy)

---

## 7. How This Connects to Future Features

### Near-Term Uses
- **VM snapshot storage** — Snapshot a VM, store blocks across network, restore anywhere
- **Template image distribution** — Distribute base images via block store (instead of HTTP downloads)
- **User file storage** — Simple put/get API for users to store data persistently

### Mid-Term Uses
- **VM migration** — Transfer VM disks via bitswap (content-addressed = only transfer unique blocks)
- **Distributed backups** — Automatic backup of VM state to multiple block store nodes
- **Content delivery** — Serve static content from nearest block store node

### Long-Term Vision
- **Full IPFS compatibility** — Upgrade to IPFS node for interoperability
- **Erasure coding** — Reed-Solomon coding for storage efficiency
- **Storage marketplace** — Users pay for persistent storage, node operators earn

---

## 8. Key Design Decisions

### Decision 1: Separate VM from DHT (not embedded)
**Why:** DHT VMs are lightweight (512 MB RAM, 2 GB disk). Block storage needs significantly more resources. Keeping them separate means:
- DHT stays fast and lean (just routing)
- Block store can be deployed selectively (only on nodes with storage)
- Independent failure domains (block store crash doesn't break DHT)
- Different scaling profiles

### Decision 2: libp2p bitswap (not custom protocol)
**Why:** Bitswap is battle-tested by IPFS with millions of nodes. It handles:
- Want/have negotiation
- Block prioritization
- Peer reputation
- Parallel downloads from multiple peers
Using a standard protocol also opens the door to IPFS interop later.

### Decision 3: FlatFS storage backend (not LevelDB for blocks)
**Why:** FlatFS stores each block as a separate file on disk. This is:
- Simple to debug (blocks are just files)
- Easy to backup/restore (cp -r)
- Good for large blocks (no LSM tree overhead)
- Proven by IPFS Kubo for years
LevelDB is only used for metadata (pin state, access times, stats).

### Decision 4: Orchestrator maintains global block index
**Why:** While the DHT handles provider records for peer-to-peer discovery, the orchestrator also maintains a lightweight index (`POST /api/blockstore/announce`). This enables:
- Fast block location without DHT queries
- Replication factor tracking
- Storage analytics
- Intelligent placement decisions

### Decision 5: Content-addressed with CIDv1
**Why:** Using IPFS-compatible CIDs (Content IDentifiers) means:
- Deduplication is automatic (same content = same CID)
- Integrity verification is built-in
- Future IPFS interop
- Industry standard format

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large Go binary size | Slow VM boot | Gzip+base64 encoding, ~5-8 MB compressed |
| Storage quota enforcement | Disk full crashes | Hard limit at 95%, GC trigger at 85% |
| Bitswap flood | Network saturation | Rate limiting per peer, bandwidth caps |
| Block loss | Data unavailability | Minimum replication factor of 3 |
| Stale provider records | Failed retrievals | TTL on records, periodic re-announcement |
| DHT dependency | Can't discover peers without DHT | Bootstrap poll as fallback, direct peer list |

---

## 10. Success Metrics

- Block Store VM deploys successfully on eligible nodes
- Blocks survive node restart (persistent FlatFS)
- Cross-node block retrieval works via bitswap
- GC maintains storage within quota
- Bootstrap polling discovers peers within 60 seconds
- Self-healing redeploys on VM failure
