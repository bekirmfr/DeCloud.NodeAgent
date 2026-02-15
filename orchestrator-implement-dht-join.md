# Orchestrator Task: Implement `POST /api/dht/join` Endpoint

## Context

The NodeAgent now deploys DHT VMs with a `dht-bootstrap-poll.sh` script that polls the orchestrator directly for bootstrap peers. This replaces the previous flow where bootstrap peers were baked into cloud-init at deploy time (causing a race condition when PeerId hadn't propagated via heartbeat yet).

The DHT VM calls `POST /api/dht/join` on the orchestrator to:
1. Register its peerId immediately (no heartbeat propagation delay)
2. Receive bootstrap peers to connect to (self-healing for empty bootstrap)

This follows the same pattern as the relay VM's `POST /api/relay/register-callback` in `RelayController.cs`.

## Changes Required

### 1. New File: `src/Orchestrator/Controllers/DhtController.cs`

Create a new controller with a single `POST /api/dht/join` endpoint.

**Request:**
- Body: `{ "nodeId": "...", "vmId": "...", "peerId": "12D3KooW..." }`
- Header: `X-DHT-Token: <HMAC-SHA256(authToken, nodeId:vmId)>`

**Response (200):**
```json
{
  "success": true,
  "bootstrapPeers": ["/ip4/10.20.1.199/tcp/4001/p2p/12D3KooW..."],
  "peerIdRegistered": true
}
```

**Implementation — follow `RelayController.RegisterCallback` pattern exactly:**

```csharp
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Orchestrator.Services;
using Orchestrator.Models;
using Orchestrator.Persistence;
using System.Security.Cryptography;
using System.Text;

namespace Orchestrator.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DhtController : ControllerBase
{
    private readonly IDhtNodeService _dhtNodeService;
    private readonly DataStore _dataStore;
    private readonly ILogger<DhtController> _logger;

    public DhtController(
        IDhtNodeService dhtNodeService,
        DataStore dataStore,
        ILogger<DhtController> logger)
    {
        _dhtNodeService = dhtNodeService;
        _dataStore = dataStore;
        _logger = logger;
    }

    /// <summary>
    /// DHT VM calls this endpoint directly to:
    ///   1. Register its peerId (eliminates heartbeat propagation delay)
    ///   2. Receive bootstrap peers to connect to (self-healing for empty bootstrap)
    ///
    /// Auth: HMAC-SHA256(auth_token, nodeId:vmId)
    /// Same pattern as relay's POST /api/relay/register-callback
    /// uses HMAC(wireguard_private_key, nodeId:vmId).
    /// </summary>
    [HttpPost("join")]
    [AllowAnonymous]
    public async Task<IActionResult> Join(
        [FromBody] DhtJoinRequest request,
        [FromHeader(Name = "X-DHT-Token")] string? token)
    {
        _logger.LogInformation(
            "Received DHT join request from node {NodeId}, VM {VmId}",
            request.NodeId, request.VmId);

        // STEP 1: Validate request
        if (string.IsNullOrEmpty(request.NodeId) ||
            string.IsNullOrEmpty(request.VmId) ||
            string.IsNullOrEmpty(request.PeerId))
        {
            _logger.LogWarning("DHT join rejected: missing required fields");
            return BadRequest(new { error = "Missing required fields (nodeId, vmId, peerId)" });
        }

        // STEP 2: Look up node and DHT info
        var node = await _dataStore.GetNodeAsync(request.NodeId);
        if (node == null)
        {
            _logger.LogWarning("DHT join rejected: node {NodeId} not found", request.NodeId);
            return NotFound(new { error = "Node not found" });
        }

        if (node.DhtInfo == null || node.DhtInfo.DhtVmId != request.VmId)
        {
            _logger.LogWarning(
                "DHT join rejected: node {NodeId} has no DHT VM {VmId}",
                request.NodeId, request.VmId);
            return NotFound(new { error = "DHT VM not found on this node" });
        }

        // STEP 3: Verify authentication token
        if (string.IsNullOrEmpty(node.DhtInfo.AuthToken))
        {
            _logger.LogError(
                "Cannot verify DHT join: AuthToken not found for node {NodeId} DHT VM {VmId}",
                request.NodeId, request.VmId);
            return StatusCode(500, new { error = "DHT auth token not available" });
        }

        var expectedToken = ComputeCallbackToken(
            request.NodeId, request.VmId, node.DhtInfo.AuthToken);

        if (string.IsNullOrEmpty(token))
        {
            _logger.LogWarning(
                "DHT join rejected: missing X-DHT-Token header from node {NodeId}",
                request.NodeId);
            return Unauthorized(new { error = "Missing authentication token" });
        }

        if (!CryptographicOperations.FixedTimeEquals(
            Encoding.UTF8.GetBytes(token),
            Encoding.UTF8.GetBytes(expectedToken)))
        {
            _logger.LogWarning(
                "DHT join rejected: invalid token from node {NodeId}/{VmId}",
                request.NodeId, request.VmId);
            return Unauthorized(new { error = "Invalid authentication token" });
        }

        _logger.LogInformation(
            "DHT join authenticated for node {NodeId}", request.NodeId);

        // STEP 4: Register peerId (immediate — no heartbeat delay)
        var peerIdUpdated = false;

        if (string.IsNullOrEmpty(node.DhtInfo.PeerId) || node.DhtInfo.PeerId != request.PeerId)
        {
            node.DhtInfo.PeerId = request.PeerId;
            node.DhtInfo.Status = DhtStatus.Active;
            node.DhtInfo.LastHealthCheck = DateTime.UtcNow;
            peerIdUpdated = true;

            await _dataStore.SaveNodeAsync(node);

            _logger.LogInformation(
                "DHT peer ID registered via /api/dht/join for node {NodeId}: {PeerId}",
                request.NodeId,
                request.PeerId.Length > 12 ? request.PeerId[..12] + "..." : request.PeerId);
        }

        // STEP 5: Return bootstrap peers (excluding this node)
        var bootstrapPeers = await _dhtNodeService.GetBootstrapPeersAsync(
            excludeNodeId: request.NodeId);

        _logger.LogInformation(
            "DHT join response for node {NodeId}: {PeerCount} bootstrap peers, peerId {Action}",
            request.NodeId, bootstrapPeers.Count,
            peerIdUpdated ? "registered" : "already known");

        return Ok(new
        {
            success = true,
            bootstrapPeers,
            peerIdRegistered = peerIdUpdated
        });
    }

    /// <summary>
    /// Compute HMAC-SHA256 callback token using the DHT auth token.
    /// Same pattern as RelayController.ComputeCallbackToken uses WireGuard private key.
    /// </summary>
    private static string ComputeCallbackToken(string nodeId, string vmId, string authToken)
    {
        var message = $"{nodeId}:{vmId}";
        var secret = authToken.Trim();

        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(secret));
        var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(message));

        return Convert.ToBase64String(hash);
    }
}

public record DhtJoinRequest(
    string NodeId,
    string VmId,
    string PeerId
);
```

### 2. Modify: `src/Orchestrator/Models/Node.cs`

Add `AuthToken` field to `DhtNodeInfo` class (after the `LastHealthCheck` property, around line 723):

```csharp
    public DhtStatus Status { get; set; } = DhtStatus.Initializing;
    public DateTime? LastHealthCheck { get; set; }

    /// <summary>
    /// HMAC auth token for DHT VM → orchestrator direct authentication.
    /// Generated at deployment time, injected into cloud-init via labels.
    /// The DHT VM uses HMAC-SHA256(AuthToken, nodeId:vmId) to authenticate
    /// with the orchestrator's /api/dht/join endpoint.
    /// Same pattern as relay uses WireGuardPrivateKey for auth.
    /// </summary>
    public string? AuthToken { get; set; }
```

Also update the PeerId doc comment from `"via qemu-guest-agent"` to `"via /api/dht/join or heartbeat"`.

### 3. Modify: `src/Orchestrator/Services/DhtNodeService.cs`

**3a.** Add using at top:

```csharp
using System.Security.Cryptography;
```

**3b.** In `DeployDhtVmAsync`, generate auth token before building labels (after the WireGuard tunnel IP override block, before `var labels = new Dictionary`):

```csharp
            // Generate auth token for DHT VM → orchestrator direct authentication.
            // Same pattern as relay uses WireGuardPrivateKey: both sides know the secret,
            // DHT VM computes HMAC(authToken, nodeId:vmId) to authenticate with /api/dht/join.
            var authToken = Convert.ToBase64String(RandomNumberGenerator.GetBytes(32));
```

**3c.** Add to the labels dictionary:

```csharp
                { "dht-auth-token", authToken },
```

**3d.** Store auth token in DhtNodeInfo:

```csharp
            node.DhtInfo = new DhtNodeInfo
            {
                DhtVmId = dhtVm.VmId,
                ListenAddress = $"{advertiseIp}:{DhtListenPort}",
                ApiPort = DhtApiPort,
                BootstrapPeerCount = bootstrapPeers.Count,
                Status = DhtStatus.Initializing,
                AuthToken = authToken,   // ← ADD THIS
            };
```

### 4. Modify: `src/Orchestrator/Services/SystemVm/SystemVmReconciliationService.cs`

Remove the bootstrap peer deferral guard (lines 144-167 approximately). This block starts with the comment `"Bootstrap peer prerequisite"` and contains the `GetPendingPeerIdCountAsync` call. Replace it with:

```csharp
        // NOTE: Bootstrap peer deferral guard was removed.
        // DHT VMs now self-heal via dht-bootstrap-poll.sh which polls
        // POST /api/dht/join for bootstrap peers. Empty bootstrap at
        // deploy time is no longer a problem — the VM will discover
        // peers within ~15 seconds of starting.
```

The removed block is the one that calls `_dhtNodeService.GetPendingPeerIdCountAsync(excludeNodeId: node.Id)` and returns early if pending > 0 and existing peers == 0. This deferral is now counterproductive because DHT VMs should deploy immediately and self-heal via polling.

### 5. Modify: `src/Orchestrator/Services/DhtNodeService.cs` — Remove `GetPendingPeerIdCountAsync`

With the deferral guard removed, `GetPendingPeerIdCountAsync` is dead code (no callers remain). Remove it from both the interface and the implementation.

**5a.** Remove from `IDhtNodeService` interface (around line 24-29):

```csharp
    // DELETE THIS:
    Task<int> GetPendingPeerIdCountAsync(string? excludeNodeId = null);
```

**5b.** Remove the entire `GetPendingPeerIdCountAsync` method implementation (around lines 235-273). This is the block starting with:

```csharp
    public async Task<int> GetPendingPeerIdCountAsync(string? excludeNodeId = null)
```

Delete all the way through its closing brace.

## How It All Connects

```
                    ┌─────────────────────────────────────────────────┐
                    │  Orchestrator deploys DHT VM                    │
                    │  1. Generates authToken (random 32 bytes)       │
                    │  2. Stores in DhtNodeInfo.AuthToken              │
                    │  3. Passes as label: dht-auth-token={token}     │
                    └────────────────────┬────────────────────────────┘
                                         │ CreateVm command
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │  NodeAgent receives command                     │
                    │  1. Reads dht-auth-token from labels            │
                    │  2. Injects __DHT_AUTH_TOKEN__ into cloud-init  │
                    │  3. Injects __ORCHESTRATOR_URL__ into cloud-init│
                    └────────────────────┬────────────────────────────┘
                                         │ VM boots
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │  DHT VM (dht-bootstrap-poll.sh)                 │
                    │  1. Waits for DHT binary, gets peerId           │
                    │  2. Computes HMAC(authToken, nodeId:vmId)       │
                    │  3. POST /api/dht/join → orchestrator           │
                    │     sends: {nodeId, vmId, peerId}               │
                    │     receives: {bootstrapPeers: [...]}           │
                    │  4. POST /connect → local DHT binary            │
                    │  5. Repeats every 15s while isolated            │
                    └────────────────────┬────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │  Orchestrator /api/dht/join handler             │
                    │  1. Verifies HMAC against stored AuthToken      │
                    │  2. Stores PeerId on node.DhtInfo               │
                    │  3. Returns GetBootstrapPeersAsync() result     │
                    └─────────────────────────────────────────────────┘
```

## Verification

After implementing, verify:
1. `DhtController` is auto-discovered by ASP.NET (route: `POST /api/dht/join`)
2. Auth token flows: `DhtNodeService.DeployDhtVmAsync` → labels → cloud-init → `dht-bootstrap-poll.sh` → `X-DHT-Token` header → `DhtController.Join` verifies against `DhtNodeInfo.AuthToken`
3. PeerId is registered immediately on first successful `/api/dht/join` call
4. Bootstrap peers are returned from `GetBootstrapPeersAsync(excludeNodeId)`
5. The deferral guard in `SystemVmReconciliationService.TryDeployAsync` is removed
6. `GetPendingPeerIdCountAsync` is removed from `IDhtNodeService` interface and `DhtNodeService` implementation (dead code)
7. Build succeeds with `dotnet build`
