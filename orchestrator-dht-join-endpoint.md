# Orchestrator: `POST /api/dht/join` Endpoint Spec

## Overview

New orchestrator endpoint that DHT VMs call directly to:
1. **Register their peerId** (replaces heartbeat-based propagation)
2. **Receive bootstrap peers** (replaces baked-in `DHT_BOOTSTRAP_PEERS`)

This follows the same pattern as the relay's `POST /api/relay/register-callback`
where the relay VM authenticates directly with the orchestrator.

## Endpoint

```
POST /api/dht/join
```

### Request Headers

| Header | Value | Description |
|--------|-------|-------------|
| `Content-Type` | `application/json` | |
| `X-DHT-Token` | `HMAC-SHA256(auth_token, nodeId:vmId)` | Same pattern as relay's `X-Relay-Token` |

### Request Body

```json
{
    "nodeId": "e9277b2c-614d-a8cb-6487-54395b6d7880",
    "vmId": "a909af8f-11ac-43a6-a206-ded998bf2ce7",
    "peerId": "12D3KooWL9W2yvrEJ6Fjv32ATWvJ5fVKTu58oNzYMXg3xeW7ZCu1"
}
```

### Response (200 OK)

```json
{
    "success": true,
    "bootstrapPeers": [
        "/ip4/10.20.1.199/tcp/4001/p2p/12D3KooWL9W2yvrEJ6Fjv32ATWvJ5fVKTu58oNzYMXg3xeW7ZCu1",
        "/ip4/10.20.2.199/tcp/4001/p2p/12D3KooWP5SkPQGKpaBJ5CGsh7rXed7Y9Ra6UUuFJwtjAzQayYHy"
    ],
    "peerIdRegistered": true
}
```

### Error Responses

| Code | Reason |
|------|--------|
| 400 | Missing required fields |
| 401 | Missing `X-DHT-Token` header |
| 403 | Invalid token (HMAC verification failed) |
| 404 | Node or VM not found |

## Authentication

### Token Generation (DHT VM side — in `dht-bootstrap-poll.sh`)

```bash
AUTH_TOKEN="__DHT_AUTH_TOKEN__"    # Injected via cloud-init from orchestrator label
NODE_ID="__NODE_ID__"
VM_ID="__VM_ID__"

MESSAGE="${NODE_ID}:${VM_ID}"
TOKEN=$(echo -n "$MESSAGE" | openssl dgst -sha256 -hmac "$AUTH_TOKEN" -binary | base64)
```

### Token Verification (Orchestrator side)

```csharp
// 1. Look up the node by nodeId
var node = await _dataStore.GetNodeAsync(request.NodeId);
if (node == null) return NotFound();

// 2. Find the DHT obligation and its auth token
var dhtObligation = node.SystemVmObligations
    .FirstOrDefault(o => o.Role == SystemVmRole.Dht && o.VmId == request.VmId);
if (dhtObligation == null) return NotFound();

// 3. Retrieve the stored auth token (set when the CreateVm command was issued)
var storedToken = dhtObligation.AuthToken;  // or node.DhtInfo.AuthToken

// 4. Compute expected HMAC
var message = $"{request.NodeId}:{request.VmId}";
using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(storedToken));
var expectedHash = Convert.ToBase64String(hmac.ComputeHash(Encoding.UTF8.GetBytes(message)));

// 5. Constant-time comparison
if (!CryptographicOperations.FixedTimeEquals(
    Encoding.UTF8.GetBytes(requestToken),
    Encoding.UTF8.GetBytes(expectedHash)))
    return Forbid();
```

### Auth Token Lifecycle

1. **Generation**: Orchestrator generates a random token when scheduling the DHT VM
   ```csharp
   var authToken = Convert.ToBase64String(RandomNumberGenerator.GetBytes(32));
   ```

2. **Storage**: Stored in the DHT obligation or node metadata
   ```csharp
   dhtObligation.AuthToken = authToken;
   ```

3. **Injection**: Passed via CreateVm command labels
   ```csharp
   labels["dht-auth-token"] = authToken;
   ```

4. **Usage**: DHT VM reads from cloud-init env, computes HMAC per request

## Orchestrator Handler Logic

```csharp
[HttpPost("join")]
public async Task<IActionResult> DhtJoin(
    [FromBody] DhtJoinRequest request,
    [FromHeader(Name = "X-DHT-Token")] string? token)
{
    // 1. Validate + authenticate (see above)

    // 2. Register peerId — eliminates heartbeat propagation delay
    var node = await _dataStore.GetNodeAsync(request.NodeId);
    node.DhtInfo ??= new DhtInfo();
    node.DhtInfo.PeerId = request.PeerId;
    await _dataStore.SaveNodeAsync(node);

    _logger.LogInformation(
        "DHT peer ID registered via /api/dht/join for node {NodeId}: {PeerId}",
        request.NodeId, request.PeerId[..12]);

    // 3. Return bootstrap peers (excluding the requesting node)
    var bootstrapPeers = await _dhtNodeService.GetBootstrapPeersAsync(
        excludeNodeId: request.NodeId);

    return Ok(new
    {
        success = true,
        bootstrapPeers,
        peerIdRegistered = true
    });
}
```

## What This Replaces

### Before (heartbeat-based flow)
```
DHT VM → dht-notify-ready.sh → NodeAgent /api/dht/ready
    → stores in StatusMessage → heartbeat (15s cycle)
    → orchestrator extracts peerId → GetBootstrapPeersAsync()
    → baked into next DHT VM's cloud-init
```
**Latency**: 4+ minutes (VM boot + cloud-init + heartbeat cycle)
**Race condition**: New node registers before peerId propagates → 0 bootstrap peers

### After (direct orchestrator polling)
```
DHT VM → dht-bootstrap-poll.sh → orchestrator /api/dht/join
    → registers peerId immediately + returns bootstrap peers
    → DHT VM calls POST /connect on local DHT binary
```
**Latency**: ~15 seconds (poll interval while isolated)
**No race condition**: peerId registered on first successful poll; bootstrap peers returned in same response

## Orchestrator Changes Required

1. **New endpoint**: `POST /api/dht/join` (controller + request/response models)
2. **Auth token generation**: In `SystemVmReconciliationService` when scheduling DHT VMs
3. **Auth token in labels**: Add `dht-auth-token` to CreateVm command labels
4. **Auth token storage**: Add `AuthToken` field to `SystemVmObligation` (or `DhtInfo`)
5. **Optional**: Remove `GetPendingPeerIdCountAsync()` deferral logic (no longer needed)
