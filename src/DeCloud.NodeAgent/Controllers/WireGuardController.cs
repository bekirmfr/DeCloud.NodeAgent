using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// API endpoints for WireGuard overlay network management
/// </summary>
[ApiController]
[Route("api/node/wireguard")]
public class WireGuardController : ControllerBase
{
    private readonly INetworkManager _networkManager;
    private readonly ILogger<WireGuardController> _logger;

    public WireGuardController(
        INetworkManager networkManager,
        ILogger<WireGuardController> logger)
    {
        _networkManager = networkManager;
        _logger = logger;
    }

    /// <summary>
    /// Get the node's WireGuard public key
    /// </summary>
    [HttpGet("pubkey")]
    [AllowAnonymous]
    [ProducesResponseType(typeof(WireGuardPubKeyResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> GetPublicKey(CancellationToken ct)
    {
        try
        {
            var publicKey = await _networkManager.GetWireGuardPublicKeyAsync(ct);

            if (string.IsNullOrEmpty(publicKey))
            {
                return StatusCode(503, new { error = "WireGuard not configured on this node" });
            }

            return Ok(new WireGuardPubKeyResponse
            {
                PublicKey = publicKey,
                Endpoint = GetNodeEndpoint()
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get WireGuard public key");
            return StatusCode(500, new { error = "Failed to retrieve public key" });
        }
    }

    /// <summary>
    /// Get WireGuard status and all peers
    /// </summary>
    [HttpGet("status")]
    [ProducesResponseType(typeof(WireGuardStatusResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetStatus(CancellationToken ct)
    {
        try
        {
            var publicKey = await _networkManager.GetWireGuardPublicKeyAsync(ct);
            var peers = await _networkManager.GetPeersAsync(ct);

            return Ok(new WireGuardStatusResponse
            {
                IsConfigured = !string.IsNullOrEmpty(publicKey),
                PublicKey = publicKey,
                Endpoint = GetNodeEndpoint(),
                PeerCount = peers.Count,
                Peers = peers.Select(p => new WireGuardPeerResponse
                {
                    PublicKey = p.PublicKey,
                    Endpoint = p.Endpoint,
                    AllowedIps = p.AllowedIps,
                    LastHandshake = p.LastHandshake,
                    TransferRx = p.TransferRx,
                    TransferTx = p.TransferTx
                }).ToList()
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get WireGuard status");
            return StatusCode(500, new { error = "Failed to retrieve status" });
        }
    }

    /// <summary>
    /// List all WireGuard peers
    /// </summary>
    [HttpGet("peers")]
    [ProducesResponseType(typeof(List<WireGuardPeerResponse>), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetPeers(CancellationToken ct)
    {
        try
        {
            var peers = await _networkManager.GetPeersAsync(ct);

            return Ok(peers.Select(p => new WireGuardPeerResponse
            {
                PublicKey = p.PublicKey,
                Endpoint = p.Endpoint,
                AllowedIps = p.AllowedIps,
                LastHandshake = p.LastHandshake,
                TransferRx = p.TransferRx,
                TransferTx = p.TransferTx
            }).ToList());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get WireGuard peers");
            return StatusCode(500, new { error = "Failed to retrieve peers" });
        }
    }

    /// <summary>
    /// Add a WireGuard peer
    /// </summary>
    [HttpPost("peers")]
    [ProducesResponseType(typeof(WireGuardPeerResponse), StatusCodes.Status201Created)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> AddPeer([FromBody] AddWireGuardPeerRequest request, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.PublicKey))
        {
            return BadRequest(new { error = "PublicKey is required" });
        }

        if (string.IsNullOrWhiteSpace(request.AllowedIps))
        {
            return BadRequest(new { error = "AllowedIps is required" });
        }

        try
        {
            _logger.LogInformation("Adding WireGuard peer: {PublicKey} with allowed IPs: {AllowedIps}",
                request.PublicKey[..Math.Min(20, request.PublicKey.Length)], request.AllowedIps);

            await _networkManager.AddPeerAsync(
                request.PublicKey,
                request.Endpoint ?? "",
                request.AllowedIps,
                ct);

            return Created($"/api/node/wireguard/peers/{request.PublicKey}", new WireGuardPeerResponse
            {
                PublicKey = request.PublicKey,
                Endpoint = request.Endpoint ?? "",
                AllowedIps = request.AllowedIps,
                LastHandshake = null,
                TransferRx = 0,
                TransferTx = 0
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to add WireGuard peer");
            return StatusCode(500, new { error = $"Failed to add peer: {ex.Message}" });
        }
    }

    /// <summary>
    /// Remove a WireGuard peer
    /// </summary>
    [HttpDelete("peers/{publicKey}")]
    [ProducesResponseType(StatusCodes.Status204NoContent)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> RemovePeer(string publicKey, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(publicKey))
        {
            return BadRequest(new { error = "PublicKey is required" });
        }

        try
        {
            // URL decode the public key (it may contain + and / characters)
            var decodedKey = Uri.UnescapeDataString(publicKey);

            _logger.LogInformation("Removing WireGuard peer: {PublicKey}",
                decodedKey[..Math.Min(20, decodedKey.Length)]);

            await _networkManager.RemovePeerAsync(decodedKey, ct);

            return NoContent();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to remove WireGuard peer");
            return StatusCode(500, new { error = $"Failed to remove peer: {ex.Message}" });
        }
    }

    /// <summary>
    /// Generate a client configuration template
    /// </summary>
    [HttpGet("client-config")]
    [AllowAnonymous]
    [ProducesResponseType(typeof(WireGuardClientConfigResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetClientConfig([FromQuery] string? clientIp, CancellationToken ct)
    {
        try
        {
            var publicKey = await _networkManager.GetWireGuardPublicKeyAsync(ct);

            if (string.IsNullOrEmpty(publicKey))
            {
                return StatusCode(503, new { error = "WireGuard not configured on this node" });
            }

            var endpoint = GetNodeEndpoint();
            var assignedIp = clientIp ?? "10.10.0.2";

            var configTemplate = $@"[Interface]
PrivateKey = <YOUR_PRIVATE_KEY>
Address = {assignedIp}/24

[Peer]
PublicKey = {publicKey}
Endpoint = {endpoint}
AllowedIPs = 10.10.0.0/24, 192.168.122.0/24
PersistentKeepalive = 25";

            return Ok(new WireGuardClientConfigResponse
            {
                HubPublicKey = publicKey,
                HubEndpoint = endpoint,
                SuggestedClientIp = assignedIp,
                ConfigTemplate = configTemplate,
                Instructions = new List<string>
                {
                    "1. Generate a private key: wg genkey",
                    "2. Derive your public key: echo '<private-key>' | wg pubkey",
                    "3. Replace <YOUR_PRIVATE_KEY> in the config with your private key",
                    "4. Save as wg-decloud.conf and import into WireGuard",
                    $"5. Register with hub: POST /api/node/wireguard/peers with your public key and allowed IP {assignedIp}/32",
                    "6. Connect and access VMs at 192.168.122.x"
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to generate client config");
            return StatusCode(500, new { error = "Failed to generate client config" });
        }
    }

    private string GetNodeEndpoint()
    {
        // Try to get from configuration or detect
        var config = HttpContext.RequestServices.GetService<IConfiguration>();
        var publicIp = config?["Node:PublicIp"] ?? "";
        var port = config?["WireGuard:ListenPort"] ?? "51820";

        if (string.IsNullOrEmpty(publicIp))
        {
            // Fallback to request host (may not be accurate behind NAT)
            publicIp = HttpContext.Connection.LocalIpAddress?.ToString() ?? "unknown";
        }

        return $"{publicIp}:{port}";
    }
}

#region DTOs

public class WireGuardPubKeyResponse
{
    public string PublicKey { get; set; } = "";
    public string Endpoint { get; set; } = "";
}

public class WireGuardStatusResponse
{
    public bool IsConfigured { get; set; }
    public string PublicKey { get; set; } = "";
    public string Endpoint { get; set; } = "";
    public int PeerCount { get; set; }
    public List<WireGuardPeerResponse> Peers { get; set; } = new();
}

public class WireGuardPeerResponse
{
    public string PublicKey { get; set; } = "";
    public string Endpoint { get; set; } = "";
    public string AllowedIps { get; set; } = "";
    public DateTime? LastHandshake { get; set; }
    public long TransferRx { get; set; }
    public long TransferTx { get; set; }
}

public class AddWireGuardPeerRequest
{
    /// <summary>
    /// The client's WireGuard public key
    /// </summary>
    public string PublicKey { get; set; } = "";

    /// <summary>
    /// Optional: The client's endpoint (IP:Port) if known
    /// </summary>
    public string? Endpoint { get; set; }

    /// <summary>
    /// The IP address(es) to allow from this peer (e.g., "10.10.0.2/32")
    /// </summary>
    public string AllowedIps { get; set; } = "";
}

public class WireGuardClientConfigResponse
{
    public string HubPublicKey { get; set; } = "";
    public string HubEndpoint { get; set; } = "";
    public string SuggestedClientIp { get; set; } = "";
    public string ConfigTemplate { get; set; } = "";
    public List<string> Instructions { get; set; } = new();
}

#endregion