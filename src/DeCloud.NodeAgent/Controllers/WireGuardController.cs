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
    private readonly ICommandExecutor _executor;
    private readonly ILogger<WireGuardController> _logger;

    public WireGuardController(
        INetworkManager networkManager,
        ICommandExecutor executor,
        ILogger<WireGuardController> logger)
    {
        _networkManager = networkManager;
        _executor = executor;
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
            return StatusCode(500, new { error = "Failed to get public key" });
        }
    }

    /// <summary>
    /// Get all active WireGuard interfaces
    /// </summary>
    [HttpGet("interfaces")]
    [ProducesResponseType(typeof(List<WireGuardInterfaceInfo>), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetInterfaces(CancellationToken ct)
    {
        try
        {
            var result = await _executor.ExecuteAsync("wg", "show interfaces", ct);

            if (!result.Success || string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                return Ok(new List<WireGuardInterfaceInfo>());
            }

            var interfaces = result.StandardOutput
                .Split(new[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(name => new WireGuardInterfaceInfo { Name = name })
                .ToList();

            return Ok(interfaces);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get WireGuard interfaces");
            return StatusCode(500, new { error = "Failed to get interfaces" });
        }
    }

    /// <summary>
    /// Get peers for a specific interface
    /// </summary>
    [HttpGet("peers/{interfaceName}")]
    [ProducesResponseType(typeof(List<WireGuardPeer>), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetPeers(string interfaceName, CancellationToken ct)
    {
        try
        {
            var peers = await _networkManager.GetPeersAsync(interfaceName, ct);
            return Ok(peers);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get peers for interface {Interface}", interfaceName);
            return StatusCode(500, new { error = "Failed to get peers" });
        }
    }

    /// <summary>
    /// Get peers for all active interfaces
    /// </summary>
    [HttpGet("peers")]
    [ProducesResponseType(typeof(Dictionary<string, List<WireGuardPeer>>), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetAllPeers(CancellationToken ct)
    {
        try
        {
            // Get all interfaces
            var interfacesResult = await _executor.ExecuteAsync("wg", "show interfaces", ct);

            if (!interfacesResult.Success || string.IsNullOrWhiteSpace(interfacesResult.StandardOutput))
            {
                return Ok(new Dictionary<string, List<WireGuardPeer>>());
            }

            var interfaces = interfacesResult.StandardOutput
                .Split(new[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                .ToList();

            var allPeers = new Dictionary<string, List<WireGuardPeer>>();

            foreach (var iface in interfaces)
            {
                var peers = await _networkManager.GetPeersAsync(iface, ct);
                allPeers[iface] = peers;
            }

            return Ok(allPeers);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get all peers");
            return StatusCode(500, new { error = "Failed to get all peers" });
        }
    }

    private string GetNodeEndpoint()
    {
        // Get public IP and port
        // This would need to be implemented based on your setup
        return "unknown";
    }
}

public class WireGuardPubKeyResponse
{
    public string PublicKey { get; set; } = string.Empty;
    public string Endpoint { get; set; } = string.Empty;
}

public class WireGuardInterfaceInfo
{
    public string Name { get; set; } = string.Empty;
}