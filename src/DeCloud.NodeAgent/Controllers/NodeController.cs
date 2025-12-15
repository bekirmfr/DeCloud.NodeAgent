using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Services;
using Microsoft.AspNetCore.Mvc;

namespace DeCloud.NodeAgent.Controllers;

[ApiController]
[Route("api/[controller]")]
public class NodeController : ControllerBase
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly INetworkManager _networkManager;
    private readonly HeartbeatService _heartBeatService;
    private readonly ILogger<NodeController> _logger;

    public NodeController(
        IResourceDiscoveryService resourceDiscovery,
        INetworkManager networkManager,
        HeartbeatService _heartBeatService,
        ILogger<NodeController> logger)
    {
        _resourceDiscovery = resourceDiscovery;
        _networkManager = networkManager;
        this._heartBeatService = _heartBeatService;
        _logger = logger;
    }

    /// <summary>
    /// Get full node resource inventory
    /// </summary>
    [HttpGet("resources")]
    public async Task<ActionResult<NodeResources>> GetResources(CancellationToken ct)
    {
        var resources = await _resourceDiscovery.DiscoverAllAsync(ct);
        return Ok(resources);
    }

    /// <summary>
    /// Get current resource snapshot (quick)
    /// </summary>
    [HttpGet("snapshot")]
    public async Task<ActionResult<ResourceSnapshot>> GetSnapshot(CancellationToken ct)
    {
        var snapshot = await _resourceDiscovery.GetCurrentSnapshotAsync(ct);
        return Ok(snapshot);
    }

    /// <summary>
    /// Get CPU information
    /// </summary>
    [HttpGet("cpu")]
    public async Task<ActionResult<CpuInfo>> GetCpu(CancellationToken ct)
    {
        var cpu = await _resourceDiscovery.GetCpuInfoAsync(ct);
        return Ok(cpu);
    }

    /// <summary>
    /// Get memory information
    /// </summary>
    [HttpGet("memory")]
    public async Task<ActionResult<MemoryInfo>> GetMemory(CancellationToken ct)
    {
        var memory = await _resourceDiscovery.GetMemoryInfoAsync(ct);
        return Ok(memory);
    }

    /// <summary>
    /// Get storage information
    /// </summary>
    [HttpGet("storage")]
    public async Task<ActionResult<List<StorageInfo>>> GetStorage(CancellationToken ct)
    {
        var storage = await _resourceDiscovery.GetStorageInfoAsync(ct);
        return Ok(storage);
    }

    /// <summary>
    /// Get GPU information
    /// </summary>
    [HttpGet("gpus")]
    public async Task<ActionResult<List<GpuInfo>>> GetGpus(CancellationToken ct)
    {
        var gpus = await _resourceDiscovery.GetGpuInfoAsync(ct);
        return Ok(gpus);
    }

    /// <summary>
    /// Get network information
    /// </summary>
    [HttpGet("network")]
    public async Task<ActionResult<NetworkInfo>> GetNetwork(CancellationToken ct)
    {
        var network = await _resourceDiscovery.GetNetworkInfoAsync(ct);
        return Ok(network);
    }

    /// <summary>
    /// Get WireGuard peers
    /// </summary>
    [HttpGet("wireguard/peers")]
    public async Task<ActionResult<List<WireGuardPeer>>> GetWireGuardPeers(CancellationToken ct)
    {
        var peers = await _networkManager.GetPeersAsync(ct);
        return Ok(peers);
    }

    /// <summary>
    /// Get WireGuard public key
    /// </summary>
    [HttpGet("wireguard/pubkey")]
    public async Task<ActionResult<string>> GetWireGuardPublicKey(CancellationToken ct)
    {
        var pubkey = await _networkManager.GetWireGuardPublicKeyAsync(ct);
        return Ok(pubkey);
    }

    /// <summary>
    /// Health check endpoint
    /// </summary>
    [HttpGet("health")]
    public IActionResult Health()
    {
        return Ok(new { status = "healthy", timestamp = DateTime.UtcNow });
    }

    [HttpGet("heartbeat")]
    public IActionResult Heartbeat()
    {
        var lastHeartbeat = _heartBeatService.GetLastHeartbeat();
        if (lastHeartbeat != null)
        {
            return Ok(lastHeartbeat);
        }
        else
        {
            return NotFound("No heartbeat data available.");
        }
    }
}
