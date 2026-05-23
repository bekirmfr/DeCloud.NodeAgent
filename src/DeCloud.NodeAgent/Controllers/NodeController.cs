using DeCloud.NodeAgent.Contracts.Response.Network;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Services;
using DeCloud.Shared.Models;
using Microsoft.AspNetCore.Mvc;

namespace DeCloud.NodeAgent.Controllers;

[ApiController]
[Route("api/[controller]")]
public class NodeController : ControllerBase
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly INodeMetadataService _nodeMetadata;
    private readonly INodeStateService _nodeStateService;
    private readonly ILogger<NodeController> _logger;

    public NodeController(
        IResourceDiscoveryService resourceDiscovery,
        IOrchestratorClient orchestratorClient,
        INodeMetadataService nodeMetadata,
        INodeStateService nodeStateService,
        ILogger<NodeController> logger)
    {
        _resourceDiscovery = resourceDiscovery;
        _orchestratorClient = orchestratorClient;
        _nodeMetadata = nodeMetadata;
        _nodeStateService = nodeStateService;
        _logger = logger;
    }

    /// <summary>
    /// Get full node resource inventory
    /// Pass ?cached=true to return the cached inventory without running
    /// a fresh CPU benchmark. Returns 404 if the cache is empty (agent
    /// just started, first heartbeat hasn't run yet).
    /// </summary>
    [HttpGet("resources")]
    public async Task<ActionResult<HardwareInventory>> GetResources(
        [FromQuery] bool cached = false, CancellationToken ct = default)
    {
        var resources = cached
            ? await _resourceDiscovery.GetInventoryCachedAsync(ct)
            : await _resourceDiscovery.DiscoverAllAsync(ct);

        if (resources == null)
            return NotFound("No cached inventory available yet — try again after the first heartbeat cycle.");

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
    /// Get network information
    /// </summary>
    [HttpGet("network/status")]
    public async Task<ActionResult<NetworkStatusResponse>> GetNetworkStatus(CancellationToken ct)
    {
        var networkStatusResponse = new NetworkStatusResponse
        {
            IsInternetReachable = _nodeStateService.IsInternetReachable,
            IsOrchestratorReachable = _nodeStateService.IsOrchestratorReachable,
        };
        return Ok(networkStatusResponse);
    }

    /// <summary>
    /// Health check endpoint
    /// </summary>
    [HttpGet("health")]
    public IActionResult Health()
    {
        return Ok(new { status = "healthy", timestamp = DateTime.UtcNow });
    }

    /// <summary>
    /// Handles HTTP GET requests to retrieve the most recent heartbeat data.
    /// </summary>
    /// <returns>An <see cref="OkObjectResult"/> containing the latest heartbeat data if available; otherwise, a <see
    /// cref="NotFoundObjectResult"/> indicating that no heartbeat data is present.</returns>
    [HttpGet("heartbeat")]
    public IActionResult Heartbeat()
    {
        var lastHeartbeat = _orchestratorClient.GetLastHeartbeat();
        if (lastHeartbeat == null)
        {
            return NotFound("No heartbeat data available.");
        }
        return Ok(lastHeartbeat);
    }


    [HttpGet("allocation")]
    public IActionResult GetAllocation()
    {
        var m = _nodeMetadata;
        return Ok(new
        {
            cpuPercent = (m.AllocatedComputePointsPercent ?? 90) / 100.0,
            memoryPercent = m.AllocatedMemoryPercent is int mp ? mp / 100.0 : AllocatedResources.DefaultPercent,
            storagePercent = m.AllocatedStoragePercent is int sp ? sp / 100.0 : AllocatedResources.DefaultPercent,
            gpuCount = m.AllocatedGpuCount,
            resolvedComputePoints = m.AllocatedComputePoints ?? 0,
            resolvedMemoryBytes = m.AllocatedMemoryBytes ?? 0,
            resolvedStorageBytes = m.AllocatedStorageBytes ?? 0,
            resolvedAt = m.AllocationResolvedAt,
            source = m.AllocationResolvedAt.HasValue ? "orchestrator" : "settings"
        });
    }
}