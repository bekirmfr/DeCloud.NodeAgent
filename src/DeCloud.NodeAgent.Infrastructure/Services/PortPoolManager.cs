using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Manages the pool of available ports for VM port mappings.
/// Port range: 40000-65535 (25,535 ports available)
///
/// Uses database-backed allocation to survive node restarts.
/// </summary>
public interface IPortPoolManager
{
    /// <summary>
    /// Allocate a free port from the pool
    /// </summary>
    Task<int?> AllocatePortAsync(CancellationToken ct = default);

    /// <summary>
    /// Release a port back to the pool (done via repository removal)
    /// </summary>
    Task ReleasePortAsync(int port, CancellationToken ct = default);

    /// <summary>
    /// Get available port count
    /// </summary>
    Task<int> GetAvailablePortCountAsync(CancellationToken ct = default);

    /// <summary>
    /// Check if a specific port is available
    /// </summary>
    Task<bool> IsPortAvailableAsync(int port, CancellationToken ct = default);

    /// <summary>
    /// Get utilization statistics
    /// </summary>
    Task<(int total, int used, double utilization)> GetUtilizationAsync(CancellationToken ct = default);
}

public class PortPoolManager : IPortPoolManager
{
    private const int POOL_START = 40000;
    private const int POOL_END = 65535;
    private const int TOTAL_PORTS = POOL_END - POOL_START + 1; // 25,535 ports

    private readonly PortMappingRepository _repository;
    private readonly ILogger<PortPoolManager> _logger;
    private readonly SemaphoreSlim _lock = new(1, 1);

    // Cache of allocated ports (loaded from database)
    private HashSet<int>? _allocatedPortsCache;
    private DateTime _cacheLastRefreshed = DateTime.MinValue;
    private static readonly TimeSpan CacheDuration = TimeSpan.FromSeconds(30);

    public PortPoolManager(
        PortMappingRepository repository,
        ILogger<PortPoolManager> logger)
    {
        _repository = repository;
        _logger = logger;
    }

    public async Task<int?> AllocatePortAsync(CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            // Get currently allocated ports
            var allocatedPorts = await GetOrRefreshAllocatedPortsAsync(ct);

            // Find first available port in the pool
            for (int port = POOL_START; port <= POOL_END; port++)
            {
                if (!allocatedPorts.Contains(port))
                {
                    _logger.LogDebug("Allocated port {Port} from pool", port);

                    // Add to cache
                    allocatedPorts.Add(port);

                    return port;
                }
            }

            _logger.LogError(
                "Port pool exhausted! All {Total} ports are allocated",
                TOTAL_PORTS);

            return null; // Pool exhausted
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task ReleasePortAsync(int port, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            if (port < POOL_START || port > POOL_END)
            {
                _logger.LogWarning(
                    "Attempted to release port {Port} outside pool range [{Start}-{End}]",
                    port, POOL_START, POOL_END);
                return;
            }

            // Remove from cache
            if (_allocatedPortsCache != null)
            {
                _allocatedPortsCache.Remove(port);
            }

            _logger.LogDebug("Released port {Port} back to pool", port);
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task<int> GetAvailablePortCountAsync(CancellationToken ct = default)
    {
        var allocated = await GetOrRefreshAllocatedPortsAsync(ct);
        return TOTAL_PORTS - allocated.Count;
    }

    public async Task<bool> IsPortAvailableAsync(int port, CancellationToken ct = default)
    {
        if (port < POOL_START || port > POOL_END)
            return false;

        var allocated = await GetOrRefreshAllocatedPortsAsync(ct);
        return !allocated.Contains(port);
    }

    public async Task<(int total, int used, double utilization)> GetUtilizationAsync(
        CancellationToken ct = default)
    {
        return await _repository.GetUtilizationAsync();
    }

    /// <summary>
    /// Get allocated ports with caching to reduce database queries
    /// </summary>
    private async Task<HashSet<int>> GetOrRefreshAllocatedPortsAsync(CancellationToken ct = default)
    {
        var now = DateTime.UtcNow;

        // Return cached if still valid
        if (_allocatedPortsCache != null &&
            now - _cacheLastRefreshed < CacheDuration)
        {
            return _allocatedPortsCache;
        }

        // Refresh from database
        _allocatedPortsCache = await _repository.GetAllocatedPortsAsync();
        _cacheLastRefreshed = now;

        _logger.LogTrace(
            "Port cache refreshed: {Count} ports allocated",
            _allocatedPortsCache.Count);

        return _allocatedPortsCache;
    }

    /// <summary>
    /// Force refresh of port cache (used after external changes)
    /// </summary>
    public void InvalidateCache()
    {
        _allocatedPortsCache = null;
        _cacheLastRefreshed = DateTime.MinValue;
    }
}
