using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Orchestrator.Models;

public interface INodeMetadataService
{
    string OrchestratorUrl { get; }
    string NodeId { get; }
    string MachineId { get; }
    string Name { get; }
    string? PublicIp { get; }
    string WalletAddress { get; }
    string Region { get; }
    string Zone { get; }
    HardwareInventory? Inventory { get; }
    /// <summary>
    /// Current scheduling configuration received from Orchestrator
    /// Used for calculating CPU quotas and resource allocations
    /// Updated during registration and heartbeat
    /// </summary>
    /// 
    NodePerformanceEvaluation PerformanceEvaluation { get; }
    public SchedulingConfig SchedulingConfig { get; }


    Task InitializeAsync(CancellationToken ct = default);
    void UpdatePublicIp(string publicIp);
    void UpdateInventory(HardwareInventory inventory);
    int GetSchedulingConfigVersion();
    void UpdateSchedulingConfig(SchedulingConfig newConfig);
    void UpdatePerformanceEvaluation(NodePerformanceEvaluation newEvaluation);
}

public class NodeMetadataService : INodeMetadataService
{
    private readonly IConfiguration _configuration;
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly ILogger<NodeMetadataService> _logger;

    public string OrchestratorUrl { get; private set; } = string.Empty;
    public string NodeId { get; private set; } = string.Empty;
    public string MachineId { get; private set; } = string.Empty;
    public string Name { get; private set; } = string.Empty;
    public string? PublicIp { get; private set; }
    public string WalletAddress { get; private set; } = string.Empty;
    public string Region { get; private set; } = string.Empty;
    public string Zone { get; private set; } = string.Empty;
    public HardwareInventory? Inventory { get; private set; } = null;
    public NodePerformanceEvaluation? PerformanceEvaluation { get; private set; } = null;
    /// <summary>
    /// Current scheduling configuration received from Orchestrator
    /// Used for calculating CPU quotas and resource allocations
    /// Updated during registration and heartbeat
    /// </summary>
    public SchedulingConfig? SchedulingConfig { get; private set; } = null;

    // Lock for thread-safe config updates
    private readonly SemaphoreSlim _configLock = new(1, 1);

    public NodeMetadataService(IConfiguration configuration, ILogger<NodeMetadataService> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        // Get machine ID
        MachineId = NodeIdGenerator.GetMachineId();

        // Get wallet from config
        WalletAddress = _configuration["OrchestratorClient:WalletAddress"] ?? "";

        OrchestratorUrl = _configuration["OrchestratorClient:BaseUrl"] ?? "";

        // Generate deterministic node ID
        NodeId = NodeIdGenerator.GenerateNodeId(MachineId, WalletAddress);

        // Get name, region, zone from config
        Name = _configuration["Node:Name"] ?? Environment.MachineName;
        Region = _configuration["Node:Region"] ?? "default";
        Zone = _configuration["Node:Zone"] ?? "default";

        // Discover public IP
        PublicIp = await DiscoverPublicIpAsync(ct);

        SchedulingConfig = new SchedulingConfig
        {
            Version = 0,  // v0 = not yet synced with Orchestrator
            BaselineBenchmark = 1000,
            BaselineOvercommitRatio = 4.0,
            MaxPerformanceMultiplier = 20.0,
            UpdatedAt = DateTime.UtcNow
        };

        _ = Task.Run(async () => {
            var inv = await _resourceDiscovery.GetInventoryCachedAsync(CancellationToken.None);
            if (inv != null) 
                UpdateInventory(inv);
        }, ct);

        _logger.LogInformation(
            "✓ Node metadata initialized: ID={NodeId}, Name={Name}, IP={PublicIp}",
            NodeId, Name, PublicIp);
    }

    public void UpdatePublicIp(string publicIp)
    {
        if (PublicIp != publicIp)
        {
            _logger.LogInformation("Public IP changed: {Old} → {New}", PublicIp, publicIp);
            PublicIp = publicIp;
        }
    }

    private async Task<string?> DiscoverPublicIpAsync(CancellationToken ct)
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var ip = await client.GetStringAsync("https://api.ipify.org", ct);
            return ip.Trim();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to discover public IP");
            return null;
        }
    }

    public void UpdateInventory(HardwareInventory inventory)
    {
        Inventory = inventory;
    }

    /// <summary>
    /// Update scheduling configuration (thread-safe)
    /// Called during registration and heartbeat when Orchestrator sends new config
    /// </summary>
    public void UpdateSchedulingConfig(SchedulingConfig newConfig)
    {
        if (newConfig == null)
        {
            _logger?.LogWarning("Attempted to update config with null, ignoring");
            return;
        }

        // Validate config
        if (newConfig.BaselineBenchmark <= 0)
        {
            _logger?.LogError(
                "Invalid config: BaselineBenchmark={Baseline} must be positive",
                newConfig.BaselineBenchmark);
            return;
        }

        if (newConfig.BaselineOvercommitRatio <= 0)
        {
            _logger?.LogError(
                "Invalid config: BaselineOvercommitRatio={Overcommit} must be positive",
                newConfig.BaselineOvercommitRatio);
            return;
        }

        // Only update if version is newer
        if (newConfig.Version > SchedulingConfig.Version)
        {
            var oldVersion = SchedulingConfig.Version;
            var oldBaseline = SchedulingConfig.BaselineBenchmark;
            var oldOvercommit = SchedulingConfig.BaselineOvercommitRatio;

            // Atomic replacement
            SchedulingConfig = newConfig;

            _logger?.LogWarning(
                "🔄 Scheduling config updated: v{OldVersion} → v{NewVersion}. " +
                "Baseline: {OldBaseline} → {NewBaseline}, " +
                "Overcommit: {OldOvercommit:F1} → {NewOvercommit:F1}",
                oldVersion, newConfig.Version,
                oldBaseline, newConfig.BaselineBenchmark,
                oldOvercommit, newConfig.BaselineOvercommitRatio);
        }
        else if (newConfig.Version < SchedulingConfig.Version)
        {
            _logger?.LogWarning(
                "Received older config v{NewVersion} (current: v{CurrentVersion}), ignoring",
                newConfig.Version, SchedulingConfig.Version);
        }
    }

    /// <summary>
    /// Get current scheduling config version for heartbeat requests
    /// </summary>
    public int GetSchedulingConfigVersion() => SchedulingConfig?.Version ?? 0;

    /// <summary>
    /// Get formatted config summary for logging
    /// </summary>
    public string GetConfigSummary()
    {
        var config = SchedulingConfig;
        return $"v{config.Version}: Baseline={config.BaselineBenchmark}, " +
               $"Overcommit={config.BaselineOvercommitRatio:F1}";
    }

    public void UpdatePerformanceEvaluation(NodePerformanceEvaluation newEvaluation)
    {
        PerformanceEvaluation = newEvaluation;
    }
}