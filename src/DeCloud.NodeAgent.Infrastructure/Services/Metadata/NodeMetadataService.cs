using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

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
    /// <summary>
    /// ISO 3166-1 alpha-2 country code from <c>Node:Country</c> config.
    /// <c>"ZZ"</c> when not configured (unknown / not yet declared).
    /// </summary>
    string Country { get; }

    NodePricing? Pricing { get; }
    HardwareInventory? Inventory { get; }

    /// <summary>
    /// Operator-allocated memory in bytes, resolved from settings.
    /// Null until UpdateInventory is called (percent mode needs TotalBytes).
    /// Null also means "use platform default (90%)".
    /// </summary>
    long? AllocatedMemoryBytes { get; }
    int? AllocatedComputePoints { get; }
    long? AllocatedStorageBytes { get; }
    int? AllocatedGpuCount { get; }

    Task InitializeAsync(CancellationToken ct = default);
    void UpdatePublicIp(string publicIp);
    void UpdateInventory(HardwareInventory inventory);
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
    public string Country { get; private set; } = "ZZ";
    public NodePricing? Pricing { get; private set; }
    public HardwareInventory? Inventory { get; private set; } = null;
    private string? _memoryAllocMode;
    private int? _memoryAllocValue;
    public long? AllocatedMemoryBytes { get; private set; }

    private string? _cpuAllocMode;
    private int? _cpuAllocValue;
    public int? AllocatedComputePoints { get; private set; }

    private string? _storageAllocMode;
    private int? _storageAllocValue;
    public long? AllocatedStorageBytes { get; private set; }

    public int? AllocatedGpuCount { get; private set; }

    public NodeMetadataService(IConfiguration configuration, ILogger<NodeMetadataService> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        // Get machine ID
        MachineId = NodeIdGenerator.GetMachineId();

        // Operator identity — settings.json (flat keys) override appsettings (nested keys).
        // settings.json: { "wallet": "0x..." }  → key "wallet"
        // appsettings:   { "OrchestratorClient": { "WalletAddress": "0x..." } } → key "OrchestratorClient:WalletAddress"
        WalletAddress = _configuration["wallet"]
                     ?? _configuration["OrchestratorClient:WalletAddress"]
                     ?? "";

        OrchestratorUrl = _configuration["orchestrator_url"]
                       ?? _configuration["OrchestratorClient:BaseUrl"]
                       ?? "";

        // Generate deterministic node ID
        NodeId = NodeIdGenerator.GenerateNodeId(MachineId, WalletAddress);

        // Operator locality — settings.json flat keys override Node:* from appsettings
        Name = _configuration["name"]
            ?? _configuration["Node:Name"]
            ?? Environment.MachineName;
        Region = _configuration["region"]
              ?? _configuration["Node:Region"]
              ?? "default";
        Zone = _configuration["zone"]
            ?? _configuration["Node:Zone"]
            ?? "default";
        Country = (_configuration["country"]
                ?? _configuration["Node:Country"]
                ?? "ZZ").ToUpperInvariant();

        _logger.LogInformation(
            "Node locality: Country={Country}, Region={Region}, Zone={Zone}",
            Country, Region, Zone);

        // ── Memory allocation ────────────────────────────────────────────
        _memoryAllocMode = _configuration["resources:memory:mode"]
                        ?? _configuration["Node:Resources:Memory:Mode"];
        var memValStr = _configuration["resources:memory:value"]
                     ?? _configuration["Node:Resources:Memory:Value"];
        if (int.TryParse(memValStr, out var memVal))
            _memoryAllocValue = memVal;

        if (string.Equals(_memoryAllocMode, "mb", StringComparison.OrdinalIgnoreCase)
            && _memoryAllocValue.HasValue)
        {
            AllocatedMemoryBytes = (long)_memoryAllocValue.Value * 1024 * 1024;
            _logger.LogInformation("Resource allocation (memory): {Mb} MB (absolute)",
                _memoryAllocValue.Value);
        }
        else if (string.Equals(_memoryAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _memoryAllocValue.HasValue)
        {
            _logger.LogInformation("Resource allocation (memory): {Pct}% (deferred until hardware discovery)",
                _memoryAllocValue.Value);
        }
        else
        {
            _logger.LogInformation("Resource allocation (memory): not configured, platform default (90%) will apply");
        }

        // ── CPU allocation ───────────────────────────────────────────────
        _cpuAllocMode = _configuration["resources:cpu:mode"];
        var cpuValStr = _configuration["resources:cpu:value"];
        if (int.TryParse(cpuValStr, out var cpuVal))
            _cpuAllocValue = cpuVal;

        if (string.Equals(_cpuAllocMode, "points", StringComparison.OrdinalIgnoreCase)
            && _cpuAllocValue.HasValue)
        {
            AllocatedComputePoints = _cpuAllocValue.Value;
            _logger.LogInformation("Resource allocation (CPU): {Pts} compute points (absolute)",
                _cpuAllocValue.Value);
        }
        else if (string.Equals(_cpuAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _cpuAllocValue.HasValue)
        {
            _logger.LogInformation("Resource allocation (CPU): {Pct}% (deferred until hardware discovery)",
                _cpuAllocValue.Value);
        }
        else
        {
            _logger.LogInformation("Resource allocation (CPU): not configured, platform default (90%) will apply");
        }

        // ── Storage allocation ───────────────────────────────────────────
        _storageAllocMode = _configuration["resources:storage:mode"];
        var storValStr = _configuration["resources:storage:value"];
        if (int.TryParse(storValStr, out var storVal))
            _storageAllocValue = storVal;

        if (string.Equals(_storageAllocMode, "mb", StringComparison.OrdinalIgnoreCase)
            && _storageAllocValue.HasValue)
        {
            AllocatedStorageBytes = (long)_storageAllocValue.Value * 1024 * 1024;
            _logger.LogInformation("Resource allocation (storage): {Mb} MB (absolute)",
                _storageAllocValue.Value);
        }
        else if (string.Equals(_storageAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _storageAllocValue.HasValue)
        {
            _logger.LogInformation("Resource allocation (storage): {Pct}% (deferred until hardware discovery)",
                _storageAllocValue.Value);
        }
        else
        {
            _logger.LogInformation("Resource allocation (storage): not configured, platform default (90%) will apply");
        }

        // ── GPU allocation ───────────────────────────────────────────────
        var gpuCountStr = _configuration["resources:gpu:count"];
        if (int.TryParse(gpuCountStr, out var gpuCount))
        {
            AllocatedGpuCount = gpuCount;
            _logger.LogInformation("Resource allocation (GPU): {Count} GPUs", gpuCount);
        }
        else
        {
            _logger.LogInformation("Resource allocation (GPU): not configured, all detected GPUs offered");
        }

        // Load operator pricing from config (optional)
        var pricingSection = _configuration.GetSection("Node:Pricing");
        if (pricingSection.Exists())
        {
            Pricing = new NodePricing();
            pricingSection.Bind(Pricing);
        }

        // Discover public IP
        PublicIp = await DiscoverPublicIpAsync(ct);

        // Background task to update inventory
        _ = Task.Run(async () => {
            var inv = await _resourceDiscovery.GetInventoryCachedAsync(CancellationToken.None);
            if (inv != null) UpdateInventory(inv);
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
        ResolveAllocatedMemory(inventory);
        ResolveAllocatedCpu(inventory);
        ResolveAllocatedStorage(inventory);
    }

    /// <summary>
    /// Resolve percent-based memory allocation once hardware info is available.
    /// Called from UpdateInventory after discovery completes.
    /// </summary>
    private void ResolveAllocatedMemory(HardwareInventory inventory)
    {
        if (AllocatedMemoryBytes.HasValue)
            return; // Already resolved (absolute mode set in InitializeAsync)

        if (string.Equals(_memoryAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _memoryAllocValue.HasValue)
        {
            var pct = Math.Clamp(_memoryAllocValue.Value, 1, 95) / 100.0;
            AllocatedMemoryBytes = (long)(inventory.Memory.TotalBytes * pct);
            _logger.LogInformation(
                "Resource allocation (memory): resolved {Pct}% of {TotalMb} MB = {AllocMb} MB",
                _memoryAllocValue.Value,
                inventory.Memory.TotalBytes / (1024 * 1024),
                AllocatedMemoryBytes.Value / (1024 * 1024));
        }
        // else: null → orchestrator applies 90% default
    }

    private void ResolveAllocatedCpu(HardwareInventory inventory)
    {
        if (AllocatedComputePoints.HasValue)
            return; // Already resolved (absolute points set in InitializeAsync)

        if (string.Equals(_cpuAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _cpuAllocValue.HasValue)
        {
            // Percent of logical cores — orchestrator derives points from cores
            var pct = Math.Clamp(_cpuAllocValue.Value, 1, 95) / 100.0;
            var allocatedCores = (int)Math.Floor(inventory.Cpu.PhysicalCores * pct);
            // Store as a negative sentinel meaning "percent-of-cores" — the
            // orchestrator can't use raw core count directly, so we resolve to
            // actual compute points using the node's benchmark on the orchestrator
            // side. For now, send null and let orchestrator apply 90% default.
            // TODO: resolve to compute points once benchmark score is available locally.
            _ = allocatedCores; // suppress unused warning
            _logger.LogInformation(
                "Resource allocation (CPU): {Pct}% → orchestrator will apply proportionally",
                _cpuAllocValue.Value);
        }
        // else: null → orchestrator applies 90% default
    }

    private void ResolveAllocatedStorage(HardwareInventory inventory)
    {
        if (AllocatedStorageBytes.HasValue)
            return; // Already resolved (absolute mode set in InitializeAsync)

        if (string.Equals(_storageAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _storageAllocValue.HasValue)
        {
            var pct = Math.Clamp(_storageAllocValue.Value, 1, 95) / 100.0;
            var totalStorage = inventory.Storage.Sum(s => s.TotalBytes);
            AllocatedStorageBytes = (long)(totalStorage * pct);
            _logger.LogInformation(
                "Resource allocation (storage): resolved {Pct}% of {TotalGb} GB = {AllocGb} GB",
                _storageAllocValue.Value,
                totalStorage / (1024 * 1024 * 1024),
                AllocatedStorageBytes.Value / (1024 * 1024 * 1024));
        }
        // else: null → orchestrator applies 90% default
    }
}