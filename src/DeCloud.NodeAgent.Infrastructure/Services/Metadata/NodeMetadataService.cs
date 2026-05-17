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
    public long? AllocatedMemoryBytes { get; private set; }

    // Raw config values — resolved against hardware in UpdateInventory
    private string? _memoryAllocMode;
    private int? _memoryAllocValue;

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
        Country = (_configuration["Node:Country"] ?? "ZZ").ToUpperInvariant();
        _logger.LogInformation(
            "Node locality: Country={Country}, Region={Region}, Zone={Zone}",
            Country, Region, Zone);

        // Resource allocation settings (from decloud configure)
        _memoryAllocMode = _configuration["Node:Resources:Memory:Mode"];
        if (int.TryParse(_configuration["Node:Resources:Memory:Value"], out var memVal))
            _memoryAllocValue = memVal;

        // If mode is "mb", resolve immediately (doesn't need hardware discovery)
        if (string.Equals(_memoryAllocMode, "mb", StringComparison.OrdinalIgnoreCase)
            && _memoryAllocValue.HasValue)
        {
            AllocatedMemoryBytes = (long)_memoryAllocValue.Value * 1024 * 1024;
            _logger.LogInformation(
                "Resource allocation (memory): {Mb} MB (absolute, from settings)",
                _memoryAllocValue.Value);
        }
        else if (string.Equals(_memoryAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _memoryAllocValue.HasValue)
        {
            // Percent mode — deferred to UpdateInventory when TotalBytes is known
            _logger.LogInformation(
                "Resource allocation (memory): {Pct}% (deferred until hardware discovery)",
                _memoryAllocValue.Value);
        }
        else
        {
            _logger.LogInformation(
                "Resource allocation (memory): not configured, platform default (90%) will apply");
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
    }

    /// <summary>
    /// Resolve percent-based memory allocation now that hardware info is available.
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
        // else: no config → AllocatedMemoryBytes stays null → orchestrator applies 90% default
    }
}