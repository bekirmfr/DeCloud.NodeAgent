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
    public HardwareInventory? Inventory { get; private set; } = null;

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
    }
}