namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Periodic heartbeat sent to orchestrator
/// </summary>
public class Heartbeat
{
    public string NodeId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public NodeStatus Status { get; set; }
    public ResourceSnapshot Resources { get; set; } = new();
    public List<VmSummary> ActiveVms { get; set; } = new();
    public CgnatNodeInfo? CgnatInfo { get; set; }
}

public class HeartbeatResponseData
{
    public bool acknowledged { get; set; }
    public List<object>? pendingCommands { get; set; }
    public CgnatInfoDto? cgnatInfo { get; set; }
}

public class CgnatInfoDto
{
    public string? assignedRelayNodeId { get; set; }
    public string tunnelIp { get; set; } = string.Empty;
    public string? wireGuardConfig { get; set; }
    public string publicEndpoint { get; set; } = string.Empty;
}

public enum NodeStatus
{
    Initializing,
    Online,
    Maintenance,   // Not accepting new VMs but running existing
    Draining,      // Migrating VMs away, preparing for shutdown
    Offline,
    Degraded
}

public class NodeHealth
{
    public bool IsHealthy { get; set; }
    public List<HealthCheck> Checks { get; set; } = new();
}

public class HealthCheck
{
    public string Name { get; set; } = string.Empty;
    public bool Passed { get; set; }
    public string? Message { get; set; }
    public DateTime CheckedAt { get; set; } = DateTime.UtcNow;
}

public class ResourceSnapshot
{
    // CPU
    public int TotalPhysicalCores { get; set; }
    public int TotalVirtualCpuCores { get; set; }
    public int UsedVirtualCpuCores { get; set; }
    public int AvailableVirtualCpuCores => TotalVirtualCpuCores - UsedVirtualCpuCores;
    public double VirtualCpuUsagePercent { get; set; }

    // Compute Points
    public int TotalComputePoints { get; set; }
    public int UsedComputePoints { get; set; }
    public int AvailableComputePoints => TotalComputePoints - UsedComputePoints;
    public double ComputePointUsagePercent => TotalComputePoints == 0 ? 0 : (double)UsedComputePoints / TotalComputePoints * 100.0;

    // Memory
    public long TotalMemoryBytes { get; set; }
    public long UsedMemoryBytes { get; set; }
    public long AvailableMemoryBytes => Math.Max(0, TotalMemoryBytes - UsedMemoryBytes);

    // Storage
    public long TotalStorageBytes { get; set; }
    public long UsedStorageBytes { get; set; }
    public long AvailableStorageBytes => Math.Max(0, TotalStorageBytes - UsedStorageBytes);

    // GPU (if any)
    public int TotalGpus { get; set; }
    public int UsedGpus { get; set; }
    public int AvailableGpus => TotalGpus - UsedGpus;
}

public class VmSummary
{
    public string VmId { get; set; } = string.Empty;
    public string? Name { get; set; }
    public VmState State { get; set; }
    public string? OwnerId { get; set; } = string.Empty;
    public int VirtualCpuCores { get; set; }
    public int QualityTier { get; set; }
    public int ComputePointCost { get; set; }
    public long MemoryBytes { get; set; }
    public long? DiskBytes { get; set; }
    public double VirtualCpuUsagePercent { get; set; }
    public bool IsIpAssigned { get; set; }
    public string? IpAddress { get; set; }
    public int? VncPort { get; set; }
    public string? MacAddress { get; set; }
    public DateTime StartedAt { get; set; }
}

/// <summary>
/// Node registration with the orchestrator
/// </summary>
public class NodeRegistration
{
    /// <summary>
    /// Deterministic node ID (calculated from MachineId + WalletAddress)
    /// </summary>
    public required string NodeId { get; set; }

    /// <summary>
    /// Machine fingerprint for validation
    /// </summary>
    public required string MachineId { get; set; }
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Wallet address for ownership/billing
    /// </summary>
    public required string WalletAddress { get; set; }
    public string PublicIp { get; set; } = string.Empty;
    public int AgentPort { get; set; }

    // Full resource inventory
    public required HardwareInventory HardwareInventory { get; set; } = new();
    public required string AgentVersion { get; set; } = string.Empty;
    public required List<string> SupportedImages { get; set; } = new();

    
    // Staking info
    public string StakingTxHash { get; set; } = string.Empty;

    public string Region { get; set; } = "default";
    public string Zone { get; set; } = "default";

    public DateTime RegisteredAt { get; set; } = DateTime.UtcNow;
}

public class NodePricing
{
    public decimal PricePerVCpuHour { get; set; }
    public decimal PricePerGbRamHour { get; set; }
    public decimal PricePerGbStorageMonth { get; set; }
    public decimal PricePerGpuHour { get; set; }
    public string Currency { get; set; } = "USDC";  // Stablecoin default
}
