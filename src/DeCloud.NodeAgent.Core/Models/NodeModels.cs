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
    public List<VmSummary> ActiveVmDetails { get; set; } = new();
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
    public int TotalVCpus { get; set; }
    public int UsedVCpus { get; set; }
    public int AvailableVCpus => TotalVCpus - UsedVCpus;
    public double CpuUsagePercent { get; set; }
    
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
    public string TenantId { get; set; } = string.Empty;
    public string LeaseId { get; set; } = string.Empty;
    public VmState State { get; set; }
    public int VCpus { get; set; }
    public long MemoryBytes { get; set; }
    public long? DiskBytes { get; set; }
    public double CpuUsagePercent { get; set; }
    public string? IpAddress { get; set; }
    public string? VncPort { get; set; }
    public string? MacAddress { get; set; }
    public string? EncryptedPassword { get; set; }
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

    /// <summary>
    /// Wallet address for ownership/billing
    /// </summary>
    public required string WalletAddress { get; set; }
    public string Name { get; set; } = string.Empty;
    public string PublicIp { get; set; } = string.Empty;
    public int AgentPort { get; set; }

    // Full resource inventory
    public NodeResources Resources { get; set; } = new();
    public string AgentVersion { get; set; } = string.Empty;
    public List<string> SupportedImages { get; set; } = new();

    public bool SupportsGpu { get; set; }
    public GpuInfo? GpuInfo { get; set; }

    // Pricing (optional, for marketplace)
    public NodePricing? Pricing { get; set; }
    
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
