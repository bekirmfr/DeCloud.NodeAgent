namespace DeCloud.NodeAgent.Core.Models;


/// <summary>
/// Heartbeat payload sent to orchestrator
/// </summary>
public class Heartbeat
{
    public ResourceSnapshot Resources { get; set; } = new();
    public List<string>? ActiveVmIds { get; set; }
    public List<VmSummary>? VmSummaries { get; set; }
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

/// <summary>
/// Current resource snapshot for heartbeat.
/// 
/// CPU FIELDS:
/// - PhysicalCpuCores: Actual CPU cores (used for capacity calculations)
/// - LogicalCpuCores: Threads/hyperthreads (informational)
/// - CpuUsagePercent: Current utilization across all cores
/// </summary>
public class ResourceSnapshot
{
    // CPU - Physical cores (TRUE capacity)
    public int PhysicalCpuCores { get; set; }

    // CPU - Logical cores/threads (informational)
    public int LogicalCpuCores { get; set; }

    // CPU - Current usage percentage (0-100)
    public double CpuUsagePercent { get; set; }

    // CPU - Load average (1 minute)
    public double LoadAverage { get; set; }

    // Memory
    public long TotalMemoryBytes { get; set; }
    public long UsedMemoryBytes { get; set; }
    public long AvailableMemoryBytes => TotalMemoryBytes - UsedMemoryBytes;

    // Storage
    public long TotalStorageBytes { get; set; }
    public long UsedStorageBytes { get; set; }
    public long AvailableStorageBytes => TotalStorageBytes - UsedStorageBytes;

    // Network
    public double NetworkInMbps { get; set; }
    public double NetworkOutMbps { get; set; }
}

/// <summary>
/// Summary of a running VM for heartbeat reporting
/// </summary>
public class VmSummary
{
    public string VmId { get; set; } = string.Empty;
    public string State { get; set; } = string.Empty;

    // Allocated resources (what the VM was provisioned with)
    public int CpuCores { get; set; }
    public long MemoryMb { get; set; }
    public long DiskGb { get; set; }

    // Network info
    public string? IpAddress { get; set; }
    public int? VncPort { get; set; }
    public string? MacAddress { get; set; }

    // Security (wallet-encrypted password)
    public string? EncryptedPassword { get; set; }
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
