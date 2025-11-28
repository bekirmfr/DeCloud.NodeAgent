namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Periodic heartbeat sent to orchestrator
/// </summary>
public class Heartbeat
{
    public string NodeId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public NodeStatus Status { get; set; }
    public NodeHealth Health { get; set; } = new();
    
    // Resource snapshot
    public ResourceSnapshot Resources { get; set; } = new();
    
    // Active VMs summary
    public List<VmSummary> ActiveVms { get; set; } = new();
    
    // Signature for verification (signed with node's private key)
    public string Signature { get; set; } = string.Empty;
}

public enum NodeStatus
{
    Initializing,
    Online,
    Maintenance,   // Not accepting new VMs but running existing
    Draining,      // Migrating VMs away, preparing for shutdown
    Offline
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
    public string TenantId { get; set; } = string.Empty;
    public string LeaseId { get; set; } = string.Empty;
    public VmState State { get; set; }
    public int VCpus { get; set; }
    public long MemoryBytes { get; set; }
    public double CpuUsagePercent { get; set; }
    public string? IpAddress { get; set; }
    public DateTime StartedAt { get; set; }
}

/// <summary>
/// Node registration with the orchestrator
/// </summary>
public class NodeRegistration
{
    public string NodeId { get; set; } = string.Empty;
    public string PublicKey { get; set; } = string.Empty;  // For signature verification
    public string WireGuardPublicKey { get; set; } = string.Empty;
    public string Endpoint { get; set; } = string.Empty;   // IP:Port for orchestrator to reach agent
    
    // Full resource inventory
    public NodeResources Resources { get; set; } = new();
    
    // Pricing (optional, for marketplace)
    public NodePricing? Pricing { get; set; }
    
    // Staking info
    public string WalletAddress { get; set; } = string.Empty;
    public string StakingTxHash { get; set; } = string.Empty;
    
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
