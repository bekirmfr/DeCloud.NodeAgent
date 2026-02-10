using DeCloud.NodeAgent.Core.Interfaces.State;
using Orchestrator.Models;

namespace DeCloud.NodeAgent.Core.Models;

public record NodeRegistrationResponse(
    string NodeId,
    NodePerformanceEvaluation PerformanceEvaluation,
    string ApiKey,
    SchedulingConfig SchedulingConfig,
    /// <summary>
    /// Orchestrator's WireGuard public key for relay configuration
    /// Null if WireGuard is not enabled on orchestrator
    /// </summary>
    string OrchestratorWireGuardPublicKey,
    TimeSpan HeartbeatInterval
);

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
    public int SchedulingConfigVersion { get; set; } = 0;
    public CgnatNodeInfo? CgnatInfo { get; set; }
}

public class HeartbeatDto
{
    public Heartbeat? Heartbeat { get; set; }
    public HttpResponseMessage? Response { get; set; }
}

public class HeartbeatResponseData
{
    public bool Acknowledged { get; set; }
    public List<object>? PendingCommands { get; set; }
    public CgnatInfoDto? CgnatInfo { get; set; }
    public DateTime? ServerTime { get; set; }
}

public class CgnatInfoDto
{
    public string? assignedRelayNodeId { get; set; }
    public string tunnelIp { get; set; } = string.Empty;
    public string? wireGuardConfig { get; set; }
    public string publicEndpoint { get; set; } = string.Empty;
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
    public List<ServiceSummary>? Services { get; set; }
}

/// <summary>
/// Lightweight service status for heartbeat reporting.
/// </summary>
public class ServiceSummary
{
    public string Name { get; set; } = string.Empty;
    public int? Port { get; set; }
    public string? Protocol { get; set; }
    public string Status { get; set; } = "Pending";
    public DateTime? ReadyAt { get; set; }
}

/// <summary>
/// Node registration with the orchestrator
/// </summary>
public class NodeRegistration
{
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

    /// <summary>
    /// Wallet signature proving ownership (from WalletConnect CLI)
    /// Optional for backward compatibility - will be required in production
    /// </summary>
    public string? Signature { get; set; }

    /// <summary>
    /// Message that was signed (includes node ID, wallet, timestamp)
    /// Optional for backward compatibility - will be required in production
    /// </summary>
    public string? Message { get; set; }

    public DateTime RegisteredAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Node operator pricing (optional). If null, platform defaults are used.
    /// </summary>
    public NodePricing? Pricing { get; set; }
}

public class NodePricing
{
    public decimal CpuPerHour { get; set; }
    public decimal MemoryPerGbPerHour { get; set; }
    public decimal StoragePerGbPerHour { get; set; }
    public decimal GpuPerHour { get; set; }
    public string Currency { get; set; } = "USDC";
}
