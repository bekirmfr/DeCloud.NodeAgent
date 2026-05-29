using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.Shared.Contracts;
using DeCloud.Shared.Dtos;
using DeCloud.Shared.Enums;
using DeCloud.Shared.Models;

namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Response from POST /api/nodes/me/evaluate.
/// Carries performance evaluation, scheduling config, obligations,
/// identity states, system templates — everything that was previously
/// in the registration response.
/// </summary>
public record EvaluateNodeResponse(
    NodePerformanceEvaluation PerformanceEvaluation,
    AgentSchedulingConfig SchedulingConfig,
    List<string>? DhtBootstrapPeers,
    Dictionary<string, ObligationStatePayload>? ObligationStates,
    Dictionary<string, SystemVmTemplatePayload>? SystemTemplates = null,
    List<ObligationDescriptorDto>? Obligations = null
);

/// <summary>
/// Periodic heartbeat sent to orchestrator.
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

    /// <summary>
    /// Versions of obligation identity state currently stored in the node
    /// agent's SQLite database, keyed by canonical role name.
    /// Allows the orchestrator to detect stale states without waiting for
    /// the next registration.
    /// </summary>
    public Dictionary<string, int>? ObligationStateVersions { get; set; }

    /// <summary>
    /// Per-role coarse reality classification ("None" | "Healthy" | "Unhealthy"),
    /// keyed by canonical role name. Populated in HeartbeatService from
    /// IRealityProjection over the obligation list (P4).
    /// </summary>
    public Dictionary<string, string>? ObligationHealth { get; set; }

    /// <summary>
    /// Revisions of system templates currently stored in the node agent's
    /// SQLite <c>system_template</c> table, keyed by canonical role name.
    /// Allows the orchestrator to detect and push template updates without
    /// waiting for the next registration.
    /// Absent or zero values mean the node has no template for that role.
    /// </summary>
    public Dictionary<string, int>? SystemTemplateVersions { get; set; }

    /// <summary>
    /// SHA-256 hash of authoritative settings (wallet, country, region, zone).
    /// Computed by SettingsHash.Compute() from DeCloud.Shared.
    /// The orchestrator compares this against its stored registration state
    /// to detect local edits that weren't committed via re-register.
    /// </summary>
    public string? SettingsHash { get; set; }
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
    public int AllocatedComputePoints { get; set; }
    public int UsedComputePoints { get; set; }
    public int AvailableComputePoints => AllocatedComputePoints - UsedComputePoints;
    public double ComputePointUsagePercent => TotalComputePoints == 0 ? 0 : (double)UsedComputePoints / TotalComputePoints * 100.0;

    // Memory
    /// <summary>Total physical RAM on the host.</summary>
    public long TotalMemoryBytes { get; set; }
    /// <summary>Operator-allocated memory ceiling (for scheduling alignment).</summary>
    public long AllocatedMemoryBytes { get; set; }
    public long UsedMemoryBytes { get; set; }
    public long AvailableMemoryBytes => Math.Max(0, AllocatedMemoryBytes - UsedMemoryBytes);

    // Storage
    public long TotalStorageBytes { get; set; }
    public long AllocatedStorageBytes { get; set; }
    public long UsedStorageBytes { get; set; }
    public long AvailableStorageBytes => Math.Max(0, AllocatedStorageBytes - UsedStorageBytes);

    // GPU (if any)
    public int TotalGpus { get; set; }
    /// <summary>
    /// Operator-configured GPU ceiling. Null = all detected GPUs offered.
    /// 0 = GPUs present but operator has opted out of offering them.
    /// </summary>
    public int AllocatedGpus { get; set; }
    /// <summary>Physical GPUs currently held by active Passthrough VMs.</summary>
    public int UsedGpus { get; set; }
    public int AvailableGpus => Math.Max(0, AllocatedGpus - UsedGpus);

    /// <summary>Sum of MemoryBytes for all detected GPUs.</summary>
    public long TotalGpuVramBytes { get; set; }
    /// <summary>
    /// Committed VRAM across all active GPU VMs.
    /// Passthrough: full GPU MemoryBytes per assigned GPU.
    /// Proxied: sum of GpuVramBytes quotas per active VM (Phase 2 — 0 until VmSpec.GpuVramBytes exists).
    /// </summary>
    public long AllocatedGpuVramBytes { get; set; }
    /// <summary>Live aggregate VRAM usage reported by nvidia-smi (sum of MemoryUsedBytes).</summary>
    public long UsedGpuVramBytes { get; set; }
    public long AvailableGpuVramBytes => Math.Max(0, AllocatedGpuVramBytes - UsedGpuVramBytes);
    /// <summary>True if any GPU supports Proxied (shared) access via the proxy daemon.</summary>
    public bool SupportsGpuProxy { get; set; }
    /// <summary>True if any GPU supports VFIO Passthrough (IOMMU + vfio-pci).</summary>
    public bool SupportsGpuPassthrough { get; set; }
}