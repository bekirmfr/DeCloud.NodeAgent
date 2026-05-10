using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.Shared.Models;
using Orchestrator.Models;

namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Wire DTO for a single obligation received in the registration response.
/// Mirrors <c>Orchestrator.Models.ObligationDescriptorPayload</c>.
/// </summary>
public class NodeObligationDescriptorDto
{
    public string Role { get; init; } = string.Empty;
    public List<string> Deps { get; init; } = [];
}

public record NodeRegistrationResponse(
    string NodeId,
    NodePerformanceEvaluation PerformanceEvaluation,
    string ApiKey,
    SchedulingConfig SchedulingConfig,
    string OrchestratorWireGuardPublicKey,
    TimeSpan HeartbeatInterval,
    List<string>? DhtBootstrapPeers,
    /// <summary>
    /// Identity state payloads keyed by canonical role name.
    /// Contains only roles where orchestrator version > node-reported version.
    /// Null or empty = all identity states current on the node.
    /// </summary>
    Dictionary<string, ObligationStatePayload>? ObligationStates,
    /// <summary>
    /// System template payloads keyed by canonical role name.
    /// Contains only roles where orchestrator revision > node-reported revision.
    /// Null or empty = all templates current on the node (or none seeded yet).
    /// </summary>
    Dictionary<string, SystemVmTemplatePayload>? SystemTemplates = null,
    List<NodeObligationDescriptorDto>? Obligations = null
);


/// <summary>
/// Mirrors Orchestrator.Models.ObligationStatePayload.
/// Defined here to avoid a project reference from Core → Orchestrator.
/// </summary>
public class ObligationStatePayload
{
    public string StateJson { get; init; } = string.Empty;
    public int Version { get; init; }
}


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
    public VmType VmType { get; set; }
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
    public string? StatusMessage { get; set; }
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

    /// <summary>
    /// SSH certificate authority public key — captured from
    /// <c>/etc/ssh/decloud_ca.pub</c> by <c>OrchestratorClient.RegisterAsync</c>
    /// at registration time. Sent to the orchestrator so tenant cloud-init
    /// templates can substitute <c>__CA_PUBLIC_KEY__</c> at render time.
    ///
    /// <para>
    /// Mirrors the orchestrator-side
    /// <c>NodeRegistrationRequest.SshCaPublicKey</c>. Same JSON shape on the
    /// wire.
    /// </para>
    ///
    /// <para>
    /// Null on the rare path where the node lacks <c>/etc/ssh/decloud_ca.pub</c>
    /// (e.g., misconfigured or freshly cloned node image). The orchestrator
    /// accepts null and stamps null into <c>Node.SshCaPublicKey</c>; tenant
    /// deploys that need the CA key fail at render time with a clear message.
    /// </para>
    /// </summary>
    public string? SshCaPublicKey { get; set; }


    // Staking info
    public string StakingTxHash { get; set; } = string.Empty;

    public string Region { get; set; } = "default";
    public string Zone { get; set; } = "default";

    /// <summary>
    /// ISO 3166-1 alpha-2 country code declared by the operator.
    /// Read from <c>Node:Country</c> in appsettings.Production.json.
    /// <c>"ZZ"</c> when not configured. Null on nodes running pre-2.3
    /// agents — orchestrator accepts null and records "ZZ".
    /// </summary>
    public string? Country { get; set; }


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

    /// <summary>
    /// Monotonic version from the orchestrator — used for conflict resolution.
    /// </summary>
    public Dictionary<string, int> ObligationStateVersions { get; set; } = new();

    /// <summary>
    /// Revisions of system templates currently stored in the node agent's
    /// SQLite database, keyed by canonical role name.
    /// Allows the orchestrator to skip sending templates already current on the node.
    /// Absent or zero-valued entries mean no template stored for that role.
    /// </summary>
    public Dictionary<string, int> SystemTemplateVersions { get; set; } = new();
}

public class NodePricing
{
    public decimal CpuPerHour { get; set; }
    public decimal MemoryPerGbPerHour { get; set; }
    public decimal StoragePerGbPerHour { get; set; }
    public decimal GpuPerHour { get; set; }
    public string Currency { get; set; } = "USDC";
}
