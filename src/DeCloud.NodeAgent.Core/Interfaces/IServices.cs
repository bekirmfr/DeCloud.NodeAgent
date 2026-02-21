using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using Orchestrator.Models;
using System.Net.NetworkInformation;

namespace DeCloud.NodeAgent.Core.Interfaces;

/// <summary>
/// Handles node registration with the orchestrator
/// </summary>
public interface INodeRegistrationService
{
    Task<RegistrationResult> RegisterAsync(CancellationToken ct = default);
}

public class RegistrationResult
{
    public bool IsSuccess { get; init; }
    public string? NodeId { get; init; }
    public string? ApiKey { get; init; }
    public SchedulingConfig? SchedulingConfig { get; init; }
    public string? Error { get; init; }

    public static RegistrationResult Success(string nodeId, string apiKey) =>
        new() { IsSuccess = true, NodeId = nodeId, ApiKey = apiKey};

    public static RegistrationResult Failure(string error) =>
        new() { IsSuccess = false, Error = error };
}

/// <summary>
/// Discovers and monitors local hardware resources
/// </summary>
public interface IResourceDiscoveryService
{
    Task<HardwareInventory?> GetInventoryCachedAsync(CancellationToken ct = default);
    Task<HardwareInventory> DiscoverAllAsync(CancellationToken ct = default);
    Task<CpuInfo> GetCpuInfoAsync(CancellationToken ct = default, bool runBenchmark = true);
    Task<MemoryInfo> GetMemoryInfoAsync(CancellationToken ct = default);
    Task<List<StorageInfo>> GetStorageInfoAsync(CancellationToken ct = default);
    Task<List<GpuInfo>> GetGpuInfoAsync(CancellationToken ct = default, bool forceRecheck = false);
    Task<NetworkInfo> GetNetworkInfoAsync(CancellationToken ct = default);
    Task<ResourceSnapshot> GetCurrentSnapshotAsync(CancellationToken ct = default);
}

/// <summary>
/// Manages VM lifecycle via libvirt/virsh
/// </summary>
public interface IVmManager
{
    Task<VmOperationResult> CreateVmAsync(VmSpec spec, string? password = null, CancellationToken ct = default);
    Task<VmOperationResult> StartVmAsync(string vmId, CancellationToken ct = default);
    Task<VmOperationResult> StopVmAsync(string vmId, bool force = false, CancellationToken ct = default);
    Task<VmOperationResult> RestartVmAsync(string vmId, bool force = false, CancellationToken ct = default);
    Task<VmOperationResult> DeleteVmAsync(string vmId, CancellationToken ct = default);
    Task<VmOperationResult> PauseVmAsync(string vmId, CancellationToken ct = default);
    Task<VmOperationResult> ResumeVmAsync(string vmId, CancellationToken ct = default);
    
    Task<VmInstance?> GetVmAsync(string vmId, CancellationToken ct = default);
    Task<List<VmInstance>> GetAllVmsAsync(CancellationToken ct = default);
    Task<VmResourceUsage> GetVmUsageAsync(string vmId, CancellationToken ct = default);
    Task ReconcileAllWithLibvirtAsync(CancellationToken ct = default);
    Task<bool> VmExistsAsync(string vmId, CancellationToken ct = default);
    Task<string?> GetVmIpAddressAsync(string vmId, CancellationToken ct = default);
    /// <summary>
    /// Apply CPU quota cap to a running VM
    /// </summary>
    Task<bool> ApplyQuotaCapAsync(
        VmInstance vm,
        int quotaMicroseconds,
        int periodMicroseconds = 100000,
        CancellationToken ct = default);
}

/// <summary>
/// Manages base VM images (download, cache, verify)
/// </summary>
public interface IImageManager
{
    Task<string> EnsureImageAvailableAsync(string imageUrl, string expectedHash, CancellationToken ct = default);
    Task<bool> VerifyImageAsync(string imagePath, string expectedHash, CancellationToken ct = default);
    Task<string> CreateOverlayDiskAsync(string baseImagePath, string vmId, long sizeBytes, CancellationToken ct = default);
    Task DeleteDiskAsync(string diskPath, CancellationToken ct = default);
    Task<List<CachedImage>> GetCachedImagesAsync(CancellationToken ct = default);
    Task PruneUnusedImagesAsync(TimeSpan maxAge, CancellationToken ct = default);
}

public class CachedImage
{
    public string Url { get; set; } = string.Empty;
    public string LocalPath { get; set; } = string.Empty;
    public string Hash { get; set; } = string.Empty;
    public long SizeBytes { get; set; }
    public DateTime DownloadedAt { get; set; }
    public DateTime LastUsedAt { get; set; }
}

/// <summary>
/// Manages WireGuard overlay network
/// </summary>
public interface INetworkManager
{
    /// <summary>
    /// Get the node's WireGuard public key
    /// </summary>
    Task<string> GetWireGuardPublicKeyAsync(CancellationToken ct = default);

    /// <summary>
    /// Add a peer to a specific WireGuard interface
    /// </summary>
    /// <param name="interfaceName">WireGuard interface name (e.g., wg-relay, wg-hub, wg-relay-server)</param>
    /// <param name="publicKey">Peer's WireGuard public key</param>
    /// <param name="endpoint">Peer's endpoint (IP:port)</param>
    /// <param name="allowedIps">Comma-separated list of allowed IPs</param>
    Task AddPeerAsync(
        string interfaceName,
        string publicKey,
        string endpoint,
        string allowedIps,
        CancellationToken ct = default);

    /// <summary>
    /// Remove a peer from a specific WireGuard interface
    /// </summary>
    /// <param name="interfaceName">WireGuard interface name (e.g., wg-relay, wg-hub, wg-relay-server)</param>
    /// <param name="publicKey">Peer's WireGuard public key</param>
    Task RemovePeerAsync(
        string interfaceName,
        string publicKey,
        CancellationToken ct = default);

    /// <summary>
    /// Get all peers from a specific WireGuard interface
    /// </summary>
    /// <param name="interfaceName">WireGuard interface name (e.g., wg-relay, wg-hub, wg-relay-server)</param>
    Task<List<WireGuardPeer>> GetPeersAsync(
        string interfaceName,
        CancellationToken ct = default);

    /// <summary>
    /// Create network interface for a VM
    /// </summary>
    Task<string> CreateVmNetworkAsync(
        string vmId,
        VmNetworkConfig config,
        CancellationToken ct = default);

    /// <summary>
    /// Delete network interface for a VM
    /// </summary>
    Task DeleteVmNetworkAsync(
        string vmId,
        CancellationToken ct = default);

    /// <summary>
    /// Start a WireGuard interface using wg-quick
    /// </summary>
    /// <param name="interfaceName">Interface name to start (e.g., wg-relay, wg-hub)</param>
    Task<bool> StartWireGuardInterfaceAsync(
        string interfaceName,
        CancellationToken ct = default);
}

public class WireGuardPeer
{
    public string PublicKey { get; set; } = string.Empty;
    public string Endpoint { get; set; } = string.Empty;
    public string AllowedIps { get; set; } = string.Empty;
    public DateTime? LastHandshake { get; set; }
    public long TransferRx { get; set; }
    public long TransferTx { get; set; }
}

/// <summary>
/// Communication with the orchestration layer
/// </summary>
public interface IOrchestratorClient
{
    Task InitializeAsync(CancellationToken ct = default);
    string? NodeId { get; }

    string? WalletAddress { get; }

    Task<RegistrationResult> RegisterWithPendingAuthAsync(CancellationToken ct = default);
    Task<RegistrationResult> RegisterNodeAsync(NodeRegistration registration, CancellationToken ct = default);
    Task<bool> SendHeartbeatAsync(Heartbeat heartbeat, CancellationToken ct = default);

    Task<List<PendingCommand>> GetPendingCommandsAsync(CancellationToken ct = default);
    /// <summary>
    /// Fetch pending commands from orchestrator via dedicated endpoint
    /// Used by hybrid push-pull command delivery system
    /// </summary>
    Task<List<PendingCommand>> FetchPendingCommandsAsync(CancellationToken ct = default);
    Task<bool> AcknowledgeCommandAsync(string commandId, bool success, string? errorMessage, string? data = null, CancellationToken ct = default);
    HeartbeatDto? GetLastHeartbeat();
    Task ReloadCredentialsAsync(CancellationToken ct = default);
    /// <summary>
    /// Get node summary information from orchestrator.
    /// Endpoint: GET /api/nodes/me
    /// </summary>
    /// <remarks>
    /// Returns node status, region, public IP, registration date, and agent version.
    /// Useful for verifying orchestrator's view of this node matches local state.
    /// </remarks>
    Task<NodeSummaryResponse?> GetNodeSummaryAsync(CancellationToken ct = default);

    /// <summary>
    /// Get current scheduling configuration from orchestrator.
    /// Endpoint: GET /api/nodes/me/config
    /// </summary>
    /// <remarks>
    /// Returns the authoritative scheduling config including:
    /// - Baseline benchmark score
    /// - Overcommit ratios per tier
    /// - Tier configurations
    /// Use this for on-demand config refresh outside heartbeat cycle.
    /// </remarks>
    Task<SchedulingConfig?> GetSchedulingConfigAsync(CancellationToken ct = default);

    /// <summary>
    /// Get node performance evaluation and tier eligibility.
    /// Endpoint: GET /api/nodes/me/performance
    /// </summary>
    /// <remarks>
    /// Returns orchestrator's evaluation of this node including:
    /// - Benchmark scores (raw and capped)
    /// - Points per core calculation
    /// - Eligible tiers and their capabilities
    /// - Highest eligible tier
    /// </remarks>
    Task<NodePerformanceEvaluation?> GetPerformanceEvaluationAsync(CancellationToken ct = default);

    /// <summary>
    /// Get node capacity and current allocation status.
    /// Endpoint: GET /api/nodes/me/capacity
    /// </summary>
    /// <remarks>
    /// Returns detailed capacity information including:
    /// - Physical resources (cores, memory, storage)
    /// - Total compute points available
    /// - Current allocations and availability
    /// - Per-VM breakdown
    /// </remarks>
    Task<NodeCapacityResponse?> GetCapacityAsync(CancellationToken ct = default);

    /// <summary>
    /// Request orchestrator to re-evaluate node performance.
    /// Endpoint: POST /api/nodes/me/evaluate
    /// </summary>
    /// <remarks>
    /// Triggers a fresh performance evaluation on the orchestrator.
    /// Use after hardware changes or benchmark updates.
    /// Returns the new evaluation result.
    /// </remarks>
    Task<NodePerformanceEvaluation?> RequestPerformanceEvaluationAsync(CancellationToken ct = default);

    /// <summary>
    /// Synchronize all node state from orchestrator.
    /// Calls multiple /api/nodes/me endpoints to refresh local state.
    /// </summary>
    /// <remarks>
    /// This is a convenience method that fetches:
    /// 1. Scheduling configuration
    /// 2. Performance evaluation
    /// 3. Node summary
    /// 
    /// Use on startup or after extended disconnection to ensure
    /// local state matches orchestrator's authoritative view.
    /// </remarks>
    Task<NodeSyncResult> SyncWithOrchestratorAsync(CancellationToken ct = default);

    Task<bool> IsOrchestratorReachableAsync(CancellationToken ct = default);
}

/// <summary>
/// Response from GET /api/nodes/me
/// </summary>
public class NodeSummaryResponse
{
    public string NodeId { get; set; } = string.Empty;
    public NodeStatus Status { get; set; }
    public string? Region { get; set; }
    public string? PublicIp { get; set; }
    public DateTime RegisteredAt { get; set; }
    public DateTime LastHeartbeat { get; set; }
    public string AgentVersion { get; set; } = string.Empty;
    public int? SchedulingConfigVersion { get; set; }
}

/// <summary>
/// Response from GET /api/nodes/me/capacity
/// </summary>
public class NodeCapacityResponse
{
    public string NodeId { get; set; } = string.Empty;

    // Physical resources
    public int PhysicalCores { get; set; }
    public long PhysicalMemoryBytes { get; set; }
    public long PhysicalStorageBytes { get; set; }

    // Point-based capacity
    public double PointsPerCore { get; set; }
    public int TotalComputePoints { get; set; }
    public int AllocatedComputePoints { get; set; }
    public int AvailableComputePoints => TotalComputePoints - AllocatedComputePoints;
    public double UtilizationPercent => TotalComputePoints > 0
        ? (double)AllocatedComputePoints / TotalComputePoints * 100
        : 0;

    // Memory
    public long AllocatedMemoryBytes { get; set; }
    public long AvailableMemoryBytes { get; set; }

    // Storage
    public long AllocatedStorageBytes { get; set; }
    public long AvailableStorageBytes { get; set; }

    // VM information
    public int ActiveVmCount { get; set; }
    public List<VmAllocationSummary> VmBreakdown { get; set; } = new();
}

/// <summary>
/// VM allocation summary within capacity response
/// </summary>
public class VmAllocationSummary
{
    public string VmId { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Tier { get; set; } = string.Empty;
    public int VCpus { get; set; }
    public int Points { get; set; }
    public long MemoryBytes { get; set; }
    public VmStatus Status { get; set; }
}

public enum VmStatus
{
    Pending,        // 0 - Waiting to be scheduled
    Scheduling,     // 1 - Finding a node
    Provisioning,   // 2 - Being created on node
    Running,        // 3 - Active and running
    Stopping,       // 4 - Being stopped
    Stopped,        // 5 - Stopped but resources reserved
    Deleting,       // 6 - Deletion in progress, waiting for node confirmation
    Migrating,      // 7 - Being moved to another node
    Error,          // 8 - Something went wrong
    Deleted         // 9 - Deletion confirmed, resources freed
}

/// <summary>
/// Result of full node synchronization
/// </summary>
public class NodeSyncResult
{
    public bool Success { get; set; }
    public string? Error { get; set; }

    public bool ConfigSynced { get; set; }
    public bool PerformanceSynced { get; set; }
    public bool SummarySynced { get; set; }

    public int? ConfigVersion { get; set; }
    public QualityTier? HighestTier { get; set; }
    public int? TotalComputePoints { get; set; }

    public DateTime SyncedAt { get; set; } = DateTime.UtcNow;

    public static NodeSyncResult Failed(string error) => new()
    {
        Success = false,
        Error = error,
        SyncedAt = DateTime.UtcNow
    };
}

/// <summary>
/// Manages iptables NAT rules for relay VM port forwarding
/// </summary>
public interface INatRuleManager
{
    /// <summary>
    /// Adds port forwarding NAT rule for a VM
    /// </summary>
    Task<bool> AddPortForwardingAsync(
        string vmIp,
        int port,
        string protocol = "udp",
        CancellationToken ct = default);

    /// <summary>
    /// Removes port forwarding NAT rule for a VM
    /// </summary>
    Task<bool> RemovePortForwardingAsync(
        string vmIp,
        int port,
        string protocol = "udp",
        CancellationToken ct = default);

    /// <summary>
    /// Checks if NAT rule exists for a VM
    /// </summary>
    Task<bool> RuleExistsAsync(
        string vmIp,
        int port,
        string protocol = "udp",
        CancellationToken ct = default);

    /// <summary>
    /// Asynchronously determines whether any rules are defined for the specified virtual machine IP address.
    /// </summary>
    /// <param name="vmIp">The IP address of the virtual machine to check for associated rules. Cannot be null or empty.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains <see langword="true"/> if one or
    /// more rules exist for the specified virtual machine; otherwise, <see langword="false"/>.</returns>
    Task<bool> HasRulesForVmAsync(string vmIp, CancellationToken ct = default);

    /// <summary>
    /// Saves iptables rules persistently
    /// </summary>
    Task<bool> SaveRulesAsync(CancellationToken ct = default);

    /// <summary>
    /// Removes all relay NAT rules (cleanup)
    /// </summary>
    Task<bool> RemoveAllRelayNatRulesAsync(CancellationToken ct = default);

    /// <summary>
    /// Gets list of existing NAT rules
    /// </summary>
    Task<List<string>> GetExistingRulesAsync(CancellationToken ct = default);
}

public class PendingCommand
{
    public string CommandId { get; set; } = string.Empty;
    public CommandType Type { get; set; }
    public string Payload { get; set; } = string.Empty;  // JSON payload
    public bool RequiresAck { get; set; } = true;
    public DateTime IssuedAt { get; set; }
}

public enum CommandType
{
    CreateVm,
    StartVm,
    StopVm,
    DeleteVm,
    UpdateNetwork,
    Benchmark,
    Shutdown,
    AllocatePort,
    RemovePort
}

/// <summary>
/// Executes shell commands (virsh, wg, etc.)
/// </summary>
public interface ICommandExecutor
{
    Task<CommandResult> ExecuteAsync(string command, string arguments, CancellationToken ct = default);
    Task<CommandResult> ExecuteAsync(string command, string arguments, TimeSpan timeout, CancellationToken ct = default);
}

public class CommandResult
{
    public int ExitCode { get; set; }
    public string StandardOutput { get; set; } = string.Empty;
    public string StandardError { get; set; } = string.Empty;
    public TimeSpan Duration { get; set; }
    public bool Success => ExitCode == 0;
}
