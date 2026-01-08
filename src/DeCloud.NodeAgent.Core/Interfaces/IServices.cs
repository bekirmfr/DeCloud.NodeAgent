using DeCloud.NodeAgent.Core.Models;

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
    public string? Error { get; init; }

    public static RegistrationResult Success(string nodeId, string apiKey) =>
        new() { IsSuccess = true, NodeId = nodeId, ApiKey = apiKey };

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
    Task<CpuInfo> GetCpuInfoAsync(CancellationToken ct = default);
    Task<MemoryInfo> GetMemoryInfoAsync(CancellationToken ct = default);
    Task<List<StorageInfo>> GetStorageInfoAsync(CancellationToken ct = default);
    Task<List<GpuInfo>> GetGpuInfoAsync(CancellationToken ct = default);
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
    Task<VmOperationResult> DeleteVmAsync(string vmId, CancellationToken ct = default);
    Task<VmOperationResult> PauseVmAsync(string vmId, CancellationToken ct = default);
    Task<VmOperationResult> ResumeVmAsync(string vmId, CancellationToken ct = default);
    
    Task<VmInstance?> GetVmAsync(string vmId, CancellationToken ct = default);
    Task<List<VmInstance>> GetAllVmsAsync(CancellationToken ct = default);
    Task<VmResourceUsage> GetVmUsageAsync(string vmId, CancellationToken ct = default);
    
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
    string? NodeId { get; }

    string? WalletAddress { get; }

    Task<RegistrationResult> RegisterWithPendingAuthAsync(CancellationToken ct = default);
    Task<RegistrationResult> RegisterNodeAsync(NodeRegistration registration, CancellationToken ct = default);
    Task<bool> SendHeartbeatAsync(Heartbeat heartbeat, CancellationToken ct = default);

    Task<List<PendingCommand>> GetPendingCommandsAsync(CancellationToken ct = default);
    Task<bool> AcknowledgeCommandAsync(string commandId, bool success, string? errorMessage, CancellationToken ct = default);
    HeartbeatDto? GetLastHeartbeat();
    Task ReloadCredentialsAsync(CancellationToken ct = default);
}

/// <summary>
/// Manages iptables NAT rules for relay VM port forwarding
/// </summary>
public interface INatRuleManager
{
    Task<bool> AddPortForwardingAsync(string vmIp, int port, string protocol = "udp", CancellationToken ct = default);
    Task<bool> RemovePortForwardingAsync(string vmIp, int port, string protocol = "udp", CancellationToken ct = default);
    Task<bool> RuleExistsAsync(string vmIp, int port, string protocol = "udp", CancellationToken ct = default);
    Task<bool> SaveRulesAsync(CancellationToken ct = default);
    Task<bool> RemoveAllRelayNatRulesAsync(CancellationToken ct = default);
    Task<List<string>> GetExistingRulesAsync(CancellationToken ct = default);
}

/// <summary>
/// Thread-safe authentication state tracking
/// </summary>
public interface IAuthenticationStateService
{
    /// <summary>
    /// Current authentication state
    /// </summary>
    AuthenticationState CurrentState { get; }

    /// <summary>
    /// Simple registered check
    /// </summary>
    bool IsRegistered { get; }

    /// <summary>
    /// Resource discovery completion check
    /// </summary>
    bool IsDiscoveryComplete { get; }

    /// <summary>
    /// Update authentication state (called by AuthenticationManager)
    /// </summary>
    void UpdateState(AuthenticationState newState);

    /// <summary>
    /// Wait until registered or cancelled
    /// </summary>
    Task WaitForRegistrationAsync(CancellationToken ct);
}

public enum AuthenticationState
{
    Initializing,
    WaitingForDiscovery,
    NotAuthenticated,
    PendingRegistration,
    Registered,
    CredentialsInvalid
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
    Shutdown
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
