using DeCloud.NodeAgent.Core.Models;

namespace DeCloud.NodeAgent.Core.Interfaces;

/// <summary>
/// Discovers and monitors local hardware resources
/// </summary>
public interface IResourceDiscoveryService
{
    Task<NodeResources> DiscoverAllAsync(CancellationToken ct = default);
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
    Task<VmOperationResult> CreateVmAsync(VmSpec spec, CancellationToken ct = default);
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
    Task<bool> InitializeWireGuardAsync(CancellationToken ct = default);
    Task<string> GetWireGuardPublicKeyAsync(CancellationToken ct = default);
    Task AddPeerAsync(string publicKey, string endpoint, string allowedIps, CancellationToken ct = default);
    Task RemovePeerAsync(string publicKey, CancellationToken ct = default);
    Task<List<WireGuardPeer>> GetPeersAsync(CancellationToken ct = default);
    
    // VM networking
    Task<string> CreateVmNetworkAsync(string vmId, VmNetworkConfig config, CancellationToken ct = default);
    Task DeleteVmNetworkAsync(string vmId, CancellationToken ct = default);
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
    bool IsRegistered { get; }

    string? WalletAddress { get; }

    Task<bool> RegisterNodeAsync(NodeRegistration registration, CancellationToken ct = default);
    Task<bool> SendHeartbeatAsync(Heartbeat heartbeat, CancellationToken ct = default);

    Task<List<PendingCommand>> GetPendingCommandsAsync(CancellationToken ct = default);
    Task<bool> AcknowledgeCommandAsync(string commandId, bool success, string? errorMessage, CancellationToken ct = default);
    Heartbeat? GetLastHeartbeat();
}

public class PendingCommand
{
    public string CommandId { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
    public string? VmId { get; set; }
    public string? Payload { get; set; }
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
