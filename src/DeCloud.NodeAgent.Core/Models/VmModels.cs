using System.Text.Json.Serialization;

namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Specification for creating a new VM
/// </summary>
public class VmSpec
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public VmType VmType { get; set; } = VmType.Relay;
    public string NodeId { get; set; } = string.Empty; // Target node ID

    // Resource allocation
    public int VirtualCpuCores { get; set; } = 1;

    public int VcpuQuotaMicroseconds { get; set; } = -1;
    public int VcpuPeriodMicroseconds { get; set; } = -1;
    /// <summary>
    /// Timestamp when CPU quota was applied (for Burstable tier)
    /// </summary>
    public DateTime? VcpuQuotaAppliedAt { get; set; } = null;
    public long MemoryBytes { get; set; } = 1024 * 1024 * 1024; // 1GB default
    public long DiskBytes { get; set; } = 10L * 1024 * 1024 * 1024; // 10GB default

    // Quality tier and point cost
    public QualityTier QualityTier { get; set; } = QualityTier.Standard;  // 0=Guaranteed, 1=Standard, 3=Balanced, 3=Burstable
    public int ComputePointCost { get; set; } = 0; // Total points (vCPUs � pointsPerVCpu)

    // Image source
    public string BaseImageUrl { get; set; } = string.Empty;  // URL to download base image
    public string BaseImageHash { get; set; } = string.Empty; // SHA256 for verification

    // GPU
    public string? GpuPciAddress { get; set; }

    /// <summary>
    /// How GPU access is provided to this VM.
    /// None = no GPU, Passthrough = VFIO (dedicated), Proxied = GPU proxy daemon (shared).
    /// Set by the orchestrator as a scheduling parameter — the orchestrator picks the mode
    /// based on node capabilities (IOMMU, GPU availability) during scheduling.
    /// </summary>
    public GpuMode GpuMode { get; set; } = GpuMode.None;

    /// <summary>
    /// Vsock context ID (CID) assigned to this VM for host↔guest communication.
    /// Used by the GPU proxy daemon to identify which VM is making requests.
    /// CID 0 = hypervisor, 1 = reserved, 2 = host, 3+ = guests.
    /// Set automatically by LibvirtVmManager when GpuMode is Proxied.
    /// Null on WSL2 (vsock unavailable).
    /// </summary>
    public uint? VsockCid { get; set; }

    /// <summary>
    /// Per-VM auth token for GPU proxy TCP connections.
    /// 32-byte hex string generated at VM creation, injected via cloud-init.
    /// Used when vsock is unavailable (WSL2) to authenticate TCP connections
    /// from the guest shim to the host GPU proxy daemon.
    /// </summary>
    public string? GpuProxyToken { get; set; }

    // Deployment mode (VM or Container)
    public DeploymentMode DeploymentMode { get; set; } = DeploymentMode.VirtualMachine;

    // Container-specific fields (used when DeploymentMode = Container)
    public string? ContainerImage { get; set; }
    public Dictionary<string, string>? EnvironmentVariables { get; set; }

    // Network configuration
    public string IpAddress { get; set; } = string.Empty;  // Within overlay network
    public string MacAddress { get; set; } = string.Empty;

    // Bandwidth limiting (enforced via libvirt QoS)
    public BandwidthTier BandwidthTier { get; set; } = BandwidthTier.Unmetered;

    // Cloud-init configuration (optional)
    public string? CloudInitUserData { get; set; }
    public string? SshPublicKey { get; set; }
    
    /// <summary>
    /// Wallet-encrypted password (stored permanently)
    /// Format: base64(iv):base64(ciphertext):base64(tag)
    /// SECURITY: Only store encrypted passwords, never plaintext
    /// </summary>
    public string? WalletEncryptedPassword { get; set; }     // Wallet-encrypted (persistent)

    // Billing and ownership
    public string? OwnerId { get; set; } = string.Empty;      // Tenant/user ID
    public Dictionary<string, string>? Labels { get; set; }
    // =========================================================================
    // REPLICATION
    // =========================================================================

    /// <summary>
    /// Number of block store providers that must hold the VM's overlay chunks
    /// before a lazysync version is considered confirmed.
    ///
    /// Supported values:
    ///   0 — Ephemeral. Lazysync skipped entirely. VM data is lost on node failure.
    ///       Use for stateless workloads, batch jobs, CI runners.
    ///   1 — Single replica. Survives if ≥1 block store provider holds the blocks.
    ///   3 — Standard (default). Survives loss of 2 provider nodes simultaneously.
    ///   5 — High availability. Survives loss of 4 provider nodes simultaneously.
    ///       Use for databases, ML checkpoints, critical stateful services.
    ///
    /// Immutable after VM creation. Default: 3 for tenant VMs, 0 for system VMs.
    /// Affects scheduling: replicationFactor > 0 requires an Active BlockStore
    /// on the target node.
    /// </summary>
    public int ReplicationFactor { get; set; } = 0;

    /// <summary>
    /// Authoritative node ID for this VM. Set at CreateVm time, updated on migration.
    /// NodeAgent compares against its own node ID on startup and each reconciliation
    /// cycle — mismatch means this VM is a zombie and must be destroyed.
    /// Null on VMs created before this field was introduced (safe — skipped by check).
    /// </summary>
    public string? TargetNodeId { get; set; }

    /// <summary>
    /// Non-null when this CreateVm is a migration, not a fresh deployment.
    /// The confirmed manifest root CID — used only for logging/diagnostics.
    /// Disk reconstruction uses OverlayChunkMap.
    /// </summary>
    public string? OverlayRootCid { get; set; }

    /// <summary>
    /// Offset→CID map from the confirmed manifest.
    /// When non-null, LibvirtVmManager fetches each block from the local
    /// BlockStore (which bitswap-fetches from DHT providers) and writes it
    /// at the correct byte offset in the newly-created overlay before boot.
    /// </summary>
    public Dictionary<long, string>? OverlayChunkMap { get; set; }
}

public class VmNetworkConfig
{
    public string MacAddress { get; set; } = string.Empty;
    public string IpAddress { get; set; } = string.Empty;  // Within overlay network
}

/// <summary>
/// Runtime state of a VM
/// </summary>
public class VmInstance
{
    public string VmId { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public VmState State { get; set; }

    /// <summary>
    /// Per-service readiness statuses.
    /// "System" (cloud-init) always first. Additional services from template.
    /// Checked via qemu-guest-agent by VmReadinessMonitor.
    /// </summary>
    public List<VmServiceStatus> Services { get; set; } = new();
    public bool IsFullyReady => Services.Count > 0 && Services.All(s => s.Status == ServiceReadiness.Ready);

    public VmSpec Spec { get; set; } = new();
    public string? NetworkInterface { get; set; }  // e.g., "vnet0"

    // Runtime info
    public int? Pid { get; set; }  // QEMU process ID
    public int? VncPort { get; set; }

    // Resource usage
    public VmResourceUsage CurrentUsage { get; set; } = new();

    // Timestamps
    public DateTime CreatedAt { get; set; }
    public DateTime? StartedAt { get; set; }
    public DateTime? StoppedAt { get; set; }
    public DateTime LastHeartbeat { get; set; }

    // Paths
    public string DiskPath { get; set; } = string.Empty;
    public string ConfigPath { get; set; } = string.Empty;
}

public enum VmState
{
    Pending,      // Spec received, not yet created
    Creating,     // Image downloading, disk creating
    Starting,     // Booting
    Running,
    Paused,
    Stopping,
    Stopped,
    Failed,
    NotFound,
    Deleted,
    Migrating
}

public enum VmType
{
    General,
    Compute,
    Memory,
    Storage,
    Gpu,
    Relay,
    Dht,
    Inference,
    BlockStore  // Distributed block storage duty (5% of node storage)
}

/// <summary>
/// How a workload is deployed on a node.
/// VirtualMachine: KVM/QEMU VM via libvirt (default, full isolation)
/// Container: Docker container with GPU sharing (for nodes without IOMMU, e.g. WSL2)
/// </summary>
public enum DeploymentMode
{
    VirtualMachine = 0,
    Container = 1
}

/// <summary>
/// How GPU access is provided to a virtual machine.
/// </summary>
public enum GpuMode
{
    /// <summary>No GPU access</summary>
    None = 0,

    /// <summary>
    /// VFIO passthrough: GPU bound to vfio-pci, passed as PCI hostdev to VM.
    /// Requires IOMMU enabled in BIOS/kernel. One GPU per VM, full performance.
    /// </summary>
    Passthrough = 1,

    /// <summary>
    /// GPU proxy: VM communicates with a host-side GPU proxy daemon over virtio-vsock.
    /// A CUDA shim (LD_PRELOAD) inside the VM intercepts CUDA calls and forwards them.
    /// Works without IOMMU. Multiple VMs can share one GPU.
    /// </summary>
    Proxied = 2
}

public enum QualityTier
{
    /// <summary>
    /// Dedicated resources, guaranteed performance
    /// Requires highest-performance nodes (4000+ benchmark)
    /// </summary>
    Guaranteed = 0,

    /// <summary>
    /// High performance for demanding applications
    /// Requires high-end nodes (2500+ benchmark)
    /// </summary>
    Standard = 1,

    /// <summary>
    /// Balanced performance for production workloads
    /// Requires mid-range nodes (1500+ benchmark)
    /// </summary>
    Balanced = 2,

    /// <summary>
    /// Best-effort, lowest cost
    /// Minimum acceptable performance (1000+ benchmark)
    /// </summary>
    Burstable = 3
}

/// <summary>
/// Bandwidth tier for VM network rate limiting.
/// Enforced via libvirt QoS on the VM's virtio network interface.
/// </summary>
public enum BandwidthTier
{
    /// <summary>10 Mbps average, 20 Mbps burst - Light browsing, text</summary>
    Basic = 0,

    /// <summary>50 Mbps average, 100 Mbps burst - General browsing, streaming</summary>
    Standard = 1,

    /// <summary>200 Mbps average, 400 Mbps burst - HD video, downloads</summary>
    Performance = 2,

    /// <summary>No artificial cap - limited only by host NIC</summary>
    Unmetered = 3
}

/// <summary>
/// Capability information for a specific tier
/// </summary>
public class TierCapability
{
    public QualityTier Tier { get; set; }
    public int MinimumBenchmark { get; set; }
    public double RequiredPointsPerVCpu { get; set; }
    public double NodePointsPerCore { get; set; }
    public int MaxVCpusPerCore { get; set; }
    public decimal PriceMultiplier { get; set; }
    public string Description { get; set; } = string.Empty;
    public bool IsEligible { get; set; }
    public string? IneligibilityReason { get; set; }
}

public class VmResourceUsage
{
    public double CpuPercent { get; set; }
    public long MemoryUsedBytes { get; set; }
    public long DiskReadBytes { get; set; }
    public long DiskWriteBytes { get; set; }
    public long NetworkRxBytes { get; set; }
    public long NetworkTxBytes { get; set; }
    public DateTime MeasuredAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Result of a VM operation
/// </summary>
public class VmOperationResult
{
    public bool Success { get; set; }
    public string VmId { get; set; } = string.Empty;
    public VmState? NewState { get; set; }
    public string? ErrorMessage { get; set; }
    public string? ErrorCode { get; set; }

    public static VmOperationResult Ok(string vmId, VmState state) => new()
    {
        Success = true,
        VmId = vmId,
        NewState = state
    };

    public static VmOperationResult Fail(string vmId, string error, string? code = null) => new()
    {
        Success = false,
        VmId = vmId,
        ErrorMessage = error,
        ErrorCode = code
    };
}

/// <summary>
/// Readiness status of a single service inside a VM.
/// Checked via qemu-guest-agent (virtio channel, no network needed).
/// </summary>
public class VmServiceStatus
{
    public string Name { get; set; } = string.Empty;
    public int? Port { get; set; }
    public string? Protocol { get; set; }
    public CheckType CheckType { get; set; } = CheckType.CloudInitDone;
    public string? HttpPath { get; set; }
    public string? ExecCommand { get; set; }

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public ServiceReadiness Status { get; set; } = ServiceReadiness.Pending;
    /// <summary>
    /// When true, this service is periodically re-verified after reaching
    /// Ready. A failed re-check reverts Ready → Failed. Default: false.
    /// </summary>
    public bool LivenessCheck { get; set; } = false;

    public string? StatusMessage { get; set; }

    /// <summary>
    /// Last successful HTTP response body from this service's health endpoint
    /// (truncated to 512 chars). Persists across status transitions so the
    /// pre-crash state (memory pressure, OOM count, peer info) survives in
    /// ServicesJson when the VM is deleted. Only populated for HttpGet checks.
    /// </summary>
    public string? LastSuccessBody { get; set; }
    public DateTime? ReadyAt { get; set; }
    public DateTime? LastCheckAt { get; set; }
    public int TimeoutSeconds { get; set; } = 300;
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum CheckType
{
    /// <summary>cloud-init status --format json → status == "done"</summary>
    CloudInitDone,

    /// <summary>nc -zv -w2 localhost {port} → exit 0</summary>
    TcpPort,

    /// <summary>curl -sf -o /dev/null http://localhost:{port}{path} → exit 0</summary>
    HttpGet,

    /// <summary>Arbitrary command via bash -c → exit 0</summary>
    ExecCommand,

    /// <summary>
    /// Host-side guest-agent ping via virsh qemu-agent-command guest-ping.
    /// Does not run through guest-exec — it IS the agent liveness test.
    /// Evaluated before the guest-agent gate in VmReadinessMonitor.
    /// </summary>
    GuestAgentPing
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum ServiceReadiness
{
    /// <summary>Waiting for System (cloud-init) to complete first</summary>
    Pending,

    /// <summary>Actively being probed</summary>
    Checking,

    /// <summary>Check passed — service is accepting traffic</summary>
    Ready,

    /// <summary>Timeout expired without passing check</summary>
    TimedOut,

    /// <summary>cloud-init reported error (System service only)</summary>
    Failed
}

/// <summary>
/// GPU usage statistics for a single VM connection, used for billing and monitoring.
/// Populated by the GPU proxy daemon via the GET_USAGE_STATS protocol command.
/// </summary>
public class GpuUsageStats
{
    /// <summary>Current GPU memory allocated (bytes)</summary>
    public long MemoryAllocated { get; set; }

    /// <summary>Configured memory quota (0 = unlimited)</summary>
    public long MemoryQuota { get; set; }

    /// <summary>Peak GPU memory usage (bytes)</summary>
    public long PeakMemory { get; set; }

    /// <summary>Cumulative total bytes allocated</summary>
    public long TotalAllocBytes { get; set; }

    /// <summary>Total kernel launches</summary>
    public int KernelLaunches { get; set; }

    /// <summary>Number of kernels killed by timeout</summary>
    public int KernelTimeouts { get; set; }

    /// <summary>Cumulative kernel execution time (microseconds)</summary>
    public long KernelTimeUs { get; set; }

    /// <summary>Time since VM connected (microseconds)</summary>
    public long ConnectTimeUs { get; set; }
}

public record VmDashboardRecord
{
    public string VmId { get; init; } = "";
    public string Name { get; init; } = "";
    public string State { get; init; } = "";
    public string VmType { get; init; } = "";
    public string? OwnerId { get; init; }
    public string? IpAddress { get; init; }
    public int? VncPort { get; init; }
    public int ReplicationFactor { get; init; }
    public int VirtualCpuCores { get; init; }
    public long MemoryBytes { get; init; }
    public long DiskBytes { get; init; }
    public string CreatedAt { get; init; } = "";
    public string LastUpdated { get; init; } = "";
    public string? TargetNodeId { get; init; }
    public string? DeletionReason { get; init; }
}