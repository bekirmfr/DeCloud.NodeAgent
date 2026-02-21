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

    // Optional GPU passthrough (VirtualMachine mode)
    public string? GpuPciAddress { get; set; }

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
    Inference
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
/// GPU setup mode sent by orchestrator in ConfigureGpu command.
/// Auto: orchestrator picks best strategy based on IOMMU support.
/// </summary>
public enum GpuSetupMode
{
    Auto = 0,
    VfioPassthrough = 1,
    ContainerToolkit = 2
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

    public string? StatusMessage { get; set; }
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
    ExecCommand
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