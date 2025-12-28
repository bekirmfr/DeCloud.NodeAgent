namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Specification for creating a new VM
/// </summary>
public class VmSpec
{
    public string VmId { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;

    // Resource allocation
    public int VCpus { get; set; } = 1;
    public long MemoryBytes { get; set; } = 1024 * 1024 * 1024; // 1GB default
    public long DiskBytes { get; set; } = 10L * 1024 * 1024 * 1024; // 10GB default

    // Quality tier and point cost
    public int QualityTier { get; set; } = 1;  // 0=Guaranteed, 1=Standard, 3=Balanced, 3=Burstable
    public int ComputePointCost { get; set; } // Total points (vCPUs × pointsPerVCpu)

    // Image source
    public string BaseImageUrl { get; set; } = string.Empty;  // URL to download base image
    public string BaseImageHash { get; set; } = string.Empty; // SHA256 for verification

    // Optional GPU passthrough
    public string? GpuPciAddress { get; set; }

    // Network configuration
    public VmNetworkConfig Network { get; set; } = new();

    // Cloud-init configuration (optional)
    public string? CloudInitUserData { get; set; }
    public string? SshPublicKey { get; set; }

    // SECURITY: Password fields - only one should be set
    public string? Password { get; set; }              // Plaintext (transient, cleared after VM creation)
    
    /// <summary>
    /// Wallet-encrypted password (stored permanently)
    /// Format: base64(iv):base64(ciphertext):base64(tag)
    /// SECURITY: Only store encrypted passwords, never plaintext
    /// </summary>
    public string? EncryptedPassword { get; set; }     // Wallet-encrypted (persistent)

    // Billing and ownership
    public string TenantId { get; set; } = string.Empty;      // Tenant/user ID
    public string TenantWalletAddress { get; set; } = string.Empty; // Tenant's wallet address
    public string LeaseId { get; set; } = string.Empty;       // On-chain lease reference
}

public class VmNetworkConfig
{
    public string MacAddress { get; set; } = string.Empty;
    public string IpAddress { get; set; } = string.Empty;  // Within overlay network
    public string Gateway { get; set; } = string.Empty;
    public int VxlanVni { get; set; }  // Tenant isolation
    public List<int> AllowedPorts { get; set; } = new();  // Security group rules
}

/// <summary>
/// Runtime state of a VM
/// </summary>
public class VmInstance
{
    public string VmId { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public VmState State { get; set; }
    public VmSpec Spec { get; set; } = new();
    public string? NetworkInterface { get; set; }  // e.g., "vnet0"

    /// <summary>
    /// Timestamp when CPU quota was applied (for Burstable tier)
    /// </summary>
    public DateTime? QuotaAppliedAt { get; set; }

    // Runtime info
    public int? Pid { get; set; }  // QEMU process ID
    public string? VncPort { get; set; }
    public string? SpicePort { get; set; }

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
    Deleted,
    Migrating
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