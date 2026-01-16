namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Complete hardware inventory of a node
/// </summary>
public class HardwareInventory
{
    public string NodeId { get; set; } = string.Empty;
    public CpuInfo Cpu { get; set; } = new();
    public MemoryInfo Memory { get; set; } = new();
    public List<StorageInfo> Storage { get; set; } = new();
    public bool SupportsGpu { get; set; }
    public List<GpuInfo> Gpus { get; set; } = new();
    public NetworkInfo Network { get; set; } = new();
    public DateTime CollectedAt { get; set; } = DateTime.UtcNow;
}

public class CpuInfo
{
    public string Model { get; set; } = string.Empty;
    /// <summary>
    /// CPU Architecture: x86_64, aarch64, etc.
    /// </summary>
    public string Architecture { get; set; } = string.Empty;
    public int PhysicalCores { get; set; }
    public int LogicalCores { get; set; }
    public double FrequencyMhz { get; set; }
    public List<string> Flags { get; set; } = new(); // e.g., "vmx", "svm" for virtualization
    public bool SupportsVirtualization { get; set; }
    
    // Current utilization (0-100)
    public double UsagePercent { get; set; }
    
    // Available for VMs (considering overcommit ratio)
    public int AvailableVCpus { get; set; }
    /// <summary>
    /// CPU benchmark score - measured during node registration
    /// 1000 = Burstable baseline
    /// 1500 = Balanced tier minimum
    /// 2500 = Standard tier minimum  
    /// 4000 = Guaranteed tier minimum
    /// </summary>
    public int BenchmarkScore { get; set; } = 1000;
}

public class MemoryInfo
{
    public long TotalBytes { get; set; }
    public long AvailableBytes { get; set; }
    public long UsedBytes { get; set; }
    
    // Reserved for host OS (configurable)
    public long ReservedBytes { get; set; }
    
    // Available for VM allocation
    public long AllocatableBytes => Math.Max(0, TotalBytes - ReservedBytes - UsedBytes);
    
    public double UsagePercent => TotalBytes > 0 ? (double)UsedBytes / TotalBytes * 100 : 0;
}

public class StorageInfo
{
    public string DevicePath { get; set; } = string.Empty;  // e.g., /dev/sda
    public string MountPoint { get; set; } = string.Empty;  // e.g., /var/lib/decloud
    public string FileSystem { get; set; } = string.Empty;  // e.g., ext4, xfs
    public StorageType Type { get; set; }
    public long TotalBytes { get; set; }
    public long AvailableBytes { get; set; }
    public long UsedBytes { get; set; }
    
    // Measured IOPS (optional, from benchmark)
    public int? ReadIops { get; set; }
    public int? WriteIops { get; set; }
}

public enum StorageType
{
    Unknown,
    HDD,
    SSD,
    NVMe
}

public class GpuInfo
{
    public string Vendor { get; set; } = string.Empty;      // NVIDIA, AMD, Intel
    public string Model { get; set; } = string.Empty;       // e.g., RTX 4090
    public string PciAddress { get; set; } = string.Empty;  // e.g., 0000:01:00.0
    public long MemoryBytes { get; set; }
    public long MemoryUsedBytes { get; set; }
    public string DriverVersion { get; set; } = string.Empty;
    
    // VFIO passthrough readiness
    public bool IsIommuEnabled { get; set; }
    public string IommuGroup { get; set; } = string.Empty;
    public bool IsAvailableForPassthrough { get; set; }
    
    // Current utilization
    public double GpuUsagePercent { get; set; }
    public double MemoryUsagePercent { get; set; }
    public int? TemperatureCelsius { get; set; }
}

public class NetworkInfo
{
    public string PublicIp { get; set; } = string.Empty;
    public string PrivateIp { get; set; } = string.Empty;
    public string WireGuardIp { get; set; } = string.Empty;
    public int? WireGuardPort { get; set; }
    
    // Bandwidth (measured via iperf3 or similar)
    public long? BandwidthBitsPerSecond { get; set; }
    
    // NAT type detection
    public NatType NatType { get; set; }
    
    public List<NetworkInterface> Interfaces { get; set; } = new();
}

public enum NatType
{
    Unknown,
    None,           // Public IP, no NAT
    FullCone,       // Easy to traverse
    RestrictedCone,
    PortRestricted,
    Symmetric       // Hardest to traverse, may need relay
}

public class NetworkInterface
{
    public string Name { get; set; } = string.Empty;       // e.g., eth0
    public string MacAddress { get; set; } = string.Empty;
    public string IpAddress { get; set; } = string.Empty;
    public long SpeedMbps { get; set; }
    public bool IsUp { get; set; }
}
