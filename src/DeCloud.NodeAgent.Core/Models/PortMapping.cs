namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Represents a port mapping from node public port to VM internal port.
/// Stored in SQLite for persistence across node restarts.
/// </summary>
public class PortMapping
{
    /// <summary>
    /// Unique identifier for this port mapping
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// VM this port mapping belongs to
    /// </summary>
    public string VmId { get; set; } = string.Empty;

    /// <summary>
    /// VM's internal IP address (e.g., 192.168.122.50)
    /// </summary>
    public string VmPrivateIp { get; set; } = string.Empty;

    /// <summary>
    /// Port on the VM (e.g., 22 for SSH, 3306 for MySQL)
    /// </summary>
    public int VmPort { get; set; }

    /// <summary>
    /// Public port on the node (e.g., 42156)
    /// Allocated from pool (40000-65535)
    /// </summary>
    public int PublicPort { get; set; }

    /// <summary>
    /// Protocol: TCP, UDP, or Both
    /// </summary>
    public PortProtocol Protocol { get; set; } = PortProtocol.TCP;

    /// <summary>
    /// Optional label for this port mapping (e.g., "SSH", "MySQL")
    /// </summary>
    public string? Label { get; set; }

    /// <summary>
    /// When this port mapping was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Whether iptables rules are currently active
    /// </summary>
    public bool IsActive { get; set; } = true;
}

/// <summary>
/// Port protocol type
/// </summary>
public enum PortProtocol
{
    TCP = 1,
    UDP = 2,
    Both = 3  // TCP + UDP
}
