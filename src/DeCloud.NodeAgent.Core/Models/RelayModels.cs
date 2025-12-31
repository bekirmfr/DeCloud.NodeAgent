// Add this to: src/DeCloud.NodeAgent.Core/Models/RelayModels.cs (NEW FILE)

namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Configuration for CGNAT nodes received from orchestrator
/// </summary>
public class CgnatNodeInfo
{
    /// <summary>
    /// ID of the relay node serving this CGNAT node
    /// </summary>
    public string? AssignedRelayNodeId { get; set; }

    /// <summary>
    /// WireGuard tunnel IP assigned to this node
    /// </summary>
    public string TunnelIp { get; set; } = string.Empty;

    /// <summary>
    /// WireGuard configuration for connecting to relay
    /// </summary>
    public string? WireGuardConfig { get; set; }

    /// <summary>
    /// Public endpoint URL for accessing VMs on this node
    /// Format: https://relay-{region}-{id}.decloud.io
    /// </summary>
    public string PublicEndpoint { get; set; } = string.Empty;

    /// <summary>
    /// Connection status to relay
    /// </summary>
    public TunnelStatus TunnelStatus { get; set; } = TunnelStatus.Disconnected;

    /// <summary>
    /// Last successful handshake with relay
    /// </summary>
    public DateTime? LastHandshake { get; set; }
}

/// <summary>
/// WireGuard tunnel connection status
/// </summary>
public enum TunnelStatus
{
    /// <summary>
    /// Not connected to relay
    /// </summary>
    Disconnected,

    /// <summary>
    /// Attempting to establish connection
    /// </summary>
    Connecting,

    /// <summary>
    /// Tunnel is established and healthy
    /// </summary>
    Connected,

    /// <summary>
    /// Tunnel configuration or connection error
    /// </summary>
    Error
}