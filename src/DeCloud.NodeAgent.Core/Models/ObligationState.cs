using System.Text.Json.Serialization;

namespace DeCloud.NodeAgent.Core.Models;

// ============================================================
// Placement: src/DeCloud.NodeAgent.Core/Models/ObligationState.cs
// ============================================================

/// <summary>
/// Base class for all obligation identity state.
///
/// Stored in SQLite obligation_state table as a JSON blob (state_json column).
/// The version field is also stored in its own column for efficient conflict
/// resolution — the JSON is not parsed to check the version.
///
/// SECURITY: State may contain private keys. The state_json column must
/// never be logged in full. Only role and version should appear in logs.
/// </summary>
public abstract class ObligationStateBase
{
    /// <summary>
    /// Monotonic version assigned by the orchestrator.
    /// Stored redundantly in the JSON blob and in its own column.
    /// Version-based conflict resolution: higher version always wins.
    /// </summary>
    public int Version { get; set; }

    /// <summary>UTC timestamp of when this state was last written by the orchestrator.</summary>
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Identity state for the Relay system VM role.
///
/// The WireGuard keypair is the stable identity anchor: every mesh-enrolled DHT
/// and BlockStore VM has the relay's public key baked into its wg-mesh.conf.
/// Preserving the keypair across relay redeployments lets all mesh peers reconnect
/// automatically without reconfiguration.
/// </summary>
public class RelayObligationState : ObligationStateBase
{
    /// <summary>WireGuard private key (base64). NEVER log this value.</summary>
    public string WireGuardPrivateKey { get; set; } = string.Empty;

    /// <summary>WireGuard public key derived from the private key. Safe to log.</summary>
    public string WireGuardPublicKey { get; set; } = string.Empty;

    /// <summary>Relay tunnel IP within the mesh (e.g. "10.20.0.1").</summary>
    public string TunnelIp { get; set; } = string.Empty;

    /// <summary>Subnet allocated to CGNAT nodes routed through this relay (e.g. "10.20.1.0/24").</summary>
    public string RelaySubnet { get; set; } = string.Empty;

    /// <summary>Bearer token used by CGNAT nodes to authenticate against the relay API. NEVER log.</summary>
    public string AuthToken { get; set; } = string.Empty;
}

/// <summary>
/// Identity state for the DHT system VM role.
///
/// The Ed25519 keypair is the libp2p peer identity. If the key changes, the node
/// gets a new peer ID and all routing table entries across the DHT network must
/// update — causing cascading reconnections. Persisting the key here eliminates
/// that disruption on redeployment.
/// </summary>
public class DhtObligationState : ObligationStateBase
{
    /// <summary>Ed25519 private key as a base64-encoded byte array. NEVER log this value.</summary>
    public string Ed25519PrivateKeyBase64 { get; set; } = string.Empty;

    /// <summary>libp2p peer ID derived from the Ed25519 public key. Safe to log.</summary>
    public string PeerId { get; set; } = string.Empty;

    /// <summary>WireGuard private key for mesh connectivity. NEVER log this value.</summary>
    public string WireGuardPrivateKey { get; set; } = string.Empty;

    /// <summary>WireGuard public key derived from the private key. Safe to log.</summary>
    public string WireGuardPublicKey { get; set; } = string.Empty;

    /// <summary>WireGuard tunnel IP within the mesh (e.g. "10.30.0.x").</summary>
    public string TunnelIp { get; set; } = string.Empty;

    /// <summary>Bearer token used to authenticate DHT API calls from the orchestrator. NEVER log.</summary>
    public string AuthToken { get; set; } = string.Empty;
}

/// <summary>
/// Identity state for the BlockStore system VM role.
///
/// Like DHT, the Ed25519 keypair is the libp2p peer identity used in Kademlia routing.
/// The storage quota is included here so the VM can self-limit without an extra
/// orchestrator call at boot time.
/// </summary>
public class BlockStoreObligationState : ObligationStateBase
{
    /// <summary>Ed25519 private key as a base64-encoded byte array. NEVER log this value.</summary>
    public string Ed25519PrivateKeyBase64 { get; set; } = string.Empty;

    /// <summary>libp2p peer ID derived from the Ed25519 public key. Safe to log.</summary>
    public string PeerId { get; set; } = string.Empty;

    /// <summary>WireGuard private key for mesh connectivity. NEVER log this value.</summary>
    public string WireGuardPrivateKey { get; set; } = string.Empty;

    /// <summary>WireGuard public key derived from the private key. Safe to log.</summary>
    public string WireGuardPublicKey { get; set; } = string.Empty;

    /// <summary>WireGuard tunnel IP within the mesh (e.g. "10.40.0.x").</summary>
    public string TunnelIp { get; set; } = string.Empty;

    /// <summary>Bearer token used to authenticate block store API calls. NEVER log.</summary>
    public string AuthToken { get; set; } = string.Empty;

    /// <summary>
    /// Storage quota in bytes this node's block store may consume.
    /// Typically 5 % of total node storage, calculated by the orchestrator at registration.
    /// </summary>
    public long StorageQuotaBytes { get; set; }
}

/// <summary>
/// Known obligation role names. Used as the primary key in the obligation_state table
/// and as path segments in the ObligationStateController route.
/// </summary>
public static class ObligationRole
{
    public const string Relay      = "relay";
    public const string Dht        = "dht";
    public const string BlockStore = "blockstore";

    private static readonly HashSet<string> _valid =
        new(StringComparer.OrdinalIgnoreCase) { Relay, Dht, BlockStore };

    /// <summary>Returns true if <paramref name="role"/> is one of the three known roles.</summary>
    public static bool IsValid(string? role) =>
        role is not null && _valid.Contains(role);

    /// <summary>Returns the canonical lower-case role name, or null if unrecognised.</summary>
    public static string? Canonicalise(string? role) =>
        IsValid(role) ? role!.ToLowerInvariant() : null;
}
