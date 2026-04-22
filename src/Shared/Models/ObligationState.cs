namespace DeCloud.Shared.Models;

// ============================================================
// Placement: src/Shared/Models/ObligationState.cs
//
// Replaces: src/DeCloud.NodeAgent.Core/Models/ObligationState.cs
//
// Both projects already include src/Shared/**/*.cs via:
//   <Compile Include="../Shared/**/*.cs" ... />
// so no .csproj changes are needed.
//
// Delete the NodeAgent.Core version after placing this file.
// Update any using directives from DeCloud.NodeAgent.Core.Models
// to DeCloud.Shared.Models in ObligationStateService.cs,
// ObligationStateController.cs, and IObligationStateService.cs.
// ============================================================

/// <summary>
/// Base class for all obligation identity state.
/// Stored in SQLite obligation_state table as a JSON blob (state_json column).
///
/// SECURITY: State may contain private keys. Never log the full JSON blob —
/// only role and version numbers should appear in application logs.
/// </summary>
public abstract class ObligationStateBase
{
    /// <summary>
    /// Monotonic version assigned by the orchestrator.
    /// Conflict resolution: higher version always wins.
    /// </summary>
    public int Version { get; set; }

    /// <summary>UTC timestamp of the last orchestrator write.</summary>
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Identity state for the Relay system VM role.
/// The WireGuard keypair is the mesh identity anchor — preserving it across
/// redeploys lets all mesh peers reconnect without reconfiguration.
/// </summary>
public class RelayObligationState : ObligationStateBase
{
    /// <summary>WireGuard private key (base64). NEVER log.</summary>
    public string WireGuardPrivateKey { get; set; } = string.Empty;

    /// <summary>WireGuard public key derived from the private key. Safe to log.</summary>
    public string WireGuardPublicKey { get; set; } = string.Empty;

    /// <summary>Relay tunnel IP within the mesh (e.g. "10.20.0.1").</summary>
    public string TunnelIp { get; set; } = string.Empty;

    /// <summary>Subnet allocated to CGNAT nodes routed through this relay (e.g. "10.20.1.0/24").</summary>
    public string RelaySubnet { get; set; } = string.Empty;

    /// <summary>Bearer token for CGNAT node authentication against the relay API. NEVER log.</summary>
    public string AuthToken { get; set; } = string.Empty;
}

/// <summary>
/// Identity state for the DHT system VM role.
/// The Ed25519 keypair is the libp2p peer identity. Preserving it across
/// redeploys eliminates cascading DHT routing table updates.
/// </summary>
public class DhtObligationState : ObligationStateBase
{
    /// <summary>Ed25519 private key as base64-encoded bytes. NEVER log.</summary>
    public string Ed25519PrivateKeyBase64 { get; set; } = string.Empty;

    /// <summary>libp2p peer ID derived from the Ed25519 public key. Safe to log.</summary>
    public string PeerId { get; set; } = string.Empty;

    /// <summary>WireGuard private key for mesh connectivity. NEVER log.</summary>
    public string WireGuardPrivateKey { get; set; } = string.Empty;

    /// <summary>WireGuard public key derived from the private key. Safe to log.</summary>
    public string WireGuardPublicKey { get; set; } = string.Empty;

    /// <summary>WireGuard tunnel IP within the mesh (e.g. "10.30.0.x").</summary>
    public string TunnelIp { get; set; } = string.Empty;

    /// <summary>Bearer token for DHT API authentication. NEVER log.</summary>
    public string AuthToken { get; set; } = string.Empty;
}

/// <summary>
/// Identity state for the BlockStore system VM role.
/// Like DHT, the Ed25519 keypair is the libp2p peer identity used in Kademlia routing.
/// </summary>
public class BlockStoreObligationState : ObligationStateBase
{
    /// <summary>Ed25519 private key as base64-encoded bytes. NEVER log.</summary>
    public string Ed25519PrivateKeyBase64 { get; set; } = string.Empty;

    /// <summary>libp2p peer ID derived from the Ed25519 public key. Safe to log.</summary>
    public string PeerId { get; set; } = string.Empty;

    /// <summary>WireGuard private key for mesh connectivity. NEVER log.</summary>
    public string WireGuardPrivateKey { get; set; } = string.Empty;

    /// <summary>WireGuard public key derived from the private key. Safe to log.</summary>
    public string WireGuardPublicKey { get; set; } = string.Empty;

    /// <summary>WireGuard tunnel IP within the mesh (e.g. "10.40.0.x").</summary>
    public string TunnelIp { get; set; } = string.Empty;

    /// <summary>Bearer token for block store API authentication. NEVER log.</summary>
    public string AuthToken { get; set; } = string.Empty;

    /// <summary>
    /// Storage quota in bytes this node's block store may consume.
    /// Typically 5% of total node storage, calculated at registration time.
    /// </summary>
    public long StorageQuotaBytes { get; set; }
}

/// <summary>
/// Canonical role name constants and validation helpers.
/// These strings are the primary key in the SQLite obligation_state table
/// and the route segment in ObligationStateController.
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

    /// <summary>
    /// Returns the canonical lower-case role name, or null if unrecognised.
    /// Use this before any DB lookup to prevent open-ended queries.
    /// </summary>
    public static string? Canonicalise(string? role) =>
        IsValid(role) ? role!.ToLowerInvariant() : null;
}
