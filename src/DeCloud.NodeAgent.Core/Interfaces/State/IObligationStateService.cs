namespace DeCloud.NodeAgent.Core.Interfaces.State;

// ============================================================
// Placement: src/DeCloud.NodeAgent.Core/Interfaces/IObligationStateService.cs
// ============================================================

/// <summary>
/// Persists system VM identity state on the node agent so it survives VM crashes
/// and redeployments. The orchestrator is the authoritative source; this service
/// stores the authoritative local copy in SQLite.
///
/// SECURITY: State contains private keys (Ed25519, WireGuard). Implementations
/// must never log the full state JSON — only role and version numbers.
///
/// CONFLICT RESOLUTION: Higher version always wins. A write with a version equal
/// to or lower than the stored version is silently ignored (idempotent).
/// </summary>
public interface IObligationStateService
{
    /// <summary>
    /// Persist obligation identity state received from the orchestrator.
    ///
    /// Only writes if <paramref name="incomingVersion"/> is strictly greater than
    /// the currently stored version. Equal or lower versions are silently ignored —
    /// this makes the operation fully idempotent.
    /// </summary>
    /// <param name="role">Canonical role name ("relay" | "dht" | "blockstore").</param>
    /// <param name="stateJson">JSON-serialised identity state blob.</param>
    /// <param name="incomingVersion">Version assigned by the orchestrator.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>
    /// <c>true</c> if the state was written (incoming version was higher);
    /// <c>false</c> if it was ignored (equal or lower version already stored).
    /// </returns>
    Task<bool> SaveStateAsync(
        string role,
        string stateJson,
        int incomingVersion,
        CancellationToken ct = default);

    /// <summary>
    /// Retrieve the raw identity state JSON for a role.
    /// Returns <c>null</c> if no state exists for this role (first boot, state cleared,
    /// or role not assigned to this node).
    /// </summary>
    /// <param name="role">Canonical role name.</param>
    /// <param name="ct">Cancellation token.</param>
    Task<string?> GetStateJsonAsync(string role, CancellationToken ct = default);

    /// <summary>
    /// Get the current stored version for a role.
    /// Returns <c>0</c> if no state exists — allows the orchestrator to detect
    /// that a full state push is required on re-registration.
    /// </summary>
    /// <param name="role">Canonical role name.</param>
    /// <param name="ct">Cancellation token.</param>
    Task<int> GetVersionAsync(string role, CancellationToken ct = default);

    /// <summary>
    /// Delete identity state for a role.
    /// Called when an obligation is permanently removed from this node.
    /// </summary>
    /// <param name="role">Canonical role name.</param>
    /// <param name="ct">Cancellation token.</param>
    Task DeleteStateAsync(string role, CancellationToken ct = default);
}
