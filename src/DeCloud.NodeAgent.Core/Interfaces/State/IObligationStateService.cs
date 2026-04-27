using DeCloud.NodeAgent.Core.Models.State;

namespace DeCloud.NodeAgent.Core.Interfaces.State;

/// <summary>
/// Persists system VM identity state and the obligation list on the node
/// agent so they survive VM crashes, agent restarts, and orchestrator
/// outages. The orchestrator is the authoritative source for both; this
/// service stores the authoritative local copy in SQLite.
///
/// Two concerns, one service:
///   • Identity state — per-role JSON blob containing private keys,
///     versioned with monotonic conflict resolution. Served to system VMs
///     over virbr0 at boot time so peer IDs survive redeployment.
///   • Obligations — the set of roles this node must run, plus each role's
///     dependency list. Read by the matrix's intent projection. Replaced
///     wholesale on each orchestrator push (no per-role updates).
///
/// SECURITY: Identity state contains private keys (Ed25519, WireGuard).
/// Implementations must never log the full state JSON — only role and
/// version numbers.
///
/// CONFLICT RESOLUTION (identity): Higher version always wins. A write with
/// a version equal to or lower than the stored version is silently ignored
/// (idempotent).
///
/// CONFLICT RESOLUTION (obligations): Whole-set replacement. The orchestrator
/// pushes the full list; the node atomically wipes and re-inserts. This
/// avoids drift from missed deletes.
/// </summary>
public interface IObligationStateService
{
    // ════════════════════════════════════════════════════════════════════
    // Identity
    // ════════════════════════════════════════════════════════════════════

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

    // ════════════════════════════════════════════════════════════════════
    // Obligations
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Persist the full obligation list for this node, replacing any prior
    /// content atomically. Called by the orchestrator-response handler when
    /// it receives an updated set of obligations (registration response, or
    /// future re-evaluation push).
    ///
    /// Whole-set semantics: any obligation present in the prior list but
    /// absent in <paramref name="obligations"/> is removed. The orchestrator
    /// is the source of truth — the node mirrors what it is told.
    ///
    /// Each entry is canonicalised before write; entries with unknown role
    /// names are silently dropped (logged at Warning).
    /// </summary>
    /// <param name="obligations">
    /// The complete obligation set. May be empty (node has no obligations
    /// — e.g., a node whose hardware no longer qualifies for any role).
    /// </param>
    /// <param name="ct">Cancellation token.</param>
    Task SaveObligationsAsync(
        IReadOnlyList<ObligationDescriptor> obligations,
        CancellationToken ct = default);

    /// <summary>
    /// Retrieve the full obligation list. Returns an empty list (never
    /// <c>null</c>) if no obligations have been persisted yet.
    ///
    /// Read every cycle by the reconciliation matrix; intended to be cheap
    /// — backed by a single SQLite SELECT.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    Task<IReadOnlyList<ObligationDescriptor>> GetObligationsAsync(
        CancellationToken ct = default);
}