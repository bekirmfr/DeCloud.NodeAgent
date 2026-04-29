using DeCloud.NodeAgent.Core.Models.State;
using DeCloud.Shared.Models;

namespace DeCloud.NodeAgent.Core.Interfaces.State;

/// <summary>
/// Persists system VM identity state, obligations, and system templates on
/// the node agent so they survive VM crashes, agent restarts, and orchestrator
/// outages. The orchestrator is the authoritative source for all three;
/// this service stores the authoritative local copy in SQLite.
///
/// Three concerns, one service (one SQLite file, three table sections):
///   • Identity state  — per-role JSON blob containing private keys,
///     versioned with monotonic conflict resolution. Served to system VMs
///     over virbr0 at boot time so peer IDs survive redeployment.
///   • Obligations     — the set of roles this node must run, plus each
///     role's dependency list. Read by the matrix's intent projection.
///     Replaced wholesale on each orchestrator push.
///   • System templates — lightweight deployment specs (cloud-init content,
///     artifact refs, resource spec, service declarations) for each role.
///     The <c>SystemVmReconciler</c> reads these to self-create system VMs
///     without contacting the orchestrator (P10).
///
/// SECURITY: Identity state contains private keys (Ed25519, WireGuard).
/// Implementations must never log the full state JSON — only role and
/// version numbers. Same discipline applies to template JSON (it may
/// contain auth tokens baked into cloud-init).
///
/// CONFLICT RESOLUTION
///   Identity:    Higher version wins. Equal or lower silently ignored.
///   Obligations: Whole-set replacement.
///   Templates:   Higher revision wins. Equal or lower silently ignored.
/// </summary>
public interface IObligationStateService
{
    // ════════════════════════════════════════════════════════════════════
    // Identity
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Persist obligation identity state received from the orchestrator.
    /// Only writes if <paramref name="incomingVersion"/> is strictly greater
    /// than the currently stored version. Idempotent.
    /// </summary>
    Task<bool> SaveStateAsync(
        string role,
        string stateJson,
        int incomingVersion,
        CancellationToken ct = default);

    /// <summary>Retrieve the raw identity state JSON for a role, or <c>null</c>.</summary>
    Task<string?> GetStateJsonAsync(string role, CancellationToken ct = default);

    /// <summary>Get the current stored version for a role (0 if absent).</summary>
    Task<int> GetVersionAsync(string role, CancellationToken ct = default);

    /// <summary>Delete identity state for a role (obligation permanently removed).</summary>
    Task DeleteStateAsync(string role, CancellationToken ct = default);

    // ════════════════════════════════════════════════════════════════════
    // Obligations
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Persist the full obligation list for this node, replacing any prior
    /// content atomically. Whole-set semantics. Unknown role names dropped
    /// with Warning log.
    /// </summary>
    Task SaveObligationsAsync(
        IReadOnlyList<ObligationDescriptor> obligations,
        CancellationToken ct = default);

    /// <summary>
    /// Retrieve the full obligation list.
    /// Returns empty list (never <c>null</c>) if none persisted yet.
    /// </summary>
    Task<IReadOnlyList<ObligationDescriptor>> GetObligationsAsync(
        CancellationToken ct = default);

    // ════════════════════════════════════════════════════════════════════
    // System templates
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Persist a system VM deployment template received from the orchestrator.
    /// Only writes if <paramref name="incomingRevision"/> is strictly greater
    /// than the currently stored revision. Idempotent.
    ///
    /// After a successful write, the caller should trigger
    /// <c>IArtifactCacheService.PrefetchAsync</c> for the template's artifacts
    /// so they are cached before the first Create dispatch.
    /// </summary>
    /// <param name="role">Canonical role name.</param>
    /// <param name="templateJson">JSON-serialised <see cref="SystemVmTemplate"/>.</param>
    /// <param name="incomingRevision">Revision number from the orchestrator payload.</param>
    /// <returns>
    /// <c>true</c> if written (incoming revision was higher);
    /// <c>false</c> if ignored (equal or lower revision already stored).
    /// </returns>
    Task<bool> SaveSystemTemplateAsync(
        string role,
        string templateJson,
        int incomingRevision,
        string? templateId = null,
        CancellationToken ct = default);

    /// <summary>
    /// Retrieve the raw system template JSON for a role.
    /// Returns <c>null</c> if no template has been received yet (node not
    /// ready to self-deploy this role).
    /// </summary>
    Task<string?> GetSystemTemplateJsonAsync(string role, CancellationToken ct = default);

    /// <summary>
    /// Get the current stored revision for a role's system template (0 if absent).
    /// Included in every heartbeat so the orchestrator can detect and push updates.
    /// </summary>
    Task<int> GetSystemTemplateRevisionAsync(string role, CancellationToken ct = default);

    /// <summary>
    /// Return a <c>{ role → revision }</c> map for all canonical roles.
    /// Roles with no stored template have value 0. Used when building the
    /// heartbeat's <c>systemTemplateVersions</c> field — parallel to the
    /// existing <c>BuildObligationStateVersionsAsync</c> helper.
    /// </summary>
    Task<Dictionary<string, int>> GetSystemTemplateRevisionsAsync(
        CancellationToken ct = default);
}