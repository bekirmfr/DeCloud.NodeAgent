namespace DeCloud.NodeAgent.Core.Models.State;

/// <summary>
/// A single obligation as the orchestrator declares it: which role this node
/// must run, and the role's dependency list.
///
/// Stored in the node's local SQLite <c>obligation</c> table, written by the
/// orchestrator at registration and on capability re-evaluation, read by the
/// matrix's <c>intent</c> projection every cycle.
///
/// The dependency list is pushed by the orchestrator alongside the role —
/// not hardcoded on the node — so the dependency graph can evolve without
/// shipping a node-agent update. The orchestrator's
/// <c>SystemVmDependencies</c> remains the source of truth; nodes consume
/// what they're told.
///
/// Keys are canonical role names: "relay" | "dht" | "blockstore" (lower-case).
/// Storage and lookup go through <c>ObligationRole.Canonicalise</c> to prevent
/// open-ended queries.
/// </summary>
public sealed record ObligationDescriptor
{
    /// <summary>
    /// Canonical role name. Primary key in the <c>obligation</c> table.
    /// </summary>
    public required string Role { get; init; }

    /// <summary>
    /// Dependency role names this obligation depends on, in declaration
    /// order. The intent computation enforces only those entries that
    /// also have an obligation in the local set ("applicable
    /// dependencies"); cluster-wide deps that don't apply to this node
    /// are skipped.
    ///
    /// Empty list = no dependencies (e.g., the Relay role).
    /// </summary>
    public required IReadOnlyList<string> Deps { get; init; }

    /// <summary>
    /// When this descriptor was last updated by the orchestrator push.
    /// Diagnostic only — the matrix does not consume this field.
    /// </summary>
    public DateTime UpdatedAt { get; init; } = DateTime.UtcNow;
}