namespace DeCloud.NodeAgent.Core.Models.State;

/// <summary>
/// The matrix's <c>intent</c> axis (SYSTEM_VM_DESIGN.md §5.2): a pure read of
/// local SQLite expressing whether this node should be running a system VM
/// for a given role, and whether its dependencies are currently satisfied.
///
/// The two fields are split rather than collapsed into a single
/// <c>ShouldDeploy</c> boolean because the reconciler logs them distinctly:
///
///   • <c>WantDeployed = false</c>           → "no obligation for this role"
///   • <c>WantDeployed = true, DepsMet=false</c> → "waiting on dependency"
///   • <c>WantDeployed = true, DepsMet=true</c>  → "go"
///
/// The <see cref="Interfaces.SystemVm.IIntentComputation"/>
/// implementation is a pure function — call it any number of times per cycle
/// without side-effects.
/// </summary>
public sealed record Intent
{
    /// <summary>
    /// True iff this node currently holds an obligation for the role.
    /// Computed by checking the local <c>obligation</c> SQLite table for a
    /// row keyed by the canonical role name.
    /// </summary>
    public required bool WantDeployed { get; init; }

    /// <summary>
    /// True iff every applicable dependency for this role is presently
    /// <see cref="Reality.Healthy"/> on this node. A dependency is
    /// "applicable" when this node also has an obligation for it — the node
    /// only enforces dependencies it is itself responsible for. This means
    /// (e.g.) a CGNAT-only node with no Relay obligation does not block its
    /// DHT obligation on a non-existent local Relay.
    ///
    /// Vacuously <c>true</c> when <see cref="WantDeployed"/> is <c>false</c>
    /// (no point gating dependencies for a role we don't run).
    /// </summary>
    public required bool DepsMet { get; init; }

    /// <summary>
    /// Pre-canned "no obligation" intent, returned when the role isn't in
    /// the local obligation table. Useful for the matrix's iteration over
    /// the canonical role set.
    /// </summary>
    public static Intent None { get; } = new() { WantDeployed = false, DepsMet = true };
}