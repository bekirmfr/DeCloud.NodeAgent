namespace DeCloud.NodeAgent.Core.Models.Reality;

/// <summary>
/// Coarse-grained state of a system VM role on this node, as observed by
/// <see cref="DeCloud.NodeAgent.Core.Interfaces.State.IRealityProjection"/>.
///
/// This is the matrix's <c>reality</c> axis (see SYSTEM_VM_DESIGN.md §5.2).
/// It deliberately collapses the underlying <see cref="VmState"/> into three
/// values — finer granularity belongs inside the VM (per-service readiness,
/// mesh health endpoints), not in the matrix.
/// </summary>
public enum Reality
{
    /// <summary>
    /// No VM of this role is tracked locally. Either the role has never been
    /// deployed on this node, or any prior VM has been fully removed
    /// (state <see cref="VmState.Deleted"/> or <see cref="VmState.NotFound"/>
    /// — those are filtered out by the projection).
    /// </summary>
    None,

    /// <summary>
    /// A VM exists for this role, is in <see cref="VmState.Running"/>, and
    /// every entry in its <c>Services</c> list is <c>Ready</c>. This is the
    /// converged state — the matrix takes no action when intent agrees.
    /// </summary>
    Healthy,

    /// <summary>
    /// A VM exists for this role but is not Healthy. Covers any of:
    /// <list type="bullet">
    ///   <item>Running but at least one service has not reached Ready
    ///         (cloud-init still running, mesh peer count below threshold,
    ///         binary version mismatch, …).</item>
    ///   <item>Transitional states: Pending, Creating, Starting, Stopping,
    ///         Migrating.</item>
    ///   <item>Terminal-but-not-gone: Stopped, Failed, Paused.</item>
    /// </list>
    /// The matrix issues a Delete for the offending VM. Once the Delete
    /// completes, reality becomes <see cref="None"/> and the next cycle
    /// issues a Create (assuming intent still says yes).
    /// </summary>
    Unhealthy
}

/// <summary>
/// A point-in-time projection of a single role's reality on this node.
///
/// Returned by <see cref="DeCloud.NodeAgent.Core.Interfaces.State.IRealityProjection.Project"/>.
/// Consumed by the reconciliation matrix (P5) as one of its three input axes.
/// </summary>
public sealed record RealitySnapshot
{
    /// <summary>
    /// The coarse-grained reality classification. The matrix branches on this.
    /// </summary>
    public required Reality State { get; init; }

    /// <summary>
    /// VM identifier of the VM this snapshot is about.
    /// <c>null</c> when <see cref="State"/> is <see cref="Reality.None"/>.
    /// Required when <see cref="State"/> is <see cref="Reality.Healthy"/> or
    /// <see cref="Reality.Unhealthy"/> — the matrix uses it as the target of
    /// any Delete it dispatches.
    /// </summary>
    public string? VmId { get; init; }

    /// <summary>
    /// Underlying VM state for diagnostics and logging.
    /// <c>null</c> when <see cref="State"/> is <see cref="Reality.None"/>.
    /// Not consumed by decision logic — only by log messages and dashboard
    /// output. The matrix decides solely on <see cref="State"/>.
    /// </summary>
    public VmState? VmState { get; init; }

    /// <summary>
    /// Convenience: a snapshot with no VM tracked.
    /// </summary>
    public static RealitySnapshot None { get; } = new() { State = Reality.None };
}