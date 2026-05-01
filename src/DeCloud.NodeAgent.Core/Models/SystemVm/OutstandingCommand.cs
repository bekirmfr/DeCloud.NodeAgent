using DeCloud.NodeAgent.Core.Interfaces.SystemVm;

namespace DeCloud.NodeAgent.Core.Models.SystemVm;

/// <summary>
/// Kind of system-VM lifecycle command currently outstanding on the node.
/// Mirrors the two operations the node-side reconciliation matrix can dispatch.
/// </summary>
public enum OutstandingCommandKind
{
    Create,
    Delete
}

/// <summary>
/// A single command issued by the node-side <c>SystemVmReconciler</c> against
/// the local command pipeline that has not yet completed.
///
/// The reconciliation matrix consults the outstanding-commands map every cycle
/// to decide whether to act or wait — the presence of an entry for a role
/// means "we already issued something for this role; don't issue again until it
/// completes or expires."
///
/// Records are written when a command is enqueued, cleared when the command
/// acks (success or failure), and swept when their age exceeds the global
/// <c>CommandTimeout</c>. They live in process memory only — outstanding
/// commands from a previous node-agent process are inherently expired (the
/// process that issued them is gone).
/// </summary>
public sealed record OutstandingCommand
{
    /// <summary>
    /// Unique identifier of the command in the local command pipeline.
    /// Returned by the queue when the command is enqueued and matched by
    /// the ack handler when the command completes.
    /// </summary>
    public required string CommandId { get; init; }

    /// <summary>
    /// What kind of operation is in flight: Create or Delete.
    /// </summary>
    public required OutstandingCommandKind Kind { get; init; }

    /// <summary>
    /// The VM identifier this command targets. Always present for Delete
    /// (the VM exists). Absent for Create until the orchestrator-side
    /// VM record is materialised — for the node-side path this is set to
    /// the node-issued VM id at enqueue time. Null is permitted only as a
    /// transient state while the matrix is wiring the request.
    /// </summary>
    public string? VmId { get; init; }

    /// <summary>
    /// Wall-clock time (UTC) at which the matrix issued this command.
    /// Used by <see cref="IOutstandingCommands.SweepExpired"/>
    /// to detect expiry against the global CommandTimeout.
    /// </summary>
    public required DateTime IssuedAt { get; init; }

    /// <summary>
    /// Revision of the system template this Create was issued against.
    /// Used to detect template drift while a Create is in flight.
    /// When the stored template revision exceeds this value and the VM
    /// has appeared (reality=Unhealthy), the Create is considered stale
    /// and is cleared immediately rather than waiting for the 20-minute sweep.
    /// Zero for Delete commands (revision is irrelevant).
    /// </summary>
    public int TemplateRevision { get; init; }
}