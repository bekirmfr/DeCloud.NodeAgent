using DeCloud.NodeAgent.Core.Models.SystemVm;

namespace DeCloud.NodeAgent.Core.Interfaces.SystemVm;

/// <summary>
/// In-memory map of system-VM lifecycle commands the node has issued but not yet
/// observed completing. Keyed by canonical role name ("relay" | "dht" | "blockstore").
///
/// The node-side reconciliation matrix (introduced in P5) reads this collection
/// every cycle as its <c>pending</c> input axis — if a role has an outstanding
/// command, the matrix waits rather than issuing again.
///
/// This is *the* single source of truth for "is anything in flight for this role
/// right now?" — replacing the previous orchestrator-side cluster of signals
/// (<c>obligation.Status == Deploying</c>, <c>vm.ActiveCommandId</c>,
/// <c>PendingCommandAcks</c>, deploy-cooldown windows). Those legacy signals are
/// deleted in P7 once the matrix is authoritative.
///
/// IMPLEMENTATION NOTES
///   • In-memory only. Outstanding commands from a previous node-agent process
///     are inherently expired — the process that issued them is gone with the
///     ack handler that would clear them.
///   • Thread-safe. Multiple background services read and write concurrently:
///     the reconciler issues, the command-ack handler clears, the sweeper expires.
///   • Idempotent on Set: re-issuing for a role overwrites the previous entry.
///     Callers should normally not Set without first checking TryGet, but the
///     overwrite semantics make a race-condition double-set harmless.
/// </summary>
public interface IOutstandingCommands
{
    /// <summary>
    /// Look up the outstanding command for <paramref name="role"/>, if any.
    /// </summary>
    /// <param name="role">Canonical role name (case-insensitive — canonicalised internally).</param>
    /// <param name="command">The outstanding command, if one is recorded.</param>
    /// <returns>
    /// <c>true</c> if a command is recorded for this role; <c>false</c> otherwise.
    /// Returns <c>false</c> for unknown role names rather than throwing —
    /// the matrix treats "no entry" and "unknown role" identically.
    /// </returns>
    bool TryGet(string role, out OutstandingCommand command);

    /// <summary>
    /// Record a newly issued command for <paramref name="role"/>. Overwrites
    /// any previous entry for the same role.
    /// </summary>
    /// <param name="role">Canonical role name. Unknown roles are silently dropped.</param>
    /// <param name="command">The command details — must include CommandId, Kind, IssuedAt.</param>
    void Set(string role, OutstandingCommand command);

    /// <summary>
    /// Remove the outstanding command for <paramref name="role"/>, if any.
    /// Called by the command-ack handler when a Create or Delete completes
    /// (whether successfully or with failure — both clear the outstanding entry).
    /// </summary>
    /// <param name="role">Canonical role name. No-op for unknown roles.</param>
    /// <returns>
    /// <c>true</c> if an entry was removed; <c>false</c> if there was none.
    /// </returns>
    bool Clear(string role);

    /// <summary>
    /// Remove every outstanding command whose age (now − IssuedAt) is greater
    /// than <paramref name="timeout"/>. Called by the reconciler at the end of
    /// each cycle to prevent permanently-stuck entries from blocking decisions.
    ///
    /// Sweeping does not retry the underlying command — it merely declares the
    /// in-flight state forgotten. The next cycle re-evaluates from current
    /// reality and may issue a fresh command if still appropriate.
    /// </summary>
    /// <param name="timeout">Maximum age before a command is considered expired.</param>
    /// <returns>The number of entries removed.</returns>
    int SweepExpired(TimeSpan timeout);

    /// <summary>
    /// Snapshot of every currently outstanding command, keyed by canonical role.
    /// Intended for diagnostics, dashboards, and tests — not for decision logic.
    /// The matrix uses <see cref="TryGet"/> per role; only observability code
    /// should enumerate.
    /// </summary>
    /// <returns>A point-in-time copy. Mutating the returned dictionary does not
    /// affect the underlying state.</returns>
    IReadOnlyDictionary<string, OutstandingCommand> Snapshot();
}