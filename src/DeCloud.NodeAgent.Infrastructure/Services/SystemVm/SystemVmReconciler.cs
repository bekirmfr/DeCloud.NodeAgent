using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Interfaces.SystemVm;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Models.Reality;
using DeCloud.NodeAgent.Core.Models.State;
using DeCloud.NodeAgent.Core.Models.SystemVm;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services.SystemVm;

// ── Supporting types ────────────────────────────────────────────────────────

/// <summary>
/// What the reconciliation matrix decided to do for a single role in this cycle.
/// </summary>
public enum MatrixAction
{
    /// <summary>No action required — system is converged or a command is already in flight.</summary>
    Wait,

    /// <summary>A system VM should be created for this role.</summary>
    IssueCreate,

    /// <summary>The existing VM for this role should be deleted (prerequisite to redeploy or cleanup).</summary>
    IssueDelete,
}

/// <summary>
/// The outcome of one matrix cell evaluation: what to do and a human-readable
/// explanation of why. Used for shadow-mode logging and, in P6, for structured
/// audit trails.
/// </summary>
public sealed record MatrixDecision(MatrixAction Action, string Reason);

// ── Reconciler ──────────────────────────────────────────────────────────────

/// <summary>
/// Node-side system VM reconciliation loop (SYSTEM_VM_DESIGN.md §5).
///
/// Every <see cref="ReconcileInterval"/> this service evaluates the three-axis
/// reconciliation matrix for each of the three canonical system VM roles
/// (relay, dht, blockstore):
///
/// <code>
///   (intent, reality, pending) → MatrixAction
/// </code>
///
/// <b>P5 — shadow mode.</b> The matrix runs fully but <b>does not act</b>.
/// Decisions are logged at <c>Information</c> level with a <c>[SHADOW]</c>
/// prefix so operators can verify they match expectations before P6 cuts over.
/// Steady-state <c>Wait</c> decisions are logged at <c>Debug</c> to avoid noise.
///
/// The week-long soak in shadow mode is the safety gate before P6. Every
/// WARNING-level discrepancy during the soak should be triaged and explained.
/// Once all decisions match orchestrator behaviour and no unexpected events
/// appear, P6 is safe to ship.
///
/// <b>P6 upgrade path.</b> Replace the shadow-mode logging block in
/// <see cref="RunCycleAsync"/> with actual command issuance. The <see cref="Decide"/>
/// method and the <see cref="MatrixDecision"/> model stay unchanged — only the
/// consumer of the decision changes.
/// </summary>
public sealed class SystemVmReconciler : BackgroundService
{
    // ── Constants ───────────────────────────────────────────────────────

    /// <summary>How often the matrix runs. 30s matches the heartbeat interval.</summary>
    private static readonly TimeSpan ReconcileInterval = TimeSpan.FromSeconds(30);

    /// <summary>
    /// How long before an outstanding command is declared timed-out and swept.
    /// 20 minutes covers cold-cache cloud-init (artifact download + binary extraction).
    /// </summary>
    private static readonly TimeSpan CommandTimeout = TimeSpan.FromMinutes(20);

    /// <summary>
    /// Startup delay before the first cycle. Gives <c>VmManagerInitializationService</c>
    /// time to populate the in-memory VM list from libvirt before we project reality.
    /// </summary>
    private static readonly TimeSpan StartupDelay = TimeSpan.FromSeconds(10);

    /// <summary>
    /// The complete set of roles this reconciler is responsible for.
    /// Iterating this set (rather than the obligation list alone) ensures the matrix
    /// also evaluates <c>intent=no</c> cells — catching stray VMs for roles the node
    /// is no longer obligated to run.
    /// </summary>
    private static readonly IReadOnlyList<string> CanonicalRoles =
        [ObligationRole.Relay, ObligationRole.Dht, ObligationRole.BlockStore];

    // ── Dependencies ────────────────────────────────────────────────────

    private readonly IObligationStateService _obligationState;
    private readonly IIntentComputation _intent;
    private readonly IRealityProjection _reality;
    private readonly IOutstandingCommands _outstanding;
    private readonly ILogger<SystemVmReconciler> _logger;

    // ── Constructor ─────────────────────────────────────────────────────

    public SystemVmReconciler(
        IObligationStateService obligationState,
        IIntentComputation intent,
        IRealityProjection reality,
        IOutstandingCommands outstanding,
        ILogger<SystemVmReconciler> logger)
    {
        _obligationState = obligationState;
        _intent = intent;
        _reality = reality;
        _outstanding = outstanding;
        _logger = logger;
    }

    // ── BackgroundService ───────────────────────────────────────────────

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        _logger.LogInformation(
            "SystemVmReconciler starting in SHADOW mode — decisions will be logged " +
            "but not acted upon. Shadow soak target: ≥1 week with no unexpected " +
            "WARNING-level discrepancies before P6 cutover.");

        // Short delay so the VM manager can populate its in-memory list from libvirt.
        await Task.Delay(StartupDelay, ct);

        while (!ct.IsCancellationRequested)
        {
            try
            {
                await RunCycleAsync(ct);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "SystemVmReconciler: unhandled exception in cycle — will retry");
            }

            await Task.Delay(ReconcileInterval, ct);
        }

        _logger.LogInformation("SystemVmReconciler stopped");
    }

    // ── Cycle ────────────────────────────────────────────────────────────

    /// <summary>
    /// One full evaluation of the matrix across all canonical roles.
    /// Shadow mode: logs decisions, does not act.
    /// </summary>
    private async Task RunCycleAsync(CancellationToken ct)
    {
        // Fetch the obligation list once per cycle — all role evaluations
        // share the same snapshot so intent decisions are consistent.
        var obligations = await _obligationState.GetObligationsAsync(ct);

        foreach (var role in CanonicalRoles)
        {
            try
            {
                EvaluateRole(role, obligations);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "SystemVmReconciler: exception evaluating role '{Role}' — skipping this role this cycle",
                    role);
            }
        }

        // Sweep commands that have been outstanding longer than CommandTimeout.
        // These are not failures per se — the ack may have been missed after a
        // restart. Sweeping returns the slot to None so the next cycle re-evaluates
        // from current reality. The libvirt duplicate-name guard makes re-issue safe.
        var swept = _outstanding.SweepExpired(CommandTimeout);
        if (swept > 0)
        {
            _logger.LogWarning(
                "SystemVmReconciler: swept {Count} timed-out outstanding commands — " +
                "next cycle will re-evaluate those roles from current reality",
                swept);
        }
    }

    // ── Per-role evaluation ──────────────────────────────────────────────

    private void EvaluateRole(string role, IReadOnlyList<ObligationDescriptor> obligations)
    {
        // ── Axis 1: Intent ──────────────────────────────────────────────
        var intent = _intent.Compute(role, obligations);

        // ── Axis 2: Reality ─────────────────────────────────────────────
        var reality = _reality.Project(role);

        // ── Axis 3: Pending ─────────────────────────────────────────────
        var hasPending = _outstanding.TryGet(role, out var pending);
        OutstandingCommand? pendingOrNull = hasPending ? pending : null;

        // ── Decision ────────────────────────────────────────────────────
        var decision = Decide(intent, reality, pendingOrNull);

        // ── Shadow-mode logging ──────────────────────────────────────────
        // Steady-state Waits are Debug (every 30s per role × 3 = 90 log lines/min
        // at Info is too noisy for production). Action decisions are Info —
        // these are the signal the soak week is looking for.

        var pendingDesc = pendingOrNull is null
            ? "none"
            : $"{pendingOrNull.Kind}:{pendingOrNull.CommandId[..8]}";

        if (decision.Action == MatrixAction.Wait)
        {
            _logger.LogDebug(
                "[SHADOW] [{Role}] intent={Want}/{Deps} reality={Reality} pending={Pending} → Wait — {Reason}",
                role,
                intent.WantDeployed ? "yes" : "no",
                intent.DepsMet ? "depsMet" : "!depsMet",
                reality.State,
                pendingDesc,
                decision.Reason);
        }
        else
        {
            // IssueCreate or IssueDelete — log at Information so it's
            // visible without enabling Debug on the whole namespace.
            _logger.LogInformation(
                "[SHADOW] [{Role}] intent={Want}/{Deps} reality={Reality}({VmState}) pending={Pending} → WOULD {Action} — {Reason}",
                role,
                intent.WantDeployed ? "yes" : "no",
                intent.DepsMet ? "depsMet" : "!depsMet",
                reality.State,
                reality.VmState?.ToString() ?? "-",
                pendingDesc,
                decision.Action,
                decision.Reason);
        }
    }

    // ── Matrix decision function ─────────────────────────────────────────

    /// <summary>
    /// The reconciliation matrix. Pure function — no I/O, no side effects.
    ///
    /// Full truth table:
    ///
    /// <code>
    /// intent.WantDeployed=false
    ///   reality=None,      pending=*         → Wait       (no obligation, no VM — converged)
    ///   reality≠None,      pending=Delete    → Wait       (stray VM, delete in flight)
    ///   reality≠None,      pending≠Delete    → IssueDelete (stray VM — obligation removed)
    ///
    /// intent.WantDeployed=true
    ///   reality=Healthy,   pending=*         → Wait       (converged — running and ready)
    ///   reality=None,      pending=Create    → Wait       (create in flight — wait)
    ///   reality=None,      pending=Delete    → Wait       (delete in flight — will create next)
    ///   reality=None,      pending=null, !depsMet → Wait  (obligation active, waiting on deps)
    ///   reality=None,      pending=null, depsMet  → IssueCreate
    ///   reality=Unhealthy, pending=Delete    → Wait       (delete in flight)
    ///   reality=Unhealthy, pending=Create    → Wait       (create in flight, not yet converged)
    ///   reality=Unhealthy, pending=null      → IssueDelete (unhealthy — delete for redeploy)
    /// </code>
    ///
    /// Note on Unhealthy+Create: the VM exists but isn't Healthy yet (still
    /// booting, cloud-init running, services not ready). This is the expected
    /// transient state after a Create. Wait until it converges to Healthy or
    /// the command expires (sweeper), which will then produce Unhealthy+null.
    ///
    /// Note on Healthy+pending: a command is in flight but the VM is already
    /// Healthy. Could be a stale Create ack that arrived late, or a Delete that
    /// hasn't happened yet. Log at Warning in the caller — this is unexpected.
    /// Conservative: Wait and let the sweeper clear the stale entry.
    /// </summary>
    private static MatrixDecision Decide(
        Intent intent,
        RealitySnapshot reality,
        OutstandingCommand? pending)
    {
        // ── No obligation for this role ──────────────────────────────────
        if (!intent.WantDeployed)
        {
            if (reality.State == Reality.None)
                return new(MatrixAction.Wait,
                    "no obligation, no VM — converged");

            if (pending?.Kind == OutstandingCommandKind.Delete)
                return new(MatrixAction.Wait,
                    $"no obligation, stray VM {reality.VmId}, delete already in flight");

            return new(MatrixAction.IssueDelete,
                $"no obligation but VM {reality.VmId} exists ({reality.VmState}) — deleting stray");
        }

        // ── Obligation exists — want a healthy VM ────────────────────────
        switch (reality.State)
        {
            case Reality.Healthy:
                // Steady state. Log at caller as Debug.
                if (pending is not null)
                {
                    // Unexpected — VM is Healthy but there's a command in flight.
                    // This can happen if a Delete was issued just as the VM finished booting,
                    // or if a Create ack arrived late. Conservative: Wait; sweeper will clear.
                    return new(MatrixAction.Wait,
                        $"VM healthy but {pending.Kind} command {pending.CommandId[..8]} still outstanding — waiting for sweep or ack");
                }
                return new(MatrixAction.Wait, "converged — VM running and fully ready");

            case Reality.None:
                if (pending?.Kind == OutstandingCommandKind.Create)
                    return new(MatrixAction.Wait, "no VM yet, create in flight");

                if (pending?.Kind == OutstandingCommandKind.Delete)
                    return new(MatrixAction.Wait, "no VM, delete in flight — will create next cycle when clear");

                // No pending command. May or may not create depending on deps.
                if (!intent.DepsMet)
                    return new(MatrixAction.Wait, "obligation active, no VM, dependencies not yet Healthy");

                return new(MatrixAction.IssueCreate, "obligation active, no VM, deps met — creating");

            case Reality.Unhealthy:
                if (pending?.Kind == OutstandingCommandKind.Delete)
                    return new(MatrixAction.Wait, $"VM {reality.VmId} unhealthy, delete in flight");

                if (pending?.Kind == OutstandingCommandKind.Create)
                    return new(MatrixAction.Wait,
                        $"VM {reality.VmId} exists but not Healthy yet ({reality.VmState}) — create in flight, waiting to converge");

                return new(MatrixAction.IssueDelete,
                    $"VM {reality.VmId} unhealthy ({reality.VmState}) — deleting for redeploy");

            default:
                // Should not be reachable — Reality only has three values.
                return new(MatrixAction.Wait,
                    $"unhandled reality state {reality.State} — conservative wait");
        }
    }
}