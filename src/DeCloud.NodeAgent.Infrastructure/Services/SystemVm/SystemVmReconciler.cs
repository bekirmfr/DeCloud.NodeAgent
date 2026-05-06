using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Interfaces.SystemVm;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Models.Reality;
using DeCloud.NodeAgent.Core.Models.State;
using DeCloud.NodeAgent.Core.Models.SystemVm;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Text.Json;

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
/// explanation of why.
/// </summary>
public sealed record MatrixDecision(MatrixAction Action, string Reason);

// ── Reconciler ──────────────────────────────────────────────────────────────

/// <summary>
/// Node-side system VM reconciliation loop (SYSTEM_VM_DESIGN.md §5).
///
/// Every <see cref="ReconcileInterval"/> this service evaluates the three-axis
/// reconciliation matrix for each of the three canonical system VM roles:
///
/// <code>
///   (intent, reality, pending) → MatrixAction
/// </code>
///
/// <b>P6 — fully live.</b>
///
/// <b>IssueDelete:</b> calls <see cref="IVmManager.DeleteVmAsync"/> directly.
/// Fully autonomous — no orchestrator contact required.
///
/// <b>IssueCreate:</b> reads the system template from local SQLite
/// (<see cref="IObligationStateService.GetSystemTemplateJsonAsync"/>), verifies
/// all artifacts are cached (<see cref="IArtifactCacheService"/>), builds a
/// <see cref="VmSpec"/> and calls <see cref="IVmManager.CreateVmAsync"/> directly.
/// Cloud-init identity state is NOT injected here — system VMs query
/// <c>http://192.168.122.1:5100/api/obligations/{role}/state</c> at boot time
/// (SYSTEM_VM_DESIGN.md §4.8). LibvirtVmManager handles <c>__NODE_ID__</c>,
/// <c>__ORCHESTRATOR_URL__</c>, <c>__HOSTNAME__</c>, <c>__TIMESTAMP__</c>
/// substitution from its own node metadata — the reconciler only needs to
/// substitute artifact URL variables.
///
/// <b>Orchestrator cooperation:</b> the orchestrator's
/// <c>SystemVmReconciliationService</c> is disabled.
/// Until P7 ships, both systems run concurrently — the node handles deletes,
/// the orchestrator handles the gaps until templates are seeded in P10.
/// This causes no conflict: the orchestrator's <c>VerifyActiveAsync</c> sees
/// a missing VM and re-deploys — which is exactly correct after a node-driven
/// delete. The node's reconciler skips Create when no template is available.
/// </summary>
public sealed class SystemVmReconciler : BackgroundService
{
    // ── Constants ───────────────────────────────────────────────────────

    private static readonly TimeSpan ReconcileInterval = TimeSpan.FromSeconds(30);
    private static readonly TimeSpan CommandTimeout = TimeSpan.FromMinutes(20);
    private static readonly TimeSpan StartupDelay = TimeSpan.FromSeconds(10);

    private static readonly IReadOnlyList<string> CanonicalRoles =
        [ObligationRole.Relay, ObligationRole.Dht, ObligationRole.BlockStore];

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    // virbr0 bridge IP from which VMs fetch artifacts and identity.
    private const string ArtifactBaseUrl = "http://192.168.122.1:5100/api/artifacts";

    // ── Dependencies ────────────────────────────────────────────────────

    private readonly IObligationStateService _obligationState;
    private readonly IIntentComputation _intent;
    private readonly IRealityProjection _reality;
    private readonly IOutstandingCommands _outstanding;
    private readonly IVmDeploymentPipeline _pipeline;
    private readonly IVmManager _vmManager;
    private readonly VmRepository _repository;
    private readonly ILogger<SystemVmReconciler> _logger;

    public SystemVmReconciler(
        IObligationStateService obligationState,
        IIntentComputation intent,
        IRealityProjection reality,
        IOutstandingCommands outstanding,
        IVmDeploymentPipeline pipeline,
        IVmManager vmManager,
        VmRepository repository,
        ILogger<SystemVmReconciler> logger)
    {
        _obligationState = obligationState;
        _intent = intent;
        _reality = reality;
        _outstanding = outstanding;
        _pipeline = pipeline;
        _vmManager = vmManager;
        _repository = repository;
        _logger = logger;
    }

    // ── BackgroundService ───────────────────────────────────────────────

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        _logger.LogInformation(
            "SystemVmReconciler started — Delete and Create are LIVE.");

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
                _logger.LogError(ex,
                    "SystemVmReconciler: unhandled exception in cycle — will retry");
            }

            await Task.Delay(ReconcileInterval, ct);
        }

        _logger.LogInformation("SystemVmReconciler stopped");
    }

    // ── Cycle ────────────────────────────────────────────────────────────

    private async Task RunCycleAsync(CancellationToken ct)
    {
        var obligations = await _obligationState.GetObligationsAsync(ct);

        foreach (var role in CanonicalRoles)
        {
            try
            {
                await EvaluateRoleAsync(role, obligations, ct);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "SystemVmReconciler: exception evaluating role '{Role}' — skipping this cycle",
                    role);
            }
        }

        var swept = _outstanding.SweepExpired(CommandTimeout);
        if (swept > 0)
        {
            _logger.LogWarning(
                "SystemVmReconciler: swept {Count} timed-out outstanding commands — " +
                "next cycle re-evaluates from current reality",
                swept);
        }
    }

    // ── Per-role evaluation ──────────────────────────────────────────────

    private async Task EvaluateRoleAsync(
        string role,
        IReadOnlyList<ObligationDescriptor> obligations,
        CancellationToken ct)
    {
        var intent = _intent.Compute(role, obligations);
        var reality = _reality.Project(role);
        var hasPending = _outstanding.TryGet(role, out var pending);
        var pendingOrNull = hasPending ? pending : null;

        // VM created (VmId set = CreateVmAsync succeeded) but never appeared in
        // libvirt. 3 minutes is enough for any host — clear and retry.
        if (pendingOrNull?.Kind == OutstandingCommandKind.Create &&
            !string.IsNullOrEmpty(pendingOrNull.VmId) &&
            reality.State == Reality.None &&
            (DateTime.UtcNow - pendingOrNull.IssuedAt) > TimeSpan.FromMinutes(3))
        {
            _outstanding.Clear(role);
            _logger.LogWarning(
                "SystemVmReconciler [{Role}]: VM {VmId} never appeared in libvirt " +
                "after 3 minutes — clearing pending to retry.",
                role, pendingOrNull.VmId);
            pendingOrNull = null;
        }

        var decision = Decide(intent, reality, pendingOrNull);

        var pendingDesc = pendingOrNull is null
            ? "none"
            : $"{pendingOrNull.Kind}:{pendingOrNull.CommandId[..8]}";

        switch (decision.Action)
        {
            case MatrixAction.Wait:
                _logger.LogDebug(
                    "[{Role}] intent={Want}/{Deps} reality={Reality} pending={Pending} → Wait — {Reason}",
                    role,
                    intent.WantDeployed ? "yes" : "no",
                    intent.DepsMet ? "depsMet" : "!depsMet",
                    reality.State, pendingDesc, decision.Reason);
                break;

            case MatrixAction.IssueDelete:
                _logger.LogInformation(
                    "[{Role}] intent={Want}/{Deps} reality={Reality}({VmState}) pending={Pending} → Delete {VmId} — {Reason}",
                    role,
                    intent.WantDeployed ? "yes" : "no",
                    intent.DepsMet ? "depsMet" : "!depsMet",
                    reality.State, reality.VmState?.ToString() ?? "-",
                    pendingDesc, reality.VmId, decision.Reason);
                await ActDeleteAsync(role, reality.VmId!, ct);
                break;

            case MatrixAction.IssueCreate:
                _logger.LogInformation(
                    "[{Role}] intent={Want}/{Deps} reality={Reality} pending={Pending} → Create — {Reason}",
                    role,
                    intent.WantDeployed ? "yes" : "no",
                    intent.DepsMet ? "depsMet" : "!depsMet",
                    reality.State, pendingDesc, decision.Reason);
                await ActCreateAsync(role, ct);
                break;
        }
    }

    // ── Delete action ────────────────────────────────────────────────────

    /// <summary>
    /// Performs a live system VM deletion. Sets the outstanding-command entry
    /// before calling so concurrent readers see the correct state. Clears it
    /// in <c>finally</c> so the next cycle always gets a fresh evaluation.
    ///
    /// On success: next cycle projects <see cref="Reality.None"/> and, if the
    /// obligation still exists, the orchestrator's reconcile loop (or the node's
    /// reconciler post-P10 seeding) deploys a replacement.
    ///
    /// On failure: next cycle projects <see cref="Reality.Unhealthy"/> → matrix
    /// issues another Delete. libvirt delete is idempotent for non-existent domains.
    ///
    /// Note on NAT cleanup: relay VM iptables rules are swept within ~60s by
    /// <c>VmHealthService.CleanStaleRelayNatIfNotRelayNodeAsync</c>.
    /// </summary>
    private async Task ActDeleteAsync(string role, string vmId, CancellationToken ct)
    {
        var commandId = Guid.NewGuid().ToString();

        _outstanding.Set(role, new OutstandingCommand
        {
            CommandId = commandId,
            Kind = OutstandingCommandKind.Delete,
            VmId = vmId,
            IssuedAt = DateTime.UtcNow,
        });

        try
        {
            var result = await _vmManager.DeleteVmAsync(vmId, ct);

            if (result.Success)
            {
                _logger.LogInformation(
                    "SystemVmReconciler [{Role}]: VM {VmId} deleted successfully",
                    role, vmId);
            }
            else
            {
                _logger.LogWarning(
                    "SystemVmReconciler [{Role}]: delete VM {VmId} returned failure — {Error}. " +
                    "Will retry on next cycle.",
                    role, vmId, result.ErrorMessage);
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex,
                "SystemVmReconciler [{Role}]: delete VM {VmId} threw — will retry on next cycle",
                role, vmId);
        }
        finally
        {
            _outstanding.Clear(role);
        }
    }

    // ── Create action ────────────────────────────────────────────────────

    /// <summary>
    /// Performs a live system VM creation using the locally-cached system template.
    ///
    /// Steps:
    ///   1. Fetch template JSON from SQLite — skip if absent (P10 hasn't seeded yet).
    ///   2. Deserialise to <see cref="SystemVmTemplate"/>.
    ///   3. Verify artifacts are cached; prefetch any that are missing.
    ///      Skip if any critical (Binary) artifact is still missing after prefetch.
    ///   4. Use template.CloudInitContent directly — fully rendered by the orchestrator.
    ///   5. Build <see cref="VmSpec"/> from the template resource spec.
    ///   6. Register an outstanding command before dispatching.
    ///   7. Call <see cref="IVmManager.CreateVmAsync"/>.
    ///   8. On success: look up the newly created <see cref="VmInstance"/>,
    ///      set <see cref="VmInstance.Services"/> from the template declarations,
    ///      persist via <see cref="VmRepository"/>.
    ///   9. Clear the outstanding command in <c>finally</c>.
    /// </summary>
    private async Task ActCreateAsync(string role, CancellationToken ct)
    {
        // ── 1. Fetch template ───────────────────────────────────────────
        var templateJson = await _obligationState.GetSystemTemplateJsonAsync(role, ct);
        if (templateJson is null)
        {
            _logger.LogInformation(
                "SystemVmReconciler [{Role}]: no system template in SQLite — " +
                "orchestrator hasn't seeded it yet (P10). Skipping Create this cycle.",
                role);
            return;
        }

        // ── 2. Deserialise ──────────────────────────────────────────────
        SystemVmTemplate template;
        try
        {
            template = JsonSerializer.Deserialize<SystemVmTemplate>(templateJson, JsonOpts)
                       ?? throw new InvalidOperationException("Deserialised to null");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "SystemVmReconciler [{Role}]: failed to deserialise system template — " +
                "template JSON may be corrupt. Skipping Create.",
                role);
            return;
        }

        // ── 3. Architecture ─────────────────────────────────────────────
        // Needed for the node-arch label and for the deployment pipeline's
        // architecture-filtered binary verification.
        // Prefetch + binary verification are in IVmDeploymentPipeline (P2.5).
        var arch = ResourceDiscoveryService.GetArchitectureNormalised();

        // ── 4. Cloud-init ────────────────────────────────────────────────
        // template.CloudInitContent is the fully-rendered string produced by
        // CloudInitRenderer on the orchestrator: all __VARNAME__ placeholders
        // substituted in Pass 1, all ${ARTIFACT_URL:...} tokens substituted
        // in Pass 2. Use it directly — no node-side substitution needed or safe.
        // SubstituteArtifactVariables is dropped (P3.1.5): calling it on an
        // already-substituted string is a no-op, and it would be wrong once
        // the legacy LibvirtVmManager substitution path is removed in Phase 4.
        var cloudInit = template.CloudInitContent;

        // ── 5. Build VmSpec ─────────────────────────────────────────────
        var vmId = Guid.NewGuid().ToString();
        var vmName = $"{role}-{vmId[..8]}";

        var vmType = role switch
        {
            ObligationRole.Relay => VmType.Relay,
            ObligationRole.Dht => VmType.Dht,
            ObligationRole.BlockStore => VmType.BlockStore,
            _ => throw new ArgumentException($"Unknown role: {role}")
        };

        var vmSpec = new VmSpec
        {
            Id = vmId,
            Name = vmName,
            VmType = vmType,
            VirtualCpuCores = template.VirtualCpuCores > 0 ? template.VirtualCpuCores : 1,
            MemoryBytes = template.MemoryBytes > 0
                                  ? template.MemoryBytes
                                  : 512L * 1024 * 1024,
            DiskBytes = template.DiskBytes > 0
                                  ? template.DiskBytes
                                  : 2L * 1024 * 1024 * 1024,
            BaseImageUrl = template.BaseImageUrl,
            BaseImageHash = template.BaseImageHash,
            CloudInitUserData = cloudInit,
            ReplicationFactor = 0,  // system VMs are never replicated
            Labels = new Dictionary<string, string>
            {
                // node-arch: consumed by LibvirtVmManager (architecture-filtered
                //   artifact cache lookups; still runs during Phase 3 legacy path).
                // system-vm-role / system-vm-revision: carried into the VM record
                //   for observability and drift detection.
                // Role-specific labels (relay-subnet, relay-region, node-public-ip,
                //   dht-advertise-ip, etc.) are NOT included here. The rendered
                //   cloud-init in CloudInitUserData already has all __VARNAME__
                //   placeholders substituted by the orchestrator renderer (P3.1.x).
                //   LibvirtVmManager STEP 5.5/5.6/5.7 will attempt to Replace()
                //   those placeholders but find nothing to replace — a no-op.
                //   Phase 4.1 removes those legacy substitution blocks entirely.
                ["node-arch"] = arch,
                ["system-vm-role"] = role,
                ["system-vm-revision"] = template.Revision.ToString(),
                // TODO (Phase 3 cleanup / BLOCKSTORE-FIX §6): replace Guid.NewGuid()
                // above with obligation.VmId once NodeService.ReconcileNodeAsync stops
                // pre-assigning VmIds and all three roles adopt the "node mints VmId,
                // orchestrator adopts via heartbeat" pattern.
            }
        };

        // relay-specific label injection block removed (P3.1.5).
        // Previously injected relay-subnet from obligation state so that
        // LibvirtVmManager STEP 5.5 could substitute __RELAY_SUBNET__ in cloud-init.
        // Now redundant: RelaySubnetResolver (P3.1.1/P3.1.2) substitutes
        // __RELAY_SUBNET__ at orchestrator render time; placeholder is absent from
        // template.CloudInitContent by the time the reconciler sees it.

        // ── 6. Register outstanding command ─────────────────────────────
        var commandId = Guid.NewGuid().ToString();
        _outstanding.Set(role, new OutstandingCommand
        {
            CommandId = commandId,
            Kind = OutstandingCommandKind.Create,
            VmId = vmId,
            IssuedAt = DateTime.UtcNow,
            TemplateRevision = template?.Revision ?? 0
        });

        // ── 7–8. Create VM, set services ────────────────────────────────
        // Outstanding command is intentionally NOT cleared on success.
        // It remains live until the next cycle observes reality=Healthy
        // (cleared in EvaluateRoleAsync before Decide) or until the
        // 20-minute CommandTimeout sweep expires it.
        // This prevents the matrix from seeing pending=none + reality=Unhealthy
        // (VM booting through cloud-init) and issuing a premature Delete.
        try
        {
            var result = await _pipeline.DeployAsync(
                vmSpec,
                template.Artifacts,
                password: null,
                ct);

            if (!result.Success)
            {
                _logger.LogError(
                    "SystemVmReconciler [{Role}]: CreateVm failed — {Error}. " +
                    "Will retry on next cycle.",
                    role, result.ErrorMessage);

                // Clear on failure — reality stays None so the next cycle retries.
                _outstanding.Clear(role);
                return;
            }

            _logger.LogInformation(
                "SystemVmReconciler [{Role}]: VM {VmId} created successfully",
                role, vmId);

            // ── Post-set services so VmReadinessMonitor knows what to probe ──
            if (template.Services.Count > 0)
            {
                await SetVmServicesAsync(vmId, template.Services, ct);
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex,
                "SystemVmReconciler [{Role}]: CreateVm threw — will retry on next cycle",
                role);
            // Clear on exception — reality stays None so the next cycle retries.
            _outstanding.Clear(role);
        }
    }

    // ── Service post-set ─────────────────────────────────────────────────

    /// <summary>
    /// After a successful CreateVm, look up the newly created
    /// <see cref="VmInstance"/>, build its service readiness list from the
    /// template declarations, and persist via <see cref="VmRepository"/>.
    ///
    /// This mirrors <c>CommandProcessorService.HandleCreateVmAsync</c>'s
    /// <c>ParseServiceDefinitions</c> + <c>SaveVmAsync</c> pattern.
    /// Without this, VmReadinessMonitor has no services to probe and the VM
    /// can never reach <see cref="ServiceReadiness.Ready"/>.
    /// </summary>
    private async Task SetVmServicesAsync(
        string vmId,
        IReadOnlyList<SystemVmServiceDeclaration> declarations,
        CancellationToken ct)
    {
        try
        {
            var vm = _vmManager.GetAllVms().FirstOrDefault(v => v.VmId == vmId);
            if (vm is null)
            {
                _logger.LogWarning(
                    "SystemVmReconciler: VM {VmId} not found in VmManager after create — " +
                    "services not registered; VmReadinessMonitor will use defaults",
                    vmId);
                return;
            }

            vm.Services = declarations.Select(d => new VmServiceStatus
            {
                Name = d.Name,
                Port = d.Port,
                Protocol = d.Protocol,
                CheckType = Enum.TryParse<CheckType>(d.CheckType, ignoreCase: true, out var ct2)
                                  ? ct2 : CheckType.CloudInitDone,
                HttpPath = d.HttpPath,
                LivenessCheck = d.LivenessCheck,
                TimeoutSeconds = d.TimeoutSeconds,
                Status = ServiceReadiness.Pending,
            }).ToList();

            await _repository.SaveVmAsync(vm);

            _logger.LogDebug(
                "SystemVmReconciler: registered {Count} service checks for VM {VmId}",
                vm.Services.Count, vmId);
        }
        catch (Exception ex)
        {
            // Non-fatal — VmReadinessMonitor has fallback logic for VMs
            // with no service declarations.
            _logger.LogWarning(ex,
                "SystemVmReconciler: failed to set services for VM {VmId} — " +
                "readiness probing may be delayed",
                vmId);
        }
    }

    // ── Matrix decision function ─────────────────────────────────────────

    /// <summary>
    /// The reconciliation matrix. Pure static function — no I/O, no side effects.
    ///
    /// <code>
    /// intent.WantDeployed=false
    ///   reality=None,      pending=*         → Wait       (converged)
    ///   reality≠None,      pending=Delete    → Wait       (delete in flight)
    ///   reality≠None,      pending≠Delete    → IssueDelete (stray VM)
    ///
    /// intent.WantDeployed=true
    ///   reality=Healthy,   pending=*         → Wait       (converged)
    ///   reality=None,      pending=Create    → Wait       (create in flight)
    ///   reality=None,      pending=Delete    → Wait       (delete in flight — create next)
    ///   reality=None,      pending=null, !depsMet → Wait  (waiting on deps)
    ///   reality=None,      pending=null, depsMet  → IssueCreate
    ///   reality=Unhealthy, pending=Delete    → Wait       (delete in flight)
    ///   reality=Unhealthy, pending=Create    → Wait       (booting — shielded until CommandTimeout)
    ///   reality=Unhealthy, pending=null      → IssueDelete (delete for redeploy)
    /// </code>
    /// </summary>
    private static MatrixDecision Decide(
    Intent intent,
    RealitySnapshot reality,
    OutstandingCommand? pending)
    {
        // A pending Create shields a booting VM from premature deletion
        // as long as the domain exists in libvirt (reality != None).
        // reality=Unhealthy during cloud-init is expected and normal.
        //
        // Shield is lifted when:
        //   - reality=Healthy → converged normally, pending no longer needed
        //   - reality=None    → domain gone (user deleted it, or create failed)
        //                       handled by the 3-min pre-check above
        //   - pending expired → SweepExpired (CommandTimeout=20min) cleared it;
        //                       next cycle sees pending=null + reality=Unhealthy → Delete
        //
        // This keeps one timeout in the system (CommandTimeout) rather than
        // introducing a separate BootGracePeriod constant.
        var effectivePending = pending;
        if (pending?.Kind == OutstandingCommandKind.Create &&
            reality.State == Reality.Healthy)
        {
            effectivePending = null;
        }

        if (!intent.WantDeployed)
        {
            if (reality.State == Reality.None)
                return new(MatrixAction.Wait, "no obligation, no VM — converged");

            if (effectivePending?.Kind == OutstandingCommandKind.Delete)
                return new(MatrixAction.Wait,
                    $"no obligation, stray VM {reality.VmId}, delete already in flight");

            return new(MatrixAction.IssueDelete,
                $"no obligation but VM {reality.VmId} exists ({reality.VmState}) — deleting stray");
        }

        switch (reality.State)
        {
            case Reality.Healthy:
                return new(MatrixAction.Wait, "converged — VM running and fully ready");

            case Reality.None:
                if (effectivePending?.Kind == OutstandingCommandKind.Create)
                    return new(MatrixAction.Wait,
                        $"create in flight since {effectivePending.IssuedAt:HH:mm:ss}");

                if (effectivePending?.Kind == OutstandingCommandKind.Delete)
                    return new(MatrixAction.Wait,
                        "delete in flight — will create next cycle when clear");

                if (!intent.DepsMet)
                    return new(MatrixAction.Wait,
                        "obligation active, no VM, dependencies not yet Healthy");

                return new(MatrixAction.IssueCreate,
                    "obligation active, no VM, deps met — creating");

            case Reality.Unhealthy:
                if (effectivePending?.Kind == OutstandingCommandKind.Delete)
                    return new(MatrixAction.Wait,
                        $"VM {reality.VmId} unhealthy, delete in flight");

                if (effectivePending?.Kind == OutstandingCommandKind.Create)
                    return new(MatrixAction.Wait,
                        $"VM {reality.VmId} booting ({reality.VmState}) — create in flight, " +
                        $"waiting up to {CommandTimeout.TotalMinutes:F0}min for services to start");

                return new(MatrixAction.IssueDelete,
                    $"VM {reality.VmId} unhealthy ({reality.VmState}) — deleting for redeploy");

            default:
                return new(MatrixAction.Wait,
                    $"unhandled reality state {reality.State} — conservative wait");
        }
    }
}