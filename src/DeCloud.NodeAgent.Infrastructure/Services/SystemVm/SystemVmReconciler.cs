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
using System.Runtime.InteropServices;
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
/// <c>SystemVmReconciliationService</c> is disabled in P7 (after soak).
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
    private readonly IVmManager _vmManager;
    private readonly IArtifactCacheService _artifactCache;
    private readonly VmRepository _repository;
    private readonly ILogger<SystemVmReconciler> _logger;

    public SystemVmReconciler(
        IObligationStateService obligationState,
        IIntentComputation intent,
        IRealityProjection reality,
        IOutstandingCommands outstanding,
        IVmManager vmManager,
        IArtifactCacheService artifactCache,
        VmRepository repository,
        ILogger<SystemVmReconciler> logger)
    {
        _obligationState = obligationState;
        _intent = intent;
        _reality = reality;
        _outstanding = outstanding;
        _vmManager = vmManager;
        _artifactCache = artifactCache;
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
    ///   4. Build the cloud-init UserData by substituting artifact URL/SHA256 variables
    ///      in the template content. LibvirtVmManager handles all other variables
    ///      (__NODE_ID__, __ORCHESTRATOR_URL__, __HOSTNAME__, __TIMESTAMP__) from its
    ///      own node metadata at ISO-creation time.
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

        // ── 3. Verify / prefetch artifacts ──────────────────────────────
        var arch = RuntimeInformation.ProcessArchitecture == Architecture.Arm64
            ? "arm64" : "amd64";

        if (template.Artifacts.Count > 0)
        {
            await _artifactCache.PrefetchAsync(template.Artifacts, arch, ct);

            // Verify every architecture-appropriate binary is cached.
            // A missing binary means the VM will fail to start — better to skip
            // and retry next cycle than to deploy a broken VM.
            var missingBinary = false;
            foreach (var artifact in template.Artifacts)
            {
                if (artifact.Type != ArtifactType.Binary) continue;
                if (artifact.Architecture is not null &&
                    !string.Equals(artifact.Architecture, arch, StringComparison.OrdinalIgnoreCase))
                    continue;

                var cached = await _artifactCache.GetLocalPathAsync(artifact.Sha256, ct);
                if (cached is null)
                {
                    _logger.LogWarning(
                        "SystemVmReconciler [{Role}]: binary artifact '{Name}' ({Sha256}) " +
                        "not in cache after prefetch — skipping Create this cycle.",
                        role, artifact.Name, artifact.Sha256[..12]);
                    missingBinary = true;
                }
            }

            if (missingBinary) return;
        }

        // ── 4. Substitute artifact variables in cloud-init ──────────────
        var cloudInit = SubstituteArtifactVariables(
            template.CloudInitContent, template.Artifacts, arch);

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
                // Carried into LibvirtVmManager's variable substitution pipeline
                // as spec.Labels["node-arch"] for ResolveArtifactVariables.
                ["node-arch"] = arch,
                ["system-vm-role"] = role,
                ["system-vm-revision"] = template.Revision.ToString(),
            }
        };

        // ── 6. Register outstanding command ─────────────────────────────
        var commandId = Guid.NewGuid().ToString();
        _outstanding.Set(role, new OutstandingCommand
        {
            CommandId = commandId,
            Kind = OutstandingCommandKind.Create,
            VmId = vmId,
            IssuedAt = DateTime.UtcNow,
        });

        // ── 7–9. Create VM, set services, clear outstanding ─────────────
        try
        {
            var result = await _vmManager.CreateVmAsync(vmSpec, password: null, ct);

            if (!result.Success)
            {
                _logger.LogError(
                    "SystemVmReconciler [{Role}]: CreateVm failed — {Error}. " +
                    "Will retry on next cycle.",
                    role, result.ErrorMessage);
                return;
            }

            _logger.LogInformation(
                "SystemVmReconciler [{Role}]: VM {VmId} created successfully",
                role, vmId);

            // ── 8. Post-set services so VmReadinessMonitor knows what to probe ──
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
        }
        finally
        {
            // Always clear. Next cycle evaluates from fresh reality projection.
            // Create succeeded → reality = Running (eventually) → Healthy.
            // Create failed   → reality = None → matrix issues Create again.
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

    // ── Artifact variable substitution ───────────────────────────────────

    /// <summary>
    /// Replace <c>${ARTIFACT_URL:name}</c> and <c>${ARTIFACT_SHA256:name}</c>
    /// placeholders in the cloud-init template with the local artifact cache
    /// URLs and SHA256 digests, filtered by architecture.
    ///
    /// All artifact URLs resolve to the local node-agent cache endpoint:
    /// <c>http://192.168.122.1:5100/api/artifacts/{sha256}</c>
    ///
    /// The VM never sees the upstream SourceUrl — it always fetches from the
    /// local cache, which the node agent has pre-verified.
    /// </summary>
    private static string SubstituteArtifactVariables(
        string cloudInitContent,
        IReadOnlyList<TemplateArtifact> artifacts,
        string nodeArch)
    {
        if (artifacts.Count == 0) return cloudInitContent;

        var result = cloudInitContent;

        foreach (var artifact in artifacts)
        {
            // Skip arch-specific artifacts that don't match this node.
            if (artifact.Architecture is not null &&
                !string.Equals(artifact.Architecture, nodeArch,
                    StringComparison.OrdinalIgnoreCase))
                continue;

            var urlKey = $"${{ARTIFACT_URL:{artifact.Name}}}";
            var sha256Key = $"${{ARTIFACT_SHA256:{artifact.Name}}}";
            var localUrl = $"{ArtifactBaseUrl}/{artifact.Sha256}";

            result = result
                .Replace(urlKey, localUrl, StringComparison.Ordinal)
                .Replace(sha256Key, artifact.Sha256, StringComparison.Ordinal);
        }

        return result;
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
    ///   reality=Unhealthy, pending=Create    → Wait       (create in flight, not converged yet)
    ///   reality=Unhealthy, pending=null      → IssueDelete (delete for redeploy)
    /// </code>
    /// </summary>
    private static MatrixDecision Decide(
        Intent intent,
        RealitySnapshot reality,
        OutstandingCommand? pending)
    {
        if (!intent.WantDeployed)
        {
            if (reality.State == Reality.None)
                return new(MatrixAction.Wait, "no obligation, no VM — converged");

            if (pending?.Kind == OutstandingCommandKind.Delete)
                return new(MatrixAction.Wait,
                    $"no obligation, stray VM {reality.VmId}, delete already in flight");

            return new(MatrixAction.IssueDelete,
                $"no obligation but VM {reality.VmId} exists ({reality.VmState}) — deleting stray");
        }

        switch (reality.State)
        {
            case Reality.Healthy:
                if (pending is not null)
                    return new(MatrixAction.Wait,
                        $"VM healthy but {pending.Kind} command {pending.CommandId[..8]} still outstanding — waiting for sweep");
                return new(MatrixAction.Wait, "converged — VM running and fully ready");

            case Reality.None:
                if (pending?.Kind == OutstandingCommandKind.Create)
                    return new(MatrixAction.Wait, "no VM yet, create in flight");

                if (pending?.Kind == OutstandingCommandKind.Delete)
                    return new(MatrixAction.Wait, "no VM, delete in flight — will create next cycle when clear");

                if (!intent.DepsMet)
                    return new(MatrixAction.Wait,
                        "obligation active, no VM, dependencies not yet Healthy");

                return new(MatrixAction.IssueCreate,
                    "obligation active, no VM, deps met — creating");

            case Reality.Unhealthy:
                if (pending?.Kind == OutstandingCommandKind.Delete)
                    return new(MatrixAction.Wait, $"VM {reality.VmId} unhealthy, delete in flight");

                if (pending?.Kind == OutstandingCommandKind.Create)
                    return new(MatrixAction.Wait,
                        $"VM {reality.VmId} not Healthy yet ({reality.VmState}) — create in flight, waiting to converge");

                return new(MatrixAction.IssueDelete,
                    $"VM {reality.VmId} unhealthy ({reality.VmState}) — deleting for redeploy");

            default:
                return new(MatrixAction.Wait,
                    $"unhandled reality state {reality.State} — conservative wait");
        }
    }
}