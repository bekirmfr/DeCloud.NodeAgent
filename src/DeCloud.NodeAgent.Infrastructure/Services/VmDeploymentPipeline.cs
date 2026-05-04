using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Docker;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Default implementation of <see cref="IVmDeploymentPipeline"/>.
///
/// Step ordering (from the 2026-05-04 design diagram):
/// <list type="number">
///   <item><b>Step 3 — prefetch.</b> Idempotent. <c>ArtifactCacheService.PrefetchAsync</c>
///         dispatches each artifact through HTTPS or <c>data:</c> URI handler internally.
///         No-op if <paramref name="artifacts"/> is empty (custom cloud-init or container).</item>
///   <item><b>Step 3a — verify.</b> For every <c>Binary</c> artifact whose
///         architecture matches the host (or is universal/null), confirm it is in
///         the cache. A binary missing here means the VM will fail to fetch its
///         binary at boot — better to refuse the deploy than to ship a broken VM.
///         Non-Binary artifacts (Script, WebAsset) are not verified — their
///         absence is degraded but not fatal (the VM can still boot).</item>
///   <item><b>Step 5 — dispatch.</b> Route by <see cref="VmSpec.DeploymentMode"/>
///         to libvirt (default) or docker (Container mode). Call <c>CreateVmAsync</c>.</item>
/// </list>
///
/// Step 4 (cloud-init rendering / variable substitution) is intentionally NOT in
/// this pipeline. System VMs render in <c>SystemVmReconciler.SubstituteArtifactVariables</c>
/// before calling here; tenant VMs receive pre-rendered cloud-init from the
/// orchestrator. Phase 4 will collapse rendering into a single
/// orchestrator-side IVmDeployer that pushes rendered cloud-init for every VM
/// type — at which point the System VM "render on node" path goes away and
/// this pipeline keeps the same shape.
/// </summary>
public class VmDeploymentPipeline : IVmDeploymentPipeline
{
    private readonly IArtifactCacheService _artifactCache;
    private readonly IVmManager _vmManager;
    private readonly DockerContainerManager _dockerManager;
    private readonly ILogger<VmDeploymentPipeline> _logger;

    public VmDeploymentPipeline(
        IArtifactCacheService artifactCache,
        IVmManager vmManager,
        DockerContainerManager dockerManager,
        ILogger<VmDeploymentPipeline> logger)
    {
        _artifactCache = artifactCache;
        _vmManager = vmManager;
        _dockerManager = dockerManager;
        _logger = logger;
    }

    public async Task<VmOperationResult> DeployAsync(
        VmSpec spec,
        IReadOnlyList<TemplateArtifact> artifacts,
        string? password,
        CancellationToken ct)
    {
        var arch = ResourceDiscoveryService.GetArchitectureNormalised();

        // ── Step 3 — prefetch ───────────────────────────────────────────
        if (artifacts is { Count: > 0 })
        {
            await _artifactCache.PrefetchAsync(artifacts, arch, ct);

            // ── Step 3a — verify binaries are cached ────────────────────
            foreach (var artifact in artifacts)
            {
                if (artifact.Type != ArtifactType.Binary) continue;

                // Skip arch-specific artifacts that don't match this host.
                // Universal (Architecture == null) artifacts always match.
                if (artifact.Architecture is not null &&
                    !string.Equals(artifact.Architecture, arch,
                                   StringComparison.OrdinalIgnoreCase))
                    continue;

                var cached = await _artifactCache.GetLocalPathAsync(artifact.Sha256, ct);
                if (cached is null)
                {
                    var msg =
                        $"Binary artifact '{artifact.Name}' " +
                        $"({artifact.Sha256[..12]}) not in cache after prefetch — " +
                        "refusing deploy.";

                    _logger.LogWarning(
                        "VmDeploymentPipeline: VM {VmId}: {Msg}", spec.Id, msg);

                    return new VmOperationResult
                    {
                        Success = false,
                        ErrorMessage = msg
                    };
                }
            }
        }

        // ── Step 5 — dispatch ───────────────────────────────────────────
        IVmManager manager = spec.DeploymentMode == DeploymentMode.Container
            ? _dockerManager
            : _vmManager;

        return await manager.CreateVmAsync(spec, password, ct);
    }
}