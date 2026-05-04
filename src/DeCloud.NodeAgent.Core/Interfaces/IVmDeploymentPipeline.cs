using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared.Models;

namespace DeCloud.NodeAgent.Core.Interfaces;

/// <summary>
/// Node-side unified deploy pipeline. Single entry point for both
/// <c>SystemVmReconciler.ActCreateAsync</c> (system VMs) and
/// <c>CommandProcessorService.HandleCreateVmAsync</c> (tenant + custom + container).
///
/// <para>Owns:</para>
/// <list type="number">
///   <item>Artifact prefetch into the local cache (idempotent, no-op if list empty).</item>
///   <item>Architecture-filtered binary verification — every binary artifact
///         that matches the host arch (or is universal) must be cached after
///         prefetch, otherwise the deploy is refused before VM creation.</item>
///   <item>Manager dispatch: <see cref="VmSpec.DeploymentMode"/> selects libvirt
///         vs docker. The chosen manager's <c>CreateVmAsync</c> is invoked.</item>
/// </list>
///
/// <para>Caller owns (out of scope):</para>
/// <list type="bullet">
///   <item>Cloud-init rendering / variable substitution. System VMs render via
///         <c>SubstituteArtifactVariables</c> in the reconciler before calling here;
///         tenant VMs receive already-rendered cloud-init from the orchestrator.</item>
///   <item>Service readiness check registration. System VMs register from template
///         declarations (<c>SetVmServicesAsync</c>); tenant VMs from the command
///         payload (<c>ParseServiceDefinitions</c>). Two different sources, two
///         different code paths — keeping this in the pipeline would force one
///         caller to translate to the other's format. Cleaner to leave it out.</item>
///   <item>Outstanding-command tracking (system-VM-only concept).</item>
///   <item>GPU PCI assignment (currently in <c>CommandProcessorService</c>).</item>
/// </list>
///
/// <para>This is the boundary correction described in the 2026-05-04 design
/// discussion. Prior to extraction, <c>SystemVmReconciler.ActCreateAsync</c>
/// prefetched artifacts and <c>CommandProcessorService.HandleCreateVmAsync</c>
/// did not — the asymmetry caused the us-1-8c15 deploy bug where tenant
/// cloud-init referenced artifact URLs the cache had never seen.</para>
/// </summary>
public interface IVmDeploymentPipeline
{
    /// <summary>
    /// Run the prefetch + dispatch pipeline.
    /// </summary>
    /// <param name="spec">Fully-populated VM spec including
    ///   <see cref="VmSpec.CloudInitUserData"/> if applicable.</param>
    /// <param name="artifacts">Template artifacts. May be empty (e.g. custom
    ///   cloud-init or container deployments). Filtering by architecture is
    ///   handled inside the pipeline.</param>
    /// <param name="password">Optional root password (tenant VMs).</param>
    /// <param name="ct">Cancellation.</param>
    /// <returns>The <see cref="VmOperationResult"/> from the underlying manager,
    ///   or a Failed result if a binary artifact is missing after prefetch.</returns>
    Task<VmOperationResult> DeployAsync(
        VmSpec spec,
        IReadOnlyList<TemplateArtifact> artifacts,
        string? password,
        CancellationToken ct);
}