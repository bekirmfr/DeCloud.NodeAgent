using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.SystemVm;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Models.Reality;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services.SystemVm;

/// <summary>
/// Default implementation of <see cref="IRealityProjection"/> that reads from
/// the node's <see cref="IVmManager"/>.
///
/// Decision rules, in order:
///
///   1. Canonicalise the role string. Unknown → <see cref="Reality.None"/>.
///
///   2. Fetch all locally-tracked VMs and filter to those whose
///      <c>Spec.VmType</c> matches the role. Filter out VMs in
///      <see cref="VmState.NotFound"/> or <see cref="VmState.Deleted"/> —
///      those are tombstones, not present reality.
///
///   3. Empty filtered set → <see cref="Reality.None"/>.
///
///   4. If exactly one matches: classify as <see cref="Reality.Healthy"/>
///      iff <c>State == Running</c> and <see cref="VmInstance.IsFullyReady"/>;
///      otherwise <see cref="Reality.Unhealthy"/>.
///
///   5. If multiple match (invariant violation — should not happen with the
///      libvirt duplicate-name guard, but defended-against here):
///        a) Log Warning identifying the offenders.
///        b) If any is Healthy, return Healthy with that VM's id (the others
///           are stragglers; cleanup happens through other paths or future
///           cycles when they're the only ones left).
///        c) Otherwise return Unhealthy with the *most recently created*
///           VM's id, on the rationale that the most recent attempt is the
///           "active" one and reflects current reconciler state best.
///
/// Logging discipline:
///   - Hot path. No allocations beyond the snapshot record itself.
///   - Debug log on every projection is too noisy (this runs every 30 s
///     for every role on every node). Only log Warning for the multi-VM
///     invariant violation; the matrix logs its decisions separately.
/// </summary>
public sealed class RealityProjection : IRealityProjection
{
    private readonly IVmManager _vmManager;
    private readonly ILogger<RealityProjection> _logger;

    public RealityProjection(
        IVmManager vmManager,
        ILogger<RealityProjection> logger)
    {
        _vmManager = vmManager;
        _logger = logger;
    }

    /// <inheritdoc/>
    public RealitySnapshot Project(string role)
    {
        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
            return RealitySnapshot.None;

        var targetVmType = MapRoleToVmType(canonical);
        if (targetVmType is null)
            return RealitySnapshot.None;

        // Fetch + filter. Allocates a List<T> only because GetAllVms() returns
        // an IReadOnlyCollection — can't filter+enumerate without materialising
        // unless we want to enumerate twice.
        var matches = _vmManager.GetAllVms()
            .Where(vm =>
                vm.Spec.VmType == targetVmType.Value &&
                vm.State != VmState.NotFound &&
                vm.State != VmState.Deleted)
            .ToList();

        return matches.Count switch
        {
            0 => RealitySnapshot.None,
            1 => Classify(matches[0]),
            _ => ClassifyMultiple(canonical, matches),
        };
    }

    // ── Classification helpers ──────────────────────────────────────────

    private static RealitySnapshot Classify(VmInstance vm)
    {
        var healthy = vm.State == VmState.Running && vm.IsFullyReady;
        return new RealitySnapshot
        {
            State = healthy ? Reality.Healthy : Reality.Unhealthy,
            VmId = vm.VmId,
            VmState = vm.State,
        };
    }

    private RealitySnapshot ClassifyMultiple(string role, List<VmInstance> matches)
    {
        // Invariant violation. The libvirt duplicate-name guard should make
        // this unreachable in practice — but the cost of defending is one
        // extra log line, and the benefit is we don't return a wrong
        // snapshot if the invariant ever breaks.
        _logger.LogWarning(
            "RealityProjection [{Role}]: {Count} VMs match this role on the node — " +
            "expected at most one. VMs: {VmIds}. " +
            "Reporting the canonical reality and surfacing one VM id; " +
            "stragglers will be cleaned up on subsequent cycles.",
            role,
            matches.Count,
            string.Join(", ", matches.Select(v => $"{v.VmId}({v.State})")));

        // Prefer a Healthy VM if any — keep the active deploy, let stragglers
        // be cleaned up later (they're not blocking the matrix from converging).
        var healthy = matches
            .FirstOrDefault(vm => vm.State == VmState.Running && vm.IsFullyReady);

        if (healthy is not null)
        {
            return new RealitySnapshot
            {
                State = Reality.Healthy,
                VmId = healthy.VmId,
                VmState = healthy.State,
            };
        }

        // No Healthy among the matches — surface the most recently created.
        // The most recent attempt is most likely to be the matrix's current
        // target; older entries are leftovers the reconciler never cleared.
        var mostRecent = matches
            .OrderByDescending(vm => vm.CreatedAt)
            .First();

        return new RealitySnapshot
        {
            State = Reality.Unhealthy,
            VmId = mostRecent.VmId,
            VmState = mostRecent.State,
        };
    }

    // ── Role ↔ VmType mapping ───────────────────────────────────────────

    /// <summary>
    /// Canonical role string → <see cref="VmType"/>.
    /// Inlined as a switch (rather than a shared mapping helper) because the
    /// set is fixed and small; abstraction would obscure rather than help.
    /// </summary>
    private static VmType? MapRoleToVmType(string canonicalRole) => canonicalRole switch
    {
        ObligationRole.Dht => VmType.Dht,
        ObligationRole.Relay => VmType.Relay,
        ObligationRole.BlockStore => VmType.BlockStore,
        _ => null,
    };
}