using DeCloud.NodeAgent.Core.Interfaces.SystemVm;
using DeCloud.NodeAgent.Core.Models.Reality;
using DeCloud.NodeAgent.Core.Models.State;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services.SystemVm;

/// <summary>
/// Default implementation of <see cref="IIntentComputation"/>.
///
/// Decision rules:
///
///   1. Canonicalise the role string. Unknown → <see cref="Intent.None"/>.
///
///   2. Look up the canonical role in the supplied obligation list. If
///      absent, return <c>Intent.None</c> — this node has no obligation for
///      the role.
///
///   3. The role is obligated. Compute its applicable dependencies:
///      every entry in the obligation's <see cref="ObligationDescriptor.Deps"/>
///      list whose canonical name *also* appears in the obligation list.
///      Dependencies the orchestrator declared but this node does not
///      itself run are silently skipped — the orchestrator is responsible
///      for cluster-wide consistency, the node only enforces what it owns.
///
///   4. <c>DepsMet</c> = every applicable dependency reports
///      <see cref="Reality.Healthy"/> via <see cref="IRealityProjection"/>.
///      With no applicable dependencies, this is vacuously <c>true</c>.
///
///   5. Return <c>Intent { WantDeployed = true, DepsMet = ... }</c>.
///
/// Logging discipline: hot path. No per-call Debug logging. The matrix logs
/// its decisions at the call site; this projection is silent unless the
/// dependency list contains an unknown role name (Warning).
/// </summary>
public sealed class IntentComputation : IIntentComputation
{
    private readonly IRealityProjection _reality;
    private readonly ILogger<IntentComputation> _logger;

    public IntentComputation(
        IRealityProjection reality,
        ILogger<IntentComputation> logger)
    {
        _reality = reality;
        _logger = logger;
    }

    /// <inheritdoc/>
    public Intent Compute(string role, IReadOnlyList<ObligationDescriptor> allObligations)
    {
        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
            return Intent.None;

        // Find this role in the obligation list. The list is small (≤ 4
        // entries in practice) so a linear scan is the right tool.
        ObligationDescriptor? mine = null;
        foreach (var o in allObligations)
        {
            if (string.Equals(o.Role, canonical, StringComparison.OrdinalIgnoreCase))
            {
                mine = o;
                break;
            }
        }

        if (mine is null)
            return Intent.None;

        // We hold an obligation for this role. Now check applicable dependencies.
        var depsMet = AreApplicableDepsHealthy(canonical, mine.Deps, allObligations);

        return new Intent
        {
            WantDeployed = true,
            DepsMet = depsMet,
        };
    }

    /// <summary>
    /// Returns <c>true</c> iff every dependency in <paramref name="deps"/>
    /// that is *also* present in <paramref name="allObligations"/> projects
    /// to <see cref="Reality.Healthy"/>. Dependencies not held locally are
    /// skipped; an empty applicable set is vacuously satisfied.
    /// </summary>
    private bool AreApplicableDepsHealthy(
        string role,
        IReadOnlyList<string> deps,
        IReadOnlyList<ObligationDescriptor> allObligations)
    {
        if (deps.Count == 0)
            return true;

        foreach (var dep in deps)
        {
            var depCanonical = ObligationRole.Canonicalise(dep);
            if (depCanonical is null)
            {
                // The orchestrator should never push an unknown dep, but if
                // it does, log it and treat the dep as not applicable —
                // safer than assuming healthy or stalling forever.
                _logger.LogWarning(
                    "IntentComputation [{Role}]: ignoring unknown dependency '{Dep}' " +
                    "in pushed obligation descriptor — treating as not applicable",
                    role, dep);
                continue;
            }

            // Applicable only if this node also holds an obligation for the dep.
            var isApplicable = false;
            foreach (var o in allObligations)
            {
                if (string.Equals(o.Role, depCanonical, StringComparison.OrdinalIgnoreCase))
                {
                    isApplicable = true;
                    break;
                }
            }
            if (!isApplicable)
                continue;

            // Applicable dep — must be Healthy.
            var depReality = _reality.Project(depCanonical);
            if (depReality.State != Reality.Healthy)
                return false;
        }

        return true;
    }
}