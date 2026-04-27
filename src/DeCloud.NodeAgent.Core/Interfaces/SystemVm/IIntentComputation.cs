using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Models.State;

namespace DeCloud.NodeAgent.Core.Interfaces.SystemVm;

/// <summary>
/// Projects the matrix's <c>intent</c> axis (SYSTEM_VM_DESIGN.md §5.2) for a
/// single role: does this node hold an obligation for it, and are the
/// dependencies satisfied?
///
/// Pure synchronous function. The caller (the reconciliation matrix) fetches
/// the obligation list once per cycle via
/// <see cref="IObligationStateService.GetObligationsAsync"/> and passes it in,
/// so this projection performs no I/O of its own — it only consults
/// <see cref="IRealityProjection"/> to evaluate dependency health.
/// </summary>
public interface IIntentComputation
{
    /// <summary>
    /// Compute intent for <paramref name="role"/>.
    /// </summary>
    /// <param name="role">
    /// Canonical role name (case-insensitive — canonicalised internally).
    /// Unknown roles produce <see cref="Intent.None"/>.
    /// </param>
    /// <param name="allObligations">
    /// The full obligation list the matrix is currently iterating over.
    /// Used to (a) find this role's descriptor and (b) determine which
    /// dependencies are applicable on this node.
    /// </param>
    /// <returns>
    /// An <see cref="Intent"/> describing whether the node should deploy
    /// this role and whether its dependencies are presently satisfied.
    /// Never <c>null</c>.
    /// </returns>
    Intent Compute(string role, IReadOnlyList<ObligationDescriptor> allObligations);
}