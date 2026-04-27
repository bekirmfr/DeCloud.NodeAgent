using DeCloud.NodeAgent.Core.Models.Reality;

namespace DeCloud.NodeAgent.Core.Interfaces.SystemVm;

/// <summary>
/// Projects the current reality of a system VM role on this node, by reading
/// the in-memory VM list maintained by <see cref="IVmManager"/>.
///
/// This is the matrix's <c>reality</c> axis (SYSTEM_VM_DESIGN.md §5.2).
/// Pure read — never mutates state, never performs I/O beyond what
/// <see cref="IVmManager.GetAllVms"/> requires (which is itself an in-memory
/// dictionary lookup).
///
/// Synchronous on purpose — the underlying VM list is in-process state,
/// updated continuously by libvirt reconciliation and the readiness monitor.
/// Making the projection async would imply a wait that doesn't exist.
/// </summary>
public interface IRealityProjection
{
    /// <summary>
    /// Compute the reality snapshot for a single role.
    /// </summary>
    /// <param name="role">
    /// Canonical role name (case-insensitive — canonicalised internally):
    /// "relay" | "dht" | "blockstore". Unknown role names produce
    /// <see cref="RealitySnapshot.None"/> rather than throwing — the matrix
    /// treats "no VM" and "unknown role" identically.
    /// </param>
    /// <returns>
    /// A snapshot describing the current state. Never <c>null</c>.
    /// </returns>
    RealitySnapshot Project(string role);
}