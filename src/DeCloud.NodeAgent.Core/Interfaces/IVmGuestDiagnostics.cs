using DeCloud.NodeAgent.Core.Enums;
using DeCloud.NodeAgent.Core.Models;

namespace DeCloud.NodeAgent.Core.Interfaces
{

    /// <summary>
    /// Captures diagnostic streams from a VM — console output, cloud-init logs,
    /// systemd journal — using the cheapest mechanism available for each source.
    ///
    /// The interface is forward-compatible: more sources land in follow-up
    /// commits. The first cut implements <see cref="DiagnosticSource.Console"/>
    /// because it is the only source that works when the qemu-guest-agent never
    /// starts (cloud-init parse error, apt failure, network never up, kernel
    /// panic) — the exact failure modes the user needs visibility into.
    /// </summary>
    public interface IVmGuestDiagnostics
    {
        /// <summary>
        /// Return the tail of the named diagnostic source, capped at
        /// <paramref name="maxBytes"/>. Returns <c>Available=false</c> rather
        /// than throwing when the source does not exist (e.g. console.log absent
        /// because the VM was created before the console-capture feature shipped,
        /// or guest-agent unreachable for a guest-side source).
        /// </summary>
        Task<DiagnosticsResult> CaptureAsync(
            string vmId,
            DiagnosticSource source,
            int maxBytes,
            CancellationToken ct = default);
    }
}
