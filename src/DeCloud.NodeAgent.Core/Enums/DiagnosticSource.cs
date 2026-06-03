namespace DeCloud.NodeAgent.Core.Enums
{
    /// <summary>
    /// Allowlist of diagnostic streams the host can return. Closed enum so a
    /// malicious caller cannot ask the node agent to read arbitrary paths.
    /// </summary>
    public enum DiagnosticSource
    {
        /// <summary>
        /// Host-side <c>console.log</c> written by libvirt's <c>&lt;log&gt;</c>
        /// on the serial chardev. Always readable — no guest cooperation.
        /// </summary>
        Console = 0,

        /// <summary>
        /// Guest <c>/var/log/cloud-init.log</c>. Requires guest agent. Not yet
        /// implemented — reserved for the cloud-init log follow-up.
        /// </summary>
        CloudInitLog = 1,

        /// <summary>
        /// Guest <c>/var/log/cloud-init-output.log</c>. Requires guest agent.
        /// Not yet implemented.
        /// </summary>
        CloudInitOutputLog = 2,

        /// <summary>
        /// Guest <c>journalctl</c> output. Requires guest agent. Not yet
        /// implemented; today this path lives in
        /// <c>SystemVmService.CaptureVmJournalAsync</c>.
        /// </summary>
        Journal = 3
    }
}
