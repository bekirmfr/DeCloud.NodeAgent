using DeCloud.NodeAgent.Core.Enums;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Infrastructure.Services
{
    public class VmGuestDiagnostics : IVmGuestDiagnostics
    {
        private readonly LibvirtVmManagerOptions _options;
        private readonly ILogger<VmGuestDiagnostics> _logger;

        // 4 MB ceiling protects against a caller asking for the whole log of a
        // long-running VM with append='on'. A normal successful Debian boot is
        // ~3 KB on the wire; cloud-init stack traces ~80 KB. 4 MB is generous.
        private const int MaxBytesAbsoluteCeiling = 4 * 1024 * 1024;
        private const int DefaultMaxBytes = 256 * 1024;

        public VmGuestDiagnostics(
            IOptions<LibvirtVmManagerOptions> options,
            ILogger<VmGuestDiagnostics> logger)
        {
            _options = options.Value;
            _logger = logger;
        }

        public async Task<DiagnosticsResult> CaptureAsync(
            string vmId,
            DiagnosticSource source,
            int maxBytes,
            CancellationToken ct = default)
        {
            // Clamp the read size — caller may pass 0 (use default) or a huge
            // value; both collapse to the safe range here.
            if (maxBytes <= 0) maxBytes = DefaultMaxBytes;
            if (maxBytes > MaxBytesAbsoluteCeiling) maxBytes = MaxBytesAbsoluteCeiling;

            return source switch
            {
                DiagnosticSource.Console => await ReadConsoleLogAsync(vmId, maxBytes, ct),

                // Guest-agent-dependent sources land in follow-up commits behind
                // the same interface — callers don't have to change.
                DiagnosticSource.CloudInitLog
                    or DiagnosticSource.CloudInitOutputLog
                    or DiagnosticSource.Journal => DiagnosticsResult.Unavailable(
                        source,
                        $"Source '{source}' not yet implemented on this node agent"),

                _ => DiagnosticsResult.Unavailable(source, "Unknown diagnostic source")
            };
        }

        /// <summary>
        /// Read the tail of the per-VM <c>console.log</c> written by libvirt's
        /// <c>&lt;log file='...'/&gt;</c> directive on the serial chardev.
        /// Survives guest-agent failure because it is a host-side file owned by
        /// libvirt-qemu — the exact case the feature was added for.
        /// </summary>
        private async Task<DiagnosticsResult> ReadConsoleLogAsync(
            string vmId,
            int maxBytes,
            CancellationToken ct)
        {
            // Every real DeCloud VM ID is a libvirt UUID (see CreateVmAsync —
            // spec.Id is the libvirt domain UUID). Reject anything else: it is
            // either a bug or a path-traversal attempt walking outside vmDir.
            if (!Guid.TryParse(vmId, out _))
            {
                _logger.LogWarning(
                    "Console log read rejected: vmId '{VmId}' is not a valid UUID",
                    vmId);
                return DiagnosticsResult.Unavailable(
                    DiagnosticSource.Console, "Invalid vmId format");
            }

            var path = Path.Combine(_options.VmStoragePath, vmId, "console.log");

            if (!File.Exists(path))
            {
                return DiagnosticsResult.Unavailable(
                    DiagnosticSource.Console,
                    "console.log not present — the VM may have been created before " +
                    "the console-capture feature shipped, or the log has been " +
                    "removed during cleanup");
            }

            try
            {
                // QEMU holds this file open in append mode. FileShare.ReadWrite
                // | Delete lets us read concurrently and survives rotation /
                // delete-while-open on Linux.
                await using var stream = new FileStream(
                    path,
                    FileMode.Open,
                    FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete);

                var totalBytes = stream.Length;
                var truncated = totalBytes > maxBytes;
                var readFrom = truncated ? totalBytes - maxBytes : 0;
                var toRead = (int)(totalBytes - readFrom);

                stream.Seek(readFrom, SeekOrigin.Begin);

                var buffer = new byte[toRead];
                var bytesRead = await stream.ReadAsync(buffer.AsMemory(0, toRead), ct);

                // Console output sometimes contains kernel \x escape codes or
                // partial multibyte sequences at the cut point. UTF-8 with
                // replacement keeps the rest readable instead of throwing.
                var content = System.Text.Encoding.UTF8.GetString(
                    buffer, 0, bytesRead);

                // If we truncated, drop the first (possibly partial) line so the
                // output starts at a clean line boundary — easier to read.
                if (truncated)
                {
                    var firstNewline = content.IndexOf('\n');
                    if (firstNewline >= 0 && firstNewline < content.Length - 1)
                        content = content[(firstNewline + 1)..];
                }

                return DiagnosticsResult.Captured(
                    DiagnosticSource.Console, content, totalBytes, truncated);
            }
            catch (UnauthorizedAccessException ex)
            {
                // libvirt-qemu owns the file; an unprivileged node agent process
                // would hit this. Surface it clearly rather than pretending the
                // file does not exist — that would send the operator chasing the
                // wrong root cause.
                _logger.LogWarning(ex,
                    "Console log read denied for VM {VmId} at {Path}", vmId, path);
                return DiagnosticsResult.Unavailable(
                    DiagnosticSource.Console,
                    "Permission denied reading console log");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "Console log read failed for VM {VmId} at {Path}", vmId, path);
                return DiagnosticsResult.Unavailable(
                    DiagnosticSource.Console, $"Read error: {ex.Message}");
            }
        }
    }
}
