using DeCloud.NodeAgent.Core.Enums;

namespace DeCloud.NodeAgent.Core.Models
{
    public record DiagnosticsResult
    {
        public DiagnosticSource Source { get; init; }
        public bool Available { get; init; }
        public string? Content { get; init; }
        public int CapturedBytes { get; init; }
        public long? TotalBytes { get; init; }
        public bool Truncated { get; init; }
        public string? Message { get; init; }

        public static DiagnosticsResult Captured(
            DiagnosticSource source,
            string content,
            long totalBytes,
            bool truncated) => new()
            {
                Source = source,
                Available = true,
                Content = content,
                CapturedBytes = content.Length,
                TotalBytes = totalBytes,
                Truncated = truncated
            };

        public static DiagnosticsResult Unavailable(
            DiagnosticSource source,
            string message) => new()
            {
                Source = source,
                Available = false,
                Message = message
            };
    }
}
