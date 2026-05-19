using System.Text.Json;
using System.Text.Json.Serialization;

namespace DeCloud.NodeAgent.Core.Json;

/// <summary>
/// Platform-wide JSON serialization options. Use these instead of creating
/// per-call options — ensures consistent casing and null handling everywhere.
///
/// Two profiles:
///   Default — human-readable (files on disk, diagnostics, metadata.json)
///   Wire    — compact (API responses, heartbeat payloads, state persistence)
/// </summary>
public static class JsonOptions
{
    /// <summary>
    /// Human-readable: indented, camelCase, case-insensitive, skip nulls.
    /// Use for files written to disk (metadata.json, diagnostics exports).
    /// </summary>
    public static readonly JsonSerializerOptions Default = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true
    };

    /// <summary>
    /// Compact: same semantics as Default but no indentation.
    /// Use for API payloads, state files, wire format.
    /// </summary>
    public static readonly JsonSerializerOptions Wire = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false
    };
}