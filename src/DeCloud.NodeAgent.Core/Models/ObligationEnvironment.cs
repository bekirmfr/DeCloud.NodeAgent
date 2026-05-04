using System.Text.Json.Serialization;

namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Response payload for <c>GET /api/obligations/{role}/environment</c>.
///
/// Consumed by <c>decloud-env-watcher.sh</c> inside running system VMs.
/// The watcher diffs <c>Values</c> against its local environment file,
/// applies the max-scope reaction across changed variables, and stores the
/// new <c>Generation</c> to detect future changes cheaply.
///
/// Wire shape (JSON):
/// <code>
/// {
///   "values":     { "VARNAME": "value", ... },
///   "scopes":     { "VARNAME": "noop|reload|restart", ... },
///   "generation": "abc123..."
/// }
/// </code>
/// </summary>
public sealed record ObligationEnvironment
{
    /// <summary>
    /// Current values of all Dynamic variables declared in the role's template.
    /// Keys are variable names as they appear in the cloud-init template.
    /// Empty when the role declares no Dynamic variables (e.g. relay in Phase 3).
    /// </summary>
    [JsonPropertyName("values")]
    public IReadOnlyDictionary<string, string> Values { get; init; }
        = new Dictionary<string, string>();

    /// <summary>
    /// Watcher scope policy per variable. Keys match <c>Values</c>.
    /// Values: "noop" | "reload" | "restart".
    /// </summary>
    [JsonPropertyName("scopes")]
    public IReadOnlyDictionary<string, string> Scopes { get; init; }
        = new Dictionary<string, string>();

    /// <summary>
    /// Deterministic SHA256-based fingerprint of <c>Values</c>.
    /// Computed by sorting keys, serialising to compact JSON, and taking the
    /// first 16 hex chars of the SHA256 digest. The watcher compares this
    /// against its locally-cached generation to skip the per-variable diff
    /// when nothing has changed.
    /// </summary>
    [JsonPropertyName("generation")]
    public string Generation { get; init; } = string.Empty;
}