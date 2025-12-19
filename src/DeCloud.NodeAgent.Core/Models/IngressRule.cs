using System.Text.Json.Serialization;

namespace DeCloud.NodeAgent.Core.Models;

/// <summary>
/// Represents an ingress rule that routes external traffic to a VM.
/// Supports HTTP/HTTPS with automatic TLS via Let's Encrypt.
/// </summary>
public class IngressRule
{
    /// <summary>
    /// Unique identifier for this ingress rule
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// The VM this ingress routes to
    /// </summary>
    public string VmId { get; set; } = string.Empty;

    /// <summary>
    /// Owner wallet address (for authorization)
    /// </summary>
    public string OwnerWallet { get; set; } = string.Empty;

    /// <summary>
    /// The domain name (e.g., "myapp.example.com")
    /// Must be a valid FQDN that the user controls
    /// </summary>
    public string Domain { get; set; } = string.Empty;

    /// <summary>
    /// Target port on the VM (e.g., 80, 3000, 8080)
    /// </summary>
    public int TargetPort { get; set; } = 80;

    /// <summary>
    /// Target protocol for backend connection
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public IngressProtocol TargetProtocol { get; set; } = IngressProtocol.Http;

    /// <summary>
    /// Whether to enable automatic TLS via Let's Encrypt
    /// </summary>
    public bool EnableTls { get; set; } = true;

    /// <summary>
    /// Whether to force HTTPS redirect
    /// </summary>
    public bool ForceHttps { get; set; } = true;

    /// <summary>
    /// Whether to enable HTTP/2
    /// </summary>
    public bool EnableHttp2 { get; set; } = true;

    /// <summary>
    /// Whether to enable WebSocket support
    /// </summary>
    public bool EnableWebSocket { get; set; } = true;

    /// <summary>
    /// Optional path prefix (e.g., "/api" routes only /api/* requests)
    /// Empty means route all paths
    /// </summary>
    public string PathPrefix { get; set; } = string.Empty;

    /// <summary>
    /// Strip the path prefix before forwarding to backend
    /// </summary>
    public bool StripPathPrefix { get; set; } = false;

    /// <summary>
    /// Custom headers to add to proxied requests
    /// </summary>
    public Dictionary<string, string> CustomHeaders { get; set; } = new();

    /// <summary>
    /// Rate limiting: requests per minute (0 = disabled)
    /// </summary>
    public int RateLimitPerMinute { get; set; } = 0;

    /// <summary>
    /// IP whitelist (empty = allow all)
    /// </summary>
    public List<string> AllowedIps { get; set; } = new();

    /// <summary>
    /// Current status of the ingress rule
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public IngressStatus Status { get; set; } = IngressStatus.Pending;

    /// <summary>
    /// Status message (error details, etc.)
    /// </summary>
    public string? StatusMessage { get; set; }

    /// <summary>
    /// TLS certificate status
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public TlsCertStatus TlsStatus { get; set; } = TlsCertStatus.Pending;

    /// <summary>
    /// When the TLS certificate expires (if provisioned)
    /// </summary>
    public DateTime? TlsExpiresAt { get; set; }

    /// <summary>
    /// When this rule was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// When this rule was last updated
    /// </summary>
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// When Caddy config was last reloaded for this rule
    /// </summary>
    public DateTime? LastReloadAt { get; set; }

    /// <summary>
    /// Total requests handled (updated periodically)
    /// </summary>
    public long TotalRequests { get; set; }

    /// <summary>
    /// Total bytes transferred (updated periodically)
    /// </summary>
    public long TotalBytesTransferred { get; set; }

    /// <summary>
    /// The VM's private IP address (cached for routing)
    /// </summary>
    public string? VmPrivateIp { get; set; }
}

public enum IngressProtocol
{
    Http,
    Https,
    Tcp,
    Udp
}

public enum IngressStatus
{
    /// <summary>
    /// Rule created, waiting for configuration
    /// </summary>
    Pending,

    /// <summary>
    /// Caddy is being configured
    /// </summary>
    Configuring,

    /// <summary>
    /// Rule is active and routing traffic
    /// </summary>
    Active,

    /// <summary>
    /// Rule is paused (not routing)
    /// </summary>
    Paused,

    /// <summary>
    /// Configuration failed
    /// </summary>
    Error,

    /// <summary>
    /// Rule is being deleted
    /// </summary>
    Deleting,

    /// <summary>
    /// Rule has been deleted
    /// </summary>
    Deleted
}

public enum TlsCertStatus
{
    /// <summary>
    /// TLS not requested
    /// </summary>
    Disabled,

    /// <summary>
    /// Waiting for certificate provisioning
    /// </summary>
    Pending,

    /// <summary>
    /// Certificate is being provisioned
    /// </summary>
    Provisioning,

    /// <summary>
    /// Certificate is valid and active
    /// </summary>
    Valid,

    /// <summary>
    /// Certificate is expiring soon (within 30 days)
    /// </summary>
    ExpiringSoon,

    /// <summary>
    /// Certificate has expired
    /// </summary>
    Expired,

    /// <summary>
    /// Certificate provisioning failed
    /// </summary>
    Failed
}

#region DTOs

public record CreateIngressRequest(
    string VmId,
    string Domain,
    int TargetPort = 80,
    bool EnableTls = true,
    bool ForceHttps = true,
    bool EnableWebSocket = true,
    string? PathPrefix = null,
    bool StripPathPrefix = false,
    int RateLimitPerMinute = 0
);

public record UpdateIngressRequest(
    int? TargetPort = null,
    bool? EnableTls = null,
    bool? ForceHttps = null,
    bool? EnableWebSocket = null,
    string? PathPrefix = null,
    bool? StripPathPrefix = null,
    int? RateLimitPerMinute = null,
    List<string>? AllowedIps = null,
    Dictionary<string, string>? CustomHeaders = null
);

public record IngressResponse(
    string Id,
    string VmId,
    string Domain,
    int TargetPort,
    bool EnableTls,
    IngressStatus Status,
    string? StatusMessage,
    TlsCertStatus TlsStatus,
    DateTime? TlsExpiresAt,
    string? PublicUrl,
    DateTime CreatedAt,
    DateTime UpdatedAt,
    long TotalRequests
);

public record IngressOperationResult(
    bool Success,
    string? IngressId = null,
    string? Error = null,
    IngressResponse? Ingress = null
)
{
    public static IngressOperationResult Ok(IngressRule rule) => new(
        Success: true,
        IngressId: rule.Id,
        Ingress: ToResponse(rule)
    );

    public static IngressOperationResult Fail(string error) => new(
        Success: false,
        Error: error
    );

    private static IngressResponse ToResponse(IngressRule rule) => new(
        Id: rule.Id,
        VmId: rule.VmId,
        Domain: rule.Domain,
        TargetPort: rule.TargetPort,
        EnableTls: rule.EnableTls,
        Status: rule.Status,
        StatusMessage: rule.StatusMessage,
        TlsStatus: rule.TlsStatus,
        TlsExpiresAt: rule.TlsExpiresAt,
        PublicUrl: rule.EnableTls ? $"https://{rule.Domain}" : $"http://{rule.Domain}",
        CreatedAt: rule.CreatedAt,
        UpdatedAt: rule.UpdatedAt,
        TotalRequests: rule.TotalRequests
    );
}

#endregion