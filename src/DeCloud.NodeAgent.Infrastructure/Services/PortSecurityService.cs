using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Configuration for port validation
/// </summary>
public class PortSecurityOptions
{
    /// <summary>
    /// Minimum allowed target port for ingress rules
    /// </summary>
    public int MinAllowedPort { get; set; } = 1;

    /// <summary>
    /// Maximum allowed target port
    /// </summary>
    public int MaxAllowedPort { get; set; } = 65535;

    /// <summary>
    /// Ports that should not be exposed via ingress (security-sensitive)
    /// </summary>
    public List<int> BlockedPorts { get; set; } = new()
    {
        // SSH - use SSH tunnels instead
        22,
        2222,
        // Databases - should never be exposed publicly
        3306,   // MySQL
        5432,   // PostgreSQL
        27017,  // MongoDB
        27018,
        27019,
        6379,   // Redis
        9200,   // Elasticsearch
        9300,
        11211,  // Memcached
        // Messaging
        5672,   // RabbitMQ
        15672,
        9092,   // Kafka
        // Infrastructure
        2375,   // Docker
        2376,
        2379,   // etcd
        2380,
        6443,   // Kubernetes API
        10250,
        10251,
        10252,
        // DeCloud internal
        5100,   // NodeAgent API
        51820,  // WireGuard
        2019,   // Caddy Admin API
        16509   // libvirt
    };
}

/// <summary>
/// Interface for port validation
/// </summary>
public interface IPortSecurityService
{
    /// <summary>
    /// Validate if a port can be used as ingress target
    /// </summary>
    PortValidationResult ValidateTargetPort(int port);

    /// <summary>
    /// Get list of blocked ports for documentation
    /// </summary>
    IReadOnlyList<int> GetBlockedPorts();
}

/// <summary>
/// Validates ports for ingress target rules.
/// Prevents exposing security-sensitive services via HTTP ingress.
/// </summary>
public class PortSecurityService : IPortSecurityService
{
    private readonly PortSecurityOptions _options;
    private readonly ILogger<PortSecurityService> _logger;

    public PortSecurityService(
        IOptions<PortSecurityOptions> options,
        ILogger<PortSecurityService> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public PortValidationResult ValidateTargetPort(int port)
    {
        // Check range
        if (port < _options.MinAllowedPort || port > _options.MaxAllowedPort)
        {
            return PortValidationResult.Invalid(
                $"Port must be between {_options.MinAllowedPort} and {_options.MaxAllowedPort}");
        }

        // Check blocked list
        if (_options.BlockedPorts.Contains(port))
        {
            _logger.LogWarning("Blocked ingress target port attempted: {Port}", port);
            return PortValidationResult.Invalid(
                $"Port {port} cannot be exposed via ingress for security reasons. " +
                "Use SSH tunnels for database access.");
        }

        return PortValidationResult.Valid();
    }

    public IReadOnlyList<int> GetBlockedPorts() => _options.BlockedPorts.AsReadOnly();
}

public class PortValidationResult
{
    public bool IsValid { get; init; }
    public string? Message { get; init; }

    public static PortValidationResult Valid() => new() { IsValid = true };

    public static PortValidationResult Invalid(string message) => new()
    {
        IsValid = false,
        Message = message
    };
}