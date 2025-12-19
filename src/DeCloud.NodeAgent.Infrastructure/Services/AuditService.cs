using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Configuration for security audit logging
/// </summary>
public class AuditLogOptions
{
    /// <summary>
    /// Enable audit logging
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Path to audit log file
    /// </summary>
    public string LogPath { get; set; } = "/var/log/decloud/audit.log";

    /// <summary>
    /// Maximum log file size in MB before rotation
    /// </summary>
    public int MaxFileSizeMb { get; set; } = 100;

    /// <summary>
    /// Number of rotated files to keep
    /// </summary>
    public int MaxFiles { get; set; } = 10;

    /// <summary>
    /// Log retention in days
    /// </summary>
    public int RetentionDays { get; set; } = 90;

    /// <summary>
    /// Include request details (may contain sensitive data)
    /// </summary>
    public bool IncludeRequestDetails { get; set; } = false;

    /// <summary>
    /// Actions to audit
    /// </summary>
    public List<AuditAction> AuditedActions { get; set; } = new()
    {
        AuditAction.IngressCreated,
        AuditAction.IngressDeleted,
        AuditAction.IngressUpdated,
        AuditAction.VmCreated,
        AuditAction.VmDeleted,
        AuditAction.VmStarted,
        AuditAction.VmStopped,
        AuditAction.AuthSuccess,
        AuditAction.AuthFailure,
        AuditAction.SecurityViolation
    };
}

/// <summary>
/// Audit actions that can be logged
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter))]
public enum AuditAction
{
    // Ingress
    IngressCreated,
    IngressDeleted,
    IngressUpdated,
    IngressPaused,
    IngressResumed,

    // VM lifecycle
    VmCreated,
    VmDeleted,
    VmStarted,
    VmStopped,

    // Authentication
    AuthSuccess,
    AuthFailure,
    TokenGenerated,
    TokenRevoked,

    // Security events
    SecurityViolation,
    RateLimitExceeded,
    BlockedPortAttempt,
    UnauthorizedAccess,
    SuspiciousActivity
}

/// <summary>
/// Audit log entry
/// </summary>
public class AuditEntry
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public AuditAction Action { get; set; }

    public string? UserId { get; set; }
    public string? WalletAddress { get; set; }
    public string? VmId { get; set; }
    public string? ResourceId { get; set; }
    public string? ResourceType { get; set; }
    public string? SourceIp { get; set; }
    public string? UserAgent { get; set; }
    public bool Success { get; set; } = true;
    public string? ErrorMessage { get; set; }
    public Dictionary<string, object?> Details { get; set; } = new();

    /// <summary>
    /// Severity level
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public AuditSeverity Severity { get; set; } = AuditSeverity.Info;
}

public enum AuditSeverity
{
    Debug,
    Info,
    Warning,
    Error,
    Critical
}

/// <summary>
/// Interface for audit logging
/// </summary>
public interface IAuditService
{
    /// <summary>
    /// Log an audit event
    /// </summary>
    Task LogAsync(AuditEntry entry);

    /// <summary>
    /// Log an audit event with builder
    /// </summary>
    Task LogAsync(AuditAction action, Action<AuditEntry> configure);

    /// <summary>
    /// Query audit logs
    /// </summary>
    Task<List<AuditEntry>> QueryAsync(AuditQuery query, CancellationToken ct = default);

    /// <summary>
    /// Get recent security events
    /// </summary>
    Task<List<AuditEntry>> GetSecurityEventsAsync(int count = 100, CancellationToken ct = default);
}

/// <summary>
/// Security audit logging service.
/// Provides comprehensive logging for all security-relevant events.
/// </summary>
public class AuditService : IAuditService, IDisposable
{
    private readonly AuditLogOptions _options;
    private readonly ILogger<AuditService> _logger;
    private readonly ConcurrentQueue<AuditEntry> _writeQueue = new();
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly Timer _flushTimer;
    private StreamWriter? _writer;
    private long _currentFileSize;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public AuditService(
        IOptions<AuditLogOptions> options,
        ILogger<AuditService> logger)
    {
        _options = options.Value;
        _logger = logger;

        if (_options.Enabled)
        {
            InitializeLogFile();
            _flushTimer = new Timer(FlushCallback, null, TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(5));
        }
        else
        {
            _flushTimer = new Timer(_ => { }, null, Timeout.Infinite, Timeout.Infinite);
        }
    }

    public Task LogAsync(AuditEntry entry)
    {
        if (!_options.Enabled) return Task.CompletedTask;
        if (!_options.AuditedActions.Contains(entry.Action)) return Task.CompletedTask;

        _writeQueue.Enqueue(entry);

        // Log to standard logger as well for critical events
        if (entry.Severity >= AuditSeverity.Warning)
        {
            var logLevel = entry.Severity switch
            {
                AuditSeverity.Warning => LogLevel.Warning,
                AuditSeverity.Error => LogLevel.Error,
                AuditSeverity.Critical => LogLevel.Critical,
                _ => LogLevel.Information
            };

            _logger.Log(logLevel,
                "AUDIT: {Action} | User: {User} | VM: {VmId} | Success: {Success} | {Message}",
                entry.Action, entry.WalletAddress ?? entry.UserId, entry.VmId,
                entry.Success, entry.ErrorMessage ?? "OK");
        }

        return Task.CompletedTask;
    }

    public Task LogAsync(AuditAction action, Action<AuditEntry> configure)
    {
        var entry = new AuditEntry { Action = action };
        configure(entry);
        return LogAsync(entry);
    }

    public async Task<List<AuditEntry>> QueryAsync(AuditQuery query, CancellationToken ct = default)
    {
        var results = new List<AuditEntry>();

        if (!File.Exists(_options.LogPath))
        {
            return results;
        }

        // Read and filter log file
        // Note: For production, consider using a proper log aggregation system
        await foreach (var line in File.ReadLinesAsync(_options.LogPath, ct))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            try
            {
                var entry = JsonSerializer.Deserialize<AuditEntry>(line, JsonOptions);
                if (entry == null) continue;

                // Apply filters
                if (query.StartTime.HasValue && entry.Timestamp < query.StartTime.Value)
                    continue;
                if (query.EndTime.HasValue && entry.Timestamp > query.EndTime.Value)
                    continue;
                if (query.Actions?.Count > 0 && !query.Actions.Contains(entry.Action))
                    continue;
                if (!string.IsNullOrEmpty(query.UserId) && entry.UserId != query.UserId)
                    continue;
                if (!string.IsNullOrEmpty(query.VmId) && entry.VmId != query.VmId)
                    continue;
                if (!string.IsNullOrEmpty(query.WalletAddress) &&
                    !string.Equals(entry.WalletAddress, query.WalletAddress, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (query.SuccessOnly.HasValue && entry.Success != query.SuccessOnly.Value)
                    continue;
                if (query.MinSeverity.HasValue && entry.Severity < query.MinSeverity.Value)
                    continue;

                results.Add(entry);

                if (query.Limit.HasValue && results.Count >= query.Limit.Value)
                    break;
            }
            catch (JsonException)
            {
                // Skip malformed entries
            }
        }

        return results;
    }

    public Task<List<AuditEntry>> GetSecurityEventsAsync(int count = 100, CancellationToken ct = default)
    {
        return QueryAsync(new AuditQuery
        {
            Actions = new List<AuditAction>
            {
                AuditAction.AuthFailure,
                AuditAction.SecurityViolation,
                AuditAction.RateLimitExceeded,
                AuditAction.BlockedPortAttempt,
                AuditAction.UnauthorizedAccess,
                AuditAction.SuspiciousActivity
            },
            Limit = count,
            MinSeverity = AuditSeverity.Warning
        }, ct);
    }

    private void InitializeLogFile()
    {
        try
        {
            var directory = Path.GetDirectoryName(_options.LogPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            _writer = new StreamWriter(_options.LogPath, append: true)
            {
                AutoFlush = false
            };

            _currentFileSize = new FileInfo(_options.LogPath).Length;

            _logger.LogInformation("Audit log initialized: {Path}", _options.LogPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize audit log");
        }
    }

    private void FlushCallback(object? state)
    {
        _ = FlushAsync();
    }

    private async Task FlushAsync()
    {
        if (!_writeQueue.IsEmpty)
        {
            await _writeLock.WaitAsync();
            try
            {
                while (_writeQueue.TryDequeue(out var entry))
                {
                    var json = JsonSerializer.Serialize(entry, JsonOptions);
                    _writer?.WriteLine(json);
                    _currentFileSize += json.Length + Environment.NewLine.Length;
                }

                _writer?.Flush();

                // Check for rotation
                if (_currentFileSize > _options.MaxFileSizeMb * 1024 * 1024)
                {
                    await RotateLogAsync();
                }
            }
            finally
            {
                _writeLock.Release();
            }
        }
    }

    private async Task RotateLogAsync()
    {
        _writer?.Close();
        _writer?.Dispose();

        // Rotate files
        for (int i = _options.MaxFiles - 1; i >= 1; i--)
        {
            var oldFile = $"{_options.LogPath}.{i}";
            var newFile = $"{_options.LogPath}.{i + 1}";

            if (File.Exists(oldFile))
            {
                if (i == _options.MaxFiles - 1)
                {
                    File.Delete(oldFile);
                }
                else
                {
                    File.Move(oldFile, newFile, overwrite: true);
                }
            }
        }

        if (File.Exists(_options.LogPath))
        {
            File.Move(_options.LogPath, $"{_options.LogPath}.1");
        }

        // Create new file
        _writer = new StreamWriter(_options.LogPath, append: false)
        {
            AutoFlush = false
        };
        _currentFileSize = 0;

        _logger.LogInformation("Audit log rotated");
    }

    public void Dispose()
    {
        _flushTimer?.Dispose();
        FlushAsync().GetAwaiter().GetResult();
        _writer?.Dispose();
        _writeLock?.Dispose();
    }
}

/// <summary>
/// Query parameters for audit log search
/// </summary>
public class AuditQuery
{
    public DateTime? StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public List<AuditAction>? Actions { get; set; }
    public string? UserId { get; set; }
    public string? VmId { get; set; }
    public string? WalletAddress { get; set; }
    public bool? SuccessOnly { get; set; }
    public AuditSeverity? MinSeverity { get; set; }
    public int? Limit { get; set; }
}

/// <summary>
/// Extension methods for audit logging in services
/// </summary>
public static class AuditExtensions
{
    public static Task LogIngressCreatedAsync(this IAuditService audit,
        string vmId, string domain, string walletAddress, string? sourceIp = null)
    {
        return audit.LogAsync(AuditAction.IngressCreated, e =>
        {
            e.VmId = vmId;
            e.WalletAddress = walletAddress;
            e.SourceIp = sourceIp;
            e.ResourceType = "Ingress";
            e.Details["domain"] = domain;
            e.Severity = AuditSeverity.Info;
        });
    }

    public static Task LogIngressDeletedAsync(this IAuditService audit,
        string vmId, string domain, string walletAddress, string? sourceIp = null)
    {
        return audit.LogAsync(AuditAction.IngressDeleted, e =>
        {
            e.VmId = vmId;
            e.WalletAddress = walletAddress;
            e.SourceIp = sourceIp;
            e.ResourceType = "Ingress";
            e.Details["domain"] = domain;
            e.Severity = AuditSeverity.Info;
        });
    }

    public static Task LogBlockedPortAttemptAsync(this IAuditService audit,
        int port, string reason, string walletAddress, string? sourceIp = null)
    {
        return audit.LogAsync(AuditAction.BlockedPortAttempt, e =>
        {
            e.WalletAddress = walletAddress;
            e.SourceIp = sourceIp;
            e.Success = false;
            e.ErrorMessage = reason;
            e.Details["port"] = port;
            e.Severity = AuditSeverity.Warning;
        });
    }

    public static Task LogSecurityViolationAsync(this IAuditService audit,
        string description, string? walletAddress = null, string? sourceIp = null,
        Dictionary<string, object?>? details = null)
    {
        return audit.LogAsync(AuditAction.SecurityViolation, e =>
        {
            e.WalletAddress = walletAddress;
            e.SourceIp = sourceIp;
            e.Success = false;
            e.ErrorMessage = description;
            e.Severity = AuditSeverity.Critical;
            if (details != null)
            {
                foreach (var kvp in details)
                {
                    e.Details[kvp.Key] = kvp.Value;
                }
            }
        });
    }
}