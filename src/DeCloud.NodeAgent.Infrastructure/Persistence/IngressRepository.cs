using DeCloud.NodeAgent.Core.Models;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Persistence;

/// <summary>
/// SQLite-based repository for persisting ingress rules.
/// Provides resilience across node agent restarts.
/// </summary>
public class IngressRepository : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _lock = new(1, 1);

    public IngressRepository(string databasePath, ILogger logger)
    {
        _logger = logger;

        var directory = Path.GetDirectoryName(databasePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        _connection = new SqliteConnection($"Data Source={databasePath}");
        _connection.Open();
        InitializeDatabase();

        _logger.LogInformation("✓ IngressRepository initialized at {Path}", databasePath);
    }

    private void InitializeDatabase()
    {
        var sql = @"
            CREATE TABLE IF NOT EXISTS IngressRules (
                Id TEXT PRIMARY KEY,
                VmId TEXT NOT NULL,
                OwnerWallet TEXT NOT NULL,
                Domain TEXT NOT NULL UNIQUE,
                TargetPort INTEGER NOT NULL DEFAULT 80,
                TargetProtocol TEXT NOT NULL DEFAULT 'Http',
                EnableTls INTEGER NOT NULL DEFAULT 1,
                ForceHttps INTEGER NOT NULL DEFAULT 1,
                EnableHttp2 INTEGER NOT NULL DEFAULT 1,
                EnableWebSocket INTEGER NOT NULL DEFAULT 1,
                PathPrefix TEXT,
                StripPathPrefix INTEGER NOT NULL DEFAULT 0,
                CustomHeaders TEXT,
                RateLimitPerMinute INTEGER NOT NULL DEFAULT 0,
                AllowedIps TEXT,
                Status TEXT NOT NULL DEFAULT 'Pending',
                StatusMessage TEXT,
                TlsStatus TEXT NOT NULL DEFAULT 'Pending',
                TlsExpiresAt TEXT,
                CreatedAt TEXT NOT NULL,
                UpdatedAt TEXT NOT NULL,
                LastReloadAt TEXT,
                TotalRequests INTEGER NOT NULL DEFAULT 0,
                TotalBytesTransferred INTEGER NOT NULL DEFAULT 0,
                VmPrivateIp TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_ingress_vmid ON IngressRules(VmId);
            CREATE INDEX IF NOT EXISTS idx_ingress_domain ON IngressRules(Domain);
            CREATE INDEX IF NOT EXISTS idx_ingress_owner ON IngressRules(OwnerWallet);
            CREATE INDEX IF NOT EXISTS idx_ingress_status ON IngressRules(Status);
        ";

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();

        _logger.LogDebug("Ingress database schema initialized");
    }

    /// <summary>
    /// Save or update an ingress rule
    /// </summary>
    public async Task SaveAsync(IngressRule rule)
    {
        await _lock.WaitAsync();
        try
        {
            rule.UpdatedAt = DateTime.UtcNow;

            var sql = @"
                INSERT OR REPLACE INTO IngressRules (
                    Id, VmId, OwnerWallet, Domain, TargetPort, TargetProtocol,
                    EnableTls, ForceHttps, EnableHttp2, EnableWebSocket,
                    PathPrefix, StripPathPrefix, CustomHeaders, RateLimitPerMinute,
                    AllowedIps, Status, StatusMessage, TlsStatus, TlsExpiresAt,
                    CreatedAt, UpdatedAt, LastReloadAt, TotalRequests, 
                    TotalBytesTransferred, VmPrivateIp
                ) VALUES (
                    @Id, @VmId, @OwnerWallet, @Domain, @TargetPort, @TargetProtocol,
                    @EnableTls, @ForceHttps, @EnableHttp2, @EnableWebSocket,
                    @PathPrefix, @StripPathPrefix, @CustomHeaders, @RateLimitPerMinute,
                    @AllowedIps, @Status, @StatusMessage, @TlsStatus, @TlsExpiresAt,
                    @CreatedAt, @UpdatedAt, @LastReloadAt, @TotalRequests,
                    @TotalBytesTransferred, @VmPrivateIp
                )";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;

            cmd.Parameters.AddWithValue("@Id", rule.Id);
            cmd.Parameters.AddWithValue("@VmId", rule.VmId);
            cmd.Parameters.AddWithValue("@OwnerWallet", rule.OwnerWallet);
            cmd.Parameters.AddWithValue("@Domain", rule.Domain);
            cmd.Parameters.AddWithValue("@TargetPort", rule.TargetPort);
            cmd.Parameters.AddWithValue("@TargetProtocol", rule.TargetProtocol.ToString());
            cmd.Parameters.AddWithValue("@EnableTls", rule.EnableTls ? 1 : 0);
            cmd.Parameters.AddWithValue("@ForceHttps", rule.ForceHttps ? 1 : 0);
            cmd.Parameters.AddWithValue("@EnableHttp2", rule.EnableHttp2 ? 1 : 0);
            cmd.Parameters.AddWithValue("@EnableWebSocket", rule.EnableWebSocket ? 1 : 0);
            cmd.Parameters.AddWithValue("@PathPrefix", rule.PathPrefix ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@StripPathPrefix", rule.StripPathPrefix ? 1 : 0);
            cmd.Parameters.AddWithValue("@CustomHeaders",
                rule.CustomHeaders.Count > 0
                    ? System.Text.Json.JsonSerializer.Serialize(rule.CustomHeaders)
                    : (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@RateLimitPerMinute", rule.RateLimitPerMinute);
            cmd.Parameters.AddWithValue("@AllowedIps",
                rule.AllowedIps.Count > 0
                    ? System.Text.Json.JsonSerializer.Serialize(rule.AllowedIps)
                    : (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@Status", rule.Status.ToString());
            cmd.Parameters.AddWithValue("@StatusMessage", rule.StatusMessage ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@TlsStatus", rule.TlsStatus.ToString());
            cmd.Parameters.AddWithValue("@TlsExpiresAt",
                rule.TlsExpiresAt?.ToString("O") ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@CreatedAt", rule.CreatedAt.ToString("O"));
            cmd.Parameters.AddWithValue("@UpdatedAt", rule.UpdatedAt.ToString("O"));
            cmd.Parameters.AddWithValue("@LastReloadAt",
                rule.LastReloadAt?.ToString("O") ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@TotalRequests", rule.TotalRequests);
            cmd.Parameters.AddWithValue("@TotalBytesTransferred", rule.TotalBytesTransferred);
            cmd.Parameters.AddWithValue("@VmPrivateIp", rule.VmPrivateIp ?? (object)DBNull.Value);

            await cmd.ExecuteNonQueryAsync();

            _logger.LogDebug("Saved ingress rule {Id} for domain {Domain}", rule.Id, rule.Domain);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get an ingress rule by ID
    /// </summary>
    public async Task<IngressRule?> GetByIdAsync(string id)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = "SELECT * FROM IngressRules WHERE Id = @Id AND Status != 'Deleted'";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@Id", id);

            using var reader = await cmd.ExecuteReaderAsync();
            if (await reader.ReadAsync())
            {
                return ParseFromReader(reader);
            }

            return null;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get an ingress rule by domain
    /// </summary>
    public async Task<IngressRule?> GetByDomainAsync(string domain)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = "SELECT * FROM IngressRules WHERE Domain = @Domain AND Status != 'Deleted'";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@Domain", domain.ToLowerInvariant());

            using var reader = await cmd.ExecuteReaderAsync();
            if (await reader.ReadAsync())
            {
                return ParseFromReader(reader);
            }

            return null;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get all ingress rules for a VM
    /// </summary>
    public async Task<List<IngressRule>> GetByVmIdAsync(string vmId)
    {
        await _lock.WaitAsync();
        try
        {
            var rules = new List<IngressRule>();
            var sql = "SELECT * FROM IngressRules WHERE VmId = @VmId AND Status != 'Deleted' ORDER BY CreatedAt";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var rule = ParseFromReader(reader);
                if (rule != null) rules.Add(rule);
            }

            return rules;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get all active ingress rules (for Caddy config generation)
    /// </summary>
    public async Task<List<IngressRule>> GetAllActiveAsync()
    {
        await _lock.WaitAsync();
        try
        {
            var rules = new List<IngressRule>();
            var sql = "SELECT * FROM IngressRules WHERE Status IN ('Active', 'Configuring') ORDER BY Domain";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var rule = ParseFromReader(reader);
                if (rule != null) rules.Add(rule);
            }

            return rules;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get all ingress rules
    /// </summary>
    public async Task<List<IngressRule>> GetAllAsync()
    {
        await _lock.WaitAsync();
        try
        {
            var rules = new List<IngressRule>();
            var sql = "SELECT * FROM IngressRules WHERE Status != 'Deleted' ORDER BY CreatedAt DESC";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var rule = ParseFromReader(reader);
                if (rule != null) rules.Add(rule);
            }

            return rules;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Delete an ingress rule (soft delete)
    /// </summary>
    public async Task DeleteAsync(string id)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = @"
                UPDATE IngressRules 
                SET Status = 'Deleted', UpdatedAt = @UpdatedAt 
                WHERE Id = @Id";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@Id", id);
            cmd.Parameters.AddWithValue("@UpdatedAt", DateTime.UtcNow.ToString("O"));

            await cmd.ExecuteNonQueryAsync();

            _logger.LogInformation("Deleted ingress rule {Id}", id);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Update ingress rule status
    /// </summary>
    public async Task UpdateStatusAsync(string id, IngressStatus status, string? message = null)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = @"
                UPDATE IngressRules 
                SET Status = @Status, 
                    StatusMessage = @Message,
                    UpdatedAt = @UpdatedAt,
                    LastReloadAt = CASE WHEN @Status = 'Active' THEN @UpdatedAt ELSE LastReloadAt END
                WHERE Id = @Id";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@Id", id);
            cmd.Parameters.AddWithValue("@Status", status.ToString());
            cmd.Parameters.AddWithValue("@Message", message ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@UpdatedAt", DateTime.UtcNow.ToString("O"));

            await cmd.ExecuteNonQueryAsync();
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Update TLS status
    /// </summary>
    public async Task UpdateTlsStatusAsync(string id, TlsCertStatus status, DateTime? expiresAt = null)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = @"
                UPDATE IngressRules 
                SET TlsStatus = @TlsStatus, 
                    TlsExpiresAt = @TlsExpiresAt,
                    UpdatedAt = @UpdatedAt
                WHERE Id = @Id";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@Id", id);
            cmd.Parameters.AddWithValue("@TlsStatus", status.ToString());
            cmd.Parameters.AddWithValue("@TlsExpiresAt", expiresAt?.ToString("O") ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@UpdatedAt", DateTime.UtcNow.ToString("O"));

            await cmd.ExecuteNonQueryAsync();
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Purge deleted rules older than specified age
    /// </summary>
    public async Task PurgeDeletedAsync(TimeSpan olderThan)
    {
        await _lock.WaitAsync();
        try
        {
            var cutoff = (DateTime.UtcNow - olderThan).ToString("O");
            var sql = "DELETE FROM IngressRules WHERE Status = 'Deleted' AND UpdatedAt < @Cutoff";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@Cutoff", cutoff);

            var rows = await cmd.ExecuteNonQueryAsync();
            if (rows > 0)
            {
                _logger.LogInformation("Purged {Count} deleted ingress rules", rows);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    private IngressRule? ParseFromReader(SqliteDataReader reader)
    {
        try
        {
            var rule = new IngressRule
            {
                Id = reader.GetString(reader.GetOrdinal("Id")),
                VmId = reader.GetString(reader.GetOrdinal("VmId")),
                OwnerWallet = reader.GetString(reader.GetOrdinal("OwnerWallet")),
                Domain = reader.GetString(reader.GetOrdinal("Domain")),
                TargetPort = reader.GetInt32(reader.GetOrdinal("TargetPort")),
                TargetProtocol = Enum.Parse<IngressProtocol>(
                    reader.GetString(reader.GetOrdinal("TargetProtocol"))),
                EnableTls = reader.GetInt32(reader.GetOrdinal("EnableTls")) == 1,
                ForceHttps = reader.GetInt32(reader.GetOrdinal("ForceHttps")) == 1,
                EnableHttp2 = reader.GetInt32(reader.GetOrdinal("EnableHttp2")) == 1,
                EnableWebSocket = reader.GetInt32(reader.GetOrdinal("EnableWebSocket")) == 1,
                StripPathPrefix = reader.GetInt32(reader.GetOrdinal("StripPathPrefix")) == 1,
                RateLimitPerMinute = reader.GetInt32(reader.GetOrdinal("RateLimitPerMinute")),
                Status = Enum.Parse<IngressStatus>(reader.GetString(reader.GetOrdinal("Status"))),
                TlsStatus = Enum.Parse<TlsCertStatus>(reader.GetString(reader.GetOrdinal("TlsStatus"))),
                TotalRequests = reader.GetInt64(reader.GetOrdinal("TotalRequests")),
                TotalBytesTransferred = reader.GetInt64(reader.GetOrdinal("TotalBytesTransferred")),
                CreatedAt = DateTime.Parse(reader.GetString(reader.GetOrdinal("CreatedAt"))),
                UpdatedAt = DateTime.Parse(reader.GetString(reader.GetOrdinal("UpdatedAt")))
            };

            // Nullable fields
            var pathPrefixOrdinal = reader.GetOrdinal("PathPrefix");
            if (!reader.IsDBNull(pathPrefixOrdinal))
                rule.PathPrefix = reader.GetString(pathPrefixOrdinal);

            var statusMsgOrdinal = reader.GetOrdinal("StatusMessage");
            if (!reader.IsDBNull(statusMsgOrdinal))
                rule.StatusMessage = reader.GetString(statusMsgOrdinal);

            var tlsExpiresOrdinal = reader.GetOrdinal("TlsExpiresAt");
            if (!reader.IsDBNull(tlsExpiresOrdinal))
                rule.TlsExpiresAt = DateTime.Parse(reader.GetString(tlsExpiresOrdinal));

            var lastReloadOrdinal = reader.GetOrdinal("LastReloadAt");
            if (!reader.IsDBNull(lastReloadOrdinal))
                rule.LastReloadAt = DateTime.Parse(reader.GetString(lastReloadOrdinal));

            var vmIpOrdinal = reader.GetOrdinal("VmPrivateIp");
            if (!reader.IsDBNull(vmIpOrdinal))
                rule.VmPrivateIp = reader.GetString(vmIpOrdinal);

            // JSON fields
            var customHeadersOrdinal = reader.GetOrdinal("CustomHeaders");
            if (!reader.IsDBNull(customHeadersOrdinal))
            {
                var json = reader.GetString(customHeadersOrdinal);
                rule.CustomHeaders = System.Text.Json.JsonSerializer
                    .Deserialize<Dictionary<string, string>>(json) ?? new();
            }

            var allowedIpsOrdinal = reader.GetOrdinal("AllowedIps");
            if (!reader.IsDBNull(allowedIpsOrdinal))
            {
                var json = reader.GetString(allowedIpsOrdinal);
                rule.AllowedIps = System.Text.Json.JsonSerializer
                    .Deserialize<List<string>>(json) ?? new();
            }

            return rule;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to parse ingress rule from database");
            return null;
        }
    }

    public void Dispose()
    {
        _lock?.Dispose();
        _connection?.Dispose();
    }
}