using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Models.State;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace DeCloud.NodeAgent.Infrastructure.Persistence;

/// <summary>
/// SQLite-backed store for all node-agent obligation data.
///
/// Three tables in one file (<c>obligation-state.db</c>):
///
/// <code>
/// obligation_state   — identity blobs (private keys, versioned)
/// obligation         — which roles to run + dependency lists
/// system_template    — deployment specs (cloud-init, artifacts, resource spec)
/// </code>
///
/// All three tables follow the same role-keyed pattern.
/// <c>system_template</c> mirrors <c>obligation_state</c> but carries a
/// <c>revision</c> (int) rather than <c>version</c> to distinguish template
/// monotonic counters from identity state monotonic counters.
///
/// SECURITY: The <c>obligation_state</c> table stores private keys.
/// File permissions enforced to 0600 on Linux. State JSON is never logged.
/// <c>system_template</c> JSON may contain auth tokens baked into
/// cloud-init — same no-log discipline applies.
/// </summary>
public sealed class ObligationStateRepository : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly ILogger<ObligationStateRepository> _logger;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private static readonly JsonSerializerOptions _jsonOptions = new() { WriteIndented = false };

    public string DatabasePath { get; }

    public ObligationStateRepository(
        string databasePath,
        ILogger<ObligationStateRepository> logger)
    {
        _logger = logger;
        DatabasePath = databasePath;

        var dir = Path.GetDirectoryName(databasePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        EnforceFilePermissions(databasePath);

        _connection = new SqliteConnection($"Data Source={databasePath}");
        _connection.Open();

        using (var cmd = _connection.CreateCommand())
        {
            cmd.CommandText = "PRAGMA journal_mode=WAL;";
            cmd.ExecuteNonQuery();
        }

        InitialiseSchema();
        _logger.LogInformation("✓ ObligationStateRepository initialised at {Path}", databasePath);
    }

    // ════════════════════════════════════════════════════════════════════
    // Schema
    // ════════════════════════════════════════════════════════════════════

    private void InitialiseSchema()
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
            CREATE TABLE IF NOT EXISTS obligation_state (
                role        TEXT PRIMARY KEY,
                state_json  TEXT NOT NULL,
                version     INTEGER NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS obligation (
                role        TEXT PRIMARY KEY,
                deps_json   TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS system_template (
                role          TEXT PRIMARY KEY,
                template_json TEXT NOT NULL,
                revision      INTEGER NOT NULL,
                updated_at    TEXT NOT NULL
            );";
        cmd.ExecuteNonQuery();
        _logger.LogDebug("obligation_state, obligation, and system_template tables ready");
    }

    // ════════════════════════════════════════════════════════════════════
    // Identity state CRUD
    // ════════════════════════════════════════════════════════════════════

    public async Task<bool> UpsertAsync(
        string role, string stateJson, int version, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var stored = await GetVersionInternalAsync(role);
            if (version <= stored)
            {
                _logger.LogDebug("ObligationState [{Role}] upsert skipped — v{In} ≤ v{Stored}", role, version, stored);
                return false;
            }

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
                INSERT INTO obligation_state (role, state_json, version, updated_at)
                VALUES (@role, @json, @version, @at)
                ON CONFLICT(role) DO UPDATE SET
                    state_json = excluded.state_json,
                    version    = excluded.version,
                    updated_at = excluded.updated_at;";
            cmd.Parameters.AddWithValue("@role", role);
            cmd.Parameters.AddWithValue("@json", stateJson);
            cmd.Parameters.AddWithValue("@version", version);
            cmd.Parameters.AddWithValue("@at", DateTime.UtcNow.ToString("O"));
            await cmd.ExecuteNonQueryAsync(ct);

            _logger.LogInformation("ObligationState [{Role}] persisted v{Version} (prev v{Prev})", role, version, stored);
            return true;
        }
        finally { _lock.Release(); }
    }

    public async Task<string?> GetStateJsonAsync(string role, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT state_json FROM obligation_state WHERE role = @role;";
            cmd.Parameters.AddWithValue("@role", role);
            var result = await cmd.ExecuteScalarAsync(ct);
            return result is string s ? s : null;
        }
        finally { _lock.Release(); }
    }

    public async Task<int> GetVersionAsync(string role, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try { return await GetVersionInternalAsync(role); }
        finally { _lock.Release(); }
    }

    public async Task<bool> DeleteAsync(string role, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "DELETE FROM obligation_state WHERE role = @role;";
            cmd.Parameters.AddWithValue("@role", role);
            var rows = await cmd.ExecuteNonQueryAsync(ct);
            if (rows > 0) _logger.LogInformation("ObligationState [{Role}] deleted", role);
            return rows > 0;
        }
        finally { _lock.Release(); }
    }

    // ════════════════════════════════════════════════════════════════════
    // Obligations CRUD
    // ════════════════════════════════════════════════════════════════════

    public async Task ReplaceObligationsAsync(
        IReadOnlyList<ObligationDescriptor> obligations, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            using var tx = _connection.BeginTransaction();

            using (var del = _connection.CreateCommand())
            {
                del.Transaction = tx;
                del.CommandText = "DELETE FROM obligation;";
                await del.ExecuteNonQueryAsync(ct);
            }

            foreach (var o in obligations)
            {
                using var ins = _connection.CreateCommand();
                ins.Transaction = tx;
                ins.CommandText = @"
                    INSERT INTO obligation (role, deps_json, updated_at)
                    VALUES (@role, @deps, @at);";
                ins.Parameters.AddWithValue("@role", o.Role);
                ins.Parameters.AddWithValue("@deps", JsonSerializer.Serialize(o.Deps, _jsonOptions));
                ins.Parameters.AddWithValue("@at", o.UpdatedAt.ToString("O"));
                await ins.ExecuteNonQueryAsync(ct);
            }

            tx.Commit();
            _logger.LogDebug("Obligation table replaced — {Count} entries", obligations.Count);
        }
        finally { _lock.Release(); }
    }

    public async Task<IReadOnlyList<ObligationDescriptor>> GetObligationsAsync(
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT role, deps_json, updated_at FROM obligation;";
            using var reader = await cmd.ExecuteReaderAsync(ct);
            var results = new List<ObligationDescriptor>();

            while (await reader.ReadAsync(ct))
            {
                var role = reader.GetString(0);
                var deps = JsonSerializer.Deserialize<List<string>>(reader.GetString(1), _jsonOptions) ?? new();
                DateTime.TryParse(reader.GetString(2), null,
                    System.Globalization.DateTimeStyles.RoundtripKind, out var updatedAt);
                results.Add(new ObligationDescriptor { Role = role, Deps = deps, UpdatedAt = updatedAt });
            }

            return results;
        }
        finally { _lock.Release(); }
    }

    // ════════════════════════════════════════════════════════════════════
    // System templates CRUD
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Upsert a system template. Returns <c>true</c> if written (incoming
    /// revision was higher); <c>false</c> if skipped (equal or lower).
    /// </summary>
    public async Task<bool> UpsertSystemTemplateAsync(
        string role, string templateJson, int revision, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var stored = await GetSystemTemplateRevisionInternalAsync(role);
            if (revision <= stored)
            {
                _logger.LogDebug("SystemTemplate [{Role}] upsert skipped — r{In} ≤ r{Stored}", role, revision, stored);
                return false;
            }

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
                INSERT INTO system_template (role, template_json, revision, updated_at)
                VALUES (@role, @json, @revision, @at)
                ON CONFLICT(role) DO UPDATE SET
                    template_json = excluded.template_json,
                    revision      = excluded.revision,
                    updated_at    = excluded.updated_at;";
            cmd.Parameters.AddWithValue("@role", role);
            cmd.Parameters.AddWithValue("@json", templateJson);
            cmd.Parameters.AddWithValue("@revision", revision);
            cmd.Parameters.AddWithValue("@at", DateTime.UtcNow.ToString("O"));
            await cmd.ExecuteNonQueryAsync(ct);

            // Log role + revision only — never the template JSON.
            _logger.LogInformation("SystemTemplate [{Role}] persisted r{Revision} (prev r{Prev})", role, revision, stored);
            return true;
        }
        finally { _lock.Release(); }
    }

    /// <summary>Retrieve raw template JSON for a role, or <c>null</c>.</summary>
    public async Task<string?> GetSystemTemplateJsonAsync(
        string role, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT template_json FROM system_template WHERE role = @role;";
            cmd.Parameters.AddWithValue("@role", role);
            var result = await cmd.ExecuteScalarAsync(ct);
            return result is string s ? s : null;
        }
        finally { _lock.Release(); }
    }

    /// <summary>Get the stored revision for a role's template (0 if absent).</summary>
    public async Task<int> GetSystemTemplateRevisionAsync(
        string role, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try { return await GetSystemTemplateRevisionInternalAsync(role); }
        finally { _lock.Release(); }
    }

    // ════════════════════════════════════════════════════════════════════
    // Internal helpers
    // ════════════════════════════════════════════════════════════════════

    private async Task<int> GetVersionInternalAsync(string role)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT version FROM obligation_state WHERE role = @role;";
        cmd.Parameters.AddWithValue("@role", role);
        var result = await cmd.ExecuteScalarAsync();
        return result is long v ? (int)v : 0;
    }

    private async Task<int> GetSystemTemplateRevisionInternalAsync(string role)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT revision FROM system_template WHERE role = @role;";
        cmd.Parameters.AddWithValue("@role", role);
        var result = await cmd.ExecuteScalarAsync();
        return result is long r ? (int)r : 0;
    }

    private void EnforceFilePermissions(string path)
    {
        if (OperatingSystem.IsWindows()) return;
        try
        {
            if (File.Exists(path))
                File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not enforce 0600 on {Path}", path);
        }
    }

    public void Dispose()
    {
        _lock.Dispose();
        _connection.Dispose();
    }
}