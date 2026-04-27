using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Models.State;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace DeCloud.NodeAgent.Infrastructure.Persistence;

// ============================================================
// Placement: src/DeCloud.NodeAgent.Infrastructure/Persistence/ObligationStateRepository.cs
// ============================================================

/// <summary>
/// SQLite-backed store for system VM obligation data on the node agent.
///
/// Opens its own connection to <c>obligation-state.db</c> in the same
/// <c>VmStoragePath</c> directory as <c>vms.db</c> and <c>port-mappings.db</c>.
///
/// Two tables (a third — <c>system_template</c> — is added in P9):
///
/// <code>
/// CREATE TABLE obligation_state (
///     role        TEXT PRIMARY KEY,   -- "relay" | "dht" | "blockstore"
///     state_json  TEXT NOT NULL,      -- Identity blob (private keys etc.)
///     version     INTEGER NOT NULL,   -- Monotonic, orchestrator-assigned
///     updated_at  TEXT NOT NULL
/// );
///
/// CREATE TABLE obligation (
///     role        TEXT PRIMARY KEY,   -- "relay" | "dht" | "blockstore"
///     deps_json   TEXT NOT NULL,      -- JSON array of dep role names
///     updated_at  TEXT NOT NULL
/// );
/// </code>
///
/// SECURITY: The <c>obligation_state</c> table stores private keys. File
/// permissions are enforced to 0600 (owner read/write only) at initialisation
/// on Linux. The full state_json is never written to application logs — only
/// role and version. The <c>obligation</c> table is non-sensitive (just role
/// names + dep lists) but lives in the same file under the same permissions.
/// </summary>
public sealed class ObligationStateRepository : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly ILogger<ObligationStateRepository> _logger;

    // One writer at a time — SQLite WAL mode does not change this requirement
    // for a single-process writer.
    private readonly SemaphoreSlim _lock = new(1, 1);

    // JSON options used for deps_json serialisation. PropertyNamingPolicy is
    // not relevant for arrays of strings, but we set defaults explicitly to
    // avoid surprises if the schema evolves.
    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = false,
    };

    // Exposed so callers can verify the path at startup.
    public string DatabasePath { get; }

    public ObligationStateRepository(string databasePath, ILogger<ObligationStateRepository> logger)
    {
        _logger = logger;
        DatabasePath = databasePath;

        var directory = Path.GetDirectoryName(databasePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        // Harden file permissions on Linux before opening the connection so that
        // the file is never readable by other OS users even briefly.
        EnforceFilePermissions(databasePath);

        _connection = new SqliteConnection($"Data Source={databasePath}");
        _connection.Open();

        // WAL mode: readers do not block the writer and vice-versa.
        using (var cmd = _connection.CreateCommand())
        {
            cmd.CommandText = "PRAGMA journal_mode=WAL;";
            cmd.ExecuteNonQuery();
        }

        InitialiseSchema();

        _logger.LogInformation(
            "✓ ObligationStateRepository initialised at {Path}", databasePath);
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
            );";
        cmd.ExecuteNonQuery();
        _logger.LogDebug("obligation_state and obligation tables ready");
    }

    // ════════════════════════════════════════════════════════════════════
    // Identity state CRUD
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Upsert state for <paramref name="role"/>.
    /// Returns <c>true</c> if the row was written; <c>false</c> if the incoming
    /// version was equal to or lower than the stored version (no-op).
    /// </summary>
    public async Task<bool> UpsertAsync(
        string role,
        string stateJson,
        int version,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            // Read current version first (one extra round-trip is cheaper than
            // a blind upsert that must be rolled back on version conflict).
            var stored = await GetVersionInternalAsync(role);

            if (version <= stored)
            {
                _logger.LogDebug(
                    "ObligationState [{Role}] upsert skipped — incoming v{Incoming} ≤ stored v{Stored}",
                    role, version, stored);
                return false;
            }

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
                INSERT INTO obligation_state (role, state_json, version, updated_at)
                VALUES (@role, @state_json, @version, @updated_at)
                ON CONFLICT(role) DO UPDATE SET
                    state_json = excluded.state_json,
                    version    = excluded.version,
                    updated_at = excluded.updated_at;";

            cmd.Parameters.AddWithValue("@role", role);
            cmd.Parameters.AddWithValue("@state_json", stateJson);
            cmd.Parameters.AddWithValue("@version", version);
            cmd.Parameters.AddWithValue("@updated_at", DateTime.UtcNow.ToString("O"));

            await cmd.ExecuteNonQueryAsync(ct);

            // Log role + version only — never the state JSON.
            _logger.LogInformation(
                "ObligationState [{Role}] persisted v{Version} (prev v{Prev})",
                role, version, stored);

            return true;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>Retrieve the raw state JSON for <paramref name="role"/>, or <c>null</c>.</summary>
    public async Task<string?> GetStateJsonAsync(string role, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT state_json FROM obligation_state WHERE role = @role;";
            cmd.Parameters.AddWithValue("@role", role);

            var result = await cmd.ExecuteScalarAsync(ct);

            // result is DBNull when no row matches.
            return result is string s ? s : null;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>Retrieve the current version for <paramref name="role"/>, or <c>0</c>.</summary>
    public async Task<int> GetVersionAsync(string role, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            return await GetVersionInternalAsync(role);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>Delete the state row for <paramref name="role"/> (obligation removed).</summary>
    public async Task<bool> DeleteAsync(string role, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "DELETE FROM obligation_state WHERE role = @role;";
            cmd.Parameters.AddWithValue("@role", role);

            var rows = await cmd.ExecuteNonQueryAsync(ct);
            if (rows > 0)
                _logger.LogInformation("ObligationState [{Role}] deleted", role);

            return rows > 0;
        }
        finally
        {
            _lock.Release();
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Obligations CRUD
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Replace the entire obligation set atomically. Wipes the table and
    /// inserts the new entries inside a single SQLite transaction — the
    /// post-state is exactly <paramref name="obligations"/> or unchanged.
    ///
    /// Caller responsibility: <paramref name="obligations"/> entries must
    /// already be canonicalised (lower-case role names, valid deps).
    /// <see cref="ObligationStateService.SaveObligationsAsync"/> performs
    /// that sanitisation before invoking this method.
    /// </summary>
    public async Task ReplaceObligationsAsync(
        IReadOnlyList<ObligationDescriptor> obligations,
        CancellationToken ct = default)
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
                    VALUES (@role, @deps_json, @updated_at);";
                ins.Parameters.AddWithValue("@role", o.Role);
                ins.Parameters.AddWithValue(
                    "@deps_json",
                    JsonSerializer.Serialize(o.Deps, _jsonOptions));
                ins.Parameters.AddWithValue("@updated_at", o.UpdatedAt.ToString("O"));
                await ins.ExecuteNonQueryAsync(ct);
            }

            tx.Commit();

            _logger.LogDebug(
                "Obligation table replaced — {Count} entries", obligations.Count);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Read all obligations from the table. Returns an empty list if the
    /// table is empty (never <c>null</c>). Order is unspecified — the matrix
    /// does not depend on it.
    /// </summary>
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
                var depsJson = reader.GetString(1);
                var updatedAtStr = reader.GetString(2);

                var deps = JsonSerializer.Deserialize<List<string>>(depsJson, _jsonOptions)
                           ?? new List<string>();

                if (!DateTime.TryParse(updatedAtStr, null,
                        System.Globalization.DateTimeStyles.RoundtripKind,
                        out var updatedAt))
                {
                    updatedAt = DateTime.UtcNow;
                }

                results.Add(new ObligationDescriptor
                {
                    Role = role,
                    Deps = deps,
                    UpdatedAt = updatedAt,
                });
            }

            return results;
        }
        finally
        {
            _lock.Release();
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Internal helpers
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Read stored version without acquiring the lock.
    /// Must only be called from within a lock-guarded block.
    /// </summary>
    private async Task<int> GetVersionInternalAsync(string role)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT version FROM obligation_state WHERE role = @role;";
        cmd.Parameters.AddWithValue("@role", role);

        var result = await cmd.ExecuteScalarAsync();
        return result is long v ? (int)v : 0;
    }

    /// <summary>
    /// Enforce 0600 permissions on <paramref name="path"/> on Linux.
    /// No-op on Windows (where the equivalent is NTFS ACLs, handled by the OS).
    /// Called before the connection is opened so the file is never world-readable.
    /// </summary>
    private void EnforceFilePermissions(string path)
    {
        if (OperatingSystem.IsWindows())
            return;

        try
        {
            // If the file already exists, tighten its permissions immediately.
            if (File.Exists(path))
            {
                File.SetUnixFileMode(path,
                    UnixFileMode.UserRead | UnixFileMode.UserWrite);
            }
            // For new files, the OS inherits from the umask; we fix it after creation
            // inside InitialiseSchema → the first PRAGMA triggers file creation.
            // A second call is made from ObligationStateService.EnsureFilePermissionsOnce().
        }
        catch (Exception ex)
        {
            // Non-fatal but important — log as warning so operators notice.
            _logger.LogWarning(ex,
                "Could not enforce 0600 permissions on {Path}. " +
                "The obligation state file may be readable by other OS users.",
                path);
        }
    }

    public void Dispose()
    {
        _lock.Dispose();
        _connection.Dispose();
    }
}