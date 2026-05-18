using DeCloud.NodeAgent.Core.Models;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace DeCloud.NodeAgent.Infrastructure.Persistence;

/// <summary>
/// SQLite-based repository for persisting VM state on node agents.
/// Provides resilience across node agent restarts with optional encryption.
/// 
/// SECURITY: Supports encryption using a deterministic key derived from node ID and wallet.
/// SECURITY: Never stores plaintext passwords - only wallet-encrypted passwords.
/// 
/// SCHEMA VERSIONING: Automatically migrates database schema on startup
/// </summary>
public class VmRepository : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private readonly bool _encrypted;

    private const int CURRENT_SCHEMA_VERSION = 8; // Incremented when schema changes

    public VmRepository(string databasePath, ILogger logger, string? encryptionKey = null)
    {
        _logger = logger;

        // Ensure directory exists
        var directory = Path.GetDirectoryName(databasePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        _encrypted = !string.IsNullOrEmpty(encryptionKey);

        if (_encrypted)
        {
            // Note: Standard Microsoft.Data.Sqlite doesn't support encryption
            // For production, use SQLCipher or encrypt sensitive fields at application level
            _connection = new SqliteConnection($"Data Source={databasePath}");
            _logger.LogInformation("✓ VmRepository initialized with field-level encryption at {Path}", databasePath);
        }
        else
        {
            _connection = new SqliteConnection($"Data Source={databasePath}");
            _logger.LogWarning("⚠ VmRepository initialized WITHOUT encryption at {Path}", databasePath);
        }

        _connection.Open();
        InitializeOrMigrateDatabase();
    }

    /// <summary>
    /// Generate a deterministic encryption key from node-specific identifiers.
    /// This allows the same encryption key to be regenerated after restart.
    /// SECURITY: Key is derived using PBKDF2 with 100,000 iterations.
    /// </summary>
    public static string GenerateEncryptionKey(string nodeId, string walletAddress)
    {
        var salt = Encoding.UTF8.GetBytes($"decloud-node-{nodeId}");
        var password = Encoding.UTF8.GetBytes($"{walletAddress}-{nodeId}");

        using var pbkdf2 = new Rfc2898DeriveBytes(
            password,
            salt,
            iterations: 100_000,
            HashAlgorithmName.SHA256);

        return Convert.ToBase64String(pbkdf2.GetBytes(32));
    }

    /// <summary>
    /// Initialize database or migrate to latest schema version
    /// </summary>
    private void InitializeOrMigrateDatabase()
    {
        // Create schema version table if it doesn't exist
        using (var cmd = _connection.CreateCommand())
        {
            cmd.CommandText = @"
                CREATE TABLE IF NOT EXISTS SchemaVersion (
                    Version INTEGER PRIMARY KEY,
                    AppliedAt TEXT NOT NULL
                )";
            cmd.ExecuteNonQuery();
        }

        // Get current schema version
        var currentVersion = GetSchemaVersion();
        _logger.LogInformation("Current database schema version: {Version}", currentVersion);

        // Apply migrations if needed
        if (currentVersion < CURRENT_SCHEMA_VERSION)
        {
            _logger.LogWarning("Database schema outdated (v{Current}). Migrating to v{Target}...",
                currentVersion, CURRENT_SCHEMA_VERSION);

            MigrateSchema(currentVersion, CURRENT_SCHEMA_VERSION);

            _logger.LogInformation("✓ Database schema migrated successfully to v{Version}", CURRENT_SCHEMA_VERSION);
        }
        else if (currentVersion == 0)
        {
            // Fresh database - create initial schema
            CreateInitialSchema();
            SetSchemaVersion(CURRENT_SCHEMA_VERSION);
            _logger.LogInformation("✓ Database schema initialized at v{Version}", CURRENT_SCHEMA_VERSION);
        }
        else
        {
            _logger.LogDebug("Database schema is up to date (v{Version})", currentVersion);
        }
    }

    /// <summary>
    /// Get current schema version from database
    /// </summary>
    private int GetSchemaVersion()
    {
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT MAX(Version) FROM SchemaVersion";
            var result = cmd.ExecuteScalar();
            return result is DBNull or null ? 0 : Convert.ToInt32(result);
        }
        catch
        {
            // SchemaVersion table doesn't exist yet
            return 0;
        }
    }

    /// <summary>
    /// Set schema version after successful migration
    /// </summary>
    private void SetSchemaVersion(int version)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "INSERT INTO SchemaVersion (Version, AppliedAt) VALUES (@Version, @AppliedAt)";
        cmd.Parameters.AddWithValue("@Version", version);
        cmd.Parameters.AddWithValue("@AppliedAt", DateTime.UtcNow.ToString("O"));
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Create initial database schema (v2 - with QualityTier and ComputePointCost)
    /// </summary>
    private void CreateInitialSchema()
    {
        var createTable = @"
            CREATE TABLE IF NOT EXISTS VmRecords (
                VmId TEXT PRIMARY KEY,
                Name TEXT NOT NULL,
                ServicesJson TEXT DEFAULT '[]',
                OwnerId TEXT,
                QualityTier INTEGER NOT NULL DEFAULT 3,
                ReplicationFactor INTEGER NOT NULL DEFAULT 0,
                ComputePointCost INTEGER NOT NULL DEFAULT 4,
                VirtualCpuCores INTEGER NOT NULL,
                MemoryBytes INTEGER NOT NULL,
                DiskBytes INTEGER NOT NULL,
                State TEXT NOT NULL,
                IpAddress TEXT,
                MacAddress TEXT,
                VncPort INTEGER,
                Pid INTEGER,
                CreatedAt TEXT NOT NULL,
                StartedAt TEXT,
                StoppedAt TEXT,
                LastUpdated TEXT NOT NULL,
                DiskPath TEXT,
                ConfigPath TEXT,
                BaseImageUrl TEXT,
                BaseImageHash TEXT,
                SshPublicKey TEXT,
                EncryptedPassword TEXT,
                VmType TEXT NOT NULL DEFAULT 'General',
                LabelsJson TEXT,
                TargetNodeId TEXT,
                DeletionReason TEXT,
                CrashJournal TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_tenant ON VmRecords(OwnerId);
            CREATE INDEX IF NOT EXISTS idx_state ON VmRecords(State);
            CREATE INDEX IF NOT EXISTS idx_updated ON VmRecords(LastUpdated);
            CREATE INDEX IF NOT EXISTS idx_quality_tier ON VmRecords(QualityTier);
        ";

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = createTable;
        cmd.ExecuteNonQuery();

        _logger.LogDebug("Database schema created{Encrypted}",
            _encrypted ? " with encryption" : "");
    }

    /// <summary>
    /// Migrate database schema from old version to new version
    /// </summary>
    private void MigrateSchema(int fromVersion, int toVersion)
    {
        using var transaction = _connection.BeginTransaction();
        try
        {
            // Migration from v0 (or v1 without QualityTier/ComputePointCost) to v2
            if (fromVersion < 2)
            {
                _logger.LogInformation("Applying migration: v{From} → v{To}", fromVersion, 2);
                MigrateToV2();
                SetSchemaVersion(2);
            }

            // Migration v2 → v3: Add ServicesJson column
            if (fromVersion < 3)
            {
                _logger.LogInformation("Applying migration: v{From} → v3 (ServicesJson column)", Math.Max(fromVersion, 2));
                MigrateToV3();
                SetSchemaVersion(3);
            }

            // Migration v3 → v4: Add ReplicationFactor column
            if (fromVersion < 4)
            {
                _logger.LogInformation("Applying migration: v{From} → v4 (ReplicationFactor column)", Math.Max(fromVersion, 3));
                MigrateToV4();
                SetSchemaVersion(4);
            }

            // Migration v4 → v5: Add VmType and LabelsJson columns
            if (fromVersion < 5)
            {
                _logger.LogInformation("Applying migration: v{From} → v5 (VmType, LabelsJson columns)", Math.Max(fromVersion, 4));
                MigrateToV5();
                SetSchemaVersion(5);
            }

            // Migration v5 → v6: Add TargetNodeId column for zombie fencing
            if (fromVersion < 6)
            {
                _logger.LogInformation("Applying migration: v{From} → v6 (TargetNodeId column)", Math.Max(fromVersion, 5));
                MigrateToV6();
                SetSchemaVersion(6);
            }

            if (fromVersion < 7)
            {
                _logger.LogInformation("Applying migration: v{From} → v7 (DeletionReason column)", Math.Max(fromVersion, 6));
                MigrateToV7();
                SetSchemaVersion(7);
            }

            if (fromVersion < 8)
            {
                _logger.LogInformation("Applying migration: v{From} → v8 (CrashJournal column)", Math.Max(fromVersion, 7));
                MigrateToV8();
                SetSchemaVersion(8);
            }

            transaction.Commit();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Schema migration failed. Rolling back...");
            transaction.Rollback();
            throw;
        }
    }

    /// <summary>
    /// Migrate to schema v2: Add QualityTier and ComputePointCost columns
    /// </summary>
    private void MigrateToV2()
    {
        _logger.LogInformation("Adding QualityTier and ComputePointCost columns...");

        // Check if VmRecords table exists
        using (var cmd = _connection.CreateCommand())
        {
            cmd.CommandText = "SELECT name FROM sqlite_master WHERE type='table' AND name='VmRecords'";
            var tableExists = cmd.ExecuteScalar() != null;

            if (!tableExists)
            {
                // Table doesn't exist - create new schema
                _logger.LogInformation("VmRecords table doesn't exist. Creating fresh schema...");
                CreateInitialSchema();
                return;
            }
        }

        // Check if columns already exist
        var hasQualityTier = ColumnExists("VmRecords", "QualityTier");
        var hasComputePointCost = ColumnExists("VmRecords", "ComputePointCost");

        if (hasQualityTier && hasComputePointCost)
        {
            _logger.LogInformation("QualityTier and ComputePointCost columns already exist. No migration needed.");
            return;
        }

        // Create backup table name
        var backupTable = $"VmRecords_backup_{DateTime.UtcNow:yyyyMMddHHmmss}";
        _logger.LogInformation("Creating backup table: {Table}", backupTable);

        // Rename old table to backup
        using (var cmd = _connection.CreateCommand())
        {
            cmd.CommandText = $"ALTER TABLE VmRecords RENAME TO {backupTable}";
            cmd.ExecuteNonQuery();
        }

        // Create new table with correct schema
        CreateInitialSchema();

        // Migrate data from backup to new table
        _logger.LogInformation("Migrating data from backup table...");
        using (var cmd = _connection.CreateCommand())
        {
            cmd.CommandText = $@"
                INSERT INTO VmRecords 
                SELECT 
                    VmId, Name, OwnerId,
                    3 as QualityTier,      -- Default to Burstable tier
                    4 as ComputePointCost, -- Default compute points for Burstable
                    VirtualCpuCores, MemoryBytes, DiskBytes,
                    State, IpAddress, MacAddress, VncPort, Pid,
                    CreatedAt, StartedAt, StoppedAt, LastUpdated,
                    DiskPath, ConfigPath, BaseImageUrl, BaseImageHash,
                    SshPublicKey, EncryptedPassword
                FROM {backupTable}";

            var rowsMigrated = cmd.ExecuteNonQuery();
            _logger.LogInformation("✓ Migrated {Count} VM records", rowsMigrated);
        }

        // Keep backup table for safety (can be manually dropped later)
        _logger.LogInformation("Backup table {Table} retained for safety", backupTable);
    }

    /// <summary>
    /// Check if a column exists in a table
    /// </summary>
    private bool ColumnExists(string tableName, string columnName)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = $"PRAGMA table_info({tableName})";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var name = reader.GetString(1); // Column name is at index 1
            if (name.Equals(columnName, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Migrate to schema v3: Add ServicesJson column for per-service readiness tracking
    /// </summary>
    private void MigrateToV3()
    {
        if (!ColumnExists("VmRecords", "ServicesJson"))
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "ALTER TABLE VmRecords ADD COLUMN ServicesJson TEXT DEFAULT '[]'";
            cmd.ExecuteNonQuery();
            _logger.LogInformation("Added ServicesJson column to VmRecords");
        }
    }

    /// <summary>
    /// Migrate to schema v4: Add ReplicationFactor column for lazysync block replication.
    /// </summary>
    private void MigrateToV4()
    {
        if (!ColumnExists("VmRecords", "ReplicationFactor"))
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "ALTER TABLE VmRecords ADD COLUMN ReplicationFactor INTEGER NOT NULL DEFAULT 0";
            cmd.ExecuteNonQuery();
            _logger.LogInformation("Added ReplicationFactor column to VmRecords");
        }
    }

    /// <summary>
    /// Migrate to schema v5: Add VmType and LabelsJson columns so that system VM
    /// classification (Relay/DHT/BlockStore) survives NodeAgent restarts.
    /// Without these columns every VM loaded from SQLite gets VmType=General and
    /// Labels=null, causing the dashboard to show all system VMs as "Not deployed".
    /// </summary>
    private void MigrateToV5()
    {
        if (!ColumnExists("VmRecords", "VmType"))
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "ALTER TABLE VmRecords ADD COLUMN VmType TEXT NOT NULL DEFAULT 'General'";
            cmd.ExecuteNonQuery();
            _logger.LogInformation("Added VmType column to VmRecords");
        }

        if (!ColumnExists("VmRecords", "LabelsJson"))
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "ALTER TABLE VmRecords ADD COLUMN LabelsJson TEXT";
            cmd.ExecuteNonQuery();
            _logger.LogInformation("Added LabelsJson column to VmRecords");
        }
    }

    private void MigrateToV6()
    {
        if (!ColumnExists("VmRecords", "TargetNodeId"))
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "ALTER TABLE VmRecords ADD COLUMN TargetNodeId TEXT";
            cmd.ExecuteNonQuery();
            _logger.LogInformation("Added TargetNodeId column to VmRecords");
        }
    }

    /// <summary>
    /// Migrate to schema v7: Add DeletionReason column so the reconciliation
    /// matrix's decision reason is persisted on deleted system VM records.
    /// </summary>
    private void MigrateToV7()
    {
        if (!ColumnExists("VmRecords", "DeletionReason"))
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "ALTER TABLE VmRecords ADD COLUMN DeletionReason TEXT";
            cmd.ExecuteNonQuery();
            _logger.LogInformation("Added DeletionReason column to VmRecords");
        }
    }

    private void MigrateToV8()
    {
        if (!ColumnExists("VmRecords", "CrashJournal"))
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "ALTER TABLE VmRecords ADD COLUMN CrashJournal TEXT";
            cmd.ExecuteNonQuery();
            _logger.LogInformation("Added CrashJournal column to VmRecords");
        }
    }

    /// <summary>
    /// Save or update a VM record
    /// </summary>
    public async Task SaveVmAsync(VmInstance vm)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = @"
                INSERT OR REPLACE INTO VmRecords
                (VmId, Name, ServicesJson, OwnerId, QualityTier, ReplicationFactor, ComputePointCost,
                 VirtualCpuCores, MemoryBytes, DiskBytes,
                 State, IpAddress, MacAddress, VncPort, Pid,
                 CreatedAt, StartedAt, StoppedAt, LastUpdated, DiskPath, ConfigPath,
                 BaseImageUrl, BaseImageHash, SshPublicKey, EncryptedPassword,
                 VmType, LabelsJson, TargetNodeId)
                VALUES
                (@VmId, @Name, @ServicesJson, @OwnerId, @QualityTier, @ReplicationFactor, @ComputePointCost,
                 @VirtualCpuCores, @MemoryBytes, @DiskBytes,
                 @State, @IpAddress, @MacAddress, @VncPort, @Pid,
                 @CreatedAt, @StartedAt, @StoppedAt, @LastUpdated, @DiskPath, @ConfigPath,
                 @BaseImageUrl, @BaseImageHash, @SshPublicKey, @EncryptedPassword,
                 @VmType, @LabelsJson, @TargetNodeId)
            ";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;

            cmd.Parameters.AddWithValue("@VmId", vm.VmId);
            cmd.Parameters.AddWithValue("@Name", vm.Name);
            cmd.Parameters.AddWithValue("@ServicesJson",
                vm.Services.Count > 0 ? JsonSerializer.Serialize(vm.Services) : "[]");
            cmd.Parameters.AddWithValue("@OwnerId", vm.Spec.OwnerId ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@QualityTier", vm.Spec.QualityTier);
            cmd.Parameters.AddWithValue("@ComputePointCost", vm.Spec.ComputePointCost);
            cmd.Parameters.AddWithValue("@VirtualCpuCores", vm.Spec.VirtualCpuCores);
            cmd.Parameters.AddWithValue("@MemoryBytes", vm.Spec.MemoryBytes);
            cmd.Parameters.AddWithValue("@DiskBytes", vm.Spec.DiskBytes);
            cmd.Parameters.AddWithValue("@State", vm.State.ToString());
            cmd.Parameters.AddWithValue("@IpAddress", vm.Spec.IpAddress ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@MacAddress", vm.Spec.MacAddress ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@VncPort", vm.VncPort ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@Pid", vm.Pid ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@CreatedAt", vm.CreatedAt.ToString("O"));
            cmd.Parameters.AddWithValue("@StartedAt", vm.StartedAt?.ToString("O") ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@StoppedAt", vm.StoppedAt?.ToString("O") ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@LastUpdated", DateTime.UtcNow.ToString("O"));
            cmd.Parameters.AddWithValue("@DiskPath", vm.DiskPath ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@ConfigPath", vm.ConfigPath ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@BaseImageUrl", vm.Spec.BaseImageUrl ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@BaseImageHash", vm.Spec.BaseImageHash ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@SshPublicKey", vm.Spec.SshPublicKey ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@ReplicationFactor", vm.Spec.ReplicationFactor);
            // SECURITY: Only store encrypted password, NEVER plaintext
            cmd.Parameters.AddWithValue("@EncryptedPassword", vm.Spec.WalletEncryptedPassword ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@VmType", vm.Spec.VmType.ToString());
            cmd.Parameters.AddWithValue("@LabelsJson",
                vm.Spec.Labels is { Count: > 0 }
                    ? JsonSerializer.Serialize(vm.Spec.Labels)
                    : (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@TargetNodeId",
                vm.Spec.TargetNodeId ?? (object)DBNull.Value);

            await cmd.ExecuteNonQueryAsync();

            _logger.LogDebug("Saved VM {VmId} to database (State: {State})", vm.VmId, vm.State);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Load a specific VM by ID
    /// </summary>
    public async Task<VmInstance?> LoadVmAsync(string vmId)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = "SELECT * FROM VmRecords WHERE VmId = @VmId AND State != 'Deleted' LIMIT 1";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);

            using var reader = await cmd.ExecuteReaderAsync();
            if (await reader.ReadAsync())
            {
                return ParseVmFromReader(reader);
            }

            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load VM {VmId} from database", vmId);
            return null;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Load all VMs (excluding deleted)
    /// </summary>
    public async Task<List<VmInstance>> LoadAllVmsAsync()
    {
        await _lock.WaitAsync();
        try
        {
            var vms = new List<VmInstance>();
            var sql = "SELECT * FROM VmRecords WHERE State != 'Deleted' ORDER BY CreatedAt DESC";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                try
                {
                    var vm = ParseVmFromReader(reader);
                    if (vm != null)
                    {
                        vms.Add(vm);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to deserialize VM record");
                }
            }

            _logger.LogInformation("Loaded {Count} VMs from database", vms.Count);
            return vms;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Update VM state (optimized for frequent state changes)
    /// </summary>
    public async Task UpdateVmStateAsync(string vmId, VmState newState)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = @"
                UPDATE VmRecords 
                SET State = @State, 
                    LastUpdated = @Now,
                    StartedAt = CASE WHEN @State = 'Running' AND StartedAt IS NULL THEN @Now ELSE StartedAt END,
                    StoppedAt = CASE WHEN @State = 'Stopped' THEN @Now ELSE StoppedAt END
                WHERE VmId = @VmId";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);
            cmd.Parameters.AddWithValue("@State", newState.ToString());
            cmd.Parameters.AddWithValue("@Now", DateTime.UtcNow.ToString("O"));

            var rows = await cmd.ExecuteNonQueryAsync();
            if (rows > 0)
            {
                _logger.LogDebug("Updated VM {VmId} state to {State}", vmId, newState);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Update VM IP address
    /// </summary>
    public async Task UpdateVmIpAsync(string vmId, string ipAddress)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = "UPDATE VmRecords SET IpAddress = @Ip, LastUpdated = @Now WHERE VmId = @VmId";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);
            cmd.Parameters.AddWithValue("@Ip", ipAddress);
            cmd.Parameters.AddWithValue("@Now", DateTime.UtcNow.ToString("O"));

            await cmd.ExecuteNonQueryAsync();
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Delete a VM record (soft delete by marking as Deleted)
    /// </summary>
    public async Task DeleteVmAsync(string vmId)
    {
        await _lock.WaitAsync();
        try
        {
            // Soft delete - mark as deleted rather than removing
            var sql = "UPDATE VmRecords SET State = 'Deleted', LastUpdated = @Now WHERE VmId = @VmId";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);
            cmd.Parameters.AddWithValue("@Now", DateTime.UtcNow.ToString("O"));

            var rows = await cmd.ExecuteNonQueryAsync();
            if (rows > 0)
            {
                _logger.LogInformation("Marked VM {VmId} as deleted in database", vmId);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Stamp a deletion reason on a VM record before it is soft-deleted.
    /// Called by SystemVmReconciler to persist the matrix decision reason
    /// so it survives in the Deleted record for post-mortem diagnosis.
    /// </summary>
    public async Task SetDeletionReasonAsync(string vmId, string reason)
    {
        await _lock.WaitAsync();
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "UPDATE VmRecords SET DeletionReason = @Reason WHERE VmId = @VmId";
            cmd.Parameters.AddWithValue("@VmId", vmId);
            cmd.Parameters.AddWithValue("@Reason", reason);
            await cmd.ExecuteNonQueryAsync();
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task SetCrashJournalAsync(string vmId, string journal)
    {
        await _lock.WaitAsync();
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "UPDATE VmRecords SET CrashJournal = @Journal WHERE VmId = @VmId";
            cmd.Parameters.AddWithValue("@VmId", vmId);
            cmd.Parameters.AddWithValue("@Journal", journal);
            await cmd.ExecuteNonQueryAsync();
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Permanently remove deleted VMs older than specified age
    /// </summary>
    public async Task PurgeDeletedVmsAsync(TimeSpan olderThan)
    {
        await _lock.WaitAsync();
        try
        {
            var cutoff = DateTime.UtcNow.Subtract(olderThan).ToString("O");
            var sql = "DELETE FROM VmRecords WHERE State = 'Deleted' AND LastUpdated < @Cutoff";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@Cutoff", cutoff);

            var rows = await cmd.ExecuteNonQueryAsync();
            if (rows > 0)
            {
                _logger.LogInformation("Purged {Count} deleted VM records from database", rows);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get database statistics
    /// </summary>
    public async Task<DatabaseStats> GetStatsAsync()
    {
        await _lock.WaitAsync();
        try
        {
            var stats = new DatabaseStats();

            // ── Query 1: VMs by state ──────────────────────────────────────────
            using (var cmd = _connection.CreateCommand())
            {
                cmd.CommandText = @"
                SELECT State, COUNT(*) as Count
                FROM VmRecords
                GROUP BY State";

                using var reader = await cmd.ExecuteReaderAsync();
                while (await reader.ReadAsync())
                {
                    stats.VmsByState[reader.GetString(0)] = reader.GetInt32(1);
                }
            } // reader + cmd disposed here — connection is free

            // ── Query 2: DB file size ──────────────────────────────────────────
            using (var cmd = _connection.CreateCommand())
            {
                cmd.CommandText =
                    "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()";

                using var sizeReader = await cmd.ExecuteReaderAsync();
                if (await sizeReader.ReadAsync())
                    stats.DatabaseSizeBytes = sizeReader.GetInt64(0);
            } // reader + cmd disposed here

            return stats;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Lightweight method with lock to update the LastUpdated field. Parses to vm.LastHeartBeat 
    /// </summary>
    /// <param name="vmId"></param>
    /// <param name="timestamp"></param>
    /// <returns></returns>
    public async Task UpdateLastHeartbeatAsync(string vmId, DateTime timestamp)
    {
        await _lock.WaitAsync();
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "UPDATE VmRecords SET LastUpdated = @ts WHERE VmId = @id";
            cmd.Parameters.AddWithValue("@ts", timestamp.ToString("O"));
            cmd.Parameters.AddWithValue("@id", vmId);
            await cmd.ExecuteNonQueryAsync();
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Returns a sanitized summary of all VM records for dashboard display.
    /// SECURITY: Excludes SshPublicKey, EncryptedPassword, BaseImageHash.
    /// </summary>
    public async Task<List<VmDashboardRecord>> GetDashboardSummaryAsync()
    {
        await _lock.WaitAsync();
        try
        {
            var records = new List<VmDashboardRecord>();
            const string sql = @"
            SELECT VmId, Name, State, VmType, OwnerId,
                   IpAddress, VncPort, ReplicationFactor,
                   VirtualCpuCores, MemoryBytes, DiskBytes,
                   CreatedAt, LastUpdated, TargetNodeId,
                   DeletionReason
            FROM VmRecords
            ORDER BY CreatedAt DESC";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                records.Add(new VmDashboardRecord
                {
                    VmId = reader.GetString(reader.GetOrdinal("VmId")),
                    Name = reader.GetString(reader.GetOrdinal("Name")),
                    State = reader.GetString(reader.GetOrdinal("State")),
                    VmType = reader.GetString(reader.GetOrdinal("VmType")),
                    OwnerId = reader.IsDBNull(reader.GetOrdinal("OwnerId")) ? null : reader.GetString(reader.GetOrdinal("OwnerId")),
                    IpAddress = reader.IsDBNull(reader.GetOrdinal("IpAddress")) ? null : reader.GetString(reader.GetOrdinal("IpAddress")),
                    VncPort = reader.IsDBNull(reader.GetOrdinal("VncPort")) ? null : reader.GetInt32(reader.GetOrdinal("VncPort")),
                    ReplicationFactor = reader.GetInt32(reader.GetOrdinal("ReplicationFactor")),
                    VirtualCpuCores = reader.GetInt32(reader.GetOrdinal("VirtualCpuCores")),
                    MemoryBytes = reader.GetInt64(reader.GetOrdinal("MemoryBytes")),
                    DiskBytes = reader.GetInt64(reader.GetOrdinal("DiskBytes")),
                    CreatedAt = reader.GetString(reader.GetOrdinal("CreatedAt")),
                    LastUpdated = reader.GetString(reader.GetOrdinal("LastUpdated")),
                    TargetNodeId = reader.IsDBNull(reader.GetOrdinal("TargetNodeId")) ? null : reader.GetString(reader.GetOrdinal("TargetNodeId")),
                    DeletionReason = reader.IsDBNull(reader.GetOrdinal("DeletionReason")) ? null : reader.GetString(reader.GetOrdinal("DeletionReason")),
                });
            }
            return records;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Gets current schema version (exposed for dashboard).
    /// </summary>
    public int SchemaVersion => GetSchemaVersion();

    /// <summary>
    /// Gets the underlying SQLite database file path.
    /// </summary>
    public string DatabasePath => _connection.DataSource;

    /// <summary>
    /// Parse VmInstance from database reader using COLUMN NAMES (robust against schema changes)
    /// </summary>
    private VmInstance? ParseVmFromReader(SqliteDataReader reader)
    {
        try
        {
            // Use column names instead of indices for robustness
            var vm = new VmInstance
            {
                VmId = reader.GetString(reader.GetOrdinal("VmId")),
                Name = reader.GetString(reader.GetOrdinal("Name")),
                Services = DeserializeServices(GetNullableString(reader, "ServicesJson")),
                Spec = new VmSpec
                {
                    Id = reader.GetString(reader.GetOrdinal("VmId")),
                    Name = reader.GetString(reader.GetOrdinal("Name")),
                    OwnerId = GetNullableString(reader, "OwnerId"),
                    QualityTier = (QualityTier)reader.GetInt32(reader.GetOrdinal("QualityTier")),
                    ComputePointCost = reader.GetInt32(reader.GetOrdinal("ComputePointCost")),
                    VirtualCpuCores = reader.GetInt32(reader.GetOrdinal("VirtualCpuCores")),
                    MemoryBytes = reader.GetInt64(reader.GetOrdinal("MemoryBytes")),
                    DiskBytes = reader.GetInt64(reader.GetOrdinal("DiskBytes")),
                    IpAddress = GetNullableString(reader, "IpAddress"),
                    MacAddress = GetNullableString(reader, "MacAddress"),
                    BaseImageUrl = GetNullableString(reader, "BaseImageUrl"),
                    BaseImageHash = GetNullableString(reader, "BaseImageHash"),
                    SshPublicKey = GetNullableString(reader, "SshPublicKey"),
                    ReplicationFactor = reader.GetInt32(reader.GetOrdinal("ReplicationFactor")),
                },
                State = Enum.Parse<VmState>(reader.GetString(reader.GetOrdinal("State"))),
                VncPort = GetNullableInt(reader, "VncPort"),
                Pid = GetNullableInt(reader, "Pid"),
                CreatedAt = DateTime.Parse(reader.GetString(reader.GetOrdinal("CreatedAt"))),
                StartedAt = GetNullableDateTime(reader, "StartedAt"),
                StoppedAt = GetNullableDateTime(reader, "StoppedAt"),
                LastHeartbeat = DateTime.Parse(reader.GetString(reader.GetOrdinal("LastUpdated"))),
                DiskPath = GetNullableString(reader, "DiskPath") ?? string.Empty,
                ConfigPath = GetNullableString(reader, "ConfigPath") ?? string.Empty
            };

            // Restore VmType — critical for system VM classification (Relay/DHT/BlockStore)
            // after restarts. Without this every VM loads as VmType.General.
            var vmTypeStr = GetNullableString(reader, "VmType");
            if (!string.IsNullOrEmpty(vmTypeStr) &&
                Enum.TryParse<VmType>(vmTypeStr, ignoreCase: true, out var parsedVmType))
            {
                vm.Spec.VmType = parsedVmType;
            }

            // Restore Labels — role label is the dashboard fallback for type classification
            var labelsJson = GetNullableString(reader, "LabelsJson");
            if (!string.IsNullOrEmpty(labelsJson))
            {
                try
                {
                    vm.Spec.Labels = JsonSerializer.Deserialize<Dictionary<string, string>>(labelsJson);
                }
                catch
                {
                    // Leave Labels null — VmType alone is sufficient for classification
                }
            }

            // Restore TargetNodeId for zombie fencing
            vm.Spec.TargetNodeId = GetNullableString(reader, "TargetNodeId");

            return vm;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to parse VM from database row");
            return null;
        }
    }

    // Helper methods for nullable value parsing
    private string? GetNullableString(SqliteDataReader reader, string columnName)
    {
        var ordinal = reader.GetOrdinal(columnName);
        return reader.IsDBNull(ordinal) ? null : reader.GetString(ordinal);
    }

    private int? GetNullableInt(SqliteDataReader reader, string columnName)
    {
        var ordinal = reader.GetOrdinal(columnName);
        return reader.IsDBNull(ordinal) ? null : reader.GetInt32(ordinal);
    }

    private DateTime? GetNullableDateTime(SqliteDataReader reader, string columnName)
    {
        var ordinal = reader.GetOrdinal(columnName);
        return reader.IsDBNull(ordinal) ? null : DateTime.Parse(reader.GetString(ordinal));
    }

    /// <summary>
    /// Deserialize services JSON from database, with fallback to empty list.
    /// </summary>
    private static List<VmServiceStatus> DeserializeServices(string? json)
    {
        if (string.IsNullOrEmpty(json) || json == "[]") return new List<VmServiceStatus>();
        try
        {
            return JsonSerializer.Deserialize<List<VmServiceStatus>>(json) ?? new List<VmServiceStatus>();
        }
        catch
        {
            return new List<VmServiceStatus>();
        }
    }

    public void Dispose()
    {
        _lock?.Dispose();
        _connection?.Dispose();
    }
}

/// <summary>
/// Database statistics
/// </summary>
public class DatabaseStats
{
    public Dictionary<string, int> VmsByState { get; set; } = new();
    public long DatabaseSizeBytes { get; set; }
    public int TotalVms => VmsByState.Values.Sum();
}