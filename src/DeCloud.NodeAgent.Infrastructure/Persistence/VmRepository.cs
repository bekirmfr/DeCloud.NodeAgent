using DeCloud.NodeAgent.Core.Models;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Persistence;

/// <summary>
/// SQLite-based repository for persisting VM state on node agents.
/// Provides resilience across node agent restarts.
/// </summary>
public class VmRepository : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _lock = new(1, 1);

    public VmRepository(string databasePath, ILogger logger)
    {
        _logger = logger;

        // Ensure directory exists
        var directory = Path.GetDirectoryName(databasePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        _connection = new SqliteConnection($"Data Source={databasePath}");
        _connection.Open();

        InitializeDatabase();

        _logger.LogInformation("VmRepository initialized at {Path}", databasePath);
    }

    private void InitializeDatabase()
    {
        var createTable = @"
            CREATE TABLE IF NOT EXISTS VmRecords (
                VmId TEXT PRIMARY KEY,
                Name TEXT NOT NULL,
                TenantId TEXT NOT NULL,
                LeaseId TEXT,
                VCpus INTEGER NOT NULL,
                MemoryBytes INTEGER NOT NULL,
                DiskBytes INTEGER NOT NULL,
                State TEXT NOT NULL,
                IpAddress TEXT,
                MacAddress TEXT,
                VncPort TEXT,
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
                Password TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_tenant ON VmRecords(TenantId);
            CREATE INDEX IF NOT EXISTS idx_state ON VmRecords(State);
            CREATE INDEX IF NOT EXISTS idx_updated ON VmRecords(LastUpdated);
        ";

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = createTable;
        cmd.ExecuteNonQuery();

        _logger.LogDebug("Database schema initialized");
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
                (VmId, Name, TenantId, LeaseId, VCpus, MemoryBytes, DiskBytes, 
                 State, IpAddress, MacAddress, VncPort, Pid,
                 CreatedAt, StartedAt, StoppedAt, LastUpdated, DiskPath, ConfigPath,
                 BaseImageUrl, BaseImageHash, SshPublicKey, Password)
                VALUES 
                (@VmId, @Name, @TenantId, @LeaseId, @VCpus, @MemoryBytes, @DiskBytes,
                 @State, @IpAddress, @MacAddress, @VncPort, @Pid,
                 @CreatedAt, @StartedAt, @StoppedAt, @LastUpdated, @DiskPath, @ConfigPath,
                 @BaseImageUrl, @BaseImageHash, @SshPublicKey, @Password)
            ";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;

            cmd.Parameters.AddWithValue("@VmId", vm.VmId);
            cmd.Parameters.AddWithValue("@Name", vm.Name);
            cmd.Parameters.AddWithValue("@TenantId", vm.Spec.TenantId ?? "");
            cmd.Parameters.AddWithValue("@LeaseId", vm.Spec.LeaseId ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@VCpus", vm.Spec.VCpus);
            cmd.Parameters.AddWithValue("@MemoryBytes", vm.Spec.MemoryBytes);
            cmd.Parameters.AddWithValue("@DiskBytes", vm.Spec.DiskBytes);
            cmd.Parameters.AddWithValue("@State", vm.State.ToString());
            cmd.Parameters.AddWithValue("@IpAddress", vm.Spec.Network.IpAddress ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@MacAddress", vm.Spec.Network.MacAddress ?? (object)DBNull.Value);
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
            cmd.Parameters.AddWithValue("@Password", vm.Spec.Password ?? (object)DBNull.Value);

            await cmd.ExecuteNonQueryAsync();

            _logger.LogDebug("Saved VM {VmId} to database", vm.VmId);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Load all VMs from database (excluding deleted ones)
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
                    var vm = new VmInstance
                    {
                        VmId = reader.GetString(0),
                        Name = reader.GetString(1),
                        Spec = new VmSpec
                        {
                            VmId = reader.GetString(0),
                            Name = reader.GetString(1),
                            TenantId = reader.GetString(2),
                            LeaseId = reader.IsDBNull(3) ? string.Empty : reader.GetString(3),
                            VCpus = reader.GetInt32(4),
                            MemoryBytes = reader.GetInt64(5),
                            DiskBytes = reader.GetInt64(6),
                            Network = new VmNetworkConfig
                            {
                                IpAddress = reader.IsDBNull(8) ? string.Empty : reader.GetString(8),
                                MacAddress = reader.IsDBNull(9) ? string.Empty : reader.GetString(9)
                            },
                            BaseImageUrl = reader.IsDBNull(18) ? string.Empty : reader.GetString(18),
                            BaseImageHash = reader.IsDBNull(19) ? string.Empty : reader.GetString(19),
                            SshPublicKey = reader.IsDBNull(20) ? null : reader.GetString(20),
                            Password = reader.IsDBNull(21) ? null : reader.GetString(21)
                        },
                        State = Enum.Parse<VmState>(reader.GetString(7)),
                        VncPort = reader.IsDBNull(10) ? null : reader.GetString(10),
                        Pid = reader.IsDBNull(11) ? null : reader.GetInt32(11),
                        CreatedAt = DateTime.Parse(reader.GetString(12)),
                        StartedAt = reader.IsDBNull(13) ? null : DateTime.Parse(reader.GetString(13)),
                        StoppedAt = reader.IsDBNull(14) ? null : DateTime.Parse(reader.GetString(14)),
                        LastHeartbeat = DateTime.Parse(reader.GetString(15)),
                        DiskPath = reader.IsDBNull(16) ? string.Empty : reader.GetString(16),
                        ConfigPath = reader.IsDBNull(17) ? string.Empty : reader.GetString(17)
                    };

                    vms.Add(vm);
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
    /// Load a specific VM by ID
    /// </summary>
    public async Task<VmInstance?> LoadVmAsync(string vmId)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = "SELECT * FROM VmRecords WHERE VmId = @VmId LIMIT 1";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);

            using var reader = await cmd.ExecuteReaderAsync();
            if (await reader.ReadAsync())
            {
                return new VmInstance
                {
                    VmId = reader.GetString(0),
                    Name = reader.GetString(1),
                    Spec = new VmSpec
                    {
                        VmId = reader.GetString(0),
                        Name = reader.GetString(1),
                        TenantId = reader.GetString(2),
                        LeaseId = reader.IsDBNull(3) ? string.Empty : reader.GetString(3),
                        VCpus = reader.GetInt32(4),
                        MemoryBytes = reader.GetInt64(5),
                        DiskBytes = reader.GetInt64(6),
                        Network = new VmNetworkConfig
                        {
                            IpAddress = reader.IsDBNull(8) ? string.Empty : reader.GetString(8),
                            MacAddress = reader.IsDBNull(9) ? string.Empty : reader.GetString(9)
                        },
                        BaseImageUrl = reader.IsDBNull(18) ? string.Empty : reader.GetString(18),
                        BaseImageHash = reader.IsDBNull(19) ? string.Empty : reader.GetString(19),
                        SshPublicKey = reader.IsDBNull(20) ? null : reader.GetString(20),
                        Password = reader.IsDBNull(21) ? null : reader.GetString(21)
                    },
                    State = Enum.Parse<VmState>(reader.GetString(7)),
                    VncPort = reader.IsDBNull(10) ? null : reader.GetString(10),
                    Pid = reader.IsDBNull(11) ? null : reader.GetInt32(11),
                    CreatedAt = DateTime.Parse(reader.GetString(12)),
                    StartedAt = reader.IsDBNull(13) ? null : DateTime.Parse(reader.GetString(13)),
                    StoppedAt = reader.IsDBNull(14) ? null : DateTime.Parse(reader.GetString(14)),
                    LastHeartbeat = DateTime.Parse(reader.GetString(15)),
                    DiskPath = reader.IsDBNull(16) ? string.Empty : reader.GetString(16),
                    ConfigPath = reader.IsDBNull(17) ? string.Empty : reader.GetString(17)
                };
            }

            return null;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Update VM state
    /// </summary>
    public async Task UpdateVmStateAsync(string vmId, VmState newState)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = @"
                UPDATE VmRecords 
                SET State = @State, 
                    LastUpdated = @LastUpdated,
                    StartedAt = CASE WHEN @State = 'Running' AND StartedAt IS NULL THEN @Now ELSE StartedAt END,
                    StoppedAt = CASE WHEN @State = 'Stopped' THEN @Now ELSE StoppedAt END
                WHERE VmId = @VmId
            ";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);
            cmd.Parameters.AddWithValue("@State", newState.ToString());
            cmd.Parameters.AddWithValue("@LastUpdated", DateTime.UtcNow.ToString("O"));
            cmd.Parameters.AddWithValue("@Now", DateTime.UtcNow.ToString("O"));

            await cmd.ExecuteNonQueryAsync();
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
    /// Delete a VM from database
    /// </summary>
    public async Task DeleteVmAsync(string vmId)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = "DELETE FROM VmRecords WHERE VmId = @VmId";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);

            var rows = await cmd.ExecuteNonQueryAsync();

            if (rows > 0)
            {
                _logger.LogInformation("Deleted VM {VmId} from database", vmId);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get count of VMs by state
    /// </summary>
    public async Task<Dictionary<VmState, int>> GetVmCountsByStateAsync()
    {
        await _lock.WaitAsync();
        try
        {
            var counts = new Dictionary<VmState, int>();
            var sql = "SELECT State, COUNT(*) FROM VmRecords WHERE State != 'Deleted' GROUP BY State";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var state = Enum.Parse<VmState>(reader.GetString(0));
                var count = reader.GetInt32(1);
                counts[state] = count;
            }

            return counts;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Clean up old deleted VMs (permanent removal from DB)
    /// </summary>
    public async Task<int> PurgeOldDeletedVmsAsync(TimeSpan olderThan)
    {
        await _lock.WaitAsync();
        try
        {
            var cutoffDate = DateTime.UtcNow.Subtract(olderThan).ToString("O");
            var sql = "DELETE FROM VmRecords WHERE State = 'Deleted' AND LastUpdated < @Cutoff";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@Cutoff", cutoffDate);

            var deleted = await cmd.ExecuteNonQueryAsync();

            if (deleted > 0)
            {
                _logger.LogInformation("Purged {Count} old deleted VMs from database", deleted);
            }

            return deleted;
        }
        finally
        {
            _lock.Release();
        }
    }

    public void Dispose()
    {
        _lock?.Dispose();
        _connection?.Dispose();
    }
}