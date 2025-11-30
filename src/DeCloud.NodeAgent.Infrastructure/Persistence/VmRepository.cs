using DeCloud.NodeAgent.Core.Models;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using System.Security.Cryptography;
using System.Text;

namespace DeCloud.NodeAgent.Infrastructure.Persistence;

/// <summary>
/// Enhanced SQLite-based repository for persisting VM state on node agents.
/// Provides resilience across node agent restarts with optional encryption.
/// 
/// SECURITY: This version adds encryption support using a deterministic key
/// derived from node ID and wallet address.
/// </summary>
public class VmRepositoryEncrypted : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private readonly bool _encrypted;

    public VmRepositoryEncrypted(string databasePath, ILogger logger, string? encryptionKey = null)
    {
        _logger = logger;

        // Ensure directory exists
        var directory = Path.GetDirectoryName(databasePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Generate encryption key if not provided
        _encrypted = !string.IsNullOrEmpty(encryptionKey);

        if (_encrypted)
        {
            // Use SQLCipher for encrypted database
            // Note: Requires SQLCipher NuGet package or Microsoft.Data.Sqlite with encryption
            _connection = new SqliteConnection($"Data Source={databasePath};Password={encryptionKey}");
            _logger.LogInformation("✓ VmRepository initialized with encryption at {Path}", databasePath);
        }
        else
        {
            _connection = new SqliteConnection($"Data Source={databasePath}");
            _logger.LogWarning("⚠ VmRepository initialized WITHOUT encryption at {Path}", databasePath);
        }

        _connection.Open();
        InitializeDatabase();
    }

    /// <summary>
    /// Generate a secure encryption key from node-specific data
    /// This provides deterministic encryption that survives node agent restarts
    /// </summary>
    public static string GenerateEncryptionKey(string nodeId, string walletAddress)
    {
        // Combine node ID and wallet for deterministic key generation
        var input = $"{nodeId}:{walletAddress}:decloud-vm-encryption";
        var bytes = Encoding.UTF8.GetBytes(input);

        // Use SHA-256 to generate a 32-byte key
        var hash = SHA256.HashData(bytes);
        return Convert.ToBase64String(hash);
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

        _logger.LogDebug("Database schema initialized{Encrypted}",
            _encrypted ? " with encryption" : " without encryption");
    }

    // ... (rest of methods remain the same as VmRepository)

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
            cmd.Parameters.AddWithValue("@TenantId", vm.Spec.TenantId);
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

            _logger.LogDebug("Saved VM {VmId} to encrypted database", vm.VmId);
        }
        finally
        {
            _lock.Release();
        }
    }

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

            _logger.LogInformation("Loaded {Count} VMs from encrypted database", vms.Count);
            return vms;
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task UpdateVmStateAsync(string vmId, VmState newState)
    {
        await _lock.WaitAsync();
        try
        {
            var sql = "UPDATE VmRecords SET State = @State, LastUpdated = @Now WHERE VmId = @VmId";

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = sql;
            cmd.Parameters.AddWithValue("@VmId", vmId);
            cmd.Parameters.AddWithValue("@State", newState.ToString());
            cmd.Parameters.AddWithValue("@Now", DateTime.UtcNow.ToString("O"));

            await cmd.ExecuteNonQueryAsync();
        }
        finally
        {
            _lock.Release();
        }
    }

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