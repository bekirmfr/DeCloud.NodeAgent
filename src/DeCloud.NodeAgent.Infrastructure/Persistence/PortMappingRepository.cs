using DeCloud.NodeAgent.Core.Models;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Persistence;

/// <summary>
/// SQLite-based repository for persisting port mappings.
/// Ensures port allocations survive node restarts.
/// </summary>
public class PortMappingRepository : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly ILogger<PortMappingRepository> _logger;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private const int CURRENT_SCHEMA_VERSION = 1;

    public PortMappingRepository(string databasePath, ILogger<PortMappingRepository> logger)
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
    }

    private void InitializeDatabase()
    {
        var createTable = @"
            CREATE TABLE IF NOT EXISTS PortMappings (
                Id TEXT PRIMARY KEY,
                VmId TEXT NOT NULL,
                VmPrivateIp TEXT NOT NULL,
                VmPort INTEGER NOT NULL,
                PublicPort INTEGER NOT NULL UNIQUE,
                Protocol INTEGER NOT NULL,
                Label TEXT,
                CreatedAt TEXT NOT NULL,
                IsActive INTEGER NOT NULL DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_port_vm ON PortMappings(VmId);
            CREATE INDEX IF NOT EXISTS idx_port_public ON PortMappings(PublicPort);
            CREATE INDEX IF NOT EXISTS idx_port_active ON PortMappings(IsActive);
        ";

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = createTable;
        cmd.ExecuteNonQuery();

        _logger.LogDebug("PortMappings database initialized");
    }

    /// <summary>
    /// Add a new port mapping
    /// </summary>
    public async Task<bool> AddAsync(PortMapping mapping)
    {
        await _lock.WaitAsync();
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
                INSERT INTO PortMappings
                (Id, VmId, VmPrivateIp, VmPort, PublicPort, Protocol, Label, CreatedAt, IsActive)
                VALUES
                (@Id, @VmId, @VmPrivateIp, @VmPort, @PublicPort, @Protocol, @Label, @CreatedAt, @IsActive)";

            cmd.Parameters.AddWithValue("@Id", mapping.Id);
            cmd.Parameters.AddWithValue("@VmId", mapping.VmId);
            cmd.Parameters.AddWithValue("@VmPrivateIp", mapping.VmPrivateIp);
            cmd.Parameters.AddWithValue("@VmPort", mapping.VmPort);
            cmd.Parameters.AddWithValue("@PublicPort", mapping.PublicPort);
            cmd.Parameters.AddWithValue("@Protocol", (int)mapping.Protocol);
            cmd.Parameters.AddWithValue("@Label", mapping.Label ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue("@CreatedAt", mapping.CreatedAt.ToString("O"));
            cmd.Parameters.AddWithValue("@IsActive", mapping.IsActive ? 1 : 0);

            await cmd.ExecuteNonQueryAsync();

            _logger.LogInformation(
                "Port mapping added: {PublicPort} → {VmIp}:{VmPort} (VM {VmId})",
                mapping.PublicPort, mapping.VmPrivateIp, mapping.VmPort, mapping.VmId);

            return true;
        }
        catch (SqliteException ex) when (ex.SqliteErrorCode == 19) // UNIQUE constraint
        {
            _logger.LogError(
                "Port {PublicPort} is already allocated",
                mapping.PublicPort);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to add port mapping");
            return false;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Remove a port mapping by VM ID and VM port
    /// </summary>
    public async Task<bool> RemoveAsync(string vmId, int vmPort)
    {
        await _lock.WaitAsync();
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "DELETE FROM PortMappings WHERE VmId = @VmId AND VmPort = @VmPort";
            cmd.Parameters.AddWithValue("@VmId", vmId);
            cmd.Parameters.AddWithValue("@VmPort", vmPort);

            var rowsAffected = await cmd.ExecuteNonQueryAsync();

            if (rowsAffected > 0)
            {
                _logger.LogInformation(
                    "Port mapping removed: VM {VmId} port {VmPort}",
                    vmId, vmPort);
            }

            return rowsAffected > 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to remove port mapping");
            return false;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Remove a port mapping by public port (used for relay nodes where all mappings have VmPort=0)
    /// </summary>
    public async Task<bool> RemoveByPublicPortAsync(int publicPort)
    {
        await _lock.WaitAsync();
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "DELETE FROM PortMappings WHERE PublicPort = @PublicPort";
            cmd.Parameters.AddWithValue("@PublicPort", publicPort);

            var rowsAffected = await cmd.ExecuteNonQueryAsync();

            if (rowsAffected > 0)
            {
                _logger.LogInformation(
                    "Port mapping removed: PublicPort {PublicPort}",
                    publicPort);
            }

            return rowsAffected > 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to remove port mapping by public port");
            return false;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Remove all port mappings for a VM
    /// </summary>
    public async Task<int> RemoveAllForVmAsync(string vmId)
    {
        await _lock.WaitAsync();
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "DELETE FROM PortMappings WHERE VmId = @VmId";
            cmd.Parameters.AddWithValue("@VmId", vmId);

            var rowsAffected = await cmd.ExecuteNonQueryAsync();

            if (rowsAffected > 0)
            {
                _logger.LogInformation(
                    "Removed {Count} port mappings for VM {VmId}",
                    rowsAffected, vmId);
            }

            return rowsAffected;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to remove port mappings for VM {VmId}", vmId);
            return 0;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get all port mappings for a VM
    /// </summary>
    public async Task<List<PortMapping>> GetByVmIdAsync(string vmId)
    {
        await _lock.WaitAsync();
        try
        {
            var mappings = new List<PortMapping>();

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT * FROM PortMappings WHERE VmId = @VmId ORDER BY VmPort";
            cmd.Parameters.AddWithValue("@VmId", vmId);

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                mappings.Add(ReadPortMapping(reader));
            }

            return mappings;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get all allocated public ports
    /// </summary>
    public async Task<HashSet<int>> GetAllocatedPortsAsync()
    {
        await _lock.WaitAsync();
        try
        {
            var ports = new HashSet<int>();

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT PublicPort FROM PortMappings WHERE IsActive = 1";

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                ports.Add(reader.GetInt32(0));
            }

            return ports;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Check if a port is already allocated
    /// </summary>
    public async Task<bool> IsPortAllocatedAsync(int port)
    {
        await _lock.WaitAsync();
        try
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT COUNT(*) FROM PortMappings WHERE PublicPort = @Port AND IsActive = 1";
            cmd.Parameters.AddWithValue("@Port", port);

            var count = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            return count > 0;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get all active port mappings (for reconciliation after restart)
    /// </summary>
    public async Task<List<PortMapping>> GetAllActiveAsync()
    {
        await _lock.WaitAsync();
        try
        {
            var mappings = new List<PortMapping>();

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT * FROM PortMappings WHERE IsActive = 1 ORDER BY VmId, VmPort";

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                mappings.Add(ReadPortMapping(reader));
            }

            return mappings;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get port utilization statistics
    /// </summary>
    public async Task<(int total, int used, double utilization)> GetUtilizationAsync()
    {
        var totalPorts = 25535; // 65535 - 40000 + 1
        var used = (await GetAllocatedPortsAsync()).Count;
        var utilization = (double)used / totalPorts;

        return (totalPorts, used, utilization);
    }

    private PortMapping ReadPortMapping(SqliteDataReader reader)
    {
        return new PortMapping
        {
            Id = reader.GetString(reader.GetOrdinal("Id")),
            VmId = reader.GetString(reader.GetOrdinal("VmId")),
            VmPrivateIp = reader.GetString(reader.GetOrdinal("VmPrivateIp")),
            VmPort = reader.GetInt32(reader.GetOrdinal("VmPort")),
            PublicPort = reader.GetInt32(reader.GetOrdinal("PublicPort")),
            Protocol = (PortProtocol)reader.GetInt32(reader.GetOrdinal("Protocol")),
            Label = reader.IsDBNull(reader.GetOrdinal("Label"))
                ? null
                : reader.GetString(reader.GetOrdinal("Label")),
            CreatedAt = DateTime.Parse(reader.GetString(reader.GetOrdinal("CreatedAt"))),
            IsActive = reader.GetInt32(reader.GetOrdinal("IsActive")) == 1
        };
    }

    public void Dispose()
    {
        _lock?.Dispose();
        _connection?.Dispose();
    }
}
