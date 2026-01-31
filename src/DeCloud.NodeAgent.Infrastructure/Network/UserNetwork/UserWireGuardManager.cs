using System.Security.Cryptography;
using System.Text;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.UserNetwork;
using DeCloud.NodeAgent.Core.Models.UserNetwork;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Network.UserNetwork;

/// <summary>
/// Manages per-user WireGuard networks for secure VM isolation
/// Each user gets a dedicated WireGuard interface with unique cryptographic keys
/// Provides stronger isolation than bridges and encrypts all traffic (even same-node)
/// </summary>
public class UserWireGuardManager : IUserWireGuardManager
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<UserWireGuardManager> _logger;

    // Track active user networks in memory
    private readonly Dictionary<string, UserWireGuardConfig> _userNetworks = new();
    private readonly SemaphoreSlim _lock = new(1, 1);

    // Port allocation starts from 51821 (51820 is for relay)
    private int _nextPort = 51821;

    public UserWireGuardManager(
        ICommandExecutor executor,
        ILogger<UserWireGuardManager> logger)
    {
        _executor = executor;
        _logger = logger;
    }

    public async Task<UserWireGuardConfig> EnsureUserNetworkAsync(
        string userId,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            // Check if already exists
            if (_userNetworks.TryGetValue(userId, out var existing))
            {
                _logger.LogDebug(
                    "User network already exists for {UserId}: {Interface}",
                    userId, existing.InterfaceName);
                return existing;
            }

            _logger.LogInformation(
                "Creating WireGuard network for user {UserId}",
                userId);

            // Compute deterministic subnet from user ID
            var subnet = ComputeUserSubnet(userId);
            // Linux interface names limited to 15 chars (wg-user- = 8 chars, so max 7 from userId)
            var interfaceName = $"wg-user-{userId.Substring(0, Math.Min(7, userId.Length))}";

            // Generate WireGuard keypair
            var privateKey = await GenerateWireGuardPrivateKeyAsync(ct);
            var publicKey = await DerivePublicKeyAsync(privateKey, ct);

            // Create configuration
            var config = new UserWireGuardConfig
            {
                UserId = userId,
                InterfaceName = interfaceName,
                PrivateKey = privateKey,
                PublicKey = publicKey,
                Subnet = subnet,
                LocalIp = $"{subnet}.1",
                ListenPort = _nextPort++,
                CreatedAt = DateTime.UtcNow,
                LastModified = DateTime.UtcNow
            };

            // Create WireGuard interface
            await CreateWireGuardInterfaceAsync(config, ct);

            // Store in memory
            _userNetworks[userId] = config;

            _logger.LogInformation(
                "✓ Created user network {Interface} for {UserId}: {Subnet}.0/24 (port {Port})",
                interfaceName, userId, subnet, config.ListenPort);

            return config;
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task<string> AllocateVmIpAsync(
        string userId,
        string vmId,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            if (!_userNetworks.TryGetValue(userId, out var config))
            {
                throw new InvalidOperationException(
                    $"User network not found for {userId}. " +
                    "Call EnsureUserNetworkAsync first.");
            }

            // Get used IPs
            var usedIps = config.VmPeers
                .Select(p => p.VmIp)
                .ToHashSet();

            // Start from .10 (skip .1 for node, .2-.9 reserved for future use)
            for (int i = 10; i < 255; i++)
            {
                var candidateIp = $"{config.Subnet}.{i}";
                if (!usedIps.Contains(candidateIp))
                {
                    _logger.LogDebug(
                        "Allocated IP {Ip} for VM {VmId} in user network {Interface}",
                        candidateIp, vmId, config.InterfaceName);
                    return candidateIp;
                }
            }

            throw new InvalidOperationException(
                $"No available IPs in subnet {config.Subnet}.0/24");
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task AddVmPeerAsync(
        string userId,
        string vmId,
        string vmIp,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            if (!_userNetworks.TryGetValue(userId, out var config))
            {
                throw new InvalidOperationException(
                    $"User network not found for {userId}");
            }

            _logger.LogInformation(
                "Adding VM {VmId} ({VmIp}) to user network {Interface}",
                vmId, vmIp, config.InterfaceName);

            // Add to peer list
            config.VmPeers.Add(new VmPeerInfo
            {
                VmId = vmId,
                VmIp = vmIp,
                AddedAt = DateTime.UtcNow
            });

            // Add route for this VM IP through WireGuard interface
            await _executor.ExecuteAsync(
                "ip",
                $"route add {vmIp}/32 dev {config.InterfaceName}",
                ct);

            config.LastModified = DateTime.UtcNow;

            _logger.LogInformation(
                "✓ VM {VmId} added to user network {Interface} at {VmIp}",
                vmId, config.InterfaceName, vmIp);
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task RemoveVmPeerAsync(
        string userId,
        string vmId,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            if (!_userNetworks.TryGetValue(userId, out var config))
            {
                _logger.LogWarning(
                    "User network not found for {UserId} when removing VM {VmId}",
                    userId, vmId);
                return;
            }

            var peer = config.VmPeers.FirstOrDefault(p => p.VmId == vmId);
            if (peer == null)
            {
                _logger.LogWarning(
                    "VM {VmId} not found in user network {Interface}",
                    vmId, config.InterfaceName);
                return;
            }

            _logger.LogInformation(
                "Removing VM {VmId} ({VmIp}) from user network {Interface}",
                vmId, peer.VmIp, config.InterfaceName);

            // Remove route
            await _executor.ExecuteAsync(
                "ip",
                $"route del {peer.VmIp}/32 dev {config.InterfaceName}",
                ct);

            // Remove from peer list
            config.VmPeers.Remove(peer);
            config.LastModified = DateTime.UtcNow;

            _logger.LogInformation(
                "✓ VM {VmId} removed from user network {Interface}",
                vmId, config.InterfaceName);
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task CleanupUserNetworkIfEmptyAsync(
        string userId,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            if (!_userNetworks.TryGetValue(userId, out var config))
            {
                return;
            }

            if (config.VmPeers.Count > 0)
            {
                _logger.LogDebug(
                    "User network {Interface} has {Count} VMs, not cleaning up",
                    config.InterfaceName, config.VmPeers.Count);
                return;
            }

            _logger.LogInformation(
                "Cleaning up empty user network {Interface} for {UserId}",
                config.InterfaceName, userId);

            // Stop and delete WireGuard interface
            await DeleteWireGuardInterfaceAsync(config, ct);

            // Remove from tracking
            _userNetworks.Remove(userId);

            _logger.LogInformation(
                "✓ Cleaned up user network {Interface}",
                config.InterfaceName);
        }
        finally
        {
            _lock.Release();
        }
    }

    public Task<UserWireGuardConfig?> GetUserNetworkAsync(
        string userId,
        CancellationToken ct = default)
    {
        _userNetworks.TryGetValue(userId, out var config);
        return Task.FromResult(config);
    }

    public Task<List<UserWireGuardConfig>> ListUserNetworksAsync(
        CancellationToken ct = default)
    {
        return Task.FromResult(_userNetworks.Values.ToList());
    }

    // ============================================================
    // Private Helper Methods
    // ============================================================

    /// <summary>
    /// Compute deterministic subnet for a user using SHA256
    /// Returns base like "10.100.42" (user adds .0/24)
    /// </summary>
    private string ComputeUserSubnet(string userId)
    {
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(userId));
        var octet = hash[0] % 255; // Use first byte mod 255

        // Avoid subnet 0 and 255
        if (octet == 0) octet = 1;
        if (octet == 255) octet = 254;

        return $"10.100.{octet}";
    }

    /// <summary>
    /// Generate a new WireGuard private key
    /// </summary>
    private async Task<string> GenerateWireGuardPrivateKeyAsync(
        CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync("wg", "genkey", ct);
        if (!result.Success)
        {
            throw new InvalidOperationException(
                $"Failed to generate WireGuard private key: {result.StandardError}");
        }
        return result.StandardOutput.Trim();
    }

    /// <summary>
    /// Derive public key from private key
    /// </summary>
    private async Task<string> DerivePublicKeyAsync(
        string privateKey,
        CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync(
            "bash",
            $"-c \"echo '{privateKey}' | wg pubkey\"",
            ct);

        if (!result.Success)
        {
            throw new InvalidOperationException(
                $"Failed to derive public key: {result.StandardError}");
        }
        return result.StandardOutput.Trim();
    }

    /// <summary>
    /// Create and configure WireGuard interface
    /// </summary>
    private async Task CreateWireGuardInterfaceAsync(
        UserWireGuardConfig config,
        CancellationToken ct)
    {
        // Create interface
        var createResult = await _executor.ExecuteAsync(
            "ip",
            $"link add {config.InterfaceName} type wireguard",
            ct);

        if (!createResult.Success &&
            !createResult.StandardError.Contains("exists"))
        {
            throw new InvalidOperationException(
                $"Failed to create WireGuard interface: {createResult.StandardError}");
        }

        // Write temporary config file for wg setconf
        var configContent = $@"[Interface]
PrivateKey = {config.PrivateKey}
ListenPort = {config.ListenPort}
";

        var tempFile = Path.Combine(
            Path.GetTempPath(),
            $"wg-{config.InterfaceName}-{Guid.NewGuid()}.conf");

        try
        {
            await File.WriteAllTextAsync(tempFile, configContent, ct);

            // Configure WireGuard
            var setResult = await _executor.ExecuteAsync(
                "wg",
                $"setconf {config.InterfaceName} {tempFile}",
                ct);

            if (!setResult.Success)
            {
                throw new InvalidOperationException(
                    $"Failed to configure WireGuard: {setResult.StandardError}");
            }

            // Assign IP address
            await _executor.ExecuteAsync(
                "ip",
                $"addr add {config.LocalIp}/24 dev {config.InterfaceName}",
                ct);

            // Bring interface up
            await _executor.ExecuteAsync(
                "ip",
                $"link set {config.InterfaceName} up",
                ct);

            // Enable IP forwarding (needed for VM-to-VM communication)
            await _executor.ExecuteAsync(
                "sysctl",
                "-w net.ipv4.ip_forward=1",
                ct);

            _logger.LogDebug(
                "WireGuard interface {Interface} created and configured",
                config.InterfaceName);
        }
        finally
        {
            // Clean up temp file
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    /// <summary>
    /// Delete WireGuard interface
    /// </summary>
    private async Task DeleteWireGuardInterfaceAsync(
        UserWireGuardConfig config,
        CancellationToken ct)
    {
        try
        {
            // Bring interface down
            await _executor.ExecuteAsync(
                "ip",
                $"link set {config.InterfaceName} down",
                ct);

            // Delete interface
            await _executor.ExecuteAsync(
                "ip",
                $"link delete {config.InterfaceName}",
                ct);

            _logger.LogDebug(
                "WireGuard interface {Interface} deleted",
                config.InterfaceName);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(
                ex,
                "Error deleting WireGuard interface {Interface}",
                config.InterfaceName);
        }
    }
}