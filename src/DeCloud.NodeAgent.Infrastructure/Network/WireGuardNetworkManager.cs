using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Infrastructure.Network;

public class WireGuardOptions
{
    public string InterfaceName { get; set; } = "wg0";
    public string ConfigPath { get; set; } = "";  // Auto-detected based on OS
    public int ListenPort { get; set; } = 51820;
    public string PrivateKeyPath { get; set; } = "";  // Auto-detected
    public string PublicKeyPath { get; set; } = "";   // Auto-detected
    public string Address { get; set; } = "";
}

public class WireGuardNetworkManager : INetworkManager
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<WireGuardNetworkManager> _logger;
    private readonly WireGuardOptions _options;
    private readonly bool _isWindows;
    private readonly string _configPath;
    private readonly string _privateKeyPath;
    private readonly string _publicKeyPath;

    public WireGuardNetworkManager(
        ICommandExecutor executor,
        IOptions<WireGuardOptions> options,
        ILogger<WireGuardNetworkManager> logger)
    {
        _executor = executor;
        _logger = logger;
        _options = options.Value;
        _isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        // Set platform-specific paths
        if (_isWindows)
        {
            _configPath = string.IsNullOrEmpty(_options.ConfigPath) 
                ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "WireGuard", "Data")
                : _options.ConfigPath;
        }
        else
        {
            _configPath = string.IsNullOrEmpty(_options.ConfigPath) ? "/etc/wireguard" : _options.ConfigPath;
        }

        _privateKeyPath = string.IsNullOrEmpty(_options.PrivateKeyPath)
            ? Path.Combine(_configPath, "private.key")
            : _options.PrivateKeyPath;
            
        _publicKeyPath = string.IsNullOrEmpty(_options.PublicKeyPath)
            ? Path.Combine(_configPath, "public.key")
            : _options.PublicKeyPath;
    }

    public async Task<bool> InitializeWireGuardAsync(CancellationToken ct = default)
    {
        _logger.LogInformation("Initializing WireGuard (Platform: {Platform})", _isWindows ? "Windows" : "Linux");

        // Check if WireGuard is installed
        var checkResult = await _executor.ExecuteAsync(
            _isWindows ? "where" : "which",
            _isWindows ? "wg" : "wg",
            ct);

        if (!checkResult.Success)
        {
            _logger.LogWarning("WireGuard not found. Install WireGuard to enable overlay networking.");
            _logger.LogWarning(_isWindows 
                ? "Download from: https://www.wireguard.com/install/" 
                : "Install with: apt install wireguard");
            return false;
        }

        try
        {
            Directory.CreateDirectory(_configPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Cannot create WireGuard config directory at {Path}", _configPath);
            return false;
        }

        // Generate keys if they don't exist
        if (!File.Exists(_privateKeyPath))
        {
            _logger.LogInformation("Generating new WireGuard keypair");

            var genKeyResult = await _executor.ExecuteAsync("wg", "genkey", ct);
            if (!genKeyResult.Success)
            {
                _logger.LogError("Failed to generate private key: {Error}", genKeyResult.StandardError);
                return false;
            }

            var privateKey = genKeyResult.StandardOutput.Trim();
            
            try
            {
                await File.WriteAllTextAsync(_privateKeyPath, privateKey, ct);
                
                if (!_isWindows)
                {
                    await _executor.ExecuteAsync("chmod", $"600 {_privateKeyPath}", ct);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to save private key");
                return false;
            }

            // Generate public key
            CommandResult pubKeyResult;
            if (_isWindows)
            {
                pubKeyResult = await _executor.ExecuteAsync("powershell",
                    $"-NoProfile -Command \"'{privateKey}' | wg pubkey\"", ct);
            }
            else
            {
                pubKeyResult = await _executor.ExecuteAsync("sh",
                    $"-c \"echo '{privateKey}' | wg pubkey\"", ct);
            }

            if (!pubKeyResult.Success)
            {
                _logger.LogError("Failed to generate public key: {Error}", pubKeyResult.StandardError);
                return false;
            }

            await File.WriteAllTextAsync(_publicKeyPath, pubKeyResult.StandardOutput.Trim(), ct);
        }

        // On Windows, WireGuard uses its own service/GUI - we just prepare the config
        if (_isWindows)
        {
            _logger.LogInformation("WireGuard keys ready. Use WireGuard GUI to manage tunnels on Windows.");
            return true;
        }

        // Linux: Create and configure interface
        var showResult = await _executor.ExecuteAsync("ip", $"link show {_options.InterfaceName}", ct);
        if (!showResult.Success)
        {
            var createResult = await _executor.ExecuteAsync("ip", 
                $"link add {_options.InterfaceName} type wireguard", ct);
            
            if (!createResult.Success && !createResult.StandardError.Contains("exists"))
            {
                _logger.LogError("Failed to create WireGuard interface: {Error}", createResult.StandardError);
                return false;
            }
        }

        // Set private key
        var privateKeyContent = await File.ReadAllTextAsync(_privateKeyPath, ct);
        var setKeyResult = await _executor.ExecuteAsync("sh",
            $"-c \"echo '{privateKeyContent.Trim()}' | wg set {_options.InterfaceName} private-key /dev/stdin listen-port {_options.ListenPort}\"",
            ct);

        if (!setKeyResult.Success)
        {
            _logger.LogError("Failed to set WireGuard private key: {Error}", setKeyResult.StandardError);
            return false;
        }

        // Set address if configured
        if (!string.IsNullOrEmpty(_options.Address))
        {
            await _executor.ExecuteAsync("ip", $"addr add {_options.Address} dev {_options.InterfaceName}", ct);
        }

        // Bring up interface
        var upResult = await _executor.ExecuteAsync("ip", $"link set {_options.InterfaceName} up", ct);
        if (!upResult.Success)
        {
            _logger.LogError("Failed to bring up WireGuard interface: {Error}", upResult.StandardError);
            return false;
        }

        _logger.LogInformation("WireGuard interface {Interface} initialized successfully", _options.InterfaceName);
        return true;
    }

    public async Task<string> GetWireGuardPublicKeyAsync(CancellationToken ct = default)
    {
        if (File.Exists(_publicKeyPath))
        {
            return (await File.ReadAllTextAsync(_publicKeyPath, ct)).Trim();
        }

        if (_isWindows)
        {
            return string.Empty;  // On Windows, keys managed by WG GUI
        }

        var result = await _executor.ExecuteAsync("wg", $"show {_options.InterfaceName} public-key", ct);
        return result.Success ? result.StandardOutput.Trim() : string.Empty;
    }

    public async Task AddPeerAsync(string publicKey, string endpoint, string allowedIps, CancellationToken ct = default)
    {
        _logger.LogInformation("Adding WireGuard peer: {PublicKey} at {Endpoint}", 
            publicKey.Length > 8 ? publicKey[..8] + "..." : publicKey, endpoint);

        if (_isWindows)
        {
            _logger.LogWarning("Peer management on Windows requires using WireGuard GUI or config files");
            return;
        }

        var args = $"set {_options.InterfaceName} peer {publicKey}";
        
        if (!string.IsNullOrEmpty(endpoint))
            args += $" endpoint {endpoint}";
        
        if (!string.IsNullOrEmpty(allowedIps))
            args += $" allowed-ips {allowedIps}";

        args += " persistent-keepalive 25";

        var result = await _executor.ExecuteAsync("wg", args, ct);
        
        if (!result.Success)
            throw new Exception($"Failed to add WireGuard peer: {result.StandardError}");

        if (!string.IsNullOrEmpty(allowedIps))
        {
            foreach (var ip in allowedIps.Split(','))
            {
                await _executor.ExecuteAsync("ip", $"route add {ip.Trim()} dev {_options.InterfaceName}", ct);
            }
        }
    }

    public async Task RemovePeerAsync(string publicKey, CancellationToken ct = default)
    {
        if (_isWindows)
        {
            _logger.LogWarning("Peer management on Windows requires using WireGuard GUI");
            return;
        }

        _logger.LogInformation("Removing WireGuard peer: {PublicKey}", 
            publicKey.Length > 8 ? publicKey[..8] + "..." : publicKey);

        var result = await _executor.ExecuteAsync("wg", 
            $"set {_options.InterfaceName} peer {publicKey} remove", ct);

        if (!result.Success && !result.StandardError.Contains("not found"))
            throw new Exception($"Failed to remove WireGuard peer: {result.StandardError}");
    }

    public async Task<List<WireGuardPeer>> GetPeersAsync(CancellationToken ct = default)
    {
        var peers = new List<WireGuardPeer>();

        var result = await _executor.ExecuteAsync("wg", $"show {_options.InterfaceName} dump", ct);
        if (!result.Success) return peers;

        var lines = result.StandardOutput.Split('\n').Skip(1);
        
        foreach (var line in lines)
        {
            var parts = line.Split('\t');
            if (parts.Length < 5) continue;

            var peer = new WireGuardPeer
            {
                PublicKey = parts[0],
                Endpoint = parts[2] == "(none)" ? string.Empty : parts[2],
                AllowedIps = parts[3],
                TransferRx = long.TryParse(parts.Length > 5 ? parts[5] : "0", out var rx) ? rx : 0,
                TransferTx = long.TryParse(parts.Length > 6 ? parts[6] : "0", out var tx) ? tx : 0
            };

            if (long.TryParse(parts[4], out var handshakeTimestamp) && handshakeTimestamp > 0)
            {
                peer.LastHandshake = DateTimeOffset.FromUnixTimeSeconds(handshakeTimestamp).UtcDateTime;
            }

            peers.Add(peer);
        }

        return peers;
    }

    public async Task<string> CreateVmNetworkAsync(string vmId, VmNetworkConfig config, CancellationToken ct = default)
    {
        if (_isWindows)
        {
            _logger.LogWarning("VM networking not supported on Windows (requires Linux bridge/tap interfaces)");
            return $"vnic-{vmId[..8]}";
        }

        _logger.LogInformation("Creating network for VM {VmId}", vmId);

        var tapName = $"tap-{vmId[..Math.Min(8, vmId.Length)]}";

        var tapResult = await _executor.ExecuteAsync("ip", $"tuntap add {tapName} mode tap", ct);
        if (!tapResult.Success && !tapResult.StandardError.Contains("exists"))
            throw new Exception($"Failed to create tap interface: {tapResult.StandardError}");

        if (!string.IsNullOrEmpty(config.MacAddress))
            await _executor.ExecuteAsync("ip", $"link set {tapName} address {config.MacAddress}", ct);

        await _executor.ExecuteAsync("ip", $"link set {tapName} up", ct);
        await _executor.ExecuteAsync("ip", $"link set {tapName} master virbr0", ct);

        return tapName;
    }

    public async Task DeleteVmNetworkAsync(string vmId, CancellationToken ct = default)
    {
        if (_isWindows) return;

        var tapName = $"tap-{vmId[..Math.Min(8, vmId.Length)]}";
        _logger.LogInformation("Deleting network for VM {VmId}", vmId);

        await _executor.ExecuteAsync("ip", $"link set {tapName} nomaster", ct);
        await _executor.ExecuteAsync("ip", $"link delete {tapName}", ct);
    }

    public async Task<bool> StartWireGuardInterfaceAsync(string interfaceName, CancellationToken ct = default)
    {
        if (_isWindows)
        {
            _logger.LogWarning("wg-quick not available on Windows - use WireGuard GUI");
            return false;
        }

        _logger.LogInformation("Starting WireGuard interface: {Interface}", interfaceName);

        var result = await _executor.ExecuteAsync("wg-quick", $"up {interfaceName}", ct);

        if (result.Success)
        {
            _logger.LogInformation("WireGuard interface {Interface} started successfully", interfaceName);
            return true;
        }

        if (result.StandardError.Contains("already exists") || result.StandardError.Contains("RTNETLINK answers: File exists"))
        {
            _logger.LogInformation("WireGuard interface {Interface} already running", interfaceName);
            return true;
        }

        _logger.LogError("Failed to start WireGuard interface {Interface}: {Error}",
            interfaceName, result.StandardError);
        return false;
    }
}
