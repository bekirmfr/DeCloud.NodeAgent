// =====================================================
// NAT Rule Manager for Relay VMs
// =====================================================
//
// Manages iptables NAT rules to forward WireGuard traffic
// from host's public IP to relay VM's internal IP.
//
// This version uses the decloud-relay-nat script which provides:
// - Automatic cleanup of old/stale rules
// - Proper FORWARD rule ordering (position 1)
// - Connection tracking cleanup
// - Complete NAT setup (PREROUTING + POSTROUTING + FORWARD)
// - Atomic operations (clean-then-add)

using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

public class NatRuleManager : INatRuleManager
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<NatRuleManager> _logger;
    private readonly bool _isLinux;
    private const string NAT_SCRIPT = "/usr/local/bin/decloud-relay-nat";

    public NatRuleManager(
        ICommandExecutor executor,
        ILogger<NatRuleManager> logger)
    {
        _executor = executor;
        _logger = logger;
        _isLinux = Environment.OSVersion.Platform == PlatformID.Unix;
    }

    /// <summary>
    /// Adds NAT forwarding for a relay VM using the decloud-relay-nat script
    /// </summary>
    public async Task<bool> AddPortForwardingAsync(
        string vmIp,
        int port,
        string protocol = "udp",
        CancellationToken ct = default)
    {
        if (!_isLinux)
        {
            _logger.LogWarning("NAT rule management not supported on non-Linux platforms");
            return false;
        }

        // Only support port 51820 (WireGuard) for relay VMs
        if (port != 51820)
        {
            _logger.LogWarning(
                "NAT forwarding only supported for port 51820 (WireGuard), not {Port}",
                port);
            return false;
        }

        try
        {
            _logger.LogInformation(
                "Configuring NAT for relay VM: {Protocol}/{Port} → {VmIp}:{Port}",
                protocol.ToUpper(), port, vmIp, port);

            // Check if NAT script exists
            if (!File.Exists(NAT_SCRIPT))
            {
                _logger.LogError(
                    "NAT script not found: {Path}. Install via install.sh or deploy-relay-nat.sh",
                    NAT_SCRIPT);
                return false;
            }

            // Detect public interface (eth0, ens3, etc.)
            var publicInterface = await DetectPublicInterfaceAsync(ct);

            _logger.LogInformation(
                "Using public interface: {Interface} for NAT configuration",
                publicInterface);

            // Execute NAT script to add rules
            // The script will:
            // 1. Clean all old relay NAT rules
            // 2. Add PREROUTING DNAT rule
            // 3. Add POSTROUTING MASQUERADE rule
            // 4. Add FORWARD rule at position 1 (ensures priority)
            // 5. Clear connection tracking
            // 6. Save rules persistently
            var result = await _executor.ExecuteAsync(
                NAT_SCRIPT,
                $"add {vmIp} {publicInterface}",
                ct);

            if (!result.Success)
            {
                _logger.LogError(
                    "Failed to configure NAT for {VmIp}: {Error}",
                    vmIp, result.StandardError);
                return false;
            }

            _logger.LogInformation(
                "✓ NAT configured successfully: {Interface}:51820 → {VmIp}:51820",
                publicInterface, vmIp);

            _logger.LogDebug("NAT script output: {Output}", result.StandardOutput);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Error configuring NAT for {VmIp}:{Port}",
                vmIp, port);
            return false;
        }
    }

    /// <summary>
    /// Removes NAT forwarding for a specific relay VM
    /// Note: The 'add' command already cleans old rules, so this is rarely needed
    /// </summary>
    public async Task<bool> RemovePortForwardingAsync(
        string vmIp,
        int port,
        string protocol = "udp",
        CancellationToken ct = default)
    {
        if (!_isLinux)
        {
            _logger.LogWarning("NAT rule management not supported on non-Linux platforms");
            return false;
        }

        if (port != 51820)
        {
            _logger.LogWarning("Only port 51820 NAT removal supported");
            return false;
        }

        try
        {
            _logger.LogInformation(
                "Removing NAT for relay VM: {VmIp}:{Port}",
                vmIp, port);

            if (!File.Exists(NAT_SCRIPT))
            {
                _logger.LogWarning("NAT script not found: {Path}", NAT_SCRIPT);
                return false;
            }

            var result = await _executor.ExecuteAsync(
                NAT_SCRIPT,
                $"remove {vmIp}",
                ct);

            if (!result.Success)
            {
                _logger.LogWarning(
                    "Failed to remove NAT for {VmIp} (may not exist): {Error}",
                    vmIp, result.StandardError);
                return false;
            }

            _logger.LogInformation("✓ NAT removed for {VmIp}", vmIp);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing NAT for {VmIp}", vmIp);
            return false;
        }
    }

    /// <summary>
    /// Checks if NAT rule exists for a relay VM
    /// </summary>
    public async Task<bool> RuleExistsAsync(
        string vmIp,
        int port,
        string protocol = "udp",
        CancellationToken ct = default)
    {
        if (!_isLinux) return false;

        try
        {
            // Check if PREROUTING rule exists using iptables directly
            var result = await _executor.ExecuteAsync(
                "iptables",
                $"-t nat -C PREROUTING -p {protocol} --dport {port} " +
                $"-j DNAT --to-destination {vmIp}:{port}",
                ct);

            return result.Success;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Saves iptables rules persistently (handled by NAT script)
    /// </summary>
    public async Task<bool> SaveRulesAsync(CancellationToken ct = default)
    {
        if (!_isLinux) return false;

        try
        {
            if (!File.Exists(NAT_SCRIPT))
            {
                _logger.LogDebug("NAT script not found, skipping save");
                return false;
            }

            var result = await _executor.ExecuteAsync(
                NAT_SCRIPT,
                "save",
                ct);

            return result.Success;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error saving NAT rules");
            return false;
        }
    }

    /// <summary>
    /// Removes all relay NAT rules (cleanup)
    /// </summary>
    public async Task<bool> RemoveAllRelayNatRulesAsync(CancellationToken ct = default)
    {
        if (!_isLinux) return false;

        try
        {
            _logger.LogInformation("Cleaning all relay NAT rules...");

            if (!File.Exists(NAT_SCRIPT))
            {
                _logger.LogWarning("NAT script not found: {Path}", NAT_SCRIPT);
                return false;
            }

            var result = await _executor.ExecuteAsync(
                NAT_SCRIPT,
                "clean",
                ct);

            if (!result.Success)
            {
                _logger.LogWarning(
                    "Failed to clean NAT rules: {Error}",
                    result.StandardError);
                return false;
            }

            _logger.LogInformation("✓ All relay NAT rules cleaned");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error cleaning relay NAT rules");
            return false;
        }
    }

    /// <summary>
    /// Gets current NAT rules from the script
    /// </summary>
    public async Task<List<string>> GetExistingRulesAsync(CancellationToken ct = default)
    {
        if (!_isLinux) return new List<string>();

        try
        {
            if (!File.Exists(NAT_SCRIPT))
            {
                return new List<string>();
            }

            var result = await _executor.ExecuteAsync(
                NAT_SCRIPT,
                "show",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                return result.StandardOutput
                    .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                    .ToList();
            }

            return new List<string>();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error getting existing NAT rules");
            return new List<string>();
        }
    }

    /// <summary>
    /// Detects the public network interface
    /// </summary>
    private async Task<string> DetectPublicInterfaceAsync(CancellationToken ct = default)
    {
        try
        {
            // Try to find default route interface
            var result = await _executor.ExecuteAsync(
                "ip",
                "route show default",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                // Parse output like: "default via 142.234.200.1 dev eth0 proto static"
                var parts = result.StandardOutput.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var devIndex = Array.IndexOf(parts, "dev");
                if (devIndex >= 0 && devIndex + 1 < parts.Length)
                {
                    var iface = parts[devIndex + 1];
                    _logger.LogDebug("Detected public interface: {Interface}", iface);
                    return iface;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error detecting public interface, using default 'eth0'");
        }

        // Default fallback
        return "eth0";
    }
}

// =====================================================
// Usage Example in CommandProcessorService
// =====================================================
/*

// In CommandProcessorService.HandleCreateVmAsync(), after relay VM is created:

if (vmType == (int)VmType.Relay && result.Success)
{
    _logger.LogInformation("Relay VM {VmId} created, configuring NAT rules...", vmId);
    
    // Wait for VM to get IP address (up to 60 seconds)
    string? vmIp = null;
    var maxRetries = 12; // 12 * 5 seconds = 60 seconds
    
    for (int i = 0; i < maxRetries; i++)
    {
        await Task.Delay(5000, ct);
        
        var vmInstance = await _vmManager.GetVmAsync(vmId, ct);
        if (vmInstance != null && !string.IsNullOrEmpty(vmInstance.IpAddress))
        {
            vmIp = vmInstance.IpAddress;
            _logger.LogInformation("✓ Relay VM {VmId} obtained IP: {IpAddress}", vmId, vmIp);
            break;
        }
        
        _logger.LogDebug(
            "Waiting for relay VM {VmId} to obtain IP address (attempt {Attempt}/{Max})", 
            vmId, i + 1, maxRetries);
    }
    
    if (vmIp != null)
    {
        // Configure NAT using decloud-relay-nat script
        // This automatically:
        // - Cleans old rules
        // - Adds PREROUTING DNAT
        // - Adds POSTROUTING MASQUERADE
        // - Adds FORWARD at position 1
        // - Clears connection tracking
        // - Saves rules persistently
        var natSuccess = await _natRuleManager.AddPortForwardingAsync(
            vmIp, 
            51820,  // WireGuard port
            "udp", 
            ct);
        
        if (natSuccess)
        {
            _logger.LogInformation(
                "✓ NAT configured for relay VM {VmId}: Public:51820 → {VmIp}:51820",
                vmId, vmIp);
        }
        else
        {
            _logger.LogError(
                "Failed to configure NAT for relay VM {VmId}. " +
                "CGNAT nodes may not be able to connect!",
                vmId);
        }
    }
    else
    {
        _logger.LogError(
            "Relay VM {VmId} did not obtain IP within 60 seconds. " +
            "NAT not configured!",
            vmId);
    }
}

// In CommandProcessorService.HandleDeleteVmAsync(), before VM deletion:
// Note: The 'add' command cleans old rules automatically, so removal is optional

var vmInstance = await _vmManager.GetVmAsync(vmId, ct);
if (vmInstance?.Spec.VmType == VmType.Relay && !string.IsNullOrEmpty(vmInstance.IpAddress))
{
    _logger.LogInformation("Removing NAT rules for relay VM {VmId}", vmId);
    
    await _natRuleManager.RemovePortForwardingAsync(
        vmInstance.IpAddress,
        51820,
        "udp",
        ct);
}

// For debugging/diagnostics:
var existingRules = await _natRuleManager.GetExistingRulesAsync(ct);
_logger.LogInformation("Current NAT rules:\n{Rules}", string.Join("\n", existingRules));

*/