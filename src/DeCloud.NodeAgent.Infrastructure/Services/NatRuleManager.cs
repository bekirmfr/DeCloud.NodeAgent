// =====================================================
// NAT Rule Manager for Relay VMs
// =====================================================
// File: src/DeCloud.NodeAgent.Infrastructure/Services/NatRuleManager.cs
//
// Manages iptables NAT rules to forward WireGuard traffic
// from host's public IP to relay VM's internal IP.
//
// This is critical for relay VMs to accept connections from
// CGNAT nodes that connect to the host's public IP:51820

using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

public class NatRuleManager : INatRuleManager
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<NatRuleManager> _logger;
    private readonly bool _isLinux;

    public NatRuleManager(
        ICommandExecutor executor,
        ILogger<NatRuleManager> logger)
    {
        _executor = executor;
        _logger = logger;
        _isLinux = Environment.OSVersion.Platform == PlatformID.Unix;
    }

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

        try
        {
            _logger.LogInformation(
                "Adding NAT rule: {Protocol}/{Port} → {VmIp}:{Port}",
                protocol.ToUpper(), port, vmIp, port);

            // Check if rule already exists
            if (await RuleExistsAsync(vmIp, port, protocol, ct))
            {
                _logger.LogInformation(
                    "NAT rule already exists for {VmIp}:{Port}/{Protocol}",
                    vmIp, port, protocol.ToUpper());
                return true;
            }

            // Add PREROUTING rule for DNAT (destination NAT)
            // This redirects incoming traffic on the host's public IP to the VM
            var preRoutingRule = $"-t nat -A PREROUTING -p {protocol} --dport {port} " +
                               $"-j DNAT --to-destination {vmIp}:{port}";

            var preRoutingResult = await _executor.ExecuteAsync(
                "iptables",
                preRoutingRule,
                ct);

            if (!preRoutingResult.Success)
            {
                _logger.LogError(
                    "Failed to add PREROUTING rule: {Error}",
                    preRoutingResult.StandardError);
                return false;
            }

            // Add FORWARD rule to allow forwarded traffic
            var forwardRule = $"-A FORWARD -p {protocol} -d {vmIp} --dport {port} -j ACCEPT";

            var forwardResult = await _executor.ExecuteAsync(
                "iptables",
                forwardRule,
                ct);

            if (!forwardResult.Success)
            {
                _logger.LogWarning(
                    "Failed to add FORWARD rule (may already exist): {Error}",
                    forwardResult.StandardError);
                // Don't fail if FORWARD rule fails - it might already exist
            }

            _logger.LogInformation(
                "✓ NAT rule added successfully: {Protocol}/{Port} → {VmIp}:{Port}",
                protocol.ToUpper(), port, vmIp, port);

            // Save rules persistently
            await SaveRulesAsync(ct);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Error adding NAT rule for {VmIp}:{Port}/{Protocol}",
                vmIp, port, protocol.ToUpper());
            return false;
        }
    }

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

        try
        {
            _logger.LogInformation(
                "Removing NAT rule: {Protocol}/{Port} → {VmIp}:{Port}",
                protocol.ToUpper(), port, vmIp, port);

            // Remove PREROUTING rule
            var preRoutingRule = $"-t nat -D PREROUTING -p {protocol} --dport {port} " +
                               $"-j DNAT --to-destination {vmIp}:{port}";

            var preRoutingResult = await _executor.ExecuteAsync(
                "iptables",
                preRoutingRule,
                ct);

            if (!preRoutingResult.Success)
            {
                _logger.LogWarning(
                    "Failed to remove PREROUTING rule (may not exist): {Error}",
                    preRoutingResult.StandardError);
            }

            // Remove FORWARD rule
            var forwardRule = $"-D FORWARD -p {protocol} -d {vmIp} --dport {port} -j ACCEPT";

            var forwardResult = await _executor.ExecuteAsync(
                "iptables",
                forwardRule,
                ct);

            if (!forwardResult.Success)
            {
                _logger.LogWarning(
                    "Failed to remove FORWARD rule (may not exist): {Error}",
                    forwardResult.StandardError);
            }

            _logger.LogInformation(
                "✓ NAT rule removed: {Protocol}/{Port} → {VmIp}:{Port}",
                protocol.ToUpper(), port, vmIp, port);

            // Save rules persistently
            await SaveRulesAsync(ct);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Error removing NAT rule for {VmIp}:{Port}/{Protocol}",
                vmIp, port, protocol.ToUpper());
            return false;
        }
    }

    public async Task<bool> RuleExistsAsync(
        string vmIp,
        int port,
        string protocol = "udp",
        CancellationToken ct = default)
    {
        if (!_isLinux) return false;

        try
        {
            // Check if PREROUTING rule exists
            var checkResult = await _executor.ExecuteAsync(
                "iptables",
                $"-t nat -C PREROUTING -p {protocol} --dport {port} " +
                $"-j DNAT --to-destination {vmIp}:{port}",
                ct);

            return checkResult.Success;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex,
                "Error checking if NAT rule exists for {VmIp}:{Port}/{Protocol}",
                vmIp, port, protocol.ToUpper());
            return false;
        }
    }

    public async Task<bool> SaveRulesAsync(CancellationToken ct = default)
    {
        if (!_isLinux) return false;

        try
        {
            _logger.LogDebug("Saving iptables rules persistently");

            // Try netfilter-persistent (Debian/Ubuntu)
            var netfilterResult = await _executor.ExecuteAsync(
                "which",
                "netfilter-persistent",
                ct);

            if (netfilterResult.Success)
            {
                var saveResult = await _executor.ExecuteAsync(
                    "netfilter-persistent",
                    "save",
                    ct);

                if (saveResult.Success)
                {
                    _logger.LogDebug("✓ Rules saved via netfilter-persistent");
                    return true;
                }
            }

            // Fallback: Try iptables-save directly
            var saveFileResult = await _executor.ExecuteAsync(
                "sh",
                "-c \"iptables-save > /etc/iptables/rules.v4\"",
                ct);

            if (saveFileResult.Success)
            {
                _logger.LogDebug("✓ Rules saved to /etc/iptables/rules.v4");
                return true;
            }

            _logger.LogWarning(
                "Could not save iptables rules persistently. " +
                "Rules will be lost on reboot. Install netfilter-persistent or configure iptables-save.");

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error saving iptables rules persistently");
            return false;
        }
    }
}

// =====================================================
// Usage Example in CommandProcessorService
// =====================================================
/*

// In CommandProcessorService.HandleCreateVmAsync(), after VM is created:

if (vmType == (int)VmType.Relay && result.Success)
{
    _logger.LogInformation("Relay VM {VmId} created, configuring NAT rules...", vmId);
    
    // Wait for VM to get IP address (up to 60 seconds)
    string? vmIp = null;
    var maxRetries = 12; // 12 * 5 seconds = 60 seconds
    
    for (int i = 0; i < maxRetries; i++)
    {
        await Task.Delay(5000, ct); // Wait 5 seconds
        
        var vmInstance = await _vmManager.GetVmAsync(vmId, ct);
        if (vmInstance != null && !string.IsNullOrEmpty(vmInstance.IpAddress))
        {
            vmIp = vmInstance.IpAddress;
            _logger.LogInformation("Relay VM {VmId} obtained IP: {IpAddress}", vmId, vmIp);
            break;
        }
        
        _logger.LogDebug("Waiting for relay VM {VmId} to obtain IP address (attempt {Attempt}/{Max})", 
            vmId, i + 1, maxRetries);
    }
    
    if (vmIp != null)
    {
        // Add NAT rule to forward WireGuard traffic to relay VM
        var natSuccess = await _natRuleManager.AddPortForwardingAsync(
            vmIp, 
            51820,  // WireGuard port
            "udp", 
            ct);
        
        if (natSuccess)
        {
            _logger.LogInformation(
                "✓ Relay VM {VmId} NAT rule configured: UDP/51820 → {VmIp}:51820",
                vmId, vmIp);
        }
        else
        {
            _logger.LogError(
                "Failed to configure NAT rule for relay VM {VmId}. " +
                "CGNAT nodes will not be able to connect!",
                vmId);
        }
    }
    else
    {
        _logger.LogError(
            "Relay VM {VmId} did not obtain IP address within 60 seconds. " +
            "NAT rule not configured!",
            vmId);
    }
}

// In CommandProcessorService.HandleDeleteVmAsync(), before VM deletion:

// Check if this is a relay VM and remove NAT rules
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

*/