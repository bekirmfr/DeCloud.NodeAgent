using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Manages iptables port forwarding rules for Smart Port Allocation.
/// Creates DNAT rules to forward public ports to VM internal ports.
///
/// Separate from NatRuleManager (which handles Relay VMs).
/// </summary>
public interface IPortForwardingManager
{
    /// <summary>
    /// Create port forwarding rule: PublicPort → VM:VmPort
    /// </summary>
    Task<bool> CreateForwardingAsync(
        string vmIp,
        int vmPort,
        int publicPort,
        PortProtocol protocol,
        CancellationToken ct = default);

    /// <summary>
    /// Remove port forwarding rule (both DNAT and FORWARD)
    /// </summary>
    Task<bool> RemoveForwardingAsync(
        string vmIp,
        int vmPort,
        int publicPort,
        PortProtocol protocol,
        CancellationToken ct = default);

    /// <summary>
    /// Remove all forwarding rules for a VM
    /// </summary>
    Task<bool> RemoveAllForVmAsync(
        string vmIp,
        CancellationToken ct = default);

    /// <summary>
    /// Reconcile iptables rules with database (after restart)
    /// </summary>
    Task ReconcileRulesAsync(CancellationToken ct = default);

    /// <summary>
    /// Check if forwarding rule exists
    /// </summary>
    Task<bool> RuleExistsAsync(int publicPort, CancellationToken ct = default);
}

public class PortForwardingManager : IPortForwardingManager
{
    private readonly ICommandExecutor _executor;
    private readonly PortMappingRepository _repository;
    private readonly IVmManager _vmManager;
    private readonly ILogger<PortForwardingManager> _logger;
    private readonly bool _isLinux;
    private readonly SemaphoreSlim _lock = new(1, 1);

    // Chain name for port forwarding rules (keeps them organized)
    private const string CHAIN_NAME = "DECLOUD_PORT_FWD";
    
    // DeCloud tunnel network range (10.20.0.0/16)
    private const string TUNNEL_IP_PREFIX = "10.20.";

    public PortForwardingManager(
        ICommandExecutor executor,
        PortMappingRepository repository,
        IVmManager vmManager,
        ILogger<PortForwardingManager> logger)
    {
        _executor = executor;
        _repository = repository;
        _vmManager = vmManager;
        _logger = logger;
        _isLinux = Environment.OSVersion.Platform == PlatformID.Unix;
    }

    public async Task<bool> CreateForwardingAsync(
        string vmIp,
        int vmPort,
        int publicPort,
        PortProtocol protocol,
        CancellationToken ct = default)
    {
        if (!_isLinux)
        {
            _logger.LogWarning("Port forwarding not supported on non-Linux platforms");
            return false;
        }

        bool lockAcquired = false;
        try
        {
            // Don't pass ct to WaitAsync - prevents cancellation from crashing host
            await _lock.WaitAsync();
            lockAcquired = true;
            
            // Ensure our custom chain exists
            await EnsureChainExistsAsync(ct);

            // Check if this is relay forwarding (tunnel IP destination)
            string actualDestination = vmIp;
            int actualPort = vmPort;
            
            if (IsTunnelIp(vmIp))
            {
                _logger.LogInformation(
                    "Detected tunnel IP {TunnelIp} - checking for local relay VM...",
                    vmIp);
                
                var relayVmIp = await GetRelayVmIpAsync(ct);
                if (relayVmIp != null)
                {
                    _logger.LogInformation(
                        "Found relay VM at {RelayVmIp} - will forward through it",
                        relayVmIp);
                    
                    // Forward to relay VM on the same public port
                    // The relay VM will then forward to the tunnel IP
                    actualDestination = relayVmIp;
                    actualPort = publicPort;
                    
                    _logger.LogInformation(
                        "Creating 2-hop forwarding: :{PublicPort} → {RelayVmIp}:{Port} → {TunnelIp}:{VmPort}",
                        publicPort, relayVmIp, publicPort, vmIp, vmPort);
                }
                else
                {
                    _logger.LogWarning(
                        "Tunnel IP {TunnelIp} detected but no relay VM found - direct forwarding may fail",
                        vmIp);
                }
            }

            _logger.LogInformation(
                "Creating port forwarding: :{PublicPort} → {Destination}:{Port} ({Protocol})",
                publicPort, actualDestination, actualPort, protocol);

            // Create forwarding rules based on protocol
            if (protocol == PortProtocol.TCP || protocol == PortProtocol.Both)
            {
                await CreateIptablesRuleAsync("tcp", actualDestination, actualPort, publicPort, ct);
            }

            if (protocol == PortProtocol.UDP || protocol == PortProtocol.Both)
            {
                await CreateIptablesRuleAsync("udp", actualDestination, actualPort, publicPort, ct);
            }

            // Save rules persistently
            await SaveRulesAsync(ct);

            _logger.LogInformation(
                "✓ Port forwarding created: :{PublicPort} → {Destination}:{Port}",
                publicPort, actualDestination, actualPort);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to create port forwarding for {VmIp}:{VmPort} → :{PublicPort}",
                vmIp, vmPort, publicPort);
            return false;
        }
        finally
        {
            if (lockAcquired)
            {
                _lock.Release();
            }
        }
    }

    public async Task<bool> RemoveForwardingAsync(
        string vmIp,
        int vmPort,
        int publicPort,
        PortProtocol protocol,
        CancellationToken ct = default)
    {
        if (!_isLinux)
        {
            return false;
        }

        bool lockAcquired = false;
        try
        {
            await _lock.WaitAsync(ct);
            lockAcquired = true;
            
            _logger.LogInformation(
                "Removing port forwarding for {VmIp}:{VmPort} → :{PublicPort} ({Protocol})",
                vmIp, vmPort, publicPort, protocol);

            bool success = true;

            // Remove TCP rules
            if (protocol == PortProtocol.TCP || protocol == PortProtocol.Both)
            {
                // Remove DNAT rule
                var result = await _executor.ExecuteAsync(
                    "iptables",
                    $"-t nat -D {CHAIN_NAME} -p tcp --dport {publicPort} -j DNAT",
                    ct);

                if (!result.Success)
                {
                    _logger.LogWarning(
                        "Failed to remove TCP DNAT rule for port {PublicPort}: {Error}",
                        publicPort, result.StandardError);
                    success = false;
                }

                // Remove FORWARD rule
                result = await _executor.ExecuteAsync(
                    "iptables",
                    $"-D FORWARD -p tcp -d {vmIp} --dport {vmPort} -j ACCEPT",
                    ct);

                if (!result.Success)
                {
                    _logger.LogWarning(
                        "Failed to remove TCP FORWARD rule: {Error}",
                        result.StandardError);
                    // Don't mark as failure - maybe it was already removed
                }
            }

            // Remove UDP rules
            if (protocol == PortProtocol.UDP || protocol == PortProtocol.Both)
            {
                // Remove DNAT rule
                var result = await _executor.ExecuteAsync(
                    "iptables",
                    $"-t nat -D {CHAIN_NAME} -p udp --dport {publicPort} -j DNAT",
                    ct);

                if (!result.Success)
                {
                    _logger.LogWarning(
                        "Failed to remove UDP DNAT rule for port {PublicPort}: {Error}",
                        publicPort, result.StandardError);
                    success = false;
                }

                // Remove FORWARD rule
                result = await _executor.ExecuteAsync(
                    "iptables",
                    $"-D FORWARD -p udp -d {vmIp} --dport {vmPort} -j ACCEPT",
                    ct);

                if (!result.Success)
                {
                    _logger.LogWarning(
                        "Failed to remove UDP FORWARD rule: {Error}",
                        result.StandardError);
                    // Don't mark as failure - maybe it was already removed
                }
            }

            if (success)
            {
                await SaveRulesAsync(ct);
                _logger.LogInformation("✓ Port forwarding removed for port {PublicPort}", publicPort);
            }

            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing port forwarding for port {PublicPort}", publicPort);
            return false;
        }
        finally
        {
            if (lockAcquired)
            {
                _lock.Release();
            }
        }
    }

    public async Task<bool> RemoveAllForVmAsync(string vmIp, CancellationToken ct = default)
    {
        if (!_isLinux)
        {
            return false;
        }

        await _lock.WaitAsync(ct);
        try
        {
            _logger.LogInformation("Removing all port forwarding rules for VM {VmIp}", vmIp);

            // This is a bit brute-force, but effective:
            // Flush all rules in our chain, then recreate rules for other VMs

            // Get all mappings except for this VM
            var allMappings = await _repository.GetAllActiveAsync();
            var otherVmMappings = allMappings.Where(m => m.VmPrivateIp != vmIp).ToList();

            // Flush our chain
            await _executor.ExecuteAsync("iptables", $"-t nat -F {CHAIN_NAME}", ct);

            // Recreate rules for other VMs
            foreach (var mapping in otherVmMappings)
            {
                await CreateForwardingAsync(
                    mapping.VmPrivateIp,
                    mapping.VmPort,
                    mapping.PublicPort,
                    mapping.Protocol,
                    ct);
            }

            await SaveRulesAsync(ct);

            _logger.LogInformation("✓ Removed all forwarding rules for VM {VmIp}", vmIp);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing forwarding rules for VM {VmIp}", vmIp);
            return false;
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task ReconcileRulesAsync(CancellationToken ct = default)
    {
        if (!_isLinux)
        {
            _logger.LogDebug("Skipping reconciliation on non-Linux platform");
            return;
        }

        await _lock.WaitAsync(ct);
        try
        {
            _logger.LogInformation("Reconciling port forwarding rules from database...");

            // Ensure chain exists
            await EnsureChainExistsAsync(ct);

            // Flush existing rules in our chain
            await _executor.ExecuteAsync("iptables", $"-t nat -F {CHAIN_NAME}", ct);

            // Get all active mappings from database
            var mappings = await _repository.GetAllActiveAsync();

            _logger.LogInformation("Recreating {Count} port forwarding rules", mappings.Count);

            // Recreate all rules
            foreach (var mapping in mappings)
            {
                await CreateForwardingAsync(
                    mapping.VmPrivateIp,
                    mapping.VmPort,
                    mapping.PublicPort,
                    mapping.Protocol,
                    ct);
            }

            await SaveRulesAsync(ct);

            _logger.LogInformation("✓ Reconciliation complete: {Count} rules restored", mappings.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to reconcile port forwarding rules");
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task<bool> RuleExistsAsync(int publicPort, CancellationToken ct = default)
    {
        if (!_isLinux)
        {
            return false;
        }

        try
        {
            var result = await _executor.ExecuteAsync(
                "iptables",
                $"-t nat -L {CHAIN_NAME} -n --line-numbers",
                ct);

            if (result.Success)
            {
                return result.StandardOutput.Contains($"dpt:{publicPort}");
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Create iptables DNAT rule for specific protocol
    /// </summary>
    private async Task CreateIptablesRuleAsync(
        string protocol,
        string vmIp,
        int vmPort,
        int publicPort,
        CancellationToken ct)
    {
        // FORWARD: Allow traffic to reach the VM (must be BEFORE DNAT to bypass Libvirt's REJECT rules)
        // Insert at position 1 to ensure it's processed before Docker/Libvirt chains
        var forwardRule = $"-I FORWARD 1 -p {protocol} -d {vmIp} --dport {vmPort} -j ACCEPT";

        var result = await _executor.ExecuteAsync("iptables", forwardRule, ct);
        if (!result.Success)
        {
            _logger.LogWarning(
                "Failed to create FORWARD rule (may already exist): {Error}",
                result.StandardError);
            // Don't fail - FORWARD rule might already exist
        }

        // PREROUTING DNAT: Rewrite destination
        var preRoutingRule = $"-t nat -A {CHAIN_NAME} -p {protocol} --dport {publicPort} -j DNAT --to-destination {vmIp}:{vmPort}";

        result = await _executor.ExecuteAsync("iptables", preRoutingRule, ct);
        if (!result.Success)
        {
            throw new Exception($"Failed to create DNAT rule: {result.StandardError}");
        }

        _logger.LogDebug(
            "Created {Protocol} iptables rules (FORWARD + DNAT): :{PublicPort} → {VmIp}:{VmPort}",
            protocol.ToUpper(), publicPort, vmIp, vmPort);
    }

    /// <summary>
    /// Ensure our custom chain exists
    /// </summary>
    private async Task EnsureChainExistsAsync(CancellationToken ct)
    {
        // Create chain if it doesn't exist
        var result = await _executor.ExecuteAsync(
            "iptables",
            $"-t nat -N {CHAIN_NAME}",
            ct);

        // If chain already exists, that's fine (exit code 1)
        if (!result.Success && !result.StandardError.Contains("Chain already exists"))
        {
            _logger.LogWarning("Failed to create chain {Chain}: {Error}",
                CHAIN_NAME, result.StandardError);
        }

        // Ensure our chain is called from PREROUTING
        result = await _executor.ExecuteAsync(
            "iptables",
            $"-t nat -C PREROUTING -j {CHAIN_NAME}",
            ct);

        if (!result.Success)
        {
            // Chain not in PREROUTING, add it
            result = await _executor.ExecuteAsync(
                "iptables",
                $"-t nat -A PREROUTING -j {CHAIN_NAME}",
                ct);

            if (result.Success)
            {
                _logger.LogInformation("✓ Created iptables chain {Chain}", CHAIN_NAME);
            }
        }
    }

    /// <summary>
    /// Save iptables rules persistently
    /// </summary>
    private async Task SaveRulesAsync(CancellationToken ct)
    {
        try
        {
            // Try iptables-save with netfilter-persistent (Debian/Ubuntu)
            var result = await _executor.ExecuteAsync(
                "netfilter-persistent",
                "save",
                ct);

            if (result.Success)
            {
                _logger.LogDebug("Saved iptables rules via netfilter-persistent");
                return;
            }

            // Fallback: iptables-save to file (RHEL/CentOS)
            result = await _executor.ExecuteAsync(
                "sh",
                "-c \"iptables-save > /etc/sysconfig/iptables\"",
                ct);

            if (result.Success)
            {
                _logger.LogDebug("Saved iptables rules to /etc/sysconfig/iptables");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to save iptables rules persistently");
        }
    }

    /// <summary>
    /// Check if IP is in the DeCloud tunnel network (10.20.0.0/16)
    /// </summary>
    private bool IsTunnelIp(string ip)
    {
        return ip.StartsWith(TUNNEL_IP_PREFIX, StringComparison.Ordinal);
    }

    /// <summary>
    /// Find the local relay VM that manages WireGuard tunnels.
    /// Returns the relay VM's IP address if found, null otherwise.
    /// </summary>
    private async Task<string?> GetRelayVmIpAsync(CancellationToken ct)
    {
        try
        {
            var vms = await _vmManager.GetAllVmsAsync(ct);
            var relayVm = vms.FirstOrDefault(vm => vm.Spec.VmType == VmType.Relay);
            
            if (relayVm != null && !string.IsNullOrEmpty(relayVm.Spec.IpAddress))
            {
                _logger.LogDebug(
                    "Found relay VM {VmId} at {IpAddress}",
                    relayVm.Spec.Id, relayVm.Spec.IpAddress);
                return relayVm.Spec.IpAddress;
            }
            
            _logger.LogDebug("No relay VM found on this node");
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to query for relay VM");
            return null;
        }
    }
}
