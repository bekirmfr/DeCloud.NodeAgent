using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Org.BouncyCastle.Pqc.Crypto.Lms;

namespace DeCloud.NodeAgent.Infrastructure.Services
{
    public class VmHealthService : BackgroundService
    {
        private readonly IVmManager _vmManager;
        private readonly INatRuleManager _natRuleManager;
        private readonly IOrchestratorClient _orchestratorClient;
        private readonly ILogger<VmHealthService> _logger;

        private static readonly TimeSpan HealthCheckInterval = TimeSpan.FromMinutes(1);
        private static readonly TimeSpan NatCheckInterval = TimeSpan.FromMinutes(10); // Only check NAT every 10 minutes
        
        // Track last NAT check time per VM to avoid excessive checking
        private readonly Dictionary<string, DateTime> _lastNatCheckByVm = new();
        private readonly SemaphoreSlim _natCheckLock = new(1, 1);

        public VmHealthService(
            IVmManager vmManager,
            INatRuleManager natRuleManager,
            IOrchestratorClient orchestratorClient,
            ILogger<VmHealthService> logger
            )
        {
            _vmManager = vmManager;
            _natRuleManager = natRuleManager;
            _orchestratorClient = orchestratorClient;
            _logger = logger;
        }
        protected override async Task ExecuteAsync(CancellationToken ct)
        {
            _logger.LogInformation("Vm health service starting...");
            var vms = new List<VmInstance>();

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    await _vmManager.ReconcileAllWithLibvirtAsync(ct);

                    vms = _vmManager.GetAllVms().ToList();

                    foreach (var vm in vms)
                    {
                        if (vm.Spec.VmType == VmType.Relay)
                            await CheckRelayVmNatRulesAsync();
                    }

                    // If this node has no relay VM, ensure no stale relay
                    // NAT rules exist (e.g. from a previous deployment or
                    // a saved /etc/iptables/rules.v4 from another node role).
                    await CleanStaleRelayNatIfNotRelayNodeAsync(ct);

                    foreach (var vm in vms)  // continue existing loop
                    {

                        if (vm.State != VmState.Failed)
                        {
                            var timeSinceLastHeartbeat = DateTime.UtcNow - vm.LastHeartbeat;
                            if (timeSinceLastHeartbeat > TimeSpan.FromMinutes(5))
                            {
                                // System VMs (Relay, Dht, BlockStore) are managed exclusively
                                // by orchestrator reconciliation. Attempting local restart
                                // conflicts with the orchestrator's cleanup state machine and
                                // causes virsh reset failures on stopped domains.
                                // Mark Failed so the orchestrator detects and reacts via heartbeat.
                                if (vm.Spec.VmType is VmType.Relay or VmType.Dht or VmType.BlockStore)
                                {
                                    _logger.LogInformation(
                                        "System VM {VmId} ({VmType}) missed heartbeat for {ElapsedMinutes:F1}m " +
                                        "— skipping local restart, orchestrator reconciliation will redeploy",
                                        vm.VmId, vm.Spec.VmType, timeSinceLastHeartbeat.TotalMinutes);
                                    vm.State = VmState.Failed;
                                    continue;
                                }

                                _logger.LogWarning(
                                    "VM {VmId} has not sent heartbeat for {ElapsedMinutes} minutes. Restarting VM.",
                                    vm.VmId, timeSinceLastHeartbeat.TotalMinutes);
                                var restarted = await _vmManager.RestartVmAsync(vm.VmId, force: true, ct);
                                if (!restarted.Success)
                                {
                                    _logger.LogError(
                                        "VM {VmId} failed to restart — marking as Failed to stop retry loop. " +
                                        "Orchestrator reconciliation will redeploy.",
                                        vm.VmId);
                                    // Mark Failed in-memory so the loop skips it next cycle
                                    vm.State = VmState.Failed;
                                }
                            }
                        }
                    }

                    // Prevent tight loop
                    await Task.Delay(HealthCheckInterval, ct);
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("AuthenticationManager stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "AuthenticationManager error");
                throw;
            }
        }

        public async Task CheckRelayVmNatRulesAsync(CancellationToken ct = default)
        {
            foreach (var vm in _vmManager.GetAllVms().Where(v =>
                v.Spec.VmType == VmType.Relay &&
                v.State == VmState.Running))
            {
                // Check if we recently verified NAT rules for this VM
                await _natCheckLock.WaitAsync(ct);
                try
                {
                    if (_lastNatCheckByVm.TryGetValue(vm.VmId, out var lastCheck))
                    {
                        var timeSinceLastCheck = DateTime.UtcNow - lastCheck;
                        if (timeSinceLastCheck < NatCheckInterval)
                        {
                            // Skip check - we verified recently
                            continue;
                        }
                    }
                }
                finally
                {
                    _natCheckLock.Release();
                }

                var vmIp = await _vmManager.GetVmIpAddressAsync(vm.VmId);

                if (vmIp == null)
                {
                    _logger.LogWarning(
                        "Relay VM {VmId} has no IP address assigned yet - skipping NAT check",
                        vm.VmId);
                    continue;
                }

                // Use the proper NAT script check method instead of raw iptables
                // This checks all three required rules: PREROUTING, POSTROUTING, FORWARD
                var hasNatRules = await _natRuleManager.HasRulesForVmAsync(vmIp, ct);

                if (!hasNatRules)
                {
                    _logger.LogWarning(
                        "Relay VM {VmId} at {Ip} missing NAT rules - reconfiguring",
                        vm.VmId, vmIp);

                    var success = await _natRuleManager.AddPortForwardingAsync(vmIp, 51820, "udp", ct);
                    
                    if (success)
                    {
                        // Update last check time after successful configuration
                        await _natCheckLock.WaitAsync(ct);
                        try
                        {
                            _lastNatCheckByVm[vm.VmId] = DateTime.UtcNow;
                        }
                        finally
                        {
                            _natCheckLock.Release();
                        }
                    }
                }
                else
                {
                    _logger.LogDebug(
                        "✓ Relay VM {VmId} at {Ip} has complete NAT rules",
                        vm.VmId, vmIp);

                    // Update last check time - rules verified OK
                    await _natCheckLock.WaitAsync(ct);
                    try
                    {
                        _lastNatCheckByVm[vm.VmId] = DateTime.UtcNow;
                    }
                    finally
                    {
                    _natCheckLock.Release();
                    }
                }
            }
        }

        private async Task CleanStaleRelayNatIfNotRelayNodeAsync(CancellationToken ct)
        {
            var hasRelayVm = _vmManager.GetAllVms()
                .Any(v => v.Spec.VmType == VmType.Relay &&
                          v.State is VmState.Running or VmState.Starting or VmState.Creating);

            if (hasRelayVm) return; // Relay node — rules are managed by CheckRelayVmNatRulesAsync

            // CGNAT nodes use wg-relay for VM routing.
            // Their iptables contain wg-relay↔virbr0 FORWARD rules whose
            // raw output includes "51820". Calling `decloud-relay-nat clean`
            // here would flush those FORWARD rules, severing all subdomain
            // traffic until WireGuardConfigManager reconciles (~60 s later).
            var isCgnatNode = _orchestratorClient.GetLastHeartbeat()?.Heartbeat?.CgnatInfo != null;
            if (isCgnatNode)
            {
                _logger.LogDebug(
                    "Skipping relay NAT cleanup — node is a CGNAT client (wg-relay active)");
                return;
            }

            // No relay VM — check if any relay NAT rules exist and wipe them
            var existingRules = await _natRuleManager.GetExistingRulesAsync(ct);

            // Only match PREROUTING DNAT rules, not arbitrary FORWARD
            // or conntrack entries that happen to mention port 51820 / 8080.
            var hasStaleRelayRules = existingRules.Any(r =>
                r.Contains("DNAT") &&
                (r.Contains("51820") || r.Contains("8080")));

            if (hasStaleRelayRules)
            {
                _logger.LogWarning(
                    "Node has stale relay PREROUTING DNAT rules but no relay VM — cleaning");
                await _natRuleManager.RemoveAllRelayNatRulesAsync(ct);
            }
        }
    }
}
