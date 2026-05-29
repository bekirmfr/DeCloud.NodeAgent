using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.Shared.Enums;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services
{
    public class VmHealthService : BackgroundService
    {
        private readonly IVmManager _vmManager;
        private readonly VmRepository _vmRepository;
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
            VmRepository vmRepository,
            INatRuleManager natRuleManager,
            IOrchestratorClient orchestratorClient,
            ILogger<VmHealthService> logger
            )
        {
            _vmManager = vmManager;
            _vmRepository = vmRepository;
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
                        if (vm.Spec.Role == VmRole.Relay)
                            await CheckRelayVmNatRulesAsync();
                    }

                    // If this node has no relay VM, ensure no stale relay
                    // NAT rules exist (e.g. from a previous deployment or
                    // a saved /etc/iptables/rules.v4 from another node role).
                    await CleanStaleRelayNatIfNotRelayNodeAsync(ct);

                    foreach (var vm in vms)  // continue existing loop
                    {

                        if (vm.Status != Shared.Enums.VmStatus.Error)
                        {
                            // System VMs are managed exclusively by orchestrator reconciliation.
                            // Their health authority is the orchestrator, which has its own
                            // heartbeat freshness check and grace period. Local heartbeat-based
                            // marking is unreliable for system VMs: LastHeartbeat depends on
                            // qemu-guest-agent, which may not respond after network isolation
                            // or guest OS issues, producing permanent false positives.
                            // The watchdog's zombie check handles local recovery instead.
                            if (vm.Spec.Role is VmRole.Relay or VmRole.Dht or VmRole.BlockStore)
                                continue;

                            // Guard: skip VMs with no established heartbeat baseline.
                            // After agent restart, LastHeartbeat may still be DateTime.MinValue
                            // if reconciliation didn't stamp it (e.g., race with initialization).
                            // The readiness monitor will establish a real baseline within its
                            // first cycle — don't force-restart before that happens.
                            if (vm.LastHeartbeat <= DateTime.MinValue.AddYears(1))
                            {
                                _logger.LogDebug(
                                    "VM {VmId} has no heartbeat baseline yet — skipping health check " +
                                    "until VmReadinessMonitor establishes one",
                                    vm.VmId);
                                continue;
                            }

                            var timeSinceLastHeartbeat = DateTime.UtcNow - vm.LastHeartbeat;
                            if (vm.IsFullyReady && timeSinceLastHeartbeat > TimeSpan.FromMinutes(5))
                            {

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
                                    vm.Status = Shared.Enums.VmStatus.Error;
                                }
                                else
                                {
                                    // Reset the heartbeat baseline so the health check doesn't immediately
                                    // re-fire on the next cycle. VmReadinessMonitor will overwrite this with
                                    // a real guest-agent ping once the VM boots and the agent responds.
                                    vm.LastHeartbeat = DateTime.UtcNow;
                                    await _vmRepository.UpdateLastHeartbeatAsync(vm.VmId, vm.LastHeartbeat);
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
                v.Spec.Role == VmRole.Relay &&
                v.Status == Shared.Enums.VmStatus.Running))
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
                .Any(v => v.Spec.Role == VmRole.Relay &&
                          v.Status is VmStatus.Running or VmStatus.Provisioning or VmStatus.Provisioning);

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
