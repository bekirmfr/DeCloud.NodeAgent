using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services
{
    public class VmHealthService : BackgroundService
    {
        private readonly IVmManager _vmManager;
        private readonly INatRuleManager _natRuleManager;
        private readonly ILogger<VmHealthService> _logger;

        private static readonly TimeSpan HealthCheckInterval = TimeSpan.FromMinutes(1);
        private static readonly TimeSpan NatCheckInterval = TimeSpan.FromMinutes(10); // Only check NAT every 10 minutes
        
        // Track last NAT check time per VM to avoid excessive checking
        private readonly Dictionary<string, DateTime> _lastNatCheckByVm = new();
        private readonly SemaphoreSlim _natCheckLock = new(1, 1);

        public VmHealthService(
            IVmManager vmManager,
            INatRuleManager natRuleManager,
            ILogger<VmHealthService> logger
            )
        {
            _vmManager = vmManager;
            _natRuleManager = natRuleManager;  
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

                        if (vm.State != VmState.Running && vm.State != VmState.Failed)
                        {
                            var timeSinceLastHeartbeat = DateTime.UtcNow - vm.LastHeartbeat;
                            if (timeSinceLastHeartbeat > TimeSpan.FromMinutes(5))
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
            var relayVms = _vmManager.GetAllVms();

            foreach (var vm in relayVms.Where(v => v.Spec.VmType == VmType.Relay))
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
    }
}
