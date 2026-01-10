using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using static AuthenticationManager;

namespace DeCloud.NodeAgent.Infrastructure.Services
{
    public class VmHealthService : BackgroundService
    {
        private readonly VmRepository _repository;
        private readonly IVmManager _vmManager;
        private readonly INatRuleManager _natRuleManager;
        private readonly ILogger<VmHealthService> _logger;

        private static readonly TimeSpan HealthCheckInterval = TimeSpan.FromMinutes(1);
        public VmHealthService(
            VmRepository repository,
            IVmManager vmManager,
            INatRuleManager natRuleManager,
            ILogger<VmHealthService> logger
            )
        {
            _repository = repository;
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

                    vms = await _vmManager.GetAllVmsAsync(ct);

                    foreach (var vm in vms)
                    {
                        if (vm.Spec.VmType == VmType.Relay)
                            await CheckRelayVmNatRulesAsync();

                        if (vm.State != VmState.Running)
                        {
                            var timeSinceLastHeartbeat = DateTime.UtcNow - vm.LastHeartbeat;
                            if (timeSinceLastHeartbeat > TimeSpan.FromMinutes(5))
                            {
                                _logger.LogWarning(
                                    "VM {VmId} has not sent heartbeat for {ElapsedMinutes} minutes. Restarting VM.",
                                    vm.VmId, timeSinceLastHeartbeat.TotalMinutes);
                                await _vmManager.RestartVmAsync(vm.VmId, true, ct);
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

        public async Task CheckRelayVmNatRulesAsync()
        {
            var relayVms = await _vmManager.GetAllVmsAsync();

            foreach (var vm in relayVms)
            {
                var natRules = await _natRuleManager.GetExistingRulesAsync();

                if (natRules.Count > 0)
                {
                    _logger.LogWarning(
                        "Relay VM {VmId} at {Ip} missing NAT rules - reconfiguring",
                        vm.VmId, vm.Spec.IpAddress);

                    await _natRuleManager.AddPortForwardingAsync(vm.Spec.IpAddress, 51820, "udp");
                }
            }
        }
    }
}
