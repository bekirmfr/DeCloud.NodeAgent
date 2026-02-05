using DeCloud.NodeAgent.Infrastructure.Services;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Background service that reconciles port forwarding rules on startup.
/// Ensures iptables rules match the database after node restart.
/// </summary>
public class PortForwardingReconciliationService : IHostedService
{
    private readonly IPortForwardingManager _portForwardingManager;
    private readonly ILogger<PortForwardingReconciliationService> _logger;

    public PortForwardingReconciliationService(
        IPortForwardingManager portForwardingManager,
        ILogger<PortForwardingReconciliationService> logger)
    {
        _portForwardingManager = portForwardingManager;
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting port forwarding reconciliation...");

        try
        {
            // Reconcile iptables rules with database
            await _portForwardingManager.ReconcileRulesAsync(cancellationToken);

            _logger.LogInformation("✓ Port forwarding reconciliation complete");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to reconcile port forwarding rules");
            // Don't throw - allow service to start even if reconciliation fails
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Port forwarding reconciliation service shutdown");
        return Task.CompletedTask;
    }
}
