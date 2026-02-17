using DeCloud.NodeAgent.Infrastructure.Services;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Background service that reconciles port forwarding rules on startup.
/// Ensures iptables rules match the database after node restart.
/// Uses BackgroundService (not IHostedService) so startup is non-blocking —
/// Kestrel binds the port immediately while reconciliation runs in the background.
/// </summary>
public class PortForwardingReconciliationService : BackgroundService
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

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Starting port forwarding reconciliation...");

        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken);
            cts.CancelAfter(TimeSpan.FromSeconds(30));

            await _portForwardingManager.ReconcileRulesAsync(cts.Token);

            _logger.LogInformation("Port forwarding reconciliation complete");
        }
        catch (OperationCanceledException) when (!stoppingToken.IsCancellationRequested)
        {
            _logger.LogWarning(
                "Port forwarding reconciliation timed out after 30s — " +
                "rules will be reconciled on next port operation.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to reconcile port forwarding rules");
        }
    }
}
