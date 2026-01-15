// =====================================================================
// NodeStateSyncService - Updated with Async Waiter
// =====================================================================
// File: src/DeCloud.NodeAgent/Services/State/NodeStateSyncService.cs
// 
// Changes from previous version:
// - Uses WaitForAuthenticationAsync instead of polling loop
// =====================================================================

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Background service for synchronizing node state with orchestrator.
/// Handles startup sync, periodic refresh, and error recovery.
/// Delegates state tracking to INodeStateService.
/// </summary>
public class NodeStateSyncService : BackgroundService
{
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly INodeStateService _nodeState;
    private readonly INodeMetadataService _nodeMetadata;
    private readonly NodeStateSyncOptions _options;
    private readonly ILogger<NodeStateSyncService> _logger;

    private bool _isInitialSyncComplete;

    public NodeStateSyncService(
        IOrchestratorClient orchestratorClient,
        INodeStateService nodeState,
        INodeMetadataService nodeMetadata,
        IOptions<NodeStateSyncOptions> options,
        ILogger<NodeStateSyncService> logger)
    {
        _orchestratorClient = orchestratorClient;
        _nodeState = nodeState;
        _nodeMetadata = nodeMetadata;
        _options = options.Value;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        _logger.LogInformation(
            "NodeStateSyncService starting (SyncInterval={Interval}s, MaxFailures={MaxFailures})",
            _options.SyncInterval.TotalSeconds,
            _options.MaxConsecutiveFailuresBeforeFullSync);

        // Wait for authentication using async waiter (no polling!)
        await WaitForAuthenticationAsync(ct);

        // Perform initial sync
        if (_options.EnableStartupSync)
        {
            await PerformInitialSyncAsync(ct);
        }
        else
        {
            _isInitialSyncComplete = true;
            _logger.LogInformation("Startup sync disabled, skipping initial sync");
        }

        // Start periodic sync loop
        await PeriodicSyncLoopAsync(ct);
    }

    // =====================================================================
    // Initialization
    // =====================================================================

    private async Task WaitForAuthenticationAsync(CancellationToken ct)
    {
        _logger.LogInformation("Waiting for node authentication...");

        try
        {
            // Use timeout to avoid hanging forever
            using var timeoutCts = new CancellationTokenSource(TimeSpan.FromMinutes(5));
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);

            await _nodeState.WaitForAuthenticationAsync(linkedCts.Token);

            _logger.LogInformation("✓ Node authenticated, proceeding with state sync");
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            _logger.LogWarning(
                "Authentication timeout after 5 minutes, will retry sync when authenticated");
        }
    }

    private async Task PerformInitialSyncAsync(CancellationToken ct)
    {
        _logger.LogInformation("═══════════════════════════════════════════════════════════");
        _logger.LogInformation("INITIAL STATE SYNCHRONIZATION");
        _logger.LogInformation("═══════════════════════════════════════════════════════════");

        var maxRetries = 5;
        var retryDelay = TimeSpan.FromSeconds(5);

        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            if (ct.IsCancellationRequested) return;

            _logger.LogInformation("Initial sync attempt {Attempt}/{Max}...", attempt, maxRetries);

            var result = await _orchestratorClient.SyncWithOrchestratorAsync(ct);

            if (result.Success)
            {
                _nodeState.RecordSync(true);
                _isInitialSyncComplete = true;

                _logger.LogInformation(
                    "✓ Initial sync successful: ConfigV{ConfigVersion}, Tier={Tier}, Points={Points}",
                    result.ConfigVersion,
                    result.HighestTier,
                    result.TotalComputePoints);
                return;
            }

            _nodeState.RecordSync(false);

            _logger.LogWarning(
                "Initial sync failed (attempt {Attempt}/{Max}): {Error}",
                attempt, maxRetries, result.Error);

            if (attempt < maxRetries)
            {
                _logger.LogInformation("Retrying in {Delay}s...", retryDelay.TotalSeconds);
                await Task.Delay(retryDelay, ct);
                retryDelay *= 2;
            }
        }

        _logger.LogError(
            "⚠ Initial sync failed after {MaxRetries} attempts. " +
            "Node will continue with config received during registration.",
            maxRetries);

        _isInitialSyncComplete = true;
    }

    // =====================================================================
    // Periodic Sync Loop
    // =====================================================================

    private async Task PeriodicSyncLoopAsync(CancellationToken ct)
    {
        _logger.LogInformation(
            "Starting periodic sync loop (interval: {Interval}s)",
            _options.SyncInterval.TotalSeconds);

        while (!ct.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(_options.SyncInterval, ct);

                if (!_nodeState.IsAuthenticated)
                {
                    _logger.LogDebug("Not authenticated, skipping sync");
                    continue;
                }

                if (_nodeState.ConsecutiveFailures >= _options.MaxConsecutiveFailuresBeforeFullSync)
                {
                    _logger.LogWarning(
                        "Too many consecutive failures ({Failures}), performing full sync",
                        _nodeState.ConsecutiveFailures);
                    await PerformFullSyncAsync(ct);
                }
                else
                {
                    await PerformConfigSyncAsync(ct);
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in periodic sync loop");
                _nodeState.RecordSync(false);

                var backoffDelay = TimeSpan.FromSeconds(
                    Math.Min(60, Math.Pow(2, _nodeState.ConsecutiveFailures)));
                _logger.LogInformation("Backing off for {Delay}s after failure", backoffDelay.TotalSeconds);
                await Task.Delay(backoffDelay, ct);
            }
        }

        _logger.LogInformation("Periodic sync loop stopped");
    }

    // =====================================================================
    // Sync Operations
    // =====================================================================

    private async Task PerformConfigSyncAsync(CancellationToken ct)
    {
        var currentVersion = _nodeMetadata.GetSchedulingConfigVersion();

        _logger.LogDebug("Checking config sync (current version: {Version})", currentVersion);

        var config = await _orchestratorClient.GetSchedulingConfigAsync(ct);

        if (config == null)
        {
            _nodeState.RecordSync(false);
            _logger.LogWarning(
                "Config sync failed (consecutive failures: {Failures})",
                _nodeState.ConsecutiveFailures);
            return;
        }

        _nodeState.RecordSync(true);

        if (config.Version > currentVersion)
        {
            _logger.LogInformation(
                "✓ Config updated: v{OldVersion} → v{NewVersion}",
                currentVersion, config.Version);
        }
        else
        {
            _logger.LogDebug("Config up to date (v{Version})", config.Version);
        }
    }

    private async Task PerformFullSyncAsync(CancellationToken ct)
    {
        _logger.LogInformation("Performing full state synchronization...");

        var result = await _orchestratorClient.SyncWithOrchestratorAsync(ct);

        if (result.Success)
        {
            _nodeState.RecordSync(true);

            _logger.LogInformation(
                "✓ Full sync successful: ConfigV{ConfigVersion}, Tier={Tier}",
                result.ConfigVersion, result.HighestTier);
        }
        else
        {
            _nodeState.RecordSync(false);
            _logger.LogWarning("Full sync failed: {Error}", result.Error);
        }
    }

    // =====================================================================
    // Public Methods
    // =====================================================================

    public async Task<bool> RequestImmediateSyncAsync(CancellationToken ct = default)
    {
        _logger.LogInformation("Immediate sync requested");

        var result = await _orchestratorClient.SyncWithOrchestratorAsync(ct);
        _nodeState.RecordSync(result.Success);

        return result.Success;
    }

    public NodeSyncStatus GetSyncStatus() => new()
    {
        IsInitialSyncComplete = _isInitialSyncComplete,
        LastSuccessfulSync = _nodeState.LastSync ?? DateTime.MinValue,
        ConsecutiveFailures = _nodeState.ConsecutiveFailures,
        CurrentConfigVersion = _nodeMetadata.GetSchedulingConfigVersion(),
        SyncInterval = _options.SyncInterval
    };
}

// =====================================================================
// DTOs (unchanged)
// =====================================================================

public class NodeStateSyncOptions
{
    public TimeSpan SyncInterval { get; set; } = TimeSpan.FromMinutes(5);
    public int MaxConsecutiveFailuresBeforeFullSync { get; set; } = 3;
    public bool EnableStartupSync { get; set; } = true;
}

public class NodeSyncStatus
{
    public bool IsInitialSyncComplete { get; init; }
    public DateTime LastSuccessfulSync { get; init; }
    public int ConsecutiveFailures { get; init; }
    public int CurrentConfigVersion { get; init; }
    public TimeSpan SyncInterval { get; init; }

    public TimeSpan TimeSinceLastSync => LastSuccessfulSync == DateTime.MinValue
        ? TimeSpan.MaxValue
        : DateTime.UtcNow - LastSuccessfulSync;

    public bool IsHealthy => ConsecutiveFailures == 0 &&
                             TimeSinceLastSync < SyncInterval * 3;
}