// =====================================================================
// OrchestratorClient Extension - /api/nodes/me Endpoints Implementation
// =====================================================================
// File: src/DeCloud.NodeAgent/Services/OrchestratorClient.NodeMe.cs
// 
// IMPORTANT: This is a partial class extension. Merge into your existing
// OrchestratorClient.cs or keep as a separate partial class file.
// =====================================================================

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Orchestrator.Models;
using System.Text.Json;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Partial class extending OrchestratorClient with /api/nodes/me endpoint support.
/// These endpoints provide self-service node information from the orchestrator.
/// </summary>
public partial class OrchestratorClient
{
    // =====================================================================
    // CONFIGURATION
    // =====================================================================

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    // Retry configuration for resilience
    private const int MaxRetries = 3;
    private static readonly TimeSpan[] RetryDelays =
    {
        TimeSpan.FromSeconds(1),
        TimeSpan.FromSeconds(2),
        TimeSpan.FromSeconds(5)
    };

    // =====================================================================
    // GET /api/nodes/me - Node Summary
    // =====================================================================

    /// <inheritdoc />
    public async Task<HttpResponse<NodeSummaryResponse>> GetNodeSummaryAsync(CancellationToken ct = default)
    {
        try
        {
            _logger.LogDebug("Fetching node summary from orchestrator...");

            var response = await _httpClient.GetAsync("/api/nodes/me", ct);

            return await HttpResponse<NodeSummaryResponse>.FromResponseAsync(response);
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error fetching node summary");
            return HttpResponse<NodeSummaryResponse>.FromException(ex);
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogDebug("Node summary request cancelled");
            return HttpResponse<NodeSummaryResponse>.FromException(ex);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching node summary");
            return HttpResponse<NodeSummaryResponse>.FromException(ex);
        }
    }

    // =====================================================================
    // GET /api/nodes/me/config - Scheduling Configuration
    // =====================================================================

    /// <inheritdoc />
    public async Task<SchedulingConfig?> GetSchedulingConfigAsync(CancellationToken ct = default)
    {
        try
        {
            _logger.LogDebug("Fetching scheduling configuration from orchestrator...");

            var response = await _httpClient.GetAsync("/api/nodes/me/config", ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning(
                    "Failed to get scheduling config: {StatusCode} - {Reason}",
                    (int)response.StatusCode, response.ReasonPhrase);
                return null;
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var config = JsonSerializer.Deserialize<SchedulingConfig>(content, _jsonOptions);

            if (config != null)
            {
                _logger.LogInformation(
                    "✓ Scheduling config received: v{Version}, Baseline={Baseline}, Overcommit={Overcommit:F1}",
                    config.Version, config.BaselineBenchmark, config.BaselineOvercommitRatio);

                // Update local metadata service
                _nodeMetadata.UpdateSchedulingConfig(new SchedulingConfig
                {
                    Version = config.Version,
                    BaselineBenchmark = config.BaselineBenchmark,
                    BaselineOvercommitRatio = config.BaselineOvercommitRatio,
                    MaxPerformanceMultiplier = config.MaxPerformanceMultiplier,
                    Tiers = config.Tiers,
                    UpdatedAt = config.UpdatedAt
                });
            }

            return config;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error fetching scheduling config");
            return null;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogDebug("Scheduling config request cancelled");
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching scheduling config");
            return null;
        }
    }

    // =====================================================================
    // GET /api/nodes/me/performance - Performance Evaluation
    // =====================================================================

    /// <inheritdoc />
    public async Task<NodePerformanceEvaluation?> GetPerformanceEvaluationAsync(CancellationToken ct = default)
    {
        try
        {
            _logger.LogDebug("Fetching performance evaluation from orchestrator...");

            var response = await _httpClient.GetAsync("/api/nodes/me/performance", ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning(
                    "Failed to get performance evaluation: {StatusCode} - {Reason}",
                    (int)response.StatusCode, response.ReasonPhrase);
                return null;
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var evaluation = JsonSerializer.Deserialize<NodePerformanceEvaluation>(content, _jsonOptions);

            if (evaluation != null)
            {
                _logger.LogInformation(
                    "✓ Performance evaluation received: " +
                    "Benchmark={Benchmark}, Points/Core={PointsPerCore:F2}, " +
                    "HighestTier={HighestTier}, TotalPoints={TotalPoints:F0}",
                    evaluation.BenchmarkScore,
                    evaluation.PointsPerCore,
                    evaluation.HighestTier,
                    evaluation.TotalComputePoints);

                // Update local metadata service
                _nodeMetadata.UpdatePerformanceEvaluation(evaluation);
            }

            return evaluation;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error fetching performance evaluation");
            return null;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogDebug("Performance evaluation request cancelled");
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching performance evaluation");
            return null;
        }
    }

    // =====================================================================
    // GET /api/nodes/me/capacity - Node Capacity
    // =====================================================================

    /// <inheritdoc />
    public async Task<NodeCapacityResponse?> GetCapacityAsync(CancellationToken ct = default)
    {
        try
        {
            _logger.LogDebug("Fetching node capacity from orchestrator...");

            var response = await _httpClient.GetAsync("/api/nodes/me/capacity", ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning(
                    "Failed to get node capacity: {StatusCode} - {Reason}",
                    (int)response.StatusCode, response.ReasonPhrase);
                return null;
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var capacity = JsonSerializer.Deserialize<NodeCapacityResponse>(content, _jsonOptions);

            if (capacity != null)
            {
                _logger.LogInformation(
                    "✓ Node capacity received: " +
                    "{AllocatedPoints}/{TotalPoints} points ({Utilization:F1}%), " +
                    "{VmCount} active VMs",
                    capacity.AllocatedComputePoints,
                    capacity.TotalComputePoints,
                    capacity.UtilizationPercent,
                    capacity.ActiveVmCount);
            }

            return capacity;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error fetching node capacity");
            return null;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogDebug("Node capacity request cancelled");
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching node capacity");
            return null;
        }
    }

    // =====================================================================
    // POST /api/nodes/me/evaluate - Request Performance Re-evaluation
    // =====================================================================

    /// <inheritdoc />
    public async Task<NodePerformanceEvaluation?> RequestPerformanceEvaluationAsync(CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Requesting performance re-evaluation from orchestrator...");

            var response = await _httpClient.PostAsync(
                "/api/nodes/me/evaluate",
                null,  // No body required
                ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning(
                    "Failed to request performance evaluation: {StatusCode} - {Reason}",
                    (int)response.StatusCode, response.ReasonPhrase);
                return null;
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var evaluation = JsonSerializer.Deserialize<NodePerformanceEvaluation>(content, _jsonOptions);

            if (evaluation != null)
            {
                _logger.LogInformation(
                    "✓ Performance re-evaluation complete: " +
                    "Benchmark={Benchmark}, Points/Core={PointsPerCore:F2}, " +
                    "HighestTier={HighestTier}, Acceptable={IsAcceptable}",
                    evaluation.BenchmarkScore,
                    evaluation.PointsPerCore,
                    evaluation.HighestTier,
                    evaluation.IsAcceptable);

                // Update local metadata service
                _nodeMetadata.UpdatePerformanceEvaluation(evaluation);
            }

            return evaluation;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error requesting performance evaluation");
            return null;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogDebug("Performance evaluation request cancelled");
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error requesting performance evaluation");
            return null;
        }
    }

    // =====================================================================
    // Full Node Synchronization
    // =====================================================================

    /// <inheritdoc />
    public async Task<NodeSyncResult> SyncWithOrchestratorAsync(CancellationToken ct = default)
    {
        _logger.LogInformation("═══════════════════════════════════════════════════════════");
        _logger.LogInformation("Starting full synchronization with orchestrator...");
        _logger.LogInformation("═══════════════════════════════════════════════════════════");

        var result = new NodeSyncResult { SyncedAt = DateTime.UtcNow };
        var errors = new List<string>();

        try
        {
            // 1. Fetch scheduling configuration
            var config = await GetSchedulingConfigWithRetryAsync(ct);
            if (config != null)
            {
                result.ConfigSynced = true;
                result.ConfigVersion = config.Version;
                _logger.LogInformation("✓ Scheduling config synced: v{Version}", config.Version);
            }
            else
            {
                errors.Add("Failed to sync scheduling config");
                _logger.LogWarning("✗ Failed to sync scheduling config");
            }

            // 2. Fetch performance evaluation
            var evaluation = await GetPerformanceEvaluationWithRetryAsync(ct);
            if (evaluation != null)
            {
                result.PerformanceSynced = true;
                result.HighestTier = evaluation.HighestTier;
                result.TotalComputePoints = (int)evaluation.TotalComputePoints;
                _logger.LogInformation(
                    "✓ Performance evaluation synced: HighestTier={Tier}, Points={Points:F0}",
                    evaluation.HighestTier, evaluation.TotalComputePoints);
            }
            else
            {
                errors.Add("Failed to sync performance evaluation");
                _logger.LogWarning("✗ Failed to sync performance evaluation");
            }

            // 3. Fetch node summary (for verification)
            var summary = await GetNodeSummaryAsync(ct);
            if (summary != null)
            {
                result.SummarySynced = true;
                _logger.LogInformation(
                    "✓ Node summary synced: Status={Status}, LastHeartbeat={Heartbeat}",
                    summary.Status, summary.LastHeartbeat);

                // Verify node ID matches
                if (summary.NodeId != _nodeMetadata.NodeId)
                {
                    _logger.LogError(
                        "⚠ Node ID mismatch! Local={LocalId}, Orchestrator={RemoteId}",
                        _nodeMetadata.NodeId, summary.NodeId);
                    errors.Add($"Node ID mismatch: {_nodeMetadata.NodeId} vs {summary.NodeId}");
                }
            }
            else
            {
                errors.Add("Failed to sync node summary");
                _logger.LogWarning("✗ Failed to sync node summary");
            }

            // Determine overall success
            result.Success = result.ConfigSynced && result.PerformanceSynced;

            if (errors.Any())
            {
                result.Error = string.Join("; ", errors);
            }

            _logger.LogInformation("═══════════════════════════════════════════════════════════");
            _logger.LogInformation(
                "Synchronization {Status}: Config={Config}, Performance={Perf}, Summary={Summary}",
                result.Success ? "COMPLETE" : "PARTIAL",
                result.ConfigSynced ? "✓" : "✗",
                result.PerformanceSynced ? "✓" : "✗",
                result.SummarySynced ? "✓" : "✗");
            _logger.LogInformation("═══════════════════════════════════════════════════════════");

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during node synchronization");
            return NodeSyncResult.Failed($"Synchronization error: {ex.Message}");
        }
    }

    // =====================================================================
    // Retry Helpers
    // =====================================================================

    private async Task<SchedulingConfig?> GetSchedulingConfigWithRetryAsync(CancellationToken ct)
    {
        for (int attempt = 0; attempt < MaxRetries; attempt++)
        {
            var result = await GetSchedulingConfigAsync(ct);
            if (result != null) return result;

            if (attempt < MaxRetries - 1)
            {
                _logger.LogDebug(
                    "Retrying scheduling config fetch in {Delay}s (attempt {Attempt}/{Max})",
                    RetryDelays[attempt].TotalSeconds, attempt + 2, MaxRetries);
                await Task.Delay(RetryDelays[attempt], ct);
            }
        }
        return null;
    }

    private async Task<NodePerformanceEvaluation?> GetPerformanceEvaluationWithRetryAsync(CancellationToken ct)
    {
        for (int attempt = 0; attempt < MaxRetries; attempt++)
        {
            var result = await GetPerformanceEvaluationAsync(ct);
            if (result != null) return result;

            if (attempt < MaxRetries - 1)
            {
                _logger.LogDebug(
                    "Retrying performance evaluation fetch in {Delay}s (attempt {Attempt}/{Max})",
                    RetryDelays[attempt].TotalSeconds, attempt + 2, MaxRetries);
                await Task.Delay(RetryDelays[attempt], ct);
            }
        }
        return null;
    }
}