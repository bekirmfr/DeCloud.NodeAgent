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
using DeCloud.NodeAgent.Core.Models.State;
using DeCloud.Shared.Contracts;
using DeCloud.Shared.Models;
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
    public async Task<NodeSummaryResponse?> GetNodeSummaryAsync(CancellationToken ct = default)
    {
        try
        {
            _logger.LogDebug("Fetching node summary from orchestrator...");

            var response = await _httpClient.GetAsync("/api/nodes/me", ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning(
                    "Failed to get node summary: {StatusCode} - {Reason}",
                    (int)response.StatusCode, response.ReasonPhrase);
                return null;
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var result = JsonSerializer.Deserialize<NodeSummaryResponse>(content, _jsonOptions);

            if (result != null)
            {
                _logger.LogDebug(
                    "✓ Node summary received: Status={Status}, LastHeartbeat={LastHeartbeat}",
                    result.Status, result.LastHeartbeat);
            }

            return result;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error fetching node summary");
            return null;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogDebug("Node summary request cancelled");
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching node summary");
            return null;
        }
    }

    // =====================================================================
    // GET /api/nodes/me/config - Scheduling Configuration
    // =====================================================================

    /// <inheritdoc />
    public async Task<AgentSchedulingConfig?> GetSchedulingConfigAsync(CancellationToken ct = default)
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
            var config = JsonSerializer.Deserialize<AgentSchedulingConfig>(content, _jsonOptions);

            if (config != null)
            {
                _logger.LogInformation(
                    "✓ Scheduling config received: v{Version}, Baseline={Baseline}, Overcommit={Overcommit:F1}",
                    config.Version, config.BaselineBenchmark, config.BaselineOvercommitRatio);

                _nodeState.UpdateSchedulingConfig(config);
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

                // Update local state service
                _nodeState.UpdatePerformanceEvaluation(evaluation);
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
    // GET /api/nodes/me/allocation - Orchestrator-Stored Allocation
    // =====================================================================

    /// <inheritdoc />
    public async Task<NodeAllocateResponse?> GetAllocationAsync(CancellationToken ct = default)
    {
        try
        {
            _logger.LogDebug("Fetching allocation from orchestrator...");

            var response = await _httpClient.GetAsync("/api/nodes/me/allocation", ct);

            if (!response.IsSuccessStatusCode)
            {
                // 404 here means the node isn't registered yet — fall back to
                // the local cache (loaded by NodeMetadataService.LoadResolvedAllocationAsync).
                // Other failures are equally non-fatal at this stage.
                _logger.LogWarning(
                    "Could not fetch orchestrator allocation: {StatusCode} - {Reason}. " +
                    "Falling back to local cache / settings-derived values.",
                    (int)response.StatusCode, response.ReasonPhrase);
                return null;
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var allocation = JsonSerializer.Deserialize<NodeAllocateResponse>(content, _jsonOptions);

            if (allocation == null)
            {
                _logger.LogWarning(
                    "Orchestrator returned empty allocation response — keeping local cache.");
                return null;
            }

            // Apply the same way AllocateAsync does — populate in-memory state
            // and persist the cache file. UpdateFromOrchestratorResolutionAsync's
            // "> 0" guards correctly skip null/zero fields, so a not-yet-evaluated
            // node won't have its non-zero in-memory values clobbered.
            await _nodeMetadata.UpdateFromOrchestratorResolutionAsync(allocation, ct);

            _logger.LogInformation(
                "✓ Allocation fetched from orchestrator: CPU={CpuPct:P0}, " +
                "Mem={MemPct:P0}, Stor={StorPct:P0}, " +
                "Resolved={Resolved}",
                allocation.EffectiveCpuPercent,
                allocation.EffectiveMemoryPercent,
                allocation.EffectiveStoragePercent,
                allocation.ResolvedComputePoints.HasValue ? "yes" : "no (not evaluated)");

            return allocation;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(ex,
                "HTTP error fetching orchestrator allocation — keeping local cache");
            return null;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogDebug("Allocation request cancelled");
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Error fetching orchestrator allocation — keeping local cache");
            return null;
        }
    }

    // =====================================================================
    // POST /api/nodes/me/evaluate - Request Performance Re-evaluation
    // =====================================================================

    /// <inheritdoc />
    /// <inheritdoc />
    public async Task<EvaluateNodeResponse?> EvaluateNodeAsync(CancellationToken ct = default)
    {
        try
        {
            var inventory = await _resourceDiscovery.DiscoverAllAsync(ct);

            _logger.LogInformation("Requesting evaluation from orchestrator...");

            var response = await _httpClient.PostAsJsonAsync(
                "/api/nodes/me/evaluate",
                inventory,
                ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning(
                    "Evaluation failed: {StatusCode} - {Reason}",
                    (int)response.StatusCode, response.ReasonPhrase);
                return null;
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var evalResponse = JsonSerializer.Deserialize<EvaluateNodeResponse>(
                content, _jsonOptions);

            if (evalResponse == null)
            {
                _logger.LogError("Failed to deserialize evaluate response");
                return null;
            }

            // ── Persist performance evaluation ───────────────────────────
            _nodeState.UpdatePerformanceEvaluation(evalResponse.PerformanceEvaluation);
            _logger.LogInformation(
                "✓ Performance evaluation: benchmark={Benchmark}, " +
                "points={Points}, tier={Tier}",
                evalResponse.PerformanceEvaluation.BenchmarkScore,
                evalResponse.PerformanceEvaluation.TotalComputePoints,
                evalResponse.PerformanceEvaluation.HighestTier);

            // ── Persist scheduling config ────────────────────────────────
            _nodeState.UpdateSchedulingConfig(evalResponse.SchedulingConfig);
            _logger.LogInformation(
                "✓ Scheduling config v{Version} received",
                evalResponse.SchedulingConfig.Version);

            // ── Persist obligation identity states ───────────────────────
            if (evalResponse.ObligationStates is { Count: > 0 } states)
            {
                foreach (var (role, payload) in states)
                {
                    if (string.IsNullOrWhiteSpace(payload.StateJson))
                    {
                        _logger.LogWarning(
                            "Evaluate response: empty state JSON for role '{Role}' — skipping",
                            role);
                        continue;
                    }

                    var written = await _obligationState.SaveStateAsync(
                        role,
                        payload.StateJson,
                        payload.Version,
                        ct);

                    _logger.LogInformation(
                        written
                            ? "✓ Obligation state saved: {Role} v{Version}"
                            : "Obligation state {Role} v{Version} already current",
                        role, payload.Version);
                }
            }

            // ── Persist system VM templates ──────────────────────────────
            if (evalResponse.SystemTemplates is { Count: > 0 } templates)
            {
                foreach (var (role, payload) in templates)
                {
                    if (string.IsNullOrWhiteSpace(payload.TemplateJson))
                    {
                        _logger.LogWarning(
                            "Evaluate response: empty template JSON for role '{Role}' — skipping",
                            role);
                        continue;
                    }

                    var written = await _obligationState.SaveSystemTemplateAsync(
                        role,
                        payload.TemplateJson,
                        payload.Revision,
                        payload.TemplateId,
                        ct);

                    _logger.LogInformation(
                        written
                            ? "✓ System template saved: {Role} r{Revision}"
                            : "System template {Role} r{Revision} already current",
                        role, payload.Revision);
                }
            }

            // ── Persist obligation descriptors ───────────────────────────
            if (evalResponse.Obligations is { Count: > 0 } obligations)
            {
                var descriptors = obligations
                    .Where(o => !string.IsNullOrWhiteSpace(o.Role))
                    .Select(o => new ObligationDescriptor
                    {
                        Role = o.Role,
                        Deps = o.Deps ?? [],
                    })
                    .ToList();

                await _obligationState.SaveObligationsAsync(descriptors, ct);
                _logger.LogInformation(
                    "✓ Obligations persisted: [{Roles}]",
                    string.Join(", ", descriptors.Select(d => d.Role)));
            }

            // ── DHT bootstrap peers ──────────────────────────────────────
            if (evalResponse.DhtBootstrapPeers is { Count: > 0 } peers)
            {
                _logger.LogInformation(
                    "✓ {Count} DHT bootstrap peer(s) received",
                    peers.Count);
            }

            return evalResponse;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error during evaluation");
            return null;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogDebug("Evaluation request cancelled");
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during evaluation");
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

    /// <inheritdoc />
    public async Task<Dictionary<string, string>> GetVmIngressUrlsAsync(
        CancellationToken ct = default)
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/nodes/me/vm-ingress", ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogDebug(
                    "GetVmIngressUrls returned {Status}", (int)response.StatusCode);
                return new Dictionary<string, string>();
            }

            var content = await response.Content.ReadAsStringAsync(ct);

            var result = JsonSerializer.Deserialize<Dictionary<string, string>>(
                content, _jsonOptions);

            return result ?? new Dictionary<string, string>();
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Could not fetch VM ingress URLs from orchestrator");
            return new Dictionary<string, string>();
        }
    }

    /// <inheritdoc />
    public async Task<List<SystemVmObligationDto>?> GetObligationsAsync(
        CancellationToken ct = default)
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/nodes/me/obligations", ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogDebug("GetObligations returned {Status}", (int)response.StatusCode);
                return null;
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var json = JsonSerializer.Deserialize<JsonElement>(content, _jsonOptions);

            if (!json.TryGetProperty("obligations", out var arr))
                return null;

            return JsonSerializer.Deserialize<List<SystemVmObligationDto>>(
                arr.GetRawText(), _jsonOptions);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Could not fetch obligations from orchestrator");
            return null;
        }
    }

    // =====================================================================
    // Retry Helpers
    // =====================================================================

    private async Task<AgentSchedulingConfig?> GetSchedulingConfigWithRetryAsync(CancellationToken ct)
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