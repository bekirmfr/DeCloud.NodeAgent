using System.Net.Http.Json;
using System.Text.Json;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Services;

public class OrchestratorClientOptions
{
    public string BaseUrl { get; set; } = "https://api.decloud.network";
    public string ApiKey { get; set; } = "";
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
}

/// <summary>
/// HTTP client for communicating with the orchestration layer.
/// This is a stub implementation - replace with real API calls.
/// </summary>
public class OrchestratorClient : IOrchestratorClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<OrchestratorClient> _logger;
    private readonly OrchestratorClientOptions _options;

    // In-memory command queue for testing without orchestrator
    private readonly Queue<PendingCommand> _mockCommands = new();

    public OrchestratorClient(
        HttpClient httpClient,
        IOptions<OrchestratorClientOptions> options,
        ILogger<OrchestratorClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
        _options = options.Value;

        _httpClient.BaseAddress = new Uri(_options.BaseUrl);
        _httpClient.Timeout = _options.Timeout;
        
        if (!string.IsNullOrEmpty(_options.ApiKey))
        {
            _httpClient.DefaultRequestHeaders.Add("X-API-Key", _options.ApiKey);
        }
    }

    public async Task<bool> RegisterNodeAsync(NodeRegistration registration, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Registering node {NodeId} with orchestrator", registration.NodeId);

            // TODO: Replace with real API call
            // var response = await _httpClient.PostAsJsonAsync("/api/v1/nodes/register", registration, ct);
            // return response.IsSuccessStatusCode;

            // Stub: Always succeed
            _logger.LogInformation("Node registered (stub mode)");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to register node");
            return false;
        }
    }

    public async Task<bool> SendHeartbeatAsync(Heartbeat heartbeat, CancellationToken ct = default)
    {
        try
        {
            // TODO: Replace with real API call
            // var response = await _httpClient.PostAsJsonAsync("/api/v1/nodes/heartbeat", heartbeat, ct);
            // return response.IsSuccessStatusCode;

            // Stub: Log and succeed
            _logger.LogDebug("Heartbeat sent (stub mode): {VmCount} VMs", heartbeat.ActiveVms.Count);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to send heartbeat");
            return false;
        }
    }

    public async Task<bool> ReportVmStateChangeAsync(string vmId, VmState newState, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Reporting VM {VmId} state change to {State}", vmId, newState);

            // TODO: Replace with real API call
            // var payload = new { vmId, state = newState.ToString() };
            // var response = await _httpClient.PostAsJsonAsync("/api/v1/vms/state", payload, ct);
            // return response.IsSuccessStatusCode;

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to report VM state change");
            return false;
        }
    }

    public Task<List<PendingCommand>> GetPendingCommandsAsync(CancellationToken ct = default)
    {
        // TODO: Replace with real API call
        // var response = await _httpClient.GetAsync("/api/v1/nodes/commands", ct);
        // if (response.IsSuccessStatusCode)
        // {
        //     return await response.Content.ReadFromJsonAsync<List<PendingCommand>>(ct) ?? new();
        // }

        // Stub: Return any mock commands queued via local API
        var commands = new List<PendingCommand>();
        while (_mockCommands.TryDequeue(out var cmd))
        {
            commands.Add(cmd);
        }
        return Task.FromResult(commands);
    }

    public async Task AcknowledgeCommandAsync(string commandId, bool success, string? error = null, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Acknowledging command {CommandId}: {Success}", commandId, success);

            // TODO: Replace with real API call
            // var payload = new { commandId, success, error };
            // await _httpClient.PostAsJsonAsync("/api/v1/nodes/commands/ack", payload, ct);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to acknowledge command");
        }
    }

    /// <summary>
    /// Enqueue a command locally for testing (bypasses orchestrator)
    /// </summary>
    public void EnqueueLocalCommand(PendingCommand command)
    {
        _mockCommands.Enqueue(command);
        _logger.LogInformation("Enqueued local command: {Type}", command.Type);
    }
}
