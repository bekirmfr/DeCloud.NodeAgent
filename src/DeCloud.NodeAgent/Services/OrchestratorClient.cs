using System.Net.Http.Json;
using System.Text.Json;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Services;

public class OrchestratorClientOptions
{
    public string BaseUrl { get; set; } = "http://localhost:5050";
    public string ApiKey { get; set; } = "";
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
    public string WalletAddress { get; set; } = "0x0000000000000000000000000000000000000000";
}

public class OrchestratorClient : IOrchestratorClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<OrchestratorClient> _logger;
    private readonly OrchestratorClientOptions _options;

    // Registration credentials - set after successful registration
    private string? _nodeId;
    private string? _authToken;

    public string? NodeId => _nodeId;
    public bool IsRegistered => !string.IsNullOrEmpty(_nodeId) && !string.IsNullOrEmpty(_authToken);

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
    }

    public async Task<bool> RegisterNodeAsync(NodeRegistration registration, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Registering node with orchestrator at {Url}", _options.BaseUrl);

            var request = new
            {
                name = registration.NodeId,
                walletAddress = _options.WalletAddress,
                publicIp = registration.Resources.Network.PublicIp,
                agentPort = 5100,
                resources = new
                {
                    cpuCores = registration.Resources.Cpu.LogicalCores,
                    memoryMb = registration.Resources.Memory.TotalBytes / 1024 / 1024,
                    storageGb = registration.Resources.Storage.Sum(s => s.TotalBytes) / 1024 / 1024 / 1024,
                    bandwidthMbps = 1000
                },
                agentVersion = "1.0.0",
                supportedImages = new[] { "ubuntu-24.04", "ubuntu-22.04", "debian-12" },
                supportsGpu = registration.Resources.Gpus.Any(),
                gpuInfo = registration.Resources.Gpus.FirstOrDefault() != null ? new
                {
                    model = registration.Resources.Gpus.First().Model,
                    vramMb = registration.Resources.Gpus.First().MemoryBytes / 1024 / 1024,
                    count = registration.Resources.Gpus.Count,
                    driver = registration.Resources.Gpus.First().DriverVersion
                } : null,
                region = "default",
                zone = "default"
            };

            var response = await _httpClient.PostAsJsonAsync("/api/nodes/register", request, ct);

            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(ct);
                var json = JsonDocument.Parse(content);

                if (json.RootElement.TryGetProperty("data", out var data))
                {
                    _nodeId = data.GetProperty("nodeId").GetString();
                    _authToken = data.GetProperty("authToken").GetString();

                    _logger.LogInformation("Node registered successfully. NodeId: {NodeId}", _nodeId);
                    return true;
                }
            }

            var errorContent = await response.Content.ReadAsStringAsync(ct);
            _logger.LogError("Registration failed: {Status} - {Content}", response.StatusCode, errorContent);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to register node");
            return false;
        }
    }

    public async Task<bool> SendHeartbeatAsync(Heartbeat heartbeat, CancellationToken ct = default)
    {
        if (!IsRegistered)
        {
            _logger.LogWarning("Cannot send heartbeat - node not registered");
            return false;
        }

        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, $"/api/nodes/{_nodeId}/heartbeat");
            request.Headers.Add("X-Node-Token", _authToken);

            var payload = new
            {
                nodeId = _nodeId,
                metrics = new
                {
                    timestamp = DateTime.UtcNow,
                    cpuUsagePercent = heartbeat.Resources.CpuUsagePercent,
                    memoryUsagePercent = (double)heartbeat.Resources.UsedMemoryBytes / heartbeat.Resources.TotalMemoryBytes * 100,
                    storageUsagePercent = (double)heartbeat.Resources.UsedStorageBytes / heartbeat.Resources.TotalStorageBytes * 100,
                    networkInMbps = 0,
                    networkOutMbps = 0,
                    activeVmCount = heartbeat.ActiveVms.Count,
                    loadAverage = 0.0
                },
                availableResources = new
                {
                    cpuCores = heartbeat.Resources.AvailableVCpus,
                    memoryMb = heartbeat.Resources.AvailableMemoryBytes / 1024 / 1024,
                    storageGb = heartbeat.Resources.AvailableStorageBytes / 1024 / 1024 / 1024,
                    bandwidthMbps = 1000
                },
                activeVmIds = heartbeat.ActiveVms.Select(v => v.VmId).ToList()
            };

            request.Content = JsonContent.Create(payload);

            var response = await _httpClient.SendAsync(request, ct);

            if (response.IsSuccessStatusCode)
            {
                _logger.LogDebug("Heartbeat sent successfully: {VmCount} VMs", heartbeat.ActiveVms.Count);
                return true;
            }

            _logger.LogWarning("Heartbeat failed with status: {Status}", response.StatusCode);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to send heartbeat");
            return false;
        }
    }

    public async Task<bool> ReportVmStateChangeAsync(string vmId, VmState newState, CancellationToken ct = default)
    {
        if (!IsRegistered)
        {
            _logger.LogWarning("Cannot report VM state - node not registered");
            return false;
        }

        try
        {
            _logger.LogInformation("Reporting VM {VmId} state change to {State}", vmId, newState);
            // TODO: Implement when Orchestrator has this endpoint
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
        // Commands come back with heartbeat response
        return Task.FromResult(new List<PendingCommand>());
    }

    public Task AcknowledgeCommandAsync(string commandId, bool success, string? error = null, CancellationToken ct = default)
    {
        _logger.LogInformation("Command {CommandId} acknowledged: {Success}", commandId, success);
        return Task.CompletedTask;
    }
}