// Sends enhanced heartbeat with detailed VM information

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using System.Text.Json;

namespace DeCloud.NodeAgent.Services;

public class OrchestratorClientOptions
{
    public string BaseUrl { get; set; } = "http://localhost:5000";
    public string? ApiKey { get; set; }
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
    public string? WalletAddress { get; set; }
}

public class OrchestratorClient : IOrchestratorClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<OrchestratorClient> _logger;
    private readonly OrchestratorClientOptions _options;

    private string? _nodeId;
    private string? _authToken;
    private Heartbeat? _lastHeartbeat = null;

    // Queue for pending commands received from heartbeat responses
    private readonly ConcurrentQueue<PendingCommand> _pendingCommands = new();

    public string? NodeId => _nodeId;
    public bool IsRegistered => !string.IsNullOrEmpty(_nodeId) && !string.IsNullOrEmpty(_authToken);
    public string? WalletAddress { get; set; }

    public OrchestratorClient(
        HttpClient httpClient,
        IOptions<OrchestratorClientOptions> options,
        ILogger<OrchestratorClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
        _options = options.Value;

        _httpClient.BaseAddress = new Uri(_options.BaseUrl.TrimEnd('/'));
        _httpClient.Timeout = _options.Timeout;
    }

    public async Task<bool> RegisterNodeAsync(NodeRegistration registration, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Registering node with orchestrator at {Url}", _options.BaseUrl);

            var request = new
            {
                nodeId = registration.NodeId,
                machineId = registration.MachineId,
                name = registration.Name,
                walletAddress = registration.WalletAddress,
                publicIp = registration.PublicIp,
                agentPort = registration.AgentPort,
                resources = new
                {
                    cpuCores = registration.Resources.Cpu.LogicalCores,
                    memoryMb = registration.Resources.Memory.TotalBytes / 1024 / 1024,
                    storageGb = registration.Resources.Storage.Sum(s => s.TotalBytes) / 1024 / 1024 / 1024,
                    bandwidthMbps = 1000
                },
                agentVersion = registration.AgentVersion,
                supportedImages = registration.SupportedImages,
                supportsGpu = registration.SupportsGpu,
                gpuInfo = registration.GpuInfo != null ? new
                {
                    model = registration.GpuInfo.Model,
                    vramMb = registration.GpuInfo.MemoryBytes / 1024 / 1024,
                    count = 1,
                    driver = registration.GpuInfo.DriverVersion
                } : null,
                region = registration.Region,
                zone = registration.Zone
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
            var request = new HttpRequestMessage(HttpMethod.Post, $"/api/nodes/{_nodeId}/heartbeat");
            request.Headers.Add("X-Node-Token", _authToken);

            // Calculate resource metrics
            var cpuUsage = heartbeat.Resources.CpuUsagePercent;
            var memUsage = heartbeat.Resources.TotalMemoryBytes > 0
                ? (double)heartbeat.Resources.UsedMemoryBytes / heartbeat.Resources.TotalMemoryBytes * 100
                : 0;
            var storageUsage = heartbeat.Resources.TotalStorageBytes > 0
                ? (double)heartbeat.Resources.UsedStorageBytes / heartbeat.Resources.TotalStorageBytes * 100
                : 0;

            var payload = new
            {
                nodeId = heartbeat.NodeId,
                metrics = new
                {
                    timestamp = heartbeat.Timestamp.ToString("O"),
                    cpuUsagePercent = cpuUsage,
                    memoryUsagePercent = memUsage,
                    storageUsagePercent = storageUsage,
                    networkInMbps = 0,
                    networkOutMbps = 0,
                    activeVmCount = heartbeat.ActiveVmDetails.Count,
                    loadAverage = 0.0
                },
                availableResources = new
                {
                    cpuCores = heartbeat.Resources.AvailableVCpus,
                    memoryMb = heartbeat.Resources.AvailableMemoryBytes / 1024 / 1024,
                    storageGb = heartbeat.Resources.AvailableStorageBytes / 1024 / 1024 / 1024,
                    bandwidthMbps = 1000
                },
                activeVms = heartbeat.ActiveVmDetails.Select(v => new
                {
                    vmId = v.VmId,
                    name = v.Name,
                    tenantId = v.TenantId,
                    state = v.State.ToString(),  // Convert enum to string
                    ipAddress = v.IpAddress,
                    cpuUsagePercent = v.CpuUsagePercent,
                    startedAt = v.StartedAt.ToString("O"),
                    vCpus = v.VCpus,
                    memoryBytes = v.MemoryBytes,
                    diskBytes = v.DiskBytes,
                    // These fields are populated by HeartbeatService from VmInstance
                    vncPort = v.VncPort,
                    macAddress = v.MacAddress,
                    encryptedPassword = v.EncryptedPassword
                }).ToList()
            };

            request.Content = JsonContent.Create(payload);

            var response = await _httpClient.SendAsync(request, ct);

            if (response.IsSuccessStatusCode)
            {
                _lastHeartbeat = heartbeat;

                var content = await response.Content.ReadAsStringAsync(ct);
                await ProcessHeartbeatResponseAsync(content, ct);

                if (heartbeat.ActiveVmDetails.Count > 0)
                {
                    _logger.LogDebug("Heartbeat sent successfully: {VmCount} VMs",
                        heartbeat.ActiveVmDetails.Count);
                }

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

    public Heartbeat? GetLastHeartbeat()
    {
        return _lastHeartbeat;
    }

    private Task ProcessHeartbeatResponseAsync(string content, CancellationToken ct)
    {
        try
        {
            var json = JsonDocument.Parse(content);

            if (json.RootElement.TryGetProperty("data", out var data) &&
                data.TryGetProperty("pendingCommands", out var commands) &&
                commands.ValueKind == JsonValueKind.Array)
            {
                foreach (var cmd in commands.EnumerateArray())
                {
                    var commandId = cmd.GetProperty("commandId").GetString() ?? "";

                    // Handle 'type' as either string OR number (enum integer)
                    CommandType commandType;
                    var typeProp = cmd.GetProperty("type");

                    if (typeProp.ValueKind == JsonValueKind.Number)
                    {
                        commandType = (CommandType)typeProp.GetInt32();
                    }
                    else if (typeProp.ValueKind == JsonValueKind.String)
                    {
                        var typeStr = typeProp.GetString() ?? "";
                        if (!Enum.TryParse<CommandType>(typeStr, true, out commandType))
                        {
                            _logger.LogWarning("Unknown command type: {Type}", typeStr);
                            continue;
                        }
                    }
                    else
                    {
                        _logger.LogWarning("Unexpected type format: {Kind}", typeProp.ValueKind);
                        continue;
                    }

                    var payload = cmd.GetProperty("payload").GetString() ?? "";

                    _logger.LogInformation("Received command from orchestrator: {Type} (ID: {CommandId})",
                        commandType, commandId);

                    // =====================================================
                    // Without this, commands are never processed!
                    // =====================================================
                    _pendingCommands.Enqueue(new PendingCommand
                    {
                        CommandId = commandId,
                        Type = commandType,
                        Payload = payload,
                        IssuedAt = DateTime.UtcNow
                    });
                }

                if (commands.GetArrayLength() > 0)
                {
                    _logger.LogInformation("Enqueued {Count} command(s) for processing",
                        commands.GetArrayLength());
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to process heartbeat response");
        }

        return Task.CompletedTask;
    }

    public Task<List<PendingCommand>> GetPendingCommandsAsync(CancellationToken ct = default)
    {
        var commands = new List<PendingCommand>();

        while (_pendingCommands.TryDequeue(out var cmd))
        {
            commands.Add(cmd);
        }

        return Task.FromResult(commands);
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
            // The state will be reported in the next heartbeat via ActiveVms
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to report VM state change");
            return false;
        }
    }

    public async Task<bool> AcknowledgeCommandAsync(
        string commandId,
        bool success,
        string? errorMessage,
        CancellationToken ct = default)
    {
        if (!IsRegistered)
        {
            _logger.LogWarning("Cannot acknowledge command - node not registered");
            return false;
        }

        try
        {
            var request = new HttpRequestMessage(
                HttpMethod.Post,
                $"/api/nodes/{_nodeId}/commands/{commandId}/acknowledge");

            request.Headers.Add("X-Node-Token", _authToken);

            var payload = new
            {
                commandId,
                success,
                errorMessage = errorMessage ?? string.Empty,
                completedAt = DateTime.UtcNow.ToString("O")
            };

            request.Content = JsonContent.Create(payload);

            var response = await _httpClient.SendAsync(request, ct);

            if (response.IsSuccessStatusCode)
            {
                _logger.LogDebug("Command {CommandId} acknowledged: {Success}", commandId, success);
                return true;
            }

            _logger.LogWarning("Failed to acknowledge command {CommandId}: {Status}",
                commandId, response.StatusCode);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error acknowledging command {CommandId}", commandId);
            return false;
        }
    }
}