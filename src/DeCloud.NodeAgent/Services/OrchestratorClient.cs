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
    private string? _walletAddress;
    private Heartbeat? _lastHeartbeat = null;

    // Queue for pending commands received from heartbeat responses
    private readonly ConcurrentQueue<PendingCommand> _pendingCommands = new();

    public string? NodeId => _nodeId;
    public bool IsRegistered => !string.IsNullOrEmpty(_nodeId) && !string.IsNullOrEmpty(_authToken);
    public string? WalletAddress => _walletAddress;

    public OrchestratorClient(
        HttpClient httpClient,
        IOptions<OrchestratorClientOptions> options,
        ILogger<OrchestratorClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
        _options = options.Value;
        _walletAddress = _options.WalletAddress;

        _httpClient.BaseAddress = new Uri(_options.BaseUrl.TrimEnd('/'));
        _httpClient.Timeout = _options.Timeout;
    }

    /// <summary>
    /// Register node with orchestrator.
    /// 
    /// CPU REPORTING:
    /// - cpuCores = PhysicalCores (used for overcommit calculations)
    /// - cpuThreads = LogicalCores (informational, shows HT/SMT capability)
    /// 
    /// This allows the orchestrator to properly calculate overcommit ratios
    /// based on actual physical capacity rather than thread count.
    /// </summary>
    public async Task<bool> RegisterNodeAsync(NodeRegistration registration, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation(
                "Registering node with orchestrator at {Url}. " +
                "CPU: {PhysicalCores} physical cores, {LogicalCores} logical threads",
                _options.BaseUrl,
                registration.Resources.Cpu.PhysicalCores,
                registration.Resources.Cpu.LogicalCores);

            // Build registration request with explicit physical/logical separation
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
                    // CRITICAL: Report PHYSICAL cores as the base capacity
                    cpuCores = registration.Resources.Cpu.PhysicalCores,

                    // Also report logical threads for reference
                    cpuThreads = registration.Resources.Cpu.LogicalCores,

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

                    _logger.LogInformation(
                        "✓ Node registered successfully. NodeId: {NodeId}, " +
                        "Reported capacity: {PhysCores} physical cores / {LogicalCores} threads",
                        _nodeId,
                        registration.Resources.Cpu.PhysicalCores,
                        registration.Resources.Cpu.LogicalCores);
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

    /// <summary>
    /// Send heartbeat with current resource availability.
    /// Reports available resources based on actual VM allocations.
    /// </summary>
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

            // Calculate available resources (total - allocated to VMs)
            var allocatedCpuCores = heartbeat.VmSummaries?.Sum(vm => vm.CpuCores) ?? 0;
            var allocatedMemoryMb = heartbeat.VmSummaries?.Sum(vm => vm.MemoryMb) ?? 0;
            var allocatedStorageGb = heartbeat.VmSummaries?.Sum(vm => vm.DiskGb) ?? 0;

            var totalPhysicalCores = heartbeat.Resources.PhysicalCpuCores;
            var totalLogicalCores = heartbeat.Resources.LogicalCpuCores;
            var totalMemoryMb = heartbeat.Resources.TotalMemoryBytes / 1024 / 1024;
            var totalStorageGb = heartbeat.Resources.TotalStorageBytes / 1024 / 1024 / 1024;

            var heartbeatPayload = new
            {
                nodeId = _nodeId,
                metrics = new
                {
                    timestamp = DateTime.UtcNow.ToString("o"),
                    cpuUsagePercent = cpuUsage,
                    memoryUsagePercent = memUsage,
                    storageUsagePercent = storageUsage,
                    networkInMbps = heartbeat.Resources.NetworkInMbps,
                    networkOutMbps = heartbeat.Resources.NetworkOutMbps,
                    activeVmCount = heartbeat.ActiveVmIds?.Count ?? 0,
                    loadAverage = heartbeat.Resources.LoadAverage
                },
                availableResources = new
                {
                    // Report PHYSICAL cores (consistent with registration)
                    cpuCores = Math.Max(0, totalPhysicalCores - allocatedCpuCores),
                    cpuThreads = Math.Max(0, totalLogicalCores - allocatedCpuCores),
                    memoryMb = Math.Max(0, totalMemoryMb - allocatedMemoryMb),
                    storageGb = Math.Max(0, totalStorageGb - allocatedStorageGb),
                    bandwidthMbps = 1000
                },
                activeVmIds = heartbeat.ActiveVmIds ?? new List<string>(),
                vmSummaries = heartbeat.VmSummaries?.Select(vm => new
                {
                    vmId = vm.VmId,
                    state = vm.State,
                    cpuCores = vm.CpuCores,
                    memoryMb = vm.MemoryMb,
                    diskGb = vm.DiskGb,
                    ipAddress = vm.IpAddress,
                    vncPort = vm.VncPort,
                    macAddress = vm.MacAddress,
                    encryptedPassword = vm.EncryptedPassword
                }).ToList()
            };

            request.Content = JsonContent.Create(heartbeatPayload);

            var response = await _httpClient.SendAsync(request, ct);

            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(ct);
                await ProcessHeartbeatResponseAsync(content, ct);

                _lastHeartbeat = heartbeat;
                return true;
            }

            _logger.LogWarning(
                "Heartbeat failed: {Status} - {Content}",
                response.StatusCode,
                await response.Content.ReadAsStringAsync(ct));
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

    private async Task ProcessHeartbeatResponseAsync(string content, CancellationToken ct)
    {
        try
        {
            var json = JsonDocument.Parse(content);

            if (json.RootElement.TryGetProperty("data", out var data))
            {
                if (data.TryGetProperty("pendingCommands", out var commands) &&
                    commands.ValueKind == JsonValueKind.Array)
                {
                    foreach (var cmd in commands.EnumerateArray())
                    {
                        var pendingCommand = new PendingCommand
                        {
                            CommandId = cmd.GetProperty("commandId").GetString() ?? "",
                            Type = cmd.GetProperty("type").GetString() ?? "",
                            VmId = cmd.TryGetProperty("vmId", out var vmId) ? vmId.GetString() : null,
                            Payload = cmd.TryGetProperty("payload", out var payload) ? payload.ToString() : null
                        };

                        _pendingCommands.Enqueue(pendingCommand);
                        _logger.LogInformation(
                            "Received command: {Type} for VM {VmId}",
                            pendingCommand.Type,
                            pendingCommand.VmId);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to parse heartbeat response");
        }
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

    public bool TryGetPendingCommand(out PendingCommand? command)
    {
        return _pendingCommands.TryDequeue(out command);
    }
}