// Sends enhanced heartbeat with detailed VM information

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using System.Net.Mail;
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
    private string? _orchestratorPublicKey;
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

    public async Task<bool> RegisterNodeAsync(NodeRegistration request, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Registering node with orchestrator at {Url}", _options.BaseUrl);

            var response = await _httpClient.PostAsJsonAsync("/api/nodes/register", request, ct);

            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(ct);
                var json = JsonDocument.Parse(content);

                if (json.RootElement.TryGetProperty("data", out var data))
                {
                    _nodeId = data.GetProperty("nodeId").GetString();
                    _authToken = data.GetProperty("authToken").GetString();

                    // =====================================================
                    // Handle orchestrator WireGuard public key
                    // =====================================================
                    if (data.TryGetProperty("orchestratorWireGuardPublicKey", out var pubKeyProp) &&
                        pubKeyProp.ValueKind == JsonValueKind.String)
                    {
                        _orchestratorPublicKey = pubKeyProp.GetString();

                        if (!string.IsNullOrWhiteSpace(_orchestratorPublicKey))
                        {
                            await SaveOrchestratorPublicKeyAsync(_orchestratorPublicKey, ct);
                        }
                    }
                    else
                    {
                        _logger.LogInformation(
                            "No orchestrator WireGuard public key in registration response - " +
                            "WireGuard may not be enabled on orchestrator");
                    }

                    _logger.LogInformation("Node registered successfully. NodeId: {NodeId} Response: {ResponseJson}", _nodeId, json.ToString());
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

    // =====================================================
    // Save orchestrator public key
    // =====================================================
    private async Task SaveOrchestratorPublicKeyAsync(string publicKey, CancellationToken ct)
    {
        const string wireGuardDir = "/etc/wireguard";
        const string publicKeyPath = "/etc/wireguard/orchestrator-public.key";

        try
        {
            // Ensure directory exists
            if (!Directory.Exists(wireGuardDir))
            {
                Directory.CreateDirectory(wireGuardDir);
                _logger.LogInformation("Created WireGuard directory: {Dir}", wireGuardDir);
            }

            // Write orchestrator public key
            await File.WriteAllTextAsync(publicKeyPath, publicKey.Trim() + "\n", ct);

            // Set proper permissions (readable by all, writable by root)
            var chmodProcess = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "chmod",
                Arguments = $"644 {publicKeyPath}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            });

            if (chmodProcess != null)
            {
                await chmodProcess.WaitForExitAsync(ct);

                if (chmodProcess.ExitCode != 0)
                {
                    var error = await chmodProcess.StandardError.ReadToEndAsync(ct);
                    _logger.LogWarning("chmod failed: {Error}", error);
                }
            }

            _logger.LogInformation(
                "✓ Saved orchestrator WireGuard public key to {Path} - " +
                "relay VMs on this node will have orchestrator peer pre-configured",
                publicKeyPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to save orchestrator WireGuard public key to {Path} - " +
                "relay VMs will not have orchestrator peer pre-configured. " +
                "Ensure node agent has write permissions to /etc/wireguard/",
                publicKeyPath);

            // Don't throw - registration can still succeed without this
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
            var cpuUsage = heartbeat.Resources.VirtualCpuUsagePercent;
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
                    //networkInMbps = 0,
                    //networkOutMbps = 0,
                    activeVmCount = heartbeat.ActiveVms.Count,
                    //loadAverage = 0.0
                },
                availableResources = new
                {
                    computePoints = heartbeat.Resources.AvailableVirtualCpuCores,
                    memoryBytes = heartbeat.Resources.AvailableMemoryBytes,
                    storageBytes = heartbeat.Resources.AvailableStorageBytes,
                    //bandwidthMbps = 1000
                },
                activeVms = heartbeat.ActiveVms.Select(v => new
                {
                    vmId = v.VmId,
                    name = v.Name,
                    state = v.State.ToString(),  // Convert enum to string
                    ownerId = v.OwnerId ?? string.Empty,
                    isIpAssigned = v.IsIpAssigned,
                    ipAddress = v.IpAddress,
                    macAddress = v.MacAddress,
                    sshPort = 2222,
                    vncPort = v.VncPort,
                    virtualCpuCores = v.VirtualCpuCores,  
                    qualityTier = v.QualityTier,
                    computePointCost = v.ComputePointCost,
                    memoryBytes = v.MemoryBytes,
                    diskBytes = v.DiskBytes,
                    startedAt = v.StartedAt.ToString("O")
                }).ToList()
            };

            request.Content = JsonContent.Create(payload);

            _logger.LogDebug("Sending heartbeat: {request}", request.Content);

            var response = await _httpClient.SendAsync(request, ct);

            if (response.IsSuccessStatusCode)
            {
                _lastHeartbeat = heartbeat;

                var content = await response.Content.ReadAsStringAsync(ct);
                await ProcessHeartbeatResponseAsync(content, ct);

                if (heartbeat.ActiveVms.Count > 0)
                {
                    _logger.LogDebug("Heartbeat sent successfully: {VmCount} VMs",
                        heartbeat.ActiveVms.Count);
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
        _logger.LogDebug("Processing heartbeat response: {ResponseContent}", content);
        try
        {
            var json = JsonDocument.Parse(content);

            if (!json.RootElement.TryGetProperty("data", out var data))
            {
                _logger.LogWarning("Heartbeat response missing 'data' property");
                throw new ArgumentException("Invalid heartbeat response format");
            }

            if (data.TryGetProperty("pendingCommands", out var commands) &&
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

            // ========================================
            // Extract and store cgnatInfo
            // ========================================
            if (data.TryGetProperty("cgnatInfo", out var cgnatInfoElement) &&
                cgnatInfoElement.ValueKind != JsonValueKind.Null)
            {
                var relayId = cgnatInfoElement.GetProperty("assignedRelayNodeId").GetString();
                var tunnelIp = cgnatInfoElement.GetProperty("tunnelIp").GetString() ?? "";
                var wgConfig = cgnatInfoElement.TryGetProperty("wireGuardConfig", out var wgConfigProp)
                    ? wgConfigProp.GetString()
                    : null;
                var publicEndpoint = cgnatInfoElement.TryGetProperty("publicEndpoint", out var endpointProp)
                    ? endpointProp.GetString() ?? ""
                    : "";

                _logger.LogInformation(
                    "Received relay assignment: Relay {RelayId}, Tunnel IP {TunnelIp}",
                    relayId, tunnelIp);

                // Store cgnatInfo in the heartbeat
                if (_lastHeartbeat != null)
                {
                    _lastHeartbeat.CgnatInfo = new CgnatNodeInfo
                    {
                        AssignedRelayNodeId = relayId,
                        TunnelIp = tunnelIp,
                        WireGuardConfig = wgConfig,
                        PublicEndpoint = publicEndpoint,
                        TunnelStatus = TunnelStatus.Disconnected, // WireGuard will update this
                        LastHandshake = null
                    };

                    _logger.LogInformation(
                        "✓ Stored CGNAT info in heartbeat for WireGuard configuration");
                }
            }
            else if (_lastHeartbeat != null && _lastHeartbeat.CgnatInfo != null)
            {
                // No cgnatInfo in response but we had it before - clear it
                _logger.LogInformation("Clearing previous CGNAT info - node no longer behind NAT");
                _lastHeartbeat.CgnatInfo = null;
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