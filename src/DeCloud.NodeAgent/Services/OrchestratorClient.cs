// OrchestratorClient with Wallet Signature Authentication
// Stateless authentication - no tokens, no expiration, no re-registration!

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Services.Auth;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;

namespace DeCloud.NodeAgent.Services;

public class OrchestratorClientOptions
{
    public string BaseUrl { get; set; } = "http://localhost:5000";
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
    public string? WalletAddress { get; set; }
    public int AgentPort { get; set; } = 5050;
}

public class OrchestratorClient : IOrchestratorClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<OrchestratorClient> _logger;
    private readonly OrchestratorClientOptions _options;
    private readonly INodeWalletService _walletService;

    private string? _nodeId;
    private string? _orchestratorPublicKey;
    private string? _walletAddress;
    private Heartbeat? _lastHeartbeat = null;

    // Queue for pending commands received from heartbeat responses
    private readonly ConcurrentQueue<PendingCommand> _pendingCommands = new();

    public string? NodeId => _nodeId;
    public bool IsRegistered => !string.IsNullOrEmpty(_nodeId);
    public string? WalletAddress => _walletAddress;

    public OrchestratorClient(
        HttpClient httpClient,
        IOptions<OrchestratorClientOptions> options,
        ILogger<OrchestratorClient> logger,
        INodeWalletService walletService)
    {
        _httpClient = httpClient;
        _logger = logger;
        _options = options.Value;
        _walletService = walletService;
        _walletAddress = _walletService.GetWalletAddress();

        _httpClient.BaseAddress = new Uri(_options.BaseUrl.TrimEnd('/'));
        _httpClient.Timeout = _options.Timeout;

        _logger.LogInformation("OrchestratorClient initialized with wallet: {Wallet}",
            _walletAddress);
    }

    /// <summary>
    /// Register the node with the orchestrator.
    /// No token returned - future requests authenticated via wallet signature!
    /// </summary>
    public async Task<bool> RegisterNodeAsync(NodeRegistration request, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Registering node with orchestrator at {Url}", _options.BaseUrl);

            // Ensure wallet address is set in request
            request.WalletAddress = _walletAddress!;

            var response = await _httpClient.PostAsJsonAsync("/api/nodes/register", request, ct);

            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(ct);
                var json = JsonDocument.Parse(content);

                if (json.RootElement.TryGetProperty("data", out var data))
                {
                    _nodeId = data.GetProperty("nodeId").GetString();

                    // ✅ NO MORE AUTH TOKEN!
                    // Authentication is now done via wallet signatures

                    // Handle orchestrator WireGuard public key
                    if (data.TryGetProperty("orchestratorWireGuardPublicKey", out var pubKeyProp) &&
                        pubKeyProp.ValueKind == JsonValueKind.String)
                    {
                        _orchestratorPublicKey = pubKeyProp.GetString();

                        if (!string.IsNullOrWhiteSpace(_orchestratorPublicKey))
                        {
                            await SaveOrchestratorPublicKeyAsync(_orchestratorPublicKey, ct);
                        }
                    }

                    _logger.LogInformation(
                        "✓ Node registered successfully: {NodeId} | Wallet: {Wallet}",
                        _nodeId, _walletAddress);

                    _logger.LogInformation(
                        "✓ Wallet-based authentication enabled - no tokens, no expiration!");

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
    /// Save orchestrator public key for WireGuard relay configuration
    /// </summary>
    private async Task SaveOrchestratorPublicKeyAsync(string publicKey, CancellationToken ct)
    {
        const string wireGuardDir = "/etc/wireguard";
        const string publicKeyPath = "/etc/wireguard/orchestrator-public.key";

        try
        {
            if (!Directory.Exists(wireGuardDir))
            {
                Directory.CreateDirectory(wireGuardDir);
                _logger.LogInformation("Created WireGuard directory: {Dir}", wireGuardDir);
            }

            await File.WriteAllTextAsync(publicKeyPath, publicKey.Trim() + "\n", ct);

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
            }

            _logger.LogInformation(
                "✓ Saved orchestrator WireGuard public key to {Path}",
                publicKeyPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save orchestrator WireGuard public key");
        }
    }

    /// <summary>
    /// Send heartbeat to orchestrator with wallet signature authentication.
    /// Stateless - no tokens, no 401 errors, no re-registration!
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
            // =====================================================
            // Create Request with Wallet Signature Headers
            // =====================================================
            var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            var requestPath = $"/api/nodes/{_nodeId}/heartbeat";

            // Message format: {nodeId}:{timestamp}:{requestPath}
            var message = $"{_nodeId}:{timestamp}:{requestPath}";

            // Sign the message with wallet
            var signature = await _walletService.SignMessageAsync(message);

            var request = new HttpRequestMessage(HttpMethod.Post, requestPath);

            // ✅ NEW: Wallet signature authentication headers
            request.Headers.Add("X-Node-Signature", signature);
            request.Headers.Add("X-Node-Timestamp", timestamp.ToString());

            // ❌ OLD: No more X-Node-Token!

            // Build heartbeat payload
            var payload = BuildHeartbeatPayload(heartbeat);
            request.Content = JsonContent.Create(payload);

            var response = await _httpClient.SendAsync(request, ct);

            // =====================================================
            // Handle Response - No More 401 Token Errors!
            // =====================================================
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(ct);

                if (response.StatusCode == HttpStatusCode.Unauthorized)
                {
                    _logger.LogError(
                        "❌ Heartbeat rejected: Signature validation failed. " +
                        "Check that node wallet matches registered wallet.");
                }
                else
                {
                    _logger.LogWarning(
                        "Heartbeat failed with status {Status}: {Error}",
                        response.StatusCode, errorContent);
                }

                return false;
            }

            // Parse successful response
            var content = await response.Content.ReadAsStringAsync(ct);
            await ProcessHeartbeatResponseAsync(content, ct);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to send heartbeat");
            return false;
        }
    }

    /// <summary>
    /// Build heartbeat payload from Heartbeat object
    /// </summary>
    private object BuildHeartbeatPayload(Heartbeat heartbeat)
    {
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
                activeVmCount = heartbeat.ActiveVms.Count
            },
            availableResources = new
            {
                computePoints = heartbeat.Resources.AvailableVirtualCpuCores,
                memoryBytes = heartbeat.Resources.AvailableMemoryBytes,
                storageBytes = heartbeat.Resources.AvailableStorageBytes
            },
            activeVms = heartbeat.ActiveVms.Select(v => new
            {
                vmId = v.VmId,
                name = v.Name,
                state = v.State.ToString(),
                ownerId = v.OwnerId ?? "unknown",
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
            }),
            cgnatInfo = heartbeat.CgnatInfo != null ? new
            {
                assignedRelayNodeId = heartbeat.CgnatInfo.AssignedRelayNodeId,
                tunnelIp = heartbeat.CgnatInfo.TunnelIp,
                wireGuardConfig = heartbeat.CgnatInfo.WireGuardConfig,
                publicEndpoint = heartbeat.CgnatInfo.PublicEndpoint,
                tunnelStatus = heartbeat.CgnatInfo.TunnelStatus.ToString(),
                lastHandshake = heartbeat.CgnatInfo.LastHandshake?.ToString("O")
            } : null
        };

        return payload;
    }

    /// <summary>
    /// Process heartbeat response including pending commands and CGNAT info
    /// </summary>
    private Task ProcessHeartbeatResponseAsync(string content, CancellationToken ct)
    {
        _logger.LogDebug("Processing heartbeat response");
        try
        {
            var json = JsonDocument.Parse(content);

            if (!json.RootElement.TryGetProperty("data", out var data))
            {
                _logger.LogWarning("Heartbeat response missing 'data' property");
                return Task.CompletedTask;
            }

            // Process Pending Commands
            if (data.TryGetProperty("pendingCommands", out var commands) &&
                commands.ValueKind == JsonValueKind.Array)
            {
                foreach (var cmd in commands.EnumerateArray())
                {
                    var commandId = cmd.GetProperty("commandId").GetString() ?? "";

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

            // Extract and Store CGNAT Info
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

                if (_lastHeartbeat != null)
                {
                    _lastHeartbeat.CgnatInfo = new CgnatNodeInfo
                    {
                        AssignedRelayNodeId = relayId,
                        TunnelIp = tunnelIp,
                        WireGuardConfig = wgConfig,
                        PublicEndpoint = publicEndpoint,
                        TunnelStatus = TunnelStatus.Disconnected,
                        LastHandshake = null
                    };

                    _logger.LogInformation("✓ Stored CGNAT info in heartbeat");
                }
            }
            else if (_lastHeartbeat != null && _lastHeartbeat.CgnatInfo != null)
            {
                _logger.LogInformation("Clearing previous CGNAT info - node no longer behind NAT");
                _lastHeartbeat.CgnatInfo = null;
            }

            _lastHeartbeat = _lastHeartbeat;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to process heartbeat response");
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Get all pending commands from the queue
    /// </summary>
    public Task<List<PendingCommand>> GetPendingCommandsAsync(CancellationToken ct = default)
    {
        var commands = new List<PendingCommand>();

        while (_pendingCommands.TryDequeue(out var cmd))
        {
            commands.Add(cmd);
        }

        return Task.FromResult(commands);
    }

    /// <summary>
    /// Report VM state change to orchestrator (via next heartbeat)
    /// </summary>
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

    /// <summary>
    /// Acknowledge command execution to orchestrator with wallet signature
    /// </summary>
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
            // =====================================================
            // Create Request with Wallet Signature Headers
            // =====================================================
            var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            var requestPath = $"/api/nodes/{_nodeId}/commands/{commandId}/acknowledge";

            var message = $"{_nodeId}:{timestamp}:{requestPath}";
            var signature = await _walletService.SignMessageAsync(message);

            var request = new HttpRequestMessage(HttpMethod.Post, requestPath);

            // ✅ NEW: Wallet signature authentication
            request.Headers.Add("X-Node-Signature", signature);
            request.Headers.Add("X-Node-Timestamp", timestamp.ToString());

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

    /// <summary>
    /// Get the last heartbeat object (for accessing CGNAT info, etc.)
    /// </summary>
    public Heartbeat? GetLastHeartbeat()
    {
        return _lastHeartbeat;
    }
}