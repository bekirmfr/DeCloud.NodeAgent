// Enhanced OrchestratorClient with robust authentication and automatic re-registration
// Handles expired tokens gracefully by re-registering on 401 Unauthorized errors

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
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

    private string? _nodeId;
    private string? _authToken;
    private string? _orchestratorPublicKey;
    private string? _walletAddress;
    private Heartbeat? _lastHeartbeat = null;

    // Track re-registration attempts to prevent infinite loops
    private int _reregistrationAttempts = 0;
    private const int MaxReregistrationAttempts = 3;
    private DateTime _lastReregistrationAttempt = DateTime.MinValue;
    private readonly TimeSpan _reregistrationCooldown = TimeSpan.FromMinutes(5);

    // Store last registration request for re-registration
    private NodeRegistration? _lastRegistrationRequest = null;

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
    /// Register or re-register the node with the orchestrator
    /// </summary>
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

                    // Reset re-registration tracking on successful registration
                    _reregistrationAttempts = 0;
                    _lastReregistrationAttempt = DateTime.MinValue;

                    // Store registration request for potential re-registration
                    _lastRegistrationRequest = request;

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

                    _logger.LogInformation(
                        "✓ Node registered successfully: {NodeId} | Token: {TokenPreview}...",
                        _nodeId, _authToken?[..Math.Min(12, _authToken.Length)]);
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

    /// <summary>
    /// Send heartbeat to orchestrator with automatic re-registration on 401 errors
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

            // Build heartbeat payload
            var payload = BuildHeartbeatPayload(heartbeat);
            request.Content = JsonContent.Create(payload);

            var response = await _httpClient.SendAsync(request, ct);

            // =====================================================
            // CRITICAL: Handle 401 Unauthorized - Token Expired/Invalid
            // =====================================================
            if (response.StatusCode == HttpStatusCode.Unauthorized)
            {
                _logger.LogWarning(
                    "❌ Heartbeat rejected with 401 Unauthorized - auth token expired or invalid");

                // Attempt automatic re-registration
                var reregistered = await AttemptReregistrationAsync(ct);

                if (reregistered)
                {
                    // Retry heartbeat with new token
                    _logger.LogInformation("Retrying heartbeat after successful re-registration");
                    return await SendHeartbeatAsync(heartbeat, ct);
                }

                _logger.LogError("Failed to re-register node after 401 error - manual intervention may be required");
                return false;
            }

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(ct);
                _logger.LogWarning(
                    "Heartbeat failed with status {Status}: {Error}",
                    response.StatusCode, errorContent);
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
    /// Attempt to re-register the node after auth failure
    /// Includes cooldown period and max attempt limits
    /// </summary>
    private async Task<bool> AttemptReregistrationAsync(CancellationToken ct)
    {
        // Check cooldown period
        if (DateTime.UtcNow - _lastReregistrationAttempt < _reregistrationCooldown)
        {
            var remainingCooldown = _reregistrationCooldown - (DateTime.UtcNow - _lastReregistrationAttempt);
            _logger.LogWarning(
                "Re-registration attempted too recently. Cooldown remaining: {Remaining:N0}s",
                remainingCooldown.TotalSeconds);
            return false;
        }

        // Check max attempts
        if (_reregistrationAttempts >= MaxReregistrationAttempts)
        {
            _logger.LogError(
                "❌ Maximum re-registration attempts ({Max}) reached. " +
                "Manual node restart may be required. " +
                "Check orchestrator connectivity and authentication system.",
                MaxReregistrationAttempts);
            return false;
        }

        _reregistrationAttempts++;
        _lastReregistrationAttempt = DateTime.UtcNow;

        _logger.LogInformation(
            "🔄 Re-registration attempt {Attempt}/{Max} due to expired/invalid token",
            _reregistrationAttempts, MaxReregistrationAttempts);

        // Use stored registration request if available, otherwise build new one
        if (_lastRegistrationRequest != null)
        {
            var success = await RegisterNodeAsync(_lastRegistrationRequest, ct);

            if (success)
            {
                _logger.LogInformation("✅ Successfully re-registered after 401 error");
            }
            else
            {
                _logger.LogError("❌ Re-registration failed (attempt {Attempt}/{Max})",
                    _reregistrationAttempts, MaxReregistrationAttempts);
            }

            return success;
        }
        else
        {
            _logger.LogError(
                "❌ Cannot re-register: No previous registration request stored. " +
                "Node must restart to re-register.");
            return false;
        }
    }

    /// <summary>
    /// Build heartbeat payload from Heartbeat object
    /// </summary>
    private object BuildHeartbeatPayload(Heartbeat heartbeat)
    {
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
                sshPort = 2222, // Standard SSH port for VMs
                vncPort = v.VncPort,
                virtualCpuCores = v.VirtualCpuCores,
                qualityTier = v.QualityTier,
                computePointCost = v.ComputePointCost,
                memoryBytes = v.MemoryBytes,
                diskBytes = v.DiskBytes,
                startedAt = v.StartedAt.ToString("O") // StartedAt is non-nullable DateTime
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
        _logger.LogDebug("Processing heartbeat response: {ResponseContent}", content);
        try
        {
            var json = JsonDocument.Parse(content);

            if (!json.RootElement.TryGetProperty("data", out var data))
            {
                _logger.LogWarning("Heartbeat response missing 'data' property");
                throw new ArgumentException("Invalid heartbeat response format");
            }

            // =====================================================
            // Process Pending Commands
            // =====================================================
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

                    // Enqueue command for processing
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

            // =====================================================
            // Extract and Store CGNAT Info
            // =====================================================
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

            _lastHeartbeat = _lastHeartbeat; // Update last heartbeat
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
    /// Acknowledge command execution to orchestrator
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

            // Handle 401 on acknowledgment (less critical than heartbeat)
            if (response.StatusCode == HttpStatusCode.Unauthorized)
            {
                _logger.LogWarning(
                    "Command acknowledgment rejected with 401 - token may have expired. " +
                    "Will be re-sent after next successful heartbeat.");
                return false;
            }

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