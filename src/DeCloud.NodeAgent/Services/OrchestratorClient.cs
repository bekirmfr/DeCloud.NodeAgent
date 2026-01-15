// OrchestratorClient with Wallet Signature Authentication
// Stateless authentication - no tokens, no expiration, no re-registration!

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;
using Orchestrator.Models;
using System.Collections.Concurrent;
using System.Net;
using System.Reflection;
using System.Text.Json;

namespace DeCloud.NodeAgent.Services;

public class OrchestratorClientOptions
{
    public string BaseUrl { get; set; } = "http://localhost:5000";
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
    public string? WalletAddress { get; set; }
    public int AgentPort { get; set; } = 5050;
    public string PendingAuthFile { get; set; } = "/etc/decloud/pending-auth";
}

public partial class OrchestratorClient : IOrchestratorClient
{
    private const string PendingAuthFile = "/etc/decloud/pending-auth";
    private readonly HttpClient _httpClient;
    private readonly ILogger<OrchestratorClient> _logger;
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly INodeStateService _nodeState;
    private readonly INodeMetadataService _nodeMetadata;
    private readonly OrchestratorClientOptions _options;

    private string? _nodeId;
    private string? _apiKey;
    private string? _orchestratorPublicKey;
    private string? _walletAddress;
    private HeartbeatDto? _lastHeartbeat = null;

    // Queue for pending commands received from heartbeat responses
    private readonly ConcurrentQueue<PendingCommand> _pendingCommands = new();

    private static readonly TimeSpan OrchestratorTimeout = TimeSpan.FromMinutes(2);
    private static readonly TimeSpan InternetCheckTimeout = TimeSpan.FromSeconds(10);
    private static readonly string[] PublicIpServices = new[]
    {
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://ifconfig.me/ip"
    };


    public string? NodeId => _nodeId;
    public string? WalletAddress => _walletAddress;

    public OrchestratorClient(
        HttpClient httpClient,
        IOptions<OrchestratorClientOptions> options,
        IResourceDiscoveryService resourceDiscovery,
        INodeStateService nodeState,
        INodeMetadataService nodeMetadata,
        ILogger<OrchestratorClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
        _options = options.Value;
        _resourceDiscovery = resourceDiscovery;
        _nodeMetadata = nodeMetadata;
        _walletAddress = _options.WalletAddress;
        _nodeState = nodeState;
        _httpClient.BaseAddress = new Uri(_options.BaseUrl.TrimEnd('/'));
        _httpClient.Timeout = _options.Timeout;

        _logger.LogInformation("OrchestratorClient initialized with wallet: {Wallet}",
            _walletAddress);
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        await LoadCredentialsAsync();
        var isInternetReachable = await IsInternetReachableAsync(ct);
        _nodeState.SetInternetReachable(isInternetReachable);
        var isOrchestratorReachable = await IsOrchestratorReachableAsync(ct);
        _nodeState.SetOrchestratorReachable(isOrchestratorReachable);

        _logger.LogInformation("OrchestratorClient initialized with wallet: {Wallet} \n" +
            "internet connection: {Internet}" +
            "orchestrator connection: {Orchestrator}",
            _walletAddress, isInternetReachable, isOrchestratorReachable);
    }
    private async Task LoadCredentialsAsync()
    {
        const string credentialsFilePath = "/etc/decloud/credentials";

        if (!File.Exists(credentialsFilePath))
        {
            _logger.LogWarning("No credentials file found - node not registered");
            return;
        }

        var lines = await File.ReadAllLinesAsync(credentialsFilePath);

        foreach (var line in lines)
        {
            if (line.StartsWith("NODE_ID="))
                _nodeId = line.Split('=')[1];
            else if (line.StartsWith("WALLET_ADDRESS="))
                _walletAddress = line.Split('=')[1];
            else if (line.StartsWith("API_KEY="))
                _apiKey = line.Split('=')[1];
        }


        if (string.IsNullOrWhiteSpace(_nodeId) ||
           string.IsNullOrWhiteSpace(_apiKey))
        {
            _logger.LogWarning("Incomplete credentials in file - node not registered");
            return;
        }

        if (!string.IsNullOrWhiteSpace(_apiKey))
        {
            // Set Authorization header
            _httpClient.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _apiKey);

            _logger.LogInformation("✓ API key loaded from {File}", credentialsFilePath);
        }
    }

    private async Task SaveCredentialsAsync(CancellationToken ct)
    {
        const string credentialsFile = "/etc/decloud/credentials";

        // first delete if old file exists
        if (File.Exists(credentialsFile))
        {
            File.Delete(credentialsFile);
        }

        File.Create(credentialsFile).Dispose();

        var content = $@"NODE_ID={_nodeId}
WALLET_ADDRESS={_walletAddress}
API_KEY={_apiKey}
REGISTERED_AT={DateTime.UtcNow:O}";
        await File.WriteAllTextAsync(credentialsFile, content, ct);

        _logger.LogInformation("✓ API key saved to {File}", credentialsFile);
    }

    /// <summary>
    /// Register node using pending authentication with retry logic
    /// Combines auth loading, registration, and credential saving
    /// </summary>
    public async Task<RegistrationResult> RegisterWithPendingAuthAsync(CancellationToken ct = default)
    {
        const int MaxRetries = 5;

        // Load pending auth
        var pendingAuth = await LoadPendingAuthAsync(ct);
        if (pendingAuth == null)
        {
            return RegistrationResult.Failure("No pending authentication found");
        }

        // Validate not expired (10 minutes)
        if (IsPendingAuthExpired(pendingAuth))
        {
            File.Delete(PendingAuthFile);
            return RegistrationResult.Failure("Authentication expired");
        }

        // Get hardware inventory
        var inventory = await _resourceDiscovery.GetInventoryCachedAsync(ct);
        if (inventory == null)
        {
            return RegistrationResult.Failure("Hardware inventory not ready");
        }

        _nodeMetadata.UpdateInventory(inventory);

        // Build registration request
        var registration = new NodeRegistration
        {
            MachineId = _nodeMetadata.MachineId,
            Name = _nodeMetadata.Name,
            WalletAddress = pendingAuth.WalletAddress,
            Signature = pendingAuth.Signature,
            Message = pendingAuth.Message,
            PublicIp = _nodeMetadata.PublicIp ?? "127.0.0.1",
            AgentPort = 5100,
            HardwareInventory = inventory,
            AgentVersion = GetAgentVersion(),
            SupportedImages = GetSupportedImages(),
            Region = _nodeMetadata.Region,
            Zone = _nodeMetadata.Zone,
            RegisteredAt = DateTime.UtcNow
        };

        // Register with retry logic
        for (int attempt = 1; attempt <= MaxRetries; attempt++)
        {
            try
            {
                _logger.LogInformation(
                    "Registration attempt {Attempt}/{Max}...",
                    attempt, MaxRetries);

                var result = await RegisterNodeAsync(registration, ct);

                if (result.IsSuccess)
                {
                    // Delete pending auth on success
                    File.Delete(PendingAuthFile);

                    _nodeId = result.NodeId;
                    _apiKey = result.ApiKey;

                    _logger.LogInformation(
                        "✓ Registration successful: {NodeId}",
                        _nodeId);
                    return result;
                }

                _logger.LogWarning(
                    "Registration attempt {Attempt} failed: {Error}",
                    attempt,
                    result.Error);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(
                    ex,
                    "Registration attempt {Attempt} error",
                    attempt);
            }

            // Exponential backoff
            if (attempt < MaxRetries)
            {
                var delay = TimeSpan.FromSeconds(10 * attempt);
                await Task.Delay(delay, ct);
            }
        }

        return RegistrationResult.Failure("Max retries exceeded");
    }

    /// <summary>
    /// Load pending authentication data
    /// </summary>
    private async Task<PendingAuth?> LoadPendingAuthAsync(CancellationToken ct)
    {
        if (!File.Exists(PendingAuthFile))
        {
            return null;
        }

        try
        {
            var json = await File.ReadAllTextAsync(PendingAuthFile, ct);
            var auth = JsonSerializer.Deserialize<PendingAuth>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            if (auth == null ||
                string.IsNullOrWhiteSpace(auth.WalletAddress) ||
                string.IsNullOrWhiteSpace(auth.Signature) ||
                string.IsNullOrWhiteSpace(auth.Message))
            {
                _logger.LogError("Invalid pending auth format: {FileJson}", json);
                return null;
            }

            return auth;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load pending auth");
            return null;
        }
    }

    /// <summary>
    /// Check if pending auth is expired (> 10 minutes old)
    /// </summary>
    private bool IsPendingAuthExpired(PendingAuth auth)
    {
        var age = DateTimeOffset.UtcNow.ToUnixTimeSeconds() - auth.Timestamp;
        return age > 600;
    }

    /// <summary>
    /// Reload credentials from file (call after registration completes)
    /// </summary>
    public async Task ReloadCredentialsAsync(CancellationToken ct = default)
    {
        await LoadCredentialsAsync();
    }

    /// <summary>
    /// Get agent version from assembly
    /// </summary>
    private string GetAgentVersion()
    {
        return Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion ?? "unknown";
    }

    /// <summary>
    /// Register the node with the orchestrator.
    /// No token returned - future requests authenticated via wallet signature!
    /// </summary>
    public async Task<RegistrationResult> RegisterNodeAsync(NodeRegistration request, CancellationToken ct = default)
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

                //_logger.LogDebug("Received registration response: {Json}", json.ToString());

                if (json.RootElement.TryGetProperty("data", out var data))
                {
                    var registrationResponse = JsonSerializer.Deserialize<NodeRegistrationResponse>(
                            data.GetRawText(),
                            new JsonSerializerOptions
                            {
                                PropertyNameCaseInsensitive = true
                            });

                    if (registrationResponse == null)
                    {
                        _logger.LogError("Failed to deserialize node registration response");
                        throw new ArgumentNullException(nameof(NodeRegistrationResponse));
                    }

                    _nodeId = registrationResponse.NodeId ?? throw new ArgumentNullException(nameof(registrationResponse.NodeId));
                    _apiKey = registrationResponse.ApiKey ?? throw new ArgumentNullException(nameof(registrationResponse.ApiKey));

                    await SaveCredentialsAsync(ct);

                    // Set Authorization header
                    _httpClient.DefaultRequestHeaders.Authorization =
                        new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _apiKey);

                    _logger.LogInformation($"✓ Node ID ({_nodeId}) and API key ({_apiKey}) received and authorization header is configured");

                    _orchestratorPublicKey = registrationResponse.OrchestratorWireGuardPublicKey ?? throw new ArgumentNullException(nameof(registrationResponse.OrchestratorWireGuardPublicKey));
                    await SaveOrchestratorPublicKeyAsync(_orchestratorPublicKey, ct);

                    var performanceEval = registrationResponse.PerformanceEvaluation ?? throw new ArgumentNullException(nameof(registrationResponse.PerformanceEvaluation));
                    _nodeMetadata.UpdatePerformanceEvaluation(performanceEval);
                    _logger.LogInformation($"✓ Performance evaluation received and updated with cpu benchmark score {performanceEval.BenchmarkScore}, total compute points {performanceEval.TotalComputePoints}");

                    var schedulingConfig = registrationResponse.SchedulingConfig ?? throw new ArgumentNullException(nameof(registrationResponse.SchedulingConfig));
                    _nodeMetadata.UpdateSchedulingConfig(schedulingConfig);
                    _logger.LogInformation($"✓ Scheduling configuration version {schedulingConfig.Version} received and updated");

                    _logger.LogInformation(
                        "✓ Node registered successfully: {NodeId} | Wallet: {Wallet}",
                        _nodeId, _walletAddress);

                    return RegistrationResult.Success(_nodeId, _apiKey);
                }
            }

            var errorContent = await response.Content.ReadAsStringAsync(ct);
            _logger.LogError("Registration failed: {Status} - {Content}", response.StatusCode, errorContent);
            return RegistrationResult.Failure($"Registration failed: {response.StatusCode} - {errorContent}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to register node");
            return RegistrationResult.Failure($"Failed to register node: {ex.Message}");
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
        if (!_nodeState.IsAuthenticated)
        {
            _logger.LogWarning("Cannot send heartbeat - node not registered");
            return false;
        }

        try
        {
            var requestPath = $"/api/nodes/{_nodeId}/heartbeat";

            // ✅ API key is already in DefaultRequestHeaders.Authorization

            var payload = BuildHeartbeatPayload(heartbeat);

            _lastHeartbeat = new HeartbeatDto
            {
                Heartbeat = heartbeat
            };

            var response = await _httpClient.PostAsJsonAsync(requestPath, payload, ct);

            _lastHeartbeat.Response = response;

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

            _nodeState.SetInternetReachable(true);
            _nodeState.SetOrchestratorReachable(true);

            await ProcessHeartbeatResponseAsync(content, ct);

            return true;
        }
        catch (Exception ex)
        {
            var internetReachable = await IsInternetReachableAsync(ct);
            _nodeState.SetInternetReachable(internetReachable);

            var orchestaratorReachable =  await IsOrchestratorReachableAsync(ct);
            _nodeState.SetOrchestratorReachable(orchestaratorReachable);

            _logger.LogError(ex, $"Failed to send heartbeat" + 
                "Internet reachable: {Internet}" +
                "Orchestrator reachable: {Orchestrator}",
                internetReachable, orchestaratorReachable);

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
            schedulingConfigVersion = heartbeat.SchedulingConfigVersion,
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
    private async Task ProcessHeartbeatResponseAsync(string content, CancellationToken ct)
    {
        _logger.LogDebug("Processing heartbeat response");
        try
        {
            var json = JsonDocument.Parse(content);

            if (!json.RootElement.TryGetProperty("data", out var data))
            {
                _logger.LogWarning("Heartbeat response missing 'data' property");
                throw new ArgumentException("Heartbeat response missing 'data' property");
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
            }

            // Process Scheduling Config Update
            if (data.TryGetProperty("schedulingConfig", out var schedulingConfigProg))
            {
                var schedulingConfig = JsonSerializer.Deserialize<SchedulingConfig>(
                    schedulingConfigProg.GetRawText(),
                    new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });
                if (schedulingConfig != null)
                {
                    _logger.LogInformation("Received updated scheduling configuration version {Version}",
                        schedulingConfig.Version);
                    _nodeMetadata.UpdateSchedulingConfig(schedulingConfig);
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
                    "Received relay data: Relay {RelayId}, Tunnel IP {TunnelIp}",
                    relayId, tunnelIp);

                if (_lastHeartbeat != null && _lastHeartbeat.Heartbeat != null)
                {
                    _lastHeartbeat.Heartbeat.CgnatInfo = new CgnatNodeInfo
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
            else if (_lastHeartbeat != null && _lastHeartbeat.Heartbeat != null && _lastHeartbeat.Heartbeat.CgnatInfo != null)
            {
                _logger.LogInformation("Clearing previous CGNAT info - node no longer behind NAT");
                _lastHeartbeat.Heartbeat.CgnatInfo = null;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to process heartbeat response");
        }
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
    /// Acknowledge command execution to orchestrator with wallet signature
    /// </summary>
    public async Task<bool> AcknowledgeCommandAsync(
        string commandId,
        bool success,
        string? errorMessage,
        CancellationToken ct = default)
    {
        if (!_nodeState.IsAuthenticated)
        {
            _logger.LogWarning("Cannot acknowledge command - node not registered");
            return false;
        }

        try
        {
            // =====================================================
            // Create Request with Wallet Signature Headers
            // =====================================================
            var requestPath = $"/api/nodes/{_nodeId}/commands/{commandId}/acknowledge";

            var request = new HttpRequestMessage(HttpMethod.Post, requestPath);

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
    /// Get supported VM images
    /// </summary>
    private List<string> GetSupportedImages()
    {
        return new List<string>
        {
            "ubuntu-22.04",
            "ubuntu-24.04",
            "debian-12"
        };
    }

    /// <summary>
    /// Get the last heartbeat object (for accessing CGNAT info, etc.)
    /// </summary>
    public HeartbeatDto? GetLastHeartbeat()
    {
        return _lastHeartbeat;
    }
    public record PendingAuth(
        string WalletAddress,
        string Signature,
        string Message,
        long Timestamp,
        string? MachineId = null
        );

    // ================================================================
    // METHODS - Internet Connectivity
    // ================================================================

    /// <summary>
    /// Check if the orchestrator is reachable by calling the health endpoint
    /// </summary>
    public async Task<bool> IsOrchestratorReachableAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/system/health", ct);

            if (response.IsSuccessStatusCode)
            {
                _logger.LogDebug("Orchestrator health check successful");
                return true;
            }

            _logger.LogWarning(
                "Orchestrator health check failed with status: {Status}",
                response.StatusCode);
            return false;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(
                ex,
                "Cannot reach orchestrator at {BaseUrl}",
                _options.BaseUrl);
            return false;
        }
        catch (TaskCanceledException ex)
        {
            _logger.LogWarning(
                ex,
                "Orchestrator health check timed out after {Timeout}",
                _options.Timeout);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Unexpected error during orchestrator health check");
            return false;
        }
    }

    /// <summary>
    /// Check if the node has internet connectivity by querying public IP services.
    /// Tries multiple services with failover for reliability.
    /// </summary>
    public async Task<bool> IsInternetReachableAsync(CancellationToken ct = default)
    {
        foreach (var service in PublicIpServices)
        {
            try
            {
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                cts.CancelAfter(InternetCheckTimeout);

                var response = await _httpClient.GetAsync(service, cts.Token);

                if (response.IsSuccessStatusCode)
                {
                    var ip = (await response.Content.ReadAsStringAsync(cts.Token)).Trim();

                    if (!string.IsNullOrWhiteSpace(ip) && IsValidIp(ip))
                    {
                        _logger.LogDebug(
                            "Internet connectivity confirmed via {Service}, public IP: {Ip}",
                            service, ip);
                        return true;
                    }
                }
            }
            catch (HttpRequestException ex)
            {
                _logger.LogDebug(
                    ex,
                    "Failed to reach {Service}, trying next service",
                    service);
            }
            catch (TaskCanceledException ex)
            {
                _logger.LogDebug(
                    ex,
                    "Timeout checking {Service}, trying next service",
                    service);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(
                    ex,
                    "Unexpected error checking {Service}",
                    service);
            }
        }

        _logger.LogWarning("No internet connectivity - all public IP services unreachable");
        return false;
    }

    private static bool IsValidIp(string ip)
    {
        return IPAddress.TryParse(ip, out _);
    }
}