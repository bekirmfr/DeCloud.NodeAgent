// OrchestratorClient with Wallet Signature Authentication
// Stateless authentication - no tokens, no expiration, no re-registration!

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Interfaces.SystemVm;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Services;
using DeCloud.Shared.Contracts;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Options;
using Orchestrator.Models;
using System.Collections.Concurrent;
using System.Net;
using System.Reflection;
using System.Text;
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
    private readonly IVmManager _vmManager;
    private readonly IObligationStateService _obligationState;
    private readonly ISystemVmService _systemVmService;
    private readonly IArtifactCacheService _artifactCache;
    private readonly INodeMetadataService _nodeMetadata;
    private readonly OrchestratorClientOptions _options;

    private string? _nodeId;
    private string? _apiKey;
    private string? _orchestratorPublicKey;
    private string? _walletAddress;
    private HeartbeatDto? _lastHeartbeat = null;
    private const string SshCaPublicKeyPath = "/etc/ssh/decloud_ca.pub";
    private const int SystemVmDashboardPort = 8080;

    // Queue for pending commands received from heartbeat responses
    // Shared singleton with CommandProcessorService via DI
    private readonly ConcurrentQueue<PendingCommand> _pendingCommands;

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
        IVmManager vmManager,
        IObligationStateService obligationState,
        ISystemVmService systemVmService,
        IArtifactCacheService artifactCache,
        INodeMetadataService nodeMetadata,
        ConcurrentQueue<PendingCommand> pendingCommands,
        ILogger<OrchestratorClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
        _options = options.Value;
        _resourceDiscovery = resourceDiscovery;
        _nodeMetadata = nodeMetadata;
        // Wallet set in InitializeAsync from NodeMetadataService (settings.json)
        _nodeState = nodeState;
        _vmManager = vmManager;
        _obligationState = obligationState;
        _systemVmService = systemVmService;
        _artifactCache = artifactCache;
        _pendingCommands = pendingCommands;
        _httpClient.BaseAddress = new Uri(_options.BaseUrl.TrimEnd('/'));
        _httpClient.Timeout = _options.Timeout;

        _logger.LogInformation("OrchestratorClient initialized with wallet: {Wallet}",
            _walletAddress);
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        // Wallet and node ID from operator settings (settings.json) via
        // NodeMetadataService — single source of truth for identity.
        _walletAddress = _nodeMetadata.WalletAddress;
        _nodeId = _nodeMetadata.NodeId;

        await LoadCredentialsAsync();

        var isInternetReachable = await IsInternetReachableAsync(ct);
        _nodeState.SetInternetReachable(isInternetReachable);

        var isOrchestratorReachable = await IsOrchestratorReachableAsync(ct);
        _nodeState.SetOrchestratorReachable(isOrchestratorReachable);

        // Pre-fetch performance evaluation and scheduling config
        // Node state is updated within these methods
        await GetPerformanceEvaluationAsync(ct);
        await GetSchedulingConfigAsync(ct);

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
            // API_KEY is the only true credential — issued by orchestrator
            // during registration. Wallet and NodeId come from
            // NodeMetadataService (settings.json + machineId).
            if (line.StartsWith("API_KEY="))
                _apiKey = line.Split('=', 2)[1];
        }

        if (string.IsNullOrWhiteSpace(_apiKey))
        {
            _logger.LogWarning("No API key in credentials file - node not registered");
            return;
        }

        _httpClient.DefaultRequestHeaders.Authorization =
            new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _apiKey);

        _logger.LogInformation("✓ API key loaded from {File}", credentialsFilePath);
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

        var content = $@"API_KEY={_apiKey}
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

        // Read the SSH CA public key — P1.9. Used by orchestrator to substitute
        // __CA_PUBLIC_KEY__ in tenant cloud-init at render time.
        var sshCaPublicKey = await ReadSshCaPublicKeyAsync(ct);

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
            Country = _nodeMetadata.Country,
            Pricing = _nodeMetadata.Pricing,
            RegisteredAt = DateTime.UtcNow,
            SshCaPublicKey = sshCaPublicKey,
            // ObligationStateVersions and SystemTemplateVersions are not used
            // during registration — obligations are delivered via the evaluate
            // endpoint, and version-aware delivery uses the heartbeat flow.
            AllocatedResources = BuildAllocatedResources(),
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
    /// Push resource allocation percentages to the orchestrator.
    /// Calls POST /api/nodes/{id}/allocate.
    /// </summary>
    public async Task<NodeAllocateResponse?> AllocateAsync(
        NodeAllocateRequest request,
        CancellationToken ct = default)
    {
        if (!_nodeState.IsAuthenticated || string.IsNullOrEmpty(_nodeId))
        {
            _logger.LogWarning("Cannot allocate — node not registered");
            return null;
        }

        try
        {
            var requestPath = $"/api/nodes/{_nodeId}/allocate";
            var json = JsonSerializer.Serialize(request, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(requestPath, content, ct);

            // Use HttpResponse<T> to unwrap the ApiResponse<T> envelope,
            // same pattern as RegisterNodeAsync and other orchestrator calls.
            var result = await HttpResponse<NodeAllocateResponse>.FromResponseAsync(response);

            if (result.IsSuccess && result.Data != null)
            {
                _logger.LogInformation(
                    "✓ Allocation updated: CPU={CpuPct:P0}, Mem={MemPct:P0}, " +
                    "Stor={StorPct:P0}, GPU={Gpu}",
                    result.Data.EffectiveCpuPercent,
                    result.Data.EffectiveMemoryPercent,
                    result.Data.EffectiveStoragePercent,
                    result.Data.GpuCount?.ToString() ?? "all");

                await _nodeMetadata.UpdateFromOrchestratorResolutionAsync(result.Data, ct);

                return result.Data;
            }

            _logger.LogWarning("Allocation failed: {Error}", result.Error);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to call allocate endpoint");
            return null;
        }
    }

    /// <summary>
    /// Build the AllocatedResources object for registration.
    /// Prefers v2 (percentage) format; falls back to v1 (absolute) for
    /// backward compatibility with old settings that use absolute modes.
    /// </summary>
    private AllocatedResources? BuildAllocatedResources()
    {
        if (!_nodeMetadata.AllocatedComputePointsPercent.HasValue &&
            !_nodeMetadata.AllocatedMemoryBytes.HasValue &&
            !_nodeMetadata.AllocatedStorageBytes.HasValue &&
            !_nodeMetadata.AllocatedGpuCount.HasValue)
        {
            return null; // No allocation configured — orchestrator applies defaults
        }

        var inventory = _nodeMetadata.Inventory;

        double? cpuPct = _nodeMetadata.AllocatedComputePointsPercent.HasValue
            ? _nodeMetadata.AllocatedComputePointsPercent.Value / 100.0
            : null;

        // Prefer raw percent values (no lossy round-trip through bytes)
        double? memPct = _nodeMetadata.AllocatedMemoryPercent.HasValue
            ? _nodeMetadata.AllocatedMemoryPercent.Value / 100.0
            : _nodeMetadata.AllocatedMemoryBytes.HasValue
                && inventory?.Memory != null
                && inventory.Memory.TotalBytes > 0
                ? Math.Clamp(
                    (double)_nodeMetadata.AllocatedMemoryBytes.Value / inventory.Memory.TotalBytes,
                    AllocatedResources.MinPercent,
                    AllocatedResources.MaxPercent)
                : null;

        double? storPct = _nodeMetadata.AllocatedStoragePercent.HasValue
            ? _nodeMetadata.AllocatedStoragePercent.Value / 100.0
            : _nodeMetadata.AllocatedStorageBytes.HasValue && inventory?.Storage != null
                ? Math.Clamp(
                    (double)_nodeMetadata.AllocatedStorageBytes.Value
                        / Math.Max(1, inventory.Storage.Sum(s => s.TotalBytes)),
                    AllocatedResources.MinPercent,
                    AllocatedResources.MaxPercent)
                : null;

        return new AllocatedResources
        {
            SchemaVersion = AllocatedResources.CurrentSchemaVersion,
            CpuPercent = cpuPct,
            MemoryPercent = memPct,
            StoragePercent = storPct,
            GpuCount = _nodeMetadata.AllocatedGpuCount
        };
    }



    public async Task<bool> LoginAsync(CancellationToken ct = default)
    {
        if (!_nodeState.IsAuthenticated || string.IsNullOrEmpty(_nodeId))
        {
            _logger.LogWarning("Cannot login — node not registered");
            return false;
        }

        try
        {
            var requestPath = $"/api/nodes/{_nodeId}/login";
            var response = await _httpClient.PostAsync(requestPath, null, ct);

            if (response.IsSuccessStatusCode)
            {
                _logger.LogInformation("✓ Login successful — scheduling-ready");
                return true;
            }

            var error = await response.Content.ReadAsStringAsync(ct);
            _logger.LogWarning("Login failed ({Status}): {Error}",
                response.StatusCode, error);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to call login endpoint");
            return false;
        }
    }

    public async Task<bool> LogoutAsync(CancellationToken ct = default)
    {
        if (!_nodeState.IsAuthenticated || string.IsNullOrEmpty(_nodeId))
        {
            _logger.LogWarning("Cannot logout — node not registered");
            return false;
        }

        try
        {
            var requestPath = $"/api/nodes/{_nodeId}/logout";
            var response = await _httpClient.PostAsync(requestPath, null, ct);

            if (response.IsSuccessStatusCode)
            {
                _logger.LogInformation("✓ Logout successful — scheduling paused");
                return true;
            }

            var error = await response.Content.ReadAsStringAsync(ct);
            _logger.LogWarning("Logout failed ({Status}): {Error}",
                response.StatusCode, error);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to call logout endpoint");
            return false;
        }
    }

    /// <summary>
    /// Builds a dictionary of { roleName → storedVersion } from the local SQLite
    /// store for inclusion in the registration request.
    /// Allows the orchestrator to skip sending states the node already has.
    /// </summary>
    private async Task<Dictionary<string, int>> BuildObligationStateVersionsAsync(
        CancellationToken ct)
    {
        var roles = new[] { "relay", "dht", "blockstore" };
        var versions = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var role in roles)
        {
            try
            {
                var version = await _obligationState.GetVersionAsync(role, ct);
                if (version > 0)
                    versions[role] = version;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Could not read stored version for obligation role '{Role}' — " +
                    "orchestrator will resend state for this role",
                    role);
            }
        }

        _logger.LogDebug(
            "Reporting obligation state versions: {@Versions}", versions);

        return versions;
    }

    /// <summary>
    /// Deserialise the system template JSON and prefetch all artifacts
    /// appropriate for this node's architecture.
    /// Errors are logged but do not fail registration.
    /// </summary>
    private async Task TriggerArtifactPrefetchAsync(
        string role, string templateJson, CancellationToken ct)
    {
        try
        {
            var template = System.Text.Json.JsonSerializer.Deserialize<SystemVmTemplate>(
                templateJson,
                new System.Text.Json.JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

            if (template?.Artifacts is not { Count: > 0 })
            {
                _logger.LogDebug("System template [{Role}]: no artifacts to prefetch", role);
                return;
            }

            var arch = ResourceDiscoveryService.GetArchitectureNormalised();

            _logger.LogInformation(
                "System template [{Role}]: prefetching {Count} artifact(s) for {Arch}",
                role, template.Artifacts.Count, arch);

            await _artifactCache.PrefetchAsync(template.Artifacts, arch, ct);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "System template [{Role}]: artifact prefetch failed — " +
                "will retry when template is re-pulled on next heartbeat",
                role);
        }
    }

    /// <summary>
    /// Pulls the current identity state for <paramref name="role"/> from
    /// the orchestrator (GET /api/nodes/me/obligations/{role}/state) and
    /// persists it via IObligationStateService.SaveStateAsync.
    ///
    /// Called when the heartbeat response signals ObligationStatesPending.
    /// Safe to call redundantly — SaveStateAsync is version-guarded and
    /// idempotent, so a duplicate fetch is always a no-op.
    /// </summary>
    private async Task FetchAndSaveStateAsync(string role, CancellationToken ct)
    {
        try
        {
            _logger.LogInformation(
                "Fetching updated obligation state for role '{Role}' from orchestrator", role);

            var response = await _httpClient.GetAsync(
                $"/api/nodes/me/obligations/{role}/state", ct);

            if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                _logger.LogWarning(
                    "Orchestrator has no state for role '{Role}' — nothing to fetch", role);
                return;
            }

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning(
                    "FetchAndSaveStateAsync [{Role}]: orchestrator returned {Status}",
                    role, (int)response.StatusCode);
                return;
            }

            var stateJson = await response.Content.ReadAsStringAsync(ct);
            if (string.IsNullOrWhiteSpace(stateJson))
            {
                _logger.LogWarning(
                    "FetchAndSaveStateAsync [{Role}]: received empty state JSON", role);
                return;
            }

            // Parse the version from the state JSON without loading the full type
            // hierarchy — avoids a dependency on the concrete state subclass here.
            int version;
            try
            {
                using var doc = JsonDocument.Parse(stateJson);
                version = doc.RootElement.TryGetProperty("version", out var vProp)
                    ? vProp.GetInt32()
                    : 0;
            }
            catch
            {
                _logger.LogWarning(
                    "FetchAndSaveStateAsync [{Role}]: could not parse version from state JSON",
                    role);
                return;
            }

            if (version < 1)
            {
                _logger.LogWarning(
                    "FetchAndSaveStateAsync [{Role}]: state JSON has invalid version {V}",
                    role, version);
                return;
            }

            var written = await _obligationState.SaveStateAsync(role, stateJson, version, ct);

            // Log role + version only — never the state JSON body.
            _logger.LogInformation(
                written
                    ? "✓ Obligation state [{Role}] v{Version} fetched and persisted"
                    : "  Obligation state [{Role}] v{Version} already current — skipped",
                role, version);
        }
        catch (Exception ex)
        {
            // Non-fatal: the VM will retry on next heartbeat cycle.
            _logger.LogWarning(ex,
                "FetchAndSaveStateAsync [{Role}]: unexpected error — will retry next cycle", role);
        }
    }

    /// <summary>
    /// Pull the current system template for <paramref name="role"/> from the
    /// orchestrator (GET /api/nodes/me/system-templates/{role}) and persist
    /// via SaveSystemTemplateAsync + artifact prefetch.
    /// Parallel to FetchAndSaveStateAsync for identity state.
    /// </summary>
    private async Task FetchAndSaveSystemTemplateAsync(string role, CancellationToken ct)
    {
        try
        {
            _logger.LogInformation("Fetching updated system template for role '{Role}'", role);

            var response = await _httpClient.GetAsync(
                $"/api/nodes/me/system-templates/{role}", ct);

            if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                _logger.LogDebug("Orchestrator has no system template for '{Role}' yet", role);
                return;
            }

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("FetchAndSaveSystemTemplateAsync [{Role}]: {Status}",
                    role, (int)response.StatusCode);
                return;
            }

            var json = await response.Content.ReadAsStringAsync(ct);
            var payload = System.Text.Json.JsonSerializer.Deserialize<SystemVmTemplatePayload>(
                json,
                new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            if (payload is null || string.IsNullOrWhiteSpace(payload.TemplateJson))
            {
                _logger.LogWarning("FetchAndSaveSystemTemplateAsync [{Role}]: empty payload", role);
                return;
            }

            var written = await _obligationState.SaveSystemTemplateAsync(
                role,
                payload.TemplateJson,
                payload.Revision,
                string.IsNullOrEmpty(payload.TemplateId) ? null : payload.TemplateId,
                ct);

            if (written)
            {
                _logger.LogInformation("✓ System template [{Role}] r{Revision} updated", role, payload.Revision);
                await TriggerArtifactPrefetchAsync(role, payload.TemplateJson, ct);
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "FetchAndSaveSystemTemplateAsync [{Role}]: failed", role);
        }
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
    /// Reads the SSH certificate authority public key from disk for inclusion
    /// in the registration payload (P1.9). The key lives on every DeCloud node
    /// and is used to sign per-VM SSH host certificates.
    ///
    /// <para>
    /// Failure modes are non-fatal — registration continues with a null value
    /// and tenant deploys that need the CA key fail loudly at orchestrator-side
    /// render time, with a clear message pointing at the node. This keeps
    /// node-side bootstrap robust against transient FS issues; the operator
    /// fixes the file and the next registration cycle picks it up.
    /// </para>
    /// </summary>
    private async Task<string?> ReadSshCaPublicKeyAsync(CancellationToken ct)
    {
        try
        {
            if (!File.Exists(SshCaPublicKeyPath))
            {
                _logger.LogWarning(
                    "SSH CA public key not found at {Path} — registering without it. " +
                    "Tenant cloud-init templates that reference __CA_PUBLIC_KEY__ will " +
                    "fail at orchestrator-side render time until this file is created " +
                    "and the agent re-registers.",
                    SshCaPublicKeyPath);
                return null;
            }

            var content = await File.ReadAllTextAsync(SshCaPublicKeyPath, ct);
            var trimmed = content.Trim();

            if (string.IsNullOrEmpty(trimmed))
            {
                _logger.LogWarning(
                    "SSH CA public key file at {Path} is empty — registering without it.",
                    SshCaPublicKeyPath);
                return null;
            }

            _logger.LogInformation(
                "✓ SSH CA public key read from {Path} ({Length} bytes)",
                SshCaPublicKeyPath, trimmed.Length);

            return trimmed;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Failed to read SSH CA public key from {Path} — registering without it.",
                SshCaPublicKeyPath);
            return null;
        }
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

                    // Registration is now identity-only. Evaluation, obligations,
                    // and system templates are delivered by the evaluate endpoint.
                    // Log non-compliant VMs if present (re-registration).
                    if (registrationResponse.NonCompliantVms is { Count: > 0 } ncVms)
                    {
                        _logger.LogWarning(
                            "{Count} VM(s) flagged for migration due to locality change",
                            ncVms.Count);
                        foreach (var vm in ncVms)
                        {
                            _logger.LogWarning("  {VmId}: {Reason}", vm.VmId, vm.Reason);
                        }
                    }

                    _logger.LogInformation(
                        "✓ Node registered: {NodeId}. " +
                        "Run 'decloud evaluate' to benchmark and receive obligations.",
                        _nodeId);

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

            var payload = await BuildHeartbeatPayload(heartbeat, ct);

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
    /// <summary>
    /// Build heartbeat payload matching orchestrator's NodeHeartbeat model
    /// </summary>
    private async Task<object> BuildHeartbeatPayload(Heartbeat heartbeat, CancellationToken ct)
    {
        // Build payload matching orchestrator's NodeHeartbeat model exactly
        var payload = new 
        {
            nodeId = heartbeat.NodeId,
            metrics = new  // NodeMetrics structure
            {
                timestamp = heartbeat.Timestamp.ToString("O"),
                cpuUsagePercent = heartbeat.Resources.VirtualCpuUsagePercent,
                // Usage percent measures against physical RAM (host-level metric),
                // not the allocated ceiling (scheduling concept).
                memoryUsagePercent = heartbeat.Resources.TotalMemoryBytes > 0
                    ? (double)heartbeat.Resources.UsedMemoryBytes / heartbeat.Resources.TotalMemoryBytes * 100
                    : 0,
                storageUsagePercent = heartbeat.Resources.TotalStorageBytes > 0
                    ? (double)heartbeat.Resources.UsedStorageBytes / heartbeat.Resources.TotalStorageBytes * 100
                    : 0,
                networkInMbps = (double)0,
                networkOutMbps = (double)0,
                activeVmCount = heartbeat.ActiveVms.Count,
                loadAverage = (double)0
            },
            schedulingConfigVersion = heartbeat.SchedulingConfigVersion,
            activeVms = heartbeat.ActiveVms.Select(v => new
            {
                vmId = v.VmId,
                name = v.Name,
                state = v.State.ToString(),
                vmType = v.VmType.ToString(),
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
                gpuMode = v.GpuMode,
                gpuVramBytes = v.GpuVramBytes,
                imageId = v.Name,  // Use name as imageId for now
                startedAt = v.StartedAt.ToString("O"),
                services = v.Services?.Select(s => new
                {
                    name = s.Name,
                    port = s.Port,
                    protocol = s.Protocol,
                    status = s.Status,
                    statusMessage = s.StatusMessage,
                    readyAt = s.ReadyAt?.ToString("O")
                })
            }),
            cgnatInfo = heartbeat.CgnatInfo != null ? new
            {
                assignedRelayNodeId = heartbeat.CgnatInfo.AssignedRelayNodeId,
                tunnelIp = heartbeat.CgnatInfo.TunnelIp,
                wireGuardConfig = heartbeat.CgnatInfo.WireGuardConfig,
                publicEndpoint = heartbeat.CgnatInfo.PublicEndpoint,
                tunnelStatus = (int)heartbeat.CgnatInfo.TunnelStatus,  // Send as int
                lastHandshake = heartbeat.CgnatInfo.LastHandshake?.ToString("O")
            } : null,
            obligationStateVersions = await BuildObligationStateVersionsAsync(ct),
            obligationHealth = heartbeat.ObligationHealth,
            systemTemplateVersions = await _systemVmService.GetTemplateRevisionsAsync(ct),
            systemBinaryVersions = await _systemVmService.GetAllBinaryVersionsAsync(ct),
            agentVersion = _nodeState.AgentVersion,
            settingsHash = heartbeat.SettingsHash,
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
                    var command = ParseCommand(cmd);
                    if (command != null)
                    {
                        _pendingCommands.Enqueue(command);
                        _logger.LogDebug("Queued command {CommandId} from heartbeat", command.CommandId);
                    }
                }
            }

            // Process Scheduling Config Update
            if (data.TryGetProperty("schedulingConfig", out var schedulingConfigProp))
            {
                var schedulingConfig = JsonSerializer.Deserialize<SchedulingConfig>(
                    schedulingConfigProp.GetRawText(),
                    Core.Json.JsonOptions.Wire);
                if (schedulingConfig != null)
                {
                    _logger.LogInformation("Received updated scheduling configuration version {Version}",
                        schedulingConfig.Version);
                    _nodeState.UpdateSchedulingConfig(schedulingConfig);
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
            // Process InvalidVmIds — destroy VMs the orchestrator says
            // this node should not be running
            if (data.TryGetProperty("invalidVmIds", out var invalidVmIdsEl) &&
                invalidVmIdsEl.ValueKind == JsonValueKind.Array)
            {
                foreach (var idEl in invalidVmIdsEl.EnumerateArray())
                {
                    var vmId = idEl.GetString();
                    if (!string.IsNullOrEmpty(vmId))
                    {
                        _logger.LogWarning(
                            "Orchestrator flagged VM {VmId} as invalid — queuing for destruction",
                            vmId);
                        _pendingCommands.Enqueue(new PendingCommand
                        {
                            CommandId = Guid.NewGuid().ToString(),
                            Type = CommandType.DeleteVm,
                            Payload = $"{{\"vmId\":\"{vmId}\"}}",
                            RequiresAck = false,
                            IssuedAt = DateTime.UtcNow
                        });
                    }
                }
            }

            // Process settings drift warning
            if (data.TryGetProperty("settingsDrift", out var driftElement) &&
                driftElement.ValueKind == JsonValueKind.Object)
            {
                var driftMessage = driftElement.TryGetProperty("message", out var msg)
                    ? msg.GetString() : "Settings drift detected";

                _logger.LogWarning(
                    "⚠ Settings drift detected by orchestrator: {Message}. " +
                    "Run 'decloud register' to re-commit settings, or revert local edits.",
                    driftMessage);
            }

            // ── Obligation state pull ──────────────────────────────────────────
            // If the orchestrator signals that one or more obligation states are
            // newer than what we have locally, fetch and persist each one.
            if (data.TryGetProperty("obligationStatesPending", out var pendingProp)
                && pendingProp.ValueKind == JsonValueKind.Array)
            {
                foreach (var roleProp in pendingProp.EnumerateArray())
                {
                    var role = roleProp.GetString();
                    if (!string.IsNullOrWhiteSpace(role))
                        await FetchAndSaveStateAsync(role, ct);
                }
            }

            // Pull system templates signalled as stale by the orchestrator.
            if (data.TryGetProperty("systemTemplatesPending", out var pendingTemplates) &&
                pendingTemplates.ValueKind == JsonValueKind.Array)
            {
                foreach (var roleEl in pendingTemplates.EnumerateArray())
                {
                    var role = roleEl.GetString();
                    if (!string.IsNullOrEmpty(role))
                    {
                        _ = Task.Run(() => FetchAndSaveSystemTemplateAsync(role, ct), ct);
                    }
                }
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
    /// Fetch pending commands from orchestrator (dedicated endpoint for hybrid push-pull)
    /// Unlike GetPendingCommandsAsync which returns queued commands from heartbeat,
    /// this makes a direct API call to GET /api/nodes/{nodeId}/commands
    /// </summary>
    public async Task<List<PendingCommand>> FetchPendingCommandsAsync(CancellationToken ct = default)
    {
        if (!_nodeState.IsAuthenticated || string.IsNullOrEmpty(_nodeId))
        {
            _logger.LogWarning("Cannot fetch commands - node not registered");
            return new List<PendingCommand>();
        }

        try
        {
            var requestPath = $"/api/nodes/{_nodeId}/commands";

            var response = await _httpClient.GetAsync(requestPath, ct);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync(ct);
                _logger.LogWarning(
                    "Failed to fetch commands: {Status} - {Error}",
                    response.StatusCode, error);
                return new List<PendingCommand>();
            }

            var content = await response.Content.ReadAsStringAsync(ct);
            var json = JsonDocument.Parse(content);

            if (!json.RootElement.TryGetProperty("data", out var data))
            {
                return new List<PendingCommand>();
            }

            var commands = new List<PendingCommand>();

            if (data.ValueKind == JsonValueKind.Array)
            {
                foreach (var cmd in data.EnumerateArray())
                {
                    var command = ParseCommand(cmd);
                    if (command != null)
                    {
                        commands.Add(command);
                    }
                }
            }

            if (commands.Count > 0)
            {
                _logger.LogDebug(
                    "Fetched {Count} pending command(s) from orchestrator",
                    commands.Count);
            }

            return commands;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching pending commands");
            return new List<PendingCommand>();
        }
    }

    /// <summary>
    /// Parse command from JSON element (shared by heartbeat and fetch)
    /// </summary>
    private PendingCommand? ParseCommand(JsonElement cmd)
    {
        try
        {
            var commandId = cmd.GetProperty("commandId").GetString() ?? "";
            var typeProp = cmd.GetProperty("type");

            CommandType commandType;
            if (typeProp.ValueKind == JsonValueKind.Number)
            {
                commandType = (CommandType)typeProp.GetInt32();
            }
            else if (typeProp.ValueKind == JsonValueKind.String)
            {
                if (!Enum.TryParse<CommandType>(typeProp.GetString(), true, out commandType))
                {
                    _logger.LogWarning("Unknown command type: {Type}", typeProp.GetString());
                    return null;
                }
            }
            else
            {
                _logger.LogWarning("Unexpected command type format: {Kind}", typeProp.ValueKind);
                return null;
            }

            var payload = cmd.GetProperty("payload").GetString() ?? "";
            var requiresAck = cmd.TryGetProperty("requiresAck", out var ackProp)
                ? ackProp.GetBoolean()
                : true;

            return new PendingCommand
            {
                CommandId = commandId,
                Type = commandType,
                Payload = payload,
                RequiresAck = requiresAck,
                IssuedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to parse command");
            return null;
        }
    }

    /// <summary>
    /// Acknowledge command execution to orchestrator with wallet signature
    /// </summary>
    public async Task<bool> AcknowledgeCommandAsync(
        string commandId,
        bool success,
        string? errorMessage,
        string? data = null,
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
                completedAt = DateTime.UtcNow.ToString("O"),
                data
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
    /// Register an updated VM overlay manifest from a lazysync cycle.
    /// POST /api/blockstore/manifest
    /// </summary>
    public async Task<bool> RegisterManifestAsync(
        string vmId,
        string rootCid,
        int version,
        List<string> changedBlockCids,
        int blockCount,
        int blockSizeKb,
        long totalBytes,
        bool isSeeding = false,
        int replicationFactor = 3,
        Dictionary<long, string>? chunkMap = null,
        CancellationToken ct = default)
    {
        if (!_nodeState.IsAuthenticated) return false;

        try
        {
            var payload = new
            {
                vmId,
                nodeId = _nodeId,
                rootCid,
                version,
                changedBlockCids,
                blockCount,
                blockSizeKb,
                manifestType = 0,
                totalBytes,
                isSeeding,
                replicationFactor,  // ← pass through from LazysyncDaemon
                chunkMap
            };

            var response = await _httpClient.PostAsJsonAsync(
                "/api/blockstore/manifest", payload, ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning(
                    "RegisterManifest failed for VM {VmId}: HTTP {Status}",
                    vmId, response.StatusCode);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "RegisterManifest error for VM {VmId}", vmId);
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
    // Internet Connectivity
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