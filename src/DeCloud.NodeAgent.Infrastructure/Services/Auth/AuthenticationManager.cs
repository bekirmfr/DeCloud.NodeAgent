using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services.Auth;

public class AuthenticationManager : BackgroundService
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly INodeStateService _nodeState;
    private readonly ILogger<AuthenticationManager> _logger;

    // File paths
    private const string CredentialsFile = "/etc/decloud/credentials";
    private const string PendingAuthFile = "/etc/decloud/pending-auth";

    // Polling intervals
    private static readonly TimeSpan DiscoveryCheckInterval = TimeSpan.FromSeconds(5);
    private static readonly TimeSpan AuthCheckInterval = TimeSpan.FromSeconds(10);

    public AuthenticationManager(
        IResourceDiscoveryService resourceDiscovery,
        IOrchestratorClient orchestratorClient,
        INodeStateService state,
        ILogger<AuthenticationManager> logger)
    {
        _resourceDiscovery = resourceDiscovery;
        _orchestratorClient = orchestratorClient;
        _nodeState = state;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        _logger.LogInformation("AuthenticationManager starting...");

        try
        {
            // Phase 1: Wait for resource discovery to complete
            await WaitForResourceDiscoveryAsync(ct);

            // Phase 2: Authentication lifecycle
            while (!ct.IsCancellationRequested)
            {
                var state = await DetermineAuthStateAsync(ct);

                // Update shared state
                _nodeState.SetAuthState(state);

                switch (state)
                {
                    case AuthenticationState.NotAuthenticated:
                        await HandleNotAuthenticatedAsync(ct);
                        break;

                    case AuthenticationState.PendingRegistration:
                        await HandlePendingRegistrationAsync(ct);
                        break;

                    case AuthenticationState.Registered:
                        _logger.LogInformation("✓ Node authenticated and registered");
                        return; // Exit - normal operation can proceed

                    case AuthenticationState.CredentialsInvalid:
                        await HandleInvalidCredentialsAsync(ct);
                        break;
                }

                // Prevent tight loop
                await Task.Delay(AuthCheckInterval, ct);
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("AuthenticationManager stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "AuthenticationManager error");
            throw;
        }
    }

    private async Task WaitForResourceDiscoveryAsync(CancellationToken ct)
    {
        _logger.LogInformation("Waiting for resource discovery to complete...");

        while (!ct.IsCancellationRequested)
        {
            if (await IsResourceDiscoveryCompleteAsync(ct))
            {
                _logger.LogInformation("✓ Resource discovery complete");
                return;
            }

            await Task.Delay(DiscoveryCheckInterval, ct);
        }
    }

    private async Task<bool> IsResourceDiscoveryCompleteAsync(CancellationToken ct)
    {
        try
        {
            // Check if ResourceDiscoveryService has completed initial discovery
            var inventory = await _resourceDiscovery.GetInventoryCachedAsync(ct);

            return inventory != null &&
                   inventory.Cpu.BenchmarkScore > 0 &&
                   inventory.Memory.TotalBytes > 0;
        }
        catch
        {
            return false;
        }
    }

    private async Task<AuthenticationState> DetermineAuthStateAsync(CancellationToken ct)
    {
        var authFileExists = File.Exists(PendingAuthFile);

        _logger.LogInformation("Auth file exists: {AuthFileExists}", authFileExists);

        // Check if pending authentication exists
        if (authFileExists)
        {
            return AuthenticationState.PendingRegistration;
        }

        var credentials = new Dictionary<string, string>();

        // Check if credentials exist and are valid
        if (File.Exists(CredentialsFile))
        {
            credentials = await LoadCredentialsAsync(ct);

            if (await ValidateCredentialsAsync(credentials, ct))
            {
                var isAuthorized = await VerifyNodeAuthorizationAsync(
                    credentials["NODE_ID"],
                    credentials["API_KEY"], ct);

                if (isAuthorized)
                {
                    return AuthenticationState.Registered;
                }
            }

            return AuthenticationState.CredentialsInvalid;
        }

        return AuthenticationState.NotAuthenticated;
    }

    private async Task HandleNotAuthenticatedAsync(CancellationToken ct)
    {
        // Log once, then wait silently
        if (!_hasLoggedAuthWarning)
        {
            _logger.LogWarning("═══════════════════════════════════════");
            _logger.LogWarning("⚠  Node Not Authenticated");
            _logger.LogWarning("═══════════════════════════════════════");
            _logger.LogWarning("");
            _logger.LogWarning("To authenticate this node, run:");
            _logger.LogWarning("  sudo cli-decloud-node login");
            _logger.LogWarning("");
            _logger.LogWarning("The node agent is running and ready.");
            _logger.LogWarning("Waiting for authentication...");
            _logger.LogWarning("");

            _hasLoggedAuthWarning = true;
        }

        await Task.Delay(AuthCheckInterval, ct);
    }

    private async Task HandlePendingRegistrationAsync(CancellationToken ct)
    {
        _logger.LogInformation("Authentication detected, starting registration...");

        try
        {
            var result = await _orchestratorClient.RegisterWithPendingAuthAsync(ct);

            if (result.IsSuccess)
            {
                _logger.LogInformation("✓ Registration successful: {NodeId}", result.NodeId);

                // Signal OrchestratorClient to reload credentials
                await _orchestratorClient.ReloadCredentialsAsync(ct);
            }
            else
            {
                _logger.LogError("Registration failed: {Error}", result.Error);

                // Don't delete pending-auth - allow retry
                await Task.Delay(TimeSpan.FromSeconds(30), ct);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Registration error");
            await Task.Delay(TimeSpan.FromSeconds(30), ct);
        }
    }

    private async Task HandleInvalidCredentialsAsync(CancellationToken ct)
    {
        _logger.LogError("═══════════════════════════════════════");
        _logger.LogError("❌ Invalid Credentials Detected");
        _logger.LogError("═══════════════════════════════════════");
        _logger.LogError("");
        _logger.LogError("The credentials file is corrupted or invalid.");
        _logger.LogError("Please re-authenticate:");
        _logger.LogError("  sudo rm /etc/decloud/credentials");
        _logger.LogError("  sudo cli-decloud-node login");
        _logger.LogError("");

        // Wait longer for invalid credentials
        await Task.Delay(TimeSpan.FromMinutes(1), ct);
    }

    private async Task<Dictionary<string, string>> LoadCredentialsAsync(CancellationToken ct)
    {
        var credentials = new Dictionary<string, string>();

        var lines = await File.ReadAllLinesAsync(CredentialsFile, ct);
        foreach (var line in lines)
        {
            if (line.Contains('='))
            {
                var parts = line.Split('=', 2);
                credentials[parts[0].Trim()] = parts[1].Trim();
            }
        }

        return credentials;
    }

    private async Task<bool> ValidateCredentialsAsync(
        Dictionary<string, string> credentials,
        CancellationToken ct)
    {
        // Check required fields
        if (!credentials.ContainsKey("NODE_ID") ||
            !credentials.ContainsKey("API_KEY") ||
            !credentials.ContainsKey("WALLET_ADDRESS"))
        {
            return false;
        }

        // Validate format
        if (string.IsNullOrWhiteSpace(credentials["NODE_ID"]) ||
            string.IsNullOrWhiteSpace(credentials["API_KEY"]))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Verify node authorization by calling the orchestrator's GET /api/nodes/{nodeId} endpoint.
    /// Returns true if the node receives a 200 OK response, indicating valid credentials.
    /// </summary>
    private async Task<bool> VerifyNodeAuthorizationAsync(string nodeId, string apiKey, CancellationToken ct)
    {
        try
        {
            using var httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(10)
            };

            // Get orchestrator base URL from OrchestratorClient configuration
            var baseUrl = _orchestratorClient.GetType()
                .GetField("_httpClient", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                ?.GetValue(_orchestratorClient) as HttpClient;

            var requestUrl = baseUrl?.BaseAddress != null
                ? $"{baseUrl.BaseAddress.ToString().TrimEnd('/')}/api/nodes/{nodeId}"
                : $"http://localhost:5000/api/nodes/{nodeId}";

            var request = new HttpRequestMessage(HttpMethod.Get, requestUrl);
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey);

            var response = await httpClient.SendAsync(request, ct);

            if (response.StatusCode == System.Net.HttpStatusCode.OK)
            {
                _logger.LogInformation("✓ Node authorization verified with orchestrator");
                return true;
            }

            _logger.LogWarning(
                "Node authorization failed: {StatusCode} - {Reason}",
                (int)response.StatusCode,
                response.ReasonPhrase);

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to verify node authorization with orchestrator");
            return false;
        }
    }

    private bool _hasLoggedAuthWarning = false;
}