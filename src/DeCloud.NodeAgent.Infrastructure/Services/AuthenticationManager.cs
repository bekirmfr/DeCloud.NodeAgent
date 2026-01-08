using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Text.Json;

public class AuthenticationManager : BackgroundService
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly ILogger<AuthenticationManager> _logger;

    // File paths
    private const string CredentialsFile = "/etc/decloud/credentials";
    private const string PendingAuthFile = "/etc/decloud/pending-auth";

    // Polling intervals
    private static readonly TimeSpan DiscoveryCheckInterval = TimeSpan.FromSeconds(5);
    private static readonly TimeSpan AuthCheckInterval = TimeSpan.FromSeconds(10);

    public enum AuthState
    {
        WaitingForDiscovery,
        NotAuthenticated,
        PendingRegistration,
        Registered,
        CredentialsInvalid
    }

    public AuthenticationManager(
        IResourceDiscoveryService resourceDiscovery,
        IOrchestratorClient orchestratorClient,
        ILogger<AuthenticationManager> logger)
    {
        _resourceDiscovery = resourceDiscovery;
        _orchestratorClient = orchestratorClient;
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

                switch (state)
                {
                    case AuthState.NotAuthenticated:
                        await HandleNotAuthenticatedAsync(ct);
                        break;

                    case AuthState.PendingRegistration:
                        await HandlePendingRegistrationAsync(ct);
                        break;

                    case AuthState.Registered:
                        _logger.LogInformation("✓ Node authenticated and registered");
                        return; // Exit - normal operation can proceed

                    case AuthState.CredentialsInvalid:
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
            var inventory = await _resourceDiscovery.GetCachedInventoryAsync(ct);

            return inventory != null &&
                   inventory.Cpu.BenchmarkScore > 0 &&
                   inventory.Memory.TotalBytes > 0;
        }
        catch
        {
            return false;
        }
    }

    private async Task<AuthState> DetermineAuthStateAsync(CancellationToken ct)
    {
        // Check if credentials exist and are valid
        if (File.Exists(CredentialsFile))
        {
            var credentials = await LoadCredentialsAsync(ct);

            if (await ValidateCredentialsAsync(credentials, ct))
            {
                return AuthState.Registered;
            }
            else
            {
                return AuthState.CredentialsInvalid;
            }
        }

        // Check if pending authentication exists
        if (File.Exists(PendingAuthFile))
        {
            return AuthState.PendingRegistration;
        }

        return AuthState.NotAuthenticated;
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

        // Optional: Ping orchestrator to verify API key is valid
        // (Can add later if needed)

        return true;
    }

    private bool _hasLoggedAuthWarning = false;
}