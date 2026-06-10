using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace DeCloud.NodeAgent.Infrastructure.Services.Auth;

public class AuthenticationManager : BackgroundService
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly INodeStateService _nodeState;
    private readonly ILogger<AuthenticationManager> _logger;

    // File paths
    private const string CredentialsFilePath = "/etc/decloud/credentials";
    private const string PendingAuthFile = "/etc/decloud/pending-auth";
    private const string LoggedOutSentinelPath = "/etc/decloud/logged-out";

    // Polling intervals
    private static readonly TimeSpan DiscoveryCheckInterval = TimeSpan.FromSeconds(5);
    private static readonly TimeSpan AuthCheckInterval = TimeSpan.FromSeconds(10);

    // When the node is registered and scheduling-ready, we only need to watch
    // for a new pending-auth (i.e. the operator ran 'decloud register' again).
    // Checking every 30s is sufficient and avoids hammering the orchestrator.
    private static readonly TimeSpan RegisteredCheckInterval = TimeSpan.FromSeconds(30);

    private readonly INodeMetadataService _nodeMetadata;

    private bool _hasLoggedAuthWarning = false;

    public AuthenticationManager(
        IResourceDiscoveryService resourceDiscovery,
        IOrchestratorClient orchestratorClient,
        INodeStateService state,
        INodeMetadataService nodeMetadata,
        ILogger<AuthenticationManager> logger)
    {
        _resourceDiscovery = resourceDiscovery;
        _orchestratorClient = orchestratorClient;
        _nodeState = state;
        _nodeMetadata = nodeMetadata;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        _logger.LogInformation("AuthenticationManager starting...");

        try
        {
            // Phase 1: Wait for resource discovery to complete
            await WaitForResourceDiscoveryAsync(ct);

            // Phase 2: Authentication lifecycle — loop forever so that
            // 'decloud register' (which writes pending-auth) is picked up
            // at any point during the agent's lifetime without a restart.
            while (!ct.IsCancellationRequested)
            {
                var state = await DetermineAuthStateAsync(ct);

                // Update shared state
                _nodeState.SetAuthState(state);

                // Reset the auth warning flag when we leave NotAuthenticated
                if (state != AuthenticationState.NotAuthenticated)
                    _hasLoggedAuthWarning = false;

                switch (state)
                {
                    case AuthenticationState.NotAuthenticated:
                        await HandleNotAuthenticatedAsync(ct);
                        break;

                    case AuthenticationState.PendingRegistration:
                        // Reset scheduling-ready cache — new registration
                        // means a new API key and unknown server-side state.
                        _nodeState.SetSchedulingReady(false);
                        await HandlePendingRegistrationAsync(ct);
                        break;

                    case AuthenticationState.Registered:
                        _logger.LogInformation("✓ Node authenticated and registered");

                        // Auto-login: set SchedulingReady at orchestrator unless
                        // the operator explicitly logged out (sentinel file present)
                        // or the orchestrator already shows the node as ready.
                        await AutoLoginIfNotLoggedOutAsync(ct);

                        // Use a longer interval while registered — we only need
                        // to detect a new pending-auth from 'decloud register'.
                        await Task.Delay(RegisteredCheckInterval, ct);
                        continue; // skip the standard AuthCheckInterval below

                    case AuthenticationState.CredentialsInvalid:
                        _nodeState.SetSchedulingReady(false);
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
        _resourceDiscovery.GetInventoryCachedAsync(ct);

        _logger.LogInformation("Waiting for resource discovery to complete...");

        while (!ct.IsCancellationRequested)
        {
            if (_nodeState.IsDiscoveryComplete)
            {
                _logger.LogInformation("✓ Resource discovery complete");
                return;
            }

            await Task.Delay(DiscoveryCheckInterval, ct);
        }
    }

    private async Task<AuthenticationState> DetermineAuthStateAsync(CancellationToken ct)
    {
        // Pending-auth always takes priority — checked first (cheapest operation).
        // Logged at Debug because in steady state this is False every iteration
        // and the value of a healthy node — Information level produces noise that
        // obscures cadence gaps caused by real state transitions.
        var authFileExists = File.Exists(PendingAuthFile);
        _logger.LogDebug("Pending-auth file exists: {Exists}", authFileExists);

        if (authFileExists)
        {
            return AuthenticationState.PendingRegistration;
        }

        // Check if credentials exist and are valid.
        // We do NOT call the orchestrator here — credential revocation is
        // detected via 401 responses on heartbeat and command operations
        // (SendHeartbeatAsync, FetchPendingCommandsAsync, AcknowledgeCommandAsync),
        // which already run continuously and transition the state machine on 401.
        // A polling verify call here is redundant, slower, and conflates transport
        // failure with credential rejection — see commit message for full rationale.
        if (File.Exists(CredentialsFilePath))
        {
            var credentials = await LoadCredentialsAsync(ct);

            // Backfill derived identity fields — the credentials file
            // stores only API_KEY. NODE_ID and WALLET_ADDRESS are derived
            // from settings.json + machine-id by NodeMetadataService.
            if (!credentials.ContainsKey("NODE_ID"))
                credentials["NODE_ID"] = _nodeMetadata.NodeId;
            if (!credentials.ContainsKey("WALLET_ADDRESS"))
                credentials["WALLET_ADDRESS"] = _nodeMetadata.WalletAddress;

            if (await ValidateCredentialsAsync(credentials, ct))
                return AuthenticationState.Registered;

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
            _logger.LogWarning("  sudo decloud register");
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

                _nodeState.SetAuthState(AuthenticationState.Registered);

                // Signal OrchestratorClient to reload credentials
                await _orchestratorClient.ReloadCredentialsAsync(ct);
            }
            else
            {
                _logger.LogError("Registration failed: {Error}", result.Error);

                // Don't delete pending-auth — allow retry
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
        _logger.LogError("  sudo decloud register");
        _logger.LogError("");

        // Wait longer for invalid credentials
        await Task.Delay(TimeSpan.FromMinutes(1), ct);
    }

    private async Task<Dictionary<string, string>> LoadCredentialsAsync(CancellationToken ct)
    {
        var credentials = new Dictionary<string, string>();

        var lines = await File.ReadAllLinesAsync(CredentialsFilePath, ct);
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
    /// Call the login endpoint to set SchedulingReady = true,
    /// unless the operator has explicitly logged out (sentinel file exists)
    /// or the orchestrator already shows the node as scheduling-ready.
    ///
    /// This enables unattended restart: a node that reboots comes back
    /// online and schedulable without operator intervention. A node that
    /// was deliberately logged out stays paused across restarts.
    /// </summary>
    private async Task AutoLoginIfNotLoggedOutAsync(CancellationToken ct)
    {
        if (File.Exists(LoggedOutSentinelPath))
        {
            _logger.LogInformation(
                "Logged-out sentinel exists at {Path} — heartbeating but not scheduling-ready. " +
                "Run 'decloud login' to resume scheduling.",
                LoggedOutSentinelPath);
            return;
        }

        // Skip login if the orchestrator already has this node as scheduling-ready.
        if (_nodeState.IsSchedulingReady)
        {
            _logger.LogInformation(
                "Node is already scheduling-ready — skipping auto-login");
            return;
        }

        // Evaluation must exist before login — login requires PerformanceEvaluation
        // to compute capacity. On a normal restart InitializeAsync already fetched
        // it from the orchestrator. If it is still null here the node has not yet
        // been evaluated (fresh registration) or the orchestrator was unreachable
        // during init. In both cases the operator must run 'decloud evaluate'
        // explicitly — auto-evaluating here would bypass the intended lifecycle
        // (register → evaluate → login) and cause obligations to deploy before
        // the operator has reviewed allocation.
        if (_nodeState.PerformanceEvaluation == null)
        {
            _logger.LogInformation(
                "No performance evaluation — run 'decloud evaluate' then 'decloud login' to go live.");
            return;
        }

        try
        {
            var success = await _orchestratorClient.LoginAsync(ct);

            if (success)
            {
                _nodeState.SetSchedulingReady(true);
                _logger.LogInformation("✓ Auto-login successful — node is scheduling-ready");
            }
            else
            {
                _logger.LogWarning(
                    "Auto-login failed — node is heartbeating but not scheduling-ready. " +
                    "Run 'decloud login' manually to resume scheduling.");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Auto-login failed — node is heartbeating but not scheduling-ready");
        }
    }
}