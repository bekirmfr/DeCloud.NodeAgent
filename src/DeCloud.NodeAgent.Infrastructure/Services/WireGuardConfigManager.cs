using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Declarative WireGuard configuration manager
/// Reconciles actual state to match desired state with automatic cleanup
/// Uses INetworkManager for low-level operations
/// </summary>
public class WireGuardConfigManager : BackgroundService
{
    private readonly ILogger<WireGuardConfigManager> _logger;
    private readonly INetworkManager _networkManager;
    private readonly ICommandExecutor _executor;
    private readonly IOrchestratorClient _orchestratorClient;

    // Reconciliation interval
    private static readonly TimeSpan ReconcileInterval = TimeSpan.FromMinutes(1);

    // Handshake-liveness self-healing for CgnatClient role.
    // Mirrors wg-mesh-watchdog STALE_THRESHOLD=180s (used inside system VMs).
    // Persistent keepalive is 25s → 7 consecutive missed keepalives before we act.
    // Dashboard classifies "stale" at 180s, "dead" at 600s — we recover before "dead".
    // Relay-side eviction kicks in at 300s — we self-heal before the relay drops us.
    private static readonly TimeSpan HandshakeStaleThreshold = TimeSpan.FromSeconds(180);

    // Grace after interface creation: don't bounce a tunnel that hasn't had a chance
    // to handshake yet. Mirrors wg-mesh-watchdog's OnBootSec=2min precedent.
    private static readonly TimeSpan HandshakeColdStartGrace = TimeSpan.FromSeconds(120);

    // Tracks when each interface was last (re)created, used for cold-start grace.
    // In-memory only: on agent restart this effectively resets the grace from the
    // restart moment, which is the safer direction (avoids a false-positive bounce
    // immediately after the agent comes up while the interface is already healthy).
    private readonly Dictionary<string, DateTime> _interfaceCreatedAt = new();

    public WireGuardConfigManager(
        ILogger<WireGuardConfigManager> logger,
        INetworkManager networkManager,
        ICommandExecutor executor,
        IOrchestratorClient orchestratorClient)
    {
        _logger = logger;
        _networkManager = networkManager;
        _executor = executor;
        _orchestratorClient = orchestratorClient;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("WireGuard Configuration Manager starting");

        // Initial delay for node registration
        await Task.Delay(TimeSpan.FromSeconds(20), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await ReconcileWireGuardStateAsync(stoppingToken);
                await Task.Delay(ReconcileInterval, stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in WireGuard reconciliation");
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);
            }
        }
    }

    /// <summary>
    /// Reconcile actual WireGuard state to match desired state
    /// Kubernetes-style declarative configuration
    /// </summary>
    private async Task ReconcileWireGuardStateAsync(CancellationToken ct)
    {
        // ========================================
        // STEP 1: Determine Desired State
        // ========================================
        var desiredConfig = await DetermineDesiredConfigAsync(ct);

        if (desiredConfig == null)
        {
            _logger.LogDebug("No WireGuard configuration needed");

            // Clean up any existing interfaces
            await CleanupAllInterfacesAsync(ct);
            return;
        }

        _logger.LogInformation(
            "Desired WireGuard state: Interface={Interface}, Role={Role}",
            desiredConfig.InterfaceName,
            desiredConfig.Role);

        // ========================================
        // STEP 2: Get Actual State
        // ========================================
        var actualInterfaces = await GetActiveInterfacesAsync(ct);

        // ========================================
        // STEP 3: Remove Unwanted Interfaces
        // ========================================
        var unwantedInterfaces = actualInterfaces
            .Where(iface => iface != desiredConfig.InterfaceName)
            .ToList();

        foreach (var iface in unwantedInterfaces)
        {
            _logger.LogWarning(
                "Removing unwanted WireGuard interface: {Interface} " +
                "(desired: {Desired})",
                iface,
                desiredConfig.InterfaceName);

            await RemoveInterfaceAsync(iface, ct);
        }

        // ========================================
        // STEP 4: Create/Update Desired Interface
        // ========================================
        if (!actualInterfaces.Contains(desiredConfig.InterfaceName))
        {
            _logger.LogInformation(
                "Creating WireGuard interface: {Interface}",
                desiredConfig.InterfaceName);

            await CreateInterfaceAsync(desiredConfig, ct);
        }
        else
        {
            // Interface exists - verify configuration matches
            if (!await VerifyConfigurationAsync(desiredConfig, ct))
            {
                _logger.LogInformation(
                    "Configuration drift detected, recreating interface: {Interface}",
                    desiredConfig.InterfaceName);

                await RemoveInterfaceAsync(desiredConfig.InterfaceName, ct);
                await CreateInterfaceAsync(desiredConfig, ct);
            }
            else if (await IsHandshakeStaleAsync(desiredConfig, ct))
            {
                // Data-plane drift: config is correct, interface is up, but the
                // tunnel hasn't handshook in a while. Typical cause on CGNAT hosts
                // is the upstream carrier-grade NAT mapping dying. A clean bounce
                // rebinds the local UDP socket and forces a fresh outbound mapping.
                // The reused /etc/wireguard/{iface}.conf preserves the private key,
                // so the relay's peer entry stays valid — only the local socket
                // changes.
                _logger.LogWarning(
                    "Handshake stale on {Interface} (>{Threshold}s) — " +
                    "bouncing interface to refresh upstream NAT mapping",
                    desiredConfig.InterfaceName,
                    (int)HandshakeStaleThreshold.TotalSeconds);

                await RemoveInterfaceAsync(desiredConfig.InterfaceName, ct);
                await CreateInterfaceAsync(desiredConfig, ct);
            }
        }

        _logger.LogInformation(
            "✓ WireGuard state reconciled: {Interface} is active",
            desiredConfig.InterfaceName);
    }

    /// <summary>
    /// Determine what WireGuard configuration is needed based on node role
    /// </summary>
    private async Task<WireGuardDesiredConfig?> DetermineDesiredConfigAsync(
        CancellationToken ct)
    {
        var heartbeat = _orchestratorClient.GetLastHeartbeat();

        if (heartbeat?.Heartbeat?.CgnatInfo != null)
        {
            // ========================================
            // CGNAT Node → Need relay tunnel
            // ========================================
            return new WireGuardDesiredConfig
            {
                InterfaceName = "wg-relay",
                Role = WireGuardRole.CgnatClient,
                Configuration = heartbeat.Heartbeat.CgnatInfo.WireGuardConfig ?? string.Empty,
                TunnelIp = heartbeat.Heartbeat.CgnatInfo.TunnelIp,
                RelayNodeId = heartbeat.Heartbeat.CgnatInfo.AssignedRelayNodeId
            };
        }

        // Check if this is a relay VM
        if (IsRelayVm())
        {
            // ========================================
            // Relay VM → Need server configuration
            // ========================================
            return new WireGuardDesiredConfig
            {
                InterfaceName = "wg-relay-server",
                Role = WireGuardRole.RelayServer,
                Configuration = await LoadRelayServerConfigAsync(ct),
                TunnelIp = "10.20.0.1" // Gateway IP for relay network
            };
        }

        // Check if hub mode is needed (peer-to-peer mesh)
        if (NeedsHubConfiguration())
        {
            // ========================================
            // Regular Node → Hub/mesh configuration
            // ========================================
            return new WireGuardDesiredConfig
            {
                InterfaceName = "wg-hub",
                Role = WireGuardRole.HubNode,
                Configuration = await LoadHubConfigAsync(ct),
                TunnelIp = "10.10.0.1" // Regular mesh network
            };
        }

        // No WireGuard needed
        return null;
    }

    /// <summary>
    /// Get list of active WireGuard interfaces
    /// </summary>
    private async Task<List<string>> GetActiveInterfacesAsync(CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync("wg", "show interfaces", ct);

        if (!result.Success || string.IsNullOrWhiteSpace(result.StandardOutput))
        {
            return new List<string>();
        }

        return result.StandardOutput
            .Split(new[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
            .ToList();
    }

    /// <summary>
    /// Remove a WireGuard interface completely
    /// </summary>
    private async Task RemoveInterfaceAsync(string interfaceName, CancellationToken ct)
    {
        try
        {
            // Check if interface actually exists first
            var checkResult = await _executor.ExecuteAsync("ip", $"link show {interfaceName}", ct);
            bool interfaceExists = checkResult.Success;

            // Stop systemd service if exists
            await _executor.ExecuteAsync(
                "systemctl",
                $"stop wg-quick@{interfaceName}",
                ct);

            await _executor.ExecuteAsync(
                "systemctl",
                $"disable wg-quick@{interfaceName}",
                ct);

            // Only attempt wg-quick down if config file exists
            var configPath = $"/etc/wireguard/{interfaceName}.conf";
            if (File.Exists(configPath))
            {
                // Remove interface using wg-quick
                await _executor.ExecuteAsync(
                    "wg-quick",
                    $"down {interfaceName}",
                    ct);
            }

            // Force remove via ip link only if interface exists
            if (interfaceExists)
            {
                await _executor.ExecuteAsync(
                    "ip",
                    $"link delete {interfaceName}",
                    ct);
            }

            // Remove config file (backup first)
            if (File.Exists(configPath))
            {
                var backupPath = $"{configPath}.removed-{DateTime.UtcNow:yyyyMMdd-HHmmss}";
                File.Move(configPath, backupPath);
                _logger.LogInformation("Backed up old config: {Path}", backupPath);
            }

            _logger.LogInformation("✓ Removed WireGuard interface: {Interface}", interfaceName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing interface: {Interface}", interfaceName);
        }
    }

    /// <summary>
    /// Create WireGuard interface from desired configuration
    /// Uses INetworkManager for the actual operations
    /// </summary>
    private async Task CreateInterfaceAsync(
        WireGuardDesiredConfig config,
        CancellationToken ct)
    {
        var configPath = $"/etc/wireguard/{config.InterfaceName}.conf";

        // Write configuration
        await File.WriteAllTextAsync(configPath, config.Configuration, ct);

        // Set permissions
        await _executor.ExecuteAsync("chmod", $"600 {configPath}", ct);

        // Start interface using INetworkManager
        var success = await _networkManager.StartWireGuardInterfaceAsync(
            config.InterfaceName,
            ct);

        if (!success)
        {
            throw new Exception($"Failed to start WireGuard interface: {config.InterfaceName}");
        }

        // Enable systemd service for persistence
        await _executor.ExecuteAsync(
            "systemctl",
            $"enable wg-quick@{config.InterfaceName}",
            ct);

        _interfaceCreatedAt[config.InterfaceName] = DateTime.UtcNow;

        _logger.LogInformation(
            "✓ Created WireGuard interface: {Interface} (Role: {Role}, IP: {TunnelIp})",
            config.InterfaceName,
            config.Role,
            config.TunnelIp);
    }

    /// <summary>
    /// Verify that interface configuration matches desired state
    /// </summary>
    private async Task<bool> VerifyConfigurationAsync(
        WireGuardDesiredConfig desired,
        CancellationToken ct)
    {
        var configPath = $"/etc/wireguard/{desired.InterfaceName}.conf";

        if (!File.Exists(configPath))
        {
            return false;
        }

        var actualConfig = await File.ReadAllTextAsync(configPath, ct);

        // Simple comparison - could be more sophisticated
        return actualConfig.Trim() == desired.Configuration.Trim();
    }

    /// <summary>
    /// CGNAT-client self-healing: the host has a single peer (the relay).
    /// If that peer's handshake is stale beyond <see cref="HandshakeStaleThreshold"/>
    /// the tunnel is broken — typically because the upstream CGNAT NAT mapping
    /// died (e.g. the host was offline long enough for the carrier to drop its
    /// UDP outbound mapping). A clean interface bounce rebinds the local UDP
    /// socket and forces a fresh outbound mapping. The reused
    /// /etc/wireguard/{iface}.conf preserves the private key, so the relay's
    /// peer entry stays valid.
    ///
    /// Scope: <see cref="WireGuardRole.CgnatClient"/> only. Multi-peer interfaces
    /// (RelayServer, HubNode) have peers that may legitimately be stale at any
    /// given moment — bouncing the whole interface to fix one would be
    /// disproportionate. They need per-peer healing, which is handled elsewhere
    /// (relay-api.py cleanup thread, orchestrator EnsurePeerRegisteredAsync).
    /// </summary>
    private async Task<bool> IsHandshakeStaleAsync(
        WireGuardDesiredConfig desired,
        CancellationToken ct)
    {
        if (desired.Role != WireGuardRole.CgnatClient)
        {
            return false;
        }

        // Cold-start grace: a freshly created interface has no handshake yet,
        // and that's expected, not stale.
        if (_interfaceCreatedAt.TryGetValue(desired.InterfaceName, out var createdAt) &&
            DateTime.UtcNow - createdAt < HandshakeColdStartGrace)
        {
            return false;
        }

        var result = await _executor.ExecuteAsync(
            "wg",
            $"show {desired.InterfaceName} latest-handshakes",
            ct);

        if (!result.Success || string.IsNullOrWhiteSpace(result.StandardOutput))
        {
            // Can't read handshake state → don't act. The interface-existence
            // check above already passed, so this is a transient wg-show failure,
            // not a missing interface.
            return false;
        }

        // Output format per peer: "<base64-pubkey>\t<unix-seconds>\n"
        // CgnatClient has exactly one peer (the relay); take the first line
        // defensively in case wg adds metadata in future versions.
        var firstLine = result.StandardOutput
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault()
            ?.Trim();

        if (string.IsNullOrEmpty(firstLine))
        {
            return false;
        }

        var parts = firstLine.Split('\t', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 2 || !long.TryParse(parts[1], out var lastUnix))
        {
            return false;
        }

        // lastUnix == 0 means "never handshook". If we're past the cold-start
        // grace window, the interface is up but has never reached the relay —
        // that's stale by any reasonable definition.
        if (lastUnix == 0)
        {
            return true;
        }

        var ageSeconds = DateTimeOffset.UtcNow.ToUnixTimeSeconds() - lastUnix;
        return ageSeconds > (long)HandshakeStaleThreshold.TotalSeconds;
    }

    /// <summary>
    /// The complete set of WireGuard interface names this service can create.
    /// Derived directly from DetermineDesiredConfigAsync return values.
    /// Only these interfaces are eligible for cleanup — any other interface
    /// on the host (e.g., orchestrator's wg-relay-client) is not our concern.
    /// </summary>
    private static readonly HashSet<string> ManagedInterfaceNames =
    [
        "wg-relay",         // WireGuardRole.CgnatClient
        "wg-relay-server",  // WireGuardRole.RelayServer
        "wg-hub",           // WireGuardRole.HubNode
    ];

    private async Task CleanupAllInterfacesAsync(CancellationToken ct)
    {
        var activeInterfaces = await GetActiveInterfacesAsync(ct);

        foreach (var iface in activeInterfaces)
        {
            if (!ManagedInterfaceNames.Contains(iface))
            {
                _logger.LogDebug(
                    "Skipping non-managed WireGuard interface: {Interface}", iface);
                continue;
            }

            _logger.LogInformation(
                "No WireGuard needed - removing managed interface: {Interface}", iface);
            await RemoveInterfaceAsync(iface, ct);
        }
    }

    // ========================================
    // Helper Methods
    // ========================================

    private bool IsRelayVm()
    {
        // Check if this instance is a relay VM
        return Environment.GetEnvironmentVariable("DECLOUD_RELAY_VM") == "true" ||
               File.Exists("/etc/decloud/relay-vm.marker");
    }

    private bool NeedsHubConfiguration()
    {
        // Check if hub/mesh mode is needed
        return Environment.GetEnvironmentVariable("DECLOUD_HUB_MODE") == "true";
    }

    private async Task<string> LoadRelayServerConfigAsync(CancellationToken ct)
    {
        // Load relay server configuration
        var configPath = "/etc/wireguard/relay-server-template.conf";

        if (File.Exists(configPath))
        {
            return await File.ReadAllTextAsync(configPath, ct);
        }

        throw new InvalidOperationException("Relay server configuration not found");
    }

    private async Task<string> LoadHubConfigAsync(CancellationToken ct)
    {
        // Load hub configuration
        var configPath = "/etc/wireguard/hub-template.conf";

        if (File.Exists(configPath))
        {
            return await File.ReadAllTextAsync(configPath, ct);
        }

        // Return default hub configuration
        return @"[Interface]
PrivateKey = <GENERATED>
Address = 10.10.0.1/16
ListenPort = 51820

# Peers added dynamically
";
    }
}

/// <summary>
/// Desired WireGuard configuration state
/// </summary>
public class WireGuardDesiredConfig
{
    public string InterfaceName { get; set; } = string.Empty;
    public WireGuardRole Role { get; set; }
    public string Configuration { get; set; } = string.Empty;
    public string? TunnelIp { get; set; }
    public string? RelayNodeId { get; set; }
}

/// <summary>
/// WireGuard interface role
/// </summary>
public enum WireGuardRole
{
    /// <summary>
    /// Hub node in peer-to-peer mesh
    /// </summary>
    HubNode,

    /// <summary>
    /// CGNAT node connecting to relay
    /// </summary>
    CgnatClient,

    /// <summary>
    /// Relay VM serving CGNAT clients
    /// </summary>
    RelayServer
}