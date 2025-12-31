using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Manages WireGuard tunnel to assigned relay node
/// </summary>
public class RelayTunnelService : BackgroundService
{
    private readonly INetworkManager _networkManager;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly ILogger<RelayTunnelService> _logger;

    // Tunnel health check interval
    private static readonly TimeSpan HealthCheckInterval = TimeSpan.FromSeconds(30);

    private string? _relayNodeId;
    private string? _tunnelIp;
    private bool _isConfigured;

    public RelayTunnelService(
        INetworkManager networkManager,
        IOrchestratorClient orchestratorClient,
        ILogger<RelayTunnelService> logger)
    {
        _networkManager = networkManager;
        _orchestratorClient = orchestratorClient;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Relay tunnel service starting");

        // Wait for initial registration
        await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await CheckAndConfigureTunnelAsync(stoppingToken);
                await Task.Delay(HealthCheckInterval, stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in relay tunnel service");
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            }
        }
    }

    private async Task CheckAndConfigureTunnelAsync(CancellationToken ct)
    {
        // Check if we need a relay (orchestrator tells us in heartbeat response)
        var heartbeat = _orchestratorClient.GetLastHeartbeat();

        if (heartbeat?.CgnatInfo == null)
        {
            // Not behind CGNAT or not yet configured
            if (_isConfigured)
            {
                _logger.LogInformation("No longer need relay tunnel");
                _isConfigured = false;
            }
            return;
        }

        var cgnatInfo = heartbeat.CgnatInfo;

        // Check if configuration changed
        if (_relayNodeId != cgnatInfo.AssignedRelayNodeId ||
            _tunnelIp != cgnatInfo.TunnelIp)
        {
            _logger.LogInformation(
                "Relay assignment changed: Relay={RelayId}, TunnelIP={TunnelIp}",
                cgnatInfo.AssignedRelayNodeId, cgnatInfo.TunnelIp);

            await ConfigureTunnelAsync(cgnatInfo, ct);
        }

        // Health check existing tunnel
        if (_isConfigured)
        {
            await CheckTunnelHealthAsync(ct);
        }
    }

    private async Task ConfigureTunnelAsync(CgnatNodeInfo cgnatInfo, CancellationToken ct)
    {
        _logger.LogInformation("Configuring WireGuard tunnel to relay {RelayId}", cgnatInfo.AssignedRelayNodeId);

        try
        {
            // Apply WireGuard configuration
            if (!string.IsNullOrEmpty(cgnatInfo.WireGuardConfig))
            {
                // Save config to file
                var configPath = "/etc/wireguard/wg-relay.conf";
                await File.WriteAllTextAsync(configPath, cgnatInfo.WireGuardConfig, ct);

                // Start WireGuard interface
                await _networkManager.StartWireGuardInterfaceAsync("wg-relay", ct);

                _relayNodeId = cgnatInfo.AssignedRelayNodeId;
                _tunnelIp = cgnatInfo.TunnelIp;
                _isConfigured = true;

                _logger.LogInformation(
                    "WireGuard tunnel configured successfully (Tunnel IP: {TunnelIp})",
                    _tunnelIp);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to configure WireGuard tunnel");
            _isConfigured = false;
        }
    }

    private async Task CheckTunnelHealthAsync(CancellationToken ct)
    {
        try
        {
            var peers = await _networkManager.GetPeersAsync(ct);
            var relayPeer = peers.FirstOrDefault();

            if (relayPeer != null && relayPeer.LastHandshake.HasValue)
            {
                var timeSinceHandshake = DateTime.UtcNow - relayPeer.LastHandshake.Value;

                if (timeSinceHandshake > TimeSpan.FromMinutes(2))
                {
                    _logger.LogWarning(
                        "Relay tunnel unhealthy: No handshake in {TimeSince}",
                        timeSinceHandshake);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to check tunnel health");
        }
    }
}