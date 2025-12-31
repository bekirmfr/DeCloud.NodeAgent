using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Diagnostics;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Automatically configures WireGuard tunnel when assigned to relay
/// </summary>
public class WireGuardAutoConfigService : BackgroundService
{
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly ILogger<WireGuardAutoConfigService> _logger;

    private string? _lastRelayId;
    private string? _lastTunnelIp;

    public WireGuardAutoConfigService(
        IOrchestratorClient orchestratorClient,
        ILogger<WireGuardAutoConfigService> logger)
    {
        _orchestratorClient = orchestratorClient;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("WireGuard auto-config service starting");

        // Wait for initial registration
        await Task.Delay(TimeSpan.FromSeconds(15), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await CheckAndConfigureAsync(stoppingToken);
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in WireGuard auto-config");
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);
            }
        }
    }

    private async Task CheckAndConfigureAsync(CancellationToken ct)
    {
        var heartbeat = _orchestratorClient.GetLastHeartbeat();

        if (heartbeat?.CgnatInfo == null)
        {
            return; // Not behind CGNAT
        }

        var cgnat = heartbeat.CgnatInfo;

        // Check if config changed
        if (_lastRelayId == cgnat.AssignedRelayNodeId &&
            _lastTunnelIp == cgnat.TunnelIp)
        {
            return; // No change
        }

        _logger.LogInformation(
            "Configuring WireGuard tunnel to relay {RelayId} (Tunnel IP: {TunnelIp})",
            cgnat.AssignedRelayNodeId, cgnat.TunnelIp);

        if (string.IsNullOrEmpty(cgnat.WireGuardConfig))
        {
            _logger.LogWarning("No WireGuard config provided");
            return;
        }

        try
        {
            // Save config
            var configPath = "/etc/wireguard/wg-relay.conf";
            await File.WriteAllTextAsync(configPath, cgnat.WireGuardConfig, ct);

            // Stop existing interface if running
            await RunCommandAsync("wg-quick", "down wg-relay", ignoreError: true);

            // Start new interface
            await RunCommandAsync("wg-quick", "up wg-relay");

            _lastRelayId = cgnat.AssignedRelayNodeId;
            _lastTunnelIp = cgnat.TunnelIp;

            _logger.LogInformation("WireGuard tunnel configured successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to configure WireGuard tunnel");
        }
    }

    private async Task RunCommandAsync(string command, string args, bool ignoreError = false)
    {
        var psi = new ProcessStartInfo
        {
            FileName = command,
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };

        using var process = Process.Start(psi);
        if (process == null)
        {
            throw new Exception($"Failed to start {command}");
        }

        await process.WaitForExitAsync();

        if (process.ExitCode != 0 && !ignoreError)
        {
            var error = await process.StandardError.ReadToEndAsync();
            throw new Exception($"{command} failed: {error}");
        }
    }
}