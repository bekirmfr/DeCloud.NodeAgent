using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;
using System.Net.Http;

namespace DeCloud.NodeAgent.Infrastructure.Services.CloudInit;

public interface INodeRelayConfigProvider
{
    /// <summary>
    /// Returns relay configuration for a mesh-participant role (DHT,
    /// BlockStore), or null if relay isn't yet available (CGNAT race or
    /// co-located relay still booting). Callers either retry (wg-config
    /// endpoint sends 202) or substitute "" for affected dynamics
    /// (environment endpoint relies on watcher's generation diff).
    /// </summary>
    Task<NodeRelayConfig?> TryGetAsync(string role, CancellationToken ct);
}

/// <summary>
/// Pre-fetched relay configuration. Mirrors the wire shape of the
/// /api/obligations/{role}/wg-config endpoint response.
/// </summary>
public sealed record NodeRelayConfig(
    string RelayEndpoint,    // e.g. "142.234.200.95:51820"
    string RelayPublicKey,
    string RelayApiUrl,      // e.g. "http://142.234.200.95:8080"
    string TunnelIp);        // e.g. "10.30.0.248/24"

public sealed class NodeRelayConfigProvider : INodeRelayConfigProvider
{
    // Tunnel IP offsets — must match orchestrator's DhtNodeService and
    // BlockStoreService offset conventions. Lifted as-is from
    // ObligationStateController; consolidate via shared constants when
    // the controller's GetWgConfig is refactored to call this provider.
    private const int DhtCgnatOffset = 230;
    private const int BlockStoreCgnatOffset = 210;
    private const int DhtRelayNodeOctet = 199;
    private const int BlockStoreRelayNodeOctet = 202;

    private readonly IOrchestratorClient _orchestratorClient;
    private readonly IPortForwardingManager _portForwardingManager;
    private readonly IVmManager _vmManager;
    private readonly HttpClient _httpClient;
    private readonly ILogger<NodeRelayConfigProvider> _logger;

    public NodeRelayConfigProvider(
        IOrchestratorClient orchestratorClient,
        IPortForwardingManager portForwardingManager,
        IVmManager vmManager,
        HttpClient httpClient,
        ILogger<NodeRelayConfigProvider> logger)
    {
        _orchestratorClient = orchestratorClient;
        _portForwardingManager = portForwardingManager;
        _vmManager = vmManager;
        _httpClient = httpClient;
        _logger = logger;
    }

    public async Task<NodeRelayConfig?> TryGetAsync(string role, CancellationToken ct)
    {
        // ── Path 1: CGNAT node (relay info from orchestrator heartbeat) ──
        var cgnatInfo = _orchestratorClient.GetLastHeartbeat()?.Heartbeat?.CgnatInfo;
        if (cgnatInfo != null && !string.IsNullOrEmpty(cgnatInfo.WireGuardConfig))
        {
            var relayEndpoint = ParseWgConfigField(cgnatInfo.WireGuardConfig, "Endpoint");
            var relayPubKey = ParseWgConfigField(cgnatInfo.WireGuardConfig, "PublicKey");
            var hostTunnelIp = cgnatInfo.TunnelIp;

            if (!string.IsNullOrEmpty(relayEndpoint) &&
                !string.IsNullOrEmpty(relayPubKey) &&
                !string.IsNullOrEmpty(hostTunnelIp))
            {
                var vmTunnelIp = ComputeVmTunnelIp(hostTunnelIp, role);
                if (vmTunnelIp != null)
                {
                    var relayHostIp = relayEndpoint.Split(':')[0];
                    return new NodeRelayConfig(
                        RelayEndpoint: relayEndpoint,
                        RelayPublicKey: relayPubKey,
                        RelayApiUrl: $"http://{relayHostIp}:8080",
                        TunnelIp: $"{vmTunnelIp}/24");
                }
            }
        }

        // ── Path 2: Public IP node with co-located relay VM ──
        var relayVmIp = await _portForwardingManager.GetRelayVmIpAsync(ct);
        var relayVm = _vmManager.GetAllVms()
            .FirstOrDefault(v => v.Spec.VmType == VmType.Relay && v.State == VmState.Running);

        if (relayVmIp != null && relayVm != null)
        {
            try
            {
                var statusJson = await _httpClient.GetStringAsync(
                    $"http://{relayVmIp}/api/relay/status", ct);

                using var doc = System.Text.Json.JsonDocument.Parse(statusJson);
                var relayPubKey = doc.RootElement
                    .GetProperty("wireguard_public_key").GetString() ?? "";

                var relaySubnetLabel = relayVm.Spec.Labels
                    ?.GetValueOrDefault("relay-subnet") ?? "248";
                int.TryParse(relaySubnetLabel, out var relaySubnet);

                var vmOctet = role == "dht"
                    ? DhtRelayNodeOctet
                    : BlockStoreRelayNodeOctet;
                var vmTunnelIp = $"10.20.{relaySubnet}.{vmOctet}";

                return new NodeRelayConfig(
                    RelayEndpoint: $"{relayVmIp}:51820",
                    RelayPublicKey: relayPubKey,
                    RelayApiUrl: $"http://{relayVmIp}:8080",
                    TunnelIp: $"{vmTunnelIp}/24");
            }
            catch (Exception ex)
            {
                _logger.LogDebug(
                    "NodeRelayConfigProvider [{Role}]: co-located relay at {Ip} not reachable — treating as not-yet-ready",
                    role, relayVmIp);
            }
        }

        // Not ready yet. Caller decides whether to retry (wg-config) or
        // return empties (environment endpoint).
        return null;
    }

    private static string? ParseWgConfigField(string wgConfig, string fieldName)
    {
        foreach (var line in wgConfig.Split('\n'))
        {
            var trimmed = line.Trim();
            if (trimmed.StartsWith(fieldName + " =", StringComparison.OrdinalIgnoreCase) ||
                trimmed.StartsWith(fieldName + "=", StringComparison.OrdinalIgnoreCase))
            {
                var idx = trimmed.IndexOf('=');
                return idx >= 0 ? trimmed[(idx + 1)..].Trim() : null;
            }
        }
        return null;
    }

    private static string? ComputeVmTunnelIp(string hostTunnelIp, string role)
    {
        var parts = hostTunnelIp.Split('.');
        if (parts.Length != 4 || !int.TryParse(parts[3], out var hostOctet))
            return null;
        var offset = role == "dht" ? DhtCgnatOffset : BlockStoreCgnatOffset;
        var vmOctet = offset + hostOctet;
        if (vmOctet > 253) return null;
        return $"{parts[0]}.{parts[1]}.{parts[2]}.{vmOctet}";
    }
}