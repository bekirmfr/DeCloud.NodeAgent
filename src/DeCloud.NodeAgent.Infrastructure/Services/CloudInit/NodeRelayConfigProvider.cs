using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.Shared.Enums;
using Microsoft.Extensions.Logging;

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

    /// <summary>
    /// Returns the per-relay Bearer token for this node's assigned relay, or null
    /// when unavailable. Role-agnostic: the token is per-relay, not per-role. On a
    /// CGNAT node it comes from the relay assignment (CgnatInfo.RelayApiToken); on a
    /// co-located relay host it is null here (the NodeAgent reads it from the local
    /// relay obligation instead).
    /// </summary>
    Task<string?> TryGetRelayApiTokenAsync(CancellationToken ct);
}

/// <summary>
/// Pre-fetched relay configuration. Mirrors the wire shape of the
/// /api/obligations/{role}/wg-config endpoint response.
/// </summary>
public sealed record NodeRelayConfig(
    string RelayEndpoint,    // e.g. "142.234.200.95:51820"
    string RelayPublicKey,
    string RelayApiUrl,      // e.g. "http://142.234.200.95:8080"
    string TunnelIp,         // e.g. "10.30.0.248/24"
    int Mtu,                 // wg-mesh interface MTU for this node's path
    string? AuthToken = null); // per-relay Bearer token; set on CGNAT nodes (from
                               // CgnatInfo), null on co-located relay hosts, whose
                               // NodeAgent reads it from the local relay obligation

// WireGuard mesh MTU per encapsulation depth. The mesh interface defaults to
// 1420 (single-WireGuard assumption). A CGNAT node carries DHT traffic as
// WireGuard-inside-WireGuard (mesh tunnel over the relay tunnel), so its path
// MTU is ~120 bytes lower; 1220 is measured-safe across that stacked path.
// Public nodes have one WireGuard layer and could run higher, but 1220 is
// used uniformly — an under-MTU costs marginal throughput, never connectivity,
// and a single value keeps both ends of any link symmetric (the path MTU is
// the min of both ends, so the lower governs regardless).
public static class WgMeshMtu
{
    public const int Cgnat = 1220;
    public const int Public = 1220;
}

public sealed class NodeRelayConfigProvider : INodeRelayConfigProvider
{
    // Relay /24 mesh addressing — single source of truth in
    // DeCloud.Shared.Models.RelaySubnetLayout. Compile-time aliases: the numbers
    // are defined once there, so this file can never drift from the controller's copy.
    private const int DhtCgnatOffset = DeCloud.Shared.Models.RelaySubnetLayout.DhtCgnatOffset;
    private const int BlockStoreCgnatOffset = DeCloud.Shared.Models.RelaySubnetLayout.BlockStoreCgnatOffset;
    private const int DhtRelayNodeOctet = DeCloud.Shared.Models.RelaySubnetLayout.DhtRelayNodeOctet;
    private const int BlockStoreRelayNodeOctet = DeCloud.Shared.Models.RelaySubnetLayout.BlockStoreRelayNodeOctet;

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
                         TunnelIp: $"{vmTunnelIp}/24",
                         Mtu: WgMeshMtu.Cgnat,
                         AuthToken: cgnatInfo.RelayApiToken);
                }
            }
        }

        // ── Path 2: Public IP node with co-located relay VM ──
        var relayVmIp = await _portForwardingManager.GetRelayVmIpAsync(ct);
        var relayVm = _vmManager.GetAllVms()
            .FirstOrDefault(v => v.Spec.Role == VmRole.Relay && v.Status == Shared.Enums.VmStatus.Running);

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
                    TunnelIp: $"{vmTunnelIp}/24",
                    Mtu: WgMeshMtu.Public);
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

    public Task<string?> TryGetRelayApiTokenAsync(CancellationToken ct)
    {
        // Role-agnostic, no network: the relay token rides the heartbeat on
        // CgnatInfo. Null on a co-located relay host (that NodeAgent uses the
        // local relay obligation instead).
        var cgnatInfo = _orchestratorClient.GetLastHeartbeat()?.Heartbeat?.CgnatInfo;
        return Task.FromResult(cgnatInfo?.RelayApiToken);
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