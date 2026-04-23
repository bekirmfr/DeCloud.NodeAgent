using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Services;
using DeCloud.Shared.Models;
using Microsoft.AspNetCore.Mvc;
using System.Net;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Serves obligation identity state to system VMs over the virbr0 bridge
/// (192.168.122.x — the libvirt internal network).
///
/// SECURITY MODEL
/// ──────────────
/// Primary boundary:  virbr0 is not routable externally; only VMs on this
///                    node's libvirt network can reach 192.168.122.1:5100.
///
/// Defense-in-depth:  Every request is validated against the virbr0 subnet
///                    (192.168.122.0/24) at the controller level. If a
///                    misconfigured firewall rule temporarily opens port 5100
///                    externally, the controller still refuses to serve keys.
///
/// No authentication token is required — the network boundary is the auth.
///
/// LOGGING DISCIPLINE
/// ──────────────────
/// State JSON is never written to logs. Only role name, version, and the
/// caller's IP address are logged, and only at DEBUG level on success.
/// A 404 or a blocked-IP attempt is logged at WARNING.
///
/// ROUTES
/// ──────
///   GET /api/obligations/{role}/state
///       Returns identity state JSON for the role.
///       404 if no state has been persisted yet (first boot).
///
///   GET /api/obligations/{role}/version
///       Returns the current version integer.
///       Lightweight check — no key material in the response.
///       Returns 0 if no state exists.
/// </summary>
[ApiController]
[Route("api/obligations")]
public class ObligationStateController : ControllerBase
{
    // virbr0 bridge network — only addresses in this subnet may call this controller.
    private static readonly IPNetwork VirbR0Network = IPNetwork.Parse("192.168.122.0/24");

    private readonly IObligationStateService _obligationState;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly IPortForwardingManager _portForwardingManager;
    private readonly IVmManager _vmManager;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<ObligationStateController> _logger;

    // Tunnel IP offsets within the relay subnet — must match orchestrator's DhtNodeService
    // and BlockStoreService offset conventions.
    private const int DhtCgnatOffset = 230; // host .2 → DHT .232
    private const int BlockStoreCgnatOffset = 210; // host .2 → BlockStore .212
    private const int DhtRelayNodeOctet = 199; // fixed for co-located relay nodes
    private const int BlockStoreRelayNodeOctet = 202; // fixed for co-located relay nodes

    public ObligationStateController(
        IObligationStateService obligationState,
        IOrchestratorClient orchestratorClient,
        IPortForwardingManager portForwardingManager,
        IVmManager vmManager,
        IHttpClientFactory httpClientFactory,
        ILogger<ObligationStateController> logger)
    {
        _obligationState = obligationState;
        _orchestratorClient = orchestratorClient;
        _portForwardingManager = portForwardingManager;
        _vmManager = vmManager;
        _httpClientFactory = httpClientFactory;
        _logger = logger;
    }

    // ----------------------------------------------------------------
    // GET /api/obligations/{role}/wg-config
    // ----------------------------------------------------------------

    /// <summary>
    /// Returns the WireGuard connection parameters needed for a system VM to
    /// enroll in the relay mesh at boot time.
    ///
    /// Called by DHT and BlockStore VMs during cloud-init when the relay assignment
    /// was not yet available at deploy time (CGNAT race). The VM polls this endpoint
    /// until the NodeAgent has relay info, then proceeds with wg-mesh-enroll.sh.
    ///
    /// Returns 202 Accepted with Retry-After header if relay is not yet assigned.
    /// </summary>
    [HttpGet("{role}/wg-config")]
    [ProducesResponseType(200)]
    [ProducesResponseType(202)]
    [ProducesResponseType(400)]
    [ProducesResponseType(403)]
    public async Task<IActionResult> GetWgConfig(string role, CancellationToken ct)
    {
        if (!EnforceVirbr0(out var callerIp))
            return StatusCode(403, "Forbidden: accessible only from the VM bridge network.");

        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
            return BadRequest($"Unknown role '{role}'. Valid values: dht, blockstore.");

        if (canonical == ObligationRole.Relay)
            return BadRequest("Relay VMs are the relay — they do not connect to one.");

        // ── Path 1: CGNAT node ────────────────────────────────────────────────
        var cgnatInfo = _orchestratorClient.GetLastHeartbeat()?.Heartbeat?.CgnatInfo;
        if (cgnatInfo != null && !string.IsNullOrEmpty(cgnatInfo.WireGuardConfig))
        {
            var relayEndpoint = ParseWgConfigField(cgnatInfo.WireGuardConfig, "Endpoint");
            var relayPubKey = ParseWgConfigField(cgnatInfo.WireGuardConfig, "PublicKey");
            var hostTunnelIp = cgnatInfo.TunnelIp; // e.g. 10.20.248.2

            if (!string.IsNullOrEmpty(relayEndpoint) &&
                !string.IsNullOrEmpty(relayPubKey) &&
                !string.IsNullOrEmpty(hostTunnelIp))
            {
                var vmTunnelIp = ComputeVmTunnelIp(hostTunnelIp, canonical);
                if (vmTunnelIp != null)
                {
                    // Relay gateway is .254 in the subnet — that's where the relay API lives
                    var parts = hostTunnelIp.Split('.');
                    var relayApiUrl = $"http://{parts[0]}.{parts[1]}.{parts[2]}.254:8080/api/relay";
                    var relayHostIp = relayEndpoint.Split(':')[0];

                    _logger.LogInformation(
                        "GetWgConfig [{Role}] → CGNAT path: endpoint={Endpoint}, tunnel={Tunnel}. Caller: {Ip}",
                        canonical, relayEndpoint, vmTunnelIp, callerIp);

                    return Ok(new
                    {
                        relayEndpoint,
                        relayPublicKey = relayPubKey,
                        relayApiUrl,
                        tunnelIp = $"{vmTunnelIp}/24"
                    });
                }
            }
        }

        // ── Path 2: Public IP node with co-located relay VM ───────────────────
        var relayVmIp = await _portForwardingManager.GetRelayVmIpAsync(ct);
        var relayVm = _vmManager.GetAllVms()
            .FirstOrDefault(v =>
                v.Spec.VmType == VmType.Relay &&
                v.State == VmState.Running);

        if (relayVmIp != null && relayVm != null)
        {
            try
            {
                var http = _httpClientFactory.CreateClient();
                http.Timeout = TimeSpan.FromSeconds(5);
                var statusJson = await http.GetStringAsync(
                    $"http://{relayVmIp}/api/relay/status", ct);

                using var doc = System.Text.Json.JsonDocument.Parse(statusJson);
                var relayPubKey = doc.RootElement
                    .GetProperty("wireguard_public_key")
                    .GetString() ?? string.Empty;

                var relaySubnetLabel = relayVm.Spec.Labels
                    ?.GetValueOrDefault("relay-subnet") ?? "248";
                int.TryParse(relaySubnetLabel, out var relaySubnet);

                var vmOctet = canonical == ObligationRole.Dht
                    ? DhtRelayNodeOctet
                    : BlockStoreRelayNodeOctet;
                var vmTunnelIp = $"10.20.{relaySubnet}.{vmOctet}";

                _logger.LogInformation(
                    "GetWgConfig [{Role}] → local relay path: vmIp={RelayVmIp}, tunnel={Tunnel}. Caller: {Ip}",
                    canonical, relayVmIp, vmTunnelIp, callerIp);

                return Ok(new
                {
                    relayEndpoint = $"192.168.122.1:51820", // NodeAgent proxy handles enrollment
                    relayPublicKey = relayPubKey,
                    relayApiUrl = $"http://{relayVmIp}:8080/api/relay",
                    tunnelIp = $"{vmTunnelIp}/24"
                });
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "GetWgConfig [{Role}]: could not query relay VM at {Ip} — returning 202",
                    canonical, relayVmIp);
            }
        }

        // ── Not ready yet — tell VM to retry ─────────────────────────────────
        _logger.LogInformation(
            "GetWgConfig [{Role}]: relay not yet available — returning 202. Caller: {Ip}",
            canonical, callerIp);

        Response.Headers["Retry-After"] = "15";
        return StatusCode(202, new
        {
            status = "pending",
            message = "Relay not yet assigned. Retry in 15 seconds."
        });
    }

    // ----------------------------------------------------------------
    // Private helpers
    // ----------------------------------------------------------------

    /// <summary>
    /// Parse a field from a WireGuard config [Peer] section.
    /// e.g. ParseWgConfigField(config, "PublicKey") → "1+MKr..."
    /// </summary>
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

    /// <summary>
    /// Compute the WireGuard mesh tunnel IP for a system VM on a CGNAT node,
    /// using the same offset convention as DhtNodeService and BlockStoreService.
    /// </summary>
    private static string? ComputeVmTunnelIp(string hostTunnelIp, string role)
    {
        var parts = hostTunnelIp.Split('.');
        if (parts.Length != 4 || !int.TryParse(parts[3], out var hostOctet))
            return null;

        var offset = role == ObligationRole.Dht
            ? DhtCgnatOffset
            : BlockStoreCgnatOffset;

        var vmOctet = offset + hostOctet;
        if (vmOctet > 253) return null;

        return $"{parts[0]}.{parts[1]}.{parts[2]}.{vmOctet}";
    }

    // ----------------------------------------------------------------
    // GET /api/obligations/{role}/state
    // ----------------------------------------------------------------

    /// <summary>
    /// Returns the raw identity state JSON for <paramref name="role"/>.
    /// Called by system VMs at cloud-init time to restore persistent identity
    /// before starting their services.
    /// </summary>
    /// <param name="role">Role name: relay | dht | blockstore (case-insensitive).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <response code="200">State JSON returned successfully.</response>
    /// <response code="400">Unknown role name.</response>
    /// <response code="403">Caller is not on the virbr0 subnet.</response>
    /// <response code="404">No state persisted for this role yet.</response>
    [HttpGet("{role}/state")]
    [ProducesResponseType(typeof(string), 200)]
    [ProducesResponseType(400)]
    [ProducesResponseType(403)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> GetState(string role, CancellationToken ct)
    {
        if (!EnforceVirbr0(out var callerIp))
            return StatusCode(403, "Forbidden: obligation state is only accessible from the VM bridge network.");

        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
        {
            _logger.LogWarning(
                "GetState: unknown role '{Role}' from {Ip}", role, callerIp);
            return BadRequest($"Unknown role '{role}'. Valid values: relay, dht, blockstore.");
        }

        var stateJson = await _obligationState.GetStateJsonAsync(canonical, ct);

        if (stateJson is null)
        {
            _logger.LogDebug(
                "GetState [{Role}]: no state found (first boot?) — returning 404. Caller: {Ip}",
                canonical, callerIp);
            return NotFound($"No obligation state found for role '{canonical}'.");
        }

        // Log access at DEBUG only — never log the state content.
        var version = await _obligationState.GetVersionAsync(canonical, ct);
        _logger.LogDebug(
            "GetState [{Role}] v{Version} served to {Ip}",
            canonical, version, callerIp);

        // Return raw JSON with the correct content-type so the VM can parse it directly.
        return Content(stateJson, "application/json");
    }

    // ----------------------------------------------------------------
    // GET /api/obligations/{role}/version
    // ----------------------------------------------------------------

    /// <summary>
    /// Returns the current stored version for <paramref name="role"/> as a plain integer.
    /// Intended for lightweight checks — no key material is included in the response.
    /// </summary>
    /// <param name="role">Role name: relay | dht | blockstore (case-insensitive).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <response code="200">Version integer (0 if no state persisted).</response>
    /// <response code="400">Unknown role name.</response>
    /// <response code="403">Caller is not on the virbr0 subnet.</response>
    [HttpGet("{role}/version")]
    [ProducesResponseType(typeof(int), 200)]
    [ProducesResponseType(400)]
    [ProducesResponseType(403)]
    public async Task<IActionResult> GetVersion(string role, CancellationToken ct)
    {
        if (!EnforceVirbr0(out var callerIp))
            return StatusCode(403, "Forbidden: obligation state is only accessible from the VM bridge network.");

        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
        {
            _logger.LogWarning(
                "GetVersion: unknown role '{Role}' from {Ip}", role, callerIp);
            return BadRequest($"Unknown role '{role}'. Valid values: relay, dht, blockstore.");
        }

        var version = await _obligationState.GetVersionAsync(canonical, ct);

        _logger.LogDebug(
            "GetVersion [{Role}] → v{Version} served to {Ip}",
            canonical, version, callerIp);

        return Ok(version);
    }

    // ----------------------------------------------------------------
    // Private helpers
    // ----------------------------------------------------------------

    /// <summary>
    /// Validates that the caller's remote IP is within the virbr0 subnet
    /// (192.168.122.0/24). Returns <c>true</c> if the request may proceed.
    /// </summary>
    /// <param name="callerIp">The caller's IP address string (for logging).</param>
    private bool EnforceVirbr0(out string callerIp)
    {
        var remote = HttpContext.Connection.RemoteIpAddress;

        // Unwrap IPv4-mapped-in-IPv6 (::ffff:192.168.122.x)
        if (remote is not null && remote.IsIPv4MappedToIPv6)
            remote = remote.MapToIPv4();

        callerIp = remote?.ToString() ?? "unknown";

        if (remote is null || !VirbR0Network.Contains(remote))
        {
            _logger.LogWarning(
                "ObligationStateController: blocked request from {Ip} — not on virbr0 subnet",
                callerIp);
            return false;
        }

        return true;
    }
}
