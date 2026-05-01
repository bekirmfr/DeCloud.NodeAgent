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
/// Layer 1 — Network boundary:
///   virbr0 is not routable externally. Only VMs on this node's libvirt
///   network can reach 192.168.122.1:5100.
///
/// Layer 2 — Subnet check (all endpoints):
///   Every request is validated against the virbr0 subnet (192.168.122.0/24).
///   Rejects requests if a misconfigured firewall rule temporarily exposes
///   port 5100 externally.
///
/// Layer 3 — Caller-IP-to-role binding (/state endpoint only):
///   The /state endpoint requires that the caller's IP matches the IP of the
///   system VM currently assigned to that role. This prevents a tenant VM on
///   the same virbr0 bridge from calling GET /dht/state to exfiltrate the
///   DHT node's Ed25519 private key.
///
///   Enforcement:
///     - VM found and IP known and matches → 200
///     - VM found and IP known and differs → 403 (log Warning)
///     - VM found but IP not yet resolvable → allow + log Warning
///       (first-boot timing window; DHCP lease still being acquired)
///     - No VM found for role → allow + log Warning
///       (state not yet delivered; GetStateJsonAsync returns 404 anyway)
///
/// LOGGING DISCIPLINE
/// ──────────────────
/// State JSON is never written to logs. Only role name, version, and the
/// caller's IP address are logged, and only at DEBUG level on success.
/// A 404, subnet block, or role-binding block is logged at WARNING.
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
                    // The relay's WG API lives at .254 of the relay's OWN subnet (e.g. 10.20.0.254),
                    // NOT .254 of the host's CGNAT subnet (e.g. 10.20.248.254 — doesn't exist).
                    // Derive the relay's subnet from its allowed-ips via `wg show`.
                    // Fall back to the NodeAgent proxy which handles routing correctly.
                    var relayHostIp = relayEndpoint.Split(':')[0];
                    // Use the relay's public IP (port 8080 is NAT-forwarded) so
                    // wg-mesh-enroll.sh Strategy 2 can reach the relay API even
                    // before the host's wg-relay tunnel has an active handshake.
                    // The tunnel gateway (10.20.x.254) is not reachable at enrollment
                    // time if the host's WireGuard handshake hasn't completed yet.
                    var relayApiUrl = $"http://{relayHostIp}:8080/api/relay";
                    var relayGatewayIp = await DiscoverRelayGatewayFromWgAsync(ct)
                        ?? $"10.20.0.254";

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
                    // Use the relay VM's direct virbr0 IP — co-located VMs on the
                    // same bridge can reach it directly. 192.168.122.1 (host bridge)
                    // is not the relay and has nothing listening on UDP 51820.
                    // Peer registration uses the NodeAgent proxy separately.
                    relayEndpoint = $"{relayVmIp}:51820",
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

    private async Task<string?> DiscoverRelayGatewayFromWgAsync(CancellationToken ct)
    {
        try
        {
            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "wg",
                    Arguments = "show all allowed-ips",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };
            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync(ct);
            await process.WaitForExitAsync(ct);

            // Parse: "<iface>  <pubkey>  10.20.0.0/16"
            // Take first 10.20.x.x network and derive .254 of that network
            foreach (var line in output.Split('\n'))
            {
                var parts = line.Split('\t', ' ', StringSplitOptions.RemoveEmptyEntries);
                foreach (var part in parts.Skip(2))
                {
                    if (!part.StartsWith("10.20.")) continue;
                    var network = part.Split('/')[0];
                    var octets = network.Split('.');
                    if (octets.Length == 4)
                        return $"{octets[0]}.{octets[1]}.{octets[2]}.254";
                }
            }
            return null;
        }
        catch { return null; }
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
        // Layer 2: subnet check
        if (!EnforceVirbr0(out var callerIp))
            return StatusCode(403, "Forbidden: obligation state is only accessible from the VM bridge network.");

        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
        {
            _logger.LogWarning(
                "GetState: unknown role '{Role}' from {Ip}", role, callerIp);
            return BadRequest($"Unknown role '{role}'. Valid values: relay, dht, blockstore.");
        }

        // Layer 3: caller-IP-to-role binding — only the assigned system VM may read its own keys
        if (!await EnforceRoleBinding(canonical, callerIp, ct))
            return StatusCode(403,
                $"Forbidden: caller IP {callerIp} is not the assigned {canonical} VM on this node.");

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
    /// Layer 3 enforcement: verifies the caller's IP matches the IP of the
    /// system VM currently assigned to <paramref name="role"/> on this node.
    ///
    /// Allow-with-warning cases (all narrow timing windows or harmless states):
    ///   - No VM in memory for role  → state will 404 anyway; allow
    ///   - VM found, IP not in spec  → DHCP fallback tried; if still null, allow
    ///   - VM found, IP not in DHCP  → cloud-init first-boot window; allow
    /// </summary>
    private async Task<bool> EnforceRoleBinding(
        string role, string callerIp, CancellationToken ct)
    {
        var expectedType = RoleToVmType(role);
        if (expectedType is null)
            return true;

        var systemVm = _vmManager.GetAllVms()
            .FirstOrDefault(v =>
                v.Spec.VmType == expectedType &&
                v.State is VmState.Running or VmState.Starting or VmState.Creating);

        if (systemVm is null)
        {
            _logger.LogWarning(
                "GetState [{Role}]: no system VM in memory for role binding — " +
                "allowing {Ip} (state will 404 if not yet delivered)",
                role, callerIp);
            return true;
        }

        // Prefer cached IP on spec; fall back to live DHCP query
        var assignedIp = systemVm.Spec.IpAddress;
        if (string.IsNullOrWhiteSpace(assignedIp))
            assignedIp = await _vmManager.GetVmIpAddressAsync(systemVm.VmId, ct);

        if (string.IsNullOrWhiteSpace(assignedIp))
        {
            _logger.LogWarning(
                "GetState [{Role}]: system VM {VmId} IP not yet resolvable — " +
                "allowing {Ip} (first-boot timing window)",
                role, systemVm.VmId[..8], callerIp);
            return true;
        }

        var assignedBare = assignedIp.Split('/')[0].Trim();
        if (!string.Equals(assignedBare, callerIp, StringComparison.OrdinalIgnoreCase))
        {
            _logger.LogWarning(
                "GetState [{Role}]: REJECTED — caller {CallerIp} does not match " +
                "assigned VM {VmId} IP {AssignedIp}. Possible tenant VM exfil attempt.",
                role, callerIp, systemVm.VmId[..8], assignedBare);
            return false;
        }

        return true;
    }

    /// <summary>
    /// Maps an obligation role name to the corresponding <see cref="VmType"/>.
    /// Returns null for unrecognised roles.
    /// </summary>
    private static VmType? RoleToVmType(string role) => role switch
    {
        ObligationRole.Relay => VmType.Relay,
        ObligationRole.Dht => VmType.Dht,
        ObligationRole.BlockStore => VmType.BlockStore,
        _ => null,
    };

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