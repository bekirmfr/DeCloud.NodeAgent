using DeCloud.NodeAgent.Infrastructure.Services;
using Microsoft.AspNetCore.Mvc;
using System.Text;
using System.Text.Json;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Proxies WireGuard mesh enrollment requests from DHT VMs to the relay VM.
///
/// DHT VMs (inside QEMU) cannot reach the relay VM's API at port 8080 because
/// only UDP/51820 is NAT-forwarded from the host. This proxy endpoint runs on
/// the host's NodeAgent (reachable from VMs via virbr0 default gateway on port 5100)
/// and forwards the enrollment request to the relay VM's bridge IP.
///
/// For CGNAT nodes (no local relay), the NodeAgent reaches the relay through
/// the host's WireGuard tunnel at the relay's gateway IP (10.20.x.254).
/// </summary>
[ApiController]
[Route("api/relay")]
public class WgMeshEnrollController : ControllerBase
{
    private readonly IPortForwardingManager _portForwardingManager;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<WgMeshEnrollController> _logger;

    public WgMeshEnrollController(
        IPortForwardingManager portForwardingManager,
        IHttpClientFactory httpClientFactory,
        ILogger<WgMeshEnrollController> logger)
    {
        _portForwardingManager = portForwardingManager;
        _httpClientFactory = httpClientFactory;
        _logger = logger;
    }

    [HttpPost("wg-mesh-enroll")]
    public async Task<IActionResult> WgMeshEnroll(
        [FromBody] WgMeshEnrollRequest request,
        CancellationToken ct)
    {
        if (string.IsNullOrEmpty(request.PublicKey) ||
            string.IsNullOrEmpty(request.AllowedIps))
        {
            return BadRequest(new { error = "Missing public_key or allowed_ips" });
        }

        _logger.LogInformation(
            "WG mesh enrollment proxy: registering peer {PubKey} with allowed_ips={AllowedIps}",
            request.PublicKey[..Math.Min(16, request.PublicKey.Length)] + "...",
            request.AllowedIps);

        // Strategy 1: Local relay VM (co-located on same host)
        var relayIp = await _portForwardingManager.GetRelayVmIpAsync(ct);
        if (!string.IsNullOrEmpty(relayIp))
        {
            _logger.LogInformation(
                "Found local relay VM at bridge IP {RelayIp}, proxying enrollment",
                relayIp);

            var result = await ProxyToRelayApiAsync(relayIp, request, ct);
            if (result != null)
                return result;
        }

        // Strategy 2: Relay tunnel gateway (CGNAT host with WG tunnel to relay)
        var relayTunnelIp = await DiscoverRelayTunnelGatewayAsync(ct);
        if (!string.IsNullOrEmpty(relayTunnelIp))
        {
            _logger.LogInformation(
                "Found relay tunnel gateway at {TunnelIp}, proxying enrollment via WG tunnel",
                relayTunnelIp);

            var result = await ProxyToRelayApiAsync(relayTunnelIp, request, ct);
            if (result != null)
                return result;
        }

        _logger.LogWarning(
            "WG mesh enrollment proxy failed: no relay reachable " +
            "(no local relay VM, no WG tunnel gateway)");

        return StatusCode(502, new
        {
            error = "No relay reachable",
            message = "Could not find local relay VM or WG tunnel gateway to proxy enrollment"
        });
    }

    private async Task<IActionResult?> ProxyToRelayApiAsync(
        string relayIp, WgMeshEnrollRequest request, CancellationToken ct)
    {
        var relayUrl = $"http://{relayIp}:8080/api/relay/add-peer";

        try
        {
            var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(10);

            var payload = new
            {
                public_key = request.PublicKey,
                allowed_ips = request.AllowedIps,
                description = request.Description ?? "mesh-peer",
                peer_type = request.PeerType ?? "system-vm",
                parent_node_id = request.ParentNodeId ?? ""
            };

            var json = JsonSerializer.Serialize(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync(relayUrl, content, ct);
            var responseBody = await response.Content.ReadAsStringAsync(ct);

            if (response.IsSuccessStatusCode)
            {
                _logger.LogInformation(
                    "WG mesh enrollment proxied successfully to {RelayIp}",
                    relayIp);

                return Content(responseBody, "application/json");
            }

            _logger.LogWarning(
                "Relay API at {RelayIp} returned {StatusCode}: {Body}",
                relayIp, (int)response.StatusCode, responseBody);

            // Return null to try next strategy
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Failed to proxy enrollment to relay at {RelayIp}",
                relayIp);
            return null;
        }
    }

    /// <summary>
    /// Discover the relay's tunnel gateway IP from the host's WireGuard interfaces.
    /// On CGNAT hosts, the host has a WG tunnel to the relay (e.g., 10.20.1.2/24).
    /// The relay gateway is at .254 in the same subnet.
    /// </summary>
    private async Task<string?> DiscoverRelayTunnelGatewayAsync(CancellationToken ct)
    {
        try
        {
            // Look for WG interfaces with 10.20.x.x addresses
            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "ip",
                    Arguments = "-4 -o addr show",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync(ct);
            await process.WaitForExitAsync(ct);

            // Parse lines like: "5: wg-relay-client inet 10.20.1.2/24 scope global wg-relay-client"
            foreach (var line in output.Split('\n'))
            {
                if (!line.Contains("10.20.")) continue;

                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < parts.Length - 1; i++)
                {
                    if (parts[i] == "inet" && parts[i + 1].StartsWith("10.20."))
                    {
                        var tunnelIpCidr = parts[i + 1]; // "10.20.1.2/24"
                        var tunnelIp = tunnelIpCidr.Split('/')[0];
                        var octets = tunnelIp.Split('.');

                        if (octets.Length == 4)
                        {
                            // Relay gateway is .254 in the same subnet
                            var gatewayIp = $"{octets[0]}.{octets[1]}.{octets[2]}.254";

                            _logger.LogDebug(
                                "Derived relay tunnel gateway {GatewayIp} from host tunnel IP {TunnelIp}",
                                gatewayIp, tunnelIp);

                            return gatewayIp;
                        }
                    }
                }
            }

            _logger.LogDebug("No WG tunnel with 10.20.x.x address found on host");
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Error discovering relay tunnel gateway");
            return null;
        }
    }
}

/// <summary>
/// Request body for WG mesh enrollment proxy.
/// Matches the relay API's add-peer format with snake_case JSON.
/// </summary>
public record WgMeshEnrollRequest(
    [property: System.Text.Json.Serialization.JsonPropertyName("public_key")]
    string PublicKey,

    [property: System.Text.Json.Serialization.JsonPropertyName("allowed_ips")]
    string AllowedIps,

    [property: System.Text.Json.Serialization.JsonPropertyName("description")]
    string? Description,

    [property: System.Text.Json.Serialization.JsonPropertyName("peer_type")]
    string? PeerType = null,

    [property: System.Text.Json.Serialization.JsonPropertyName("parent_node_id")]
    string? ParentNodeId = null
);
