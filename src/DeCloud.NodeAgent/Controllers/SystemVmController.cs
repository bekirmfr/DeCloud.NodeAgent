using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.SystemVm;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.AspNetCore.Mvc;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Proxies operator requests to system VM dashboards (relay, dht, blockstore).
///
/// System VMs are managed entirely by the node agent and have no ingress subdomain.
/// This controller provides the only external access path to their management dashboards,
/// scoped to operators who already have access to the node agent.
///
/// Route: GET /api/system-vms/{role}/proxy/{**path}
///
/// Examples:
///   GET /api/system-vms/relay/proxy/              → relay-api dashboard
///   GET /api/system-vms/relay/proxy/api/relay/status
///   GET /api/system-vms/dht/proxy/
///   GET /api/system-vms/blockstore/proxy/
/// </summary>
[ApiController]
[Route("api/system-vms")]
public class SystemVmController : ControllerBase
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly IVmManager _vmManager;
    private readonly ObligationStateRepository _obligationStore;
    private readonly ISystemVmService _systemVmService;
    private readonly ILogger<SystemVmController> _logger;

    public SystemVmController(
        IHttpClientFactory httpClientFactory,
        IVmManager vmManager,
        ObligationStateRepository obligationStore,
        ISystemVmService systemVmService,
        ILogger<SystemVmController> logger)
    {
        _httpClientFactory = httpClientFactory;
        _vmManager = vmManager;
        _obligationStore = obligationStore;
        _systemVmService = systemVmService;
        _logger = logger;
    }

    // ── GET / HEAD / POST / PUT / DELETE / OPTIONS / PATCH ──────────────

    [HttpGet("{role}/proxy/{**path}")]
    [HttpHead("{role}/proxy/{**path}")]
    [HttpPost("{role}/proxy/{**path}")]
    [HttpPut("{role}/proxy/{**path}")]
    [HttpDelete("{role}/proxy/{**path}")]
    [HttpOptions("{role}/proxy/{**path}")]
    [HttpPatch("{role}/proxy/{**path}")]
    public async Task<IActionResult> ProxyDashboard(
        string role,
        string? path = null,
        CancellationToken ct = default)
    {
        // ── 1. Validate role ────────────────────────────────────────────
        if (!ISystemVmService.RoleToVmType.ContainsKey(role))
        {
            return NotFound(new
            {
                error = $"Unknown system VM role '{role}'. Valid roles: relay, dht, blockstore."
            });
        }

        // ── 2. Resolve dashboard URL ────────────────────────────────────
        var vm = _systemVmService.GetRunningVm(role);
        if (vm is null)
        {
            return NotFound(new
            {
                error = $"No running {role} VM found. " +
                        "The VM may still be booting or has not been deployed on this node."
            });
        }

        // ── 3. Build target URL ─────────────────────────────────────────
        var targetPath  = string.IsNullOrEmpty(path) ? "/" : "/" + path;
        var queryString = Request.QueryString.HasValue ? Request.QueryString.Value : "";
        var targetUrl = $"{_systemVmService.GetDashboardBaseUrl(role)}{targetPath}{queryString}";

        _logger.LogDebug(
            "System VM dashboard proxy: {Role} {Method} {Path} → {TargetUrl}",
            role, Request.Method, targetPath, targetUrl);

        // ── 4. Forward the request ──────────────────────────────────────
        try
        {
            var httpClient = _httpClientFactory.CreateClient("VmProxy");

            var upstream = new HttpRequestMessage(
                new HttpMethod(Request.Method),
                targetUrl);

            // Copy request headers (skip hop-by-hop)
            foreach (var header in Request.Headers)
            {
                if (HopByHopHeaders.Contains(header.Key))
                    continue;

                upstream.Headers.TryAddWithoutValidation(header.Key, header.Value.ToArray());
            }

            // Copy request body
            if (Request.ContentLength > 0 || Request.Headers.ContainsKey("Transfer-Encoding"))
            {
                upstream.Content = new StreamContent(Request.Body);
                if (!string.IsNullOrEmpty(Request.ContentType))
                    upstream.Content.Headers.TryAddWithoutValidation(
                        "Content-Type", Request.ContentType);
            }

            using var response = await httpClient.SendAsync(
                upstream,
                HttpCompletionOption.ResponseHeadersRead,
                ct);

            // ── 5. Stream response back ─────────────────────────────────
            Response.StatusCode = (int)response.StatusCode;

            var contentType = response.Content.Headers.ContentType?.MediaType ?? "";

            foreach (var header in response.Headers)
            {
                if (!HopByHopHeaders.Contains(header.Key))
                    Response.Headers[header.Key] = header.Value.ToArray();
            }

            foreach (var header in response.Content.Headers)
            {
                if (!HopByHopHeaders.Contains(header.Key) &&
                    !header.Key.Equals("Content-Length", StringComparison.OrdinalIgnoreCase))
                    Response.Headers[header.Key] = header.Value.ToArray();
            }

            // HTML responses: rewrite absolute asset URLs so the browser
            // routes /static/* back through this proxy instead of hitting
            // the node agent root (where those paths don't exist).
            if (contentType.Contains("text/html", StringComparison.OrdinalIgnoreCase))
            {
                var proxyBase = $"/api/system-vms/{role}/proxy";
                var html = await response.Content.ReadAsStringAsync(ct);

                // Inject <base> tag — all relative URLs (static assets, JS API
                // calls) resolve through this proxy regardless of trailing slash.
                html = html.Replace("<head>", $"<head>\n    <base href=\"{proxyBase}/\">");

                // Strip leading slash from static refs so they become relative
                // and are resolved by the <base> tag rather than the root.
                html = html
                    .Replace("href=\"/static/", "href=\"static/")
                    .Replace("src=\"/static/", "src=\"static/");

                var bytes = System.Text.Encoding.UTF8.GetBytes(html);
                Response.Headers["Content-Length"] = bytes.Length.ToString();
                await Response.Body.WriteAsync(bytes, ct);
            }
            else
            {
                await response.Content.CopyToAsync(Response.Body, ct);
            }

            return new EmptyResult();
        }
        catch (TaskCanceledException)
        {
            return StatusCode(504, new { error = $"{role} dashboard request timed out." });
        }
        catch (HttpRequestException ex)
        {
            var isRefused = ex.InnerException is
                System.Net.Sockets.SocketException { SocketErrorCode: System.Net.Sockets.SocketError.ConnectionRefused };

            if (isRefused)
            {
                _logger.LogWarning(
                    "System VM dashboard {Role} at {Ip}:{Port} refused connection — service may still be starting",
                    role, vm.Spec.IpAddress, ISystemVmService.DashboardPort);

                return StatusCode(503, new
                {
                    error = $"{role} dashboard is not yet reachable. The service may still be starting.",
                    vmId  = vm.VmId,
                    vmIp  = vm.Spec.IpAddress
                });
            }

            _logger.LogWarning(ex,
                "System VM dashboard proxy failed for {Role} at {Ip}:{Port}",
                role, vm.Spec.IpAddress, ISystemVmService.DashboardPort);

            return StatusCode(502, new { error = "Failed to reach system VM dashboard.", details = ex.Message });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "System VM dashboard proxy error for {Role}",
                role);

            return StatusCode(500, new { error = "Internal proxy error.", details = ex.Message });
        }
    }

    /// <summary>
    /// Returns connection details for a co-located system VM so that sibling
    /// system VMs (e.g. blockstore → dht) can establish direct libp2p
    /// connections over virbr0 without involving the orchestrator.
    ///
    /// 404 means "not yet ready" — callers poll until they get 200.
    /// </summary>
    [HttpGet("{role}/peer-info")]
    public async Task<IActionResult> GetPeerInfo(string role)
    {
        var vmType = ISystemVmService.RoleToVmType.TryGetValue(role, out var vt) ? vt : (VmType?)null;
        if (vmType is null) return NotFound();

        var vm = _vmManager.GetRunningVms()
            .FirstOrDefault(v => v.Spec.VmType == vmType.Value && v.IsFullyReady);
        if (vm is null || string.IsNullOrEmpty(vm.Spec.IpAddress))
            return NotFound();  // not yet running — caller polls

        var state = await _obligationStore.GetStateJsonAsync(role);
        var peerId = ExtractPeerId(state);
        if (string.IsNullOrEmpty(peerId))
            return NotFound();

        return Ok(new
        {
            peerId,
            ipAddress = vm.Spec.IpAddress,   // virbr0 — same-host direct route
            port = vmType == VmType.Dht ? 4001 : 5001
        });
    }

    /// <summary>
    /// Extract the libp2p peerId from a serialized obligation state JSON
    /// document. Returns null if the field is missing or the JSON is invalid —
    /// callers treat that as "not yet ready" and the client polls.
    /// </summary>
    private static string? ExtractPeerId(string? stateJson)
    {
        if (string.IsNullOrWhiteSpace(stateJson)) return null;
        try
        {
            using var doc = System.Text.Json.JsonDocument.Parse(stateJson);
            return doc.RootElement.TryGetProperty("peerId", out var p)
                && p.ValueKind == System.Text.Json.JsonValueKind.String
                ? p.GetString()
                : null;
        }
        catch (System.Text.Json.JsonException)
        {
            return null;
        }
    }

    // ── Headers that must not be forwarded ──────────────────────────────

    private static readonly HashSet<string> HopByHopHeaders =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "Connection",
            "Keep-Alive",
            "Transfer-Encoding",
            "TE",
            "Trailer",
            "Upgrade",
            "Proxy-Authorization",
            "Proxy-Authenticate",
        };
}
