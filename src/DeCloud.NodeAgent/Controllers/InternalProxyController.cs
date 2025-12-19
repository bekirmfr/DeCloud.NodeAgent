using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.AspNetCore.Mvc;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Internal proxy controller for central ingress traffic.
/// Receives requests from the Orchestrator's Caddy and forwards them to VMs.
/// 
/// This endpoint is used when the Orchestrator routes *.vms.decloud.io traffic
/// to the appropriate node, which then proxies to the VM's private IP.
/// 
/// Route: /internal/proxy/{vmId}/*
/// </summary>
[ApiController]
[Route("internal/proxy")]
public class InternalProxyController : ControllerBase
{
    private readonly IVmManager _vmManager;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<InternalProxyController> _logger;

    public InternalProxyController(
        IVmManager vmManager,
        IHttpClientFactory httpClientFactory,
        ILogger<InternalProxyController> logger)
    {
        _vmManager = vmManager;
        _httpClientFactory = httpClientFactory;
        _logger = logger;
    }

    /// <summary>
    /// Proxy HTTP requests to a VM's private IP.
    /// The target port is specified via X-DeCloud-Target-Port header.
    /// </summary>
    [HttpGet("{vmId}/{**path}")]
    [HttpPost("{vmId}/{**path}")]
    [HttpPut("{vmId}/{**path}")]
    [HttpDelete("{vmId}/{**path}")]
    [HttpPatch("{vmId}/{**path}")]
    [HttpOptions("{vmId}/{**path}")]
    [HttpHead("{vmId}/{**path}")]
    public async Task ProxyToVm(string vmId, string? path = null)
    {
        try
        {
            // Get VM
            var vm = await _vmManager.GetVmAsync(vmId, HttpContext.RequestAborted);
            if (vm == null)
            {
                _logger.LogWarning("Proxy request for unknown VM: {VmId}", vmId);
                HttpContext.Response.StatusCode = 404;
                await HttpContext.Response.WriteAsJsonAsync(new
                {
                    error = "VM not found",
                    vmId = vmId
                });
                return;
            }

            // Get VM's private IP
            var vmIp = vm.Spec.Network?.IpAddress;
            if (string.IsNullOrEmpty(vmIp))
            {
                _logger.LogWarning("VM {VmId} has no IP address", vmId);
                HttpContext.Response.StatusCode = 503;
                await HttpContext.Response.WriteAsJsonAsync(new
                {
                    error = "VM not ready",
                    message = "VM does not have an IP address yet"
                });
                return;
            }

            // Get target port from header or default to 80
            var targetPort = 80;
            if (HttpContext.Request.Headers.TryGetValue("X-DeCloud-Target-Port", out var portHeader))
            {
                int.TryParse(portHeader.FirstOrDefault(), out targetPort);
            }

            // Build target URL
            var targetPath = string.IsNullOrEmpty(path) ? "/" : $"/{path}";
            var queryString = HttpContext.Request.QueryString.Value ?? "";
            var targetUrl = $"http://{vmIp}:{targetPort}{targetPath}{queryString}";

            _logger.LogDebug(
                "Proxying {Method} {Path} → {VmId} ({VmIp}:{Port})",
                HttpContext.Request.Method, path, vmId, vmIp, targetPort);

            // Create HTTP client and forward request
            var client = _httpClientFactory.CreateClient("VmProxy");
            client.Timeout = TimeSpan.FromSeconds(30);

            // Build forwarded request
            var request = new HttpRequestMessage
            {
                Method = new HttpMethod(HttpContext.Request.Method),
                RequestUri = new Uri(targetUrl)
            };

            // Copy headers (except hop-by-hop headers)
            foreach (var header in HttpContext.Request.Headers)
            {
                if (IsHopByHopHeader(header.Key))
                    continue;

                // Don't copy our internal headers
                if (header.Key.StartsWith("X-DeCloud-"))
                    continue;

                request.Headers.TryAddWithoutValidation(header.Key, header.Value.ToArray());
            }

            // Add/update forwarded headers
            var clientIp = HttpContext.Request.Headers["X-Real-IP"].FirstOrDefault()
                ?? HttpContext.Connection.RemoteIpAddress?.ToString()
                ?? "unknown";

            request.Headers.TryAddWithoutValidation("X-Forwarded-For", clientIp);
            request.Headers.TryAddWithoutValidation("X-Forwarded-Proto",
                HttpContext.Request.Headers["X-Forwarded-Proto"].FirstOrDefault() ?? "https");
            request.Headers.TryAddWithoutValidation("X-Forwarded-Host",
                HttpContext.Request.Headers["X-Forwarded-Host"].FirstOrDefault()
                ?? HttpContext.Request.Host.Value);

            // Copy body for methods that have one
            if (HttpContext.Request.ContentLength > 0 || HttpContext.Request.Headers.ContainsKey("Transfer-Encoding"))
            {
                request.Content = new StreamContent(HttpContext.Request.Body);

                if (HttpContext.Request.ContentType != null)
                {
                    request.Content.Headers.TryAddWithoutValidation("Content-Type", HttpContext.Request.ContentType);
                }
            }

            // Send request to VM
            HttpResponseMessage response;
            try
            {
                response = await client.SendAsync(
                    request,
                    HttpCompletionOption.ResponseHeadersRead,
                    HttpContext.RequestAborted);
            }
            catch (HttpRequestException ex)
            {
                _logger.LogWarning(ex, "Failed to connect to VM {VmId} at {VmIp}:{Port}", vmId, vmIp, targetPort);
                HttpContext.Response.StatusCode = 502;
                await HttpContext.Response.WriteAsJsonAsync(new
                {
                    error = "Bad Gateway",
                    message = "Failed to connect to the application",
                    details = "The application may not be running or listening on the configured port"
                });
                return;
            }

            // Copy response status
            HttpContext.Response.StatusCode = (int)response.StatusCode;

            // Copy response headers
            foreach (var header in response.Headers)
            {
                if (IsHopByHopHeader(header.Key))
                    continue;

                HttpContext.Response.Headers[header.Key] = header.Value.ToArray();
            }

            foreach (var header in response.Content.Headers)
            {
                HttpContext.Response.Headers[header.Key] = header.Value.ToArray();
            }

            // Stream response body
            await using var responseStream = await response.Content.ReadAsStreamAsync(HttpContext.RequestAborted);
            await responseStream.CopyToAsync(HttpContext.Response.Body, HttpContext.RequestAborted);
        }
        catch (OperationCanceledException)
        {
            _logger.LogDebug("Proxy request cancelled for VM {VmId}", vmId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error proxying request to VM {VmId}", vmId);

            if (!HttpContext.Response.HasStarted)
            {
                HttpContext.Response.StatusCode = 500;
                await HttpContext.Response.WriteAsJsonAsync(new
                {
                    error = "Internal Server Error",
                    message = "An error occurred while proxying the request"
                });
            }
        }
    }

    /// <summary>
    /// WebSocket proxy for real-time applications
    /// </summary>
    [HttpGet("{vmId}/ws/{**path}")]
    public async Task ProxyWebSocket(string vmId, string? path = null)
    {
        if (!HttpContext.WebSockets.IsWebSocketRequest)
        {
            HttpContext.Response.StatusCode = 400;
            await HttpContext.Response.WriteAsync("WebSocket connection required");
            return;
        }

        // Get VM
        var vm = await _vmManager.GetVmAsync(vmId, HttpContext.RequestAborted);
        if (vm == null)
        {
            HttpContext.Response.StatusCode = 404;
            await HttpContext.Response.WriteAsync("VM not found");
            return;
        }

        var vmIp = vm.Spec.Network?.IpAddress;
        if (string.IsNullOrEmpty(vmIp))
        {
            HttpContext.Response.StatusCode = 503;
            await HttpContext.Response.WriteAsync("VM not ready");
            return;
        }

        var targetPort = 80;
        if (HttpContext.Request.Headers.TryGetValue("X-DeCloud-Target-Port", out var portHeader))
        {
            int.TryParse(portHeader.FirstOrDefault(), out targetPort);
        }

        _logger.LogDebug("Proxying WebSocket to VM {VmId} ({VmIp}:{Port})", vmId, vmIp, targetPort);

        try
        {
            // Accept incoming WebSocket
            using var clientWs = await HttpContext.WebSockets.AcceptWebSocketAsync();

            // Connect to VM's WebSocket
            using var vmWsClient = new System.Net.WebSockets.ClientWebSocket();

            var wsPath = string.IsNullOrEmpty(path) ? "/" : $"/{path}";
            var wsUri = new Uri($"ws://{vmIp}:{targetPort}{wsPath}");

            await vmWsClient.ConnectAsync(wsUri, HttpContext.RequestAborted);

            // Bidirectional proxy
            var clientToVm = ProxyWebSocketOneWay(clientWs, vmWsClient, "client→vm", HttpContext.RequestAborted);
            var vmToClient = ProxyWebSocketOneWay(vmWsClient, clientWs, "vm→client", HttpContext.RequestAborted);

            await Task.WhenAny(clientToVm, vmToClient);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "WebSocket proxy error for VM {VmId}", vmId);
        }
    }

    private async Task ProxyWebSocketOneWay(
        System.Net.WebSockets.WebSocket source,
        System.Net.WebSockets.WebSocket dest,
        string direction,
        CancellationToken ct)
    {
        var buffer = new byte[4096];

        while (source.State == System.Net.WebSockets.WebSocketState.Open && !ct.IsCancellationRequested)
        {
            var result = await source.ReceiveAsync(buffer, ct);

            if (result.MessageType == System.Net.WebSockets.WebSocketMessageType.Close)
            {
                await dest.CloseAsync(
                    System.Net.WebSockets.WebSocketCloseStatus.NormalClosure,
                    "Closed",
                    ct);
                break;
            }

            await dest.SendAsync(
                new ArraySegment<byte>(buffer, 0, result.Count),
                result.MessageType,
                result.EndOfMessage,
                ct);
        }
    }

    private static bool IsHopByHopHeader(string header)
    {
        return header.Equals("Connection", StringComparison.OrdinalIgnoreCase) ||
               header.Equals("Keep-Alive", StringComparison.OrdinalIgnoreCase) ||
               header.Equals("Proxy-Authenticate", StringComparison.OrdinalIgnoreCase) ||
               header.Equals("Proxy-Authorization", StringComparison.OrdinalIgnoreCase) ||
               header.Equals("TE", StringComparison.OrdinalIgnoreCase) ||
               header.Equals("Trailer", StringComparison.OrdinalIgnoreCase) ||
               header.Equals("Transfer-Encoding", StringComparison.OrdinalIgnoreCase) ||
               header.Equals("Upgrade", StringComparison.OrdinalIgnoreCase);
    }
}