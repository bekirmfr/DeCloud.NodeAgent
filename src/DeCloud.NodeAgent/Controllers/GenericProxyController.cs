using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.AspNetCore.Mvc;
using System.Net.WebSockets;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Unified proxy controller for all VM network access.
/// Replaces separate SSH, SFTP, attestation, and HTTP proxy endpoints.
/// 
/// Routes:
///   /api/vms/{vmId}/proxy/http/{port}/{**path}       - HTTP/HTTPS proxy
///   /api/vms/{vmId}/proxy/ws/{port}                  - WebSocket tunnel (for SSH/SFTP/etc)
///   /api/vms/{vmId}/proxy/tcp/{port}                 - Raw TCP tunnel (WebSocket-based)
/// 
/// Examples:
///   /api/vms/{vmId}/proxy/http/9999/challenge        - Attestation agent
///   /api/vms/{vmId}/proxy/http/80/index.html         - Web server
///   /api/vms/{vmId}/proxy/ws/22                      - SSH over WebSocket
///   /api/vms/{vmId}/proxy/tcp/3306                   - MySQL over WebSocket
/// </summary>
[ApiController]
[Route("api/vms/{vmId}/proxy")]
public class GenericProxyController : ControllerBase
{
    private readonly IVmManager _vmManager;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<GenericProxyController> _logger;

    // Security: Allowed ports (can be configured via appsettings)
    private static readonly HashSet<int> AllowedPorts = new()
    {
        22,    // SSH
        80,    // HTTP
        443,   // HTTPS
        3306,  // MySQL
        5432,  // PostgreSQL
        6379,  // Redis
        8080,  // Common HTTP alt
        8443,  // Common HTTPS alt
        9999,  // Attestation agent
    };

    // Ports that require special handling (e.g., authentication)
    private static readonly HashSet<int> ProtectedPorts = new() { 22, 3306, 5432, 6379 };

    public GenericProxyController(
        IVmManager vmManager,
        IHttpClientFactory httpClientFactory,
        ILogger<GenericProxyController> logger)
    {
        _vmManager = vmManager;
        _httpClientFactory = httpClientFactory;
        _logger = logger;
    }

    #region HTTP Proxy

    /// <summary>
    /// Proxy HTTP requests to VM's internal services
    /// 
    /// Examples:
    ///   GET  /api/vms/{vmId}/proxy/http/9999/health          → http://{vm-ip}:9999/health
    ///   POST /api/vms/{vmId}/proxy/http/9999/challenge       → http://{vm-ip}:9999/challenge
    ///   GET  /api/vms/{vmId}/proxy/http/80/api/status        → http://{vm-ip}:80/api/status
    /// </summary>
    [HttpGet("http/{port:int}/{**path}")]
    [HttpPost("http/{port:int}/{**path}")]
    [HttpPut("http/{port:int}/{**path}")]
    [HttpDelete("http/{port:int}/{**path}")]
    [HttpPatch("http/{port:int}/{**path}")]
    [HttpOptions("http/{port:int}/{**path}")]
    [HttpHead("http/{port:int}/{**path}")]
    public async Task<IActionResult> ProxyHttp(
        string vmId,
        int port,
        string? path = null,
        CancellationToken ct = default)
    {
        try
        {
            // Security validation
            if (!AllowedPorts.Contains(port))
            {
                _logger.LogWarning(
                    "Blocked proxy attempt to unauthorized port {Port} for VM {VmId}",
                    port, vmId);
                return StatusCode(403, new { error = $"Port {port} is not allowed" });
            }

            // Get VM details
            var vm = await _vmManager.GetVmAsync(vmId, ct);
            if (vm == null)
            {
                return NotFound(new { error = "VM not found" });
            }

            var vmIp = vm.Spec.IpAddress;
            if (string.IsNullOrEmpty(vmIp))
            {
                return BadRequest(new { error = "VM IP not available" });
            }

            // Build target URL
            var targetPath = string.IsNullOrEmpty(path) ? "" : "/" + path;
            var queryString = Request.QueryString.HasValue ? Request.QueryString.Value : "";
            var targetUrl = $"http://{vmIp}:{port}{targetPath}{queryString}";

            _logger.LogDebug(
                "Proxying HTTP {Method} for VM {VmId} to {Url}",
                Request.Method, vmId, targetUrl);

            // Create HTTP client with appropriate timeout
            var httpClient = _httpClientFactory.CreateClient();
            httpClient.Timeout = GetTimeoutForPort(port);

            // Forward request
            return await ForwardHttpRequestAsync(httpClient, targetUrl, ct);
        }
        catch (TaskCanceledException)
        {
            _logger.LogWarning("HTTP proxy timeout for VM {VmId} port {Port}", vmId, port);
            return StatusCode(504, new { error = "Request timeout" });
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP proxy failed for VM {VmId} port {Port}", vmId, port);
            return StatusCode(502, new { error = "Failed to reach VM service", details = ex.Message });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "HTTP proxy error for VM {VmId} port {Port}", vmId, port);
            return StatusCode(500, new { error = "Internal error", details = ex.Message });
        }
    }

    private async Task<IActionResult> ForwardHttpRequestAsync(
        HttpClient client,
        string targetUrl,
        CancellationToken ct)
    {
        // Build request message
        var request = new HttpRequestMessage(
            new HttpMethod(Request.Method),
            targetUrl);

        // Copy headers (excluding some)
        foreach (var header in Request.Headers)
        {
            if (!ShouldSkipHeader(header.Key))
            {
                request.Headers.TryAddWithoutValidation(header.Key, header.Value.ToArray());
            }
        }

        // Copy body if present
        if (Request.ContentLength > 0)
        {
            var bodyStream = new MemoryStream();
            await Request.Body.CopyToAsync(bodyStream, ct);
            bodyStream.Position = 0;
            request.Content = new StreamContent(bodyStream);

            if (Request.ContentType != null)
            {
                request.Content.Headers.ContentType =
                    System.Net.Http.Headers.MediaTypeHeaderValue.Parse(Request.ContentType);
            }
        }

        // Send request
        var response = await client.SendAsync(request, ct);

        // Copy response
        Response.StatusCode = (int)response.StatusCode;

        foreach (var header in response.Headers)
        {
            Response.Headers[header.Key] = header.Value.ToArray();
        }

        foreach (var header in response.Content.Headers)
        {
            Response.Headers[header.Key] = header.Value.ToArray();
        }

        await response.Content.CopyToAsync(Response.Body, ct);

        return new EmptyResult();
    }

    private static bool ShouldSkipHeader(string headerName)
    {
        var skip = new[] { "host", "connection", "transfer-encoding", "content-length" };
        return skip.Contains(headerName.ToLowerInvariant());
    }

    #endregion

    #region WebSocket/TCP Tunnel

    /// <summary>
    /// WebSocket tunnel for raw TCP connections (SSH, databases, etc)
    /// 
    /// Usage:
    ///   ws://node:5100/api/vms/{vmId}/proxy/ws/22?user=root&password=...
    ///   ws://node:5100/api/vms/{vmId}/proxy/ws/3306
    /// </summary>
    [HttpGet("ws/{port:int}")]
    public async Task WebSocketTunnel(
        string vmId,
        int port,
        CancellationToken ct = default)
    {
        if (!HttpContext.WebSockets.IsWebSocketRequest)
        {
            HttpContext.Response.StatusCode = 400;
            await HttpContext.Response.WriteAsync("WebSocket connection required");
            return;
        }

        // Security validation
        if (!AllowedPorts.Contains(port))
        {
            _logger.LogWarning(
                "Blocked WebSocket tunnel to unauthorized port {Port} for VM {VmId}",
                port, vmId);
            HttpContext.Response.StatusCode = 403;
            await HttpContext.Response.WriteAsync($"Port {port} is not allowed");
            return;
        }

        try
        {
            // Get VM details
            var vm = await _vmManager.GetVmAsync(vmId, ct);
            if (vm == null)
            {
                HttpContext.Response.StatusCode = 404;
                await HttpContext.Response.WriteAsync("VM not found");
                return;
            }

            var vmIp = vm.Spec.IpAddress;
            if (string.IsNullOrEmpty(vmIp))
            {
                HttpContext.Response.StatusCode = 400;
                await HttpContext.Response.WriteAsync("VM IP not available");
                return;
            }

            _logger.LogInformation(
                "Opening WebSocket tunnel for VM {VmId} to {VmIp}:{Port}",
                vmId, vmIp, port);

            // Accept WebSocket connection
            var webSocket = await HttpContext.WebSockets.AcceptWebSocketAsync();

            // Create TCP connection to VM
            using var tcpClient = new System.Net.Sockets.TcpClient();
            await tcpClient.ConnectAsync(vmIp, port, ct);

            using var stream = tcpClient.GetStream();

            // Bidirectional proxy
            await Task.WhenAny(
                ProxyWebSocketToTcpAsync(webSocket, stream, ct),
                ProxyTcpToWebSocketAsync(stream, webSocket, ct)
            );

            _logger.LogInformation(
                "WebSocket tunnel closed for VM {VmId} port {Port}",
                vmId, port);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "WebSocket tunnel error for VM {VmId} port {Port}", vmId, port);
        }
    }

    private async Task ProxyWebSocketToTcpAsync(
        WebSocket webSocket,
        System.Net.Sockets.NetworkStream stream,
        CancellationToken ct)
    {
        var buffer = new byte[8192];

        while (webSocket.State == WebSocketState.Open && !ct.IsCancellationRequested)
        {
            var result = await webSocket.ReceiveAsync(buffer, ct);

            if (result.MessageType == WebSocketMessageType.Close)
            {
                await webSocket.CloseAsync(
                    WebSocketCloseStatus.NormalClosure,
                    "Client closed",
                    ct);
                break;
            }

            if (result.Count > 0)
            {
                await stream.WriteAsync(buffer.AsMemory(0, result.Count), ct);
            }
        }
    }

    private async Task ProxyTcpToWebSocketAsync(
        System.Net.Sockets.NetworkStream stream,
        WebSocket webSocket,
        CancellationToken ct)
    {
        var buffer = new byte[8192];

        while (webSocket.State == WebSocketState.Open && !ct.IsCancellationRequested)
        {
            var bytesRead = await stream.ReadAsync(buffer, ct);

            if (bytesRead == 0)
            {
                break; // Connection closed
            }

            await webSocket.SendAsync(
                buffer.AsMemory(0, bytesRead),
                WebSocketMessageType.Binary,
                true,
                ct);
        }
    }

    #endregion

    #region Helper Methods

    private static TimeSpan GetTimeoutForPort(int port)
    {
        return port switch
        {
            9999 => TimeSpan.FromMilliseconds(150), // Attestation - fast
            22 => TimeSpan.FromSeconds(30),          // SSH - medium
            80 or 443 or 8080 => TimeSpan.FromSeconds(10), // HTTP - medium
            _ => TimeSpan.FromSeconds(30)            // Default
        };
    }

    #endregion

    #region Health Check

    /// <summary>
    /// Quick health check for a VM service
    /// GET /api/vms/{vmId}/proxy/health/{port}
    /// </summary>
    [HttpGet("health/{port:int}")]
    public async Task<IActionResult> HealthCheck(
        string vmId,
        int port,
        CancellationToken ct = default)
    {
        try
        {
            var vm = await _vmManager.GetVmAsync(vmId, ct);
            if (vm == null)
            {
                return NotFound(new { error = "VM not found" });
            }

            var vmIp = vm.Spec.IpAddress;
            if (string.IsNullOrEmpty(vmIp))
            {
                return BadRequest(new { error = "VM IP not available" });
            }

            // Try to connect
            using var tcpClient = new System.Net.Sockets.TcpClient();
            await tcpClient.ConnectAsync(vmIp, port, ct);

            return Ok(new
            {
                vmId,
                port,
                status = "reachable",
                vmIp,
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            return Ok(new
            {
                vmId,
                port,
                status = "unreachable",
                error = ex.Message,
                timestamp = DateTime.UtcNow
            });
        }
    }

    #endregion
}