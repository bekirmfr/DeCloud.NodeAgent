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

            // Check if this is a WebSocket upgrade request
            if (IsWebSocketUpgradeRequest())
            {
                _logger.LogInformation(
                    "Detected WebSocket upgrade request for VM {VmId} port {Port}",
                    vmId, port);
                
                // Handle as WebSocket tunnel
                await HandleWebSocketTunnel(vmId, vmIp, port, ct);
                return new EmptyResult();
            }

            // Build target URL for regular HTTP request
            var targetPath = string.IsNullOrEmpty(path) ? "" : "/" + path;
            var queryString = Request.QueryString.HasValue ? Request.QueryString.Value : "";
            var targetUrl = $"http://{vmIp}:{port}{targetPath}{queryString}";

            // Use Trace for routine proxy requests to reduce log noise
            _logger.LogTrace(
                "Proxying HTTP {Method} for VM {VmId} to {Url}",
                Request.Method, vmId, targetUrl);

            // Create HTTP client with appropriate timeout
            var httpClient = _httpClientFactory.CreateClient("VmProxy");
            httpClient.Timeout = GetTimeoutForPort(port);

            // Forward request
            return await ForwardHttpRequestAsync(httpClient, targetUrl, vmId, port, ct);
        }
        catch (TaskCanceledException)
        {
            // Only log warning if timeout is unusual (not for known slow endpoints)
            _logger.LogDebug("HTTP proxy timeout for VM {VmId} port {Port}", vmId, port);
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
        string vmId,
        int port,
        CancellationToken ct)
    {
        // Build request message
        var request = new HttpRequestMessage(
            new HttpMethod(Request.Method),
            targetUrl);

        // Copy headers (excluding some). We will set Host and X-Forwarded-* explicitly below.
        foreach (var header in Request.Headers)
        {
            if (!ShouldSkipHeader(header.Key))
            {
                request.Headers.TryAddWithoutValidation(header.Key, header.Value.ToArray());
            }
        }

        // Use the PUBLIC hostname for the backend (e.g. code-server). When Caddy proxies to us,
        // it may rewrite Host to our address; X-Forwarded-Host keeps the original client host.
        // Backends need the public host to set cookies/redirects correctly (e.g. code-server login).
        var publicHost = Request.Headers["X-Forwarded-Host"].FirstOrDefault()
            ?? Request.Headers.Host.ToString();
        if (!string.IsNullOrEmpty(publicHost))
        {
            request.Headers.Host = publicHost;
            request.Headers.Remove("X-Forwarded-Host");
            request.Headers.TryAddWithoutValidation("X-Forwarded-Host", publicHost);
        }

        var proto = Request.Headers["X-Forwarded-Proto"].FirstOrDefault() ?? Request.Scheme;
        request.Headers.Remove("X-Forwarded-Proto");
        request.Headers.TryAddWithoutValidation("X-Forwarded-Proto", proto);
        var clientIp = HttpContext.Connection.RemoteIpAddress?.ToString();
        if (!string.IsNullOrEmpty(clientIp))
        {
            var existingFor = Request.Headers["X-Forwarded-For"].FirstOrDefault();
            var forValue = string.IsNullOrEmpty(existingFor) ? clientIp : $"{existingFor}, {clientIp}";
            request.Headers.Remove("X-Forwarded-For");
            request.Headers.TryAddWithoutValidation("X-Forwarded-For", forValue);
        }

        // Copy body if present
        var hasBody = Request.ContentLength > 0
    || (Request.ContentLength == null && HttpMethods.IsPost(Request.Method))
    || (Request.ContentLength == null && HttpMethods.IsPut(Request.Method))
    || (Request.ContentLength == null && HttpMethods.IsPatch(Request.Method));

        if (hasBody)
        {
            Request.EnableBuffering();

            if (Request.Body.CanSeek)
            {
                Request.Body.Position = 0;
            }

            var bodyStream = new MemoryStream();
            await Request.Body.CopyToAsync(bodyStream, ct);

            if (bodyStream.Length > 0)
            {
                bodyStream.Position = 0;
                request.Content = new StreamContent(bodyStream);
                if (Request.ContentType != null)
                {
                    request.Content.Headers.ContentType =
                        System.Net.Http.Headers.MediaTypeHeaderValue.Parse(Request.ContentType);
                }

                request.Content.Headers.ContentLength = bodyStream.Length;
            }
        }

        // Send request
        var response = await client.SendAsync(request, ct);

        // Copy response
        Response.StatusCode = (int)response.StatusCode;

        // Copy response headers (excluding problematic ones)
        foreach (var header in response.Headers)
        {
            if (!ShouldSkipResponseHeader(header.Key))
            {
                Response.Headers[header.Key] = header.Value.ToArray();
            }
        }

        // Copy content headers - IMPORTANT: Include Content-Length if present
        foreach (var header in response.Content.Headers)
        {
            if (!ShouldSkipResponseHeader(header.Key))
            {
                Response.Headers[header.Key] = header.Value.ToArray();
            }
        }

        // Only copy body for status codes that allow it
        // 304 Not Modified, 204 No Content, and 1xx/HEAD responses must not have bodies
        if (response.StatusCode != System.Net.HttpStatusCode.NotModified &&
            response.StatusCode != System.Net.HttpStatusCode.NoContent &&
            (int)response.StatusCode >= 200 &&
            Request.Method != "HEAD")
        {
            // If Content-Length is known, disable chunked encoding
            if (response.Content.Headers.ContentLength.HasValue)
            {
                Response.ContentLength = response.Content.Headers.ContentLength.Value;
            }
            
            await response.Content.CopyToAsync(Response.Body, ct);
        }

        return new EmptyResult();
    }

    private static bool ShouldSkipHeader(string headerName)
    {
        var skip = new[] { "host", "connection", "transfer-encoding", "content-length" };
        return skip.Contains(headerName.ToLowerInvariant());
    }
    
    private static bool ShouldSkipResponseHeader(string headerName)
    {
        // Skip headers that ASP.NET Core manages automatically
        var skip = new[] { "connection", "transfer-encoding" };
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

    private bool IsWebSocketUpgradeRequest()
    {
        var connectionHeader = Request.Headers["Connection"].ToString();
        var upgradeHeader = Request.Headers["Upgrade"].ToString();
        
        return connectionHeader.Contains("Upgrade", StringComparison.OrdinalIgnoreCase) &&
               upgradeHeader.Equals("websocket", StringComparison.OrdinalIgnoreCase);
    }

    private async Task HandleWebSocketTunnel(string vmId, string vmIp, int port, CancellationToken ct)
    {
        if (!HttpContext.WebSockets.IsWebSocketRequest)
        {
            HttpContext.Response.StatusCode = 400;
            await HttpContext.Response.WriteAsync("WebSocket connection required");
            return;
        }

        try
        {
            _logger.LogInformation(
                "Proxying WebSocket upgrade for VM {VmId} to {VmIp}:{Port}, path: {Path}",
                vmId, vmIp, port, Request.Path);

            // Build the backend WebSocket URL
            var path = Request.Path.ToString();
            var queryString = Request.QueryString.HasValue ? Request.QueryString.Value : "";
            
            // Extract the actual path after /proxy/http/{port}/
            var proxyPrefix = $"/api/vms/{vmId}/proxy/http/{port}";
            if (path.StartsWith(proxyPrefix))
            {
                path = path.Substring(proxyPrefix.Length);
            }
            
            var backendWsUrl = $"wss://{vmIp}:{port}{path}{queryString}";
            
            _logger.LogDebug("Backend WebSocket URL: {Url}", backendWsUrl);

            // Create WebSocket client to backend
            using var backendWs = new System.Net.WebSockets.ClientWebSocket();
            
            // Copy headers from client request to backend request
            foreach (var header in Request.Headers)
            {
                if (IsWebSocketHeader(header.Key))
                {
                    continue; // Skip WebSocket-specific headers, ClientWebSocket handles these
                }
                if (!ShouldSkipHeader(header.Key))
                {
                    try
                    {
                        backendWs.Options.SetRequestHeader(header.Key, header.Value.ToString());
                    }
                    catch
                    {
                        // Skip headers that can't be set
                    }
                }
            }

            // For code-server, we need to connect via HTTP (not HTTPS) to the VM's internal IP
            var httpBackendWsUrl = $"ws://{vmIp}:{port}{path}{queryString}";
            
            _logger.LogDebug("Connecting to backend via: {Url}", httpBackendWsUrl);
            
            // Connect to backend WebSocket
            await backendWs.ConnectAsync(new Uri(httpBackendWsUrl), ct);

            _logger.LogInformation("WebSocket tunnel established for VM {VmId}", vmId);

            // Accept client WebSocket
            var clientWs = await HttpContext.WebSockets.AcceptWebSocketAsync();

            // Bidirectional bridge
            await Task.WhenAll(
                RelayWebSocketAsync(clientWs, backendWs, "client->backend", ct),
                RelayWebSocketAsync(backendWs, clientWs, "backend->client", ct)
            );

            _logger.LogInformation("WebSocket tunnel closed for VM {VmId}", vmId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "WebSocket tunnel error for VM {VmId} port {Port}", vmId, port);
            throw;
        }
    }

    private static bool IsWebSocketHeader(string headerName)
    {
        var wsHeaders = new[] { 
            "sec-websocket-key", "sec-websocket-version", "sec-websocket-protocol",
            "sec-websocket-extensions", "upgrade", "connection"
        };
        return wsHeaders.Contains(headerName.ToLowerInvariant());
    }

    private async Task RelayWebSocketAsync(
        System.Net.WebSockets.WebSocket source,
        System.Net.WebSockets.WebSocket destination,
        string direction,
        CancellationToken ct)
    {
        var buffer = new byte[8192];
        
        try
        {
            while (source.State == System.Net.WebSockets.WebSocketState.Open &&
                   destination.State == System.Net.WebSockets.WebSocketState.Open &&
                   !ct.IsCancellationRequested)
            {
                var result = await source.ReceiveAsync(new ArraySegment<byte>(buffer), ct);

                if (result.MessageType == System.Net.WebSockets.WebSocketMessageType.Close)
                {
                    await destination.CloseAsync(
                        System.Net.WebSockets.WebSocketCloseStatus.NormalClosure,
                        result.CloseStatusDescription,
                        ct);
                    break;
                }

                await destination.SendAsync(
                    new ArraySegment<byte>(buffer, 0, result.Count),
                    result.MessageType,
                    result.EndOfMessage,
                    ct);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "WebSocket relay {Direction} ended", direction);
        }
    }

    private static TimeSpan GetTimeoutForPort(int port)
    {
        return port switch
        {
            9999 => TimeSpan.FromSeconds(3),
            22 => TimeSpan.FromSeconds(30),
            80 or 443 or 8080 => TimeSpan.FromSeconds(10),
            _ => TimeSpan.FromSeconds(30)
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