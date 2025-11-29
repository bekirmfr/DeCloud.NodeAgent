using System.Net.WebSockets;
using System.Text;
using Microsoft.AspNetCore.Mvc;
using Renci.SshNet;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// WebSocket-based SSH proxy for browser-based terminal access to VMs
/// </summary>
[ApiController]
[Route("api/vms")]
public class SshProxyController : ControllerBase
{
    private readonly ILogger<SshProxyController> _logger;
    private readonly IConfiguration _configuration;

    private const int BufferSize = 4096;
    private const int TerminalCols = 120;
    private const int TerminalRows = 40;

    public SshProxyController(
        ILogger<SshProxyController> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
    }

    /// <summary>
    /// WebSocket endpoint for SSH proxy to a VM
    /// Connect via: ws://node:5100/api/vms/{vmId}/terminal?ip={vmIp}&user={username}
    /// 
    /// Authentication can be done via:
    /// - Query param: privateKey (base64 encoded)
    /// - Query param: password
    /// - Header: X-SSH-PrivateKey (base64 encoded)
    /// - Header: X-SSH-Password
    /// </summary>
    [HttpGet("{vmId}/terminal")]
    public async Task Terminal(
        string vmId,
        [FromQuery] string ip,
        [FromQuery] string user = "ubuntu",
        [FromQuery] int port = 22,
        [FromQuery] string? password = null,
        [FromQuery] string? privateKey = null)
    {
        if (!HttpContext.WebSockets.IsWebSocketRequest)
        {
            HttpContext.Response.StatusCode = 400;
            await HttpContext.Response.WriteAsync("WebSocket connection required");
            return;
        }

        if (string.IsNullOrEmpty(ip))
        {
            HttpContext.Response.StatusCode = 400;
            await HttpContext.Response.WriteAsync("VM IP address required");
            return;
        }

        // Get credentials from headers if not in query
        password ??= HttpContext.Request.Headers["X-SSH-Password"].FirstOrDefault();
        privateKey ??= HttpContext.Request.Headers["X-SSH-PrivateKey"].FirstOrDefault();

        _logger.LogInformation("SSH terminal requested for VM {VmId} at {Ip}:{Port} as {User}",
            vmId, ip, port, user);

        using var webSocket = await HttpContext.WebSockets.AcceptWebSocketAsync();

        try
        {
            await ProxySshConnection(webSocket, ip, port, user, password, privateKey, vmId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "SSH proxy error for VM {VmId}", vmId);

            if (webSocket.State == WebSocketState.Open)
            {
                var errorMessage = Encoding.UTF8.GetBytes($"\r\n\x1b[31mConnection error: {ex.Message}\x1b[0m\r\n");
                await webSocket.SendAsync(errorMessage, WebSocketMessageType.Binary, true, CancellationToken.None);
                await webSocket.CloseAsync(WebSocketCloseStatus.InternalServerError, ex.Message, CancellationToken.None);
            }
        }
    }

    private async Task ProxySshConnection(
        WebSocket webSocket,
        string host,
        int port,
        string username,
        string? password,
        string? privateKeyBase64,
        string vmId)
    {
        // Build authentication methods
        var authMethods = new List<AuthenticationMethod>();

        if (!string.IsNullOrEmpty(privateKeyBase64))
        {
            try
            {
                var keyBytes = Convert.FromBase64String(privateKeyBase64);
                var keyStream = new MemoryStream(keyBytes);
                var privateKeyFile = new PrivateKeyFile(keyStream);
                authMethods.Add(new PrivateKeyAuthenticationMethod(username, privateKeyFile));
                _logger.LogDebug("Using private key authentication for {User}@{Host}", username, host);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to parse private key, falling back to other methods");
            }
        }

        if (!string.IsNullOrEmpty(password))
        {
            authMethods.Add(new PasswordAuthenticationMethod(username, password));
            _logger.LogDebug("Using password authentication for {User}@{Host}", username, host);
        }

        // If no auth provided, try to use node's default key
        if (authMethods.Count == 0)
        {
            var defaultKeyPath = "/root/.ssh/id_rsa";
            if (System.IO.File.Exists(defaultKeyPath))
            {
                try
                {
                    var privateKeyFile = new PrivateKeyFile(defaultKeyPath);
                    authMethods.Add(new PrivateKeyAuthenticationMethod(username, privateKeyFile));
                    _logger.LogDebug("Using node's default SSH key for {User}@{Host}", username, host);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to load default SSH key");
                }
            }
        }

        if (authMethods.Count == 0)
        {
            throw new InvalidOperationException("No SSH authentication method available. Provide password or privateKey.");
        }

        var connectionInfo = new Renci.SshNet.ConnectionInfo(host, port, username, authMethods.ToArray())
        {
            Timeout = TimeSpan.FromSeconds(30),
            RetryAttempts = 2
        };

        using var sshClient = new SshClient(connectionInfo);

        // Send connecting message
        var connectingMsg = Encoding.UTF8.GetBytes($"Connecting to {username}@{host}:{port}...\r\n");
        await webSocket.SendAsync(connectingMsg, WebSocketMessageType.Binary, true, CancellationToken.None);

        try
        {
            sshClient.Connect();
        }
        catch (Exception ex)
        {
            var errorMsg = Encoding.UTF8.GetBytes($"\x1b[31mFailed to connect: {ex.Message}\x1b[0m\r\n");
            await webSocket.SendAsync(errorMsg, WebSocketMessageType.Binary, true, CancellationToken.None);
            throw;
        }

        _logger.LogInformation("SSH connected to {User}@{Host}:{Port} for VM {VmId}", username, host, port, vmId);

        // Send connected message
        var connectedMsg = Encoding.UTF8.GetBytes($"\x1b[32mConnected!\x1b[0m\r\n\r\n");
        await webSocket.SendAsync(connectedMsg, WebSocketMessageType.Binary, true, CancellationToken.None);

        using var shellStream = sshClient.CreateShellStream(
            "xterm-256color",
            (uint)TerminalCols,
            (uint)TerminalRows,
            800,
            600,
            BufferSize);

        var cts = new CancellationTokenSource();

        // Task to read from SSH and send to WebSocket
        var sshToWsTask = Task.Run(async () =>
        {
            var buffer = new byte[BufferSize];
            try
            {
                while (!cts.Token.IsCancellationRequested && sshClient.IsConnected)
                {
                    if (shellStream.DataAvailable)
                    {
                        var bytesRead = await shellStream.ReadAsync(buffer, 0, buffer.Length, cts.Token);
                        if (bytesRead > 0 && webSocket.State == WebSocketState.Open)
                        {
                            await webSocket.SendAsync(
                                new ArraySegment<byte>(buffer, 0, bytesRead),
                                WebSocketMessageType.Binary,
                                true,
                                cts.Token);
                        }
                    }
                    else
                    {
                        await Task.Delay(10, cts.Token);
                    }
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "SSH to WebSocket stream ended");
            }
        }, cts.Token);

        // Task to read from WebSocket and send to SSH
        var wsToSshTask = Task.Run(async () =>
        {
            var buffer = new byte[BufferSize];
            try
            {
                while (!cts.Token.IsCancellationRequested && webSocket.State == WebSocketState.Open)
                {
                    var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);

                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        _logger.LogInformation("WebSocket closed by client for VM {VmId}", vmId);
                        break;
                    }

                    if (result.MessageType == WebSocketMessageType.Text || result.MessageType == WebSocketMessageType.Binary)
                    {
                        // Check for resize command (JSON: {"type":"resize","cols":120,"rows":40})
                        var data = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        if (data.StartsWith("{") && data.Contains("resize"))
                        {
                            try
                            {
                                var resizeCmd = System.Text.Json.JsonSerializer.Deserialize<ResizeCommand>(data);
                                if (resizeCmd?.Type == "resize" && resizeCmd.Cols > 0 && resizeCmd.Rows > 0)
                                {
                                    shellStream.ChangeWindowSize(
                                        (uint)resizeCmd.Cols,
                                        (uint)resizeCmd.Rows,
                                        800,
                                        600);
                                    _logger.LogDebug("Terminal resized to {Cols}x{Rows}", resizeCmd.Cols, resizeCmd.Rows);
                                    continue;
                                }
                            }
                            catch { /* Not a resize command, treat as regular input */ }
                        }

                        // Regular input - send to shell
                        if (result.Count > 0 && sshClient.IsConnected)
                        {
                            shellStream.Write(buffer, 0, result.Count);
                            shellStream.Flush();
                        }
                    }
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "WebSocket to SSH stream ended");
            }
        }, cts.Token);

        // Wait for either task to complete
        await Task.WhenAny(sshToWsTask, wsToSshTask);

        // Cancel the other task
        cts.Cancel();

        // Cleanup
        if (sshClient.IsConnected)
        {
            sshClient.Disconnect();
        }

        if (webSocket.State == WebSocketState.Open)
        {
            await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Session ended", CancellationToken.None);
        }

        _logger.LogInformation("SSH session ended for VM {VmId}", vmId);
    }

    private class ResizeCommand
    {
        [System.Text.Json.Serialization.JsonPropertyName("type")]
        public string Type { get; set; } = "";

        [System.Text.Json.Serialization.JsonPropertyName("cols")]
        public int Cols { get; set; }

        [System.Text.Json.Serialization.JsonPropertyName("rows")]
        public int Rows { get; set; }
    }
}