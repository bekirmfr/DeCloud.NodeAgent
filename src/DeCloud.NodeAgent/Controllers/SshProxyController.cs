using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using DeCloud.NodeAgent.Infrastructure.Services;
using Microsoft.AspNetCore.Mvc;
using Renci.SshNet;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// WebSocket-based SSH proxy for browser-based terminal access to VMs.
/// Supports ephemeral key injection for secure access.
/// </summary>
[ApiController]
[Route("api/vms")]
public class SshProxyController : ControllerBase
{
    private readonly IEphemeralSshKeyService _ephemeralKeyService;
    private readonly ILogger<SshProxyController> _logger;
    private readonly IConfiguration _configuration;

    private const int BufferSize = 4096;
    private const int TerminalCols = 120;
    private const int TerminalRows = 40;

    public SshProxyController(
        IEphemeralSshKeyService ephemeralKeyService,
        ILogger<SshProxyController> logger,
        IConfiguration configuration)
    {
        _ephemeralKeyService = ephemeralKeyService;
        _logger = logger;
        _configuration = configuration;
    }

    /// <summary>
    /// WebSocket endpoint for SSH proxy to a VM.
    /// Connect via: ws://node:5100/api/vms/{vmId}/terminal?ip={vmIp}&user={username}
    /// 
    /// Authentication methods (in order of preference):
    /// 1. ephemeral=true - Auto-generates and injects ephemeral key (most secure)
    /// 2. privateKey - Base64 encoded private key (via query or X-SSH-PrivateKey header)
    /// 3. password - Password auth (via query or X-SSH-Password header)
    /// </summary>
    [HttpGet("{vmId}/terminal")]
    public async Task Terminal(
        string vmId,
        [FromQuery] string ip,
        [FromQuery] string user = "ubuntu",
        [FromQuery] int port = 22,
        [FromQuery] string? password = null,
        [FromQuery] string? privateKey = null,
        [FromQuery] bool ephemeral = false)
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

        // Check for ephemeral key request header
        if (!ephemeral && HttpContext.Request.Headers.ContainsKey("X-Use-Ephemeral-Key"))
        {
            ephemeral = true;
        }

        _logger.LogInformation(
            "SSH terminal requested for VM {VmId} at {Ip}:{Port} as {User}, ephemeral={Ephemeral}",
            vmId, ip, port, user, ephemeral);

        using var webSocket = await HttpContext.WebSockets.AcceptWebSocketAsync();

        string? ephemeralFingerprint = null;

        try
        {
            // If ephemeral mode, generate and inject key
            if (ephemeral && string.IsNullOrEmpty(privateKey) && string.IsNullOrEmpty(password))
            {
                var setupMessage = Encoding.UTF8.GetBytes(
                    "\x1b[33m[DeCloud] Setting up secure terminal access...\x1b[0m\r\n");
                await webSocket.SendAsync(setupMessage, WebSocketMessageType.Binary, true, CancellationToken.None);

                try
                {
                    // Generate keypair
                    var keypair = _ephemeralKeyService.GenerateKeyPair($"terminal-{vmId[..8]}");

                    // Inject into VM
                    var injectResult = await _ephemeralKeyService.InjectKeyAsync(
                        vmId,
                        keypair.PublicKey,
                        user,
                        TimeSpan.FromMinutes(5));

                    if (!injectResult.Success)
                    {
                        var errorMsg = Encoding.UTF8.GetBytes(
                            $"\x1b[31m[DeCloud] Failed to setup access: {injectResult.Error}\x1b[0m\r\n" +
                            "\x1b[33m[DeCloud] Falling back to password auth if available...\x1b[0m\r\n");
                        await webSocket.SendAsync(errorMsg, WebSocketMessageType.Binary, true, CancellationToken.None);
                    }
                    else
                    {
                        privateKey = Convert.ToBase64String(Encoding.UTF8.GetBytes(keypair.PrivateKey));
                        ephemeralFingerprint = keypair.Fingerprint;

                        var successMsg = Encoding.UTF8.GetBytes(
                            $"\x1b[32m[DeCloud] Secure access established (expires in 5 min)\x1b[0m\r\n");
                        await webSocket.SendAsync(successMsg, WebSocketMessageType.Binary, true, CancellationToken.None);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Ephemeral key setup failed for VM {VmId}", vmId);
                    var errorMsg = Encoding.UTF8.GetBytes(
                        $"\x1b[31m[DeCloud] Key injection error: {ex.Message}\x1b[0m\r\n");
                    await webSocket.SendAsync(errorMsg, WebSocketMessageType.Binary, true, CancellationToken.None);
                }
            }

            await ProxySshConnection(webSocket, ip, port, user, password, privateKey, vmId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "SSH proxy error for VM {VmId}", vmId);

            if (webSocket.State == WebSocketState.Open)
            {
                var errorMessage = Encoding.UTF8.GetBytes(
                    $"\r\n\x1b[31mConnection error: {ex.Message}\x1b[0m\r\n");
                await webSocket.SendAsync(errorMessage, WebSocketMessageType.Binary, true, CancellationToken.None);
                await webSocket.CloseAsync(WebSocketCloseStatus.InternalServerError, ex.Message, CancellationToken.None);
            }
        }
        finally
        {
            // Cleanup ephemeral key
            if (!string.IsNullOrEmpty(ephemeralFingerprint))
            {
                try
                {
                    await _ephemeralKeyService.RemoveKeyAsync(vmId, ephemeralFingerprint, user);
                    _logger.LogDebug("Cleaned up ephemeral key {Fingerprint} from VM {VmId}",
                        ephemeralFingerprint, vmId);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to cleanup ephemeral key from VM {VmId}", vmId);
                }
            }
        }
    }

    /// <summary>
    /// Endpoint for orchestrator to initiate terminal with ephemeral key.
    /// Returns the WebSocket URL with credentials embedded.
    /// </summary>
    [HttpPost("{vmId}/terminal/connect")]
    public async Task<ActionResult<TerminalConnectResponse>> InitiateTerminal(
    string vmId,
    [FromBody] TerminalConnectRequest request)
    {
        _logger.LogInformation("Initiating terminal connection for VM {VmId}", vmId);

        try
        {
            // Generate and inject ephemeral key
            var keypair = _ephemeralKeyService.GenerateKeyPair($"terminal-{vmId[..8]}");

            var injectResult = await _ephemeralKeyService.InjectKeyAsync(
                vmId,
                keypair.PublicKey,
                request.Username ?? "ubuntu",
                TimeSpan.FromSeconds(request.TtlSeconds > 0 ? request.TtlSeconds : 300));

            if (injectResult.Success)
            {
                // Build WebSocket URL with private key
                var privateKeyBase64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(keypair.PrivateKey));
                var wsUrl = $"/api/vms/{vmId}/terminal?ip={request.VmIp}&user={request.Username ?? "ubuntu"}&port={request.Port}";

                return Ok(new TerminalConnectResponse
                {
                    Success = true,
                    WebSocketPath = wsUrl,
                    PrivateKey = keypair.PrivateKey,
                    PrivateKeyBase64 = privateKeyBase64,
                    Fingerprint = keypair.Fingerprint,
                    ExpiresAt = injectResult.ExpiresAt,
                    MethodUsed = injectResult.MethodUsed.ToString()
                });
            }

            // Ephemeral key failed - fall back to password if available
            _logger.LogWarning("Ephemeral key injection failed for VM {VmId}: {Error}. Checking for password fallback.",
                vmId, injectResult.Error);

            if (!string.IsNullOrEmpty(request.Password))
            {
                _logger.LogInformation("Using password fallback for VM {VmId}", vmId);
                var wsUrl = $"/api/vms/{vmId}/terminal?ip={request.VmIp}&user={request.Username ?? "ubuntu"}&port={request.Port}";

                return Ok(new TerminalConnectResponse
                {
                    Success = true,
                    WebSocketPath = wsUrl,
                    Password = request.Password,
                    MethodUsed = "PasswordFallback"
                });
            }

            // No fallback available
            return StatusCode(500, new TerminalConnectResponse
            {
                Success = false,
                Error = $"Failed to inject key: {injectResult.Error}. No password fallback available."
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initiate terminal for VM {VmId}", vmId);
            return StatusCode(500, new TerminalConnectResponse
            {
                Success = false,
                Error = ex.Message
            });
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
        var authMethods = new List<AuthenticationMethod>();

        // Decode and add private key auth
        if (!string.IsNullOrEmpty(privateKeyBase64))
        {
            try
            {
                var keyBytes = Convert.FromBase64String(privateKeyBase64);
                var keyString = Encoding.UTF8.GetString(keyBytes);

                using var keyStream = new MemoryStream(Encoding.UTF8.GetBytes(keyString));
                var privateKeyFile = new PrivateKeyFile(keyStream);
                authMethods.Add(new PrivateKeyAuthenticationMethod(username, privateKeyFile));

                _logger.LogDebug("Using private key authentication for {User}@{Host}", username, host);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to parse private key, trying password if available");
            }
        }

        // Add password auth as fallback
        if (!string.IsNullOrEmpty(password))
        {
            authMethods.Add(new PasswordAuthenticationMethod(username, password));
            _logger.LogDebug("Using password authentication for {User}@{Host}", username, host);
        }

        if (!authMethods.Any())
        {
            throw new InvalidOperationException(
                "No authentication method available. Provide password or privateKey.");
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
                while (!cts.Token.IsCancellationRequested && webSocket.State == WebSocketState.Open)
                {
                    var bytesRead = await shellStream.ReadAsync(buffer, 0, buffer.Length, cts.Token);
                    if (bytesRead > 0)
                    {
                        await webSocket.SendAsync(
                            new ArraySegment<byte>(buffer, 0, bytesRead),
                            WebSocketMessageType.Binary,
                            true,
                            cts.Token);
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
                    var result = await webSocket.ReceiveAsync(buffer, cts.Token);

                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        break;
                    }

                    if (result.Count > 0)
                    {
                        var data = Encoding.UTF8.GetString(buffer, 0, result.Count);

                        // Check for resize command
                        if (data.StartsWith("{") && data.Contains("resize"))
                        {
                            try
                            {
                                var resize = JsonSerializer.Deserialize<ResizeCommand>(data);
                                if (resize?.Type == "resize" && resize.Cols > 0 && resize.Rows > 0)
                                {
                                    shellStream.ChangeWindowSize(
                                        (uint)resize.Cols,
                                        (uint)resize.Rows,
                                        800,
                                        600);
                                }
                            }
                            catch { }
                            continue;
                        }

                        shellStream.Write(data);
                        shellStream.Flush();
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

        // Clean close
        if (webSocket.State == WebSocketState.Open)
        {
            await webSocket.CloseAsync(
                WebSocketCloseStatus.NormalClosure,
                "Session ended",
                CancellationToken.None);
        }
    }

    private class ResizeCommand
    {
        public string Type { get; set; } = "";
        public int Cols { get; set; }
        public int Rows { get; set; }
    }
}

public class TerminalConnectRequest
{
    public string VmIp { get; init; } = "";
    public string? Username { get; init; }
    public int Port { get; init; } = 22;
    public int TtlSeconds { get; init; } = 300;
    public string? Password { get; init; }
}

public class TerminalConnectResponse
{
    public bool Success { get; init; }
    public string? Error { get; init; }
    public string WebSocketPath { get; init; } = "";
    public string PrivateKey { get; init; } = "";
    public string PrivateKeyBase64 { get; init; } = "";
    public string Fingerprint { get; init; } = "";
    public DateTime? ExpiresAt { get; init; }
    public string? MethodUsed { get; init; }
    public string? Password { get; init; }
}