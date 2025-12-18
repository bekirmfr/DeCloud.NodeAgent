using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using DeCloud.NodeAgent.Infrastructure.Services;
using Microsoft.AspNetCore.Mvc;
using Renci.SshNet;
using Renci.SshNet.Sftp;
using ConnectionInfo = Renci.SshNet.ConnectionInfo;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// WebSocket-based SFTP proxy for browser-based file management on VMs.
/// Provides secure file browsing, upload, and download capabilities.
/// Reuses existing SSH authentication (password or ephemeral keys).
/// </summary>
[ApiController]
[Route("api/vms")]
public class SftpProxyController : ControllerBase
{
    private readonly IEphemeralSshKeyService _ephemeralKeyService;
    private readonly ILogger<SftpProxyController> _logger;
    private readonly IConfiguration _configuration;

    private const int ChunkSize = 64 * 1024; // 64KB chunks for file transfer
    private const long MaxUploadSize = 500 * 1024 * 1024; // 500MB max file size

    // Reuse ephemeral key cache from SshProxyController
    private static readonly ConcurrentDictionary<string, (string PrivateKey, DateTime ExpiresAt)>
        _ephemeralKeyCache = new();

    public SftpProxyController(
        IEphemeralSshKeyService ephemeralKeyService,
        ILogger<SftpProxyController> logger,
        IConfiguration configuration)
    {
        _ephemeralKeyService = ephemeralKeyService;
        _logger = logger;
        _configuration = configuration;
    }

    /// <summary>
    /// WebSocket endpoint for SFTP operations on a VM.
    /// Connect via: ws://node:5100/api/vms/{vmId}/sftp?ip={vmIp}&user={username}&password={password}
    /// 
    /// Supports operations: list, download, upload, mkdir, delete, rename, stat
    /// </summary>
    [HttpGet("{vmId}/sftp")]
    public async Task SftpProxy(
        string vmId,
        [FromQuery] string ip,
        [FromQuery] string user = "ubuntu",
        [FromQuery] int port = 22,
        [FromQuery] string? password = null,
        [FromQuery] string? privateKey = null,
        [FromQuery] string? keyRef = null)
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

        // Get authentication credentials (same logic as SshProxyController)
        password ??= HttpContext.Request.Headers["X-SSH-Password"].FirstOrDefault();
        privateKey ??= HttpContext.Request.Headers["X-SSH-PrivateKey"].FirstOrDefault();

        // Check ephemeral key cache
        if (!string.IsNullOrEmpty(keyRef) && _ephemeralKeyCache.TryGetValue(keyRef, out var cached))
        {
            if (DateTime.UtcNow < cached.ExpiresAt)
            {
                privateKey = Convert.ToBase64String(Encoding.UTF8.GetBytes(cached.PrivateKey));
                _logger.LogDebug("Using cached ephemeral key for SFTP: {KeyRef}", keyRef);
            }
            else
            {
                _ephemeralKeyCache.TryRemove(keyRef, out _);
                _logger.LogDebug("Ephemeral key expired: {KeyRef}", keyRef);
            }
        }

        if (string.IsNullOrEmpty(password) && string.IsNullOrEmpty(privateKey))
        {
            HttpContext.Response.StatusCode = 401;
            await HttpContext.Response.WriteAsync("Authentication required (password or privateKey)");
            return;
        }

        var webSocket = await HttpContext.WebSockets.AcceptWebSocketAsync();
        _logger.LogInformation("SFTP WebSocket connected for VM {VmId} at {Ip}:{Port}", vmId, ip, port);

        await HandleSftpSession(webSocket, ip, port, user, password, privateKey, vmId);
    }

    private async Task HandleSftpSession(
        WebSocket webSocket,
        string host,
        int port,
        string username,
        string? password,
        string? privateKeyBase64,
        string vmId)
    {
        SftpClient? sftpClient = null;

        try
        {
            // Setup authentication methods
            var authMethods = new List<AuthenticationMethod>();

            if (!string.IsNullOrEmpty(privateKeyBase64))
            {
                try
                {
                    var keyBytes = Convert.FromBase64String(privateKeyBase64);
                    var keyString = Encoding.UTF8.GetString(keyBytes);
                    using var keyStream = new MemoryStream(Encoding.UTF8.GetBytes(keyString));
                    var privateKeyFile = new PrivateKeyFile(keyStream);
                    authMethods.Add(new PrivateKeyAuthenticationMethod(username, privateKeyFile));
                    _logger.LogDebug("SFTP using private key auth for {User}@{Host}", username, host);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "SFTP failed to parse private key");
                }
            }

            if (!string.IsNullOrEmpty(password))
            {
                authMethods.Add(new PasswordAuthenticationMethod(username, password));
                _logger.LogDebug("SFTP using password auth for {User}@{Host}", username, host);
            }

            if (authMethods.Count == 0)
            {
                await SendErrorAsync(webSocket, "NO_AUTH", "No authentication method available");
                return;
            }

            // Connect SFTP
            var connectionInfo = new ConnectionInfo(host, port, username, authMethods.ToArray());
            connectionInfo.Timeout = TimeSpan.FromSeconds(30);

            sftpClient = new SftpClient(connectionInfo);
            sftpClient.Connect();

            _logger.LogInformation("SFTP connected to {User}@{Host}:{Port} for VM {VmId}",
                username, host, port, vmId);

            // Send connected confirmation
            await SendResponseAsync(webSocket, new SftpResponse
            {
                Type = "connected",
                Success = true,
                Message = "SFTP session established"
            });

            // Handle commands
            var buffer = new byte[ChunkSize * 2]; // Buffer for receiving data
            var messageBuffer = new List<byte>();

            while (webSocket.State == WebSocketState.Open)
            {
                var result = await webSocket.ReceiveAsync(
                    new ArraySegment<byte>(buffer),
                    CancellationToken.None);

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    _logger.LogDebug("SFTP WebSocket close received for VM {VmId}", vmId);
                    break;
                }

                messageBuffer.AddRange(buffer.Take(result.Count));

                if (result.EndOfMessage)
                {
                    var messageData = messageBuffer.ToArray();
                    messageBuffer.Clear();

                    await ProcessCommandAsync(webSocket, sftpClient, messageData, vmId);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "SFTP session error for VM {VmId}", vmId);
            await SendErrorAsync(webSocket, "SESSION_ERROR", ex.Message);
        }
        finally
        {
            sftpClient?.Disconnect();
            sftpClient?.Dispose();

            if (webSocket.State == WebSocketState.Open)
            {
                try
                {
                    await webSocket.CloseAsync(
                        WebSocketCloseStatus.NormalClosure,
                        "Session ended",
                        CancellationToken.None);
                }
                catch { }
            }

            _logger.LogInformation("SFTP session ended for VM {VmId}", vmId);
        }
    }

    private async Task ProcessCommandAsync(
        WebSocket webSocket,
        SftpClient sftp,
        byte[] data,
        string vmId)
    {
        try
        {
            // Check if this is a binary upload chunk or JSON command
            if (data.Length > 0 && data[0] != '{')
            {
                // Binary data - this is an upload chunk, handled separately
                return;
            }

            var json = Encoding.UTF8.GetString(data);
            var command = JsonSerializer.Deserialize<SftpCommand>(json, JsonOptions);

            if (command == null)
            {
                await SendErrorAsync(webSocket, "INVALID_COMMAND", "Failed to parse command");
                return;
            }

            _logger.LogDebug("SFTP command: {Type} for VM {VmId}", command.Type, vmId);

            switch (command.Type?.ToLowerInvariant())
            {
                case "list":
                    await HandleListAsync(webSocket, sftp, command);
                    break;

                case "download":
                    await HandleDownloadAsync(webSocket, sftp, command);
                    break;

                case "upload_start":
                    await HandleUploadStartAsync(webSocket, sftp, command);
                    break;

                case "upload_chunk":
                    await HandleUploadChunkAsync(webSocket, sftp, command);
                    break;

                case "upload_complete":
                    await HandleUploadCompleteAsync(webSocket, sftp, command);
                    break;

                case "mkdir":
                    await HandleMkdirAsync(webSocket, sftp, command);
                    break;

                case "delete":
                    await HandleDeleteAsync(webSocket, sftp, command);
                    break;

                case "rename":
                    await HandleRenameAsync(webSocket, sftp, command);
                    break;

                case "stat":
                    await HandleStatAsync(webSocket, sftp, command);
                    break;

                case "pwd":
                    await HandlePwdAsync(webSocket, sftp);
                    break;

                default:
                    await SendErrorAsync(webSocket, "UNKNOWN_COMMAND", $"Unknown command: {command.Type}");
                    break;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "SFTP command processing error for VM {VmId}", vmId);
            await SendErrorAsync(webSocket, "COMMAND_ERROR", ex.Message);
        }
    }

    private async Task HandleListAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        var path = SanitizePath(command.Path ?? "/home");

        if (!sftp.Exists(path))
        {
            await SendErrorAsync(webSocket, "PATH_NOT_FOUND", $"Path does not exist: {path}");
            return;
        }

        var entries = new List<SftpFileEntry>();

        foreach (var file in sftp.ListDirectory(path))
        {
            // Skip . and .. entries
            if (file.Name == "." || file.Name == "..") continue;

            entries.Add(new SftpFileEntry
            {
                Name = file.Name,
                Path = file.FullName,
                IsDirectory = file.IsDirectory,
                IsSymbolicLink = file.IsSymbolicLink,
                Size = file.Length,
                Modified = file.LastWriteTime,
                Permissions = GetPermissionString(file),
                Owner = file.UserId.ToString(),
                Group = file.GroupId.ToString()
            });
        }

        // Sort: directories first, then by name
        entries = entries
            .OrderByDescending(e => e.IsDirectory)
            .ThenBy(e => e.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();

        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "list",
            Success = true,
            Path = path,
            Files = entries
        });
    }

    private async Task HandleDownloadAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        var path = SanitizePath(command.Path);

        if (string.IsNullOrEmpty(path))
        {
            await SendErrorAsync(webSocket, "INVALID_PATH", "Path is required");
            return;
        }

        if (!sftp.Exists(path))
        {
            await SendErrorAsync(webSocket, "FILE_NOT_FOUND", $"File not found: {path}");
            return;
        }

        var fileInfo = sftp.Get(path);
        if (fileInfo.IsDirectory)
        {
            await SendErrorAsync(webSocket, "IS_DIRECTORY", "Cannot download a directory");
            return;
        }

        var fileSize = fileInfo.Length;
        var fileName = Path.GetFileName(path);

        // Send download start
        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "download_start",
            Success = true,
            Path = path,
            FileName = fileName,
            FileSize = fileSize,
            TotalChunks = (int)Math.Ceiling((double)fileSize / ChunkSize)
        });

        // Stream file in chunks
        using var stream = sftp.OpenRead(path);
        var buffer = new byte[ChunkSize];
        int bytesRead;
        int chunkIndex = 0;
        long totalSent = 0;

        while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
        {
            var chunk = new byte[bytesRead];
            Array.Copy(buffer, chunk, bytesRead);

            await SendResponseAsync(webSocket, new SftpResponse
            {
                Type = "download_chunk",
                Success = true,
                Path = path,
                ChunkIndex = chunkIndex,
                ChunkData = Convert.ToBase64String(chunk),
                BytesSent = totalSent + bytesRead
            });

            totalSent += bytesRead;
            chunkIndex++;
        }

        // Send download complete
        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "download_complete",
            Success = true,
            Path = path,
            FileName = fileName,
            FileSize = fileSize
        });
    }

    // Track active uploads
    private static readonly ConcurrentDictionary<string, UploadSession> _uploadSessions = new();

    private async Task HandleUploadStartAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        var path = SanitizePath(command.Path);
        var fileSize = command.FileSize ?? 0;

        if (string.IsNullOrEmpty(path))
        {
            await SendErrorAsync(webSocket, "INVALID_PATH", "Path is required");
            return;
        }

        if (fileSize > MaxUploadSize)
        {
            await SendErrorAsync(webSocket, "FILE_TOO_LARGE",
                $"File exceeds maximum size of {MaxUploadSize / 1024 / 1024}MB");
            return;
        }

        // Ensure parent directory exists
        var parentDir = Path.GetDirectoryName(path)?.Replace("\\", "/");
        if (!string.IsNullOrEmpty(parentDir) && !sftp.Exists(parentDir))
        {
            await SendErrorAsync(webSocket, "PARENT_NOT_FOUND",
                $"Parent directory does not exist: {parentDir}");
            return;
        }

        // Create upload session
        var sessionId = Guid.NewGuid().ToString("N")[..8];
        var session = new UploadSession
        {
            SessionId = sessionId,
            Path = path,
            ExpectedSize = fileSize,
            StartedAt = DateTime.UtcNow,
            TempStream = new MemoryStream()
        };

        _uploadSessions[sessionId] = session;

        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "upload_ready",
            Success = true,
            SessionId = sessionId,
            Path = path
        });
    }

    private async Task HandleUploadChunkAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        if (string.IsNullOrEmpty(command.SessionId) ||
            !_uploadSessions.TryGetValue(command.SessionId, out var session))
        {
            await SendErrorAsync(webSocket, "INVALID_SESSION", "Upload session not found");
            return;
        }

        if (string.IsNullOrEmpty(command.ChunkData))
        {
            await SendErrorAsync(webSocket, "INVALID_CHUNK", "Chunk data is required");
            return;
        }

        try
        {
            var chunkBytes = Convert.FromBase64String(command.ChunkData);
            await session.TempStream.WriteAsync(chunkBytes);
            session.BytesReceived += chunkBytes.Length;

            await SendResponseAsync(webSocket, new SftpResponse
            {
                Type = "upload_progress",
                Success = true,
                SessionId = command.SessionId,
                BytesReceived = session.BytesReceived,
                ExpectedSize = session.ExpectedSize
            });
        }
        catch (Exception ex)
        {
            await SendErrorAsync(webSocket, "CHUNK_ERROR", $"Failed to process chunk: {ex.Message}");
        }
    }

    private async Task HandleUploadCompleteAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        if (string.IsNullOrEmpty(command.SessionId) ||
            !_uploadSessions.TryRemove(command.SessionId, out var session))
        {
            await SendErrorAsync(webSocket, "INVALID_SESSION", "Upload session not found");
            return;
        }

        try
        {
            // Reset stream position and write to SFTP
            session.TempStream.Position = 0;

            using var sftpStream = sftp.Create(session.Path);
            await session.TempStream.CopyToAsync(sftpStream);

            session.TempStream.Dispose();

            await SendResponseAsync(webSocket, new SftpResponse
            {
                Type = "upload_complete",
                Success = true,
                Path = session.Path,
                FileSize = session.BytesReceived
            });
        }
        catch (Exception ex)
        {
            session.TempStream.Dispose();
            await SendErrorAsync(webSocket, "UPLOAD_ERROR", $"Failed to write file: {ex.Message}");
        }
    }

    private async Task HandleMkdirAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        var path = SanitizePath(command.Path);

        if (string.IsNullOrEmpty(path))
        {
            await SendErrorAsync(webSocket, "INVALID_PATH", "Path is required");
            return;
        }

        if (sftp.Exists(path))
        {
            await SendErrorAsync(webSocket, "ALREADY_EXISTS", $"Path already exists: {path}");
            return;
        }

        sftp.CreateDirectory(path);

        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "mkdir",
            Success = true,
            Path = path
        });
    }

    private async Task HandleDeleteAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        var path = SanitizePath(command.Path);

        if (string.IsNullOrEmpty(path))
        {
            await SendErrorAsync(webSocket, "INVALID_PATH", "Path is required");
            return;
        }

        // Security: Prevent deletion of critical system paths
        var protectedPaths = new[] { "/", "/etc", "/usr", "/bin", "/sbin", "/boot", "/var", "/root" };
        if (protectedPaths.Contains(path))
        {
            await SendErrorAsync(webSocket, "PROTECTED_PATH", "Cannot delete protected system path");
            return;
        }

        if (!sftp.Exists(path))
        {
            await SendErrorAsync(webSocket, "NOT_FOUND", $"Path not found: {path}");
            return;
        }

        var fileInfo = sftp.Get(path);
        if (fileInfo.IsDirectory)
        {
            // Recursive delete for directories
            DeleteDirectoryRecursive(sftp, path);
        }
        else
        {
            sftp.DeleteFile(path);
        }

        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "delete",
            Success = true,
            Path = path
        });
    }

    private void DeleteDirectoryRecursive(SftpClient sftp, string path)
    {
        foreach (var file in sftp.ListDirectory(path))
        {
            if (file.Name == "." || file.Name == "..") continue;

            if (file.IsDirectory)
            {
                DeleteDirectoryRecursive(sftp, file.FullName);
            }
            else
            {
                sftp.DeleteFile(file.FullName);
            }
        }
        sftp.DeleteDirectory(path);
    }

    private async Task HandleRenameAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        var oldPath = SanitizePath(command.Path);
        var newPath = SanitizePath(command.NewPath);

        if (string.IsNullOrEmpty(oldPath) || string.IsNullOrEmpty(newPath))
        {
            await SendErrorAsync(webSocket, "INVALID_PATH", "Both path and newPath are required");
            return;
        }

        if (!sftp.Exists(oldPath))
        {
            await SendErrorAsync(webSocket, "NOT_FOUND", $"Source path not found: {oldPath}");
            return;
        }

        if (sftp.Exists(newPath))
        {
            await SendErrorAsync(webSocket, "ALREADY_EXISTS", $"Destination already exists: {newPath}");
            return;
        }

        sftp.RenameFile(oldPath, newPath);

        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "rename",
            Success = true,
            Path = oldPath,
            NewPath = newPath
        });
    }

    private async Task HandleStatAsync(WebSocket webSocket, SftpClient sftp, SftpCommand command)
    {
        var path = SanitizePath(command.Path);

        if (string.IsNullOrEmpty(path))
        {
            await SendErrorAsync(webSocket, "INVALID_PATH", "Path is required");
            return;
        }

        if (!sftp.Exists(path))
        {
            await SendErrorAsync(webSocket, "NOT_FOUND", $"Path not found: {path}");
            return;
        }

        var file = sftp.Get(path);

        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "stat",
            Success = true,
            Path = path,
            File = new SftpFileEntry
            {
                Name = file.Name,
                Path = file.FullName,
                IsDirectory = file.IsDirectory,
                IsSymbolicLink = file.IsSymbolicLink,
                Size = file.Length,
                Modified = file.LastWriteTime,
                Permissions = GetPermissionString(file),
                Owner = file.UserId.ToString(),
                Group = file.GroupId.ToString()
            }
        });
    }

    private async Task HandlePwdAsync(WebSocket webSocket, SftpClient sftp)
    {
        var pwd = sftp.WorkingDirectory;

        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "pwd",
            Success = true,
            Path = pwd
        });
    }

    private static string SanitizePath(string? path)
    {
        if (string.IsNullOrEmpty(path)) return "/home";

        // Normalize path separators
        path = path.Replace("\\", "/");

        // Remove null bytes and other dangerous characters
        path = path.Replace("\0", "");

        // Resolve .. to prevent path traversal
        var parts = path.Split('/', StringSplitOptions.RemoveEmptyEntries);
        var stack = new Stack<string>();

        foreach (var part in parts)
        {
            if (part == "..")
            {
                if (stack.Count > 0) stack.Pop();
            }
            else if (part != ".")
            {
                stack.Push(part);
            }
        }

        var result = "/" + string.Join("/", stack.Reverse());
        return result;
    }

    private static string GetPermissionString(ISftpFile file)
    {
        var perms = new char[10];

        // File type
        perms[0] = file.IsDirectory ? 'd' : file.IsSymbolicLink ? 'l' : '-';

        // Owner permissions
        perms[1] = file.OwnerCanRead ? 'r' : '-';
        perms[2] = file.OwnerCanWrite ? 'w' : '-';
        perms[3] = file.OwnerCanExecute ? 'x' : '-';

        // Group permissions
        perms[4] = file.GroupCanRead ? 'r' : '-';
        perms[5] = file.GroupCanWrite ? 'w' : '-';
        perms[6] = file.GroupCanExecute ? 'x' : '-';

        // Others permissions
        perms[7] = file.OthersCanRead ? 'r' : '-';
        perms[8] = file.OthersCanWrite ? 'w' : '-';
        perms[9] = file.OthersCanExecute ? 'x' : '-';

        return new string(perms);
    }

    private async Task SendResponseAsync(WebSocket webSocket, SftpResponse response)
    {
        if (webSocket.State != WebSocketState.Open) return;

        var json = JsonSerializer.Serialize(response, JsonOptions);
        var bytes = Encoding.UTF8.GetBytes(json);
        await webSocket.SendAsync(bytes, WebSocketMessageType.Text, true, CancellationToken.None);
    }

    private async Task SendErrorAsync(WebSocket webSocket, string code, string message)
    {
        await SendResponseAsync(webSocket, new SftpResponse
        {
            Type = "error",
            Success = false,
            ErrorCode = code,
            Message = message
        });
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };
}

#region DTOs

public class SftpCommand
{
    public string? Type { get; set; }
    public string? Path { get; set; }
    public string? NewPath { get; set; }
    public long? FileSize { get; set; }
    public string? SessionId { get; set; }
    public string? ChunkData { get; set; }
    public int? ChunkIndex { get; set; }
}

public class SftpResponse
{
    public string Type { get; set; } = "";
    public bool Success { get; set; }
    public string? Message { get; set; }
    public string? ErrorCode { get; set; }
    public string? Path { get; set; }
    public string? NewPath { get; set; }
    public string? FileName { get; set; }
    public long? FileSize { get; set; }
    public int? TotalChunks { get; set; }
    public int? ChunkIndex { get; set; }
    public string? ChunkData { get; set; }
    public long? BytesSent { get; set; }
    public long? BytesReceived { get; set; }
    public long? ExpectedSize { get; set; }
    public string? SessionId { get; set; }
    public List<SftpFileEntry>? Files { get; set; }
    public SftpFileEntry? File { get; set; }
}

public class SftpFileEntry
{
    public string Name { get; set; } = "";
    public string Path { get; set; } = "";
    public bool IsDirectory { get; set; }
    public bool IsSymbolicLink { get; set; }
    public long Size { get; set; }
    public DateTime Modified { get; set; }
    public string Permissions { get; set; } = "";
    public string Owner { get; set; } = "";
    public string Group { get; set; } = "";
}

public class UploadSession
{
    public string SessionId { get; set; } = "";
    public string Path { get; set; } = "";
    public long ExpectedSize { get; set; }
    public long BytesReceived { get; set; }
    public DateTime StartedAt { get; set; }
    public MemoryStream TempStream { get; set; } = new();
}

#endregion