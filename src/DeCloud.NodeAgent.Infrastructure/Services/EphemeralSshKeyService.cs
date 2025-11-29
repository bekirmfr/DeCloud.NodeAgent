using System.Security.Cryptography;
using System.Text;
using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Service for injecting ephemeral SSH keys into running VMs.
/// Supports multiple injection methods with fallback.
/// </summary>
public interface IEphemeralSshKeyService
{
    /// <summary>
    /// Inject a temporary SSH public key into a running VM.
    /// The key will be added to the specified user's authorized_keys.
    /// </summary>
    Task<SshKeyInjectionResult> InjectKeyAsync(
        string vmId,
        string publicKey,
        string username = "ubuntu",
        TimeSpan? ttl = null,
        CancellationToken ct = default);

    /// <summary>
    /// Remove an injected SSH key from a VM.
    /// </summary>
    Task<bool> RemoveKeyAsync(
        string vmId,
        string publicKeyFingerprint,
        string username = "ubuntu",
        CancellationToken ct = default);

    /// <summary>
    /// Generate a new ephemeral Ed25519 keypair.
    /// </summary>
    EphemeralKeyPair GenerateKeyPair(string comment = "decloud-terminal");
}

public class SshKeyInjectionResult
{
    public bool Success { get; init; }
    public string? Error { get; init; }
    public InjectionMethod MethodUsed { get; init; }
    public string? KeyFingerprint { get; init; }
    public DateTime? ExpiresAt { get; init; }

    public static SshKeyInjectionResult Ok(InjectionMethod method, string fingerprint, DateTime? expiresAt = null)
        => new() { Success = true, MethodUsed = method, KeyFingerprint = fingerprint, ExpiresAt = expiresAt };

    public static SshKeyInjectionResult Fail(string error)
        => new() { Success = false, Error = error, MethodUsed = InjectionMethod.None };
}

public enum InjectionMethod
{
    None,
    QemuGuestAgent,
    VirtCustomize,
    SshExec
}

public class EphemeralKeyPair
{
    public string PublicKey { get; init; } = "";
    public string PrivateKey { get; init; } = "";
    public string Fingerprint { get; init; } = "";
    public DateTime GeneratedAt { get; init; } = DateTime.UtcNow;
}

public class EphemeralSshKeyService : IEphemeralSshKeyService
{
    private readonly ICommandExecutor _executor;
    private readonly IVmManager _vmManager;
    private readonly ILogger<EphemeralSshKeyService> _logger;

    // Track injected keys for cleanup
    private readonly Dictionary<string, List<InjectedKey>> _injectedKeys = new();
    private readonly object _lock = new();

    public EphemeralSshKeyService(
        ICommandExecutor executor,
        IVmManager vmManager,
        ILogger<EphemeralSshKeyService> logger)
    {
        _executor = executor;
        _vmManager = vmManager;
        _logger = logger;
    }

    public EphemeralKeyPair GenerateKeyPair(string comment = "decloud-terminal")
    {
        // Generate Ed25519 keypair using ssh-keygen
        var tempDir = Path.Combine(Path.GetTempPath(), $"decloud-keygen-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            var keyPath = Path.Combine(tempDir, "ephemeral_key");

            // Generate key with no passphrase
            var result = _executor.ExecuteAsync(
                "ssh-keygen",
                $"-t ed25519 -f {keyPath} -N \"\" -C \"{comment}\" -q",
                TimeSpan.FromSeconds(10),
                CancellationToken.None).GetAwaiter().GetResult();

            if (!result.Success)
            {
                _logger.LogError("Failed to generate SSH key: {Error}", result.StandardError);
                throw new Exception($"ssh-keygen failed: {result.StandardError}");
            }

            var privateKey = File.ReadAllText(keyPath);
            var publicKey = File.ReadAllText(keyPath + ".pub").Trim();

            // Calculate fingerprint
            var fingerprint = CalculateFingerprint(publicKey);

            _logger.LogDebug("Generated ephemeral keypair with fingerprint {Fingerprint}", fingerprint);

            return new EphemeralKeyPair
            {
                PublicKey = publicKey,
                PrivateKey = privateKey,
                Fingerprint = fingerprint,
                GeneratedAt = DateTime.UtcNow
            };
        }
        finally
        {
            // Secure cleanup
            try
            {
                foreach (var file in Directory.GetFiles(tempDir))
                {
                    // Overwrite with zeros before deleting
                    var length = new FileInfo(file).Length;
                    File.WriteAllBytes(file, new byte[length]);
                    File.Delete(file);
                }
                Directory.Delete(tempDir, true);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to cleanup temp key directory");
            }
        }
    }

    public async Task<SshKeyInjectionResult> InjectKeyAsync(
        string vmId,
        string publicKey,
        string username = "ubuntu",
        TimeSpan? ttl = null,
        CancellationToken ct = default)
    {
        _logger.LogInformation("Injecting ephemeral SSH key into VM {VmId} for user {Username}", vmId, username);

        // Validate VM exists and is running
        var vm = await _vmManager.GetVmAsync(vmId, ct);
        if (vm == null)
        {
            return SshKeyInjectionResult.Fail($"VM {vmId} not found");
        }

        if (vm.State != Core.Models.VmState.Running)
        {
            return SshKeyInjectionResult.Fail($"VM {vmId} is not running (state: {vm.State})");
        }

        var fingerprint = CalculateFingerprint(publicKey);
        var expiresAt = ttl.HasValue ? DateTime.UtcNow.Add(ttl.Value) : (DateTime?)null;

        // Try injection methods in order of preference
        var methods = new Func<string, string, string, CancellationToken, Task<bool>>[]
        {
            TryQemuGuestAgentAsync,
            TryVirtCustomizeAsync,
            TrySshExecAsync
        };

        var methodNames = new[] { InjectionMethod.QemuGuestAgent, InjectionMethod.VirtCustomize, InjectionMethod.SshExec };

        for (int i = 0; i < methods.Length; i++)
        {
            try
            {
                if (await methods[i](vmId, publicKey, username, ct))
                {
                    _logger.LogInformation(
                        "Successfully injected SSH key into VM {VmId} using {Method}",
                        vmId, methodNames[i]);

                    // Track for cleanup
                    TrackInjectedKey(vmId, fingerprint, username, expiresAt);

                    return SshKeyInjectionResult.Ok(methodNames[i], fingerprint, expiresAt);
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Injection method {Method} failed for VM {VmId}", methodNames[i], vmId);
            }
        }

        return SshKeyInjectionResult.Fail("All injection methods failed");
    }

    public async Task<bool> RemoveKeyAsync(
        string vmId,
        string publicKeyFingerprint,
        string username = "ubuntu",
        CancellationToken ct = default)
    {
        _logger.LogInformation("Removing SSH key {Fingerprint} from VM {VmId}", publicKeyFingerprint, vmId);

        // Try to remove via QEMU Guest Agent first
        var script = $@"
            sed -i '/{publicKeyFingerprint}/d' /home/{username}/.ssh/authorized_keys 2>/dev/null || true
        ";

        var result = await ExecuteInVmAsync(vmId, script, ct);

        // Remove from tracking
        lock (_lock)
        {
            if (_injectedKeys.TryGetValue(vmId, out var keys))
            {
                keys.RemoveAll(k => k.Fingerprint == publicKeyFingerprint);
            }
        }

        return result;
    }

    /// <summary>
    /// Method 1: QEMU Guest Agent (fastest, most reliable for running VMs)
    /// </summary>
    private async Task<bool> TryQemuGuestAgentAsync(
        string vmId,
        string publicKey,
        string username,
        CancellationToken ct)
    {
        _logger.LogDebug("Trying QEMU Guest Agent injection for VM {VmId}", vmId);

        // Check if guest agent is available
        var checkResult = await _executor.ExecuteAsync(
            "bash",
            $"-c \"virsh qemu-agent-command {vmId} '{{\\\"execute\\\":\\\"guest-ping\\\"}}' \"",
            TimeSpan.FromSeconds(5),
            ct);

        if (!checkResult.Success)
        {
            _logger.LogDebug("QEMU Guest Agent not available for VM {VmId}", vmId);
            return false;
        }

        // Escape the public key for JSON
        var escapedKey = publicKey.Replace("\"", "\\\"");

        // Create script to add key
        var script = $@"
mkdir -p /home/{username}/.ssh && \
chmod 700 /home/{username}/.ssh && \
echo '{publicKey}' >> /home/{username}/.ssh/authorized_keys && \
chmod 600 /home/{username}/.ssh/authorized_keys && \
chown -R {username}:{username} /home/{username}/.ssh
";
        var base64Script = Convert.ToBase64String(Encoding.UTF8.GetBytes(script));

        // Execute via guest-exec
        var execCmd = $@"{{""execute"":""guest-exec"",""arguments"":{{""path"":""/bin/bash"",""arg"":[""-c"",""echo {base64Script} | base64 -d | bash""],""capture-output"":true}}}}";

        var execResult = await _executor.ExecuteAsync(
            "bash",
            $"-c \"virsh qemu-agent-command {vmId} '{execCmd.Replace("\"", "\\\"")}' \"",
            TimeSpan.FromSeconds(30),
            ct);

        if (!execResult.Success)
        {
            _logger.LogDebug("QEMU Guest Agent exec failed: {Error}", execResult.StandardError);
            return false;
        }

        // Check if we got a PID back (indicates command was accepted)
        if (execResult.StandardOutput.Contains("pid"))
        {
            // Extract PID and wait for completion with success verification
            try
            {
                var pidMatch = System.Text.RegularExpressions.Regex.Match(
                    execResult.StandardOutput, @"""pid""\s*:\s*(\d+)");
                if (pidMatch.Success)
                {
                    var pid = pidMatch.Groups[1].Value;
                    _logger.LogDebug("Guest exec started with PID {Pid} for VM {VmId}", pid, vmId);

                    bool completed = false;
                    bool succeeded = false;

                    // Wait for command to complete (poll guest-exec-status)
                    for (int i = 0; i < 20; i++)  // Max 10 seconds (20 * 500ms)
                    {
                        await Task.Delay(500, ct);

                        var statusJson = $"{{\"execute\":\"guest-exec-status\",\"arguments\":{{\"pid\":{pid}}}}}";
                        var statusResult = await _executor.ExecuteAsync(
                            "bash",
                            $"-c \"virsh qemu-agent-command {vmId} '{statusJson.Replace("\"", "\\\"")}' \"",
                            TimeSpan.FromSeconds(5),
                            ct);

                        if (statusResult.Success)
                        {
                            _logger.LogDebug("guest-exec-status response: {Response}", statusResult.StandardOutput);

                            if (statusResult.StandardOutput.Contains("\"exited\":true"))
                            {
                                completed = true;
                                // Check exit code - 0 means success
                                var exitCodeMatch = System.Text.RegularExpressions.Regex.Match(
                                    statusResult.StandardOutput, @"""exitcode""\s*:\s*(\d+)");
                                if (exitCodeMatch.Success && exitCodeMatch.Groups[1].Value == "0")
                                {
                                    succeeded = true;
                                }
                                break;
                            }
                        }
                    }

                    if (completed)
                    {
                        if (succeeded)
                        {
                            // Add small additional delay to ensure filesystem sync
                            await Task.Delay(200, ct);
                            _logger.LogDebug("QEMU Guest Agent injection completed successfully for VM {VmId}", vmId);
                            return true;
                        }
                        else
                        {
                            _logger.LogWarning("QEMU Guest Agent script exited with non-zero code for VM {VmId}", vmId);
                            return false;
                        }
                    }
                    else
                    {
                        // Timeout - command might still be running, proceed anyway with delay
                        _logger.LogWarning("QEMU Guest Agent command timed out for VM {VmId}, proceeding with delay", vmId);
                        await Task.Delay(1000, ct);
                        return true;  // Optimistic - assume it will complete
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Failed to wait for guest-exec completion, adding fallback delay");
                await Task.Delay(1000, ct);  // Fallback delay
            }

            _logger.LogDebug("QEMU Guest Agent injection assumed successful for VM {VmId}", vmId);
            return true;
        }

        // No PID in response - command didn't execute properly
        _logger.LogDebug("QEMU Guest Agent exec response didn't contain PID for VM {VmId}", vmId);
        return false;
    }

    /// <summary>
    /// Method 2: virt-customize (works on disk, requires VM pause or uses live)
    /// </summary>
    private async Task<bool> TryVirtCustomizeAsync(
        string vmId,
        string publicKey,
        string username,
        CancellationToken ct)
    {
        _logger.LogDebug("Trying virt-customize injection for VM {VmId}", vmId);

        // Get VM disk path
        var diskPath = await GetVmDiskPathAsync(vmId, ct);
        if (string.IsNullOrEmpty(diskPath))
        {
            _logger.LogDebug("Could not determine disk path for VM {VmId}", vmId);
            return false;
        }

        // virt-customize can work on running VMs with --live flag (experimental)
        // For safety, we'll create a temp script file
        var scriptPath = Path.GetTempFileName();
        try
        {
            var script = $@"
mkdir -p /home/{username}/.ssh
chmod 700 /home/{username}/.ssh
echo '{publicKey}' >> /home/{username}/.ssh/authorized_keys
chmod 600 /home/{username}/.ssh/authorized_keys
chown -R {username}:{username} /home/{username}/.ssh
";
            await File.WriteAllTextAsync(scriptPath, script, ct);

            var result = await _executor.ExecuteAsync(
                "/bin/bash",
                $"-c \"LIBGUESTFS_BACKEND=direct virt-customize -d {vmId} --run {scriptPath} 2>&1\"",
                TimeSpan.FromMinutes(2),
                ct);

            if (result.Success)
            {
                _logger.LogDebug("virt-customize injection succeeded for VM {VmId}", vmId);
                return true;
            }

            _logger.LogDebug("virt-customize failed: {Error}", result.StandardError);
            return false;
        }
        finally
        {
            try { File.Delete(scriptPath); } catch { }
        }
    }

    /// <summary>
    /// Method 3: SSH exec using existing credentials (fallback)
    /// This requires the VM to already have some form of access
    /// </summary>
    private async Task<bool> TrySshExecAsync(
        string vmId,
        string publicKey,
        string username,
        CancellationToken ct)
    {
        _logger.LogDebug("Trying SSH exec injection for VM {VmId}", vmId);

        // Get VM IP
        var vmIp = await _vmManager.GetVmIpAddressAsync(vmId, ct);
        if (string.IsNullOrEmpty(vmIp))
        {
            _logger.LogDebug("Could not determine IP for VM {VmId}", vmId);
            return false;
        }

        // Try with default key or password
        var script = $@"mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '{publicKey}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys";

        // Try without host key checking for internal VMs
        var result = await _executor.ExecuteAsync(
            "ssh",
            $"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 {username}@{vmIp} \"{script}\"",
            TimeSpan.FromSeconds(15),
            ct);

        if (result.Success)
        {
            _logger.LogDebug("SSH exec injection succeeded for VM {VmId}", vmId);
            return true;
        }

        _logger.LogDebug("SSH exec failed: {Error}", result.StandardError);
        return false;
    }

    private async Task<bool> ExecuteInVmAsync(string vmId, string script, CancellationToken ct)
    {
        var base64Script = Convert.ToBase64String(Encoding.UTF8.GetBytes(script));
        var execCmd = $@"{{""execute"":""guest-exec"",""arguments"":{{""path"":""/bin/bash"",""arg"":[""-c"",""echo {base64Script} | base64 -d | bash""],""capture-output"":true}}}}";

        var result = await _executor.ExecuteAsync(
            "bash",
            $"-c \"virsh qemu-agent-command {vmId} '{execCmd.Replace("\"", "\\\"")}' \"",
            TimeSpan.FromSeconds(30),
            ct);

        return result.Success;
    }

    private async Task<string?> GetVmDiskPathAsync(string vmId, CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync(
            "virsh",
            $"domblklist {vmId} --details",
            TimeSpan.FromSeconds(10),
            ct);

        if (!result.Success) return null;

        foreach (var line in result.StandardOutput.Split('\n'))
        {
            if (line.Contains("disk") && line.Contains("vda"))
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 4)
                {
                    return parts[3];
                }
            }
        }

        return null;
    }

    private void TrackInjectedKey(string vmId, string fingerprint, string username, DateTime? expiresAt)
    {
        lock (_lock)
        {
            if (!_injectedKeys.ContainsKey(vmId))
            {
                _injectedKeys[vmId] = new List<InjectedKey>();
            }

            _injectedKeys[vmId].Add(new InjectedKey
            {
                Fingerprint = fingerprint,
                Username = username,
                InjectedAt = DateTime.UtcNow,
                ExpiresAt = expiresAt
            });
        }
    }

    private static string CalculateFingerprint(string publicKey)
    {
        try
        {
            // Extract the base64 part of the key
            var parts = publicKey.Split(' ');
            if (parts.Length < 2) return publicKey.GetHashCode().ToString("X");

            var keyData = Convert.FromBase64String(parts[1]);
            var hash = SHA256.HashData(keyData);
            return "SHA256:" + Convert.ToBase64String(hash).TrimEnd('=');
        }
        catch
        {
            return publicKey.GetHashCode().ToString("X");
        }
    }

    private class InjectedKey
    {
        public string Fingerprint { get; init; } = "";
        public string Username { get; init; } = "";
        public DateTime InjectedAt { get; init; }
        public DateTime? ExpiresAt { get; init; }
    }
}