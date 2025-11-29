using System.Text;
using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Resilient cloud-init state cleaner that ensures VMs boot with fresh cloud-init configuration.
/// Uses multiple fallback methods and graceful degradation.
/// </summary>
public interface ICloudInitCleaner
{
    /// <summary>
    /// Clean cloud-init state from a disk image (base or overlay).
    /// Returns true if cleaned successfully, false if cleaning was skipped but VM can still work.
    /// Throws only on critical errors that should prevent VM creation.
    /// </summary>
    Task<CloudInitCleanResult> CleanAsync(string diskPath, CancellationToken ct = default);

    /// <summary>
    /// Check if required tools are available on this system
    /// </summary>
    Task<CloudInitToolsStatus> CheckToolsAsync(CancellationToken ct = default);
}

public enum CleanMethod
{
    None,
    VirtCustomize,
    GuestMount,
    QemuNbd,
    Skipped
}

public class CloudInitCleanResult
{
    public bool Success { get; init; }
    public CleanMethod MethodUsed { get; init; }
    public string? Message { get; init; }
    public TimeSpan Duration { get; init; }

    public static CloudInitCleanResult Cleaned(CleanMethod method, TimeSpan duration)
        => new() { Success = true, MethodUsed = method, Duration = duration };

    public static CloudInitCleanResult Skipped(string reason)
        => new() { Success = true, MethodUsed = CleanMethod.Skipped, Message = reason };

    public static CloudInitCleanResult Failed(string error)
        => new() { Success = false, MethodUsed = CleanMethod.None, Message = error };
}

public class CloudInitToolsStatus
{
    public bool VirtCustomizeAvailable { get; init; }
    public bool GuestMountAvailable { get; init; }
    public bool QemuNbdAvailable { get; init; }
    public bool AnyToolAvailable => VirtCustomizeAvailable || GuestMountAvailable || QemuNbdAvailable;
    public string RecommendedInstallCommand { get; init; } = "";
}

public class CloudInitCleaner : ICloudInitCleaner
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<CloudInitCleaner> _logger;

    // Lock to prevent concurrent NBD operations (only 16 nbd devices available)
    private static readonly SemaphoreSlim _nbdLock = new(4);

    // Cache tool availability check
    private CloudInitToolsStatus? _toolsStatus;
    private DateTime _toolsStatusCheckedAt;
    private static readonly TimeSpan ToolsStatusCacheDuration = TimeSpan.FromMinutes(10);

    // Marker file to track already-cleaned base images
    private const string CleanedMarkerSuffix = ".cloudinit-cleaned";

    public CloudInitCleaner(ICommandExecutor executor, ILogger<CloudInitCleaner> logger)
    {
        _executor = executor;
        _logger = logger;
    }

    public async Task<CloudInitCleanResult> CleanAsync(string diskPath, CancellationToken ct = default)
    {
        var startTime = DateTime.UtcNow;

        // Validate input
        if (string.IsNullOrEmpty(diskPath))
        {
            return CloudInitCleanResult.Failed("Disk path is null or empty");
        }

        if (!File.Exists(diskPath))
        {
            return CloudInitCleanResult.Failed($"Disk not found: {diskPath}");
        }

        // Check if this is a base image that was already cleaned
        if (await IsAlreadyCleanedAsync(diskPath, ct))
        {
            _logger.LogDebug("Disk already cleaned (marker exists): {Path}", diskPath);
            return CloudInitCleanResult.Skipped("Already cleaned");
        }

        // Check available tools
        var tools = await CheckToolsAsync(ct);
        if (!tools.AnyToolAvailable)
        {
            _logger.LogWarning(
                "No cloud-init cleaning tools available. Install with: {Command}. " +
                "VMs may have stale cloud-init state.",
                tools.RecommendedInstallCommand);
            return CloudInitCleanResult.Skipped("No cleaning tools available");
        }

        _logger.LogInformation("Cleaning cloud-init state from: {Path}", diskPath);

        // Try each method in order of preference
        var methods = new List<Func<string, CancellationToken, Task<bool>>>
        {
            TryVirtCustomizeAsync,
            TryGuestMountAsync,
            TryQemuNbdAsync
        };

        foreach (var method in methods)
        {
            try
            {
                ct.ThrowIfCancellationRequested();

                if (await method(diskPath, ct))
                {
                    var duration = DateTime.UtcNow - startTime;
                    var methodName = method.Method.Name.Replace("Try", "").Replace("Async", "");
                    var cleanMethod = methodName switch
                    {
                        "VirtCustomize" => CleanMethod.VirtCustomize,
                        "GuestMount" => CleanMethod.GuestMount,
                        "QemuNbd" => CleanMethod.QemuNbd,
                        _ => CleanMethod.None
                    };

                    // Mark as cleaned for base images
                    await MarkAsCleanedAsync(diskPath, cleanMethod, ct);

                    _logger.LogInformation(
                        "Cloud-init state cleaned successfully using {Method} in {Duration}ms",
                        cleanMethod, duration.TotalMilliseconds);

                    return CloudInitCleanResult.Cleaned(cleanMethod, duration);
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Clean method {Method} failed, trying next", method.Method.Name);
            }
        }

        // All methods failed - this is not critical, VM might still work
        _logger.LogWarning(
            "All cloud-init cleaning methods failed for {Path}. " +
            "VM may boot with stale cloud-init state. " +
            "Consider installing libguestfs-tools.",
            diskPath);

        return CloudInitCleanResult.Skipped("All cleaning methods failed");
    }

    public async Task<CloudInitToolsStatus> CheckToolsAsync(CancellationToken ct = default)
    {
        // Return cached result if fresh
        if (_toolsStatus != null && DateTime.UtcNow - _toolsStatusCheckedAt < ToolsStatusCacheDuration)
        {
            return _toolsStatus;
        }

        var virtCustomize = await CheckToolExistsAsync("virt-customize", ct);
        var guestMount = await CheckToolExistsAsync("guestmount", ct);
        var qemuNbd = await CheckToolExistsAsync("qemu-nbd", ct);

        _toolsStatus = new CloudInitToolsStatus
        {
            VirtCustomizeAvailable = virtCustomize,
            GuestMountAvailable = guestMount,
            QemuNbdAvailable = qemuNbd,
            RecommendedInstallCommand = "sudo apt install -y libguestfs-tools qemu-utils"
        };
        _toolsStatusCheckedAt = DateTime.UtcNow;

        if (!_toolsStatus.AnyToolAvailable)
        {
            _logger.LogWarning(
                "No cloud-init cleaning tools found. Recommended: {Command}",
                _toolsStatus.RecommendedInstallCommand);
        }

        return _toolsStatus;
    }

    /// <summary>
    /// Method 1: virt-customize (fastest, most reliable)
    /// </summary>
    private async Task<bool> TryVirtCustomizeAsync(string diskPath, CancellationToken ct)
    {
        var tools = await CheckToolsAsync(ct);
        if (!tools.VirtCustomizeAvailable)
        {
            _logger.LogDebug("virt-customize not available");
            return false;
        }

        _logger.LogDebug("Attempting cloud-init clean with virt-customize");

        // Build command to clean cloud-init state
        // We clean multiple locations to handle different distros
        var commands = new[]
        {
            "rm -rf /var/lib/cloud/*",
            "rm -rf /var/log/cloud-init*",
            "rm -f /etc/machine-id",
            "rm -f /var/lib/dbus/machine-id",
            "truncate -s 0 /etc/machine-id 2>/dev/null || true"
        };

        var runCommands = string.Join(" ", commands.Select(c => $"--run-command \"{c}\""));

        // Set LIBGUESTFS_BACKEND to direct to avoid issues with libvirt
        var result = await _executor.ExecuteAsync(
            "/bin/bash",
            $"-c \"LIBGUESTFS_BACKEND=direct virt-customize -a {diskPath} {runCommands} 2>&1\"",
            TimeSpan.FromMinutes(3),
            ct);

        if (result.Success)
        {
            _logger.LogDebug("virt-customize completed successfully");
            return true;
        }

        _logger.LogDebug("virt-customize failed: {Error}", result.StandardError);
        return false;
    }

    /// <summary>
    /// Method 2: guestmount (fallback, requires FUSE)
    /// </summary>
    private async Task<bool> TryGuestMountAsync(string diskPath, CancellationToken ct)
    {
        var tools = await CheckToolsAsync(ct);
        if (!tools.GuestMountAvailable)
        {
            _logger.LogDebug("guestmount not available");
            return false;
        }

        _logger.LogDebug("Attempting cloud-init clean with guestmount");

        var mountPoint = Path.Combine(Path.GetTempPath(), $"decloud-mount-{Guid.NewGuid():N}");

        try
        {
            Directory.CreateDirectory(mountPoint);

            // Mount the image
            var mountResult = await _executor.ExecuteAsync(
                "/bin/bash",
                $"-c \"LIBGUESTFS_BACKEND=direct guestmount -a {diskPath} -i --rw {mountPoint} 2>&1\"",
                TimeSpan.FromMinutes(2),
                ct);

            if (!mountResult.Success)
            {
                _logger.LogDebug("guestmount failed: {Error}", mountResult.StandardError);
                return false;
            }

            // Clean cloud-init directories
            var cloudLibPath = Path.Combine(mountPoint, "var/lib/cloud");
            var cloudLogPath = Path.Combine(mountPoint, "var/log");
            var machineIdPath = Path.Combine(mountPoint, "etc/machine-id");

            if (Directory.Exists(cloudLibPath))
            {
                await DeleteDirectoryContentsAsync(cloudLibPath);
                _logger.LogDebug("Cleaned /var/lib/cloud");
            }

            // Remove cloud-init logs
            if (Directory.Exists(cloudLogPath))
            {
                foreach (var file in Directory.GetFiles(cloudLogPath, "cloud-init*"))
                {
                    File.Delete(file);
                }
            }

            // Clear machine-id to force regeneration
            if (File.Exists(machineIdPath))
            {
                await File.WriteAllTextAsync(machineIdPath, "", ct);
            }

            return true;
        }
        finally
        {
            // Always try to unmount
            await _executor.ExecuteAsync("guestunmount", mountPoint, TimeSpan.FromSeconds(30), ct);

            // Clean up mount point
            try
            {
                if (Directory.Exists(mountPoint))
                {
                    Directory.Delete(mountPoint, false);
                }
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    /// <summary>
    /// Method 3: qemu-nbd (most compatible, but requires nbd kernel module)
    /// </summary>
    private async Task<bool> TryQemuNbdAsync(string diskPath, CancellationToken ct)
    {
        var tools = await CheckToolsAsync(ct);
        if (!tools.QemuNbdAvailable)
        {
            _logger.LogDebug("qemu-nbd not available");
            return false;
        }

        _logger.LogDebug("Attempting cloud-init clean with qemu-nbd");

        // Limit concurrent NBD operations
        if (!await _nbdLock.WaitAsync(TimeSpan.FromSeconds(30), ct))
        {
            _logger.LogDebug("Could not acquire NBD lock");
            return false;
        }

        string? nbdDevice = null;
        string? mountPoint = null;

        try
        {
            // Ensure nbd module is loaded
            await _executor.ExecuteAsync("modprobe", "nbd max_part=8", TimeSpan.FromSeconds(10), ct);

            // Find available nbd device
            nbdDevice = await FindAvailableNbdDeviceAsync(ct);
            if (nbdDevice == null)
            {
                _logger.LogDebug("No available NBD device found");
                return false;
            }

            // Connect image to NBD device
            var connectResult = await _executor.ExecuteAsync(
                "qemu-nbd",
                $"--connect={nbdDevice} {diskPath}",
                TimeSpan.FromSeconds(30),
                ct);

            if (!connectResult.Success)
            {
                _logger.LogDebug("qemu-nbd connect failed: {Error}", connectResult.StandardError);
                return false;
            }

            // Wait for device to be ready
            await Task.Delay(500, ct);

            // Find the right partition (try common ones)
            string? partition = await FindRootPartitionAsync(nbdDevice, ct);
            if (partition == null)
            {
                _logger.LogDebug("Could not find root partition on {Device}", nbdDevice);
                return false;
            }

            // Mount the partition
            mountPoint = Path.Combine(Path.GetTempPath(), $"decloud-nbd-{Guid.NewGuid():N}");
            Directory.CreateDirectory(mountPoint);

            var mountResult = await _executor.ExecuteAsync(
                "mount",
                $"{partition} {mountPoint}",
                TimeSpan.FromSeconds(30),
                ct);

            if (!mountResult.Success)
            {
                _logger.LogDebug("mount failed: {Error}", mountResult.StandardError);
                return false;
            }

            // Clean cloud-init
            var cloudLibPath = Path.Combine(mountPoint, "var/lib/cloud");
            if (Directory.Exists(cloudLibPath))
            {
                await DeleteDirectoryContentsAsync(cloudLibPath);
            }

            // Clear machine-id
            var machineIdPath = Path.Combine(mountPoint, "etc/machine-id");
            if (File.Exists(machineIdPath))
            {
                await File.WriteAllTextAsync(machineIdPath, "", ct);
            }

            _logger.LogDebug("qemu-nbd clean completed successfully");
            return true;
        }
        finally
        {
            // Cleanup in reverse order
            if (mountPoint != null)
            {
                await _executor.ExecuteAsync("umount", mountPoint, TimeSpan.FromSeconds(30), ct);
                try { Directory.Delete(mountPoint, false); } catch { }
            }

            if (nbdDevice != null)
            {
                await _executor.ExecuteAsync("qemu-nbd", $"--disconnect {nbdDevice}", TimeSpan.FromSeconds(30), ct);
            }

            _nbdLock.Release();
        }
    }

    private async Task<string?> FindAvailableNbdDeviceAsync(CancellationToken ct)
    {
        // Check /dev/nbd0 through /dev/nbd15
        for (int i = 0; i < 16; i++)
        {
            var device = $"/dev/nbd{i}";
            if (!File.Exists(device))
            {
                continue;
            }

            // Check if device is in use by looking at its size
            var result = await _executor.ExecuteAsync(
                "blockdev",
                $"--getsize64 {device}",
                TimeSpan.FromSeconds(5),
                ct);

            if (result.Success && long.TryParse(result.StandardOutput.Trim(), out var size) && size == 0)
            {
                return device;
            }
        }

        return null;
    }

    private async Task<string?> FindRootPartitionAsync(string nbdDevice, CancellationToken ct)
    {
        // Try common partition layouts
        var candidates = new[]
        {
            $"{nbdDevice}p1",  // /dev/nbd0p1 - most common for cloud images
            $"{nbdDevice}p2",  // /dev/nbd0p2 - some images have boot as p1
            $"{nbdDevice}p3",  // /dev/nbd0p3 - Ubuntu with EFI
            nbdDevice         // No partition table
        };

        foreach (var partition in candidates)
        {
            if (!File.Exists(partition))
            {
                continue;
            }

            // Try to identify filesystem
            var result = await _executor.ExecuteAsync(
                "blkid",
                $"-o value -s TYPE {partition}",
                TimeSpan.FromSeconds(5),
                ct);

            if (result.Success)
            {
                var fsType = result.StandardOutput.Trim();
                if (fsType is "ext4" or "ext3" or "xfs" or "btrfs")
                {
                    return partition;
                }
            }
        }

        return null;
    }

    private async Task<bool> CheckToolExistsAsync(string tool, CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync("which", tool, TimeSpan.FromSeconds(5), ct);
        return result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput);
    }

    private async Task<bool> IsAlreadyCleanedAsync(string diskPath, CancellationToken ct)
    {
        var markerPath = diskPath + CleanedMarkerSuffix;

        if (!File.Exists(markerPath))
        {
            return false;
        }

        // Check if marker is newer than the disk (disk might have been re-downloaded)
        var diskInfo = new FileInfo(diskPath);
        var markerInfo = new FileInfo(markerPath);

        return markerInfo.LastWriteTimeUtc >= diskInfo.LastWriteTimeUtc;
    }

    private async Task MarkAsCleanedAsync(string diskPath, CleanMethod method, CancellationToken ct)
    {
        try
        {
            var markerPath = diskPath + CleanedMarkerSuffix;
            var content = $"cleaned={DateTime.UtcNow:O}\nmethod={method}\n";
            await File.WriteAllTextAsync(markerPath, content, ct);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Could not write cleaned marker for {Path}", diskPath);
        }
    }

    private static async Task DeleteDirectoryContentsAsync(string path)
    {
        if (!Directory.Exists(path))
        {
            return;
        }

        // Delete all files
        foreach (var file in Directory.GetFiles(path, "*", SearchOption.AllDirectories))
        {
            try
            {
                File.Delete(file);
            }
            catch
            {
                // Ignore individual file deletion errors
            }
        }

        // Delete all subdirectories
        foreach (var dir in Directory.GetDirectories(path))
        {
            try
            {
                Directory.Delete(dir, true);
            }
            catch
            {
                // Ignore individual directory deletion errors
            }
        }

        await Task.CompletedTask;
    }
}