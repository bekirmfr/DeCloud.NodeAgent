using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;
using System.Diagnostics;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Manages the lifecycle of the GPU proxy daemon — a host-side process
/// that bridges CUDA calls from guest VMs over virtio-vsock.
///
/// Started automatically when the node has GPU(s) but no IOMMU (proxy mode).
/// Stopped when no GPU-proxied VMs remain.
/// </summary>
public class GpuProxyService
{
    private readonly ICommandExecutor _executor;
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly ILogger<GpuProxyService> _logger;

    private Process? _daemonProcess;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private bool _isRunning;
    private int _consecutiveCrashes;

    /// <summary>
    /// Path to the gpu-proxy-daemon binary.
    /// Set via configuration or auto-detected.
    /// </summary>
    public string DaemonPath { get; set; } = "/usr/local/bin/gpu-proxy-daemon";

    /// <summary>
    /// Path to the CUDA shim .so (injected into VMs via cloud-init).
    /// </summary>
    public string ShimPath { get; set; } = "/usr/local/lib/libdecloud_cuda_shim.so";

    /// <summary>
    /// Host-side directory exposed to VMs via virtiofs for shim delivery.
    /// The shim .so is symlinked/copied here so the guest can mount it.
    /// </summary>
    public string ShimShareDir { get; set; } = "/usr/local/lib/decloud-gpu-shim";

    /// <summary>
    /// Port the daemon listens on (vsock port, must match proto/gpu_proxy_proto.h).
    /// </summary>
    public int DaemonPort { get; set; } = 9999;

    /// <summary>
    /// Maximum consecutive crashes before giving up on auto-restart.
    /// </summary>
    public int MaxCrashRestarts { get; set; } = 5;

    /// <summary>
    /// Default per-VM GPU memory quota in bytes. 0 = unlimited.
    /// Overridden per-VM via the SET_MEMORY_QUOTA protocol command.
    /// </summary>
    public long DefaultMemoryQuotaBytes { get; set; } = 0;

    /// <summary>
    /// Kernel execution timeout in seconds. 0 = disabled.
    /// Protects against runaway GPU kernels monopolizing the device.
    /// </summary>
    public int KernelTimeoutSeconds { get; set; } = 30;

    public bool IsRunning => _isRunning;

    public GpuProxyService(
        ICommandExecutor executor,
        IResourceDiscoveryService resourceDiscovery,
        ILogger<GpuProxyService> logger)
    {
        _executor = executor;
        _resourceDiscovery = resourceDiscovery;
        _logger = logger;
    }

    /// <summary>
    /// Ensure the host-side virtiofs share directory exists and contains the shim .so.
    /// Called before starting the daemon so that VMs can mount the share at boot.
    /// </summary>
    private void EnsureShimShareDirectory()
    {
        try
        {
            if (!Directory.Exists(ShimShareDir))
            {
                Directory.CreateDirectory(ShimShareDir);
                _logger.LogInformation(
                    "Created GPU shim share directory: {Dir}", ShimShareDir);
            }

            var targetPath = Path.Combine(ShimShareDir, "libdecloud_cuda_shim.so");
            if (File.Exists(ShimPath) && !File.Exists(targetPath))
            {
                File.Copy(ShimPath, targetPath, overwrite: true);
                _logger.LogInformation(
                    "Copied CUDA shim to virtiofs share: {Src} -> {Dst}",
                    ShimPath, targetPath);
            }
            else if (!File.Exists(ShimPath))
            {
                _logger.LogWarning(
                    "CUDA shim .so not found at {Path}. " +
                    "VMs in proxy mode will not have GPU access until " +
                    "the shim is built and placed at this path.",
                    ShimPath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Failed to set up GPU shim share directory at {Dir}", ShimShareDir);
        }
    }

    /// <summary>
    /// Start the GPU proxy daemon if the node supports GPU proxy mode.
    /// Idempotent — does nothing if already running.
    /// </summary>
    public async Task<bool> EnsureStartedAsync(CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            if (_isRunning && _daemonProcess is { HasExited: false })
            {
                return true;
            }

            // Check if proxy mode is supported
            var inventory = await _resourceDiscovery.GetInventoryCachedAsync(ct);
            if (inventory?.SupportsGpuProxy != true)
            {
                _logger.LogDebug(
                    "GPU proxy mode not supported on this node (no GPU or IOMMU available)");
                return false;
            }

            // Ensure the virtiofs share directory is ready
            EnsureShimShareDirectory();

            // Check daemon binary exists
            if (!File.Exists(DaemonPath))
            {
                _logger.LogWarning(
                    "GPU proxy daemon not found at {Path}. " +
                    "Build it with: cd src/gpu-proxy && make daemon",
                    DaemonPath);
                return false;
            }

            // Check if already running externally (e.g. via systemd)
            var checkResult = await _executor.ExecuteAsync(
                "pgrep", "-f gpu-proxy-daemon", ct);
            if (checkResult.Success && !string.IsNullOrWhiteSpace(checkResult.StandardOutput))
            {
                _logger.LogInformation(
                    "GPU proxy daemon already running (PID: {Pid})",
                    checkResult.StandardOutput.Trim().Split('\n')[0]);
                _isRunning = true;
                return true;
            }

            // Start the daemon
            _logger.LogInformation(
                "Starting GPU proxy daemon: {Path} -p {Port}",
                DaemonPath, DaemonPort);

            var psi = new ProcessStartInfo
            {
                FileName = DaemonPath,
                Arguments = $"-p {DaemonPort} -t {KernelTimeoutSeconds} -v",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };

            _daemonProcess = Process.Start(psi);
            if (_daemonProcess == null)
            {
                _logger.LogError("Failed to start GPU proxy daemon");
                return false;
            }

            // Forward daemon output to our logger
            _daemonProcess.OutputDataReceived += (_, e) =>
            {
                if (e.Data != null) _logger.LogInformation("[gpu-proxy] {Line}", e.Data);
            };
            _daemonProcess.ErrorDataReceived += (_, e) =>
            {
                if (e.Data != null) _logger.LogWarning("[gpu-proxy] {Line}", e.Data);
            };
            _daemonProcess.BeginOutputReadLine();
            _daemonProcess.BeginErrorReadLine();

            // Give it a moment to start
            await Task.Delay(500, ct);

            if (_daemonProcess.HasExited)
            {
                _logger.LogError(
                    "GPU proxy daemon exited immediately with code {Code}",
                    _daemonProcess.ExitCode);
                _daemonProcess = null;
                return false;
            }

            _isRunning = true;
            _consecutiveCrashes = 0;
            _logger.LogInformation(
                "GPU proxy daemon started (PID: {Pid}, vsock port: {Port})",
                _daemonProcess.Id, DaemonPort);

            return true;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Stop the GPU proxy daemon.
    /// </summary>
    public async Task StopAsync(CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            if (_daemonProcess == null || _daemonProcess.HasExited)
            {
                _isRunning = false;
                _daemonProcess = null;
                return;
            }

            _logger.LogInformation("Stopping GPU proxy daemon (PID: {Pid})", _daemonProcess.Id);

            // Send SIGTERM for graceful shutdown
            try
            {
                _daemonProcess.Kill(entireProcessTree: false);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to send SIGTERM to GPU proxy daemon");
            }

            // Wait up to 5 seconds for graceful exit
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(5));

            try
            {
                await _daemonProcess.WaitForExitAsync(cts.Token);
                _logger.LogInformation("GPU proxy daemon stopped gracefully");
            }
            catch (OperationCanceledException)
            {
                // Force kill
                _logger.LogWarning("GPU proxy daemon did not exit gracefully, force killing");
                _daemonProcess.Kill(entireProcessTree: true);
            }

            _daemonProcess = null;
            _isRunning = false;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Check if the daemon is healthy (still running).
    /// If it has crashed, attempt automatic restart with backoff.
    /// </summary>
    public bool HealthCheck()
    {
        if (!_isRunning) return false;

        if (_daemonProcess is { HasExited: false })
            return true;

        // Daemon crashed — mark as not running
        if (_daemonProcess?.HasExited == true)
        {
            _logger.LogWarning(
                "GPU proxy daemon exited unexpectedly (code: {Code})",
                _daemonProcess.ExitCode);
            _daemonProcess = null;
            _isRunning = false;
        }

        return false;
    }

    /// <summary>
    /// Attempt to restart the daemon if it has crashed.
    /// Returns true if the daemon is now running (either it was already running,
    /// or it was successfully restarted).
    /// Uses exponential backoff: 2s, 4s, 8s, 16s between restart attempts.
    /// Gives up after MaxCrashRestarts consecutive failures.
    /// </summary>
    public async Task<bool> EnsureHealthyAsync(CancellationToken ct = default)
    {
        if (HealthCheck())
            return true;

        if (_consecutiveCrashes >= MaxCrashRestarts)
        {
            _logger.LogError(
                "GPU proxy daemon has crashed {Count} times consecutively. " +
                "Not restarting — manual intervention required.",
                _consecutiveCrashes);
            return false;
        }

        _consecutiveCrashes++;
        var backoffMs = (int)Math.Pow(2, _consecutiveCrashes) * 1000;
        backoffMs = Math.Min(backoffMs, 16000);

        _logger.LogWarning(
            "GPU proxy daemon crashed (attempt {Attempt}/{Max}). " +
            "Restarting in {BackoffMs}ms...",
            _consecutiveCrashes, MaxCrashRestarts, backoffMs);

        await Task.Delay(backoffMs, ct);

        var started = await EnsureStartedAsync(ct);
        if (started)
        {
            _logger.LogInformation(
                "GPU proxy daemon restarted successfully after crash");
            _consecutiveCrashes = 0;
        }

        return started;
    }
}
