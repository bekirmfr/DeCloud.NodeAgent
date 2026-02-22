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
    /// Port the daemon listens on (vsock port, must match proto/gpu_proxy_proto.h).
    /// </summary>
    public int DaemonPort { get; set; } = 9999;

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
                Arguments = $"-p {DaemonPort} -v",
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
}
