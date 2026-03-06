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
    public string ShimPath { get; set; } = "/usr/local/lib/decloud-gpu-shim/libdecloud_cuda_shim.so";

    /// <summary>
    /// Host-side directory exposed to VMs via virtiofs for shim delivery.
    /// The shim .so is symlinked/copied here so the guest can mount it.
    /// </summary>
    public string ShimShareDir { get; set; } = "/usr/local/lib/decloud-gpu-shim";

    /// <summary>
    /// Path to the gpu-proxy source tree for auto-rebuilding missing shims.
    /// </summary>
    public string GpuProxySrcDir { get; set; } = "/opt/decloud/DeCloud.NodeAgent/src/gpu-proxy";

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
    /// Expected shim artifacts in the 9p share directory.
    /// If any are missing, triggers a rebuild from source.
    /// </summary>
    private static readonly string[] RequiredShimFiles =
    {
        "libdecloud_cuda_shim.so",
        "libcuda.so.1",
        "libnvidia-ml.so.1",
        "libcublas_stub.so",
        "libcublasLt_stub.so"
    };

    /// <summary>
    /// Ensure the host-side 9p share directory exists and contains all shim artifacts.
    /// Self-healing: if any artifacts are missing and gpu-proxy source is available,
    /// triggers 'make install-all-shims-compat' to rebuild and install automatically.
    /// </summary>
    private async Task EnsureShimShareDirectoryAsync()
    {
        try
        {
            if (!Directory.Exists(ShimShareDir))
            {
                Directory.CreateDirectory(ShimShareDir);
                _logger.LogInformation(
                    "Created GPU shim share directory: {Dir}", ShimShareDir);
            }

            // Check which artifacts are missing
            var missing = RequiredShimFiles
                .Where(f => !File.Exists(Path.Combine(ShimShareDir, f)))
                .ToList();

            if (missing.Count == 0)
            {
                _logger.LogDebug("All GPU shim artifacts present in {Dir}", ShimShareDir);
                return;
            }

            // Check if gpu-proxy source is available for rebuild
            var makefilePath = Path.Combine(GpuProxySrcDir, "Makefile");
            if (!File.Exists(makefilePath))
            {
                _logger.LogWarning(
                    "GPU shim artifacts missing: {Missing}. " +
                    "Source not found at {Src} — cannot auto-rebuild.",
                    string.Join(", ", missing), GpuProxySrcDir);
                return;
            }

            _logger.LogInformation(
                "GPU shim artifacts missing from 9p share: {Missing}. Triggering rebuild...",
                string.Join(", ", missing));

            // Detect Docker availability for compat builds
            var dockerCheck = await _executor.ExecuteAsync(
                "docker", "info", CancellationToken.None);
            var makeTarget = dockerCheck.Success
                ? "install-all-shims-compat"
                : "install";

            // 'make install-all-shims-compat' builds ALL shims in Docker (glibc 2.31)
            // and installs directly to /usr/local/lib/decloud-gpu-shim/ (= ShimShareDir).
            var result = await _executor.ExecuteAsync(
                "make", $"-C {GpuProxySrcDir} {makeTarget}",
                CancellationToken.None);

            if (result.Success)
            {
                var stillMissing = RequiredShimFiles
                    .Where(f => !File.Exists(Path.Combine(ShimShareDir, f)))
                    .ToList();

                if (stillMissing.Count == 0)
                    _logger.LogInformation(
                        "GPU shims rebuilt and installed to {Dir}", ShimShareDir);
                else
                    _logger.LogWarning(
                        "GPU shim rebuild completed but still missing: {Missing}",
                        string.Join(", ", stillMissing));
            }
            else
            {
                _logger.LogWarning(
                    "GPU shim rebuild failed (exit={Exit}): {Err}",
                    result.ExitCode, result.StandardError?.Trim());
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

            // Ensure the 9p share directory is ready (auto-rebuilds missing shims)
            await EnsureShimShareDirectoryAsync();

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

            // Always enable TCP listener — needed for WSL2 and as fallback
            var psi = new ProcessStartInfo
            {
                FileName = DaemonPath,
                Arguments = $"-p {DaemonPort} -t 0 -T 192.168.122.1",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };

            // WSL2: The daemon binary has -rpath=/usr/local/cuda/lib baked in from the build,
            // which points to nvidia-cuda-toolkit stubs that can't see the real GPU.
            // The actual WSL2 CUDA driver lives at /usr/lib/wsl/lib/libcuda.so.1.
            // LD_LIBRARY_PATH is searched BEFORE rpath, so prepending it here ensures
            // the daemon loads the real driver and can see the GPU.
            const string wslCudaLibPath = "/usr/lib/wsl/lib";
            if (Directory.Exists(wslCudaLibPath))
            {
                var existingLdPath = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
                psi.Environment["LD_LIBRARY_PATH"] = string.IsNullOrEmpty(existingLdPath)
                    ? wslCudaLibPath
                    : $"{wslCudaLibPath}:{existingLdPath}";

                _logger.LogInformation(
                    "WSL2 CUDA driver detected at {Path} — prepending to LD_LIBRARY_PATH",
                    wslCudaLibPath);
            }

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
                "Starting GPU proxy daemon: {Path} {Args}",
                DaemonPath, psi.Arguments);

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
    /// Handles both locally-started daemons (tracked via _daemonProcess)
    /// and externally-started ones (detected via pgrep).
    /// </summary>
    public async Task<bool> HealthCheckAsync(CancellationToken ct = default)
    {
        if (!_isRunning) return false;

        // Case 1: We started the daemon ourselves — check the Process object
        if (_daemonProcess is { HasExited: false })
            return true;

        // Case 2: Daemon we started has exited
        if (_daemonProcess?.HasExited == true)
        {
            _logger.LogWarning(
                "GPU proxy daemon exited unexpectedly (code: {Code})",
                _daemonProcess.ExitCode);
            _daemonProcess = null;
            _isRunning = false;
            return false;
        }

        // Case 3: Daemon was started externally (_daemonProcess is null,
        // _isRunning is true). Verify it's still alive via pgrep.
        try
        {
            var result = await _executor.ExecuteAsync(
                "pgrep", "-f gpu-proxy-daemon", ct);
            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
                return true;
        }
        catch
        {
            // pgrep failed — treat as unhealthy
        }

        _isRunning = false;
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
        if (await HealthCheckAsync(ct))
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
