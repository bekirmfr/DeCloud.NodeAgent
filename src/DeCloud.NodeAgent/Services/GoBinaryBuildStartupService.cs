using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Infrastructure.Services;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Background service that proactively builds any missing Go binaries
/// (DHT node, Block Store node) on NodeAgent startup.
///
/// This eliminates the dependency on install.sh build order — binaries are
/// guaranteed to exist before the first VM deployment attempt, regardless of
/// whether install.sh ran the build steps or was partially executed.
///
/// Build is hash-checked: if the binary already exists and the source is
/// unchanged, build.sh exits in ~1 second (cache hit). On first run or after
/// a source update, the full Go build runs (~3-4 minutes with network,
/// ~30 seconds with cached Go modules).
///
/// Runs once at startup, non-blocking (BackgroundService). The NodeAgent
/// accepts connections immediately — VM deployments requesting a missing
/// binary will wait for LoadDhtBinaryAsync / LoadBlockStoreBinaryAsync
/// to trigger their own build if this service hasn't finished yet.
/// </summary>
public class GoBinaryBuildStartupService : BackgroundService
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<GoBinaryBuildStartupService> _logger;
    private readonly string _templateBasePath;

    // Per-binary build timeout. Go download + compile can take several minutes
    // on first run with no cache. Subsequent runs with cached modules: ~30s.
    private static readonly TimeSpan BuildTimeout = TimeSpan.FromMinutes(10);

    public GoBinaryBuildStartupService(
        ICommandExecutor executor,
        ILogger<GoBinaryBuildStartupService> logger)
    {
        _executor = executor;
        _logger = logger;
        _templateBasePath = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "CloudInit",
            "Templates");
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Short delay so the NodeAgent is fully initialized before we start
        // a potentially long-running build. The 5-second delay is enough for
        // Kestrel to bind and start accepting connections.
        await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);

        _logger.LogInformation("GoBinaryBuildStartupService: checking Go binaries...");

        // Detect host architecture once
        var arch = await DetectArchitectureAsync(stoppingToken);
        _logger.LogInformation("Host architecture: {Arch}", arch);

        // Build each binary in sequence (avoids parallel Go module downloads
        // racing over the shared GOPATH/GOCACHE)
        await BuildBinaryAsync("DHT node", "dht-vm", "dht-node-src", arch, stoppingToken);
        await BuildBinaryAsync("Block Store node", "blockstore-vm", "blockstore-node-src", arch, stoppingToken);

        _logger.LogInformation("GoBinaryBuildStartupService: binary check complete.");
    }

    private async Task BuildBinaryAsync(
        string displayName,
        string vmDir,
        string srcDir,
        string arch,
        CancellationToken ct)
    {
        var buildScript = Path.Combine(_templateBasePath, vmDir, srcDir, "build.sh");

        if (!File.Exists(buildScript))
        {
            _logger.LogWarning(
                "{Binary}: build script not found at {Path} — skipping. " +
                "Ensure the source files were included in the deployment.",
                displayName, buildScript);
            return;
        }

        // Check if the binary already exists (fast path — avoid logging a warning
        // that triggers unnecessary alarm when everything is fine)
        var binaryName = vmDir switch
        {
            "dht-vm" => "dht-node",
            "blockstore-vm" => "blockstore-node",
            _ => vmDir
        };
        var expectedBinary = Path.Combine(_templateBasePath, vmDir, $"{binaryName}-{arch}.gz.b64");
        var hashFile = Path.Combine(_templateBasePath, vmDir, $".{binaryName}-source.sha256");

        if (File.Exists(expectedBinary))
        {
            _logger.LogInformation(
                "{Binary}: binary exists at {Path} — running hash check for staleness",
                displayName, expectedBinary);
        }
        else
        {
            _logger.LogWarning(
                "{Binary}: binary not found at {Path} — building from source. " +
                "This may take several minutes on first run.",
                displayName, expectedBinary);
        }

        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(BuildTimeout);

            _logger.LogInformation("{Binary}: running build.sh {Arch}...", displayName, arch);

            var result = await _executor.ExecuteAsync(
                "bash",
                $"{buildScript} {arch}",
                BuildTimeout,
                cts.Token);

            if (result.Success)
            {
                if (File.Exists(expectedBinary))
                {
                    var sizeKB = new FileInfo(expectedBinary).Length / 1024;
                    _logger.LogInformation(
                        "{Binary}: build complete — {Path} ({SizeKB} KB gz+b64)",
                        displayName, expectedBinary, sizeKB);
                }
                else
                {
                    // build.sh reported success but binary not found — log as warning
                    _logger.LogWarning(
                        "{Binary}: build.sh succeeded but binary not found at {Path}. " +
                        "Check build.sh output for errors.",
                        displayName, expectedBinary);
                }
            }
            else
            {
                _logger.LogError(
                    "{Binary}: build failed (exit {ExitCode}). " +
                    "VM deployments requiring this binary will fail. " +
                    "Ensure Go 1.23+ is installed (apt install golang-go). " +
                    "Stderr: {Stderr}",
                    displayName, result.ExitCode,
                    result.StandardError?.Length > 500
                        ? result.StandardError[..500] + "…"
                        : result.StandardError);
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // NodeAgent shutting down — exit gracefully
        }
        catch (OperationCanceledException)
        {
            _logger.LogError(
                "{Binary}: build timed out after {Minutes} minutes. " +
                "Check Go installation and network access.",
                displayName, BuildTimeout.TotalMinutes);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "{Binary}: unexpected error during build. " +
                "VM deployments requiring this binary may fail.",
                displayName);
        }
    }

    private async Task<string> DetectArchitectureAsync(CancellationToken ct)
    {
        try
        {
            var result = await _executor.ExecuteAsync("uname", "-m", ct);
            if (result.Success)
            {
                return result.StandardOutput.Trim() switch
                {
                    "x86_64" or "amd64"    => "amd64",
                    "aarch64" or "arm64"   => "arm64",
                    var other              => "amd64" // safe default
                };
            }
        }
        catch { /* fall through to default */ }

        return "amd64";
    }
}
