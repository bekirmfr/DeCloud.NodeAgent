using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Self-configuring GPU setup service.
/// Runs once on startup after resource discovery: detects NVIDIA GPUs and
/// auto-installs Docker + NVIDIA Container Toolkit if missing.
/// Also configures VFIO passthrough modules when IOMMU is available.
/// Results are picked up by the next heartbeat via ResourceDiscoveryService.
/// </summary>
public class GpuAutoConfigureService : BackgroundService
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly ICommandExecutor _executor;
    private readonly ILogger<GpuAutoConfigureService> _logger;

    public GpuAutoConfigureService(
        IResourceDiscoveryService resourceDiscovery,
        ICommandExecutor executor,
        ILogger<GpuAutoConfigureService> logger)
    {
        _resourceDiscovery = resourceDiscovery;
        _executor = executor;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Wait for initial resource discovery + registration to complete
        await Task.Delay(TimeSpan.FromSeconds(20), stoppingToken);

        try
        {
            await AutoConfigureGpuAsync(stoppingToken);
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Normal shutdown
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "GPU auto-configuration failed");
        }
    }

    private async Task AutoConfigureGpuAsync(CancellationToken ct)
    {
        _logger.LogInformation("GPU auto-configuration: checking hardware...");

        var inventory = await _resourceDiscovery.GetInventoryCachedAsync(ct);
        if (inventory == null || !inventory.SupportsGpu || inventory.Gpus.Count == 0)
        {
            _logger.LogInformation("No GPU detected — skipping auto-configuration");
            return;
        }

        _logger.LogInformation(
            "GPU detected: {Count} GPU(s), container sharing={ContainerReady}, passthrough={Passthrough}",
            inventory.Gpus.Count,
            inventory.SupportsGpuContainers,
            inventory.Gpus.Any(g => g.IsAvailableForPassthrough));

        // Already fully configured?
        if (inventory.SupportsGpuContainers)
        {
            _logger.LogInformation("GPU container sharing already configured — nothing to do");
            return;
        }

        // ─── Step 1: Verify NVIDIA drivers ───
        var (hasDriver, driverVersion) = await FindNvidiaSmiAsync(ct);
        if (!hasDriver)
        {
            _logger.LogWarning(
                "NVIDIA GPU detected but nvidia-smi not available. " +
                "Install NVIDIA drivers manually, then restart the node agent.");
            return;
        }

        _logger.LogInformation("NVIDIA driver version: {Version}", driverVersion);

        // ─── Step 2: Ensure Docker is installed and running ───
        var dockerReady = await EnsureDockerAsync(ct);
        if (!dockerReady)
        {
            _logger.LogWarning("Docker not available — GPU container sharing will not work");
            return;
        }

        // ─── Step 3: Ensure NVIDIA Container Toolkit ───
        var toolkitReady = await EnsureNvidiaContainerToolkitAsync(ct);
        if (toolkitReady)
        {
            _logger.LogInformation("GPU container sharing ready (Docker + NVIDIA Container Toolkit)");
        }
        else
        {
            _logger.LogWarning("NVIDIA Container Toolkit setup failed — GPU container sharing unavailable");
        }

        // ─── Step 4: VFIO passthrough (if IOMMU enabled) ───
        var hasIommu = inventory.Gpus.Any(g => g.IsIommuEnabled);
        if (hasIommu)
        {
            var vfioReady = await ConfigureVfioAsync(ct);
            if (vfioReady)
            {
                _logger.LogInformation("VFIO passthrough modules configured");
            }
        }

        // Force re-discovery so the next heartbeat reports updated capabilities
        _logger.LogInformation("Re-running resource discovery after GPU setup...");
        await _resourceDiscovery.DiscoverAllAsync(ct);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private async Task<(bool success, string? driverVersion)> FindNvidiaSmiAsync(CancellationToken ct)
    {
        var paths = new[]
        {
            "nvidia-smi",
            "/usr/bin/nvidia-smi",
            "/usr/lib/wsl/lib/nvidia-smi"
        };

        foreach (var path in paths)
        {
            var result = await _executor.ExecuteAsync(
                path, "--query-gpu=driver_version --format=csv,noheader,nounits", ct);
            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                return (true, result.StandardOutput.Trim().Split('\n')[0].Trim());
            }
        }

        return (false, null);
    }

    private async Task<bool> EnsureDockerAsync(CancellationToken ct)
    {
        // Already running?
        var info = await _executor.ExecuteAsync("docker", "info", ct);
        if (info.Success)
        {
            _logger.LogDebug("Docker already running");
            return true;
        }

        // Installed but not running?
        var ver = await _executor.ExecuteAsync("docker", "--version", ct);
        if (ver.Success)
        {
            _logger.LogInformation("Docker installed but not running — starting...");
            await _executor.ExecuteAsync("systemctl", "enable docker --quiet", ct);
            await _executor.ExecuteAsync("systemctl", "start docker", ct);
            await Task.Delay(3000, ct);
            info = await _executor.ExecuteAsync("docker", "info", ct);
            if (info.Success) return true;
        }

        // Install via convenience script
        _logger.LogInformation("Installing Docker...");
        var dl = await _executor.ExecuteAsync(
            "bash", "-c \"curl -fsSL https://get.docker.com -o /tmp/get-docker.sh\"",
            TimeSpan.FromMinutes(2), ct);
        if (!dl.Success)
        {
            _logger.LogError("Failed to download Docker install script");
            return false;
        }

        var install = await _executor.ExecuteAsync(
            "bash", "-c \"sh /tmp/get-docker.sh\"",
            TimeSpan.FromMinutes(10), ct);
        if (!install.Success)
        {
            _logger.LogError("Docker installation failed: {Error}", install.StandardError);
            return false;
        }

        await _executor.ExecuteAsync("systemctl", "enable docker --quiet", ct);
        await _executor.ExecuteAsync("systemctl", "start docker", ct);
        await Task.Delay(3000, ct);

        info = await _executor.ExecuteAsync("docker", "info", ct);
        if (info.Success)
        {
            _logger.LogInformation("Docker installed and running");
            return true;
        }

        _logger.LogError("Docker installed but failed to start");
        return false;
    }

    private async Task<bool> EnsureNvidiaContainerToolkitAsync(CancellationToken ct)
    {
        // Already configured?
        var info = await _executor.ExecuteAsync("docker", "info", ct);
        if (info.Success && info.StandardOutput.Contains("nvidia", StringComparison.OrdinalIgnoreCase))
        {
            _logger.LogDebug("NVIDIA Container Toolkit already configured");
            return true;
        }

        _logger.LogInformation("Installing NVIDIA Container Toolkit...");

        // GPG key
        if (!File.Exists("/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"))
        {
            var gpg = await _executor.ExecuteAsync(
                "bash", "-c \"curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\"",
                TimeSpan.FromSeconds(30), ct);
            if (!gpg.Success)
            {
                _logger.LogError("Failed to add NVIDIA GPG key: {Error}", gpg.StandardError);
                return false;
            }
        }

        // Repository (stable/deb — works on Ubuntu 22.04+/24.04)
        var repo = await _executor.ExecuteAsync(
            "bash", "-c \"curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list\"",
            TimeSpan.FromSeconds(30), ct);
        if (!repo.Success)
        {
            _logger.LogError("Failed to add NVIDIA toolkit repo: {Error}", repo.StandardError);
            return false;
        }

        // Install
        await _executor.ExecuteAsync("bash", "-c \"apt-get update -qq\"", TimeSpan.FromMinutes(2), ct);
        var pkg = await _executor.ExecuteAsync(
            "bash", "-c \"apt-get install -y nvidia-container-toolkit\"",
            TimeSpan.FromMinutes(5), ct);
        if (!pkg.Success)
        {
            _logger.LogError("Failed to install nvidia-container-toolkit: {Error}", pkg.StandardError);
            return false;
        }

        // Configure Docker runtime
        var cfg = await _executor.ExecuteAsync(
            "nvidia-ctk", "runtime configure --runtime=docker",
            TimeSpan.FromSeconds(30), ct);
        if (!cfg.Success)
        {
            _logger.LogError("Failed to configure NVIDIA runtime: {Error}", cfg.StandardError);
            return false;
        }

        // Restart Docker
        await _executor.ExecuteAsync("systemctl", "restart docker", ct);
        await Task.Delay(3000, ct);

        // Verify
        info = await _executor.ExecuteAsync("docker", "info", ct);
        if (info.Success && info.StandardOutput.Contains("nvidia", StringComparison.OrdinalIgnoreCase))
        {
            _logger.LogInformation("NVIDIA Container Toolkit installed and configured");
            return true;
        }

        _logger.LogWarning("NVIDIA Container Toolkit installed but runtime not detected in Docker");
        return false;
    }

    private async Task<bool> ConfigureVfioAsync(CancellationToken ct)
    {
        try
        {
            var modules = new[] { "vfio", "vfio_iommu_type1", "vfio_pci" };
            foreach (var mod in modules)
            {
                await _executor.ExecuteAsync("modprobe", mod, ct);
            }

            var modulesConf = string.Join("\n", modules);
            var write = await _executor.ExecuteAsync(
                "bash", $"-c \"echo '{modulesConf}' > /etc/modules-load.d/vfio.conf\"", ct);

            var blacklist = await _executor.ExecuteAsync(
                "bash", "-c \"echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf\"", ct);

            if (write.Success && blacklist.Success)
            {
                await _executor.ExecuteAsync(
                    "bash", "-c \"update-initramfs -u 2>/dev/null || dracut -f 2>/dev/null || true\"",
                    TimeSpan.FromMinutes(2), ct);
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to configure VFIO passthrough");
        }

        return false;
    }
}
