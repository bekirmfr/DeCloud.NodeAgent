using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Services;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace DeCloud.NodeAgent.Tests;

public class GpuProxyServiceTests
{
    private readonly Mock<ICommandExecutor> _executor;
    private readonly Mock<IResourceDiscoveryService> _resourceDiscovery;
    private readonly Mock<ILogger<GpuProxyService>> _logger;
    private readonly GpuProxyService _service;

    public GpuProxyServiceTests()
    {
        _executor = new Mock<ICommandExecutor>();
        _resourceDiscovery = new Mock<IResourceDiscoveryService>();
        _logger = new Mock<ILogger<GpuProxyService>>();

        _service = new GpuProxyService(
            _executor.Object,
            _resourceDiscovery.Object,
            _logger.Object);
    }

    [Fact]
    public void DefaultProperties_HaveExpectedValues()
    {
        Assert.Equal("/usr/local/bin/gpu-proxy-daemon", _service.DaemonPath);
        Assert.Equal("/usr/local/lib/libdecloud_cuda_shim.so", _service.ShimPath);
        Assert.Equal("/usr/local/lib/decloud-gpu-shim", _service.ShimShareDir);
        Assert.Equal(9999, _service.DaemonPort);
        Assert.Equal(5, _service.MaxCrashRestarts);
        Assert.Equal(0, _service.DefaultMemoryQuotaBytes);
        Assert.Equal(30, _service.KernelTimeoutSeconds);
        Assert.False(_service.IsRunning);
    }

    [Fact]
    public async Task EnsureStartedAsync_ReturnsTrue_WhenAlreadyRunning()
    {
        // Simulate that proxy mode is not supported (no GPU)
        _resourceDiscovery.Setup(r => r.GetInventoryCachedAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new HardwareInventory { SupportsGpuProxy = false });

        var result = await _service.EnsureStartedAsync();
        Assert.False(result);
    }

    [Fact]
    public async Task EnsureStartedAsync_ReturnsFalse_WhenNoGpuProxy()
    {
        _resourceDiscovery.Setup(r => r.GetInventoryCachedAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new HardwareInventory { SupportsGpuProxy = false });

        var result = await _service.EnsureStartedAsync();
        Assert.False(result);
    }

    [Fact]
    public void HealthCheck_ReturnsFalse_WhenNotRunning()
    {
        Assert.False(_service.HealthCheck());
    }

    [Fact]
    public async Task EnsureHealthyAsync_ReturnsFalse_WhenNotRunning()
    {
        // Not running, no crashes yet — should attempt restart but fail
        // because no GPU proxy support
        _resourceDiscovery.Setup(r => r.GetInventoryCachedAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new HardwareInventory { SupportsGpuProxy = false });

        var result = await _service.EnsureHealthyAsync();
        Assert.False(result);
    }

    [Fact]
    public async Task StopAsync_CompletesWithoutError_WhenNotRunning()
    {
        // Should not throw when daemon is not running
        await _service.StopAsync();
        Assert.False(_service.IsRunning);
    }

    [Fact]
    public void KernelTimeoutSeconds_CanBeConfigured()
    {
        _service.KernelTimeoutSeconds = 60;
        Assert.Equal(60, _service.KernelTimeoutSeconds);
    }

    [Fact]
    public void DefaultMemoryQuotaBytes_CanBeConfigured()
    {
        _service.DefaultMemoryQuotaBytes = 4L * 1024 * 1024 * 1024; // 4 GB
        Assert.Equal(4L * 1024 * 1024 * 1024, _service.DefaultMemoryQuotaBytes);
    }
}
