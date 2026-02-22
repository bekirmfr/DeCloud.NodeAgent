using DeCloud.NodeAgent.Core.Models;
using Xunit;

namespace DeCloud.NodeAgent.Tests;

public class GpuUsageStatsTests
{
    [Fact]
    public void GpuUsageStats_DefaultsToZero()
    {
        var stats = new GpuUsageStats();

        Assert.Equal(0, stats.MemoryAllocated);
        Assert.Equal(0, stats.MemoryQuota);
        Assert.Equal(0, stats.PeakMemory);
        Assert.Equal(0, stats.TotalAllocBytes);
        Assert.Equal(0, stats.KernelLaunches);
        Assert.Equal(0, stats.KernelTimeouts);
        Assert.Equal(0, stats.KernelTimeUs);
        Assert.Equal(0, stats.ConnectTimeUs);
    }

    [Fact]
    public void GpuUsageStats_CanSetProperties()
    {
        var stats = new GpuUsageStats
        {
            MemoryAllocated = 512 * 1024 * 1024L,
            MemoryQuota = 2L * 1024 * 1024 * 1024,
            PeakMemory = 768 * 1024 * 1024L,
            TotalAllocBytes = 1L * 1024 * 1024 * 1024,
            KernelLaunches = 42,
            KernelTimeouts = 1,
            KernelTimeUs = 123456789L,
            ConnectTimeUs = 987654321L,
        };

        Assert.Equal(512 * 1024 * 1024L, stats.MemoryAllocated);
        Assert.Equal(2L * 1024 * 1024 * 1024, stats.MemoryQuota);
        Assert.Equal(768 * 1024 * 1024L, stats.PeakMemory);
        Assert.Equal(1L * 1024 * 1024 * 1024, stats.TotalAllocBytes);
        Assert.Equal(42, stats.KernelLaunches);
        Assert.Equal(1, stats.KernelTimeouts);
        Assert.Equal(123456789L, stats.KernelTimeUs);
        Assert.Equal(987654321L, stats.ConnectTimeUs);
    }

    [Fact]
    public void GpuUsageStats_QuotaZeroMeansUnlimited()
    {
        var stats = new GpuUsageStats { MemoryQuota = 0 };

        // By convention, 0 = unlimited
        Assert.Equal(0, stats.MemoryQuota);
    }
}
