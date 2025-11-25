using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Services;

public class HeartbeatOptions
{
    public TimeSpan Interval { get; set; } = TimeSpan.FromSeconds(15);
    public string OrchestratorUrl { get; set; } = "https://api.decloud.network";
}

public class HeartbeatService : BackgroundService
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly IVmManager _vmManager;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly ILogger<HeartbeatService> _logger;
    private readonly HeartbeatOptions _options;
    private readonly string _nodeId;

    private NodeStatus _currentStatus = NodeStatus.Initializing;

    public HeartbeatService(
        IResourceDiscoveryService resourceDiscovery,
        IVmManager vmManager,
        IOrchestratorClient orchestratorClient,
        IOptions<HeartbeatOptions> options,
        ILogger<HeartbeatService> logger)
    {
        _resourceDiscovery = resourceDiscovery;
        _vmManager = vmManager;
        _orchestratorClient = orchestratorClient;
        _logger = logger;
        _options = options.Value;
        
        // Get node ID from resource discovery
        _nodeId = Environment.MachineName; // Simplified - get from proper source
    }

    public void SetStatus(NodeStatus status)
    {
        _currentStatus = status;
        _logger.LogInformation("Node status changed to {Status}", status);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Heartbeat service starting with interval {Interval}s", 
            _options.Interval.TotalSeconds);

        // Initial delay to let other services start
        await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
        
        _currentStatus = NodeStatus.Online;

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await SendHeartbeatAsync(stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to send heartbeat");
            }

            await Task.Delay(_options.Interval, stoppingToken);
        }

        _logger.LogInformation("Heartbeat service stopping");
    }

    private async Task SendHeartbeatAsync(CancellationToken ct)
    {
        var snapshot = await _resourceDiscovery.GetCurrentSnapshotAsync(ct);
        var vms = await _vmManager.GetAllVmsAsync(ct);

        var heartbeat = new Heartbeat
        {
            NodeId = _nodeId,
            Timestamp = DateTime.UtcNow,
            Status = _currentStatus,
            Health = await PerformHealthChecksAsync(ct),
            Resources = snapshot,
            ActiveVms = vms.Select(vm => new VmSummary
            {
                VmId = vm.VmId,
                TenantId = vm.Spec.TenantId,
                LeaseId = vm.Spec.LeaseId,
                State = vm.State,
                VCpus = vm.Spec.VCpus,
                MemoryBytes = vm.Spec.MemoryBytes,
                CpuUsagePercent = vm.CurrentUsage.CpuPercent,
                StartedAt = vm.StartedAt ?? vm.CreatedAt
            }).ToList()
        };

        // Update resource usage based on running VMs
        heartbeat.Resources.UsedVCpus = vms.Where(v => v.State == VmState.Running).Sum(v => v.Spec.VCpus);
        heartbeat.Resources.UsedMemoryBytes = vms.Where(v => v.State == VmState.Running).Sum(v => v.Spec.MemoryBytes);

        var success = await _orchestratorClient.SendHeartbeatAsync(heartbeat, ct);
        
        if (success)
        {
            _logger.LogDebug("Heartbeat sent: {VmCount} VMs, {CpuAvail} vCPUs available",
                vms.Count, heartbeat.Resources.AvailableVCpus);
        }
        else
        {
            _logger.LogWarning("Failed to deliver heartbeat to orchestrator");
        }
    }

    private async Task<NodeHealth> PerformHealthChecksAsync(CancellationToken ct)
    {
        var checks = new List<HealthCheck>();

        // Check libvirt
        checks.Add(await CheckLibvirtAsync(ct));

        // Check WireGuard
        checks.Add(await CheckWireGuardAsync(ct));

        // Check disk space
        checks.Add(await CheckDiskSpaceAsync(ct));

        // Check memory
        checks.Add(await CheckMemoryAsync(ct));

        return new NodeHealth
        {
            IsHealthy = checks.All(c => c.Passed),
            Checks = checks
        };
    }

    private async Task<HealthCheck> CheckLibvirtAsync(CancellationToken ct)
    {
        var isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
            System.Runtime.InteropServices.OSPlatform.Windows);
            
        if (isWindows)
        {
            // On Windows, check if Hyper-V is available (optional)
            return new HealthCheck
            {
                Name = "hypervisor",
                Passed = true,
                Message = "Windows - VM management via Hyper-V (if enabled)",
                CheckedAt = DateTime.UtcNow
            };
        }
        
        try
        {
            var result = await RunCommandAsync("virsh", "version", ct);
            return new HealthCheck
            {
                Name = "libvirt",
                Passed = result.exitCode == 0,
                Message = result.exitCode == 0 ? "OK" : result.stderr,
                CheckedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            return new HealthCheck
            {
                Name = "libvirt",
                Passed = false,
                Message = ex.Message,
                CheckedAt = DateTime.UtcNow
            };
        }
    }

    private async Task<HealthCheck> CheckWireGuardAsync(CancellationToken ct)
    {
        var isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
            System.Runtime.InteropServices.OSPlatform.Windows);
            
        try
        {
            // Check if wg command exists
            var whereCmd = isWindows ? "where" : "which";
            var checkResult = await RunCommandAsync(whereCmd, "wg", ct);
            
            if (checkResult.exitCode != 0)
            {
                return new HealthCheck
                {
                    Name = "wireguard",
                    Passed = false,
                    Message = "WireGuard not installed",
                    CheckedAt = DateTime.UtcNow
                };
            }

            // On Windows, just check if WG is installed (interface managed by WG GUI)
            if (isWindows)
            {
                return new HealthCheck
                {
                    Name = "wireguard",
                    Passed = true,
                    Message = "WireGuard installed (managed via WireGuard GUI)",
                    CheckedAt = DateTime.UtcNow
                };
            }
            
            var result = await RunCommandAsync("wg", "show wg0", ct);
            return new HealthCheck
            {
                Name = "wireguard",
                Passed = result.exitCode == 0,
                Message = result.exitCode == 0 ? "OK" : "WireGuard interface not found",
                CheckedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            return new HealthCheck
            {
                Name = "wireguard",
                Passed = false,
                Message = ex.Message,
                CheckedAt = DateTime.UtcNow
            };
        }
    }

    private async Task<HealthCheck> CheckDiskSpaceAsync(CancellationToken ct)
    {
        var memory = await _resourceDiscovery.GetMemoryInfoAsync(ct);
        var storage = await _resourceDiscovery.GetStorageInfoAsync(ct);
        
        // Find the storage for /var/lib/decloud
        var vmStorage = storage.FirstOrDefault(s => 
            s.MountPoint == "/" || s.MountPoint.StartsWith("/var")) ?? storage.FirstOrDefault();

        if (vmStorage == null)
        {
            return new HealthCheck
            {
                Name = "disk_space",
                Passed = false,
                Message = "No storage found",
                CheckedAt = DateTime.UtcNow
            };
        }

        var freePercent = (double)vmStorage.AvailableBytes / vmStorage.TotalBytes * 100;
        var passed = freePercent > 10; // At least 10% free

        return new HealthCheck
        {
            Name = "disk_space",
            Passed = passed,
            Message = $"{freePercent:F1}% free ({vmStorage.AvailableBytes / 1024 / 1024 / 1024}GB)",
            CheckedAt = DateTime.UtcNow
        };
    }

    private async Task<HealthCheck> CheckMemoryAsync(CancellationToken ct)
    {
        var memory = await _resourceDiscovery.GetMemoryInfoAsync(ct);
        var freePercent = (double)memory.AvailableBytes / memory.TotalBytes * 100;
        var passed = freePercent > 5; // At least 5% free

        return new HealthCheck
        {
            Name = "memory",
            Passed = passed,
            Message = $"{freePercent:F1}% free ({memory.AvailableBytes / 1024 / 1024}MB)",
            CheckedAt = DateTime.UtcNow
        };
    }

    private async Task<(int exitCode, string stdout, string stderr)> RunCommandAsync(
        string command, string args, CancellationToken ct)
    {
        using var process = new System.Diagnostics.Process
        {
            StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = command,
                Arguments = args,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            }
        };

        process.Start();
        var stdout = await process.StandardOutput.ReadToEndAsync(ct);
        var stderr = await process.StandardError.ReadToEndAsync(ct);
        await process.WaitForExitAsync(ct);

        return (process.ExitCode, stdout, stderr);
    }
}
