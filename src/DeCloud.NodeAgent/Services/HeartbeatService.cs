// Updated HeartbeatService.cs for Node Agent
// Sends detailed VM information with each heartbeat

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Services;

public class HeartbeatService : BackgroundService
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly IVmManager _vmManager;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly HeartbeatOptions _options;
    private readonly ILogger<HeartbeatService> _logger;
    private Heartbeat? _lastHeartbeat = null;
    private NodeStatus _currentStatus = NodeStatus.Offline;

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
        _options = options.Value;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        _logger.LogInformation("Heartbeat service starting with interval {Interval}s",
            _options.Interval.TotalSeconds);

        // Initial delay to let other services start
        await Task.Delay(TimeSpan.FromSeconds(10), ct);

        // Wait for node to be registered
        while (!_orchestratorClient.IsRegistered && !ct.IsCancellationRequested)
            {
                _logger.LogInformation("Waiting for node registration...");
                await Task.Delay(TimeSpan.FromSeconds(10), ct);
            }

        _currentStatus = NodeStatus.Online;
        _logger.LogInformation("✓ Node registered, starting heartbeats");

        // Send heartbeats
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await SendHeartbeatAsync(ct);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to send heartbeat");
            }

            await Task.Delay(_options.Interval, ct);
        }

        _logger.LogInformation("Heartbeat service stopping");
    }

    // HeartbeatService.cs - Updated snippet showing proper VmSummary population

    private async Task SendHeartbeatAsync(CancellationToken ct)
    {
        try
        {
            // Get current resource snapshot
            var snapshot = await _resourceDiscovery.GetCurrentSnapshotAsync(ct);

            // Get all active VMs with detailed information
            var allVms = await _vmManager.GetAllVmsAsync(ct);
            var activeVms = allVms
                .Where(vm => vm.State != VmState.Deleted && vm.State != VmState.Failed)
                .ToList();

            // =====================================================
            // Apply quota to Burstable VMs after boot
            // =====================================================
            foreach (var vm in activeVms)
            {
                if (ShouldApplyQuota(vm))
                {
                    await ApplyBurstableQuotaAsync(vm, ct);
                }
            }

            // =====================================================
            // Build detailed VM summaries for heartbeat
            // =====================================================
            var vmSummaries = new List<VmSummary>();

            foreach (var vm in activeVms)
            {
                try
                {
                    // Get current usage metrics if VM is running
                    var usage = vm.State == VmState.Running
                        ? await _vmManager.GetVmUsageAsync(vm.VmId, ct)
                        : null;

                    // Get IP address for running VMs
                    string? ipAddress = null;
                    bool isIpAssigned = false;
                    if (vm.State == VmState.Running)
                    {
                        // Always get fresh libvirt IP first, fall back to stored IP
                        var vmIpAddress = await _vmManager.GetVmIpAddressAsync(vm.VmId, ct);
                        isIpAssigned = !string.IsNullOrEmpty(vmIpAddress);
                        ipAddress = vmIpAddress;
                        var vncPort = vm.VncPort;
                    }

                    vmSummaries.Add(new VmSummary
                    {
                        VmId = vm.VmId,
                        Name = vm.Name,
                        OwnerId = vm.Spec.OwnerId,
                        State = vm.State,
                        VirtualCpuCores = vm.Spec.VirtualCpuCores,
                        QualityTier = vm.Spec.QualityTier,
                        ComputePointCost = vm.Spec.ComputePointCost,
                        MemoryBytes = vm.Spec.MemoryBytes,
                        DiskBytes = vm.Spec.DiskBytes,
                        VirtualCpuUsagePercent = usage?.CpuPercent ?? 0,
                        StartedAt = vm.StartedAt ?? vm.CreatedAt,
                        IsIpAssigned = isIpAssigned,
                        IpAddress = ipAddress,
                        VncPort = vm.VncPort,
                        MacAddress = vm.Spec.MacAddress
                    });
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to get details for VM {VmId}", vm.VmId);
                }
            }

            // Update resource usage based on running VMs
            snapshot.UsedVirtualCpuCores = activeVms.Where(v => v.State == VmState.Running).Sum(v => v.Spec.VirtualCpuCores);
            snapshot.UsedMemoryBytes = activeVms.Where(v => v.State == VmState.Running).Sum(v => v.Spec.MemoryBytes);

            // Create heartbeat using the standard Heartbeat model
            var heartbeat = new Heartbeat
            {
                NodeId = (_orchestratorClient as OrchestratorClient)?.NodeId ?? Environment.MachineName,
                Timestamp = DateTime.UtcNow,
                Status = _currentStatus,
                Resources = snapshot,
                ActiveVms = vmSummaries  // Now includes VncPort, MacAddress, EncryptedPassword
            };

            // Send heartbeat - OrchestratorClient will transform to API format
            var success = await _orchestratorClient.SendHeartbeatAsync(heartbeat, ct);

            if (success)
            {
                _lastHeartbeat = heartbeat;
                _logger.LogDebug("Heartbeat sent: {VmCount} VMs, CPU {Cpu}%, MEM {Mem}%",
                    activeVms.Count,
                    snapshot.VirtualCpuUsagePercent,
                    snapshot.TotalMemoryBytes > 0
                        ? (double)snapshot.UsedMemoryBytes / snapshot.TotalMemoryBytes * 100
                        : 0);
            }
            else
            {
                _logger.LogWarning("Heartbeat failed - orchestrator may be unreachable");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending heartbeat");
        }
    }
    /// <summary>
    /// Check if a Burstable VM needs quota applied
    /// </summary>
    private bool ShouldApplyQuota(VmInstance vm)
    {
        // Only apply to Burstable tier (QualityTier = 3)
        if (vm.Spec.QualityTier != 3)
            return false;

        // Only apply to running VMs
        if (vm.State != VmState.Running)
            return false;

        // Already applied?
        if (vm.QuotaAppliedAt.HasValue)
            return false;

        // VM must have an IP (indicates guest agent is working)
        if (string.IsNullOrEmpty(vm.Spec.IpAddress))
            return false;

        // VM must be running for at least 60 seconds
        if (!vm.StartedAt.HasValue)
            return false;

        var uptime = DateTime.UtcNow - vm.StartedAt.Value;
        if (uptime < TimeSpan.FromSeconds(60))
            return false;

        return true;
    }

    /// <summary>
    /// Apply quota cap to a Burstable VM
    /// </summary>
    private async Task ApplyBurstableQuotaAsync(VmInstance vm, CancellationToken ct)
    {
        try
        {
            _logger.LogInformation(
                "VM {VmId} ({Name}) is ready - applying Burstable tier quota cap (uptime: {Uptime}s)",
                vm.VmId, vm.Name,
                vm.StartedAt.HasValue ? (DateTime.UtcNow - vm.StartedAt.Value).TotalSeconds : 0);

            // Calculate quota: 50% per vCPU
            var quotaPerVCpu = 50000; // 50ms per 100ms period = 50%
            var totalQuota = vm.Spec.VirtualCpuCores * quotaPerVCpu;

            var success = await _vmManager.ApplyQuotaCapAsync(
                vm,
                totalQuota,
                periodMicroseconds: 100000,
                ct);

            if (success)
            {
                _logger.LogInformation(
                    "✓ Applied {Percent}% CPU quota to Burstable VM {VmId} ({VCpus} vCPUs)",
                    50, vm.VmId, vm.Spec.VirtualCpuCores);
            }
            else
            {
                _logger.LogWarning(
                    "Failed to apply quota to VM {VmId} - will retry on next heartbeat",
                    vm.VmId);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying quota to VM {VmId}", vm.VmId);
        }
    }
}

// =====================================================
// Heartbeat Configuration
// =====================================================

public class HeartbeatOptions
{
    public TimeSpan Interval { get; set; } = TimeSpan.FromSeconds(15);
    public string OrchestratorUrl { get; set; } = "http://localhost:5000";
    public string? WalletAddress { get; set; }
    public int AgentPort { get; set; } = 5100;
}