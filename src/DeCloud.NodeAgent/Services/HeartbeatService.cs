// Updated HeartbeatService.cs for Node Agent
// Sends detailed VM information with each heartbeat

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Services;

public class HeartbeatService : BackgroundService
{
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly IVmManager _vmManager;
    private readonly VmRepository _repository;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly INodeStateService _nodeState;
    private readonly INodeMetadataService _nodeMetadata;
    private readonly HeartbeatOptions _options;
    private readonly ILogger<HeartbeatService> _logger;
    private Heartbeat? _lastHeartbeat = null;

    public HeartbeatService(
        IResourceDiscoveryService resourceDiscovery,
        IVmManager vmManager,
        VmRepository repository,
        IOrchestratorClient orchestratorClient,
        INodeStateService nodeState,
        INodeMetadataService nodeMetadata,
        IOptions<HeartbeatOptions> options,
        ILogger<HeartbeatService> logger)
    {
        _resourceDiscovery = resourceDiscovery;
        _vmManager = vmManager;
        _repository = repository;
        _orchestratorClient = orchestratorClient;
        _nodeState = nodeState;
        _nodeMetadata = nodeMetadata;
        _options = options.Value;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        _logger.LogInformation("Heartbeat service starting with interval {Interval}s",
            _options.Interval.TotalSeconds);

        // Wait for node to be registered
        await _nodeState.WaitForAuthenticationAsync(ct);

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
            // Apply quota to Burstable non-relay type VMs after boot
            // =====================================================
            foreach (var vm in activeVms.Where(v => v.Spec.QualityTier == QualityTier.Burstable && v.Spec.VmType == VmType.General))
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
                    var actualState = vm.State; //Fetvh actual virsh state using CommandExecutor
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

                        if (isIpAssigned && vm.Spec.IpAddress != ipAddress)
                        {
                            vm.Spec.IpAddress = ipAddress!;
                            await _repository.SaveVmAsync(vm);
                            _logger.LogInformation(
                                "Updated VM {VmId} IP address: {IpAddress}",
                                vm.VmId, ipAddress);
                        }
                    }

                    vmSummaries.Add(new VmSummary
                    {
                        VmId = vm.VmId,
                        Name = vm.Name,
                        OwnerId = vm.Spec.OwnerId,
                        State = vm.State,
                        VirtualCpuCores = vm.Spec.VirtualCpuCores,
                        QualityTier = (int)vm.Spec.QualityTier,
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
            snapshot.UsedVirtualCpuCores = activeVms.Where(v => v.State == VmState.Running)
                .Sum(v => v.Spec.VirtualCpuCores);
            snapshot.UsedMemoryBytes = activeVms.Where(v => v.State == VmState.Running)
                .Sum(v => v.Spec.MemoryBytes);
            // Calculate used compute points
            snapshot.UsedComputePoints = activeVms.Where(v => v.State == VmState.Running)
                .Sum(v => v.Spec.ComputePointCost);

            // Get CGNAT info from last heartbeat response (if available)
            var lastHeartbeat = _orchestratorClient.GetLastHeartbeat();
            var cgnatInfo = lastHeartbeat?.Heartbeat?.CgnatInfo;

            // Create heartbeat using the standard Heartbeat model
            var heartbeat = new Heartbeat
            {
                NodeId = (_orchestratorClient as OrchestratorClient)?.NodeId ?? Environment.MachineName,
                Timestamp = DateTime.UtcNow,
                Status = _nodeState.Status,
                Resources = snapshot,
                ActiveVms = vmSummaries,  // Now includes VncPort, MacAddress, EncryptedPassword
                SchedulingConfigVersion = _nodeMetadata.GetSchedulingConfigVersion(),
                CgnatInfo = cgnatInfo
            };

            // Send heartbeat - OrchestratorClient will transform to API format
            var success = await _orchestratorClient.SendHeartbeatAsync(heartbeat, ct);

            _nodeState.RecordHeartbeat(success);

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
        if (vm.Spec.QualityTier != QualityTier.Burstable)
            return false;

        // Only apply to running VMs
        if (vm.State != VmState.Running)
            return false;

        // Already applied?
        if (vm.Spec.VcpuQuotaAppliedAt.HasValue)
            return false;

        // VM must have an IP (indicates guest agent is working)
        if (string.IsNullOrEmpty(vm.Spec.IpAddress))
            return false;

        // VM must be running for at least 120 seconds
        if (!vm.StartedAt.HasValue)
            return false;

        var uptime = DateTime.UtcNow - vm.StartedAt.Value;
        if (uptime < TimeSpan.FromSeconds(120))
            return false;

        return true;
    }

    /// <summary>
    /// Apply quota cap to a Burstable VM based on node performance
    /// Formula: quota = 1 / (points_per_core × overcommit)
    /// </summary>
    private async Task ApplyBurstableQuotaAsync(VmInstance vm, CancellationToken ct)
    {
        try
        {
            var uptime = vm.StartedAt.HasValue
                ? (DateTime.UtcNow - vm.StartedAt.Value).TotalSeconds
                : 0;

            _logger.LogInformation(
                "VM {VmId} ({Name}) is ready - calculating Burstable tier quota (uptime: {Uptime:F0}s)",
                vm.VmId, vm.Name, uptime);

            var performanceEval = _nodeMetadata.PerformanceEvaluation;
            if (performanceEval == null)
            {
                _logger.LogWarning("VM {VmId}: Performance evaluation not available, skipping quota", vm.VmId);
                return;
            }

            // =====================================================
            // Calculate quota based on point-fair share with 4x burst
            // =====================================================
            var vmPoints = (double)vm.Spec.ComputePointCost;
            var totalNodePoints = (double)performanceEval.TotalComputePoints;
            var physicalCores = performanceEval.PhysicalCores;

            // Fair share percentage
            var fairSharePercent = vmPoints / totalNodePoints;

            // Fixed period (100ms standard)
            const int periodMicroseconds = 100000;

            // Physical capacity in µs (what 100% of node = in quota terms)
            var physicalCapacity = physicalCores * periodMicroseconds;

            // Fair share quota
            var fairShareQuota = (int)(physicalCapacity * fairSharePercent);

            // 4x burst (allows burst up to non-overcommitted fair share)
            const double burstMultiplier = 4.0;
            var burstQuota = (int)(fairShareQuota * burstMultiplier);

            // Cap at 50% of physical node (no single Burstable VM dominates)
            var maxNodeShareQuota = (int)(physicalCapacity * 0.5);

            // Apply all caps
            var finalQuota = Math.Min(burstQuota, Math.Min(physicalCapacity, maxNodeShareQuota));

            // Safety floor (at least 1%)
            finalQuota = Math.Max(finalQuota, periodMicroseconds / 100);

            _logger.LogInformation(
                "VM {VmId}: Quota calculation - Points={VmPts}/{TotalPts} ({FairShare:F2}%), " +
                "Fair={Fair}µs, 4xBurst={Burst}µs, 50%Cap={Cap}µs, Final={Final}µs ({FinalCores:F2} cores)",
                vm.VmId,
                vmPoints, totalNodePoints, fairSharePercent * 100,
                fairShareQuota, burstQuota, maxNodeShareQuota, finalQuota,
                (double)finalQuota / periodMicroseconds);

            // =====================================================
            // Apply quota via virsh schedinfo
            // =====================================================
            var success = await _vmManager.ApplyQuotaCapAsync(
                vm,
                finalQuota,
                periodMicroseconds,
                ct);

            if (success)
            {
                _logger.LogInformation(
                    "✓ Applied quota to Burstable VM {VmId}: {Final}µs/{Period}µs ({Percent:F2}% of node)",
                    vm.VmId, finalQuota, periodMicroseconds,
                    (double)finalQuota / physicalCapacity * 100);
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