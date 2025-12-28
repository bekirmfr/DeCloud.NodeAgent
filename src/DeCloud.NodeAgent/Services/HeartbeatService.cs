// Updated HeartbeatService.cs for Node Agent
// Sends detailed VM information with each heartbeat

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared;
using Microsoft.Extensions.Options;
using System.Security.Cryptography;
using System.Text;

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

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Heartbeat service starting with interval {Interval}s",
            _options.Interval.TotalSeconds);

        // Initial delay to let other services start
        await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);

        // Register with orchestrator
        var orchestratorClient = _orchestratorClient as OrchestratorClient;
        if (orchestratorClient != null && !orchestratorClient.IsRegistered)
        {
            await RegisterWithOrchestratorAsync(stoppingToken);
        }

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

    // Updated HeartbeatService.RegisterWithOrchestratorAsync
    // Generates deterministic node ID before registration

    private async Task RegisterWithOrchestratorAsync(CancellationToken ct)
    {
        var maxRetries = 5;
        var retryDelay = TimeSpan.FromSeconds(10);

        for (var i = 0; i < maxRetries; i++)
        {
            try
            {
                _logger.LogInformation("Registration attempt {Attempt}/{Max}", i + 1, maxRetries);

                // =====================================================
                // STEP 1: Get Machine ID
                // =====================================================
                string machineId;
                try
                {
                    machineId = NodeIdGenerator.GetMachineId();
                    _logger.LogInformation("Machine ID: {MachineId}", machineId);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to get machine ID");
                    throw;
                }

                // =====================================================
                // STEP 2: Get Wallet Address
                // =====================================================
                var walletAddress = _orchestratorClient.WalletAddress;

                // Validate wallet address
                if (string.IsNullOrWhiteSpace(walletAddress) ||
                    walletAddress == "0x0000000000000000000000000000000000000000")
                {
                    _logger.LogError(
                        "❌ CRITICAL: No valid wallet address configured! " +
                        "Set 'Orchestrator:WalletAddress' in appsettings.json or NODE_WALLET_ADDRESS environment variable.");

                    throw new InvalidOperationException("No valid wallet address configured");
                }

                _logger.LogInformation("Wallet address: {Wallet}", walletAddress);

                // =====================================================
                // STEP 3: Generate Deterministic Node ID
                // =====================================================
                string nodeId;
                try
                {
                    nodeId = NodeIdGenerator.GenerateNodeId(machineId, walletAddress);
                    _logger.LogInformation("✓ Generated deterministic node ID: {NodeId}", nodeId);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to generate node ID");
                    throw;
                }

                // =====================================================
                // STEP 4: Build Registration Request
                // =====================================================
                var resources = await _resourceDiscovery.DiscoverAllAsync(ct);
                var publicIp = await GetPublicIpAsync(ct);

                var registration = new NodeRegistration
                {
                    NodeId = nodeId,           // ← Deterministic node ID
                    MachineId = machineId,     // ← For validation
                    Name = Environment.MachineName,
                    WalletAddress = walletAddress,
                    PublicIp = publicIp ?? "127.0.0.1",
                    AgentPort = 5100,
                    Resources = resources,
                    AgentVersion = "2.0.0",
                    SupportedImages = new List<string>
                {
                    "ubuntu-24.04", "ubuntu-22.04", "ubuntu-20.04",
                    "debian-12", "debian-11",
                    "fedora-40", "fedora-39",
                    "alpine-3.19", "alpine-3.18"
                },
                    SupportsGpu = resources.Gpus.Any(),
                    GpuInfo = resources.Gpus.FirstOrDefault(),
                    Region = "default",
                    Zone = "default"
                };

                // =====================================================
                // STEP 5: Register with Orchestrator
                // =====================================================
                var success = await _orchestratorClient.RegisterNodeAsync(registration, ct);

                if (success)
                {
                    _logger.LogInformation(
                        "✓ Successfully registered with orchestrator\n" +
                        "  Node ID: {NodeId}\n" +
                        "  Machine: {MachineId}\n" +
                        "  Wallet:  {Wallet}",
                        nodeId, machineId, walletAddress);
                    return;
                }

                _logger.LogWarning("Registration attempt {Attempt} failed", i + 1);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Registration attempt {Attempt} failed", i + 1);
            }

            if (i < maxRetries - 1)
            {
                await Task.Delay(retryDelay, ct);
            }
        }

        _logger.LogError("Failed to register with orchestrator after {Retries} attempts", maxRetries);
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
                    if (vm.State == VmState.Running)
                    {
                        // CORRECT: Always get fresh libvirt IP first, fall back to stored IP
                        ipAddress = await _vmManager.GetVmIpAddressAsync(vm.VmId, ct) ?? vm.Spec.Network.IpAddress;
                        var vncPort = vm.VncPort;
                    }

                    vmSummaries.Add(new VmSummary
                    {
                        VmId = vm.VmId,
                        Name = vm.Name,
                        TenantId = vm.Spec.TenantId,
                        LeaseId = vm.Spec.LeaseId,
                        State = vm.State,
                        VCpus = vm.Spec.VCpus,
                        MemoryBytes = vm.Spec.MemoryBytes,
                        DiskBytes = vm.Spec.DiskBytes,
                        CpuUsagePercent = usage?.CpuPercent ?? 0,
                        StartedAt = vm.StartedAt ?? vm.CreatedAt,
                        IpAddress = ipAddress,
                        VncPort = vm.VncPort,
                        MacAddress = vm.Spec.Network.MacAddress,
                        EncryptedPassword = vm.Spec.EncryptedPassword
                    });
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to get details for VM {VmId}", vm.VmId);
                }
            }

            // Update resource usage based on running VMs
            snapshot.UsedVCpus = activeVms.Where(v => v.State == VmState.Running).Sum(v => v.Spec.VCpus);
            snapshot.UsedMemoryBytes = activeVms.Where(v => v.State == VmState.Running).Sum(v => v.Spec.MemoryBytes);

            // Create heartbeat using the standard Heartbeat model
            var heartbeat = new Heartbeat
            {
                NodeId = (_orchestratorClient as OrchestratorClient)?.NodeId ?? Environment.MachineName,
                Timestamp = DateTime.UtcNow,
                Status = _currentStatus,
                Resources = snapshot,
                ActiveVmDetails = vmSummaries  // Now includes VncPort, MacAddress, EncryptedPassword
            };

            // Send heartbeat - OrchestratorClient will transform to API format
            var success = await _orchestratorClient.SendHeartbeatAsync(heartbeat, ct);

            if (success)
            {
                _lastHeartbeat = heartbeat;
                _logger.LogDebug("Heartbeat sent: {VmCount} VMs, CPU {Cpu}%, MEM {Mem}%",
                    activeVms.Count,
                    snapshot.CpuUsagePercent,
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
        if (string.IsNullOrEmpty(vm.Spec.Network.IpAddress))
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
            var totalQuota = vm.Spec.VCpus * quotaPerVCpu;

            var success = await _vmManager.ApplyQuotaCapAsync(
                vm.VmId,
                totalQuota,
                periodMicroseconds: 100000,
                ct);

            if (success)
            {
                _logger.LogInformation(
                    "✓ Applied {Percent}% CPU quota to Burstable VM {VmId} ({VCpus} vCPUs)",
                    50, vm.VmId, vm.Spec.VCpus);
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

    private async Task<string> GetPublicIpAsync(CancellationToken ct)
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            return (await client.GetStringAsync("https://api.ipify.org", ct)).Trim();
        }
        catch
        {
            return "127.0.0.1";
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