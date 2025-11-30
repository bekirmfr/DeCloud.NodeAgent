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

    private async Task RegisterWithOrchestratorAsync(CancellationToken ct)
    {
        var maxRetries = 5;
        var retryDelay = TimeSpan.FromSeconds(10);

        for (var i = 0; i < maxRetries; i++)
        {
            try
            {
                _logger.LogInformation("Attempting to register with orchestrator (attempt {Attempt}/{Max})",
                    i + 1, maxRetries);

                var resources = await _resourceDiscovery.DiscoverAllAsync(ct);
                var publicIp = await GetPublicIpAsync(ct);

                // Build registration using correct NodeRegistration properties
                var registration = new NodeRegistration
                {
                    Name = Environment.MachineName,
                    NodeId = Environment.MachineName,
                    PublicIp = publicIp,                          // FIXED: Direct property
                    AgentPort = _options.AgentPort,               // FIXED: Direct property
                    Resources = resources,
                    AgentVersion = "1.0.0",                       // FIXED: Direct property
                    SupportedImages = new List<string>            // FIXED: Direct property
                    {
                        "ubuntu-24.04",
                        "ubuntu-22.04",
                        "debian-12"
                    },
                    SupportsGpu = resources.Gpus.Any(),           // FIXED: Direct property
                    GpuInfo = resources.Gpus.FirstOrDefault(),    // FIXED: Direct property
                    WalletAddress = _options.WalletAddress ?? "0x0000000000000000000000000000000000000000",
                    Region = "default",                           // FIXED: Direct property
                    Zone = "default"                              // FIXED: Direct property
                };

                var success = await _orchestratorClient.RegisterNodeAsync(registration, ct);

                if (success)
                {
                    _logger.LogInformation("Successfully registered with orchestrator");
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
            // Build detailed VM summaries for heartbeat
            // UPDATED: Include VncPort, MacAddress, EncryptedPassword
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
                        ipAddress = vm.Spec.Network.IpAddress ?? await _vmManager.GetVmIpAddressAsync(vm.VmId, ct);
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
                        IpAddress = ipAddress,
                        CpuUsagePercent = usage?.CpuPercent ?? 0,
                        StartedAt = vm.StartedAt ?? vm.CreatedAt,

                        // ADDED: Recovery fields from VmInstance
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