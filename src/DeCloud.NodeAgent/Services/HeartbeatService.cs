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

    /// <summary>
    /// Register with orchestrator, reporting proper physical/logical core counts
    /// </summary>
    private async Task RegisterWithOrchestratorAsync(CancellationToken ct)
    {
        var maxRetries = 5;
        var retryDelay = TimeSpan.FromSeconds(10);

        for (var i = 0; i < maxRetries; i++)
        {
            try
            {
                _logger.LogInformation("Registration attempt {Attempt}/{Max}", i + 1, maxRetries);

                // Discover all resources
                var resources = await _resourceDiscovery.DiscoverAllAsync(ct);

                // Generate deterministic node ID
                var machineId = await GetMachineIdAsync(ct);
                var walletAddress = (_orchestratorClient as OrchestratorClient)?.WalletAddress
                    ?? "0x0000000000000000000000000000000000000000";
                var nodeId = GenerateDeterministicNodeId(machineId, walletAddress);

                _logger.LogInformation(
                    "Discovered resources: " +
                    "CPU: {PhysicalCores} physical / {LogicalCores} logical, " +
                    "Memory: {MemoryGB}GB, " +
                    "Storage: {StorageGB}GB",
                    resources.Cpu.PhysicalCores,
                    resources.Cpu.LogicalCores,
                    resources.Memory.TotalBytes / 1024 / 1024 / 1024,
                    resources.Storage.Sum(s => s.TotalBytes) / 1024 / 1024 / 1024);

                var registration = new NodeRegistration
                {
                    NodeId = nodeId,
                    MachineId = machineId,
                    Name = Environment.MachineName,
                    WalletAddress = walletAddress,
                    PublicIp = await GetPublicIpAsync(ct) ?? "127.0.0.1",
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

                var success = await _orchestratorClient.RegisterNodeAsync(registration, ct);

                if (success)
                {
                    _logger.LogInformation(
                        "✓ Successfully registered with orchestrator\n" +
                        "  Node ID: {NodeId}\n" +
                        "  Machine: {MachineId}\n" +
                        "  Wallet:  {Wallet}\n" +
                        "  CPU:     {PhysCores} physical / {LogicalCores} logical cores",
                        nodeId, machineId, walletAddress,
                        resources.Cpu.PhysicalCores,
                        resources.Cpu.LogicalCores);
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

    /// <summary>
    /// Send heartbeat with current resource snapshot
    /// </summary>
    private async Task SendHeartbeatAsync(CancellationToken ct)
    {
        // Get current resource snapshot
        var snapshot = await _resourceDiscovery.GetCurrentSnapshotAsync(ct);

        // Get all active VMs with detailed information
        var allVms = await _vmManager.GetAllVmsAsync(ct);
        var activeVms = allVms
            .Where(vm => vm.State != VmState.Deleted && vm.State != VmState.Failed)
            .ToList();

        // Build detailed VM summaries
        var vmSummaries = new List<VmSummary>();
        foreach (var vm in activeVms)
        {
            try
            {
                vmSummaries.Add(new VmSummary
                {
                    VmId = vm.VmId,
                    State = vm.State.ToString(),
                    CpuCores = vm.Spec.VCpus,
                    MemoryMb = vm.Spec.MemoryBytes / 1024 / 1024,
                    DiskGb = vm.Spec.DiskBytes / 1024 / 1024 / 1024,
                    IpAddress = vm.Spec.Network?.IpAddress,
                    VncPort = int.TryParse(vm.VncPort, out var port) ? port : null,
                    MacAddress = vm.Spec.Network?.MacAddress,
                    EncryptedPassword = vm.Spec.EncryptedPassword
                });
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to build summary for VM {VmId}", vm.VmId);
            }
        }

        // Build heartbeat with proper physical/logical core separation
        var heartbeat = new Heartbeat
        {
            Resources = new ResourceSnapshot
            {
                // CRITICAL: Report both physical and logical cores
                PhysicalCpuCores = snapshot.PhysicalCpuCores,
                LogicalCpuCores = snapshot.LogicalCpuCores,
                CpuUsagePercent = snapshot.CpuUsagePercent,
                LoadAverage = await GetLoadAverageAsync(ct),

                TotalMemoryBytes = snapshot.TotalMemoryBytes,
                UsedMemoryBytes = snapshot.UsedMemoryBytes,

                TotalStorageBytes = snapshot.TotalStorageBytes,
                UsedStorageBytes = snapshot.UsedStorageBytes,

                NetworkInMbps = 0, // TODO: Implement network monitoring
                NetworkOutMbps = 0
            },
            ActiveVmIds = activeVms.Select(vm => vm.VmId).ToList(),
            VmSummaries = vmSummaries
        };

        var success = await _orchestratorClient.SendHeartbeatAsync(heartbeat, ct);

        if (success)
        {
            _lastHeartbeat = heartbeat;

            // Log resource summary periodically
            if (DateTime.UtcNow.Second < 15) // Log once per minute-ish
            {
                var allocatedCores = vmSummaries.Sum(vm => vm.CpuCores);
                var allocatedMemMb = vmSummaries.Sum(vm => vm.MemoryMb);

                _logger.LogDebug(
                    "Heartbeat sent: {VmCount} VMs, " +
                    "Allocated: {AllocCores}c/{AllocMem}MB, " +
                    "Physical capacity: {PhysCores}c",
                    activeVms.Count,
                    allocatedCores,
                    allocatedMemMb,
                    snapshot.PhysicalCpuCores);
            }
        }
        else
        {
            _logger.LogWarning("Heartbeat failed");
        }
    }

    private async Task<string> GetMachineIdAsync(CancellationToken ct)
    {
        try
        {
            // Try /etc/machine-id first (Linux)
            if (File.Exists("/etc/machine-id"))
            {
                return (await File.ReadAllTextAsync("/etc/machine-id", ct)).Trim();
            }

            // Fallback to hostname + some hardware info
            return $"{Environment.MachineName}-{Environment.ProcessorCount}";
        }
        catch
        {
            return Environment.MachineName;
        }
    }

    private string GenerateDeterministicNodeId(string machineId, string walletAddress)
    {
        var input = $"{machineId.ToLowerInvariant()}:{walletAddress.ToLowerInvariant()}";
        using var sha256 = SHA256.Create();
        var hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(input));
        return $"node-{Convert.ToHexString(hash[..8]).ToLowerInvariant()}";
    }

    private async Task<string?> GetPublicIpAsync(CancellationToken ct)
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            return await client.GetStringAsync("https://api.ipify.org", ct);
        }
        catch
        {
            return null;
        }
    }

    private async Task<double> GetLoadAverageAsync(CancellationToken ct)
    {
        try
        {
            if (File.Exists("/proc/loadavg"))
            {
                var content = await File.ReadAllTextAsync("/proc/loadavg", ct);
                var parts = content.Split(' ');
                if (parts.Length > 0 && double.TryParse(parts[0], out var load))
                {
                    return load;
                }
            }
        }
        catch { }

        return 0;
    }
}

// =====================================================
// Heartbeat Configuration
// =====================================================

public class HeartbeatOptions
{
    public TimeSpan Interval { get; set; } = TimeSpan.FromSeconds(15);
}