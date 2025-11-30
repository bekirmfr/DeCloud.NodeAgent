using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace DeCloud.NodeAgent.Infrastructure.Libvirt;

public class LibvirtVmManagerOptions
{
    public string VmStoragePath { get; set; } = "/var/lib/decloud/vms";
    public string ImageCachePath { get; set; } = "/var/lib/decloud/images";
    public string LibvirtUri { get; set; } = "qemu:///system";
    public int VncPortStart { get; set; } = 5900;
    public bool ReconcileOnStartup { get; set; } = true;
}

public class LibvirtVmManager : IVmManager
{
    private readonly ICommandExecutor _executor;
    private readonly IImageManager _imageManager;
    private readonly ILogger<LibvirtVmManager> _logger;
    private readonly LibvirtVmManagerOptions _options;
    private readonly VmRepository _repository;
    private readonly bool _isWindows;

    // Track our VMs in memory
    private readonly Dictionary<string, VmInstance> _vms = new();
    private readonly SemaphoreSlim _lock = new(1, 1);
    private int _nextVncPort;
    private bool _initialized = false;

    public LibvirtVmManager(
        ICommandExecutor executor,
        IImageManager imageManager,
        IOptions<LibvirtVmManagerOptions> options,
        VmRepository repository,
        ILogger<LibvirtVmManager> logger)
    {
        _executor = executor;
        _imageManager = imageManager;
        _logger = logger;
        _options = options.Value;
        _repository = repository;
        _nextVncPort = _options.VncPortStart;
        _isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
            System.Runtime.InteropServices.OSPlatform.Windows);

        if (!_isWindows)
        {
            Directory.CreateDirectory(_options.VmStoragePath);
            Directory.CreateDirectory(_options.ImageCachePath);

            _logger.LogInformation("✓ LibvirtVmManager initialized with persistent storage");
        }
        else
        {
            _logger.LogWarning(
                "⚠️  Running on Windows - VM management via libvirt/KVM is not available. " +
                "The API will run in simulation mode. Deploy to Linux for full VM functionality.");
        }
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        if (_initialized || _isWindows)
        {
            _initialized = true;
            return;
        }

        await _lock.WaitAsync(ct);
        try
        {
            _logger.LogInformation("Initializing LibvirtVmManager with state recovery...");

            // Ensure QEMU guest agent channel directory exists
            var channelDir = "/var/lib/libvirt/qemu/channel/target";
            if (Directory.Exists("/var/lib/libvirt") && !Directory.Exists(channelDir))
            {
                Directory.CreateDirectory(channelDir);
                _logger.LogDebug("Created QEMU guest agent channel directory");
            }

            // =====================================================
            // STEP 1: Load VMs from SQLite database
            // =====================================================
            var savedVms = await _repository.LoadAllVmsAsync();
            foreach (var vm in savedVms)
            {
                _vms[vm.VmId] = vm;
                _logger.LogDebug("Loaded VM {VmId} from database (State: {State})",
                    vm.VmId, vm.State);
            }

            if (savedVms.Any())
            {
                _logger.LogInformation("✓ Recovered {Count} VMs from local database", savedVms.Count);
            }

            // =====================================================
            // STEP 2: Reconcile with libvirt to get actual state
            // =====================================================
            if (_options.ReconcileOnStartup)
            {
                await ReconcileWithLibvirtAsync(ct);
            }

            // =====================================================
            // STEP 3: Update next VNC port
            // =====================================================
            await UpdateNextVncPortAsync(ct);

            _initialized = true;
            _logger.LogInformation("✓ LibvirtVmManager initialized with {Count} VMs", _vms.Count);
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task ReconcileWithLibvirtAsync(CancellationToken ct = default)
    {
        _logger.LogInformation("Reconciling VM state with libvirt...");

        try
        {
            var listResult = await _executor.ExecuteAsync("virsh", "list --all --uuid", ct);
            if (!listResult.Success)
            {
                _logger.LogWarning("Failed to list VMs from libvirt: {Error}", listResult.StandardError);
                return;
            }

            var libvirtVmIds = listResult.StandardOutput
                .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Select(line => line.Trim())
                .Where(uuid => !string.IsNullOrEmpty(uuid) && Guid.TryParse(uuid, out _))
                .ToList();

            _logger.LogInformation("Found {Count} VMs in libvirt", libvirtVmIds.Count);

            // Update state for VMs we know about
            foreach (var vmId in libvirtVmIds)
            {
                var actualState = await GetVmStateFromLibvirtAsync(vmId, ct);

                if (_vms.ContainsKey(vmId))
                {
                    var vm = _vms[vmId];
                    if (vm.State != actualState)
                    {
                        _logger.LogInformation("VM {VmId} state changed: {OldState} -> {NewState}",
                            vmId, vm.State, actualState);
                        vm.State = actualState;

                        // Persist state change
                        await _repository.UpdateVmStateAsync(vmId, actualState);
                        await _repository.SaveVmAsync(_vms[vmId]);
                    }
                }
                else
                {
                    // VM exists in libvirt but not in our tracking - attempt recovery
                    var vmInstance = await ReconstructVmInstanceAsync(vmId, ct);
                    if (vmInstance != null)
                    {
                        _vms[vmId] = vmInstance;

                        // NEW: Save recovered VM to database
                        await _repository.SaveVmAsync(vmInstance);

                        _logger.LogWarning("Recovered orphaned VM {VmId} from libvirt (state: {State})",
                            vmId, vmInstance.State);
                    }
                }
            }

            // Check for VMs in our tracking that no longer exist in libvirt
            var missingVms = _vms.Keys.Except(libvirtVmIds).ToList();
            foreach (var vmId in missingVms)
            {
                var vm = _vms[vmId];
                if (vm.State != VmState.Stopped && vm.State != VmState.Failed)
                {
                    _logger.LogWarning("VM {VmId} missing from libvirt - marking as failed", vmId);
                    vm.State = VmState.Failed;

                    // NEW: Update database
                    await _repository.UpdateVmStateAsync(vmId, VmState.Failed);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to reconcile with libvirt");
        }
    }

    private async Task<VmInstance?> ReconstructVmInstanceAsync(string vmId, CancellationToken ct)
    {
        try
        {
            var xmlResult = await _executor.ExecuteAsync("virsh", $"dumpxml {vmId}", ct);
            if (!xmlResult.Success)
            {
                _logger.LogWarning("Failed to get XML for VM {VmId}: {Error}", vmId, xmlResult.StandardError);
                return null;
            }

            var xml = XDocument.Parse(xmlResult.StandardOutput);
            var domain = xml.Element("domain");
            if (domain == null) return null;

            var vcpus = int.TryParse(domain.Element("vcpu")?.Value, out var v) ? v : 1;
            var memoryKiB = long.TryParse(domain.Element("memory")?.Value, out var m) ? m : 1048576;
            var memoryBytes = memoryKiB * 1024;

            var diskPath = domain.Descendants("disk")
                .Where(d => d.Attribute("device")?.Value == "disk")
                .SelectMany(d => d.Elements("source"))
                .Select(s => s.Attribute("file")?.Value)
                .FirstOrDefault();

            var graphics = domain.Descendants("graphics")
                .FirstOrDefault(g => g.Attribute("type")?.Value == "vnc");
            var vncPort = graphics?.Attribute("port")?.Value;

            var state = await GetVmStateFromLibvirtAsync(vmId, ct);

            var vmDir = Path.Combine(_options.VmStoragePath, vmId);
            string? tenantId = null;
            string? leaseId = null;
            string? vmName = null;

            var metadataPath = Path.Combine(vmDir, "metadata.json");
            if (File.Exists(metadataPath))
            {
                try
                {
                    var metadata = System.Text.Json.JsonDocument.Parse(
                        await File.ReadAllTextAsync(metadataPath, ct));
                    tenantId = metadata.RootElement.TryGetProperty("tenantId", out var t) ? t.GetString() : null;
                    leaseId = metadata.RootElement.TryGetProperty("leaseId", out var l) ? l.GetString() : null;
                    vmName = metadata.RootElement.TryGetProperty("name", out var n) ? n.GetString() : null;
                }
                catch { }
            }

            var instance = new VmInstance
            {
                VmId = vmId,
                Name = vmName ?? domain.Element("name")?.Value ?? vmId,
                Spec = new VmSpec
                {
                    VmId = vmId,
                    Name = vmName ?? domain.Element("name")?.Value ?? vmId,
                    VCpus = vcpus,
                    MemoryBytes = memoryBytes,
                    DiskBytes = await GetDiskSizeAsync(diskPath, ct),
                    TenantId = tenantId ?? "unknown",
                    LeaseId = leaseId ?? "unknown"
                },
                State = state,
                DiskPath = diskPath,
                ConfigPath = Path.Combine(vmDir, "domain.xml"),
                VncPort = vncPort,
                CreatedAt = Directory.Exists(vmDir)
                    ? Directory.GetCreationTimeUtc(vmDir)
                    : DateTime.UtcNow,
                StartedAt = state == VmState.Running ? DateTime.UtcNow : null
            };

            return instance;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to reconstruct VM {VmId}", vmId);
            return null;
        }
    }

    private async Task<long> GetDiskSizeAsync(string? diskPath, CancellationToken ct)
    {
        if (string.IsNullOrEmpty(diskPath) || !File.Exists(diskPath))
            return 0;

        try
        {
            var result = await _executor.ExecuteAsync("qemu-img", $"info --output=json {diskPath}", ct);
            if (result.Success)
            {
                var json = System.Text.Json.JsonDocument.Parse(result.StandardOutput);
                if (json.RootElement.TryGetProperty("virtual-size", out var size))
                {
                    return size.GetInt64();
                }
            }
        }
        catch { }

        return new FileInfo(diskPath).Length;
    }

    private async Task UpdateNextVncPortAsync(CancellationToken ct)
    {
        var maxPort = _options.VncPortStart;
        foreach (var vm in _vms.Values)
        {
            if (int.TryParse(vm.VncPort, out var port) && port >= maxPort)
            {
                maxPort = port + 1;
            }
        }
        _nextVncPort = maxPort;
        _logger.LogDebug("Next VNC port set to {Port}", _nextVncPort);
    }

    private async Task<VmState> GetVmStateFromLibvirtAsync(string vmId, CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync("virsh", $"domstate {vmId}", ct);
        if (!result.Success) return VmState.Stopped;

        return result.StandardOutput.Trim().ToLower() switch
        {
            "running" => VmState.Running,
            "paused" => VmState.Paused,
            "shut off" => VmState.Stopped,
            "crashed" => VmState.Failed,
            "in shutdown" => VmState.Stopping,
            _ => VmState.Stopped
        };
    }

    public async Task<VmOperationResult> CreateVmAsync(VmSpec spec, CancellationToken ct = default)
    {
        if (_isWindows)
        {
            _logger.LogWarning("VM creation not supported on Windows - requires Linux with KVM/libvirt");
            return VmOperationResult.Fail(spec.VmId,
                "VM creation requires Linux with KVM/libvirt. Windows detected.", "PLATFORM_UNSUPPORTED");
        }

        if (!_initialized)
        {
            await InitializeAsync(ct);
        }

        await _lock.WaitAsync(ct);
        try
        {
            if (_vms.ContainsKey(spec.VmId))
            {
                return VmOperationResult.Fail(spec.VmId, "VM already exists", "DUPLICATE");
            }

            // Create VM instance tracking object
            var instance = new VmInstance
            {
                VmId = spec.VmId,
                Name = spec.Name,
                Spec = spec,
                State = VmState.Creating,
                CreatedAt = DateTime.UtcNow,
                LastHeartbeat = DateTime.UtcNow,
                VncPort = _nextVncPort++.ToString()
            };

            _vms[spec.VmId] = instance;

            // Persist to database immediately
            await _repository.SaveVmAsync(instance);

            _logger.LogInformation("Creating VM {VmId}: {VCpus} vCPUs, {MemMB}MB RAM, {DiskGB}GB disk",
                spec.VmId, spec.VCpus, spec.MemoryBytes / 1024 / 1024, spec.DiskBytes / 1024 / 1024 / 1024);

            // Log authentication method
            if (!string.IsNullOrEmpty(spec.SshPublicKey))
            {
                _logger.LogInformation("VM {VmId} will use SSH key authentication ({KeyLength} chars)",
                    spec.VmId, spec.SshPublicKey.Length);
            }
            else if (!string.IsNullOrEmpty(spec.Password))
            {
                _logger.LogInformation("VM {VmId} will use password authentication", spec.VmId);
            }
            else
            {
                _logger.LogWarning("VM {VmId} has no SSH key or password - will use fallback password 'decloud'", spec.VmId);
            }

            var vmDir = Path.Combine(_options.VmStoragePath, spec.VmId);
            Directory.CreateDirectory(vmDir);

            // After successful creation, update state
            instance.State = VmState.Starting;
            await _repository.UpdateVmStateAsync(spec.VmId, VmState.Starting);

            // Start the VM
            var startResult = await _executor.ExecuteAsync("virsh", $"start {spec.VmId}", ct);
            if (startResult.Success)
            {
                instance.State = VmState.Running;
                instance.StartedAt = DateTime.UtcNow;

                // NEW: Update database
                await _repository.UpdateVmStateAsync(spec.VmId, VmState.Running);

                _logger.LogInformation("VM {VmId} started successfully", spec.VmId);

                // Background task to get IP address
                _ = Task.Run(async () =>
                {
                    await Task.Delay(TimeSpan.FromSeconds(10), ct);
                    var ip = await GetVmIpAddressAsync(spec.VmId, ct);
                    if (!string.IsNullOrEmpty(ip))
                    {
                        instance.Spec.Network.IpAddress = ip;
                        await _repository.UpdateVmIpAsync(spec.VmId, ip);
                        _logger.LogInformation("VM {VmId} obtained IP: {Ip}", spec.VmId, ip);
                    }
                }, ct);

                return VmOperationResult.Ok(spec.VmId, VmState.Running);
            }
            else
            {
                instance.State = VmState.Failed;
                await _repository.UpdateVmStateAsync(spec.VmId, VmState.Failed);

                _logger.LogError("Failed to start VM {VmId}: {Error}", spec.VmId, startResult.StandardError);
                return VmOperationResult.Fail(spec.VmId, startResult.StandardError, "START_FAILED");
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    private async Task SaveVmMetadataAsync(string vmDir, VmSpec spec, CancellationToken ct)
    {
        var metadata = new
        {
            vmId = spec.VmId,
            name = spec.Name,
            tenantId = spec.TenantId,
            leaseId = spec.LeaseId,
            createdAt = DateTime.UtcNow,
            vcpus = spec.VCpus,
            memoryBytes = spec.MemoryBytes,
            diskBytes = spec.DiskBytes
        };

        var metadataPath = Path.Combine(vmDir, "metadata.json");
        await File.WriteAllTextAsync(metadataPath,
            System.Text.Json.JsonSerializer.Serialize(metadata, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            }), ct);
    }

    public async Task<VmOperationResult> StartVmAsync(string vmId, CancellationToken ct = default)
    {
        if (!_initialized) await InitializeAsync(ct);

        if (!_vms.TryGetValue(vmId, out var instance))
        {
            // Try to recover from database
            instance = await _repository.LoadVmAsync(vmId);
            if (instance != null)
            {
                _vms[vmId] = instance;
                _logger.LogInformation("Recovered VM {VmId} from database", vmId);
            }
            else if (await VmExistsAsync(vmId, ct))
            {
                instance = await ReconstructVmInstanceAsync(vmId, ct);
                if (instance != null)
                {
                    _vms[vmId] = instance;
                    await _repository.SaveVmAsync(instance);
                }
            }

            if (instance == null)
            {
                return VmOperationResult.Fail(vmId, "VM not found", "NOT_FOUND");
            }
        }

        _logger.LogInformation("Starting VM {VmId}", vmId);

        var result = await _executor.ExecuteAsync("virsh", $"start {vmId}", ct);

        if (result.Success || result.StandardError.Contains("already active"))
        {
            instance.State = VmState.Running;
            instance.StartedAt = DateTime.UtcNow;

            // NEW: Persist state change
            await _repository.UpdateVmStateAsync(vmId, VmState.Running);

            return VmOperationResult.Ok(vmId, VmState.Running);
        }

        _logger.LogError("Failed to start VM {VmId}: {Error}", vmId, result.StandardError);
        return VmOperationResult.Fail(vmId, result.StandardError, "START_FAILED");
    }

    public async Task<VmOperationResult> StopVmAsync(string vmId, bool force = false, CancellationToken ct = default)
    {
        if (!_initialized) await InitializeAsync(ct);

        _logger.LogInformation("Stopping VM {VmId} (force={Force})", vmId, force);

        var existsInLibvirt = await VmExistsAsync(vmId, ct);
        if (!existsInLibvirt)
        {
            _vms.Remove(vmId);
            await _repository.DeleteVmAsync(vmId);  // ✅ ADD THIS
            _logger.LogInformation("Removed non-existent VM {VmId} from tracking and database", vmId);
            return VmOperationResult.Fail(vmId, "VM not found in libvirt", "NOT_FOUND");
        }

        if (_vms.TryGetValue(vmId, out var instance))
        {
            instance.State = VmState.Stopping;
            await _repository.UpdateVmStateAsync(vmId, VmState.Stopping);
        }

        var command = force ? "destroy" : "shutdown";
        var result = await _executor.ExecuteAsync("virsh", $"{command} {vmId}", ct);

        if (result.Success || result.StandardError.Contains("not running") || result.StandardError.Contains("shut off"))
        {
            if (instance != null)
            {
                instance.State = VmState.Stopped;
                instance.StoppedAt = DateTime.UtcNow;

                // NEW: Persist state change
                await _repository.UpdateVmStateAsync(vmId, VmState.Stopped);
            }

            return VmOperationResult.Ok(vmId, VmState.Stopped);
        }

        _logger.LogError("Failed to stop VM {VmId}: {Error}", vmId, result.StandardError);
        return VmOperationResult.Fail(vmId, result.StandardError, "STOP_FAILED");
    }

    public async Task<VmOperationResult> DeleteVmAsync(string vmId, CancellationToken ct = default)
    {
        if (!_initialized) await InitializeAsync(ct);

        _logger.LogInformation("Deleting VM {VmId}", vmId);

        // Stop if running
        var state = await GetVmStateFromLibvirtAsync(vmId, ct);
        if (state == VmState.Running || state == VmState.Paused)
        {
            await _executor.ExecuteAsync("virsh", $"destroy {vmId}", ct);
            await Task.Delay(500, ct);
        }

        // Undefine from libvirt
        var undefResult = await _executor.ExecuteAsync("virsh", $"undefine {vmId} --remove-all-storage --nvram", ct);

        if (!undefResult.Success && !undefResult.StandardError.Contains("not found"))
        {
            _logger.LogWarning("Failed to undefine VM {VmId}: {Error}", vmId, undefResult.StandardError);
        }

        // Remove from tracking
        _vms.Remove(vmId);

        // NEW: Remove from database
        await _repository.DeleteVmAsync(vmId);

        // Clean up directory
        var vmDirectory = Path.Combine(_options.VmStoragePath, vmId);
        if (Directory.Exists(vmDirectory))
        {
            try
            {
                Directory.Delete(vmDirectory, recursive: true);
                _logger.LogInformation("Deleted VM directory: {VmDir}", vmDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to delete VM directory {VmDir}", vmDirectory);
            }
        }

        return VmOperationResult.Ok(vmId, VmState.Stopped);
    }

    public async Task<VmOperationResult> PauseVmAsync(string vmId, CancellationToken ct = default)
    {
        if (!_initialized) await InitializeAsync(ct);

        if (!_vms.TryGetValue(vmId, out var instance))
        {
            return VmOperationResult.Fail(vmId, "VM not found", "NOT_FOUND");
        }

        var result = await _executor.ExecuteAsync("virsh", $"suspend {vmId}", ct);

        if (result.Success)
        {
            instance.State = VmState.Paused;
            await _repository.UpdateVmStateAsync(vmId, VmState.Paused);
            return VmOperationResult.Ok(vmId, VmState.Paused);
        }

        return VmOperationResult.Fail(vmId, result.StandardError, "PAUSE_FAILED");
    }

    public async Task<VmOperationResult> ResumeVmAsync(string vmId, CancellationToken ct = default)
    {
        if (!_initialized) await InitializeAsync(ct);

        if (!_vms.TryGetValue(vmId, out var instance))
        {
            return VmOperationResult.Fail(vmId, "VM not found", "NOT_FOUND");
        }

        var result = await _executor.ExecuteAsync("virsh", $"resume {vmId}", ct);

        if (result.Success)
        {
            instance.State = VmState.Running;
            await _repository.UpdateVmStateAsync(vmId, VmState.Running);
            return VmOperationResult.Ok(vmId, VmState.Running);
        }

        return VmOperationResult.Fail(vmId, result.StandardError, "RESUME_FAILED");
    }

    public Task<VmInstance?> GetVmAsync(string vmId, CancellationToken ct = default)
    {
        _vms.TryGetValue(vmId, out var vm);
        return Task.FromResult(vm);
    }

    public Task<List<VmInstance>> GetAllVmsAsync(CancellationToken ct = default)
    {
        return Task.FromResult(_vms.Values.ToList());
    }

    public async Task<VmResourceUsage> GetVmUsageAsync(string vmId, CancellationToken ct = default)
    {
        var usage = new VmResourceUsage { MeasuredAt = DateTime.UtcNow };

        if (!_vms.TryGetValue(vmId, out var vm) || vm.State != VmState.Running)
        {
            return usage;
        }

        var cpuResult = await _executor.ExecuteAsync("virsh", $"domstats {vmId} --cpu-total", ct);
        if (cpuResult.Success)
        {
            var match = Regex.Match(cpuResult.StandardOutput, @"cpu\.time=(\d+)");
            if (match.Success && long.TryParse(match.Groups[1].Value, out var cpuTime))
            {
                usage.CpuPercent = (cpuTime / 1_000_000_000.0) % 100;
            }
        }

        var memResult = await _executor.ExecuteAsync("virsh", $"dommemstat {vmId}", ct);
        if (memResult.Success)
        {
            foreach (var line in memResult.StandardOutput.Split('\n'))
            {
                if (line.StartsWith("actual"))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2 && long.TryParse(parts[1], out var actualKb))
                    {
                        usage.MemoryUsedBytes = actualKb * 1024;
                    }
                }
            }
        }

        var blockResult = await _executor.ExecuteAsync("virsh", $"domblkstat {vmId} vda", ct);
        if (blockResult.Success)
        {
            foreach (var line in blockResult.StandardOutput.Split('\n'))
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2)
                {
                    if (parts[0].EndsWith("rd_bytes") && long.TryParse(parts[1], out var rd))
                        usage.DiskReadBytes = rd;
                    if (parts[0].EndsWith("wr_bytes") && long.TryParse(parts[1], out var wr))
                        usage.DiskWriteBytes = wr;
                }
            }
        }

        foreach (var iface in new[] { "vnet0", "vnet1", "vnet2", "vnet3", "vnet4", "vnet5", "vnet6", "vnet7", "vnet8", "vnet9" })
        {
            var netResult = await _executor.ExecuteAsync("virsh", $"domifstat {vmId} {iface}", ct);
            if (netResult.Success && !netResult.StandardOutput.Contains("error"))
            {
                foreach (var line in netResult.StandardOutput.Split('\n'))
                {
                    if (line.Contains("rx_bytes"))
                    {
                        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 2 && long.TryParse(parts.Last(), out var rx))
                            usage.NetworkRxBytes = rx;
                    }
                    if (line.Contains("tx_bytes"))
                    {
                        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 2 && long.TryParse(parts.Last(), out var tx))
                            usage.NetworkTxBytes = tx;
                    }
                }
                break;
            }
        }

        return usage;
    }

    public async Task<bool> VmExistsAsync(string vmId, CancellationToken ct = default)
    {
        var result = await _executor.ExecuteAsync("virsh", $"dominfo {vmId}", ct);
        return result.Success;
    }

    public async Task<string?> GetVmIpAddressAsync(string vmId, CancellationToken ct = default)
    {
        var result = await _executor.ExecuteAsync("virsh", "net-dhcp-leases default", ct);
        if (!result.Success) return null;

        var domiflistResult = await _executor.ExecuteAsync("virsh", $"domiflist {vmId}", ct);
        if (!domiflistResult.Success) return null;

        var macMatch = Regex.Match(domiflistResult.StandardOutput, @"([0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2})",
            RegexOptions.IgnoreCase);
        if (!macMatch.Success) return null;

        var vmMac = macMatch.Value.ToLower();

        foreach (var line in result.StandardOutput.Split('\n'))
        {
            if (line.ToLower().Contains(vmMac))
            {
                var ipMatch = Regex.Match(line, @"(\d+\.\d+\.\d+\.\d+)");
                if (ipMatch.Success)
                {
                    return ipMatch.Value;
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Create cloud-init ISO with proper network config and authentication.
    /// Includes machine-id regeneration to prevent DHCP IP collisions.
    /// </summary>
    private async Task<string> CreateCloudInitIsoAsync(VmSpec spec, string vmDir, CancellationToken ct)
    {
        var hasPassword = !string.IsNullOrEmpty(spec.Password);
        var hasSshKey = !string.IsNullOrEmpty(spec.SshPublicKey);

        var sb = new StringBuilder();
        sb.AppendLine("#cloud-config");
        sb.AppendLine($"hostname: {spec.Name}");
        sb.AppendLine("manage_etc_hosts: true");
        sb.AppendLine();

        // CRITICAL: Regenerate machine-id BEFORE networking starts
        // This prevents DHCP collisions when VMs are cloned from the same base image
        sb.AppendLine("bootcmd:");
        sb.AppendLine("  - rm -f /etc/machine-id /var/lib/dbus/machine-id");
        sb.AppendLine("  - systemd-machine-id-setup");
        sb.AppendLine();

        // User configuration
        sb.AppendLine("users:");
        sb.AppendLine("  - name: ubuntu");
        sb.AppendLine("    sudo: ALL=(ALL) NOPASSWD:ALL");
        sb.AppendLine("    shell: /bin/bash");
        sb.AppendLine(hasPassword ? "    lock_passwd: false" : "    lock_passwd: true");

        // SSH key(s) - support multiple keys (newline separated)
        if (hasSshKey)
        {
            sb.AppendLine("    ssh_authorized_keys:");
            foreach (var key in spec.SshPublicKey!.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                var trimmedKey = key.Trim();
                if (!string.IsNullOrEmpty(trimmedKey))
                {
                    sb.AppendLine($"      - {trimmedKey}");
                }
            }
        }

        // Password configuration
        if (hasPassword)
        {
            sb.AppendLine();
            sb.AppendLine("chpasswd:");
            sb.AppendLine("  list: |");
            sb.AppendLine($"    ubuntu:{spec.Password}");
            sb.AppendLine("  expire: false");
            sb.AppendLine();
            sb.AppendLine("ssh_pwauth: true");
        }

        sb.AppendLine();
        sb.AppendLine("packages:");
        sb.AppendLine("  - qemu-guest-agent");
        sb.AppendLine();

        // Runtime commands
        sb.AppendLine("runcmd:");
        sb.AppendLine("  - systemctl enable qemu-guest-agent");
        sb.AppendLine("  - systemctl start qemu-guest-agent");

        // Ensure SSH allows password auth (some images disable it)
        if (hasPassword)
        {
            sb.AppendLine("  - sed -i 's/^PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config");
            sb.AppendLine("  - sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config");
            sb.AppendLine("  - sed -i 's/^#PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config");
            sb.AppendLine("  - systemctl restart sshd || systemctl restart ssh");
        }

        var userData = sb.ToString();

        _logger.LogDebug("Cloud-init user-data for VM {VmId}:\n{UserData}", spec.VmId, userData);

        var userDataPath = Path.Combine(vmDir, "user-data");
        var metaDataPath = Path.Combine(vmDir, "meta-data");

        await File.WriteAllTextAsync(userDataPath, userData, ct);

        var metaData = $"instance-id: {spec.VmId}\nlocal-hostname: {spec.Name}\n";
        await File.WriteAllTextAsync(metaDataPath, metaData, ct);

        var isoPath = Path.Combine(vmDir, "cloud-init.iso");
        var result = await _executor.ExecuteAsync(
            "genisoimage",
            $"-output {isoPath} -volid cidata -joliet -rock {userDataPath} {metaDataPath}",
            ct);

        if (!result.Success)
        {
            result = await _executor.ExecuteAsync(
                "cloud-localds",
                $"{isoPath} {userDataPath} {metaDataPath}",
                ct);
        }

        if (!result.Success)
        {
            _logger.LogWarning("Failed to create cloud-init ISO: {Error}", result.StandardError);
            return string.Empty;
        }

        _logger.LogInformation("Created cloud-init ISO at {Path} (password auth: {HasPassword}, ssh key: {HasSshKey})",
            isoPath, hasPassword, hasSshKey);

        return isoPath;
    }

    private string GenerateLibvirtXml(VmSpec spec, string diskPath, string? cloudInitIso, int vncPort)
    {
        var cloudInitDisk = string.IsNullOrEmpty(cloudInitIso) ? "" : $@"
            <disk type='file' device='cdrom'>
              <driver name='qemu' type='raw'/>
              <source file='{cloudInitIso}'/>
              <target dev='sda' bus='sata'/>
              <readonly/>
            </disk>";

        return $@"<domain type='kvm'>
              <name>{spec.VmId}</name>
              <uuid>{spec.VmId}</uuid>
              <memory unit='bytes'>{spec.MemoryBytes}</memory>
              <vcpu placement='static'>{spec.VCpus}</vcpu>
              <os>
                <type arch='x86_64' machine='q35'>hvm</type>
                <boot dev='hd'/>
              </os>
              <features>
                <acpi/>
                <apic/>
              </features>
              <cpu mode='host-passthrough' check='none'/>
              <clock offset='utc'>
                <timer name='rtc' tickpolicy='catchup'/>
                <timer name='pit' tickpolicy='delay'/>
                <timer name='hpet' present='no'/>
              </clock>
              <on_poweroff>destroy</on_poweroff>
              <on_reboot>restart</on_reboot>
              <on_crash>destroy</on_crash>
              <devices>
                <emulator>/usr/bin/qemu-system-x86_64</emulator>
                <disk type='file' device='disk'>
                  <driver name='qemu' type='qcow2'/>
                  <source file='{diskPath}'/>
                  <target dev='vda' bus='virtio'/>
                </disk>{cloudInitDisk}
                <interface type='network'>
                  <source network='default'/>
                  <model type='virtio'/>
                </interface>
                <serial type='pty'>
                  <target port='0'/>
                </serial>
                <console type='pty'>
                  <target type='serial' port='0'/>
                </console>
                <graphics type='vnc' port='{vncPort}' autoport='no' listen='0.0.0.0'>
                  <listen type='address' address='0.0.0.0'/>
                </graphics>
                <video>
                  <model type='qxl' ram='65536' vram='65536' vgamem='16384' heads='1'/>
                </video>
                <rng model='virtio'>
                  <backend model='random'>/dev/urandom</backend>
                </rng>
                <channel type='unix'>
                  <source mode='bind' path='/var/lib/libvirt/qemu/channel/target/{spec.VmId}.org.qemu.guest_agent.0'/>
                  <target type='virtio' name='org.qemu.guest_agent.0'/>
                </channel>
              </devices>
            </domain>";
    }
}