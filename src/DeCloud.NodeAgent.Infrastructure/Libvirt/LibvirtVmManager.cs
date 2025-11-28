using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
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
        ILogger<LibvirtVmManager> logger)
    {
        _executor = executor;
        _imageManager = imageManager;
        _logger = logger;
        _options = options.Value;
        _nextVncPort = _options.VncPortStart;
        _isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
            System.Runtime.InteropServices.OSPlatform.Windows);

        if (!_isWindows)
        {
            Directory.CreateDirectory(_options.VmStoragePath);
            Directory.CreateDirectory(_options.ImageCachePath);
        }
        else
        {
            _logger.LogWarning("Running on Windows - VM management via libvirt/KVM is not available. " +
                "The API will run in simulation mode. Deploy to Linux for full VM functionality.");
        }
    }

    /// <summary>
    /// Initialize the VM manager and reconcile with libvirt state.
    /// Should be called on startup before processing any commands.
    /// </summary>
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
            if (_options.ReconcileOnStartup)
            {
                await ReconcileWithLibvirtAsync(ct);
            }
            _initialized = true;
            _logger.LogInformation("LibvirtVmManager initialized with {Count} VMs", _vms.Count);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Reconcile in-memory VM tracking with actual libvirt state.
    /// This discovers VMs that exist in libvirt but aren't in our tracking.
    /// </summary>
    public async Task ReconcileWithLibvirtAsync(CancellationToken ct = default)
    {
        _logger.LogInformation("Reconciling VM state with libvirt...");

        try
        {
            // Get all VMs from libvirt
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

            foreach (var vmId in libvirtVmIds)
            {
                if (_vms.ContainsKey(vmId))
                {
                    // Already tracking - just update state
                    var state = await GetVmStateFromLibvirtAsync(vmId, ct);
                    _vms[vmId].State = state;
                    _logger.LogDebug("Updated existing VM {VmId} state to {State}", vmId, state);
                }
                else
                {
                    // Not tracking - reconstruct from libvirt
                    var vmInstance = await ReconstructVmInstanceAsync(vmId, ct);
                    if (vmInstance != null)
                    {
                        _vms[vmId] = vmInstance;
                        _logger.LogInformation("Recovered VM {VmId} from libvirt (state: {State})",
                            vmId, vmInstance.State);
                    }
                }
            }

            // Check for VMs we're tracking that no longer exist in libvirt
            var orphanedTracking = _vms.Keys.Except(libvirtVmIds).ToList();
            foreach (var vmId in orphanedTracking)
            {
                _logger.LogWarning("Removing stale tracking for VM {VmId} (no longer in libvirt)", vmId);
                _vms.Remove(vmId);
            }

            // Update next VNC port based on what's in use
            await UpdateNextVncPortAsync(ct);

            _logger.LogInformation("Reconciliation complete: tracking {Count} VMs", _vms.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during libvirt reconciliation");
        }
    }

    /// <summary>
    /// Reconstruct a VmInstance from libvirt domain XML
    /// </summary>
    private async Task<VmInstance?> ReconstructVmInstanceAsync(string vmId, CancellationToken ct)
    {
        try
        {
            // Get domain XML
            var xmlResult = await _executor.ExecuteAsync("virsh", $"dumpxml {vmId}", ct);
            if (!xmlResult.Success)
            {
                _logger.LogWarning("Failed to get XML for VM {VmId}: {Error}", vmId, xmlResult.StandardError);
                return null;
            }

            var xml = XDocument.Parse(xmlResult.StandardOutput);
            var domain = xml.Element("domain");
            if (domain == null) return null;

            // Parse resources from XML
            var vcpus = int.TryParse(domain.Element("vcpu")?.Value, out var v) ? v : 1;
            var memoryKiB = long.TryParse(domain.Element("memory")?.Value, out var m) ? m : 1048576;
            var memoryBytes = memoryKiB * 1024;

            // Find disk path
            var diskPath = domain.Descendants("disk")
                .Where(d => d.Attribute("device")?.Value == "disk")
                .SelectMany(d => d.Elements("source"))
                .Select(s => s.Attribute("file")?.Value)
                .FirstOrDefault();

            // Find VNC port
            var graphics = domain.Descendants("graphics")
                .FirstOrDefault(g => g.Attribute("type")?.Value == "vnc");
            var vncPort = graphics?.Attribute("port")?.Value;

            // Get current state
            var state = await GetVmStateFromLibvirtAsync(vmId, ct);

            // Check for cloud-init metadata in VM directory
            var vmDir = Path.Combine(_options.VmStoragePath, vmId);
            string? tenantId = null;
            string? leaseId = null;

            var metadataPath = Path.Combine(vmDir, "metadata.json");
            if (File.Exists(metadataPath))
            {
                try
                {
                    var metadata = System.Text.Json.JsonDocument.Parse(
                        await File.ReadAllTextAsync(metadataPath, ct));
                    tenantId = metadata.RootElement.TryGetProperty("tenantId", out var t) ? t.GetString() : null;
                    leaseId = metadata.RootElement.TryGetProperty("leaseId", out var l) ? l.GetString() : null;
                }
                catch { /* ignore metadata parse errors */ }
            }

            var instance = new VmInstance
            {
                VmId = vmId,
                Spec = new VmSpec
                {
                    VmId = vmId,
                    Name = domain.Element("name")?.Value ?? vmId,
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
            return 10L * 1024 * 1024 * 1024; // Default 10GB

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
        catch { /* ignore */ }

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
            "pmsuspended" => VmState.Paused,
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

        // Ensure initialized
        if (!_initialized)
        {
            await InitializeAsync(ct);
        }

        await _lock.WaitAsync(ct);
        try
        {
            _logger.LogInformation("Creating VM {VmId}: {VCpus} vCPUs, {MemMB}MB RAM, {DiskGB}GB disk",
                spec.VmId, spec.VCpus, spec.MemoryBytes / 1024 / 1024, spec.DiskBytes / 1024 / 1024 / 1024);

            // 1. Create VM directory
            var vmDir = Path.Combine(_options.VmStoragePath, spec.VmId);
            Directory.CreateDirectory(vmDir);

            var instance = new VmInstance
            {
                VmId = spec.VmId,
                Spec = spec,
                State = VmState.Creating,
                CreatedAt = DateTime.UtcNow
            };
            _vms[spec.VmId] = instance;

            // Save metadata for recovery
            await SaveVmMetadataAsync(vmDir, spec, ct);

            // 2. Download/prepare base image using IImageManager
            _logger.LogInformation("Preparing base image from {ImageUrl}", spec.BaseImageUrl);

            string baseImagePath;
            try
            {
                // Use EnsureImageAvailableAsync - pass empty hash to skip verification
                baseImagePath = await _imageManager.EnsureImageAvailableAsync(
                    spec.BaseImageUrl ?? "",
                    string.Empty, // No hash verification for now
                    ct);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to download base image");
                instance.State = VmState.Failed;
                return VmOperationResult.Fail(spec.VmId, $"Failed to download base image: {ex.Message}", "IMAGE_DOWNLOAD_FAILED");
            }

            if (string.IsNullOrEmpty(baseImagePath))
            {
                instance.State = VmState.Failed;
                return VmOperationResult.Fail(spec.VmId, "Failed to download base image", "IMAGE_DOWNLOAD_FAILED");
            }

            // 3. Create disk overlay using IImageManager
            string diskPath;
            try
            {
                diskPath = await _imageManager.CreateOverlayDiskAsync(baseImagePath, spec.VmId, spec.DiskBytes, ct);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to create disk overlay");
                instance.State = VmState.Failed;
                return VmOperationResult.Fail(spec.VmId, ex.Message, "DISK_CREATE_FAILED");
            }

            instance.DiskPath = diskPath;

            // 4. Always generate cloud-init ISO (for network config at minimum)
            var cloudInitIso = await CreateCloudInitIsoAsync(spec, vmDir, ct);

            // 5. Generate libvirt XML
            var vncPort = Interlocked.Increment(ref _nextVncPort);
            instance.VncPort = vncPort.ToString();

            var xml = GenerateLibvirtXml(spec, instance.DiskPath, cloudInitIso, vncPort);
            instance.ConfigPath = Path.Combine(vmDir, "domain.xml");
            await File.WriteAllTextAsync(instance.ConfigPath, xml, ct);

            // 6. Define domain with virsh
            var defineResult = await _executor.ExecuteAsync("virsh",
                $"define {instance.ConfigPath}", ct);

            if (!defineResult.Success)
            {
                _logger.LogError("Failed to define VM: {Error}", defineResult.StandardError);
                instance.State = VmState.Failed;
                return VmOperationResult.Fail(spec.VmId, defineResult.StandardError, "DEFINE_FAILED");
            }

            instance.State = VmState.Stopped;
            _logger.LogInformation("VM {VmId} created successfully", spec.VmId);

            return VmOperationResult.Ok(spec.VmId, VmState.Stopped);
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
        // Ensure initialized
        if (!_initialized) await InitializeAsync(ct);

        // Check libvirt directly if not in tracking
        if (!_vms.TryGetValue(vmId, out var instance))
        {
            if (!await VmExistsAsync(vmId, ct))
            {
                return VmOperationResult.Fail(vmId, "VM not found", "NOT_FOUND");
            }

            // Reconcile this specific VM
            instance = await ReconstructVmInstanceAsync(vmId, ct);
            if (instance != null)
            {
                _vms[vmId] = instance;
            }
            else
            {
                return VmOperationResult.Fail(vmId, "VM not found in tracking", "NOT_FOUND");
            }
        }

        _logger.LogInformation("Starting VM {VmId}", vmId);

        var result = await _executor.ExecuteAsync("virsh", $"start {vmId}", ct);

        if (result.Success)
        {
            instance.State = VmState.Running;
            instance.StartedAt = DateTime.UtcNow;
            return VmOperationResult.Ok(vmId, VmState.Running);
        }

        // Check if already running
        if (result.StandardError.Contains("already active"))
        {
            instance.State = VmState.Running;
            instance.StartedAt ??= DateTime.UtcNow;
            return VmOperationResult.Ok(vmId, VmState.Running);
        }

        _logger.LogError("Failed to start VM {VmId}: {Error}", vmId, result.StandardError);
        return VmOperationResult.Fail(vmId, result.StandardError, "START_FAILED");
    }

    public async Task<VmOperationResult> StopVmAsync(string vmId, bool force = false, CancellationToken ct = default)
    {
        // Ensure initialized
        if (!_initialized) await InitializeAsync(ct);

        _logger.LogInformation("Stopping VM {VmId} (force={Force})", vmId, force);

        // Check libvirt directly - don't require in-memory tracking
        var existsInLibvirt = await VmExistsAsync(vmId, ct);
        if (!existsInLibvirt)
        {
            _vms.Remove(vmId);
            return VmOperationResult.Fail(vmId, "VM not found in libvirt", "NOT_FOUND");
        }

        // Update tracking if we have it
        if (_vms.TryGetValue(vmId, out var instance))
        {
            instance.State = VmState.Stopping;
        }

        var command = force ? "destroy" : "shutdown";
        var result = await _executor.ExecuteAsync("virsh", $"{command} {vmId}", ct);

        if (result.Success)
        {
            // Wait for actual shutdown if graceful
            if (!force)
            {
                for (var i = 0; i < 30; i++)
                {
                    await Task.Delay(1000, ct);
                    var state = await GetVmStateFromLibvirtAsync(vmId, ct);
                    if (state == VmState.Stopped) break;
                }
            }

            if (instance != null)
            {
                instance.State = VmState.Stopped;
                instance.StoppedAt = DateTime.UtcNow;
            }
            return VmOperationResult.Ok(vmId, VmState.Stopped);
        }

        // Check if already stopped
        if (result.StandardError.Contains("not running") || result.StandardError.Contains("shut off"))
        {
            if (instance != null)
            {
                instance.State = VmState.Stopped;
                instance.StoppedAt ??= DateTime.UtcNow;
            }
            return VmOperationResult.Ok(vmId, VmState.Stopped);
        }

        _logger.LogError("Failed to stop VM {VmId}: {Error}", vmId, result.StandardError);
        return VmOperationResult.Fail(vmId, result.StandardError, "STOP_FAILED");
    }

    public async Task<VmOperationResult> DeleteVmAsync(string vmId, CancellationToken ct = default)
    {
        // Ensure initialized
        if (!_initialized) await InitializeAsync(ct);

        _logger.LogInformation("Deleting VM {VmId}", vmId);

        // Check if VM exists in libvirt (not just our in-memory tracking)
        var existsInLibvirt = await VmExistsAsync(vmId, ct);

        if (!existsInLibvirt)
        {
            // Not in libvirt - just clean up our tracking if present
            _vms.Remove(vmId);

            // Also clean up directory if it exists
            var vmDir = Path.Combine(_options.VmStoragePath, vmId);
            if (Directory.Exists(vmDir))
            {
                try
                {
                    Directory.Delete(vmDir, recursive: true);
                    _logger.LogInformation("Cleaned up orphaned VM directory: {VmDir}", vmDir);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to delete VM directory {VmDir}", vmDir);
                }
            }

            _logger.LogInformation("VM {VmId} not found in libvirt, cleaned up tracking", vmId);
            return VmOperationResult.Ok(vmId, VmState.Stopped);
        }

        // Stop if running - use virsh directly instead of StopVmAsync to ensure it works
        // even if VM isn't in our tracking dictionary
        var state = await GetVmStateFromLibvirtAsync(vmId, ct);
        if (state == VmState.Running || state == VmState.Paused)
        {
            _logger.LogInformation("VM {VmId} is {State}, forcing shutdown", vmId, state);
            var destroyResult = await _executor.ExecuteAsync("virsh", $"destroy {vmId}", ct);
            if (!destroyResult.Success && !destroyResult.StandardError.Contains("not running"))
            {
                _logger.LogWarning("Failed to destroy VM {VmId}: {Error}", vmId, destroyResult.StandardError);
                // Continue anyway - undefine might still work
            }

            // Wait a moment for the destroy to complete
            await Task.Delay(500, ct);
        }

        // Undefine domain with --remove-all-storage to clean up disks
        var undefResult = await _executor.ExecuteAsync("virsh",
            $"undefine {vmId} --remove-all-storage --nvram", ct);

        if (!undefResult.Success)
        {
            // Try without --nvram (some VMs don't have NVRAM)
            undefResult = await _executor.ExecuteAsync("virsh",
                $"undefine {vmId} --remove-all-storage", ct);
        }

        if (!undefResult.Success && !undefResult.StandardError.Contains("not found"))
        {
            _logger.LogError("Failed to undefine VM {VmId}: {Error}", vmId, undefResult.StandardError);
            return VmOperationResult.Fail(vmId, undefResult.StandardError, "UNDEFINE_FAILED");
        }

        // Clean up our VM directory (cloud-init, metadata, etc.)
        var vmDirectory = Path.Combine(_options.VmStoragePath, vmId);
        if (Directory.Exists(vmDirectory))
        {
            try
            {
                Directory.Delete(vmDirectory, recursive: true);
                _logger.LogInformation("Cleaned up VM directory: {VmDir}", vmDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to delete VM directory {VmDir}", vmDirectory);
            }
        }

        // Also clean up disk via IImageManager if it still exists
        if (_vms.TryGetValue(vmId, out var vmInstance) && !string.IsNullOrEmpty(vmInstance.DiskPath))
        {
            try
            {
                await _imageManager.DeleteDiskAsync(vmInstance.DiskPath, ct);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to delete disk via ImageManager");
            }
        }

        // Remove from tracking
        _vms.Remove(vmId);
        _logger.LogInformation("VM {VmId} deleted successfully", vmId);

        return VmOperationResult.Ok(vmId, VmState.Stopped);
    }

    public async Task<VmOperationResult> PauseVmAsync(string vmId, CancellationToken ct = default)
    {
        var result = await _executor.ExecuteAsync("virsh", $"suspend {vmId}", ct);

        if (result.Success && _vms.TryGetValue(vmId, out var instance))
        {
            instance.State = VmState.Paused;
            return VmOperationResult.Ok(vmId, VmState.Paused);
        }

        return VmOperationResult.Fail(vmId, result.StandardError, "PAUSE_FAILED");
    }

    public async Task<VmOperationResult> ResumeVmAsync(string vmId, CancellationToken ct = default)
    {
        var result = await _executor.ExecuteAsync("virsh", $"resume {vmId}", ct);

        if (result.Success && _vms.TryGetValue(vmId, out var instance))
        {
            instance.State = VmState.Running;
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

        // Get CPU stats - calculate percentage from cpu_time
        var cpuResult = await _executor.ExecuteAsync("virsh", $"cpu-stats {vmId} --total", ct);
        if (cpuResult.Success)
        {
            // Try to get CPU percentage from domstats instead
            var statsResult = await _executor.ExecuteAsync("virsh", $"domstats {vmId} --cpu-total", ct);
            if (statsResult.Success)
            {
                // Parse cpu.time and calculate approximate percentage
                var match = Regex.Match(statsResult.StandardOutput, @"cpu\.time=(\d+)");
                if (match.Success && long.TryParse(match.Groups[1].Value, out var cpuTimeNs))
                {
                    // Get VM's vCPU count to estimate usage
                    if (_vms.TryGetValue(vmId, out var vm))
                    {
                        // This is a rough estimate - for accurate %, would need to track delta over time
                        // For now, just report a placeholder based on whether VM is active
                        usage.CpuPercent = cpuTimeNs > 0 ? 5.0 : 0.0; // Placeholder
                    }
                }
            }
        }

        // Get memory stats
        var memResult = await _executor.ExecuteAsync("virsh", $"dommemstat {vmId}", ct);
        if (memResult.Success)
        {
            foreach (var line in memResult.StandardOutput.Split('\n'))
            {
                if (line.StartsWith("rss"))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2 && long.TryParse(parts[1], out var kib))
                    {
                        usage.MemoryUsedBytes = kib * 1024;
                    }
                }
            }
        }

        // Get block stats
        var blkResult = await _executor.ExecuteAsync("virsh", $"domblkstat {vmId} vda", ct);
        if (blkResult.Success)
        {
            foreach (var line in blkResult.StandardOutput.Split('\n'))
            {
                if (line.Contains("rd_bytes"))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2 && long.TryParse(parts.Last(), out var rb))
                        usage.DiskReadBytes = rb;
                }
                if (line.Contains("wr_bytes"))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2 && long.TryParse(parts.Last(), out var wb))
                        usage.DiskWriteBytes = wb;
                }
            }
        }

        // Get network stats - try different interface names
        foreach (var ifName in new[] { "vnet0", "virbr0-nic", "default" })
        {
            var netResult = await _executor.ExecuteAsync("virsh", $"domifstat {vmId} {ifName}", ct);
            if (netResult.Success)
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
                break; // Found working interface
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
        // Try to get IP from DHCP leases
        var result = await _executor.ExecuteAsync("virsh", "net-dhcp-leases default", ct);
        if (!result.Success) return null;

        // Parse output looking for our VM's MAC address
        // First get the VM's MAC
        var domiflistResult = await _executor.ExecuteAsync("virsh", $"domiflist {vmId}", ct);
        if (!domiflistResult.Success) return null;

        var macMatch = Regex.Match(domiflistResult.StandardOutput, @"([0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2})",
            RegexOptions.IgnoreCase);
        if (!macMatch.Success) return null;

        var vmMac = macMatch.Value.ToLower();

        // Find IP for this MAC in DHCP leases
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

    private async Task<string> CreateCloudInitIsoAsync(VmSpec spec, string vmDir, CancellationToken ct)
    {
        var metaDataPath = Path.Combine(vmDir, "meta-data");
        var userDataPath = Path.Combine(vmDir, "user-data");
        var networkConfigPath = Path.Combine(vmDir, "network-config");
        var isoPath = Path.Combine(vmDir, "cloud-init.iso");

        // meta-data (instance identity)
        var metaData = $@"instance-id: {spec.VmId}
local-hostname: {spec.Name}
";
        await File.WriteAllTextAsync(metaDataPath, metaData, ct);

        // network-config (always enable DHCP)
        var networkConfig = @"version: 2
ethernets:
  id0:
    match:
      driver: virtio
    dhcp4: true
    dhcp6: false
";
        await File.WriteAllTextAsync(networkConfigPath, networkConfig, ct);

        // user-data (SSH key OR password)
        var userDataBuilder = new StringBuilder();
        userDataBuilder.AppendLine("#cloud-config");

        // Set hostname
        userDataBuilder.AppendLine($"hostname: {spec.Name}");
        userDataBuilder.AppendLine("manage_etc_hosts: true");
        userDataBuilder.AppendLine();

        // Default user configuration
        userDataBuilder.AppendLine("users:");
        userDataBuilder.AppendLine("  - name: ubuntu");
        userDataBuilder.AppendLine("    sudo: ALL=(ALL) NOPASSWD:ALL");
        userDataBuilder.AppendLine("    shell: /bin/bash");
        userDataBuilder.AppendLine("    groups: [adm, sudo, docker]");

        if (!string.IsNullOrEmpty(spec.SshPublicKey))
        {
            // SSH key authentication
            userDataBuilder.AppendLine("    lock_passwd: true");
            userDataBuilder.AppendLine("    ssh_authorized_keys:");
            foreach (var key in spec.SshPublicKey.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                userDataBuilder.AppendLine($"      - {key.Trim()}");
            }
        }
        else if (!string.IsNullOrEmpty(spec.Password))
        {
            // Password authentication
            userDataBuilder.AppendLine("    lock_passwd: false");
            userDataBuilder.AppendLine($"    plain_text_passwd: {spec.Password}");
            userDataBuilder.AppendLine();
            userDataBuilder.AppendLine("ssh_pwauth: true");  // Enable SSH password auth
        }
        else
        {
            // Fallback: no auth configured (shouldn't happen, but be safe)
            _logger.LogWarning("VM {VmId} created without SSH key or password", spec.VmId);
            userDataBuilder.AppendLine("    lock_passwd: true");
        }

        userDataBuilder.AppendLine();

        // Add any custom user data
        if (!string.IsNullOrEmpty(spec.CloudInitUserData))
        {
            userDataBuilder.AppendLine("# Custom user data");
            userDataBuilder.AppendLine(spec.CloudInitUserData);
        }

        // Package updates (optional, can slow down boot)
        userDataBuilder.AppendLine();
        userDataBuilder.AppendLine("package_update: false");
        userDataBuilder.AppendLine("package_upgrade: false");

        // Final message
        userDataBuilder.AppendLine();
        userDataBuilder.AppendLine("final_message: \"DeCloud VM ready after $UPTIME seconds\"");

        await File.WriteAllTextAsync(userDataPath, userDataBuilder.ToString(), ct);

        // Generate ISO using cloud-localds or genisoimage
        var result = await _executor.ExecuteAsync(
            "cloud-localds",
            $"-N {networkConfigPath} {isoPath} {userDataPath} {metaDataPath}",
            ct);

        if (!result.Success)
        {
            // Fallback to genisoimage
            result = await _executor.ExecuteAsync(
                "genisoimage",
                $"-output {isoPath} -volid cidata -joliet -rock {userDataPath} {metaDataPath} {networkConfigPath}",
                ct);
        }

        if (!result.Success)
        {
            throw new InvalidOperationException($"Failed to create cloud-init ISO: {result.StandardError}");
        }

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
  </devices>
</domain>";
    }
}