using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Interfaces.UserNetwork;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.NodeAgent.Infrastructure.Services;
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
    public string Uri { get; set; } = "qemu:///system";
    public int VncPortStart { get; set; } = 5900;
    public bool ReconcileOnStartup { get; set; } = true;
}

public class LibvirtVmManager : IVmManager
{
    private readonly INodeMetadataService _nodeMetadata;
    private readonly INodeStateService _nodeState;
    private readonly ICommandExecutor _executor;
    private readonly IImageManager _imageManager;
    private readonly ICloudInitTemplateService _templateService;
    private readonly ILogger<LibvirtVmManager> _logger;
    private readonly LibvirtVmManagerOptions _options;
    private readonly VmRepository _repository;
    private readonly IUserWireGuardManager _userWireGuardManager;
    private readonly bool _isWindows;

    // Track our VMs in memory
    private readonly Dictionary<string, VmInstance> _vms = new();
    private readonly SemaphoreSlim _lock = new(1, 1);
    private int _nextVncPort;
    private bool _initialized = false;
    private string _hostArchitecture = "x86_64"; // Default, will be detected

    public LibvirtVmManager(
        ICommandExecutor executor,
        IImageManager imageManager,
        IOptions<LibvirtVmManagerOptions> options,
        VmRepository repository,
        ICloudInitTemplateService templateService,
        IUserWireGuardManager userWireGuardManager,
        INodeMetadataService nodeMetadata,
        INodeStateService nodeState,
        ILogger<LibvirtVmManager> logger)
    {
        _executor = executor;
        _imageManager = imageManager;
        _templateService = templateService;
        _nodeMetadata = nodeMetadata;
        _nodeState = nodeState;
        _userWireGuardManager = userWireGuardManager;
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
                await ReconcileAllWithLibvirtAsync(ct);
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

    public async Task ReconcileAllWithLibvirtAsync(CancellationToken ct = default)
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
                var isDirty = false;

                var actualState = await GetVmStateFromLibvirtAsync(vmId, ct);

                if (_vms.ContainsKey(vmId))
                {
                    var vm = _vms[vmId];
                    if (vm.State != actualState)
                    {
                        _logger.LogInformation("VM {VmId} state changed: {OldState} -> {NewState}",
                            vmId, vm.State, actualState);
                        vm.State = actualState;

                        isDirty = true;
                    }

                    // Resolve IP for running VMs that are missing one
                    if (actualState == VmState.Running && string.IsNullOrEmpty(vm.Spec.IpAddress))
                    {
                        var ip = await GetVmIpAddressAsync(vmId, ct);
                        if (!string.IsNullOrEmpty(ip))
                        {
                            vm.Spec.IpAddress = ip;
                            await _repository.UpdateVmIpAsync(vmId, ip);
                            _logger.LogInformation("VM {VmId} IP resolved during reconciliation: {Ip}", vmId, ip);
                            isDirty = true;
                        }
                    }
                }
                else
                {
                    // VM exists in libvirt but not in our tracking - attempt recovery
                    var vmInstance = await ReconstructVmInstanceAsync(vmId, ct);
                    if (vmInstance != null)
                    {
                        _vms[vmId] = vmInstance;

                        _logger.LogWarning("Recovered orphaned VM {VmId} from libvirt (state: {State})",
                            vmId, vmInstance.State);

                        isDirty = true;
                    }
                }

                // Save recovered VM to database
                if (isDirty)
                    await _repository.SaveVmAsync(_vms[vmId]);
            }

            // Check for VMs in our tracking that no longer exist in libvirt
            var missingVms = _vms.Keys.Except(libvirtVmIds).ToList();
            foreach (var vmId in missingVms)
            {
                var vm = _vms[vmId];
                // Skip VMs that are being created - they legitimately aren't in libvirt yet
                // Also skip already Stopped/Failed VMs
                if (vm.State != VmState.Stopped && vm.State != VmState.Failed && vm.State != VmState.Creating)
                {
                    _logger.LogWarning("VM {VmId} missing from libvirt - marking as failed", vmId);
                    vm.State = VmState.Failed;

                    // Update database
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
            int? vncPort = graphics?.Attribute("port")?.Value != null ? int.Parse(graphics?.Attribute("port")?.Value) : null;

            var state = await GetVmStateFromLibvirtAsync(vmId, ct);

            var vmDir = Path.Combine(_options.VmStoragePath, vmId);
            string? ownerId = null;
            string? vmName = null;
            var vmType = VmType.General;

            var metadataPath = Path.Combine(vmDir, "metadata.json");
            if (File.Exists(metadataPath))
            {
                try
                {
                    var metadata = System.Text.Json.JsonDocument.Parse(
                        await File.ReadAllTextAsync(metadataPath, ct));
                    ownerId = metadata.RootElement.TryGetProperty("ownerId", out var t) ? t.GetString() : null;
                    vmName = metadata.RootElement.TryGetProperty("name", out var n) ? n.GetString() : null;

                    if (metadata.RootElement.TryGetProperty("vmType", out var vt) &&
                        Enum.TryParse<VmType>(vt.GetString(), ignoreCase: true, out var parsedType))
                    {
                        vmType = parsedType;
                    }
                }
                catch { }
            }

            var resolvedName = vmName ?? domain.Element("name")?.Value ?? vmId;

            var instance = new VmInstance
            {
                VmId = vmId,
                Name = resolvedName,
                Spec = new VmSpec
                {
                    Id = vmId,
                    Name = resolvedName,
                    VirtualCpuCores = vcpus,
                    MemoryBytes = memoryBytes,
                    DiskBytes = await GetDiskSizeAsync(diskPath, ct),
                    OwnerId = ownerId ?? "unknown",
                    VmType = vmType,
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

            // -----------------------------------------------------------------
            // Recover services from SQLite if the DB record still exists.
            // After a crash the in-memory dict is empty but the DB may be intact.
            // -----------------------------------------------------------------
            try
            {
                var dbVm = await _repository.LoadVmAsync(vmId);
                if (dbVm?.Services.Count > 0)
                {
                    instance.Services = dbVm.Services;
                    _logger.LogInformation(
                        "Recovered {Count} service(s) from database for VM {VmId}",
                        instance.Services.Count, vmId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not load services from database for VM {VmId}", vmId);
            }

            // -----------------------------------------------------------------
            // For DHT VMs: recover peer ID from the dht-peer-id file on disk.
            // The DHT callback writes this file, and it survives crashes.
            // Re-populate the System service StatusMessage so heartbeats report
            // the peer ID without waiting for a callback that won't re-fire.
            // -----------------------------------------------------------------
            if (vmType == VmType.Dht)
            {
                var peerIdPath = Path.Combine(vmDir, "dht-peer-id");
                if (File.Exists(peerIdPath))
                {
                    try
                    {
                        var peerId = (await File.ReadAllTextAsync(peerIdPath, ct)).Trim();
                        if (!string.IsNullOrEmpty(peerId))
                        {
                            var systemService = instance.Services.FirstOrDefault(s => s.Name == "System");
                            if (systemService == null)
                            {
                                systemService = new VmServiceStatus
                                {
                                    Name = "System",
                                    CheckType = CheckType.CloudInitDone,
                                    TimeoutSeconds = 300
                                };
                                instance.Services.Add(systemService);
                            }

                            systemService.Status = ServiceReadiness.Ready;
                            systemService.StatusMessage = $"peerId={peerId}";
                            systemService.ReadyAt ??= DateTime.UtcNow;
                            systemService.LastCheckAt = DateTime.UtcNow;

                            _logger.LogInformation(
                                "Recovered DHT peer ID from disk for VM {VmId}: {PeerId}",
                                vmId, peerId.Length > 16 ? peerId[..16] + "..." : peerId);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Could not read dht-peer-id for VM {VmId}", vmId);
                    }
                }
            }

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
            int? port = vm.VncPort;
            if (port != null && port >= maxPort)
            {
                maxPort = (int)(port + 1);
            }
        }
        _nextVncPort = maxPort;
        _logger.LogDebug("Next VNC port set to {Port}", _nextVncPort);
    }

    public async Task<VmState> GetVmStateFromLibvirtAsync(string vmId, CancellationToken ct)
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

    public async Task<VmOperationResult> CreateVmAsync(VmSpec spec, string? password = null, CancellationToken ct = default)
    {
        if (_isWindows)
        {
            _logger.LogWarning("VM creation not supported on Windows - requires Linux with KVM/libvirt");
            return VmOperationResult.Fail(spec.Id,
                "VM creation requires Linux with KVM/libvirt. Windows detected.", "PLATFORM_UNSUPPORTED");
        }

        if (!_initialized)
        {
            await InitializeAsync(ct);
        }

        await _lock.WaitAsync(ct);
        try
        {
            if (_vms.ContainsKey(spec.Id))
            {
                return VmOperationResult.Fail(spec.Id, "VM already exists", "DUPLICATE");
            }

            // Create VM instance tracking object
            var instance = new VmInstance
            {
                VmId = spec.Id,
                Name = spec.Name,
                Spec = spec,
                State = VmState.Creating,
                CreatedAt = DateTime.UtcNow,
                LastHeartbeat = DateTime.UtcNow,
                VncPort = _nextVncPort++
            };

            _vms.Add(spec.Id, instance);
            await _repository.SaveVmAsync(instance);

            _logger.LogInformation("Creating VM {VmId}: {VCpus} vCPUs, {MemMB}MB RAM, {DiskGB}GB disk",
                spec.Id, spec.VirtualCpuCores, spec.MemoryBytes / 1024 / 1024, spec.DiskBytes / 1024 / 1024 / 1024);

            var vmDir = Path.Combine(_options.VmStoragePath, spec.Id);
            Directory.CreateDirectory(vmDir);

            // =====================================================
            // STEP 1: Prepare base image
            // =====================================================
            _logger.LogInformation("VM {VmId}: Downloading/preparing base image from {Url}",
                spec.Id, spec.BaseImageUrl);

            string baseImagePath;
            try
            {
                baseImagePath = await _imageManager.EnsureImageAvailableAsync(
                    spec.BaseImageUrl,
                    spec.BaseImageHash,
                    ct);
                _logger.LogInformation("VM {VmId}: Base image ready at {Path}", spec.Id, baseImagePath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "VM {VmId}: Failed to prepare base image", spec.Id);
                instance.State = VmState.Failed;
                await _repository.UpdateVmStateAsync(spec.Id, VmState.Failed);
                return VmOperationResult.Fail(spec.Id, $"Failed to prepare base image: {ex.Message}", "IMAGE_ERROR");
            }

            // =====================================================
            // STEP 1.5: Setup User WireGuard Network
            // =====================================================
            if (!string.IsNullOrEmpty(spec.OwnerId))
            {
                _logger.LogInformation(
                    "VM {VmId}: Setting up user network for owner {OwnerId}",
                    spec.Id, spec.OwnerId);

                var userNetwork = await _userWireGuardManager.EnsureUserNetworkAsync(
                    spec.OwnerId, ct);

                var vmIp = await _userWireGuardManager.AllocateVmIpAsync(
                    spec.OwnerId, spec.Id, ct);

                spec.IpAddress = vmIp;

                _logger.LogInformation(
                    "VM {VmId}: Assigned IP {VmIp} in user network {Subnet}.0/24",
                    spec.Id, vmIp, userNetwork.Subnet);
            }

            // =====================================================
            // STEP 2: Create overlay disk
            // =====================================================
            _logger.LogInformation("VM {VmId}: Creating overlay disk ({DiskGB}GB)",
                spec.Id, spec.DiskBytes / 1024 / 1024 / 1024);

            string diskPath;
            try
            {
                diskPath = await _imageManager.CreateOverlayDiskAsync(
                    baseImagePath,
                    spec.Id,
                    spec.DiskBytes,
                    ct);
                instance.DiskPath = diskPath;
                _logger.LogInformation("VM {VmId}: Overlay disk created at {Path}", spec.Id, diskPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "VM {VmId}: Failed to create overlay disk", spec.Id);
                instance.State = VmState.Failed;
                await _repository.UpdateVmStateAsync(spec.Id, VmState.Failed);
                return VmOperationResult.Fail(spec.Id, $"Failed to create disk: {ex.Message}", "DISK_ERROR");
            }

            // =====================================================
            // STEP 3: Create cloud-init ISO
            // =====================================================
            _logger.LogInformation("VM {VmId}: Creating cloud-init ISO", spec.Id);

            string cloudInitIso;
            try
            {
                cloudInitIso = await CreateCloudInitIsoAsync(spec, vmDir, password, ct);
                if (!string.IsNullOrEmpty(cloudInitIso))
                {
                    _logger.LogInformation("VM {VmId}: Cloud-init ISO created at {Path}", spec.Id, cloudInitIso);
                }
                else
                {
                    _logger.LogWarning("VM {VmId}: Cloud-init ISO creation failed, VM may not configure properly", spec.Id);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "VM {VmId}: Cloud-init ISO creation failed, continuing without it", spec.Id);
                cloudInitIso = string.Empty;
            }

            // =====================================================
            // STEP 4: Generate libvirt XML
            // =====================================================
            var vncPort = instance.VncPort ?? 5900;
            var domainXml = GenerateLibvirtXml(spec, diskPath, cloudInitIso, vncPort);

            var xmlPath = Path.Combine(vmDir, "domain.xml");
            await File.WriteAllTextAsync(xmlPath, domainXml, ct);
            instance.ConfigPath = xmlPath;

            _logger.LogInformation("VM {VmId}: Generated libvirt XML at {Path}", spec.Id, xmlPath);
            _logger.LogDebug("VM {VmId} XML content:\n{Xml}", spec.Id, domainXml);

            // =====================================================
            // STEP 5: Save VM metadata
            // =====================================================
            await SaveVmMetadataAsync(vmDir, spec, ct);

            // =====================================================
            // STEP 6: Define VM in libvirt
            // =====================================================
            _logger.LogInformation("VM {VmId}: Defining VM in libvirt", spec.Id);

            var defineResult = await _executor.ExecuteAsync("virsh", $"define {xmlPath}", ct);
            if (!defineResult.Success)
            {
                _logger.LogError("VM {VmId}: Failed to define VM in libvirt: {Error}",
                    spec.Id, defineResult.StandardError);

                instance.State = VmState.Failed;
                await _repository.UpdateVmStateAsync(spec.Id, VmState.Failed);

                // Cleanup
                try { Directory.Delete(vmDir, recursive: true); } catch { }

                return VmOperationResult.Fail(spec.Id,
                    $"Failed to define VM: {defineResult.StandardError}", "DEFINE_FAILED");
            }

            _logger.LogInformation("VM {VmId}: Successfully defined in libvirt", spec.Id);

            // Update database with all paths
            await _repository.SaveVmAsync(instance);

            var autostartResult = await _executor.ExecuteAsync("virsh", $"autostart {spec.Id}", ct);

            if (autostartResult.Success)
            {
                _logger.LogInformation(
                    "✓ VM {VmId} configured for auto-start on host boot",
                    spec.Id);
            }
            else
            {
                _logger.LogWarning(
                    "⚠ Failed to enable auto-start for VM {VmId}: {Error}",
                    spec.Id, autostartResult.StandardError);
                // Non-fatal - continue with VM creation
            }

            // =====================================================
            // STEP 6.5: Bind GPU to VFIO (if GPU passthrough requested)
            // =====================================================
            if (!string.IsNullOrEmpty(spec.GpuPciAddress))
            {
                _logger.LogInformation("VM {VmId}: GPU passthrough requested for {PciAddr}",
                    spec.Id, spec.GpuPciAddress);

                var vfioBound = await BindToVfioAsync(spec.GpuPciAddress, spec.Id, ct);
                if (!vfioBound)
                {
                    _logger.LogError(
                        "VM {VmId}: Failed to bind GPU {PciAddr} to vfio-pci. " +
                        "VM will be created without GPU passthrough.",
                        spec.Id, spec.GpuPciAddress);
                    // Don't fail the VM — it was already defined. Log the error and let
                    // the operator investigate. The VM start will fail if libvirt can't
                    // access the device, which gives a clear error message.
                }
            }

            // =====================================================
            // STEP 7: Start the VM
            // =====================================================
            instance.State = VmState.Starting;
            await _repository.UpdateVmStateAsync(spec.Id, VmState.Starting);

            _logger.LogInformation("VM {VmId}: Starting VM", spec.Id);

            var startResult = await _executor.ExecuteAsync("virsh", $"start {spec.Id}", ct);
            if (startResult.Success)
            {
                instance.State = VmState.Running;
                instance.StartedAt = DateTime.UtcNow;
                await _repository.UpdateVmStateAsync(spec.Id, VmState.Running);

                _logger.LogInformation("VM {VmId} started successfully", spec.Id);

                // Background task to get IP address
                _ = Task.Run(async () =>
                {
                    await Task.Delay(TimeSpan.FromSeconds(15), ct);

                    await _lock.WaitAsync(ct);
                    try
                    {
                        if (_vms.TryGetValue(spec.Id, out var vm))
                        {
                            var ip = await GetVmIpAddressAsync(spec.Id, ct);
                            if (!string.IsNullOrEmpty(ip))
                            {
                                vm.Spec.IpAddress = ip;
                                await _repository.SaveVmAsync(vm);

                                // Add VM host key to jump user's known_hosts for seamless SSH
                                await AddVmHostKeyToJumpUserAsync(ip, spec.Id, ct);

                                await _repository.UpdateVmIpAsync(spec.Id, ip);
                                _logger.LogInformation("VM {VmId} obtained IP: {Ip}", spec.Id, ip);
                            }

                            if (startResult.Success && !string.IsNullOrEmpty(spec.OwnerId) &&
                                !string.IsNullOrEmpty(spec.IpAddress))
                            {
                                try
                                {
                                    await _userWireGuardManager.AddVmPeerAsync(
                                        spec.OwnerId,
                                        spec.Id,
                                        spec.IpAddress,
                                        ct);

                                    _logger.LogInformation(
                                        "✓ VM {VmId} added to user network (IP: {VmIp})",
                                        spec.Id, spec.IpAddress);
                                }
                                catch (Exception ex)
                                {
                                    _logger.LogError(
                                        ex,
                                        "Failed to add VM {VmId} to user network, " +
                                        "but VM is running. Network isolation may not work correctly.",
                                        spec.Id);
                                }
                            }
                        }
                    }
                    finally
                    {
                        _lock.Release();
                    }
                }, ct);

                return VmOperationResult.Ok(spec.Id, VmState.Running);
            }
            else
            {
                instance.State = VmState.Failed;
                await _repository.UpdateVmStateAsync(spec.Id, VmState.Failed);

                _logger.LogError("Failed to start VM {VmId}: {Error}", spec.Id, startResult.StandardError);
                return VmOperationResult.Fail(spec.Id, startResult.StandardError, "START_FAILED");
            }

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "VM {VmId}: Unexpected error during creation", spec.Id);

            if (_vms.TryGetValue(spec.Id, out var instance))
            {
                instance.State = VmState.Failed;
                await _repository.UpdateVmStateAsync(spec.Id, VmState.Failed);
            }

            return VmOperationResult.Fail(spec.Id, ex.Message, "UNEXPECTED_ERROR");
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
            vmId = spec.Id,
            name = spec.Name,
            ownerId = spec.OwnerId,
            vmType = spec.VmType.ToString(),
            createdAt = DateTime.UtcNow,
            virtualCpuCores = spec.VirtualCpuCores,
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

    public async Task<VmOperationResult> RestartVmAsync(string vmId, bool force = false, CancellationToken ct = default)
    {
        if (!_initialized) await InitializeAsync(ct);
        _logger.LogInformation("Restarting VM {VmId} (force={Force})", vmId, force);
        var command = force ? "reset" : "reboot";
        var result = await _executor.ExecuteAsync("virsh", $"{command} {vmId}", ct);
        if (result.Success)
        {
            if (_vms.TryGetValue(vmId, out var instance))
            {
                instance.State = VmState.Starting;
                instance.StartedAt = DateTime.UtcNow;
                // Persist state change
                await _repository.UpdateVmStateAsync(vmId, VmState.Starting);
            }
            return VmOperationResult.Ok(vmId, VmState.Running);
        }
        _logger.LogError("Failed to restart VM {VmId}: {Error}", vmId, result.StandardError);
        return VmOperationResult.Fail(vmId, result.StandardError, "RESTART_FAILED");
    }

    public async Task<VmOperationResult> DeleteVmAsync(string vmId, CancellationToken ct = default)
    {
        if (!_initialized) await InitializeAsync(ct);

        _logger.LogInformation("Deleting VM {VmId}", vmId);

        if (!_vms.TryGetValue(vmId, out var instance)) { 
            // Try to recover from database
            instance = await _repository.LoadVmAsync(vmId);
            if (instance != null)
            {
                _vms[vmId] = instance;
                _logger.LogInformation("Recovered VM {VmId} from database for deletion", vmId);
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
                // ====================================================================
                // RECONCILIATION: VM doesn't exist - treat as successful deletion
                // This makes delete operations idempotent and prevents stuck VMs
                // ====================================================================
                _logger.LogInformation(
                    "✓ VM {VmId} not found for deletion - already deleted (idempotent success). " +
                    "Cleaning up any remaining local state.",
                    vmId);
                
                // Clean up any orphaned resources that might still exist
                await _repository.DeleteVmAsync(vmId);
                CleanupVmDirectory(vmId, orphaned: true);
                
                return VmOperationResult.Ok(vmId, VmState.Stopped);
            }
        }

        // Stop if running
        var state = await GetVmStateFromLibvirtAsync(vmId, ct);
        if (state == VmState.Running || state == VmState.Paused)
        {
            await _executor.ExecuteAsync("virsh", $"destroy {vmId}", ct);
            await Task.Delay(500, ct);
        }

        // Return GPU to host if passthrough was active
        if (!string.IsNullOrEmpty(instance.Spec.GpuPciAddress))
        {
            await UnbindFromVfioAsync(instance.Spec.GpuPciAddress, vmId, ct);
        }

        // Undefine from libvirt
        var undefResult = await _executor.ExecuteAsync("virsh", $"undefine {vmId} --remove-all-storage --nvram", ct);

        if (!undefResult.Success && !undefResult.StandardError.Contains("not found"))
        {
            _logger.LogWarning("Failed to undefine VM {VmId}: {Error}", vmId, undefResult.StandardError);
        }

        // Remove from tracking
        _vms.Remove(vmId);

        // Remove from database
        await _repository.DeleteVmAsync(vmId);

        // Clean up directory
        CleanupVmDirectory(vmId, orphaned: false);

        if (!string.IsNullOrEmpty(instance.Spec.OwnerId))
        {
            try
            {
                await _userWireGuardManager.RemoveVmPeerAsync(
                    instance.Spec.OwnerId,
                    vmId,
                    ct);

                // Cleanup user network if this was the last VM
                await _userWireGuardManager.CleanupUserNetworkIfEmptyAsync(
                    instance.Spec.OwnerId,
                    ct);

                _logger.LogInformation(
                    "✓ VM {VmId} removed from user network",
                    vmId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(
                    ex,
                    "Failed to remove VM {VmId} from user network during deletion",
                    vmId);
            }
        }

        return VmOperationResult.Ok(vmId, VmState.Stopped);
    }

    /// <summary>
    /// Clean up VM directory
    /// </summary>
    private void CleanupVmDirectory(string vmId, bool orphaned)
    {
        var vmDirectory = Path.Combine(_options.VmStoragePath, vmId);
        if (Directory.Exists(vmDirectory))
        {
            try
            {
                Directory.Delete(vmDirectory, recursive: true);
                _logger.LogInformation(
                    orphaned 
                        ? "Cleaned up orphaned VM directory: {VmDir}" 
                        : "Deleted VM directory: {VmDir}", 
                    vmDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(
                    ex, 
                    orphaned 
                        ? "Failed to clean up orphaned VM directory {VmDir}" 
                        : "Failed to delete VM directory {VmDir}", 
                    vmDirectory);
            }
        }
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

        // Dynamically discover network interface using virsh domiflist
        var iflistResult = await _executor.ExecuteAsync("virsh", $"domiflist {vmId}", ct);
        if (iflistResult.Success)
        {
            // Parse output to find the interface name
            // Format: "vnet165     network   default   virtio   52:54:00:3f:cf:bf"
            var lines = iflistResult.StandardOutput.Split('\n');
            foreach (var line in lines)
            {
                var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2 && parts[0].StartsWith("vnet"))
                {
                    var interfaceName = parts[0];

                    // Query network stats for the discovered interface
                    var netResult = await _executor.ExecuteAsync("virsh",
                        $"domifstat {vmId} {interfaceName}", ct);

                    if (netResult.Success)
                    {
                        var rxMatch = Regex.Match(netResult.StandardOutput, @"rx_bytes\s+(\d+)");
                        var txMatch = Regex.Match(netResult.StandardOutput, @"tx_bytes\s+(\d+)");

                        if (rxMatch.Success && txMatch.Success)
                        {
                            usage.NetworkRxBytes = long.Parse(rxMatch.Groups[1].Value);
                            usage.NetworkTxBytes = long.Parse(txMatch.Groups[1].Value);

                            _logger.LogDebug("VM {VmId}: Network stats from interface {Interface}: RX={RxBytes}, TX={TxBytes}",
                                vmId, interfaceName, usage.NetworkRxBytes, usage.NetworkTxBytes);
                            break; // Found it, stop searching
                        }
                    }
                }
            }
        }

        return usage;
    }

    public async Task<bool> VmExistsAsync(string vmId, CancellationToken ct = default)
    {
        var result = await _executor.ExecuteAsync("virsh", $"dominfo {vmId}", ct);
        return result.Success;
    }

    /// <summary>
    /// Asynchronously retrieves the IPv4 address assigned to the specified virtual machine.
    /// </summary>
    /// <remarks>This method queries the DHCP leases of the default network and matches them to the MAC
    /// address of the specified virtual machine. Returns null if the IP address cannot be determined, such as when the
    /// VM is not running or not connected to the default network.</remarks>
    /// <param name="vmId">The unique identifier of the virtual machine for which to obtain the IP address. Cannot be null or empty.</param>
    /// <param name="ct">A cancellation token that can be used to cancel the operation.</param>
    /// <returns>A string containing the IPv4 address of the virtual machine if found; otherwise, null.</returns>
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
    /// Add VM's SSH host key to decloud user's known_hosts for seamless SSH access.
    /// This prevents "Host key verification failed" errors when users connect through the jump host.
    /// </summary>
    private async Task AddVmHostKeyToJumpUserAsync(string vmIp, string vmId, CancellationToken ct)
    {
        try
        {
            _logger.LogInformation("VM {VmId}: Scanning SSH host key for {VmIp}", vmId, vmIp);

            // Wait for SSH to be ready (up to 60 seconds)
            var sshReady = false;
            for (int i = 0; i < 12; i++)
            {
                var testResult = await _executor.ExecuteAsync("nc", $"-zv -w 2 {vmIp} 22", ct);
                if (testResult.Success || testResult.StandardError.Contains("succeeded") ||
                    testResult.StandardError.Contains("open"))
                {
                    sshReady = true;
                    break;
                }
                await Task.Delay(5000, ct);
            }

            if (!sshReady)
            {
                _logger.LogWarning("VM {VmId}: SSH not ready after 60 seconds, skipping host key scan", vmId);
                return;
            }

            _logger.LogInformation("VM {VmId}: SSH is ready, scanning host keys", vmId);

            // Scan the host key using ssh-keyscan
            var scanResult = await _executor.ExecuteAsync("ssh-keyscan",
                $"-H -T 10 {vmIp}", ct);

            if (!scanResult.Success || string.IsNullOrWhiteSpace(scanResult.StandardOutput))
            {
                _logger.LogWarning("VM {VmId}: Failed to scan host key: {Error}",
                    vmId, scanResult.StandardError);
                return;
            }

            var hostKey = scanResult.StandardOutput.Trim();
            if (string.IsNullOrEmpty(hostKey) || hostKey.StartsWith("#"))
            {
                _logger.LogWarning("VM {VmId}: No valid host key returned from ssh-keyscan", vmId);
                return;
            }

            // Append to decloud's known_hosts
            var knownHostsPath = "/home/decloud/.ssh/known_hosts";

            // Ensure the file exists
            if (!File.Exists(knownHostsPath))
            {
                var dir = Path.GetDirectoryName(knownHostsPath);
                if (!string.IsNullOrEmpty(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                await File.WriteAllTextAsync(knownHostsPath, "", ct);

                // Set proper ownership
                await _executor.ExecuteAsync("chown", "decloud:decloud /home/decloud/.ssh/known_hosts", ct);
                await _executor.ExecuteAsync("chmod", "600 /home/decloud/.ssh/known_hosts", ct);
            }

            // Append the host key (with newline)
            await File.AppendAllTextAsync(knownHostsPath, hostKey + "\n", ct);

            _logger.LogInformation("✓ VM {VmId}: Host key added to jump user's known_hosts", vmId);
        }
        catch (Exception ex)
        {
            // Don't fail VM creation if host key scanning fails
            _logger.LogWarning(ex, "VM {VmId}: Failed to add host key to jump user, manual acceptance required", vmId);
        }
    }

    /// <summary>
    /// Create cloud-init ISO using template-based configuration.
    /// Supports multiple VM types through specialized templates.
    /// </summary>
    /// <remarks>
    /// This method completely replaces inline cloud-init generation with a template-based approach.
    /// Templates are loaded from CloudInit/Templates/ directory and processed with runtime variables.
    /// 
    /// Template placeholders are replaced with actual values:
    /// - __VM_ID__: Unique VM identifier
    /// - __VM_NAME__: Human-readable VM name
    /// - __HOSTNAME__: System hostname
    /// - __WIREGUARD_PRIVATE_KEY__: Generated WireGuard private key (relay VMs)
    /// - __WIREGUARD_PUBLIC_KEY__: Generated WireGuard public key (relay VMs)
    /// - __SSH_AUTHORIZED_KEYS_BLOCK__: User's SSH public keys
    /// - __PASSWORD_CONFIG_BLOCK__: Password configuration block
    /// - __CA_PUBLIC_KEY__: SSH CA public key for certificate auth
    /// - And many more...
    /// </remarks>
    private async Task<string> CreateCloudInitIsoAsync(
        VmSpec spec,
        string vmDir,
        string? password = null,
        CancellationToken ct = default)
    {
        _logger.LogInformation(
            "VM {VmId}: Creating cloud-init ISO using template for VM type {VmType}",
            spec.Id, spec.VmType);

        try
        {
            // =====================================================
            // STEP 1: Build base template variables
            // =====================================================
            var variables = new Dictionary<string, string>
            {
                ["__VM_ID__"] = spec.Id,
                ["__VM_NAME__"] = spec.Name,
                ["__HOSTNAME__"] = spec.Name
            };

            // =====================================================
            // STEP 2: Build SSH Authorized Keys Block
            // =====================================================
            var hasSshKey = !string.IsNullOrEmpty(spec.SshPublicKey);
            string sshKeysBlock = "";

            if (hasSshKey && spec.VmType != VmType.Relay)
            {
                var sb = new StringBuilder();
                sb.AppendLine("ssh_authorized_keys:");

                foreach (var key in spec.SshPublicKey!.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                {
                    var trimmedKey = key.Trim();
                    if (!string.IsNullOrEmpty(trimmedKey))
                    {
                        sb.AppendLine($"  - {trimmedKey}");
                    }
                }

                sshKeysBlock = sb.ToString().TrimEnd();
                _logger.LogInformation("VM {VmId}: Including SSH public keys", spec.Id);
            }
            else
            {
                sshKeysBlock = "# No SSH keys provided";
            }

            variables["__SSH_AUTHORIZED_KEYS_BLOCK__"] = sshKeysBlock;

            // =====================================================
            // STEP 3: Build Password Configuration Block
            // =====================================================
            var hasPassword = !string.IsNullOrEmpty(password);
            string passwordBlock = "";

            if (hasPassword && spec.VmType != VmType.Relay)
            {
                var sb = new StringBuilder();
                sb.AppendLine("chpasswd:");
                sb.AppendLine("  list: |");
                sb.AppendLine($"    root:{password}");
                sb.AppendLine("  expire: false");

                passwordBlock = sb.ToString().TrimEnd();
                variables["__SSH_PASSWORD_AUTH__"] = "true";

                _logger.LogInformation("VM {VmId}: Using root password authentication", spec.Id);
            }
            else
            {
                passwordBlock = "# No password authentication";
                variables["__SSH_PASSWORD_AUTH__"] = "false";
            }

            variables["__PASSWORD_CONFIG_BLOCK__"] = passwordBlock;

            // =====================================================
            // STEP 4: Build Password SSH Commands Block
            // =====================================================
            string passwordSshCommandsBlock = "";

            if (hasPassword && spec.VmType != VmType.Relay)
            {
                // Generate a single compound command with && to chain multiple operations
                var commands = new[]
                {
                    "sed -i 's/^#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config",
                    "sed -i 's/^PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config",
                    "sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config",
                    "sed -i 's/^#PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config",
                    "systemctl restart sshd || systemctl restart ssh"
                };

                // Join with && to make one compound command
                passwordSshCommandsBlock = string.Join(" && ", commands);

                _logger.LogInformation("VM {VmId}: Password authentication will be enabled via compound command", spec.Id);
            }
            else
            {
                // When no password, use a no-op command
                passwordSshCommandsBlock = "echo 'Password authentication not configured'";
            }

            variables["__PASSWORD_SSH_COMMANDS_BLOCK__"] = passwordSshCommandsBlock;

            // =====================================================
            // STEP 5: SSH Certificate Authority Public Key
            // =====================================================
            var caPublicKeyPath = "/etc/ssh/decloud_ca.pub";
            if (File.Exists(caPublicKeyPath))
            {
                var caPublicKey = await File.ReadAllTextAsync(caPublicKeyPath, ct);

                // Indent the CA key content for YAML (6 spaces for write_files content)
                var caKeyIndented = new StringBuilder();
                foreach (var line in caPublicKey.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                {
                    caKeyIndented.AppendLine($"      {line.Trim()}");
                }

                variables["__CA_PUBLIC_KEY__"] = caKeyIndented.ToString().TrimEnd();
                _logger.LogInformation("VM {VmId}: Including SSH CA public key", spec.Id);
            }
            else
            {
                _logger.LogWarning(
                    "VM {VmId}: SSH CA public key not found at {Path} - certificate auth will not work!",
                    spec.Id, caPublicKeyPath);
                variables["__CA_PUBLIC_KEY__"] = "      # ERROR: CA public key not available";
            }

            // =====================================================
            // STEP 5.5: Orchestrator Public Key (for Relay VMs)
            // =====================================================
            if (spec.VmType == VmType.Relay)
            {
                const string orchestratorPubKeyPath = "/etc/wireguard/orchestrator-public.key";

                var orchestratorUrl = _nodeMetadata.OrchestratorUrl;
                var relayCapacity = spec.Labels?.GetValueOrDefault("relay-capacity", "10") ?? "10";
                var relayRegion = spec.Labels?.GetValueOrDefault("relay-region")
                               ?? _nodeMetadata.Region
                               ?? "default";
                var publicIp = spec.Labels?.GetValueOrDefault("node-public-ip")
                            ?? _nodeMetadata.PublicIp
                            ?? "";

                if (File.Exists(orchestratorPubKeyPath))
                {
                    var orchestratorPublicKey = await File.ReadAllTextAsync(orchestratorPubKeyPath, ct);
                    variables["__ORCHESTRATOR_PUBLIC_KEY__"] = orchestratorPublicKey.Trim();

                    _logger.LogInformation(
                        "VM {VmId}: Including orchestrator WireGuard public key for relay peer pre-configuration",
                        spec.Id);
                }
                else
                {
                    _logger.LogWarning(
                        "VM {VmId}: Orchestrator WireGuard public key not found at {Path} - " +
                        "relay will not have orchestrator peer pre-configured! " +
                        "Ensure orchestrator was installed with --enable-wireguard flag.",
                        spec.Id, orchestratorPubKeyPath);

                    variables["__ORCHESTRATOR_PUBLIC_KEY__"] = "# ERROR: Orchestrator public key not available";
                }

                // Relay VM metadata placeholders
                variables["__ORCHESTRATOR_URL__"] = orchestratorUrl;
                var orchestratorUri = new Uri(orchestratorUrl);
                variables["__ORCHESTRATOR_IP__"] = orchestratorUri.Host;
                variables["__ORCHESTRATOR_PORT__"] = "51821";
                variables["__NODE_ID__"] = _nodeMetadata.NodeId;
                variables["__HOST_MACHINE_ID__"] = _nodeMetadata.MachineId;
                variables["__PUBLIC_IP__"] = publicIp;
                variables["__RELAY_CAPACITY__"] = relayCapacity;
                variables["__RELAY_REGION__"] = relayRegion;
                variables["__TIMESTAMP__"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ");
                var relaySubnet = spec.Labels?.GetValueOrDefault("relay-subnet") ?? "0";
                variables["__RELAY_SUBNET__"] = relaySubnet;

                _logger.LogInformation(
                    "VM {VmId}: Set relay metadata - Subnet={Subnet}, Capacity={Capacity}, Region={Region}",
                    spec.Id, relaySubnet, relayCapacity, relayRegion);
            }

            // =====================================================
            // STEP 5.6: DHT VM metadata (from orchestrator labels)
            // =====================================================
            if (spec.VmType == VmType.Dht)
            {
                variables["__NODE_ID__"] = spec.Labels?.GetValueOrDefault("node-id")
                                        ?? _nodeMetadata.NodeId;
                variables["__HOST_MACHINE_ID__"] = _nodeMetadata.MachineId;
                variables["__TIMESTAMP__"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ");

                // Orchestrator URL for direct DHT→orchestrator bootstrap polling
                // (same pattern as relay VMs use for notify-orchestrator.sh)
                variables["__ORCHESTRATOR_URL__"] = _nodeMetadata.OrchestratorUrl;

                _logger.LogInformation(
                    "VM {VmId}: Set DHT metadata - NodeId={NodeId}, AdvertiseIP={AdvIP}, OrchestratorUrl={Url}",
                    spec.Id,
                    variables["__NODE_ID__"],
                    spec.Labels?.GetValueOrDefault("dht-advertise-ip") ?? "(from template)",
                    _nodeMetadata.OrchestratorUrl);
            }

            // =====================================================
            // STEP 6: Process template using CloudInitTemplateService
            // =====================================================
            string cloudInitYaml;

            // Check if custom UserData is provided (from orchestrator template)
            if (!string.IsNullOrEmpty(spec.CloudInitUserData))
            {
                _logger.LogInformation(
                    "VM {VmId}: Using custom cloud-init UserData from orchestrator ({Bytes} bytes)",
                    spec.Id, spec.CloudInitUserData.Length);
                
                // IMPORTANT: Custom UserData from orchestrator needs to be merged with
                // base configuration (hostname, password, SSH keys, etc.)
                cloudInitYaml = MergeCustomUserDataWithBaseConfig(
                    spec.CloudInitUserData,
                    spec.Name,
                    sshKeysBlock,
                    passwordBlock,
                    hasPassword);
                
                _logger.LogInformation(
                    "VM {VmId}: Merged custom UserData with base configuration",
                    spec.Id);
            }
            else
            {
                // Use built-in template processing
                try
                {
                    cloudInitYaml = await _templateService.ProcessTemplateAsync(
                        spec.VmType,
                        spec,
                        variables,
                        ct);

                    _logger.LogInformation(
                        "VM {VmId}: Successfully processed cloud-init template for {VmType}",
                        spec.Id, spec.VmType);
                }
                catch (Exception ex)
                {
                    _logger.LogError(
                        ex,
                        "VM {VmId}: Failed to process cloud-init template for {VmType}",
                        spec.Id, spec.VmType);
                    throw;
                }
            }

            // =====================================================
            // STEP 6.5: Ensure qemu-guest-agent is installed (required for readiness monitoring)
            // =====================================================
            cloudInitYaml = EnsureQemuGuestAgent(cloudInitYaml);

            // =====================================================
            // STEP 7: Write user-data and meta-data files
            // =====================================================
            var userDataPath = Path.Combine(vmDir, "user-data");
            var metaDataPath = Path.Combine(vmDir, "meta-data");

            await File.WriteAllTextAsync(userDataPath, cloudInitYaml, ct);

            var metaData = $"instance-id: {spec.Id}\nlocal-hostname: {spec.Name}\n";
            await File.WriteAllTextAsync(metaDataPath, metaData, ct);

            _logger.LogDebug(
                "VM {VmId}: Cloud-init configuration:\n{UserData}",
                spec.Id, cloudInitYaml);

            // =====================================================
            // STEP 8: Create cloud-init ISO
            // =====================================================
            var isoPath = Path.Combine(vmDir, "cloud-init.iso");

            // Try genisoimage first (most common)
            var result = await _executor.ExecuteAsync(
                "genisoimage",
                $"-output {isoPath} -volid cidata -joliet -rock {userDataPath} {metaDataPath}",
                ct);

            // Fallback to cloud-localds if genisoimage not available
            if (!result.Success)
            {
                _logger.LogDebug(
                    "VM {VmId}: genisoimage failed, trying cloud-localds: {Error}",
                    spec.Id, result.StandardError);

                result = await _executor.ExecuteAsync(
                    "cloud-localds",
                    $"{isoPath} {userDataPath} {metaDataPath}",
                    ct);
            }

            if (!result.Success)
            {
                _logger.LogError(
                    "VM {VmId}: Failed to create cloud-init ISO: {Error}",
                    spec.Id, result.StandardError);
                return string.Empty;
            }

            _logger.LogInformation(
                "✓ VM {VmId}: Created cloud-init ISO at {Path} (type: {VmType}, password: {HasPassword}, ssh: {HasSshKey}, CA: {HasCA})",
                spec.Id,
                isoPath,
                spec.VmType,
                hasPassword,
                hasSshKey,
                File.Exists(caPublicKeyPath));

            return isoPath;
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "VM {VmId}: Unexpected error creating cloud-init ISO",
                spec.Id);
            throw;
        }
    }

    /// <summary>
    /// Ensure qemu-guest-agent is installed and explicitly started in cloud-init.
    /// Required for VmReadinessMonitor to probe service readiness via virtio channel.
    /// Both packages: and runcmd: injection are needed because the systemd unit has
    /// ConditionPathExists=/dev/virtio-ports/org.qemu.guest_agent.0 — if the virtio
    /// device isn't ready at package install time, the service silently skips starting
    /// and systemd won't retry.
    /// </summary>
    private static string EnsureQemuGuestAgent(string cloudInitYaml)
    {
        const string pkg = "qemu-guest-agent";
        if (cloudInitYaml.Contains(pkg))
            return cloudInitYaml;

        var result = cloudInitYaml;

        // 1. Inject into packages: section for early installation
        var packagesIndex = result.IndexOf("\npackages:", StringComparison.Ordinal);
        if (packagesIndex >= 0)
        {
            var lineEnd = result.IndexOf('\n', packagesIndex + 1);
            if (lineEnd >= 0)
                result = result.Insert(lineEnd + 1, $"  - {pkg}\n");
        }
        else
        {
            // No packages: section — inject one before runcmd:
            var runcmdPos = result.IndexOf("\nruncmd:", StringComparison.Ordinal);
            if (runcmdPos >= 0)
                result = result.Insert(runcmdPos, $"\npackages:\n  - {pkg}\n");
            else
                result += $"\n\npackages:\n  - {pkg}\n";
        }

        // 2. Inject runcmd step to ensure the service is running after boot.
        //    packages: installs early, but the systemd ConditionPathExists may fail
        //    if the virtio device isn't ready yet. runcmd runs later when devices
        //    are guaranteed available.
        var runcmdIndex = result.IndexOf("\nruncmd:", StringComparison.Ordinal);
        if (runcmdIndex >= 0)
        {
            var runcmdLineEnd = result.IndexOf('\n', runcmdIndex + 1);
            if (runcmdLineEnd >= 0)
                result = result.Insert(runcmdLineEnd + 1,
                    "  - systemctl enable --now qemu-guest-agent || true\n");
        }

        return result;
    }


    /// <summary>
    /// Merge custom cloud-init UserData with base configuration (hostname, password, SSH)
    /// This ensures marketplace templates get proper authentication even when using custom cloud-init
    /// </summary>
    private string MergeCustomUserDataWithBaseConfig(
        string customUserData,
        string hostname,
        string sshKeysBlock,
        string passwordBlock,
        bool hasPassword)
    {
        var lines = new List<string>();
        
        // Parse custom UserData line by line
        using var reader = new StringReader(customUserData);
        string? line;
        bool foundRuncmd = false;
        bool afterRuncmd = false;
        
        while ((line = reader.ReadLine()) != null)
        {
            // Skip the first #cloud-config line, we'll add it back at the start
            if (line.Trim() == "#cloud-config" && lines.Count == 0)
            {
                continue;
            }
            
            // Check if we're entering the runcmd section
            if (line.TrimStart().StartsWith("runcmd:") && !foundRuncmd)
            {
                foundRuncmd = true;
                
                // Before runcmd, inject base configuration if not already present
                if (!customUserData.Contains("hostname:", StringComparison.OrdinalIgnoreCase))
                {
                    lines.Add($"hostname: {hostname}");
                    lines.Add("manage_etc_hosts: true");
                    lines.Add("");
                }
                
                if (!customUserData.Contains("disable_root:", StringComparison.OrdinalIgnoreCase))
                {
                    lines.Add("disable_root: false");
                    lines.Add("");
                }
                
                // Add SSH keys if provided
                if (!string.IsNullOrEmpty(sshKeysBlock) && !sshKeysBlock.Contains("# No SSH keys"))
                {
                    lines.Add(sshKeysBlock);
                    lines.Add("");
                }
                
                // Add password configuration if provided (root user only)
                if (hasPassword && !string.IsNullOrEmpty(passwordBlock) && !passwordBlock.Contains("# No password"))
                {
                    // Use the password block as-is (already configured for root)
                    lines.Add(passwordBlock);
                    lines.Add("");
                }
                
                // Add SSH password auth flag
                if (hasPassword)
                {
                    lines.Add("ssh_pwauth: true");
                    lines.Add("");
                }
                
                // Now add the runcmd line
                lines.Add(line);
                afterRuncmd = true;
                continue;
            }
            
            // If we just added runcmd, inject SSH setup commands as first runcmd items
            if (afterRuncmd && hasPassword && line.TrimStart().StartsWith("-"))
            {
                // Add SSH configuration commands before other runcmd items
                lines.Add("  # Enable password authentication for SSH");
                lines.Add("  - mkdir -p /etc/ssh/sshd_config.d");
                lines.Add("  - |");
                lines.Add("    cat > /etc/ssh/sshd_config.d/99-decloud-password-auth.conf <<'SSHEOF'");
                lines.Add("    # DeCloud: Enable password authentication");
                lines.Add("    PasswordAuthentication yes");
                lines.Add("    PermitRootLogin yes");
                lines.Add("    ChallengeResponseAuthentication no");
                lines.Add("    UsePAM yes");
                lines.Add("    SSHEOF");
                lines.Add("  - systemctl restart sshd || systemctl restart ssh");
                lines.Add("");
                
                // Now add the first actual runcmd item
                afterRuncmd = false;
            }
            
            lines.Add(line);
        }
        
        // Rebuild with #cloud-config header
        var merged = "#cloud-config\n\n" + string.Join("\n", lines);
        
        _logger.LogDebug(
            "Merged custom UserData with base config (hostname: {Hostname}, password: {HasPassword}, ssh: {HasSsh})",
            hostname, hasPassword, !sshKeysBlock.Contains("# No SSH keys"));
        
        return merged;
    }

    // =====================================================================
    // VFIO GPU PASSTHROUGH HELPERS
    // =====================================================================

    /// <summary>
    /// Parse a PCI address like "0000:01:00.0" into its domain/bus/slot/function components.
    /// Returns null if the address cannot be parsed.
    /// </summary>
    internal static (int Domain, int Bus, int Slot, int Function)? ParsePciAddress(string pciAddress)
    {
        if (string.IsNullOrWhiteSpace(pciAddress))
            return null;

        var addr = ResourceDiscoveryService.NormalizePciAddress(pciAddress);

        // Expected format: DDDD:BB:SS.F (e.g., 0000:01:00.0)
        var parts = addr.Split(new[] { ':', '.' });
        if (parts.Length != 4)
            return null;

        if (int.TryParse(parts[0], System.Globalization.NumberStyles.HexNumber, null, out var domain) &&
            int.TryParse(parts[1], System.Globalization.NumberStyles.HexNumber, null, out var bus) &&
            int.TryParse(parts[2], System.Globalization.NumberStyles.HexNumber, null, out var slot) &&
            int.TryParse(parts[3], System.Globalization.NumberStyles.HexNumber, null, out var func))
        {
            return (domain, bus, slot, func);
        }

        return null;
    }

    /// <summary>
    /// Bind a PCI device to the vfio-pci driver so it can be passed through to a VM.
    /// Steps:
    ///   1. Unbind from current driver (e.g. nvidia)
    ///   2. Set driver_override to vfio-pci
    ///   3. Trigger reprobe
    /// </summary>
    private async Task<bool> BindToVfioAsync(string pciAddress, string vmId, CancellationToken ct)
    {
        var addr = ResourceDiscoveryService.NormalizePciAddress(pciAddress);
        _logger.LogInformation("VM {VmId}: Binding GPU {PciAddr} to vfio-pci", vmId, addr);

        try
        {
            // Ensure vfio-pci module is loaded
            var modprobeResult = await _executor.ExecuteAsync("modprobe", "vfio-pci", ct);
            if (!modprobeResult.Success)
            {
                _logger.LogWarning("VM {VmId}: modprobe vfio-pci failed: {Error}. " +
                    "Continuing anyway (module may already be loaded).",
                    vmId, modprobeResult.StandardError);
            }

            var driverPath = $"/sys/bus/pci/devices/{addr}/driver";

            // Step 1: Unbind from current driver
            if (Directory.Exists(driverPath))
            {
                var unbindResult = await _executor.ExecuteAsync(
                    "bash", $"-c \"echo '{addr}' > /sys/bus/pci/devices/{addr}/driver/unbind\"", ct);

                if (!unbindResult.Success)
                {
                    _logger.LogError("VM {VmId}: Failed to unbind GPU {PciAddr} from current driver: {Error}",
                        vmId, addr, unbindResult.StandardError);
                    return false;
                }

                _logger.LogInformation("VM {VmId}: Unbound GPU {PciAddr} from existing driver", vmId, addr);
            }

            // Step 2: Set driver_override to vfio-pci
            var overrideResult = await _executor.ExecuteAsync(
                "bash", $"-c \"echo 'vfio-pci' > /sys/bus/pci/devices/{addr}/driver_override\"", ct);

            if (!overrideResult.Success)
            {
                _logger.LogError("VM {VmId}: Failed to set driver_override for GPU {PciAddr}: {Error}",
                    vmId, addr, overrideResult.StandardError);
                return false;
            }

            // Step 3: Trigger reprobe so vfio-pci picks up the device
            var probeResult = await _executor.ExecuteAsync(
                "bash", $"-c \"echo '{addr}' > /sys/bus/pci/drivers_probe\"", ct);

            if (!probeResult.Success)
            {
                _logger.LogError("VM {VmId}: Failed to reprobe GPU {PciAddr}: {Error}",
                    vmId, addr, probeResult.StandardError);
                return false;
            }

            // Verify the device is now bound to vfio-pci
            await Task.Delay(500, ct); // Give kernel time to bind
            var verifyResult = await _executor.ExecuteAsync(
                "bash", $"-c \"readlink /sys/bus/pci/devices/{addr}/driver\"", ct);

            if (verifyResult.Success && verifyResult.StandardOutput.Contains("vfio-pci"))
            {
                _logger.LogInformation("VM {VmId}: GPU {PciAddr} successfully bound to vfio-pci", vmId, addr);
                return true;
            }

            _logger.LogWarning("VM {VmId}: GPU {PciAddr} driver binding could not be verified (driver: {Driver})",
                vmId, addr, verifyResult.StandardOutput.Trim());
            return true; // Proceed anyway — libvirt may still handle it
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "VM {VmId}: Exception during VFIO bind for GPU {PciAddr}", vmId, addr);
            return false;
        }
    }

    /// <summary>
    /// Unbind a PCI device from vfio-pci and clear driver_override so the
    /// original host driver (e.g. nvidia) can reclaim it.
    /// Called during VM deletion to return the GPU to the host.
    /// </summary>
    private async Task UnbindFromVfioAsync(string pciAddress, string vmId, CancellationToken ct)
    {
        var addr = ResourceDiscoveryService.NormalizePciAddress(pciAddress);
        _logger.LogInformation("VM {VmId}: Returning GPU {PciAddr} to host (unbinding from vfio-pci)", vmId, addr);

        try
        {
            // Step 1: Unbind from vfio-pci
            var unbindPath = $"/sys/bus/pci/devices/{addr}/driver";
            if (Directory.Exists(unbindPath))
            {
                await _executor.ExecuteAsync(
                    "bash", $"-c \"echo '{addr}' > /sys/bus/pci/devices/{addr}/driver/unbind\"", ct);
            }

            // Step 2: Clear driver_override so original driver can bind
            await _executor.ExecuteAsync(
                "bash", $"-c \"echo '' > /sys/bus/pci/devices/{addr}/driver_override\"", ct);

            // Step 3: Reprobe to let original driver reclaim
            await _executor.ExecuteAsync(
                "bash", $"-c \"echo '{addr}' > /sys/bus/pci/drivers_probe\"", ct);

            _logger.LogInformation("VM {VmId}: GPU {PciAddr} returned to host", vmId, addr);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "VM {VmId}: Failed to return GPU {PciAddr} to host. " +
                "Manual intervention may be needed (echo '' > /sys/bus/pci/devices/{PciAddr}/driver_override)",
                vmId, addr, addr);
        }
    }

    /// <summary>
    /// Generate libvirt XML for PCI hostdev passthrough of a GPU.
    /// </summary>
    private static string GenerateGpuPassthroughXml(string pciAddress)
    {
        var parsed = ParsePciAddress(pciAddress);
        if (parsed == null)
            return string.Empty;

        var (domain, bus, slot, function) = parsed.Value;

        return $@"
                <hostdev mode='subsystem' type='pci' managed='yes'>
                  <source>
                    <address domain='0x{domain:X4}' bus='0x{bus:X2}' slot='0x{slot:X2}' function='0x{function:X1}'/>
                  </source>
                </hostdev>";
    }

    private string GenerateLibvirtXml(VmSpec spec, string diskPath, string? cloudInitIso, int vncPort)
    {
        // ========================================
        // CPU SHARES CALCULATION
        // ========================================
        // Each compute point = 1000 shares (libvirt default unit)
        // This ensures VMs get CPU time proportional to their tier pricing

        // Formula: PointsPerVCpu = baselineOvercommit × (baselineOvercommit / tierOvercommit)

        // Defensive: Ensure node is fully initialized before creating VMs
        if (_nodeState.SchedulingConfig == null)
        {
            throw new InvalidOperationException(
                "Cannot create VM: Node has not received scheduling configuration. " +
                "Please ensure the node has successfully registered with the orchestrator.");
        }
        
        if (_nodeState.PerformanceEvaluation == null)
        {
            throw new InvalidOperationException(
                "Cannot create VM: Node has not completed performance evaluation. " +
                "Please ensure the node has successfully registered with the orchestrator.");
        }
        
        var config = _nodeState.SchedulingConfig;
        var nodeTotalPoints = _nodeState.PerformanceEvaluation.TotalComputePoints;

        // Defensive: Ensure tier exists in config, fallback to Standard if not
        if (!config.Tiers.TryGetValue(spec.QualityTier, out var tierConfig))
        {
            _logger.LogWarning(
                "VM {VmId}: Tier {Tier} not found in scheduling config, falling back to Standard tier",
                spec.Id, spec.QualityTier);
            tierConfig = config.Tiers[QualityTier.Standard];
        }

        var pointsPerVCpu = (tierConfig.MinimumBenchmark/config.BaselineBenchmark) *
                           (config.BaselineOvercommitRatio / tierConfig.CpuOvercommitRatio);

        spec.ComputePointCost = (int)(spec.VirtualCpuCores * pointsPerVCpu);
        int cpuShares;

        if (nodeTotalPoints > 0)
        {
            var shareRatio = (double)spec.ComputePointCost / nodeTotalPoints;
            cpuShares = (int)(shareRatio * 10000);
        }
        else
        {
            // Fallback: shouldn't happen, but defensive programming
            cpuShares = spec.ComputePointCost * 100;
        }

        // Apply bounds
        cpuShares = Math.Min(10000, cpuShares);  // libvirt cgroups v2 maximum
        cpuShares = Math.Max(100, cpuShares);    // Minimum 1% of max shares

        // ========================================
        // TIER-SPECIFIC CPU CONFIGURATION
        // ========================================
        var cpuTune = "";

        switch (spec.QualityTier)
        {
            case QualityTier.Guaranteed: // Guaranteed - Dedicated cores with high shares
                    // For Guaranteed tier, we use high CPU shares
                    // TODO: Implement CPU pinning when core allocation tracking is added
                cpuTune = $@"
                <cputune>
                  <shares>{cpuShares}</shares>
                </cputune>";
                break;

            case QualityTier.Standard: // Standard - Balanced shares
                cpuTune = $@"
                <cputune>
                  <shares>{cpuShares}</shares>
                </cputune>";
                break;

            case QualityTier.Balanced: // Balanced - Medium shares
                cpuTune = $@"
                <cputune>
                  <shares>{cpuShares}</shares>
                </cputune>";
                break;

            case QualityTier.Burstable: // Burstable - Low shares + Hard quota cap
                    // Burstable tier: Low shares (1000 per vCPU) + Hard CPU quota
                    // Quota limits burst capacity even when node is idle

                // Calculate quota: Allow ComputePointCost as % of total CPU time
                // Example: 4 points on 16-point node = 25% quota
                // With 4 vCPUs, that's 6.25% per vCPU

                // Quota is in microseconds per period (100ms = 100,000 microseconds)
                // quota = (ComputePointCost / TotalPoints) × period × 1000
                // For safety, cap at points × 12,500 microseconds (12.5% per point)
                // var quotaMicroseconds = spec.VCpus * 50000; // 12.5ms per point per 100ms
                // var periodMicroseconds = 100000; // 100ms period (stsandard)

                cpuTune = $@"
                <cputune>
                  <shares>{cpuShares}</shares>
                  <!-- Quota applied after 120s via monitoring service -->
                </cputune>";
                break;

            default:
                // Fallback to shares only
                cpuTune = $@"
                <cputune>
                  <shares>{cpuShares}</shares>
                </cputune>";
                break;
        }

        // ========================================
        // CLOUD-INIT ISO (if provided)
        // ========================================
        var cloudInitDisk = string.IsNullOrEmpty(cloudInitIso) ? "" : $@"
            <disk type='file' device='cdrom'>
              <driver name='qemu' type='raw'/>
              <source file='{cloudInitIso}'/>
              <target dev='sda' bus='sata'/>
              <readonly/>
            </disk>";

        // ========================================
        // BANDWIDTH QoS (libvirt rate limiting)
        // ========================================
        // Values are in KB/s. Burst is in KB.
        var bandwidthXml = spec.BandwidthTier switch
        {
            BandwidthTier.Basic =>       // 10 Mbps avg, 20 Mbps peak
                @"
                  <bandwidth>
                    <inbound average='1250' peak='2500' burst='1250'/>
                    <outbound average='1250' peak='2500' burst='1250'/>
                  </bandwidth>",
            BandwidthTier.Standard =>    // 50 Mbps avg, 100 Mbps peak
                @"
                  <bandwidth>
                    <inbound average='6250' peak='12500' burst='6250'/>
                    <outbound average='6250' peak='12500' burst='6250'/>
                  </bandwidth>",
            BandwidthTier.Performance => // 200 Mbps avg, 400 Mbps peak
                @"
                  <bandwidth>
                    <inbound average='25000' peak='50000' burst='25000'/>
                    <outbound average='25000' peak='50000' burst='25000'/>
                  </bandwidth>",
            _ => "" // Unmetered: no bandwidth element = no cap
        };

        // ========================================
        // GPU PASSTHROUGH (VFIO)
        // ========================================
        var gpuPassthroughXml = string.Empty;
        var hasGpuPassthrough = !string.IsNullOrEmpty(spec.GpuPciAddress);

        if (hasGpuPassthrough)
        {
            gpuPassthroughXml = GenerateGpuPassthroughXml(spec.GpuPciAddress!);
            if (string.IsNullOrEmpty(gpuPassthroughXml))
            {
                _logger.LogWarning(
                    "VM {VmId}: GPU PCI address '{PciAddr}' could not be parsed, skipping GPU passthrough",
                    spec.Id, spec.GpuPciAddress);
                hasGpuPassthrough = false;
            }
        }

        // IOMMU must be enabled in the <features> block for VFIO passthrough
        var iommuFeature = hasGpuPassthrough
            ? @"
                <ioapic driver='qemu'/>"
            : "";

        // ========================================
        // COMPLETE LIBVIRT XML
        // ========================================
        return $@"
            <domain type='kvm'>
              <name>{spec.Id}</name>
              <uuid>{spec.Id}</uuid>
              <memory unit='bytes'>{spec.MemoryBytes}</memory>
              <vcpu placement='static'>{spec.VirtualCpuCores}</vcpu>
              {cpuTune}
              <os>
                <type arch='x86_64' machine='q35'>hvm</type>
                <boot dev='hd'/>
              </os>
              <features>
                <acpi/>
                <apic/>{iommuFeature}
              </features>
              <cpu mode='host-passthrough' check='none'{(hasGpuPassthrough ? " migratable='off'" : "")}/>
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
                  <model type='virtio'/>{bandwidthXml}
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
                </rng>{gpuPassthroughXml}
                <channel type='unix'>
                  <source mode='bind' path='/var/lib/libvirt/qemu/channel/target/{spec.Id}.org.qemu.guest_agent.0'/>
                  <target type='virtio' name='org.qemu.guest_agent.0'/>
                </channel>
              </devices>
            </domain>";
    }

    /// <summary>
    /// Apply CPU quota cap to a running VM (for Burstable tier after boot)
    /// </summary>
    public async Task<bool> ApplyQuotaCapAsync(
        VmInstance vm,
        int quotaMicroseconds,
        int periodMicroseconds = 100000,
        CancellationToken ct = default)
    {
        var vmId = vm.VmId;
        if (vm.Spec.VmType == VmType.Relay)
        {
            _logger.LogInformation("VM {VmId} is Relay type - skipping quota application", vmId);
            return false;
        }
        try
        {
            _logger.LogInformation(
                "Applying quota cap to VM {VmId}: {Quota}µs per {Period}µs ({Percent}%)",
                vmId, quotaMicroseconds, periodMicroseconds,
                (quotaMicroseconds * 100.0 / periodMicroseconds));

            // Use virsh schedinfo to dynamically update CPU quota
            var result = await _executor.ExecuteAsync(
                "virsh",
                $"schedinfo {vmId} --set vcpu_quota={quotaMicroseconds} --set vcpu_period={periodMicroseconds}",
                ct);

            if (!result.Success)
            {
                _logger.LogError(
                    "Failed to apply quota to VM {VmId}: {Error}",
                    vmId, result.StandardError);
                return false;
            }

            _logger.LogInformation("Successfully applied quota cap to VM {VmId}", vmId);

            // Mark quota as applied in VM metadata
            if (_vms.TryGetValue(vmId, out var instance))
            {
                instance.Spec.VcpuQuotaMicroseconds = quotaMicroseconds;
                instance.Spec.VcpuPeriodMicroseconds = periodMicroseconds;
                instance.Spec.VcpuQuotaAppliedAt = DateTime.UtcNow;
                await _repository.SaveVmAsync(instance);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Exception applying quota to VM {VmId}", vmId);
            return false;
        }
    }

    /// <summary>
    /// Initialize with architecture detection
    /// </summary>
    private async Task InitializeArchitectureAsync(CancellationToken ct)
    {
        try
        {
            // Get architecture from node metadata
            var inventory = _nodeMetadata.Inventory;
            if (inventory?.Cpu != null && !string.IsNullOrEmpty(inventory.Cpu.Architecture))
            {
                _hostArchitecture = inventory.Cpu.Architecture;
                _logger.LogInformation("Detected host architecture: {Architecture}", _hostArchitecture);
            }
            else
            {
                _logger.LogWarning("Could not detect architecture from inventory, using default: x86_64");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to detect architecture, using default: x86_64");
        }
    }

    /// <summary>
    /// Generate architecture-aware libvirt XML
    /// </summary>
    private string GenerateLibvirtXmlMultiArch(VmSpec spec, string diskPath, string? cloudInitIso, int vncPort)
    {
        // Get architecture configuration
        var archConfig = ArchitectureHelper.GetHostArchConfig(_hostArchitecture);

        _logger.LogInformation(
            "Generating VM XML for {Architecture} architecture: emulator={Emulator}, machine={Machine}",
            archConfig.Architecture,
            archConfig.QemuEmulator,
            archConfig.MachineType);

        // ========================================
        // CPU SHARES CALCULATION
        // ========================================
        var config = _nodeState.SchedulingConfig;
        
        // Defensive: Ensure tier exists in config, fallback to Standard if not
        if (!config.Tiers.TryGetValue(spec.QualityTier, out var tierConfig))
        {
            _logger.LogWarning(
                "VM {VmId}: Tier {Tier} not found in scheduling config, falling back to Standard tier",
                spec.Id, spec.QualityTier);
            tierConfig = config.Tiers[QualityTier.Standard];
        }
        
        var pointsPerVCpu = config.BaselineOvercommitRatio *
                           (config.BaselineOvercommitRatio / tierConfig.CpuOvercommitRatio);

        spec.ComputePointCost = (int)(spec.VirtualCpuCores * pointsPerVCpu);
        var cpuShares = spec.ComputePointCost * 1000;
        cpuShares = Math.Min(262144, cpuShares);
        cpuShares = Math.Max(1000, cpuShares);

        // ========================================
        // TIER-SPECIFIC CPU CONFIGURATION
        // ========================================
        var cpuTune = spec.QualityTier switch
        {
            QualityTier.Guaranteed => $@"
              <cputune>
                <shares>{cpuShares}</shares>
                <period>100000</period>
                <quota>-1</quota>
              </cputune>",

            QualityTier.Standard => $@"
              <cputune>
                <shares>{cpuShares}</shares>
              </cputune>",

            _ => $@"
              <cputune>
                <shares>{cpuShares}</shares>
              </cputune>"
        };

        // ========================================
        // ARCHITECTURE-SPECIFIC CONFIGURATION
        // ========================================

        // ARM64-specific: UEFI firmware for proper boot
        var osSection = archConfig.Architecture == "aarch64"
            ? $@"
              <os>
                <type arch='aarch64' machine='{archConfig.MachineType}'>hvm</type>
                <loader readonly='yes' type='pflash'>/usr/share/AAVMF/AAVMF_CODE.fd</loader>
                <boot dev='hd'/>
              </os>"
            : $@"
              <os>
                <type arch='x86_64' machine='{archConfig.MachineType}'>hvm</type>
                <boot dev='hd'/>
              </os>";

        // ARM64 uses virtio for optimal performance
        var diskBus = archConfig.Architecture == "aarch64" ? "virtio" : "virtio";

        // Cloud-init disk (optional)
        var cloudInitDisk = !string.IsNullOrEmpty(cloudInitIso)
            ? $@"
            <disk type='file' device='cdrom'>
              <driver name='qemu' type='raw'/>
              <source file='{cloudInitIso}'/>
              <target dev='sda' bus='sata'/>
              <readonly/>
            </disk>"
            : "";

        // ========================================
        // BANDWIDTH QoS (libvirt rate limiting)
        // ========================================
        var bandwidthXml = spec.BandwidthTier switch
        {
            BandwidthTier.Basic =>
                @"
                  <bandwidth>
                    <inbound average='1250' peak='2500' burst='1250'/>
                    <outbound average='1250' peak='2500' burst='1250'/>
                  </bandwidth>",
            BandwidthTier.Standard =>
                @"
                  <bandwidth>
                    <inbound average='6250' peak='12500' burst='6250'/>
                    <outbound average='6250' peak='12500' burst='6250'/>
                  </bandwidth>",
            BandwidthTier.Performance =>
                @"
                  <bandwidth>
                    <inbound average='25000' peak='50000' burst='25000'/>
                    <outbound average='25000' peak='50000' burst='25000'/>
                  </bandwidth>",
            _ => ""
        };

        // ========================================
        // GPU PASSTHROUGH (VFIO)
        // ========================================
        var gpuPassthroughXml = string.Empty;
        var hasGpuPassthrough = !string.IsNullOrEmpty(spec.GpuPciAddress);

        if (hasGpuPassthrough)
        {
            gpuPassthroughXml = GenerateGpuPassthroughXml(spec.GpuPciAddress!);
            if (string.IsNullOrEmpty(gpuPassthroughXml))
            {
                _logger.LogWarning(
                    "VM {VmId}: GPU PCI address '{PciAddr}' could not be parsed, skipping GPU passthrough",
                    spec.Id, spec.GpuPciAddress);
                hasGpuPassthrough = false;
            }
        }

        var iommuFeature = hasGpuPassthrough
            ? @"
                <ioapic driver='qemu'/>"
            : "";

        // ========================================
        // COMPLETE LIBVIRT XML
        // ========================================
        return $@"
            <domain type='kvm'>
              <name>{spec.Id}</name>
              <uuid>{spec.Id}</uuid>
              <memory unit='bytes'>{spec.MemoryBytes}</memory>
              <vcpu placement='static'>{spec.VirtualCpuCores}</vcpu>
              {cpuTune}
              {osSection}
              <features>
                <acpi/>
                <apic/>{iommuFeature}
              </features>
              <cpu mode='host-passthrough' check='none'{(hasGpuPassthrough ? " migratable='off'" : "")}/>
              <clock offset='utc'>
                <timer name='rtc' tickpolicy='catchup'/>
                <timer name='pit' tickpolicy='delay'/>
                <timer name='hpet' present='no'/>
              </clock>
              <on_poweroff>destroy</on_poweroff>
              <on_reboot>restart</on_reboot>
              <on_crash>destroy</on_crash>
              <devices>
                <emulator>{archConfig.QemuEmulator}</emulator>
                <disk type='file' device='disk'>
                  <driver name='qemu' type='qcow2'/>
                  <source file='{diskPath}'/>
                  <target dev='vda' bus='{diskBus}'/>
                  <address type='pci' domain='0x0000' bus='0x00' slot='0x04' function='0x0'/>
                </disk>
                {cloudInitDisk}
                <interface type='network'>
                  <source network='default'/>
                  <model type='virtio'/>{bandwidthXml}
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
                  <model type='virtio' heads='1'/>
                </video>
                <rng model='virtio'>
                  <backend model='random'>/dev/urandom</backend>
                </rng>{gpuPassthroughXml}
                <channel type='unix'>
                  <source mode='bind' path='/var/lib/libvirt/qemu/channel/target/{spec.Id}.org.qemu.guest_agent.0'/>
                  <target type='virtio' name='org.qemu.guest_agent.0'/>
                </channel>
              </devices>
            </domain>";
    }
}