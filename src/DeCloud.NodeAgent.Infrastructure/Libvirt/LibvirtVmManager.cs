using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Infrastructure.Libvirt;

public class LibvirtVmManagerOptions
{
    public string VmStoragePath { get; set; } = "/var/lib/decloud/vms";
    public string ImageCachePath { get; set; } = "/var/lib/decloud/images";
    public string LibvirtUri { get; set; } = "qemu:///system";
    public int VncPortStart { get; set; } = 5900;
}

public class LibvirtVmManager : IVmManager
{
    private readonly ICommandExecutor _executor;
    private readonly IImageManager _imageManager;
    private readonly ILogger<LibvirtVmManager> _logger;
    private readonly LibvirtVmManagerOptions _options;
    private readonly bool _isWindows;
    
    // Track our VMs in memory (in production, persist to local DB)
    private readonly Dictionary<string, VmInstance> _vms = new();
    private readonly SemaphoreSlim _lock = new(1, 1);
    private int _nextVncPort;

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

    public async Task<VmOperationResult> CreateVmAsync(VmSpec spec, CancellationToken ct = default)
    {
        if (_isWindows)
        {
            _logger.LogWarning("VM creation not supported on Windows - requires Linux with KVM/libvirt");
            return VmOperationResult.Fail(spec.VmId, 
                "VM creation requires Linux with KVM/libvirt. Windows detected.", "PLATFORM_NOT_SUPPORTED");
        }
        await _lock.WaitAsync(ct);
        try
        {
            _logger.LogInformation("Creating VM {VmId} with {VCpus} vCPUs, {MemoryMB}MB RAM",
                spec.VmId, spec.VCpus, spec.MemoryBytes / 1024 / 1024);

            // Create VM instance record
            var instance = new VmInstance
            {
                VmId = spec.VmId,
                Name = spec.Name,
                Spec = spec,
                State = VmState.Creating,
                CreatedAt = DateTime.UtcNow
            };

            _vms[spec.VmId] = instance;

            // 1. Ensure base image is available
            string baseImagePath;
            try
            {
                baseImagePath = await _imageManager.EnsureImageAvailableAsync(
                    spec.BaseImageUrl, spec.BaseImageHash, ct);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to download base image for VM {VmId}", spec.VmId);
                instance.State = VmState.Failed;
                return VmOperationResult.Fail(spec.VmId, $"Image download failed: {ex.Message}", "IMAGE_DOWNLOAD_FAILED");
            }

            // 2. Create overlay disk
            var vmDir = Path.Combine(_options.VmStoragePath, spec.VmId);
            Directory.CreateDirectory(vmDir);
            
            instance.DiskPath = await _imageManager.CreateOverlayDiskAsync(
                baseImagePath, spec.VmId, spec.DiskBytes, ct);

            // 3. Generate cloud-init ISO if needed
            string? cloudInitIso = null;
            if (!string.IsNullOrEmpty(spec.CloudInitUserData) || !string.IsNullOrEmpty(spec.SshPublicKey))
            {
                cloudInitIso = await CreateCloudInitIsoAsync(spec, vmDir, ct);
            }

            // 4. Generate libvirt XML
            var vncPort = Interlocked.Increment(ref _nextVncPort);
            instance.VncPort = vncPort.ToString();
            
            var xml = GenerateLibvirtXml(spec, instance.DiskPath, cloudInitIso, vncPort);
            instance.ConfigPath = Path.Combine(vmDir, "domain.xml");
            await File.WriteAllTextAsync(instance.ConfigPath, xml, ct);

            // 5. Define domain with virsh
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

    public async Task<VmOperationResult> StartVmAsync(string vmId, CancellationToken ct = default)
    {
        if (!_vms.TryGetValue(vmId, out var instance))
            return VmOperationResult.Fail(vmId, "VM not found", "NOT_FOUND");

        _logger.LogInformation("Starting VM {VmId}", vmId);

        var result = await _executor.ExecuteAsync("virsh", $"start {vmId}", ct);

        if (result.Success)
        {
            instance.State = VmState.Running;
            instance.StartedAt = DateTime.UtcNow;
            
            // Get PID
            var pidResult = await _executor.ExecuteAsync("virsh", $"qemu-monitor-command {vmId} --hmp info status", ct);
            
            return VmOperationResult.Ok(vmId, VmState.Running);
        }

        _logger.LogError("Failed to start VM {VmId}: {Error}", vmId, result.StandardError);
        return VmOperationResult.Fail(vmId, result.StandardError, "START_FAILED");
    }

    public async Task<VmOperationResult> StopVmAsync(string vmId, bool force = false, CancellationToken ct = default)
    {
        if (!_vms.TryGetValue(vmId, out var instance))
            return VmOperationResult.Fail(vmId, "VM not found", "NOT_FOUND");

        _logger.LogInformation("Stopping VM {VmId} (force={Force})", vmId, force);

        instance.State = VmState.Stopping;

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
                    var state = await GetVmStateAsync(vmId, ct);
                    if (state == VmState.Stopped) break;
                }
            }
            
            instance.State = VmState.Stopped;
            instance.StoppedAt = DateTime.UtcNow;
            return VmOperationResult.Ok(vmId, VmState.Stopped);
        }

        _logger.LogError("Failed to stop VM {VmId}: {Error}", vmId, result.StandardError);
        return VmOperationResult.Fail(vmId, result.StandardError, "STOP_FAILED");
    }

    public async Task<VmOperationResult> DeleteVmAsync(string vmId, CancellationToken ct = default)
    {
        _logger.LogInformation("Deleting VM {VmId}", vmId);

        // Stop if running
        var state = await GetVmStateAsync(vmId, ct);
        if (state == VmState.Running)
        {
            await StopVmAsync(vmId, force: true, ct);
        }

        // Undefine domain
        var undefResult = await _executor.ExecuteAsync("virsh", 
            $"undefine {vmId} --remove-all-storage", ct);

        if (!undefResult.Success && !undefResult.StandardError.Contains("not found"))
        {
            return VmOperationResult.Fail(vmId, undefResult.StandardError, "UNDEFINE_FAILED");
        }

        // Clean up our directory
        var vmDir = Path.Combine(_options.VmStoragePath, vmId);
        if (Directory.Exists(vmDir))
        {
            Directory.Delete(vmDir, recursive: true);
        }

        _vms.Remove(vmId);
        _logger.LogInformation("VM {VmId} deleted", vmId);
        
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
        _vms.TryGetValue(vmId, out var instance);
        return Task.FromResult(instance);
    }

    public Task<List<VmInstance>> GetAllVmsAsync(CancellationToken ct = default)
    {
        return Task.FromResult(_vms.Values.ToList());
    }

    public async Task<VmResourceUsage> GetVmUsageAsync(string vmId, CancellationToken ct = default)
    {
        var usage = new VmResourceUsage { MeasuredAt = DateTime.UtcNow };

        // Get CPU stats
        var cpuResult = await _executor.ExecuteAsync("virsh", $"cpu-stats {vmId} --total", ct);
        if (cpuResult.Success)
        {
            var match = Regex.Match(cpuResult.StandardOutput, @"cpu_time\s+([\d.]+)");
            // CPU time is in nanoseconds - would need to track delta for %
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
                if (line.Contains("rd_bytes") && long.TryParse(line.Split(' ').Last(), out var rb))
                    usage.DiskReadBytes = rb;
                if (line.Contains("wr_bytes") && long.TryParse(line.Split(' ').Last(), out var wb))
                    usage.DiskWriteBytes = wb;
            }
        }

        return usage;
    }

    public async Task<bool> VmExistsAsync(string vmId, CancellationToken ct = default)
    {
        var result = await _executor.ExecuteAsync("virsh", $"dominfo {vmId}", ct);
        return result.Success;
    }

    private async Task<VmState> GetVmStateAsync(string vmId, CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync("virsh", $"domstate {vmId}", ct);
        if (!result.Success) return VmState.Stopped;

        return result.StandardOutput.Trim().ToLower() switch
        {
            "running" => VmState.Running,
            "paused" => VmState.Paused,
            "shut off" => VmState.Stopped,
            "crashed" => VmState.Failed,
            _ => VmState.Stopped
        };
    }

    private string GenerateLibvirtXml(VmSpec spec, string diskPath, string? cloudInitIso, int vncPort)
    {
        var xml = new XDocument(
            new XElement("domain", new XAttribute("type", "kvm"),
                new XElement("name", spec.VmId),
                new XElement("uuid", Guid.NewGuid().ToString()),
                new XElement("memory", new XAttribute("unit", "bytes"), spec.MemoryBytes),
                new XElement("vcpu", spec.VCpus),
                new XElement("os",
                    new XElement("type", new XAttribute("arch", "x86_64"), "hvm"),
                    new XElement("boot", new XAttribute("dev", "hd"))
                ),
                new XElement("features",
                    new XElement("acpi"),
                    new XElement("apic")
                ),
                new XElement("cpu", new XAttribute("mode", "host-passthrough")),
                new XElement("clock", new XAttribute("offset", "utc")),
                new XElement("devices",
                    new XElement("emulator", "/usr/bin/qemu-system-x86_64"),
                    // Main disk
                    new XElement("disk", new XAttribute("type", "file"), new XAttribute("device", "disk"),
                        new XElement("driver", new XAttribute("name", "qemu"), new XAttribute("type", "qcow2")),
                        new XElement("source", new XAttribute("file", diskPath)),
                        new XElement("target", new XAttribute("dev", "vda"), new XAttribute("bus", "virtio"))
                    ),
                    // Network
                    new XElement("interface", new XAttribute("type", "bridge"),
                        new XElement("source", new XAttribute("bridge", "virbr0")),
                        new XElement("model", new XAttribute("type", "virtio")),
                        string.IsNullOrEmpty(spec.Network.MacAddress) ? null :
                            new XElement("mac", new XAttribute("address", spec.Network.MacAddress))
                    ),
                    // VNC for console access
                    new XElement("graphics", 
                        new XAttribute("type", "vnc"),
                        new XAttribute("port", vncPort),
                        new XAttribute("listen", "127.0.0.1")
                    ),
                    // Serial console
                    new XElement("serial", new XAttribute("type", "pty"),
                        new XElement("target", new XAttribute("port", "0"))
                    ),
                    new XElement("console", new XAttribute("type", "pty"),
                        new XElement("target", new XAttribute("type", "serial"), new XAttribute("port", "0"))
                    )
                )
            )
        );

        // Add cloud-init ISO if present
        if (!string.IsNullOrEmpty(cloudInitIso))
        {
            var devices = xml.Root!.Element("devices")!;
            devices.Add(
                new XElement("disk", new XAttribute("type", "file"), new XAttribute("device", "cdrom"),
                    new XElement("driver", new XAttribute("name", "qemu"), new XAttribute("type", "raw")),
                    new XElement("source", new XAttribute("file", cloudInitIso)),
                    new XElement("target", new XAttribute("dev", "sda"), new XAttribute("bus", "sata")),
                    new XElement("readonly")
                )
            );
        }

        // Add GPU passthrough if specified
        if (!string.IsNullOrEmpty(spec.GpuPciAddress))
        {
            var devices = xml.Root!.Element("devices")!;
            // Parse PCI address like "0000:01:00.0"
            var parts = spec.GpuPciAddress.Split(':');
            if (parts.Length >= 2)
            {
                var bus = parts[1].Split('.')[0];
                var slot = parts.Length > 2 ? parts[2].Split('.')[0] : "00";
                var function = spec.GpuPciAddress.Split('.').LastOrDefault() ?? "0";

                devices.Add(
                    new XElement("hostdev", 
                        new XAttribute("mode", "subsystem"),
                        new XAttribute("type", "pci"),
                        new XAttribute("managed", "yes"),
                        new XElement("source",
                            new XElement("address",
                                new XAttribute("domain", "0x0000"),
                                new XAttribute("bus", $"0x{bus}"),
                                new XAttribute("slot", $"0x{slot}"),
                                new XAttribute("function", $"0x{function}")
                            )
                        )
                    )
                );
            }
        }

        return xml.ToString();
    }

    private async Task<string> CreateCloudInitIsoAsync(VmSpec spec, string vmDir, CancellationToken ct)
    {
        var ciDir = Path.Combine(vmDir, "cloud-init");
        Directory.CreateDirectory(ciDir);

        // Create meta-data
        var metaData = $"instance-id: {spec.VmId}\nlocal-hostname: {spec.Name}";
        await File.WriteAllTextAsync(Path.Combine(ciDir, "meta-data"), metaData, ct);

        // Create user-data
        var userData = new StringBuilder();
        userData.AppendLine("#cloud-config");
        
        if (!string.IsNullOrEmpty(spec.SshPublicKey))
        {
            userData.AppendLine("ssh_authorized_keys:");
            userData.AppendLine($"  - {spec.SshPublicKey}");
        }

        if (!string.IsNullOrEmpty(spec.CloudInitUserData))
        {
            // Append custom user data
            userData.AppendLine(spec.CloudInitUserData);
        }

        await File.WriteAllTextAsync(Path.Combine(ciDir, "user-data"), userData.ToString(), ct);

        // Create ISO
        var isoPath = Path.Combine(vmDir, "cloud-init.iso");
        var mkisofsResult = await _executor.ExecuteAsync("genisoimage",
            $"-output {isoPath} -volid cidata -joliet -rock {Path.Combine(ciDir, "user-data")} {Path.Combine(ciDir, "meta-data")}",
            ct);

        if (!mkisofsResult.Success)
        {
            // Try cloud-localds as alternative
            var cloudLocaldsResult = await _executor.ExecuteAsync("cloud-localds",
                $"{isoPath} {Path.Combine(ciDir, "user-data")} {Path.Combine(ciDir, "meta-data")}",
                ct);
            
            if (!cloudLocaldsResult.Success)
            {
                throw new Exception($"Failed to create cloud-init ISO: {cloudLocaldsResult.StandardError}");
            }
        }

        return isoPath;
    }
}
