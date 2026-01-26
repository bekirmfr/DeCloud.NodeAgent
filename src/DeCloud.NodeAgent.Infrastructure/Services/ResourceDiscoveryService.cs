using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

public class ResourceDiscoveryService : IResourceDiscoveryService
{
    private readonly ICommandExecutor _executor;
    private readonly INodeStateService _nodeState;
    private readonly ILogger<ResourceDiscoveryService> _logger;
    private readonly ICpuBenchmarkService _benchmarkService;
    private readonly bool _isWindows;

    // Caching fields
    private HardwareInventory? _cachedInventory;
    private readonly SemaphoreSlim _discoverySemaphore = new(1, 1);
    private DateTime _lastDiscoveryTime = DateTime.MinValue;
    private static readonly TimeSpan DiscoveryCacheDuration = TimeSpan.FromHours(1);

    public ResourceDiscoveryService(
        ICommandExecutor executor,
        INodeStateService nodeState,
        ILogger<ResourceDiscoveryService> logger,
        ICpuBenchmarkService benchmarkService)
    {
        _executor = executor;
        _nodeState = nodeState;
        _logger = logger;
        _benchmarkService = benchmarkService;
        _isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
    }

    /// <summary>
    /// Get cached hardware inventory
    /// Returns null if discovery hasn't completed yet
    /// </summary>
    public async Task<HardwareInventory?> GetInventoryCachedAsync(CancellationToken ct = default)
    {
        try
        {
            if (_cachedInventory == null || DateTime.UtcNow - _lastDiscoveryTime > DiscoveryCacheDuration)
            {
                _logger.LogInformation("Cached inventory is stale or not available");
                await DiscoverAllAsync(ct);
            }

            _nodeState.SetDiscoveryComplete();
            return _cachedInventory;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Resource discovery failed");
            return null;
        }
    }

    /// <summary>
    /// Perform full hardware discovery with CPU benchmark
    /// This operation takes 15-30 seconds and caches the result
    /// </summary>
    public async Task<HardwareInventory> DiscoverAllAsync(CancellationToken ct = default)
    {
        // Prevent concurrent discoveries
        await _discoverySemaphore.WaitAsync(ct);
        try
        {
            _logger.LogInformation("Starting full resource discovery (Platform: {Platform})",
                _isWindows ? "Windows" : "Linux");

            var cpu = await GetCpuInfoAsync(ct);
            var memory = await GetMemoryInfoAsync(ct);
            var storage = await GetStorageInfoAsync(ct);
            var gpus = await GetGpuInfoAsync(ct);
            var supportsGpu = gpus.Any();
            var network = await GetNetworkInfoAsync(ct);

            var inventory = new HardwareInventory
            {
                Cpu = cpu,
                Memory = memory,
                Storage = storage,
                SupportsGpu = supportsGpu,
                Gpus = gpus,
                Network = network,
                CollectedAt = DateTime.UtcNow
            };

            // Update cache
            _cachedInventory = inventory;
            _lastDiscoveryTime = DateTime.UtcNow;

            _logger.LogInformation(
                "✓ Discovery complete: {Cores} cores, {Memory}GB RAM, {Storage}GB disk, {Gpus} GPUs, Benchmark: {Score}",
                inventory.Cpu.LogicalCores,
                inventory.Memory.TotalBytes / 1024 / 1024 / 1024,
                inventory.Storage.Sum(s => s.TotalBytes) / 1024 / 1024 / 1024,
                inventory.Gpus.Count,
                inventory.Cpu.BenchmarkScore);

            return inventory;
        }
        finally
        {
            _discoverySemaphore.Release();
        }
    }

    public async Task<CpuInfo> GetCpuInfoAsync(CancellationToken ct = default)
    {
        return _isWindows
            ? await GetCpuInfoWindowsAsync(ct)
            : await GetCpuInfoLinuxAsync(ct);
    }

    /// <summary>
    /// Enhanced Windows CPU detection with architecture
    /// </summary>
    private async Task<CpuInfo> GetCpuInfoWindowsAsync(CancellationToken ct)
    {
        var info = new CpuInfo { Flags = new List<string>() };

        // Detect architecture
        info.Architecture = DetectArchitecture();
        _logger.LogInformation("Detected CPU architecture: {Architecture}", info.Architecture);

        try
        {
            var result = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, VirtualizationFirmwareEnabled | ConvertTo-Json\"",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                var json = result.StandardOutput.Trim();
                info.Model = ExtractJsonValue(json, "Name") ?? "Unknown";
                info.PhysicalCores = int.TryParse(ExtractJsonValue(json, "NumberOfCores"), out var cores) ? cores : Environment.ProcessorCount;
                info.LogicalCores = int.TryParse(ExtractJsonValue(json, "NumberOfLogicalProcessors"), out var logical) ? logical : Environment.ProcessorCount;
                info.FrequencyMhz = double.TryParse(ExtractJsonValue(json, "MaxClockSpeed"), out var freq) ? freq : 0;
                info.SupportsVirtualization = ExtractJsonValue(json, "VirtualizationFirmwareEnabled")?.ToLower() == "true";
            }
            else
            {
                // Fallback
                info.LogicalCores = Environment.ProcessorCount;
                info.PhysicalCores = Environment.ProcessorCount;
            }

            // Get current CPU usage
            var usageResult = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"(Get-CimInstance Win32_Processor).LoadPercentage\"",
                ct);

            if (usageResult.Success && double.TryParse(usageResult.StandardOutput.Trim(), out var usage))
            {
                info.UsagePercent = usage;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to get Windows CPU info, using fallback");
            info.LogicalCores = Environment.ProcessorCount;
            info.PhysicalCores = Environment.ProcessorCount;
        }

        info.AvailableVCpus = Math.Max(1, info.LogicalCores);

        try
        {
            _logger.LogInformation("Running CPU benchmark for node performance evaluation...");
            var benchmarkResult = await _benchmarkService.RunBenchmarkAsync(ct);
            info.BenchmarkScore = benchmarkResult.Score;

            _logger.LogInformation(
                "CPU Benchmark: {Score} score via {Method} ({Details})",
                benchmarkResult.Score,
                benchmarkResult.Method,
                benchmarkResult.Details);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "CPU benchmark failed, using default score of 1000");
            info.BenchmarkScore = 1000; // Fallback to baseline
        }

        return info;
    }

    /// <summary>
    /// Enhanced Linux CPU detection with architecture
    /// </summary>
    private async Task<CpuInfo> GetCpuInfoLinuxAsync(CancellationToken ct)
    {
        var info = new CpuInfo { Flags = new List<string>() };

        // Detect architecture first
        info.Architecture = DetectArchitecture();
        _logger.LogInformation("Detected CPU architecture: {Architecture}", info.Architecture);

        var cpuinfoResult = await _executor.ExecuteAsync("cat", "/proc/cpuinfo", ct);
        if (cpuinfoResult.Success)
        {
            foreach (var line in cpuinfoResult.StandardOutput.Split('\n'))
            {
                if (line.StartsWith("model name"))
                {
                    info.Model = line.Split(':').LastOrDefault()?.Trim() ?? "";
                }
                else if (line.StartsWith("cpu MHz") && info.FrequencyMhz == 0)
                {
                    double.TryParse(line.Split(':').LastOrDefault()?.Trim(), out var mhz);
                    info.FrequencyMhz = mhz;
                }
                else if (line.StartsWith("flags"))
                {
                    info.Flags = line.Split(':').LastOrDefault()?.Trim().Split(' ').ToList() ?? new();
                    info.SupportsVirtualization = info.Flags.Contains("vmx") || info.Flags.Contains("svm");
                }
                // ARM-specific CPU info
                else if (line.StartsWith("CPU implementer") || line.StartsWith("CPU architecture"))
                {
                    // ARM processors show model differently
                    if (string.IsNullOrEmpty(info.Model))
                    {
                        info.Model = line.Split(':').LastOrDefault()?.Trim() ?? "";
                    }
                }
            }
        }

        var lscpuResult = await _executor.ExecuteAsync("lscpu", "", ct);
        if (lscpuResult.Success)
        {
            foreach (var line in lscpuResult.StandardOutput.Split('\n'))
            {
                if (line.StartsWith("CPU(s):"))
                {
                    if (int.TryParse(line.Split(':').LastOrDefault()?.Trim(), out var cores))
                    {
                        info.LogicalCores = cores;
                    }
                }
                else if (line.StartsWith("Core(s) per socket:"))
                {
                    if (int.TryParse(line.Split(':').LastOrDefault()?.Trim(), out var coresPerSocket))
                    {
                        info.PhysicalCores = coresPerSocket;
                    }
                }
                else if (line.StartsWith("Socket(s):"))
                {
                    if (int.TryParse(line.Split(':').LastOrDefault()?.Trim(), out var sockets))
                    {
                        if (info.PhysicalCores > 0)
                        {
                            info.PhysicalCores *= sockets;
                        }
                    }
                }
                else if (line.StartsWith("Architecture:") && string.IsNullOrEmpty(info.Architecture))
                {
                    // Fallback if runtime detection failed
                    info.Architecture = line.Split(':').LastOrDefault()?.Trim() ?? "unknown";
                }
            }
        }

        // Fallback only if lscpu completely failed
        if (info.LogicalCores == 0) info.LogicalCores = Environment.ProcessorCount;
        if (info.PhysicalCores == 0) info.PhysicalCores = info.LogicalCores;
        info.AvailableVCpus = info.LogicalCores;

        try
        {
            _logger.LogInformation("Running CPU benchmark for node performance evaluation...");
            var benchmarkResult = await _benchmarkService.RunBenchmarkAsync(ct);
            info.BenchmarkScore = benchmarkResult.Score;

            _logger.LogInformation(
                "CPU Benchmark: {Score} score via {Method} ({Details})",
                benchmarkResult.Score,
                benchmarkResult.Method,
                benchmarkResult.Details);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "CPU benchmark failed, using default score of 1000");
            info.BenchmarkScore = 1000; // Fallback to baseline
        }

        return info;
    }

    public async Task<MemoryInfo> GetMemoryInfoAsync(CancellationToken ct = default)
    {
        return _isWindows
            ? await GetMemoryInfoWindowsAsync(ct)
            : await GetMemoryInfoLinuxAsync(ct);
    }

    private async Task<MemoryInfo> GetMemoryInfoWindowsAsync(CancellationToken ct)
    {
        var info = new MemoryInfo { ReservedBytes = 2L * 1024 * 1024 * 1024 };

        try
        {
            var result = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | ConvertTo-Json\"",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                var totalKb = long.TryParse(ExtractJsonValue(result.StandardOutput, "TotalVisibleMemorySize"), out var t) ? t : 0;
                var freeKb = long.TryParse(ExtractJsonValue(result.StandardOutput, "FreePhysicalMemory"), out var f) ? f : 0;

                info.TotalBytes = totalKb * 1024;
                info.AvailableBytes = freeKb * 1024;
                info.UsedBytes = info.TotalBytes - info.AvailableBytes;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to get Windows memory info");
        }

        return info;
    }

    private async Task<MemoryInfo> GetMemoryInfoLinuxAsync(CancellationToken ct)
    {
        var info = new MemoryInfo { ReservedBytes = 2L * 1024 * 1024 * 1024 };

        var result = await _executor.ExecuteAsync("cat", "/proc/meminfo", ct);
        if (result.Success)
        {
            foreach (var line in result.StandardOutput.Split('\n'))
            {
                var parts = line.Split(':', StringSplitOptions.TrimEntries);
                if (parts.Length < 2) continue;

                var value = ParseMemoryValue(parts[1]);
                if (parts[0] == "MemTotal") info.TotalBytes = value;
                else if (parts[0] == "MemAvailable") info.AvailableBytes = value;
            }
            info.UsedBytes = info.TotalBytes - info.AvailableBytes;
        }

        return info;
    }

    public async Task<List<StorageInfo>> GetStorageInfoAsync(CancellationToken ct = default)
    {
        return _isWindows
            ? await GetStorageInfoWindowsAsync(ct)
            : await GetStorageInfoLinuxAsync(ct);
    }

    private async Task<List<StorageInfo>> GetStorageInfoWindowsAsync(CancellationToken ct)
    {
        var storageList = new List<StorageInfo>();

        try
        {
            var result = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID, Size, FreeSpace, FileSystem | ConvertTo-Json\"",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                var json = result.StandardOutput.Trim();
                var items = json.TrimStart().StartsWith("[") ? SplitJsonArray(json) : new List<string> { json };

                foreach (var item in items)
                {
                    var deviceId = ExtractJsonValue(item, "DeviceID");
                    if (string.IsNullOrEmpty(deviceId)) continue;

                    var size = long.TryParse(ExtractJsonValue(item, "Size"), out var s) ? s : 0;
                    var free = long.TryParse(ExtractJsonValue(item, "FreeSpace"), out var f) ? f : 0;

                    storageList.Add(new StorageInfo
                    {
                        DevicePath = deviceId,
                        MountPoint = deviceId + "\\",
                        FileSystem = ExtractJsonValue(item, "FileSystem") ?? "NTFS",
                        TotalBytes = size,
                        AvailableBytes = free,
                        UsedBytes = size - free,
                        Type = StorageType.SSD
                    });
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to get Windows storage info");
        }

        return storageList;
    }

    private async Task<List<StorageInfo>> GetStorageInfoLinuxAsync(CancellationToken ct)
    {
        var storageList = new List<StorageInfo>();

        var dfResult = await _executor.ExecuteAsync("df", "-B1 --output=source,target,fstype,size,used,avail", ct);
        if (dfResult.Success)
        {
            foreach (var line in dfResult.StandardOutput.Split('\n').Skip(1))
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 6 || !parts[0].StartsWith("/dev/")) continue;

                storageList.Add(new StorageInfo
                {
                    DevicePath = parts[0],
                    MountPoint = parts[1],
                    FileSystem = parts[2],
                    TotalBytes = long.TryParse(parts[3], out var total) ? total : 0,
                    UsedBytes = long.TryParse(parts[4], out var used) ? used : 0,
                    AvailableBytes = long.TryParse(parts[5], out var avail) ? avail : 0,
                    Type = parts[0].Contains("nvme") ? StorageType.NVMe : StorageType.SSD
                });
            }
        }

        return storageList;
    }

    public async Task<List<GpuInfo>> GetGpuInfoAsync(CancellationToken ct = default)
    {
        var gpus = new List<GpuInfo>();

        // Quick check: does nvidia-smi exist?
        if (!File.Exists("/usr/bin/nvidia-smi") &&
            !File.Exists("/usr/local/bin/nvidia-smi"))
        {
            _logger.LogDebug("nvidia-smi not found - no GPU detected");
            return gpus; // Return empty list immediately
        }

        try
        {
            // Try nvidia-smi first (works on both Windows and Linux)
            var nvidiaSmi = await _executor.ExecuteAsync("nvidia-smi",
            "--query-gpu=name,pci.bus_id,memory.total,memory.used,driver_version,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits",
            ct);

            if (!nvidiaSmi.Success)
            {
                _logger.LogDebug("No NVIDIA GPUs detected");
                return gpus;
            }

            if (nvidiaSmi.Success && !string.IsNullOrWhiteSpace(nvidiaSmi.StandardOutput))
            {
                foreach (var line in nvidiaSmi.StandardOutput.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                {
                    var parts = line.Split(',').Select(p => p.Trim()).ToArray();
                    if (parts.Length < 8) continue;

                    gpus.Add(new GpuInfo
                    {
                        Vendor = "NVIDIA",
                        Model = parts[0],
                        PciAddress = parts[1],
                        MemoryBytes = long.TryParse(parts[2], out var mem) ? mem * 1024 * 1024 : 0,
                        MemoryUsedBytes = long.TryParse(parts[3], out var used) ? used * 1024 * 1024 : 0,
                        DriverVersion = parts[4],
                        GpuUsagePercent = double.TryParse(parts[5], out var gpuUsage) ? gpuUsage : 0,
                        MemoryUsagePercent = double.TryParse(parts[6], out var memUsage) ? memUsage : 0,
                        TemperatureCelsius = int.TryParse(parts[7], out var temp) ? temp : null,
                        IsAvailableForPassthrough = !_isWindows
                    });
                }
            }

            // Fallback for non-NVIDIA or if nvidia-smi failed
            if (gpus.Count == 0 && _isWindows)
            {
                var wmicResult = await _executor.ExecuteAsync("powershell",
                    "-NoProfile -Command \"Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json\"",
                    ct);

                if (wmicResult.Success && !string.IsNullOrWhiteSpace(wmicResult.StandardOutput))
                {
                    var name = ExtractJsonValue(wmicResult.StandardOutput, "Name") ?? "Unknown GPU";
                    var vendor = name.Contains("NVIDIA") ? "NVIDIA" : name.Contains("AMD") ? "AMD" : "Intel";

                    gpus.Add(new GpuInfo
                    {
                        Vendor = vendor,
                        Model = name,
                        DriverVersion = ExtractJsonValue(wmicResult.StandardOutput, "DriverVersion") ?? "",
                        MemoryBytes = long.TryParse(ExtractJsonValue(wmicResult.StandardOutput, "AdapterRAM"), out var ram) ? ram : 0
                    });
                }
            }

            return gpus;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "GPU detection failed (this is normal if no GPU installed)");
            return gpus;
        }
    }

    public async Task<NetworkInfo> GetNetworkInfoAsync(CancellationToken ct = default)
    {
        var info = new NetworkInfo { Interfaces = new List<NetworkInterface>() };

        // Get public IP (works on both platforms)
        try
        {
            var cmd = _isWindows ? "curl.exe" : "curl";
            var publicIpResult = await _executor.ExecuteAsync(cmd, "-s --max-time 5 https://api.ipify.org", ct);
            if (publicIpResult.Success)
                info.PublicIp = publicIpResult.StandardOutput.Trim();
        }
        catch { /* Optional */ }

        if (_isWindows)
        {
            var ipResult = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike '*Loopback*' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object -First 1).IPAddress\"",
                ct);
            if (ipResult.Success)
                info.PrivateIp = ipResult.StandardOutput.Trim();

            var ifResult = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object -First 1 Name, MacAddress, LinkSpeed | ConvertTo-Json\"",
                ct);
            if (ifResult.Success && !string.IsNullOrWhiteSpace(ifResult.StandardOutput))
            {
                info.Interfaces.Add(new NetworkInterface
                {
                    Name = ExtractJsonValue(ifResult.StandardOutput, "Name") ?? "Ethernet",
                    MacAddress = ExtractJsonValue(ifResult.StandardOutput, "MacAddress") ?? "",
                    IpAddress = info.PrivateIp,
                    IsUp = true
                });
            }
        }
        else
        {
            var ipResult = await _executor.ExecuteAsync("hostname", "-I", ct);
            if (ipResult.Success)
                info.PrivateIp = ipResult.StandardOutput.Trim().Split(' ').FirstOrDefault() ?? "";
        }

        // Determine NAT type by comparing public and private IPs
        if (!string.IsNullOrEmpty(info.PublicIp) && !string.IsNullOrEmpty(info.PrivateIp))
        {
            // Check if public IP matches any of the private IPs (for multi-interface systems)
            var privateIps = info.PrivateIp.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (privateIps.Contains(info.PublicIp))
            {
                // Public IP matches one of the private IPs = direct connection, no NAT
                info.NatType = NatType.None;
                _logger.LogInformation(
                    "✓ NAT detection: Public IP {PublicIp} matches private IP - NatType.None (direct internet connection)",
                    info.PublicIp);
            }
            else
            {
                // Public IP differs from all private IPs = behind NAT
                // For now, mark as Unknown (could be enhanced with STUN for specific NAT type)
                info.NatType = NatType.Unknown;
                _logger.LogInformation(
                    "⚠ NAT detection: Public IP {PublicIp} != Private IP {PrivateIp} - behind NAT/CGNAT (NatType.Unknown)",
                    info.PublicIp, info.PrivateIp);
            }
        }
        else
        {
            info.NatType = NatType.Unknown;
            _logger.LogWarning(
                "⚠ NAT detection: Could not determine NAT type (PublicIP: {PublicIp}, PrivateIP: {PrivateIp})",
                info.PublicIp ?? "null",
                info.PrivateIp ?? "null");
        }

        return info;
    }

    /// <summary>
    /// Detect CPU architecture using platform APIs
    /// Returns: x86_64, aarch64, arm64, etc.
    /// </summary>
    private string DetectArchitecture()
    {
        // Use .NET's built-in architecture detection
        var runtimeArch = RuntimeInformation.ProcessArchitecture;

        return runtimeArch switch
        {
            System.Runtime.InteropServices.Architecture.X64 => "x86_64",
            System.Runtime.InteropServices.Architecture.Arm64 => "aarch64",
            System.Runtime.InteropServices.Architecture.X86 => "i686",
            System.Runtime.InteropServices.Architecture.Arm => "armv7l",
            _ => "unknown"
        };
    }

    // Add to constructor
    private readonly INodeMetadataService _nodeMetadata;

    public async Task<ResourceSnapshot> GetCurrentSnapshotAsync(CancellationToken ct = default)
    {
        var cpu = await GetCpuInfoAsync(ct);
        var memory = await GetMemoryInfoAsync(ct);
        var storage = await GetStorageInfoAsync(ct);
        var gpus = await GetGpuInfoAsync(ct);

        // ✅ Get compute points from performance evaluation
        var performanceEval = _nodeMetadata.PerformanceEvaluation;
        var totalComputePoints = performanceEval?.TotalComputePoints ?? 0;

        return new ResourceSnapshot
        {
            TotalPhysicalCores = cpu.PhysicalCores,
            TotalVirtualCpuCores = cpu.LogicalCores,
            UsedVirtualCpuCores = cpu.LogicalCores - cpu.AvailableVCpus,
            VirtualCpuUsagePercent = cpu.UsagePercent,

            // ✅ FIXED: Compute points from performance evaluation
            TotalComputePoints = (int)totalComputePoints,
            UsedComputePoints = 0, // Will be calculated by HeartbeatService

            TotalMemoryBytes = memory.TotalBytes,
            UsedMemoryBytes = memory.UsedBytes,
            TotalStorageBytes = storage.Sum(s => s.TotalBytes),
            UsedStorageBytes = storage.Sum(s => s.UsedBytes),
            TotalGpus = gpus.Count,
            UsedGpus = 0
        };
    }

    private static long ParseMemoryValue(string value)
    {
        var parts = value.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0 || !long.TryParse(parts[0], out var num)) return 0;
        var unit = parts.Length > 1 ? parts[1].ToLower() : "kb";
        return unit switch
        {
            "kb" => num * 1024,
            "mb" => num * 1024 * 1024,
            "gb" => num * 1024 * 1024 * 1024,
            _ => num * 1024
        };
    }

    private static string? ExtractJsonValue(string json, string key)
    {
        var pattern = $"\"{key}\"\\s*:\\s*\"?([^,\"\\}}\\]]+)\"?";
        var match = Regex.Match(json, pattern, RegexOptions.IgnoreCase);
        return match.Success ? match.Groups[1].Value.Trim() : null;
    }

    private static List<string> SplitJsonArray(string json)
    {
        var items = new List<string>();
        var depth = 0;
        var start = -1;

        for (var i = 0; i < json.Length; i++)
        {
            if (json[i] == '{') { if (depth++ == 0) start = i; }
            else if (json[i] == '}') { if (--depth == 0 && start >= 0) { items.Add(json.Substring(start, i - start + 1)); start = -1; } }
        }

        return items;
    }
}