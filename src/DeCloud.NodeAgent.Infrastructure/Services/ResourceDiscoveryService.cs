using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Resource discovery service with caching support
/// Initial discovery is expensive (15-30s due to CPU benchmark)
/// Subsequent calls use cached data
/// </summary>
public class ResourceDiscoveryService : IResourceDiscoveryService
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<ResourceDiscoveryService> _logger;
    private readonly ICpuBenchmarkService _benchmarkService;
    private readonly string _nodeId;
    private readonly bool _isWindows;

    // Caching fields
    private HardwareInventory? _cachedInventory;
    private bool _discoveryComplete;
    private readonly SemaphoreSlim _discoverySemaphore = new(1, 1);
    private DateTime _lastDiscoveryTime = DateTime.MinValue;

    public ResourceDiscoveryService(
        ICommandExecutor executor,
        ILogger<ResourceDiscoveryService> logger,
        ICpuBenchmarkService benchmarkService)
    {
        _executor = executor;
        _logger = logger;
        _benchmarkService = benchmarkService;
        _isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        _nodeId = GetOrCreateNodeId();
    }

    /// <summary>
    /// Check if initial discovery has completed
    /// </summary>
    public bool IsDiscoveryComplete() => _discoveryComplete;

    /// <summary>
    /// Get cached hardware inventory
    /// Returns null if discovery hasn't completed yet
    /// </summary>
    public Task<HardwareInventory?> GetCachedInventoryAsync(CancellationToken ct = default)
    {
        return Task.FromResult(_cachedInventory);
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
                NodeId = _nodeId,
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
            _discoveryComplete = true;
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

    private async Task<CpuInfo> GetCpuInfoWindowsAsync(CancellationToken ct)
    {
        var info = new CpuInfo { Flags = new List<string>() };

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

        // Run CPU benchmark (expensive operation)
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

    private async Task<CpuInfo> GetCpuInfoLinuxAsync(CancellationToken ct)
    {
        var info = new CpuInfo { Flags = new List<string>() };

        var cpuinfoResult = await _executor.ExecuteAsync("cat", "/proc/cpuinfo", ct);
        if (cpuinfoResult.Success)
        {
            foreach (var line in cpuinfoResult.StandardOutput.Split('\n'))
            {
                if (line.StartsWith("model name"))
                    info.Model = line.Split(':').LastOrDefault()?.Trim() ?? "";
                else if (line.StartsWith("cpu MHz") && info.FrequencyMhz == 0)
                    double.TryParse(line.Split(':').LastOrDefault()?.Trim(), out info.FrequencyMhz);
                else if (line.StartsWith("flags"))
                {
                    info.Flags = line.Split(':').LastOrDefault()?.Trim().Split(' ').ToList() ?? new();
                    info.SupportsVirtualization = info.Flags.Contains("vmx") || info.Flags.Contains("svm");
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
            }
        }

        // Fallback only if lscpu completely failed
        if (info.LogicalCores == 0) info.LogicalCores = Environment.ProcessorCount;
        if (info.PhysicalCores == 0) info.PhysicalCores = info.LogicalCores;
        info.AvailableVCpus = info.LogicalCores;

        // Run CPU benchmark (expensive operation)
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
        var info = new MemoryInfo();

        try
        {
            var result = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | ConvertTo-Json\"",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                var json = result.StandardOutput.Trim();
                info.TotalBytes = ParseMemoryValue(ExtractJsonValue(json, "TotalVisibleMemorySize") ?? "0 KB");
                info.AvailableBytes = ParseMemoryValue(ExtractJsonValue(json, "FreePhysicalMemory") ?? "0 KB");
                info.UsedBytes = info.TotalBytes - info.AvailableBytes;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to get Windows memory info");
        }

        // Reserve 1GB for host OS
        info.ReservedBytes = 1024L * 1024 * 1024;

        return info;
    }

    private async Task<MemoryInfo> GetMemoryInfoLinuxAsync(CancellationToken ct)
    {
        var info = new MemoryInfo();

        var result = await _executor.ExecuteAsync("cat", "/proc/meminfo", ct);
        if (result.Success)
        {
            foreach (var line in result.StandardOutput.Split('\n'))
            {
                if (line.StartsWith("MemTotal:"))
                {
                    var value = line.Split(':')[1].Trim();
                    info.TotalBytes = ParseMemoryValue(value);
                }
                else if (line.StartsWith("MemAvailable:"))
                {
                    var value = line.Split(':')[1].Trim();
                    info.AvailableBytes = ParseMemoryValue(value);
                }
            }

            info.UsedBytes = info.TotalBytes - info.AvailableBytes;
        }

        // Reserve 1GB for host OS
        info.ReservedBytes = 1024L * 1024 * 1024;

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
                "-NoProfile -Command \"Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Used -ne $null} | Select-Object Name, @{n='TotalBytes';e={$_.Used + $_.Free}}, @{n='UsedBytes';e={$_.Used}}, @{n='FreeBytes';e={$_.Free}} | ConvertTo-Json\"",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                var json = result.StandardOutput.Trim();
                var drives = json.StartsWith("[") ? SplitJsonArray(json) : new List<string> { json };

                foreach (var drive in drives)
                {
                    var name = ExtractJsonValue(drive, "Name");
                    if (string.IsNullOrWhiteSpace(name)) continue;

                    storageList.Add(new StorageInfo
                    {
                        DevicePath = $"{name}:",
                        MountPoint = $"{name}:\\",
                        FileSystem = "NTFS",
                        Type = StorageType.HDD,
                        TotalBytes = long.TryParse(ExtractJsonValue(drive, "TotalBytes"), out var total) ? total : 0,
                        UsedBytes = long.TryParse(ExtractJsonValue(drive, "UsedBytes"), out var used) ? used : 0,
                        AvailableBytes = long.TryParse(ExtractJsonValue(drive, "FreeBytes"), out var free) ? free : 0
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

        var result = await _executor.ExecuteAsync("df", "-B1 -T -x tmpfs -x devtmpfs", ct);
        if (result.Success)
        {
            var lines = result.StandardOutput.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            foreach (var line in lines.Skip(1))
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 7) continue;

                storageList.Add(new StorageInfo
                {
                    DevicePath = parts[0],
                    FileSystem = parts[1],
                    Type = DetermineStorageType(parts[0]),
                    TotalBytes = long.TryParse(parts[2], out var total) ? total : 0,
                    UsedBytes = long.TryParse(parts[3], out var used) ? used : 0,
                    AvailableBytes = long.TryParse(parts[4], out var avail) ? avail : 0,
                    MountPoint = parts[6]
                });
            }
        }

        return storageList;
    }

    public async Task<List<GpuInfo>> GetGpuInfoAsync(CancellationToken ct = default)
    {
        return _isWindows
            ? await GetGpuInfoWindowsAsync(ct)
            : await GetGpuInfoLinuxAsync(ct);
    }

    private async Task<List<GpuInfo>> GetGpuInfoWindowsAsync(CancellationToken ct)
    {
        var gpuList = new List<GpuInfo>();

        try
        {
            var result = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json\"",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                var json = result.StandardOutput.Trim();
                var gpus = json.StartsWith("[") ? SplitJsonArray(json) : new List<string> { json };

                foreach (var gpu in gpus)
                {
                    gpuList.Add(new GpuInfo
                    {
                        Model = ExtractJsonValue(gpu, "Name") ?? "Unknown GPU",
                        MemoryBytes = long.TryParse(ExtractJsonValue(gpu, "AdapterRAM"), out var mem) ? mem : 0,
                        DriverVersion = ExtractJsonValue(gpu, "DriverVersion") ?? "Unknown"
                    });
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to get Windows GPU info");
        }

        return gpuList;
    }

    private async Task<List<GpuInfo>> GetGpuInfoLinuxAsync(CancellationToken ct)
    {
        var gpuList = new List<GpuInfo>();

        // Try nvidia-smi first
        var nvidiaResult = await _executor.ExecuteAsync("nvidia-smi",
            "--query-gpu=name,memory.total,driver_version --format=csv,noheader",
            ct);

        if (nvidiaResult.Success && !string.IsNullOrWhiteSpace(nvidiaResult.StandardOutput))
        {
            var lines = nvidiaResult.StandardOutput.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            foreach (var line in lines)
            {
                var parts = line.Split(',');
                if (parts.Length < 3) continue;

                gpuList.Add(new GpuInfo
                {
                    Model = parts[0].Trim(),
                    MemoryBytes = ParseMemoryValue(parts[1].Trim()),
                    DriverVersion = parts[2].Trim()
                });
            }
        }

        // Fallback to lspci
        if (gpuList.Count == 0)
        {
            var lspciResult = await _executor.ExecuteAsync("lspci", "", ct);
            if (lspciResult.Success)
            {
                var lines = lspciResult.StandardOutput.Split('\n');
                foreach (var line in lines)
                {
                    if (line.Contains("VGA compatible controller") || line.Contains("3D controller"))
                    {
                        var parts = line.Split(':');
                        if (parts.Length > 2)
                        {
                            gpuList.Add(new GpuInfo
                            {
                                Model = parts[2].Trim(),
                                MemoryBytes = 0,
                                DriverVersion = "Unknown"
                            });
                        }
                    }
                }
            }
        }

        return gpuList;
    }

    public async Task<NetworkInfo> GetNetworkInfoAsync(CancellationToken ct = default)
    {
        return _isWindows
            ? await GetNetworkInfoWindowsAsync(ct)
            : await GetNetworkInfoLinuxAsync(ct);
    }

    private async Task<NetworkInfo> GetNetworkInfoWindowsAsync(CancellationToken ct)
    {
        var info = new NetworkInfo { Interfaces = new List<NetworkInterface>() };

        try
        {
            var result = await _executor.ExecuteAsync("powershell",
                "-NoProfile -Command \"Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notlike '*Loopback*'} | Select-Object InterfaceAlias, IPAddress | ConvertTo-Json\"",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                var json = result.StandardOutput.Trim();
                var interfaces = json.StartsWith("[") ? SplitJsonArray(json) : new List<string> { json };

                foreach (var iface in interfaces)
                {
                    info.Interfaces.Add(new NetworkInterface
                    {
                        Name = ExtractJsonValue(iface, "InterfaceAlias") ?? "Unknown",
                        PrivateIp = ExtractJsonValue(iface, "IPAddress") ?? "Unknown"
                    });
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to get Windows network info");
        }

        return info;
    }

    private async Task<NetworkInfo> GetNetworkInfoLinuxAsync(CancellationToken ct)
    {
        var info = new NetworkInfo { Interfaces = new List<NetworkInterface>() };

        var result = await _executor.ExecuteAsync("ip", "-4 addr show", ct);
        if (result.Success)
        {
            string? currentInterface = null;
            foreach (var line in result.StandardOutput.Split('\n'))
            {
                if (line.Contains(':') && !line.StartsWith(' '))
                {
                    var parts = line.Split(':', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                    {
                        currentInterface = parts[1].Trim().Split(' ')[0];
                    }
                }
                else if (line.Trim().StartsWith("inet ") && currentInterface != null && currentInterface != "lo")
                {
                    var ipPart = line.Trim().Split(' ')[1].Split('/')[0];
                    info.Interfaces.Add(new NetworkInterface
                    {
                        Name = currentInterface,
                        PrivateIp = ipPart
                    });
                }
            }
        }

        if (info.Interfaces.Count > 0)
        {
            info.PrimaryInterface = info.Interfaces[0].Name;
            info.PrivateIp = info.Interfaces[0].PrivateIp;

            _logger.LogInformation(
                "Network: Primary={Primary}, IP={Ip}",
                info.PrimaryInterface ?? "null",
                info.PrivateIp ?? "null");
        }

        return info;
    }

    public async Task<ResourceSnapshot> GetCurrentSnapshotAsync(CancellationToken ct = default)
    {
        var cpu = await GetCpuInfoAsync(ct);
        var memory = await GetMemoryInfoAsync(ct);
        var storage = await GetStorageInfoAsync(ct);
        var gpus = await GetGpuInfoAsync(ct);

        return new ResourceSnapshot
        {
            TotalPhysicalCores = cpu.PhysicalCores,
            TotalVirtualCpuCores = cpu.LogicalCores,
            UsedVirtualCpuCores = cpu.LogicalCores - cpu.AvailableVCpus,
            VirtualCpuUsagePercent = cpu.UsagePercent,
            TotalMemoryBytes = memory.TotalBytes,
            UsedMemoryBytes = memory.UsedBytes,
            TotalStorageBytes = storage.Sum(s => s.TotalBytes),
            UsedStorageBytes = storage.Sum(s => s.UsedBytes),
            TotalGpus = gpus.Count,
            UsedGpus = 0
        };
    }

    // Helper methods

    private static StorageType DetermineStorageType(string devicePath)
    {
        if (devicePath.Contains("nvme")) return StorageType.NVMe;
        if (devicePath.Contains("sd")) return StorageType.SSD;
        return StorageType.HDD;
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
            "mib" => num * 1024 * 1024,
            "gib" => num * 1024 * 1024 * 1024,
            _ => num * 1024
        };
    }

    private string GetOrCreateNodeId()
    {
        var configDir = _isWindows
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "decloud")
            : "/etc/decloud";
        var idPath = Path.Combine(configDir, "node-id");

        if (File.Exists(idPath))
            return File.ReadAllText(idPath).Trim();

        var nodeId = $"node-{Environment.MachineName.ToLower()}-{Guid.NewGuid().ToString("N")[..8]}";

        try
        {
            Directory.CreateDirectory(configDir);
            File.WriteAllText(idPath, nodeId);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not persist node ID");
        }

        return nodeId;
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