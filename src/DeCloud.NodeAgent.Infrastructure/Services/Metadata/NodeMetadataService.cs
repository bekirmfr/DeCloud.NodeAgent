using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared;
using DeCloud.Shared.Contracts;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Text.Json;

public class NodeMetadataService : INodeMetadataService
{
    private readonly IConfiguration _configuration;
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly ILogger<NodeMetadataService> _logger;

    public string OrchestratorUrl { get; private set; } = string.Empty;
    public string NodeId { get; private set; } = string.Empty;
    public string MachineId { get; private set; } = string.Empty;
    public string Name { get; private set; } = string.Empty;
    public string? PublicIp { get; private set; }
    public string WalletAddress { get; private set; } = string.Empty;
    public string Region { get; private set; } = string.Empty;
    public string Zone { get; private set; } = string.Empty;
    public string Country { get; private set; } = "ZZ";
    public NodePricing? Pricing { get; private set; }
    public HardwareInventory? Inventory { get; private set; } = null;
    private string? _memoryAllocMode;
    private int? _memoryAllocValue;
    public long? AllocatedMemoryBytes { get; private set; }

    private string? _cpuAllocMode;
    private int? _cpuAllocValue;
    public int? AllocatedComputePoints { get; private set; }
    public int? AllocatedComputePointsPercent { get; private set; }

    private string? _storageAllocMode;
    private int? _storageAllocValue;
    public long? AllocatedStorageBytes { get; private set; }
    public int? AllocatedMemoryPercent { get; private set; }
    public int? AllocatedStoragePercent { get; private set; }

    public int? AllocatedGpuCount { get; private set; }

    private string? _gpuVramAllocMode;
    private int? _gpuVramAllocValue;
    public long? AllocatedGpuVramBytes { get; private set; }
    public int? AllocatedGpuVramPercent { get; private set; }

    /// <summary>
    /// When the orchestrator last confirmed allocation, as persisted in
    /// allocation-resolved.json. Null if the cache has never been written.
    /// </summary>
    public DateTime? AllocationResolvedAt { get; private set; }

    private const string ResolvedAllocationFile = "/etc/decloud/allocation-resolved.json";

    private record ResolvedAllocationCache(
        DateTime ResolvedAt,
        int ComputePoints,
        long MemoryBytes,
        long StorageBytes,
        long GpuVramBytes,
        double CpuPercent,
        double MemoryPercent,
        double StoragePercent,
        double GpuVramPercent);

    public NodeMetadataService(IConfiguration configuration, ILogger<NodeMetadataService> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        // Get machine ID
        MachineId = NodeIdGenerator.GetMachineId();

        // Operator identity — settings.json (flat keys) override appsettings (nested keys).
        // settings.json: { "wallet": "0x..." }  → key "wallet"
        // appsettings:   { "OrchestratorClient": { "WalletAddress": "0x..." } } → key "OrchestratorClient:WalletAddress"
        WalletAddress = _configuration["wallet"]
                     ?? _configuration["OrchestratorClient:WalletAddress"]
                     ?? "";

        OrchestratorUrl = _configuration["orchestrator_url"]
                       ?? _configuration["OrchestratorClient:BaseUrl"]
                       ?? "";

        // Generate deterministic node ID
        NodeId = NodeIdGenerator.GenerateNodeId(MachineId, WalletAddress);

        // Operator locality — settings.json flat keys override Node:* from appsettings
        Name = _configuration["name"]
            ?? _configuration["Node:Name"]
            ?? Environment.MachineName;
        Region = _configuration["region"]
              ?? _configuration["Node:Region"]
              ?? "default";
        Zone = _configuration["zone"]
            ?? _configuration["Node:Zone"]
            ?? "default";
        Country = (_configuration["country"]
                ?? _configuration["Node:Country"]
                ?? "ZZ").ToUpperInvariant();

        _logger.LogInformation(
            "Node locality: Country={Country}, Region={Region}, Zone={Zone}",
            Country, Region, Zone);

        // ── Memory allocation ────────────────────────────────────────────
        _memoryAllocMode = _configuration["resources:memory:mode"]
                        ?? _configuration["Node:Resources:Memory:Mode"];
        var memValStr = _configuration["resources:memory:value"]
                     ?? _configuration["Node:Resources:Memory:Value"];
        if (int.TryParse(memValStr, out var memVal))
            _memoryAllocValue = memVal;

        if (string.Equals(_memoryAllocMode, "mb", StringComparison.OrdinalIgnoreCase)
            && _memoryAllocValue.HasValue)
        {
            AllocatedMemoryBytes = (long)_memoryAllocValue.Value * 1024 * 1024;
            _logger.LogInformation("Resource allocation (memory): {Mb} MB (absolute)",
                _memoryAllocValue.Value);
        }
        else if (string.Equals(_memoryAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _memoryAllocValue.HasValue)
        {
            _logger.LogInformation("Resource allocation (memory): {Pct}% (deferred until hardware discovery)",
                _memoryAllocValue.Value);
        }
        else
        {
            _logger.LogInformation("Resource allocation (memory): not configured, platform default (90%) will apply");
        }

        // ── CPU allocation ───────────────────────────────────────────────
        _cpuAllocMode = _configuration["resources:cpu:mode"];
        var cpuValStr = _configuration["resources:cpu:value"];
        if (int.TryParse(cpuValStr, out var cpuVal))
            _cpuAllocValue = cpuVal;

        if (string.Equals(_cpuAllocMode, "points", StringComparison.OrdinalIgnoreCase)
            && _cpuAllocValue.HasValue)
        {
            AllocatedComputePoints = _cpuAllocValue.Value;
            _logger.LogInformation("Resource allocation (CPU): {Pts} compute points (absolute)",
                _cpuAllocValue.Value);
        }
        else if (string.Equals(_cpuAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _cpuAllocValue.HasValue)
        {
            _logger.LogInformation("Resource allocation (CPU): {Pct}% (deferred until hardware discovery)",
                _cpuAllocValue.Value);
        }
        else
        {
            _logger.LogInformation("Resource allocation (CPU): not configured, platform default (90%) will apply");
        }

        // ── Storage allocation ───────────────────────────────────────────
        _storageAllocMode = _configuration["resources:storage:mode"];
        var storValStr = _configuration["resources:storage:value"];
        if (int.TryParse(storValStr, out var storVal))
            _storageAllocValue = storVal;

        if (string.Equals(_storageAllocMode, "mb", StringComparison.OrdinalIgnoreCase)
            && _storageAllocValue.HasValue)
        {
            AllocatedStorageBytes = (long)_storageAllocValue.Value * 1024 * 1024;
            _logger.LogInformation("Resource allocation (storage): {Mb} MB (absolute)",
                _storageAllocValue.Value);
        }
        else if (string.Equals(_storageAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _storageAllocValue.HasValue)
        {
            _logger.LogInformation("Resource allocation (storage): {Pct}% (deferred until hardware discovery)",
                _storageAllocValue.Value);
        }
        else
        {
            _logger.LogInformation("Resource allocation (storage): not configured, platform default (90%) will apply");
        }

        // ── GPU allocation ───────────────────────────────────────────────
        var gpuCountStr = _configuration["resources:gpu:count"];
        if (int.TryParse(gpuCountStr, out var gpuCount))
        {
            AllocatedGpuCount = gpuCount;
            _logger.LogInformation("Resource allocation (GPU): {Count} GPUs", gpuCount);
        }
        else
        {
            _logger.LogInformation("Resource allocation (GPU): not configured, all detected GPUs offered");
        }

        // ── GPU VRAM allocation (Proxied mode) ───────────────────────────
        _gpuVramAllocMode = _configuration["resources:gpu:vram:mode"];
        var gpuVramValStr = _configuration["resources:gpu:vram:value"];
        if (int.TryParse(gpuVramValStr, out var gpuVramVal))
            _gpuVramAllocValue = gpuVramVal;

        if (string.Equals(_gpuVramAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _gpuVramAllocValue.HasValue)
            _logger.LogInformation(
                "Resource allocation (GPU VRAM): {Pct}% (deferred until hardware discovery)",
                _gpuVramAllocValue.Value);
        else
            _logger.LogInformation(
                "Resource allocation (GPU VRAM): not configured, 100% of proxy-eligible VRAM offered");

        // ── Load persisted orchestrator-resolved allocation ──────────────────
        // These are the last confirmed concrete values from the orchestrator.
        // They take precedence over locally-derived settings values so the agent
        // starts with an accurate view even before the next allocate/heartbeat.
        // The Resolve* methods (called from UpdateInventory) guard with HasValue,
        // so they will skip re-derivation when these values are already set.
        await LoadResolvedAllocationAsync(ct);

        // Load operator pricing from settings.json flat keys.
        // Written by: decloud pricing --cpu 0.012 --memory 0.006 ...
        // Stored as a nested "pricing" object in settings.json; ASP.NET Core
        // config layering exposes them as pricing:cpu_per_hour etc.
        var cpuStr = _configuration["pricing:cpu_per_hour"];
        var memStr = _configuration["pricing:memory_per_gb_per_hour"];
        var storageStr = _configuration["pricing:storage_per_gb_per_hour"];
        var gpuStr = _configuration["pricing:gpu_vram_per_gb_per_hour"];

        if (cpuStr != null || memStr != null || storageStr != null || gpuStr != null)
        {
            Pricing = new NodePricing();
            if (decimal.TryParse(cpuStr, System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out var cpu) && cpu > 0)
                Pricing.CpuPerHour = cpu;
            if (decimal.TryParse(memStr, System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out var mem) && mem > 0)
                Pricing.MemoryPerGbPerHour = mem;
            if (decimal.TryParse(storageStr, System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out var storage) && storage > 0)
                Pricing.StoragePerGbPerHour = storage;
            if (decimal.TryParse(gpuStr, System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out var gpu) && gpu > 0)
                Pricing.GpuVramPerGbPerHour = gpu;

            _logger.LogInformation(
                "Operator pricing loaded: CPU={Cpu}/hr, Memory={Mem}/GB/hr, " +
                "Storage={Storage}/GB/hr, GpuVram={Gpu}/GB/hr",
                Pricing.CpuPerHour, Pricing.MemoryPerGbPerHour,
                Pricing.StoragePerGbPerHour, Pricing.GpuVramPerGbPerHour);
        }

        // Discover public IP
        PublicIp = await DiscoverPublicIpAsync(ct);

        // Background task to update inventory
        _ = Task.Run(async () => {
            var inv = await _resourceDiscovery.GetInventoryCachedAsync(CancellationToken.None);
            if (inv != null) UpdateInventory(inv);
        }, ct);

        _logger.LogInformation(
            "✓ Node metadata initialized: ID={NodeId}, Name={Name}, IP={PublicIp}",
            NodeId, Name, PublicIp);
    }

    public void UpdatePublicIp(string publicIp)
    {
        if (PublicIp != publicIp)
        {
            _logger.LogInformation("Public IP changed: {Old} → {New}", PublicIp, publicIp);
            PublicIp = publicIp;
        }
    }

    private async Task<string?> DiscoverPublicIpAsync(CancellationToken ct)
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var ip = await client.GetStringAsync("https://api.ipify.org", ct);
            return ip.Trim();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to discover public IP");
            return null;
        }
    }

    private async Task LoadResolvedAllocationAsync(CancellationToken ct)
    {
        if (!File.Exists(ResolvedAllocationFile))
            return;

        try
        {
            var json = await File.ReadAllTextAsync(ResolvedAllocationFile, ct);
            var cached = JsonSerializer.Deserialize<ResolvedAllocationCache>(json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            if (cached == null) return;

            if (cached.ComputePoints > 0) AllocatedComputePoints = cached.ComputePoints;
            if (cached.MemoryBytes > 0) AllocatedMemoryBytes = cached.MemoryBytes;
            if (cached.StorageBytes > 0) AllocatedStorageBytes = cached.StorageBytes;
            if (cached.GpuVramBytes > 0) AllocatedGpuVramBytes = cached.GpuVramBytes;
            AllocationResolvedAt = cached.ResolvedAt;

            _logger.LogInformation(
                "Loaded persisted orchestrator allocation: {Pts} pts, " +
                "{MemGb:F1} GB RAM, {StorGb:F1} GB storage (resolved {At:u})",
                cached.ComputePoints,
                cached.MemoryBytes / (1024.0 * 1024 * 1024),
                cached.StorageBytes / (1024.0 * 1024 * 1024),
                cached.ResolvedAt);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Could not load persisted allocation from {File} — using settings-derived values",
                ResolvedAllocationFile);
        }
    }

    public async Task UpdateFromOrchestratorResolutionAsync(
    NodeAllocateResponse response,
    CancellationToken ct = default)
    {
        // Update in-memory state with whatever the orchestrator confirmed.
        // Only overwrite fields that the orchestrator actually resolved (non-null, non-zero).
        if (response.ResolvedComputePoints is > 0) AllocatedComputePoints = response.ResolvedComputePoints.Value;
        if (response.ResolvedMemoryBytes is > 0) AllocatedMemoryBytes = response.ResolvedMemoryBytes.Value;
        if (response.ResolvedStorageBytes is > 0) AllocatedStorageBytes = response.ResolvedStorageBytes.Value;
        if (response.ResolvedGpuVramBytes is > 0) AllocatedGpuVramBytes = response.ResolvedGpuVramBytes.Value;
        AllocationResolvedAt = DateTime.UtcNow;

        try
        {
            var cache = new ResolvedAllocationCache(
                ResolvedAt: DateTime.UtcNow,
                ComputePoints: response.ResolvedComputePoints ?? 0,
                MemoryBytes: response.ResolvedMemoryBytes ?? 0,
                StorageBytes: response.ResolvedStorageBytes ?? 0,
                GpuVramBytes: response.ResolvedGpuVramBytes ?? 0,
                CpuPercent: response.EffectiveCpuPercent,
                MemoryPercent: response.EffectiveMemoryPercent,
                StoragePercent: response.EffectiveStoragePercent,
                GpuVramPercent: response.EffectiveGpuVramPercent ?? 0);

            var json = JsonSerializer.Serialize(cache, new JsonSerializerOptions { WriteIndented = true });

            // Atomic write: write to .tmp then rename so a crash mid-write
            // never leaves a corrupt file.
            var tmp = ResolvedAllocationFile + ".tmp";
            await File.WriteAllTextAsync(tmp, json, ct);
            File.Move(tmp, ResolvedAllocationFile, overwrite: true);
            File.SetUnixFileMode(ResolvedAllocationFile, UnixFileMode.UserRead | UnixFileMode.UserWrite);

            _logger.LogInformation(
                "✓ Orchestrator allocation persisted: {Pts} pts, " +
                "{MemGb:F1} GB RAM, {StorGb:F1} GB storage",
                cache.ComputePoints,
                cache.MemoryBytes / (1024.0 * 1024 * 1024),
                cache.StorageBytes / (1024.0 * 1024 * 1024));
        }
        catch (Exception ex)
        {
            // Non-fatal: in-memory state is already updated; persistence failure
            // only affects the next restart, where settings-derived values apply.
            _logger.LogWarning(ex,
                "Could not persist orchestrator allocation to {File}",
                ResolvedAllocationFile);
        }
    }

    public void UpdateInventory(HardwareInventory inventory)
    {
        Inventory = inventory;
        ResolveAllocatedMemory(inventory);
        ResolveAllocatedCpu(inventory);
        ResolveAllocatedStorage(inventory);
        ResolveAllocatedGpuVram(inventory);
    }

    private void ResolveAllocatedGpuVram(HardwareInventory inventory)
    {
        if (AllocatedGpuVramBytes.HasValue)
            return; // Already resolved (from persisted cache)

        if (string.Equals(_gpuVramAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _gpuVramAllocValue.HasValue)
        {
            AllocatedGpuVramPercent = Math.Clamp(_gpuVramAllocValue.Value, 1, 95);
            var pct = AllocatedGpuVramPercent.Value / 100.0;
            var totalProxiedVram = inventory.Gpus
                .Where(g => g.IsAvailableForProxiedSharing)
                .Sum(g => g.MemoryBytes);
            AllocatedGpuVramBytes = (long)(totalProxiedVram * pct);
            _logger.LogInformation(
                "Resource allocation (GPU VRAM): resolved {Pct}% of {TotalGb:F1} GB = {AllocGb:F1} GB",
                _gpuVramAllocValue.Value,
                totalProxiedVram / (1024.0 * 1024 * 1024),
                AllocatedGpuVramBytes.Value / (1024.0 * 1024 * 1024));
        }
        // else: null → orchestrator uses full proxy-eligible VRAM as the pool
    }

    /// <summary>
    /// Resolve percent-based memory allocation once hardware info is available.
    /// Called from UpdateInventory after discovery completes.
    /// </summary>
    private void ResolveAllocatedMemory(HardwareInventory inventory)
    {
        if (AllocatedMemoryBytes.HasValue)
            return; // Already resolved (absolute mode set in InitializeAsync)

        if (string.Equals(_memoryAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _memoryAllocValue.HasValue)
        {
            AllocatedMemoryPercent = Math.Clamp(_memoryAllocValue.Value, 1, 95);
            var pct = AllocatedMemoryPercent.Value / 100.0;
            AllocatedMemoryBytes = (long)(inventory.Memory.TotalBytes * pct);
            _logger.LogInformation(
                "Resource allocation (memory): resolved {Pct}% of {TotalMb} MB = {AllocMb} MB",
                _memoryAllocValue.Value,
                inventory.Memory.TotalBytes / (1024 * 1024),
                AllocatedMemoryBytes.Value / (1024 * 1024));
        }
        // else: null → orchestrator applies 90% default
    }

    private void ResolveAllocatedCpu(HardwareInventory inventory)
    {
        if (AllocatedComputePoints.HasValue)
            return; // Already resolved (absolute points set in InitializeAsync)

        if (string.Equals(_cpuAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _cpuAllocValue.HasValue)
        {
            // Node agent doesn't know its own benchmark score — send the percent
            // to the orchestrator which applies it after evaluation.
            AllocatedComputePointsPercent = Math.Clamp(_cpuAllocValue.Value, 1, 100);
            _logger.LogInformation(
                "Resource allocation (CPU): {Pct}% → orchestrator will apply to evaluated points",
                _cpuAllocValue.Value);
        }
        // else: null → orchestrator applies 90% default
    }

    private void ResolveAllocatedStorage(HardwareInventory inventory)
    {
        if (AllocatedStorageBytes.HasValue)
            return; // Already resolved (absolute mode set in InitializeAsync)

        if (string.Equals(_storageAllocMode, "percent", StringComparison.OrdinalIgnoreCase)
            && _storageAllocValue.HasValue)
        {
            AllocatedStoragePercent = Math.Clamp(_storageAllocValue.Value, 1, 95);
            var pct = AllocatedStoragePercent.Value / 100.0;
            var totalStorage = inventory.Storage.Sum(s => s.TotalBytes);
            AllocatedStorageBytes = (long)(totalStorage * pct);
            _logger.LogInformation(
                "Resource allocation (storage): resolved {Pct}% of {TotalGb} GB = {AllocGb} GB",
                _storageAllocValue.Value,
                totalStorage / (1024 * 1024 * 1024),
                AllocatedStorageBytes.Value / (1024 * 1024 * 1024));
        }
        // else: null → orchestrator applies 90% default
    }
}