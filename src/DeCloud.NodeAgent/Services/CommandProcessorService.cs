using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using System.Text.Json;

namespace DeCloud.NodeAgent.Services;

public class CommandProcessorOptions
{
    public TimeSpan PollInterval { get; set; } = TimeSpan.FromSeconds(5);
}

public class CommandProcessorService : BackgroundService
{
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly ConcurrentQueue<PendingCommand> _pushedCommands;
    private readonly IVmManager _vmManager;
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly INatRuleManager _natRuleManager;
    private readonly ILogger<CommandProcessorService> _logger;
    private readonly CommandProcessorOptions _options;

    // Default base image URLs (fallback)
    private static readonly Dictionary<string, string> ImageUrls = new(StringComparer.OrdinalIgnoreCase)
    {
        ["ubuntu-24.04"] = "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img",
        ["ubuntu-22.04"] = "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img",
        ["ubuntu-20.04"] = "https://cloud-images.ubuntu.com/focal/current/focal-server-cloudimg-amd64.img",
        ["debian-12"] = "https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2",
        ["debian-11"] = "https://cloud.debian.org/images/cloud/bullseye/latest/debian-11-generic-amd64.qcow2",
        ["fedora-40"] = "https://download.fedoraproject.org/pub/fedora/linux/releases/40/Cloud/x86_64/images/Fedora-Cloud-Base-Generic.x86_64-40-1.14.qcow2",
        ["alpine-3.19"] = "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/cloud/nocloud_alpine-3.19.1-x86_64-bios-cloudinit-r0.qcow2"
    };

    public CommandProcessorService(
        IOrchestratorClient orchestratorClient,
        ConcurrentQueue<PendingCommand> pushedCommands,
        IVmManager vmManager,
        IResourceDiscoveryService resourceDiscovery,
        INatRuleManager natRuleManager,
        IOptions<CommandProcessorOptions> options,
        ILogger<CommandProcessorService> logger)
    {
        _orchestratorClient = orchestratorClient;
        _pushedCommands = pushedCommands;
        _vmManager = vmManager;
        _resourceDiscovery = resourceDiscovery;
        _natRuleManager = natRuleManager;
        _logger = logger;
        _options = options.Value;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation(
            "Command processor starting (hybrid push-pull mode) with poll interval {Interval}s",
            _options.PollInterval.TotalSeconds);

        // Wait for node registration
        await Task.Delay(TimeSpan.FromSeconds(15), stoppingToken);

        // Track processed commands for deduplication
        var processedCommands = new HashSet<string>();
        const int maxProcessedCache = 1000;

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // ============================================================
                // PRIORITY 1: Process pushed commands (instant delivery)
                // ============================================================
                var pushedCount = 0;
                while (_pushedCommands.TryDequeue(out var pushedCommand))
                {
                    pushedCount++;
                    await ProcessCommandAsync(pushedCommand, processedCommands, stoppingToken);
                }

                if (pushedCount > 0)
                {
                    _logger.LogInformation(
                        "✓ Processed {Count} pushed command(s)",
                        pushedCount);
                }

                // ============================================================
                // PRIORITY 2: Poll for queued commands (fallback)
                // ============================================================
                var pulledCommands = await _orchestratorClient.FetchPendingCommandsAsync(
                    stoppingToken);

                foreach (var command in pulledCommands)
                {
                    await ProcessCommandAsync(command, processedCommands, stoppingToken);
                }

                if (pulledCommands.Count > 0)
                {
                    _logger.LogInformation(
                        "✓ Retrieved {Count} command(s) via pull",
                        pulledCommands.Count);
                }

                // ============================================================
                // Cleanup processed commands cache
                // ============================================================
                if (processedCommands.Count > maxProcessedCache)
                {
                    processedCommands.Clear();
                    _logger.LogDebug("Cleared processed commands cache");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in command processing cycle");
            }

            // Poll interval (pushed commands are processed immediately)
            await Task.Delay(_options.PollInterval, stoppingToken);
        }

        _logger.LogInformation("Command processor stopping");
    }

    /// <summary>
    /// Process command with deduplication
    /// </summary>
    private async Task ProcessCommandAsync(
        PendingCommand command,
        HashSet<string> processedCommands,
        CancellationToken ct)
    {
        // Deduplication: command might arrive via both push and pull
        if (processedCommands.Contains(command.CommandId))
        {
            _logger.LogWarning(
                "⚠️  Command {CommandId} already processed (duplicate delivery, likely push+pull race)",
                command.CommandId);
            return;
        }

        processedCommands.Add(command.CommandId);

        _logger.LogInformation(
            "⚙️  Processing command {CommandId}: {Type}",
            command.CommandId, command.Type);

        try
        {
            var success = await ExecuteCommandAsync(command, ct);

            if (command.RequiresAck)
            {
                await _orchestratorClient.AcknowledgeCommandAsync(
                    command.CommandId, success, null, ct);
            }

            _logger.LogInformation(
                "✓ Command {CommandId} completed: {Success}",
                command.CommandId, success);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "✗ Command {CommandId} failed", command.CommandId);

            if (command.RequiresAck)
            {
                await _orchestratorClient.AcknowledgeCommandAsync(
                    command.CommandId, false, ex.Message, ct);
            }
        }
    }

    private async Task<bool> ExecuteCommandAsync(PendingCommand command, CancellationToken ct)
    {
        switch (command.Type)
        {
            case CommandType.CreateVm:
                return await HandleCreateVmAsync(command.Payload, ct);

            case CommandType.StartVm:
                return await HandleStartVmAsync(command.Payload, ct);

            case CommandType.StopVm:
                return await HandleStopVmAsync(command.Payload, ct);

            case CommandType.DeleteVm:
                return await HandleDeleteVmAsync(command.Payload, ct);

            case CommandType.Benchmark:
                return await HandleBenchmarkAsync(ct);

            default:
                _logger.LogWarning("Unknown command type: {Type}", command.Type);
                return false;
        }
    }

    private async Task<bool> HandleCreateVmAsync(string payload, CancellationToken ct)
    {
        var root = JsonDocument.Parse(payload).RootElement;

        string? vmId = GetStringProperty(root, "vmId", "VmId");
        string name = GetStringProperty(root, "name", "Name") ?? $"vm-{vmId?[..8]}";
        string ownerId = GetStringProperty(root, "ownerId", "OwnerId") ?? "unknown";
        string? password = GetStringProperty(root, "password", "Password");
        int vmType = GetIntProperty(root, "vmType", "VmType") ?? 0;
        int qualityTier = GetIntProperty(root, "qualityTier", "QualityTier") ?? 3;
        int virtualCpuCores = GetIntProperty(root, "virtualCpuCores", "VirtualCpuCores") ?? 1;
        int computePointCost = GetIntProperty(root, "computePointCost", "ComputePointCost") ?? 0;
        long memoryBytes = GetLongProperty(root, "memoryBytes", "MemoryBytes") ?? 1024;
        long diskBytes = GetLongProperty(root, "diskBytes", "DiskBytes") ?? 10;
        string? baseImageUrl = GetStringProperty(root, "baseImageUrl", "BaseImageUrl");
        string? imageId = GetStringProperty(root, "imageId", "ImageId");
        string? sshPublicKey = GetStringProperty(root, "sshPublicKey", "SshPublicKey");

        Dictionary<string, string>? labels = null;
        if (root.TryGetProperty("Labels", out var labelsElement) ||
            root.TryGetProperty("labels", out labelsElement))
        {
            labels = JsonSerializer.Deserialize<Dictionary<string, string>>(
                labelsElement.GetRawText());
        }

        // Resolve image URL with architecture awareness
        baseImageUrl = await ResolveImageUrlAsync(imageId, baseImageUrl, ct);

        var vmSpec = new VmSpec
        {
            Id = vmId,
            Name = name,
            OwnerId = ownerId,
            VmType = (VmType)vmType,
            VirtualCpuCores = virtualCpuCores,
            QualityTier = (QualityTier)qualityTier,
            ComputePointCost = computePointCost,
            MemoryBytes = memoryBytes,
            DiskBytes = diskBytes,
            BaseImageUrl = baseImageUrl,
            SshPublicKey = sshPublicKey,
            Labels = labels
        };

        _logger.LogInformation(
            "Creating {VmType} type VM {VmId}: {VCpus} vCPUs, {MemoryMB}MB RAM, " +
            "{DiskGB}GB disk, image: {ImageUrl}, quality tier: {QualityTier}",
            vmSpec.VmType.ToString(), vmId, virtualCpuCores,
            memoryBytes / 1024 / 1024, diskBytes / 1024 / 1024 / 1024,
            baseImageUrl, qualityTier);

        var result = await _vmManager.CreateVmAsync(vmSpec, password, ct);

        if (result.Success)
        {
            _logger.LogInformation(
                "{VmType} VM {VmId} created and started successfully on {Architecture} host",
                vmSpec.VmType.ToString(), vmId,
                (await _resourceDiscovery.GetInventoryCachedAsync(ct))?.Cpu?.Architecture ?? "unknown");

            // Notify orchestrator of successful creation
            //await _orchestratorClient.NotifyVmStatusAsync(vmId, VmState.Running, ct);
            return true;
        }
        else
        {
            _logger.LogError(
                "{VmType} VM {VmId} creation failed: {Error}",
                vmSpec.VmType.ToString(), vmId, result.ErrorMessage);

            // Notify orchestrator of failure
            //await _orchestratorClient.NotifyVmStatusAsync(vmId, VmState.Failed, ct);
            return false;
        }
    }

    private async Task<bool> HandleStartVmAsync(string payload, CancellationToken ct)
    {
        using var doc = JsonDocument.Parse(payload);
        var vmId = GetStringProperty(doc.RootElement, "VmId", "vmId");
        if (string.IsNullOrEmpty(vmId)) return false;

        _logger.LogInformation("Starting VM {VmId}", vmId);
        var result = await _vmManager.StartVmAsync(vmId, ct);
        return result.Success;
    }

    private async Task<bool> HandleStopVmAsync(string payload, CancellationToken ct)
    {
        using var doc = JsonDocument.Parse(payload);
        var vmId = GetStringProperty(doc.RootElement, "VmId", "vmId");
        var force = doc.RootElement.TryGetProperty("Force", out var forceProp) && forceProp.GetBoolean();
        if (string.IsNullOrEmpty(vmId)) return false;

        _logger.LogInformation("Stopping VM {VmId} (force={Force})", vmId, force);
        var result = await _vmManager.StopVmAsync(vmId, force, ct);
        return result.Success;
    }

    private async Task<bool> HandleDeleteVmAsync(string payload, CancellationToken ct)
    {
        using var doc = JsonDocument.Parse(payload);
        var vmId = GetStringProperty(doc.RootElement, "VmId", "vmId");
        if (string.IsNullOrEmpty(vmId)) return false;

        var vmInstance = await _vmManager.GetVmAsync(vmId, ct);
        if (vmInstance?.Spec.VmType == VmType.Relay &&
            !string.IsNullOrEmpty(vmInstance.Spec.IpAddress))
        {
            _logger.LogInformation(
                "Removing NAT rules for relay VM {VmId} ({IpAddress})",
                vmId, vmInstance.Spec.IpAddress);

            await _natRuleManager.RemovePortForwardingAsync(
                vmInstance.Spec.IpAddress,
                51820,
                "udp",
                ct);
        }

        _logger.LogInformation("Deleting VM {VmId}", vmId);
        var result = await _vmManager.DeleteVmAsync(vmId, ct);
        return result.Success;
    }

    private async Task<bool> HandleBenchmarkAsync(CancellationToken ct)
    {
        _logger.LogInformation("Running benchmark...");
        var resources = await _resourceDiscovery.DiscoverAllAsync(ct);
        _logger.LogInformation("Benchmark complete");
        return true;
    }

    /// <summary>
    /// Resolve base image URL with architecture awareness
    /// </summary>
    private async Task<string> ResolveImageUrlAsync(string? imageId, string? providedUrl, CancellationToken ct)
    {
        // If URL provided directly, use it (orchestrator override)
        if (!string.IsNullOrEmpty(providedUrl))
        {
            _logger.LogInformation("Using provided image URL: {Url}", providedUrl);
            return providedUrl;
        }

        // Get host architecture
        var inventory = await _resourceDiscovery.GetInventoryCachedAsync(ct);
        var hostArchitecture = inventory?.Cpu?.Architecture ?? "x86_64";

        _logger.LogInformation("Resolving image for architecture: {Architecture}", hostArchitecture);

        // Normalize architecture to archTag (amd64/arm64)
        var archConfig = ArchitectureHelper.GetHostArchConfig(hostArchitecture);
        var archTag = archConfig.ArchTag;

        // Default to ubuntu-22.04 if no image specified
        var imageIdToResolve = imageId ?? "ubuntu-22.04";

        // Resolve from architecture-specific image map
        var resolvedUrl = ArchitectureHelper.ResolveImageUrl(archTag, imageIdToResolve);

        if (resolvedUrl == null)
        {
            _logger.LogWarning(
                "Image {ImageId} not available for {Architecture}, falling back to default",
                imageIdToResolve, archTag);

            // Fallback to ubuntu-22.04 for the architecture
            resolvedUrl = ArchitectureHelper.ResolveImageUrl(archTag, "ubuntu-22.04");
        }

        if (resolvedUrl == null)
        {
            throw new NotSupportedException(
                $"No images available for architecture {hostArchitecture}. " +
                $"Supported architectures: {string.Join(", ", ArchitectureHelper.SupportedArchitectures.Keys)}");
        }

        _logger.LogInformation(
            "Resolved image '{ImageId}' for {Architecture} ({ArchTag}): {Url}",
            imageIdToResolve, hostArchitecture, archTag, resolvedUrl);

        return resolvedUrl;
    }

    // Helper methods to handle case-insensitive property names
    private static string? GetStringProperty(JsonElement element, params string[] propertyNames)
    {
        foreach (var name in propertyNames)
        {
            if (element.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.String)
            {
                return prop.GetString();
            }
        }
        return null;
    }

    private static int? GetIntProperty(JsonElement element, params string[] propertyNames)
    {
        foreach (var name in propertyNames)
        {
            if (element.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.Number)
            {
                return prop.GetInt32();
            }
        }
        return null;
    }

    private static long? GetLongProperty(JsonElement element, params string[] propertyNames)
    {
        foreach (var name in propertyNames)
        {
            if (element.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.Number)
            {
                return prop.GetInt64();
            }
        }
        return null;
    }
}