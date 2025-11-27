using System.Text.Json;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Services;

public class CommandProcessorOptions
{
    public TimeSpan PollInterval { get; set; } = TimeSpan.FromSeconds(5);
}

public class CommandProcessorService : BackgroundService
{
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly IVmManager _vmManager;
    private readonly IResourceDiscoveryService _resourceDiscovery;
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
        IVmManager vmManager,
        IResourceDiscoveryService resourceDiscovery,
        IOptions<CommandProcessorOptions> options,
        ILogger<CommandProcessorService> logger)
    {
        _orchestratorClient = orchestratorClient;
        _vmManager = vmManager;
        _resourceDiscovery = resourceDiscovery;
        _logger = logger;
        _options = options.Value;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Command processor starting with poll interval {Interval}s",
            _options.PollInterval.TotalSeconds);

        // Wait for heartbeat service to initialize and register
        await Task.Delay(TimeSpan.FromSeconds(15), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await ProcessPendingCommandsAsync(stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing commands");
            }

            await Task.Delay(_options.PollInterval, stoppingToken);
        }
    }

    private async Task ProcessPendingCommandsAsync(CancellationToken ct)
    {
        var commands = await _orchestratorClient.GetPendingCommandsAsync(ct);

        if (commands.Count > 0)
        {
            _logger.LogInformation("Processing {Count} pending command(s)", commands.Count);
        }

        foreach (var command in commands)
        {
            _logger.LogInformation("Processing command {CommandId}: {Type}",
                command.CommandId, command.Type);

            try
            {
                var success = await ExecuteCommandAsync(command, ct);
                await _orchestratorClient.AcknowledgeCommandAsync(
                    command.CommandId, success, null, ct);

                _logger.LogInformation("Command {CommandId} completed: {Success}", command.CommandId, success);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Command {CommandId} failed", command.CommandId);
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
        try
        {
            _logger.LogInformation("Handling CreateVm command: {Payload}", payload);

            using var doc = JsonDocument.Parse(payload);
            var root = doc.RootElement;

            // Parse the new flat format from Orchestrator
            var vmId = GetStringProperty(root, "VmId", "vmId") ?? Guid.NewGuid().ToString();
            var name = GetStringProperty(root, "Name", "name") ?? vmId;

            // Try new flat format first, then fall back to nested Spec format
            int cpuCores;
            long memoryBytes;
            long diskBytes;
            string? imageUrl;
            string? imageId = null;
            string? sshPublicKey;
            string? tenantId;
            string? leaseId;

            if (root.TryGetProperty("VCpus", out _) || root.TryGetProperty("MemoryBytes", out _))
            {
                // New flat format from updated Orchestrator
                cpuCores = GetIntProperty(root, "VCpus", "vCpus") ?? 1;
                memoryBytes = GetLongProperty(root, "MemoryBytes", "memoryBytes") ?? 1024L * 1024L * 1024L;
                diskBytes = GetLongProperty(root, "DiskBytes", "diskBytes") ?? 10L * 1024L * 1024L * 1024L;
                imageUrl = GetStringProperty(root, "BaseImageUrl", "baseImageUrl");
                sshPublicKey = GetStringProperty(root, "SshPublicKey", "sshPublicKey");
                tenantId = GetStringProperty(root, "TenantId", "tenantId") ?? "orchestrator";
                leaseId = GetStringProperty(root, "LeaseId", "leaseId") ?? vmId;

                _logger.LogInformation("Parsed flat format: {VCpus} vCPUs, {MemoryMB}MB, imageUrl={ImageUrl}",
                    cpuCores, memoryBytes / 1024 / 1024, imageUrl ?? "null");
            }
            else if (root.TryGetProperty("Spec", out var spec))
            {
                // Old nested format
                cpuCores = GetIntProperty(spec, "cpuCores", "CpuCores") ?? 1;
                var memoryMb = GetLongProperty(spec, "memoryMb", "MemoryMb") ?? 1024;
                var diskGb = GetLongProperty(spec, "diskGb", "DiskGb") ?? 10;
                memoryBytes = memoryMb * 1024 * 1024;
                diskBytes = diskGb * 1024 * 1024 * 1024;
                imageId = GetStringProperty(spec, "imageId", "ImageId");
                imageUrl = GetStringProperty(spec, "imageUrl", "ImageUrl");
                sshPublicKey = GetStringProperty(spec, "sshPublicKey", "SshPublicKey");
                tenantId = "orchestrator";
                leaseId = vmId;

                _logger.LogInformation("Parsed nested Spec format: {VCpus} vCPUs, {MemoryMB}MB, imageId={ImageId}",
                    cpuCores, memoryMb, imageId ?? "null");
            }
            else
            {
                _logger.LogError("Unknown payload format - neither flat nor nested Spec found");
                return false;
            }

            // Resolve image URL if not provided directly
            if (string.IsNullOrEmpty(imageUrl))
            {
                imageUrl = ImageUrls.GetValueOrDefault(imageId ?? "ubuntu-22.04", ImageUrls["ubuntu-22.04"]);
                _logger.LogInformation("Resolved imageId '{ImageId}' to URL: {ImageUrl}", imageId, imageUrl);
            }

            var vmSpec = new VmSpec
            {
                VmId = vmId,
                Name = name,
                VCpus = cpuCores,
                MemoryBytes = memoryBytes,
                DiskBytes = diskBytes,
                BaseImageUrl = imageUrl,
                BaseImageHash = "",
                SshPublicKey = sshPublicKey,
                TenantId = tenantId,
                LeaseId = leaseId
            };

            _logger.LogInformation("Creating VM {VmId}: {VCpus} vCPUs, {MemoryMB}MB RAM, {DiskGB}GB disk, image: {ImageUrl}, SSH key: {HasKey}",
                vmId, cpuCores, memoryBytes / 1024 / 1024, diskBytes / 1024 / 1024 / 1024, imageUrl,
                !string.IsNullOrEmpty(sshPublicKey) ? "yes" : "no");

            var result = await _vmManager.CreateVmAsync(vmSpec, ct);

            if (result.Success)
            {
                _logger.LogInformation("VM {VmId} created successfully, starting...", vmId);
                var startResult = await _vmManager.StartVmAsync(vmId, ct);

                if (startResult.Success)
                {
                    _logger.LogInformation("VM {VmId} started successfully", vmId);
                    return true;
                }

                _logger.LogError("Failed to start VM {VmId}: {Error}", vmId, startResult.ErrorMessage);
                return false;
            }

            _logger.LogError("CreateVm failed: {Error}", result.ErrorMessage);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling CreateVm command");
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