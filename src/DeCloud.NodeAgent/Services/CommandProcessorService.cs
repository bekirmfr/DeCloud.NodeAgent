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

    // Default base image URLs
    private static readonly Dictionary<string, string> ImageUrls = new()
    {
        ["ubuntu-24.04"] = "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img",
        ["ubuntu-22.04"] = "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img",
        ["debian-12"] = "https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2"
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

        foreach (var command in commands)
        {
            _logger.LogInformation("Processing command {CommandId}: {Type}",
                command.CommandId, command.Type);

            try
            {
                var success = await ExecuteCommandAsync(command, ct);
                await _orchestratorClient.AcknowledgeCommandAsync(
                    command.CommandId, success, null, ct);
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

            var vmId = root.GetProperty("VmId").GetString() ?? Guid.NewGuid().ToString();
            var name = root.GetProperty("Name").GetString() ?? vmId;

            // Parse spec
            var spec = root.GetProperty("Spec");
            var cpuCores = spec.GetProperty("cpuCores").GetInt32();
            var memoryMb = spec.GetProperty("memoryMb").GetInt64();
            var diskGb = spec.GetProperty("diskGb").GetInt64();
            var imageId = spec.GetProperty("imageId").GetString() ?? "ubuntu-22.04";

            // Get SSH key if provided
            string? sshPublicKey = null;
            if (spec.TryGetProperty("sshPublicKey", out var sshKeyProp) &&
                sshKeyProp.ValueKind == JsonValueKind.String)
            {
                sshPublicKey = sshKeyProp.GetString();
            }

            // Get image URL
            var imageUrl = spec.TryGetProperty("imageUrl", out var urlProp) &&
                          urlProp.ValueKind == JsonValueKind.String &&
                          !string.IsNullOrEmpty(urlProp.GetString())
                ? urlProp.GetString()!
                : ImageUrls.GetValueOrDefault(imageId, ImageUrls["ubuntu-22.04"]);

            var vmSpec = new VmSpec
            {
                VmId = vmId,
                Name = name,
                VCpus = cpuCores,
                MemoryBytes = memoryMb * 1024 * 1024,
                DiskBytes = diskGb * 1024 * 1024 * 1024,
                BaseImageUrl = imageUrl,
                BaseImageHash = "",
                SshPublicKey = sshPublicKey,
                TenantId = "orchestrator",
                LeaseId = vmId
            };

            _logger.LogInformation("Creating VM {VmId}: {VCpus} vCPUs, {MemoryMb}MB RAM, image: {ImageId}",
                vmId, cpuCores, memoryMb, imageId);

            var result = await _vmManager.CreateVmAsync(vmSpec, ct);

            if (result.Success)
            {
                _logger.LogInformation("VM {VmId} created, starting...", vmId);
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
        var vmId = doc.RootElement.GetProperty("VmId").GetString();
        if (string.IsNullOrEmpty(vmId)) return false;

        var result = await _vmManager.StartVmAsync(vmId, ct);
        return result.Success;
    }

    private async Task<bool> HandleStopVmAsync(string payload, CancellationToken ct)
    {
        using var doc = JsonDocument.Parse(payload);
        var vmId = doc.RootElement.GetProperty("VmId").GetString();
        var force = doc.RootElement.TryGetProperty("Force", out var forceProp) && forceProp.GetBoolean();
        if (string.IsNullOrEmpty(vmId)) return false;

        var result = await _vmManager.StopVmAsync(vmId, force, ct);
        return result.Success;
    }

    private async Task<bool> HandleDeleteVmAsync(string payload, CancellationToken ct)
    {
        using var doc = JsonDocument.Parse(payload);
        var vmId = doc.RootElement.GetProperty("VmId").GetString();
        if (string.IsNullOrEmpty(vmId)) return false;

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
}