using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.NodeAgent.Infrastructure.Services;
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
    private readonly INodeMetadataService _nodeMetadata;
    private readonly INodeStateService _nodeState;
    private readonly IPortPoolManager _portPoolManager;
    private readonly IPortForwardingManager _portForwardingManager;
    private readonly PortMappingRepository _portMappingRepository;
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
        INodeMetadataService nodeMetadata,
        INodeStateService nodeState,
        IPortPoolManager portPoolManager,
        IPortForwardingManager portForwardingManager,
        PortMappingRepository portMappingRepository,
        IOptions<CommandProcessorOptions> options,
        ILogger<CommandProcessorService> logger)
    {
        _orchestratorClient = orchestratorClient;
        _pushedCommands = pushedCommands;
        _vmManager = vmManager;
        _resourceDiscovery = resourceDiscovery;
        _natRuleManager = natRuleManager;
        _nodeMetadata = nodeMetadata;
        _nodeState = nodeState;
        _portPoolManager = portPoolManager;
        _portForwardingManager = portForwardingManager;
        _portMappingRepository = portMappingRepository;
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

        // Exponential backoff for polling when idle
        var currentPollInterval = _options.PollInterval;
        var maxPollInterval = TimeSpan.FromSeconds(30);
        var consecutiveEmptyPolls = 0;

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
                    
                    // Reset backoff when commands are processed
                    currentPollInterval = _options.PollInterval;
                    consecutiveEmptyPolls = 0;
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
                    
                    // Reset backoff when commands found
                    currentPollInterval = _options.PollInterval;
                    consecutiveEmptyPolls = 0;
                }
                else
                {
                    // No commands found - increase backoff
                    consecutiveEmptyPolls++;
                    if (consecutiveEmptyPolls > 3)
                    {
                        // Exponential backoff up to max
                        currentPollInterval = TimeSpan.FromSeconds(
                            Math.Min(
                                currentPollInterval.TotalSeconds * 1.5,
                                maxPollInterval.TotalSeconds
                            )
                        );
                    }
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

            // Use current poll interval (with backoff)
            await Task.Delay(currentPollInterval, stoppingToken);
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
            var (success, data) = await ExecuteCommandAsync(command, ct);

            if (command.RequiresAck)
            {
                await _orchestratorClient.AcknowledgeCommandAsync(
                    command.CommandId, success, null, data, ct);
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
                    command.CommandId, false, ex.Message, null, ct);
            }
        }
    }

    private async Task<(bool success, string? data)> ExecuteCommandAsync(PendingCommand command, CancellationToken ct)
    {
        switch (command.Type)
        {
            case CommandType.CreateVm:
                return (await HandleCreateVmAsync(command.Payload, ct), null);

            case CommandType.StartVm:
                return (await HandleStartVmAsync(command.Payload, ct), null);

            case CommandType.StopVm:
                return (await HandleStopVmAsync(command.Payload, ct), null);

            case CommandType.DeleteVm:
                return (await HandleDeleteVmAsync(command.Payload, ct), null);

            case CommandType.Benchmark:
                return (await HandleBenchmarkAsync(ct), null);

            case CommandType.AllocatePort:
                return await HandleAllocatePortAsync(command.Payload, ct);

            case CommandType.RemovePort:
                return (await HandleRemovePortAsync(command.Payload, ct), null);

            default:
                _logger.LogWarning("Unknown command type: {Type}", command.Type);
                return (false, null);
        }
    }

    private async Task<bool> HandleCreateVmAsync(string payload, CancellationToken ct)
    {
        // ========================================
        // INITIALIZATION CHECK
        // ========================================
        // Refuse VM creation commands until node has received SchedulingConfig and PerformanceEvaluation
        // from orchestrator. This prevents crashes when targeted deployment sends commands before first registration/heartbeat.
        if (!_nodeState.IsFullyInitialized)
        {
            _logger.LogWarning(
                "❌ Refusing CreateVM command: Node not fully initialized. " +
                "Waiting for SchedulingConfig (v{ConfigVersion}) and PerformanceEvaluation ({PerfStatus}) from orchestrator. " +
                "This command will be retried.",
                _nodeState.SchedulingConfig?.Version ?? 0,
                _nodeState.PerformanceEvaluation != null ? "received" : "pending");
            return false; // Command will be retried by orchestrator
        }

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
        string? userData = GetStringProperty(root, "userData", "UserData");

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
            CloudInitUserData = userData,
            Labels = labels
        };

        _logger.LogInformation(
            "Creating {VmType} type VM {VmId}: {VCpus} vCPUs, {MemoryMB}MB RAM, " +
            "{DiskGB}GB disk, image: {ImageUrl}, quality tier: {QualityTier}",
            vmSpec.VmType.ToString(), vmId, virtualCpuCores,
            memoryBytes / 1024 / 1024, diskBytes / 1024 / 1024 / 1024,
            baseImageUrl, qualityTier);

        if (!string.IsNullOrEmpty(userData))
        {
            _logger.LogInformation(
                "VM {VmId}: Using cloud-init UserData ({Bytes} bytes)",
                vmId, userData.Length);
        }

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

    /// <summary>
    /// Handle AllocatePort command for Smart Port Allocation.
    /// Payload format: { "VmId": "...", "VmPrivateIp": "...", "VmPort": 22, "Protocol": 1, "Label": "SSH" }
    /// </summary>
    private async Task<(bool success, string? data)> HandleAllocatePortAsync(string payload, CancellationToken ct)
    {
        try
        {
            using var doc = JsonDocument.Parse(payload);
            var root = doc.RootElement;

            string? vmId = GetStringProperty(root, "vmId", "VmId");
            string? vmPrivateIp = GetStringProperty(root, "vmPrivateIp", "VmPrivateIp");
            int? vmPort = GetIntProperty(root, "vmPort", "VmPort");
            int? protocolInt = GetIntProperty(root, "protocol", "Protocol");
            string? label = GetStringProperty(root, "label", "Label");

            if (string.IsNullOrEmpty(vmId) || string.IsNullOrEmpty(vmPrivateIp) || vmPort == null || protocolInt == null)
            {
                _logger.LogError("Invalid AllocatePort payload: missing required fields");
                return (false, null);
            }

            var protocol = (PortProtocol)protocolInt.Value;

            _logger.LogInformation(
                "Allocating port for VM {VmId} ({VmIp}:{VmPort}) - {Protocol}",
                vmId, vmPrivateIp, vmPort, protocol);

            // Allocate public port from pool
            var publicPort = await _portPoolManager.AllocatePortAsync(ct);
            if (publicPort == null)
            {
                _logger.LogError("Port pool exhausted - cannot allocate port for VM {VmId}", vmId);
                return (false, null);
            }

            // Create port mapping in database
            var mapping = new PortMapping
            {
                VmId = vmId,
                VmPrivateIp = vmPrivateIp,
                VmPort = vmPort.Value,
                PublicPort = publicPort.Value,
                Protocol = protocol,
                Label = label
            };

            var added = await _portMappingRepository.AddAsync(mapping);
            if (!added)
            {
                _logger.LogError("Failed to save port mapping to database");
                await _portPoolManager.ReleasePortAsync(publicPort.Value, ct);
                return (false, null);
            }

            // Create iptables forwarding rules
            var success = await _portForwardingManager.CreateForwardingAsync(
                vmPrivateIp,
                vmPort.Value,
                publicPort.Value,
                protocol,
                ct);

            if (!success)
            {
                _logger.LogError("Failed to create iptables rules");
                await _portMappingRepository.RemoveAsync(vmId, vmPort.Value);
                await _portPoolManager.ReleasePortAsync(publicPort.Value, ct);
                return (false, null);
            }

            _logger.LogInformation(
                "✓ Port allocated: {PublicPort} → {VmIp}:{VmPort} (VM {VmId})",
                publicPort.Value, vmPrivateIp, vmPort.Value, vmId);

            // Create acknowledgment data with allocated port info
            var ackData = JsonSerializer.Serialize(new
            {
                VmPort = vmPort.Value,
                PublicPort = publicPort.Value,
                Protocol = (int)protocol
            });

            return (true, ackData);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling AllocatePort command");
            return (false, null);
        }
    }

    /// <summary>
    /// Handle RemovePort command for Smart Port Allocation.
    /// Payload format: { "VmId": "...", "VmPort": 22, "Protocol": 1 }
    /// </summary>
    private async Task<bool> HandleRemovePortAsync(string payload, CancellationToken ct)
    {
        try
        {
            using var doc = JsonDocument.Parse(payload);
            var root = doc.RootElement;

            string? vmId = GetStringProperty(root, "vmId", "VmId");
            int? vmPort = GetIntProperty(root, "vmPort", "VmPort");
            int? protocolInt = GetIntProperty(root, "protocol", "Protocol");

            if (string.IsNullOrEmpty(vmId) || vmPort == null || protocolInt == null)
            {
                _logger.LogError("Invalid RemovePort payload: missing required fields");
                return false;
            }

            var protocol = (PortProtocol)protocolInt.Value;

            _logger.LogInformation(
                "Removing port mapping for VM {VmId} port {VmPort} ({Protocol})",
                vmId, vmPort, protocol);

            // Get mapping to find public port
            var mappings = await _portMappingRepository.GetByVmIdAsync(vmId);
            var mapping = mappings.FirstOrDefault(m => m.VmPort == vmPort.Value);

            if (mapping == null)
            {
                _logger.LogWarning("Port mapping not found for VM {VmId} port {VmPort}", vmId, vmPort);
                return false;
            }

            // Remove iptables rules
            var success = await _portForwardingManager.RemoveForwardingAsync(
                mapping.PublicPort,
                protocol,
                ct);

            if (!success)
            {
                _logger.LogWarning("Failed to remove iptables rules (may not exist)");
            }

            // Remove from database
            var removed = await _portMappingRepository.RemoveAsync(vmId, vmPort.Value);
            if (!removed)
            {
                _logger.LogError("Failed to remove port mapping from database");
                return false;
            }

            // Release port back to pool
            await _portPoolManager.ReleasePortAsync(mapping.PublicPort, ct);

            _logger.LogInformation(
                "✓ Port mapping removed: {PublicPort} → VM {VmId}:{VmPort}",
                mapping.PublicPort, vmId, vmPort.Value);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling RemovePort command");
            return false;
        }
    }
}