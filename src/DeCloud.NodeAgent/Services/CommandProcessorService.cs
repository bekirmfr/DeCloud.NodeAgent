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
    private readonly VmRepository _repository;
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
        VmRepository repository,
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
        _repository = repository;
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

        // Parse service definitions from orchestrator payload
        if (result.Success)
        {
            var vm = (await _vmManager.GetAllVmsAsync(ct)).FirstOrDefault(v => v.VmId == vmId);
            if (vm != null)
            {
                vm.Services = ParseServiceDefinitions(root);
                await _repository.SaveVmAsync(vm);
                _logger.LogInformation("VM {VmId}: {Count} service readiness checks registered",
                    vmId, vm.Services.Count);
            }
        }

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

    /// <summary>
    /// Parse service definitions from the orchestrator's CreateVm command payload.
    /// Falls back to default System-only service if none provided.
    /// Uses GetIntProperty/GetStringProperty helpers for null-safe JSON parsing.
    /// </summary>
    private static List<VmServiceStatus> ParseServiceDefinitions(JsonElement root)
    {
        var services = new List<VmServiceStatus>();

        if (root.TryGetProperty("Services", out var servicesElement) ||
            root.TryGetProperty("services", out servicesElement))
        {
            if (servicesElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var svcElement in servicesElement.EnumerateArray())
                {
                    // Use existing helpers — they check ValueKind before calling
                    // GetInt32/GetString, preventing crashes on null JSON values
                    var name = GetStringProperty(svcElement, "Name", "name");
                    var port = GetIntProperty(svcElement, "Port", "port");
                    var protocol = GetStringProperty(svcElement, "Protocol", "protocol");

                    var checkTypeStr = GetStringProperty(svcElement, "CheckType", "checkType") ?? "CloudInitDone";
                    Enum.TryParse<CheckType>(checkTypeStr, true, out var checkType);

                    var httpPath = GetStringProperty(svcElement, "HttpPath", "httpPath");
                    var execCommand = GetStringProperty(svcElement, "ExecCommand", "execCommand");
                    var timeout = GetIntProperty(svcElement, "TimeoutSeconds", "timeoutSeconds") ?? 300;

                    services.Add(new VmServiceStatus
                    {
                        Name = name ?? "Unknown",
                        Port = port,
                        Protocol = protocol,
                        CheckType = checkType,
                        HttpPath = httpPath,
                        ExecCommand = execCommand,
                        Status = ServiceReadiness.Pending,
                        TimeoutSeconds = timeout
                    });
                }
            }
        }

        // Default: at least a System service
        if (services.Count == 0)
        {
            services.Add(new VmServiceStatus
            {
                Name = "System",
                CheckType = CheckType.CloudInitDone,
                Status = ServiceReadiness.Pending,
                TimeoutSeconds = 300
            });
        }

        return services;
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
        
        // Clean up relay VM NAT rules
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

        // Clean up Direct Access ports (Smart Port Allocation)
        try
        {
            var portMappings = await _portMappingRepository.GetByVmIdAsync(vmId);
            if (portMappings.Any())
            {
                _logger.LogInformation(
                    "Cleaning up {Count} Direct Access port(s) for VM {VmId}",
                    portMappings.Count, vmId);

                foreach (var mapping in portMappings)
                {
                    _logger.LogInformation(
                        "Removing port {PublicPort} → {Destination}:{VmPort} (VM {VmId})",
                        mapping.PublicPort, mapping.VmPrivateIp, mapping.VmPort, vmId);

                    // Remove iptables rules
                    try
                    {
                        await _portForwardingManager.RemoveForwardingAsync(
                            mapping.VmPrivateIp,  // Could be VM IP or tunnel IP for relay forwarding
                            mapping.VmPort,
                            mapping.PublicPort,
                            mapping.Protocol,
                            ct);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex,
                            "Failed to remove iptables rules for port {PublicPort}, continuing cleanup",
                            mapping.PublicPort);
                    }

                    // Release port back to pool
                    try
                    {
                        await _portPoolManager.ReleasePortAsync(mapping.PublicPort, ct);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex,
                            "Failed to release port {PublicPort} back to pool",
                            mapping.PublicPort);
                    }

                    // Remove from database
                    await _portMappingRepository.RemoveAsync(vmId, mapping.VmPort);
                }

                _logger.LogInformation(
                    "✓ Cleaned up {Count} Direct Access port(s) for VM {VmId}",
                    portMappings.Count, vmId);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error cleaning up Direct Access ports for VM {VmId}", vmId);
            // Continue with VM deletion even if port cleanup fails
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
            
            // NEW: Relay forwarding fields
            bool? isRelayForwarding = root.TryGetProperty("isRelayForwarding", out var relayProp) || root.TryGetProperty("IsRelayForwarding", out relayProp) 
                ? relayProp.GetBoolean() 
                : false;
            string? tunnelDestinationIp = GetStringProperty(root, "tunnelDestinationIp", "TunnelDestinationIp");

            if (string.IsNullOrEmpty(vmId) || string.IsNullOrEmpty(vmPrivateIp) || vmPort == null || protocolInt == null)
            {
                _logger.LogError("Invalid AllocatePort payload: missing required fields");
                return (false, null);
            }

            var protocol = (PortProtocol)protocolInt.Value;

            // Determine forwarding destination
            string forwardingDestination;
            string forwardingType;
            
            if (isRelayForwarding == true && !string.IsNullOrEmpty(tunnelDestinationIp))
            {
                // Relay node - forward to CGNAT node via tunnel
                forwardingDestination = tunnelDestinationIp;
                forwardingType = "relay→tunnel";
                
                _logger.LogInformation(
                    "Relay forwarding for CGNAT VM {VmId}: will forward to tunnel {TunnelIp}:{VmPort}",
                    vmId, tunnelDestinationIp, vmPort);
            }
            else
            {
                // Direct access - forward to local VM
                forwardingDestination = vmPrivateIp;
                forwardingType = "direct→vm";
                
                _logger.LogInformation(
                    "Direct forwarding for VM {VmId}: will forward to {VmIp}:{VmPort}",
                    vmId, vmPrivateIp, vmPort);
            }

            _logger.LogInformation(
                "Allocating port for VM {VmId} ({Destination}:{VmPort}) - {Protocol} [{Type}]",
                vmId, forwardingDestination, vmPort, protocol, forwardingType);

            // Check if a specific public port was requested (for 3-hop CGNAT forwarding)
            int? requestedPublicPort = GetIntProperty(root, "publicPort", "PublicPort");
            
            int? publicPort;
            if (requestedPublicPort.HasValue && requestedPublicPort.Value > 0)
            {
                // Use the specified port (orchestrator is coordinating multi-hop forwarding)
                _logger.LogInformation(
                    "Using specified public port {PublicPort} for VM {VmId} (3-hop forwarding)",
                    requestedPublicPort.Value, vmId);
                    
                // Reserve this specific port in the pool
                bool reserved = await _portPoolManager.ReserveSpecificPortAsync(requestedPublicPort.Value, ct);
                if (!reserved)
                {
                    _logger.LogError(
                        "Cannot allocate requested port {PublicPort} for VM {VmId} - port already in use or out of range",
                        requestedPublicPort.Value, vmId);
                    return (false, null);
                }
                
                publicPort = requestedPublicPort.Value;
            }
            else
            {
                // Allocate any available port from pool (normal Direct Access)
                publicPort = await _portPoolManager.AllocatePortAsync(ct);
                if (publicPort == null)
                {
                    _logger.LogError("Port pool exhausted - cannot allocate port for VM {VmId}", vmId);
                    return (false, null);
                }
            }

            // Create port mapping in database
            var mapping = new PortMapping
            {
                VmId = vmId,
                VmPrivateIp = forwardingDestination,  // Store actual forwarding destination
                VmPort = vmPort.Value,
                PublicPort = publicPort.Value,
                Protocol = protocol,
                Label = $"{label} ({forwardingType})"  // Include forwarding type in label
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
                forwardingDestination,  // Tunnel IP for relay, VM IP for direct
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
                "✓ Port allocated: {PublicPort} → {Destination}:{VmPort} (VM {VmId}) [{Type}]",
                publicPort.Value, forwardingDestination, vmPort.Value, vmId, forwardingType);

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
            int? publicPort = GetIntProperty(root, "publicPort", "PublicPort");
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
            
            // For relay forwarding (VmPort=0), match by PublicPort instead
            // All relay mappings have VmPort=0, so matching by VmPort would return first mapping
            PortMapping? mapping;
            if (vmPort.Value == 0 && publicPort.HasValue)
            {
                mapping = mappings.FirstOrDefault(m => m.PublicPort == publicPort.Value);
                _logger.LogDebug(
                    "Relay port removal - matching by PublicPort {PublicPort}",
                    publicPort.Value);
            }
            else
            {
                mapping = mappings.FirstOrDefault(m => m.VmPort == vmPort.Value);
            }

            if (mapping == null)
            {
                _logger.LogWarning("Port mapping not found for VM {VmId} port {VmPort}", vmId, vmPort);
                return false;
            }


            // Remove iptables rules (both DNAT and FORWARD)
            // For relay forwarding, iptables rules use relay VM IP, not tunnel IP
            string ipForIptables = mapping.VmPrivateIp;
            
            // Check if this is relay forwarding (tunnel IP)
            if (mapping.VmPrivateIp.StartsWith("10.20.") || mapping.VmPrivateIp.StartsWith("10.30."))
            {
                // This is a tunnel IP - need to resolve relay VM IP for iptables deletion
                var relayVmIp = await _portForwardingManager.GetRelayVmIpAsync(ct);
                if (relayVmIp != null)
                {
                    ipForIptables = relayVmIp;
                    _logger.LogInformation(
                        "Detected relay forwarding - using relay VM IP {RelayVmIp} instead of tunnel IP {TunnelIp}",
                        relayVmIp, mapping.VmPrivateIp);
                }
                else
                {
                    _logger.LogWarning(
                        "Tunnel IP detected but no relay VM found - deletion may fail");
                }
            }
            
            var success = await _portForwardingManager.RemoveForwardingAsync(
                ipForIptables,
                mapping.VmPort,
                mapping.PublicPort,
                protocol,
                ct);

            if (!success)
            {
                _logger.LogWarning("Failed to remove iptables rules (may not exist)");
            }

            // Remove from database
            // For relay mappings (VmPort=0), remove by PublicPort to avoid deleting all relay mappings
            bool removed;
            if (vmPort.Value == 0)
            {
                removed = await _portMappingRepository.RemoveByPublicPortAsync(mapping.PublicPort);
                _logger.LogDebug("Removed relay mapping by PublicPort {PublicPort}", mapping.PublicPort);
            }
            else
            {
                removed = await _portMappingRepository.RemoveAsync(vmId, vmPort.Value);
            }
            
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