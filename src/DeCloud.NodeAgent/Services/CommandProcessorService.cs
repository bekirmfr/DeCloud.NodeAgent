using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Docker;
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
    private readonly DockerContainerManager _dockerManager;
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly INatRuleManager _natRuleManager;
    private readonly INodeMetadataService _nodeMetadata;
    private readonly INodeStateService _nodeState;
    private readonly IPortPoolManager _portPoolManager;
    private readonly IPortForwardingManager _portForwardingManager;
    private readonly PortMappingRepository _portMappingRepository;
    private readonly VmRepository _repository;
    private readonly ICommandExecutor _executor;
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
        DockerContainerManager dockerManager,
        IResourceDiscoveryService resourceDiscovery,
        INatRuleManager natRuleManager,
        INodeMetadataService nodeMetadata,
        INodeStateService nodeState,
        IPortPoolManager portPoolManager,
        IPortForwardingManager portForwardingManager,
        PortMappingRepository portMappingRepository,
        VmRepository repository,
        ICommandExecutor executor,
        IOptions<CommandProcessorOptions> options,
        ILogger<CommandProcessorService> logger)
    {
        _orchestratorClient = orchestratorClient;
        _pushedCommands = pushedCommands;
        _vmManager = vmManager;
        _dockerManager = dockerManager;
        _resourceDiscovery = resourceDiscovery;
        _natRuleManager = natRuleManager;
        _nodeMetadata = nodeMetadata;
        _nodeState = nodeState;
        _portPoolManager = portPoolManager;
        _portForwardingManager = portForwardingManager;
        _portMappingRepository = portMappingRepository;
        _repository = repository;
        _executor = executor;
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

            case CommandType.ConfigureGpu:
                return await HandleConfigureGpuAsync(command.Payload, ct);

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
        string? gpuPciAddress = GetStringProperty(root, "gpuPciAddress", "GpuPciAddress");
        int deploymentModeInt = GetIntProperty(root, "deploymentMode", "DeploymentMode") ?? 0;
        var deploymentMode = (DeploymentMode)deploymentModeInt;
        string? containerImage = GetStringProperty(root, "containerImage", "ContainerImage");

        // Parse environment variables for containers
        Dictionary<string, string>? environmentVariables = null;
        if (root.TryGetProperty("EnvironmentVariables", out var envElement) ||
            root.TryGetProperty("environmentVariables", out envElement))
        {
            environmentVariables = JsonSerializer.Deserialize<Dictionary<string, string>>(
                envElement.GetRawText());
        }

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
            GpuPciAddress = gpuPciAddress,
            DeploymentMode = deploymentMode,
            ContainerImage = containerImage,
            EnvironmentVariables = environmentVariables,
            Labels = labels
        };

        // Defense-in-depth: reject duplicate system VMs (DHT/Relay) with the same name.
        // The orchestrator's reconciliation loop can race and issue two CreateVm commands
        // for the same role before the first one is acknowledged.
        if (vmSpec.VmType is VmType.Dht or VmType.Relay)
        {
            var existingVms = await _vmManager.GetAllVmsAsync(ct);
            var duplicate = existingVms.FirstOrDefault(v =>
                v.Name == name && v.VmId != vmId &&
                v.State is not (VmState.Deleted or VmState.Failed));
            if (duplicate != null)
            {
                _logger.LogWarning(
                    "Rejecting duplicate {VmType} VM {VmId} — a VM with the same name already exists: {ExistingVmId} (state: {State})",
                    vmSpec.VmType, vmId, duplicate.VmId, duplicate.State);
                return true; // Return true to ACK and prevent retry
            }
        }

        _logger.LogInformation(
            "Creating {VmType} type {Mode} {VmId}: {VCpus} vCPUs, {MemoryMB}MB RAM, " +
            "{DiskGB}GB disk, image: {ImageUrl}, quality tier: {QualityTier}",
            vmSpec.VmType.ToString(), deploymentMode, vmId, virtualCpuCores,
            memoryBytes / 1024 / 1024, diskBytes / 1024 / 1024 / 1024,
            deploymentMode == DeploymentMode.Container ? containerImage : baseImageUrl,
            qualityTier);

        if (deploymentMode == DeploymentMode.Container)
        {
            _logger.LogInformation(
                "VM {VmId}: Container mode - image: {Image}, GPU shared via NVIDIA runtime",
                vmId, containerImage);
        }
        else if (!string.IsNullOrEmpty(userData))
        {
            _logger.LogInformation(
                "VM {VmId}: Using cloud-init UserData ({Bytes} bytes)",
                vmId, userData.Length);
        }

        // Route to the correct manager based on deployment mode
        IVmManager manager = deploymentMode == DeploymentMode.Container
            ? _dockerManager
            : _vmManager;

        var result = await manager.CreateVmAsync(vmSpec, password, ct);

        // Parse service definitions from orchestrator payload
        if (result.Success)
        {
            var vm = (await manager.GetAllVmsAsync(ct)).FirstOrDefault(v => v.VmId == vmId);
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

        var manager = await ResolveManagerForVmAsync(vmId, ct);
        _logger.LogInformation("Starting VM {VmId}", vmId);
        var result = await manager.StartVmAsync(vmId, ct);
        return result.Success;
    }

    private async Task<bool> HandleStopVmAsync(string payload, CancellationToken ct)
    {
        using var doc = JsonDocument.Parse(payload);
        var vmId = GetStringProperty(doc.RootElement, "VmId", "vmId");
        var force = doc.RootElement.TryGetProperty("Force", out var forceProp) && forceProp.GetBoolean();
        if (string.IsNullOrEmpty(vmId)) return false;

        var manager = await ResolveManagerForVmAsync(vmId, ct);
        _logger.LogInformation("Stopping VM {VmId} (force={Force})", vmId, force);
        var result = await manager.StopVmAsync(vmId, force, ct);
        return result.Success;
    }

    private async Task<bool> HandleDeleteVmAsync(string payload, CancellationToken ct)
    {
        using var doc = JsonDocument.Parse(payload);
        var vmId = GetStringProperty(doc.RootElement, "VmId", "vmId");
        if (string.IsNullOrEmpty(vmId)) return false;

        var manager = await ResolveManagerForVmAsync(vmId, ct);
        var vmInstance = await manager.GetVmAsync(vmId, ct);
        
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
        var result = await manager.DeleteVmAsync(vmId, ct);
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

    /// <summary>
    /// Handle ConfigureGpu command from orchestrator.
    /// Installs NVIDIA drivers, Docker, and NVIDIA Container Toolkit as needed.
    /// Acknowledges with GPU capability status.
    /// </summary>
    private async Task<(bool success, string? data)> HandleConfigureGpuAsync(string payload, CancellationToken ct)
    {
        try
        {
            using var doc = JsonDocument.Parse(payload);
            var root = doc.RootElement;

            int modeInt = GetIntProperty(root, "mode", "Mode") ?? 0;
            var setupMode = (GpuSetupMode)modeInt;

            _logger.LogInformation("ConfigureGpu command received: mode={Mode}", setupMode);

            var containerSharingReady = false;
            var vfioPassthroughReady = false;
            var iommuEnabled = false;
            var rebootRequired = false;
            string? driverVersion = null;
            string? errorMessage = null;

            // Detect current state
            var inventory = await _resourceDiscovery.GetInventoryCachedAsync(ct);
            iommuEnabled = inventory?.Gpus.Any(g => g.IsIommuEnabled) ?? false;

            // ─── Container Toolkit setup (Auto or ContainerToolkit) ───
            if (setupMode is GpuSetupMode.Auto or GpuSetupMode.ContainerToolkit)
            {
                _logger.LogInformation("Setting up GPU container sharing (Docker + NVIDIA Container Toolkit)...");

                // Step 1: Verify nvidia-smi / drivers
                var smiResult = await FindAndRunNvidiaSmiAsync(ct);
                if (smiResult.success)
                {
                    driverVersion = smiResult.driverVersion;
                    _logger.LogInformation("NVIDIA driver detected: {Version}", driverVersion);
                }
                else
                {
                    // Try installing NVIDIA drivers
                    _logger.LogInformation("Installing NVIDIA drivers...");
                    var driverInstall = await _executor.ExecuteAsync(
                        "bash", "-c \"apt-get update -qq && apt-get install -y nvidia-driver-535 2>&1 || apt-get install -y nvidia-driver-550 2>&1 || true\"",
                        TimeSpan.FromMinutes(5), ct);

                    if (driverInstall.Success)
                    {
                        smiResult = await FindAndRunNvidiaSmiAsync(ct);
                        if (smiResult.success)
                        {
                            driverVersion = smiResult.driverVersion;
                        }
                        else
                        {
                            _logger.LogWarning("NVIDIA drivers installed but nvidia-smi still not working — reboot may be required");
                            rebootRequired = true;
                        }
                    }
                    else
                    {
                        _logger.LogWarning("NVIDIA driver installation failed: {Error}", driverInstall.StandardError);
                    }
                }

                // Step 2: Ensure Docker is installed and running
                var dockerReady = await EnsureDockerInstalledAsync(ct);

                // Step 3: Ensure NVIDIA Container Toolkit
                if (dockerReady && driverVersion != null)
                {
                    containerSharingReady = await EnsureNvidiaContainerToolkitAsync(ct);
                }
                else if (dockerReady && rebootRequired)
                {
                    _logger.LogInformation("Docker ready but NVIDIA driver requires reboot before Container Toolkit can work");
                }

                if (containerSharingReady)
                {
                    _logger.LogInformation("GPU container sharing ready (Docker + NVIDIA Container Toolkit)");
                }
            }

            // ─── VFIO Passthrough setup (Auto or VfioPassthrough) ───
            if (setupMode is GpuSetupMode.Auto or GpuSetupMode.VfioPassthrough)
            {
                if (iommuEnabled)
                {
                    _logger.LogInformation("IOMMU enabled — configuring VFIO passthrough...");
                    vfioPassthroughReady = await ConfigureVfioPassthroughAsync(ct);
                    if (vfioPassthroughReady)
                    {
                        _logger.LogInformation("VFIO passthrough configured (reboot may be needed to take effect)");
                    }
                }
                else
                {
                    _logger.LogInformation("IOMMU not enabled — skipping VFIO passthrough configuration");
                    if (setupMode == GpuSetupMode.VfioPassthrough)
                    {
                        errorMessage = "VFIO passthrough requested but IOMMU is not enabled. Enable IOMMU in BIOS and add intel_iommu=on or amd_iommu=on to kernel parameters.";
                    }
                }
            }

            // Force re-discovery so heartbeat picks up new capabilities
            await _resourceDiscovery.DiscoverAllAsync(ct);

            // Build ack data
            var ackData = JsonSerializer.Serialize(new
            {
                ContainerSharingReady = containerSharingReady,
                VfioPassthroughReady = vfioPassthroughReady,
                IommuEnabled = iommuEnabled,
                RebootRequired = rebootRequired,
                DriverVersion = driverVersion ?? "",
                ErrorMessage = errorMessage ?? ""
            });

            var overallSuccess = containerSharingReady || vfioPassthroughReady || rebootRequired;
            _logger.LogInformation(
                "ConfigureGpu complete: containerSharing={Container}, vfioPassthrough={Vfio}, reboot={Reboot}, driver={Driver}",
                containerSharingReady, vfioPassthroughReady, rebootRequired, driverVersion ?? "none");

            return (overallSuccess, ackData);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ConfigureGpu command failed");
            var errorAck = JsonSerializer.Serialize(new
            {
                ContainerSharingReady = false,
                VfioPassthroughReady = false,
                IommuEnabled = false,
                RebootRequired = false,
                DriverVersion = "",
                ErrorMessage = ex.Message
            });
            return (false, errorAck);
        }
    }

    // =========================================================================
    // GPU Setup Helpers
    // =========================================================================

    private async Task<(bool success, string? driverVersion)> FindAndRunNvidiaSmiAsync(CancellationToken ct)
    {
        // Check common nvidia-smi paths (same logic as ResourceDiscoveryService)
        var paths = new[]
        {
            "nvidia-smi",
            "/usr/bin/nvidia-smi",
            "/usr/lib/wsl/lib/nvidia-smi"
        };

        foreach (var path in paths)
        {
            var result = await _executor.ExecuteAsync(path, "--query-gpu=driver_version --format=csv,noheader,nounits", ct);
            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                return (true, result.StandardOutput.Trim().Split('\n')[0].Trim());
            }
        }

        return (false, null);
    }

    private async Task<bool> EnsureDockerInstalledAsync(CancellationToken ct)
    {
        // Check if already running
        var dockerInfo = await _executor.ExecuteAsync("docker", "info", ct);
        if (dockerInfo.Success)
        {
            _logger.LogInformation("Docker already installed and running");
            return true;
        }

        // Try starting if installed but not running
        var dockerVersion = await _executor.ExecuteAsync("docker", "--version", ct);
        if (dockerVersion.Success)
        {
            _logger.LogInformation("Docker installed but not running — starting...");
            await _executor.ExecuteAsync("systemctl", "enable docker --quiet", ct);
            await _executor.ExecuteAsync("systemctl", "start docker", ct);
            await Task.Delay(3000, ct);

            dockerInfo = await _executor.ExecuteAsync("docker", "info", ct);
            if (dockerInfo.Success) return true;
        }

        // Install Docker via convenience script
        _logger.LogInformation("Installing Docker...");
        var download = await _executor.ExecuteAsync(
            "bash", "-c \"curl -fsSL https://get.docker.com -o /tmp/get-docker.sh\"",
            TimeSpan.FromMinutes(2), ct);
        if (!download.Success)
        {
            _logger.LogError("Failed to download Docker install script");
            return false;
        }

        var install = await _executor.ExecuteAsync(
            "bash", "-c \"sh /tmp/get-docker.sh\"",
            TimeSpan.FromMinutes(10), ct);
        if (!install.Success)
        {
            _logger.LogError("Docker installation failed: {Error}", install.StandardError);
            return false;
        }

        await _executor.ExecuteAsync("systemctl", "enable docker --quiet", ct);
        await _executor.ExecuteAsync("systemctl", "start docker", ct);
        await Task.Delay(3000, ct);

        dockerInfo = await _executor.ExecuteAsync("docker", "info", ct);
        if (dockerInfo.Success)
        {
            _logger.LogInformation("Docker installed and running");
            return true;
        }

        _logger.LogError("Docker installed but failed to start");
        return false;
    }

    private async Task<bool> EnsureNvidiaContainerToolkitAsync(CancellationToken ct)
    {
        // Check if already configured
        var dockerInfo = await _executor.ExecuteAsync("docker", "info", ct);
        if (dockerInfo.Success && dockerInfo.StandardOutput.Contains("nvidia", StringComparison.OrdinalIgnoreCase))
        {
            _logger.LogInformation("NVIDIA Container Toolkit already configured");
            return true;
        }

        _logger.LogInformation("Installing NVIDIA Container Toolkit...");

        // Add GPG key (if not present)
        if (!File.Exists("/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"))
        {
            var gpgResult = await _executor.ExecuteAsync(
                "bash", "-c \"curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\"",
                TimeSpan.FromSeconds(30), ct);
            if (!gpgResult.Success)
            {
                _logger.LogError("Failed to add NVIDIA GPG key: {Error}", gpgResult.StandardError);
                return false;
            }
        }

        // Add repository (stable/deb format — works on Ubuntu 22.04+/24.04)
        var repoResult = await _executor.ExecuteAsync(
            "bash", "-c \"curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list\"",
            TimeSpan.FromSeconds(30), ct);
        if (!repoResult.Success)
        {
            _logger.LogError("Failed to add NVIDIA container toolkit repo: {Error}", repoResult.StandardError);
            return false;
        }

        // Install
        var aptUpdate = await _executor.ExecuteAsync("bash", "-c \"apt-get update -qq\"", TimeSpan.FromMinutes(2), ct);
        var installResult = await _executor.ExecuteAsync(
            "bash", "-c \"apt-get install -y nvidia-container-toolkit\"",
            TimeSpan.FromMinutes(5), ct);
        if (!installResult.Success)
        {
            _logger.LogError("Failed to install nvidia-container-toolkit: {Error}", installResult.StandardError);
            return false;
        }

        // Configure Docker runtime
        var configResult = await _executor.ExecuteAsync(
            "nvidia-ctk", "runtime configure --runtime=docker",
            TimeSpan.FromSeconds(30), ct);
        if (!configResult.Success)
        {
            _logger.LogError("Failed to configure NVIDIA runtime: {Error}", configResult.StandardError);
            return false;
        }

        // Restart Docker
        await _executor.ExecuteAsync("systemctl", "restart docker", ct);
        await Task.Delay(3000, ct);

        // Verify
        dockerInfo = await _executor.ExecuteAsync("docker", "info", ct);
        if (dockerInfo.Success && dockerInfo.StandardOutput.Contains("nvidia", StringComparison.OrdinalIgnoreCase))
        {
            _logger.LogInformation("NVIDIA Container Toolkit installed and configured");
            return true;
        }

        _logger.LogWarning("NVIDIA Container Toolkit installed but runtime not detected in Docker");
        return false;
    }

    private async Task<bool> ConfigureVfioPassthroughAsync(CancellationToken ct)
    {
        try
        {
            // Load VFIO kernel modules
            var modules = new[] { "vfio", "vfio_iommu_type1", "vfio_pci" };
            foreach (var mod in modules)
            {
                await _executor.ExecuteAsync("modprobe", mod, ct);
            }

            // Ensure modules load on boot
            var modulesConf = string.Join("\n", modules);
            var writeResult = await _executor.ExecuteAsync(
                "bash", $"-c \"echo '{modulesConf}' > /etc/modules-load.d/vfio.conf\"", ct);

            // Blacklist nouveau driver
            var blacklistResult = await _executor.ExecuteAsync(
                "bash", "-c \"echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf\"", ct);

            if (writeResult.Success && blacklistResult.Success)
            {
                // Update initramfs
                await _executor.ExecuteAsync(
                    "bash", "-c \"update-initramfs -u 2>/dev/null || dracut -f 2>/dev/null || true\"",
                    TimeSpan.FromMinutes(2), ct);

                _logger.LogInformation("VFIO passthrough configured (vfio modules + nouveau blacklisted)");
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to configure VFIO passthrough");
        }

        return false;
    }

    /// <summary>
    /// Resolve which manager owns a given VM (libvirt or Docker)
    /// </summary>
    private async Task<IVmManager> ResolveManagerForVmAsync(string vmId, CancellationToken ct)
    {
        // Check Docker manager first (faster, in-memory lookup)
        var dockerVm = await _dockerManager.GetVmAsync(vmId, ct);
        if (dockerVm != null) return _dockerManager;

        // Default to libvirt VM manager
        return _vmManager;
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