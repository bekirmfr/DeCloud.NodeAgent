using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace DeCloud.NodeAgent.Infrastructure.Docker;

/// <summary>
/// Manages workloads as Docker containers with GPU sharing.
/// Implements IVmManager so containers appear as lightweight "VMs" to the rest of the system.
/// Used on nodes without IOMMU support (e.g., WSL2) where VFIO passthrough is unavailable.
/// </summary>
public class DockerContainerManager : IVmManager
{
    private readonly ICommandExecutor _executor;
    private readonly VmRepository _repository;
    private readonly ILogger<DockerContainerManager> _logger;

    // Track containers in memory (synced with repository)
    private readonly Dictionary<string, VmInstance> _containers = new();
    private bool _initialized;

    private const string LabelPrefix = "decloud";

    public DockerContainerManager(
        ICommandExecutor executor,
        VmRepository repository,
        ILogger<DockerContainerManager> logger)
    {
        _executor = executor;
        _repository = repository;
        _logger = logger;
    }

    public async Task<VmOperationResult> CreateVmAsync(VmSpec spec, string? password = null, CancellationToken ct = default)
    {
        var vmId = spec.Id;
        var containerName = SanitizeContainerName(spec.Name, vmId);

        _logger.LogInformation(
            "Creating GPU container {ContainerId} ({Name}): image={Image}, {VCpus} vCPUs, {MemMB}MB RAM",
            vmId, containerName, spec.ContainerImage, spec.VirtualCpuCores,
            spec.MemoryBytes / (1024 * 1024));

        if (string.IsNullOrEmpty(spec.ContainerImage))
        {
            return VmOperationResult.Fail(vmId, "ContainerImage is required for container deployment mode");
        }

        try
        {
            // Pull image first
            _logger.LogInformation("Pulling container image: {Image}", spec.ContainerImage);
            var pullResult = await _executor.ExecuteAsync(
                "docker", $"pull {spec.ContainerImage}",
                TimeSpan.FromMinutes(10), ct);

            if (!pullResult.Success)
            {
                _logger.LogError("Failed to pull image {Image}: {Error}",
                    spec.ContainerImage, pullResult.StandardError);
                return VmOperationResult.Fail(vmId, $"Failed to pull image: {pullResult.StandardError}");
            }

            // Build docker run command
            var args = BuildDockerRunArgs(spec, containerName, password);

            _logger.LogInformation("Running: docker {Args}", args);

            var runResult = await _executor.ExecuteAsync("docker", args, TimeSpan.FromMinutes(2), ct);

            if (!runResult.Success)
            {
                _logger.LogError("Failed to create container {Id}: {Error}",
                    vmId, runResult.StandardError);
                return VmOperationResult.Fail(vmId, $"docker run failed: {runResult.StandardError}");
            }

            var containerId = runResult.StandardOutput.Trim();
            _logger.LogInformation("Container created: {ContainerId} (short: {Short})",
                containerId, containerId.Length > 12 ? containerId[..12] : containerId);

            // Get container IP
            var ip = await GetContainerIpAsync(containerName, ct);

            // Create VmInstance to track the container
            var instance = new VmInstance
            {
                VmId = vmId,
                Name = containerName,
                State = VmState.Running,
                Spec = spec,
                CreatedAt = DateTime.UtcNow,
                StartedAt = DateTime.UtcNow,
                LastHeartbeat = DateTime.UtcNow,
                DiskPath = containerId, // Store Docker container ID in DiskPath
                ConfigPath = $"container:{spec.ContainerImage}"
            };

            _containers[vmId] = instance;
            await _repository.SaveVmAsync(instance);

            _logger.LogInformation(
                "GPU container {VmId} ({Name}) created and running. Image: {Image}, IP: {Ip}",
                vmId, containerName, spec.ContainerImage, ip ?? "pending");

            return VmOperationResult.Ok(vmId, VmState.Running);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create container {VmId}", vmId);
            return VmOperationResult.Fail(vmId, ex.Message);
        }
    }

    public async Task<VmOperationResult> StartVmAsync(string vmId, CancellationToken ct = default)
    {
        var instance = await GetVmAsync(vmId, ct);
        if (instance == null)
            return VmOperationResult.Fail(vmId, "Container not found");

        var result = await _executor.ExecuteAsync("docker", $"start {instance.Name}", ct);
        if (!result.Success)
            return VmOperationResult.Fail(vmId, $"docker start failed: {result.StandardError}");

        instance.State = VmState.Running;
        instance.StartedAt = DateTime.UtcNow;
        await _repository.SaveVmAsync(instance);

        _logger.LogInformation("Container {VmId} started", vmId);
        return VmOperationResult.Ok(vmId, VmState.Running);
    }

    public async Task<VmOperationResult> StopVmAsync(string vmId, bool force = false, CancellationToken ct = default)
    {
        var instance = await GetVmAsync(vmId, ct);
        if (instance == null)
            return VmOperationResult.Fail(vmId, "Container not found");

        var cmd = force ? $"kill {instance.Name}" : $"stop -t 30 {instance.Name}";
        var result = await _executor.ExecuteAsync("docker", cmd, ct);
        if (!result.Success)
            return VmOperationResult.Fail(vmId, $"docker stop failed: {result.StandardError}");

        instance.State = VmState.Stopped;
        instance.StoppedAt = DateTime.UtcNow;
        await _repository.SaveVmAsync(instance);

        _logger.LogInformation("Container {VmId} stopped (force={Force})", vmId, force);
        return VmOperationResult.Ok(vmId, VmState.Stopped);
    }

    public async Task<VmOperationResult> RestartVmAsync(string vmId, bool force = false, CancellationToken ct = default)
    {
        var instance = await GetVmAsync(vmId, ct);
        if (instance == null)
            return VmOperationResult.Fail(vmId, "Container not found");

        var result = await _executor.ExecuteAsync("docker", $"restart {instance.Name}", ct);
        if (!result.Success)
            return VmOperationResult.Fail(vmId, $"docker restart failed: {result.StandardError}");

        instance.State = VmState.Running;
        instance.StartedAt = DateTime.UtcNow;
        await _repository.SaveVmAsync(instance);

        _logger.LogInformation("Container {VmId} restarted", vmId);
        return VmOperationResult.Ok(vmId, VmState.Running);
    }

    public async Task<VmOperationResult> DeleteVmAsync(string vmId, CancellationToken ct = default)
    {
        var instance = await GetVmAsync(vmId, ct);
        if (instance == null)
            return VmOperationResult.Fail(vmId, "Container not found");

        // Force remove (stops if running)
        var result = await _executor.ExecuteAsync("docker", $"rm -f {instance.Name}", ct);
        if (!result.Success)
        {
            _logger.LogWarning("docker rm failed for {VmId}: {Error}", vmId, result.StandardError);
            // Continue cleanup even if docker rm fails
        }

        instance.State = VmState.Deleted;
        _containers.Remove(vmId);
        await _repository.SaveVmAsync(instance);

        _logger.LogInformation("Container {VmId} deleted", vmId);
        return VmOperationResult.Ok(vmId, VmState.Deleted);
    }

    public async Task<VmOperationResult> PauseVmAsync(string vmId, CancellationToken ct = default)
    {
        var instance = await GetVmAsync(vmId, ct);
        if (instance == null)
            return VmOperationResult.Fail(vmId, "Container not found");

        var result = await _executor.ExecuteAsync("docker", $"pause {instance.Name}", ct);
        if (!result.Success)
            return VmOperationResult.Fail(vmId, $"docker pause failed: {result.StandardError}");

        instance.State = VmState.Paused;
        await _repository.SaveVmAsync(instance);
        return VmOperationResult.Ok(vmId, VmState.Paused);
    }

    public async Task<VmOperationResult> ResumeVmAsync(string vmId, CancellationToken ct = default)
    {
        var instance = await GetVmAsync(vmId, ct);
        if (instance == null)
            return VmOperationResult.Fail(vmId, "Container not found");

        var result = await _executor.ExecuteAsync("docker", $"unpause {instance.Name}", ct);
        if (!result.Success)
            return VmOperationResult.Fail(vmId, $"docker unpause failed: {result.StandardError}");

        instance.State = VmState.Running;
        await _repository.SaveVmAsync(instance);
        return VmOperationResult.Ok(vmId, VmState.Running);
    }

    public Task<VmInstance?> GetVmAsync(string vmId, CancellationToken ct = default)
    {
        _containers.TryGetValue(vmId, out var instance);
        return Task.FromResult(instance);
    }

    public Task<List<VmInstance>> GetAllVmsAsync(CancellationToken ct = default)
    {
        return Task.FromResult(_containers.Values.ToList());
    }

    public async Task<VmResourceUsage> GetVmUsageAsync(string vmId, CancellationToken ct = default)
    {
        var usage = new VmResourceUsage { MeasuredAt = DateTime.UtcNow };

        var instance = await GetVmAsync(vmId, ct);
        if (instance == null) return usage;

        try
        {
            var result = await _executor.ExecuteAsync(
                "docker",
                $"stats --no-stream --format \"{{{{.CPUPerc}}}} {{{{.MemUsage}}}}\" {instance.Name}",
                ct);

            if (result.Success && !string.IsNullOrWhiteSpace(result.StandardOutput))
            {
                var parts = result.StandardOutput.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 1)
                {
                    var cpuStr = parts[0].TrimEnd('%');
                    if (double.TryParse(cpuStr, out var cpu))
                        usage.CpuPercent = cpu;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to get container stats for {VmId}", vmId);
        }

        return usage;
    }

    public async Task ReconcileAllWithLibvirtAsync(CancellationToken ct = default)
    {
        // Reconcile containers: check Docker state matches our records
        _logger.LogInformation("Reconciling Docker containers with database...");

        foreach (var (vmId, instance) in _containers.ToList())
        {
            if (instance.State == VmState.Deleted) continue;

            try
            {
                var result = await _executor.ExecuteAsync(
                    "docker", $"inspect --format \"{{{{.State.Status}}}}\" {instance.Name}", ct);

                if (!result.Success)
                {
                    _logger.LogWarning("Container {VmId} ({Name}) not found in Docker - marking as deleted",
                        vmId, instance.Name);
                    instance.State = VmState.Deleted;
                    await _repository.SaveVmAsync(instance);
                    continue;
                }

                var dockerState = result.StandardOutput.Trim().ToLowerInvariant();
                var newState = dockerState switch
                {
                    "running" => VmState.Running,
                    "paused" => VmState.Paused,
                    "exited" or "dead" => VmState.Stopped,
                    "created" or "restarting" => VmState.Starting,
                    _ => instance.State
                };

                if (newState != instance.State)
                {
                    _logger.LogInformation("Container {VmId} state reconciled: {Old} → {New}",
                        vmId, instance.State, newState);
                    instance.State = newState;
                    await _repository.SaveVmAsync(instance);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to reconcile container {VmId}", vmId);
            }
        }
    }

    public Task<bool> VmExistsAsync(string vmId, CancellationToken ct = default)
    {
        return Task.FromResult(_containers.ContainsKey(vmId));
    }

    public async Task<string?> GetVmIpAddressAsync(string vmId, CancellationToken ct = default)
    {
        var instance = await GetVmAsync(vmId, ct);
        if (instance == null) return null;
        return await GetContainerIpAsync(instance.Name, ct);
    }

    public Task<bool> ApplyQuotaCapAsync(
        VmInstance vm, int quotaMicroseconds, int periodMicroseconds = 100000,
        CancellationToken ct = default)
    {
        // Docker CPU limits are set at container creation via --cpus
        // Dynamic quota adjustment is not supported for containers
        _logger.LogDebug("ApplyQuotaCap not applicable for Docker containers (set at creation)");
        return Task.FromResult(true);
    }

    /// <summary>
    /// Initialize: load containers from database and reconcile with Docker
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        if (_initialized) return;

        _logger.LogInformation("Initializing DockerContainerManager...");

        try
        {
            // Load from database
            var savedVms = await _repository.LoadAllVmsAsync();
            foreach (var vm in savedVms)
            {
                // Only track container-mode VMs
                if (vm.Spec.DeploymentMode == DeploymentMode.Container)
                {
                    _containers[vm.VmId] = vm;
                }
            }

            _logger.LogInformation("Loaded {Count} containers from database", _containers.Count);

            // Reconcile with Docker
            await ReconcileAllWithLibvirtAsync(ct);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize DockerContainerManager");
        }

        _initialized = true;
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    private string BuildDockerRunArgs(VmSpec spec, string containerName, string? password)
    {
        var args = new List<string>
        {
            "run", "-d",
            "--name", containerName,
            "--gpus", "all",
            "--restart", "unless-stopped",
            $"--cpus={spec.VirtualCpuCores}",
            $"--memory={spec.MemoryBytes}",
            $"--label={LabelPrefix}.vm-id={spec.Id}",
            $"--label={LabelPrefix}.owner-id={spec.OwnerId ?? "unknown"}",
            $"--label={LabelPrefix}.managed=true"
        };

        // Environment variables
        if (spec.EnvironmentVariables != null)
        {
            foreach (var (key, value) in spec.EnvironmentVariables)
            {
                args.Add("-e");
                args.Add($"{key}={value}");
            }
        }

        // Pass password as env var if provided
        if (!string.IsNullOrEmpty(password))
        {
            args.Add("-e");
            args.Add($"DECLOUD_PASSWORD={password}");
        }

        // Container image (must be last)
        args.Add(spec.ContainerImage!);

        return string.Join(" ", args.Select(EscapeArg));
    }

    private async Task<string?> GetContainerIpAsync(string containerName, CancellationToken ct)
    {
        try
        {
            var result = await _executor.ExecuteAsync(
                "docker",
                $"inspect --format \"{{{{.NetworkSettings.IPAddress}}}}\" {containerName}",
                ct);

            if (result.Success)
            {
                var ip = result.StandardOutput.Trim().Trim('"');
                return string.IsNullOrEmpty(ip) ? null : ip;
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to get container IP for {Name}", containerName);
        }

        return null;
    }

    private static string SanitizeContainerName(string name, string vmId)
    {
        // Docker container names: [a-zA-Z0-9][a-zA-Z0-9_.-]
        var sanitized = new string(name
            .Where(c => char.IsLetterOrDigit(c) || c == '-' || c == '_' || c == '.')
            .ToArray());

        if (string.IsNullOrEmpty(sanitized))
            sanitized = "decloud";

        // Prefix with "dc-" and suffix with short VM ID for uniqueness
        var shortId = vmId.Length >= 8 ? vmId[..8] : vmId;
        return $"dc-{sanitized}-{shortId}";
    }

    private static string EscapeArg(string arg)
    {
        if (arg.Contains(' ') && !arg.StartsWith('"'))
            return $"\"{arg}\"";
        return arg;
    }
}
