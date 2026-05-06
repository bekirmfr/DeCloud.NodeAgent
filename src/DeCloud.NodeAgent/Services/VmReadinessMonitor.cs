using System.Text;
using System.Text.Json;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Infrastructure.Persistence;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Background service that monitors per-service readiness of running VMs
/// using qemu-guest-agent commands through the virtio channel.
/// Polls every 10 seconds. Reports results via HeartbeatService.
/// </summary>
public class VmReadinessMonitor : BackgroundService
{
    private readonly IVmManager _vmManager;
    private readonly VmRepository _repository;
    private readonly ICommandExecutor _commandExecutor;
    private readonly ILogger<VmReadinessMonitor> _logger;
    private static readonly TimeSpan PollInterval = TimeSpan.FromSeconds(10);
    private static readonly TimeSpan GuestExecWait = TimeSpan.FromSeconds(3);
    private static readonly TimeSpan RecheckInterval = TimeSpan.FromSeconds(60);

    // ── Periodic liveness for fully-ready system VMs ─────────────────────
    //
    // System VMs are long-lived; the reconciler depends on accurate
    // IsFullyReady to detect Unhealthy reality. These fields support
    // periodic re-verification after initial readiness is established.
    //
    // Guest-agent failure tracking: if the agent is persistently
    // unreachable, service checks cannot run and the only signal is
    // the ping itself. After the threshold, System service is reverted
    // to Failed so IsFullyReady becomes false.
    //
    // Liveness cadence: service re-checks run at LivenessRecheckInterval
    // (60s). SystemVm services run under Restart=always in systemd — a
    // transient crash self-heals within seconds. A service still dead at
    // the next 60s check is persistently failed and warrants redeploy.

    private readonly Dictionary<string, int> _guestAgentConsecutiveFailures = new();
    private readonly Dictionary<string, DateTime> _lastLivenessCheck = new();
    private const int GuestAgentFailureThreshold = 18; // ~3 min at 10s poll
    private static readonly TimeSpan LivenessRecheckInterval = TimeSpan.FromSeconds(60);
    private static readonly HashSet<VmType> SystemVmTypes =
        [VmType.Relay, VmType.Dht, VmType.BlockStore];

    public VmReadinessMonitor(
        IVmManager vmManager,
        VmRepository repository,
        ICommandExecutor commandExecutor,
        ILogger<VmReadinessMonitor> logger)
    {
        _vmManager = vmManager;
        _repository = repository;
        _commandExecutor = commandExecutor;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("VmReadinessMonitor started — polling every {Interval}s", PollInterval.TotalSeconds);

        // Wait for VMs to be loaded from database
        await Task.Delay(TimeSpan.FromSeconds(15), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await CheckAllVmsAsync(stoppingToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogError(ex, "VmReadinessMonitor cycle failed");
            }

            await Task.Delay(PollInterval, stoppingToken);
        }
    }

    private async Task CheckAllVmsAsync(CancellationToken ct)
    {
        var allVms = _vmManager.GetAllVms();

        // Service readiness checks — only for VMs not yet fully ready.
        var runningVms = allVms
            .Where(vm => vm.State == VmState.Running && vm.Services.Count > 0 && !vm.IsFullyReady)
            .ToList();

        foreach (var vm in runningVms)
        {
            try
            {
                await CheckVmServicesAsync(vm, ct);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogWarning(ex, "Failed to check readiness for VM {VmId}", vm.VmId);
            }
        }

        // Guest-agent liveness ping for fully-ready VMs.
        // Updates LastHeartbeat so VmHealthService can detect stale tenant VMs.
        // For system VMs, also runs periodic service re-verification so
        // IsFullyReady reflects reality and the reconciler can act on failures.
        var fullyReadyVms = allVms
            .Where(vm => vm.State == VmState.Running && vm.IsFullyReady)
            .ToList();

        foreach (var vm in fullyReadyVms)
        {
            try
            {
                var agentAlive = await IsGuestAgentReady(vm.VmId, ct);

                if (agentAlive)
                {
                    vm.LastHeartbeat = DateTime.UtcNow;
                    await _repository.UpdateLastHeartbeatAsync(vm.VmId, vm.LastHeartbeat);
                    _guestAgentConsecutiveFailures.Remove(vm.VmId);

                    // Periodic service liveness re-check for system VMs.
                    if (SystemVmTypes.Contains(vm.Spec.VmType))
                        await RecheckSystemVmServicesAsync(vm, ct);
                }
                else if (SystemVmTypes.Contains(vm.Spec.VmType))
                {
                    await HandleSystemVmAgentFailureAsync(vm);
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogWarning(ex, "Failed to ping guest agent for VM {VmId}", vm.VmId);
            }
        }

        // Prune tracking for VMs that no longer exist (deleted/redeployed).
        var activeVmIds = allVms.Select(v => v.VmId).ToHashSet();
        foreach (var staleId in _guestAgentConsecutiveFailures.Keys
                     .Where(id => !activeVmIds.Contains(id)).ToList())
            _guestAgentConsecutiveFailures.Remove(staleId);
        foreach (var staleId in _lastLivenessCheck.Keys
                     .Where(id => !activeVmIds.Contains(id)).ToList())
            _lastLivenessCheck.Remove(staleId);
    }

    private async Task CheckVmServicesAsync(VmInstance vm, CancellationToken ct)
    {
        var domainName = vm.VmId;
        bool anyChanged = false;

        // Evaluate timeouts BEFORE the guest agent gate so that services don't stay
        // Pending forever when qemu-guest-agent never starts (e.g. missing package).
        var systemService = vm.Services.FirstOrDefault(s => s.Name == "System");
        foreach (var service in vm.Services)
        {
            if (service.Status is ServiceReadiness.Ready
                or ServiceReadiness.TimedOut
                or ServiceReadiness.Failed)
                continue;

            var timeoutBase = service.Name == "System"
                ? (vm.StartedAt ?? vm.CreatedAt)
                : (systemService?.ReadyAt ?? vm.StartedAt ?? vm.CreatedAt);
            var elapsed = (DateTime.UtcNow - timeoutBase).TotalSeconds;
            if (elapsed > service.TimeoutSeconds)
            {
                service.Status = ServiceReadiness.TimedOut;
                service.LastCheckAt = DateTime.UtcNow;
                anyChanged = true;
                _logger.LogWarning(
                    "Service {Service} on VM {VmId} timed out after {Timeout}s (guest agent may not be running)",
                    service.Name, vm.VmId, service.TimeoutSeconds);
            }
        }

        // Check if guest agent is responding — needed for active probing
        if (!await IsGuestAgentReady(domainName, ct))
        {
            _logger.LogDebug("Guest agent not ready for VM {VmId}, skipping active checks", vm.VmId);
            if (anyChanged)
                await _repository.SaveVmAsync(vm);
            return;
        }

        bool systemReady = systemService?.Status == ServiceReadiness.Ready;

        foreach (var service in vm.Services)
        {
            if (service.Status == ServiceReadiness.Ready)
                continue;

            // Self-healing: recheck TimedOut/Failed services at a slower interval
            if (service.Status is ServiceReadiness.TimedOut or ServiceReadiness.Failed)
            {
                var sinceLastCheck = service.LastCheckAt.HasValue
                    ? (DateTime.UtcNow - service.LastCheckAt.Value).TotalSeconds
                    : double.MaxValue;
                if (sinceLastCheck < RecheckInterval.TotalSeconds)
                    continue;
            }

            // Gate: non-System services wait for System to be Ready
            if (service.Name != "System" && !systemReady)
            {
                if (service.Status != ServiceReadiness.Pending)
                {
                    service.Status = ServiceReadiness.Pending;
                    anyChanged = true;
                }
                continue;
            }

            // Execute the check
            var (success, failed) = await ExecuteServiceCheckAsync(domainName, service, ct);
            var previousStatus = service.Status;

            if (success)
            {
                if (previousStatus is ServiceReadiness.TimedOut or ServiceReadiness.Failed)
                {
                    _logger.LogInformation(
                        "Service {Service} on VM {VmId} RECOVERED from {PreviousStatus} (port: {Port})",
                        service.Name, vm.VmId, previousStatus, service.Port);
                }
                else
                {
                    _logger.LogInformation("Service {Service} on VM {VmId} is READY (port: {Port})",
                        service.Name, vm.VmId, service.Port);
                }
                service.Status = ServiceReadiness.Ready;
                service.ReadyAt = DateTime.UtcNow;
                service.LastCheckAt = DateTime.UtcNow;

                // If System just became ready, allow other services to start checking
                if (service.Name == "System") systemReady = true;
            }
            else if (failed)
            {
                service.Status = ServiceReadiness.Failed;
                service.LastCheckAt = DateTime.UtcNow;
                if (previousStatus != ServiceReadiness.Failed)
                    _logger.LogWarning("Service {Service} on VM {VmId} FAILED", service.Name, vm.VmId);
            }
            else
            {
                // Don't downgrade TimedOut/Failed to Checking — keep the terminal status
                // so the recheck interval continues to apply
                if (previousStatus is not (ServiceReadiness.TimedOut or ServiceReadiness.Failed))
                    service.Status = ServiceReadiness.Checking;
                service.LastCheckAt = DateTime.UtcNow;
            }

            if (service.Status != previousStatus) anyChanged = true;
        }

        if (anyChanged)
        {
            await _repository.SaveVmAsync(vm);
        }
    }

    /// <summary>
    /// Periodic liveness re-verification for a fully-ready system VM.
    /// Re-runs HttpGet/TcpPort service checks at <see cref="LivenessRecheckInterval"/>
    /// cadence. If a previously-Ready service now fails, reverts it to Failed so
    /// <see cref="VmInstance.IsFullyReady"/> becomes false and the reconciler
    /// detects Unhealthy reality.
    ///
    /// CloudInitDone checks are skipped — cloud-init completion is permanent.
    /// Only called when the guest agent is confirmed alive (the caller gates on
    /// <see cref="IsGuestAgentReady"/>).
    /// </summary>
    private async Task RecheckSystemVmServicesAsync(VmInstance vm, CancellationToken ct)
    {
        // Throttle: one re-check per LivenessRecheckInterval per VM.
        if (_lastLivenessCheck.TryGetValue(vm.VmId, out var lastCheck)
            && (DateTime.UtcNow - lastCheck) < LivenessRecheckInterval)
            return;

        _lastLivenessCheck[vm.VmId] = DateTime.UtcNow;

        var anyReverted = false;

        foreach (var service in vm.Services)
        {
            if (service.Status != ServiceReadiness.Ready)
                continue;

            // CloudInitDone is permanent — boot-finished doesn't disappear.
            if (service.CheckType == CheckType.CloudInitDone)
                continue;

            var (success, _) = await ExecuteServiceCheckAsync(vm.VmId, service, ct);
            service.LastCheckAt = DateTime.UtcNow;

            if (!success)
            {
                _logger.LogWarning(
                    "System VM {VmId} ({VmType}): service {Service} (port {Port}) " +
                    "failed periodic liveness check — reverting to Failed",
                    vm.VmId, vm.Spec.VmType, service.Name, service.Port);

                service.Status = ServiceReadiness.Failed;
                service.StatusMessage = "Periodic liveness check failed";
                anyReverted = true;
            }
        }

        if (anyReverted)
            await _repository.SaveVmAsync(vm);
    }

    /// <summary>
    /// Handles persistent guest-agent unreachability for a fully-ready system VM.
    /// After <see cref="GuestAgentFailureThreshold"/> consecutive failures (~3 min),
    /// reverts all services to Failed. This is the fallback for the case where the
    /// guest OS itself is dead — service checks cannot run via guest-exec, so the
    /// agent ping is the only signal.
    /// </summary>
    private async Task HandleSystemVmAgentFailureAsync(VmInstance vm)
    {
        _guestAgentConsecutiveFailures.TryGetValue(vm.VmId, out var count);
        count++;
        _guestAgentConsecutiveFailures[vm.VmId] = count;

        if (count < GuestAgentFailureThreshold)
            return;

        _logger.LogWarning(
            "System VM {VmId} ({VmType}) guest agent unreachable for {Count} " +
            "consecutive polls (~{Seconds}s) — reverting all services to Failed",
            vm.VmId, vm.Spec.VmType, count,
            count * (int)PollInterval.TotalSeconds);

        foreach (var svc in vm.Services)
        {
            svc.Status = ServiceReadiness.Failed;
            svc.StatusMessage = "Guest agent unreachable";
            svc.LastCheckAt = DateTime.UtcNow;
        }

        await _repository.SaveVmAsync(vm);

        // Reset counter. If the reconciler doesn't act immediately (e.g.
        // a pending Delete is in flight), the next threshold crossing will
        // re-trigger, keeping reality accurate.
        _guestAgentConsecutiveFailures.Remove(vm.VmId);
    }

    /// <summary>
    /// Check if qemu-guest-agent is responding via guest-ping.
    /// </summary>
    private async Task<bool> IsGuestAgentReady(string domain, CancellationToken ct)
    {
        try
        {
            var result = await _commandExecutor.ExecuteAsync(
                "virsh", VirshQemuAgentArgs(domain, "{\"execute\":\"guest-ping\"}"),
                TimeSpan.FromSeconds(5), ct);
            return result.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Build virsh arguments for qemu-agent-command with proper escaping.
    /// .NET Process.Start splits Arguments using Windows-style rules even on Linux:
    /// double quotes delimit argument groups, backslash-quote (\") is a literal quote.
    /// Single quotes have NO special meaning, so '{"execute":"guest-ping"}' fails
    /// because the inner double quotes are consumed as group delimiters.
    /// </summary>
    private static string VirshQemuAgentArgs(string domain, string jsonCommand)
    {
        var escaped = jsonCommand.Replace("\\", "\\\\").Replace("\"", "\\\"");
        return $"qemu-agent-command {domain} \"{escaped}\"";
    }

    /// <summary>
    /// Execute a service check via qemu-guest-agent guest-exec.
    /// Returns (success, hardFailed). hardFailed is true only for cloud-init error state.
    /// </summary>
    private async Task<(bool success, bool failed)> ExecuteServiceCheckAsync(
        string domain, VmServiceStatus service, CancellationToken ct)
    {
        string path;
        string[] args;

        switch (service.CheckType)
        {
            case CheckType.CloudInitDone:
                path = "/bin/bash";
                args = new[] { "-c", "test -f /var/lib/cloud/instance/boot-finished" };
                break;

            case CheckType.TcpPort:
                path = "/usr/bin/nc";
                args = new[] { "-zv", "-w2", "localhost", service.Port?.ToString() ?? "0" };
                break;

            case CheckType.HttpGet:
                path = "/usr/bin/curl";
                var url = $"http://localhost:{service.Port}{service.HttpPath ?? "/"}";
                // Use -s (silent) without -f: any HTTP response (including 401 auth)
                // means the service is running. Only connection refused/timeout = not ready.
                args = new[] { "-s", "-o", "/dev/null", "-m", "5", url };
                break;

            case CheckType.ExecCommand:
                path = "/bin/bash";
                args = new[] { "-c", service.ExecCommand ?? "true" };
                break;

            default:
                return (false, false);
        }

        // Build guest-exec JSON
        var argsJson = string.Join(",", args.Select(a => $"\"{EscapeJson(a)}\""));
        var execCmd = $"{{\"execute\":\"guest-exec\",\"arguments\":{{\"path\":\"{EscapeJson(path)}\",\"arg\":[{argsJson}],\"capture-output\":true}}}}";

        try
        {
            // Step 1: Send guest-exec, get PID
            var execResult = await _commandExecutor.ExecuteAsync(
                "virsh", VirshQemuAgentArgs(domain, execCmd),
                TimeSpan.FromSeconds(10), ct);

            if (execResult.ExitCode != 0) return (false, false);

            var execJson = JsonDocument.Parse(execResult.StandardOutput.Trim());
            var pid = execJson.RootElement.GetProperty("return").GetProperty("pid").GetInt64();

            // Step 2: Wait, then get exit status
            await Task.Delay(GuestExecWait, ct);

            var statusCmd = $"{{\"execute\":\"guest-exec-status\",\"arguments\":{{\"pid\":{pid}}}}}";
            var statusResult = await _commandExecutor.ExecuteAsync(
                "virsh", VirshQemuAgentArgs(domain, statusCmd),
                TimeSpan.FromSeconds(10), ct);

            if (statusResult.ExitCode != 0) return (false, false);

            var statusJson = JsonDocument.Parse(statusResult.StandardOutput.Trim());
            var ret = statusJson.RootElement.GetProperty("return");

            if (!ret.GetProperty("exited").GetBoolean())
            {
                // Process still running — not ready yet
                return (false, false);
            }

            var exitCode = ret.GetProperty("exitcode").GetInt32();

            return (exitCode == 0, false);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Guest-exec failed for {Service} on {Domain}", service.Name, domain);
            return (false, false);
        }
    }

    private static string EscapeJson(string s) =>
        s.Replace("\\", "\\\\").Replace("\"", "\\\"");
}
