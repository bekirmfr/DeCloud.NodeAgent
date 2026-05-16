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

        // Boot-time readiness + liveness re-verification for opted-in services.
        var runningVms = allVms
            .Where(vm => vm.State == VmState.Running
                      && vm.Services.Count > 0
                      && (!vm.IsFullyReady || vm.Services.Any(s => s.LivenessCheck)))
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

        // Guest-agent liveness ping for fully-ready VMs (LastHeartbeat update).
        // VmHealthService uses LastHeartbeat to detect stale tenant VMs.
        var fullyReadyVms = allVms
            .Where(vm => vm.State == VmState.Running && vm.IsFullyReady)
            .ToList();

        foreach (var vm in fullyReadyVms)
        {
            try
            {
                if (await IsGuestAgentReady(vm.VmId, ct))
                {
                    vm.LastHeartbeat = DateTime.UtcNow;
                    await _repository.UpdateLastHeartbeatAsync(vm.VmId, vm.LastHeartbeat);
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogWarning(ex, "Failed to ping guest agent for VM {VmId}", vm.VmId);
            }
        }
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

        // ── GuestAgentPing: host-side, runs before the gate ──────────────
        foreach (var service in vm.Services.Where(s => s.CheckType == CheckType.GuestAgentPing))
        {
            // Apply same cadence rules as other services
            if (service.Status == ServiceReadiness.Ready)
            {
                if (!service.LivenessCheck) continue;
                var since = service.LastCheckAt.HasValue
                    ? (DateTime.UtcNow - service.LastCheckAt.Value).TotalSeconds
                    : double.MaxValue;
                if (since < RecheckInterval.TotalSeconds) continue;
            }
            else if (service.Status is ServiceReadiness.TimedOut or ServiceReadiness.Failed)
            {
                var since = service.LastCheckAt.HasValue
                    ? (DateTime.UtcNow - service.LastCheckAt.Value).TotalSeconds
                    : double.MaxValue;
                if (since < RecheckInterval.TotalSeconds) continue;
            }

            var alive = await IsGuestAgentReady(domainName, ct);
            var prev = service.Status;

            if (alive)
            {
                if (prev != ServiceReadiness.Ready)
                {
                    service.Status = ServiceReadiness.Ready;
                    service.ReadyAt = DateTime.UtcNow;
                    service.StatusMessage = null;
                    _logger.LogInformation(prev is ServiceReadiness.TimedOut or ServiceReadiness.Failed
                        ? "Service {Service} on VM {VmId} RECOVERED from {Prev}"
                        : "Service {Service} on VM {VmId} is READY",
                        service.Name, vm.VmId, prev);
                }
            }
            else
            {
                service.Status = ServiceReadiness.Failed;
                service.StatusMessage = "Guest agent unreachable";
                if (prev != ServiceReadiness.Failed)
                    _logger.LogWarning(
                        "Service {Service} on VM {VmId} FAILED — guest agent unreachable",
                        service.Name, vm.VmId);
            }

            service.LastCheckAt = DateTime.UtcNow;
            if (service.Status != prev) anyChanged = true;
        }

        // Check if guest agent is responding — needed for active probing
        // Use GuestAgentPing result if declared; otherwise probe directly.
        var agentPingService = vm.Services
            .FirstOrDefault(s => s.CheckType == CheckType.GuestAgentPing);
        var agentAlive = agentPingService != null
            ? agentPingService.Status == ServiceReadiness.Ready
            : await IsGuestAgentReady(domainName, ct);

        if (!agentAlive)
        {
            _logger.LogDebug("Guest agent not ready for VM {VmId}, skipping active checks", vm.VmId);
            if (anyChanged)
                await _repository.SaveVmAsync(vm);
            return;
        }

        bool systemReady = systemService?.Status == ServiceReadiness.Ready;

        foreach (var service in vm.Services)
        {
            if (service.CheckType == CheckType.GuestAgentPing)
                continue;

            if (service.Status == ServiceReadiness.Ready)
            {
                if (!service.LivenessCheck)
                    continue;
                var sinceLast = service.LastCheckAt.HasValue
                    ? (DateTime.UtcNow - service.LastCheckAt.Value).TotalSeconds
                    : double.MaxValue;
                if (sinceLast < RecheckInterval.TotalSeconds)
                    continue;
                // Fall through to execute the check.
            }

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
            var (success, failed, diagnostic) = await ExecuteServiceCheckAsync(domainName, service, ct);
            var previousStatus = service.Status;

            if (success)
            {
                if (previousStatus is ServiceReadiness.TimedOut or ServiceReadiness.Failed)
                {
                    _logger.LogInformation(
                        "Service {Service} on VM {VmId} RECOVERED from {PreviousStatus} (port: {Port})",
                        service.Name, vm.VmId, previousStatus, service.Port);
                }
                else if (previousStatus != ServiceReadiness.Ready)
                {
                    _logger.LogInformation("Service {Service} on VM {VmId} is READY (port: {Port})",
                        service.Name, vm.VmId, service.Port);
                }
                service.Status = ServiceReadiness.Ready;
                service.ReadyAt = DateTime.UtcNow;
                service.LastCheckAt = DateTime.UtcNow;
                service.StatusMessage = null; // clear stale failure message

                // If System just became ready, allow other services to start checking
                if (service.Name == "System") systemReady = true;
            }
            else if (failed)
            {
                service.Status = ServiceReadiness.Failed;
                service.StatusMessage = diagnostic;
                service.LastCheckAt = DateTime.UtcNow;
                if (previousStatus != ServiceReadiness.Failed)
                    _logger.LogWarning("Service {Service} on VM {VmId} FAILED", service.Name, vm.VmId);
            }
            else
            {
                if (previousStatus == ServiceReadiness.Ready)
                {
                    // Liveness re-check failure: service was Ready but no longer
                    // responding. Revert so IsFullyReady reflects reality.
                    service.Status = ServiceReadiness.Failed;
                    service.StatusMessage = diagnostic ?? "Liveness check failed";
                    service.LastCheckAt = DateTime.UtcNow;
                    _logger.LogWarning(
                        "Service {Service} on VM {VmId} FAILED liveness check (port {Port})",
                        service.Name, vm.VmId, service.Port);
                }
                else
                {
                    // Boot-time path: don't downgrade TimedOut/Failed to Checking
                    if (previousStatus is not (ServiceReadiness.TimedOut or ServiceReadiness.Failed))
                        service.Status = ServiceReadiness.Checking;
                    service.LastCheckAt = DateTime.UtcNow;
                }
            }

            if (service.Status != previousStatus) anyChanged = true;
        }

        if (anyChanged)
        {
            await _repository.SaveVmAsync(vm);
        }
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
    private async Task<(bool success, bool failed, string? diagnostic)> ExecuteServiceCheckAsync(
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
                // -s: silent (no progress bar). No -f: any HTTP response means
                // the service is running. No -o /dev/null: response body is
                // captured by guest-exec so it can be read on failure for diagnostics.
                args = new[] { "-s", "-m", "5", url };
                break;

            case CheckType.ExecCommand:
                path = "/bin/bash";
                args = new[] { "-c", service.ExecCommand ?? "true" };
                break;

            default:
                return (false, false, null);
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

            if (execResult.ExitCode != 0) return (false, false, null);

            var execJson = JsonDocument.Parse(execResult.StandardOutput.Trim());
            var pid = execJson.RootElement.GetProperty("return").GetProperty("pid").GetInt64();

            // Step 2: Wait, then get exit status
            await Task.Delay(GuestExecWait, ct);

            var statusCmd = $"{{\"execute\":\"guest-exec-status\",\"arguments\":{{\"pid\":{pid}}}}}";
            var statusResult = await _commandExecutor.ExecuteAsync(
                "virsh", VirshQemuAgentArgs(domain, statusCmd),
                TimeSpan.FromSeconds(10), ct);

            if (statusResult.ExitCode != 0) return (false, false, null);

            var statusJson = JsonDocument.Parse(statusResult.StandardOutput.Trim());
            var ret = statusJson.RootElement.GetProperty("return");

            if (!ret.GetProperty("exited").GetBoolean())
            {
                // Process still running — not ready yet
                return (false, false, null);
            }

            var exitCode = ret.GetProperty("exitcode").GetInt32();

            // Extract stdout/stderr from guest-exec-status for diagnostics.
            // Fields are base64-encoded and optional — only present when the
            // command produced output and capture-output was true.
            string ? diagnostic = null;
            if (exitCode != 0)
            {
                diagnostic = DecodeGuestExecOutput(ret);
            }
            
            return (exitCode == 0, false, diagnostic);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Guest-exec failed for {Service} on {Domain}", service.Name, domain);
            return (false, false, ex.Message);
        }
    }

    private static string EscapeJson(string s) =>
        s.Replace("\\", "\\\\").Replace("\"", "\\\"");

    /// <summary>
    /// Decode stdout and stderr from a guest-exec-status return object.
    /// Both fields are base64-encoded and optional. Returns a combined
    /// string truncated to 512 chars to avoid bloating ServicesJson.
    /// </summary>
    private static string? DecodeGuestExecOutput(JsonElement ret)
    {
        const int MaxLen = 512;

        var parts = new List<string>(2);

        if (ret.TryGetProperty("out-data", out var outData))
        {
            try
            {
                var decoded = Encoding.UTF8.GetString(
                    Convert.FromBase64String(outData.GetString() ?? ""));
                var trimmed = decoded.Trim();
                if (trimmed.Length > 0)
                    parts.Add(trimmed);
            }
            catch { /* malformed base64 — skip */ }
        }

        if (ret.TryGetProperty("err-data", out var errData))
        {
            try
            {
                var decoded = Encoding.UTF8.GetString(
                    Convert.FromBase64String(errData.GetString() ?? ""));
                var trimmed = decoded.Trim();
                if (trimmed.Length > 0)
                    parts.Add("stderr: " + trimmed);
            }
            catch { /* malformed base64 — skip */ }
        }

        if (parts.Count == 0)
            return null;

        var combined = string.Join(" | ", parts);
        return combined.Length > MaxLen
            ? combined[..MaxLen]
            : combined;
    }
}
