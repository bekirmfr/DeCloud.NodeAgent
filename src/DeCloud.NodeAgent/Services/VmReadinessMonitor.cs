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
        var allVms = await _vmManager.GetAllVmsAsync(ct);
        var runningVms = allVms
            .Where(vm => vm.State == VmState.Running && vm.Services.Count > 0 && !vm.IsFullyReady)
            .ToList();

        if (runningVms.Count == 0) return;

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
    }

    private async Task CheckVmServicesAsync(VmInstance vm, CancellationToken ct)
    {
        var domainName = vm.VmId;
        bool anyChanged = false;

        // Check if guest agent is responding first
        if (!await IsGuestAgentReady(domainName, ct))
        {
            _logger.LogDebug("Guest agent not ready for VM {VmId}, skipping", vm.VmId);
            return;
        }

        var systemService = vm.Services.FirstOrDefault(s => s.Name == "System");
        bool systemReady = systemService?.Status == ServiceReadiness.Ready;

        foreach (var service in vm.Services)
        {
            if (service.Status == ServiceReadiness.Ready || service.Status == ServiceReadiness.TimedOut)
                continue;

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

            // Check timeout — System counts from VM start; other services count from
            // when System became Ready, since they're gated behind it.
            var timeoutBase = service.Name == "System"
                            ? (vm.StartedAt ?? vm.CreatedAt)
                            : (systemService?.ReadyAt ?? vm.StartedAt ?? vm.CreatedAt);
            var elapsed = (DateTime.UtcNow - timeoutBase).TotalSeconds;
            if (elapsed > service.TimeoutSeconds)
            {
                if (service.Status != ServiceReadiness.TimedOut)
                {
                    service.Status = ServiceReadiness.TimedOut;
                    service.LastCheckAt = DateTime.UtcNow;
                    anyChanged = true;
                    _logger.LogWarning("Service {Service} on VM {VmId} timed out after {Timeout}s",
                        service.Name, vm.VmId, service.TimeoutSeconds);
                }
                continue;
            }

            // Execute the check
            var (success, failed) = await ExecuteServiceCheckAsync(domainName, service, ct);
            var previousStatus = service.Status;

            if (success)
            {
                service.Status = ServiceReadiness.Ready;
                service.ReadyAt = DateTime.UtcNow;
                service.LastCheckAt = DateTime.UtcNow;
                _logger.LogInformation("Service {Service} on VM {VmId} is READY (port: {Port})",
                    service.Name, vm.VmId, service.Port);

                // If System just became ready, allow other services to start checking
                if (service.Name == "System") systemReady = true;
            }
            else if (failed)
            {
                service.Status = ServiceReadiness.Failed;
                service.LastCheckAt = DateTime.UtcNow;
                _logger.LogWarning("Service {Service} on VM {VmId} FAILED", service.Name, vm.VmId);
            }
            else
            {
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
                path = "/usr/bin/cloud-init";
                args = new[] { "status", "--format", "json" };
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

            // Special handling for cloud-init: parse JSON output to detect "error" status
            if (service.CheckType == CheckType.CloudInitDone && exitCode == 0 &&
                ret.TryGetProperty("out-data", out var outData))
            {
                var stdout = Encoding.UTF8.GetString(Convert.FromBase64String(outData.GetString() ?? ""));
                try
                {
                    var cloudInitJson = JsonDocument.Parse(stdout);
                    var status = cloudInitJson.RootElement.GetProperty("status").GetString();
                    return status switch
                    {
                        "done" => (true, false),
                        "error" => (false, true),
                        _ => (false, false) // "running" or other
                    };
                }
                catch
                {
                    return (false, false);
                }
            }

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
