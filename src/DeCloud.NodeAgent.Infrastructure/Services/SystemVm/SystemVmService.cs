using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Interfaces.SystemVm;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Services.State;
using DeCloud.Shared.Enums;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;

namespace DeCloud.NodeAgent.Infrastructure.Services.SystemVm;

/// <summary>
/// Central service for system VM operations.
///
/// Owns:
///   • VM lookup by role (single source for "give me the running DHT VM")
///   • Binary version cache (query /version once per VM lifecycle, not per heartbeat)
///   • Template revision reporting (delegates to obligation-state SQLite)
///   • Dashboard URL resolution (role → http://{ip}:{port})
///
/// Consumed by:
///   • OrchestratorClient — heartbeat payload (versions + revisions)
///   • SystemVmController — proxy target resolution
///   • SystemVmReconciler — cache invalidation on VM deletion
/// </summary>
public sealed class SystemVmService : ISystemVmService
{
    private readonly IVmManager _vmManager;
    private readonly IObligationStateService _obligationState;
    private readonly ICommandExecutor _commandExecutor;
    private readonly ILogger<SystemVmService> _logger;

    /// <summary>
    /// Cached binary versions keyed by role. Entry is (vmId, version) — the
    /// vmId is used to detect redeploys: if the running VM's ID differs from
    /// the cached one, the cache is stale and a fresh /version query fires.
    /// </summary>
    private readonly ConcurrentDictionary<string, (string vmId, string version)> _versionCache = new();

    public SystemVmService(
        IVmManager vmManager,
        IObligationStateService obligationState,
        ICommandExecutor commandExecutor,
        ILogger<SystemVmService> logger)
    {
        _vmManager = vmManager;
        _obligationState = obligationState;
        _commandExecutor = commandExecutor;
        _logger = logger;
    }

    // ── VM lookup ────────────────────────────────────────────────────────

    public VmInstance? GetRunningVm(string role)
    {
        if (!ObligationStateService.ObligationToVmRole.TryGetValue(role, out var vmRole))
            return null;

        return _vmManager
            .GetAllVms()
            .FirstOrDefault(v =>
                v.Spec.Role == vmRole &&
                v.Status == Shared.Enums.VmStatus.Running &&
                !string.IsNullOrEmpty(v.Spec.IpAddress));
    }

    public string? GetDashboardBaseUrl(string role)
    {
        var vm = GetRunningVm(role);
        return vm is null
            ? null
            : $"http://{vm.Spec.IpAddress}:{ISystemVmService.DashboardPort}";
    }

    // ── Binary version tracking ──────────────────────────────────────────

    public string? GetCachedBinaryVersion(string role) =>
        _versionCache.TryGetValue(role, out var entry) ? entry.version : null;

    public async Task<Dictionary<string, string?>> GetAllBinaryVersionsAsync(
        CancellationToken ct = default)
    {
        var result = new Dictionary<string, string?>();

        foreach (var role in ISystemVmService.Roles)
        {
            try
            {
                var vm = GetRunningVm(role);
                if (vm is null) continue;

                // Cache hit: same VM ID means same binary — skip the HTTP call.
                if (_versionCache.TryGetValue(role, out var cached) &&
                    cached.vmId == vm.VmId)
                {
                    result[role] = cached.version;
                    continue;
                }

                var version = await QueryVersionAsync(vm, ct);
                if (version is not null)
                {
                    _versionCache[role] = (vm.VmId, version);
                    result[role] = version;
                }
            }
            catch (HttpRequestException)
            {
                // Expected during VM boot — dashboard port not listening yet.
                // Version will be cached on the next successful query.
            }
            catch (TaskCanceledException)
            {
                // Timeout — VM is slow to respond, will retry next heartbeat.
            }
            catch (Exception ex)
            {
                _logger.LogDebug("Could not query /version for {Role}: {Message}", role, ex.Message);
            }
        }

        return result;
    }

    public void InvalidateVersionCache(string role)
    {
        if (_versionCache.TryRemove(role, out _))
            _logger.LogDebug("SystemVmService: invalidated version cache for {Role}", role);
    }

    // ── Template revisions ───────────────────────────────────────────────

    public Task<Dictionary<string, int>> GetTemplateRevisionsAsync(
        CancellationToken ct = default) =>
        _obligationState.GetSystemTemplateRevisionsAsync(ct);

    public async Task<string?> CaptureVmJournalAsync(string vmId, int lines = 100, CancellationToken ct = default)
    {
        try
        {
            // Step 1: guest-exec journalctl inside the VM
            var argsJson = string.Join(",",
                new[] { "--no-pager", "-n", lines.ToString(), "-p", "0..5", "--since=-30min" }
                    .Select(a => $"\"{a}\""));
            var execCmd = $"{{\"execute\":\"guest-exec\",\"arguments\":{{\"path\":\"/usr/bin/journalctl\",\"arg\":[{argsJson}],\"capture-output\":true}}}}";

            var escapedExec = execCmd.Replace("\\", "\\\\").Replace("\"", "\\\"");
            var execResult = await _commandExecutor.ExecuteAsync(
                "virsh",
                $"-c qemu:///system qemu-agent-command {vmId} \"{escapedExec}\"",
                TimeSpan.FromSeconds(5), ct);

            if (execResult.ExitCode != 0) return null;

            var execJson = JsonDocument.Parse(execResult.StandardOutput.Trim());
            var pid = execJson.RootElement.GetProperty("return").GetProperty("pid").GetInt64();

            // Step 2: Wait briefly, then fetch output
            await Task.Delay(2000, ct);

            var statusCmd = $"{{\"execute\":\"guest-exec-status\",\"arguments\":{{\"pid\":{pid}}}}}";
            var escapedStatus = statusCmd.Replace("\\", "\\\\").Replace("\"", "\\\"");
            var statusResult = await _commandExecutor.ExecuteAsync(
                "virsh",
                $"-c qemu:///system qemu-agent-command {vmId} \"{escapedStatus}\"",
                TimeSpan.FromSeconds(5), ct);

            if (statusResult.ExitCode != 0) return null;

            var statusJson = JsonDocument.Parse(statusResult.StandardOutput.Trim());
            var ret = statusJson.RootElement.GetProperty("return");

            if (!ret.GetProperty("exited").GetBoolean()) return null;

            // Decode base64 stdout
            if (ret.TryGetProperty("out-data", out var outData))
            {
                var decoded = Encoding.UTF8.GetString(
                    Convert.FromBase64String(outData.GetString() ?? ""));
                var trimmed = decoded.Trim();
                // Cap at 8KB to avoid bloating SQLite
                return trimmed.Length > 8192 ? trimmed[..8192] : trimmed;
            }

            return null;
        }
        catch
        {
            // Guest agent unreachable — expected when that's the failure mode.
            return null;
        }
    }

    // ── Internal ─────────────────────────────────────────────────────────

    /// <summary>
    /// Query the /version endpoint on a running system VM's dashboard port.
    /// Returns the version string or null if the endpoint is unavailable.
    /// </summary>
    private async Task<string?> QueryVersionAsync(VmInstance vm, CancellationToken ct)
    {
        var url = $"http://{vm.Spec.IpAddress}:{ISystemVmService.DashboardPort}/version";

        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
        var response = await client.GetAsync(url, ct);

        if (!response.IsSuccessStatusCode)
            return null;

        var json = await response.Content.ReadAsStringAsync(ct);
        using var doc = JsonDocument.Parse(json);

        return doc.RootElement.TryGetProperty("version", out var ver)
            ? ver.GetString()
            : null;
    }
}