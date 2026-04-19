using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.Qmp;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace DeCloud.NodeAgent.Infrastructure.Services.Resilience;

/// <summary>
/// IQmpClient implementation via 'virsh qemu-monitor-command'.
/// Consistent with the existing virsh pattern used in VmReadinessMonitor
/// and EphemeralSshKeyService — no raw Unix socket management required.
/// </summary>
public class QmpClient : IQmpClient
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<QmpClient> _logger;

    public QmpClient(ICommandExecutor executor, ILogger<QmpClient> logger)
    {
        _executor = executor;
        _logger = logger;
    }

    public async Task<JsonElement> SendAsync(
        string vmId, string execute, object? arguments = null, CancellationToken ct = default)
    {
        var json = arguments == null
            ? $"{{\"execute\":\"{execute}\"}}"
            : $"{{\"execute\":\"{execute}\",\"arguments\":{JsonSerializer.Serialize(arguments)}}}";

        var escaped = json.Replace("\\", "\\\\").Replace("\"", "\\\"");
        var args = $"qemu-monitor-command {vmId} --pretty \"{escaped}\"";

        var result = await _executor.ExecuteAsync("virsh", args, TimeSpan.FromSeconds(10), ct);
        if (!result.Success)
            throw new InvalidOperationException(
                $"QMP {execute} failed for VM {vmId}: {result.StandardError}");

        using var doc = JsonDocument.Parse(result.StandardOutput.Trim());
        // virsh --pretty wraps the QMP response in {"return": ...}
        // Detect QMP-level error and surface as exception so callers can handle it.
        if (doc.RootElement.TryGetProperty("error", out var err))
        {
            var desc = err.TryGetProperty("desc", out var d) ? d.GetString() : "unknown";
            throw new InvalidOperationException(
                $"QMP {execute} error for VM {vmId}: {desc}");
        }
        if (doc.RootElement.TryGetProperty("return", out var ret))
            return ret.Clone();

        return doc.RootElement.Clone();
    }

    public async Task<string?> GetPrimaryDriveNodeAsync(string vmId, CancellationToken ct = default)
    {
        try
        {
            var ret = await SendAsync(vmId, "query-block", ct: ct);
            // Find the first device that has a backing file (the overlay drive)
            foreach (var dev in ret.EnumerateArray())
            {
                if (dev.TryGetProperty("inserted", out var ins) &&
                    ins.TryGetProperty("node-name", out var node))
                    return node.GetString();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "query-block failed for VM {VmId}", vmId);
        }
        return null;
    }

    public async Task AddDirtyBitmapAsync(
        string vmId, string driveNode, string bitmapName, CancellationToken ct = default)
    {
        await SendAsync(vmId, "block-dirty-bitmap-add", new
        {
            node = driveNode,
            name = bitmapName,
            persistent = true
        }, ct);
        _logger.LogDebug("Added dirty bitmap {Name} to {Drive} on VM {VmId}", bitmapName, driveNode, vmId);
    }

    public async Task ClearDirtyBitmapAsync(
        string vmId, string driveNode, string bitmapName, CancellationToken ct = default)
    {
        await SendAsync(vmId, "block-dirty-bitmap-clear", new
        {
            node = driveNode,
            name = bitmapName
        }, ct);
        _logger.LogDebug("Cleared dirty bitmap {Name} on VM {VmId}", bitmapName, vmId);
    }

    public async Task<string> DriveBackupIncrementalAsync(
        string vmId, string driveNode, string bitmapName, string targetPath, CancellationToken ct = default)
    {
        // QEMU requires a non-empty job-id for drive-backup.
        // Without it, QEMU returns "Invalid job ID ''" and creates an empty sparse
        // file at targetPath before failing — producing all-zero output that
        // ScanChunksAsync silently treats as "no changed chunks".
        // job-id contains a hyphen so use Dictionary instead of anonymous object.
        var jobId = $"ls-{vmId[..8]}-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}";

        await SendAsync(vmId, "drive-backup", new Dictionary<string, object>
        {
            ["job-id"] = jobId,
            ["device"] = driveNode,
            ["target"] = targetPath,
            ["format"] = "raw",
            ["sync"] = "incremental",
            ["bitmap"] = bitmapName,
        }, ct);

        // Poll query-block-jobs by job-id until the job completes (max 10 min).
        // Matching by job-id is reliable; "device" field varies across QEMU versions.
        var deadline = DateTime.UtcNow.AddMinutes(10);
        while (DateTime.UtcNow < deadline)
        {
            await Task.Delay(2000, ct);
            var jobs = await SendAsync(vmId, "query-block-jobs", ct: ct);
            var running = false;
            foreach (var job in jobs.EnumerateArray())
            {
                if (job.TryGetProperty("id", out var id) && id.GetString() == jobId)
                {
                    running = true;
                    if (job.TryGetProperty("status", out var status) &&
                        status.GetString() == "concluded")
                        return targetPath;
                }
            }
            if (!running) return targetPath; // job finished and removed from list
        }
        throw new TimeoutException($"drive-backup timed out for VM {vmId}");
    }
}