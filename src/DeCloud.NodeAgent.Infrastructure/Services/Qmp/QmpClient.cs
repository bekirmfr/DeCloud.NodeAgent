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
            // QEMU 8.2.2 / libvirt: the top-level "device" field is always empty.
            // Identify the primary disk by two stable invariants:
            //   1. Not read-only (excludes cloud-init ISO CDROM)
            //   2. backing_file_depth >= 1 (is an overlay on top of the base image;
            //      excludes the base image node and raw file storage layer nodes)
            // Return inserted.node-name (e.g. "libvirt-2-format") — the block node
            // identifier accepted by blockdev-snapshot's "node" argument.
            // The "file" field is not used — after blockdev-add + blockdev-snapshot
            // it becomes a JSON blob string, not a plain path.
            foreach (var dev in ret.EnumerateArray())
            {
                if (!dev.TryGetProperty("inserted", out var ins)) continue;
                if (ins.TryGetProperty("ro", out var ro) && ro.GetBoolean()) continue;
                if (!ins.TryGetProperty("backing_file_depth", out var depth)) continue;
                if (depth.GetInt32() < 1) continue;
                if (ins.TryGetProperty("node-name", out var nodeName))
                {
                    var name = nodeName.GetString();
                    if (!string.IsNullOrEmpty(name))
                    {
                        _logger.LogDebug(
                            "VM {VmId}: primary disk node={Node}", vmId, name);
                        return name;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "query-block failed for VM {VmId}", vmId);
        }
        return null;
    }

    public async Task<string> CreateSnapshotAsync(
        string vmId, string driveNode, string snapshotPath, CancellationToken ct = default)
    {
        // Phase J: blockdev-snapshot-sync atomically redirects the VM's write
        // path to a new qcow2 at snapshotPath, making the current disk immutable
        // at this exact point in time.
        //
        // Why this is coherent and drive-backup is not:
        //   drive-backup installs a COW snapshot point at job creation but then
        //   reads clusters sequentially over tens of seconds, unfrozen. Clusters
        //   are individually correct at T=0 but coupled ext4 metadata structures
        //   (journal, inode bitmap, inode table) are read at different wall-clock
        //   moments, producing temporal incoherence. Observed on msi-1867:
        //   journal transactions 9331 and 9332 absent from capture, dozens of
        //   directory entries pointing to "deleted/unused" inodes, VM unbootable.
        //
        //   blockdev-snapshot-sync performs no data movement. The snapshot IS the
        //   original disk file, frozen at the moment of the QMP call. The guest
        //   is frozen for one QMP round-trip; no writes can occur; every byte in
        //   the original file is at the same filesystem transaction boundary.
        //
        // Caller must hold guest-fsfreeze across this call.
        // snapshotPath must be pre-created and AppArmor-granted before this call.
        //
        // The return value is the QEMU block node name of the new live overlay
        // (auto-generated by QEMU). Pass it to DeleteSnapshotNodeAsync after
        // lazysync completes to detach the backing file relationship.
        // Node name for the new overlay — unique per VM, short (under QEMU's
        // 31-char node-name limit), identifiable in query-named-block-nodes output.
        var newNodeName = $"ls-ov-{vmId[..8]}";

        // QEMU 8.2.2 (libvirt-managed): blockdev-snapshot-sync does not accept
        // the libvirt node-name as a device identifier, and the top-level device
        // field is empty. The correct sequence is:
        //   1. blockdev-add: register the pre-created qcow2 as a named block node
        //   2. blockdev-snapshot: atomically redirect writes from the disk node
        //      to the new overlay node
        // Both tested and confirmed working on QEMU 8.2.2 / libvirt.
        // The overlay file must be chown'd to libvirt-qemu before blockdev-add.
        await SendAsync(vmId, "blockdev-add", new Dictionary<string, object>
        {
            ["driver"] = "qcow2",
            ["node-name"] = newNodeName,
            ["file"] = new Dictionary<string, object>
            {
                ["driver"] = "file",
                ["filename"] = snapshotPath,
            },
        }, ct);

        await SendAsync(vmId, "blockdev-snapshot", new Dictionary<string, object>
        {
            ["node"] = driveNode,       // libvirt-2-format — disk's block node name
            ["overlay"] = newNodeName,  // ls-ov-{vmId[..8]} — just registered above
        }, ct);

        _logger.LogDebug(
            "VM {VmId}: snapshot created via blockdev-add + blockdev-snapshot, " +
            "overlay node={Node}", vmId, newNodeName);

        return newNodeName;
    }

    public async Task DeleteSnapshotNodeAsync(
        string vmId, string newOverlayNode, string snapshotPath, CancellationToken ct = default)
    {
        // Phase J: merge the overlay back into disk.qcow2 and remove it.
        //
        // After blockdev-snapshot, the block graph is:
        //   disk.qcow2 (libvirt-2-format, backing/frozen)
        //     ← newOverlayNode (ls-ov-*, live, receives all guest writes)
        //
        // Teardown sequence confirmed on QEMU 8.2.2:
        //   1. block-commit: merge newOverlayNode's content into libvirt-2-format
        //      (disk.qcow2). Runs as a background job. On a fresh snapshot with
        //      minimal writes the job completes near-instantly (status=ready).
        //   2. job-complete: signal QEMU to finalize the commit job and restore
        //      libvirt-2-format as the active write destination. After this,
        //      disk.qcow2 is the live disk again.
        //   3. blockdev-del: remove the now-detached overlay block node.
        //   4. Delete the overlay file from disk.
        //
        // All steps non-fatal: if any QMP step fails, log at Warning and proceed.
        // The overlay file is always deleted in the finally path. An orphaned
        // overlay node (if blockdev-del fails) is handled by crash-recovery in
        // LibvirtVmManager on next startup.
        var jobId = $"ls-commit-{vmId[..8]}";
        try
        {
            // Step 1: block-commit — merge overlay into disk.qcow2
            await SendAsync(vmId, "block-commit", new Dictionary<string, object>
            {
                ["device"] = newOverlayNode,
                ["top-node"] = newOverlayNode,
                ["base-node"] = "libvirt-2-format",
                ["job-id"] = jobId,
            }, ct);

            // Step 2: poll until job is ready (offset == len), then finalize
            var deadline = DateTime.UtcNow.AddMinutes(10);
            while (DateTime.UtcNow < deadline)
            {
                await Task.Delay(1000, ct);
                var jobs = await SendAsync(vmId, "query-block-jobs", ct: ct);
                var found = false;
                foreach (var job in jobs.EnumerateArray())
                {
                    if (!job.TryGetProperty("device", out var id)) continue;
                    if (id.GetString() != jobId) continue;
                    found = true;
                    if (job.TryGetProperty("ready", out var ready) && ready.GetBoolean())
                        goto jobReady;
                }
                if (!found) goto jobReady; // job auto-dismissed
            }
            throw new TimeoutException($"block-commit job {jobId} timed out for VM {vmId}");

        jobReady:
            await SendAsync(vmId, "job-complete", new Dictionary<string, object>
            {
                ["id"] = jobId,
            }, ct);

            _logger.LogDebug(
                "VM {VmId}: block-commit complete, disk.qcow2 restored as live disk",
                vmId);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "VM {VmId}: block-commit teardown failed for overlay {Node} — " +
                "crash-recovery will handle orphan on next startup", vmId, newOverlayNode);
        }

        // Step 3: remove the overlay block node from QEMU's block graph
        try
        {
            await SendAsync(vmId, "blockdev-del", new Dictionary<string, object>
            {
                ["node-name"] = newOverlayNode,
            }, ct);
            _logger.LogDebug("VM {VmId}: overlay block node {Node} removed", vmId, newOverlayNode);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "VM {VmId}: blockdev-del failed for node {Node}", vmId, newOverlayNode);
        }

        // Step 4: delete overlay file from disk
        if (File.Exists(snapshotPath))
        {
            try { File.Delete(snapshotPath); }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "VM {VmId}: failed to delete overlay file {Path}", vmId, snapshotPath);
            }
        }
    }

    public async Task GrantScratchAppArmorAsync(string vmId, string path, CancellationToken ct = default)
    {
        // libvirt generates /etc/apparmor.d/libvirt/libvirt-<uuid>.files at VM start
        // and updates it on disk attach/detach. We write the same rule format directly,
        // then reload the profile — no PCI slot required, scope remains per-VM.
        var filesProfile = $"/etc/apparmor.d/libvirt/libvirt-{vmId}.files";
        if (!File.Exists(filesProfile))
        {
            _logger.LogDebug("VM {VmId}: AppArmor .files profile not found — skipping grant", vmId);
            return; // AppArmor not active on this host
        }

        await File.AppendAllTextAsync(filesProfile, $"  \"{path}\" rw,\n", ct);

        var result = await _executor.ExecuteAsync(
            "apparmor_parser",
            $"-r /etc/apparmor.d/libvirt/libvirt-{vmId}",
            TimeSpan.FromSeconds(10), ct);

        if (!result.Success)
            throw new InvalidOperationException(
                $"apparmor_parser reload failed for VM {vmId}: {result.StandardError}");

        _logger.LogDebug("VM {VmId}: AppArmor scratch access granted ({Path})", vmId, path);
    }

    public async Task RevokeScratchAppArmorAsync(string vmId, string path, CancellationToken ct = default)
    {
        var filesProfile = $"/etc/apparmor.d/libvirt/libvirt-{vmId}.files";
        if (!File.Exists(filesProfile)) return;

        var rule = $"  \"{path}\" rw,";
        var lines = await File.ReadAllLinesAsync(filesProfile, ct);
        var filtered = lines.Where(l => l.TrimEnd() != rule).ToArray();

        if (filtered.Length == lines.Length) return; // rule already absent — nothing to do

        await File.WriteAllLinesAsync(filesProfile, filtered, ct);

        var result = await _executor.ExecuteAsync(
            "apparmor_parser",
            $"-r /etc/apparmor.d/libvirt/libvirt-{vmId}",
            TimeSpan.FromSeconds(10), ct);

        if (!result.Success)
            _logger.LogWarning("VM {VmId}: apparmor_parser reload failed during revoke: {Err}",
                vmId, result.StandardError);
        else
            _logger.LogDebug("VM {VmId}: AppArmor scratch access revoked ({Path})", vmId, path);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase D+2: guest agent commands (qemu-agent-command, not qemu-monitor-command)
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Send a raw guest-agent command. Same escaping pattern as SendAsync
    /// but targets qemu-agent-command, which goes to the guest agent socket
    /// (vsock or virtio-serial) rather than the QMP monitor socket.
    /// </summary>
    private async Task<JsonElement> SendAgentAsync(
        string vmId, string execute, object? arguments = null,
        TimeSpan? timeout = null, CancellationToken ct = default)
    {
        var json = arguments == null
            ? $"{{\"execute\":\"{execute}\"}}"
            : $"{{\"execute\":\"{execute}\",\"arguments\":{JsonSerializer.Serialize(arguments)}}}";

        var escaped = json.Replace("\\", "\\\\").Replace("\"", "\\\"");
        var args = $"qemu-agent-command {vmId} \"{escaped}\"";

        var result = await _executor.ExecuteAsync(
            "virsh", args, timeout ?? TimeSpan.FromSeconds(10), ct);

        if (!result.Success)
            throw new InvalidOperationException(
                $"Guest agent {execute} failed for VM {vmId}: {result.StandardError}");

        using var doc = JsonDocument.Parse(result.StandardOutput.Trim());
        if (doc.RootElement.TryGetProperty("error", out var err))
        {
            var desc = err.TryGetProperty("desc", out var d) ? d.GetString() : "unknown";
            throw new InvalidOperationException(
                $"Guest agent {execute} error for VM {vmId}: {desc}");
        }
        if (doc.RootElement.TryGetProperty("return", out var ret))
            return ret.Clone();

        return doc.RootElement.Clone();
    }

    public async Task<int> FsFreezeAsync(string vmId, CancellationToken ct = default)
    {
        // Generous timeout: freeze drives a sync inside the guest, which on a
        // VM under write pressure can take many seconds. 30s is the floor used
        // by libvirt's own virsh domfsfreeze when no override is given.
        var ret = await SendAgentAsync(
            vmId, "guest-fsfreeze-freeze",
            timeout: TimeSpan.FromSeconds(30), ct: ct);

        var count = ret.ValueKind == JsonValueKind.Number ? ret.GetInt32() : 0;
        _logger.LogDebug("VM {VmId}: froze {Count} filesystem(s)", vmId, count);
        return count;
    }

    public async Task<int> FsThawAsync(string vmId, CancellationToken ct = default)
    {
        var ret = await SendAgentAsync(
            vmId, "guest-fsfreeze-thaw",
            timeout: TimeSpan.FromSeconds(30), ct: ct);

        var count = ret.ValueKind == JsonValueKind.Number ? ret.GetInt32() : 0;
        _logger.LogDebug("VM {VmId}: thawed {Count} filesystem(s)", vmId, count);
        return count;
    }

    public async Task<bool> GuestAgentPingAsync(string vmId, CancellationToken ct = default)
    {
        try
        {
            await SendAgentAsync(
                vmId, "guest-ping",
                timeout: TimeSpan.FromSeconds(2), ct: ct);
            return true;
        }
        catch
        {
            return false;
        }
    }
}