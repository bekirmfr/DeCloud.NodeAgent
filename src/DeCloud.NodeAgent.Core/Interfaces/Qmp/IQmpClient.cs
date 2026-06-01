namespace DeCloud.NodeAgent.Core.Interfaces.Qmp;

/// <summary>
/// Thin wrapper over 'virsh qemu-monitor-command' (QMP) and
/// 'virsh qemu-agent-command' (guest agent). Both flow through virsh so
/// the implementation can share execution and escaping machinery, even
/// though they target different sockets on the QEMU process.
///
/// Phase D:   query-block (drive node name discovery).
/// Phase D+1: dirty bitmap add/clear + drive-backup for incremental export.
/// Phase D+2: guest-fsfreeze for application-consistent capture.
/// Phase H:   drive-backup split into Start + Wait so the freeze envelope
///            brackets only the snapshot-point creation (one QMP round-trip),
///            not the whole copy. Adds sync=top for first-cycle full overlay
///            capture inside the live qemu's coherent block layer, eliminating
///            the cross-process coherence gap of qemu-img convert --force-share.
/// </summary>
public interface IQmpClient
{
    /// <summary>Send a raw QMP command to the QEMU monitor for a running VM.</summary>
    Task<System.Text.Json.JsonElement> SendAsync(
        string vmId,
        string execute,
        object? arguments = null,
        CancellationToken ct = default);

    /// <summary>
    /// Return the QEMU block node name for the primary virtio disk (e.g. "virtio0").
    /// Used to target dirty bitmap and drive-backup commands at the correct drive.
    /// </summary>
    Task<string?> GetPrimaryDriveNodeAsync(string vmId, CancellationToken ct = default);

    /// <summary>Phase D+1: Add persistent dirty bitmap to the primary drive.</summary>
    Task AddDirtyBitmapAsync(string vmId, string driveNode, string bitmapName, CancellationToken ct = default);

    /// <summary>Phase D+1: Remove a dirty bitmap (e.g. before re-adding on migrated VMs).</summary>
    Task RemoveDirtyBitmapAsync(string vmId, string driveNode, string bitmapName, CancellationToken ct = default);

    /// <summary>Phase D+1: Clear (reset) a dirty bitmap after successful export.</summary>
    Task ClearDirtyBitmapAsync(string vmId, string driveNode, string bitmapName, CancellationToken ct = default);

    /// <summary>
    /// Phase H: Start an incremental drive-backup block job and return its
    /// job-id immediately. QEMU installs the copy-on-write snapshot point
    /// synchronously when this call returns; the actual copy runs
    /// asynchronously, observed via WaitForBackupJobAsync.
    ///
    /// Callers wrap only this Start call in the freeze envelope. Freeze
    /// duration becomes one QMP round-trip regardless of how many clusters
    /// the backup will copy — pre-Phase-H this scaled with disk delta size.
    /// </summary>
    Task<string> StartDriveBackupIncrementalAsync(
        string vmId,
        string driveNode,
        string bitmapName,
        string targetPath,
        CancellationToken ct = default);

    /// <summary>
    /// Phase H: Start a full-overlay drive-backup via sync=top and return
    /// its job-id. Captures every cluster present in the topmost BDS (the
    /// overlay), skipping backing-chain fall-through. Output is overlay-only
    /// — no external whitelist required.
    ///
    /// Runs as a block job inside the live qemu process, reading through
    /// the same coherent block layer that committed the freeze-flushed data.
    /// Eliminates the cross-process coherence gap that qemu-img convert
    /// --force-share introduces (a separate process maintains its own qcow2
    /// driver state — L2 cache, refcount cache — which doesn't synchronize
    /// with the live qemu's freshly-flushed writes).
    ///
    /// Callers wrap only this Start call in the freeze envelope. Pair with
    /// WaitForBackupJobAsync to await the background copy.
    /// </summary>
    Task<string> StartDriveBackupTopAsync(
        string vmId,
        string driveNode,
        string targetPath,
        CancellationToken ct = default);

    /// <summary>
    /// Phase H: Block until the named drive-backup job completes. Polls
    /// query-block-jobs every 2s; returns when the job reports
    /// status=concluded or disappears from the list (older QEMU versions).
    ///
    /// Runs OUTSIDE the freeze envelope — guest writes proceed normally
    /// during the wait, protected by drive-backup's before-write notifier
    /// which copies old cluster contents to the target before allowing any
    /// in-place overwrite. The captured snapshot is fixed at job creation
    /// time; the wait just observes the copy completing.
    /// </summary>
    Task WaitForBackupJobAsync(
        string vmId,
        string jobId,
        TimeSpan timeout,
        CancellationToken ct = default);

    /// <summary>
    /// Grant this VM's QEMU process rw access to a scratch file by appending
    /// a rule to its per-VM AppArmor .files profile and reloading it.
    /// No-op when AppArmor is not in use (profile file absent).
    /// </summary>
    Task GrantScratchAppArmorAsync(string vmId, string path, CancellationToken ct = default);

    /// <summary>
    /// Revoke the scratch file rule added by GrantScratchAppArmorAsync.
    /// Non-fatal on failure — always called in a finally block.
    /// </summary>
    Task RevokeScratchAppArmorAsync(string vmId, string path, CancellationToken ct = default);

    /// <summary>
    /// Phase D+2: Freeze all mounted filesystems inside the guest via
    /// qemu-guest-agent's guest-fsfreeze-freeze. Forces ext4/xfs journal
    /// commit and dirty-page writeback to the virtio device, producing
    /// an application-consistent snapshot point.
    ///
    /// Returns the number of frozen filesystems on success. Throws if
    /// the guest agent is unreachable or returns an error — callers
    /// should catch and fall back to crash-consistent capture.
    ///
    /// The freeze MUST be matched with a thaw; failure to thaw leaves
    /// the guest unable to write. Callers wrap in try/finally.
    /// </summary>
    Task<int> FsFreezeAsync(string vmId, CancellationToken ct = default);

    /// <summary>
    /// Phase D+2: Thaw guest filesystems previously frozen by
    /// FsFreezeAsync. Returns the number of thawed filesystems.
    ///
    /// Safe to call even if no freeze is currently active (returns 0
    /// or an error that the caller should swallow).
    /// </summary>
    Task<int> FsThawAsync(string vmId, CancellationToken ct = default);

    /// <summary>
    /// Phase D+2: Cheap host-side liveness check on the guest agent.
    /// Returns true if guest-ping returns within ~2 seconds.
    /// Used to decide whether to attempt fsfreeze at all — there's no
    /// point calling freeze if ping is already failing.
    /// </summary>
    Task<bool> GuestAgentPingAsync(string vmId, CancellationToken ct = default);
}