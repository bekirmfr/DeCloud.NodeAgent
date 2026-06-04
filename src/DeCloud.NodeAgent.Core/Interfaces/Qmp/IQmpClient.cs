namespace DeCloud.NodeAgent.Core.Interfaces.Qmp;

/// <summary>
/// Thin wrapper over 'virsh qemu-monitor-command' (QMP) and
/// 'virsh qemu-agent-command' (guest agent). Both flow through virsh so
/// the implementation can share execution and escaping machinery, even
/// though they target different sockets on the QEMU process.
///
/// Phase D:   query-block (drive node name discovery).
/// Phase D+2: guest-fsfreeze for application-consistent capture.
/// Phase J:   blockdev-snapshot-sync for coherent point-in-time disk
///            snapshots. Replaces drive-backup (Phases H/I) and the
///            dirty bitmap (Phase D+1). See MIGRATION_SYSTEM_DESIGN.md
///            §6.1.8 for why drive-backup's COW model cannot produce
///            coherent snapshots on an active guest.
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
    /// Used to target snapshot commands at the correct drive.
    /// </summary>
    Task<string?> GetPrimaryDriveNodeAsync(string vmId, CancellationToken ct = default);

    /// <summary>
    /// Phase J: Atomically redirect the VM's write path to a new qcow2
    /// overlay at <paramref name="snapshotPath"/>, making the current disk
    /// (identified by <paramref name="driveNode"/>) read-only and coherent
    /// as of the moment this call returns.
    ///
    /// Must be called inside a guest-fsfreeze envelope so that the guest's
    /// page cache is flushed and no writes are in flight when the snapshot
    /// point is fixed. The freeze envelopes both the creation of the new
    /// overlay and the redirection of the write path — total freeze duration
    /// is one QMP round-trip (sub-second).
    ///
    /// After this call the guest can be thawed: all new writes go to the
    /// new overlay. The original disk file is immutable and can be read by
    /// the lazysync daemon at leisure without any COW race.
    ///
    /// Returns the QEMU block node name of the new (now live) overlay,
    /// needed by DeleteSnapshotNodeAsync to detach it after lazysync.
    /// </summary>
    Task<string> CreateSnapshotAsync(
        string vmId,
        string driveNode,
        string snapshotPath,
        CancellationToken ct = default);

    /// <summary>
    /// Phase J: Detach and delete the snapshot backing file created by
    /// CreateSnapshotAsync after lazysync has finished reading it.
    ///
    /// Issues 'blockdev-snapshot-delete' (or equivalent commit-and-remove
    /// sequence) to disconnect the backing file relationship, then deletes
    /// the file at <paramref name="snapshotPath"/> from disk.
    ///
    /// Safe to call even if the snapshot file no longer exists (idempotent).
    /// </summary>
    Task DeleteSnapshotNodeAsync(
        string vmId,
        string newOverlayNode,
        string snapshotPath,
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