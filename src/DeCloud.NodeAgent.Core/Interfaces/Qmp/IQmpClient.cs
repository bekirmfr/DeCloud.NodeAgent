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

    /// <summary>Phase D+1: Start incremental drive-backup block job and wait for completion.</summary>
    Task<string> DriveBackupIncrementalAsync(
        string vmId,
        string driveNode,
        string bitmapName,
        string targetPath,
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