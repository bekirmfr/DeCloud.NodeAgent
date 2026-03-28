namespace DeCloud.NodeAgent.Core.Interfaces.Qmp;

/// <summary>
/// Thin wrapper over 'virsh qemu-monitor-command' for QMP interactions.
/// Phase D: used for query-block (drive node name discovery).
/// Phase D+1: dirty bitmap add/clear + drive-backup for incremental export.
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

    /// <summary>Phase D+1: Clear (reset) a dirty bitmap after successful export.</summary>
    Task ClearDirtyBitmapAsync(string vmId, string driveNode, string bitmapName, CancellationToken ct = default);

    /// <summary>Phase D+1: Start incremental drive-backup block job and wait for completion.</summary>
    Task<string> DriveBackupIncrementalAsync(
        string vmId,
        string driveNode,
        string bitmapName,
        string targetPath,
        CancellationToken ct = default);
}