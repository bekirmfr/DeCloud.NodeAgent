using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.Qmp;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using DeCloud.Shared.Enums;
using DeCloud.Shared.Json;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Background service that continuously replicates VM disks to the local
/// BlockStore VM via the lazysync protocol.
///
/// Cycle (every 5 minutes, 5 minute startup delay):
///   For each Running tenant VM with ReplicationFactor > 0:
///   1. Capture the full merged disk as a coherent point-in-time snapshot
///      via QMP drive-backup, bracketed by a guest-fsfreeze envelope:
///         • First cycle: block-dirty-bitmap-add (drive-level), then
///           drive-backup sync=full inside the freeze envelope. Output is
///           a self-contained raw image — no backing-file dependency on
///           reconstruction.
///         • Subsequent cycles: drive-backup sync=incremental against the
///           dirty bitmap inside the freeze envelope. Only clusters
///           written since the last successful clear are exported.
///      The freeze brackets ONLY the QMP Start call (one round-trip,
///      sub-second). drive-backup's copy-on-write notifier fixes the
///      snapshot point at job creation; the actual copy runs unfrozen.
///   2. Scan the captured raw file in 1 MB chunks. Skip all-zero chunks
///      (sparse free space). No qcow2-layer reasoning — the captured
///      image is what the guest sees.
///   3. Compute CIDv1 (raw codec, sha2-256 multihash, base32lower) for
///      each non-zero chunk.
///   4. Compare against stored manifest: push only genuinely changed
///      chunks (content-addressed dedup across all VMs sharing an image).
///   5. POST each new block to the local BlockStore VM HTTP API
///      (POST /blocks).
///   6. Update the in-memory + on-disk manifest state.
///   7. POST updated manifest to the orchestrator.
///   8. Clear the dirty bitmap so the next cycle tracks only writes that
///      occur after this successful push.
///   9. Delete the temp raw file.
///
/// Phase I — Full-disk snapshot replication:
///   Replaces the overlay-only model (Phase D–H) with full-disk capture
///   via drive-backup sync=full. Eliminates the metadata temporal
///   incoherence failure mode where ext4 structures (inode bitmap,
///   journal, inode table) captured at different lazysync versions
///   formed internally inconsistent on-disk state that neither journal
///   replay nor fsck could reconcile (observed 2026-06-03 on msi-0d04).
///   Every confirmed manifest version is now an application-consistent
///   coherent point-in-time snapshot — planned and unplanned recovery
///   produce identical correctness guarantees; only their RPO differs.
///   See MIGRATION_SYSTEM_DESIGN.md §6.1.6 / §6.1.7 and
///   PHASE_I_FULL_DISK_REPLICATION.md.
///
///   Storage cost is bounded by content-addressed dedup: base image
///   CIDs appear once on the network regardless of how many VMs share
///   the image. Aggregate network growth is K × BaseImage × RF, not
///   N × Disk × RF. At platform scale (N≥1000 VMs, K≤20 base images)
///   growth is under 2%.
///
/// State persisted per VM: {VmStoragePath}/{vmId}/lazysync.json
/// </summary>
public class LazysyncDaemon : BackgroundService
{
    private static readonly TimeSpan CycleInterval = TimeSpan.FromMinutes(5);
    private static readonly TimeSpan StartupDelay = TimeSpan.FromMinutes(5);
    private const int BlockSizeBytes = 1024 * 1024; // 1 MB
    private const int BlockSizeKb = 1024;
    private const long SparseScanThreshold = 65536; // -S value for qemu-img convert

    // Bitmap name written to the qcow2 file. Fixed across all VMs so the daemon
    // can find it by name on startup without querying QEMU for the list.
    private const string BitmapName = "lazysync";

    // No per-cycle block cap is applied during seeding.
    //
    // Design rationale:
    //   The receiver side uses a fixed worker pool (GossipSubFetchWorkers=4, FetchQueueSize=2000)
    //   that absorbs any burst without goroutine explosion or block loss. A 1.2 GB overlay
    //   (~1200 blocks) generates ~1200 GossipSub announcements over ~60 seconds — the receiver
    //   queue drains them at 4×parallel, completing in ~25 minutes vs ~65 minutes with a cap.
    //
    // Scale note:
    //   Bitswap load on the source is bounded by ReplicationFactor (max 5), not node count.
    //   XOR threshold filtering ensures only the RF closest nodes pull blocks — the other
    //   N-RF nodes in the network receive GossipSub messages but do not fetch.
    //   Worst case: 5 receivers × 4 workers = 20 concurrent bitswap readers on the source.
    //   This is acceptable at current RF constraints (max RF=5).
    //
    //   If RF is raised significantly (e.g. RF=50+) or overlay sizes grow to 100+ GB,
    //   reinstate a rate-limiter here — not a block count cap but a token bucket or
    //   configurable BlockPushDelayMs between PushBlocksAsync iterations to keep the
    //   pipeline continuous without saturating the source node's uplink.

    private readonly LibvirtVmManager _vmManager;
    private readonly ICommandExecutor _executor;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly LibvirtVmManagerOptions _vmOptions;
    private readonly IQmpClient _qmpClient;
    private readonly HttpClient _blockstoreClient;
    private readonly ILogger<LazysyncDaemon> _logger;

    // Per-VM in-memory state. Populated from lazysync.json on first access.
    private readonly Dictionary<string, LazysyncState> _states = new();

    public LazysyncDaemon(
        LibvirtVmManager vmManager,
        ICommandExecutor executor,
        IOrchestratorClient orchestratorClient,
        IOptions<LibvirtVmManagerOptions> vmOptions,
        IQmpClient qmpClient,
        HttpClient blockstoreClient,
        ILogger<LazysyncDaemon> logger)
    {
        _vmManager = vmManager;
        _executor = executor;
        _orchestratorClient = orchestratorClient;
        _vmOptions = vmOptions.Value;
        _qmpClient = qmpClient;
        _blockstoreClient = blockstoreClient;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("LazysyncDaemon started — first cycle in {Delay}", StartupDelay);
        await Task.Delay(StartupDelay, stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try { await RunCycleAsync(stoppingToken); }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested) { break; }
            catch (Exception ex) { _logger.LogError(ex, "LazysyncDaemon cycle failed"); }

            await Task.Delay(CycleInterval, stoppingToken);
        }

        _logger.LogInformation("LazysyncDaemon stopped");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Cycle
    // ═══════════════════════════════════════════════════════════════════════

    private async Task RunCycleAsync(CancellationToken ct)
    {
        var blockstoreAddr = await FindBlockstoreApiAsync(ct);
        if (blockstoreAddr == null)
        {
            _logger.LogDebug("LazysyncDaemon: no local BlockStore VM running — skipping cycle");
            return;
        }

        var vms = _vmManager.GetRunningVms()
            .Where(v => v.Spec.ReplicationFactor > 0 &&
                        v.Spec.Role == VmRole.General &&
                        v.IsFullyReady)  // wait for cloud-init + guest agent
            .ToList();

        if (vms.Count == 0) return;

        _logger.LogInformation("LazysyncDaemon: syncing {Count} VM(s)", vms.Count);

        foreach (var vm in vms)
        {
            if (ct.IsCancellationRequested) break;
            try { await SyncVmAsync(vm, blockstoreAddr, ct); }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Lazysync failed for VM {VmId} — skipping", vm.VmId);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Per-VM sync
    // ═══════════════════════════════════════════════════════════════════════

    private async Task SyncVmAsync(VmInstance vm, string blockstoreAddr, CancellationToken ct)
    {
        var diskPath = Path.Combine(_vmOptions.VmStoragePath, vm.VmId, "disk.qcow2");
        if (!File.Exists(diskPath))
        {
            _logger.LogDebug("VM {VmId}: no disk.qcow2 found at {Path}", vm.VmId, diskPath);
            return;
        }

        var state = await LoadStateAsync(vm.VmId);
        // Use VM's storage directory. NodeAgent pre-creates the file here;
        // AppArmor access is granted per-VM via GrantScratchAppArmorAsync.
        // /tmp/ is blocked by the per-VM AppArmor confinement regardless.
        var tmpPath = Path.Combine(_vmOptions.VmStoragePath, vm.VmId,
            $"lazysync-tmp-{DateTimeOffset.UtcNow.ToUnixTimeSeconds()}.raw");

        try
        {
            // Tracked so step 7.5 doesn't clear the bitmap immediately after V1.
            // Under Phase G fallback (crash-consistent capture when guest agent is
            // unresponsive), writes can occur during V1's backup window and must
            // remain tracked for V2 to pick up.
            var wasFirstCycle = !state.BitmapCreated;

            if (wasFirstCycle)
            {
                // ── Phase I: First cycle — drive-backup sync=full inside live qemu ──
                // Captures the full merged disk as a coherent point-in-time snapshot:
                // backing chain is dereferenced inline by QEMU's block layer, output
                // is a self-contained raw image. No backing-file dependency on
                // reconstruction; the target needs only the chunks.
                //
                // The bitmap is added BEFORE the backup so V2 captures every write
                // from V1 onwards. The freeze brackets only the Start call: once
                // drive-backup returns the job-id, the copy-on-write snapshot point
                // is fixed and the guest can be thawed. The copy runs unfrozen.
                var driveNode = await TryGetDriveNodeAsync(vm.VmId, ct);
                if (driveNode == null)
                {
                    _logger.LogWarning(
                        "VM {VmId}: QMP drive node unavailable on first cycle — " +
                        "skipping cycle, will retry next interval", vm.VmId);
                    return;
                }

                // Pre-create target as sparse file at virtual disk size.
                // drive-backup mode=existing requires size match; SetLength uses
                // ftruncate() — no actual disk space consumed for the sparse region.
                var virtualSize = await GetDiskVirtualSizeAsync(diskPath, ct);
                using (var fs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write,
                           FileShare.None, bufferSize: 4096, useAsync: false))
                    fs.SetLength(virtualSize);

                // 0600 before chown so there is no window where the file is
                // root:root readable by group/other in a world-traversable directory.
                File.SetUnixFileMode(tmpPath,
                    UnixFileMode.UserRead | UnixFileMode.UserWrite);

                var chownResult = await _executor.ExecuteAsync(
                    "chown", $"libvirt-qemu:libvirt-qemu \"{tmpPath}\"",
                    TimeSpan.FromSeconds(5), ct);
                if (!chownResult.Success)
                    throw new InvalidOperationException(
                        $"chown libvirt-qemu failed for {tmpPath}: {chownResult.StandardError}");

                // Bitmap may already exist if this VM was migrated from another node.
                // Remove then re-add to ensure a clean tracking window from V1 onwards.
                try
                {
                    await _qmpClient.RemoveDirtyBitmapAsync(vm.VmId, driveNode, BitmapName, ct);
                    _logger.LogDebug(
                        "VM {VmId}: removed pre-existing lazysync bitmap", vm.VmId);
                }
                catch
                {
                    // Not present — fine, ignore.
                }

                try
                {
                    await _qmpClient.AddDirtyBitmapAsync(vm.VmId, driveNode, BitmapName, ct);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex,
                        "VM {VmId}: AddDirtyBitmap failed on first cycle — " +
                        "skipping cycle, will retry next interval", vm.VmId);
                    if (File.Exists(tmpPath)) File.Delete(tmpPath);
                    return;
                }

                var attached = false;
                string jobId;
                try
                {
                    await _qmpClient.GrantScratchAppArmorAsync(vm.VmId, tmpPath, ct);
                    attached = true;

                    // Freeze brackets ONLY the Start call. Snapshot point is fixed
                    // at the moment drive-backup returns its job-id; thaw immediately
                    // and let the copy run in the background.
                    jobId = await RunUnderGuestFreezeAsync(vm.VmId,
                        () => _qmpClient.StartDriveBackupFullAsync(
                            vm.VmId, driveNode, tmpPath, ct),
                        ct);

                    // Wait for the copy unfrozen. drive-backup's before-write notifier
                    // protects the target while the guest continues to write.
                    await _qmpClient.WaitForBackupJobAsync(
                        vm.VmId, jobId, TimeSpan.FromMinutes(10), ct);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex,
                        "VM {VmId}: first-cycle drive-backup sync=full failed — " +
                        "leaving BitmapCreated=false; next cycle will retry V1",
                        vm.VmId);
                    if (File.Exists(tmpPath)) File.Delete(tmpPath);
                    await SaveStateAsync(vm.VmId, state);
                    return;
                }
                finally
                {
                    // Only revoke if grant succeeded — avoids a spurious apparmor_parser
                    // error when the catch path is entered due to a grant failure.
                    if (attached)
                        await _qmpClient.RevokeScratchAppArmorAsync(vm.VmId, tmpPath, ct);
                }

                state.BitmapCreated = true;
            }
            else
            {
                // ── Subsequent cycles: incremental export via QMP ─────────────────
                // Only clusters written since last ClearDirtyBitmap are exported.
                // Output raw file is already overlay-only — no whitelist filter needed.
                var driveNode = await TryGetDriveNodeAsync(vm.VmId, ct);
                if (driveNode == null)
                {
                    _logger.LogWarning(
                        "VM {VmId}: QMP drive node unavailable — resetting to full export next cycle",
                        vm.VmId);
                    state.BitmapCreated = false;
                    await SaveStateAsync(vm.VmId, state);
                    return;
                }

                // Pre-create as a sparse file at the exact virtual disk size.
                // drive-backup mode=existing requires size match; SetLength uses
                // ftruncate() — no actual disk space consumed for the sparse region.
                var virtualSize = await GetDiskVirtualSizeAsync(diskPath, ct);
                using (var fs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write,
                           FileShare.None, bufferSize: 4096, useAsync: false))
                    fs.SetLength(virtualSize);

                // 0600 before chown so there is no window where the file is
                // root:root readable by group/other in a world-traversable directory.
                File.SetUnixFileMode(tmpPath,
                    UnixFileMode.UserRead | UnixFileMode.UserWrite);

                var chownResult = await _executor.ExecuteAsync(
                    "chown", $"libvirt-qemu:libvirt-qemu \"{tmpPath}\"",
                    TimeSpan.FromSeconds(5), ct);
                if (!chownResult.Success)
                    throw new InvalidOperationException(
                        $"chown libvirt-qemu failed for {tmpPath}: {chownResult.StandardError}");

                // Grant this VM's QEMU process rw access via per-VM AppArmor .files profile.
                var attached = false;
                try
                {
                    await _qmpClient.GrantScratchAppArmorAsync(vm.VmId, tmpPath, ct);
                    attached = true;

                    // Phase H: freeze brackets only the Start call. drive-backup's
                    // copy-on-write notifier installs at job creation, fixing the
                    // snapshot point — the copy runs unfrozen in the background.
                    // Freeze duration is bounded by one QMP round-trip; pre-Phase-H
                    // this scaled with the incremental delta size, which on a busy
                    // VM with a large dirty bitmap could be many seconds.
                    //
                    // Freeze still matters: it forces the guest to flush its page
                    // cache into QEMU before the snapshot point is taken, so the
                    // bitmap snapshot reflects the freshly-flushed clusters rather
                    // than deferring them to the next cycle.
                    var jobId = await RunUnderGuestFreezeAsync(vm.VmId,
                        () => _qmpClient.StartDriveBackupIncrementalAsync(
                            vm.VmId, driveNode, BitmapName, tmpPath, ct),
                        ct);
                    await _qmpClient.WaitForBackupJobAsync(
                        vm.VmId, jobId, TimeSpan.FromMinutes(10), ct);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex,
                        "VM {VmId}: incremental backup failed — resetting to full export next cycle",
                        vm.VmId);
                    state.BitmapCreated = false;
                    if (File.Exists(tmpPath)) File.Delete(tmpPath);
                    // Persist the BitmapCreated=false reset so the next cycle uses full export.
                    // Without this, LoadStateAsync would read BitmapCreated=true from disk and
                    // retry the failing incremental path indefinitely, never falling back.
                    await SaveStateAsync(vm.VmId, state);
                    return;
                }
                finally
                {
                    // Only revoke if grant succeeded — avoids a spurious apparmor_parser
                    // error when the catch path is entered due to a grant failure.
                    if (attached)
                        await _qmpClient.RevokeScratchAppArmorAsync(vm.VmId, tmpPath, ct);
                }
            }

            // Step 2-4: Scan raw, skip all-zero chunks, compute CIDs, diff against manifest.
            var (changedChunks, totalBytes) = await ScanChunksAsync(tmpPath, state, ct);

            if (changedChunks.Count == 0)
            {
                _logger.LogDebug("VM {VmId}: no changed chunks this cycle", vm.VmId);
                return;
            }

            // All changed blocks are pushed in a single cycle — no cap applied.
            // The receiver's worker pool handles the burst safely. See class-level
            // comment on SeedingMaxBlocksPerCycle for full rationale.
            var isSeeding = state.Version == 0;

            // Step 5: Push new blocks to BlockStore
            await PushBlocksAsync(vm.VmId, tmpPath, changedChunks, blockstoreAddr, state.Version + 1, ct);

            // Step 6: Update state
            foreach (var (offset, cid) in changedChunks)
                state.Chunks[offset] = cid;
            state.Version++;
            state.TotalBytes = totalBytes;
            state.LastSyncAt = DateTime.UtcNow;

            // Step 7: Report manifest to orchestrator
            var rootCid = ComputeRootCid(state.Chunks);
            await _orchestratorClient.RegisterManifestAsync(
                vm.VmId,
                rootCid,
                state.Version,
                changedChunks.Select(c => c.Cid).ToList(),
                state.Chunks.Count,
                BlockSizeKb,
                totalBytes,
                isSeeding,
                vm.Spec.ReplicationFactor,
                state.Chunks,
                ct);

            // Step 7.5: Clear the dirty bitmap so the next cycle tracks only
            // writes that occur after this successful push.
            //
            // Skipped on the V1 cycle (wasFirstCycle) because the bitmap was
            // added BEFORE V1's drive-backup; under Phase G fallback (crash-
            // consistent capture when the guest agent is unresponsive), writes
            // can occur during V1's backup window and must remain tracked for
            // V2 to pick up. With Phase G's freeze active, the bitmap is empty
            // at V1's end and clearing would be a no-op anyway; the guard
            // matters only for the fallback path.
            if (state.BitmapCreated && !wasFirstCycle)
            {
                var driveNode = await TryGetDriveNodeAsync(vm.VmId, ct);
                if (driveNode != null)
                {
                    try
                    {
                        await _qmpClient.ClearDirtyBitmapAsync(vm.VmId, driveNode, BitmapName, ct);
                    }
                    catch (Exception ex)
                    {
                        // Non-fatal: next cycle falls back to full export.
                        _logger.LogWarning(ex,
                            "VM {VmId}: ClearDirtyBitmap failed — resetting to full export next cycle",
                            vm.VmId);
                        state.BitmapCreated = false;
                    }
                }
                else
                {
                    _logger.LogWarning(
                        "VM {VmId}: drive node lost after export — resetting to full export next cycle",
                        vm.VmId);
                    state.BitmapCreated = false;
                }
            }

            // Step 8: Persist state
            await SaveStateAsync(vm.VmId, state);

            // Step 9: Notify local blockstore VM so its /manifests endpoint
            // reflects the current manifest — populates the dashboard Resources table.
            // Fire-and-forget: dashboard display is non-critical, never block the cycle.
            // ReplicationFactor flows from VmSpec → blockstore's owners/{vmId}.meta →
            // GossipSub new-blocks announcements, letting peers self-organize repair
            // without orchestrator round-trips (Phase B+).
            _ = NotifyBlockstoreManifestAsync(
                blockstoreAddr, vm.VmId, rootCid, state, changedChunks, totalBytes,
                vm.Spec.ReplicationFactor, ct);

            _logger.LogInformation(
                "VM {VmId}: lazysync v{Version} — {Changed} changed / {Total} chunks, {Bytes} bytes",
                vm.VmId, state.Version, changedChunks.Count, state.Chunks.Count, totalBytes);
        }
        finally
        {
            if (File.Exists(tmpPath))
                File.Delete(tmpPath);
        }
    }

    private async Task NotifyBlockstoreManifestAsync(
        string blockstoreAddr,
        string vmId,
        string rootCid,
        LazysyncState state,
        List<(long Offset, string Cid)> changedChunks,
        long totalBytes,
        int replicationFactor,
        CancellationToken ct)
    {
        try
        {
            var payload = new
            {
                vmId,
                rootCid,
                version = state.Version,
                blockCount = state.Chunks.Count,
                blockSizeKb = BlockSizeKb,
                totalBytes,
                resourceType = "VMOverlay",
                resourceId = vmId,
                chunkCids = changedChunks.Select(c => c.Cid).ToList(),
                replicationFactor
            };

            var json = System.Text.Json.JsonSerializer.Serialize(payload);
            var content = new System.Net.Http.StringContent(
                json, System.Text.Encoding.UTF8, "application/json");

            var response = await _blockstoreClient.PostAsync(
                $"{blockstoreAddr}/manifests", content, ct);

            if (!response.IsSuccessStatusCode)
                _logger.LogDebug(
                    "VM {VmId}: blockstore manifest notify returned {Status}",
                    vmId, response.StatusCode);
        }
        catch (Exception ex)
        {
            // Non-fatal — dashboard display only
            _logger.LogDebug(ex, "VM {VmId}: blockstore manifest notify failed", vmId);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Chunk scan + CID computation
    // ═══════════════════════════════════════════════════════════════════════

    private async Task<(List<(long Offset, string Cid)> Changed, long TotalBytes)>
        ScanChunksAsync(string rawPath, LazysyncState state, CancellationToken ct)
    {
        var changed = new List<(long Offset, string Cid)>();
        long totalBytes = 0;
        var buf = new byte[BlockSizeBytes];

        await using var fs = new FileStream(
            rawPath, FileMode.Open, FileAccess.Read, FileShare.None,
            bufferSize: 1024 * 1024, useAsync: true);

        long offset = 0;
        while (true)
        {
            ct.ThrowIfCancellationRequested();

            var read = await fs.ReadAsync(buf, 0, BlockSizeBytes, ct);
            if (read == 0) break;

            // Skip all-zero chunks — unallocated / sparse free space.
            // Base image clusters that the guest hasn't read or written through
            // are dereferenced by qemu's block layer into either real bytes
            // (then they get CIDs and replicate) or zeros (then they're skipped
            // here, same as a hole in a fresh-deploy overlay).
            if (IsAllZero(buf, read))
            {
                offset += read;
                continue;
            }

            totalBytes += read;
            var cid = ComputeCidV1(buf.AsSpan(0, read));

            // Only include if CID actually changed (content-addressed dedup).
            if (!state.Chunks.TryGetValue(offset, out var existing) || existing != cid)
                changed.Add((offset, cid));

            offset += read;
        }

        return (changed, totalBytes);
    }

    private static bool IsAllZero(byte[] buf, int count)
    {
        for (var i = 0; i < count; i++)
            if (buf[i] != 0) return false;
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CIDv1 computation (raw codec, sha2-256, base32lower multibase)
    // CIDv1 binary: [0x01 version][0x55 raw codec][multihash]
    // multihash:    [0x12 sha2-256][0x20 = 32 bytes][sha256 digest]
    // ═══════════════════════════════════════════════════════════════════════

    private static string ComputeCidV1(ReadOnlySpan<byte> data)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(data, hash);

        // Build CID bytes: version(1) + codec(0x55) + multihash(0x12, 0x20, hash)
        var cid = new byte[36];
        cid[0] = 0x01; cid[1] = 0x55; // version + raw codec
        cid[2] = 0x12; cid[3] = 0x20; // sha2-256 code + length
        hash.CopyTo(cid.AsSpan(4));

        return "b" + ToBase32Lower(cid);
    }

    private static string ComputeRootCid(Dictionary<long, string> chunks)
    {
        // Root CID = CIDv1 of the sorted concatenated chunk CIDs (manifest fingerprint)
        var sorted = chunks.OrderBy(kv => kv.Key).Select(kv => kv.Value);
        var payload = Encoding.UTF8.GetBytes(string.Join("\n", sorted));
        return ComputeCidV1(payload);
    }

    private static readonly char[] Base32Alphabet = "abcdefghijklmnopqrstuvwxyz234567".ToCharArray();

    private static string ToBase32Lower(byte[] data)
    {
        var sb = new StringBuilder();
        int buf = 0, bitsLeft = 0;
        foreach (var b in data)
        {
            buf = (buf << 8) | b;
            bitsLeft += 8;
            while (bitsLeft >= 5)
            {
                bitsLeft -= 5;
                sb.Append(Base32Alphabet[(buf >> bitsLeft) & 0x1F]);
            }
        }
        if (bitsLeft > 0)
            sb.Append(Base32Alphabet[(buf << (5 - bitsLeft)) & 0x1F]);
        return sb.ToString();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Block push to local BlockStore VM
    // ═══════════════════════════════════════════════════════════════════════

    private async Task PushBlocksAsync(
        string vmId, 
        string rawPath,
        List<(long Offset, string Cid)> changedChunks,
        string blockstoreAddr, 
        int manifestVersion, 
        CancellationToken ct)
    {
        var buf = new byte[BlockSizeBytes];

        await using var fs = new FileStream(
            rawPath, FileMode.Open, FileAccess.Read, FileShare.None,
            bufferSize: 1024 * 1024, useAsync: true);

        foreach (var (offset, cid) in changedChunks)
        {
            ct.ThrowIfCancellationRequested();

            fs.Seek(offset, SeekOrigin.Begin);
            var read = await fs.ReadAsync(buf, 0, BlockSizeBytes, ct);
            if (read == 0) continue;

            var content = new ByteArrayContent(buf, 0, read);
            content.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue("application/octet-stream");

            var url = $"{blockstoreAddr}/blocks?cid={Uri.EscapeDataString(cid)}&owner={Uri.EscapeDataString(vmId)}&manifestVersion={manifestVersion}";
            var response = await _blockstoreClient.PostAsync(url, content, ct);

            if (!response.IsSuccessStatusCode)
            {
                var err = await response.Content.ReadAsStringAsync(ct);
                _logger.LogWarning(
                    "VM {VmId}: block push failed for CID {Cid}: HTTP {Status} — {Err}",
                    vmId, cid[..16], response.StatusCode, err);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BlockStore VM discovery
    // ═══════════════════════════════════════════════════════════════════════

    private async Task<string?> FindBlockstoreApiAsync(CancellationToken ct)
    {
        var vms = _vmManager.GetRunningVms();
        var blockstoreVm = vms.FirstOrDefault(v => v.Spec.Role == VmRole.BlockStore);
        if (blockstoreVm == null || string.IsNullOrEmpty(blockstoreVm.Spec.IpAddress))
            return null;

        // BlockStore VM API port: 5090 (BlockStoreVmSpec.ApiPort)
        return $"http://{blockstoreVm.Spec.IpAddress}:5090";
    }

    private async Task<long> GetDiskVirtualSizeAsync(string diskPath, CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync(
            "qemu-img", $"info --force-share --output=json \"{diskPath}\"",
            TimeSpan.FromSeconds(30), ct);

        if (!result.Success)
            throw new Exception($"qemu-img info failed for {diskPath}: {result.StandardError}");

        using var doc = JsonDocument.Parse(result.StandardOutput);
        return doc.RootElement.GetProperty("virtual-size").GetInt64();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Application-consistent capture via guest agent fsfreeze
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Execute <paramref name="captureBody"/> inside a guest-fsfreeze /
    /// guest-fsthaw bracket and return its result. Callers pass the minimal
    /// operation that must happen while the guest is quiesced — in Phase H,
    /// that's the QMP Start call which creates a drive-backup job and fixes
    /// its copy-on-write snapshot point. The actual copy runs unfrozen in
    /// the background, awaited via WaitForBackupJobAsync.
    ///
    /// Bounded freeze duration: the bracket holds for one QMP round-trip
    /// (sub-millisecond on the same host) regardless of VM disk size or
    /// backup data volume. Pre-Phase-H the body was the full backup copy,
    /// which scaled with delta size and did not bound the guest write-pause.
    ///
    /// Best-effort: if the guest agent is unreachable or freeze fails,
    /// runs <paramref name="captureBody"/> without freeze (crash-consistent
    /// fallback — equivalent to pre-D+2 behaviour). Logs a warning so
    /// repeated agent failures are visible.
    ///
    /// Thaw runs in a finally block. If the body throws, the guest is still
    /// thawed before the exception propagates. If thaw itself fails after a
    /// successful freeze, that's a logged-but-non-fatal condition — the host
    /// can't unfreeze a guest agent that has stopped responding mid-capture,
    /// but the guest's own fsfreeze-timeout (libvirt's domfsfreeze adds one
    /// by default; we rely on it) will auto-thaw after a bounded interval.
    ///
    /// Why we bother:
    ///   crash-consistent capture sees torn writes — file metadata says the
    ///   file is N bytes long but the last extents may be zero-padded because
    ///   the guest's page cache hadn't flushed when the qcow2 was sampled.
    ///   This was observed on a migrated VM whose libnetplan.cpython-311.pyc
    ///   had a valid header and a tail of zeros, causing cloud-init's
    ///   init-local to throw ValueError: bad marshal data on the target.
    /// </summary>
    private async Task<T> RunUnderGuestFreezeAsync<T>(
        string vmId, Func<Task<T>> captureBody, CancellationToken ct)
    {
        // Cheap precheck — skip the whole dance if the agent isn't responding.
        // Avoids a 30s timeout per cycle for VMs without qemu-guest-agent.
        var agentOk = await _qmpClient.GuestAgentPingAsync(vmId, ct);
        if (!agentOk)
        {
            _logger.LogWarning(
                "VM {VmId}: guest agent unresponsive — capturing crash-consistent " +
                "(application-level data may be torn until agent recovers)",
                vmId);
            return await captureBody();
        }

        var frozen = false;
        try
        {
            try
            {
                var count = await _qmpClient.FsFreezeAsync(vmId, ct);
                frozen = count > 0;
                if (!frozen)
                    _logger.LogWarning(
                        "VM {VmId}: fsfreeze returned 0 filesystems — proceeding without freeze",
                        vmId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "VM {VmId}: fsfreeze failed — capturing crash-consistent", vmId);
            }

            // The body is awaited inside the try; its result is held while the
            // finally runs (thaw), then returned. Exceptions from the body
            // propagate after the finally, still leaving the guest thawed.
            return await captureBody();
        }
        finally
        {
            if (frozen)
            {
                try
                {
                    await _qmpClient.FsThawAsync(vmId, CancellationToken.None);
                }
                catch (Exception ex)
                {
                    // Cannot rethrow from finally without masking the body's exception.
                    // Guest's libvirt-side freeze timeout will auto-thaw eventually.
                    _logger.LogError(ex,
                        "VM {VmId}: fsthaw failed after successful freeze — " +
                        "guest may be paused until libvirt fsfreeze-timeout expires", vmId);
                }
            }
        }
    }

    /// <summary>
    /// Returns the QEMU block node name for a VM's primary disk, or null if
    /// QMP is unavailable (VM starting, stopped, or not a KVM guest).
    /// Failure is non-fatal — caller falls back to full export.
    /// </summary>
    private async Task<string?> TryGetDriveNodeAsync(string vmId, CancellationToken ct)
    {
        try
        {
            return await _qmpClient.GetPrimaryDriveNodeAsync(vmId, ct);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "VM {VmId}: QMP GetPrimaryDriveNode failed", vmId);
            return null;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // State persistence
    // ═══════════════════════════════════════════════════════════════════════

    private async Task<LazysyncState> LoadStateAsync(string vmId)
    {
        if (_states.TryGetValue(vmId, out var cached)) return cached;

        var path = StatePath(vmId);
        if (File.Exists(path))
        {
            try
            {
                var json = await File.ReadAllTextAsync(path);
                var state = JsonSerializer.Deserialize<LazysyncState>(json, JsonOptions.Wire);
                if (state != null)
                {
                    _states[vmId] = state;
                    return state;
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "VM {VmId}: corrupt lazysync state — starting fresh", vmId);
            }
        }

        var fresh = new LazysyncState();
        _states[vmId] = fresh;
        return fresh;
    }

    private async Task SaveStateAsync(string vmId, LazysyncState state)
    {
        var path = StatePath(vmId);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllTextAsync(path,
            JsonSerializer.Serialize(state, JsonOptions.Wire));
    }

    private string StatePath(string vmId) =>
        Path.Combine(_vmOptions.VmStoragePath, vmId, "lazysync.json");
}

// ════════════════════════════════════════════════════════════════════════════
// State model
// ════════════════════════════════════════════════════════════════════════════

public class LazysyncState
{
    /// <summary>Monotonically increasing version. Matches ManifestRecord.Version on orchestrator.</summary>
    public int Version { get; set; } = 0;

    /// <summary>Total overlay bytes in the last sync (for billing estimate).</summary>
    public long TotalBytes { get; set; }

    /// <summary>Chunk map: virtual disk offset (bytes) → CIDv1 string.</summary>
    public Dictionary<long, string> Chunks { get; set; } = new();

    /// <summary>When the last successful sync completed.</summary>
    public DateTime LastSyncAt { get; set; } = DateTime.MinValue;

    /// <summary>
    /// True once a persistent dirty bitmap named "lazysync" has been added to
    /// the VM's primary qcow2 via QMP. When true, SyncVmAsync uses incremental
    /// drive-backup instead of full qemu-img map + convert, exporting only
    /// clusters written since the last successful cycle.
    ///
    /// Reset to false on any QMP failure so the next cycle falls back to
    /// the full export path, which recreates the bitmap cleanly.
    /// Also reset when lazysync.json is deleted (reseed command).
    /// </summary>
    public bool BitmapCreated { get; set; } = false;
}