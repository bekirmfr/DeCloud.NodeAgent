using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Background service that continuously replicates VM overlay disks to the
/// local BlockStore VM via the lazysync protocol.
///
/// Cycle (every 5 minutes, 3 minute startup delay):
///   For each Running tenant VM with ReplicationFactor > 0:
///   1a. Map the overlay extents via:
///           qemu-img map --force-share --output=json disk.qcow2
///       Collect all 1 MB chunk offsets where depth=0 (owned by the overlay,
///       not the backing image) and zero=false. This is the replication whitelist.
///   1b. Export the full merged raw file via:
///           qemu-img convert --force-share -f qcow2 -O raw -S 65536 disk.qcow2 tmp.raw
///       Used only to read actual byte data for CID computation.
///       NOTE: -S skips zero regions in the merged output (free disk space).
///             It does NOT filter backing-file data — that is handled by step 1a.
///   2. Scan the raw file in 1 MB chunks. Skip chunks not in the overlay whitelist
///      (backing-file data) and skip all-zero chunks (sparse free space).
///   3. Compute CIDv1 (raw codec, sha2-256 multihash, base32lower) for each chunk.
///   4. Compare against stored manifest: push only genuinely changed chunks.
///   5. POST each new block to the local BlockStore VM HTTP API (POST /blocks).
///   6. Update the in-memory + on-disk manifest state.
///   7. POST updated manifest to the orchestrator.
///   8. Delete the temp raw file.
///
/// Why overlay-only matters:
///   qemu-img convert merges the full backing chain (overlay + base OS image).
///   The base image (~2 GB, non-zero EXT4 data) would be replicated every cycle
///   without the depth=0 whitelist from step 1a. With the whitelist, only chunks
///   that live in disk.qcow2 itself are replicated — typically 100–500 MB on first
///   boot, then small deltas each cycle. Base image and cloud-init are excluded
///   because they are reconstructible artifacts (image registry + VM labels).
///
/// Dirty bitmap optimization (Phase D+1):
///   IQmpClient is injected but not used in Phase D. In Phase D+1, replace steps
///   1a+1b with block-dirty-bitmap-add + drive-backup sync=incremental to export
///   only actually-written clusters, eliminating the need for the overlay map step.
///
/// State persisted per VM: {VmStoragePath}/{vmId}/lazysync.json
/// </summary>
public class LazysyncDaemon : BackgroundService
{
    private static readonly TimeSpan CycleInterval = TimeSpan.FromMinutes(5);
    private static readonly TimeSpan StartupDelay = TimeSpan.FromMinutes(3);
    private const int BlockSizeBytes = 1024 * 1024; // 1 MB
    private const int BlockSizeKb = 1024;
    private const long SparseScanThreshold = 65536; // -S value for qemu-img convert

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
    private readonly HttpClient _blockstoreClient;
    private readonly ILogger<LazysyncDaemon> _logger;

    // Per-VM in-memory state. Populated from lazysync.json on first access.
    private readonly Dictionary<string, LazysyncState> _states = new();

    public LazysyncDaemon(
        LibvirtVmManager vmManager,
        ICommandExecutor executor,
        IOrchestratorClient orchestratorClient,
        IOptions<LibvirtVmManagerOptions> vmOptions,
        HttpClient blockstoreClient,
        ILogger<LazysyncDaemon> logger)
    {
        _vmManager = vmManager;
        _executor = executor;
        _orchestratorClient = orchestratorClient;
        _vmOptions = vmOptions.Value;
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
                        v.Spec.VmType == VmType.General)
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
        var tmpPath = Path.Combine(Path.GetTempPath(),
            $"lazysync-{vm.VmId[..8]}-{DateTimeOffset.UtcNow.ToUnixTimeSeconds()}.raw");

        try
        {
            // Step 1a: Build the overlay whitelist — depth=0 chunk offsets only.
            // Must run before the raw export so we know which chunks belong to the
            // overlay vs. the backing base image.
            var overlayOffsets = await GetOverlayChunkOffsetsAsync(diskPath, ct);

            if (overlayOffsets.Count == 0)
            {
                _logger.LogDebug("VM {VmId}: overlay has no allocated chunks — skipping", vm.VmId);
                return;
            }

            // Step 1b: Export full merged raw (overlay + backing chain merged).
            // We need this to read actual byte data for CID computation.
            // Backing-file chunks are filtered out in step 2 via overlayOffsets.
            await ExportOverlayAsync(diskPath, tmpPath, ct);

            // Step 2-4: Scan raw, skip non-overlay chunks, compute CIDs, diff against manifest.
            var (changedChunks, totalBytes) = await ScanChunksAsync(tmpPath, overlayOffsets, state, ct);

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
                ct);

            // Step 8: Persist state
            await SaveStateAsync(vm.VmId, state);

            // Step 9: Notify local blockstore VM so its /manifests endpoint
            // reflects the current manifest — populates the dashboard Resources table.
            // Fire-and-forget: dashboard display is non-critical, never block the cycle.
            _ = NotifyBlockstoreManifestAsync(
                blockstoreAddr, vm.VmId, rootCid, state, changedChunks, totalBytes, ct);

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
                chunkCids = changedChunks.Select(c => c.Cid).Take(50).ToList()
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
    // Overlay extent map — depth=0 whitelist
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Returns the set of 1 MB chunk offsets that belong to the overlay itself
    /// (depth=0 in qemu-img map output). Chunks at depth≥1 come from the backing
    /// file (the base OS image) and must NOT be replicated — they are a shared,
    /// downloadable artifact reconstructible from the image registry.
    ///
    /// qemu-img map JSON fields used:
    ///   start  — byte offset of the extent in the virtual disk address space
    ///   length — byte count of the extent
    ///   depth  — 0 = in the overlay (disk.qcow2), 1+ = in a backing file
    ///   zero   — true = extent is all-zero (unallocated / sparse free space)
    /// </summary>
    private async Task<HashSet<long>> GetOverlayChunkOffsetsAsync(string diskPath, CancellationToken ct)
    {
        var result = await _executor.ExecuteAsync(
            "qemu-img",
            $"map --force-share --output=json \"{diskPath}\"",
            TimeSpan.FromMinutes(5),
            ct);

        if (!result.Success)
            throw new Exception($"qemu-img map failed: {result.StandardError}");

        var offsets = new HashSet<long>();
        var extents = JsonSerializer.Deserialize<JsonElement[]>(result.StandardOutput) ?? [];

        foreach (var extent in extents)
        {
            // depth=0: extent lives in the overlay file — replicate it.
            // depth≥1: extent comes from the backing chain — skip it.
            if (extent.GetProperty("depth").GetInt32() != 0) continue;

            // zero=true: unallocated / all-zero region — nothing useful to replicate.
            if (extent.GetProperty("zero").GetBoolean()) continue;

            var start = extent.GetProperty("start").GetInt64();
            var length = extent.GetProperty("length").GetInt64();

            // Mark every 1 MB chunk that overlaps this extent.
            // An extent may span a partial leading chunk and a partial trailing chunk,
            // both of which still need to be included.
            var firstChunk = (start / BlockSizeBytes) * BlockSizeBytes;
            var lastChunk = ((start + length - 1) / BlockSizeBytes) * BlockSizeBytes;
            for (var off = firstChunk; off <= lastChunk; off += BlockSizeBytes)
                offsets.Add(off);
        }

        _logger.LogDebug(
            "VM {DiskFile}: overlay map — {Count} chunk offset(s) at depth=0",
            Path.GetFileName(diskPath), offsets.Count);

        return offsets;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // qemu-img export
    // ═══════════════════════════════════════════════════════════════════════

    private async Task ExportOverlayAsync(string diskPath, string tmpPath, CancellationToken ct)
    {
        // --force-share: read QEMU-locked qcow2 (POSIX advisory locks, safe for reads).
        // -S 65536: skip zero/unallocated regions in the merged output (free disk space).
        //
        // WARNING: qemu-img convert merges the full backing chain — the output raw file
        //          contains both overlay data AND base OS image data. The -S flag only
        //          skips truly zero regions; it does NOT filter backing-file extents.
        //          Use GetOverlayChunkOffsetsAsync() + the overlayOffsets whitelist in
        //          ScanChunksAsync() to exclude backing-file chunks from replication.
        var result = await _executor.ExecuteAsync(
            "qemu-img",
            $"convert --force-share -f qcow2 -O raw -S {SparseScanThreshold} \"{diskPath}\" \"{tmpPath}\"",
            TimeSpan.FromMinutes(10),
            ct);

        if (!result.Success)
            throw new Exception($"qemu-img convert failed: {result.StandardError}");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Chunk scan + CID computation
    // ═══════════════════════════════════════════════════════════════════════

    private async Task<(List<(long Offset, string Cid)> Changed, long TotalBytes)>
        ScanChunksAsync(string rawPath, HashSet<long> overlayOffsets, LazysyncState state, CancellationToken ct)
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

            // Skip backing-file chunks (depth≥1 extents — base OS image data).
            // qemu-img convert merges the backing chain into the raw output, so
            // without this guard we would replicate the entire OS image every cycle.
            if (!overlayOffsets.Contains(offset))
            {
                offset += read;
                continue;
            }

            // Skip all-zero chunks — unallocated / sparse free space within the overlay.
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
        var blockstoreVm = vms.FirstOrDefault(v => v.Spec.VmType == VmType.BlockStore);
        if (blockstoreVm == null || string.IsNullOrEmpty(blockstoreVm.Spec.IpAddress))
            return null;

        // BlockStore VM API port: 5090 (BlockStoreVmSpec.ApiPort)
        return $"http://{blockstoreVm.Spec.IpAddress}:5090";
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
                var state = JsonSerializer.Deserialize<LazysyncState>(json);
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
            JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = false }));
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
}