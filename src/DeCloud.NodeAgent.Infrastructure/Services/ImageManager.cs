using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using System.Security.Cryptography;

namespace DeCloud.NodeAgent.Infrastructure.Services;

public class ImageManagerOptions
{
    public string CachePath { get; set; } = "/var/lib/decloud/images";
    public string VmStoragePath { get; set; } = "/var/lib/decloud/vms";
    public TimeSpan DownloadTimeout { get; set; } = TimeSpan.FromMinutes(30);
}

/// <summary>
/// Manages base VM images: download, content-addressed cache, verification.
///
/// CACHE LAYOUT
/// Files are stored at <c>{CachePath}/{sha256}.img</c> — the filename IS the
/// hash of the file's bytes. Two nodes that downloaded byte-identical upstream
/// bytes have byte-identical cached files at the same path. The cache key is
/// the content, never the URL.
///
/// VERIFICATION
/// When the caller provides an expected hash, this manager strictly enforces
/// it: cached file's hash must match, downloaded bytes must hash to the
/// expected value, otherwise the deploy fails. When the expected hash is
/// empty (permissive mode during initial rollout — see BASE_IMAGE_DESIGN.md
/// §4.6), the manager hashes the downloaded bytes on the fly and returns
/// the computed hash so the caller can record it back into VmSpec.
///
/// CLEANING IS NOT DONE HERE
/// Cloud-init state cleaning used to be done in this class against the
/// cached file. That broke byte-level content identity (different nodes'
/// cleaning produced different bytes for the same upstream image). Cleaning
/// has moved to <see cref="DeCloud.NodeAgent.Infrastructure.Libvirt.LibvirtVmManager.CreateVmAsync"/>
/// where it operates on the per-VM overlay disk, leaving the base
/// content-addressable. See BASE_IMAGE_DESIGN.md §4.4.
///
/// LEGACY CACHE FILES
/// Files matching the old <c>{url_hash}_{nohash|hashprefix}.img</c> naming
/// are migrated lazily: on cache miss, this manager scans the cache directory
/// for any legacy file whose bytes hash to the expected (or computed) value
/// and renames it to the content-addressed name. No flag day; legacy state
/// is repaired in place on first encounter.
/// </summary>
public class ImageManager : IImageManager
{
    private readonly ICommandExecutor _executor;
    private readonly HttpClient _httpClient;
    private readonly ILogger<ImageManager> _logger;
    private readonly ImageManagerOptions _options;
    private readonly SemaphoreSlim _downloadLock = new(3); // Max 3 concurrent downloads
    private readonly ConcurrentDictionary<string, ImageDownloadProgress> _activeDownloads = new();
    // Maps imageUrl → vmId so the download loop can update by vmId
    private readonly ConcurrentDictionary<string, string> _downloadVmMap = new();

    public IReadOnlyDictionary<string, ImageDownloadProgress> ActiveDownloads => _activeDownloads;

    public void TrackDownload(string vmId, string imageUrl)
    {
        _downloadVmMap[imageUrl] = vmId;
        _activeDownloads[vmId] = new ImageDownloadProgress(vmId, imageUrl, 0, 0, 0);
    }

    public ImageManager(
        ICommandExecutor executor,
        HttpClient httpClient,
        IOptions<ImageManagerOptions> options,
        ILogger<ImageManager> logger)
    {
        _executor = executor;
        _httpClient = httpClient;
        _logger = logger;
        _options = options.Value;

        Directory.CreateDirectory(_options.CachePath);
        Directory.CreateDirectory(_options.VmStoragePath);
    }

    public async Task<EnsureImageResult> EnsureImageAvailableAsync(
        string imageUrl, string expectedHash, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(imageUrl))
            throw new ArgumentException("imageUrl is required", nameof(imageUrl));

        var normalisedExpected = NormaliseHash(expectedHash);

        // ── Strict-mode fast path: cache hit by content-addressed name ────
        if (!string.IsNullOrEmpty(normalisedExpected))
        {
            var contentAddressedPath = CachePathForHash(normalisedExpected);
            if (File.Exists(contentAddressedPath))
            {
                // Cache filename IS the hash; trust the filename as the
                // primary check, but periodically the operator may want full
                // re-verification. Re-hashing every cache hit would cost ~30s
                // for a 2 GiB image — too expensive. Filename match is
                // sufficient: the file got into the cache only after passing
                // verification in a previous call.
                _logger.LogDebug(
                    "Image cache hit (content-addressed): {Path}", contentAddressedPath);
                return new EnsureImageResult(contentAddressedPath, normalisedExpected);
            }

            // Legacy cache: scan for a *_nohash.* or *_hashprefix.* file that
            // hashes to the expected value. If found, rename in place.
            var rescued = await TryRescueLegacyCacheEntryAsync(
                normalisedExpected, contentAddressedPath, ct);
            if (rescued)
            {
                _logger.LogInformation(
                    "Image cache: legacy file matched expected hash and was migrated to {Path}",
                    contentAddressedPath);
                return new EnsureImageResult(contentAddressedPath, normalisedExpected);
            }
        }

        // ── Download path (with serialisation) ────────────────────────────
        await _downloadLock.WaitAsync(ct);
        try
        {
            // Re-check after acquiring the lock — another caller may have
            // completed the download.
            if (!string.IsNullOrEmpty(normalisedExpected))
            {
                var contentAddressedPath = CachePathForHash(normalisedExpected);
                if (File.Exists(contentAddressedPath))
                    return new EnsureImageResult(contentAddressedPath, normalisedExpected);
            }

            var (finalPath, computedHash) = await DownloadAndHashAsync(
                imageUrl, normalisedExpected, ct);

            return new EnsureImageResult(finalPath, computedHash);
        }
        finally
        {
            _downloadLock.Release();
        }
    }

    /// <summary>
    /// Public verification helper retained for diagnostics. Hashes the file
    /// at <paramref name="imagePath"/> and compares against the expected
    /// value. Returns true on match, false on mismatch. When the expected
    /// hash is empty, this method returns true unconditionally — there is
    /// nothing to verify against. Callers wanting strict verification must
    /// pass a non-empty expected hash.
    /// </summary>
    public async Task<bool> VerifyImageAsync(
        string imagePath, string expectedHash, CancellationToken ct = default)
    {
        var normalisedExpected = NormaliseHash(expectedHash);
        if (string.IsNullOrEmpty(normalisedExpected))
            return true;

        if (!File.Exists(imagePath))
        {
            _logger.LogWarning(
                "VerifyImageAsync: file does not exist at {Path}", imagePath);
            return false;
        }

        var actual = await ComputeFileSha256Async(imagePath, ct);
        var match = string.Equals(actual, normalisedExpected, StringComparison.Ordinal);
        if (!match)
        {
            _logger.LogWarning(
                "Image hash mismatch at {Path}: expected {Expected}, got {Actual}",
                imagePath, normalisedExpected[..16], actual[..16]);
        }
        return match;
    }

    public async Task<string> CreateOverlayDiskAsync(
        string baseImagePath, string vmId, long sizeBytes, CancellationToken ct = default)
    {
        var vmDir = Path.Combine(_options.VmStoragePath, vmId);
        Directory.CreateDirectory(vmDir);
        // Harden: restrict to root only — VM disk and state files are tenant data.
        if (OperatingSystem.IsLinux())
        {
            await _executor.ExecuteAsync("chown", $"root:kvm \"{vmDir}\"", ct);
            File.SetUnixFileMode(vmDir,
                UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute |
                UnixFileMode.GroupRead | UnixFileMode.GroupExecute); // 0750
        }

        var overlayPath = Path.Combine(vmDir, "disk.qcow2");
        var requestedSizeGb = Math.Max(1, sizeBytes / 1024 / 1024 / 1024);

        // Detect backing file format and virtual size via a single qemu-img info call.
        var (backingFormat, backingVirtualSize) = await GetBackingImageInfoAsync(baseImagePath, ct);

        // The overlay must be at least as large as the backing image's virtual size,
        // otherwise the partition table references blocks beyond the overlay boundary
        // and the kernel drops to initramfs unable to mount root.
        var backingSizeGb = (long)Math.Ceiling((double)backingVirtualSize / 1024 / 1024 / 1024);
        var sizeGb = Math.Max(requestedSizeGb, backingSizeGb);

        if (sizeGb > requestedSizeGb)
        {
            _logger.LogWarning(
                "Overlay size {RequestedGB}GB is smaller than backing image virtual size {BackingGB}GB — " +
                "increasing to {ActualGB}GB to prevent boot failure",
                requestedSizeGb, backingSizeGb, sizeGb);
        }

        _logger.LogInformation("Creating overlay disk at {Path}, size {Size}GB, backing {Base} (format: {Format})",
            overlayPath, sizeGb, baseImagePath, backingFormat);

        // Create qcow2 overlay with backing file
        var result = await _executor.ExecuteAsync("qemu-img",
            $"create -f qcow2 -F {backingFormat} -b {baseImagePath} {overlayPath} {sizeGb}G",
            TimeSpan.FromMinutes(5),
            ct);

        if (!result.Success)
        {
            throw new Exception($"Failed to create overlay disk: {result.StandardError}");
        }

        return overlayPath;
    }

    /// <summary>
    /// Detect the format and virtual size of a disk image using qemu-img info.
    /// Returns (format, virtualSizeBytes). Falls back to ("qcow2", 0) if detection fails.
    /// </summary>
    private async Task<(string Format, long VirtualSizeBytes)> GetBackingImageInfoAsync(string imagePath, CancellationToken ct)
    {
        var format = "qcow2";
        long virtualSize = 0;

        try
        {
            // --force-share: read through POSIX advisory locks held by concurrent
            // processes (running QEMU instances). Without this flag, qemu-img info
            // fails with "Failed to get shared write lock" when the base image is
            // open by a VM.
            var result = await _executor.ExecuteAsync("qemu-img",
                $"info --force-share --output=json \"{imagePath}\"",
                TimeSpan.FromSeconds(30),
                ct);

            if (result.Success)
            {
                using var doc = System.Text.Json.JsonDocument.Parse(result.StandardOutput);
                var root = doc.RootElement;
                if (root.TryGetProperty("format", out var fmtEl))
                    format = fmtEl.GetString() ?? "qcow2";
                if (root.TryGetProperty("virtual-size", out var vsEl))
                    virtualSize = vsEl.GetInt64();
            }
            else
            {
                _logger.LogWarning(
                    "qemu-img info failed for {Path}: {Error}",
                    imagePath, result.StandardError);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to parse qemu-img info output for {Path}", imagePath);
        }

        return (format, virtualSize);
    }

    public Task DeleteDiskAsync(string diskPath, CancellationToken ct = default)
    {
        if (File.Exists(diskPath))
        {
            File.Delete(diskPath);
            _logger.LogDebug("Deleted disk: {Path}", diskPath);
        }

        // Also delete parent directory if empty
        var dir = Path.GetDirectoryName(diskPath);
        if (dir != null && Directory.Exists(dir) && !Directory.EnumerateFileSystemEntries(dir).Any())
        {
            Directory.Delete(dir);
        }

        return Task.CompletedTask;
    }

    public Task<List<CachedImage>> GetCachedImagesAsync(CancellationToken ct = default)
    {
        var images = new List<CachedImage>();

        // Match both new content-addressed (.img) and legacy (.qcow2) names.
        foreach (var pattern in new[] { "*.img", "*.qcow2" })
        {
            foreach (var file in Directory.GetFiles(_options.CachePath, pattern))
            {
                var info = new FileInfo(file);
                images.Add(new CachedImage
                {
                    LocalPath = file,
                    SizeBytes = info.Length,
                    DownloadedAt = info.CreationTimeUtc,
                    LastUsedAt = info.LastAccessTimeUtc
                });
            }
        }

        return Task.FromResult(images);
    }

    public async Task PruneUnusedImagesAsync(TimeSpan maxAge, CancellationToken ct = default)
    {
        var cutoff = DateTime.UtcNow - maxAge;
        var pruned = 0;
        long freedBytes = 0;

        // Prune both content-addressed and legacy files.
        foreach (var pattern in new[] { "*.img", "*.qcow2" })
        {
            foreach (var file in Directory.GetFiles(_options.CachePath, pattern))
            {
                var info = new FileInfo(file);
                if (info.LastAccessTimeUtc >= cutoff) continue;

                var isInUse = await IsImageInUseAsync(file, ct);
                if (isInUse) continue;

                freedBytes += info.Length;
                File.Delete(file);
                pruned++;
                _logger.LogDebug("Pruned unused image: {Path}", file);
            }
        }

        if (pruned > 0)
        {
            _logger.LogInformation("Pruned {Count} unused images, freed {Bytes}MB",
                pruned, freedBytes / 1024 / 1024);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Download + content-addressed storage
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Download the image at <paramref name="imageUrl"/>, hashing on the fly,
    /// and atomically rename the temp file to <c>{cache}/{sha256}.img</c>.
    ///
    /// Strict mode (non-empty <paramref name="expectedHash"/>): if the
    /// computed hash differs from expected, the temp file is deleted and an
    /// exception is thrown. Permissive mode (empty expected): the computed
    /// hash becomes the cache key; no comparison.
    ///
    /// Same pattern as <c>ArtifactCacheService.DownloadAndVerifyAsync</c> —
    /// kept consistent on purpose.
    /// </summary>
    private async Task<(string LocalPath, string Sha256)> DownloadAndHashAsync(
        string imageUrl, string expectedHash, CancellationToken ct)
    {
        // Temp file lives next to the eventual destination so the final
        // rename is intra-directory (atomic on POSIX). Random suffix prevents
        // collisions when two callers race on different URLs.
        var tempPath = Path.Combine(
            _options.CachePath,
            $".downloading-{Guid.NewGuid():N}.tmp");

        _logger.LogInformation(
            "Downloading image from {Url} (mode={Mode})",
            imageUrl,
            string.IsNullOrEmpty(expectedHash) ? "permissive" : "strict");

        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(_options.DownloadTimeout);

            using var response = await _httpClient.GetAsync(
                imageUrl, HttpCompletionOption.ResponseHeadersRead, cts.Token);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? 0;

            string computedHash;
            await using (var contentStream = await response.Content.ReadAsStreamAsync(cts.Token))
            await using (var fileStream = new FileStream(
                tempPath, FileMode.Create, FileAccess.Write,
                FileShare.None, bufferSize: 81920, useAsync: true))
            using (var sha = SHA256.Create())
            using (var cryptoStream = new CryptoStream(
                contentStream, sha, CryptoStreamMode.Read))
            {
                var buffer = new byte[81920];
                var downloadedBytes = 0L;
                var lastLogTime = DateTime.UtcNow;
                int bytesRead;

                while ((bytesRead = await cryptoStream.ReadAsync(
                    buffer.AsMemory(0, buffer.Length), cts.Token)) > 0)
                {
                    await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cts.Token);
                    downloadedBytes += bytesRead;

                    // Update progress tracking (every iteration for responsive UI)
                    if (_downloadVmMap.TryGetValue(imageUrl, out var trackingVmId))
                    {
                        var pct = totalBytes > 0
                            ? (int)(downloadedBytes * 100 / totalBytes)
                            : 0;
                        _activeDownloads[trackingVmId] = new ImageDownloadProgress(
                            trackingVmId, imageUrl, downloadedBytes, totalBytes, pct);
                    }

                    // Log progress every 10 seconds
                    if ((DateTime.UtcNow - lastLogTime).TotalSeconds >= 10)
                    {
                        var percent = totalBytes > 0
                            ? (double)downloadedBytes / totalBytes * 100
                            : 0;
                        _logger.LogInformation(
                            "Download progress: {Downloaded}MB / {Total}MB ({Percent:F1}%)",
                            downloadedBytes / 1024 / 1024,
                            totalBytes / 1024 / 1024,
                            percent);
                        lastLogTime = DateTime.UtcNow;
                    }
                }

                await fileStream.FlushAsync(cts.Token);
                computedHash = Convert.ToHexString(sha.Hash!).ToLowerInvariant();
            }

            // Strict mode: enforce hash match.
            if (!string.IsNullOrEmpty(expectedHash) &&
                !string.Equals(computedHash, expectedHash, StringComparison.Ordinal))
            {
                _logger.LogError(
                    "Downloaded image hash mismatch! URL={Url}, expected={Expected}, got={Actual}. " +
                    "The artifact at the URL does not match the orchestrator's recorded hash. " +
                    "The URL may have drifted to a newer build, or the bytes may have been tampered with.",
                    imageUrl, expectedHash[..16], computedHash[..16]);
                TryDelete(tempPath);
                throw new Exception(
                    $"Downloaded image hash mismatch for {imageUrl}: " +
                    $"expected {expectedHash[..16]}, got {computedHash[..16]}");
            }

            // Atomic rename to the content-addressed final path.
            var finalPath = CachePathForHash(computedHash);
            if (File.Exists(finalPath))
            {
                // Another concurrent caller materialised the same hash
                // between our cache miss and this rename. Drop our temp; both
                // files have identical bytes by construction.
                TryDelete(tempPath);
            }
            else
            {
                File.Move(tempPath, finalPath, overwrite: false);
            }

            _logger.LogInformation(
                "Download complete: {Path} (sha256={Sha256})",
                finalPath, computedHash[..16]);

            // Clean up progress tracking
            if (_downloadVmMap.TryRemove(imageUrl, out var completedVmId))
                _activeDownloads.TryRemove(completedVmId, out _);

            return (finalPath, computedHash);
        }
        catch
        {
            TryDelete(tempPath);
            if (_downloadVmMap.TryRemove(imageUrl, out var failedVmId))
                _activeDownloads.TryRemove(failedVmId, out _);
            throw;
        }
    }

    /// <summary>
    /// Scan the cache directory for files matching the old naming scheme
    /// (<c>*_nohash.*</c> or <c>*_*.*</c>) and, if any of them hash to the
    /// expected value, rename to the content-addressed name. Returns true
    /// if a legacy file was successfully migrated.
    ///
    /// Only invoked when the caller has a non-empty expected hash. In
    /// permissive mode we don't know what to look for.
    /// </summary>
    private async Task<bool> TryRescueLegacyCacheEntryAsync(
        string expectedHash, string contentAddressedPath, CancellationToken ct)
    {
        if (!Directory.Exists(_options.CachePath)) return false;

        // Candidates: anything that isn't already a content-addressed .img.
        // The old naming had {urlHash}_{hashprefix|nohash}.{qcow2|img}.
        var candidates = new List<string>();
        foreach (var pattern in new[] { "*_nohash.*", "*_*.qcow2", "*_*.img" })
        {
            foreach (var path in Directory.GetFiles(_options.CachePath, pattern))
            {
                // Skip files that already look content-addressed
                // (filename is a 64-char hex string + extension).
                var fileName = Path.GetFileNameWithoutExtension(path);
                if (fileName.Length == 64 &&
                    fileName.All(c => (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                {
                    continue;
                }
                candidates.Add(path);
            }
        }

        foreach (var candidate in candidates)
        {
            try
            {
                var actual = await ComputeFileSha256Async(candidate, ct);
                if (!string.Equals(actual, expectedHash, StringComparison.Ordinal))
                {
                    continue;
                }

                // Match — migrate to content-addressed name.
                if (File.Exists(contentAddressedPath))
                {
                    // Concurrent producer beat us. Drop the legacy copy.
                    File.Delete(candidate);
                }
                else
                {
                    File.Move(candidate, contentAddressedPath, overwrite: false);
                }
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex,
                    "Legacy cache rescue: could not hash/migrate {Path} — skipping",
                    candidate);
            }
        }

        return false;
    }

    private async Task<bool> IsImageInUseAsync(string imagePath, CancellationToken ct)
    {
        // Check if any qcow2 in VM storage uses this as backing file
        foreach (var vmDir in Directory.GetDirectories(_options.VmStoragePath))
        {
            var diskPath = Path.Combine(vmDir, "disk.qcow2");
            if (!File.Exists(diskPath)) continue;

            var result = await _executor.ExecuteAsync(
                "qemu-img", $"info --force-share \"{diskPath}\"", ct);
            if (result.Success && result.StandardOutput.Contains(imagePath))
            {
                return true;
            }
        }

        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private string CachePathForHash(string sha256) =>
        Path.Combine(_options.CachePath, $"{sha256}.img");

    private static string NormaliseHash(string? hash)
    {
        if (string.IsNullOrWhiteSpace(hash)) return string.Empty;
        var trimmed = hash.Trim();
        if (trimmed.StartsWith("sha256:", StringComparison.OrdinalIgnoreCase))
            trimmed = trimmed["sha256:".Length..];
        return trimmed.ToLowerInvariant();
    }

    private static async Task<string> ComputeFileSha256Async(string path, CancellationToken ct)
    {
        using var sha = SHA256.Create();
        await using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 81920, useAsync: true);
        var hashBytes = await sha.ComputeHashAsync(stream, ct);
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
        }
        catch
        {
            // Best-effort cleanup. The next download attempt will overwrite.
        }
    }
}
