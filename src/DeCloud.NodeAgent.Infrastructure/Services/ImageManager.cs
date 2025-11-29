using System.Security.Cryptography;
using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Infrastructure.Services;

public class ImageManagerOptions
{
    public string CachePath { get; set; } = "/var/lib/decloud/images";
    public string VmStoragePath { get; set; } = "/var/lib/decloud/vms";
    public TimeSpan DownloadTimeout { get; set; } = TimeSpan.FromMinutes(30);
}

public class ImageManager : IImageManager
{
    private readonly ICommandExecutor _executor;
    private readonly ICloudInitCleaner _cloudInitCleaner;
    private readonly HttpClient _httpClient;
    private readonly ILogger<ImageManager> _logger;
    private readonly ImageManagerOptions _options;
    private readonly SemaphoreSlim _downloadLock = new(3); // Max 3 concurrent downloads

    public ImageManager(
        ICommandExecutor executor,
        ICloudInitCleaner cloudInitCleaner,
        HttpClient httpClient,
        IOptions<ImageManagerOptions> options,
        ILogger<ImageManager> logger)
    {
        _executor = executor;
        _cloudInitCleaner = cloudInitCleaner;
        _httpClient = httpClient;
        _logger = logger;
        _options = options.Value;
        
        Directory.CreateDirectory(_options.CachePath);
        Directory.CreateDirectory(_options.VmStoragePath);
    }

    public async Task<string> EnsureImageAvailableAsync(string imageUrl, string expectedHash, CancellationToken ct = default)
    {
        var fileName = GetCacheFileName(imageUrl, expectedHash);
        var localPath = Path.Combine(_options.CachePath, fileName);

        // Check if already cached
        if (File.Exists(localPath))
        {
            _logger.LogDebug("Image found in cache: {Path}", localPath);

            if (await VerifyImageAsync(localPath, expectedHash, ct))
            {
                // Image is cached and verified - cloud-init should already be cleaned
                return localPath;
            }

            _logger.LogWarning("Cached image hash mismatch, re-downloading");
            File.Delete(localPath);
        }

        // Download
        await _downloadLock.WaitAsync(ct);
        try
        {
            // Double-check after acquiring lock
            if (File.Exists(localPath) && await VerifyImageAsync(localPath, expectedHash, ct))
            {
                return localPath;
            }

            await DownloadImageAsync(imageUrl, localPath, ct);

            // Verify downloaded image
            if (!await VerifyImageAsync(localPath, expectedHash, ct))
            {
                File.Delete(localPath);
                throw new Exception($"Downloaded image hash does not match expected: {expectedHash}");
            }

            // ========================================
            // Clean cloud-init state from base image
            // ========================================
            var cleanResult = await _cloudInitCleaner.CleanAsync(localPath, ct);
            if (cleanResult.Success)
            {
                _logger.LogInformation(
                    "Base image cloud-init cleaned: {Method} in {Duration}ms",
                    cleanResult.MethodUsed,
                    cleanResult.Duration.TotalMilliseconds);
            }
            else
            {
                _logger.LogWarning(
                    "Could not clean cloud-init from base image: {Message}. " +
                    "VMs may boot with stale configuration.",
                    cleanResult.Message);
            }
            // =========================================================

            return localPath;
        }
        finally
        {
            _downloadLock.Release();
        }
    }
}

    public async Task<bool> VerifyImageAsync(string imagePath, string expectedHash, CancellationToken ct = default)
    {
        if (string.IsNullOrEmpty(expectedHash))
        {
            _logger.LogWarning("No hash provided for image verification, skipping");
            return true;
        }

        _logger.LogDebug("Verifying hash for {Path}", imagePath);

        using var sha256 = SHA256.Create();
        await using var stream = File.OpenRead(imagePath);
        
        var hashBytes = await sha256.ComputeHashAsync(stream, ct);
        var actualHash = Convert.ToHexString(hashBytes).ToLowerInvariant();

        var expected = expectedHash.ToLowerInvariant().Replace("sha256:", "");
        var match = actualHash == expected;

        if (!match)
        {
            _logger.LogWarning("Hash mismatch: expected {Expected}, got {Actual}", expected, actualHash);
        }

        return match;
    }

    public async Task<string> CreateOverlayDiskAsync(string baseImagePath, string vmId, long sizeBytes, CancellationToken ct = default)
    {
        var vmDir = Path.Combine(_options.VmStoragePath, vmId);
        Directory.CreateDirectory(vmDir);

        var overlayPath = Path.Combine(vmDir, "disk.qcow2");
        var sizeGb = Math.Max(1, sizeBytes / 1024 / 1024 / 1024);

        _logger.LogInformation("Creating overlay disk at {Path}, size {Size}GB, backing {Base}",
            overlayPath, sizeGb, baseImagePath);

        // Create qcow2 overlay with backing file
        var result = await _executor.ExecuteAsync("qemu-img",
            $"create -f qcow2 -F qcow2 -b {baseImagePath} {overlayPath} {sizeGb}G",
            TimeSpan.FromMinutes(5),
            ct);

        if (!result.Success)
        {
            throw new Exception($"Failed to create overlay disk: {result.StandardError}");
        }

        return overlayPath;
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

        foreach (var file in Directory.GetFiles(_options.CachePath, "*.qcow2"))
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

        return Task.FromResult(images);
    }

    public async Task PruneUnusedImagesAsync(TimeSpan maxAge, CancellationToken ct = default)
    {
        var cutoff = DateTime.UtcNow - maxAge;
        var pruned = 0;
        long freedBytes = 0;

        foreach (var file in Directory.GetFiles(_options.CachePath, "*.qcow2"))
        {
            var info = new FileInfo(file);
            if (info.LastAccessTimeUtc < cutoff)
            {
                // Check if any VM is using this as backing file
                var isInUse = await IsImageInUseAsync(file, ct);
                if (!isInUse)
                {
                    freedBytes += info.Length;
                    File.Delete(file);
                    pruned++;
                    _logger.LogDebug("Pruned unused image: {Path}", file);
                }
            }
        }

        if (pruned > 0)
        {
            _logger.LogInformation("Pruned {Count} unused images, freed {Bytes}MB",
                pruned, freedBytes / 1024 / 1024);
        }
    }

    private async Task DownloadImageAsync(string url, string destPath, CancellationToken ct)
    {
        _logger.LogInformation("Downloading image from {Url} to {Path}", url, destPath);

        var tempPath = destPath + ".downloading";

        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(_options.DownloadTimeout);

            using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cts.Token);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? 0;
            var downloadedBytes = 0L;
            var lastLogTime = DateTime.UtcNow;

            await using var contentStream = await response.Content.ReadAsStreamAsync(cts.Token);
            await using var fileStream = File.Create(tempPath);

            var buffer = new byte[81920]; // 80KB buffer
            int bytesRead;

            while ((bytesRead = await contentStream.ReadAsync(buffer, cts.Token)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cts.Token);
                downloadedBytes += bytesRead;

                // Log progress every 10 seconds
                if ((DateTime.UtcNow - lastLogTime).TotalSeconds >= 10)
                {
                    var percent = totalBytes > 0 ? (double)downloadedBytes / totalBytes * 100 : 0;
                    _logger.LogInformation("Download progress: {Downloaded}MB / {Total}MB ({Percent:F1}%)",
                        downloadedBytes / 1024 / 1024,
                        totalBytes / 1024 / 1024,
                        percent);
                    lastLogTime = DateTime.UtcNow;
                }
            }

            // Move to final location
            File.Move(tempPath, destPath, overwrite: true);
            
            _logger.LogInformation("Download complete: {Path} ({Size}MB)",
                destPath, downloadedBytes / 1024 / 1024);
        }
        catch
        {
            // Clean up temp file on failure
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
            throw;
        }
    }

    private async Task<bool> IsImageInUseAsync(string imagePath, CancellationToken ct)
    {
        // Check if any qcow2 in VM storage uses this as backing file
        foreach (var vmDir in Directory.GetDirectories(_options.VmStoragePath))
        {
            var diskPath = Path.Combine(vmDir, "disk.qcow2");
            if (!File.Exists(diskPath)) continue;

            var result = await _executor.ExecuteAsync("qemu-img", $"info {diskPath}", ct);
            if (result.Success && result.StandardOutput.Contains(imagePath))
            {
                return true;
            }
        }

        return false;
    }

    private static string GetCacheFileName(string url, string hash)
    {
        // Create a stable filename from URL and hash
        var urlHash = Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(url)))[..16];
        var hashPrefix = string.IsNullOrEmpty(hash) ? "nohash" : hash.Replace("sha256:", "")[..16];
        
        // Extract original extension
        var uri = new Uri(url);
        var ext = Path.GetExtension(uri.LocalPath);
        if (string.IsNullOrEmpty(ext)) ext = ".qcow2";

        return $"{urlHash}_{hashPrefix}{ext}";
    }
}
