using System.Security.Cryptography;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

// ============================================================
// Placement: src/DeCloud.NodeAgent.Infrastructure/Services/ArtifactCacheService.cs
// ============================================================

/// <summary>
/// Default implementation of <see cref="IArtifactCacheService"/>.
///
/// Stores artifacts as flat files in <c>{cacheDir}/{sha256}</c>. In-progress
/// downloads land at <c>{cacheDir}/{sha256}.tmp</c> so a crash mid-download
/// never leaves a corrupt completed-looking entry.
///
/// Thread safety: multiple concurrent calls for the same SHA256 are
/// de-duplicated via a per-SHA256 <see cref="SemaphoreSlim"/>. Calls for
/// different SHA256 values proceed in parallel.
///
/// Logging discipline: Info on first download; Debug on cache-hit (hot path,
/// every VM prefetch would be noisy at Info); Warning on SHA256 mismatch,
/// download failure, or verify-and-purge.
/// </summary>
public sealed class ArtifactCacheService : IArtifactCacheService
{
    private readonly string _cacheDir;
    private readonly HttpClient _httpClient;
    private readonly ILogger<ArtifactCacheService> _logger;

    // Per-SHA256 semaphores prevent concurrent downloads of the same artifact.
    // The outer lock protects the dictionary; inner semaphores protect individual downloads.
    private readonly Dictionary<string, SemaphoreSlim> _downloadLocks = new(StringComparer.OrdinalIgnoreCase);
    private readonly SemaphoreSlim _dictLock = new(1, 1);

    // Named HttpClient registered in Program.cs with a 10-minute timeout for large binary downloads.
    private const string HttpClientName = "ArtifactDownload";

    public ArtifactCacheService(
        string cacheDir,
        HttpClient httpClient,
        ILogger<ArtifactCacheService> logger)
    {
        _cacheDir         = cacheDir;
        _httpClient = httpClient;
        _logger           = logger;

        EnsureCacheDirectory();
    }

    // ── IArtifactCacheService ────────────────────────────────────────────

    /// <inheritdoc/>
    public Task<string?> GetLocalPathAsync(string sha256, CancellationToken ct = default)
    {
        ValidateSha256(sha256);
        var path = CachePath(sha256);
        return Task.FromResult(File.Exists(path) ? path : null);
    }

    /// <inheritdoc/>
    public async Task<string> EnsureCachedAsync(
        string sha256,
        string sourceUrl,
        CancellationToken ct = default)
    {
        ValidateSha256(sha256);
        ValidateSourceUrl(sourceUrl);

        var finalPath = CachePath(sha256);

        // Fast path: already present and verified.
        if (File.Exists(finalPath) && await VerifyFileAsync(finalPath, sha256, ct))
        {
            _logger.LogDebug("ArtifactCache: hit {Sha256}", sha256[..12]);
            return finalPath;
        }

        // Acquire per-SHA256 lock to prevent concurrent downloads.
        var sem = await GetOrCreateLockAsync(sha256, ct);
        await sem.WaitAsync(ct);
        try
        {
            // Re-check after acquiring — another thread may have completed the download.
            if (File.Exists(finalPath) && await VerifyFileAsync(finalPath, sha256, ct))
            {
                _logger.LogDebug("ArtifactCache: hit {Sha256} (post-lock)", sha256[..12]);
                return finalPath;
            }

            await DownloadAndVerifyAsync(sha256, sourceUrl, finalPath, ct);
            return finalPath;
        }
        finally
        {
            sem.Release();
        }
    }

    /// <inheritdoc/>
    public async Task PrefetchAsync(
        IReadOnlyList<TemplateArtifact> artifacts,
        string nodeArchitecture,
        CancellationToken ct = default)
    {
        foreach (var artifact in artifacts)
        {
            // Skip arch-specific artifacts for other architectures.
            if (artifact.Architecture is not null &&
                !string.Equals(artifact.Architecture, nodeArchitecture, StringComparison.OrdinalIgnoreCase))
            {
                _logger.LogDebug(
                    "ArtifactCache: skipping {Name} (arch={Arch}, node={NodeArch})",
                    artifact.Name, artifact.Architecture, nodeArchitecture);
                continue;
            }

            try
            {
                await EnsureCachedAsync(artifact.Sha256, artifact.SourceUrl, ct);
                _logger.LogDebug("ArtifactCache: prefetched {Name} ({Sha256})",
                    artifact.Name, artifact.Sha256[..12]);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex)
            {
                // One artifact failing must not abort the rest of the prefetch.
                // The reconciler checks all artifacts are cached before dispatching Create.
                _logger.LogWarning(ex,
                    "ArtifactCache: prefetch failed for {Name} ({Sha256}) — will retry on next template update",
                    artifact.Name, artifact.Sha256[..12]);
            }
        }
    }

    /// <inheritdoc/>
    public async Task<bool> VerifyAsync(string sha256, CancellationToken ct = default)
    {
        ValidateSha256(sha256);
        var path = CachePath(sha256);

        if (!File.Exists(path))
            return false;

        var ok = await VerifyFileAsync(path, sha256, ct);
        if (!ok)
        {
            _logger.LogWarning(
                "ArtifactCache: SHA256 mismatch on verify for {Sha256} — purging corrupt file",
                sha256[..12]);
            TryDeleteFile(path);
        }

        return ok;
    }

    /// <inheritdoc/>
    public Task<bool> PurgeAsync(string sha256, CancellationToken ct = default)
    {
        ValidateSha256(sha256);
        var path = CachePath(sha256);
        var deleted = TryDeleteFile(path);

        if (deleted)
            _logger.LogInformation("ArtifactCache: purged {Sha256}", sha256[..12]);

        return Task.FromResult(deleted);
    }

    /// <inheritdoc/>
    public async Task PruneAsync(long maxCacheBytes, CancellationToken ct = default)
    {
        var files = new DirectoryInfo(_cacheDir)
            .GetFiles()
            .Where(f => !f.Name.EndsWith(".tmp", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f.LastAccessTimeUtc)
            .ToList();

        var totalBytes = files.Sum(f => f.Length);

        if (totalBytes <= maxCacheBytes)
        {
            _logger.LogDebug(
                "ArtifactCache: {TotalMb:F0} MB in cache, under {MaxMb:F0} MB limit — no pruning needed",
                totalBytes / 1_048_576.0, maxCacheBytes / 1_048_576.0);
            return;
        }

        _logger.LogInformation(
            "ArtifactCache: {TotalMb:F0} MB exceeds {MaxMb:F0} MB limit — pruning LRU artifacts",
            totalBytes / 1_048_576.0, maxCacheBytes / 1_048_576.0);

        foreach (var file in files)
        {
            if (totalBytes <= maxCacheBytes) break;
            ct.ThrowIfCancellationRequested();

            var size = file.Length;
            if (TryDeleteFile(file.FullName))
            {
                totalBytes -= size;
                _logger.LogDebug("ArtifactCache: pruned {FileName} ({Kb:F0} KB)", file.Name, size / 1024.0);
            }
        }
    }

    // ── Download and verify ──────────────────────────────────────────────

    private async Task DownloadAndVerifyAsync(
        string sha256,
        string sourceUrl,
        string finalPath,
        CancellationToken ct)
    {
        var tmpPath = finalPath + ".tmp";

        _logger.LogInformation(
            "ArtifactCache: downloading {Sha256} from {Url}",
            sha256[..12], sourceUrl);

        var client = _httpClient;

        try
        {
            using var response = await client.GetAsync(sourceUrl, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();

            using var downloadStream = await response.Content.ReadAsStreamAsync(ct);

            // Write to temp file while computing SHA256 in one streaming pass.
            using var sha = SHA256.Create();
            using var cryptoStream = new CryptoStream(downloadStream, sha, CryptoStreamMode.Read);
            using var fileStream = new FileStream(
                tmpPath, FileMode.Create, FileAccess.Write,
                FileShare.None, bufferSize: 81920, useAsync: true);

            await cryptoStream.CopyToAsync(fileStream, ct);
            await fileStream.FlushAsync(ct);

            // Finalize the hash — required before reading sha.Hash.
            cryptoStream.FlushFinalBlock();

            var actualHash = Convert.ToHexString(sha.Hash!).ToLowerInvariant();

            if (!string.Equals(actualHash, sha256, StringComparison.OrdinalIgnoreCase))
            {
                TryDeleteFile(tmpPath);
                throw new InvalidOperationException(
                    $"ArtifactCache: SHA256 mismatch for artifact from {sourceUrl}. " +
                    $"Expected {sha256[..12]}, got {actualHash[..12]}. " +
                    "The artifact at the source URL may have been tampered with or replaced.");
            }
        }
        catch (Exception ex) when (ex is not InvalidOperationException && ex is not OperationCanceledException)
        {
            TryDeleteFile(tmpPath);
            throw;
        }

        // Atomic rename: temp → final.
        File.Move(tmpPath, finalPath, overwrite: true);

        // Harden file permissions on Linux.
        EnforceFilePermissions(finalPath);

        _logger.LogInformation(
            "ArtifactCache: stored {Sha256} ({Kb:F0} KB)",
            sha256[..12], new FileInfo(finalPath).Length / 1024.0);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private string CachePath(string sha256) =>
        Path.Combine(_cacheDir, sha256.ToLowerInvariant());

    private static async Task<bool> VerifyFileAsync(string path, string expectedSha256, CancellationToken ct)
    {
        try
        {
            using var sha = SHA256.Create();
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true);
            var hash = await sha.ComputeHashAsync(stream, ct);
            return string.Equals(
                Convert.ToHexString(hash).ToLowerInvariant(),
                expectedSha256.ToLowerInvariant(),
                StringComparison.Ordinal);
        }
        catch
        {
            return false;
        }
    }

    private async Task<SemaphoreSlim> GetOrCreateLockAsync(string sha256, CancellationToken ct)
    {
        await _dictLock.WaitAsync(ct);
        try
        {
            if (!_downloadLocks.TryGetValue(sha256, out var sem))
            {
                sem = new SemaphoreSlim(1, 1);
                _downloadLocks[sha256] = sem;
            }
            return sem;
        }
        finally
        {
            _dictLock.Release();
        }
    }

    private void EnsureCacheDirectory()
    {
        if (!Directory.Exists(_cacheDir))
        {
            Directory.CreateDirectory(_cacheDir);
            _logger.LogInformation("ArtifactCache: created cache directory at {Dir}", _cacheDir);
        }

        EnforceDirectoryPermissions(_cacheDir);
    }

    private static void EnforceFilePermissions(string path)
    {
        if (OperatingSystem.IsWindows()) return;
        try
        {
            File.SetUnixFileMode(path,
                UnixFileMode.UserRead | UnixFileMode.UserWrite |
                UnixFileMode.GroupRead);
        }
        catch { /* Non-fatal — log at call site if needed */ }
    }

    private static void EnforceDirectoryPermissions(string path)
    {
        if (OperatingSystem.IsWindows()) return;
        try
        {
            File.SetUnixFileMode(path,
                UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute |
                UnixFileMode.GroupRead | UnixFileMode.GroupExecute);
        }
        catch { /* Non-fatal */ }
    }

    private static bool TryDeleteFile(string path)
    {
        try { File.Delete(path); return true; }
        catch { return false; }
    }

    // ── Validation ───────────────────────────────────────────────────────

    private static void ValidateSha256(string sha256)
    {
        if (sha256.Length != 64 || !sha256.All(c => (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
            throw new ArgumentException($"Invalid SHA256: must be 64 lowercase hex characters. Got: '{sha256}'", nameof(sha256));
    }

    private static void ValidateSourceUrl(string sourceUrl)
    {
        if (!Uri.TryCreate(sourceUrl, UriKind.Absolute, out var uri) || uri.Scheme != "https")
            throw new ArgumentException($"Invalid source URL: only HTTPS is accepted. Got: '{sourceUrl}'", nameof(sourceUrl));
    }
}
