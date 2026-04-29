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
/// HTTPS downloads land at <c>{cacheDir}/{sha256}.tmp</c> so a crash
/// mid-download never leaves a corrupt entry.
///
/// Supports two source schemes:
///
/// <b>HTTPS:</b> Downloads from the author-controlled URL, verifies SHA256,
/// stores atomically. Per-SHA256 semaphore de-duplicates concurrent downloads.
///
/// <b>data: URI:</b> Decodes the base64 payload inline (no network call),
/// verifies SHA256, stores atomically. Because the bytes are already present
/// in the artifact descriptor, this is always instantaneous.
///
/// Thread safety: concurrent calls for the same SHA256 are de-duplicated
/// via per-SHA256 <see cref="SemaphoreSlim"/>. Different SHA256 values
/// proceed in parallel.
///
/// Logging discipline: Info on first write; Debug on cache-hit; Warning on
/// SHA256 mismatch, download failure, or corrupt-file purge.
/// </summary>
public sealed class ArtifactCacheService : IArtifactCacheService
{
    private readonly string _cacheDir;
    private readonly HttpClient _httpClient;
    private readonly ILogger<ArtifactCacheService> _logger;

    private readonly Dictionary<string, SemaphoreSlim> _locks = new(StringComparer.OrdinalIgnoreCase);
    private readonly SemaphoreSlim _dictLock = new(1, 1);

    public ArtifactCacheService(
        string cacheDir,
        HttpClient httpClient,
        ILogger<ArtifactCacheService> logger)
    {
        _cacheDir   = cacheDir;
        _httpClient = httpClient;
        _logger     = logger;

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

        // Acquire per-SHA256 lock to prevent concurrent duplicate work.
        var sem = await GetOrCreateLockAsync(sha256, ct);
        await sem.WaitAsync(ct);
        try
        {
            // Re-check after lock — another caller may have completed the work.
            if (File.Exists(finalPath) && await VerifyFileAsync(finalPath, sha256, ct))
            {
                _logger.LogDebug("ArtifactCache: hit {Sha256} (post-lock)", sha256[..12]);
                return finalPath;
            }

            if (IsDataUri(sourceUrl))
                await StoreFromDataUriAsync(sha256, sourceUrl, finalPath, ct);
            else
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
            if (artifact.Architecture is not null &&
                !string.Equals(artifact.Architecture, nodeArchitecture,
                    StringComparison.OrdinalIgnoreCase))
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

        if (!File.Exists(path)) return false;

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
        var deleted = TryDeleteFile(CachePath(sha256));
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
                "ArtifactCache: {TotalMb:F0} MB under {MaxMb:F0} MB limit — no pruning",
                totalBytes / 1_048_576.0, maxCacheBytes / 1_048_576.0);
            return;
        }

        _logger.LogInformation(
            "ArtifactCache: {TotalMb:F0} MB exceeds {MaxMb:F0} MB limit — pruning LRU",
            totalBytes / 1_048_576.0, maxCacheBytes / 1_048_576.0);

        foreach (var file in files)
        {
            if (totalBytes <= maxCacheBytes) break;
            ct.ThrowIfCancellationRequested();

            var size = file.Length;
            if (TryDeleteFile(file.FullName))
            {
                totalBytes -= size;
                _logger.LogDebug("ArtifactCache: pruned {FileName} ({Kb:F0} KB)",
                    file.Name, size / 1024.0);
            }
        }
    }

    // ── Inline (data: URI) storage ───────────────────────────────────────

    /// <summary>
    /// Decode a <c>data:</c> URI and store the bytes as a cached artifact.
    ///
    /// Format: <c>data:{mediaType};base64,{base64payload}</c>
    ///
    /// Verifies SHA256 of the decoded bytes before writing to the final path.
    /// Atomic: bytes land in a temp file, verified, then renamed.
    ///
    /// No network call — the bytes are already present in the URI.
    /// Inline artifacts are instantaneous to "download".
    /// </summary>
    private async Task StoreFromDataUriAsync(
        string sha256,
        string dataUri,
        string finalPath,
        CancellationToken ct)
    {
        _logger.LogInformation(
            "ArtifactCache: storing inline attachment {Sha256}",
            sha256[..12]);

        // Parse: data:{mediaType};base64,{payload}
        // The base64 payload starts after the first comma.
        var commaIndex = dataUri.IndexOf(',');
        if (commaIndex < 0)
            throw new ArgumentException(
                $"Malformed data: URI for artifact {sha256[..12]} — no comma separator.");

        var header = dataUri[..commaIndex];
        if (!header.Contains(";base64", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException(
                $"data: URI for artifact {sha256[..12]} must use base64 encoding " +
                "(e.g., data:text/x-sh;base64,...).");

        var base64Payload = dataUri[(commaIndex + 1)..].Trim();

        byte[] bytes;
        try
        {
            bytes = Convert.FromBase64String(base64Payload);
        }
        catch (FormatException ex)
        {
            throw new ArgumentException(
                $"data: URI for artifact {sha256[..12]} contains invalid base64.", ex);
        }

        // Verify SHA256 of decoded bytes.
        var actualHash = Convert.ToHexString(
            SHA256.HashData(bytes)).ToLowerInvariant();

        if (!string.Equals(actualHash, sha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(
                $"ArtifactCache: SHA256 mismatch for inline artifact. " +
                $"Expected {sha256[..12]}, got {actualHash[..12]}. " +
                "The data: URI bytes do not match the declared Sha256.");

        // Atomic write: temp → rename.
        var tmpPath = finalPath + ".tmp";
        try
        {
            await File.WriteAllBytesAsync(tmpPath, bytes, ct);
            File.Move(tmpPath, finalPath, overwrite: true);
            EnforceFilePermissions(finalPath);

            _logger.LogInformation(
                "ArtifactCache: stored inline attachment {Sha256} ({Kb:F0} KB)",
                sha256[..12], bytes.Length / 1024.0);
        }
        catch
        {
            TryDeleteFile(tmpPath);
            throw;
        }
    }

    // ── HTTPS download ───────────────────────────────────────────────────

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

        try
        {
            using var response = await _httpClient.GetAsync(
                sourceUrl, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();

            using var downloadStream = await response.Content.ReadAsStreamAsync(ct);
            using var sha = SHA256.Create();
            using var cryptoStream = new CryptoStream(downloadStream, sha, CryptoStreamMode.Read);
            using var fileStream = new FileStream(
                tmpPath, FileMode.Create, FileAccess.Write,
                FileShare.None, bufferSize: 81920, useAsync: true);

            await cryptoStream.CopyToAsync(fileStream, ct);
            await fileStream.FlushAsync(ct);

            var actualHash = Convert.ToHexString(sha.Hash!).ToLowerInvariant();

            if (!string.Equals(actualHash, sha256, StringComparison.OrdinalIgnoreCase))
            {
                TryDeleteFile(tmpPath);
                throw new InvalidOperationException(
                    $"ArtifactCache: SHA256 mismatch for {sourceUrl}. " +
                    $"Expected {sha256[..12]}, got {actualHash[..12]}. " +
                    "The artifact at the source URL may have been tampered with.");
            }
        }
        catch (Exception ex) when (ex is not InvalidOperationException && ex is not OperationCanceledException)
        {
            TryDeleteFile(tmpPath);
            throw;
        }

        File.Move(tmpPath, finalPath, overwrite: true);
        EnforceFilePermissions(finalPath);

        _logger.LogInformation(
            "ArtifactCache: stored {Sha256} ({Kb:F0} KB)",
            sha256[..12], new FileInfo(finalPath).Length / 1024.0);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private string CachePath(string sha256) =>
        Path.Combine(_cacheDir, sha256.ToLowerInvariant());

    private static async Task<bool> VerifyFileAsync(
        string path, string expectedSha256, CancellationToken ct)
    {
        try
        {
            using var sha = SHA256.Create();
            using var stream = new FileStream(
                path, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true);
            var hash = await sha.ComputeHashAsync(stream, ct);
            return string.Equals(
                Convert.ToHexString(hash).ToLowerInvariant(),
                expectedSha256.ToLowerInvariant(),
                StringComparison.Ordinal);
        }
        catch { return false; }
    }

    private async Task<SemaphoreSlim> GetOrCreateLockAsync(string sha256, CancellationToken ct)
    {
        await _dictLock.WaitAsync(ct);
        try
        {
            if (!_locks.TryGetValue(sha256, out var sem))
            {
                sem = new SemaphoreSlim(1, 1);
                _locks[sha256] = sem;
            }
            return sem;
        }
        finally { _dictLock.Release(); }
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
        try { File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.GroupRead); }
        catch { }
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
        catch { }
    }

    private static bool TryDeleteFile(string path)
    {
        try { File.Delete(path); return true; }
        catch { return false; }
    }

    // ── Validation ───────────────────────────────────────────────────────

    private static void ValidateSha256(string sha256)
    {
        if (sha256.Length != 64 ||
            !sha256.All(c =>
                (c >= '0' && c <= '9') ||
                (c >= 'a' && c <= 'f') ||
                (c >= 'A' && c <= 'F')))
            throw new ArgumentException(
                $"Invalid SHA256: must be 64 hex characters. Got: '{sha256}'",
                nameof(sha256));
    }

    private static void ValidateSourceUrl(string sourceUrl)
    {
        if (IsDataUri(sourceUrl))
        {
            // data: URIs are allowed — validated at decode time.
            return;
        }

        if (!Uri.TryCreate(sourceUrl, UriKind.Absolute, out var uri) || uri.Scheme != "https")
            throw new ArgumentException(
                $"Invalid source URL: only HTTPS or data: URIs are accepted. Got: '{sourceUrl}'",
                nameof(sourceUrl));
    }

    private static bool IsDataUri(string url) =>
        url.StartsWith("data:", StringComparison.OrdinalIgnoreCase);
}
