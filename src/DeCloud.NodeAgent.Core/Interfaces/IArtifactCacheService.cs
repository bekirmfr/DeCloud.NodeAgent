using DeCloud.Shared.Models;

namespace DeCloud.NodeAgent.Core.Interfaces;

// ============================================================
// Placement: src/DeCloud.NodeAgent.Core/Interfaces/IArtifactCacheService.cs
// ============================================================

/// <summary>
/// Manages the node-local artifact cache: a flat directory of files keyed by
/// their SHA256 digest. Artifacts are downloaded from author-controlled URLs,
/// verified, and served to VMs over virbr0 via <c>ArtifactCacheController</c>.
///
/// SECURITY CONTRACT
///   • Only HTTPS source URLs are accepted.
///   • SHA256 is verified on every download. A mismatch causes the temp file
///     to be discarded; the method throws to signal the failure.
///   • Files in the cache are 0640 (owner rw, group r, world none).
///   • The cache directory is 0750.
///   • Callers must never expose the local file path outside the process —
///     only serve via the controller, which enforces the virbr0 restriction.
///
/// CACHE SEMANTICS
///   • Keyed by SHA256 (lower-case hex). Two identical blobs from different
///     URLs share one cache entry.
///   • <see cref="EnsureCachedAsync"/> is idempotent: if the artifact is
///     already present and verified, it returns immediately.
///   • Download is atomic: bytes land in a temp file first, SHA256 is
///     verified, then the temp file is renamed to the final path. A crash
///     mid-download leaves only the temp file, which is cleaned up on next
///     startup or by the next call to <see cref="EnsureCachedAsync"/>.
/// </summary>
public interface IArtifactCacheService
{
    /// <summary>
    /// Return the local file path of a cached artifact, or <c>null</c> if the
    /// artifact is not cached. Does not initiate a download.
    ///
    /// Callers: <c>ArtifactCacheController.GetArtifact</c> (check before serve).
    /// </summary>
    /// <param name="sha256">Lower-case SHA256 hex digest (64 chars).</param>
    Task<string?> GetLocalPathAsync(string sha256, CancellationToken ct = default);

    /// <summary>
    /// Ensure the artifact identified by <paramref name="sha256"/> is present
    /// in the cache. If it is already present and the SHA256 matches, returns
    /// immediately without any I/O. Otherwise, downloads from
    /// <paramref name="sourceUrl"/>, verifies SHA256, and stores atomically.
    ///
    /// Callers: P9 prefetch triggered when a system template lands in SQLite;
    /// future tenant-VM prefetch triggered at deployment time.
    /// </summary>
    /// <param name="sha256">Expected SHA256 (lower-case hex, 64 chars).</param>
    /// <param name="sourceUrl">HTTPS URL to fetch from on cache miss. Must be HTTPS.</param>
    /// <returns>Local file path of the cached artifact.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown if <paramref name="sha256"/> is not a valid 64-char hex string, or if
    /// <paramref name="sourceUrl"/> is not HTTPS.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the downloaded bytes do not match <paramref name="sha256"/>.
    /// The corrupted temp file is discarded before throwing.
    /// </exception>
    Task<string> EnsureCachedAsync(
        string sha256,
        string sourceUrl,
        CancellationToken ct = default);

    /// <summary>
    /// Prefetch all architecture-appropriate artifacts from a template.
    /// Calls <see cref="EnsureCachedAsync"/> for each artifact in
    /// <paramref name="artifacts"/> whose <c>Architecture</c> matches
    /// <paramref name="nodeArchitecture"/> (or is null / universal).
    ///
    /// Errors for individual artifacts are logged at Warning and do not abort
    /// the remaining prefetch — a partial failure is better than a full stop.
    /// Callers should check <see cref="GetLocalPathAsync"/> for each artifact
    /// before dispatching a deploy command.
    ///
    /// Callers: P9 (system template received), P10 (reconciler before Create).
    /// </summary>
    /// <param name="artifacts">Artifact list from the system template.</param>
    /// <param name="nodeArchitecture">"amd64" or "arm64" — filters binary artifacts.</param>
    Task PrefetchAsync(
        IReadOnlyList<TemplateArtifact> artifacts,
        string nodeArchitecture,
        CancellationToken ct = default);

    /// <summary>
    /// Verify that <paramref name="sha256"/> is cached and the stored bytes
    /// still match the digest. Returns <c>false</c> if the artifact is absent
    /// or corrupted (the corrupt file is purged in that case).
    /// </summary>
    Task<bool> VerifyAsync(string sha256, CancellationToken ct = default);

    /// <summary>
    /// Remove a cached artifact. No-op if the artifact is not cached.
    /// Returns <c>true</c> if a file was deleted.
    /// </summary>
    Task<bool> PurgeAsync(string sha256, CancellationToken ct = default);

    /// <summary>
    /// Evict least-recently-accessed artifacts until the cache total is below
    /// <paramref name="maxCacheBytes"/>. No-op if already within budget.
    ///
    /// Intended to be called periodically or when disk pressure is detected.
    /// Node operators configure <paramref name="maxCacheBytes"/> via
    /// <c>LibvirtVmManagerOptions.ArtifactCacheMaxBytes</c> (default: 2 GB).
    /// </summary>
    Task PruneAsync(long maxCacheBytes, CancellationToken ct = default);
}
