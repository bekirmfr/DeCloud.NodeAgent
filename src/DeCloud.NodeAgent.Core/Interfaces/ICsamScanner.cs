// NEW FILE — src/DeCloud.NodeAgent.Core/Interfaces/ICsamScanner.cs
// Phase 6 pass 1, build item 1 (handout §6). The seam the real matcher slots
// into later with no pipeline change.

namespace DeCloud.NodeAgent.Core.Interfaces;

/// <summary>
/// Truthful scan states. The four values are NOT a clean/not-clean binary
/// (plan Decision 9) and no code path may upgrade any of them to Clean:
///   NotScanned  — no matcher wired, or the scan did not run. Honest default.
///   Clean       — a real matcher hashed the file(s) and found no match.
///   Match       — a positive known-CSAM hash match. The ONLY blocking state.
///   Unscannable — the scanner ran but could not hash a file (too large for
///                 the per-file budget, unreadable, etc.). Falls to the
///                 reactive path; never rendered as Clean.
/// </summary>
public enum CsamOutcome
{
    NotScanned,
    Clean,
    Match,
    Unscannable
}

/// <summary>Per-file result. Only hashes and metadata — never file content.</summary>
public sealed record CsamFileResult(
    string Path,
    CsamOutcome Outcome,
    string? MatchHash = null,
    string? DbSource = null);

/// <summary>
/// Whole-scan result. <see cref="Overall"/> is the summary the caller records
/// and gates on: Match if any file matched; NotScanned when no matcher is
/// wired; Unscannable if any file could not be hashed (and none matched);
/// Clean only when a real matcher hashed every changed media file clean.
/// Deferral ("not finished in budget") is deliberately NOT an outcome — it is
/// expressed by the scan being cancelled by the caller's budget token, because
/// "not finished yet" is a scheduling fact, not a truth about the content.
/// </summary>
public sealed record CsamScanResult(CsamOutcome Overall, IReadOnlyList<CsamFileResult> Files)
{
    public static readonly CsamScanResult NotScanned =
        new(CsamOutcome.NotScanned, System.Array.Empty<CsamFileResult>());
}

/// <summary>
/// CSAM scanner seam (plan Decisions 1, 9, 15). Called once per lazysync cycle
/// per enrolled tenant VM, on the FROZEN, coherent, plaintext disk.qcow2 —
/// after the blockdev-snapshot, before anything leaves the node.
///
/// The scanner owns its own filesystem access: it receives the frozen disk
/// path, not a mount point. The null implementation therefore never mounts
/// anything (zero cost), and the real matcher encapsulates the mount
/// discipline in one place:
///   - libguestfs/guestmount with LIBGUESTFS_BACKEND=direct, READ-ONLY
///     (reuse the CloudInitCleaner pattern; the appliance QEMU isolates the
///     host kernel from the adversarial guest FS).
///   - NEVER a host-kernel nbd mount of a tenant filesystem.
///   - Magic-byte type gate: only image/video files enter the hash path.
///   - Per-file change detection via a persisted { path → size, mtime, hash }
///     map; per-file size/time caps resolve to Unscannable, never Clean.
///   - Hashes only ever leave the node — never file content.
/// These mechanics land with the real matcher; they are recorded here so the
/// contract is pinned now.
/// </summary>
public interface ICsamScanner
{
    /// <summary>
    /// False while no real matcher is wired (the null scanner). The CALLER owns
    /// the honesty wiring: when Enabled is false the recorded scan state is
    /// NotScanned — never Clean — regardless of what ScanAsync returns.
    /// </summary>
    bool Enabled { get; }

    /// <summary>
    /// Scan the frozen disk image. Must honor <paramref name="ct"/> promptly:
    /// the caller enforces the per-cycle scan budget through it, and a
    /// cancelled scan means "defer this cycle", not any content outcome.
    /// </summary>
    Task<CsamScanResult> ScanAsync(string vmId, string frozenDiskPath, CancellationToken ct);
}
