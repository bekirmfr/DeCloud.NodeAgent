// NEW FILE — src/DeCloud.NodeAgent.Infrastructure/Services/Compliance/NullCsamScanner.cs
// Phase 6 pass 1, build item 1 (handout §6). The honest stub: it manufactures
// no coverage. Enabled = false, and it never returns Clean for anything.

using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services.Compliance;

/// <summary>
/// Matcher-less scanner used until the real perceptual-hash matcher (gated on
/// the Microsoft CSAM Matching API agreement) is wired. Two invariants:
///
///   1. Enabled == false, so the caller records every VM's scan state as
///      NotScanned. A stub that returned Clean would be worse than no scanner
///      at all — it would manufacture false assurance about child-safety
///      coverage (handout §0 hard rule 2).
///   2. It performs no I/O: no mount, no read of the disk. Zero cost.
///
/// Test hook (smoke tests only — handout §9): configuration key
/// "Csam:Stub:ForceOutcome" set to "Match" or "Unscannable" forces that
/// outcome so the result-gate and the report→queue→suspend chain can be
/// exercised end-to-end without a matcher. A forced "Clean" is REFUSED —
/// the honesty invariant admits no exception, not even for tests. Any forced
/// value logs a loud warning every scan; it must never be set in production.
/// </summary>
public sealed class NullCsamScanner : ICsamScanner
{
    private readonly CsamOutcome? _forced;
    private readonly ILogger<NullCsamScanner> _logger;

    public NullCsamScanner(IConfiguration configuration, ILogger<NullCsamScanner> logger)
    {
        _logger = logger;

        var forced = configuration["Csam:Stub:ForceOutcome"];
        if (string.IsNullOrWhiteSpace(forced))
            return;

        if (Enum.TryParse<CsamOutcome>(forced, ignoreCase: true, out var outcome) &&
            outcome is CsamOutcome.Match or CsamOutcome.Unscannable)
        {
            _forced = outcome;
            _logger.LogWarning(
                "NullCsamScanner: Csam:Stub:ForceOutcome={Outcome} is set — TEST MODE. " +
                "Every scan will report this outcome. Remove this setting in production.",
                outcome);
        }
        else
        {
            // "Clean" (or garbage) is refused: a matcher-less stub must not be
            // configurable into claiming coverage.
            _logger.LogError(
                "NullCsamScanner: invalid Csam:Stub:ForceOutcome '{Value}' ignored — " +
                "only Match or Unscannable may be forced; Clean can never be forced.",
                forced);
        }
    }

    /// <inheritdoc />
    public bool Enabled => false;

    /// <inheritdoc />
    public Task<CsamScanResult> ScanAsync(string vmId, string frozenDiskPath, CancellationToken ct)
    {
        if (_forced is CsamOutcome.Match)
        {
            _logger.LogWarning("NullCsamScanner: forcing Match for VM {VmId} (test mode)", vmId);
            return Task.FromResult(new CsamScanResult(
                CsamOutcome.Match,
                new[] { new CsamFileResult("(forced-by-config)", CsamOutcome.Match,
                    MatchHash: "stub-forced-match", DbSource: "stub") }));
        }

        if (_forced is CsamOutcome.Unscannable)
        {
            _logger.LogWarning("NullCsamScanner: forcing Unscannable for VM {VmId} (test mode)", vmId);
            return Task.FromResult(new CsamScanResult(
                CsamOutcome.Unscannable,
                new[] { new CsamFileResult("(forced-by-config)", CsamOutcome.Unscannable) }));
        }

        return Task.FromResult(CsamScanResult.NotScanned);
    }
}
