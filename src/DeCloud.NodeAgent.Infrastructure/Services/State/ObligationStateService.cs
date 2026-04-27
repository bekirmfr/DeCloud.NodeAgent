using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Models.State;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services.State;

// ============================================================
// Placement: src/DeCloud.NodeAgent.Infrastructure/Services/State/ObligationStateService.cs
// ============================================================

/// <summary>
/// Implements <see cref="IObligationStateService"/> on top of
/// <see cref="ObligationStateRepository"/>.
///
/// Responsibilities:
///   • Validate and canonicalise role names before hitting the repository.
///   • Enforce version-based conflict resolution for identity state
///     (incoming &gt; stored → write).
///   • Filter pushed obligation lists through canonicalisation and warn on
///     unknown role names rather than letting them reach SQL.
///   • Ensure DB file permissions remain 0600 after the first write (covers
///     the case where SQLite created the file after
///     <c>EnforceFilePermissions</c> ran in the repository constructor).
///   • Never log the state JSON content — only role and version numbers.
/// </summary>
public sealed class ObligationStateService : IObligationStateService
{
    private readonly ObligationStateRepository _repository;
    private readonly ILogger<ObligationStateService> _logger;

    // Track whether we've done the post-creation permission check.
    private int _permissionCheckDone = 0; // 0 = not done, 1 = done

    public ObligationStateService(
        ObligationStateRepository repository,
        ILogger<ObligationStateService> logger)
    {
        _repository = repository;
        _logger = logger;
    }

    // ════════════════════════════════════════════════════════════════════
    // Identity
    // ════════════════════════════════════════════════════════════════════

    /// <inheritdoc/>
    public async Task<bool> SaveStateAsync(
        string role,
        string stateJson,
        int incomingVersion,
        CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null)
        {
            _logger.LogWarning("SaveStateAsync called with unknown role '{Role}' — ignored", role);
            return false;
        }

        if (string.IsNullOrWhiteSpace(stateJson))
        {
            _logger.LogWarning(
                "SaveStateAsync [{Role}] called with empty stateJson — ignored", canonical);
            return false;
        }

        if (incomingVersion < 1)
        {
            _logger.LogWarning(
                "SaveStateAsync [{Role}] called with invalid version {Version} — ignored",
                canonical, incomingVersion);
            return false;
        }

        var written = await _repository.UpsertAsync(canonical, stateJson, incomingVersion, ct);

        // After the first successful write the DB file definitely exists.
        // Re-enforce 0600 once in case SQLite created it after the constructor ran.
        if (written)
            EnsureFilePermissionsOnce();

        return written;
    }

    /// <inheritdoc/>
    public async Task<string?> GetStateJsonAsync(string role, CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null)
        {
            _logger.LogWarning("GetStateJsonAsync called with unknown role '{Role}'", role);
            return null;
        }

        return await _repository.GetStateJsonAsync(canonical, ct);
    }

    /// <inheritdoc/>
    public async Task<int> GetVersionAsync(string role, CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null)
        {
            _logger.LogWarning("GetVersionAsync called with unknown role '{Role}'", role);
            return 0;
        }

        return await _repository.GetVersionAsync(canonical, ct);
    }

    /// <inheritdoc/>
    public async Task DeleteStateAsync(string role, CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null)
        {
            _logger.LogWarning("DeleteStateAsync called with unknown role '{Role}'", role);
            return;
        }

        await _repository.DeleteAsync(canonical, ct);
    }

    // ════════════════════════════════════════════════════════════════════
    // Obligations
    // ════════════════════════════════════════════════════════════════════

    /// <inheritdoc/>
    public async Task SaveObligationsAsync(
        IReadOnlyList<ObligationDescriptor> obligations,
        CancellationToken ct = default)
    {
        // Canonicalise + filter unknown roles before hitting the repository.
        // The repository's own role validation is a defence-in-depth layer;
        // this layer handles the "Warning" logging cleanly.
        var sanitised = new List<ObligationDescriptor>(obligations.Count);
        foreach (var o in obligations)
        {
            var canonical = ValidateRole(o.Role);
            if (canonical is null)
            {
                _logger.LogWarning(
                    "SaveObligationsAsync: dropping obligation with unknown role '{Role}'",
                    o.Role);
                continue;
            }

            // Sanitise dep names too — preserve the descriptor shape but
            // filter out unknown ones with a warning. IntentComputation also
            // tolerates unknown deps, so this is belt-and-braces.
            var sanitisedDeps = new List<string>(o.Deps.Count);
            foreach (var dep in o.Deps)
            {
                var depCanonical = ValidateRole(dep);
                if (depCanonical is null)
                {
                    _logger.LogWarning(
                        "SaveObligationsAsync [{Role}]: dropping unknown dependency '{Dep}'",
                        canonical, dep);
                    continue;
                }
                sanitisedDeps.Add(depCanonical);
            }

            sanitised.Add(new ObligationDescriptor
            {
                Role = canonical,
                Deps = sanitisedDeps,
                UpdatedAt = o.UpdatedAt,
            });
        }

        await _repository.ReplaceObligationsAsync(sanitised, ct);

        _logger.LogInformation(
            "Obligations replaced — {Count} entries: [{Roles}]",
            sanitised.Count,
            string.Join(", ", sanitised.Select(o => o.Role)));
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ObligationDescriptor>> GetObligationsAsync(
        CancellationToken ct = default)
    {
        return await _repository.GetObligationsAsync(ct);
    }

    // ════════════════════════════════════════════════════════════════════
    // Private helpers
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Validates and canonicalises <paramref name="role"/>.
    /// Returns the lower-case canonical name, or <c>null</c> if unrecognised.
    /// Prevents open-ended database lookups from controller inputs.
    /// </summary>
    private static string? ValidateRole(string? role) =>
        ObligationRole.Canonicalise(role);

    /// <summary>
    /// Re-enforces 0600 on the SQLite file exactly once per service lifetime.
    /// Idempotent — safe to call from multiple threads; the Interlocked compare-exchange
    /// guarantees only one thread runs the check.
    /// </summary>
    private void EnsureFilePermissionsOnce()
    {
        if (Interlocked.CompareExchange(ref _permissionCheckDone, 1, 0) != 0)
            return;

        if (OperatingSystem.IsWindows())
            return;

        try
        {
            var path = _repository.DatabasePath;
            if (File.Exists(path))
            {
                File.SetUnixFileMode(path,
                    UnixFileMode.UserRead | UnixFileMode.UserWrite);

                _logger.LogDebug(
                    "Confirmed 0600 permissions on obligation-state.db at {Path}", path);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Could not enforce 0600 permissions on obligation-state.db — " +
                "the file may be readable by other OS users");
        }
    }
}