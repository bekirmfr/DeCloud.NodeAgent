using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Core.Models.State;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.Shared.Enums;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services.State;

/// <summary>
/// Implements <see cref="IObligationStateService"/> on top of
/// <see cref="ObligationStateRepository"/>.
///
/// Responsibilities:
///   • Validate and canonicalise role names before hitting the repository.
///   • Enforce version/revision-based conflict resolution.
///   • Filter pushed obligation lists through canonicalisation.
///   • Ensure DB file permissions remain 0600 after the first write.
///   • Never log the state or template JSON content.
/// </summary>
public sealed class ObligationStateService : IObligationStateService
{
    private readonly ObligationStateRepository _repository;
    private readonly ILogger<ObligationStateService> _logger;

    private int _permissionCheckDone = 0;

    public ObligationStateService(
        ObligationStateRepository repository,
        ILogger<ObligationStateService> logger)
    {
        _repository = repository;
        _logger = logger;
    }

    public static Dictionary<string, VmRole> RoleToVmType = new Dictionary<string, VmRole>
    {
        ["relay"] = VmRole.Relay,
        ["dht"] = VmRole.Dht,
        ["blockstore"] = VmRole.BlockStore,
    };

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
            _logger.LogWarning("SaveStateAsync: unknown role '{Role}' — ignored", role);
            return false;
        }
        if (string.IsNullOrWhiteSpace(stateJson))
        {
            _logger.LogWarning("SaveStateAsync [{Role}]: empty stateJson — ignored", canonical);
            return false;
        }
        if (incomingVersion < 1)
        {
            _logger.LogWarning("SaveStateAsync [{Role}]: invalid version {V} — ignored", canonical, incomingVersion);
            return false;
        }

        var written = await _repository.UpsertAsync(canonical, stateJson, incomingVersion, ct);
        if (written) EnsureFilePermissionsOnce();
        return written;
    }

    /// <inheritdoc/>
    public async Task<string?> GetStateJsonAsync(string role, CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null) { _logger.LogWarning("GetStateJsonAsync: unknown role '{Role}'", role); return null; }
        return await _repository.GetStateJsonAsync(canonical, ct);
    }

    /// <inheritdoc/>
    public async Task<int> GetVersionAsync(string role, CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null) { _logger.LogWarning("GetVersionAsync: unknown role '{Role}'", role); return 0; }
        return await _repository.GetVersionAsync(canonical, ct);
    }

    /// <inheritdoc/>
    public async Task DeleteStateAsync(string role, CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null) { _logger.LogWarning("DeleteStateAsync: unknown role '{Role}'", role); return; }
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
        var sanitised = new List<ObligationDescriptor>(obligations.Count);
        foreach (var o in obligations)
        {
            var canonical = ValidateRole(o.Role);
            if (canonical is null)
            {
                _logger.LogWarning("SaveObligationsAsync: dropping unknown role '{Role}'", o.Role);
                continue;
            }

            var sanitisedDeps = new List<string>(o.Deps.Count);
            foreach (var dep in o.Deps)
            {
                var depCanonical = ValidateRole(dep);
                if (depCanonical is null)
                {
                    _logger.LogWarning("SaveObligationsAsync [{Role}]: dropping unknown dep '{Dep}'", canonical, dep);
                    continue;
                }
                sanitisedDeps.Add(depCanonical);
            }

            sanitised.Add(new ObligationDescriptor { Role = canonical, Deps = sanitisedDeps, UpdatedAt = o.UpdatedAt });
        }

        await _repository.ReplaceObligationsAsync(sanitised, ct);
        _logger.LogInformation("Obligations replaced — {Count} entries: [{Roles}]",
            sanitised.Count, string.Join(", ", sanitised.Select(o => o.Role)));
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ObligationDescriptor>> GetObligationsAsync(
        CancellationToken ct = default)
        => await _repository.GetObligationsAsync(ct);

    // ════════════════════════════════════════════════════════════════════
    // System templates
    // ════════════════════════════════════════════════════════════════════

    /// <inheritdoc/>
    public async Task<bool> SaveSystemTemplateAsync(
        string role,
        string templateJson,
        int incomingRevision,
        string? templateId = null,
        CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null)
        {
            _logger.LogWarning("SaveSystemTemplateAsync: unknown role '{Role}' — ignored", role);
            return false;
        }
        if (string.IsNullOrWhiteSpace(templateJson))
        {
            _logger.LogWarning("SaveSystemTemplateAsync [{Role}]: empty templateJson — ignored", canonical);
            return false;
        }
        if (incomingRevision < 1)
        {
            _logger.LogWarning("SaveSystemTemplateAsync [{Role}]: invalid revision {R} — ignored", canonical, incomingRevision);
            return false;
        }

        return await _repository.UpsertSystemTemplateAsync(canonical, templateJson, incomingRevision, templateId, ct);
    }

    /// <inheritdoc/>
    public async Task<string?> GetSystemTemplateJsonAsync(string role, CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null) return null;
        return await _repository.GetSystemTemplateJsonAsync(canonical, ct);
    }

    /// <inheritdoc/>
    public async Task<int> GetSystemTemplateRevisionAsync(string role, CancellationToken ct = default)
    {
        var canonical = ValidateRole(role);
        if (canonical is null) return 0;
        return await _repository.GetSystemTemplateRevisionAsync(canonical, ct);
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, int>> GetSystemTemplateRevisionsAsync(
        CancellationToken ct = default)
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var role in ObligationRole.All)
        {
            try
            {
                var revision = await _repository.GetSystemTemplateRevisionAsync(role, ct);
                if (revision > 0)
                    result[role] = revision;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Could not read system template revision for role '{Role}' — reporting 0",
                    role);
            }
        }

        return result;
    }

    // ════════════════════════════════════════════════════════════════════
    // Private helpers
    // ════════════════════════════════════════════════════════════════════

    private static string? ValidateRole(string? role) => ObligationRole.Canonicalise(role);

    private void EnsureFilePermissionsOnce()
    {
        if (Interlocked.CompareExchange(ref _permissionCheckDone, 1, 0) != 0) return;
        if (OperatingSystem.IsWindows()) return;

        try
        {
            var path = _repository.DatabasePath;
            if (File.Exists(path))
            {
                File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite);
                _logger.LogDebug("Confirmed 0600 permissions on obligation-state.db at {Path}", path);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not enforce 0600 on obligation-state.db — may be world-readable");
        }
    }
}