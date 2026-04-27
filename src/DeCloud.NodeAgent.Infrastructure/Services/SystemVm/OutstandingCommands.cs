using DeCloud.NodeAgent.Core.Interfaces.SystemVm;
using DeCloud.NodeAgent.Core.Models.SystemVm;
using DeCloud.Shared.Models;
using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace DeCloud.NodeAgent.Infrastructure.Services.SystemVm;

/// <summary>
/// Default in-memory implementation of <see cref="IOutstandingCommands"/>.
///
/// Backed by a <see cref="ConcurrentDictionary{TKey,TValue}"/> keyed by
/// canonical role name. All operations are O(1) and lock-free for reads;
/// writes use the dictionary's internal striped locking. <see cref="SweepExpired"/>
/// snapshots the keys before iterating so it never blocks concurrent writers.
///
/// Logging discipline:
///   • Set / Clear / SweepExpired log at Debug — the volume is one entry per
///     role per command per cycle, low enough to leave on in production.
///   • The CommandId is logged (it's not sensitive — just a UUID).
///   • Sweep logs at Warning when it actually expires entries — that's a
///     genuine "something didn't ack in 20 minutes" signal worth investigating.
/// </summary>
public sealed class OutstandingCommands : IOutstandingCommands
{
    private readonly ConcurrentDictionary<string, OutstandingCommand> _entries
        = new(StringComparer.OrdinalIgnoreCase);

    private readonly ILogger<OutstandingCommands> _logger;
    private readonly TimeProvider _time;

    public OutstandingCommands(ILogger<OutstandingCommands> logger)
        : this(logger, TimeProvider.System)
    {
    }

    /// <summary>
    /// Test-friendly constructor — accepts a custom <see cref="TimeProvider"/>
    /// so unit tests can drive expiry deterministically without sleeping.
    /// </summary>
    internal OutstandingCommands(ILogger<OutstandingCommands> logger, TimeProvider time)
    {
        _logger = logger;
        _time = time;
    }

    /// <inheritdoc/>
    public bool TryGet(string role, out OutstandingCommand command)
    {
        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
        {
            command = null!;
            return false;
        }

        return _entries.TryGetValue(canonical, out command!);
    }

    /// <inheritdoc/>
    public void Set(string role, OutstandingCommand command)
    {
        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
        {
            _logger.LogWarning(
                "OutstandingCommands.Set: unknown role '{Role}' — entry dropped (commandId: {CommandId})",
                role, command.CommandId);
            return;
        }

        _entries[canonical] = command;

        _logger.LogDebug(
            "OutstandingCommands.Set [{Role}] {Kind} commandId={CommandId} vmId={VmId} issuedAt={IssuedAt:o}",
            canonical, command.Kind, command.CommandId, command.VmId ?? "(none)", command.IssuedAt);
    }

    /// <inheritdoc/>
    public bool Clear(string role)
    {
        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
            return false;

        var removed = _entries.TryRemove(canonical, out var entry);
        if (removed)
        {
            _logger.LogDebug(
                "OutstandingCommands.Clear [{Role}] removed {Kind} commandId={CommandId}",
                canonical, entry!.Kind, entry.CommandId);
        }
        return removed;
    }

    /// <inheritdoc/>
    public int SweepExpired(TimeSpan timeout)
    {
        if (timeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be positive.");

        var now = _time.GetUtcNow().UtcDateTime;
        var swept = 0;

        // Snapshot keys first — ConcurrentDictionary's enumerator is consistent
        // but we want to avoid holding the snapshot lifetime over the removals.
        foreach (var key in _entries.Keys.ToArray())
        {
            if (!_entries.TryGetValue(key, out var entry))
                continue;

            var age = now - entry.IssuedAt;
            if (age <= timeout)
                continue;

            // Compare-and-remove: only remove if the entry hasn't been replaced
            // since we read it. Avoids racing with a fresh Set on the same role.
            var removed = ((ICollection<KeyValuePair<string, OutstandingCommand>>)_entries)
                .Remove(new KeyValuePair<string, OutstandingCommand>(key, entry));

            if (removed)
            {
                swept++;
                _logger.LogWarning(
                    "OutstandingCommands.SweepExpired [{Role}] timed out after {AgeSeconds:F0}s — " +
                    "{Kind} commandId={CommandId} vmId={VmId}. Next cycle will re-evaluate.",
                    key, age.TotalSeconds, entry.Kind, entry.CommandId, entry.VmId ?? "(none)");
            }
        }

        return swept;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, OutstandingCommand> Snapshot()
    {
        // ConcurrentDictionary.ToArray gives a consistent snapshot — safe to
        // hand to callers that may iterate or mutate without affecting state.
        return _entries.ToArray()
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value, StringComparer.OrdinalIgnoreCase);
    }
}