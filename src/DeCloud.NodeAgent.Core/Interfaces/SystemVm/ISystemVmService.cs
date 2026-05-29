using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared.Enums;
using DeCloud.Shared.Models;

namespace DeCloud.NodeAgent.Core.Interfaces.SystemVm;

/// <summary>
/// Central service for system VM operations. Owns VM lookup by role, version
/// tracking, template revision reporting, and dashboard proxy resolution.
///
/// Consolidates logic previously scattered across SystemVmController,
/// OrchestratorClient, SystemVmReconciler, and PortForwardingManager.
/// Each consumer was duplicating role mapping, VM lookup, or version caching.
/// </summary>
public interface ISystemVmService
{
    /// <summary>Dashboard proxy port inside all system VMs.</summary>
    const int DashboardPort = 8080;

    /// <summary>Canonical role names in dependency order.</summary>
    static readonly IReadOnlyList<string> Roles = ObligationRole.All;

    /// <summary>Role string → VmType mapping.</summary>
    static readonly IReadOnlyDictionary<string, VmRole> RoleToVmType =
        new Dictionary<string, VmRole>(StringComparer.OrdinalIgnoreCase)
        {
            [ObligationRole.Relay] = VmRole.Relay,
            [ObligationRole.Dht] = VmRole.Dht,
            [ObligationRole.BlockStore] = VmRole.BlockStore,
        };

    // ── VM lookup ────────────────────────────────────────────────────────

    /// <summary>
    /// Find the running system VM for a role. Returns null if no VM of that
    /// role is running or has no IP address assigned.
    /// </summary>
    VmInstance? GetRunningVm(string role);

    /// <summary>
    /// Build the dashboard base URL for a role's running VM.
    /// Returns null if the VM is not running.
    /// Example: "http://192.168.122.168:8080"
    /// </summary>
    string? GetDashboardBaseUrl(string role);

    // ── Binary version tracking ──────────────────────────────────────────

    /// <summary>
    /// Get the cached binary version for a role. Returns null if the VM
    /// hasn't been queried yet or doesn't expose /version.
    /// </summary>
    string? GetCachedBinaryVersion(string role);

    /// <summary>
    /// Query /version on all running system VMs. Uses cache — only queries
    /// VMs whose cached version is missing or whose VM ID has changed
    /// (indicating a redeploy). Returns { role → version } for responding VMs.
    /// </summary>
    Task<Dictionary<string, string?>> GetAllBinaryVersionsAsync(CancellationToken ct = default);

    /// <summary>
    /// Clear the cached binary version for a role. Call when a system VM
    /// is deleted so the next query fetches the fresh version from the
    /// replacement VM.
    /// </summary>
    void InvalidateVersionCache(string role);

    // ── Template revisions ───────────────────────────────────────────────

    /// <summary>
    /// Get stored template revisions for all roles from obligation-state SQLite.
    /// Returns { role → revision }. Used in heartbeat payload.
    /// </summary>
    Task<Dictionary<string, int>> GetTemplateRevisionsAsync(CancellationToken ct = default);

    /// <summary>
    /// Capture the last N lines of systemd journal from inside a VM via guest-exec.
    /// Best-effort with a short timeout — returns null if the guest agent is dead.
    /// </summary>
    Task<string?> CaptureVmJournalAsync(string vmId, int lines = 100, CancellationToken ct = default);
}
