// =====================================================================
// INodeStateService - Node State Interface
// =====================================================================
// File: src/DeCloud.NodeAgent.Core/Interfaces/INodeStateService.cs
//
// Single source of truth for node runtime state.
// Includes authentication state (replaces IAuthenticationStateService)
// =====================================================================

using DeCloud.NodeAgent.Core.Models;

namespace DeCloud.NodeAgent.Core.Interfaces;

/// <summary>
/// Global access to node runtime state.
/// Thread-safe singleton for checking/updating operational status.
/// </summary>
public interface INodeStateService
{
    // ================================================================
    // OPERATIONAL STATUS
    // ================================================================

    /// <summary>
    /// Current operational status (Online, Offline, Degraded, etc.)
    /// </summary>
    NodeOperationalStatus Status { get; }

    /// <summary>
    /// Quick health check - true if authenticated and connected
    /// </summary>
    bool IsHealthy { get; }

    // ================================================================
    // AUTHENTICATION STATE
    // ================================================================

    /// <summary>
    /// Detailed authentication state (for state machine tracking)
    /// </summary>
    AuthenticationState AuthState { get; }

    /// <summary>
    /// Whether node has completed authentication with orchestrator
    /// Equivalent to AuthState == AuthenticationState.Registered
    /// </summary>
    bool IsAuthenticated { get; }
    /// <summary>
    /// Whether node has completed registration with orchestrator
    /// Equivalent to AuthState == AuthenticationState.Registered
    /// </summary>
    bool IsRegistered { get; }

    /// <summary>
    /// Whether resource discovery is complete
    /// </summary>
    bool IsDiscoveryComplete { get; }

    // ================================================================
    // CONNECTION STATE
    // ================================================================

    /// <summary>
    /// Whether orchestrator is reachable (based on recent heartbeat/sync)
    /// </summary>
    bool IsOrchestratorReachable { get; }

    // ================================================================
    // TIMESTAMPS
    // ================================================================

    /// <summary>
    /// When node agent started
    /// </summary>
    DateTime StartedAt { get; }

    /// <summary>
    /// Last successful heartbeat to orchestrator
    /// </summary>
    DateTime? LastHeartbeat { get; }

    /// <summary>
    /// Last successful state sync with orchestrator
    /// </summary>
    DateTime? LastSync { get; }

    // ================================================================
    // FAILURE TRACKING
    // ================================================================

    /// <summary>
    /// Consecutive failed operations (heartbeat or sync)
    /// </summary>
    int ConsecutiveFailures { get; }

    // ================================================================
    // METHODS - State Updates
    // ================================================================

    void SetStatus(NodeOperationalStatus status);
    void SetAuthState(AuthenticationState state);
    void RecordHeartbeat(bool success);
    void RecordSync(bool success);

    // ================================================================
    // ASYNC WAITERS
    // ================================================================

    /// <summary>
    /// Wait until node is registered with orchestrator
    /// </summary>
    Task WaitForAuthenticationAsync(CancellationToken ct = default);

    /// <summary>
    /// Wait until resource discovery is complete
    /// </summary>
    Task WaitForDiscoveryAsync(CancellationToken ct = default);

    // ================================================================
    // SNAPSHOT
    // ================================================================

    /// <summary>
    /// Get complete state snapshot (for logging/diagnostics)
    /// </summary>
    NodeStateSnapshot GetSnapshot();
}

// ================================================================
// ENUMS
// ================================================================

public enum NodeOperationalStatus
{
    /// <summary>Node agent is starting up</summary>
    Initializing,

    /// <summary>Node is operational and accepting workloads</summary>
    Online,

    /// <summary>Node is running but has issues (e.g., can't reach orchestrator)</summary>
    Degraded,

    /// <summary>Node is not accepting new VMs, existing VMs continue</summary>
    Maintenance,

    /// <summary>Node is migrating VMs away, preparing for shutdown</summary>
    Draining,

    /// <summary>Node is not operational</summary>
    Offline
}

// ================================================================
// SNAPSHOT DTO
// ================================================================

/// <summary>
/// Immutable snapshot of node state at a point in time
/// </summary>
public record NodeStateSnapshot
{
    public NodeOperationalStatus Status { get; init; }
    public AuthenticationState AuthState { get; init; }
    public bool IsHealthy { get; init; }
    public bool IsAuthenticated { get; init; }
    public bool IsDiscoveryComplete { get; init; }
    public bool IsOrchestratorReachable { get; init; }
    public DateTime StartedAt { get; init; }
    public DateTime? LastHeartbeat { get; init; }
    public DateTime? LastSync { get; init; }
    public int ConsecutiveFailures { get; init; }
    public TimeSpan Uptime { get; init; }
    public DateTime CapturedAt { get; init; }
}