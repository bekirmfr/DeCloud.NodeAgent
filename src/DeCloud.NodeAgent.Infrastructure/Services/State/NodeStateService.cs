// =====================================================================
// NodeStateService - Fixed Implementation
// =====================================================================
// File: src/DeCloud.NodeAgent.Infrastructure/Services/State/NodeStateService.cs
//
// FIXES:
// 1. IsDiscoveryComplete now tracks actual discovery, not auth state
// 2. Added SetDiscoveryComplete() method for ResourceDiscoveryService to call
// 3. WaitForDiscoveryAsync now works correctly
// =====================================================================

using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Thread-safe implementation of node state tracking.
/// Register as Singleton - single instance shared across all services.
/// </summary>
public class NodeStateService : INodeStateService
{
    private readonly ILogger<NodeStateService> _logger;
    private readonly object _lock = new();

    // ================================================================
    // STATE FIELDS
    // ================================================================

    private NodeStatus _status = NodeStatus.Initializing;
    private AuthenticationState _authState = AuthenticationState.NotAuthenticated;
    private bool _isDiscoveryComplete;  // Explicit tracking
    private bool _isOrchestratorReachable;  // Explicit tracking
    private bool _isInternetReachable;  // (not exposed yet)
    private DateTime? _lastHeartbeat;
    private DateTime? _lastSync;
    private int _consecutiveFailures;

    // ================================================================
    // ASYNC WAITERS
    // ================================================================

    private TaskCompletionSource _registrationComplete = new();
    private TaskCompletionSource _discoveryComplete = new();
    private TaskCompletionSource _internetComplete = new();
    private TaskCompletionSource _orchestratorComplete = new();

    // ================================================================
    // CONSTANTS
    // ================================================================

    private static readonly TimeSpan OrchestratorTimeout = TimeSpan.FromMinutes(2);

    // ================================================================
    // CONSTRUCTOR
    // ================================================================

    public NodeStateService(ILogger<NodeStateService> logger)
    {
        _logger = logger;
        StartedAt = DateTime.UtcNow;

        _logger.LogInformation("NodeStateService initialized at {StartedAt}", StartedAt);
    }

    // ================================================================
    // PROPERTIES - Operational Status
    // ================================================================

    public NodeStatus Status
    {
        get { lock (_lock) return _status; }
    }

    public bool IsHealthy
    {
        get
        {
            lock (_lock)
            {
                return IsInternetReachable &&
                       IsOrchestratorReachable &&
                       IsAuthenticated &&
                       IsOnline;
            }
        }
    }

    public bool IsOnline
    {
        get { lock (_lock) return _status == NodeStatus.Online; }
    }

    // ================================================================
    // PROPERTIES - Authentication State
    // ================================================================

    public AuthenticationState AuthState
    {
        get { lock (_lock) return _authState; }
    }

    public bool IsAuthenticated
    {
        get { lock (_lock) return _authState == AuthenticationState.Registered; }
    }

    // ================================================================
    // PROPERTIES - Discovery State (FIXED)
    // ================================================================

    /// <summary>
    /// Whether hardware discovery has completed.
    /// Set by ResourceDiscoveryService after initial scan.
    /// </summary>
    public bool IsDiscoveryComplete
    {
        get { lock (_lock) return _isDiscoveryComplete; }
    }

    // ================================================================
    // PROPERTIES - Connection State
    // ================================================================

    public bool IsOrchestratorReachable
    {
        get { lock (_lock) return _isOrchestratorReachable; }
    }

    public bool IsInternetReachable
    {
        get { lock (_lock) return _isInternetReachable; }
    }

    // ================================================================
    // PROPERTIES - Timestamps
    // ================================================================

    public DateTime StartedAt { get; }

    public DateTime? LastHeartbeat
    {
        get { lock (_lock) return _lastHeartbeat; }
    }

    public DateTime? LastSync
    {
        get { lock (_lock) return _lastSync; }
    }

    // ================================================================
    // PROPERTIES - Failure Tracking
    // ================================================================

    public int ConsecutiveFailures
    {
        get { lock (_lock) return _consecutiveFailures; }
    }



    // ================================================================
    // METHODS - State Updates
    // ================================================================

    public void SetStatus(NodeStatus status)
    {
        lock (_lock)
        {
            if (_status != status)
            {
                var oldStatus = _status;
                _status = status;

                _logger.LogInformation(
                    "Node status changed: {OldStatus} → {NewStatus}",
                    oldStatus, status);
            }
        }
    }

    public void SetAuthState(AuthenticationState state)
    {
        lock (_lock)
        {
            if (_authState != state)
            {
                var oldState = _authState;
                _authState = state;

                // Reset TCS if moving back to unauthenticated state
                if (state == AuthenticationState.NotAuthenticated)
                {
                    _registrationComplete = new TaskCompletionSource();
                    _discoveryComplete = new TaskCompletionSource();
                }

                if (state == AuthenticationState.Registered)
                {
                    _registrationComplete.TrySetResult();
                }
            }
        }
    }

    /// <summary>
    /// Signal that hardware discovery has completed.
    /// Called by ResourceDiscoveryService after initial scan.
    /// </summary>
    public void SetDiscoveryComplete()
    {
        lock (_lock)
        {
            if (!_isDiscoveryComplete)
            {
                _isDiscoveryComplete = true;
                _discoveryComplete.TrySetResult();

                _logger.LogInformation("✓ Hardware discovery marked complete");
            }
        }
    }

    public void SetInternetReachable(bool reachable)
    {
        lock (_lock)
        {
            if (_isInternetReachable != reachable)
            {
                _isInternetReachable = reachable;
                if (reachable)
                {
                    _internetComplete.TrySetResult();
                }
                _logger.LogInformation(
                    "Internet reachability changed: {Status}",
                    reachable ? "Reachable" : "Unreachable");
            }
        }
    }

    public void SetOrchestratorReachable(bool reachable)
    {
        lock (_lock)
        {
            if (_isOrchestratorReachable != reachable)
            {
                _isOrchestratorReachable = reachable;
                if (reachable)
                {
                    _orchestratorComplete.TrySetResult();
                }
                _logger.LogInformation(
                    "Orchestrator reachability changed: {Status}",
                    reachable ? "Reachable" : "Unreachable");
            }
        }
    }

    public void RecordHeartbeat(bool success)
    {
        lock (_lock)
        {
            if (success)
            {
                _lastHeartbeat = DateTime.UtcNow;
                _consecutiveFailures = 0;

                // Auto-transition to Online if healthy
                if (_status == NodeStatus.Initializing ||
                    _status == NodeStatus.Degraded)
                {
                    SetStatusInternal(NodeStatus.Online);
                }

                SetInternetReachable(success);
                SetOrchestratorReachable(success);
            }
            else
            {
                _consecutiveFailures++;

                _logger.LogWarning(
                    "Heartbeat failed (consecutive failures: {Count})",
                    _consecutiveFailures);

                // Auto-transition to Degraded after multiple failures
                if (_consecutiveFailures >= 3 && _status == NodeStatus.Online)
                {
                    SetStatusInternal(NodeStatus.Degraded);
                }
            }
        }
    }

    public void RecordSync(bool success)
    {
        lock (_lock)
        {
            if (success)
            {
                _lastSync = DateTime.UtcNow;
                _consecutiveFailures = 0;
            }
            else
            {
                _consecutiveFailures++;

                _logger.LogWarning(
                    "Sync failed (consecutive failures: {Count})",
                    _consecutiveFailures);
            }
        }
    }

    // ================================================================
    // ASYNC WAITERS
    // ================================================================

    public async Task WaitForAuthenticationAsync(CancellationToken ct = default)
    {
        if (IsAuthenticated)
            return;

        await _registrationComplete.Task.WaitAsync(ct);
    }

    public async Task WaitForDiscoveryAsync(CancellationToken ct = default)
    {
        if (IsDiscoveryComplete)
            return;

        await _discoveryComplete.Task.WaitAsync(ct);
    }

    public async Task WaitForInternetAsync(CancellationToken ct = default)
    {
        if (IsInternetReachable)
            return;

        await  _internetComplete.Task.WaitAsync(ct);
    }

    public async Task WaitForOrchestratorAsync(CancellationToken ct = default)
    {
        if (IsOrchestratorReachable)
            return;
        await _orchestratorComplete.Task.WaitAsync(ct);
    }

    // ================================================================
    // SNAPSHOT
    // ================================================================

    public NodeStateSnapshot GetSnapshot()
    {
        lock (_lock)
        {
            return new NodeStateSnapshot
            {
                Status = _status,
                AuthState = _authState,
                IsHealthy = IsHealthy,
                IsAuthenticated = IsAuthenticated,
                IsDiscoveryComplete = _isDiscoveryComplete,
                IsOrchestratorReachable = IsOrchestratorReachable,
                StartedAt = StartedAt,
                LastHeartbeat = _lastHeartbeat,
                LastSync = _lastSync,
                ConsecutiveFailures = _consecutiveFailures,
                Uptime = DateTime.UtcNow - StartedAt,
                CapturedAt = DateTime.UtcNow
            };
        }
    }

    // ================================================================
    // PRIVATE HELPERS
    // ================================================================

    private DateTime? GetLastSuccessfulContact()
    {
        if (_lastHeartbeat == null && _lastSync == null)
            return null;

        if (_lastHeartbeat == null) return _lastSync;
        if (_lastSync == null) return _lastHeartbeat;

        return _lastHeartbeat > _lastSync ? _lastHeartbeat : _lastSync;
    }

    private void SetStatusInternal(NodeStatus status)
    {
        if (_status != status)
        {
            var oldStatus = _status;
            _status = status;

            _logger.LogInformation(
                "Node status changed: {OldStatus} → {NewStatus}",
                oldStatus, status);
        }
    }
}