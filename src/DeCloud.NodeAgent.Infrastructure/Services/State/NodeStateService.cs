// =====================================================================
// NodeStateService - Implementation
// =====================================================================
// File: src/DeCloud.NodeAgent.Infrastructure/Services/State/NodeStateService.cs
//
// Thread-safe singleton implementation.
// Register as Singleton in DI container.
// Includes authentication state (replaces AuthenticationStateService)
// =====================================================================

using DeCloud.NodeAgent.Core.Interfaces;
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

    private NodeOperationalStatus _status = NodeOperationalStatus.Initializing;
    private AuthenticationState _authState = AuthenticationState.NotAuthenticated;
    private DateTime? _lastHeartbeat;
    private DateTime? _lastSync;
    private int _consecutiveFailures;

    // ================================================================
    // ASYNC WAITERS
    // ================================================================

    private readonly TaskCompletionSource _registrationComplete = new();
    private readonly TaskCompletionSource _discoveryComplete = new();

    // ================================================================
    // CONSTANTS
    // ================================================================

    /// <summary>
    /// Orchestrator considered unreachable if no successful contact in this period
    /// </summary>
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

    public NodeOperationalStatus Status
    {
        get { lock (_lock) return _status; }
    }

    public bool IsHealthy
    {
        get
        {
            lock (_lock)
            {
                return
                    IsAuthenticated &&
                    IsRegistered &&
                    IsOrchestratorReachable &&
                    IsOnline;
            }
        }
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
        get { lock (_lock) return _authState == AuthenticationState.PendingRegistration; }
    }

    public bool IsRegistered
    {
               get { lock (_lock) return _authState == AuthenticationState.Registered; }
    }

    public bool IsOnline
    {
               get { lock (_lock) return _status == NodeOperationalStatus.Online; }
    }

    public bool IsDiscoveryComplete
    {
        get
        {
            lock (_lock)
            {
                return _authState == AuthenticationState.PendingRegistration;
            }
        }
    }

    // ================================================================
    // PROPERTIES - Connection State
    // ================================================================

    public bool IsOrchestratorReachable
    {
        get
        {
            lock (_lock)
            {
                var lastContact = GetLastSuccessfulContact();
                if (lastContact == null) return false;
                return (DateTime.UtcNow - lastContact.Value) < OrchestratorTimeout;
            }
        }
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

    public void SetStatus(NodeOperationalStatus status)
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

                _logger.LogInformation(
                    "Auth state changed: {OldState} → {NewState}",
                    oldState, state);

                // Signal waiters
                if (state == AuthenticationState.Registered)
                {
                    _registrationComplete.TrySetResult();
                }

                if (state != AuthenticationState.NotAuthenticated &&
                    state != AuthenticationState.WaitingForDiscovery)
                {
                    _discoveryComplete.TrySetResult();
                }
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
                if (_status == NodeOperationalStatus.Initializing ||
                    _status == NodeOperationalStatus.Degraded)
                {
                    SetStatusInternal(NodeOperationalStatus.Online);
                }
            }
            else
            {
                _consecutiveFailures++;

                _logger.LogWarning(
                    "Heartbeat failed (consecutive failures: {Count})",
                    _consecutiveFailures);

                // Auto-transition to Degraded after multiple failures
                if (_consecutiveFailures >= 3 && _status == NodeOperationalStatus.Online)
                {
                    SetStatusInternal(NodeOperationalStatus.Degraded);
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
                IsDiscoveryComplete = IsDiscoveryComplete,
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

    /// <summary>
    /// Internal status change (already inside lock)
    /// </summary>
    private void SetStatusInternal(NodeOperationalStatus status)
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