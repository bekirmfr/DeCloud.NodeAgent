// src/DeCloud.NodeAgent.Infrastructure/Services/AuthenticationStateService.cs
using DeCloud.NodeAgent.Core.Interfaces;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Thread-safe singleton for authentication state coordination
/// </summary>
public class AuthenticationStateService : IAuthenticationStateService
{
    private AuthenticationState _currentState = AuthenticationState.Initializing;
    private readonly SemaphoreSlim _stateLock = new(1, 1);
    private readonly TaskCompletionSource _registrationComplete = new();

    public AuthenticationState CurrentState => _currentState;

    public bool IsRegistered => _currentState == AuthenticationState.Registered;

    public bool IsDiscoveryComplete => _currentState != AuthenticationState.Initializing
        && _currentState != AuthenticationState.WaitingForDiscovery;

    public void UpdateState(AuthenticationState newState)
    {
        _stateLock.Wait();
        try
        {
            var oldState = _currentState;
            _currentState = newState;

            // Signal registration completion
            if (newState == AuthenticationState.Registered && !_registrationComplete.Task.IsCompleted)
            {
                _registrationComplete.TrySetResult();
            }
        }
        finally
        {
            _stateLock.Release();
        }
    }

    public async Task WaitForRegistrationAsync(CancellationToken ct)
    {
        if (IsRegistered)
            return;

        await _registrationComplete.Task.WaitAsync(ct);
    }
}