namespace DeCloud.NodeAgent.Core.Models
{
    public enum AuthenticationState
    {
        NotAuthenticated,
        CredentialsInvalid,
        WaitingForDiscovery,
        PendingRegistration,
        Registered
    }
}
