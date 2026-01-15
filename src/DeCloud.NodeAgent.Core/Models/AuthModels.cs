namespace DeCloud.NodeAgent.Core.Models
{
    public enum AuthenticationState
    {
        WaitingForDiscovery,
        NotAuthenticated,
        PendingRegistration,
        Registered,
        CredentialsInvalid
    }
}
