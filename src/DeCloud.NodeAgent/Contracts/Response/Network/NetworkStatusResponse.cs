namespace DeCloud.NodeAgent.Contracts.Response.Network
{
    public class NetworkStatusResponse
    {
        public bool IsInternetReachable { get; set; }
        public bool IsOrchestratorReachable { get; set; }
    }
}
