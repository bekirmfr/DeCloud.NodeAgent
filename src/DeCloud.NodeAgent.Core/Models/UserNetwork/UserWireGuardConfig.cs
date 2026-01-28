namespace DeCloud.NodeAgent.Core.Models.UserNetwork;

public class UserWireGuardConfig
{
    public string UserId { get; set; } = string.Empty;
    public string InterfaceName { get; set; } = string.Empty;
    public string PrivateKey { get; set; } = string.Empty;
    public string PublicKey { get; set; } = string.Empty;
    public string Subnet { get; set; } = string.Empty;
    public string LocalIp { get; set; } = string.Empty;
    public int ListenPort { get; set; } = 51820;
    public List<VmPeerInfo> VmPeers { get; set; } = new();
    public List<NodePeerInfo> NodePeers { get; set; } = new();
    public DateTime CreatedAt { get; set; }
    public DateTime LastModified { get; set; }
}

public class VmPeerInfo
{
    public string VmId { get; set; } = string.Empty;
    public string VmIp { get; set; } = string.Empty;
    public DateTime AddedAt { get; set; }
}

public class NodePeerInfo
{
    public string NodeId { get; set; } = string.Empty;
    public string PublicKey { get; set; } = string.Empty;
    public string Endpoint { get; set; } = string.Empty;
    public List<string> AllowedIps { get; set; } = new();
    public DateTime AddedAt { get; set; }
}