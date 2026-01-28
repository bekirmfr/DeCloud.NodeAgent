using DeCloud.NodeAgent.Core.Models.UserNetwork;

namespace DeCloud.NodeAgent.Core.Interfaces.UserNetwork;

public interface IUserWireGuardManager
{
    Task<UserWireGuardConfig> EnsureUserNetworkAsync(string userId, CancellationToken ct = default);
    Task<string> AllocateVmIpAsync(string userId, string vmId, CancellationToken ct = default);
    Task AddVmPeerAsync(string userId, string vmId, string vmIp, CancellationToken ct = default);
    Task RemoveVmPeerAsync(string userId, string vmId, CancellationToken ct = default);
    Task CleanupUserNetworkIfEmptyAsync(string userId, CancellationToken ct = default);
    Task<UserWireGuardConfig?> GetUserNetworkAsync(string userId, CancellationToken ct = default);
    Task<List<UserWireGuardConfig>> ListUserNetworksAsync(CancellationToken ct = default);
}