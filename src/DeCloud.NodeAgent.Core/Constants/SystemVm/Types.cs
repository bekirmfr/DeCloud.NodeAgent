using DeCloud.NodeAgent.Core.Models;

namespace DeCloud.NodeAgent.Core.Constants
{
    public static class SystemVmConstants
    {
        public static readonly HashSet<VmType> Types =
        [
            VmType.Relay,
            VmType.Dht,
            VmType.BlockStore
        ];
    }
}
