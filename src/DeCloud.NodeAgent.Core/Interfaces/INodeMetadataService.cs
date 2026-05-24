using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared.Contracts;

namespace DeCloud.NodeAgent.Core.Interfaces
{
    public interface INodeMetadataService
    {
        string OrchestratorUrl { get; }
        string NodeId { get; }
        string MachineId { get; }
        string Name { get; }
        string? PublicIp { get; }
        string WalletAddress { get; }
        string Region { get; }
        string Zone { get; }
        /// <summary>
        /// ISO 3166-1 alpha-2 country code from <c>Node:Country</c> config.
        /// <c>"ZZ"</c> when not configured (unknown / not yet declared).
        /// </summary>
        string Country { get; }

        NodePricing? Pricing { get; }
        HardwareInventory? Inventory { get; }

        /// <summary>
        /// Operator-allocated memory in bytes, resolved from settings.
        /// Null until UpdateInventory is called (percent mode needs TotalBytes).
        /// Null also means "use platform default (90%)".
        /// </summary>
        long? AllocatedMemoryBytes { get; }
        int? AllocatedComputePoints { get; }
        int? AllocatedComputePointsPercent { get; }
        /// <summary>Raw memory allocation percentage from settings (1-95). Null = not configured.</summary>
        int? AllocatedMemoryPercent { get; }
        long? AllocatedStorageBytes { get; }
        /// <summary>Raw storage allocation percentage from settings (1-95). Null = not configured.</summary>
        int? AllocatedStoragePercent { get; }
        int? AllocatedGpuCount { get; }
        long? AllocatedGpuVramBytes { get; }
        int? AllocatedGpuVramPercent { get; }
        /// <summary>
        /// When the orchestrator last confirmed allocation, as persisted in
        /// allocation-resolved.json. Null if the cache has never been written
        /// (values are settings-derived, not orchestrator-confirmed).
        /// </summary>
        DateTime? AllocationResolvedAt { get; }

        Task InitializeAsync(CancellationToken ct = default);
        void UpdatePublicIp(string publicIp);
        void UpdateInventory(HardwareInventory inventory);
        /// <summary>
        /// Updates in-memory allocation state from the orchestrator's resolved concrete
        /// values and persists them to /etc/decloud/allocation-resolved.json.
        /// Called after every successful allocate response so the agent survives
        /// restarts with the last-known orchestrator-confirmed allocation.
        /// </summary>
        Task UpdateFromOrchestratorResolutionAsync(NodeAllocateResponse response, CancellationToken ct = default);
    }
}
