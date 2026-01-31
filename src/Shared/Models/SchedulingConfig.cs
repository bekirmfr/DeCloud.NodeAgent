// DeCloud.Shared/Models/SchedulingConfig.cs
namespace DeCloud.Shared.Models;

/// <summary>
/// Lightweight scheduling configuration snapshot shared between Orchestrator and Node Agents
/// Contains only the essential parameters needed for VM CPU quota calculations
/// </summary>
public class SchedulingConfig
{
    /// <summary>
    /// Configuration version for tracking changes
    /// Node compares this to detect when config needs updating
    /// </summary>
    public int Version { get; set; }

    /// <summary>
    /// Baseline benchmark score (e.g., 1000 for Intel i3-10100)
    /// Used for calculating nodePointsPerCore = BenchmarkScore / BaselineBenchmark
    /// </summary>
    public int BaselineBenchmark { get; set; } = 1000;

    /// <summary>
    /// Burstable tier overcommit ratio (e.g., 4.0)
    /// Used for calculating CPU quotas with burstable VMs
    /// </summary>
    public double BaselineOvercommitRatio { get; set; } = 4.0;

    /// <summary>
    /// Maximum performance multiplier cap
    /// Prevents nodes with extremely high benchmarks from dominating
    /// </summary>
    public double MaxPerformanceMultiplier { get; set; } = 20.0;

    /// <summary>
    /// Configuration for each quality tier
    /// </summary>
    public Dictionary<QualityTier, TierConfiguration> Tiers { get; set; } = new();

    /// <summary>
    /// Last update timestamp (for debugging/logging)
    /// </summary>
    public DateTime UpdatedAt { get; set; }
}

/// <summary>
/// Configuration for a specific quality tier
/// </summary>
public class TierConfiguration
{
    /// <summary>
    /// Minimum benchmark score required to host this tier
    /// </summary>
    public int MinimumBenchmark { get; set; }

    /// <summary>
    /// CPU overcommit ratio for this tier
    /// Guaranteed: 1.0 (no overcommit), Standard: 1.6, Balanced: 2.7, Burstable: 4.0
    /// </summary>
    public double CpuOvercommitRatio { get; set; }

    /// <summary>
    /// Storage overcommit ratio for this tier (safe with qcow2 thin provisioning)
    /// </summary>
    public double StorageOvercommitRatio { get; set; }

    /// <summary>
    /// Price multiplier for this tier
    /// </summary>
    public decimal PriceMultiplier { get; set; }

    /// <summary>
    /// Human-readable description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Target use case for this tier
    /// </summary>
    public string TargetUseCase { get; set; } = string.Empty;
}

/// <summary>
/// Quality tier enumeration
/// </summary>
public enum QualityTier
{
    /// <summary>
    /// Dedicated resources, guaranteed performance
    /// Requires highest-performance nodes (4000+ benchmark)
    /// </summary>
    Guaranteed = 0,

    /// <summary>
    /// High performance for demanding applications
    /// Requires high-end nodes (2500+ benchmark)
    /// </summary>
    Standard = 1,

    /// <summary>
    /// Balanced performance for production workloads
    /// Requires mid-range nodes (1500+ benchmark)
    /// </summary>
    Balanced = 2,

    /// <summary>
    /// Best-effort, lowest cost
    /// Minimum acceptable performance (1000+ benchmark)
    /// </summary>
    Burstable = 3
}
