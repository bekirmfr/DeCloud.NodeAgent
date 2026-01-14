namespace Orchestrator.Models; // or NodeAgent.Models

/// <summary>
/// Lightweight scheduling configuration snapshot for Node Agents
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
    /// Last update timestamp (for debugging/logging)
    /// </summary>
    public DateTime UpdatedAt { get; set; }
}