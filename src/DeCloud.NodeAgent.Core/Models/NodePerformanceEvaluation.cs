namespace DeCloud.NodeAgent.Core.Models
{
    /// <summary>
    /// Result of node performance evaluation
    /// </summary>
    public class NodePerformanceEvaluation
    {
        public string NodeId { get; set; } = string.Empty;
        public string CpuModel { get; set; } = string.Empty;
        public int PhysicalCores { get; set; }
        public int BenchmarkScore { get; set; }
        public int CappedBenchmarkScore { get; set; }
        public int BaselineBenchmark { get; set; }

        /// <summary>
        /// Performance multiplier before capping
        /// </summary>
        public double PerformanceMultiplier { get; set; }

        /// <summary>
        /// Performance multiplier after capping (same as PointsPerCore)
        /// </summary>
        public double CappedPerformanceMultiplier { get; set; }

        /// <summary>
        /// Single source of truth: How many points this node provides per physical core
        /// Formula: CappedBenchmarkScore / BurstableBaseline
        /// </summary>
        public double PointsPerCore { get; set; }
        /// <summary>
        /// Gets or sets the total number of compute points granted to the node.
        /// </summary>
        public double TotalComputePoints { get; set; }

        public bool IsAcceptable { get; set; }
        public string? RejectionReason { get; set; }

        public List<QualityTier> EligibleTiers { get; set; } = new();
        public QualityTier? HighestTier { get; set; }
       
        public Dictionary<QualityTier, TierCapability> TierCapabilities { get; set; } = new();
    }
}
