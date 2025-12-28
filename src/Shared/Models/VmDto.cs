// DeCloud.Shared/Models/VmDto.cs
namespace DeCloud.Shared.Models;

/// <summary>
/// Canonical VM representation for orchestrator ↔ node communication
/// </summary>
public class VmDto
{
    // Identity (aligned naming)
    public required string VmId { get; set; }
    public required string Name { get; set; }
    public required string TenantId { get; set; }
    public required string TenantWallet { get; set; }
    
    // Resources (use smallest unit - bytes)
    public required int VCpus { get; set; }
    public required long MemoryBytes { get; set; }
    public required long DiskBytes { get; set; }
    
    // Quality & Billing
    public required int QualityTier { get; set; }
    public required int ComputePointCost { get; set; }
    
    // State (use string for serialization)
    public required string State { get; set; }
    
    // Network (aligned naming)
    public required string? MacAddress { get; set; }
    public required string? IpAddress { get; set; }
    public required string? VncPort { get; set; }
    
    // Auth (support both methods)
    public required string? SshPublicKey { get; set; }
    public required string? EncryptedPassword { get; set; }
    
    // Metadata
    public required string? ImageId { get; set; }
    public required string? LeaseId { get; set; }
    
    // Timestamps (ISO format strings for JSON)
    public required DateTime CreatedAt { get; set; }
    public required DateTime? StartedAt { get; set; }
}