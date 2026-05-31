namespace DeCloud.NodeAgent.Infrastructure.Libvirt;

/// <summary>
/// Multi-architecture support for LibvirtVmManager
/// 
/// SECURITY: Architecture validation prevents incompatible VM creation
/// KISS: Simple architecture mapping without complex emulation layers
/// PERFORMANCE: Native architecture for maximum efficiency on low-end ARM devices
/// </summary>
public static class ArchitectureHelper
{
    /// <summary>
    /// Architecture configuration for VM creation
    /// </summary>
    public class ArchConfig
    {
        public required string Architecture { get; init; }
        public required string QemuEmulator { get; init; }
        public required string MachineType { get; init; }
        public required string ArchTag { get; init; } // For image URLs
    }

    /// <summary>
    /// Supported architectures
    /// </summary>
    public static readonly Dictionary<string, ArchConfig> SupportedArchitectures = new()
    {
        ["x86_64"] = new ArchConfig
        {
            Architecture = "x86_64",
            QemuEmulator = "/usr/bin/qemu-system-x86_64",
            MachineType = "q35",
            ArchTag = "amd64"
        },
        ["aarch64"] = new ArchConfig
        {
            Architecture = "aarch64",
            QemuEmulator = "/usr/bin/qemu-system-aarch64",
            MachineType = "virt",
            ArchTag = "arm64"
        }
    };

    /// <summary>
    /// Get architecture configuration for host
    /// </summary>
    public static ArchConfig GetHostArchConfig(string architecture)
    {
        // Normalize architecture names
        var normalized = architecture.ToLower() switch
        {
            "x86_64" or "amd64" or "x64" => "x86_64",
            "aarch64" or "arm64" => "aarch64",
            _ => throw new NotSupportedException($"Unsupported architecture: {architecture}")
        };

        if (!SupportedArchitectures.TryGetValue(normalized, out var config))
        {
            throw new NotSupportedException($"Architecture {normalized} not supported for VM hosting");
        }

        return config;
    }
}