using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;

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

    /// <summary>
    /// ARM64 cloud images from major distributions
    /// SECURITY: Official cloud image URLs only - no third-party sources
    /// </summary>
    public static readonly Dictionary<string, Dictionary<string, string>> ImageUrlsByArchitecture = new()
    {
        ["amd64"] = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["ubuntu-24.04"] = "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img",
            ["ubuntu-22.04"] = "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img",
            ["ubuntu-20.04"] = "https://cloud-images.ubuntu.com/focal/current/focal-server-cloudimg-amd64.img",
            ["debian-12"] = "https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2",
            ["debian-11"] = "https://cloud.debian.org/images/cloud/bullseye/latest/debian-11-generic-amd64.qcow2",
            ["fedora-40"] = "https://download.fedoraproject.org/pub/fedora/linux/releases/40/Cloud/x86_64/images/Fedora-Cloud-Base-Generic.x86_64-40-1.14.qcow2",
            ["alpine-3.19"] = "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/cloud/nocloud_alpine-3.19.1-x86_64-bios-cloudinit-r0.qcow2"
        },
        ["arm64"] = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            // Ubuntu ARM64 images
            ["ubuntu-24.04"] = "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-arm64.img",
            ["ubuntu-22.04"] = "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-arm64.img",
            ["ubuntu-20.04"] = "https://cloud-images.ubuntu.com/focal/current/focal-server-cloudimg-arm64.img",

            // Debian ARM64 images
            ["debian-12"] = "https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-arm64.qcow2",
            ["debian-11"] = "https://cloud.debian.org/images/cloud/bullseye/latest/debian-11-generic-arm64.qcow2",

            // Fedora ARM64 images
            ["fedora-40"] = "https://download.fedoraproject.org/pub/fedora/linux/releases/40/Cloud/aarch64/images/Fedora-Cloud-Base-Generic-40-1.14.aarch64.qcow2",

            // Alpine ARM64 images
            ["alpine-3.19"] = "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/cloud/nocloud_alpine-3.19.1-aarch64-uefi-cloudinit-r0.qcow2"
        }
    };

    /// <summary>
    /// Resolve image URL based on architecture and image ID
    /// </summary>
    public static string? ResolveImageUrl(string archTag, string imageId)
    {
        if (!ImageUrlsByArchitecture.TryGetValue(archTag, out var images))
        {
            return null;
        }

        return images.GetValueOrDefault(imageId);
    }
}