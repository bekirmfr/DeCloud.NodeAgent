// src/Shared/NodeIdGenerator.cs
// Deterministic node ID generation from hardware fingerprint + wallet address
// This file should be shared between Orchestrator and NodeAgent projects

using System.Security.Cryptography;
using System.Text;

namespace DeCloud.Shared;

/// <summary>
/// Generates deterministic node IDs from machine identity and wallet address.
/// Same hardware + same wallet = always same node ID.
/// </summary>
public static class NodeIdGenerator
{
    // Version salt - increment if algorithm changes
    private const string Salt = "decloud-node-v1";
    
    /// <summary>
    /// Generate deterministic node ID from machine ID and wallet address
    /// </summary>
    /// <param name="machineId">Hardware fingerprint (from /etc/machine-id or fallback)</param>
    /// <param name="walletAddress">Ethereum wallet address (must not be null address)</param>
    /// <returns>Deterministic UUID as string</returns>
    /// <exception cref="ArgumentException">If wallet is null address or inputs invalid</exception>
    public static string GenerateNodeId(string machineId, string walletAddress)
    {
        // Validate inputs
        if (string.IsNullOrWhiteSpace(machineId))
            throw new ArgumentException("Machine ID cannot be null or empty", nameof(machineId));
        
        if (string.IsNullOrWhiteSpace(walletAddress))
            throw new ArgumentException("Wallet address cannot be null or empty", nameof(walletAddress));
        
        // Normalize inputs (case-insensitive, trimmed)
        machineId = machineId.Trim().ToLowerInvariant();
        walletAddress = walletAddress.Trim().ToLowerInvariant();
        
        // Reject null wallet address
        if (walletAddress == "0x0000000000000000000000000000000000000000" ||
            walletAddress == "0000000000000000000000000000000000000000")
        {
            throw new ArgumentException(
                "Cannot generate node ID with null wallet address. " +
                "Provide a valid wallet address for node identity.",
                nameof(walletAddress));
        }
        
        // Create deterministic hash
        var input = $"{machineId}:{walletAddress}:{Salt}";
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(input));
        
        // Convert to UUID format for compatibility with existing code
        // Take first 16 bytes of SHA256 hash
        var guidBytes = new byte[16];
        Array.Copy(hashBytes, guidBytes, 16);
        
        return new Guid(guidBytes).ToString();
    }
    
    /// <summary>
    /// Validate that a claimed node ID matches the machine ID and wallet
    /// </summary>
    public static bool ValidateNodeId(string claimedNodeId, string machineId, string walletAddress)
    {
        try
        {
            var expectedNodeId = GenerateNodeId(machineId, walletAddress);
            return claimedNodeId.Equals(expectedNodeId, StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }
    
    /// <summary>
    /// Get machine ID from system (Linux)
    /// Falls back to hostname + MAC address hash if /etc/machine-id unavailable
    /// </summary>
    public static string GetMachineId()
    {
        try
        {
            // Primary: Use Linux machine-id (stable across reboots/OS reinstalls)
            if (File.Exists("/etc/machine-id"))
            {
                var machineId = File.ReadAllText("/etc/machine-id").Trim();
                if (!string.IsNullOrWhiteSpace(machineId))
                    return machineId;
            }
            
            // Fallback: Generate from hostname + first MAC address
            return GenerateFallbackMachineId();
        }
        catch (Exception)
        {
            // Last resort fallback
            return GenerateFallbackMachineId();
        }
    }
    
    private static string GenerateFallbackMachineId()
    {
        var hostname = Environment.MachineName;
        var mac = GetFirstMacAddress();
        var combined = $"{hostname}:{mac}";
        
        // Create stable hash
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(combined));
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLower();
    }
    
    private static string GetFirstMacAddress()
    {
        try
        {
            var networkInterfaces = System.Net.NetworkInformation.NetworkInterface.GetAllNetworkInterfaces();
            foreach (var nic in networkInterfaces)
            {
                // Skip loopback and down interfaces
                if (nic.OperationalStatus == System.Net.NetworkInformation.OperationalStatus.Up &&
                    nic.NetworkInterfaceType != System.Net.NetworkInformation.NetworkInterfaceType.Loopback)
                {
                    var mac = nic.GetPhysicalAddress().ToString();
                    if (!string.IsNullOrEmpty(mac) && mac != "000000000000")
                        return mac.ToLower();
                }
            }
        }
        catch
        {
            // Ignore errors
        }
        
        return "unknown";
    }
}
