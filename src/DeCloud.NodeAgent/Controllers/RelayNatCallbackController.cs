using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Infrastructure.Network;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System.Security.Cryptography;
using System.Text;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Handles callbacks from relay VMs to configure NAT port forwarding
/// when the VM obtains its IP address
/// </summary>
[ApiController]
[Route("api/relay")]
public class RelayNatCallbackController : ControllerBase
{
    private readonly INatRuleManager _natRuleManager;
    private readonly ILogger<RelayNatCallbackController> _logger;

    public RelayNatCallbackController(
        INatRuleManager natRuleManager,
        ILogger<RelayNatCallbackController> logger)
    {
        _natRuleManager = natRuleManager;
        _logger = logger;
    }

    /// <summary>
    /// Relay VM calls this endpoint when it obtains an IP address
    /// to trigger NAT port forwarding configuration on the host
    /// </summary>
    [HttpPost("nat-ready")]
    public async Task<IActionResult> NatReady(
        [FromBody] RelayNatNotification notification,
        [FromHeader(Name = "X-Relay-Token")] string? token)
    {
        _logger.LogInformation(
            "Received NAT configuration callback from relay VM {VmId} with IP {VmIp}",
            notification.VmId, notification.VmIp);

        // =====================================================
        // STEP 1: Validate request
        // =====================================================
        if (string.IsNullOrEmpty(notification.VmId) ||
            string.IsNullOrEmpty(notification.VmIp))
        {
            _logger.LogWarning("Invalid NAT callback: missing VmId or VmIp");
            return BadRequest(new { error = "Missing required fields" });
        }

        // =====================================================
        // STEP 2: Verify authentication token
        // =====================================================
        var expectedToken = ComputeCallbackToken(notification.VmId, notification.VmIp);

        if (string.IsNullOrEmpty(token))
        {
            _logger.LogWarning(
                "NAT callback rejected: Missing X-Relay-Token header from VM {VmId}",
                notification.VmId);
            return Unauthorized(new { error = "Missing authentication token" });
        }

        // Constant-time comparison to prevent timing attacks
        if (!CryptographicOperations.FixedTimeEquals(
            Encoding.UTF8.GetBytes(token),
            Encoding.UTF8.GetBytes(expectedToken)))
        {
            _logger.LogWarning(
                "NAT callback rejected: Invalid token from VM {VmId}",
                notification.VmId);
            return Unauthorized(new { error = "Invalid authentication token" });
        }

        _logger.LogInformation(
            "✓ NAT callback authenticated successfully for relay VM {VmId}",
            notification.VmId);

        // =====================================================
        // STEP 3: Configure NAT port forwarding
        // =====================================================
        try
        {
            _logger.LogInformation(
                "Configuring NAT rule: UDP/51820 → {VmIp}:51820",
                notification.VmIp);

            var success = await _natRuleManager.AddPortForwardingAsync(
                notification.VmIp,
                51820,  // WireGuard port
                "udp",
                HttpContext.RequestAborted);

            if (success)
            {
                _logger.LogInformation(
                    "✓ Relay VM {VmId} NAT configured successfully via callback: " +
                    "Public UDP/51820 → {VmIp}:51820",
                    notification.VmId, notification.VmIp);

                _logger.LogInformation(
                    "✓ CGNAT nodes can now connect to this relay at the host's public IP");

                return Ok(new
                {
                    success = true,
                    message = "NAT rule configured successfully",
                    vmId = notification.VmId,
                    vmIp = notification.VmIp,
                    natRule = $"UDP/51820 → {notification.VmIp}:51820"
                });
            }
            else
            {
                _logger.LogError(
                    "❌ Failed to configure NAT rule for relay VM {VmId}",
                    notification.VmId);

                return StatusCode(500, new
                {
                    success = false,
                    error = "Failed to configure NAT rule",
                    vmId = notification.VmId
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Error configuring NAT for relay VM {VmId}",
                notification.VmId);

            return StatusCode(500, new
            {
                success = false,
                error = "Internal server error",
                message = ex.Message
            });
        }
    }

    /// <summary>
    /// Compute HMAC-SHA256 callback token for authentication
    /// Uses a deterministic secret based on the host's machine ID
    /// </summary>
    private string ComputeCallbackToken(string vmId, string vmIp)
    {
        // Message: vmId:vmIp
        var message = $"{vmId}:{vmIp}";

        // Secret: Machine ID (consistent across reboots, unique per host)
        var secret = GetMachineId();

        // HMAC-SHA256
        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(secret));
        var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(message));

        return Convert.ToBase64String(hash);
    }

    /// <summary>
    /// Get machine ID for authentication secret
    /// </summary>
    private string GetMachineId()
    {
        try
        {
            // Linux: /etc/machine-id
            if (System.IO.File.Exists("/etc/machine-id"))
            {
                return System.IO.File.ReadAllText("/etc/machine-id").Trim();
            }

            // Fallback: hostname
            return Environment.MachineName;
        }
        catch
        {
            // Ultimate fallback
            return "decloud-default-secret";
        }
    }
}

/// <summary>
/// Notification sent by relay VM when it obtains an IP address
/// </summary>
public record RelayNatNotification(
    string VmId,
    string VmIp
);