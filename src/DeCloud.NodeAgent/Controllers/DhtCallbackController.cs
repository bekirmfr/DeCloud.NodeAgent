using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.AspNetCore.Mvc;
using System.Security.Cryptography;
using System.Text;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Handles callbacks from DHT VMs when the DHT binary starts and obtains a peer ID.
/// Mirrors the relay callback pattern (RelayNatCallbackController).
/// </summary>
[ApiController]
[Route("api/dht")]
public class DhtCallbackController : ControllerBase
{
    private readonly IVmManager _vmManager;
    private readonly VmRepository _repository;
    private readonly ILogger<DhtCallbackController> _logger;

    public DhtCallbackController(
        IVmManager vmManager,
        VmRepository repository,
        ILogger<DhtCallbackController> logger)
    {
        _vmManager = vmManager;
        _repository = repository;
        _logger = logger;
    }

    /// <summary>
    /// DHT VM calls this endpoint when its DHT binary starts and reports a peer ID.
    /// Updates the VM's System service to Ready so the orchestrator can mark the
    /// DHT obligation as Active and proceed with relay deployment.
    /// </summary>
    [HttpPost("ready")]
    public async Task<IActionResult> DhtReady(
        [FromBody] DhtReadyNotification notification,
        [FromHeader(Name = "X-DHT-Token")] string? token)
    {
        _logger.LogInformation(
            "Received DHT ready callback from VM {VmId} with peer ID {PeerId}",
            notification.VmId, notification.PeerId);

        // =====================================================
        // STEP 1: Validate request
        // =====================================================
        if (string.IsNullOrEmpty(notification.VmId) ||
            string.IsNullOrEmpty(notification.PeerId))
        {
            _logger.LogWarning("Invalid DHT callback: missing VmId or PeerId");
            return BadRequest(new { error = "Missing required fields" });
        }

        // =====================================================
        // STEP 2: Verify authentication token
        // =====================================================
        var expectedToken = ComputeCallbackToken(notification.VmId, notification.PeerId);

        if (string.IsNullOrEmpty(token))
        {
            _logger.LogWarning(
                "DHT callback rejected: Missing X-DHT-Token header from VM {VmId}",
                notification.VmId);
            return Unauthorized(new { error = "Missing authentication token" });
        }

        if (!CryptographicOperations.FixedTimeEquals(
            Encoding.UTF8.GetBytes(token),
            Encoding.UTF8.GetBytes(expectedToken)))
        {
            _logger.LogWarning(
                "DHT callback rejected: Invalid token from VM {VmId}",
                notification.VmId);
            return Unauthorized(new { error = "Invalid authentication token" });
        }

        _logger.LogInformation(
            "DHT callback authenticated for VM {VmId}", notification.VmId);

        // =====================================================
        // STEP 3: Find the VM and update service readiness
        // =====================================================
        try
        {
            var vm = await _vmManager.GetVmAsync(notification.VmId, HttpContext.RequestAborted);
            if (vm == null)
            {
                _logger.LogWarning("DHT callback: VM {VmId} not found", notification.VmId);
                return NotFound(new { error = "VM not found" });
            }

            var systemService = vm.Services.FirstOrDefault(s => s.Name == "System");
            if (systemService == null)
            {
                _logger.LogWarning("DHT callback: VM {VmId} has no System service", notification.VmId);
                return BadRequest(new { error = "VM has no System service" });
            }

            // Always update StatusMessage with peer ID, even if VmReadinessMonitor
            // already marked the service Ready via cloud-init check (race condition:
            // cloud-init "done" fires before DHT binary reports peerId, monitor marks
            // Ready with null StatusMessage, then this callback's old idempotency guard
            // would short-circuit and the peerId was never captured).
            var previousStatus = systemService.Status;
            var alreadyReady = systemService.Status == ServiceReadiness.Ready;

            systemService.Status = ServiceReadiness.Ready;
            systemService.StatusMessage = $"peerId={notification.PeerId}";
            systemService.LastCheckAt = DateTime.UtcNow;
            if (!alreadyReady)
                systemService.ReadyAt = DateTime.UtcNow;

            await _repository.SaveVmAsync(vm);

            _logger.LogInformation(
                "DHT VM {VmId} System service {Action} via callback " +
                "(was: {PreviousStatus}, peer ID: {PeerId})",
                notification.VmId,
                alreadyReady ? "updated peerId (was already Ready)" : "marked Ready",
                previousStatus, notification.PeerId);

            // Store peer ID in VM directory for heartbeat reporting
            await StorePeerIdAsync(notification.VmId, notification.PeerId);

            return Ok(new
            {
                success = true,
                message = alreadyReady
                    ? "DHT peer ID captured (service was already Ready)"
                    : "DHT node registered successfully",
                vmId = notification.VmId,
                peerId = notification.PeerId,
                alreadyReady
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Error processing DHT callback for VM {VmId}",
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
    /// Store peer ID to disk so it can be picked up by heartbeat reporting.
    /// </summary>
    private async Task StorePeerIdAsync(string vmId, string peerId)
    {
        try
        {
            var vmDir = Path.Combine("/var/lib/decloud/vms", vmId);
            if (Directory.Exists(vmDir))
            {
                var peerIdPath = Path.Combine(vmDir, "dht-peer-id");
                await System.IO.File.WriteAllTextAsync(peerIdPath, peerId);
                _logger.LogDebug("Stored DHT peer ID at {Path}", peerIdPath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to store DHT peer ID to disk for VM {VmId}", vmId);
        }
    }

    /// <summary>
    /// Compute HMAC-SHA256 callback token for authentication.
    /// Uses machine ID as secret (same pattern as relay callback).
    /// </summary>
    private string ComputeCallbackToken(string vmId, string peerId)
    {
        var message = $"{vmId}:{peerId}";
        var secret = GetMachineId();

        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(secret));
        var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(message));
        return Convert.ToBase64String(hash);
    }

    private string GetMachineId()
    {
        try
        {
            if (System.IO.File.Exists("/etc/machine-id"))
            {
                return System.IO.File.ReadAllText("/etc/machine-id").Trim();
            }
            return Environment.MachineName;
        }
        catch
        {
            return "decloud-default-secret";
        }
    }
}

public record DhtReadyNotification(
    string VmId,
    string PeerId
);
