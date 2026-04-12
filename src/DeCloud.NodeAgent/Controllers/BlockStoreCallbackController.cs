using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.AspNetCore.Mvc;
using System.Security.Cryptography;
using System.Text;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Handles callbacks from block store VMs when the blockstore-node binary
/// starts and obtains a libp2p peer ID.
/// Mirrors DhtCallbackController — same authentication pattern, same service update.
///
/// Authentication: HMAC-SHA256(machineId, vmId:peerId) via X-BlockStore-Token header.
/// This is the NodeAgent-side token (machineId as secret), distinct from the
/// orchestrator-side token (authToken as secret) used in POST /api/blockstore/join.
/// </summary>
[ApiController]
[Route("api/blockstore")]
public class BlockStoreCallbackController : ControllerBase
{
    private readonly IVmManager _vmManager;
    private readonly VmRepository _repository;
    private readonly ILogger<BlockStoreCallbackController> _logger;

    public BlockStoreCallbackController(
        IVmManager vmManager,
        VmRepository repository,
        ILogger<BlockStoreCallbackController> logger)
    {
        _vmManager = vmManager;
        _repository = repository;
        _logger = logger;
    }

    /// <summary>
    /// Block store VM calls this on startup when the blockstore-node binary reports
    /// its libp2p peer ID. Updates the VM's System service to Ready so the
    /// orchestrator can mark the BlockStore obligation as Active.
    /// </summary>
    [HttpPost("ready")]
    public async Task<IActionResult> BlockStoreReady(
        [FromBody] BlockStoreReadyNotification notification,
        [FromHeader(Name = "X-BlockStore-Token")] string? token)
    {
        _logger.LogInformation(
            "Received block store ready callback from VM {VmId} with peer ID {PeerId}",
            notification.VmId, notification.PeerId);

        // ── Validate request ──────────────────────────────────────────
        if (string.IsNullOrEmpty(notification.VmId) ||
            string.IsNullOrEmpty(notification.PeerId))
        {
            _logger.LogWarning("Invalid block store callback: missing VmId or PeerId");
            return BadRequest(new { error = "Missing required fields" });
        }

        // ── Verify authentication token ───────────────────────────────
        var expectedToken = ComputeCallbackToken(notification.VmId, notification.PeerId);

        if (string.IsNullOrEmpty(token))
        {
            _logger.LogWarning(
                "Block store callback rejected: missing X-BlockStore-Token from VM {VmId}",
                notification.VmId);
            return Unauthorized(new { error = "Missing authentication token" });
        }

        if (!CryptographicOperations.FixedTimeEquals(
            Encoding.UTF8.GetBytes(token),
            Encoding.UTF8.GetBytes(expectedToken)))
        {
            _logger.LogWarning(
                "Block store callback rejected: invalid token from VM {VmId}",
                notification.VmId);
            return Unauthorized(new { error = "Invalid authentication token" });
        }

        _logger.LogInformation(
            "Block store callback authenticated for VM {VmId}", notification.VmId);

        // ── Find VM and update service readiness ──────────────────────
        try
        {
            var vm = await _vmManager.GetVmAsync(notification.VmId, HttpContext.RequestAborted);
            if (vm == null)
            {
                _logger.LogWarning("Block store callback: VM {VmId} not found", notification.VmId);
                return NotFound(new { error = "VM not found" });
            }

            var systemService = vm.Services.FirstOrDefault(s => s.Name == "System");
            if (systemService == null)
            {
                _logger.LogWarning(
                    "Block store callback: VM {VmId} has no System service", notification.VmId);
                return BadRequest(new { error = "VM has no System service" });
            }

            // Always update StatusMessage with peer ID — even if already Ready
            // (handles race where cloud-init readiness fires before peerId is known)
            var previousStatus = systemService.Status;
            var alreadyReady = systemService.Status == ServiceReadiness.Ready;

            systemService.Status = ServiceReadiness.Ready;
            systemService.StatusMessage = $"peerId={notification.PeerId}";
            systemService.LastCheckAt = DateTime.UtcNow;
            if (!alreadyReady)
                systemService.ReadyAt = DateTime.UtcNow;

            await _repository.SaveVmAsync(vm);

            _logger.LogInformation(
                "Block store VM {VmId} System service {Action} via callback " +
                "(was: {PreviousStatus}, peer ID: {PeerId})",
                notification.VmId,
                alreadyReady ? "updated peerId (was already Ready)" : "marked Ready",
                previousStatus, notification.PeerId);

            // Persist peer ID to disk for heartbeat reporting
            await StorePeerIdAsync(notification.VmId, notification.PeerId);

            return Ok(new
            {
                success = true,
                message = alreadyReady
                    ? "Block store peer ID captured (service was already Ready)"
                    : "Block store node registered successfully",
                vmId = notification.VmId,
                peerId = notification.PeerId,
                alreadyReady,
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Error processing block store callback for VM {VmId}",
                notification.VmId);

            return StatusCode(500, new
            {
                success = false,
                error = "Internal server error",
                message = ex.Message,
            });
        }
    }

    /// <summary>
    /// Called by the orchestrator's LazysyncManager after each ConfirmedVersion
    /// advance to push the confirmed CID list to the local BlockStore VM binary.
    ///
    /// The binary writes the CIDs to confirmed/{vmId}.cids. During GC, it evicts
    /// these confirmed-remote blocks before falling back to LRU, freeing the local
    /// blockstore's 5% duty for blocks still needing scatter.
    ///
    /// No auth required — endpoint is on port 5100 (agent port), only accessible
    /// to the orchestrator. Data (CID list) is not sensitive.
    /// </summary>
    [HttpPost("confirmed")]
    public async Task<IActionResult> PushConfirmedBlocks(
        [FromBody] ConfirmedBlocksNotification notification)
    {
        if (string.IsNullOrEmpty(notification.VmId) || notification.Cids == null)
            return BadRequest(new { error = "Missing vmId or cids" });

        // Find the local BlockStore VM
        var allVms = _vmManager.GetAllVms();
        var blockstoreVm = allVms.FirstOrDefault(v =>
            v.Spec.VmType == VmType.BlockStore &&
            v.State == VmState.Running);

        if (blockstoreVm == null || string.IsNullOrEmpty(blockstoreVm.Spec.IpAddress))
        {
            _logger.LogDebug(
                "PushConfirmedBlocks: no running BlockStore VM — skipping for VM {VmId}",
                notification.VmId);
            return Ok(new { success = true, skipped = true });
        }

        var blockstoreUrl = $"http://{blockstoreVm.Spec.IpAddress}:5090";

        try
        {
            var payload = System.Text.Json.JsonSerializer.Serialize(new
            {
                cids = notification.Cids
            });

            using var content = new StringContent(
                payload, System.Text.Encoding.UTF8, "application/json");

            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
            var response = await http.PostAsync(
                $"{blockstoreUrl}/confirmed/{Uri.EscapeDataString(notification.VmId)}",
                content,
                HttpContext.RequestAborted);

            if (response.IsSuccessStatusCode)
            {
                _logger.LogDebug(
                    "Forwarded {Count} confirmed CIDs to blockstore for VM {VmId}",
                    notification.Cids.Count, notification.VmId);
                return Ok(new { success = true, count = notification.Cids.Count });
            }

            _logger.LogWarning(
                "Blockstore returned {Status} for confirmed CIDs (VM {VmId})",
                response.StatusCode, notification.VmId);
            return Ok(new { success = false, status = (int)response.StatusCode });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Failed to forward confirmed CIDs to blockstore for VM {VmId}",
                notification.VmId);
            return Ok(new { success = false, error = ex.Message });
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Compute HMAC-SHA256 callback token.
    /// Secret: machine ID (same pattern as DhtCallbackController).
    /// Message: vmId:peerId
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
                return System.IO.File.ReadAllText("/etc/machine-id").Trim();
            return Environment.MachineName;
        }
        catch
        {
            return "decloud-default-secret";
        }
    }

    /// <summary>
    /// Store peer ID to disk for heartbeat reporting.
    /// Path mirrors the DHT pattern: /var/lib/decloud/vms/{vmId}/blockstore-peer-id
    /// </summary>
    private async Task StorePeerIdAsync(string vmId, string peerId)
    {
        try
        {
            var vmDir = Path.Combine("/var/lib/decloud/vms", vmId);
            if (Directory.Exists(vmDir))
            {
                var peerIdPath = Path.Combine(vmDir, "blockstore-peer-id");
                await System.IO.File.WriteAllTextAsync(peerIdPath, peerId);
                _logger.LogDebug("Stored block store peer ID at {Path}", peerIdPath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Failed to store block store peer ID to disk for VM {VmId}", vmId);
        }
    }
}

public record BlockStoreReadyNotification(
    string VmId,
    string PeerId
);

public record ConfirmedBlocksNotification(
    string VmId,
    List<string> Cids
);
