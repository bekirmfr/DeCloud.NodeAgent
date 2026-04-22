using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared.Models;
using Microsoft.AspNetCore.Mvc;
using System.Net;

namespace DeCloud.NodeAgent.Controllers;

// ============================================================
// Placement: src/DeCloud.NodeAgent/Controllers/ObligationStateController.cs
// ============================================================

/// <summary>
/// Serves obligation identity state to system VMs over the virbr0 bridge
/// (192.168.122.x — the libvirt internal network).
///
/// SECURITY MODEL
/// ──────────────
/// Primary boundary:  virbr0 is not routable externally; only VMs on this
///                    node's libvirt network can reach 192.168.122.1:5100.
///
/// Defense-in-depth:  Every request is validated against the virbr0 subnet
///                    (192.168.122.0/24) at the controller level. If a
///                    misconfigured firewall rule temporarily opens port 5100
///                    externally, the controller still refuses to serve keys.
///
/// No authentication token is required — the network boundary is the auth.
///
/// LOGGING DISCIPLINE
/// ──────────────────
/// State JSON is never written to logs. Only role name, version, and the
/// caller's IP address are logged, and only at DEBUG level on success.
/// A 404 or a blocked-IP attempt is logged at WARNING.
///
/// ROUTES
/// ──────
///   GET /api/obligations/{role}/state
///       Returns identity state JSON for the role.
///       404 if no state has been persisted yet (first boot).
///
///   GET /api/obligations/{role}/version
///       Returns the current version integer.
///       Lightweight check — no key material in the response.
///       Returns 0 if no state exists.
/// </summary>
[ApiController]
[Route("api/obligations")]
public class ObligationStateController : ControllerBase
{
    // virbr0 bridge network — only addresses in this subnet may call this controller.
    private static readonly IPNetwork VirbR0Network =
        IPNetwork.Parse("192.168.122.0/24");

    private readonly IObligationStateService _obligationState;
    private readonly ILogger<ObligationStateController> _logger;

    public ObligationStateController(
        IObligationStateService obligationState,
        ILogger<ObligationStateController> logger)
    {
        _obligationState = obligationState;
        _logger = logger;
    }

    // ----------------------------------------------------------------
    // GET /api/obligations/{role}/state
    // ----------------------------------------------------------------

    /// <summary>
    /// Returns the raw identity state JSON for <paramref name="role"/>.
    /// Called by system VMs at cloud-init time to restore persistent identity
    /// before starting their services.
    /// </summary>
    /// <param name="role">Role name: relay | dht | blockstore (case-insensitive).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <response code="200">State JSON returned successfully.</response>
    /// <response code="400">Unknown role name.</response>
    /// <response code="403">Caller is not on the virbr0 subnet.</response>
    /// <response code="404">No state persisted for this role yet.</response>
    [HttpGet("{role}/state")]
    [ProducesResponseType(typeof(string), 200)]
    [ProducesResponseType(400)]
    [ProducesResponseType(403)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> GetState(string role, CancellationToken ct)
    {
        if (!EnforceVirbr0(out var callerIp))
            return StatusCode(403, "Forbidden: obligation state is only accessible from the VM bridge network.");

        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
        {
            _logger.LogWarning(
                "GetState: unknown role '{Role}' from {Ip}", role, callerIp);
            return BadRequest($"Unknown role '{role}'. Valid values: relay, dht, blockstore.");
        }

        var stateJson = await _obligationState.GetStateJsonAsync(canonical, ct);

        if (stateJson is null)
        {
            _logger.LogDebug(
                "GetState [{Role}]: no state found (first boot?) — returning 404. Caller: {Ip}",
                canonical, callerIp);
            return NotFound($"No obligation state found for role '{canonical}'.");
        }

        // Log access at DEBUG only — never log the state content.
        var version = await _obligationState.GetVersionAsync(canonical, ct);
        _logger.LogDebug(
            "GetState [{Role}] v{Version} served to {Ip}",
            canonical, version, callerIp);

        // Return raw JSON with the correct content-type so the VM can parse it directly.
        return Content(stateJson, "application/json");
    }

    // ----------------------------------------------------------------
    // GET /api/obligations/{role}/version
    // ----------------------------------------------------------------

    /// <summary>
    /// Returns the current stored version for <paramref name="role"/> as a plain integer.
    /// Intended for lightweight checks — no key material is included in the response.
    /// </summary>
    /// <param name="role">Role name: relay | dht | blockstore (case-insensitive).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <response code="200">Version integer (0 if no state persisted).</response>
    /// <response code="400">Unknown role name.</response>
    /// <response code="403">Caller is not on the virbr0 subnet.</response>
    [HttpGet("{role}/version")]
    [ProducesResponseType(typeof(int), 200)]
    [ProducesResponseType(400)]
    [ProducesResponseType(403)]
    public async Task<IActionResult> GetVersion(string role, CancellationToken ct)
    {
        if (!EnforceVirbr0(out var callerIp))
            return StatusCode(403, "Forbidden: obligation state is only accessible from the VM bridge network.");

        var canonical = ObligationRole.Canonicalise(role);
        if (canonical is null)
        {
            _logger.LogWarning(
                "GetVersion: unknown role '{Role}' from {Ip}", role, callerIp);
            return BadRequest($"Unknown role '{role}'. Valid values: relay, dht, blockstore.");
        }

        var version = await _obligationState.GetVersionAsync(canonical, ct);

        _logger.LogDebug(
            "GetVersion [{Role}] → v{Version} served to {Ip}",
            canonical, version, callerIp);

        return Ok(version);
    }

    // ----------------------------------------------------------------
    // Private helpers
    // ----------------------------------------------------------------

    /// <summary>
    /// Validates that the caller's remote IP is within the virbr0 subnet
    /// (192.168.122.0/24). Returns <c>true</c> if the request may proceed.
    /// </summary>
    /// <param name="callerIp">The caller's IP address string (for logging).</param>
    private bool EnforceVirbr0(out string callerIp)
    {
        var remote = HttpContext.Connection.RemoteIpAddress;

        // Unwrap IPv4-mapped-in-IPv6 (::ffff:192.168.122.x)
        if (remote is not null && remote.IsIPv4MappedToIPv6)
            remote = remote.MapToIPv4();

        callerIp = remote?.ToString() ?? "unknown";

        if (remote is null || !VirbR0Network.Contains(remote))
        {
            _logger.LogWarning(
                "ObligationStateController: blocked request from {Ip} — not on virbr0 subnet",
                callerIp);
            return false;
        }

        return true;
    }
}
