using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.AspNetCore.Mvc;
using System.Net;
using System.Net.NetworkInformation;

namespace DeCloud.NodeAgent.Controllers;

// ============================================================
// Placement: src/DeCloud.NodeAgent/Controllers/ArtifactCacheController.cs
// ============================================================

/// <summary>
/// Serves cached artifacts to VMs over the libvirt bridge network (virbr0).
///
/// SECURITY: Only accessible from the virbr0 subnet (192.168.122.0/24).
/// All requests from other sources are rejected with 403. This mirrors the
/// access restriction on <c>ObligationStateController</c> — both endpoints
/// expose data that must not leave the local VM bridge.
///
/// USAGE: Cloud-init scripts inside VMs call this endpoint to download
/// template artifacts (binaries, scripts, config files). The URL is
/// pre-substituted by the orchestrator's <c>ResolveArtifactVariables</c>
/// method into the cloud-init template before deployment:
///
/// <code>
///   ${ARTIFACT_URL:dht-node} → http://192.168.122.1:5100/api/artifacts/{sha256}
/// </code>
///
/// NO DOWNLOAD ON MISS: If the artifact is not in the local cache the
/// controller returns 404. The artifact must be prefetched before the VM
/// is deployed (in P9 this happens when the system template arrives;
/// for tenant VMs it happens when the deploy command is processed).
/// This keeps VM boot time predictable — it does not depend on external
/// network availability at the moment the VM runs its cloud-init.
/// </summary>
[ApiController]
[Route("api/artifacts")]
public class ArtifactCacheController : ControllerBase
{
    // virbr0 default network (libvirt NAT bridge)
    private static readonly IPNetwork VirbR0Network = IPNetwork.Parse("192.168.122.0/24");

    private readonly IArtifactCacheService _cache;
    private readonly ILogger<ArtifactCacheController> _logger;

    public ArtifactCacheController(
        IArtifactCacheService cache,
        ILogger<ArtifactCacheController> logger)
    {
        _cache  = cache;
        _logger = logger;
    }

    // ── GET /api/artifacts/{sha256} ──────────────────────────────────────

    /// <summary>
    /// Stream a cached artifact to a VM.
    /// </summary>
    /// <param name="sha256">
    /// Lower-case SHA256 hex digest (64 chars). Must match the digest the
    /// orchestrator recorded at template publish time — the node agent
    /// verified it on download, and this endpoint serves only what passed
    /// verification.
    /// </param>
    /// <param name="ct">Cancellation token.</param>
    /// <response code="200">Artifact bytes. Content-Type: application/octet-stream.</response>
    /// <response code="400">Invalid SHA256 format.</response>
    /// <response code="403">Caller is not on the virbr0 subnet.</response>
    /// <response code="404">Artifact not in local cache (not yet prefetched).</response>
    [HttpGet("{sha256}")]
    [ProducesResponseType(200)]
    [ProducesResponseType(400)]
    [ProducesResponseType(403)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> GetArtifact(string sha256, CancellationToken ct)
    {
        if (!EnforceVirbr0(out var callerIp))
        {
            _logger.LogWarning(
                "ArtifactCacheController: rejected request for {Sha256} from {Ip} — not on virbr0",
                sha256.Length > 12 ? sha256[..12] : sha256, callerIp);
            return StatusCode(403, "Artifacts are only accessible from the VM bridge network (virbr0).");
        }

        if (!IsValidSha256(sha256))
        {
            _logger.LogWarning(
                "ArtifactCacheController: invalid SHA256 format '{Sha256}' from {Ip}",
                sha256, callerIp);
            return BadRequest("Invalid SHA256 format. Must be 64 lowercase hexadecimal characters.");
        }

        var localPath = await _cache.GetLocalPathAsync(sha256.ToLowerInvariant(), ct);

        if (localPath is null)
        {
            _logger.LogWarning(
                "ArtifactCacheController: {Sha256} not in cache (requested by {Ip}) — prefetch may have failed",
                sha256[..12], callerIp);
            return NotFound(
                $"Artifact {sha256[..12]}... is not in the local cache. " +
                "The node agent should have prefetched it when the system template arrived. " +
                "Check ArtifactCacheService logs for prefetch failures.");
        }

        _logger.LogDebug(
            "ArtifactCacheController: serving {Sha256} to {Ip}",
            sha256[..12], callerIp);

        // Stream directly from the cached file. FileStreamResult handles range
        // requests and keeps memory usage flat regardless of artifact size.
        var stream = new FileStream(
            localPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 81920,
            useAsync: true);

        return File(stream, "application/octet-stream");
    }

    // ── virbr0 enforcement ───────────────────────────────────────────────

    /// <summary>
    /// Returns true if the caller's remote IP is on the virbr0 subnet
    /// (192.168.122.0/24). Out-param is the caller's IP for logging.
    /// </summary>
    private bool EnforceVirbr0(out string callerIp)
    {
        var remoteIp = HttpContext.Connection.RemoteIpAddress;

        if (remoteIp is null)
        {
            callerIp = "unknown";
            return false;
        }

        // Normalise IPv4-mapped IPv6 (::ffff:192.168.122.x → 192.168.122.x)
        if (remoteIp.IsIPv4MappedToIPv6)
            remoteIp = remoteIp.MapToIPv4();

        callerIp = remoteIp.ToString();
        return VirbR0Network.Contains(remoteIp);
    }

    // ── Validation ───────────────────────────────────────────────────────

    private static bool IsValidSha256(string sha256) =>
        sha256.Length == 64 &&
        sha256.All(c =>
            (c >= '0' && c <= '9') ||
            (c >= 'a' && c <= 'f') ||
            (c >= 'A' && c <= 'F'));
}
