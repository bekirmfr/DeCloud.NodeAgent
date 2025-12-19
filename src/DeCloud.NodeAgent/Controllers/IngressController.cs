using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Services;
using Microsoft.AspNetCore.Mvc;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// API controller for managing ingress rules (domain routing to VMs).
/// Provides CRUD operations for HTTP/HTTPS reverse proxy configuration.
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class IngressController : ControllerBase
{
    private readonly IIngressService _ingressService;
    private readonly ICaddyManager _caddyManager;
    private readonly ILogger<IngressController> _logger;

    public IngressController(
        IIngressService ingressService,
        ICaddyManager caddyManager,
        ILogger<IngressController> logger)
    {
        _ingressService = ingressService;
        _caddyManager = caddyManager;
        _logger = logger;
    }

    /// <summary>
    /// Get all ingress rules
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<List<IngressResponse>>> GetAll(CancellationToken ct)
    {
        var rules = await _ingressService.GetAllAsync(ct);
        var responses = rules.Select(ToResponse).ToList();
        return Ok(responses);
    }

    /// <summary>
    /// Get ingress rules for a specific VM
    /// </summary>
    [HttpGet("vm/{vmId}")]
    public async Task<ActionResult<List<IngressResponse>>> GetByVmId(string vmId, CancellationToken ct)
    {
        var rules = await _ingressService.GetByVmIdAsync(vmId, ct);
        var responses = rules.Select(ToResponse).ToList();
        return Ok(responses);
    }

    /// <summary>
    /// Get a specific ingress rule
    /// </summary>
    [HttpGet("{ingressId}")]
    public async Task<ActionResult<IngressResponse>> GetById(string ingressId, CancellationToken ct)
    {
        var rule = await _ingressService.GetByIdAsync(ingressId, ct);
        if (rule == null)
        {
            return NotFound(new { error = "Ingress rule not found" });
        }

        return Ok(ToResponse(rule));
    }

    /// <summary>
    /// Create a new ingress rule to route a domain to a VM
    /// </summary>
    /// <remarks>
    /// Creates a reverse proxy rule that routes traffic from the specified domain
    /// to the target VM and port. Optionally enables automatic TLS via Let's Encrypt.
    /// 
    /// The user must configure their DNS to point the domain to this node's public IP.
    /// 
    /// Example:
    /// ```json
    /// {
    ///   "vmId": "abc123",
    ///   "domain": "myapp.example.com",
    ///   "targetPort": 3000,
    ///   "enableTls": true
    /// }
    /// ```
    /// </remarks>
    [HttpPost]
    public async Task<ActionResult<IngressOperationResult>> Create(
        [FromBody] CreateIngressRequest request,
        [FromHeader(Name = "X-Owner-Wallet")] string? ownerWallet,
        CancellationToken ct)
    {
        if (string.IsNullOrEmpty(ownerWallet))
        {
            return BadRequest(IngressOperationResult.Fail("X-Owner-Wallet header is required"));
        }

        _logger.LogInformation(
            "Creating ingress: {Domain} → VM {VmId}:{Port} (TLS: {Tls})",
            request.Domain, request.VmId, request.TargetPort, request.EnableTls);

        var result = await _ingressService.CreateAsync(request, ownerWallet, ct);

        if (!result.Success)
        {
            _logger.LogWarning("Failed to create ingress for {Domain}: {Error}", request.Domain, result.Error);
            return BadRequest(result);
        }

        return CreatedAtAction(nameof(GetById), new { ingressId = result.IngressId }, result);
    }

    /// <summary>
    /// Update an existing ingress rule
    /// </summary>
    [HttpPatch("{ingressId}")]
    public async Task<ActionResult<IngressOperationResult>> Update(
        string ingressId,
        [FromBody] UpdateIngressRequest request,
        [FromHeader(Name = "X-Owner-Wallet")] string? ownerWallet,
        CancellationToken ct)
    {
        if (string.IsNullOrEmpty(ownerWallet))
        {
            return BadRequest(IngressOperationResult.Fail("X-Owner-Wallet header is required"));
        }

        _logger.LogInformation("Updating ingress {IngressId}", ingressId);

        var result = await _ingressService.UpdateAsync(ingressId, request, ownerWallet, ct);

        if (!result.Success)
        {
            if (result.Error?.Contains("Not authorized") == true)
            {
                return Forbid();
            }
            if (result.Error?.Contains("not found") == true)
            {
                return NotFound(result);
            }
            return BadRequest(result);
        }

        return Ok(result);
    }

    /// <summary>
    /// Delete an ingress rule
    /// </summary>
    [HttpDelete("{ingressId}")]
    public async Task<ActionResult<IngressOperationResult>> Delete(
        string ingressId,
        [FromHeader(Name = "X-Owner-Wallet")] string? ownerWallet,
        CancellationToken ct)
    {
        if (string.IsNullOrEmpty(ownerWallet))
        {
            return BadRequest(IngressOperationResult.Fail("X-Owner-Wallet header is required"));
        }

        _logger.LogInformation("Deleting ingress {IngressId}", ingressId);

        var result = await _ingressService.DeleteAsync(ingressId, ownerWallet, ct);

        if (!result.Success)
        {
            if (result.Error?.Contains("Not authorized") == true)
            {
                return Forbid();
            }
            if (result.Error?.Contains("not found") == true)
            {
                return NotFound(result);
            }
            return BadRequest(result);
        }

        return Ok(result);
    }

    /// <summary>
    /// Pause an ingress rule (stop routing traffic)
    /// </summary>
    [HttpPost("{ingressId}/pause")]
    public async Task<ActionResult<IngressOperationResult>> Pause(
        string ingressId,
        [FromHeader(Name = "X-Owner-Wallet")] string? ownerWallet,
        CancellationToken ct)
    {
        if (string.IsNullOrEmpty(ownerWallet))
        {
            return BadRequest(IngressOperationResult.Fail("X-Owner-Wallet header is required"));
        }

        var result = await _ingressService.PauseAsync(ingressId, ownerWallet, ct);

        if (!result.Success)
        {
            if (result.Error?.Contains("Not authorized") == true)
            {
                return Forbid();
            }
            return BadRequest(result);
        }

        return Ok(result);
    }

    /// <summary>
    /// Resume a paused ingress rule
    /// </summary>
    [HttpPost("{ingressId}/resume")]
    public async Task<ActionResult<IngressOperationResult>> Resume(
        string ingressId,
        [FromHeader(Name = "X-Owner-Wallet")] string? ownerWallet,
        CancellationToken ct)
    {
        if (string.IsNullOrEmpty(ownerWallet))
        {
            return BadRequest(IngressOperationResult.Fail("X-Owner-Wallet header is required"));
        }

        var result = await _ingressService.ResumeAsync(ingressId, ownerWallet, ct);

        if (!result.Success)
        {
            if (result.Error?.Contains("Not authorized") == true)
            {
                return Forbid();
            }
            return BadRequest(result);
        }

        return Ok(result);
    }

    /// <summary>
    /// Force reload all ingress rules
    /// </summary>
    [HttpPost("reload")]
    public async Task<ActionResult> Reload(CancellationToken ct)
    {
        _logger.LogInformation("Manual reload of all ingress rules requested");

        var success = await _ingressService.ReloadAllAsync(ct);

        if (!success)
        {
            return StatusCode(500, new { error = "Failed to reload ingress configuration" });
        }

        return Ok(new { message = "Ingress configuration reloaded successfully" });
    }

    /// <summary>
    /// Get Caddy health status
    /// </summary>
    [HttpGet("health")]
    public async Task<ActionResult> Health(CancellationToken ct)
    {
        var healthy = await _caddyManager.IsHealthyAsync(ct);

        if (!healthy)
        {
            return StatusCode(503, new
            {
                status = "unhealthy",
                message = "Caddy reverse proxy is not responding"
            });
        }

        var rules = await _ingressService.GetAllAsync(ct);
        var activeCount = rules.Count(r => r.Status == IngressStatus.Active);
        var errorCount = rules.Count(r => r.Status == IngressStatus.Error);

        return Ok(new
        {
            status = "healthy",
            caddy = "running",
            totalRules = rules.Count,
            activeRules = activeCount,
            errorRules = errorCount
        });
    }

    /// <summary>
    /// Validate a domain name without creating an ingress rule
    /// </summary>
    [HttpPost("validate-domain")]
    public ActionResult ValidateDomain([FromBody] ValidateDomainRequest request)
    {
        if (string.IsNullOrEmpty(request.Domain))
        {
            return BadRequest(new { valid = false, error = "Domain is required" });
        }

        var isValid = _ingressService.ValidateDomain(request.Domain, out var error);

        return Ok(new
        {
            valid = isValid,
            domain = request.Domain.ToLowerInvariant().Trim(),
            error = error
        });
    }

    /// <summary>
    /// Get DNS instructions for setting up a domain
    /// </summary>
    [HttpGet("dns-instructions")]
    public ActionResult GetDnsInstructions([FromQuery] string nodePublicIp)
    {
        if (string.IsNullOrEmpty(nodePublicIp))
        {
            return BadRequest(new { error = "nodePublicIp query parameter is required" });
        }

        return Ok(new
        {
            instructions = new[]
            {
                $"1. Log in to your domain registrar or DNS provider",
                $"2. Create an A record pointing to: {nodePublicIp}",
                $"3. Example: myapp.example.com → A → {nodePublicIp}",
                $"4. Wait for DNS propagation (may take up to 48 hours, usually faster)",
                $"5. Create the ingress rule using the API",
                $"6. TLS certificates will be automatically provisioned via Let's Encrypt"
            },
            example = new
            {
                type = "A",
                name = "myapp",
                value = nodePublicIp,
                ttl = 3600
            },
            note = "For subdomains like api.myapp.example.com, create the A record for 'api.myapp'"
        });
    }

    private static IngressResponse ToResponse(IngressRule rule) => new(
        Id: rule.Id,
        VmId: rule.VmId,
        Domain: rule.Domain,
        TargetPort: rule.TargetPort,
        EnableTls: rule.EnableTls,
        Status: rule.Status,
        StatusMessage: rule.StatusMessage,
        TlsStatus: rule.TlsStatus,
        TlsExpiresAt: rule.TlsExpiresAt,
        PublicUrl: rule.EnableTls ? $"https://{rule.Domain}" : $"http://{rule.Domain}",
        CreatedAt: rule.CreatedAt,
        UpdatedAt: rule.UpdatedAt,
        TotalRequests: rule.TotalRequests
    );
}

public record ValidateDomainRequest(string Domain);