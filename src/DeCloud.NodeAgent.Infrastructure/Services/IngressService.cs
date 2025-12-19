using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Interface for ingress management service
/// </summary>
public interface IIngressService
{
    /// <summary>
    /// Create a new ingress rule
    /// </summary>
    Task<IngressOperationResult> CreateAsync(CreateIngressRequest request, string ownerWallet, CancellationToken ct = default);

    /// <summary>
    /// Update an existing ingress rule
    /// </summary>
    Task<IngressOperationResult> UpdateAsync(string ingressId, UpdateIngressRequest request, string ownerWallet, CancellationToken ct = default);

    /// <summary>
    /// Delete an ingress rule
    /// </summary>
    Task<IngressOperationResult> DeleteAsync(string ingressId, string ownerWallet, CancellationToken ct = default);

    /// <summary>
    /// Get an ingress rule by ID
    /// </summary>
    Task<IngressRule?> GetByIdAsync(string ingressId, CancellationToken ct = default);

    /// <summary>
    /// Get all ingress rules for a VM
    /// </summary>
    Task<List<IngressRule>> GetByVmIdAsync(string vmId, CancellationToken ct = default);

    /// <summary>
    /// Get all ingress rules
    /// </summary>
    Task<List<IngressRule>> GetAllAsync(CancellationToken ct = default);

    /// <summary>
    /// Pause an ingress rule (stop routing)
    /// </summary>
    Task<IngressOperationResult> PauseAsync(string ingressId, string ownerWallet, CancellationToken ct = default);

    /// <summary>
    /// Resume a paused ingress rule
    /// </summary>
    Task<IngressOperationResult> ResumeAsync(string ingressId, string ownerWallet, CancellationToken ct = default);

    /// <summary>
    /// Force reload all active ingress rules
    /// </summary>
    Task<bool> ReloadAllAsync(CancellationToken ct = default);

    /// <summary>
    /// Validate a domain name
    /// </summary>
    bool ValidateDomain(string domain, out string? error);
}

/// <summary>
/// Service for managing ingress rules and Caddy configuration.
/// Handles validation, persistence, and Caddy configuration updates.
/// </summary>
public class IngressService : IIngressService
{
    private readonly IngressRepository _repository;
    private readonly ICaddyManager _caddyManager;
    private readonly IVmManager _vmManager;
    private readonly ILogger<IngressService> _logger;
    private readonly SemaphoreSlim _reloadLock = new(1, 1);

    // Domain validation regex - RFC 1123
    private static readonly Regex DomainRegex = new(
        @"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,}$",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    // Reserved/blocked domains
    private static readonly HashSet<string> BlockedDomains = new(StringComparer.OrdinalIgnoreCase)
    {
        "localhost",
        "localhost.localdomain",
        "local",
        "example.com",
        "example.org",
        "example.net",
        "test",
        "invalid"
    };

    // Blocked TLDs
    private static readonly HashSet<string> BlockedTlds = new(StringComparer.OrdinalIgnoreCase)
    {
        "local",
        "localhost",
        "internal",
        "lan",
        "home",
        "corp",
        "mail"
    };

    public IngressService(
        IngressRepository repository,
        ICaddyManager caddyManager,
        IVmManager vmManager,
        ILogger<IngressService> logger)
    {
        _repository = repository;
        _caddyManager = caddyManager;
        _vmManager = vmManager;
        _logger = logger;
    }

    public async Task<IngressOperationResult> CreateAsync(
        CreateIngressRequest request,
        string ownerWallet,
        CancellationToken ct = default)
    {
        try
        {
            // Validate domain
            if (!ValidateDomain(request.Domain, out var domainError))
            {
                return IngressOperationResult.Fail(domainError!);
            }

            var normalizedDomain = request.Domain.ToLowerInvariant().Trim();

            // Check if domain already exists
            var existing = await _repository.GetByDomainAsync(normalizedDomain);
            if (existing != null)
            {
                return IngressOperationResult.Fail($"Domain '{normalizedDomain}' is already registered");
            }

            // Validate VM exists and belongs to owner
            var vm = await _vmManager.GetVmAsync(request.VmId, ct);
            if (vm == null)
            {
                return IngressOperationResult.Fail($"VM '{request.VmId}' not found");
            }

            // Get VM IP address
            var vmIp = vm.Spec.Network?.IpAddress;
            if (string.IsNullOrEmpty(vmIp))
            {
                return IngressOperationResult.Fail("VM does not have an IP address yet. Please wait for VM to start.");
            }

            // Validate port
            if (request.TargetPort < 1 || request.TargetPort > 65535)
            {
                return IngressOperationResult.Fail("Target port must be between 1 and 65535");
            }

            // Create the rule
            var rule = new IngressRule
            {
                VmId = request.VmId,
                OwnerWallet = ownerWallet,
                Domain = normalizedDomain,
                TargetPort = request.TargetPort,
                EnableTls = request.EnableTls,
                ForceHttps = request.ForceHttps,
                EnableWebSocket = request.EnableWebSocket,
                PathPrefix = request.PathPrefix ?? string.Empty,
                StripPathPrefix = request.StripPathPrefix,
                RateLimitPerMinute = request.RateLimitPerMinute,
                VmPrivateIp = vmIp,
                Status = IngressStatus.Configuring,
                TlsStatus = request.EnableTls ? TlsCertStatus.Pending : TlsCertStatus.Disabled
            };

            // Save to database
            await _repository.SaveAsync(rule);

            _logger.LogInformation(
                "Created ingress rule {Id}: {Domain} → {VmId}:{Port}",
                rule.Id, rule.Domain, rule.VmId, rule.TargetPort);

            // Reload Caddy configuration
            var reloadSuccess = await ReloadAllAsync(ct);

            if (reloadSuccess)
            {
                rule.Status = IngressStatus.Active;
                rule.TlsStatus = request.EnableTls ? TlsCertStatus.Provisioning : TlsCertStatus.Disabled;
                await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Active, "Route configured successfully");

                _logger.LogInformation("✓ Ingress {Domain} is now active", rule.Domain);
            }
            else
            {
                rule.Status = IngressStatus.Error;
                await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Error, "Failed to apply Caddy configuration");

                _logger.LogError("Failed to configure Caddy for {Domain}", rule.Domain);
            }

            return IngressOperationResult.Ok(rule);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating ingress rule for {Domain}", request.Domain);
            return IngressOperationResult.Fail($"Internal error: {ex.Message}");
        }
    }

    public async Task<IngressOperationResult> UpdateAsync(
        string ingressId,
        UpdateIngressRequest request,
        string ownerWallet,
        CancellationToken ct = default)
    {
        try
        {
            var rule = await _repository.GetByIdAsync(ingressId);
            if (rule == null)
            {
                return IngressOperationResult.Fail("Ingress rule not found");
            }

            // Authorization check
            if (!string.Equals(rule.OwnerWallet, ownerWallet, StringComparison.OrdinalIgnoreCase))
            {
                return IngressOperationResult.Fail("Not authorized to modify this ingress rule");
            }

            // Apply updates
            if (request.TargetPort.HasValue)
            {
                if (request.TargetPort.Value < 1 || request.TargetPort.Value > 65535)
                {
                    return IngressOperationResult.Fail("Target port must be between 1 and 65535");
                }
                rule.TargetPort = request.TargetPort.Value;
            }

            if (request.EnableTls.HasValue)
            {
                rule.EnableTls = request.EnableTls.Value;
                if (request.EnableTls.Value && rule.TlsStatus == TlsCertStatus.Disabled)
                {
                    rule.TlsStatus = TlsCertStatus.Pending;
                }
            }

            if (request.ForceHttps.HasValue)
                rule.ForceHttps = request.ForceHttps.Value;

            if (request.EnableWebSocket.HasValue)
                rule.EnableWebSocket = request.EnableWebSocket.Value;

            if (request.PathPrefix != null)
                rule.PathPrefix = request.PathPrefix;

            if (request.StripPathPrefix.HasValue)
                rule.StripPathPrefix = request.StripPathPrefix.Value;

            if (request.RateLimitPerMinute.HasValue)
                rule.RateLimitPerMinute = request.RateLimitPerMinute.Value;

            if (request.AllowedIps != null)
                rule.AllowedIps = request.AllowedIps;

            if (request.CustomHeaders != null)
                rule.CustomHeaders = request.CustomHeaders;

            rule.Status = IngressStatus.Configuring;
            await _repository.SaveAsync(rule);

            _logger.LogInformation("Updated ingress rule {Id}", ingressId);

            // Reload configuration
            var reloadSuccess = await ReloadAllAsync(ct);

            if (reloadSuccess)
            {
                rule.Status = IngressStatus.Active;
                await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Active, "Configuration updated");
            }
            else
            {
                rule.Status = IngressStatus.Error;
                await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Error, "Failed to apply configuration update");
            }

            return IngressOperationResult.Ok(rule);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating ingress rule {Id}", ingressId);
            return IngressOperationResult.Fail($"Internal error: {ex.Message}");
        }
    }

    public async Task<IngressOperationResult> DeleteAsync(
        string ingressId,
        string ownerWallet,
        CancellationToken ct = default)
    {
        try
        {
            var rule = await _repository.GetByIdAsync(ingressId);
            if (rule == null)
            {
                return IngressOperationResult.Fail("Ingress rule not found");
            }

            // Authorization check
            if (!string.Equals(rule.OwnerWallet, ownerWallet, StringComparison.OrdinalIgnoreCase))
            {
                return IngressOperationResult.Fail("Not authorized to delete this ingress rule");
            }

            rule.Status = IngressStatus.Deleting;
            await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Deleting);

            _logger.LogInformation("Deleting ingress rule {Id} for domain {Domain}", ingressId, rule.Domain);

            // Soft delete
            await _repository.DeleteAsync(ingressId);

            // Reload Caddy to remove the route
            await ReloadAllAsync(ct);

            _logger.LogInformation("✓ Ingress {Domain} deleted successfully", rule.Domain);

            return IngressOperationResult.Ok(rule);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting ingress rule {Id}", ingressId);
            return IngressOperationResult.Fail($"Internal error: {ex.Message}");
        }
    }

    public async Task<IngressRule?> GetByIdAsync(string ingressId, CancellationToken ct = default)
    {
        return await _repository.GetByIdAsync(ingressId);
    }

    public async Task<List<IngressRule>> GetByVmIdAsync(string vmId, CancellationToken ct = default)
    {
        return await _repository.GetByVmIdAsync(vmId);
    }

    public async Task<List<IngressRule>> GetAllAsync(CancellationToken ct = default)
    {
        return await _repository.GetAllAsync();
    }

    public async Task<IngressOperationResult> PauseAsync(
        string ingressId,
        string ownerWallet,
        CancellationToken ct = default)
    {
        try
        {
            var rule = await _repository.GetByIdAsync(ingressId);
            if (rule == null)
            {
                return IngressOperationResult.Fail("Ingress rule not found");
            }

            if (!string.Equals(rule.OwnerWallet, ownerWallet, StringComparison.OrdinalIgnoreCase))
            {
                return IngressOperationResult.Fail("Not authorized to modify this ingress rule");
            }

            if (rule.Status == IngressStatus.Paused)
            {
                return IngressOperationResult.Fail("Ingress rule is already paused");
            }

            rule.Status = IngressStatus.Paused;
            await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Paused, "Paused by user");

            await ReloadAllAsync(ct);

            _logger.LogInformation("Paused ingress rule {Id}", ingressId);

            return IngressOperationResult.Ok(rule);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error pausing ingress rule {Id}", ingressId);
            return IngressOperationResult.Fail($"Internal error: {ex.Message}");
        }
    }

    public async Task<IngressOperationResult> ResumeAsync(
        string ingressId,
        string ownerWallet,
        CancellationToken ct = default)
    {
        try
        {
            var rule = await _repository.GetByIdAsync(ingressId);
            if (rule == null)
            {
                return IngressOperationResult.Fail("Ingress rule not found");
            }

            if (!string.Equals(rule.OwnerWallet, ownerWallet, StringComparison.OrdinalIgnoreCase))
            {
                return IngressOperationResult.Fail("Not authorized to modify this ingress rule");
            }

            if (rule.Status != IngressStatus.Paused)
            {
                return IngressOperationResult.Fail("Ingress rule is not paused");
            }

            rule.Status = IngressStatus.Configuring;
            await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Configuring, "Resuming...");

            var reloadSuccess = await ReloadAllAsync(ct);

            if (reloadSuccess)
            {
                rule.Status = IngressStatus.Active;
                await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Active, "Resumed successfully");
            }
            else
            {
                rule.Status = IngressStatus.Error;
                await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Error, "Failed to resume");
            }

            _logger.LogInformation("Resumed ingress rule {Id}", ingressId);

            return IngressOperationResult.Ok(rule);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error resuming ingress rule {Id}", ingressId);
            return IngressOperationResult.Fail($"Internal error: {ex.Message}");
        }
    }

    public async Task<bool> ReloadAllAsync(CancellationToken ct = default)
    {
        await _reloadLock.WaitAsync(ct);
        try
        {
            // Check Caddy health first
            var healthy = await _caddyManager.IsHealthyAsync(ct);
            if (!healthy)
            {
                _logger.LogError("Caddy is not healthy - cannot reload configuration");
                return false;
            }

            // Get all active rules
            var activeRules = await _repository.GetAllActiveAsync();

            // Update VM IPs in case they changed
            foreach (var rule in activeRules)
            {
                try
                {
                    var vm = await _vmManager.GetVmAsync(rule.VmId, ct);
                    if (vm?.Spec.Network?.IpAddress != null)
                    {
                        if (rule.VmPrivateIp != vm.Spec.Network.IpAddress)
                        {
                            rule.VmPrivateIp = vm.Spec.Network.IpAddress;
                            await _repository.SaveAsync(rule);
                            _logger.LogDebug("Updated VM IP for rule {Id}: {Ip}", rule.Id, rule.VmPrivateIp);
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Could not refresh VM IP for rule {Id}", rule.Id);
                }
            }

            // Filter out rules without valid IPs
            var validRules = activeRules.Where(r => !string.IsNullOrEmpty(r.VmPrivateIp)).ToList();

            if (validRules.Count != activeRules.Count)
            {
                _logger.LogWarning(
                    "Skipping {Count} rules without valid VM IPs",
                    activeRules.Count - validRules.Count);
            }

            // Reload Caddy
            var success = await _caddyManager.ReloadFromRulesAsync(validRules, ct);

            if (success)
            {
                _logger.LogInformation("✓ Caddy reloaded with {Count} active routes", validRules.Count);
            }

            return success;
        }
        finally
        {
            _reloadLock.Release();
        }
    }

    public bool ValidateDomain(string domain, out string? error)
    {
        error = null;

        if (string.IsNullOrWhiteSpace(domain))
        {
            error = "Domain is required";
            return false;
        }

        var normalized = domain.Trim().ToLowerInvariant();

        // Length check
        if (normalized.Length > 253)
        {
            error = "Domain name too long (max 253 characters)";
            return false;
        }

        // Format check
        if (!DomainRegex.IsMatch(normalized))
        {
            error = "Invalid domain format. Must be a valid FQDN (e.g., myapp.example.com)";
            return false;
        }

        // Check for blocked domains
        if (BlockedDomains.Contains(normalized))
        {
            error = $"Domain '{normalized}' is reserved and cannot be used";
            return false;
        }

        // Check for blocked TLDs
        var tld = normalized.Split('.').LastOrDefault() ?? "";
        if (BlockedTlds.Contains(tld))
        {
            error = $"TLD '.{tld}' is not allowed. Please use a publicly routable domain.";
            return false;
        }

        // Check for IP addresses
        if (System.Net.IPAddress.TryParse(normalized.Replace(".", ""), out _))
        {
            error = "IP addresses are not allowed. Please use a domain name.";
            return false;
        }

        // Check for wildcards (not supported)
        if (normalized.Contains('*'))
        {
            error = "Wildcard domains are not supported";
            return false;
        }

        return true;
    }
}