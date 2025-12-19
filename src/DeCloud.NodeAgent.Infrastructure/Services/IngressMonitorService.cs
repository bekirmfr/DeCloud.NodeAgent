using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Background service that monitors ingress rules health and TLS certificate status.
/// Periodically checks:
/// - VM availability (IP addresses)
/// - Caddy configuration consistency
/// - TLS certificate expiration
/// - Route health checks
/// </summary>
public class IngressMonitorService : BackgroundService
{
    private readonly IngressRepository _repository;
    private readonly ICaddyManager _caddyManager;
    private readonly IIngressService _ingressService;
    private readonly ILogger<IngressMonitorService> _logger;

    private readonly TimeSpan _checkInterval = TimeSpan.FromMinutes(5);
    private readonly TimeSpan _tlsCheckInterval = TimeSpan.FromHours(6);

    private DateTime _lastTlsCheck = DateTime.MinValue;

    public IngressMonitorService(
        IngressRepository repository,
        ICaddyManager caddyManager,
        IIngressService ingressService,
        ILogger<IngressMonitorService> logger)
    {
        _repository = repository;
        _caddyManager = caddyManager;
        _ingressService = ingressService;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Ingress monitor service started");

        // Initial delay to let services start
        await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);

        // Initial reload to ensure Caddy is configured
        await InitialLoadAsync(stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await PerformHealthCheckAsync(stoppingToken);

                // Check TLS certificates periodically
                if (DateTime.UtcNow - _lastTlsCheck > _tlsCheckInterval)
                {
                    await CheckTlsCertificatesAsync(stoppingToken);
                    _lastTlsCheck = DateTime.UtcNow;
                }

                await Task.Delay(_checkInterval, stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in ingress monitor loop");
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            }
        }

        _logger.LogInformation("Ingress monitor service stopped");
    }

    private async Task InitialLoadAsync(CancellationToken ct)
    {
        try
        {
            // Check if Caddy is running
            var healthy = await _caddyManager.IsHealthyAsync(ct);

            if (!healthy)
            {
                _logger.LogWarning("Caddy is not healthy on startup - ingress routing will not work");
                return;
            }

            // Load and apply all active rules
            var rules = await _repository.GetAllActiveAsync();

            if (rules.Count > 0)
            {
                _logger.LogInformation("Loading {Count} active ingress rules on startup", rules.Count);
                await _ingressService.ReloadAllAsync(ct);
            }
            else
            {
                _logger.LogInformation("No active ingress rules to load on startup");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during initial ingress load");
        }
    }

    private async Task PerformHealthCheckAsync(CancellationToken ct)
    {
        try
        {
            // Check Caddy health
            var caddyHealthy = await _caddyManager.IsHealthyAsync(ct);

            if (!caddyHealthy)
            {
                _logger.LogWarning("Caddy health check failed - attempting to recover");

                // Mark all active rules as needing attention
                var activeRules = await _repository.GetAllActiveAsync();
                foreach (var rule in activeRules)
                {
                    await _repository.UpdateStatusAsync(
                        rule.Id,
                        IngressStatus.Error,
                        "Caddy proxy is unhealthy");
                }

                return;
            }

            // Get all rules and check their status
            var rules = await _repository.GetAllAsync();
            var needsReload = false;

            foreach (var rule in rules)
            {
                if (rule.Status == IngressStatus.Deleted || rule.Status == IngressStatus.Deleting)
                    continue;

                // Check for rules that were in error but might be recoverable
                if (rule.Status == IngressStatus.Error)
                {
                    if (!string.IsNullOrEmpty(rule.VmPrivateIp))
                    {
                        _logger.LogInformation(
                            "Attempting to recover error rule {Id} for {Domain}",
                            rule.Id, rule.Domain);

                        rule.Status = IngressStatus.Configuring;
                        await _repository.UpdateStatusAsync(rule.Id, IngressStatus.Configuring, "Attempting recovery");
                        needsReload = true;
                    }
                }

                // Check for active rules without VM IP
                if (rule.Status == IngressStatus.Active && string.IsNullOrEmpty(rule.VmPrivateIp))
                {
                    _logger.LogWarning(
                        "Active rule {Id} for {Domain} has no VM IP - marking as error",
                        rule.Id, rule.Domain);

                    await _repository.UpdateStatusAsync(
                        rule.Id,
                        IngressStatus.Error,
                        "VM IP address is missing");
                }
            }

            if (needsReload)
            {
                _logger.LogInformation("Triggering reload due to recovered rules");
                await _ingressService.ReloadAllAsync(ct);
            }

            // Log summary
            var activeCount = rules.Count(r => r.Status == IngressStatus.Active);
            var errorCount = rules.Count(r => r.Status == IngressStatus.Error);

            if (errorCount > 0)
            {
                _logger.LogWarning(
                    "Ingress status: {Active} active, {Error} in error state",
                    activeCount, errorCount);
            }
            else
            {
                _logger.LogDebug("Ingress status: {Active} active routes", activeCount);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error performing ingress health check");
        }
    }

    private async Task CheckTlsCertificatesAsync(CancellationToken ct)
    {
        try
        {
            var rules = await _repository.GetAllActiveAsync();
            var tlsRules = rules.Where(r => r.EnableTls).ToList();

            if (tlsRules.Count == 0)
            {
                return;
            }

            _logger.LogInformation("Checking TLS certificates for {Count} domains", tlsRules.Count);

            foreach (var rule in tlsRules)
            {
                try
                {
                    var certInfo = await _caddyManager.GetCertificateInfoAsync(rule.Domain, ct);

                    if (certInfo != null)
                    {
                        var newStatus = certInfo.IsValid
                            ? (certInfo.IsExpiringSoon ? TlsCertStatus.ExpiringSoon : TlsCertStatus.Valid)
                            : TlsCertStatus.Expired;

                        if (rule.TlsStatus != newStatus || rule.TlsExpiresAt != certInfo.NotAfter)
                        {
                            await _repository.UpdateTlsStatusAsync(rule.Id, newStatus, certInfo.NotAfter);

                            if (certInfo.IsExpiringSoon)
                            {
                                _logger.LogWarning(
                                    "TLS certificate for {Domain} expires in {Days} days",
                                    rule.Domain, certInfo.DaysUntilExpiry);
                            }
                        }
                    }
                    else if (rule.TlsStatus == TlsCertStatus.Pending || rule.TlsStatus == TlsCertStatus.Provisioning)
                    {
                        // Certificate might still be provisioning
                        _logger.LogDebug(
                            "Certificate for {Domain} still provisioning (status: {Status})",
                            rule.Domain, rule.TlsStatus);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error checking TLS certificate for {Domain}", rule.Domain);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking TLS certificates");
        }
    }
}

/// <summary>
/// Service for periodic cleanup of deleted ingress rules
/// </summary>
public class IngressMaintenanceService : BackgroundService
{
    private readonly IngressRepository _repository;
    private readonly ILogger<IngressMaintenanceService> _logger;

    public IngressMaintenanceService(
        IngressRepository repository,
        ILogger<IngressMaintenanceService> logger)
    {
        _repository = repository;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Ingress maintenance service started");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Run daily
                await Task.Delay(TimeSpan.FromHours(24), stoppingToken);

                // Purge deleted rules older than 7 days
                await _repository.PurgeDeletedAsync(TimeSpan.FromDays(7));

                _logger.LogInformation("Ingress maintenance completed");
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in ingress maintenance");
            }
        }
    }
}