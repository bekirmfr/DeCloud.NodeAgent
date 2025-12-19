using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.NodeAgent.Infrastructure.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Infrastructure;

/// <summary>
/// Extension methods for registering ingress and security services in DI container
/// </summary>
public static class IngressServiceExtensions
{
    /// <summary>
    /// Add ingress gateway services to the service collection.
    /// This includes:
    /// - IngressRepository (SQLite persistence)
    /// - CaddyManager (reverse proxy management)
    /// - IngressService (business logic)
    /// - Background monitoring services
    /// </summary>
    public static IServiceCollection AddIngressServices(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        // Configure options
        services.Configure<CaddyOptions>(configuration.GetSection("Caddy"));

        // Register repository
        services.AddSingleton<IngressRepository>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<IngressRepository>>();
            var config = sp.GetRequiredService<IConfiguration>();

            // Default path for ingress database
            var basePath = config.GetValue<string>("Libvirt:VmStoragePath") ?? "/var/lib/decloud/vms";
            var dbPath = Path.Combine(basePath, "ingress.db");

            return new IngressRepository(dbPath, logger);
        });

        // Register HttpClient for Caddy API
        services.AddHttpClient<ICaddyManager, CaddyManager>((sp, client) =>
        {
            var options = sp.GetRequiredService<IOptions<CaddyOptions>>().Value;
            client.BaseAddress = new Uri(options.AdminApiUrl);
            client.Timeout = TimeSpan.FromSeconds(30);
        });

        // Register services
        services.AddSingleton<IIngressService, IngressService>();

        // Register background services
        services.AddHostedService<IngressMonitorService>();
        services.AddHostedService<IngressMaintenanceService>();

        return services;
    }

    /// <summary>
    /// Add security services for port validation and auditing.
    /// This includes:
    /// - PortSecurityService (target port validation for ingress)
    /// - AuditService (security logging)
    /// </summary>
    public static IServiceCollection AddSecurityServices(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        // Configure options
        services.Configure<PortSecurityOptions>(configuration.GetSection("PortSecurity"));
        services.Configure<AuditLogOptions>(configuration.GetSection("AuditLog"));

        // Port security (validates ingress target ports)
        services.AddSingleton<IPortSecurityService, PortSecurityService>();

        // Security audit logging
        services.AddSingleton<IAuditService, AuditService>();

        return services;
    }

    /// <summary>
    /// Add all ingress and security services
    /// </summary>
    public static IServiceCollection AddIngressAndSecurityServices(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        services.AddIngressServices(configuration);
        services.AddSecurityServices(configuration);
        return services;
    }
}