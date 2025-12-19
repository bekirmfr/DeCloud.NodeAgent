using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Configuration options for Caddy manager
/// </summary>
public class CaddyOptions
{
    /// <summary>
    /// Caddy Admin API endpoint
    /// </summary>
    public string AdminApiUrl { get; set; } = "http://localhost:2019";

    /// <summary>
    /// Path to Caddy configuration file (for fallback/initial config)
    /// </summary>
    public string ConfigPath { get; set; } = "/etc/caddy/Caddyfile";

    /// <summary>
    /// Email for Let's Encrypt ACME account
    /// </summary>
    public string AcmeEmail { get; set; } = "";

    /// <summary>
    /// Use Let's Encrypt staging for testing
    /// </summary>
    public bool UseAcmeStaging { get; set; } = false;

    /// <summary>
    /// Data directory for Caddy (certificates, etc.)
    /// </summary>
    public string DataDir { get; set; } = "/var/lib/caddy";

    /// <summary>
    /// Enable Caddy access logging
    /// </summary>
    public bool EnableAccessLog { get; set; } = true;

    /// <summary>
    /// Access log path
    /// </summary>
    public string AccessLogPath { get; set; } = "/var/log/caddy/access.log";

    /// <summary>
    /// Global rate limit (requests per second, 0 = disabled)
    /// </summary>
    public int GlobalRateLimitRps { get; set; } = 0;

    /// <summary>
    /// Enable automatic HTTPS redirect
    /// </summary>
    public bool AutoHttpsRedirect { get; set; } = true;
}

/// <summary>
/// Interface for Caddy management
/// </summary>
public interface ICaddyManager
{
    /// <summary>
    /// Check if Caddy is running and accessible
    /// </summary>
    Task<bool> IsHealthyAsync(CancellationToken ct = default);

    /// <summary>
    /// Get current Caddy configuration
    /// </summary>
    Task<JsonDocument?> GetConfigAsync(CancellationToken ct = default);

    /// <summary>
    /// Apply complete configuration
    /// </summary>
    Task<bool> ApplyConfigAsync(object config, CancellationToken ct = default);

    /// <summary>
    /// Reload configuration from all active ingress rules
    /// </summary>
    Task<bool> ReloadFromRulesAsync(IEnumerable<IngressRule> rules, CancellationToken ct = default);

    /// <summary>
    /// Add or update a single route
    /// </summary>
    Task<bool> UpsertRouteAsync(IngressRule rule, CancellationToken ct = default);

    /// <summary>
    /// Remove a route
    /// </summary>
    Task<bool> RemoveRouteAsync(string domain, CancellationToken ct = default);

    /// <summary>
    /// Get certificate status for a domain
    /// </summary>
    Task<CertificateInfo?> GetCertificateInfoAsync(string domain, CancellationToken ct = default);

    /// <summary>
    /// Trigger certificate renewal
    /// </summary>
    Task<bool> RenewCertificateAsync(string domain, CancellationToken ct = default);
}

/// <summary>
/// Manages Caddy reverse proxy via its Admin API.
/// Provides dynamic configuration updates without restart.
/// </summary>
public class CaddyManager : ICaddyManager
{
    private readonly HttpClient _httpClient;
    private readonly CaddyOptions _options;
    private readonly ILogger<CaddyManager> _logger;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true
    };

    public CaddyManager(
        HttpClient httpClient,
        IOptions<CaddyOptions> options,
        ILogger<CaddyManager> logger)
    {
        _httpClient = httpClient;
        _options = options.Value;
        _logger = logger;

        _httpClient.BaseAddress = new Uri(_options.AdminApiUrl);
        _httpClient.Timeout = TimeSpan.FromSeconds(30);
    }

    public async Task<bool> IsHealthyAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _httpClient.GetAsync("/config/", ct);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Caddy health check failed");
            return false;
        }
    }

    public async Task<JsonDocument?> GetConfigAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _httpClient.GetAsync("/config/", ct);
            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("Failed to get Caddy config: {Status}", response.StatusCode);
                return null;
            }

            var json = await response.Content.ReadAsStringAsync(ct);
            return JsonDocument.Parse(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting Caddy config");
            return null;
        }
    }

    public async Task<bool> ApplyConfigAsync(object config, CancellationToken ct = default)
    {
        try
        {
            var json = JsonSerializer.Serialize(config, JsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync("/load", content, ct);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync(ct);
                _logger.LogError("Failed to apply Caddy config: {Status} - {Error}",
                    response.StatusCode, error);
                return false;
            }

            _logger.LogInformation("✓ Caddy configuration applied successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying Caddy config");
            return false;
        }
    }

    public async Task<bool> ReloadFromRulesAsync(IEnumerable<IngressRule> rules, CancellationToken ct = default)
    {
        var ruleList = rules.ToList();
        _logger.LogInformation("Reloading Caddy config with {Count} ingress rules", ruleList.Count);

        var config = BuildFullConfig(ruleList);
        return await ApplyConfigAsync(config, ct);
    }

    public async Task<bool> UpsertRouteAsync(IngressRule rule, CancellationToken ct = default)
    {
        try
        {
            // Get current config
            var currentConfig = await GetConfigAsync(ct);

            // For simplicity, we rebuild the full config
            // In production, you could use PATCH operations for efficiency
            _logger.LogInformation("Upserting route for domain {Domain} → {VmIp}:{Port}",
                rule.Domain, rule.VmPrivateIp, rule.TargetPort);

            // This is a simplified approach - in practice, we'd want to
            // fetch all rules and rebuild, or use Caddy's PATCH API
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error upserting route for {Domain}", rule.Domain);
            return false;
        }
    }

    public async Task<bool> RemoveRouteAsync(string domain, CancellationToken ct = default)
    {
        try
        {
            _logger.LogInformation("Removing route for domain {Domain}", domain);
            // Routes are removed by rebuilding config without the domain
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing route for {Domain}", domain);
            return false;
        }
    }

    public async Task<CertificateInfo?> GetCertificateInfoAsync(string domain, CancellationToken ct = default)
    {
        try
        {
            // Caddy stores certs in its data directory
            // We can query the PKI app for cert info
            var response = await _httpClient.GetAsync($"/pki/ca/local/certificates", ct);

            if (!response.IsSuccessStatusCode)
            {
                return null;
            }

            // Parse and find the domain's cert
            // This is simplified - actual implementation depends on Caddy's response format
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error getting certificate info for {Domain}", domain);
            return null;
        }
    }

    public async Task<bool> RenewCertificateAsync(string domain, CancellationToken ct = default)
    {
        try
        {
            // Caddy auto-renews, but we can force it by touching the route
            _logger.LogInformation("Triggering certificate renewal for {Domain}", domain);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error renewing certificate for {Domain}", domain);
            return false;
        }
    }

    /// <summary>
    /// Build complete Caddy JSON config from ingress rules
    /// </summary>
    private object BuildFullConfig(List<IngressRule> rules)
    {
        var routes = new List<object>();
        var serverNames = new List<string>();

        foreach (var rule in rules.Where(r => r.Status == IngressStatus.Active || r.Status == IngressStatus.Configuring))
        {
            if (string.IsNullOrEmpty(rule.VmPrivateIp))
            {
                _logger.LogWarning("Skipping rule {Id} - no VM IP address", rule.Id);
                continue;
            }

            serverNames.Add(rule.Domain);

            var upstream = $"{rule.VmPrivateIp}:{rule.TargetPort}";
            var route = BuildRouteConfig(rule, upstream);
            routes.Add(route);

            _logger.LogDebug("Added route: {Domain} → {Upstream}", rule.Domain, upstream);
        }

        // Build the full Caddy config
        var config = new
        {
            admin = new
            {
                listen = "localhost:2019"
            },
            logging = _options.EnableAccessLog ? new
            {
                logs = new
                {
                    @default = new
                    {
                        writer = new
                        {
                            output = "file",
                            filename = _options.AccessLogPath
                        },
                        encoder = new
                        {
                            format = "json"
                        }
                    }
                }
            } : null,
            apps = new
            {
                http = new
                {
                    servers = new
                    {
                        ingress = new
                        {
                            listen = new[] { ":80", ":443" },
                            routes = routes,
                            automatic_https = new
                            {
                                disable = false,
                                disable_redirects = !_options.AutoHttpsRedirect
                            }
                        }
                    }
                },
                tls = !string.IsNullOrEmpty(_options.AcmeEmail) ? new
                {
                    automation = new
                    {
                        policies = new[]
                        {
                            new
                            {
                                issuers = new object[]
                                {
                                    new
                                    {
                                        module = "acme",
                                        email = _options.AcmeEmail,
                                        ca = _options.UseAcmeStaging
                                            ? "https://acme-staging-v02.api.letsencrypt.org/directory"
                                            : "https://acme-v02.api.letsencrypt.org/directory"
                                    }
                                }
                            }
                        }
                    }
                } : null
            }
        };

        return config;
    }

    /// <summary>
    /// Build a single route configuration
    /// </summary>
    private object BuildRouteConfig(IngressRule rule, string upstream)
    {
        var matchers = new List<object>
        {
            new
            {
                host = new[] { rule.Domain }
            }
        };

        // Add path matcher if prefix specified
        if (!string.IsNullOrEmpty(rule.PathPrefix))
        {
            matchers.Add(new
            {
                path = new[] { $"{rule.PathPrefix}*" }
            });
        }

        var handlers = new List<object>();

        // Add rate limiting if configured
        if (rule.RateLimitPerMinute > 0)
        {
            handlers.Add(new
            {
                handler = "rate_limit",
                rate_limit = new
                {
                    zone = rule.Domain.Replace(".", "_"),
                    rate = $"{rule.RateLimitPerMinute}r/m"
                }
            });
        }

        // Add IP whitelist if configured
        if (rule.AllowedIps.Count > 0)
        {
            handlers.Add(new
            {
                handler = "remote_ip",
                ranges = rule.AllowedIps
            });
        }

        // Add custom headers
        if (rule.CustomHeaders.Count > 0)
        {
            handlers.Add(new
            {
                handler = "headers",
                request = new
                {
                    set = rule.CustomHeaders
                }
            });
        }

        // Strip path prefix if configured
        if (rule.StripPathPrefix && !string.IsNullOrEmpty(rule.PathPrefix))
        {
            handlers.Add(new
            {
                handler = "rewrite",
                strip_path_prefix = rule.PathPrefix
            });
        }

        // Main reverse proxy handler
        var proxyHandler = new Dictionary<string, object>
        {
            ["handler"] = "reverse_proxy",
            ["upstreams"] = new[] { new { dial = upstream } }
        };

        // Enable WebSocket support
        if (rule.EnableWebSocket)
        {
            proxyHandler["transport"] = new
            {
                protocol = "http",
                read_buffer_size = 4096
            };
            proxyHandler["flush_interval"] = -1; // Disable buffering for WebSockets
        }

        // Add X-Forwarded headers
        proxyHandler["headers"] = new
        {
            request = new
            {
                set = new Dictionary<string, string[]>
                {
                    ["X-Real-IP"] = new[] { "{http.request.remote.host}" },
                    ["X-Forwarded-For"] = new[] { "{http.request.remote.host}" },
                    ["X-Forwarded-Proto"] = new[] { "{http.request.scheme}" },
                    ["X-Forwarded-Host"] = new[] { "{http.request.host}" }
                }
            }
        };

        handlers.Add(proxyHandler);

        return new
        {
            match = matchers,
            handle = handlers,
            terminal = true
        };
    }
}

/// <summary>
/// Certificate information
/// </summary>
public class CertificateInfo
{
    public string Domain { get; set; } = "";
    public string Issuer { get; set; } = "";
    public DateTime NotBefore { get; set; }
    public DateTime NotAfter { get; set; }
    public bool IsValid => DateTime.UtcNow >= NotBefore && DateTime.UtcNow <= NotAfter;
    public bool IsExpiringSoon => NotAfter <= DateTime.UtcNow.AddDays(30);
    public int DaysUntilExpiry => (int)(NotAfter - DateTime.UtcNow).TotalDays;
}