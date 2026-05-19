using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.NodeAgent.Infrastructure.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Read-only host-level diagnostic dashboard.
/// Serves the dashboard HTML at / and /dashboard.
/// All data endpoints are under /api/dashboard/*.
/// No authentication required — accessible at port 5100.
/// </summary>
[ApiController]
public class NodeDashboardController : ControllerBase
{
    private readonly ICommandExecutor _executor;
    private readonly IVmManager _vmManager;
    private readonly IImageManager _imageManager;
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly INodeStateService _nodeStateService;
    private readonly IWebHostEnvironment _env;
    private readonly IConfiguration _config;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly VmRepository _vmRepository;
    private readonly PortMappingRepository _portMappingRepository;
    private readonly IObligationStateService _obligationStateService;
    private readonly ILogger<NodeDashboardController> _logger;

    // Cached ingress base domain — fetched once from orchestrator, TTL 5 minutes
    private static string? _cachedIngressBaseDomain;
    private static DateTime _ingressCacheExpiry = DateTime.MinValue;
    private static readonly SemaphoreSlim _ingressCacheLock = new(1, 1);

    /// <summary>
    /// System services always checked, plus any wg-quick@ units found dynamically.
    /// </summary>
    private static readonly string[] BaseTrackedServices =
    [
        "decloud-node-agent",
        "libvirtd",
        "libvirt-guests",
        "qemu-kvm",
        "ufw",
        "fail2ban",
        "ssh"
    ];

    public NodeDashboardController(
        ICommandExecutor executor,
        IVmManager vmManager,
        IImageManager imageManager,
        IOrchestratorClient orchestratorClient,
        INodeStateService nodeStateService,
        IWebHostEnvironment env,
        IConfiguration config,
        VmRepository vmRepository,
        PortMappingRepository portMappingRepository,
        IHttpClientFactory httpClientFactory,
        IObligationStateService obligationStateService,
        ILogger<NodeDashboardController> logger)
    {
        _executor = executor;
        _vmManager = vmManager;
        _imageManager = imageManager;
        _orchestratorClient = orchestratorClient;
        _nodeStateService = nodeStateService;
        _env = env;
        _config = config;
        _httpClientFactory = httpClientFactory;
        _vmRepository = vmRepository;
        _portMappingRepository = portMappingRepository;
        _obligationStateService = obligationStateService;
        _logger = logger;
    }

    // ==========================================================================
    // Dashboard HTML — served at / and /dashboard
    // ==========================================================================

    [HttpGet("/")]
    [HttpGet("/dashboard")]
    public IActionResult GetDashboard()
    {
        // Resolve wwwroot/dashboard.html from multiple candidate paths.
        // Works both in development (current directory) and deployed (binary directory).
        var candidates = new[]
        {
            Path.Combine(_env.WebRootPath ?? "", "dashboard.html"),
            Path.Combine(AppContext.BaseDirectory, "wwwroot", "dashboard.html"),
            Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "dashboard.html")
        };

        foreach (var path in candidates.Where(p => !string.IsNullOrEmpty(p)))
        {
            if (System.IO.File.Exists(path))
                return Content(System.IO.File.ReadAllText(path), "text/html; charset=utf-8");
        }

        _logger.LogWarning("Dashboard HTML not found in any candidate path");
        return Content(FallbackHtml(), "text/html; charset=utf-8");
    }

    // ==========================================================================
    // GET /api/dashboard/summary
    // Node identity, uptime, orchestrator connection, agent version
    // ==========================================================================

    [HttpGet("/api/dashboard/summary")]
    public async Task<IActionResult> GetSummary(CancellationToken ct)
    {
        var uptimeTask = GetUptimeSecondsAsync(ct);
        var osTask = GetOsInfoAsync(ct);
        var ingressTask = GetIngressBaseDomainAsync(ct);

        await Task.WhenAll(uptimeTask, osTask, ingressTask);

        var lastHeartbeat = _orchestratorClient.GetLastHeartbeat();
        var nodeId = _orchestratorClient.NodeId;

        long? heartbeatSecondsAgo = null;
        if (lastHeartbeat != null)
        {
            try { heartbeatSecondsAgo = (long)(DateTime.UtcNow - lastHeartbeat.Heartbeat.Timestamp).TotalSeconds; }
            catch { /* leave null if property name differs */ }
        }

        return Ok(new
        {
            nodeId = nodeId ?? "unregistered",
            hostname = Environment.MachineName,
            walletAddress = _orchestratorClient.WalletAddress,
            agentVersion = _nodeStateService.AgentVersion,
            uptimeSeconds = uptimeTask.Result,
            os = osTask.Result,
            ingressBaseDomain = ingressTask.Result,   // e.g. "vms.stackfi.tech" — null if not configured
            orchestrator = new
            {
                connected = !string.IsNullOrEmpty(nodeId),
                lastHeartbeatAt = lastHeartbeat != null ? lastHeartbeat.Heartbeat.Timestamp : (DateTime?)null,
                secondsAgo = heartbeatSecondsAgo
            },
            collectedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        });
    }

    // ==========================================================================
    // GET /api/dashboard/vm-ingress
    // Returns a vmId → publicUrl map for all VMs on this node, sourced from the
    // orchestrator database. Cached for 30 seconds.
    // Falls back gracefully: orchestrator unreachable → empty dict (JS handles
    // fallback to formula).
    // ==========================================================================

    private static Dictionary<string, string> _cachedVmIngress = new();
    private static DateTime _vmIngressCacheExpiry = DateTime.MinValue;
    private static readonly SemaphoreSlim _vmIngressCacheLock = new(1, 1);

    [HttpGet("/api/dashboard/vm-ingress")]
    public async Task<IActionResult> GetVmIngress(CancellationToken ct)
    {
        // Fast path — valid cache
        if (_vmIngressCacheExpiry > DateTime.UtcNow)
            return Ok(_cachedVmIngress);

        if (!await _vmIngressCacheLock.WaitAsync(TimeSpan.FromSeconds(2), ct))
            return Ok(_cachedVmIngress); // return stale rather than block

        try
        {
            // Re-check inside lock
            if (_vmIngressCacheExpiry > DateTime.UtcNow)
                return Ok(_cachedVmIngress);

            var urls = await _orchestratorClient.GetVmIngressUrlsAsync(ct);

            _cachedVmIngress = urls;
            _vmIngressCacheExpiry = DateTime.UtcNow.AddSeconds(30);

            return Ok(urls);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to fetch VM ingress URLs");
            return Ok(_cachedVmIngress); // return whatever we have
        }
        finally
        {
            _vmIngressCacheLock.Release();
        }
    }

    [HttpGet("/api/downloads")]
    public IActionResult GetActiveDownloads()
    {
        var downloads = _imageManager.ActiveDownloads.Values.ToList();
        return Ok(downloads);
    }

    // ==========================================================================
    // Obligation state helpers
    // ==========================================================================

    /// <summary>
    /// Private-material fields present in every obligation state blob.
    /// These are NEVER surfaced through the dashboard API.
    ///
    /// SECURITY: If new private fields are added to RelayObligationState,
    /// DhtObligationState, or BlockStoreObligationState in the future,
    /// they must also be added to this set.
    /// </summary>
    private static readonly HashSet<string> _sensitiveStateKeys =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "WireGuardPrivateKey",
            "Ed25519PrivateKeyBase64",
            "AuthToken",
        };

    /// <summary>
    /// Parses raw obligation state JSON from the local SQLite store and returns
    /// a sanitised key/value dictionary with all private-material fields removed.
    ///
    /// Returns null when:
    ///   - json is null or whitespace (state not yet generated)
    ///   - JSON cannot be parsed (should never happen in practice)
    ///
    /// Callers treat null as "no state data available" and omit the field
    /// from the HTTP response so the dashboard handles it gracefully.
    /// </summary>
    private static Dictionary<string, object?>? SanitiseStateJson(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        try
        {
            using var doc = JsonDocument.Parse(json);
            var dict = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (_sensitiveStateKeys.Contains(prop.Name)) continue;
                dict[prop.Name] = prop.Value.ValueKind switch
                {
                    JsonValueKind.String => prop.Value.GetString(),
                    JsonValueKind.Number => prop.Value.TryGetInt64(out var l)
                                                ? (object)l
                                                : prop.Value.GetDouble(),
                    JsonValueKind.True   => true,
                    JsonValueKind.False  => false,
                    JsonValueKind.Null   => null,
                    _                   => prop.Value.GetRawText(),
                };
            }
            return dict.Count > 0 ? dict : null;
        }
        catch
        {
            return null;
        }
    }

    // ==========================================================================
    // GET /api/dashboard/obligations
    // Fetches SystemVmObligations for this node from the orchestrator.
    // Returns which system VM roles are obligated and their fulfilment status.
    // Cached for 30 seconds.
    // ==========================================================================

    private static List<object> _cachedObligations = new();
    private static DateTime _obligationsCacheExpiry = DateTime.MinValue;
    private static readonly SemaphoreSlim _obligationsCacheLock = new(1, 1);

    [HttpGet("/api/dashboard/obligations")]
    public async Task<IActionResult> GetObligations(CancellationToken ct)
    {
        if (_obligationsCacheExpiry > DateTime.UtcNow)
            return Ok(new { obligations = _cachedObligations });

        if (!await _obligationsCacheLock.WaitAsync(TimeSpan.FromSeconds(2), ct))
            return Ok(new { obligations = _cachedObligations });

        try
        {
            if (_obligationsCacheExpiry > DateTime.UtcNow)
                return Ok(new { obligations = _cachedObligations });

            // Use the orchestrator client — its internal _httpClient already carries
            // the node JWT in the default Authorization header, so no manual token
            // threading is needed. This mirrors GetVmIngressUrlsAsync pattern.
            var result = await _orchestratorClient.GetObligationsAsync(ct);

            if (result != null)
            {
                var enriched = new List<object>(result.Count);
                foreach (var o in result)
                {
                    // Map role int to the canonical string used by IObligationStateService / SQLite.
                    // SystemVmRole enum: Dht=0, Relay=1, BlockStore=2, Ingress=3.
                    // Ingress has no identity state yet — roleCanonical stays null and stateData is omitted.
                    var roleCanonical = o.Role switch
                    {
                        0 => "dht",
                        1 => "relay",
                        2 => "blockstore",
                        _ => null
                    };

                    Dictionary<string, object?>? stateData = null;
                    if (roleCanonical is not null)
                    {
                        var stateJson = await _obligationStateService.GetStateJsonAsync(roleCanonical, ct);
                        stateData = SanitiseStateJson(stateJson);
                    }

                    enriched.Add((object)new
                    {
                        role = o.Role,
                        roleName = o.RoleName,
                        vmId = o.VmId,
                        status = o.Status,
                        statusName = o.StatusName,
                        failureCount = o.FailureCount,
                        lastError = o.LastError,
                        deployedAt = o.DeployedAt,
                        activeAt = o.ActiveAt,
                        runningBinaryVersion = o.RunningBinaryVersion,
                        currentBinaryVersion = o.CurrentBinaryVersion,
                        stateVersion = o.StateVersion,
                        stateData,
                    });
                }
                _cachedObligations = enriched;
            }

            // AFTER — 10 s matches the heartbeat interval, ensures UI reflects
            // obligation state changes (Active, Deploying) within one poll cycle.
            _obligationsCacheExpiry = DateTime.UtcNow.AddSeconds(10);
            return Ok(new { obligations = _cachedObligations });
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to fetch obligations from orchestrator");
            return Ok(new { obligations = _cachedObligations });
        }
        finally
        {
            _obligationsCacheLock.Release();
        }
    }

    // ==========================================================================
    // GET /api/dashboard/database
    // Local SQLite state: VmRecords + PortMappings + schema metadata.
    // Read-only. Sensitive columns (SshPublicKey, EncryptedPassword, BaseImageHash)
    // are never included.
    // ==========================================================================

    [HttpGet("/api/dashboard/database")]
    public async Task<IActionResult> GetDatabase(CancellationToken ct)
    {
        try
        {
            var dbStats = await _vmRepository.GetStatsAsync();
            var vmRows = await _vmRepository.GetDashboardSummaryAsync();
            var portRows = await _portMappingRepository.GetAllActiveAsync();

            // Determine DB file size on disk
            long dbSizeBytes = 0;
            try
            {
                var dbPath = _vmRepository.DatabasePath;
                if (!string.IsNullOrEmpty(dbPath) && System.IO.File.Exists(dbPath))
                    dbSizeBytes = new FileInfo(dbPath).Length;
            }
            catch { /* non-critical */ }

            return Ok(new
            {
                schemaVersion = _vmRepository.SchemaVersion,
                databasePath = _vmRepository.DatabasePath,
                sizeBytes = dbSizeBytes,
                stats = new
                {
                    totalVms = vmRows.Count,
                    vmsByState = dbStats.VmsByState,
                },
                vmRecords = vmRows.Select(v => new
                {
                    v.VmId,
                    v.Name,
                    v.State,
                    v.VmType,
                    v.OwnerId,
                    v.IpAddress,
                    v.VncPort,
                    v.ReplicationFactor,
                    v.VirtualCpuCores,
                    memoryGb = Math.Round(v.MemoryBytes / 1073741824.0, 2),
                    diskGb = Math.Round(v.DiskBytes / 1073741824.0, 2),
                    v.CreatedAt,
                    v.LastUpdated,
                    v.TargetNodeId,
                    v.DeletionReason
                }),
                portMappings = portRows.Select(p => new
                {
                    p.VmId,
                    p.VmPrivateIp,
                    p.VmPort,
                    p.PublicPort,
                    protocol = p.Protocol.ToString(),
                    p.Label,
                    isActive = p.IsActive,
                    createdAt = p.CreatedAt.ToString("O")
                }),
                collectedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to collect local database info");
            return StatusCode(500, new { error = "Failed to read local database" });
        }
    }

    // ==========================================================================
    // GetIngressBaseDomainAsync
    // Fetches the central ingress base domain from the orchestrator's public
    // /api/central-ingress/status endpoint (AllowAnonymous — no auth needed).
    // Result is cached for 5 minutes to avoid per-request overhead.
    // Falls back to IConfiguration["CentralIngress:BaseDomain"] if set locally.
    // ==========================================================================

    private async Task<string?> GetIngressBaseDomainAsync(CancellationToken ct)
    {
        // 1. Fast path — valid cache
        if (_cachedIngressBaseDomain != null && DateTime.UtcNow < _ingressCacheExpiry)
            return _cachedIngressBaseDomain;

        // 2. Local config override (optional, for offline/dev scenarios)
        var localOverride = _config["CentralIngress:BaseDomain"];
        if (!string.IsNullOrWhiteSpace(localOverride))
        {
            _cachedIngressBaseDomain = localOverride;
            _ingressCacheExpiry = DateTime.UtcNow.AddMinutes(5);
            return localOverride;
        }

        // 3. Fetch from orchestrator — serialize with a semaphore so concurrent
        //    requests don't all race to fetch on a cold cache.
        if (!await _ingressCacheLock.WaitAsync(TimeSpan.FromSeconds(2), ct))
            return _cachedIngressBaseDomain; // return stale rather than block

        try
        {
            // Re-check inside lock
            if (_cachedIngressBaseDomain != null && DateTime.UtcNow < _ingressCacheExpiry)
                return _cachedIngressBaseDomain;

            var orchestratorUrl = _config["Orchestrator:Url"]
                               ?? _config["OrchestratorUrl"]
                               ?? _config["ORCHESTRATOR_URL"];

            if (string.IsNullOrWhiteSpace(orchestratorUrl))
                return null;

            var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(4);

            var response = await client.GetAsync(
                $"{orchestratorUrl.TrimEnd('/')}/api/central-ingress/status", ct);

            if (!response.IsSuccessStatusCode)
                return null;

            var json = JsonNode.Parse(await response.Content.ReadAsStringAsync(ct));
            var data = json?["data"];
            var enabled = data?["isEnabled"]?.GetValue<bool>() ?? false;
            var domain = data?["baseDomain"]?.GetValue<string>();

            if (!enabled || string.IsNullOrWhiteSpace(domain))
                return null;

            _cachedIngressBaseDomain = domain;
            _ingressCacheExpiry = DateTime.UtcNow.AddMinutes(5);
            return domain;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Could not fetch ingress base domain from orchestrator");
            return null;
        }
        finally
        {
            _ingressCacheLock.Release();
        }
    }

    // ==========================================================================
    // GET /api/dashboard/network
    // Full network topology:
    //   - All host interfaces (physical + WG + bridge + tap) with rx/tx stats
    //   - WireGuard peers with handshake health classification
    //   - libvirt/virbr0 bridge ports with VM-to-tap mapping
    //   - IP routing table
    // ==========================================================================

    [HttpGet("/api/dashboard/network")]
    public async Task<IActionResult> GetNetwork(CancellationToken ct)
    {
        var addrTask = _executor.ExecuteAsync("ip", "-j addr show", TimeSpan.FromSeconds(5), ct);
        var wgTask = _executor.ExecuteAsync("wg", "show all dump", TimeSpan.FromSeconds(5), ct);
        var brctlTask = _executor.ExecuteAsync("brctl", "show", TimeSpan.FromSeconds(5), ct);
        var routeTask = _executor.ExecuteAsync("ip", "-j route show", TimeSpan.FromSeconds(5), ct);
        var netdevTask = _executor.ExecuteAsync("cat", "/proc/net/dev", TimeSpan.FromSeconds(3), ct);

        await Task.WhenAll(addrTask, wgTask, brctlTask, routeTask, netdevTask);

        var interfaces = ParseIpAddrJson(addrTask.Result.StandardOutput);
        var wgInterfaces = ParseWgDump(wgTask.Result.StandardOutput);
        var bridges = await ParseBridgesAsync(brctlTask.Result.StandardOutput, ct);
        var routes = ParseIpRouteJson(routeTask.Result.StandardOutput);
        var stats = ParseProcNetDev(netdevTask.Result.StandardOutput);

        // Classify and enrich each interface
        var wgNames = wgInterfaces.Select(w => w.Name).ToHashSet(StringComparer.Ordinal);
        var bridgeNames = bridges.Select(b => b.Name).ToHashSet(StringComparer.Ordinal);

        foreach (var iface in interfaces)
        {
            // Inject rx/tx stats from /proc/net/dev
            if (stats.TryGetValue(iface.Name, out var s))
            {
                iface.RxBytes = s.Rx;
                iface.TxBytes = s.Tx;
            }

            // Push IP addresses into the corresponding bridge record
            var bridge = bridges.FirstOrDefault(b => b.Name == iface.Name);
            if (bridge != null)
                bridge.IpAddresses = iface.IpAddresses;

            // Classify interface type
            iface.Type = iface.Name switch
            {
                "lo" => "loopback",
                _ when wgNames.Contains(iface.Name) => "wireguard",
                _ when bridgeNames.Contains(iface.Name) => "bridge",
                _ when iface.Name.StartsWith("virbr") => "bridge",
                _ when iface.Name.StartsWith("vnet") || iface.Name.StartsWith("tap") => "tap",
                _ when iface.Name.StartsWith("docker") || iface.Name.StartsWith("br-") => "bridge",
                _ => "physical"
            };
        }

        return Ok(new
        {
            interfaces,
            wireguard = wgInterfaces,
            bridges,
            routes,
            collectedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        });
    }

    // ==========================================================================
    // GET /api/dashboard/ports
    // All listening TCP and UDP ports with process names
    // ==========================================================================

    [HttpGet("/api/dashboard/ports")]
    public async Task<IActionResult> GetPorts(CancellationToken ct)
    {
        var tcpTask = _executor.ExecuteAsync("ss", "-tlnp", TimeSpan.FromSeconds(5), ct);
        var udpTask = _executor.ExecuteAsync("ss", "-ulnp", TimeSpan.FromSeconds(5), ct);

        await Task.WhenAll(tcpTask, udpTask);

        return Ok(new
        {
            tcp = ParseSsOutput(tcpTask.Result.StandardOutput),
            udp = ParseSsOutput(udpTask.Result.StandardOutput),
            collectedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        });
    }

    // ==========================================================================
    // GET /api/dashboard/firewall
    // UFW status + rules, iptables INPUT / FORWARD / NAT POSTROUTING chains
    // All commands are wrapped individually — ufw/iptables may not be installed
    // on every host (minimal installs, containers, WSL). Missing binaries return
    // a graceful "not available" result rather than a 500.
    // ==========================================================================

    [HttpGet("/api/dashboard/firewall")]
    public async Task<IActionResult> GetFirewall(CancellationToken ct)
    {
        static async Task<CommandResult> SafeRun(
            ICommandExecutor exec, string cmd, string args, TimeSpan timeout, CancellationToken ct)
        {
            try
            {
                return await exec.ExecuteAsync(cmd, args, timeout, ct);
            }
            catch (Exception ex) when (ex is System.ComponentModel.Win32Exception
                                            or System.IO.IOException
                                            or InvalidOperationException)
            {
                // Binary not found or not executable — return empty success-like result
                return new CommandResult
                {
                    ExitCode = 127,  // POSIX "command not found"
                    StandardOutput = "",
                    StandardError = $"{cmd}: command not found"
                };
            }
        }

        var ufwTask = SafeRun(_executor, "ufw", "status verbose", TimeSpan.FromSeconds(5), ct);
        var inputTask = SafeRun(_executor, "iptables", "-L INPUT -v -n --line-numbers", TimeSpan.FromSeconds(5), ct);
        var fwdTask = SafeRun(_executor, "iptables", "-L FORWARD -v -n --line-numbers", TimeSpan.FromSeconds(5), ct);
        var natTask = SafeRun(_executor, "iptables", "-t nat -L POSTROUTING -v -n --line-numbers", TimeSpan.FromSeconds(5), ct);

        await Task.WhenAll(ufwTask, inputTask, fwdTask, natTask);

        var ufwResult = ufwTask.Result;
        var inputResult = inputTask.Result;
        var fwdResult = fwdTask.Result;
        var natResult = natTask.Result;

        return Ok(new
        {
            ufw = ParseUfwOutput(ufwResult.Success ? ufwResult.StandardOutput : ""),
            iptables = new
            {
                input = new
                {
                    chain = "INPUT",
                    policy = ParseChainPolicy(inputResult.StandardOutput),
                    rules = ParseIptablesRules(inputResult.StandardOutput),
                    raw = inputResult.StandardOutput,
                    available = inputResult.ExitCode != 127
                },
                forward = new
                {
                    chain = "FORWARD",
                    policy = ParseChainPolicy(fwdResult.StandardOutput),
                    rules = ParseIptablesRules(fwdResult.StandardOutput),
                    raw = fwdResult.StandardOutput,
                    available = fwdResult.ExitCode != 127
                },
                natPostrouting = new
                {
                    chain = "POSTROUTING (nat)",
                    policy = ParseChainPolicy(natResult.StandardOutput),
                    rules = ParseIptablesRules(natResult.StandardOutput),
                    raw = natResult.StandardOutput,
                    available = natResult.ExitCode != 127
                }
            },
            collectedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        });
    }

    // ==========================================================================
    // GET /api/dashboard/services
    // Systemd status for all DeCloud-relevant services + dynamic wg-quick@ units
    // ==========================================================================

    [HttpGet("/api/dashboard/services")]
    public async Task<IActionResult> GetServices(CancellationToken ct)
    {
        // Discover active wg-quick@ units dynamically
        var wgListResult = await _executor.ExecuteAsync(
            "bash",
            "-c \"systemctl list-units 'wg-quick@*' --no-legend --no-pager 2>/dev/null | awk '{print $1}'\"",
            TimeSpan.FromSeconds(5), ct);

        var dynamicWgServices = wgListResult.Success
            ? wgListResult.StandardOutput
                .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim().Replace(".service", ""))
                .Where(s => s.StartsWith("wg-quick@"))
                .ToArray()
            : Array.Empty<string>();

        var allServices = BaseTrackedServices
            .Concat(dynamicWgServices)
            .Distinct()
            .ToArray();

        var tasks = allServices.Select(async svc =>
        {
            var result = await _executor.ExecuteAsync(
                "systemctl",
                $"show {svc} --no-pager --property=ActiveState,SubState,Description,LoadState",
                TimeSpan.FromSeconds(5), ct);

            var props = result.StandardOutput
                .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Select(l => l.Split('=', 2))
                .Where(p => p.Length == 2)
                .ToDictionary(p => p[0].Trim(), p => p[1].Trim());

            var loadState = props.GetValueOrDefault("LoadState", "not-found");
            var activeState = props.GetValueOrDefault("ActiveState", "unknown");
            var subState = props.GetValueOrDefault("SubState", "unknown");

            return new
            {
                name = svc,
                loadState,
                activeState,
                subState,
                description = props.GetValueOrDefault("Description", ""),
                isActive = activeState == "active",
                isLoaded = loadState == "loaded"
            };
        });

        var services = await Task.WhenAll(tasks);

        return Ok(new { services, collectedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds() });
    }

    // ==========================================================================
    // GET /api/dashboard/logs?lines=100
    // Returns the last N lines from the systemd journal (primary) or
    // /var/log/decloud/nodeagent.log (fallback for pre-journal installs).
    // ==========================================================================

    [HttpGet("/api/dashboard/logs")]
    public async Task<IActionResult> GetLogs(
    [FromQuery] int lines = 100,
    CancellationToken ct = default)
    {
        lines = Math.Clamp(lines, 10, 500);

        // --- Primary: systemd journal (timestamped, filterable) ---
        var result = await _executor.ExecuteAsync(
            "journalctl",
            $"-u decloud-node-agent -n {lines} --no-pager --output=short-iso",
            TimeSpan.FromSeconds(10), ct);

        if (result.Success)
        {
            var journalLines = result.StandardOutput
                .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .ToList();

            if (journalLines.Count > 0)
            {
                return Ok(new
                {
                    source = "journal",
                    logFile = (string?)null,
                    logLines = journalLines,
                    count = journalLines.Count,
                    collectedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
                });
            }
        }

        return Ok(new
        {
            source = "none",
            logFile = (string?)null,
            logLines = new List<string> { "[no logs available — journal empty and log file not found]" },
            count = 0,
            collectedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        });
    }

    /// <summary>
    /// Efficiently reads the last N lines of a file without loading the entire
    /// file into memory — seeks backwards from the end in 8 KB chunks.
    /// </summary>
    private static async Task<List<string>> ReadLastLinesAsync(
        string path, int lineCount, CancellationToken ct)
    {
        const int bufferSize = 8192;
        var lines = new List<string>(lineCount + 1);

        await using var fs = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite,
            bufferSize: bufferSize, useAsync: true);

        if (fs.Length == 0) return lines;

        var buffer = new byte[bufferSize];
        var remainder = "";
        var position = fs.Length;

        while (position > 0 && lines.Count < lineCount)
        {
            var readSize = (int)Math.Min(bufferSize, position);
            position -= readSize;
            fs.Seek(position, SeekOrigin.Begin);

            var bytesRead = await fs.ReadAsync(buffer.AsMemory(0, readSize), ct);
            var chunk = System.Text.Encoding.UTF8.GetString(buffer, 0, bytesRead) + remainder;
            var parts = chunk.Split('\n');

            // parts[0] is a partial line — carry it forward as the remainder
            remainder = parts[0];

            // parts[1..] are complete lines, in reverse order
            for (var i = parts.Length - 1; i >= 1; i--)
            {
                var line = parts[i].TrimEnd('\r');
                if (!string.IsNullOrEmpty(line))
                    lines.Add(line);
                if (lines.Count >= lineCount) break;
            }
        }

        // Don't forget the remainder (the very first line of the file)
        if (!string.IsNullOrEmpty(remainder) && lines.Count < lineCount)
            lines.Add(remainder.TrimEnd('\r'));

        lines.Reverse(); // restore chronological order
        return lines;
    }

    // ==========================================================================
    // Parsing Helpers
    // ==========================================================================

    private static List<NetInterface> ParseIpAddrJson(string json)
    {
        var result = new List<NetInterface>();
        if (string.IsNullOrWhiteSpace(json)) return result;

        try
        {
            var arr = JsonNode.Parse(json)?.AsArray();
            if (arr == null) return result;

            foreach (var item in arr)
            {
                if (item == null) continue;

                var flags = item["flags"]?.AsArray()
                    .Select(f => f?.GetValue<string>() ?? "")
                    .ToList() ?? [];

                var addresses = new List<string>();
                foreach (var a in item["addr_info"]?.AsArray() ?? [])
                {
                    var local = a?["local"]?.GetValue<string>();
                    var plen = a?["prefixlen"]?.GetValue<int>() ?? 0;
                    if (!string.IsNullOrEmpty(local))
                        addresses.Add($"{local}/{plen}");
                }

                result.Add(new NetInterface
                {
                    Name = item["ifname"]?.GetValue<string>() ?? "",
                    MacAddress = item["address"]?.GetValue<string>(),
                    Mtu = item["mtu"]?.GetValue<int>() ?? 0,
                    Flags = flags,
                    IsUp = flags.Contains("UP"),
                    IpAddresses = addresses,
                    Type = "physical"
                });
            }
        }
        catch { /* return partial result */ }

        return result;
    }

    /// <summary>
    /// Parses `wg show all dump` output.
    /// Interface lines have 5 tab-separated fields: ifname, privkey, pubkey, port, fwmark
    /// Peer lines have 9 tab-separated fields: ifname, pubkey, preshared, endpoint,
    ///   allowedips, latest-handshake, rx-bytes, tx-bytes, persistent-keepalive
    /// </summary>
    private static List<WgInterfaceInfo> ParseWgDump(string output)
    {
        var map = new Dictionary<string, WgInterfaceInfo>(StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(output)) return [];

        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var p = line.Split('\t');

            if (p.Length == 5)
            {
                // Interface header line
                map[p[0]] = new WgInterfaceInfo
                {
                    Name = p[0],
                    PublicKey = p[2],
                    ListenPort = int.TryParse(p[3], out var port) ? port : 0,
                    Peers = []
                };
            }
            else if (p.Length == 9 && map.TryGetValue(p[0], out var iface))
            {
                var ts = long.TryParse(p[5], out var hs) ? hs : 0L;
                var secsAgo = ts > 0 ? (int)(now - ts) : -1;
                var keepalive = int.TryParse(p[8], out var ka) && ka > 0 ? ka : 0;

                iface.Peers.Add(new WgPeerInfo
                {
                    PublicKey = p[1],
                    Endpoint = p[3] == "(none)" ? null : p[3],
                    AllowedIps = p[4],
                    LatestHandshake = ts,
                    HandshakeSecondsAgo = secsAgo >= 0 ? secsAgo : null,
                    HandshakeStatus = ClassifyHandshake(ts, secsAgo),
                    RxBytes = long.TryParse(p[6], out var rx) ? rx : 0,
                    TxBytes = long.TryParse(p[7], out var tx) ? tx : 0,
                    PersistentKeepalive = keepalive
                });
            }
        }

        return [.. map.Values];
    }

    private static string ClassifyHandshake(long ts, int secsAgo)
    {
        if (ts == 0 || secsAgo < 0) return "never";
        if (secsAgo < 180) return "ok";      // < 3 min
        if (secsAgo < 600) return "stale";   // 3–10 min
        return "dead";    // > 10 min
    }

    /// <summary>
    /// Parses `brctl show` and cross-references tap interfaces with running VMs
    /// via `virsh domiflist` to build the complete bridge→tap→VM topology.
    /// </summary>
    private async Task<List<BridgeInfo>> ParseBridgesAsync(string brctlOutput, CancellationToken ct)
    {
        var bridges = new List<BridgeInfo>();
        if (string.IsNullOrWhiteSpace(brctlOutput)) return bridges;

        // Build tap-interface → VM mapping from all running VMs
        var tapToVm = new Dictionary<string, (string Id, string Name)>(StringComparer.Ordinal);
        foreach (var vm in _vmManager.GetAllVms().Where(v => v.State == VmState.Running))
        {
            var r = await _executor.ExecuteAsync("virsh", $"domiflist {vm.VmId}",
                TimeSpan.FromSeconds(3), ct);
            if (!r.Success) continue;

            foreach (var line in r.StandardOutput.Split('\n').Skip(2))
            {
                var cols = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (cols.Length >= 1 && !cols[0].StartsWith('-'))
                    tapToVm[cols[0]] = (vm.VmId, vm.Name ?? vm.VmId);
            }
        }

        // Parse brctl show — bridge name is only present on the first port line
        BridgeInfo? current = null;

        foreach (var line in brctlOutput.Split('\n').Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            if (!char.IsWhiteSpace(line[0]))
            {
                // New bridge: name  bridge-id  stp  [first-interface]
                var parts = line.Split(new[] { '\t', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                current = new BridgeInfo { Name = parts[0] };
                bridges.Add(current);

                if (parts.Length >= 4)
                    current.Ports.Add(MakeBridgePort(parts[3], tapToVm));
            }
            else if (current != null)
            {
                var iface = line.Trim();
                if (!string.IsNullOrEmpty(iface))
                    current.Ports.Add(MakeBridgePort(iface, tapToVm));
            }
        }

        return bridges;
    }

    private static BridgePort MakeBridgePort(
        string iface,
        Dictionary<string, (string Id, string Name)> tapToVm)
    {
        tapToVm.TryGetValue(iface, out var vm);
        return new BridgePort { Interface = iface, VmId = vm.Id, VmName = vm.Name };
    }

    private static List<RouteEntry> ParseIpRouteJson(string json)
    {
        var routes = new List<RouteEntry>();
        if (string.IsNullOrWhiteSpace(json)) return routes;

        try
        {
            var arr = JsonNode.Parse(json)?.AsArray();
            if (arr == null) return routes;

            foreach (var item in arr)
            {
                if (item == null) continue;
                routes.Add(new RouteEntry
                {
                    Destination = item["dst"]?.GetValue<string>() ?? "",
                    Gateway = item["gateway"]?.GetValue<string>(),
                    Interface = item["dev"]?.GetValue<string>() ?? "",
                    Protocol = item["protocol"]?.GetValue<string>(),
                    Scope = item["scope"]?.GetValue<string>(),
                    Metric = item["metric"]?.GetValue<int>() ?? 0
                });
            }
        }
        catch { }

        return routes;
    }

    /// <summary>
    /// Parses /proc/net/dev for per-interface rx/tx byte counters.
    /// Format: iface: rx_bytes ... tx_bytes (8 fields each side, tx is field index 8 from start)
    /// </summary>
    private static Dictionary<string, (long Rx, long Tx)> ParseProcNetDev(string content)
    {
        var result = new Dictionary<string, (long, long)>(StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(content)) return result;

        foreach (var line in content.Split('\n').Skip(2))
        {
            var idx = line.IndexOf(':');
            if (idx < 0) continue;

            var name = line[..idx].Trim();
            var fields = line[(idx + 1)..].Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (fields.Length >= 9
                && long.TryParse(fields[0], out var rx)
                && long.TryParse(fields[8], out var tx))
            {
                result[name] = (rx, tx);
            }
        }

        return result;
    }

    /// <summary>
    /// Parses `ss -tlnp` or `ss -ulnp` output.
    /// Field layout: State Recv-Q Send-Q Local Peer [Process]
    /// </summary>
    private static List<PortEntry> ParseSsOutput(string output)
    {
        var ports = new List<PortEntry>();
        if (string.IsNullOrWhiteSpace(output)) return ports;

        foreach (var line in output.Split('\n').Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 5) continue;

            var localAddr = parts[4];
            var lastColon = localAddr.LastIndexOf(':');
            if (lastColon < 0) continue;

            var process = "";
            if (parts.Length > 5)
            {
                var m = Regex.Match(string.Join(" ", parts[5..]), @"\(\(""([^""]+)""");
                if (m.Success) process = m.Groups[1].Value;
            }

            ports.Add(new PortEntry
            {
                LocalAddress = localAddr[..lastColon],
                Port = int.TryParse(localAddr[(lastColon + 1)..], out var p) ? p : 0,
                Process = process,
                State = parts[0]
            });
        }

        return ports;
    }

    private static UfwStatusInfo ParseUfwOutput(string output)
    {
        var info = new UfwStatusInfo { RawOutput = output };
        if (string.IsNullOrWhiteSpace(output)) return info;

        info.Active = output.Contains("Status: active");

        var def = Regex.Match(output,
            @"Default:\s*(\w+)\s*\(incoming\),\s*(\w+)\s*\(outgoing\),\s*(\w+)\s*\(routed\)",
            RegexOptions.IgnoreCase);
        if (def.Success)
        {
            info.DefaultIncoming = def.Groups[1].Value;
            info.DefaultOutgoing = def.Groups[2].Value;
            info.DefaultForward = def.Groups[3].Value;
        }

        foreach (var line in output.Split('\n'))
        {
            var m = Regex.Match(line,
                @"\[\s*(\d+)\]\s+(.+?)\s+(ALLOW|DENY|REJECT|LIMIT)\s*(IN|OUT|FWD|)\s+(.*)",
                RegexOptions.IgnoreCase);
            if (m.Success)
            {
                info.Rules.Add(new UfwRule
                {
                    Number = int.Parse(m.Groups[1].Value),
                    To = m.Groups[2].Value.Trim(),
                    Action = m.Groups[3].Value.Trim(),
                    Direction = m.Groups[4].Value.Trim(),
                    From = m.Groups[5].Value.Trim()
                });
            }
        }

        return info;
    }

    /// <summary>
    /// Parses `iptables -L {chain} -v -n --line-numbers` output.
    /// Skips Chain headers and column labels; parses numeric rule rows.
    /// </summary>
    private static List<IptablesRule> ParseIptablesRules(string output)
    {
        var rules = new List<IptablesRule>();
        if (string.IsNullOrWhiteSpace(output)) return rules;

        foreach (var line in output.Split('\n'))
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            if (trimmed.StartsWith("Chain") || trimmed.StartsWith("pkts") || trimmed.StartsWith("num"))
                continue;

            var parts = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            // Line-numbered output: num pkts bytes target prot opt in out src dst [extras]
            int offset = 0;
            int lineNum = 0;
            if (parts.Length > 0 && int.TryParse(parts[0], out lineNum))
                offset = 1;
            else if (parts.Length > 0 && !long.TryParse(parts[0], out _))
                continue; // not a rule line

            if (parts.Length < offset + 7) continue;

            rules.Add(new IptablesRule
            {
                LineNumber = lineNum,
                Packets = long.TryParse(parts[offset], out var pk) ? pk : 0,
                Bytes = long.TryParse(parts[offset + 1], out var by) ? by : 0,
                Target = parts[offset + 2],
                Protocol = parts[offset + 3],
                In = parts[offset + 5] == "*" ? "any" : parts[offset + 5],
                Out = parts[offset + 6] == "*" ? "any" : parts[offset + 6],
                Source = parts.Length > offset + 7 ? parts[offset + 7] : "0.0.0.0/0",
                Destination = parts.Length > offset + 8 ? parts[offset + 8] : "0.0.0.0/0",
                Options = parts.Length > offset + 9
                              ? string.Join(" ", parts[(offset + 9)..])
                              : ""
            });
        }

        return rules;
    }

    private static string ParseChainPolicy(string output)
    {
        var m = Regex.Match(output, @"Chain \S+ \(policy (\w+)");
        return m.Success ? m.Groups[1].Value : "unknown";
    }

    // ==========================================================================
    // Utility
    // ==========================================================================

    private async Task<long> GetUptimeSecondsAsync(CancellationToken ct)
    {
        try
        {
            var content = await System.IO.File.ReadAllTextAsync("/proc/uptime", ct);
            if (double.TryParse(
                    content.Trim().Split(' ')[0],
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out var s))
                return (long)s;
        }
        catch { }
        return 0;
    }

    private async Task<string> GetOsInfoAsync(CancellationToken ct)
    {
        try
        {
            var r = await _executor.ExecuteAsync(
                "bash", "-c \"source /etc/os-release 2>/dev/null && echo $PRETTY_NAME\"",
                TimeSpan.FromSeconds(3), ct);
            if (r.Success && !string.IsNullOrWhiteSpace(r.StandardOutput))
                return r.StandardOutput.Trim();
        }
        catch { }
        return Environment.OSVersion.ToString();
    }

    private static string FallbackHtml() =>
        """
        <!DOCTYPE html>
        <html><body style="background:#0a0b0f;color:#00d4aa;font-family:monospace;padding:2rem">
          <h1>DeCloud Node Agent</h1>
          <p style="color:#f0f2f5">Dashboard HTML not found.</p>
          <p>Ensure <code>wwwroot/dashboard.html</code> exists next to the binary.</p>
          <p>API: <a style="color:#00d4aa" href="/api/dashboard/summary">/api/dashboard/summary</a></p>
        </body></html>
        """;

    // ==========================================================================
    // Data models (co-located — no separate file needed for KISS)
    // ==========================================================================

    public class NetInterface
    {
        public string Name { get; set; } = "";
        public string Type { get; set; } = "physical"; // wireguard|bridge|tap|loopback|physical
        public string? MacAddress { get; set; }
        public int Mtu { get; set; }
        public bool IsUp { get; set; }
        public List<string> Flags { get; set; } = [];
        public List<string> IpAddresses { get; set; } = [];
        public long RxBytes { get; set; }
        public long TxBytes { get; set; }
    }

    public class WgInterfaceInfo
    {
        public string Name { get; set; } = "";
        public string PublicKey { get; set; } = "";
        public int ListenPort { get; set; }
        public List<WgPeerInfo> Peers { get; set; } = [];
    }

    public class WgPeerInfo
    {
        public string PublicKey { get; set; } = "";
        public string? Endpoint { get; set; }
        public string AllowedIps { get; set; } = "";
        public long LatestHandshake { get; set; }
        public int? HandshakeSecondsAgo { get; set; }
        public string HandshakeStatus { get; set; } = "never";  // ok|stale|dead|never
        public long RxBytes { get; set; }
        public long TxBytes { get; set; }
        public int PersistentKeepalive { get; set; }
    }

    public class BridgeInfo
    {
        public string Name { get; set; } = "";
        public List<string> IpAddresses { get; set; } = [];
        public List<BridgePort> Ports { get; set; } = [];
    }

    public class BridgePort
    {
        public string Interface { get; set; } = "";
        public string? VmId { get; set; }
        public string? VmName { get; set; }
    }

    public class RouteEntry
    {
        public string Destination { get; set; } = "";
        public string? Gateway { get; set; }
        public string Interface { get; set; } = "";
        public string? Protocol { get; set; }
        public string? Scope { get; set; }
        public int Metric { get; set; }
    }

    public class PortEntry
    {
        public string LocalAddress { get; set; } = "";
        public int Port { get; set; }
        public string Process { get; set; } = "";
        public string State { get; set; } = "";
    }

    public class UfwStatusInfo
    {
        public bool Active { get; set; }
        public string DefaultIncoming { get; set; } = "";
        public string DefaultOutgoing { get; set; } = "";
        public string DefaultForward { get; set; } = "";
        public List<UfwRule> Rules { get; set; } = [];
        public string RawOutput { get; set; } = "";
    }

    public class UfwRule
    {
        public int Number { get; set; }
        public string To { get; set; } = "";
        public string Action { get; set; } = "";
        public string Direction { get; set; } = "";
        public string From { get; set; } = "";
    }

    public class IptablesRule
    {
        public int LineNumber { get; set; }
        public long Packets { get; set; }
        public long Bytes { get; set; }
        public string Target { get; set; } = "";
        public string Protocol { get; set; } = "";
        public string In { get; set; } = "";
        public string Out { get; set; } = "";
        public string Source { get; set; } = "";
        public string Destination { get; set; } = "";
        public string Options { get; set; } = "";
    }
}