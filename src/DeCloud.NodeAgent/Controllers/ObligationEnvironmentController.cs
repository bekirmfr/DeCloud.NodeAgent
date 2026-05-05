using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Infrastructure.Services.CloudInit;
using DeCloud.Shared.Models;
using Microsoft.AspNetCore.Mvc;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Serves runtime-mutable environment variable values and scope policy
/// to system VMs over the local virbr0 bridge.
///
/// <para>
/// <b>Route:</b> <c>GET /api/obligations/{role}/environment</c>
/// </para>
///
/// <para>
/// Called periodically by <c>decloud-env-watcher.sh</c> running inside
/// each system VM. The watcher diffs the returned <c>values</c> against
/// its locally-cached environment file and applies the max-scope reaction
/// across any changed variables.
/// </para>
///
/// <para>
/// <b>Security:</b> virbr0-only (192.168.122.0/24). Same enforcement as
/// <c>ObligationStateController</c> and <c>ArtifactCacheController</c>.
/// No authentication — network isolation is the security boundary.
/// Dynamic variable values may include sensitive fields (e.g. WG keys);
/// virbr0 isolation ensures only local VMs can reach this endpoint.
/// </para>
///
/// <para>
/// <b>Phase 3.1 behaviour for relay:</b> relay declares no Dynamic variables,
/// so <c>values</c> and <c>scopes</c> are both empty. The watcher sees an
/// empty scope file on the VM and exits immediately each tick.
/// </para>
/// </summary>
[ApiController]
[Route("api/obligations")]
public class ObligationEnvironmentController : ControllerBase
{
    private static readonly IPNetwork VirbR0Network = IPNetwork.Parse("192.168.122.0/24");

    private static readonly JsonSerializerOptions TemplateJsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private readonly IObligationStateService _obligationState;
    private readonly INodeRelayConfigProvider _relayConfigProvider;
    private readonly INodeMetadataService _nodeMetadata;
    private readonly ILogger<ObligationEnvironmentController> _logger;

    public ObligationEnvironmentController(
        IObligationStateService obligationState,
        INodeRelayConfigProvider relayConfigProvider,
        INodeMetadataService nodeMetadata,
        ILogger<ObligationEnvironmentController> logger)
    {
        _obligationState = obligationState;
        _relayConfigProvider = relayConfigProvider;
        _nodeMetadata = nodeMetadata;
        _logger = logger;
    }

    // ── GET /api/obligations/{role}/environment ──────────────────────────────

    /// <summary>
    /// Returns the current Dynamic variable values and scope policy for the
    /// given system VM role.
    ///
    /// <para>Returns 404 if no template has been received for this role yet
    /// (node not yet registered / template not pushed). The watcher treats
    /// any non-200 response as a transient error and retries next tick.</para>
    /// </summary>
    [HttpGet("{role}/environment")]
    public async Task<IActionResult> GetEnvironment(string role, CancellationToken ct)
    {
        if (!EnforceVirbr0(out var callerIp))
        {
            _logger.LogWarning(
                "ObligationEnvironmentController: rejected {Role} request from {Ip} — not on virbr0",
                role, callerIp);
            return StatusCode(403, "Environment endpoint is only accessible from the VM bridge network.");
        }

        role = role.ToLowerInvariant();
        if (!IsKnownRole(role))
        {
            _logger.LogWarning(
                "ObligationEnvironmentController: unknown role '{Role}' from {Ip}", role, callerIp);
            return BadRequest($"Unknown system VM role: {role}");
        }

        // Load the system template from SQLite to get the Variables declaration.
        var templateJson = await _obligationState.GetSystemTemplateJsonAsync(role, ct);
        if (templateJson is null)
        {
            _logger.LogDebug(
                "ObligationEnvironmentController: no template for role '{Role}' — returning 404",
                role);
            return NotFound($"No system template stored for role '{role}'. " +
                            "The node agent will push it on next registration or heartbeat.");
        }

        SystemVmTemplate template;
        try
        {
            template = JsonSerializer.Deserialize<SystemVmTemplate>(templateJson, TemplateJsonOpts)
                       ?? throw new InvalidOperationException("Deserialized template is null.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "ObligationEnvironmentController: failed to deserialize template for role '{Role}'",
                role);
            return StatusCode(500, "Failed to read system template.");
        }

        var dynamicVars = template.Variables
            .Where(v => v.Kind == VariableKind.Dynamic)
            .ToList();

        var values = new SortedDictionary<string, string>(StringComparer.Ordinal);
        var scopes = new SortedDictionary<string, string>(StringComparer.Ordinal);

        // Pre-fetch relay config once per request — shared by any dynamic
        // that needs it (currently DHT_ADVERTISE_IP). Returns null on
        // transient unavailability; affected dynamics resolve to "" and
        // the watcher detects the transition via generation diff.
        // Skipped for relay role (relay is the relay).
        NodeRelayConfig? relayConfig = null;
        if (role != "relay")
            relayConfig = await _relayConfigProvider.TryGetAsync(role, ct);

        foreach (var variable in dynamicVars)
        {
            scopes[variable.Name] = variable.Scope switch
            {
                WatcherScope.Noop => "noop",
                WatcherScope.Reload => "reload",
                WatcherScope.Restart => "restart",
                _ => "restart",
            };

            // Inline resolution. Universe is small (2 entries for DHT;
            // ~2 more land in P3.3 for BlockStore). Resolver pattern would
            // earn nothing at this scale; revisit if it grows past ~10
            // cases or starts needing serious per-variable logic.
            values[variable.Name] = (role, variable.Name) switch
            {
                ("dht", "DHT_ADVERTISE_IP") => DeriveAdvertiseIp(relayConfig),
                ("dht", "DHT_BOOTSTRAP_PEERS") => "",  // TODO post-Phase-3: source from heartbeat if needed
                _ => ""
            };
        }

        var generation = ComputeGeneration(values);

        _logger.LogDebug(
            "ObligationEnvironmentController: served {Role} environment ({Count} dynamics, gen={Gen})",
            role, values.Count, generation);

        return Ok(new ObligationEnvironment
        {
            Values = values,
            Scopes = scopes,
            Generation = generation,
        });
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Compute a deterministic generation fingerprint from the values dict.
    /// Sort keys, serialise to compact JSON, SHA256, take first 16 hex chars.
    /// Stable across .NET versions because SortedDictionary + System.Text.Json
    /// with no custom converters produces deterministic output for string maps.
    /// </summary>
    internal static string ComputeGeneration(IReadOnlyDictionary<string, string> values)
    {
        // Copy into a SortedDictionary to guarantee key order before serialising.
        var sorted = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (var (k, v) in values)
            sorted[k] = v;

        var json = JsonSerializer.Serialize(sorted);
        var hashBytes = SHA256.HashData(Encoding.UTF8.GetBytes(json));
        return Convert.ToHexString(hashBytes).ToLowerInvariant()[..16];
    }

    /// <summary>
    /// Derives the bare IP address from a tunnel IP that may include a CIDR
    /// suffix (e.g. "10.30.0.248/24" → "10.30.0.248"). Returns "" when the
    /// relay config is null (transient unavailability — the watcher's
    /// generation-diff handles the eventual transition).
    /// </summary>
    private static string DeriveAdvertiseIp(NodeRelayConfig? cfg)
    {
        var tunnel = cfg?.TunnelIp ?? "";
        var slash = tunnel.IndexOf('/');
        return slash >= 0 ? tunnel[..slash] : tunnel;
    }

    private bool EnforceVirbr0(out string callerIp)
    {
        var remoteIp = HttpContext.Connection.RemoteIpAddress;
        if (remoteIp is null) { callerIp = "unknown"; return false; }

        if (remoteIp.IsIPv4MappedToIPv6)
            remoteIp = remoteIp.MapToIPv4();

        callerIp = remoteIp.ToString();
        return VirbR0Network.Contains(remoteIp);
    }

    private static bool IsKnownRole(string role) =>
        role is "relay" or "dht" or "blockstore";
}