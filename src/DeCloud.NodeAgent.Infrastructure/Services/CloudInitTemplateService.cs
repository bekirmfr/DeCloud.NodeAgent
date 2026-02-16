using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// Service for loading and processing cloud-init templates for specialized VM types.
/// Handles template variable replacement and secret generation.
/// </summary>
public interface ICloudInitTemplateService
{
    /// <summary>
    /// Load and process a cloud-init template for the specified VM type.
    /// Returns the processed cloud-init YAML with all variables replaced.
    /// </summary>
    Task<string> ProcessTemplateAsync(
        VmType vmType,
        VmSpec spec,
        Dictionary<string, string>? additionalVariables = null,
        CancellationToken ct = default);

    /// <summary>
    /// Check if a template exists for the given VM type
    /// </summary>
    bool HasTemplate(VmType vmType);
}

/// <summary>
/// Template variables that can be used in cloud-init templates
/// </summary>
public class CloudInitTemplateVariables
{
    // VM Identity
    public string VmId { get; set; } = "";
    public string VmName { get; set; } = "";
    public string Hostname { get; set; } = "";

    // Network
    public string PublicIp { get; set; } = "";
    public string PrivateIp { get; set; } = "";

    // WireGuard (for relay VMs)
    public string WireGuardPrivateKey { get; set; } = "";
    public string WireGuardPublicKey { get; set; } = "";
    public string WireGuardListenPort { get; set; } = "51820";
    public string WireGuardAddress { get; set; } = "10.20.0.1/16";

    // Attestation Agent
    public string AttestationAgent { get; set; } = "";

    // Security
    public string SshPublicKey { get; set; } = "";
    public string AdminPassword { get; set; } = "";

    // Custom variables (for future VM types)
    public Dictionary<string, string> Custom { get; set; } = new();

    /// <summary>
    /// Convert to dictionary for template replacement
    /// </summary>
    public Dictionary<string, string> ToDictionary()
    {
        var dict = new Dictionary<string, string>
        {
            ["__VM_ID__"] = VmId,
            ["__VM_NAME__"] = VmName,
            ["__HOSTNAME__"] = Hostname,
            ["__PUBLIC_IP__"] = PublicIp,
            ["__PRIVATE_IP__"] = PrivateIp,
            ["__WIREGUARD_PRIVATE_KEY__"] = WireGuardPrivateKey,
            ["__WIREGUARD_PUBLIC_KEY__"] = WireGuardPublicKey,
            ["__WIREGUARD_LISTEN_PORT__"] = WireGuardListenPort,
            ["__WIREGUARD_ADDRESS__"] = WireGuardAddress,
            ["__SSH_PUBLIC_KEY__"] = SshPublicKey,
            ["__ADMIN_PASSWORD__"] = AdminPassword,
            ["__ATTESTATION_AGENT_BASE64__"] = AttestationAgent,
        };

        // Add custom variables
        foreach (var (key, value) in Custom)
        {
            dict[$"__{key.ToUpper()}__"] = value;
        }

        return dict;
    }
}

public class CloudInitTemplateService : ICloudInitTemplateService
{
    private readonly ICommandExecutor _executor;
    private readonly IVmManager _vmManager;
    private readonly ILogger<CloudInitTemplateService> _logger;
    private readonly string _templateBasePath;

    // Cache for loaded templates
    private readonly Dictionary<VmType, string> _templateCache = new();
    private readonly Dictionary<string, string> _externalTemplateCache = new();
    private readonly SemaphoreSlim _cacheLock = new(1, 1);

    public CloudInitTemplateService(
        ICommandExecutor executor,
        IVmManager vmManager,
        ILogger<CloudInitTemplateService> logger)
    {
        _executor = executor;
        _vmManager = vmManager;
        _logger = logger;

        // Templates are embedded in the NodeAgent assembly or loaded from disk
        // For now, we'll use a known path relative to the application
        _templateBasePath = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "CloudInit",
            "Templates");

        EnsureTemplateDirectory();
    }

    private void EnsureTemplateDirectory()
    {
        if (!Directory.Exists(_templateBasePath))
        {
            Directory.CreateDirectory(_templateBasePath);
            _logger.LogInformation(
                "Created cloud-init template directory at {Path}",
                _templateBasePath);
        }
    }

    public bool HasTemplate(VmType vmType)
    {
        var templatePath = GetTemplatePath(vmType);
        return File.Exists(templatePath);
    }

    public async Task<string> ProcessTemplateAsync(
        VmType vmType,
        VmSpec spec,
        Dictionary<string, string>? additionalVariables = null,
        CancellationToken ct = default)
    {
        _logger.LogInformation(
            "Processing cloud-init template for VM type {VmType}",
            vmType);

        // Load template
        var template = await LoadTemplateAsync(vmType, ct);

        if (string.IsNullOrEmpty(template))
        {
            throw new InvalidOperationException(
                $"No cloud-init template found for VM type {vmType}");
        }

        switch(vmType)
        {
            case VmType.General:
                template = await InjectGeneralExternalTemplatesAsync(template, ct);
                break;
            case VmType.Dht:
                template = await InjectDhtExternalTemplatesAsync(template, ct);
                break;
            case VmType.Inference:
                // Future VM types can have their own external templates if needed
                break;
            case VmType.Relay:
                template = await InjectRelayExternalTemplatesAsync(template, ct);
                break;
            default:

                break;
        }

        // Build template variables
        var variables = await BuildTemplateVariablesAsync(vmType, spec, ct);

        // Get base replacements dictionary first
        var replacements = variables.ToDictionary();

        // Merge additionalVariables DIRECTLY (don't go through Custom)
        if (additionalVariables != null)
        {
            foreach (var (key, value) in additionalVariables)
            {
                // Keys already have __ prefix/suffix from LibvirtVmManager
                replacements[key] = value;
            }
        }

        // Replace variables in template
        var result = template;
        foreach (var (placeholder, value) in replacements)
        {
            result = result.Replace(placeholder, value);
        }

        // Validate no unreplaced placeholders remain
        if (result.Contains("__"))
        {
            var unreplaced = System.Text.RegularExpressions.Regex.Matches(
                result, @"__[A-Z_]+__")
                .Select(m => m.Value)
                .Distinct()
                .ToList();

            if (unreplaced.Any())
            {
                _logger.LogWarning(
                    "Template has unreplaced placeholders: {Placeholders}",
                    string.Join(", ", unreplaced));
            }
            else
            {
                _logger.LogDebug("✓ All placeholders successfully replaced");
            }
        }

        _logger.LogInformation(
            "Cloud-init template processed for VM {VmId} (type: {VmType})",
            spec.Id, vmType);

        return result;
    }

    /// <summary>
    /// Load and inject external template files for general VMs with proper YAML indentation
    /// </summary>
    private async Task<string> InjectGeneralExternalTemplatesAsync(
        string template,
        CancellationToken ct)
    {
        _logger.LogInformation("Loading external general VM templates...");

        try
        {
            // Load external template files
            var indexHtml = await LoadExternalTemplateAsync("index.html", "general-vm", ct);
            var generalApi = await LoadExternalTemplateAsync("general-api.py", "general-vm", ct);

            // Replace placeholders with properly indented content
            var result = ReplaceWithIndentation(template, "__INDEX_HTML__", indexHtml);
            result = ReplaceWithIndentation(result, "__WELCOME_SERVER_PY__", generalApi);

            _logger.LogInformation(
                "✓ Injected external templates: " +
                "HTML ({HtmlSize} chars), API ({ApiSize} chars)",
                indexHtml.Length, generalApi.Length);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load external general VM templates");
            throw new InvalidOperationException(
                "Failed to load general VM templates. Ensure all template files exist in " +
                $"{Path.Combine(_templateBasePath, "relay-vm")}/", ex);
        }
    }

    /// <summary>
    /// Load and inject external template files for relay VMs with proper YAML indentation
    /// </summary>
    private async Task<string> InjectRelayExternalTemplatesAsync(
        string template,
        CancellationToken ct)
    {
        _logger.LogInformation("Loading external relay VM templates...");

        try
        {
            // Load external template files
            var dashboardHtml = await LoadExternalTemplateAsync("dashboard.html", "relay-vm", ct);
            var dashboardCss = await LoadExternalTemplateAsync("dashboard.css", "relay-vm", ct);
            var dashboardJs = await LoadExternalTemplateAsync("dashboard.js", "relay-vm", ct);
            var relayApi = await LoadExternalTemplateAsync("relay-api.py", "relay-vm", ct);
            var relayHttpProxyContent = await LoadExternalTemplateAsync("relay-http-proxy.py", "relay-vm", ct);
            var notifyNatReady = await LoadExternalTemplateAsync("notify-nat-ready.sh", "relay-vm", ct);

            // Replace placeholders with properly indented content
            var result = ReplaceWithIndentation(template, "__DASHBOARD_HTML__", dashboardHtml);
            result = ReplaceWithIndentation(result, "__DASHBOARD_CSS__", dashboardCss);
            result = ReplaceWithIndentation(result, "__DASHBOARD_JS__", dashboardJs);
            result = ReplaceWithIndentation(result, "__RELAY_API__", relayApi);
            result = ReplaceWithIndentation(result, "__RELAY_HTTP_PROXY__", relayHttpProxyContent);
            result = ReplaceWithIndentation(result, "__NOTIFY_NAT_READY__", notifyNatReady);

            _logger.LogInformation(
                "✓ Injected external templates: " +
                "HTML ({HtmlSize} chars), CSS ({CssSize} chars), " +
                "JS ({JsSize} chars), API ({ApiSize} chars)",
                dashboardHtml.Length, dashboardCss.Length,
                dashboardJs.Length, relayApi.Length);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load external relay VM templates");
            throw new InvalidOperationException(
                "Failed to load relay VM templates. Ensure all template files exist in " +
                $"{Path.Combine(_templateBasePath, "relay-vm")}/", ex);
        }
    }

    /// <summary>
    /// Load and inject external DHT VM template files (health check, ready callback).
    /// Mirrors InjectRelayExternalTemplatesAsync pattern.
    /// </summary>
    private async Task<string> InjectDhtExternalTemplatesAsync(
        string template,
        CancellationToken ct)
    {
        _logger.LogInformation("Loading external DHT VM templates...");

        try
        {
            var healthCheck = await LoadExternalTemplateAsync("dht-health-check.sh", "dht-vm", ct);
            var notifyReady = await LoadExternalTemplateAsync("dht-notify-ready.sh", "dht-vm", ct);
            var bootstrapPoll = await LoadExternalTemplateAsync("dht-bootstrap-poll.sh", "dht-vm", ct);
            var dashboardPy = await LoadExternalTemplateAsync("dht-dashboard.py", "dht-vm", ct);
            var dashboardHtml = await LoadExternalTemplateAsync("dashboard.html", "dht-vm", ct);
            var dashboardCss = await LoadExternalTemplateAsync("dashboard.css", "dht-vm", ct);
            var dashboardJs = await LoadExternalTemplateAsync("dashboard.js", "dht-vm", ct);
            var wgMeshEnroll = await LoadExternalTemplateAsync("wg-mesh-enroll.sh", "shared", ct);

            var result = ReplaceWithIndentation(template, "__DHT_HEALTH_CHECK__", healthCheck);
            result = ReplaceWithIndentation(result, "__DHT_NOTIFY_READY__", notifyReady);
            result = ReplaceWithIndentation(result, "__DHT_BOOTSTRAP_POLL__", bootstrapPoll);
            result = ReplaceWithIndentation(result, "__DHT_DASHBOARD_PY__", dashboardPy);
            result = ReplaceWithIndentation(result, "__DHT_DASHBOARD_HTML__", dashboardHtml);
            result = ReplaceWithIndentation(result, "__DHT_DASHBOARD_CSS__", dashboardCss);
            result = ReplaceWithIndentation(result, "__DHT_DASHBOARD_JS__", dashboardJs);
            result = ReplaceWithIndentation(result, "__WG_MESH_ENROLL__", wgMeshEnroll);

            _logger.LogInformation(
                "Injected DHT external templates: health-check ({HealthSize} chars), notify-ready ({ReadySize} chars), " +
                "bootstrap-poll ({PollSize} chars), dashboard-server ({DashPy} chars), " +
                "dashboard ({DashHtml}+{DashCss}+{DashJs} chars), wg-enroll ({WgSize} chars)",
                healthCheck.Length, notifyReady.Length, bootstrapPoll.Length, dashboardPy.Length,
                dashboardHtml.Length, dashboardCss.Length, dashboardJs.Length, wgMeshEnroll.Length);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load external DHT VM templates");
            throw new InvalidOperationException(
                "Failed to load DHT VM templates. Ensure all template files exist in " +
                $"{Path.Combine(_templateBasePath, "dht-vm")}/", ex);
        }
    }

    /// <summary>
    /// Replace placeholder with content, preserving YAML indentation
    /// </summary>
    private string ReplaceWithIndentation(string template, string placeholder, string content)
    {
        // Find all occurrences of the placeholder and detect indentation
        var regex = new Regex($@"^(\s*){Regex.Escape(placeholder)}$", RegexOptions.Multiline);

        return regex.Replace(template, match =>
        {
            var indentation = match.Groups[1].Value;

            // Split content into lines
            var lines = content.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

            // First line doesn't need extra indentation (it inherits from placeholder position)
            // All subsequent lines need the same indentation
            for (int i = 1; i < lines.Length; i++)
            {
                lines[i] = indentation + lines[i];
            }

            return indentation + string.Join("\n", lines);
        });
    }

    /// <summary>
    /// Load an external template file from the vm subdirectory
    /// </summary>
    private async Task<string> LoadExternalTemplateAsync(
        string filename,
        string dirName,
        CancellationToken ct)
    {
        await _cacheLock.WaitAsync(ct);
        try
        {
            // Cache key must include directory to avoid collisions between
            // VM types with identically-named files (e.g., relay-vm/dashboard.html
            // vs dht-vm/dashboard.html). Without this, whichever VM type is
            // processed first poisons the cache for the other.
            var cacheKey = $"{dirName}/{filename}";

            // Check cache first
            if (_externalTemplateCache.TryGetValue(cacheKey, out var cached))
            {
                _logger.LogDebug("Using cached external template: {CacheKey}", cacheKey);
                return cached;
            }

            // Load from disk
            var templatePath = Path.Combine(_templateBasePath, dirName, filename);

            _logger.LogDebug("Loading external template: {Path}", templatePath);

            if (!File.Exists(templatePath))
            {
                throw new FileNotFoundException(
                    $"External template file not found: {filename}. " +
                    $"Expected at: {templatePath}",
                    templatePath);
            }

            var content = await File.ReadAllTextAsync(templatePath, ct);

            // Cache for future use
            _externalTemplateCache[cacheKey] = content;

            _logger.LogDebug(
                "Loaded external template: {CacheKey} ({Length} chars)",
                cacheKey, content.Length);

            return content;
        }
        finally
        {
            _cacheLock.Release();
        }
    }

    private async Task<string> LoadTemplateAsync(VmType vmType, CancellationToken ct)
    {
        await _cacheLock.WaitAsync(ct);
        try
        {
            // Check cache first
            if (_templateCache.TryGetValue(vmType, out var cached))
            {
                return cached;
            }

            var templatePath = GetTemplatePath(vmType);

            if (!File.Exists(templatePath))
            {
                _logger.LogWarning(
                    "Template not found for VM type {VmType} at {Path}",
                    vmType, templatePath);
                return string.Empty;
            }

            var template = await File.ReadAllTextAsync(templatePath, ct);

            // Cache the template
            _templateCache[vmType] = template;

            _logger.LogDebug(
                "Loaded cloud-init template for {VmType} from {Path}",
                vmType, templatePath);

            return template;
        }
        finally
        {
            _cacheLock.Release();
        }
    }

    private string GetTemplatePath(VmType vmType)
    {
        var fileName = vmType switch
        {
            VmType.General => "general-vm-cloudinit.yaml",
            VmType.Relay => "relay-vm-cloudinit.yaml",
            VmType.Dht => "dht-vm-cloudinit.yaml",
            VmType.Inference => "inference-vm-cloudinit.yaml",
            _ => "general-vm-cloudinit.yaml"
        };

        return Path.Combine(_templateBasePath, fileName);
    }

    private async Task<CloudInitTemplateVariables> BuildTemplateVariablesAsync(
        VmType vmType,
        VmSpec spec,
        CancellationToken ct)
    {
        var variables = new CloudInitTemplateVariables
        {
            VmId = spec.Id,
            VmName = spec.Name,
            Hostname = spec.Name,
            SshPublicKey = spec.SshPublicKey ?? "",
            AdminPassword = GenerateSecurePassword(),
        };
        // Determine target architecture
        var architecture = GetTargetArchitecture(spec);

        // Load correct attestation agent binary based on architecture
        var agentFileName = $"decloud-agent-{architecture}.b64";
        var agentBinaryPath = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "CloudInit", "Templates",
            agentFileName);

        if (File.Exists(agentBinaryPath))
        {
            variables.AttestationAgent =
                await File.ReadAllTextAsync(agentBinaryPath, ct);
        }

        // VM type-specific variable generation
        switch (vmType)
        {
            case VmType.Relay:
                await PopulateRelayVariablesAsync(variables, spec, ct);
                break;

            case VmType.Dht:
                await PopulateDhtVariablesAsync(variables, spec, ct);
                break;

            case VmType.Inference:
                await PopulateInferenceVariablesAsync(variables, spec, ct);
                break;

            case VmType.General:
                // General VMs don't need special variables
                // All core variables are already set
                break;
        }

        return variables;
    }

    private string GetTargetArchitecture(VmSpec spec)
    {
        // Check if architecture is specified in spec (labels or dedicated field)
        if (spec.Labels?.TryGetValue("architecture", out var arch) == true)
        {
            return arch.ToLowerInvariant() switch
            {
                "arm64" or "aarch64" or "arm" => "arm64",
                "x86_64" or "amd64" or "x64" => "amd64",
                _ => "amd64" // Default to amd64
            };
        }

        // Default to amd64 (most common)
        return "amd64";
    }

    private async Task PopulateRelayVariablesAsync(
        CloudInitTemplateVariables variables,
        VmSpec spec,
        CancellationToken ct)
    {
        string privateKey;

        // Check if orchestrator provided a WireGuard private key via Labels
        if (spec.Labels?.TryGetValue("wireguard-private-key", out var providedKey) == true
            && !string.IsNullOrWhiteSpace(providedKey))
        {
            _logger.LogInformation(
                "Using WireGuard private key from orchestrator for relay VM {VmId}",
                spec.Id);

            privateKey = providedKey;
        }
        else
        {
            _logger.LogInformation(
                "No WireGuard key provided - generating new keypair for relay VM {VmId}",
                spec.Id);

            try
            {
                // Generate WireGuard private key
                var genKeyResult = await _executor.ExecuteAsync("wg", "genkey", ct);

                if (!genKeyResult.Success)
                {
                    throw new InvalidOperationException(
                        $"Failed to generate WireGuard private key: {genKeyResult.StandardError}");
                }

                privateKey = genKeyResult.StandardOutput.Trim();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate WireGuard keys for relay VM {VmId}", spec.Id);
                throw;
            }
        }

        variables.WireGuardPrivateKey = privateKey;

        // Derive public key from private key
        try
        {
            var pubKeyResult = await _executor.ExecuteAsync(
                "sh",
                $"-c \"echo '{privateKey}' | wg pubkey\"",
                ct);

            if (!pubKeyResult.Success)
            {
                throw new InvalidOperationException(
                    $"Failed to generate WireGuard public key: {pubKeyResult.StandardError}");
            }

            variables.WireGuardPublicKey = pubKeyResult.StandardOutput.Trim();

            _logger.LogInformation(
                "Configured WireGuard keypair for relay VM {VmId} (pubkey: {PubKey})",
                spec.Id,
                variables.WireGuardPublicKey.Substring(0, Math.Min(12, variables.WireGuardPublicKey.Length)) + "...");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to derive WireGuard public key for relay VM {VmId}", spec.Id);
            throw;
        }
    }

    private async Task PopulateDhtVariablesAsync(
        CloudInitTemplateVariables variables,
        VmSpec spec,
        CancellationToken ct)
    {
        // DHT libp2p configuration — ports and addresses come from orchestrator labels
        variables.Custom["DHT_LISTEN_PORT"] = spec.Labels?.GetValueOrDefault("dht-listen-port", "4001") ?? "4001";
        variables.Custom["DHT_API_PORT"] = spec.Labels?.GetValueOrDefault("dht-api-port", "5080") ?? "5080";
        variables.Custom["DHT_BOOTSTRAP_PEERS"] = spec.Labels?.GetValueOrDefault("dht-bootstrap-peers") ?? "";
        variables.Custom["DHT_REGION"] = spec.Labels?.GetValueOrDefault("node-region") ?? "default";

        // WireGuard mesh enrollment variables (passed from orchestrator via labels)
        variables.Custom["WG_RELAY_ENDPOINT"] = spec.Labels?.GetValueOrDefault("wg-relay-endpoint") ?? "";
        variables.Custom["WG_RELAY_PUBKEY"] = spec.Labels?.GetValueOrDefault("wg-relay-pubkey") ?? "";
        variables.Custom["WG_TUNNEL_IP"] = spec.Labels?.GetValueOrDefault("wg-tunnel-ip") ?? "";
        variables.Custom["WG_RELAY_API"] = spec.Labels?.GetValueOrDefault("wg-relay-api") ?? "";

        // Warn if WireGuard config is incomplete
        if (string.IsNullOrEmpty(variables.Custom["WG_RELAY_ENDPOINT"]) ||
            string.IsNullOrEmpty(variables.Custom["WG_RELAY_PUBKEY"]) ||
            string.IsNullOrEmpty(variables.Custom["WG_TUNNEL_IP"]) ||
            string.IsNullOrEmpty(variables.Custom["WG_RELAY_API"]))
        {
            _logger.LogWarning(
                "DHT VM {VmId} has incomplete WireGuard mesh config — " +
                "endpoint={Endpoint}, pubkey={PubKey}, tunnelIp={TunnelIp}, api={Api}",
                spec.Id,
                variables.Custom["WG_RELAY_ENDPOINT"],
                variables.Custom["WG_RELAY_PUBKEY"] is { Length: > 12 } pk ? pk[..12] + "..." : "(empty)",
                variables.Custom["WG_TUNNEL_IP"],
                variables.Custom["WG_RELAY_API"]);
        }

        // Bootstrap poll: orchestrator URL and auth token (relay callback pattern).
        // The orchestrator passes the auth token via labels when deploying the DHT VM.
        // The DHT VM uses it to compute HMAC(auth_token, nodeId:vmId) for direct
        // orchestrator authentication — same pattern as relay's notify-orchestrator.sh
        // uses HMAC(wireguard_private_key, nodeId:vmId).
        variables.Custom["DHT_AUTH_TOKEN"] = spec.Labels?.GetValueOrDefault("dht-auth-token") ?? "";
        variables.Custom["DHT_API_TOKEN"] = spec.Labels?.GetValueOrDefault("dht-api-token") ?? "";

        if (string.IsNullOrEmpty(variables.Custom["DHT_AUTH_TOKEN"]))
        {
            _logger.LogWarning(
                "DHT VM {VmId} has no dht-auth-token label — " +
                "bootstrap poll service will not be able to authenticate with orchestrator",
                spec.Id);
        }

        // Load architecture-specific DHT binary (gzip+base64 encoded, pre-built via build.sh)
        var architecture = GetTargetArchitecture(spec);
        var dhtBinaryGzB64 = await LoadDhtBinaryAsync(architecture, ct);
        variables.Custom["DHT_NODE_BINARY_GZ_BASE64"] = dhtBinaryGzB64;

        // =====================================================
        // WireGuard mesh variables — resolve from local relay VM
        // =====================================================
        // The DHT VM joins the relay's WireGuard mesh using the
        // reusable wg-mesh-enroll.sh building block.
        await PopulateWireGuardMeshVariablesAsync(variables, spec, ct);

        _logger.LogInformation(
            "Configured DHT variables for VM {VmId}: listenPort={ListenPort}, apiPort={ApiPort}, " +
            "advertiseIp={AdvIP}, arch={Arch}, binarySize={BinaryKB}KB (gz+b64), " +
            "bootstrapPeers={Peers}, authToken={HasToken}",
            spec.Id,
            variables.Custom["DHT_LISTEN_PORT"],
            variables.Custom["DHT_API_PORT"],
            variables.Custom.GetValueOrDefault("WG_TUNNEL_IP", ""),
            architecture,
            dhtBinaryGzB64.Length / 1024,
            string.IsNullOrEmpty(variables.Custom["DHT_BOOTSTRAP_PEERS"]) ? "(none — first node)" : "present",
            string.IsNullOrEmpty(variables.Custom["DHT_AUTH_TOKEN"]) ? "missing" : "present");
    }

    /// <summary>
    /// Resolve WireGuard mesh variables from the local relay VM.
    /// This is a generic building block — any VM type can use it
    /// to join the relay's WireGuard network.
    /// </summary>
    private async Task PopulateWireGuardMeshVariablesAsync(
        CloudInitTemplateVariables variables,
        VmSpec spec,
        CancellationToken ct)
    {
        // Allow explicit override via orchestrator labels
        var tunnelIp = spec.Labels?.GetValueOrDefault("wg-tunnel-ip") ?? "";
        var relayEndpoint = spec.Labels?.GetValueOrDefault("wg-relay-endpoint") ?? "";
        var relayPubkey = spec.Labels?.GetValueOrDefault("wg-relay-pubkey") ?? "";
        var relayApi = spec.Labels?.GetValueOrDefault("wg-relay-api") ?? "";

        // Auto-resolve from local relay VM if not explicitly provided
        if (string.IsNullOrEmpty(relayEndpoint) || string.IsNullOrEmpty(relayPubkey))
        {
            var relayInfo = await ResolveLocalRelayInfoAsync(ct);
            if (relayInfo != null)
            {
                if (string.IsNullOrEmpty(relayEndpoint))
                    relayEndpoint = $"{relayInfo.Value.Ip}:51820";
                if (string.IsNullOrEmpty(relayPubkey))
                    relayPubkey = relayInfo.Value.PublicKey;
                if (string.IsNullOrEmpty(relayApi))
                    relayApi = $"http://{relayInfo.Value.Ip}:8080";
                if (string.IsNullOrEmpty(tunnelIp))
                    tunnelIp = relayInfo.Value.DhtTunnelIp;

                _logger.LogInformation(
                    "Resolved relay WireGuard info: endpoint={Endpoint}, pubkey={PubKey}..., " +
                    "tunnelIp={TunnelIp}",
                    relayEndpoint,
                    relayPubkey.Length > 12 ? relayPubkey[..12] : relayPubkey,
                    tunnelIp);
            }
            else
            {
                _logger.LogWarning(
                    "No relay VM found on this host — DHT VM will not have WireGuard mesh connectivity. " +
                    "Provide wg-relay-endpoint, wg-relay-pubkey, and wg-tunnel-ip labels explicitly.");
            }
        }

        variables.Custom["WG_TUNNEL_IP"] = tunnelIp;
        variables.Custom["WG_RELAY_ENDPOINT"] = relayEndpoint;
        variables.Custom["WG_RELAY_PUBKEY"] = relayPubkey;
        variables.Custom["WG_RELAY_API"] = relayApi;
    }

    /// <summary>
    /// Find the relay VM running on this host and extract its WireGuard info.
    /// Returns null if no relay VM exists.
    /// </summary>
    private async Task<(string Ip, string PublicKey, string DhtTunnelIp)?> ResolveLocalRelayInfoAsync(
        CancellationToken ct)
    {
        try
        {
            var allVms = await _vmManager.GetAllVmsAsync(ct);
            var relayVm = allVms.FirstOrDefault(v =>
                v.Spec.VmType == VmType.Relay &&
                v.State is VmState.Running or VmState.Ready);

            if (relayVm == null)
                return null;

            var relayIp = await _vmManager.GetVmIpAddressAsync(relayVm.VmId, ct);
            if (string.IsNullOrEmpty(relayIp))
                return null;

            // Get relay's WireGuard public key via its API
            var relayPubkey = "";
            try
            {
                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
                var response = await client.GetStringAsync(
                    $"http://{relayIp}:8080/api/relay/status", ct);
                var json = System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(response);
                if (json.TryGetProperty("public_key", out var pk))
                    relayPubkey = pk.GetString() ?? "";
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Could not fetch relay public key from {Ip} — trying disk fallback",
                    relayIp);
            }

            // Fallback: read public key from relay VM's stored config
            if (string.IsNullOrEmpty(relayPubkey))
            {
                var pubkeyPath = Path.Combine("/var/lib/decloud/vms", relayVm.VmId, "wg-pubkey");
                if (File.Exists(pubkeyPath))
                    relayPubkey = (await File.ReadAllTextAsync(pubkeyPath, ct)).Trim();
            }

            // Determine DHT tunnel IP: use .253 on the relay's subnet
            // Relay address is 10.20.{subnet}.254, so DHT gets .253
            var relaySubnet = relayVm.Spec.Labels?.GetValueOrDefault("relay-subnet") ?? "";
            var dhtTunnelIp = !string.IsNullOrEmpty(relaySubnet)
                ? $"10.20.{relaySubnet}.253"
                : "";

            return (relayIp, relayPubkey, dhtTunnelIp);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to resolve local relay VM info");
            return null;
        }
    }

    /// <summary>
    /// Load the pre-built DHT binary (gzip+base64-encoded) from disk.
    /// If the .gz.b64 file doesn't exist, attempts to build it from the Go source
    /// included in the deployment (dht-node-src/build.sh).
    /// </summary>
    private async Task<string> LoadDhtBinaryAsync(string architecture, CancellationToken ct)
    {
        var fileName = $"dht-node-{architecture}.gz.b64";
        var filePath = Path.Combine(_templateBasePath, "dht-vm", fileName);

        if (!File.Exists(filePath))
        {
            _logger.LogWarning(
                "DHT binary not found at {Path} — attempting to build from source...",
                filePath);

            await BuildDhtBinaryFromSourceAsync(architecture, ct);
        }

        if (!File.Exists(filePath))
        {
            _logger.LogError(
                "DHT binary not found at {Path} after build attempt. " +
                "Ensure Go 1.23+ is installed on this node, or pre-build with: " +
                "bash CloudInit/Templates/dht-vm/dht-node-src/build.sh",
                filePath);
            throw new FileNotFoundException(
                $"DHT binary not found at: {filePath}. " +
                "Install Go 1.23+ and retry, or pre-build with: " +
                "bash CloudInit/Templates/dht-vm/dht-node-src/build.sh",
                filePath);
        }

        var content = await File.ReadAllTextAsync(filePath, ct);

        if (string.IsNullOrWhiteSpace(content))
            throw new InvalidOperationException($"DHT binary file is empty: {filePath}");

        _logger.LogInformation(
            "Loaded DHT binary: {Path} ({SizeKB}KB gz+b64)",
            filePath, content.Length / 1024);

        return content.Trim();
    }

    /// <summary>
    /// Build the DHT binary from Go source on the current node.
    /// The Go source and build.sh are included in the deployment output.
    /// Requires Go 1.23+ installed on the host.
    /// </summary>
    private async Task BuildDhtBinaryFromSourceAsync(string architecture, CancellationToken ct)
    {
        var buildScript = Path.Combine(_templateBasePath, "dht-vm", "dht-node-src", "build.sh");

        if (!File.Exists(buildScript))
        {
            _logger.LogError(
                "DHT build script not found at {Path}. " +
                "The Go source files may not have been included in the deployment.",
                buildScript);
            return;
        }

        _logger.LogInformation(
            "Building DHT binary for {Architecture} from Go source at {ScriptPath}...",
            architecture, buildScript);

        try
        {
            // Go build with dependency download can take several minutes on first run
            var buildTimeout = TimeSpan.FromMinutes(5);

            var result = await _executor.ExecuteAsync(
                "bash",
                $"{buildScript} {architecture}",
                buildTimeout,
                ct);

            if (result.Success)
            {
                _logger.LogInformation(
                    "DHT binary built successfully for {Architecture}",
                    architecture);
            }
            else
            {
                _logger.LogError(
                    "Failed to build DHT binary for {Architecture}. Exit code: {ExitCode}. " +
                    "Stderr: {Error}. Ensure Go 1.23+ is installed (apt install golang-go or via https://go.dev/dl/)",
                    architecture,
                    result.ExitCode,
                    result.StandardError);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Exception while building DHT binary for {Architecture}. " +
                "Ensure Go 1.23+ is installed on this node.",
                architecture);
        }
    }

    private Task PopulateInferenceVariablesAsync(
        CloudInitTemplateVariables variables,
        VmSpec spec,
        CancellationToken ct)
    {
        // Inference-specific configuration
        variables.Custom["INFERENCE_PORT"] = "8000";
        variables.Custom["MODEL_CACHE_PATH"] = "/var/lib/decloud-models";
        variables.Custom["INFERENCE_MAX_BATCH_SIZE"] = "32";
        variables.Custom["INFERENCE_MAX_CONCURRENT"] = "4";
        variables.Custom["INFERENCE_GPU_ENABLED"] = "true";

        _logger.LogInformation(
            "Configured Inference variables for VM {VmId}: port={Port}, models={Path}, gpu={Gpu}",
            spec.Id,
            variables.Custom["INFERENCE_PORT"],
            variables.Custom["MODEL_CACHE_PATH"],
            variables.Custom["INFERENCE_GPU_ENABLED"]);

        return Task.CompletedTask;
    }

    private string GenerateSecurePassword(int length = 16)
    {
        // Generate cryptographically secure random password
        const string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*";
        var random = System.Security.Cryptography.RandomNumberGenerator.Create();
        var bytes = new byte[length];
        random.GetBytes(bytes);

        return new string(bytes.Select(b => chars[b % chars.Length]).ToArray());
    }

    /// <summary>
    /// Read and parse a WireGuard config file from the host filesystem.
    /// Used as CGNAT fallback when no local relay VM is available —
    /// reads /etc/wireguard/wg-relay.conf to discover relay info.
    /// </summary>
    /// <param name="configFileName">Config file name (e.g., "wg-relay.conf")</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Parsed config or null if file doesn't exist / can't be parsed</returns>
    public async Task<HostWireGuardConfig?> ReadHostWireGuardConfigAsync(
        string configFileName, CancellationToken ct)
    {
        var configPath = $"/etc/wireguard/{configFileName}";

        try
        {
            // Read the config file via shell command (runs as root on the node agent host)
            var result = await _executor.ExecuteAsync("cat", configPath, ct);

            if (!result.Success)
            {
                _logger.LogDebug(
                    "WireGuard config not found at {Path}: {Error}",
                    configPath, result.StandardError);
                return null;
            }

            var configText = result.StandardOutput;
            if (string.IsNullOrWhiteSpace(configText))
            {
                _logger.LogWarning("WireGuard config at {Path} is empty", configPath);
                return null;
            }

            return ParseWireGuardConfig(configText, configPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading WireGuard config from {Path}", configPath);
            return null;
        }
    }

    /// <summary>
    /// Parse a standard WireGuard config file format to extract relay info.
    /// 
    /// Expected format:
    ///   [Interface]
    ///   Address = 10.20.1.2/24
    ///   PrivateKey = ...
    ///
    ///   [Peer]
    ///   PublicKey = ...
    ///   Endpoint = 1.2.3.4:51820
    ///   AllowedIPs = 10.20.1.0/24
    /// </summary>
    private HostWireGuardConfig? ParseWireGuardConfig(string configText, string sourcePath)
    {
        var config = new HostWireGuardConfig();

        // Parse Address from [Interface] section (e.g., "10.20.1.2/24")
        var addressMatch = Regex.Match(configText,
            @"Address\s*=\s*(\d+\.\d+\.\d+\.\d+)/(\d+)",
            RegexOptions.Multiline);

        if (addressMatch.Success)
        {
            config.TunnelIp = addressMatch.Groups[1].Value;

            // Extract subnet number from 10.20.X.Y
            var ipParts = config.TunnelIp.Split('.');
            if (ipParts.Length == 4 && ipParts[0] == "10" && ipParts[1] == "20")
            {
                config.Subnet = ipParts[2];
            }
        }

        // Parse PublicKey from [Peer] section 
        var pubKeyMatch = Regex.Match(configText,
            @"\[Peer\][\s\S]*?PublicKey\s*=\s*([A-Za-z0-9+/=]+)",
            RegexOptions.Multiline);

        if (pubKeyMatch.Success)
        {
            config.PublicKey = pubKeyMatch.Groups[1].Value.Trim();
        }

        // Parse Endpoint from [Peer] section
        var endpointMatch = Regex.Match(configText,
            @"\[Peer\][\s\S]*?Endpoint\s*=\s*([^\s]+)",
            RegexOptions.Multiline);

        if (endpointMatch.Success)
        {
            config.Endpoint = endpointMatch.Groups[1].Value.Trim();

            // Extract relay IP from endpoint (ip:port)
            var colonIdx = config.Endpoint.LastIndexOf(':');
            if (colonIdx > 0)
            {
                config.RelayIp = config.Endpoint.Substring(0, colonIdx);
            }
        }

        // Validate we got the essential fields
        if (string.IsNullOrEmpty(config.Endpoint) ||
            string.IsNullOrEmpty(config.PublicKey) ||
            string.IsNullOrEmpty(config.TunnelIp))
        {
            _logger.LogWarning(
                "WireGuard config at {Path} is incomplete — " +
                "Endpoint={Endpoint}, PublicKey={PubKey}, TunnelIp={TunnelIp}",
                sourcePath,
                config.Endpoint ?? "(missing)",
                config.PublicKey != null ? config.PublicKey[..Math.Min(12, config.PublicKey.Length)] + "..." : "(missing)",
                config.TunnelIp ?? "(missing)");
            return null;
        }

        _logger.LogInformation(
            "Parsed WireGuard config from {Path}: endpoint={Endpoint}, " +
            "tunnelIp={TunnelIp}, subnet={Subnet}",
            sourcePath, config.Endpoint, config.TunnelIp, config.Subnet);

        return config;
    }
}

/// <summary>
/// Parsed host WireGuard configuration from /etc/wireguard/*.conf.
/// Used for CGNAT fallback relay info resolution.
/// </summary>
public class HostWireGuardConfig
{
    /// <summary>Relay's WireGuard endpoint (ip:port)</summary>
    public string? Endpoint { get; set; }

    /// <summary>Relay's WireGuard public key</summary>
    public string? PublicKey { get; set; }

    /// <summary>This host's tunnel IP (e.g., "10.20.1.2")</summary>
    public string? TunnelIp { get; set; }

    /// <summary>Relay IP extracted from Endpoint</summary>
    public string? RelayIp { get; set; }

    /// <summary>Subnet number (e.g., "1" from 10.20.1.0/24)</summary>
    public string? Subnet { get; set; }
}