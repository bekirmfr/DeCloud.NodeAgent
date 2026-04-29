using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.DependencyInjection;
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
    private readonly IServiceProvider _serviceProvider;
    private IVmManager? _vmManager;
    private readonly ILogger<CloudInitTemplateService> _logger;
    private readonly string _templateBasePath;

    // Cache for loaded templates
    private readonly Dictionary<VmType, string> _templateCache = new();
    private readonly Dictionary<string, string> _externalTemplateCache = new();
    private readonly SemaphoreSlim _cacheLock = new(1, 1);

    // IVmManager is resolved lazily to break a circular dependency:
    // LibvirtVmManager → ICloudInitTemplateService → CloudInitTemplateService → IVmManager → LibvirtVmManager
    // Resolving IVmManager eagerly in the constructor deadlocks the DI container.
    private IVmManager VmManager => _vmManager ??= _serviceProvider.GetRequiredService<IVmManager>();

    public CloudInitTemplateService(
        ICommandExecutor executor,
        IServiceProvider serviceProvider,
        ILogger<CloudInitTemplateService> logger)
    {
        _executor = executor;
        _serviceProvider = serviceProvider;
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

        switch (vmType)
        {
            case VmType.General:
                template = await InjectGeneralExternalTemplatesAsync(template, ct);
                break;
            case VmType.Inference:
                // Future VM types can have their own external templates if needed
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
        var architecture = ResourceDiscoveryService.GetArchitectureNormalised();

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