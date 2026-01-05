using System.Text;
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
            ["__ADMIN_PASSWORD__"] = AdminPassword
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
    private readonly ILogger<CloudInitTemplateService> _logger;
    private readonly string _templateBasePath;

    // Cache for loaded templates
    private readonly Dictionary<VmType, string> _templateCache = new();
    private readonly Dictionary<string, string> _externalTemplateCache = new();
    private readonly SemaphoreSlim _cacheLock = new(1, 1);

    public CloudInitTemplateService(
        ICommandExecutor executor,
        ILogger<CloudInitTemplateService> logger)
    {
        _executor = executor;
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

        // For relay VMs, inject external templates (dashboard and API)
        if (vmType == VmType.Relay)
        {
            template = await InjectRelayExternalTemplatesAsync(template, ct);
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
            var dashboardHtml = await LoadExternalTemplateAsync("dashboard.html", ct);
            var dashboardCss = await LoadExternalTemplateAsync("dashboard.css", ct);
            var dashboardJs = await LoadExternalTemplateAsync("dashboard.js", ct);
            var relayApi = await LoadExternalTemplateAsync("relay-api.py", ct);

            // Replace placeholders with properly indented content
            var result = ReplaceWithIndentation(template, "__DASHBOARD_HTML__", dashboardHtml);
            result = ReplaceWithIndentation(result, "__DASHBOARD_CSS__", dashboardCss);
            result = ReplaceWithIndentation(result, "__DASHBOARD_JS__", dashboardJs);
            result = ReplaceWithIndentation(result, "__RELAY_API__", relayApi);

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
    /// Load an external template file from the relay-vm subdirectory
    /// </summary>
    private async Task<string> LoadExternalTemplateAsync(
        string filename,
        CancellationToken ct)
    {
        await _cacheLock.WaitAsync(ct);
        try
        {
            // Check cache first
            if (_externalTemplateCache.TryGetValue(filename, out var cached))
            {
                _logger.LogDebug("Using cached external template: {FileName}", filename);
                return cached;
            }

            // Load from disk
            var templatePath = Path.Combine(_templateBasePath, "relay-vm", filename);

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
            _externalTemplateCache[filename] = content;

            _logger.LogDebug(
                "Loaded external template: {FileName} ({Length} chars)",
                filename, content.Length);

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
            AdminPassword = GenerateSecurePassword()
        };

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

    private Task PopulateDhtVariablesAsync(
        CloudInitTemplateVariables variables,
        VmSpec spec,
        CancellationToken ct)
    {
        // DHT-specific configuration
        variables.Custom["DHT_PORT"] = "6881";
        variables.Custom["DHT_STORAGE_PATH"] = "/var/lib/decloud-dht/storage";
        variables.Custom["DHT_MAX_STORAGE_GB"] = "100";

        _logger.LogInformation(
            "Configured DHT variables for VM {VmId}: port={Port}, storage={Path}",
            spec.Id,
            variables.Custom["DHT_PORT"],
            variables.Custom["DHT_STORAGE_PATH"]);

        return Task.CompletedTask;
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
}