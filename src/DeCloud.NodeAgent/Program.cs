using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Interfaces.UserNetwork;
using DeCloud.NodeAgent.Core.Settings;
using DeCloud.NodeAgent.Infrastructure.Docker;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using DeCloud.NodeAgent.Infrastructure.Network;
using DeCloud.NodeAgent.Infrastructure.Network.UserNetwork;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.NodeAgent.Infrastructure.Services;
using DeCloud.NodeAgent.Infrastructure.Services.Auth;
using DeCloud.NodeAgent.Services;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

// Configuration
builder.Services.Configure<LibvirtVmManagerOptions>(
    builder.Configuration.GetSection("Libvirt"));
builder.Services.Configure<ImageManagerOptions>(
    builder.Configuration.GetSection("Images"));
builder.Services.Configure<WireGuardOptions>(
    builder.Configuration.GetSection("WireGuard"));
builder.Services.Configure<HeartbeatOptions>(
    builder.Configuration.GetSection("Heartbeat"));
builder.Services.Configure<CommandProcessorOptions>(
    builder.Configuration.GetSection("CommandProcessor"));
builder.Services.Configure<OrchestratorClientOptions>(
    builder.Configuration.GetSection("OrchestratorClient"));
builder.Services.Configure<PortSecurityOptions>(
    builder.Configuration.GetSection("PortSecurity"));
builder.Services.Configure<AuditLogOptions>(
    builder.Configuration.GetSection("AuditLog"));

// =====================================================
// GenericProxyController Configuration
// =====================================================
// Configure ProxySettings for GenericProxyController
// Note: GenericProxyController has sensible defaults and works without this,
// but this allows runtime customization via appsettings.json
builder.Services.Configure<ProxySettings>(
    builder.Configuration.GetSection("ProxySettings"));

// =====================================================
// HTTP Clients
// =====================================================
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
    options.SerializerOptions.PropertyNameCaseInsensitive = true;
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
});

builder.Services.AddHttpClient<IImageManager, ImageManager>();
builder.Services.AddHttpClient<OrchestratorClient>()
    .ConfigureHttpClient(client =>
    {
        client.Timeout = TimeSpan.FromMinutes(5);
    });

// HttpClient for VM proxy (used by InternalProxyController and GenericProxyController)
// Both controllers share this named HttpClient for proxying to VMs
builder.Services.AddHttpClient("VmProxy", client =>
{
    client.Timeout = TimeSpan.FromSeconds(30);
})
.ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler
{
    AllowAutoRedirect = false,
    UseCookies = false
});

// =====================================================
// In-Memory Queues
// =====================================================
builder.Services.AddSingleton<ConcurrentQueue<PendingCommand>>();

// =====================================================
// Core Services
// =====================================================
builder.Services.AddSingleton<INodeMetadataService, NodeMetadataService>();
builder.Services.AddSingleton<INodeStateService, NodeStateService>();
builder.Services.AddSingleton<ICommandExecutor, CommandExecutor>();
builder.Services.AddSingleton<INatRuleManager, NatRuleManager>();
builder.Services.AddSingleton<IResourceDiscoveryService, ResourceDiscoveryService>();
builder.Services.AddSingleton<ICpuBenchmarkService, CpuBenchmarkService>();
builder.Services.AddSingleton<IImageManager, ImageManager>();
builder.Services.AddSingleton<IOrchestratorClient>(sp =>
    sp.GetRequiredService<OrchestratorClient>());

// =====================================================
// Port Mapping Repository (Smart Port Allocation)
// =====================================================
builder.Services.AddSingleton<PortMappingRepository>(sp =>
{
    var options = sp.GetRequiredService<IOptions<LibvirtVmManagerOptions>>().Value;
    var logger = sp.GetRequiredService<ILogger<PortMappingRepository>>();

    // Check if running on Windows
    var isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
        System.Runtime.InteropServices.OSPlatform.Windows);

    if (isWindows)
    {
        logger.LogWarning("Running on Windows - PortMappingRepository will not be used");
        var tempPath = Path.Combine(Path.GetTempPath(), "decloud-port-mappings-dummy.db");
        return new PortMappingRepository(tempPath, logger);
    }

    var dbPath = Path.Combine(options.VmStoragePath, "port-mappings.db");
    return new PortMappingRepository(dbPath, logger);
});

// =====================================================
// Smart Port Allocation Services
// =====================================================
builder.Services.AddSingleton<IPortPoolManager, PortPoolManager>();
builder.Services.AddSingleton<IPortForwardingManager, PortForwardingManager>();

// =====================================================
// VM Repository with Encryption Support
// =====================================================
builder.Services.AddSingleton<VmRepository>(sp =>
{
    var options = sp.GetRequiredService<IOptions<LibvirtVmManagerOptions>>().Value;
    var logger = sp.GetRequiredService<ILogger<VmRepository>>();
    var config = sp.GetRequiredService<IConfiguration>();

    // Check if running on Windows
    var isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
        System.Runtime.InteropServices.OSPlatform.Windows);

    if (isWindows)
    {
        logger.LogWarning("Running on Windows - VmRepository will not be used");
        var tempPath = Path.Combine(Path.GetTempPath(), "decloud-dummy.db");
        return new VmRepository(tempPath, logger);
    }

    var dbPath = Path.Combine(options.VmStoragePath, "vms.db");
    string? encryptionKey = null;

    try
    {
        // Get wallet address from configuration
        var walletAddress = config.GetValue<string>("Orchestrator:WalletAddress");

        if (!string.IsNullOrEmpty(walletAddress))
        {
            // Generate deterministic encryption key using machine identifier + wallet
            // This ensures the same key is generated across node agent restarts
            var machineId = File.ReadAllText("/etc/machine-id").Trim();
            encryptionKey = VmRepository.GenerateEncryptionKey(machineId, walletAddress);

            logger.LogInformation(
                "✓ VmRepository encryption ENABLED (machine: {Machine}, wallet: {Wallet})",
                machineId,
                walletAddress.Substring(0, Math.Min(10, walletAddress.Length)) + "...");
        }
        else
        {
            logger.LogWarning(
                "⚠️  VmRepository encryption DISABLED - no wallet address configured. " +
                "VM data will be stored UNENCRYPTED in SQLite. " +
                "Set 'Orchestrator:WalletAddress' in appsettings.json to enable encryption.");
        }
    }
    catch (Exception ex)
    {
        logger.LogError(ex,
            "❌ Failed to generate encryption key - VmRepository will run UNENCRYPTED");
    }

    return new VmRepository(dbPath, logger, encryptionKey);
});

// =====================================================
// VM Manager with Repository Integration
// =====================================================
builder.Services.AddSingleton<ICloudInitTemplateService, CloudInitTemplateService>();
builder.Services.AddSingleton<LibvirtVmManager>();
builder.Services.AddSingleton<IVmManager>(sp => sp.GetRequiredService<LibvirtVmManager>());

// =====================================================
// GPU Proxy Service (host-side daemon for proxied GPU access via vsock)
// =====================================================
builder.Services.AddSingleton<GpuProxyService>();

// =====================================================
// Docker Container Manager (GPU sharing for WSL2/non-IOMMU nodes)
// =====================================================
builder.Services.AddSingleton<DockerContainerManager>();

// =====================================================
// Network Services
// =====================================================
builder.Services.AddSingleton<INetworkManager, WireGuardNetworkManager>();
builder.Services.AddSingleton<IUserWireGuardManager, UserWireGuardManager>();
builder.Services.AddSingleton<ICloudInitCleaner, CloudInitCleaner>();
builder.Services.AddSingleton<IEphemeralSshKeyService, EphemeralSshKeyService>();

// =====================================================
// Background Services
// =====================================================
builder.Services.AddHostedService<AuthenticationManager>();
builder.Services.AddHostedService<HeartbeatService>();
builder.Services.AddHostedService<WireGuardConfigManager>();
builder.Services.AddHostedService<CommandProcessorService>();
builder.Services.AddHostedService<DatabaseMaintenanceService>();
builder.Services.AddHostedService<VmHealthService>();

// Initialize VM Manager on startup to load VMs from database
builder.Services.AddHostedService<VmManagerInitializationService>();

// Reconcile port forwarding rules on startup (Smart Port Allocation)
builder.Services.AddHostedService<PortForwardingReconciliationService>();

// Periodic cleanup of orphaned ports (hourly)
builder.Services.AddHostedService<OrphanedPortCleanupService>();

// Per-service VM readiness monitoring via qemu-guest-agent
builder.Services.AddHostedService<VmReadinessMonitor>();

// Auto-start GPU proxy daemon on non-IOMMU nodes with GPUs
builder.Services.AddHostedService<GpuProxyStartupService>();

// =====================================================
// Security services for port validation and auditing
// =====================================================

// Port security (validates ingress target ports)
builder.Services.AddSingleton<IPortSecurityService, PortSecurityService>();

// Security audit logging
builder.Services.AddSingleton<IAuditService, AuditService>();

// =====================================================
// API
// =====================================================
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
        options.JsonSerializerOptions.PropertyNameCaseInsensitive = true;
    });
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new()
    {
        Title = "DeCloud Node Agent API",
        Version = "v1",
        Description = "Local API for managing VMs and monitoring node resources"
    });
});

var app = builder.Build();

var nodeMetadata = app.Services.GetRequiredService<INodeMetadataService>();
await nodeMetadata.InitializeAsync();

var orchestratorClient = app.Services.GetRequiredService<IOrchestratorClient>();
await orchestratorClient.InitializeAsync();

// =====================================================
// Startup Checks
// =====================================================
using (var scope = app.Services.CreateScope())
{
    var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();

    // Check cloud-init cleaner tools
    var cleaner = scope.ServiceProvider.GetRequiredService<ICloudInitCleaner>();
    var tools = await cleaner.CheckToolsAsync(CancellationToken.None);

    if (!tools.AnyToolAvailable)
    {
        logger.LogWarning(
            "⚠️  Cloud-init cleaning tools not found! VMs may boot with stale configuration. " +
            "Install 'virt-sysprep' or 'guestfs-tools' for best results.");
    }
    else
    {
        logger.LogInformation(
            "Cloud-init tools available: virt-customize={Virt}, guestmount={Guest}, qemu-nbd={Nbd}",
            tools.VirtCustomizeAvailable,
            tools.GuestMountAvailable,
            tools.QemuNbdAvailable);
    }

    // Platform check
    var isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
        System.Runtime.InteropServices.OSPlatform.Windows);

    if (isWindows)
    {
        logger.LogWarning(
            "⚠️  Running on Windows - VM management disabled. " +
            "Deploy to Linux with KVM/libvirt for full functionality.");
    }

    // Verify VmRepository encryption status
    var repository = scope.ServiceProvider.GetRequiredService<VmRepository>();
    // Note: VmRepository doesn't expose encryption status, but logs will show it

    // =====================================================
    // Log GenericProxyController configuration
    // =====================================================
    try
    {
        var proxySettings = scope.ServiceProvider.GetService<IOptions<ProxySettings>>()?.Value;
        if (proxySettings != null && proxySettings.EnablePortWhitelist)
        {
            logger.LogInformation(
                "✓ GenericProxyController configured with {Count} allowed ports: {Ports}",
                proxySettings.AllowedPorts?.Count ?? 0,
                string.Join(", ", proxySettings.AllowedPorts ?? new List<int>()));
        }
        else
        {
            logger.LogInformation(
                "✓ GenericProxyController using default configuration (port whitelist: all ports)");
        }
    }
    catch (Exception ex)
    {
        logger.LogWarning(ex, "Could not load ProxySettings - using defaults");
    }
}

// =====================================================
// Middleware Pipeline
// =====================================================
// Enable request body buffering early so proxy can read POST bodies.
// Without this, middleware may consume the body and GenericProxyController sees empty body.
app.Use(async (context, next) =>
{
    context.Request.EnableBuffering();
    await next();
});
app.UseWebSockets(new WebSocketOptions
{
    KeepAliveInterval = TimeSpan.FromSeconds(120)
});
app.UseSwagger();
app.UseSwaggerUI();

app.MapControllers();

app.Run();

// =====================================================
// VM Manager Initialization Service
// =====================================================
/// <summary>
/// Background service that initializes the VM Manager on startup.
/// Uses BackgroundService (not IHostedService) so startup is non-blocking —
/// Kestrel binds the port immediately while VM reconciliation runs in the background.
/// </summary>
public class VmManagerInitializationService : BackgroundService
{
    private readonly LibvirtVmManager _vmManager;
    private readonly ILogger<VmManagerInitializationService> _logger;

    public VmManagerInitializationService(
        LibvirtVmManager vmManager,
        ILogger<VmManagerInitializationService> logger)
    {
        _vmManager = vmManager;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Initializing VM Manager and loading state from database...");

        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken);
            cts.CancelAfter(TimeSpan.FromSeconds(60));

            await _vmManager.InitializeAsync(cts.Token);
            _logger.LogInformation("VM Manager initialization complete");
        }
        catch (OperationCanceledException) when (!stoppingToken.IsCancellationRequested)
        {
            _logger.LogWarning(
                "VM Manager initialization timed out after 60s — " +
                "continuing without full reconciliation. " +
                "Check libvirt connectivity (virsh list --all).");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize VM Manager");
        }
    }
}

// =====================================================
// GPU Proxy Daemon Auto-Start Service
// =====================================================
/// <summary>
/// Background service that auto-starts the GPU proxy daemon on nodes
/// that have GPU(s) but no IOMMU (i.e., proxy mode is required).
/// Runs once at startup so the daemon is ready before any VM boots.
/// </summary>
public class GpuProxyStartupService : BackgroundService
{
    private readonly GpuProxyService _gpuProxy;
    private readonly ILogger<GpuProxyStartupService> _logger;

    public GpuProxyStartupService(
        GpuProxyService gpuProxy,
        ILogger<GpuProxyStartupService> logger)
    {
        _gpuProxy = gpuProxy;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Small delay to let resource discovery complete first
        await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);

        try
        {
            var started = await _gpuProxy.EnsureStartedAsync(stoppingToken);
            if (started)
            {
                _logger.LogInformation(
                    "GPU proxy daemon auto-started — node is in proxy mode (no IOMMU)");
            }
            // If not started, EnsureStartedAsync already logged the reason
            // (no GPU, IOMMU available, daemon binary missing, etc.)
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Normal shutdown
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to auto-start GPU proxy daemon");
        }
    }

    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Stopping GPU proxy daemon...");
        await _gpuProxy.StopAsync(cancellationToken);
        await base.StopAsync(cancellationToken);
    }
}