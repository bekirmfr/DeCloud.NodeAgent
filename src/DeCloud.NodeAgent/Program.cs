using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using DeCloud.NodeAgent.Infrastructure.Network;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.NodeAgent.Infrastructure.Services;
using DeCloud.NodeAgent.Services;
using Microsoft.Extensions.Options;

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
    builder.Configuration.GetSection("Orchestrator"));

// =====================================================
// Core Services
// =====================================================
builder.Services.AddSingleton<ICommandExecutor, CommandExecutor>();
builder.Services.AddSingleton<IResourceDiscoveryService, ResourceDiscoveryService>();
builder.Services.AddSingleton<IImageManager, ImageManager>();

// =====================================================
// VM Repository with Encryption Support
// =====================================================
builder.Services.AddSingleton<VmRepository>(sp =>
{
    var options = sp.GetRequiredService<IOptions<LibvirtVmManagerOptions>>().Value;
    var logger = sp.GetRequiredService<ILogger<VmRepository>>();

    // Check if running on Windows
    var isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
        System.Runtime.InteropServices.OSPlatform.Windows);

    if (isWindows)
    {
        logger.LogWarning("Running on Windows - VmRepository will not be used");
        // Return a dummy repository (won't be used on Windows)
        var tempPath = Path.Combine(Path.GetTempPath(), "decloud-dummy.db");
        return new VmRepository(tempPath, logger);
    }

    var dbPath = Path.Combine(options.VmStoragePath, "vms.db");

    // Generate encryption key from node credentials
    // Note: This will be null until node is registered, so repository starts unencrypted
    // and can be upgraded to encrypted once credentials are available
    string? encryptionKey = null;

    try
    {
        var walletAddress = builder.Configuration.GetValue<string>("Orchestrator:WalletAddress");

        if (!string.IsNullOrEmpty(walletAddress))
        {
            // Use wallet address as basis for encryption key
            // Note: Node ID not available yet, so we use wallet + machine ID
            var machineId = Environment.MachineName;
            encryptionKey = VmRepository.GenerateEncryptionKey(machineId, walletAddress);

            logger.LogInformation("VmRepository encryption enabled using wallet-based key");
        }
        else
        {
            logger.LogWarning("No wallet address configured - VmRepository encryption disabled");
        }
    }
    catch (Exception ex)
    {
        logger.LogWarning(ex, "Failed to generate encryption key - using unencrypted repository");
    }

    return new VmRepository(dbPath, logger, encryptionKey);
});

// =====================================================
// VM Manager with Repository Integration
// =====================================================
// Add VmRepository as singleton
builder.Services.AddSingleton<VmRepository>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<VmRepository>>();
    var options = sp.GetRequiredService<IOptions<LibvirtVmManagerOptions>>();
    var dbPath = Path.Combine(options.Value.VmStoragePath, "vms.db");

    // Optional: Generate encryption key from node config
    var config = sp.GetRequiredService<IConfiguration>();
    var nodeId = config.GetValue<string>("Node:Id");
    var walletAddress = config.GetValue<string>("Orchestrator:WalletAddress");
    string? encryptionKey = null;

    if (!string.IsNullOrEmpty(nodeId) && !string.IsNullOrEmpty(walletAddress))
    {
        encryptionKey = VmRepository.GenerateEncryptionKey(nodeId, walletAddress);
    }

    return new VmRepository(dbPath, logger, encryptionKey);
});

builder.Services.AddSingleton<LibvirtVmManager>();
builder.Services.AddSingleton<IVmManager>(sp => sp.GetRequiredService<LibvirtVmManager>());

// =====================================================
// Network Services
// =====================================================
builder.Services.AddSingleton<INetworkManager, WireGuardNetworkManager>();
builder.Services.AddSingleton<ICloudInitCleaner, CloudInitCleaner>();
builder.Services.AddSingleton<IEphemeralSshKeyService, EphemeralSshKeyService>();

// =====================================================
// HTTP Clients
// =====================================================
builder.Services.AddHttpClient<IImageManager, ImageManager>();
builder.Services.AddHttpClient<OrchestratorClient>();
builder.Services.AddSingleton<IOrchestratorClient>(sp =>
    sp.GetRequiredService<OrchestratorClient>());

// =====================================================
// Background Services
// =====================================================
builder.Services.AddHostedService<HeartbeatService>();
builder.Services.AddHostedService<CommandProcessorService>();
builder.Services.AddHostedService<DatabaseMaintenanceService>();

// Initialize VM Manager on startup to load VMs from database
builder.Services.AddHostedService<VmManagerInitializationService>();

// =====================================================
// API
// =====================================================
builder.Services.AddControllers();
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
        var isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
            System.Runtime.InteropServices.OSPlatform.Windows);

        if (isWindows)
        {
            logger.LogWarning(
                "⚠️  Running on Windows - VM management disabled. " +
                "Deploy to Linux with KVM/libvirt for full functionality.");
        }
    }
}

// =====================================================
// Middleware Pipeline
// =====================================================
app.UseSwagger();
app.UseSwaggerUI();

app.MapControllers();

app.Run();

// =====================================================
// VM Manager Initialization Service
// =====================================================
/// <summary>
/// Background service that initializes the VM Manager on startup.
/// This ensures VMs are loaded from the SQLite database before processing begins.
/// </summary>
public class VmManagerInitializationService : IHostedService
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

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Initializing VM Manager and loading state from database...");

        try
        {
            await _vmManager.InitializeAsync(cancellationToken);
            _logger.LogInformation("✓ VM Manager initialization complete");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize VM Manager");
            // Don't throw - allow service to start even if initialization fails
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("VM Manager shutdown");
        return Task.CompletedTask;
    }
}