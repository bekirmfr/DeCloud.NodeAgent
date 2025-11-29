using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using DeCloud.NodeAgent.Infrastructure.Network;
using DeCloud.NodeAgent.Infrastructure.Services;
using DeCloud.NodeAgent.Services;

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

// Core services
builder.Services.AddSingleton<ICommandExecutor, CommandExecutor>();
builder.Services.AddSingleton<IResourceDiscoveryService, ResourceDiscoveryService>();
builder.Services.AddSingleton<IImageManager, ImageManager>();
builder.Services.AddSingleton<LibvirtVmManager>();  // Register concrete type for initialization
builder.Services.AddSingleton<IVmManager>(sp => sp.GetRequiredService<LibvirtVmManager>());  // Alias to interface
builder.Services.AddSingleton<INetworkManager, WireGuardNetworkManager>();
builder.Services.AddSingleton<ICloudInitCleaner, CloudInitCleaner>();

// HTTP client for image downloads and orchestrator communication
builder.Services.AddHttpClient<IImageManager, ImageManager>();
builder.Services.AddHttpClient<OrchestratorClient>();
builder.Services.AddSingleton<IOrchestratorClient>(sp =>
    sp.GetRequiredService<OrchestratorClient>());

// Background services
builder.Services.AddHostedService<HeartbeatService>();
builder.Services.AddHostedService<CommandProcessorService>();

// API
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

// Check cloud-init cleaner tools on startup
using (var scope = app.Services.CreateScope())
{
    var cleaner = scope.ServiceProvider.GetRequiredService<ICloudInitCleaner>();
    var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();

    var tools = await cleaner.CheckToolsAsync(CancellationToken.None);
    if (!tools.AnyToolAvailable)
    {
        logger.LogWarning(
            "⚠️  Cloud-init cleaning tools not found! VMs may boot with stale configuration. " +
            "Install with: {Command}",
            tools.RecommendedInstallCommand);
    }
    else
    {
        logger.LogInformation(
            "Cloud-init tools available: virt-customize={Virt}, guestmount={Guest}, qemu-nbd={Nbd}",
            tools.VirtCustomizeAvailable,
            tools.GuestMountAvailable,
            tools.QemuNbdAvailable);
    }
}

// Initialize services on startup
using (var scope = app.Services.CreateScope())
{
    var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();

    // Initialize LibvirtVmManager - reconcile with libvirt state
    var vmManager = scope.ServiceProvider.GetRequiredService<LibvirtVmManager>();
    try
    {
        logger.LogInformation("Initializing VM Manager and reconciling with libvirt...");
        await vmManager.InitializeAsync();
        var vms = await vmManager.GetAllVmsAsync();
        logger.LogInformation("VM Manager initialized with {Count} existing VMs", vms.Count);

        foreach (var vm in vms)
        {
            logger.LogInformation("  - VM {VmId}: {State}", vm.VmId, vm.State);
        }
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to initialize VM Manager");
    }

    // Initialize WireGuard
    var networkManager = scope.ServiceProvider.GetRequiredService<INetworkManager>();
    try
    {
        var wgInitialized = await networkManager.InitializeWireGuardAsync();
        if (wgInitialized)
        {
            var pubkey = await networkManager.GetWireGuardPublicKeyAsync();
            logger.LogInformation("WireGuard initialized. Public key: {PublicKey}", pubkey);
        }
        else
        {
            logger.LogWarning("WireGuard initialization failed - overlay networking will be unavailable");
        }
    }
    catch (Exception ex)
    {
        logger.LogWarning(ex, "WireGuard initialization error - continuing without overlay networking");
    }
}

// Configure pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseWebSockets();
app.MapControllers();

// Health check endpoint
app.MapGet("/health", () => Results.Ok(new { status = "healthy", timestamp = DateTime.UtcNow }));

app.Run();