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
builder.Services.AddSingleton<IVmManager, LibvirtVmManager>();
builder.Services.AddSingleton<INetworkManager, WireGuardNetworkManager>();

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

// Initialize services on startup
using (var scope = app.Services.CreateScope())
{
    var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
    
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
            logger.LogWarning("WireGuard initialization failed - network features limited");
        }
    }
    catch (Exception ex)
    {
        logger.LogWarning(ex, "WireGuard initialization failed - continuing without overlay network");
    }

    // Discover initial resources
    var resourceDiscovery = scope.ServiceProvider.GetRequiredService<IResourceDiscoveryService>();
    try
    {
        var resources = await resourceDiscovery.DiscoverAllAsync();
        logger.LogInformation(
            "Node resources: {Cores} cores, {MemoryGB}GB RAM, {Gpus} GPUs, Virtualization: {VirtSupport}",
            resources.Cpu.LogicalCores,
            resources.Memory.TotalBytes / 1024 / 1024 / 1024,
            resources.Gpus.Count,
            resources.Cpu.SupportsVirtualization ? "Yes" : "No");
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to discover resources");
    }
}

// Configure pipeline - Enable Swagger in all environments
app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "DeCloud Node Agent API v1");
    c.RoutePrefix = "swagger";
});

app.UseAuthorization();
app.MapControllers();

// Health endpoint at root
app.MapGet("/", () => Results.Ok(new 
{ 
    service = "DeCloud Node Agent",
    version = "1.0.0",
    status = "running",
    timestamp = DateTime.UtcNow
}));

app.Run();