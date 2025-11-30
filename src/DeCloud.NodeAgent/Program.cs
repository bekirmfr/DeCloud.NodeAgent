// Updated Program.cs for Node Agent
// Adds VmRepository registration

using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Infrastructure.Libvirt;
using DeCloud.NodeAgent.Infrastructure.Network;
using DeCloud.NodeAgent.Infrastructure.Persistence;
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

// VM Manager with repository
builder.Services.AddSingleton<LibvirtVmManager>();
builder.Services.AddSingleton<IVmManager>(sp => sp.GetRequiredService<LibvirtVmManager>());

// Network services
builder.Services.AddSingleton<INetworkManager, WireGuardNetworkManager>();
builder.Services.AddSingleton<ICloudInitCleaner, CloudInitCleaner>();
builder.Services.AddSingleton<IEphemeralSshKeyService, EphemeralSshKeyService>();

// HTTP client for image downloads and orchestrator communication
builder.Services.AddHttpClient<IImageManager, ImageManager>();
builder.Services.AddHttpClient<OrchestratorClient>();
builder.Services.AddSingleton<IOrchestratorClient>(sp =>
    sp.GetRequiredService<OrchestratorClient>());

// Background services
builder.Services.AddHostedService<HeartbeatService>();
builder.Services.AddHostedService<CommandProcessorService>();

// =====================================================
// NEW: Initialize LibvirtVmManager on startup
// This ensures VMs are loaded from SQLite database
// =====================================================
builder.Services.AddHostedService<VmManagerInitializationService>();

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
            "Install one of: virt-sysprep, guestfish, or cloud-init");
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

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapControllers();

app.MapGet("/health", () => new
{
    status = "healthy",
    timestamp = DateTime.UtcNow,
    version = "1.0.0"
});

app.Run();

// =====================================================
// NEW: Background service to initialize VM Manager
// =====================================================
public class VmManagerInitializationService : IHostedService
{
    private readonly IVmManager _vmManager;
    private readonly ILogger<VmManagerInitializationService> _logger;

    public VmManagerInitializationService(
        IVmManager vmManager,
        ILogger<VmManagerInitializationService> logger)
    {
        _vmManager = vmManager;
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Initializing VM Manager with state recovery...");

        try
        {
            // Initialize will load VMs from SQLite and reconcile with libvirt
            if (_vmManager is LibvirtVmManager libvirtManager)
            {
                await libvirtManager.InitializeAsync(cancellationToken);

                var vms = await libvirtManager.GetAllVmsAsync(cancellationToken);
                _logger.LogInformation(
                    "VM Manager initialized: {VmCount} VMs recovered from database",
                    vms.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize VM Manager");
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }
}