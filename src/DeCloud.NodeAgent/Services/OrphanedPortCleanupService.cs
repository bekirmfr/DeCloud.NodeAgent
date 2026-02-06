using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Infrastructure.Persistence;
using DeCloud.NodeAgent.Infrastructure.Services;

namespace DeCloud.NodeAgent.Services;

/// <summary>
/// Background service that periodically detects and cleans up orphaned port mappings.
/// Orphaned ports occur when VMs are manually deleted outside the API (e.g., via virsh).
/// </summary>
public class OrphanedPortCleanupService : BackgroundService
{
    private readonly IVmManager _vmManager;
    private readonly PortMappingRepository _portMappingRepository;
    private readonly IPortForwardingManager _portForwardingManager;
    private readonly IPortPoolManager _portPoolManager;
    private readonly ILogger<OrphanedPortCleanupService> _logger;

    private static readonly TimeSpan CleanupInterval = TimeSpan.FromHours(1);  // Run every hour

    public OrphanedPortCleanupService(
        IVmManager vmManager,
        PortMappingRepository portMappingRepository,
        IPortForwardingManager portForwardingManager,
        IPortPoolManager portPoolManager,
        ILogger<OrphanedPortCleanupService> logger)
    {
        _vmManager = vmManager;
        _portMappingRepository = portMappingRepository;
        _portForwardingManager = portForwardingManager;
        _portPoolManager = portPoolManager;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Wait 5 minutes after startup before first cleanup
        await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await DetectAndCleanupOrphanedPortsAsync(stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during orphaned port cleanup");
            }

            // Wait for next cleanup cycle
            await Task.Delay(CleanupInterval, stoppingToken);
        }
    }

    private async Task DetectAndCleanupOrphanedPortsAsync(CancellationToken ct)
    {
        _logger.LogInformation("Starting orphaned port detection...");

        try
        {
            // Get all active port mappings from database
            var allMappings = await _portMappingRepository.GetAllActiveAsync();
            if (!allMappings.Any())
            {
                _logger.LogDebug("No port mappings found, skipping orphan detection");
                return;
            }

            // Get all active VM IDs from libvirt
            var activeVms = await _vmManager.GetAllVmsAsync(ct);
            var activeVmIds = activeVms.Select(vm => vm.Spec.VmId).ToHashSet();

            _logger.LogDebug(
                "Checking {MappingCount} port mappings against {VmCount} active VMs",
                allMappings.Count, activeVmIds.Count);

            // Find orphaned mappings
            var orphanedMappings = allMappings
                .Where(m => !activeVmIds.Contains(m.VmId))
                .ToList();

            if (!orphanedMappings.Any())
            {
                _logger.LogInformation(
                    "✓ No orphaned ports found ({MappingCount} mappings, {VmCount} active VMs)",
                    allMappings.Count, activeVmIds.Count);
                return;
            }

            _logger.LogWarning(
                "Found {OrphanCount} orphaned port mapping(s) from deleted VMs, cleaning up...",
                orphanedMappings.Count);

            // Clean up each orphaned mapping
            int successCount = 0;
            foreach (var mapping in orphanedMappings)
            {
                try
                {
                    _logger.LogInformation(
                        "Cleaning up orphaned port {PublicPort} → {Destination}:{VmPort} (deleted VM {VmId})",
                        mapping.PublicPort, mapping.VmPrivateIp, mapping.VmPort, mapping.VmId);

                    // Remove iptables rules
                    await _portForwardingManager.RemoveForwardingAsync(
                        mapping.VmPrivateIp,
                        mapping.VmPort,
                        mapping.PublicPort,
                        mapping.Protocol,
                        ct);

                    // Release port back to pool
                    await _portPoolManager.ReleasePortAsync(mapping.PublicPort, ct);

                    // Remove from database
                    await _portMappingRepository.RemoveAsync(mapping.VmId, mapping.VmPort);

                    successCount++;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex,
                        "Failed to clean up orphaned port {PublicPort} for deleted VM {VmId}",
                        mapping.PublicPort, mapping.VmId);
                }
            }

            _logger.LogInformation(
                "✓ Orphaned port cleanup complete: {SuccessCount}/{TotalCount} cleaned up",
                successCount, orphanedMappings.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to detect orphaned ports");
        }
    }
}
