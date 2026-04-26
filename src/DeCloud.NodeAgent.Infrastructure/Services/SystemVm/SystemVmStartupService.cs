using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services.SystemVm;

/// <summary>
/// One-shot startup service that restarts system VMs (Relay, DHT, BlockStore)
/// found in Stopped or Failed state after a node restart.
///
/// System VMs are long-lived: their disks, cloud-init done marker, WireGuard config,
/// and service binaries all survive a host reboot. There is no need to redeploy —
/// a virsh start is sufficient. The VM's services (DHT binary, blockstore binary)
/// start automatically via systemd inside the guest.
///
/// Runs once after VmManagerInitializationService so the VM list is populated,
/// and before the first heartbeat so the orchestrator sees Running state immediately.
/// </summary>
public class SystemVmStartupService : BackgroundService
{
    private readonly IVmManager _vmManager;
    private readonly ICommandExecutor _executor;
    private readonly ILogger<SystemVmStartupService> _logger;

    private static readonly HashSet<VmType> SystemVmTypes =
    [
        VmType.Relay,
        VmType.Dht,
        VmType.BlockStore
    ];

    public SystemVmStartupService(
        IVmManager vmManager,
        ICommandExecutor executor,
        ILogger<SystemVmStartupService> logger)
    {
        _vmManager = vmManager;
        _executor = executor;
        _logger = logger;
    }

    /// <summary>
    /// For every running VM, verifies its tap interface (vnet*) is attached to virbr0.
    /// After a libvirtd restart or virbr0 recreation the tap interfaces may be orphaned —
    /// the QEMU process is alive but the bridge attachment is lost, breaking all traffic.
    /// Re-attaching is a non-destructive kernel operation: no VM restart required.
    /// </summary>
    private async Task ReattachOrphanedTapInterfacesAsync(CancellationToken ct)
    {
        var runningVms = _vmManager.GetAllVms()
            .Where(v => v.State == VmState.Running)
            .ToList();

        if (runningVms.Count == 0) return;

        var reattached = 0;

        foreach (var vm in runningVms)
        {
            try
            {
                // Get the tap interface name for this VM
                var iflist = await _executor.ExecuteAsync(
                    "virsh", $"domiflist {vm.VmId}", ct);

                if (!iflist.Success) continue;

                // Parse: "vnet5   network   default   virtio   52:54:00:xx:xx:xx"
                var tapIface = iflist.StandardOutput
                    .Split('\n')
                    .Skip(2)
                    .Select(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                    .Where(cols => cols.Length >= 3 && cols[2] == "default")
                    .Select(cols => cols[0])
                    .FirstOrDefault();

                if (string.IsNullOrEmpty(tapIface)) continue;

                // Check if the tap is already attached to virbr0
                var linkCheck = await _executor.ExecuteAsync(
                    "ip", $"link show {tapIface}", ct);

                if (!linkCheck.Success) continue;

                // "master virbr0" in ip link show means it's attached
                if (linkCheck.StandardOutput.Contains("master virbr0")) continue;

                // Tap exists but has no master — re-attach it
                _logger.LogWarning(
                    "SystemVmStartupService: tap interface {Tap} for VM {VmId} is detached " +
                    "from virbr0 — re-attaching",
                    tapIface, vm.VmId);

                var attach = await _executor.ExecuteAsync(
                    "ip", $"link set {tapIface} master virbr0", ct);

                if (attach.Success)
                {
                    reattached++;
                    _logger.LogInformation(
                        "✓ Re-attached {Tap} to virbr0 for VM {VmId} ({Name})",
                        tapIface, vm.VmId, vm.Name);
                }
                else
                {
                    _logger.LogWarning(
                        "Failed to re-attach {Tap} for VM {VmId}: {Error}",
                        tapIface, vm.VmId, attach.StandardError);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Error checking tap interface for VM {VmId}", vm.VmId);
            }
        }

        if (reattached > 0)
            _logger.LogInformation(
                "SystemVmStartupService: re-attached {Count} orphaned tap interface(s) to virbr0",
                reattached);
    }

    // WSL detection delegated to ResourceDiscoveryService.IsWsl()

    private async Task EnsureLibvirtNetworkActiveAsync(CancellationToken ct)
    {
        try
        {
            var infoResult = await _executor.ExecuteAsync("virsh", "net-info default", ct);
            if (!infoResult.Success)
            {
                _logger.LogWarning(
                    "SystemVmStartupService: could not query libvirt default network — VM starts may fail");
                return;
            }

            if (infoResult.StandardOutput.Contains("Active:           yes"))
            {
                _logger.LogDebug("SystemVmStartupService: libvirt default network already active");
                return;
            }

            // Network is inactive. The virbr0 bridge may still exist from before the
            // restart, causing net-start to fail with "already in use by interface virbr0".
            // Force-destroy the stale state first, then start clean.
            _logger.LogInformation(
                "SystemVmStartupService: libvirt default network inactive — resetting and starting");

            await _executor.ExecuteAsync("virsh", "net-destroy default", ct);
            // net-destroy is best-effort — ignore failure if already inactive

            var startResult = await _executor.ExecuteAsync("virsh", "net-start default", ct);

            // "already active" — nothing to do
            if (startResult.Success || startResult.StandardError.Contains("already active"))
            {
                _logger.LogInformation("✓ libvirt default network started");
                return;
            }

            // "already in use by interface virbr0" — virbr0 exists and VMs are connected
            // to it. The network is functionally active; only libvirt's internal state is
            // inconsistent. Touching virbr0 would disconnect running VMs — never do that.
            // The VMs we want to start will fail here, but that's the correct outcome:
            // fall through and let the orchestrator redeploy them on next reconciliation.
            if (startResult.StandardError.Contains("already in use by interface virbr0"))
            {
                // virbr0 bridge exists but libvirt considers the network inactive.
                // This is a libvirtd state inconsistency after WSL/host restart.
                // The only safe fix is restarting libvirtd — it will re-attach all
                // existing tap interfaces (vnet*) to virbr0, restoring full connectivity
                // for running VMs AND allowing stopped VMs to start.
                _logger.LogWarning(
                    "SystemVmStartupService: virbr0 exists but libvirt state is inconsistent — " +
                    "restarting libvirtd to recover. Running VMs will reconnect automatically.");

                var restart = await _executor.ExecuteAsync(
                    "systemctl", "restart libvirtd", ct);

                if (!restart.Success)
                {
                    _logger.LogWarning(
                        "Could not restart libvirtd: {Error} — stopped system VMs will be " +
                        "redeployed by orchestrator reconciliation",
                        restart.StandardError);
                    return;
                }

                // Give libvirtd time to restore tap interface attachments
                await Task.Delay(TimeSpan.FromSeconds(5), ct);

                var retryResult = await _executor.ExecuteAsync("virsh", "net-start default", ct);
                if (retryResult.Success || retryResult.StandardError.Contains("already active"))
                {
                    _logger.LogInformation(
                        "✓ libvirt default network started after libvirtd restart");
                    return;
                }

                // libvirtd restarted but virbr0 still exists as a stale kernel artifact.
                // On WSL, kernel interfaces survive daemon restarts and must be torn down
                // manually. On bare-metal Linux, libvirtd restart properly reclaims virbr0
                // — if it still fails here something else is wrong, don't touch the bridge.
                if (retryResult.StandardError.Contains("already in use by interface virbr0")
                    && ResourceDiscoveryService.IsWsl())
                {
                    _logger.LogInformation(
                        "SystemVmStartupService: stale virbr0 persists after libvirtd restart " +
                        "(WSL environment) — removing and doing final net-start");

                    await _executor.ExecuteAsync("ip", "link set virbr0 down", ct);
                    await _executor.ExecuteAsync("ip", "link delete virbr0", ct);

                    var finalResult = await _executor.ExecuteAsync("virsh", "net-start default", ct);
                    if (finalResult.Success || finalResult.StandardError.Contains("already active"))
                        _logger.LogInformation(
                            "✓ libvirt default network started after stale virbr0 removal");
                    else
                        _logger.LogWarning(
                            "Could not start libvirt default network after stale virbr0 removal: {Error}",
                            finalResult.StandardError);
                }
                else if (retryResult.StandardError.Contains("already in use by interface virbr0"))
                {
                    _logger.LogWarning(
                        "SystemVmStartupService: virbr0 conflict after libvirtd restart on " +
                        "bare-metal — not touching bridge. Manual libvirtd investigation required.");
                }
                else
                {
                    _logger.LogWarning(
                        "Could not start libvirt default network after libvirtd restart: {Error}",
                        retryResult.StandardError);
                }

                return;
            }

            _logger.LogWarning(
                "Could not start libvirt default network: {Error} — VM starts may fail",
                startResult.StandardError);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "SystemVmStartupService: could not check/start libvirt network — VM starts may fail");
        }
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        // Wait for VmManagerInitializationService to complete (it has a 60s timeout).
        // A fixed delay is sufficient — if libvirt isn't ready in 90s, neither are we.
        await Task.Delay(TimeSpan.FromSeconds(90), ct);

        if (ct.IsCancellationRequested) return;

        _logger.LogInformation("SystemVmStartupService: scanning for stopped system VMs...");

        // Ensure the libvirt default network is active before starting VMs.
        // After a host/WSL restart the network is defined but not started, causing
        // every virsh start to fail with "network 'default' is not active".
        await EnsureLibvirtNetworkActiveAsync(ct);
        await ReattachOrphanedTapInterfacesAsync(ct);

        // Detect zombie system VMs: Running in libvirt but LastHeartbeat is MinValue
        // (internal services never started — QEMU alive, systemd dead inside).
        // This happens when a VM survives a WSL/host restart with broken internal state.
        // virsh reset is a hard power-cycle at the QEMU level — no guest cooperation needed.
        var zombieVms = _vmManager.GetAllVms()
            .Where(v => SystemVmTypes.Contains(v.Spec.VmType)
                     && v.State == VmState.Running
                     && v.LastHeartbeat == DateTime.MinValue
                     && v.StartedAt.HasValue
                     && (DateTime.UtcNow - v.StartedAt.Value).TotalMinutes > 5)
            .ToList();

        foreach (var vm in zombieVms)
        {
            _logger.LogWarning(
                "SystemVmStartupService: system VM {VmId} ({VmType}) is running but has never " +
                "sent a heartbeat (zombie state — internal services dead). Hard resetting via virsh reset.",
                vm.VmId, vm.Spec.VmType);

            // virsh destroy + start = clean hard power cycle. No guest cooperation needed.
            await _executor.ExecuteAsync("virsh", $"destroy {vm.VmId}", ct);
            await Task.Delay(TimeSpan.FromSeconds(2), ct);
            var resetResult = await _executor.ExecuteAsync("virsh", $"start {vm.VmId}", ct);

            if (resetResult.Success)
                _logger.LogInformation(
                    "✓ Zombie system VM {VmId} ({VmType}) hard reset — services will restart via systemd",
                    vm.VmId, vm.Spec.VmType);
            else
                _logger.LogWarning(
                    "✗ Failed to hard reset zombie VM {VmId}: {Error}",
                    vm.VmId, resetResult.StandardError);
        }

        if (zombieVms.Count > 0)
            _logger.LogInformation(
                "SystemVmStartupService: processed {Count} zombie system VM(s)",
                zombieVms.Count);

        var vms = _vmManager.GetAllVms()
            .Where(v => SystemVmTypes.Contains(v.Spec.VmType)
                     && v.State is VmState.Stopped or VmState.Failed)
            .ToList();

        if (vms.Count == 0)
        {
            _logger.LogInformation("SystemVmStartupService: no stopped system VMs found");
            return;
        }

        _logger.LogInformation(
            "SystemVmStartupService: found {Count} stopped system VM(s) — attempting restart",
            vms.Count);

        foreach (var vm in vms)
        {
            try
            {
                _logger.LogInformation(
                    "Starting system VM {VmId} ({VmType}) after node restart",
                    vm.VmId, vm.Spec.VmType);

                var result = await _vmManager.StartVmAsync(vm.VmId, ct);

                if (result.Success)
                {
                    _logger.LogInformation(
                        "✓ System VM {VmId} ({VmType}) started — services will re-register via heartbeat",
                        vm.VmId, vm.Spec.VmType);
                }
                else
                {
                    _logger.LogWarning(
                        "✗ System VM {VmId} ({VmType}) failed to start: {Error} — " +
                        "orchestrator reconciliation will redeploy",
                        vm.VmId, vm.Spec.VmType, result.ErrorMessage);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "Exception starting system VM {VmId} ({VmType}) — " +
                    "orchestrator reconciliation will redeploy",
                    vm.VmId, vm.Spec.VmType);
            }
        }
    }
}