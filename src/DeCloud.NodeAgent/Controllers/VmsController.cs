using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.Shared.Enums;
using DeCloud.Shared.Models;
using Microsoft.AspNetCore.Mvc;

namespace DeCloud.NodeAgent.Controllers;

[ApiController]
[Route("api/[controller]")]
public class VmsController : ControllerBase
{
    private readonly IVmManager _vmManager;
    private readonly IVmGuestDiagnostics _diagnostics;
    private readonly ILogger<VmsController> _logger;

    public VmsController(
        IVmManager vmManager,
        IVmGuestDiagnostics diagnostics,
        ILogger<VmsController> logger)
    {
        _vmManager = vmManager;
        _diagnostics = diagnostics;
        _logger = logger;
    }

    /// <summary>
    /// List all VMs on this node
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<List<VmInstance>>> GetAll(CancellationToken ct)
    {
        var vms = _vmManager.GetAllVms();
        return Ok(vms);
    }

    /// <summary>
    /// Get a specific VM
    /// </summary>
    [HttpGet("{vmId}")]
    public async Task<ActionResult<VmInstance>> Get(string vmId, CancellationToken ct)
    {
        var vm = await _vmManager.GetVmAsync(vmId, ct);
        if (vm == null)
            return NotFound();
        return Ok(vm);
    }

    /// <summary>
    /// Create a new VM
    /// </summary>
    [HttpPost]
    public async Task<ActionResult<VmOperationResult>> Create([FromBody] VmSpec spec, string password, CancellationToken ct)
    {
        _logger.LogInformation("API: Creating VM {Name}", spec.Name);
        
        var result = await _vmManager.CreateVmAsync(spec, password, ct);
        
        if (!result.Success)
            return BadRequest(result);
        
        return CreatedAtAction(nameof(Get), new { vmId = result.VmId }, result);
    }

    [HttpPost]
    public async Task<ActionResult<VmOperationResult>> Sync(CancellationToken ct)
    {
        _logger.LogInformation("API: Syncing all vms");
        try
        {
            await _vmManager.ReconcileAllWithLibvirtAsync(ct);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "API: Error during VM sync");
            return BadRequest("Failed to sync virtual machines");
        }            

        return Ok("");
    }

    /// <summary>
    /// Start a VM
    /// </summary>
    [HttpPost("{vmId}/start")]
    public async Task<ActionResult<VmOperationResult>> Start(string vmId, CancellationToken ct)
    {
        _logger.LogInformation("API: Starting VM {VmId}", vmId);
        
        var result = await _vmManager.StartVmAsync(vmId, ct);
        
        if (!result.Success)
            return BadRequest(result);
        
        return Ok(result);
    }

    /// <summary>
    /// Stop a VM
    /// </summary>
    [HttpPost("{vmId}/stop")]
    public async Task<ActionResult<VmOperationResult>> Stop(
        string vmId, 
        [FromQuery] bool force = false, 
        CancellationToken ct = default)
    {
        _logger.LogInformation("API: Stopping VM {VmId} (force={Force})", vmId, force);
        
        var result = await _vmManager.StopVmAsync(vmId, force, ct);
        
        if (!result.Success)
            return BadRequest(result);
        
        return Ok(result);
    }

    /// <summary>
    /// Delete a VM
    /// </summary>
    [HttpDelete("{vmId}")]
    public async Task<ActionResult<VmOperationResult>> Delete(string vmId, CancellationToken ct)
    {
        _logger.LogInformation("API: Deleting VM {VmId}", vmId);
        
        var result = await _vmManager.DeleteVmAsync(vmId, ct);
        
        if (!result.Success)
            return BadRequest(result);
        
        return Ok(result);
    }

    /// <summary>
    /// Hard reboot a VM via virsh destroy + start.
    /// Works regardless of guest agent or ACPI support.
    /// Used by CLI to recover zombie VMs where virsh reboot/reset won't work.
    /// </summary>
    [HttpPost("{vmId}/hard-reboot")]
    public async Task<ActionResult<VmOperationResult>> HardReboot(string vmId, CancellationToken ct)
    {
        var vm = _vmManager.GetAllVms().FirstOrDefault(v => v.VmId == vmId);
        if (vm == null)
            return NotFound(new { error = $"VM {vmId} not found" });

        _logger.LogInformation("Hard reboot requested for VM {VmId} via CLI", vmId);

        // RestartVmAsync(force: true) issues virsh destroy + start internally.
        // This is a hard power cycle — no guest cooperation needed.
        var result = await _vmManager.RestartVmAsync(vmId, force: true, ct);

        if (result.Success)
            return Ok(result);

        return StatusCode(500, result);
    }

    /// <summary>
    /// Get resource usage for a VM
    /// </summary>
    [HttpGet("{vmId}/usage")]
    public async Task<ActionResult<VmResourceUsage>> GetUsage(string vmId, CancellationToken ct)
    {
        if (!await _vmManager.VmExistsAsync(vmId, ct))
            return NotFound();

        var usage = await _vmManager.GetVmUsageAsync(vmId, ct);
        return Ok(usage);
    }

    /// <summary>
    /// Get a diagnostic log stream for a VM.
    ///
    /// Default source <c>Console</c> returns the tail of the host-side
    /// <c>console.log</c> captured by libvirt's <c>&lt;log file='...'/&gt;</c>
    /// directive — works even when the guest agent never started (cloud-init
    /// parse error, apt failure, network never up, kernel panic). Other
    /// sources (guest-side cloud-init.log, journal) land in follow-ups.
    ///
    /// Ownership check is performed by the orchestrator-side proxy endpoint —
    /// the node agent API has no per-user identity and follows the existing
    /// pattern of the surrounding VM endpoints.
    /// </summary>
    [HttpGet("{vmId}/logs")]
    public async Task<ActionResult<DiagnosticsResult>> GetLogs(
        string vmId,
        [FromQuery] DiagnosticSource source = DiagnosticSource.Console,
        [FromQuery] int maxBytes = 0,
        CancellationToken ct = default)
    {
        var result = await _diagnostics.CaptureAsync(vmId, source, maxBytes, ct);

        if (!result.Available)
            return NotFound(result);

        return Ok(result);
    }
}