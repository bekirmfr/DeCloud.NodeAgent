using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.AspNetCore.Mvc;
using Nethereum.Contracts.QueryHandlers.MultiCall;

namespace DeCloud.NodeAgent.Controllers;

[ApiController]
[Route("api/[controller]")]
public class VmsController : ControllerBase
{
    private readonly IVmManager _vmManager;
    private readonly ILogger<VmsController> _logger;

    public VmsController(IVmManager vmManager, ILogger<VmsController> logger)
    {
        _vmManager = vmManager;
        _logger = logger;
    }

    /// <summary>
    /// List all VMs on this node
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<List<VmInstance>>> GetAll(CancellationToken ct)
    {
        var vms = await _vmManager.GetAllVmsAsync(ct);
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
            await _vmManager.ReconcileWithLibvirtAsync(ct);
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
}
