using DeCloud.NodeAgent.Contracts.Response.Network;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Interfaces.State;
using DeCloud.NodeAgent.Core.Models;
using DeCloud.NodeAgent.Services;
using DeCloud.Shared.Contracts;
using Microsoft.AspNetCore.Mvc;
using Orchestrator.Models;

namespace DeCloud.NodeAgent.Controllers;

[ApiController]
[Route("api/[controller]")]
public class OrchestratorController : ControllerBase
{
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly INodeStateService _nodeStateService;
    private readonly ILogger<NodeController> _logger;

    public OrchestratorController(
        IOrchestratorClient orchestratorClient,
        INodeStateService nodeStateService,
        ILogger<NodeController> logger)
    {
        _orchestratorClient = orchestratorClient;
        _nodeStateService = nodeStateService;
        _logger = logger;
    }

    /// <summary>
    /// Get node performance information
    /// </summary>
    [HttpGet("config")]
    public async Task<ActionResult<SchedulingConfig>> GetSchedulingonfig(CancellationToken ct)
    {
        var config = await _orchestratorClient.GetSchedulingConfigAsync(ct);

        if (config == null)
        {
            return NotFound("Scheduling configuration not available.");
        }

        return Ok(config);
    }

    /// <summary>
    /// Get node performance information
    /// </summary>
    [HttpGet("performance")]
    public async Task<ActionResult<NodePerformanceEvaluation>> GetPerformance(CancellationToken ct)
    {
        var performance = await _orchestratorClient.GetPerformanceEvaluationAsync(ct);

        if (performance == null) {
            return NotFound("Performance evaluation not available.");
        }

        return Ok(performance);
    }

    /// <summary>
    /// Get node capacity information
    /// </summary>
    [HttpGet("capacity")]
    public async Task<ActionResult<NodeCapacityResponse>> GetCapacity(CancellationToken ct)
    {
        var capacity = await _orchestratorClient.GetCapacityAsync(ct);

        if(capacity == null) {
            return NotFound("Node capacity not available.");
        }
        return Ok(capacity);
    }

    /// <summary>
    /// Get node summary
    /// </summary>
    [HttpGet("summary")]
    public async Task<ActionResult<NodeSummaryResponse>> GetTier(CancellationToken ct)
    {
        var summary = await _orchestratorClient.GetNodeSummaryAsync(ct);

        if (summary == null)
        {
            return NotFound("Node summary not available.");
        }
        return Ok(summary);
    }

    /// <summary>
    /// Request full evaluation (benchmark + obligations + templates).
    /// Primary lifecycle step after register, before login.
    /// </summary>
    [HttpPost("evaluate")]
    public async Task<ActionResult<EvaluateNodeResponse>> RequestEvaluate(CancellationToken ct)
    {
        var evaluation = await _orchestratorClient.EvaluateNodeAsync(ct);

        if (evaluation == null)
        {
            return StatusCode(502, "Evaluation failed — could not reach orchestrator.");
        }
        return Ok(evaluation);
    }

    /// <summary>
    /// Push resource allocation to orchestrator.
    /// Called by CLI: POST /api/orchestrator/allocate
    /// </summary>
    [HttpPost("allocate")]
    public async Task<ActionResult<NodeAllocateResponse>> Allocate(
        [FromBody] NodeAllocateRequest request,
        CancellationToken ct)
    {
        var response = await _orchestratorClient.AllocateAsync(request, ct);

        if (response == null)
        {
            return StatusCode(502, "Failed to reach orchestrator");
        }

        if (!response.Success)
        {
            return BadRequest(response);
        }

        return Ok(response);
    }

}