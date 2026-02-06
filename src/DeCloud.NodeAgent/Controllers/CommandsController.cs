using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.AspNetCore.Mvc;
using System.Collections.Concurrent;

namespace DeCloud.NodeAgent.Controllers;

/// <summary>
/// Controller for receiving pushed commands from orchestrator
/// Part of hybrid push-pull command delivery system
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class CommandsController : ControllerBase
{
    private readonly ConcurrentQueue<PendingCommand> _pushedCommands;
    private readonly ILogger<CommandsController> _logger;

    public CommandsController(
        ConcurrentQueue<PendingCommand> pushedCommands,
        ILogger<CommandsController> logger)
    {
        _pushedCommands = pushedCommands;
        _logger = logger;
    }

    /// <summary>
    /// Receive pushed command from orchestrator
    /// This endpoint allows orchestrator to push commands for instant delivery (~100-200ms)
    /// </summary>
    [HttpPost("receive")]
    [ProducesResponseType(typeof(CommandReceiveResponse), 200)]
    [ProducesResponseType(400)]
    public IActionResult ReceivePushedCommand([FromBody] PushedCommand command)
    {
        if (string.IsNullOrEmpty(command.CommandId))
        {
            _logger.LogWarning("Received push command with missing CommandId");
            return BadRequest(new { error = "CommandId required" });
        }

        _logger.LogInformation(
            "📥 Received pushed command {CommandId}: {Type}",
            command.CommandId, command.Type);

        // Convert to internal format
        var pendingCommand = new PendingCommand
        {
            CommandId = command.CommandId,
            Type = ConvertCommandType(command.Type),
            Payload = command.Payload,
            RequiresAck = command.RequiresAck,
            IssuedAt = DateTime.UtcNow
        };

        // Queue for immediate processing
        _pushedCommands.Enqueue(pendingCommand);

        _logger.LogDebug(
            "Command {CommandId} queued for processing (pushed queue size: {QueueSize})",
            command.CommandId, _pushedCommands.Count);

        return Ok(new CommandReceiveResponse
        {
            Received = true,
            CommandId = command.CommandId,
            QueuedAt = DateTime.UtcNow
        });
    }

    /// <summary>
    /// Health check endpoint for push capability testing
    /// Used by orchestrator to verify node can receive pushes
    /// </summary>
    [HttpGet("health")]
    [ProducesResponseType(typeof(PushHealthResponse), 200)]
    public IActionResult HealthCheck()
    {
        return Ok(new PushHealthResponse
        {
            PushCapable = true,
            QueueSize = _pushedCommands.Count,
            Timestamp = DateTime.UtcNow
        });
    }

    /// <summary>
    /// Convert orchestrator command type to node agent command type
    /// </summary>
    private CommandType ConvertCommandType(int typeValue)
    {
        return typeValue switch
        {
            0 => CommandType.CreateVm,       // NodeCommandType.CreateVm
            1 => CommandType.StartVm,        // NodeCommandType.StartVm (FIXED: was backwards)
            2 => CommandType.StopVm,         // NodeCommandType.StopVm (FIXED: was backwards)
            3 => CommandType.DeleteVm,       // NodeCommandType.DeleteVm
            4 => CommandType.UpdateNetwork,  // NodeCommandType.UpdateNetwork
            5 => CommandType.Benchmark,      // NodeCommandType.Benchmark
            6 => CommandType.Shutdown,       // NodeCommandType.Shutdown
            7 => CommandType.AllocatePort,   // NodeCommandType.AllocatePort
            8 => CommandType.RemovePort,     // NodeCommandType.RemovePort
            _ => throw new ArgumentException($"Unknown command type: {typeValue}")
        };
    }
}

/// <summary>
/// Command as received from orchestrator push
/// </summary>
public class PushedCommand
{
    public string CommandId { get; set; } = string.Empty;
    public int Type { get; set; }  // NodeCommandType as int
    public string Payload { get; set; } = string.Empty;
    public bool RequiresAck { get; set; } = true;
}

/// <summary>
/// Response to pushed command
/// </summary>
public class CommandReceiveResponse
{
    public bool Received { get; set; }
    public string CommandId { get; set; } = string.Empty;
    public DateTime QueuedAt { get; set; }
}

/// <summary>
/// Push capability health check response
/// </summary>
public class PushHealthResponse
{
    public bool PushCapable { get; set; }
    public int QueueSize { get; set; }
    public DateTime Timestamp { get; set; }
}