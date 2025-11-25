using System.Text.Json;
using DeCloud.NodeAgent.Core.Interfaces;
using DeCloud.NodeAgent.Core.Models;
using Microsoft.Extensions.Options;

namespace DeCloud.NodeAgent.Services;

public class CommandProcessorOptions
{
    public TimeSpan PollInterval { get; set; } = TimeSpan.FromSeconds(5);
}

/// <summary>
/// Polls for and executes commands from the orchestrator
/// </summary>
public class CommandProcessorService : BackgroundService
{
    private readonly IOrchestratorClient _orchestratorClient;
    private readonly IVmManager _vmManager;
    private readonly IResourceDiscoveryService _resourceDiscovery;
    private readonly ILogger<CommandProcessorService> _logger;
    private readonly CommandProcessorOptions _options;

    public CommandProcessorService(
        IOrchestratorClient orchestratorClient,
        IVmManager vmManager,
        IResourceDiscoveryService resourceDiscovery,
        IOptions<CommandProcessorOptions> options,
        ILogger<CommandProcessorService> logger)
    {
        _orchestratorClient = orchestratorClient;
        _vmManager = vmManager;
        _resourceDiscovery = resourceDiscovery;
        _logger = logger;
        _options = options.Value;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Command processor starting with poll interval {Interval}s",
            _options.PollInterval.TotalSeconds);

        // Wait for heartbeat service to initialize
        await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await ProcessPendingCommandsAsync(stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing commands");
            }

            await Task.Delay(_options.PollInterval, stoppingToken);
        }
    }

    private async Task ProcessPendingCommandsAsync(CancellationToken ct)
    {
        var commands = await _orchestratorClient.GetPendingCommandsAsync(ct);

        foreach (var command in commands)
        {
            _logger.LogInformation("Processing command {CommandId}: {Type}", 
                command.CommandId, command.Type);

            try
            {
                var success = await ExecuteCommandAsync(command, ct);
                await _orchestratorClient.AcknowledgeCommandAsync(
                    command.CommandId, success, null, ct);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Command {CommandId} failed", command.CommandId);
                await _orchestratorClient.AcknowledgeCommandAsync(
                    command.CommandId, false, ex.Message, ct);
            }
        }
    }

    private async Task<bool> ExecuteCommandAsync(PendingCommand command, CancellationToken ct)
    {
        switch (command.Type)
        {
            case CommandType.CreateVm:
                return await HandleCreateVmAsync(command.Payload, ct);

            case CommandType.StartVm:
                return await HandleStartVmAsync(command.Payload, ct);

            case CommandType.StopVm:
                return await HandleStopVmAsync(command.Payload, ct);

            case CommandType.DeleteVm:
                return await HandleDeleteVmAsync(command.Payload, ct);

            case CommandType.Benchmark:
                return await HandleBenchmarkAsync(ct);

            default:
                _logger.LogWarning("Unknown command type: {Type}", command.Type);
                return false;
        }
    }

    private async Task<bool> HandleCreateVmAsync(string payload, CancellationToken ct)
    {
        var spec = JsonSerializer.Deserialize<VmSpec>(payload);
        if (spec == null)
        {
            _logger.LogError("Invalid CreateVm payload");
            return false;
        }

        var result = await _vmManager.CreateVmAsync(spec, ct);
        
        if (result.Success)
        {
            // Auto-start the VM
            var startResult = await _vmManager.StartVmAsync(spec.VmId, ct);
            return startResult.Success;
        }

        _logger.LogError("CreateVm failed: {Error}", result.ErrorMessage);
        return false;
    }

    private async Task<bool> HandleStartVmAsync(string payload, CancellationToken ct)
    {
        var request = JsonSerializer.Deserialize<VmCommandPayload>(payload);
        if (request == null) return false;

        var result = await _vmManager.StartVmAsync(request.VmId, ct);
        return result.Success;
    }

    private async Task<bool> HandleStopVmAsync(string payload, CancellationToken ct)
    {
        var request = JsonSerializer.Deserialize<VmCommandPayload>(payload);
        if (request == null) return false;

        var result = await _vmManager.StopVmAsync(request.VmId, request.Force, ct);
        return result.Success;
    }

    private async Task<bool> HandleDeleteVmAsync(string payload, CancellationToken ct)
    {
        var request = JsonSerializer.Deserialize<VmCommandPayload>(payload);
        if (request == null) return false;

        var result = await _vmManager.DeleteVmAsync(request.VmId, ct);
        return result.Success;
    }

    private async Task<bool> HandleBenchmarkAsync(CancellationToken ct)
    {
        _logger.LogInformation("Running benchmark...");
        
        // Re-discover resources to update benchmarks
        var resources = await _resourceDiscovery.DiscoverAllAsync(ct);
        
        // TODO: Run actual benchmarks (sysbench, fio, etc.)
        
        _logger.LogInformation("Benchmark complete");
        return true;
    }

    private class VmCommandPayload
    {
        public string VmId { get; set; } = string.Empty;
        public bool Force { get; set; }
    }
}
