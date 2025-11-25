using System.Diagnostics;
using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

public class CommandExecutor : ICommandExecutor
{
    private readonly ILogger<CommandExecutor> _logger;
    private static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(30);

    public CommandExecutor(ILogger<CommandExecutor> logger)
    {
        _logger = logger;
    }

    public Task<CommandResult> ExecuteAsync(string command, string arguments, CancellationToken ct = default)
    {
        return ExecuteAsync(command, arguments, DefaultTimeout, ct);
    }

    public async Task<CommandResult> ExecuteAsync(string command, string arguments, TimeSpan timeout, CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        
        _logger.LogDebug("Executing: {Command} {Arguments}", command, arguments);

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(timeout);

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = command,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = new Process { StartInfo = psi };
            
            var stdoutTask = new TaskCompletionSource<string>();
            var stderrTask = new TaskCompletionSource<string>();

            process.OutputDataReceived += (_, e) =>
            {
                if (e.Data == null)
                    stdoutTask.TrySetResult(string.Empty);
            };
            
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data == null)
                    stderrTask.TrySetResult(string.Empty);
            };

            process.Start();
            
            var stdout = await process.StandardOutput.ReadToEndAsync(cts.Token);
            var stderr = await process.StandardError.ReadToEndAsync(cts.Token);
            
            await process.WaitForExitAsync(cts.Token);
            
            sw.Stop();

            var result = new CommandResult
            {
                ExitCode = process.ExitCode,
                StandardOutput = stdout,
                StandardError = stderr,
                Duration = sw.Elapsed
            };

            if (result.Success)
            {
                _logger.LogDebug("Command succeeded in {Duration}ms", sw.ElapsedMilliseconds);
            }
            else
            {
                _logger.LogWarning("Command failed with exit code {ExitCode}: {Stderr}", 
                    result.ExitCode, result.StandardError);
            }

            return result;
        }
        catch (OperationCanceledException)
        {
            sw.Stop();
            _logger.LogError("Command timed out after {Timeout}ms", timeout.TotalMilliseconds);
            
            return new CommandResult
            {
                ExitCode = -1,
                StandardError = $"Command timed out after {timeout.TotalSeconds}s",
                Duration = sw.Elapsed
            };
        }
        catch (Exception ex)
        {
            sw.Stop();
            _logger.LogError(ex, "Command execution failed: {Message}", ex.Message);
            
            return new CommandResult
            {
                ExitCode = -1,
                StandardError = ex.Message,
                Duration = sw.Elapsed
            };
        }
    }
}
