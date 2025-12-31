using System.Diagnostics;
using System.Text.RegularExpressions;
using DeCloud.NodeAgent.Core.Interfaces;
using Microsoft.Extensions.Logging;

namespace DeCloud.NodeAgent.Infrastructure.Services;

/// <summary>
/// CPU benchmarking service for measuring node performance
/// Runs on node agent during registration to determine tier eligibility
/// </summary>
public interface ICpuBenchmarkService
{
    Task<BenchmarkResult> RunBenchmarkAsync(CancellationToken ct = default);
}

public class CpuBenchmarkService : ICpuBenchmarkService
{
    private readonly ICommandExecutor _executor;
    private readonly ILogger<CpuBenchmarkService> _logger;

    public CpuBenchmarkService(
        ICommandExecutor executor,
        ILogger<CpuBenchmarkService> logger)
    {
        _executor = executor;
        _logger = logger;
    }

    public async Task<BenchmarkResult> RunBenchmarkAsync(CancellationToken ct = default)
    {
        _logger.LogInformation("Starting CPU benchmark...");
        var startTime = DateTime.UtcNow;

        try
        {
            // Try sysbench first (preferred for Linux)
            var sysbenchResult = await TrySysbenchAsync(ct);
            if (sysbenchResult != null)
            {
                _logger.LogInformation(
                    "✓ Sysbench completed: {Score} score in {Duration:F1}s ({Details})",
                    sysbenchResult.Score,
                    (DateTime.UtcNow - startTime).TotalSeconds,
                    sysbenchResult.Details);
                return sysbenchResult;
            }

            // Fallback to custom benchmark
            _logger.LogInformation("Sysbench not available, using custom benchmark...");
            var customResult = await RunCustomBenchmarkAsync(ct);

            _logger.LogInformation(
                "✓ Custom benchmark completed: {Score} score in {Duration:F1}s ({Details})",
                customResult.Score,
                (DateTime.UtcNow - startTime).TotalSeconds,
                customResult.Details);

            return customResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Benchmark failed, using minimum score");
            return new BenchmarkResult
            {
                Score = 500, // Conservative fallback
                Method = "error-fallback",
                Duration = DateTime.UtcNow - startTime,
                Error = ex.Message
            };
        }
    }

    private async Task<BenchmarkResult?> TrySysbenchAsync(CancellationToken ct)
    {
        try
        {
            var startTime = DateTime.UtcNow;

            // First check if sysbench is available
            var checkResult = await _executor.ExecuteAsync("which", "sysbench", ct);
            if (!checkResult.Success)
            {
                _logger.LogDebug("sysbench not found in PATH");
                return null;
            }

            // Run sysbench CPU test (prime number calculation)
            // Single-threaded, 15 second test
            var result = await _executor.ExecuteAsync(
                "sysbench",
                "cpu --cpu-max-prime=20000 --threads=1 --time=15 run",
                ct);

            if (!result.Success)
            {
                _logger.LogWarning("sysbench execution failed: {Error}", result.StandardError);
                return null;
            }

            // Parse events per second from output
            var match = Regex.Match(result.StandardOutput, @"events per second:\s+([\d.]+)");
            if (!match.Success)
            {
                _logger.LogWarning("Could not parse sysbench output");
                return null;
            }

            if (!double.TryParse(match.Groups[1].Value, out var eventsPerSecond))
            {
                return null;
            }

            // Normalize to our scoring scale
            // Baseline: ~200 events/sec = 1000 score
            // Formula: score = events_per_sec × 5
            // Examples:
            //   - 200 events/sec → 1000 (Burstable baseline)
            //   - 300 events/sec → 1500 (Balanced tier)
            //   - 500 events/sec → 2500 (Standard tier)
            //   - 800 events/sec → 4000 (Guaranteed tier)
            var score = (int)(eventsPerSecond * 5);

            return new BenchmarkResult
            {
                Score = score,
                Method = "sysbench",
                Duration = DateTime.UtcNow - startTime,
                RawMetric = eventsPerSecond,
                Details = $"{eventsPerSecond:F2} events/sec"
            };
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "sysbench failed");
            return null;
        }
    }

    private Task<BenchmarkResult> RunCustomBenchmarkAsync(CancellationToken ct)
    {
        return Task.Run(() =>
        {
            var startTime = DateTime.UtcNow;
            var stopwatch = Stopwatch.StartNew();

            // Run single-threaded prime calculation benchmark
            int primeCount = 0;
            const int maxNumber = 100000;

            for (int n = 2; n < maxNumber && !ct.IsCancellationRequested; n++)
            {
                if (IsPrime(n))
                    primeCount++;
            }

            stopwatch.Stop();

            // Calculate score based on primes found per second
            // Baseline: ~5000 primes in 1 second = 1000 score
            // Formula: score = (primes/sec) / 5
            // Examples:
            //   - 5000 primes/sec → 1000 (Burstable baseline)
            //   - 7500 primes/sec → 1500 (Balanced tier)
            //   - 12500 primes/sec → 2500 (Standard tier)
            //   - 20000 primes/sec → 4000 (Guaranteed tier)
            var primesPerSecond = primeCount / stopwatch.Elapsed.TotalSeconds;
            var score = (int)(primesPerSecond / 5);

            // Clamp to reasonable range (prevent crazy high scores)
            score = Math.Clamp(score, 100, 10000);

            return new BenchmarkResult
            {
                Score = score,
                Method = "custom-prime",
                Duration = DateTime.UtcNow - startTime,
                RawMetric = primesPerSecond,
                Details = $"{primeCount} primes in {stopwatch.Elapsed.TotalSeconds:F2}s ({primesPerSecond:F2} primes/sec)"
            };
        }, ct);
    }

    private static bool IsPrime(int n)
    {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;

        var sqrt = (int)Math.Sqrt(n);
        for (int i = 3; i <= sqrt; i += 2)
        {
            if (n % i == 0)
                return false;
        }
        return true;
    }
}

/// <summary>
/// Benchmark result with normalized score
/// </summary>
public class BenchmarkResult
{
    /// <summary>
    /// Normalized benchmark score (100-10000 scale)
    /// 1000 = Burstable baseline
    /// 1500 = Balanced tier minimum
    /// 2500 = Standard tier minimum
    /// 4000 = Guaranteed tier minimum
    /// </summary>
    public int Score { get; set; }

    /// <summary>
    /// Benchmark method used (sysbench, custom-prime, error-fallback)
    /// </summary>
    public string Method { get; set; } = string.Empty;

    /// <summary>
    /// Time taken to run benchmark
    /// </summary>
    public TimeSpan Duration { get; set; }

    /// <summary>
    /// Raw metric from benchmark tool
    /// For sysbench: events per second
    /// For custom: primes per second
    /// </summary>
    public double RawMetric { get; set; }

    /// <summary>
    /// Human-readable details
    /// </summary>
    public string? Details { get; set; }

    /// <summary>
    /// Error message if benchmark failed
    /// </summary>
    public string? Error { get; set; }
}