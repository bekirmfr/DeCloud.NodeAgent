using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;

namespace DeCloud.NodeAgent.Infrastructure.Persistence
{
    public class DatabaseMaintenanceService : BackgroundService
    {
        private readonly VmRepository _repository;
        private readonly ILogger<DatabaseMaintenanceService> _logger;

        public DatabaseMaintenanceService(VmRepository repository, ILogger<DatabaseMaintenanceService> logger)
        {
            _repository = repository;
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Database maintenance service started");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    // Run daily at 3 AM or every 24 hours
                    await Task.Delay(TimeSpan.FromHours(24), stoppingToken);

                    // Purge VMs deleted more than 7 days ago
                    await _repository.PurgeDeletedVmsAsync(TimeSpan.FromDays(7));

                    var stats = await _repository.GetStatsAsync();
                    _logger.LogInformation(
                        "Database stats: {TotalVms} VMs, {SizeKB} KB",
                        stats.TotalVms,
                        stats.DatabaseSizeBytes / 1024);
                }
                catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
                {
                    // Graceful shutdown. This service spends essentially all its time in
                    // the 24h Task.Delay above, so cancellation there is its normal exit
                    // path — not a maintenance failure. Previously logged at Error with a
                    // stack trace on every single stop (observed 2026-08-05 02:10:21).
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Database maintenance failed");
                }
            }
        }
    }
}
