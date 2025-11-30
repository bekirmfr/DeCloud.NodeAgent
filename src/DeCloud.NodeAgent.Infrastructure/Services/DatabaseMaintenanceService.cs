using DeCloud.NodeAgent.Infrastructure.Persistence;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;

namespace DeCloud.NodeAgent.Infrastructure.Services
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
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Database maintenance failed");
                }
            }
        }
    }
}
