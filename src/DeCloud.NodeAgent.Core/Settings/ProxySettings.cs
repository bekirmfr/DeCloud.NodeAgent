using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeCloud.NodeAgent.Core.Settings
{
    // =====================================================
    // ProxySettings Model (Optional Configuration)
    // =====================================================
    /// <summary>
    /// Configuration model for GenericProxyController.
    /// GenericProxyController has sensible defaults and works without this,
    /// but this allows runtime customization via appsettings.json.
    /// </summary>
    public class ProxySettings
    {
        /// <summary>
        /// List of ports that can be proxied. If null or empty, uses controller defaults.
        /// </summary>
        public List<int>? AllowedPorts { get; set; }

        /// <summary>
        /// Ports that require authentication (future enhancement)
        /// </summary>
        public List<int>? ProtectedPorts { get; set; }

        /// <summary>
        /// Timeout in milliseconds per port. Key is port number (0 = default).
        /// </summary>
        public Dictionary<int, int>? PortTimeouts { get; set; }

        /// <summary>
        /// Enable port whitelist validation (default: true)
        /// </summary>
        public bool EnablePortWhitelist { get; set; } = true;

        /// <summary>
        /// Block system ports (1-1024) unless explicitly allowed
        /// </summary>
        public bool BlockSystemPorts { get; set; } = true;

        /// <summary>
        /// Maximum concurrent TCP tunnels per VM
        /// </summary>
        public int MaxConcurrentTunnelsPerVm { get; set; } = 10;

        /// <summary>
        /// Enable rate limiting (future enhancement)
        /// </summary>
        public bool EnableRateLimiting { get; set; } = false;

        /// <summary>
        /// Max requests per minute per VM if rate limiting enabled
        /// </summary>
        public int MaxRequestsPerMinutePerVm { get; set; } = 1000;
    }
}
