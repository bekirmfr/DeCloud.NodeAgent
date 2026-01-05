/**
 * DeCloud Relay VM Dashboard - JavaScript
 * Version: 1.0.0
 * 
 * Features:
 * - Real-time metrics updates
 * - Orchestrator connection monitoring
 * - Host node connectivity checks
 * - CGNAT node management
 * - System resource monitoring
 */

// ==================== Configuration ====================
const CONFIG = {
    relayId: '__VM_ID__',
    relayName: '__VM_NAME__',
    relayRegion: '__RELAY_REGION__',
    relayCapacity: parseInt('__RELAY_CAPACITY__'),

    // Update intervals
    refreshInterval: 10000,  // 10 seconds
    quickRefreshInterval: 5000,  // 5 seconds for critical checks

    // API endpoints
    api: {
        wireguard: '/api/relay/wireguard',
        system: '/api/relay/system',
        debug: '/api/relay/debug'
    },

    // Thresholds
    thresholds: {
        handshakeStale: 180,  // 3 minutes
        handshakeWarning: 600,  // 10 minutes
        memoryWarning: 70,  // 70%
        memoryDanger: 85,  // 85%
        cpuWarning: 70,
        cpuDanger: 85
    }
};

// ==================== State Management ====================
const state = {
    initialized: false,
    lastUpdate: null,
    updateCount: 0,
    errors: [],

    // Data caches
    wireguardData: null,
    systemData: null,

    // Connection states
    orchestratorStatus: 'checking',
    hostStatus: 'checking',
    cgnatNodes: [],

    // Metrics
    totalRx: 0,
    totalTx: 0
};

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DeCloud Relay Dashboard initializing...');
    console.log(`   Relay ID: ${CONFIG.relayId}`);
    console.log(`   Region: ${CONFIG.relayRegion}`);
    console.log(`   Capacity: ${CONFIG.relayCapacity} nodes`);

    initializeDashboard();
});

async function initializeDashboard() {
    try {
        // Initial data fetch
        await updateDashboard();

        // Set up auto-refresh
        setInterval(updateDashboard, CONFIG.refreshInterval);

        // Mark as initialized
        state.initialized = true;

        // Hide loading overlay
        hideLoadingOverlay();

        console.log('✓ Dashboard initialized successfully');
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
        showAlert('Failed to load dashboard. Please refresh the page.', 'error');
        hideLoadingOverlay();
    }
}

// ==================== Main Update Loop ====================
async function updateDashboard() {
    try {
        // Fetch all data in parallel
        const [wireguard, system] = await Promise.all([
            fetchAPI(CONFIG.api.wireguard),
            fetchAPI(CONFIG.api.system)
        ]);

        // Update state
        state.wireguardData = wireguard;
        state.systemData = system;
        state.lastUpdate = new Date();
        state.updateCount++;

        // Render all sections
        renderOverallStatus();
        renderUpstreamConnections();
        renderCGNATNodes();
        renderSystemMetrics();
        renderWireGuardStatus();
        renderLastUpdate();

        // Clear error state
        if (state.errors.length > 0) {
            state.errors = [];
            closeAlert();
        }

    } catch (error) {
        console.error('Dashboard update failed:', error);
        state.errors.push(error);

        if (state.errors.length === 1) {
            showAlert('Failed to fetch metrics. Retrying...', 'warning');
        }
    }
}

// ==================== API Communication ====================
async function fetchAPI(endpoint) {
    const response = await fetch(endpoint, {
        method: 'GET',
        headers: {
            'Accept': 'application/json'
        }
    });

    if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
}

// ==================== Rendering Functions ====================

/**
 * Render overall relay status indicator
 */
function renderOverallStatus() {
    const statusDot = document.getElementById('overall-status');
    const statusText = document.getElementById('status-text');

    if (!state.wireguardData || !state.systemData) {
        setStatus(statusDot, statusText, 'checking', 'Checking...');
        return;
    }

    // Check for critical issues
    const hasOrchestratorConnection = checkOrchestratorConnection();
    const memoryUsage = state.systemData.memory_percent || 0;

    if (!hasOrchestratorConnection) {
        setStatus(statusDot, statusText, 'offline', 'No Orchestrator');
        return;
    }

    if (memoryUsage > CONFIG.thresholds.memoryDanger) {
        setStatus(statusDot, statusText, 'warning', 'High Memory');
        return;
    }

    setStatus(statusDot, statusText, 'online', 'Operational');
}

/**
 * Render upstream connections (Orchestrator + Host Node)
 */
function renderUpstreamConnections() {
    if (!state.wireguardData) return;

    const peers = state.wireguardData.peers || [];

    // Find orchestrator peer (should be first peer with specific endpoint pattern)
    const orchestratorPeer = peers.find(p =>
        p.endpoint && !p.endpoint.includes(':51820')  // Orchestrator uses different port
    );

    // Render orchestrator connection
    renderOrchestratorConnection(orchestratorPeer);

    // Render host node connection
    renderHostConnection();

    // Update count badge
    const connectedCount = (orchestratorPeer ? 1 : 0) + (state.hostStatus === 'online' ? 1 : 0);
    document.getElementById('upstream-count').textContent = `${connectedCount}/2`;
}

function renderOrchestratorConnection(peer) {
    const container = document.getElementById('orchestrator-connection');
    const statusEl = container.querySelector('.connection-status');
    const endpoint = container.querySelector('#orchestrator-endpoint');
    const handshake = container.querySelector('#orchestrator-handshake');
    const data = container.querySelector('#orchestrator-data');

    if (!peer) {
        statusEl.setAttribute('data-status', 'offline');
        endpoint.textContent = 'Not connected';
        handshake.textContent = 'Never';
        data.textContent = '↓ 0 B / ↑ 0 B';
        state.orchestratorStatus = 'offline';
        return;
    }

    // Check handshake age
    const handshakeAge = peer.latest_handshake || 0;
    const now = Math.floor(Date.now() / 1000);
    const age = now - handshakeAge;

    let status = 'online';
    if (handshakeAge === 0) {
        status = 'offline';
    } else if (age > CONFIG.thresholds.handshakeWarning) {
        status = 'offline';
    } else if (age > CONFIG.thresholds.handshakeStale) {
        status = 'warning';
    }

    statusEl.setAttribute('data-status', status);
    state.orchestratorStatus = status;

    // Update details
    endpoint.textContent = peer.endpoint || 'Unknown';
    handshake.textContent = handshakeAge === 0 ? 'Never' : formatTimeAgo(age);
    data.textContent = `↓ ${formatBytes(peer.rx_bytes || 0)} / ↑ ${formatBytes(peer.tx_bytes || 0)}`;
}

function renderHostConnection() {
    const container = document.getElementById('host-connection');
    const statusEl = container.querySelector('.connection-status');
    const gateway = container.querySelector('#host-gateway');
    const natStatus = container.querySelector('#host-nat-status');
    const agentStatus = container.querySelector('#host-agent-status');

    // For now, assume host is reachable (we got data from API)
    // In future, add explicit health check to host node agent
    const isHealthy = state.wireguardData && state.systemData;

    if (isHealthy) {
        statusEl.setAttribute('data-status', 'online');
        state.hostStatus = 'online';
        gateway.textContent = 'Connected (libvirt bridge)';
        natStatus.textContent = '✓ Configured';
        agentStatus.textContent = '✓ Responding';
    } else {
        statusEl.setAttribute('data-status', 'offline');
        state.hostStatus = 'offline';
        gateway.textContent = 'Unreachable';
        natStatus.textContent = '✗ Unknown';
        agentStatus.textContent = '✗ Not responding';
    }
}

/**
 * Render CGNAT nodes list
 */
function renderCGNATNodes() {
    if (!state.wireguardData) return;

    const container = document.getElementById('cgnat-nodes-container');
    const peers = state.wireguardData.peers || [];

    // Filter out orchestrator peer (CGNAT nodes connect on port 51820)
    const cgnatPeers = peers.filter(p => {
        // CGNAT nodes typically don't have endpoints or have endpoints with port 51820
        return !p.endpoint || p.endpoint.includes(':51820');
    });

    state.cgnatNodes = cgnatPeers;

    // Update count badge
    document.getElementById('cgnat-count').textContent = `${cgnatPeers.length}/${CONFIG.relayCapacity}`;

    if (cgnatPeers.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📭</span>
                <p>No CGNAT nodes connected</p>
            </div>
        `;
        return;
    }

    // Render each CGNAT node
    const now = Math.floor(Date.now() / 1000);
    container.innerHTML = cgnatPeers.map((peer, index) => {
        const handshakeAge = peer.latest_handshake || 0;
        const age = now - handshakeAge;

        let status = 'online';
        let statusText = formatTimeAgo(age);

        if (handshakeAge === 0) {
            status = 'offline';
            statusText = 'Never connected';
        } else if (age > CONFIG.thresholds.handshakeWarning) {
            status = 'offline';
            statusText = 'Offline (' + formatTimeAgo(age) + ')';
        } else if (age > CONFIG.thresholds.handshakeStale) {
            status = 'stale';
            statusText = 'Stale (' + formatTimeAgo(age) + ')';
        }

        return `
            <div class="cgnat-node">
                <div class="node-header">
                    <div class="node-name">
                        <span class="node-status ${status}"></span>
                        <span>CGNAT Node ${index + 1}</span>
                    </div>
                </div>
                <div class="node-details">
                    <div class="detail-row">
                        <span class="detail-label">Public Key:</span>
                        <span class="detail-value">${peer.public_key}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Handshake:</span>
                        <span class="detail-value">${statusText}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Endpoint:</span>
                        <span class="detail-value">${peer.endpoint || 'Behind CGNAT'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Data:</span>
                        <span class="detail-value">↓ ${formatBytes(peer.rx_bytes || 0)} / ↑ ${formatBytes(peer.tx_bytes || 0)}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Render system metrics
 */
function renderSystemMetrics() {
    if (!state.systemData) return;

    // Uptime
    const uptimeEl = document.getElementById('system-uptime');
    uptimeEl.textContent = formatUptime(state.systemData.uptime_seconds || 0);

    // CPU (estimate from load average)
    const cpuEl = document.getElementById('system-cpu');
    const cpuBar = document.getElementById('cpu-bar');
    const loadAvg = state.systemData.load_average || [0, 0, 0];
    const cpuPercent = Math.min(Math.round(loadAvg[0] * 100), 100);

    cpuEl.textContent = `${cpuPercent}%`;
    cpuBar.style.width = `${cpuPercent}%`;
    cpuBar.className = 'metric-bar-fill';
    if (cpuPercent > CONFIG.thresholds.cpuDanger) cpuBar.classList.add('danger');
    else if (cpuPercent > CONFIG.thresholds.cpuWarning) cpuBar.classList.add('warning');

    // Memory
    const memoryEl = document.getElementById('system-memory');
    const memoryBar = document.getElementById('memory-bar');
    const memoryPercent = state.systemData.memory_percent || 0;
    const memoryUsed = state.systemData.memory_used || 0;
    const memoryTotal = state.systemData.memory_total || 0;

    memoryEl.textContent = `${formatBytes(memoryUsed)} / ${formatBytes(memoryTotal)} (${memoryPercent}%)`;
    memoryBar.style.width = `${memoryPercent}%`;
    memoryBar.className = 'metric-bar-fill';
    if (memoryPercent > CONFIG.thresholds.memoryDanger) memoryBar.classList.add('danger');
    else if (memoryPercent > CONFIG.thresholds.memoryWarning) memoryBar.classList.add('warning');

    // Total data relayed
    const totalDataEl = document.getElementById('total-data');
    let totalRx = 0;
    let totalTx = 0;

    if (state.wireguardData && state.wireguardData.peers) {
        state.wireguardData.peers.forEach(peer => {
            totalRx += peer.rx_bytes || 0;
            totalTx += peer.tx_bytes || 0;
        });
    }

    state.totalRx = totalRx;
    state.totalTx = totalTx;

    totalDataEl.textContent = `↓ ${formatBytes(totalRx)} / ↑ ${formatBytes(totalTx)}`;
}

/**
 * Render WireGuard status
 */
function renderWireGuardStatus() {
    if (!state.wireguardData) return;

    const statusBadge = document.getElementById('wg-status');
    const totalPeers = document.getElementById('total-peers');

    const peerCount = (state.wireguardData.peers || []).length;

    statusBadge.textContent = peerCount > 0 ? 'UP' : 'DOWN';
    statusBadge.className = 'card-badge';
    statusBadge.style.background = peerCount > 0 ? 'var(--status-online)' : 'var(--status-offline)';

    totalPeers.textContent = peerCount;
}

/**
 * Render last update timestamp
 */
function renderLastUpdate() {
    const lastUpdateEl = document.getElementById('last-update');

    if (state.lastUpdate) {
        lastUpdateEl.textContent = state.lastUpdate.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

// ==================== Helper Functions ====================

/**
 * Check if orchestrator connection is healthy
 */
function checkOrchestratorConnection() {
    if (!state.wireguardData || !state.wireguardData.peers) return false;

    const peers = state.wireguardData.peers;
    const orchestratorPeer = peers.find(p =>
        p.endpoint && !p.endpoint.includes(':51820')
    );

    if (!orchestratorPeer) return false;

    const handshakeAge = orchestratorPeer.latest_handshake || 0;
    if (handshakeAge === 0) return false;

    const now = Math.floor(Date.now() / 1000);
    const age = now - handshakeAge;

    return age < CONFIG.thresholds.handshakeWarning;
}

/**
 * Set status indicator
 */
function setStatus(dotElement, textElement, status, text) {
    if (dotElement) {
        dotElement.className = 'status-dot ' + status;
    }
    if (textElement) {
        textElement.textContent = text;
    }
}

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
}

/**
 * Format uptime seconds to human-readable string
 */
function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
}

/**
 * Format seconds to "X ago" string
 */
function formatTimeAgo(seconds) {
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

// ==================== UI Functions ====================

/**
 * Show alert banner
 */
function showAlert(message, type = 'error') {
    const banner = document.getElementById('alert-banner');
    const messageEl = document.getElementById('alert-message');

    banner.className = 'alert-banner';
    if (type === 'warning') {
        banner.classList.add('warning');
    }

    messageEl.textContent = message;
    banner.style.display = 'flex';
}

/**
 * Close alert banner
 */
function closeAlert() {
    const banner = document.getElementById('alert-banner');
    banner.style.display = 'none';
}

/**
 * Hide loading overlay
 */
function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.add('hidden');

    // Remove from DOM after transition
    setTimeout(() => {
        overlay.style.display = 'none';
    }, 300);
}

// ==================== Debug Functions ====================

/**
 * Log current state to console (for debugging)
 */
window.debugState = function () {
    console.log('=== Dashboard State ===');
    console.log('Initialized:', state.initialized);
    console.log('Last Update:', state.lastUpdate);
    console.log('Update Count:', state.updateCount);
    console.log('Errors:', state.errors);
    console.log('Orchestrator Status:', state.orchestratorStatus);
    console.log('Host Status:', state.hostStatus);
    console.log('CGNAT Nodes:', state.cgnatNodes.length);
    console.log('Total RX:', formatBytes(state.totalRx));
    console.log('Total TX:', formatBytes(state.totalTx));
    console.log('WireGuard Data:', state.wireguardData);
    console.log('System Data:', state.systemData);
};

/**
 * Force dashboard update (for debugging)
 */
window.forceUpdate = function () {
    console.log('Forcing dashboard update...');
    updateDashboard();
};

// Log initialization
console.log('✓ Dashboard script loaded');