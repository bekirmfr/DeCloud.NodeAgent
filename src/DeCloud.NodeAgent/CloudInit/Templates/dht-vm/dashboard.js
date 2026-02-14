/**
 * DeCloud DHT Node Dashboard - JavaScript
 * Version: 1.0.0
 *
 * Fetches data from the DHT node's HTTP API (/health, /peers)
 * and renders live stats on the dashboard.
 */

// ==================== Configuration ====================
const CONFIG = {
    vmId: '__VM_ID__',
    vmName: '__VM_NAME__',
    region: '__DHT_REGION__',
    nodeId: '__NODE_ID__',

    refreshInterval: 10000,  // 10 seconds

    api: {
        health: '/health',
        peers: '/peers'
    }
};

// ==================== State ====================
const state = {
    initialized: false,
    lastUpdate: null,
    updateCount: 0,
    errors: [],
    healthData: null,
    peersData: null
};

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DeCloud DHT Dashboard initializing...');
    console.log('  VM ID: ' + CONFIG.vmId);
    console.log('  Region: ' + CONFIG.region);
    initializeDashboard();
});

async function initializeDashboard() {
    try {
        await updateDashboard();
        setInterval(updateDashboard, CONFIG.refreshInterval);
        state.initialized = true;
        hideLoadingOverlay();
        console.log('Dashboard initialized');
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
        showAlert('Failed to load dashboard. Please refresh the page.');
        hideLoadingOverlay();
    }
}

// ==================== Main Update Loop ====================
async function updateDashboard() {
    try {
        const [health, peers] = await Promise.all([
            fetchAPI(CONFIG.api.health),
            fetchAPI(CONFIG.api.peers)
        ]);

        state.healthData = health;
        state.peersData = peers;
        state.lastUpdate = new Date();
        state.updateCount++;

        renderOverallStatus();
        renderNodeIdentity();
        renderNetworkStats();
        renderPeersList();
        renderAddresses();
        renderLastUpdate();

        if (state.errors.length > 0) {
            state.errors = [];
            closeAlert();
        }
    } catch (error) {
        console.error('Dashboard update failed:', error);
        state.errors.push(error);
        if (state.errors.length === 1) {
            showAlert('Failed to fetch DHT node data. Retrying...');
        }
    }
}

// ==================== API ====================
async function fetchAPI(endpoint) {
    const response = await fetch(endpoint, {
        method: 'GET',
        headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
    });
    if (!response.ok) throw new Error('API ' + endpoint + ' returned ' + response.status);
    return response.json();
}

// ==================== Rendering ====================

function renderOverallStatus() {
    const dot = document.getElementById('overall-status');
    const text = document.getElementById('status-text');
    if (!state.healthData) {
        setStatus(dot, text, 'checking', 'Checking...');
        return;
    }

    var peerCount = state.healthData.connectedPeers || 0;
    var nodeStatus = state.healthData.status || 'unknown';

    if (nodeStatus === 'shutting_down') {
        setStatus(dot, text, 'offline', 'Shutting Down');
    } else if (peerCount === 0) {
        setStatus(dot, text, 'warning', 'No Peers');
    } else {
        setStatus(dot, text, 'online', 'Active (' + peerCount + ' peers)');
    }
}

function renderNodeIdentity() {
    if (!state.healthData) return;
    var d = state.healthData;

    var peerId = d.peerId || '-';
    document.getElementById('peer-id').textContent = peerId;
    document.getElementById('peer-id').title = peerId;
    document.getElementById('node-status').textContent = d.status || '-';
    document.getElementById('uptime-value').textContent = d.uptime || '-';

    var advIp = document.getElementById('advertise-ip');
    if (advIp && advIp.textContent === '') {
        advIp.textContent = '(not set)';
    }
}

function renderNetworkStats() {
    if (!state.healthData) return;
    var d = state.healthData;

    document.getElementById('connected-peers').textContent = d.connectedPeers || 0;
    document.getElementById('routing-table').textContent = d.routingTable || 0;
    document.getElementById('address-count').textContent =
        (d.addresses ? d.addresses.length : 0);
}

function renderPeersList() {
    var container = document.getElementById('peers-list');
    if (!state.peersData) return;

    var peers = state.peersData.peers || [];
    var count = state.peersData.count || 0;

    document.getElementById('peer-count-badge').textContent = count;
    document.getElementById('footer-peers').textContent = count + ' peer' + (count !== 1 ? 's' : '');

    if (count === 0) {
        container.innerHTML =
            '<div class="empty-state">' +
            '<span class="empty-icon">&#x1f50d;</span>' +
            '<p>No peers connected</p>' +
            '<p style="font-size: 0.75rem; margin-top: 0.5rem;">Waiting for bootstrap peers or peer discovery</p>' +
            '</div>';
        return;
    }

    var html = '';
    for (var i = 0; i < peers.length; i++) {
        var peerId = peers[i];
        var shortId = peerId.length > 16
            ? peerId.substring(0, 12) + '...' + peerId.substring(peerId.length - 4)
            : peerId;

        html +=
            '<div class="connection-item">' +
            '  <div class="connection-header">' +
            '    <span class="connection-status" data-status="online"></span>' +
            '    <span class="connection-name">Peer ' + (i + 1) + '</span>' +
            '  </div>' +
            '  <div class="connection-details">' +
            '    <div class="detail-row">' +
            '      <span class="detail-label">Peer ID:</span>' +
            '      <span class="detail-value" title="' + peerId + '">' + shortId + '</span>' +
            '    </div>' +
            '  </div>' +
            '</div>';
    }

    container.innerHTML = html;
}

function renderAddresses() {
    var container = document.getElementById('addresses-list');
    if (!state.healthData || !state.healthData.addresses) return;

    var addrs = state.healthData.addresses;
    if (addrs.length === 0) {
        container.innerHTML =
            '<p style="text-align: center; color: var(--text-muted); padding: 1rem;">No addresses</p>';
        return;
    }

    var html = '<div class="wireguard-info">';
    for (var i = 0; i < addrs.length; i++) {
        html +=
            '<div class="info-row">' +
            '  <span class="info-label">Address ' + (i + 1) + ':</span>' +
            '  <span class="info-value mono" style="font-size: 0.75rem; word-break: break-all;">' +
                addrs[i] +
            '  </span>' +
            '</div>';
    }
    html += '</div>';
    container.innerHTML = html;
}

function renderLastUpdate() {
    if (!state.lastUpdate) return;
    var el = document.getElementById('last-update');
    var seconds = Math.floor((new Date() - state.lastUpdate) / 1000);
    el.textContent = seconds < 60 ? seconds + 's ago' : state.lastUpdate.toLocaleTimeString();
}

// ==================== Helpers ====================

function setStatus(dot, text, status, label) {
    if (dot) dot.className = 'status-dot ' + status;
    if (text) text.textContent = label;
}

function showAlert(message) {
    var banner = document.getElementById('alert-banner');
    document.getElementById('alert-message').textContent = message;
    banner.className = 'alert-banner warning';
    banner.style.display = 'flex';
}

function closeAlert() {
    document.getElementById('alert-banner').style.display = 'none';
}

function hideLoadingOverlay() {
    var overlay = document.getElementById('loading-overlay');
    if (!overlay) return;
    overlay.classList.add('hidden');
    setTimeout(function() { overlay.style.display = 'none'; }, 300);
}

// ==================== Debug ====================
window.debugState = function() {
    console.log('=== DHT Dashboard State ===');
    console.log('Health:', state.healthData);
    console.log('Peers:', state.peersData);
    console.log('Updates:', state.updateCount);
    console.log('Last Update:', state.lastUpdate);
};
