/**
 * DeCloud Block Store Dashboard - JavaScript
 * Version: 1.0.0
 *
 * Fetches data from /health, /stats, /manifests and renders live stats.
 * Resource types: VMOverlay, ModelShard, LoRAAdapter, ImageTemplate, Unknown.
 */

// ==================== Configuration ====================
const CONFIG = {
    vmId:     '__VM_ID__',
    vmName:   '__VM_NAME__',
    region:   '__NODE_REGION__',
    nodeId:   '__NODE_ID__',
    advertiseIp: '__BLOCKSTORE_ADVERTISE_IP__',

    refreshInterval: 15000,  // 15 seconds

    api: {
        health:    '/health',
        manifests: '/manifests'
    },

    // GC thresholds (must match Go binary constants)
    gcTrigger: 85,
    gcHardLimit: 95
};

// ==================== State ====================
const state = {
    initialized:   false,
    lastUpdate:    null,
    errors:        [],
    healthData:    null,
    manifestsData: null
};

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DeCloud Block Store Dashboard initializing...');
    initializeDashboard();
});

async function initializeDashboard() {
    try {
        await updateDashboard();
        setInterval(updateDashboard, CONFIG.refreshInterval);
        state.initialized = true;
        hideLoadingOverlay();
    } catch (err) {
        console.error('Dashboard init failed:', err);
        showAlert('Failed to load dashboard. Please refresh the page.');
        hideLoadingOverlay();
    }
}

// ==================== Main Update Loop ====================
async function updateDashboard() {
    try {
        const [health, manifests] = await Promise.all([
            fetchAPI(CONFIG.api.health),
            fetchAPI(CONFIG.api.manifests)
        ]);

        state.healthData    = health;
        state.manifestsData = manifests;
        state.lastUpdate    = new Date();

        renderStatus();
        renderStorage();
        renderIdentity();
        renderResourcesTable();
        renderPeersList();
        renderFooter();
        renderLastUpdate();

        if (state.errors.length > 0) {
            state.errors = [];
            closeAlert();
        }
    } catch (err) {
        console.error('Dashboard update failed:', err);
        state.errors.push(err);
        if (state.errors.length === 1) {
            showAlert('Failed to fetch block store data. Retrying...');
        }
    }
}

// ==================== API ====================
async function fetchAPI(endpoint) {
    const resp = await fetch(endpoint, {
        headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
    });
    if (!resp.ok) throw new Error('API ' + endpoint + ' returned ' + resp.status);
    return resp.json();
}

// ==================== Rendering ====================

function renderStatus() {
    const dot  = document.getElementById('overall-status');
    const text = document.getElementById('status-text');
    if (!state.healthData) {
        setStatus(dot, text, 'checking', 'Checking...');
        return;
    }
    const d = state.healthData;
    const usage = d.usagePercent || 0;
    const peers = d.connectedPeers || 0;

    if (usage >= CONFIG.gcHardLimit) {
        setStatus(dot, text, 'offline', 'Storage Full');
    } else if (usage >= CONFIG.gcTrigger) {
        setStatus(dot, text, 'warning', 'GC Triggered (' + usage.toFixed(1) + '%)');
    } else if (peers === 0) {
        setStatus(dot, text, 'warning', 'No Peers');
    } else {
        setStatus(dot, text, 'online', 'Active (' + peers + ' peer' + (peers !== 1 ? 's' : '') + ')');
    }
}

function renderStorage() {
    if (!state.healthData) return;
    const d = state.healthData;

    const usedBytes  = d.usedBytes     || 0;
    const capBytes   = d.capacityBytes || 1;
    const pct        = d.usagePercent  || 0;

    // Progress bar
    const bar = document.getElementById('storage-bar');
    bar.style.width = Math.min(pct, 100).toFixed(1) + '%';
    bar.className = 'storage-bar-fill';
    if (pct >= CONFIG.gcHardLimit) bar.classList.add('danger');
    else if (pct >= CONFIG.gcTrigger) bar.classList.add('warning');

    document.getElementById('used-label').textContent     = formatBytes(usedBytes) + ' used';
    document.getElementById('capacity-label').textContent = formatBytes(capBytes) + ' total';
    document.getElementById('usage-badge').textContent    = pct.toFixed(1) + '%';

    document.getElementById('block-count').textContent    = formatNum(d.blockCount      || 0);
    document.getElementById('manifest-count').textContent = formatNum(d.manifestCount   || 0);
    document.getElementById('bitswap-sent').textContent   = formatNum(d.bitswapSent     || 0);
    document.getElementById('bitswap-recv').textContent   = formatNum(d.bitswapReceived || 0);
}

function renderIdentity() {
    if (!state.healthData) return;
    const d = state.healthData;
    const peerId = d.peerId || '—';
    const el = document.getElementById('peer-id');
    el.textContent = peerId.length > 20
        ? peerId.substring(0, 12) + '...' + peerId.substring(peerId.length - 6)
        : peerId;
    el.title = peerId;

    document.getElementById('connected-peers').textContent = d.connectedPeers || 0;
}

function renderResourcesTable() {
    const wrap = document.getElementById('resources-table-wrap');
    const badge = document.getElementById('resource-count-badge');

    if (!state.manifestsData) return;

    const manifests = state.manifestsData.manifests || [];
    badge.textContent = manifests.length;

    if (manifests.length === 0) {
        wrap.innerHTML =
            '<div class="empty-state">' +
            '<span class="empty-icon">&#x1f4e6;</span>' +
            '<p>No resources stored yet</p>' +
            '<p style="font-size:0.75rem;margin-top:0.5rem;">Blocks will appear here as VM overlays and AI model shards are replicated</p>' +
            '</div>';
        return;
    }

    let html =
        '<table class="resources-table">' +
        '<thead><tr>' +
        '<th>Type</th>' +
        '<th>Resource</th>' +
        '<th>Owner</th>' +
        '<th>Root CID</th>' +
        '<th>Size</th>' +
        '<th>Chunks</th>' +
        '<th>Updated</th>' +
        '</tr></thead>' +
        '<tbody>';

    for (const m of manifests) {
        html += renderResourceRow(m);
    }

    html += '</tbody></table>';
    wrap.innerHTML = html;
}

function renderResourceRow(m) {
    const typeBadge  = resourceTypeBadge(m.resourceType);
    const resourceId = truncate(m.resourceId || '—', 24);
    const owner      = truncateWallet(m.resourceOwner || '—');
    const rootCid    = truncate(m.rootCid || '—', 20);
    const size       = formatBytes(m.totalBytes || 0);
    const chunks     = formatNum(m.chunkCids ? m.chunkCids.length : 0);
    const updated    = timeAgo(m.updatedAt);

    // For model shards, show layer range
    let extraInfo = '';
    if (m.shardMeta) {
        const s = m.shardMeta;
        extraInfo =
            '<div class="shard-info">' +
            s.modelName + ' v' + (s.modelVersion || '?') + ' &bull; ' +
            'shard ' + s.shardIndex + '/' + s.totalShards + ' &bull; ' +
            'layers ' + s.layerStart + '–' + s.layerEnd +
            (s.quantBits ? ' &bull; ' + s.quantBits + 'bit' : '') +
            '</div>';
    }

    return '<tr>' +
        '<td>' + typeBadge + '</td>' +
        '<td>' +
            '<span class="td-mono" title="' + (m.resourceId || '') + '">' + resourceId + '</span>' +
            extraInfo +
        '</td>' +
        '<td><span class="td-mono" title="' + (m.resourceOwner || '') + '">' + owner + '</span></td>' +
        '<td>' +
            '<span class="td-cid" title="' + (m.rootCid || '') + '" onclick="copyText(this)">' +
            rootCid + '</span>' +
        '</td>' +
        '<td class="td-mono">' + size + '</td>' +
        '<td class="td-mono">' + chunks + '</td>' +
        '<td class="td-mono">' + updated + '</td>' +
        '</tr>';
}

function resourceTypeBadge(type) {
    const map = {
        'VMOverlay':     { cls: 'badge-vm',    label: 'VM Overlay' },
        'ModelShard':    { cls: 'badge-model',  label: 'Model Shard' },
        'LoRAAdapter':   { cls: 'badge-lora',   label: 'LoRA' },
        'ImageTemplate': { cls: 'badge-image',  label: 'Image' },
        'Unknown':       { cls: 'badge-unknown', label: 'Unknown' }
    };
    const def = map[type] || map['Unknown'];
    return '<span class="resource-badge ' + def.cls + '">' + def.label + '</span>';
}

function renderPeersList() {
    const container = document.getElementById('peers-list');
    const badge     = document.getElementById('peer-count-badge');
    if (!state.healthData) return;

    const peerCount = state.healthData.connectedPeers || 0;
    badge.textContent = peerCount;

    if (peerCount === 0) {
        container.innerHTML =
            '<div class="empty-state">' +
            '<span class="empty-icon">&#x1f50d;</span>' +
            '<p>No peers connected</p>' +
            '<p style="font-size:0.75rem;margin-top:0.5rem;">Waiting for bootstrap peers via DHT network</p>' +
            '</div>';
        return;
    }

    // The /health endpoint returns connectedPeers count but not peer IDs.
    // Render a summary — individual peer details would require a /peers endpoint
    // which can be added to main.go in a future iteration.
    container.innerHTML =
        '<div class="connection-item">' +
        '<div class="connection-header">' +
        '<span class="connection-status" data-status="online"></span>' +
        '<span class="connection-name">' + peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + ' connected via libp2p</span>' +
        '</div>' +
        '<div class="connection-details">' +
        '<div class="detail-row">' +
        '<span class="detail-label">Protocol</span>' +
        '<span class="detail-value">Kademlia DHT + Bitswap</span>' +
        '</div>' +
        '<div class="detail-row">' +
        '<span class="detail-label">GossipSub</span>' +
        '<span class="detail-value">decloud/blockstore/new-blocks</span>' +
        '</div>' +
        '</div>' +
        '</div>';
}

function renderFooter() {
    if (!state.healthData) return;
    const d = state.healthData;
    document.getElementById('footer-peers').textContent =
        (d.connectedPeers || 0) + ' peer' + ((d.connectedPeers || 0) !== 1 ? 's' : '');
    document.getElementById('footer-storage').textContent =
        'Storage: ' + (d.usagePercent || 0).toFixed(1) + '% used';
}

function renderLastUpdate() {
    if (!state.lastUpdate) return;
    const seconds = Math.floor((new Date() - state.lastUpdate) / 1000);
    document.getElementById('last-update').textContent =
        seconds < 60 ? 'just now' : Math.floor(seconds / 60) + 'm ago';
}

// ==================== Helpers ====================

function setStatus(dot, text, cls, label) {
    dot.className = 'status-dot ' + cls;
    text.textContent = label;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(i === 0 ? 0 : 1) + ' ' + sizes[i];
}

function formatNum(n) {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
    return String(n);
}

function truncate(s, maxLen) {
    if (!s || s.length <= maxLen) return s || '—';
    return s.substring(0, Math.floor(maxLen / 2)) + '…' +
           s.substring(s.length - Math.floor(maxLen / 4));
}

function truncateWallet(addr) {
    if (!addr || addr.length < 10) return addr || '—';
    return addr.substring(0, 6) + '…' + addr.substring(addr.length - 4);
}

function timeAgo(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d)) return '—';
    const seconds = Math.floor((new Date() - d) / 1000);
    if (seconds < 60)   return seconds + 's ago';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
    if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
    return Math.floor(seconds / 86400) + 'd ago';
}

function copyText(el) {
    const text = el.title || el.textContent;
    navigator.clipboard.writeText(text).then(() => {
        const orig = el.textContent;
        el.textContent = 'Copied!';
        setTimeout(() => { el.textContent = orig; }, 1200);
    }).catch(() => {});
}

function showAlert(msg) {
    const banner = document.getElementById('alert-banner');
    document.getElementById('alert-message').textContent = msg;
    banner.style.display = 'flex';
}

function closeAlert() {
    document.getElementById('alert-banner').style.display = 'none';
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.add('hidden');
    setTimeout(() => { overlay.style.display = 'none'; }, 300);
}
