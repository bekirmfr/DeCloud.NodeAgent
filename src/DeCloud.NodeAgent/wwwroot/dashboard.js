'use strict';

// ============================================================
// State
// ============================================================
const S = {
    summary: null, snapshot: null, vms: null, downloads: [],
    network: null, ports: null, firewall: null, services: null,
    logs: null, logsVisible: false, allLogs: [],
    activeCmapTab: 'interfaces',
    ingressBaseDomain: null,  // e.g. "vms.stackfi.tech" — set from summary
    vmIngress: {},            // vmId → "https://..." from orchestrator DB
    obligations: [],          // SystemVmObligations from orchestrator
    allocation: null,         // orchestrator-confirmed allocation from /api/node/allocation
};

// ============================================================
// Boot
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initScrollSpy();
    loadFast();
    loadFirewall();
    setInterval(loadFast, 10_000);
    setInterval(loadFirewall, 60_000);
});

// Keyboard: press C to collapse all collapsibles, X to expand all
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'c' || e.key === 'C') {
        document.querySelectorAll('.collapsible-body.open').forEach(el => {
            el.classList.remove('open');
            const chev = document.getElementById(el.id.replace('-body-', '-chev-').replace('-wrap', '-chev').replace('-body', '-chev'));
            if (chev) chev.classList.remove('open');
        });
    }
    if (e.key === 'x' || e.key === 'X') {
        document.querySelectorAll('.collapsible-body:not(.open)').forEach(el => {
            el.classList.add('open');
            const chev = document.getElementById(el.id.replace('-body-', '-chev-').replace('-wrap', '-chev').replace('-body', '-chev'));
            if (chev) chev.classList.add('open');
        });
    }
});

async function loadFast() {
    const [sum, snap, vms, net, ports, svcs, vmIngress, oblResp, downloads, alloc] = await Promise.all([
        get('/api/dashboard/summary'),
        get('/api/node/snapshot'),
        get('/api/vms'),
        get('/api/dashboard/network'),
        get('/api/dashboard/ports'),
        get('/api/dashboard/services'),
        get('/api/dashboard/vm-ingress'),
        get('/api/dashboard/obligations'),
        get('/api/downloads'),
        get('/api/node/allocation'),
    ]);
    S.summary = sum;
    S.snapshot = snap;
    S.vms = vms;
    S.downloads = downloads ?? [];
    S.network = net;
    S.ports = ports;
    S.services = svcs;
    S.vmIngress = vmIngress ?? {};
    S.obligations = oblResp?.obligations ?? [];
    S.allocation = alloc;
    renderAll();
    $('last-refresh').textContent = 'Updated ' + new Date().toLocaleTimeString();
    if (S.logsVisible) loadLogs();
}

async function loadFirewall() {
    S.firewall = await get('/api/dashboard/firewall');
    renderFirewall();
}

async function loadLogs() {
    S.logs = await get('/api/dashboard/logs?lines=150');
    renderLogs();
}

async function get(url) {
    try {
        const r = await fetch(url);
        if (!r.ok) return null;
        return r.json();
    } catch { return null; }
}

function $(id) { return document.getElementById(id); }
function set(id, v) { const el = $(id); if (el) el.textContent = v ?? '—'; }
function esc(s) { return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

// ============================================================
// Render All
// ============================================================
async function renderAll() {
    renderBanner();
    renderQuickStats();
    renderHardware();
    renderVMs();
    renderNetwork();
    renderServices();
    await fetchAndRenderDatabase();
}

// ============================================================
// Banner
// ============================================================
function renderBanner() {
    const d = S.summary;
    if (!d) return;
    // Cache ingress base domain for use in VM link construction
    if (d.ingressBaseDomain) S.ingressBaseDomain = d.ingressBaseDomain;
    set('node-id', d.nodeId ?? 'unregistered');
    set('hostname', d.hostname);
    set('os-info', d.os);
    set('uptime', fmtUptime(d.uptimeSeconds ?? 0));
    set('agent-version', d.agentVersion);
    set('wallet', d.walletAddress ? trunc(d.walletAddress, 10) : '—');

    const orch = d.orchestrator ?? {};
    const el = $('orch-status');
    const val = $('orch-val');
    if (orch.connected) {
        el.className = 'orch-status connected';
        const ago = orch.secondsAgo != null ? ` · ${fmtAgo(orch.secondsAgo)}` : '';
        val.textContent = 'Connected' + ago;
    } else {
        el.className = 'orch-status disconnected';
        val.textContent = 'Disconnected';
    }

    // Health pill
    const issues = collectIssues();
    const pill = $('health-pill');
    const htxt = $('health-text');
    if (issues.length === 0) {
        pill.className = 'health-pill healthy';
        htxt.textContent = 'Healthy';
    } else if (issues.length <= 2) {
        pill.className = 'health-pill degraded';
        htxt.textContent = `Degraded (${issues.length})`;
    } else {
        pill.className = 'health-pill critical';
        htxt.textContent = `Critical (${issues.length})`;
    }
    pill.title = issues.join('\n') || 'All systems operational';

    // Populate status strip (reuse orch already declared above)
    const orchDot = $('strip-orch-dot');
    if (orchDot) orchDot.className = 'strip-dot ' + (orch.connected ? 'ok' : 'err');
    set('strip-orch', orch.connected ? 'Connected' : 'Disconnected');
    set('strip-uptime', fmtUptime(d.uptimeSeconds ?? 0));
    set('strip-agent', d.agentVersion ?? '—');
}

function collectIssues() {
    const issues = [];
    if (S.summary && !S.summary.orchestrator?.connected) issues.push('Orchestrator disconnected');
    for (const wg of S.network?.wireguard ?? []) {
        const dead = wg.peers?.filter(p => p.handshakeStatus === 'dead') ?? [];
        if (dead.length) issues.push(`${wg.name}: ${dead.length} dead WG peer(s)`);
    }
    for (const svc of S.services?.services ?? []) {
        if (['libvirtd', 'decloud-node-agent'].includes(svc.name) && !svc.isActive)
            issues.push(`Service down: ${svc.name}`);
    }
    return issues;
}

// ============================================================
// Quick Stats
// ============================================================
function renderQuickStats() {
    const snap = S.snapshot;
    if (snap) {
        const cpu = snap.virtualCpuUsagePercent ?? 0;
        const ramP = snap.totalMemoryBytes > 0 ? snap.usedMemoryBytes / snap.totalMemoryBytes * 100 : 0;
        const stP = snap.totalStorageBytes > 0 ? snap.usedStorageBytes / snap.totalStorageBytes * 100 : 0;
        set('qs-cpu', cpu.toFixed(1) + '%'); setBar('qs-cpu-bar', cpu);
        set('qs-ram', ramP.toFixed(1) + '%'); setBar('qs-ram-bar', ramP);
        set('qs-storage', stP.toFixed(1) + '%'); setBar('qs-storage-bar', stP);
    }
    if (S.vms) {
        const all = S.vms;
        const run = all.filter(v => isRunning(v)).length;
        set('qs-vms', `${run}/${all.length}`);
    }
    if (S.network) {
        const wgs = S.network.wireguard ?? [];
        const peers = wgs.reduce((s, w) => s + (w.peers?.length ?? 0), 0);
        set('qs-wg', wgs.length);
        set('qs-peers', peers);
    }
    if (S.ports) {
        set('qs-ports', (S.ports.tcp?.length ?? 0) + (S.ports.udp?.length ?? 0));
    }

    // Keep status strip in sync
    if (S.snapshot) {
        const cpu = S.snapshot.virtualCpuUsagePercent ?? 0;
        const ramP = S.snapshot.totalMemoryBytes > 0
            ? S.snapshot.usedMemoryBytes / S.snapshot.totalMemoryBytes * 100 : 0;
        set('strip-cpu', cpu.toFixed(1) + '%');
        set('strip-ram', ramP.toFixed(1) + '%');
    }
    if (S.vms) {
        const run = S.vms.filter(v => isRunning(v)).length;
        set('strip-vms', `${run}/${S.vms.length}`);
    }
    if (S.network) {
        const peers = (S.network.wireguard ?? []).reduce((s, w) => s + (w.peers?.length ?? 0), 0);
        set('strip-peers', peers);
    }
}

function setBar(id, pct) {
    const el = $(id); if (!el) return;
    el.style.width = Math.min(pct, 100).toFixed(1) + '%';
    el.className = 'stat-bar-fill' + (pct >= 90 ? ' crit' : pct >= 75 ? ' warn' : '');
}

function setHwBar(id, pct) {
    const el = $(id); if (!el) return;
    el.style.width = Math.min(pct, 100).toFixed(1) + '%';
    el.className = 'hw-fill' + (pct >= 90 ? ' crit' : pct >= 75 ? ' warn' : '');
}

// ============================================================
// Hardware
// ============================================================
// ── Metric history ring buffers (last 30 samples = 5 min at 10s poll) ────────
const HW_HISTORY = { cpu: [], ram: [], stor: [] };
const HW_MAX_PTS = 30;

/**
 * Build an inline SVG sparkline / area chart from an array of 0-100 values.
 * w × h are the viewBox dimensions; actual display size is CSS-controlled.
 * color  — stroke + fill colour (CSS variable string or hex)
 * Returns an SVG string ready for innerHTML.
 */
function sparklineSvg(data, color) {
    if (data.length < 2) return '';

    const W = 200, H = 36, PAD = 2;
    const plotH = H - PAD * 2;
    const pts = data.length;
    const xStep = (W - PAD * 2) / (HW_MAX_PTS - 1);

    // Map value → y coordinate (inverted: 0% at bottom, 100% at top)
    const y = v => PAD + plotH * (1 - Math.min(v, 100) / 100);
    // Map index → x (right-align: newest point is at the right edge)
    const x = i => PAD + (HW_MAX_PTS - pts + i) * xStep;

    // Build polyline points
    const linePts = data.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');

    // Area: close to the bottom
    const firstX = x(0).toFixed(1);
    const lastX = x(data.length - 1).toFixed(1);
    const bottom = (PAD + plotH).toFixed(1);
    const areaD = `M${firstX},${bottom} L${linePts.split(' ').map(p => `${p.split(',')[0]},${p.split(',')[1]}`).join(' L')} L${lastX},${bottom} Z`;

    // Color based on latest value
    const latest = data[data.length - 1];
    const c = latest >= 90 ? 'var(--danger)'
        : latest >= 75 ? 'var(--warning)'
            : color;

    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <path class="hw-spark-area" d="${areaD}" fill="${c}"/>
        <polyline class="hw-spark-line" points="${linePts}" stroke="${c}"/>
      </svg>`;
}

function pushHistory(key, value) {
    HW_HISTORY[key].push(Math.round(value * 10) / 10);
    if (HW_HISTORY[key].length > HW_MAX_PTS) HW_HISTORY[key].shift();
}

/**
 * Render one allocation card in the Resource Allocation section.
 * Shows the operator-configured slice (allocBytes) against the physical total
 * (physBytes), with vm-spec-based usage (vmUsedBytes) as the bar fill.
 * Falls back to platform default (90%) when allocBytes is absent, which
 * happens when the /api/node/allocation endpoint is not yet deployed and
 * the snapshot field is also missing.
 */
function renderAllocCard(allocBytes, physBytes, vmUsedBytes, pctId, detailId, barId) {
    if (allocBytes > 0 && physBytes > 0) {
        const allocPct = allocBytes / physBytes * 100;
        const usedOfAlloc = vmUsedBytes / allocBytes * 100;
        set(pctId, `${allocPct.toFixed(0)}% of physical`);
        set(detailId, `${fmtBytes(vmUsedBytes)} used / ${fmtBytes(allocBytes)} allocated / ${fmtBytes(physBytes)} physical`);
        setHwBar(barId, usedOfAlloc);
    } else if (physBytes > 0) {
        const defaultAlloc = Math.floor(physBytes * 0.9);
        const usedOfDefault = defaultAlloc > 0 ? vmUsedBytes / defaultAlloc * 100 : 0;
        set(pctId, 'Default (90%)');
        set(detailId, `${fmtBytes(vmUsedBytes)} used / ${fmtBytes(defaultAlloc)} allocated / ${fmtBytes(physBytes)} physical`);
        setHwBar(barId, usedOfDefault);
    } else {
        set(pctId, '—');
        set(detailId, '—');
        setHwBar(barId, 0);
    }
}

function renderHardware() {
    const snap = S.snapshot; if (!snap) return;

    // ── Physical hardware stats (measure against physical totals) ────
    const cpu = snap.virtualCpuUsagePercent ?? 0;
    const physMem = snap.totalMemoryBytes ?? 0;
    const ramP = physMem > 0 ? (snap.usedMemoryBytes ?? 0) / physMem * 100 : 0;
    const stP = snap.totalStorageBytes > 0 ? snap.usedStorageBytes / snap.totalStorageBytes * 100 : 0;

    set('hw-cpu-pct', cpu.toFixed(1) + '%');
    set('hw-cpu-info', snap.cpuCores ? `${snap.cpuCores} cores` : '—');
    setHwBar('hw-cpu-bar', cpu);

    set('hw-ram-pct', ramP.toFixed(1) + '%');
    set('hw-ram-detail', `${fmtBytes(snap.usedMemoryBytes ?? 0)} / ${fmtBytes(physMem)}`);
    setHwBar('hw-ram-bar', ramP);

    set('hw-stor-pct', stP.toFixed(1) + '%');
    set('hw-stor-detail', `${fmtBytes(snap.usedStorageBytes ?? 0)} / ${fmtBytes(snap.totalStorageBytes ?? 0)}`);
    setHwBar('hw-stor-bar', stP);

    const kvm = snap.kvmAvailable !== false;
    const kvmEl = $('hw-kvm');
    if (kvmEl) { kvmEl.textContent = kvm ? 'KVM Available' : 'KVM Unavailable'; kvmEl.style.color = kvm ? 'var(--primary)' : 'var(--danger)'; }
    set('hw-virt-sub', snap.cpuArchitecture ?? '—');

    // Update ring buffers and render sparklines
    pushHistory('cpu', cpu);
    pushHistory('ram', ramP);
    pushHistory('stor', stP);

    const sparkEl = (id, key, color) => {
        const el = $(id); if (!el) return;
        el.innerHTML = sparklineSvg(HW_HISTORY[key], color);
    };
    sparkEl('hw-cpu-spark', 'cpu', 'var(--primary)');
    sparkEl('hw-ram-spark', 'ram', 'var(--primary)');
    sparkEl('hw-stor-spark', 'stor', 'var(--primary)');

    // ── Resource Allocation section ──────────────────────────────────
    // Compute VM-spec-based usage from S.vms (same data source as heartbeat).
    // This matches what the orchestrator sees in UsedResources.
    const allVms = (S.vms ?? []).filter(v =>
        v.state !== 9 &&   // Deleted
        v.state !== 7 &&   // Stopped (resources not in use)
        v.state !== 8);    // Deleting
    const vmUsedPts = allVms.reduce((s, v) =>
        s + ((v.spec?.computePointCost ?? v.computePointCost) || 0), 0);
    const vmUsedMem = allVms.reduce((s, v) =>
        s + ((v.spec?.memoryBytes ?? v.memoryBytes) || 0), 0);
    const vmUsedStor = allVms.reduce((s, v) =>
        s + ((v.spec?.diskBytes ?? v.diskBytes) || 0), 0);

    // Prefer orchestrator-confirmed values from S.allocation (/api/node/allocation);
    // fall back to snapshot fields for nodes running older agent versions.
    const totalPts = S.allocation?.resolvedComputePoints ?? snap.totalComputePoints ?? 0;
    const allocMem = S.allocation?.resolvedMemoryBytes ?? snap.allocatedMemoryBytes ?? 0;
    const allocStor = S.allocation?.resolvedStorageBytes ?? snap.allocatedStorageBytes ?? 0;
    // physMem already declared above; derive physStor from snap directly
    const physStor = snap.totalStorageBytes ?? 0;

    // Compute points: vm-used / allocated total with bar
    if (totalPts > 0) {
        const cpuAllocPct = vmUsedPts / totalPts * 100;
        set('hw-alloc-cpu', `${vmUsedPts} / ${totalPts} pts`);
        set('hw-alloc-cpu-detail', `${totalPts - vmUsedPts} available`);
        setHwBar('hw-alloc-cpu-bar', cpuAllocPct);
    }

    // Memory and Storage: operator-configured slice vs physical total
    renderAllocCard(allocMem, physMem, vmUsedMem, 'hw-alloc-mem-pct', 'hw-alloc-mem-detail', 'hw-alloc-mem-bar');
    renderAllocCard(allocStor, physStor, vmUsedStor, 'hw-alloc-stor', 'hw-alloc-stor-detail', 'hw-alloc-stor-bar');

    // GPUs — GpuMode: None=0, Passthrough=1, Proxied=2.
    // Count by mode, not by gpuPciAddress presence — Proxied VMs have no PCI address.
    const gpus = snap.totalGpus ?? 0;
    const passthroughVms = allVms.filter(v => (v.spec?.gpuMode ?? v.gpuMode) === 1).length;
    const proxiedVms = allVms.filter(v => (v.spec?.gpuMode ?? v.gpuMode) === 2).length;
    const isProxyNode = snap.supportsGpuProxy === true;

    const gpuModeBadge = $('hw-alloc-gpu-mode-badge');

    if (gpus > 0) {
        // Mode badge — green for Proxied, blue for Passthrough
        if (gpuModeBadge) {
            gpuModeBadge.textContent = isProxyNode ? 'Proxied' : 'Passthrough';
            gpuModeBadge.style.cssText = isProxyNode
                ? 'font-size:10px;font-weight:500;padding:1px 6px;border-radius:20px;margin-left:4px;background:#E1F5EE;color:#0F6E56'
                : 'font-size:10px;font-weight:500;padding:1px 6px;border-radius:20px;margin-left:4px;background:#E6F1FB;color:#185FA5';
        }

        if (isProxyNode) {
            // Proxied: VRAM allocation bar — same renderAllocCard pattern as memory/storage.
            // allocBytes  = operator ceiling (resolvedGpuVramBytes from allocate response).
            // physBytes   = total physical GPU VRAM from snapshot.
            // vmUsedBytes = committed VRAM (passthrough full-GPU + proxied quotas) from snapshot.
            const resolvedGpuVram = S.allocation?.resolvedGpuVramBytes ?? snap.totalGpuVramBytes ?? 0;
            const physGpuVram = snap.totalGpuVramBytes ?? 0;
            const usedGpuVram = snap.allocatedGpuVramBytes ?? 0;
            renderAllocCard(resolvedGpuVram, physGpuVram, usedGpuVram,
                'hw-alloc-gpu-vram-pct', 'hw-alloc-gpu-vram-detail', 'hw-alloc-gpu-vram-bar');
        } else {
            // Passthrough: VRAM sharing is not applicable — bar represents GPU slot usage.
            const usedPct = gpus > 0 ? passthroughVms / gpus * 100 : 0;
            set('hw-alloc-gpu-vram-pct', `${gpus} GPU${gpus !== 1 ? 's' : ''}`);
            set('hw-alloc-gpu-vram-detail', `${passthroughVms} assigned · ${gpus - passthroughVms} free`);
            setHwBar('hw-alloc-gpu-vram-bar', usedPct);
        }

        // Secondary line — VM-level GPU breakdown (shown for both modes)
        const parts = [];
        if (passthroughVms > 0) parts.push(`${passthroughVms} passthrough`);
        if (proxiedVms > 0) parts.push(`${proxiedVms} proxied`);
        set('hw-alloc-gpu-count',
            parts.length > 0
                ? `${parts.join(', ')} · ${gpus - passthroughVms} free for passthrough`
                : `${gpus} GPU${gpus !== 1 ? 's' : ''} available`);
    } else {
        if (gpuModeBadge) { gpuModeBadge.textContent = ''; gpuModeBadge.style.cssText = ''; }
        set('hw-alloc-gpu-vram-pct', 'None');
        set('hw-alloc-gpu-vram-detail', '—');
        set('hw-alloc-gpu-count', '—');
        setHwBar('hw-alloc-gpu-vram-bar', 0);
    }
}

// ============================================================
// VMs
// ============================================================

/**
 * Find the system VM that belongs to a given obligation.
 *
 * Priority:
 *   1. Exact match on obligation.vmId — always picks the CURRENT VM,
 *      even when a same-role VM from a previous deployment is still
 *      present in the list (e.g. Deleting state with stale Ready services).
 *   2. Role-type fallback — used when obligations haven't loaded yet
 *      (orchestrator unreachable) so the card still renders with whatever
 *      VM is running.
 *
 * This is the root fix for Bug 1: without vmId-first lookup the old VM's
 * services remain visible during a redeploy because role-type matching
 * returns the first VM of that type regardless of which deployment cycle
 * it belongs to.
 */
function findSysVm(all, roleFn, obl) {
    if (obl?.vmId) {
        const byId = all.find(v => v.vmId === obl.vmId);
        if (byId) return byId;
    }
    return all.find(roleFn) ?? null;
}

function renderVMs() {
    if (!S.vms) return;
    const all = S.vms;

    // v.role — top-level VmInstance.Role serialised as a string by [JsonStringEnumConverter].
    // v.spec.labels — labels dict lives on Spec, not at the top level of VmInstance.
    // v.category — "System" / "Tenant"; used for tenant filter so future system roles
    //              don't leak into the tenant table without a dashboard change.
    const isRelay = v => v.spec?.role === 'Relay' || v.spec?.labels?.['system-vm-role'] === 'relay';
    const isDht = v => v.spec?.role === 'Dht' || v.spec?.labels?.['system-vm-role'] === 'dht';
    const isBs = v => v.spec?.role === 'BlockStore' || v.spec?.labels?.['system-vm-role'] === 'blockstore';

    // Resolve each obligation once so findSysVm can use the vmId.
    const relayObl = S.obligations.find(o => o.role === ROLE_FOR_KEY.relay);
    const dhtObl = S.obligations.find(o => o.role === ROLE_FOR_KEY.dht);
    const bsObl = S.obligations.find(o => o.role === ROLE_FOR_KEY.bs);

    const relay = findSysVm(all, isRelay, relayObl);
    const dht = findSysVm(all, isDht, dhtObl);
    const bs = findSysVm(all, isBs, bsObl);
    const tenants = all.filter(v => v.spec?.category !== 'System');

    $('vm-count-badge').textContent = all.length;
    // Update tenant collapsible meta
    const tenantMeta = $('tenant-count-meta');
    if (tenantMeta) tenantMeta.textContent = tenants.length ? `${tenants.length} VM${tenants.length === 1 ? '' : 's'}` : 'empty';

    paintSysVm('relay', relay);
    paintSysVm('dht', dht);
    paintSysVm('bs', bs);

    // Dim section badge when all roles are ineligible
    if (S.obligations.length > 0) {
        const eligibleCount = [1, 0, 2].filter(role => S.obligations.some(o => o.role === role)).length;
        const secBadge = $('vm-count-badge');
        if (secBadge) secBadge.style.opacity = eligibleCount === 0 ? '0.4' : '1';
    }

    // Tenant table
    const tbody = $('tenant-tbody');
    if (!tenants.length) {
        tbody.innerHTML = '<tr><td colspan="10" class="empty">No tenant VMs</td></tr>';
        return;
    }
    autoExpand('tenant-table-body', 'tenant-chev');
    tbody.innerHTML = tenants.map(vm => {
        const spec = vm.spec ?? {};
        const uptime = vm.startedAt ? fmtUptime(Math.floor((Date.now() - new Date(vm.startedAt).getTime()) / 1000)) : '—';
        const resolved = isRunning(vm) ? resolveVmUrl(vm) : null;
        const urlCell = resolved
            ? `<a href="${resolved.url}" target="_blank"
               style="color:${resolved.source === 'direct' ? 'var(--text-muted)' : 'var(--primary)'};font-size:0.76rem"
               title="${resolved.source === 'ingress' ? 'Ingress (DB)' : resolved.source === 'formula' ? 'Ingress (formula)' : 'Direct IP'}"
             >${spec.ipAddress || '—'} ↗</a>`
            : `<span>${spec.ipAddress || '—'}</span>`;
        return `<tr>
          <td class="vm-name-cell">${esc(vm.name ?? spec.name ?? '—')}</td>
          <td class="vm-id-cell">${truncId(vm.vmId ?? '—')}</td>
          <td>${stateBadge(vm.status, vm.vmId ?? vm.spec?.id)}</td>
          <td class="health-cell">${renderHealthCell(vm)}</td>
          <td>${spec.virtualCpuCores ?? '—'}</td>
          <td>${spec.memoryBytes ? fmtBytes(spec.memoryBytes) : '—'}</td>
          <td class="vm-ip-cell">${urlCell}</td>
          <td>${vm.vncPort ?? '—'}</td>
          <td>${uptime}</td>
          <td class="vm-id-cell">${truncId(spec.ownerId ?? '—')}</td>
        </tr>`;
    }).join('');
}

// ── fetch & render database section ──────────────────────────────
async function fetchAndRenderDatabase() {
    try {
        const r = await fetch('/api/dashboard/database');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();

        // Meta row
        set('db-path', d.databasePath ?? '—');
        set('db-size', d.sizeBytes ? fmtBytes(d.sizeBytes) : '—');
        set('db-schema-ver', `v${d.schemaVersion ?? '?'}`);
        set('db-schema-badge', `v${d.schemaVersion ?? '?'}`);
        set('db-total-vms', d.stats?.totalVms ?? '—');

        // VmRecords table
        const vmRows = d.vmRecords ?? [];
        set('db-vmrecords-meta', `${vmRows.length} record${vmRows.length !== 1 ? 's' : ''}`);
        const vmTbody = document.getElementById('db-vmrecords-tbody');
        if (vmTbody) {
            vmTbody.innerHTML = vmRows.length === 0
                ? '<tr><td colspan="12" class="empty">No records</td></tr>'
                : vmRows.map(v => `<tr>
                    <td>${esc(v.name)}</td>
                    <td class="mono" title="${esc(v.vmId)}">${esc(v.vmId.slice(0, 8))}…</td>
                    <td>${esc(v.vmRole)}</td>
                    <td>${stateBadgeStr(v.state)}</td>
                    <td class="mono">${v.ipAddress ?? '—'}</td>
                    <td>${v.virtualCpuCores}</td>
                    <td>${v.memoryGb}</td>
                    <td>${v.diskGb}</td>
                    <td>${v.replicationFactor}</td>
                    <td>${v.vncPort ?? '—'}</td>
                    <td class="mono" title="${esc(v.ownerId ?? '')}">${v.ownerId ? v.ownerId.slice(0, 8) + '…' : '—'}</td>
                    <td>${fmtRelativeTime(v.lastUpdated)}</td>
                    <td title="${esc(v.deletionReason ?? '')}" style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:0.72rem;color:var(--text-muted)">${v.deletionReason ? esc(v.deletionReason) : '—'}</td>
                </tr>`).join('');
        }

        // PortMappings table
        const portRows = d.portMappings ?? [];
        set('db-ports-meta', `${portRows.length} mapping${portRows.length !== 1 ? 's' : ''}`);
        const portTbody = document.getElementById('db-ports-tbody');
        if (portTbody) {
            portTbody.innerHTML = portRows.length === 0
                ? '<tr><td colspan="8" class="empty">No port mappings</td></tr>'
                : portRows.map(p => `<tr>
                    <td class="mono" title="${esc(p.vmId)}">${esc(p.vmId.slice(0, 8))}…</td>
                    <td class="mono">${esc(p.vmPrivateIp)}</td>
                    <td>${p.vmPort === 0 ? '<em>relay</em>' : p.vmPort}</td>
                    <td class="mono"><strong>${p.publicPort}</strong></td>
                    <td>${esc(p.protocol)}</td>
                    <td>${p.label ? esc(p.label) : '—'}</td>
                    <td>${p.isActive ? '✅' : '❌'}</td>
                    <td>${fmtRelativeTime(p.createdAt)}</td>
                </tr>`).join('');
        }

    } catch (err) {
        console.warn('Database section fetch failed:', err);
    }
}

function stateBadgeStr(s) {
    const cls = s === 'Running' ? 'state-running'
        : s === 'Failed' || s === 'NotFound' ? 'state-error'
            : s === 'Stopped' || s === 'Deleted' ? 'state-stopped'
                : 'state-other';
    return `<span class="state-badge ${cls}">${esc(s)}</span>`;
}

// Helper: relative time from ISO string
function fmtRelativeTime(iso) {
    if (!iso) return '—';
    try {
        const diffSec = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
        if (diffSec < 60) return `${diffSec}s ago`;
        if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
        if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
        return `${Math.floor(diffSec / 86400)}d ago`;
    } catch { return iso; }
}

// Maps dashboard card key → SystemVmRole integer
const ROLE_FOR_KEY = { relay: 1, dht: 0, bs: 2 };

function paintSysVm(key, vm) {
    const card = $(`svc-${key}`);
    const badge = $(`svc-${key}-badge`);
    const detail = $(`svc-${key}-detail`);
    if (!card) return;

    const roleInt = ROLE_FOR_KEY[key] ?? -1;
    const obl = S.obligations.find(o => o.role === roleInt);
    const obligationsLoaded = S.obligations.length > 0;

    // ── No obligation ────────────────────────────────────────────────────────
    if (!obl) {
        if (!obligationsLoaded) {
            // Obligations not fetched yet (orchestrator unreachable / patch not applied).
            // Fall through to VM-presence-based rendering below — don't grey out.
        } else {
            // Obligations DID load but this role is absent → genuinely ineligible.
            card.className = 'sysvm-card ineligible';
            badge.className = 'sysvm-badge other';
            badge.innerHTML = '<span class="obl-badge ineligible">Not eligible</span>';
            detail.textContent = 'Not assigned to this node.';
            const chips = card.querySelector('.svc-chips');
            if (chips) chips.innerHTML = '';
            return;
        }
    }

    // ── Obligation present (or not loaded) — determine display state ──────────
    const oblStatus = obl?.status ?? (vm && isRunning(vm) ? 2 : -1);
    const oblName = obl?.statusName ?? (vm && isRunning(vm) ? 'Active' : 'Unknown');
    const running = vm ? isRunning(vm) : false;

    // Obligation Active but VM genuinely absent or stopped = unmet.
    //
    // Bug 2 fix: suppress the "UNMET" false-positive that fires during a
    // redeploy transition.  The 30-second obligation cache on the node agent
    // proxy can return a stale Active (status=2) while the new VM is still
    // Provisioning / Creating / Starting (state ≠ 3 → !running).  Without
    // this guard the card briefly shows UNMET even though the deployment is
    // progressing normally.
    //
    // We suppress UNMET when the matched VM is in a clearly transitional
    // state — the VM exists but hasn't reached Running yet, which is
    // expected during a fresh deployment.  Genuinely unmet obligations
    // (VM crashed, was force-deleted, etc.) will still be caught because
    // those VMs are either absent or stuck in Stopped/Failed.
    //
    // VmState: 0=Pending,1=Creating,2=Starting,3=Running,4=Paused,
    //          5=Stopping,6=Stopped,7=Failed,8=NotFound,9=Deleted,10=Migrating
    const TRANSITIONAL_VM_STATES = new Set([0, 1, 2, 10]); // Pending/Creating/Starting/Migrating
    const vmIsTransitioning = vm != null && TRANSITIONAL_VM_STATES.has(vm.status);
    const isUnmet = oblStatus === 2 && !running && !vmIsTransitioning;

    // Card border class
    let cardClass = 'sysvm-card';
    if (isUnmet) cardClass += ' unmet';
    else if (running) cardClass += ' running';
    else cardClass += ' stopped';
    card.className = cardClass;

    // State badge: shows obligation status when VM not deployed, VM state when running
    badge.className = 'sysvm-badge other';
    let badgeHtml;
    // Check for active download on this obligation's VM
    const dl = obl?.vmId ? (S.downloads ?? []).find(d => d.vmId === obl.vmId) : null;

    if (running) {
        badgeHtml = `<span class="obl-badge active">Running</span>`;
    } else if (isUnmet) {
        badgeHtml = `<span class="obl-badge unmet" title="Obligation Active but VM not running">Unmet</span>`;
    } else if (dl && dl.totalBytes > 0) {
        const mb = (dl.downloadedBytes / 1024 / 1024).toFixed(0);
        const totalMb = (dl.totalBytes / 1024 / 1024).toFixed(0);
        badgeHtml = `<span class="obl-badge deploying">Downloading ${dl.percentComplete}%</span>`
            + `<div class="dl-progress-bar"><div class="dl-progress-fill" style="width:${dl.percentComplete}%"></div></div>`
            + `<span class="dl-detail">${mb} / ${totalMb} MB</span>`;
    } else if (oblStatus === 1) {
        badgeHtml = `<span class="obl-badge deploying">Deploying</span>`;
    } else if (oblStatus === 0) {
        badgeHtml = `<span class="obl-badge pending">Pending</span>`;
    } else if (oblStatus === 3 || (obl?.failureCount ?? 0) > 0) {
        const errTip = obl?.lastError ? esc(obl.lastError).slice(0, 120) : '';
        badgeHtml = `<span class="obl-badge failed" title="${errTip}">Failed (${obl?.failureCount ?? 0}x)</span>`;
    } else {
        badgeHtml = `<span class="obl-badge ${oblName.toLowerCase()}">${oblName}</span>`;
    }
    badge.innerHTML = badgeHtml;

    // ── Detail row: IP + open link ────────────────────────────────────────────
    const labels = vm?.spec?.labels ?? vm?.labels ?? {};
    const ip = vm?.spec?.ipAddress || vm?.ipAddress
        || labels['dht-advertise-ip']
        || labels['blockstore-advertise-ip'];

    // System VM dashboards are always accessed through the node agent proxy.
    // resolveVmUrl() uses ingress subdomains which system VMs don't have.
    const ROLE_SLUG = { 0: 'dht', 1: 'relay', 2: 'blockstore' };
    const roleSlug = obl ? ROLE_SLUG[obl.role] : null;

    let detailHtml = '';
    if (ip) {
        let dashLink = '';
        if (running) {
            const resolved = roleSlug
                ? { url: `/api/system-vms/${roleSlug}/proxy/`, source: 'proxy' }
                : resolveVmUrl(vm);
            if (resolved) {
                const title = resolved.source === 'proxy' ? `Via node agent proxy — ${resolved.url}` :
                    resolved.source === 'direct' ? 'Direct IP — may not be reachable outside this host' : resolved.url;
                const color = resolved.source === 'direct' ? 'var(--text-muted)' : 'var(--primary)';
                const label = resolved.source === 'direct' ? 'Open (direct) →' : 'Open →';
                dashLink = ` &nbsp;<a href="${resolved.url}" target="_blank" style="font-size:0.7rem;color:${color}" title="${title}">${label}</a>`;
            }
        }
        detailHtml = `<span style="font-family:var(--mono);font-size:0.78rem">${esc(ip)}</span>${dashLink}`;
    } else if (obl?.vmId) {
        const dlInfo = dl && dl.totalBytes > 0
            ? `downloading base image (${dl.percentComplete}%)`
            : 'booting';
        detailHtml = `<span style="font-size:0.73rem;color:var(--text-muted)">VM ${truncId(obl.vmId)} — ${dlInfo}</span>`;
    } else if (oblStatus === 0) {
        detailHtml = `<span style="font-size:0.73rem;color:var(--text-muted)">Waiting for scheduler…</span>`;
    } else {
        detailHtml = `<span style="font-size:0.73rem;color:var(--text-muted)">—</span>`;
    }

    // Add obligation error note if any failures
    if (obl?.failureCount > 0 && obl.lastError) {
        detailHtml += `<div style="font-size:0.68rem;color:var(--danger);margin-top:0.2rem;font-family:var(--mono)">${esc(obl.lastError).slice(0, 80)}</div>`;
    }

    // Binary version row — shows the running binary hash, with a pending-update
    // indicator (warning colour) when currentBinaryVersion differs.
    if (obl?.runningBinaryVersion || obl?.currentBinaryVersion) {
        const runVer = obl.runningBinaryVersion ? obl.runningBinaryVersion.slice(0, 8) : '—';
        const curVer = obl.currentBinaryVersion ? obl.currentBinaryVersion.slice(0, 8) : '—';
        const updatePending = obl.runningBinaryVersion && obl.currentBinaryVersion
            && obl.runningBinaryVersion !== obl.currentBinaryVersion;
        if (updatePending) {
            detailHtml += `<div style="font-size:0.68rem;color:var(--warning);margin-top:0.25rem;font-family:var(--mono)">`
                + `bin: ${runVer} → ${curVer} <span style="opacity:0.75">↑ update pending</span></div>`;
        } else {
            detailHtml += `<div style="font-size:0.68rem;color:var(--text-muted);margin-top:0.25rem;font-family:var(--mono)">`
                + `bin: ${runVer}</div>`;
        }
    }

    // State version row — only shown once the orchestrator has generated identity state.
    if (obl?.stateVersion > 0) {
        detailHtml += `<div style="font-size:0.65rem;color:var(--text-muted);margin-top:0.1rem;font-family:var(--mono)">`
            + `state v${obl.stateVersion}</div>`;
    }

    // State data key-value panel — sanitised public fields only (private keys
    // are stripped server-side). 'version' / 'updatedAt' are suppressed here.
    if (obl?.stateData && typeof obl.stateData === 'object') {
        const SUPPRESS = new Set(['version', 'updatedAt']);
        const rows = Object.entries(obl.stateData)
            .filter(([k]) => !SUPPRESS.has(k))
            .map(([k, v]) => {
                const display = (k === 'storageQuotaBytes' && typeof v === 'number')
                    ? fmtBytes(v)
                    : String(v ?? '—');
                const truncated = display.length > 44 ? display.slice(0, 41) + '…' : display;
                return `<div style="display:flex;gap:0.5rem;font-size:0.65rem;`
                    + `font-family:var(--mono);line-height:1.6;min-width:0">`
                    + `<span style="color:var(--text-muted);min-width:6rem;`
                    + `flex-shrink:0;overflow:hidden;text-overflow:ellipsis">${esc(k)}</span>`
                    + `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap"`
                    + ` title="${esc(display)}">${esc(truncated)}</span>`
                    + `</div>`;
            }).join('');

        if (rows) {
            detailHtml += `<div style="margin-top:0.4rem;padding:0.3rem 0.45rem;`
                + `background:rgba(255,255,255,0.04);border-radius:4px;`
                + `border:1px solid rgba(255,255,255,0.06)">${rows}</div>`;
        }
    }

    detail.innerHTML = detailHtml;

    // ── Service health chips ──────────────────────────────────────────────────
    //
    // Bug 1 fix: service chips must only reflect the CURRENT VM's services.
    // Two guards are applied:
    //
    //   1. vmMatchesObligation — if the obligation has a vmId, the resolved
    //      VM must be the same one.  Without this, role-type lookup could
    //      return a stale VM from a previous deployment (still in the DB,
    //      Deleting state, services=Ready) and display its chips alongside
    //      a Deploying obligation badge.
    //
    //   2. oblIsActive — service chips are only meaningful when the
    //      obligation is Active (the VM has completed cloud-init).  During
    //      Deploying/Pending the current VM's services are in a transient
    //      initialisation state that is misleading to show; clear the chips
    //      so the operator sees a clean slate while the new VM boots.
    //
    const vmMatchesObligation = !obl?.vmId || (vm?.vmId === obl.vmId);
    const oblIsActive = oblStatus === 2;
    let servicesEl = card.querySelector('.svc-chips');
    if (!servicesEl) {
        servicesEl = document.createElement('div');
        servicesEl.className = 'svc-chips';
        card.appendChild(servicesEl);
    }
    const svcs = (running && vmMatchesObligation && oblIsActive) ? (vm?.services ?? []) : [];
    const readyCount = svcs.filter(s => svcReadiness(s) === 'ready').length;
    const readyAll = vm?.isFullyReady ?? false;
    const rdotFailed = svcs.some(s => ['timedout', 'failed'].includes(svcReadiness(s)));
    const rdot = readyAll ? 'ready' : readyCount > 0 ? 'partial' : 'pending';
    if (svcs.length) {
        servicesEl.innerHTML =
            `<div class="sysvm-ready-row">
            <span class="sysvm-ready-dot ${rdotFailed ? 'failed' : rdot}"></span>
            <span>${readyAll ? 'All services ready' : readyCount + '/' + svcs.length + ' ready'}</span>
          </div>` +
            svcs.map(s => svcChip(s)).join('');
    } else {
        servicesEl.innerHTML = `<div class="sysvm-ready-row">
            <span class="sysvm-ready-dot ${rdotFailed ? 'failed' : rdot}"></span>
            <span>No services found</span>
          </div>`;
    }
}

// ============================================================
// Network
// ============================================================
function renderNetwork() {
    if (!S.network) return;
    renderInterfaces();
    renderWireGuard();
    renderBridges();
    renderRoutes();
    if (S.ports) renderPorts();
}

/* -- Interfaces -- */
function renderInterfaces() {
    const el = $('iface-list');
    const ifaces = S.network.interfaces ?? [];
    const ifCt = $('tab-ct-iface');
    if (ifCt) ifCt.textContent = ifaces.length > 0 ? ifaces.length : '';
    if (!ifaces.length) { el.innerHTML = '<div class="empty-state"><div class="empty-icon">⬡</div><div>No interfaces detected</div></div>'; return; }

    el.innerHTML = ifaces.map(i => {
        const rxTx = i.rxBytes > 0 || i.txBytes > 0
            ? `<span class="rx">↓${fmtBytes(i.rxBytes)}</span> <span class="tx">↑${fmtBytes(i.txBytes)}</span>`
            : '';
        const ips = (i.ipAddresses ?? []).map(a => `<span class="iface-ip-tag">${esc(a)}</span>`).join('');
        return `
        <div class="iface-row ${i.isUp ? '' : 'down'}">
          <div>
            <div class="iface-name-cell">${esc(i.name)}</div>
            ${i.macAddress ? `<div class="iface-mac">${esc(i.macAddress)}</div>` : ''}
          </div>
          <span class="type-badge ${i.type}">${i.type}</span>
          <div class="iface-addrs">${ips || '<span style="color:var(--text-muted);font-size:0.73rem">no address</span>'}</div>
          <div class="iface-xfer">${rxTx}<br><span style="color:var(--text-muted)">MTU ${i.mtu}</span></div>
          <div class="up-dot ${i.isUp ? 'up' : 'down'}" title="${i.isUp ? 'UP' : 'DOWN'}"></div>
        </div>`;
    }).join('');
}

/* -- WireGuard -- */
function renderWireGuard() {
    const el = $('wg-panels');
    const wgs = S.network.wireguard ?? [];
    const wgCt = $('tab-ct-wg');
    const totalWgPeers = wgs.reduce((s, w) => s + (w.peers?.length ?? 0), 0);
    if (wgCt) wgCt.textContent = totalWgPeers > 0 ? totalWgPeers : '';
    if (!wgs.length) {
        el.innerHTML = '<div class="empty-state"><div class="empty-icon">⬡</div><div>No WireGuard interfaces configured</div></div>';
        return;
    }

    el.innerHTML = wgs.map((wg, wi) => {
        const bodyId = `wg-body-${wi}`;
        const chevId = `wg-chev-${wi}`;
        const peerCount = wg.peers?.length ?? 0;
        const stale = wg.peers?.filter(p => p.handshakeStatus === 'dead').length ?? 0;
        const isOpen = peerCount > 0;  // auto-open when peers exist

        const peersHtml = !peerCount
            ? '<tr><td colspan="8" class="empty">No peers configured</td></tr>'
            : wg.peers.map(p => {
                const hs = p.handshakeStatus ?? 'never';
                const hsAge = p.handshakeSecondsAgo != null ? fmtAge(p.handshakeSecondsAgo) : 'Never';
                const endpt = p.endpoint ?? '—';
                const ka = p.persistentKeepalive > 0 ? p.persistentKeepalive + 's' : '—';
                return `<tr>
                <td><div class="hs-badge"><span class="hs-dot ${hs}"></span><span class="hs-txt ${hs}">${hs.toUpperCase()}</span></div></td>
                <td><div style="display:flex;align-items:center;gap:0.4rem">
                  <span style="font-family:var(--mono);font-size:0.7rem">${trunc(p.publicKey, 16)}</span>
                  <button class="copy-btn" onclick="copyKey('${esc(p.publicKey)}',this)">copy</button>
                </div></td>
                <td style="font-family:var(--mono);font-size:0.7rem">${esc(endpt)}</td>
                <td style="font-family:var(--mono);font-size:0.7rem">${esc(p.allowedIps)}</td>
                <td><span class="hs-txt ${hs}">${hsAge}</span></td>
                <td class="xfer-cell"><span class="rx">↓${fmtBytes(p.rxBytes)}</span></td>
                <td class="xfer-cell"><span class="tx">↑${fmtBytes(p.txBytes)}</span></td>
                <td style="font-family:var(--mono);font-size:0.7rem">${ka}</td>
              </tr>`;
            }).join('');

        const staleNote = stale ? `<span style="color:var(--danger);margin-left:0.5rem">${stale} stale</span>` : '';
        const metaText = `${peerCount} peer${peerCount !== 1 ? 's' : ''}  ·  port ${wg.listenPort ?? '—'}${staleNote}`;

        return `
        <div class="collapsible-wrap" style="margin-bottom:0.75rem">
          <div class="collapsible-head" onclick="toggleCollapsible('${bodyId}','${chevId}')">
            <div class="collapsible-toggle">
              <span class="collapsible-title" style="font-size:0.8rem;font-family:var(--mono);color:var(--primary)">${esc(wg.name)}</span>
              <span class="collapsible-meta" style="margin-left:0.75rem">${metaText}</span>
            </div>
            <span class="collapsible-chevron ${isOpen ? 'open' : ''}" id="${chevId}">▼</span>
          </div>
          <div class="collapsible-body ${isOpen ? 'open' : ''}" id="${bodyId}">
            <div class="wg-meta-row">
              <span class="wg-meta-item"><span class="wg-meta-label">Public Key</span>${trunc(wg.publicKey, 20)}
                <button class="copy-btn" onclick="copyKey('${esc(wg.publicKey)}',this)">copy</button></span>
              <span class="wg-meta-item"><span class="wg-meta-label">Address</span>${esc(wg.address ?? '—')}</span>
            </div>
            <div class="wg-table-scroll">
              <table class="wg-table">
                <thead><tr><th>Status</th><th>Public Key</th><th>Endpoint</th><th>Allowed IPs</th><th>Last Handshake</th><th>RX</th><th>TX</th><th>KA</th></tr></thead>
                <tbody>${peersHtml}</tbody>
              </table>
            </div>
          </div>
        </div>`;
    }).join('');
}

/* -- Bridges -- */
function renderBridges() {
    const el = $('bridge-list');
    const bridges = S.network.bridges ?? [];
    const bCt = $('tab-ct-bridge');
    if (bCt) bCt.textContent = bridges.length > 0 ? bridges.length : '';
    if (!bridges.length) { el.innerHTML = '<div class="empty-state"><div class="empty-icon">⬡</div><div>No bridge interfaces</div></div>'; return; }

    el.innerHTML = bridges.map(b => {
        const ipStr = b.ipAddresses?.join(', ') || 'no address';
        const portsHtml = !b.ports?.length
            ? '<div class="bridge-empty">No ports attached</div>'
            : b.ports.map(p => {
                const vmPart = p.vmId
                    ? `<span class="bp-vmname">${esc(p.vmName ?? p.vmId)}</span><span class="bp-vmid">${truncId(p.vmId ?? '—')}</span>`
                    : `<span class="bp-none">host / unassigned</span>`;
                return `<div class="bridge-port-item">
                <span class="bp-iface">${esc(p.interface)}</span>
                <span class="bp-arrow">→</span>
                ${vmPart}
              </div>`;
            }).join('');

        return `
        <div class="bridge-card">
          <div class="bridge-head">
            <span class="bridge-name">${esc(b.name)}</span>
            <span class="bridge-ips">${esc(ipStr)}</span>
            <span class="bridge-portcount">${b.ports?.length ?? 0} port(s)</span>
          </div>
          <div class="bridge-ports-body">${portsHtml}</div>
        </div>`;
    }).join('');
}

/* -- Ports -- */
function renderPorts() {
    const p = S.ports; if (!p) return;
    $('tcp-body').innerHTML = portRows(p.tcp ?? []);
    $('udp-body').innerHTML = portRows(p.udp ?? []);
    // Update ports tab count badge
    const totalPorts = (p.tcp?.length ?? 0) + (p.udp?.length ?? 0);
    const portsCt = $('tab-ct-ports');
    if (portsCt) portsCt.textContent = totalPorts > 0 ? totalPorts : '';
}

function portRows(list) {
    if (!list.length) return '<tr><td colspan="3" class="empty">None</td></tr>';
    return list.sort((a, b) => a.port - b.port).map(p => `
        <tr>
          <td class="port-num-cell">${p.port}</td>
          <td class="port-addr-cell">${esc(p.localAddress || '*')}</td>
          <td class="port-proc-cell">${esc(p.process || '—')}</td>
        </tr>`).join('');
}

/* -- Routes -- */
function renderRoutes() {
    const tbody = $('routes-body');
    const routes = S.network.routes ?? [];
    const rtCt = $('tab-ct-routes');
    if (rtCt) rtCt.textContent = routes.length > 0 ? routes.length : '';
    if (!routes.length) { tbody.innerHTML = '<tr><td colspan="6" class="empty">No routes</td></tr>'; return; }

    tbody.innerHTML = routes.map(r => {
        const isDefault = r.destination === 'default' || r.destination === '0.0.0.0/0';
        return `<tr class="${isDefault ? 'default-row' : ''}">
          <td class="route-dst">${esc(r.destination)}</td>
          <td class="route-gw">${esc(r.gateway ?? '—')}</td>
          <td class="route-dev">${esc(r.interface)}</td>
          <td>${esc(r.protocol ?? '—')}</td>
          <td>${esc(r.scope ?? '—')}</td>
          <td>${r.metric}</td>
        </tr>`;
    }).join('');
}

// ============================================================
// Firewall
// ============================================================
function renderFirewall() {
    const fw = S.firewall; if (!fw) return;

    // UFW Banner
    const ufw = fw.ufw ?? {};
    const banner = $('ufw-banner');
    const statusWord = $('ufw-status-word');
    if (ufw.active !== undefined) {
        banner.className = `ufw-banner ${ufw.active ? 'active' : 'inactive'}`;
        statusWord.textContent = ufw.active ? '✓ UFW Active' : '✗ UFW Inactive';
    }
    const polEl = $('ufw-policies');
    if (polEl && ufw.active) {
        polEl.innerHTML = [
            { label: 'Incoming', val: ufw.defaultIncoming },
            { label: 'Outgoing', val: ufw.defaultOutgoing },
            { label: 'Forward', val: ufw.defaultForward },
        ].map(p => `<div class="pol-item"><span class="pol-label">${p.label}</span><span class="pol-val ${(p.val || '').toLowerCase()}">${p.val || '—'}</span></div>`).join('');
    }

    // UFW meta count
    const ufwMeta = $('ufw-rule-count');
    if (ufwMeta) ufwMeta.textContent = ufw.active
        ? `${ufw.rules?.length ?? 0} rule${ufw.rules?.length !== 1 ? 's' : ''}`
        : 'inactive';
    // Auto-expand UFW section only when active with rules
    if (ufw.active && ufw.rules?.length > 0) autoExpand('ufw-rules-body-wrap', 'ufw-rules-chev');

    // UFW Rules
    const ufwBody = $('ufw-rules-body');
    if (!ufw.rules?.length) {
        ufwBody.innerHTML = '<tr><td colspan="5" class="empty">No UFW rules</td></tr>';
    } else {
        ufwBody.innerHTML = ufw.rules.map(r => `
          <tr>
            <td class="num-col">${r.number}</td>
            <td>${esc(r.to)}</td>
            <td><span class="ufw-action-chip ${r.action.toUpperCase()}">${r.action}</span></td>
            <td>${esc(r.direction || '—')}</td>
            <td>${esc(r.from || 'Anywhere')}</td>
          </tr>`).join('');
    }

    // IPTables
    renderIptChain('ipt-input-body', 'ipt-input-policy', fw.iptables?.input);
    renderIptChain('ipt-fwd-body', 'ipt-fwd-policy', fw.iptables?.forward);
    renderIptChain('ipt-nat-body', 'ipt-nat-policy', fw.iptables?.natPostrouting);
}

// Map of chain bodyId → collapsible wrap id
const IPT_WRAP = {
    'ipt-input-body': 'ipt-input-wrap',
    'ipt-fwd-body': 'ipt-fwd-wrap',
    'ipt-nat-body': 'ipt-nat-wrap',
};
const IPT_CHEV = {
    'ipt-input-body': 'ipt-input-chev',
    'ipt-fwd-body': 'ipt-fwd-chev',
    'ipt-nat-body': 'ipt-nat-chev',
};

function renderIptChain(bodyId, polId, chain) {
    const tbody = $(bodyId); if (!tbody) return;
    const polEl = $(polId);

    // available=false means the binary (iptables) was not found on this host
    if (chain?.available === false) {
        polEl && (polEl.textContent = '—');
        tbody.innerHTML = '<tr><td colspan="9" class="empty" style="color:var(--text-muted)">iptables not available on this host</td></tr>';
        return;
    }

    if (polEl && chain?.policy) {
        polEl.textContent = chain.policy;
        polEl.className = `fw-policy-pill ${chain.policy} collapsible-meta`;
    }
    const rules = chain?.rules ?? [];
    if (!rules.length) {
        tbody.innerHTML = '<tr><td colspan="9" class="empty">No rules (chain empty)</td></tr>';
        return;
    }
    // Auto-expand chains that have non-trivial rules (non-libvirt OR has traffic)
    const wrapId = IPT_WRAP[bodyId];
    const chevId = IPT_CHEV[bodyId];
    if (wrapId) {
        const pktNum = r => parseInt(r.packets) || 0;
        const hasInteresting = rules.some(r => {
            const t = r.target?.toUpperCase() ?? '';
            const isNoise = t.startsWith('LIBVIRT') || t.startsWith('DOCKER');
            return !isNoise || pktNum(r) > 0;
        });
        if (hasInteresting) autoExpand(wrapId, chevId);
    }
    tbody.innerHTML = rules.map(r => `
        <tr>
          <td class="num-col">${r.lineNumber || ''}</td>
          <td><span class="target-chip ${r.target}">${esc(r.target)}</span></td>
          <td>${esc(r.protocol)}</td>
          <td>${esc(r.in)}</td>
          <td>${esc(r.out)}</td>
          <td>${esc(r.source)}</td>
          <td>${esc(r.destination)}</td>
          <td style="font-size:0.68rem;color:var(--text-muted)">${esc(r.options)}</td>
          <td class="pkts-col">${fmtNum(r.packets)}</td>
        </tr>`).join('');
}

// ============================================================
// Services
// ============================================================
function renderServices() {
    const el = $('services-grid');
    const svcs = S.services?.services ?? [];
    if (!svcs.length) { el.innerHTML = '<div class="empty">No service data</div>'; return; }

    const active = svcs.filter(s => s.isActive).length;
    const failed = svcs.filter(s => s.subState === 'failed').length;
    const metaEl = $('services-meta');
    if (metaEl) metaEl.textContent = `${active}/${svcs.length} active${failed ? ' · ' + failed + ' failed' : ''}`;

    el.innerHTML = svcs.map(s => {
        const cls = s.loadState === 'not-found' ? 'not-found'
            : !s.isLoaded ? 'not-found'
                : s.subState === 'failed' ? 'failed'
                    : s.isActive ? 'active' : 'inactive';
        const sub = s.activeState + (s.subState ? ` (${s.subState})` : '');
        return `<div class="svc-card ${cls}">
          <div class="svc-dot ${cls}"></div>
          <div class="svc-info">
            <div class="svc-name">${esc(s.name)}</div>
            <div class="svc-sub">${esc(sub)}</div>
          </div>
        </div>`;
    }).join('');
    // Auto-expand if any service is failed
    if (failed > 0) autoExpand('services-body', 'services-chev');
}

// ============================================================
// Logs
// ============================================================
function toggleLogs() {
    S.logsVisible = !S.logsVisible;
    $('log-wrap').style.display = S.logsVisible ? '' : 'none';
    $('log-btn').textContent = S.logsVisible ? 'Hide Logs' : 'Show Logs';
    if (S.logsVisible) loadLogs();
}

function renderLogs() {
    const data = S.logs;
    const lines = data?.logLines ?? [];
    S.allLogs = lines;

    // Show source badge next to the toggle button
    const btn = $('log-btn');
    if (btn && data?.source) {
        const srcLabel = data.source === 'file'
            ? `<span style="font-size:0.65rem;font-family:var(--mono);color:var(--primary);margin-left:0.5rem" title="${data.logFile ?? ''}">● file</span>`
            : `<span style="font-size:0.65rem;font-family:var(--mono);color:var(--text-muted);margin-left:0.5rem">● journal</span>`;
        // Inject after button (as sibling, not inside)
        const existing = document.getElementById('log-src-badge');
        if (existing) existing.remove();
        const span = document.createElement('span');
        span.id = 'log-src-badge';
        span.innerHTML = srcLabel;
        btn.after(span);
    }

    displayLogs(lines);
}

function filterLogs() {
    const q = $('log-filter').value.toLowerCase();
    displayLogs(q ? S.allLogs.filter(l => l.toLowerCase().includes(q)) : S.allLogs);
}

function displayLogs(lines) {
    const box = $('log-box'); if (!box) return;
    box.innerHTML = lines.map(l => {
        const low = l.toLowerCase();
        const cls = low.includes('error') || low.includes('crit') ? 'err'
            : low.includes('warn') ? 'warn'
                : low.includes('debug') ? 'dbg' : 'info';
        return `<div class="log-line ${cls}">${esc(l)}</div>`;
    }).join('');
    box.scrollTop = box.scrollHeight;
}

// ============================================================
// UI helpers
// ============================================================

/** Toggle a collapsible body open/closed */
function toggleCollapsible(bodyId, chevId, forceOpen) {
    const body = $(bodyId);
    const chev = $(chevId);
    if (!body) return;
    const isOpen = body.classList.contains('open');
    const open = forceOpen !== undefined ? forceOpen : !isOpen;
    body.classList.toggle('open', open);
    if (chev) chev.classList.toggle('open', open);
}

/** Open a collapsible if it contains data (non-empty tbody or grid) */
function autoExpand(bodyId, chevId) {
    const body = $(bodyId); if (!body) return;
    // Already open — leave it
    if (body.classList.contains('open')) return;
    toggleCollapsible(bodyId, chevId, true);
}

/** Scroll-spy: highlight nav tab matching the section currently in view */
function initScrollSpy() {
    const sections = ['hardware', 'vms', 'local-db', 'network', 'firewall', 'services', 'logs'];
    const tabs = document.querySelectorAll('.nav-tab');

    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                const id = e.target.id;
                tabs.forEach(t => {
                    t.classList.toggle('spy-active',
                        t.getAttribute('onclick')?.includes(`'${id}'`));
                });
            }
        });
    }, { threshold: 0.15, rootMargin: '-60px 0px -40% 0px' });

    sections.forEach(id => {
        const el = $(id);
        if (el) obs.observe(el);
    });
}

function scrollSec(id, btn) {
    const el = $(id); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active', 'spy-active'));
    if (btn) btn.classList.add('active');
}

function cmapTab(name, btn) {
    document.querySelectorAll('.net-tab').forEach(t => t.style.display = 'none');
    document.querySelectorAll('.cmap-tab').forEach(t => t.classList.remove('active'));
    const el = $(`tab-${name}`);
    if (el) { el.style.display = ''; el.classList.add('active'); }
    if (btn) btn.classList.add('active');
    S.activeCmapTab = name;
}

function copyKey(key, btn) {
    navigator.clipboard?.writeText(key).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'copied!'; btn.classList.add('copied');
        setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1500);
    });
}

// ============================================================
// Formatting
// ============================================================
function fmtUptime(secs) {
    if (!secs || secs < 0) return '—';
    const d = Math.floor(secs / 86400);
    const h = Math.floor((secs % 86400) / 3600);
    const m = Math.floor((secs % 3600) / 60);
    if (d > 0) return `${d}d ${h}h ${m}m`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m ${secs % 60}s`;
}

function fmtAgo(secs) {
    return fmtUptime(secs) + ' ago';
}

function fmtAge(secs) {
    if (secs < 0) return 'Never';
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    return `${Math.floor(secs / 3600)}h ago`;
}

function fmtBytes(b) {
    if (!b || b === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(Math.abs(b)) / Math.log(1024));
    return (b / Math.pow(1024, i)).toFixed(1) + ' ' + units[Math.min(i, 4)];
}

function fmtNum(n) {
    if (!n) return '0';
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'G';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n;
}

function trunc(s, n) {
    if (!s || s === '—') return '—';
    return s.length > n ? s.slice(0, n) + '…' : s;
}

/** Shorten a UUID/hex ID to "e927…7880" style — shows both ends */
function truncId(id, head = 4, tail = 4) {
    if (!id || id === '—') return '—';
    const s = id.replace(/-/g, '');          // strip hyphens for UUID
    if (s.length <= head + tail) return id;  // already short enough
    return s.slice(0, head) + '…' + s.slice(-tail);
}

function isRunning(vm) {
    // VmStatus.Running = 3 — serialised as integer (no [JsonStringEnumConverter])
    return vm.status === 3;
}

// ── Service health helpers ────────────────────────────────────────────────

/** Normalise ServiceStatus to a CSS class key */
function svcReadiness(svc) {
    const s = (svc.status ?? '').toLowerCase();
    if (s === 'ready') return 'ready';
    if (s === 'checking') return 'checking';
    if (s === 'timedout') return 'timedout';
    if (s === 'failed') return 'failed';
    return 'pending';
}

/** Build a single service chip with tooltip */
function svcChip(svc) {
    const cls = svcReadiness(svc);
    const label = svc.port ? `${svc.name}:${svc.port}` : svc.name;
    const readyAt = svc.readyAt ? 'Ready ' + fmtAge(Math.floor((Date.now() - new Date(svc.readyAt).getTime()) / 1000)) : '';
    const checkAt = svc.lastCheckAt ? 'Checked ' + fmtAge(Math.floor((Date.now() - new Date(svc.lastCheckAt).getTime()) / 1000)) : '';
    const msg = svc.statusMessage ? esc(svc.statusMessage).slice(0, 120) : '';
    const tipLines = [
        `<strong>${esc(svc.name)}</strong>`,
        svc.port ? `port: ${svc.port}${svc.protocol ? '/' + svc.protocol : ''}` : '',
        `type: ${svc.checkType ?? ''}`,
        readyAt,
        checkAt,
        msg,
    ].filter(Boolean).join('<br>');

    return `<span class="svc-chip ${cls}" title="">
        <span class="svc-chip-dot"></span>${esc(label)}
        <span class="svc-chip-tip">${tipLines}</span>
      </span>`;
}

/** Compute overall health for a VM — returns 'ready'|'partial'|'failed'|'pending' */
function vmHealthClass(vm) {
    if (!vm.services?.length) return 'pending';
    if (vm.isFullyReady) return 'ready';
    const states = vm.services.map(svcReadiness);
    if (states.some(s => s === 'failed' || s === 'timedout')) return 'failed';
    if (states.some(s => s === 'ready')) return 'partial';
    return 'pending';
}

/** Render the compact health cell for the tenant VM table */
function renderHealthCell(vm) {
    const svcs = vm.services ?? [];
    if (!svcs.length) return '<span style="color:var(--text-muted);font-size:0.72rem">—</span>';

    const cls = vmHealthClass(vm);
    const label = vm.isFullyReady ? 'Ready'
        : cls === 'failed' ? 'Failed'
            : cls === 'partial' ? `${svcs.filter(s => svcReadiness(s) === 'ready').length}/${svcs.length}`
                : 'Pending';

    const chips = svcs.map(s => svcChip(s)).join('');
    return `<div>
        <span class="health-overall ${cls}">
          <span class="svc-chip-dot" style="width:5px;height:5px;border-radius:50%;background:currentColor;display:inline-block"></span>
          ${label}
        </span>
        <div class="svc-chips" style="margin-top:0.3rem">${chips}</div>
      </div>`;
}

/**
 * Resolve the best public URL for a VM.
 * Priority:
 *   1. Actual stored ingress URL from orchestrator DB (S.vmIngress)
 *   2. Constructed from ingressBaseDomain + vm.name (same formula as server)
 *   3. Direct internal IP with a warning (not publicly routable)
 */
function resolveVmUrl(vm) {
    const vmId = vm.vmId ?? vm.spec?.id;
    const vmName = vm.name ?? vm.spec?.name;
    const labels = vm.spec?.labels ?? vm.labels ?? {};
    const ip = vm.spec?.ipAddress || vm.ipAddress
        || labels['dht-advertise-ip']
        || labels['blockstore-advertise-ip'];

    // 1. Orchestrator DB
    if (vmId && S.vmIngress[vmId]) {
        return { url: S.vmIngress[vmId], source: 'ingress' };
    }

    // 2. Constructed formula
    if (S.ingressBaseDomain && vmName) {
        return { url: `https://${vmName}.${S.ingressBaseDomain}`, source: 'formula' };
    }

    // 3. Direct IP fallback
    if (ip) {
        return { url: `http://${ip}`, source: 'direct' };
    }

    return null;
}

function vmStateName(state) {
    // Must match VmStatus enum exactly:
    // 0=Pending, 1=Scheduling, 2=Provisioning, 3=Running, 4=Paused,
    // 5=Suspended, 6=Stopping, 7=Stopped, 8=Deleting, 9=Deleted, 10=Migrating, 11=Error
    const map = {
        0: 'Pending', 1: 'Scheduling', 2: 'Provisioning', 3: 'Running',
        4: 'Paused', 5: 'Suspended', 6: 'Stopping', 7: 'Stopped',
        8: 'Deleting', 9: 'Deleted', 10: 'Migrating', 11: 'Error'
    };
    return map[state] ?? String(state);
}

function stateBadge(state, vmId) {
    const name = vmStateName(state);
    const cls = name === 'Running' ? 'state-running'
        : name === 'Failed' || name === 'NotFound' ? 'state-error'
            : name === 'Stopped' || name === 'Deleted' ? 'state-stopped'
                : 'state-other';

    // Show download progress for VMs in Creating state
    if (name === 'Creating' && vmId) {
        const dl = (S.downloads ?? []).find(d => d.vmId === vmId);
        if (dl && dl.totalBytes > 0) {
            const mb = (dl.downloadedBytes / 1024 / 1024).toFixed(0);
            const totalMb = (dl.totalBytes / 1024 / 1024).toFixed(0);
            return `<span class="state-badge state-other">Downloading ${dl.percentComplete}%</span>`
                + `<div class="dl-progress-bar"><div class="dl-progress-fill" style="width:${dl.percentComplete}%"></div></div>`
                + `<span class="dl-detail">${mb} / ${totalMb} MB</span>`;
        }
    }

    return `<span class="state-badge ${cls}">${name}</span>`;
}

// ============================================================
// Export / Snapshot
// ============================================================

// All endpoints to collect, with human labels
const EXPORT_ENDPOINTS = [
    { key: 'summary', label: 'Node Summary', url: '/api/dashboard/summary' },
    { key: 'snapshot', label: 'Resource Snapshot', url: '/api/node/snapshot' },
    { key: 'allocation', label: 'Resource Allocation', url: '/api/node/allocation' },
    { key: 'vms', label: 'Virtual Machines', url: '/api/vms' },
    { key: 'obligations', label: 'Obligations', url: '/api/dashboard/obligations' },
    { key: 'network', label: 'Network Topology', url: '/api/dashboard/network' },
    { key: 'ports', label: 'Listening Ports', url: '/api/dashboard/ports' },
    { key: 'firewall', label: 'Firewall & IPTables', url: '/api/dashboard/firewall' },
    { key: 'services', label: 'System Services', url: '/api/dashboard/services' },
    { key: 'logs', label: 'Node Agent Logs', url: '/api/dashboard/logs?lines=500' },
    { key: 'database', label: 'Local Database', url: '/api/dashboard/database' },
];

// Which sections are enabled — all on by default
const exportEnabled = new Set(EXPORT_ENDPOINTS.map(ep => ep.key));

// Raw fetched data per key — persists across re-collects so toggling
// an already-fetched section on/off updates the preview instantly
const exportData = {};

let exportPayload = null;  // the last assembled (filtered) snapshot

// ── open / close ─────────────────────────────────────────────
function openExport() {
    $('export-overlay').classList.add('open');
    // Build section list with toggles on first open
    if (!$('exp-item-summary')) buildSectionList();
    if (exportPayload) renderExportPreview(exportPayload);
}

function closeExport() {
    $('export-overlay').classList.remove('open');
}

function closeExportOnBackdrop(e) {
    if (e.target === $('export-overlay')) closeExport();
}

// ── build the section checklist (once, persists across collects) ──
function buildSectionList() {
    const container = $('export-sections');
    container.innerHTML = EXPORT_ENDPOINTS.map(ep => `
        <div class="export-section-item" id="exp-item-${ep.key}" data-key="${ep.key}">
          <label class="tog" title="Include in export">
            <input type="checkbox" id="exp-tog-${ep.key}" checked
                   onchange="onToggleSection('${ep.key}', this.checked)">
            <span class="tog-slider"></span>
          </label>
          <span class="export-sec-dot pending" id="exp-dot-${ep.key}"></span>
          <span class="export-sec-name" id="exp-lbl-${ep.key}">${ep.label}</span>
          <span class="export-sec-size" id="exp-size-${ep.key}"></span>
        </div>`).join('');
}

// ── toggle handler — called when user flips a switch ─────────
function onToggleSection(key, enabled) {
    if (enabled) {
        exportEnabled.add(key);
    } else {
        exportEnabled.delete(key);
    }

    // Update item appearance immediately
    const item = $(`exp-item-${key}`);
    const lbl = $(`exp-lbl-${key}`);
    const dot = $(`exp-dot-${key}`);
    const size = $(`exp-size-${key}`);

    if (item) item.style.opacity = enabled ? '1' : '0.4';
    if (lbl) lbl.style.color = enabled ? '' : 'var(--text-muted)';

    if (!enabled) {
        if (dot) { dot.className = 'export-sec-dot pending'; }
        if (size) { size.textContent = 'skipped'; size.style.color = 'var(--text-muted)'; }
    } else if (exportData[key]) {
        // Data was already fetched — restore done state
        if (dot) { dot.className = 'export-sec-dot done'; }
        if (size) { size.textContent = fmtBytes(JSON.stringify(exportData[key]).length); size.style.color = ''; }
    }

    // Rebuild payload from cached data and re-render preview
    rebuildPayload();
}

// ── assemble payload from cached data respecting enabled set ──
function rebuildPayload() {
    const snapshot = {
        _meta: {
            collectedAt: new Date().toISOString(),
            collectedAtUnix: Math.floor(Date.now() / 1000),
            nodeUrl: window.location.origin,
            sections: [...exportEnabled],
            dashboardVersion: '1.1',
        }
    };
    for (const ep of EXPORT_ENDPOINTS) {
        if (exportEnabled.has(ep.key) && exportData[ep.key] !== undefined) {
            snapshot[ep.key] = exportData[ep.key];
        }
    }
    exportPayload = snapshot;
    if (Object.keys(snapshot).length > 1) renderExportPreview(snapshot);
    return snapshot;
}

// ── collect (fetch) all enabled sections ─────────────────────
async function collectExport() {
    // Build section list if it hasn't been built yet
    if (!$('exp-item-summary')) buildSectionList();

    const collectBtn = $('export-collect-btn');
    const statusText = $('export-status-text');
    const copyBtn = $('export-copy-btn');
    const dlBtn = $('export-dl-btn');

    collectBtn.disabled = true;
    collectBtn.textContent = 'Collecting…';
    if (copyBtn) copyBtn.disabled = true;
    if (dlBtn) dlBtn.disabled = true;

    $('export-preview').textContent = 'Collecting…';
    $('export-preview-size').textContent = '';

    let completed = 0, skipped = 0, errors = 0;

    // Fetch enabled sections concurrently; skip disabled ones
    const tasks = EXPORT_ENDPOINTS.map(async ep => {
        const item = $(`exp-item-${ep.key}`);
        const dot = $(`exp-dot-${ep.key}`);
        const size = $(`exp-size-${ep.key}`);
        const tog = $(`exp-tog-${ep.key}`);

        if (!exportEnabled.has(ep.key)) {
            // Disabled — skip without touching stored data
            if (dot) dot.className = 'export-sec-dot pending';
            if (size) { size.textContent = 'skipped'; size.style.color = 'var(--text-muted)'; }
            if (item) item.style.opacity = '0.4';
            skipped++;
            return;
        }

        // Mark as loading
        if (item) { item.className = 'export-section-item loading'; item.style.opacity = '1'; }
        if (dot) dot.className = 'export-sec-dot loading';
        if (size) { size.textContent = '…'; size.style.color = ''; }
        if (tog) tog.disabled = true;

        try {
            const r = await fetch(ep.url);
            const data = r.ok ? await r.json() : null;

            exportData[ep.key] = r.ok ? data : { _error: `HTTP ${r.status}` };

            const bytes = JSON.stringify(exportData[ep.key]).length;

            if (r.ok) {
                if (item) item.className = 'export-section-item done';
                if (dot) dot.className = 'export-sec-dot done';
                if (size) { size.textContent = fmtBytes(bytes); size.style.color = ''; }
                completed++;
            } else {
                if (item) item.className = 'export-section-item error';
                if (dot) dot.className = 'export-sec-dot error';
                if (size) { size.textContent = 'error'; size.style.color = 'var(--danger)'; }
                errors++;
            }
        } catch (e) {
            exportData[ep.key] = { _error: e.message };
            if (item) item.className = 'export-section-item error';
            if (dot) dot.className = 'export-sec-dot error';
            if (size) { size.textContent = 'failed'; size.style.color = 'var(--danger)'; }
            errors++;
        } finally {
            if (tog) tog.disabled = false;
        }
    });

    await Promise.all(tasks);

    rebuildPayload();

    collectBtn.disabled = false;
    collectBtn.textContent = 'Re-collect';
    if (copyBtn) copyBtn.disabled = false;
    if (dlBtn) dlBtn.disabled = false;

    const enabled = EXPORT_ENDPOINTS.length - skipped;
    statusText.textContent = errors === 0
        ? `${completed}/${enabled} collected${skipped ? `, ${skipped} skipped` : ''}`
        : `${completed}/${enabled} ok, ${errors} failed${skipped ? `, ${skipped} skipped` : ''}`;
}

function renderExportPreview(payload) {
    const json = JSON.stringify(payload, null, 2);
    const preview = $('export-preview');
    const sizeEl = $('export-preview-size');

    // Truncate preview to keep DOM fast — full data is in exportPayload
    const MAX_PREVIEW = 80_000;
    if (json.length > MAX_PREVIEW) {
        preview.textContent = json.slice(0, MAX_PREVIEW) + '\n\n… (truncated in preview — full data available via Copy/Download)';
    } else {
        preview.textContent = json;
    }

    if (sizeEl) sizeEl.textContent = fmtBytes(json.length);
}

function exportCopy() {
    const payload = rebuildPayload();
    if (Object.keys(payload).length <= 1) { alert('Click "Collect All Data" first.'); return; }
    const json = JSON.stringify(payload, null, 2);
    navigator.clipboard?.writeText(json).then(() => {
        const btn = $('export-copy-btn');
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('success');
        setTimeout(() => { btn.textContent = orig; btn.classList.remove('success'); }, 2000);
    }).catch(() => alert('Clipboard write failed — use Download instead.'));
}

function exportDownload() {
    const payload = rebuildPayload();
    if (Object.keys(payload).length <= 1) { alert('Click "Collect All Data" first.'); return; }
    const json = JSON.stringify(payload, null, 2);
    const nodeId = payload.summary?.nodeId ?? 'node';
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `decloud-diag-${nodeId}-${ts}.json`;
    triggerDownload(json, filename, 'application/json');
}

function exportDownloadText() {
    const payload = rebuildPayload();
    if (Object.keys(payload).length <= 1) { alert('Click "Collect All Data" first.'); return; }
    const text = buildTextReport(payload);
    const nodeId = payload.summary?.nodeId ?? 'node';
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `decloud-diag-${nodeId}-${ts}.txt`;
    triggerDownload(text, filename, 'text/plain');
}

function triggerDownload(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/** Builds a human-readable plain-text diagnostic report */
function buildTextReport(p) {
    const lines = [];
    const hr = (char = '-') => lines.push(char.repeat(72));
    const h1 = (t) => { hr('='); lines.push(t.toUpperCase()); hr('='); };
    const h2 = (t) => { lines.push(''); lines.push('── ' + t + ' ──'); hr('-'); };
    const kv = (k, v) => lines.push(`  ${(k + ':').padEnd(28)} ${v ?? '—'}`);

    h1('DeCloud Node Diagnostic Report');
    lines.push(`  Generated: ${new Date().toISOString()}`);
    lines.push(`  Node URL:  ${window.location.origin}`);

    // Summary
    const s = p.summary;
    if (s) {
        h2('Node Identity');
        kv('Node ID', s.nodeId);
        kv('Hostname', s.hostname);
        kv('OS', s.os);
        kv('Uptime', fmtUptime(s.uptimeSeconds));
        kv('Agent Version', s.agentVersion);
        kv('Wallet', s.walletAddress);
        kv('Orchestrator', s.orchestrator?.connected ? `Connected (${fmtAgo(s.orchestrator.secondsAgo)})` : 'DISCONNECTED');
    }

    // Snapshot
    const snap = p.snapshot;
    if (snap) {
        h2('Hardware Resources');
        kv('CPU Usage', (snap.virtualCpuUsagePercent ?? 0).toFixed(1) + '%');
        kv('Memory', `${fmtBytes(snap.usedMemoryBytes)} / ${fmtBytes(snap.totalMemoryBytes)}`);
        kv('Storage', `${fmtBytes(snap.usedStorageBytes)} / ${fmtBytes(snap.totalStorageBytes)}`);
        kv('KVM', snap.kvmAvailable !== false ? 'Available' : 'Unavailable');
    }

    // VMs
    const vms = p.vms;
    if (vms?.length) {
        h2('Virtual Machines');
        for (const vm of vms) {
            const spec = vm.spec ?? {};
            lines.push(`  [${vmStateName(vm.status)}] ${vm.name ?? spec.name ?? vm.vmId}`);
            lines.push(`    ID:    ${vm.vmId}`);
            lines.push(`    Type:  vmRole=${spec.vmRole}  role=${spec.labels?.role ?? '—'}`);
            lines.push(`    IP:    ${spec.ipAddress || '—'}   VNC: ${vm.vncPort ?? '—'}`);
            lines.push(`    CPU:   ${spec.virtualCpuCores ?? '—'} vCPU   RAM: ${fmtBytes(spec.memoryBytes ?? 0)}`);
        }
    }

    // Network
    const net = p.network;
    if (net) {
        h2('Network Interfaces');
        for (const i of net.interfaces ?? []) {
            lines.push(`  ${i.isUp ? '[UP]  ' : '[DOWN]'} ${i.name.padEnd(18)} ${i.type.padEnd(12)} ${(i.ipAddresses ?? []).join(', ')}`);
        }

        h2('WireGuard Interfaces');
        for (const wg of net.wireguard ?? []) {
            lines.push(`  ${wg.name}  port=${wg.listenPort}  pubkey=${wg.publicKey}`);
            for (const peer of wg.peers ?? []) {
                const hs = peer.handshakeSecondsAgo != null ? fmtAge(peer.handshakeSecondsAgo) : 'Never';
                lines.push(`    PEER  ${peer.publicKey}`);
                lines.push(`      endpoint=${peer.endpoint ?? 'none'}  allowed=${peer.allowedIps}`);
                lines.push(`      handshake=${hs} [${peer.handshakeStatus?.toUpperCase()}]  rx=${fmtBytes(peer.rxBytes)}  tx=${fmtBytes(peer.txBytes)}`);
            }
        }

        h2('Bridge Interfaces');
        for (const b of net.bridges ?? []) {
            lines.push(`  ${b.name}  ${(b.ipAddresses ?? []).join(', ')}`);
            for (const port of b.ports ?? []) {
                const vm = port.vmName ? `${port.vmName} (${port.vmId})` : 'host/unassigned';
                lines.push(`    ${port.interface.padEnd(16)} → ${vm}`);
            }
        }

        h2('Routing Table');
        for (const r of net.routes ?? []) {
            lines.push(`  ${(r.destination ?? '').padEnd(22)} via ${(r.gateway ?? 'direct').padEnd(18)} dev ${r.interface}`);
        }
    }

    // Ports
    const ports = p.ports;
    if (ports) {
        h2('Listening Ports — TCP');
        for (const p of (ports.tcp ?? []).sort((a, b) => a.port - b.port))
            lines.push(`  ${String(p.port).padEnd(7)} ${(p.localAddress || '*').padEnd(20)} ${p.process || '—'}`);
        h2('Listening Ports — UDP');
        for (const p of (ports.udp ?? []).sort((a, b) => a.port - b.port))
            lines.push(`  ${String(p.port).padEnd(7)} ${(p.localAddress || '*').padEnd(20)} ${p.process || '—'}`);
    }

    // Firewall
    const fw = p.firewall;
    if (fw) {
        h2('Firewall — UFW');
        const ufw = fw.ufw ?? {};
        lines.push(`  Status:   ${ufw.active ? 'ACTIVE' : 'INACTIVE'}`);
        if (ufw.active) {
            lines.push(`  Incoming: ${ufw.defaultIncoming}  Outgoing: ${ufw.defaultOutgoing}  Forward: ${ufw.defaultForward}`);
            lines.push('');
            for (const r of ufw.rules ?? [])
                lines.push(`  [${r.number}] ${r.to.padEnd(20)} ${r.action.padEnd(7)} ${r.direction.padEnd(4)} from ${r.from}`);
        }

        const printChain = (title, chain) => {
            h2(title + `  [policy: ${chain?.policy ?? '?'}]`);
            for (const r of chain?.rules ?? [])
                lines.push(`  ${String(r.lineNumber).padEnd(4)} ${r.target.padEnd(12)} ${r.protocol.padEnd(6)} in=${r.in.padEnd(10)} out=${r.out.padEnd(10)} ${r.source.padEnd(20)} ${r.destination}  ${r.options}`);
        };
        printChain('iptables INPUT', fw.iptables?.input);
        printChain('iptables FORWARD', fw.iptables?.forward);
        printChain('iptables NAT POSTROUTING', fw.iptables?.natPostrouting);
    }

    // Services
    const svcs = p.services?.services ?? [];
    if (svcs.length) {
        h2('System Services');
        for (const s of svcs)
            lines.push(`  ${s.isActive ? '[ACTIVE  ]' : '[INACTIVE]'} ${s.name.padEnd(32)} ${s.activeState}/${s.subState}`);
    }

    // Logs
    const logs = p.logs?.logLines ?? [];
    if (logs.length) {
        h2(`Node Agent Logs (${logs.length} lines, source: ${p.logs?.source ?? 'unknown'})`);
        for (const l of logs) lines.push('  ' + l);
    }

    hr('=');
    lines.push('END OF REPORT');
    hr('=');
    return lines.join('\n');
}