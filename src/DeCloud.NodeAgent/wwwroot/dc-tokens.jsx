/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   dc-tokens.jsx
   Design tokens · mock data · API helper · icons
   Exports everything to window for downstream scripts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const { createContext, useContext, useState: _useState, useEffect: _useEffect } = React;

function useBreakpoint() {
  const [w, setW] = _useState(() => window.innerWidth);
  _useEffect(() => {
    const h = () => setW(window.innerWidth);
    window.addEventListener('resize', h);
    return () => window.removeEventListener('resize', h);
  }, []);
  return { mobile: w < 768, tablet: w >= 768 && w < 1024, desktop: w >= 1024 };
}

const BASE = {
  bg0:'#0a0b0f', bg1:'#0f1117', bg2:'#12141a', bg3:'#1a1d26', bg4:'#22262f', bg5:'#2a2f3a',
  a2:'#00a8ff', a3:'#8b5cf6', aw:'#f59e0b', ad:'#ef4444',
  t1:'#f0f2f5', t2:'#9ca3af', t3:'#6b7280',
  bd:'rgba(255,255,255,0.07)', bh:'rgba(255,255,255,0.14)',
  ff:"'Outfit',sans-serif", fm:"'JetBrains Mono',monospace",
};

const ThemeCtx = createContext({ accent:'#00d4aa', compact:false });
const useTheme = () => useContext(ThemeCtx);

const VM_STATUS = {
  0:{text:'Unknown',      color:'#6b7280'},
  1:{text:'Queued',       color:'#f59e0b'},
  2:{text:'Provisioning', color:'#00a8ff'},
  3:{text:'Running',      color:'#00d4aa'},
  4:{text:'Stopped',      color:'#6b7280'},
  5:{text:'Error',        color:'#ef4444'},
  6:{text:'Stopping',     color:'#f59e0b'},
  7:{text:'Starting',     color:'#00a8ff'},
  8:{text:'Scheduled',    color:'#8b5cf6'},
  9:{text:'Migrating',    color:'#8b5cf6'},
};

const NAV_GROUPS = [
  { label:'Command Center', items:[{id:'dashboard',    label:'Dashboard'}] },
  { label:'Infrastructure', items:[
    {id:'templates',    label:'Template Marketplace'},
    {id:'my-templates', label:'My Templates'},
    {id:'vms',          label:'Virtual Machines'},
    {id:'nodes',        label:'Nodes'},
  ]},
  { label:'Configuration', items:[
    {id:'ssh-keys', label:'SSH Keys'},
    {id:'settings', label:'Settings'},
  ]},
];

/* ── Mock data (fallback when API unavailable) ─────────────── */
const MOCK_VMS = [
  { id:'v1', name:'stable-diffusion-1', nodeId:'n1', cpu:4,  mem:8192,  disk:80,  sc:3, tier:'Standard',   tmpl:'Stable Diffusion', ip:'192.168.122.10', msgs:2,
    services:[{name:'System',status:'Ready'},{name:'WebUI :7860',status:'Ready'}] },
  { id:'v2', name:'postgres-dev',       nodeId:'n2', cpu:2,  mem:2048,  disk:40,  sc:3, tier:'Burstable',  tmpl:'PostgreSQL',        ip:'192.168.122.11', msgs:0,
    services:[{name:'System',status:'Ready'},{name:'Postgres :5432',status:'Ready'}] },
  { id:'v3', name:'vscode-server',      nodeId:'n3', cpu:2,  mem:4096,  disk:20,  sc:2, tier:'Standard',   tmpl:'VS Code Server',    ip:'pending',        msgs:1,
    services:[{name:'System',status:'Checking'},{name:'VS Code :8080',status:'Pending'}] },
  { id:'v4', name:'ollama-gpu-1',        nodeId:'n1', cpu:8,  mem:16384, disk:100, sc:4, tier:'Guaranteed', tmpl:'Ollama',            ip:'192.168.122.12', msgs:0,
    services:[{name:'System',status:'Ready'},{name:'Ollama :11434',status:'Ready'}] },
];

const MOCK_NODES = [
  { id:'n1', name:'us-east-node-1',    region:'us-east',   uptime:99.8, vms:3, cpu:16, mem:64,  pts:128, gpu:false },
  { id:'n2', name:'tr-south-node-3',   region:'tr-south',  uptime:97.2, vms:1, cpu:8,  mem:32,  pts:48,  gpu:false },
  { id:'n3', name:'eu-central-node-2', region:'eu-central',uptime:99.9, vms:2, cpu:32, mem:128, pts:256, gpu:true  },
  { id:'n4', name:'us-west-gpu-1',     region:'us-west',   uptime:98.5, vms:0, cpu:16, mem:96,  pts:192, gpu:true  },
];

const MOCK_TEMPLATES = [
  { id:'t1', name:'Stable Diffusion WebUI', cat:'AI & ML',  gpu:true,  deploys:1240, rating:4.8, desc:'AUTOMATIC1111 with GPU acceleration. 1-click deploy.' },
  { id:'t2', name:'PostgreSQL Database',    cat:'Databases',gpu:false, deploys:892,  rating:4.9, desc:'Production-ready Postgres 16 with persistent storage.' },
  { id:'t3', name:'VS Code Server',         cat:'Dev Tools',gpu:false, deploys:2100, rating:4.7, desc:'Full-featured browser IDE via code-server.' },
  { id:'t4', name:'Ollama + Open WebUI',    cat:'AI & ML',  gpu:true,  deploys:3200, rating:4.9, desc:'Local LLM inference. Run Llama 3, Mistral, Gemma.' },
  { id:'t5', name:'Shadowsocks Proxy',      cat:'Privacy',  gpu:false, deploys:445,  rating:4.6, desc:'High-performance encrypted SOCKS5 proxy.' },
  { id:'t6', name:'Private Browser',        cat:'Privacy',  gpu:false, deploys:678,  rating:4.5, desc:'Neko WebRTC remote desktop. Browse from VM IP.' },
];

const MOCK_KEYS = [
  { id:'k1', name:'macbook-pro',  fp:'SHA256:xK9mQ2vPL3nR8sF4tJ7uB1wE6aD5cH0iG', added:'2026-03-12' },
  { id:'k2', name:'work-laptop',  fp:'SHA256:zR4wT7pK1mN8bX2yL9cQ3sF6vH0jG5dE',  added:'2026-04-01' },
];

const MOCK_STATS = {
  totalVms:8, onlineNodes:12,
  availableComputePoints:240, totalComputePoints:480,
  availableMemoryGb:64, availableStorageGb:820,
};

/* ── API helper — tries backend, returns null on failure ───── */
const getOrchestratorUrl = () => localStorage.getItem('orchestratorUrl') || 'http://localhost:5000';

async function api(endpoint, opts={}) {
  const url   = getOrchestratorUrl();
  const token = localStorage.getItem('authToken') || '';
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 5000);
  try {
    const r = await fetch(url + endpoint, {
      ...opts,
      headers: {
        'Content-Type':'application/json',
        ...(token ? { 'Authorization':`Bearer ${token}` } : {}),
        ...opts.headers,
      },
      signal: ctrl.signal,
    });
    clearTimeout(timer);
    if (!r.ok) throw new Error(r.status);
    return r.json();
  } catch { clearTimeout(timer); return null; }
}

/* ── Icon path library ─────────────────────────────────────── */
const IP = {
  dashboard:     'M3 3h7v9H3zm11 0h7v5h-7zm0 8h7v9h-7zM3 16h7v5H3z',
  templates:     'M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16zM3.27 6.96L12 12.01l8.73-5.05M12 22.08V12',
  'my-templates':'M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4z',
  vms:           'M2 3h20v14H2zM8 21h8M12 17v4',
  nodes:         'M4 4h6v6H4zm10 0h6v6h-6zm0 10h6v6h-6zM4 14h6v6H4z',
  'ssh-keys':    'M21 2l-2 2m-7.61 7.61a5.5 5.5 0 11-7.778 7.778 5.5 5.5 0 017.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4',
  settings:      'M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z',
  search:   'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z',
  plus:     'M12 5v14M5 12h14',
  refresh:  'M23 4v6h-6M1 20v-6h6m16.49-9A9 9 0 015.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0118.36 18.36',
  bell:     'M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9m-4.27 13a2 2 0 01-3.46 0',
  wallet:   'M21 12V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2h7m7-7h-4a2 2 0 00-2 2v1a2 2 0 002 2h4v-5z',
  terminal: 'M4 17l6-6-6-6M12 19h8',
  link:     'M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71',
  trash:    'M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2',
  play:     'M5 3l14 9-14 9V3z',
  pause:    'M6 4h4v16H6zM14 4h4v16h-4z',
  x:        'M18 6L6 18M6 6l12 12',
  check:    'M20 6L9 17l-5-5',
  globe:    'M12 2a10 10 0 100 20A10 10 0 0012 2zm0 0c-2.76 0-5 4.48-5 10s2.24 10 5 10 5-4.48 5-10S14.76 2 12 2zm-10 10h20',
  message:  'M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z',
  sliders:  'M4 6h16M8 12h12M4 18h16M4 6V4m4 8V8m12 6v4',
  shield:   'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z',
  deposit:  'M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6',
  copy:     'M8 4H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-2M8 4a2 2 0 012-2h4a2 2 0 012 2v0a2 2 0 01-2 2h-4a2 2 0 01-2-2v0z',
  menu:     'M3 12h18M3 6h18M3 18h18',
  folder:   'M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z',
};

Object.assign(window, {
  BASE, ThemeCtx, useTheme, useBreakpoint,
  VM_STATUS, NAV_GROUPS,
  MOCK_VMS, MOCK_NODES, MOCK_TEMPLATES, MOCK_KEYS, MOCK_STATS,
  getOrchestratorUrl, api, IP,
});
