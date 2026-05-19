/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   dc-pages.jsx
   useData hook (API ↔ mock fallback) + all 7 pages
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const { useState, useEffect, useCallback } = React;
const {
  BASE, useTheme, useBreakpoint, VM_STATUS,
  MOCK_VMS, MOCK_NODES, MOCK_TEMPLATES, MOCK_KEYS, MOCK_STATS,
  api, getOrchestratorUrl,
  Ico, PBtn, SBtn, IBtn, Badge,
  ServiceReadiness, StatCard, PageHeader, THead,
  Card, CardHead, EmptyState, inputSty,
} = window;

/* ━━ DATA HOOK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
function useData() {
  const [vms,   setVms]   = useState(MOCK_VMS);
  const [nodes, setNodes] = useState(MOCK_NODES);
  const [keys,  setKeys]  = useState(MOCK_KEYS);
  const [stats, setStats] = useState(MOCK_STATS);
  const [apiConnected, setApiConnected] = useState(false);
  const [loading, setLoading] = useState(false);

  const normalizeVM = vm => ({
    id:       vm.id,
    name:     vm.name,
    nodeId:   vm.nodeId,
    cpu:      vm.spec?.virtualCpuCores || 0,
    mem:      Math.round((vm.spec?.memoryBytes || 0) / (1024*1024)),
    disk:     Math.round((vm.spec?.diskBytes   || 0) / (1024*1024*1024)),
    sc:       vm.status,
    tier:     (['Guaranteed','Standard','Balanced','Burstable'])[vm.spec?.qualityTier ?? 1] || 'Standard',
    tmpl:     vm.templateName || '',
    ip:       vm.networkConfig?.privateIp || 'pending',
    msgs:     vm.messages?.length || 0,
    services: vm.services || [],
    nodeHost: vm.networkConfig?.nodeAgentHost || '',
    nodePort: vm.networkConfig?.nodeAgentPort || 5100,
    sshHost:  vm.networkConfig?.sshJumpHost   || '',
    sshPort:  vm.networkConfig?.sshJumpPort   || 2222,
  });

  const fetchAll = useCallback(async () => {
    setLoading(true);
    const [vmsR, nodesR, statsR] = await Promise.all([
      api('/api/vms'),
      api('/api/marketplace/nodes?featured=true'),
      api('/api/system/stats'),
    ]);
    if (vmsR?.success  && vmsR.data?.items)  { setVms(vmsR.data.items.map(normalizeVM)); setApiConnected(true); }
    if (nodesR?.success && nodesR.data)       setNodes(nodesR.data);
    if (statsR?.success && statsR.data)       setStats(statsR.data);
    setLoading(false);
  }, []);

  useEffect(() => { fetchAll(); }, []);

  return { vms, setVms, nodes, keys, setKeys, stats, loading, apiConnected, refresh: fetchAll };
}

/* ━━ DASHBOARD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
function DashboardPage({ setModal }) {
  const { compact } = useTheme();
  const { vms, stats, loading, refresh } = useData();

  const dashStats = [
    { label:'Active Nodes',    val:String(stats.onlineNodes||0),            sub:'Heartbeat OK',         color:'#00d4aa', icon:'M4 4h6v6H4zm10 0h6v6h-6zm0 10h6v6h-6zM4 14h6v6H4z' },
    { label:'Virtual Machines',val:String(stats.totalVms||0),               sub:`${vms.filter(v=>v.sc===3).length} running`, color:'#00a8ff', icon:'M2 3h20v14H2zM8 21h8M12 17v4' },
    { label:'Compute Points',  val:String(stats.availableComputePoints||0), sub:`of ${stats.totalComputePoints||0} available`, color:'#8b5cf6', icon:'M4 4h16v16H4zM9 9h6v6H9zM9 1v3m6-3v3M9 20v3m6-3v3M1 9h3m16 0h3M1 15h3m16 0h3' },
    { label:'Free Memory',     val:`${stats.availableMemoryGb||0} GB`,      sub:'Network-wide',          color:'#f59e0b', icon:'M6 19V8m4 11V4m4 15V10m4 9V3' },
  ];

  const bp = useBreakpoint();
  const pad = compact ? '16px 16px' : bp.mobile ? '20px 16px' : bp.tablet ? '24px 24px' : '32px 40px';
  return (
    <div style={{padding:pad, overflowY:'auto', flex:1}}>
      <PageHeader title="Dashboard" subtitle="Real-time compute infrastructure overview">
        <SBtn onClick={refresh}><Ico id="refresh" size={13}/> Refresh</SBtn>
        <PBtn onClick={()=>setModal({type:'create-vm'})}><Ico id="plus" size={13} color="#0a0b0f"/> Create VM</PBtn>
      </PageHeader>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(180px,1fr))',gap:16,marginBottom:24}}>
        {dashStats.map(s=><StatCard key={s.label} {...s}/>)}
      </div>
      <Card>
        <CardHead title="Recent Virtual Machines" action={<span style={{fontSize:12,color:'#00d4aa',cursor:'pointer'}}>View all →</span>}/>
        <div style={{overflowX:'auto'}}>
        <table style={{width:'100%',borderCollapse:'collapse',minWidth:480}}>
          <THead cols={['Name','Node','CPU','Memory','Status']}/>
          <tbody>
            {vms.slice(0,5).map((vm,i)=>{
              const s = VM_STATUS[vm.sc]||VM_STATUS[4];
              return (
                <tr key={vm.id} style={{borderBottom:i<4?`1px solid ${BASE.bd}`:'none'}}>
                  <td style={{padding:'11px 16px'}}>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <div style={{width:6,height:6,borderRadius:'50%',background:s.color,boxShadow:`0 0 5px ${s.color}`,flexShrink:0}}/>
                      <div>
                        <div style={{fontFamily:BASE.fm,fontSize:12,color:BASE.t1,fontWeight:500}}>{vm.name}</div>
                        <ServiceReadiness services={vm.services}/>
                      </div>
                    </div>
                  </td>
                  <td style={{padding:'11px 16px',fontSize:12,color:BASE.t3,fontFamily:BASE.fm}}>{vm.nodeId}</td>
                  <td style={{padding:'11px 16px',fontSize:12,color:BASE.t2}}>{vm.cpu} cores</td>
                  <td style={{padding:'11px 16px',fontSize:12,color:BASE.t2}}>{vm.mem>=1024?vm.mem/1024+' GB':vm.mem+' MB'}</td>
                  <td style={{padding:'11px 16px'}}><Badge text={s.text} color={s.color}/></td>
                </tr>
              );
            })}
          </tbody>
        </table>
        </div>
      </Card>
    </div>
  );
}

/* ━━ VIRTUAL MACHINES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
function VMsPage({ setModal, toast, hub }) {
  const { compact } = useTheme();
  const { vms, setVms, refresh } = useData();
  const [search, setSearch] = useState('');

  const shown = vms.filter(v => v.name.toLowerCase().includes(search.toLowerCase()));

  /* Real-time VM status updates via SignalR */
  useEffect(() => {
    if (!hub) return;
    const onStatus = (vmId, sc) => setVms(vs => vs.map(v => v.id===vmId ? {...v, sc} : v));
    const onCreate = (vm)        => setVms(vs => [...vs, normalizeVM ? {id:vm.id,name:vm.name,nodeId:vm.nodeId,cpu:vm.spec?.virtualCpuCores||0,mem:Math.round((vm.spec?.memoryBytes||0)/(1024*1024)),disk:Math.round((vm.spec?.diskBytes||0)/(1024*1024*1024)),sc:vm.status,tier:'Standard',tmpl:'',ip:vm.networkConfig?.privateIp||'pending',msgs:0,services:[],nodeHost:vm.networkConfig?.nodeAgentHost||'',nodePort:vm.networkConfig?.nodeAgentPort||5100} : vm]);
    const onDelete = (vmId)      => setVms(vs => vs.filter(v => v.id!==vmId));
    hub.on('VmStatusChanged', onStatus);
    hub.on('VmCreated',       onCreate);
    hub.on('VmDeleted',       onDelete);
    return () => { hub.off('VmStatusChanged'); hub.off('VmCreated'); hub.off('VmDeleted'); };
  }, [hub]);

  const toggleVm = id => setVms(vs => vs.map(v => v.id===id ? {...v, sc:v.sc===3?4:3} : v));
  const deleteVm = (id,name) => {
    setVms(vs => vs.filter(v => v.id!==id));
    toast(`VM "${name}" deleted.`,'info');
  };

  const bp = useBreakpoint();
  const pad = compact ? '16px' : bp.mobile ? '20px 16px' : bp.tablet ? '24px' : '32px 40px';
  return (
    <div style={{padding:pad, overflowY:'auto', flex:1}}>
      <PageHeader title="Virtual Machines" subtitle="Deploy, monitor and manage compute instances">
        <SBtn onClick={refresh}><Ico id="refresh" size={13}/></SBtn>
        <PBtn onClick={()=>setModal({type:'create-vm'})}><Ico id="plus" size={13} color="#0a0b0f"/> Create VM</PBtn>
      </PageHeader>
      <Card>
        <div style={{padding:'10px 14px',borderBottom:`1px solid ${BASE.bd}`}}>
          <div style={{display:'flex',alignItems:'center',gap:8,background:BASE.bg3,border:`1px solid ${BASE.bd}`,borderRadius:8,padding:'0 12px',height:33,maxWidth:360}}>
            <Ico id="search" size={13} color={BASE.t3}/>
            <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Filter VMs…"
              style={{background:'none',border:'none',outline:'none',fontSize:13,color:BASE.t1,flex:1}}/>
          </div>
        </div>
        <div style={{overflowX:'auto'}}>
        <table style={{width:'100%',borderCollapse:'collapse',minWidth:680}}>
          <THead cols={['Name / Services','Node','Specs','Tier','Status','Actions']}/>
          <tbody>
            {shown.map((vm,i)=>{
              const s = VM_STATUS[vm.sc]||VM_STATUS[4];
              return (
                <tr key={vm.id} style={{borderBottom:i<shown.length-1?`1px solid ${BASE.bd}`:'none'}}>
                  <td style={{padding:'11px 16px'}}>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <div style={{width:6,height:6,borderRadius:'50%',background:s.color,boxShadow:`0 0 4px ${s.color}`,flexShrink:0}}/>
                      <div>
                        <div style={{fontFamily:BASE.fm,fontSize:12,color:BASE.t1,fontWeight:500}}>{vm.name}</div>
                        <div style={{fontSize:10,color:BASE.t3,marginTop:1}}>{vm.tmpl}</div>
                        <ServiceReadiness services={vm.services}/>
                      </div>
                    </div>
                  </td>
                  <td style={{padding:'11px 16px',fontSize:12,color:BASE.t3,fontFamily:BASE.fm}}>{vm.nodeId}</td>
                  <td style={{padding:'11px 16px',fontSize:11,color:BASE.t2,whiteSpace:'nowrap'}}>{vm.cpu} vCPU · {vm.mem>=1024?vm.mem/1024+'GB':vm.mem+'MB'} · {vm.disk}GB</td>
                  <td style={{padding:'11px 16px',fontSize:11,color:BASE.t3}}>{vm.tier}</td>
                  <td style={{padding:'11px 16px'}}><Badge text={s.text} color={s.color}/></td>
                  <td style={{padding:'11px 16px'}}>
                    <div style={{display:'flex',gap:3}}>
                      <IBtn id="link"     title="Connection info"    onClick={()=>toast(`SSH: ssh user@${vm.sshHost||'<node>'} -p ${vm.sshPort||2222}`, 'info')}/>
                      <IBtn id="terminal" title="Open terminal"      onClick={()=>window.open(`${getOrchestratorUrl()}/terminal.html?nodeHost=${encodeURIComponent(vm.nodeHost||'')}&nodePort=${vm.nodePort||5100}&vmId=${vm.id}&vmIp=${encodeURIComponent(vm.ip||'')}`, '_blank','noopener')}/>
                      <IBtn id="folder"   title="File browser"       onClick={()=>window.open(`${getOrchestratorUrl()}/file-browser.html?nodeHost=${encodeURIComponent(vm.nodeHost||'')}&nodePort=${vm.nodePort||5100}&vmId=${vm.id}&vmIp=${encodeURIComponent(vm.ip||'')}`, '_blank','noopener')}/>
                      <IBtn id="globe"    title="Custom domains"     onClick={()=>setModal({type:'custom-domains',vm})}/>
                      <IBtn id="sliders"  title="Direct access"      onClick={()=>setModal({type:'direct-access',vm})}/>
                      <IBtn id={vm.sc===3?'pause':'play'} title={vm.sc===3?'Stop VM':'Start VM'} onClick={()=>toggleVm(vm.id)}/>
                      <IBtn id="message"  title={`Events (${vm.msgs})`} onClick={()=>setModal({type:'vm-events',vm})}/>
                      <IBtn id="trash"    title="Delete VM"          danger onClick={()=>deleteVm(vm.id,vm.name)}/>
                    </div>
                  </td>
                </tr>
              );
            })}
            {shown.length===0 && <tr><td colSpan={6} style={{padding:'40px',textAlign:'center',color:BASE.t3,fontSize:13}}>No VMs match your filter.</td></tr>}
          </tbody>
        </table>
        </div>
      </Card>
    </div>
  );
}

/* ━━ NODES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
function NodesPage({ setModal }) {
  const { compact } = useTheme();
  const { nodes, refresh } = useData();
  const bp = useBreakpoint();
  const pad = compact ? '16px' : bp.mobile ? '20px 16px' : bp.tablet ? '24px' : '32px 40px';
  return (
    <div style={{padding:pad, overflowY:'auto', flex:1}}>
      <PageHeader title="Nodes" subtitle="Explore the decentralised compute network">
        <SBtn onClick={refresh}><Ico id="refresh" size={13}/> Refresh</SBtn>
      </PageHeader>
      <p style={{fontSize:11,fontWeight:700,color:BASE.t3,textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:16}}>Featured Nodes</p>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(280px,1fr))',gap:16}}>
        {nodes.map(n=>(
          <div key={n.id} style={{background:BASE.bg2,border:`1px solid ${BASE.bd}`,borderRadius:14,padding:'20px 22px'}}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:14}}>
              <div>
                <div style={{display:'flex',alignItems:'center',gap:7,marginBottom:3}}>
                  <div style={{width:6,height:6,borderRadius:'50%',background:'#00d4aa',boxShadow:'0 0 5px #00d4aa'}}/>
                  <span style={{fontFamily:BASE.fm,fontSize:12,color:BASE.t1,fontWeight:600}}>{n.name}</span>
                </div>
                <span style={{fontSize:11,color:BASE.t3}}>{n.region}</span>
              </div>
              <div style={{display:'flex',gap:5,flexWrap:'wrap',justifyContent:'flex-end'}}>
                {n.gpu && <Badge text="GPU" color={BASE.a3}/>}
                <Badge text={n.uptime+'% uptime'} color={n.uptime>=99?'#00d4aa':BASE.aw}/>
              </div>
            </div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:10,marginBottom:16}}>
              {[['CPU',n.cpu+' cores'],['RAM',n.mem+' GB'],['Points',n.pts+' pts']].map(([l,v])=>(
                <div key={l} style={{background:BASE.bg3,borderRadius:8,padding:'8px 10px'}}>
                  <div style={{fontSize:9,color:BASE.t3,fontWeight:700,textTransform:'uppercase',letterSpacing:'0.07em',marginBottom:2}}>{l}</div>
                  <div style={{fontFamily:BASE.fm,fontSize:13,color:BASE.t1,fontWeight:600}}>{v}</div>
                </div>
              ))}
            </div>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
              <span style={{fontSize:11,color:BASE.t3}}>{n.vms} active VM{n.vms!==1?'s':''}</span>
              <PBtn sm onClick={()=>setModal({type:'create-vm'})}><Ico id="plus" size={12} color="#0a0b0f"/> Deploy VM</PBtn>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ━━ TEMPLATE MARKETPLACE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const TCATS = ['All','AI & ML','Databases','Dev Tools','Privacy'];

function TemplatesPage({ setModal }) {
  const { compact, accent } = useTheme();
  const [cat, setCat] = useState('All');
  const shown = MOCK_TEMPLATES.filter(t => cat==='All' || t.cat===cat);
  const catColor = c => ({ 'AI & ML':BASE.a3, Databases:BASE.a2, Privacy:BASE.aw })[c] || accent;
  const bp = useBreakpoint();
  const pad = compact ? '16px' : bp.mobile ? '20px 16px' : bp.tablet ? '24px' : '32px 40px';
  return (
    <div style={{padding:pad, overflowY:'auto', flex:1}}>
      <PageHeader title="Template Marketplace" subtitle="One-click deployable infrastructure templates"/>
      <div style={{display:'flex',gap:6,marginBottom:24,flexWrap:'wrap'}}>
        {TCATS.map(c=>(
          <button key={c} onClick={()=>setCat(c)}
            style={{padding:'6px 14px',borderRadius:20,border:`1px solid ${cat===c?accent+'44':BASE.bd}`,background:cat===c?`${accent}12`:'transparent',color:cat===c?accent:BASE.t2,fontSize:12,fontWeight:cat===c?600:400,cursor:'pointer'}}>
            {c}
          </button>
        ))}
      </div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(240px,1fr))',gap:16}}>
        {shown.map(t=>(
          <div key={t.id} style={{background:BASE.bg2,border:`1px solid ${BASE.bd}`,borderRadius:14,padding:'20px',display:'flex',flexDirection:'column',gap:12}}>
            <div>
              <div style={{fontSize:14,fontWeight:600,color:BASE.t1,marginBottom:6}}>{t.name}</div>
              <div style={{display:'flex',gap:5,flexWrap:'wrap'}}>
                <Badge text={t.cat} color={catColor(t.cat)}/>
                {t.gpu && <Badge text="GPU" color={BASE.a3}/>}
              </div>
            </div>
            <p style={{fontSize:12,color:BASE.t3,lineHeight:1.6,flex:1}}>{t.desc}</p>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
              <div style={{display:'flex',gap:10}}>
                <span style={{fontSize:11,color:BASE.t3}}>{t.deploys.toLocaleString()} deploys</span>
                <span style={{fontSize:11,color:BASE.aw}}>★ {t.rating}</span>
              </div>
              <PBtn sm onClick={()=>setModal({type:'create-vm'})}><Ico id="play" size={11} color="#0a0b0f"/> Deploy</PBtn>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ━━ MY TEMPLATES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
function MyTemplatesPage() {
  const { compact } = useTheme();
  return (
    <div style={{padding:compact?'20px 32px':'32px 40px', overflowY:'auto', flex:1}}>
      <PageHeader title="My Templates" subtitle="Your published infrastructure blueprints">
        <PBtn><Ico id="plus" size={13} color="#0a0b0f"/> Create Template</PBtn>
      </PageHeader>
      <EmptyState icon="my-templates" title="No templates yet"
        subtitle="Create a reusable cloud-init template and share it with the DeCloud community."
        action={<PBtn><Ico id="plus" size={13} color="#0a0b0f"/> Create Template</PBtn>}/>
    </div>
  );
}

/* ━━ SSH KEYS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
function SSHKeysPage({ setModal }) {
  const { compact } = useTheme();
  const { keys, setKeys } = useData();
  const bp = useBreakpoint();
  const pad = compact ? '16px' : bp.mobile ? '20px 16px' : bp.tablet ? '24px' : '32px 40px';
  return (
    <div style={{padding:pad, overflowY:'auto', flex:1}}>
      <PageHeader title="SSH Keys" subtitle="Secure access credentials for your instances">
        <PBtn onClick={()=>setModal({type:'add-ssh-key'})}><Ico id="plus" size={13} color="#0a0b0f"/> Add SSH Key</PBtn>
      </PageHeader>
      <Card>
        <div style={{overflowX:'auto'}}>
        <table style={{width:'100%',borderCollapse:'collapse',minWidth:400}}>
          <THead cols={['Name','Fingerprint','Added','']}/>
          <tbody>
            {keys.map((k,i)=>(
              <tr key={k.id} style={{borderBottom:i<keys.length-1?`1px solid ${BASE.bd}`:'none'}}>
                <td style={{padding:'12px 16px',fontSize:13,color:BASE.t1,fontWeight:500}}>{k.name}</td>
                <td style={{padding:'12px 16px',fontFamily:BASE.fm,fontSize:11,color:BASE.t3}}>{k.fp}</td>
                <td style={{padding:'12px 16px',fontSize:12,color:BASE.t3}}>{k.added}</td>
                <td style={{padding:'12px 16px'}}>
                  <IBtn id="trash" title="Delete key" danger onClick={()=>setKeys(ks=>ks.filter(x=>x.id!==k.id))}/>
                </td>
              </tr>
            ))}
            {keys.length===0 && (
              <tr><td colSpan={4} style={{padding:'40px',textAlign:'center',color:BASE.t3,fontSize:13}}>No SSH keys. Add one to connect to your VMs.</td></tr>
            )}
          </tbody>
        </table>
        </div>
      </Card>
    </div>
  );
}

/* ━━ SETTINGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
function SettingsPage({ toast, auth }) {
  const { compact } = useTheme();
  const [url, setUrl] = useState(localStorage.getItem('orchestratorUrl') || 'http://localhost:5000');
  const save = () => { localStorage.setItem('orchestratorUrl', url); toast('Settings saved.','success'); };
  const bp = useBreakpoint();
  const pad = compact ? '16px' : bp.mobile ? '20px 16px' : bp.tablet ? '24px' : '32px 40px';
  return (
    <div style={{padding:pad, overflowY:'auto', flex:1}}>
      <PageHeader title="Settings" subtitle="Platform configuration and security"/>
      <div style={{maxWidth:540}}>
        <Card style={{overflow:'visible'}}>
          <div style={{padding:'22px'}}>
            <div style={{marginBottom:16}}>
              <label style={{display:'block',fontSize:12,fontWeight:600,color:BASE.t2,marginBottom:5}}>Orchestrator URL</label>
              <input value={url} onChange={e=>setUrl(e.target.value)} style={inputSty}/>
              <p style={{fontSize:11,color:BASE.t3,marginTop:4}}>The backend API server for all DeCloud services.</p>
            </div>
            <div style={{marginBottom:20}}>
              <label style={{display:'block',fontSize:12,fontWeight:600,color:BASE.t2,marginBottom:5}}>Connected Wallet</label>
              <input readOnly value={auth?.wallet||''} style={{...inputSty,color:BASE.t3,fontFamily:BASE.fm,fontSize:11}}/>
              <p style={{fontSize:11,color:BASE.t3,marginTop:4}}>Your Ethereum wallet used for authentication.</p>
            </div>
            <div style={{display:'flex',gap:8}}>
              <PBtn onClick={save}><Ico id="check" size={13} color="#0a0b0f"/> Save Settings</PBtn>
              <SBtn onClick={auth?.logout}>Disconnect Wallet</SBtn>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

Object.assign(window, {
  useData,
  DashboardPage, VMsPage, NodesPage, TemplatesPage,
  MyTemplatesPage, SSHKeysPage, SettingsPage,
});
