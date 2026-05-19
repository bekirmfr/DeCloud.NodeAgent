/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   dc-shell.jsx
   CommandBar · Sidebar · TweaksPanel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const { useState, useEffect } = React;
const { BASE, useTheme, useBreakpoint, NAV_GROUPS, Ico, Logo } = window;

const ACCENTS = ['#00d4aa','#00a8ff','#8b5cf6','#f59e0b'];

/* ── Command Bar ───────────────────────────────────────────── */
function CommandBar({ auth, setModal, sidebarOpen, setSidebarOpen, rtStatus='idle' }) {
  const { accent } = useTheme();
  const bp = useBreakpoint();
  const [search, setSearch] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);

  const short = auth.wallet
    ? `${auth.wallet.slice(0,6)}…${auth.wallet.slice(-4)}`
    : 'Not connected';

  return (
    <div style={{height:54,flexShrink:0,background:BASE.bg1,borderBottom:`1px solid ${BASE.bd}`,display:'flex',alignItems:'center',padding:`0 ${bp.mobile?14:28}px`,gap:bp.mobile?8:14,zIndex:10}}>

      {/* Hamburger (mobile + tablet) */}
      {!bp.desktop && (
        <button onClick={()=>setSidebarOpen(o=>!o)}
          style={{width:32,height:32,display:'flex',alignItems:'center',justifyContent:'center',background:'transparent',border:`1px solid ${BASE.bd}`,borderRadius:8,cursor:'pointer',flexShrink:0}}>
          <Ico id={sidebarOpen?'x':'menu'} size={16} color={BASE.t2}/>
        </button>
      )}

      <Logo/>

      {/* Divider — desktop only */}
      {bp.desktop && <div style={{width:1,height:22,background:BASE.bd,flexShrink:0,margin:'0 4px'}}/>}

      {/* API / RT status pill — desktop only */}
      {bp.desktop && (() => {
        const live = rtStatus === 'connected';
        const busy = rtStatus === 'connecting';
        const col  = live ? '#00d4aa' : busy ? BASE.aw : BASE.t3;
        return (
          <div style={{display:'flex',alignItems:'center',gap:5,padding:'3px 9px',background:BASE.bg3,border:`1px solid ${live?'#00d4aa30':BASE.bd}`,borderRadius:20,flexShrink:0}}>
            <div style={{width:5,height:5,borderRadius:'50%',background:col,boxShadow:live?`0 0 4px ${col}`:'none'}}/>
            <span style={{fontSize:10,fontWeight:600,color:col,fontFamily:BASE.fm}}>
              {live?'Live':busy?'•••':'Demo'}
            </span>
          </div>
        );
      })()}

      {/* Search — full on desktop/tablet, icon toggle on mobile */}
      {bp.desktop || bp.tablet ? (
        <div style={{flex:1,maxWidth:520,display:'flex',alignItems:'center',gap:9,background:BASE.bg3,border:`1px solid ${BASE.bd}`,borderRadius:8,padding:'0 14px',height:33}}>
          <Ico id="search" size={13} color={BASE.t3}/>
          <input value={search} onChange={e=>setSearch(e.target.value)}
            placeholder="Search VMs, nodes, templates…"
            style={{background:'none',border:'none',outline:'none',fontSize:13,color:BASE.t1,flex:1,minWidth:0}}/>
          {bp.desktop && <span style={{fontSize:10,color:BASE.t3,background:BASE.bg5,padding:'1px 6px',borderRadius:4,border:`1px solid ${BASE.bd}`,fontFamily:BASE.fm,flexShrink:0}}>⌘K</span>}
        </div>
      ) : (
        <button onClick={()=>setSearchOpen(o=>!o)}
          style={{width:32,height:32,display:'flex',alignItems:'center',justifyContent:'center',background:BASE.bg3,border:`1px solid ${BASE.bd}`,borderRadius:8,cursor:'pointer',flexShrink:0}}>
          <Ico id="search" size={14} color={BASE.t3}/>
        </button>
      )}

      {/* Right controls */}
      <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:8}}>
        {/* Bell — hidden on mobile */}
        {!bp.mobile && (
          <div style={{position:'relative',width:32,height:32,display:'flex',alignItems:'center',justifyContent:'center',background:BASE.bg3,border:`1px solid ${BASE.bd}`,borderRadius:8,cursor:'pointer'}}>
            <Ico id="bell" size={14} color={BASE.t3}/>
            <div style={{position:'absolute',top:5,right:5,width:5,height:5,borderRadius:'50%',background:BASE.ad,border:`1.5px solid ${BASE.bg1}`}}/>
          </div>
        )}

        {/* Balance */}
        <div onClick={()=>setModal({type:'payment'})}
          style={{display:'flex',alignItems:'center',gap:6,padding:'5px 10px',background:BASE.bg3,border:`1px solid ${BASE.bd}`,borderRadius:8,cursor:'pointer'}}>
          <Ico id="wallet" size={13} color={accent}/>
          <span style={{fontFamily:BASE.fm,fontSize:12,color:BASE.t1,fontWeight:600}}>{(auth.balance||0).toFixed(2)}</span>
          {!bp.mobile && <span style={{fontSize:11,color:accent}}>USDC</span>}
        </div>

        {/* Wallet address — tablet + desktop only */}
        {bp.desktop && (
          <div onClick={auth.logout} title="Click to disconnect"
            style={{display:'flex',alignItems:'center',gap:7,padding:'5px 12px',background:BASE.bg3,border:`1px solid ${BASE.bd}`,borderRadius:8,cursor:'pointer'}}>
            <div style={{width:6,height:6,borderRadius:'50%',background:accent,boxShadow:`0 0 5px ${accent}`}}/>
            <span style={{fontFamily:BASE.fm,fontSize:11,color:BASE.t2}}>{short}</span>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Sidebar ───────────────────────────────────────────────── */
function Sidebar({ page, navigate, open, onClose }) {
  const { accent } = useTheme();
  const bp = useBreakpoint();

  const navItems = (
    <div style={{padding:'10px 8px'}}>
      {NAV_GROUPS.map(g=>(
        <div key={g.label}>
          <div style={{fontSize:10,fontWeight:700,color:BASE.t3,letterSpacing:'0.1em',textTransform:'uppercase',padding:'10px 10px 5px'}}>{g.label}</div>
          {g.items.map(item=>{
            const active = item.id===page;
            return (
              <div key={item.id} onClick={()=>navigate(item.id)}
                style={{display:'flex',alignItems:'center',gap:9,padding:'9px 10px',borderRadius:8,marginBottom:2,cursor:'pointer',background:active?`${accent}14`:'transparent',color:active?BASE.t1:BASE.t2,fontSize:13,fontWeight:active?600:400,transition:'background 120ms ease'}}>
                <Ico id={item.id} size={14} color={active?accent:BASE.t3}/>
                {item.label}
                {active && <div style={{width:5,height:5,borderRadius:'50%',background:accent,marginLeft:'auto',boxShadow:`0 0 4px ${accent}`}}/>}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );

  if (bp.desktop) {
    return (
      <div style={{width:216,flexShrink:0,background:BASE.bg1,borderRight:`1px solid ${BASE.bd}`,overflowY:'auto'}}>
        {navItems}
      </div>
    );
  }

  return (
    <>
      {open && (
        <div onClick={onClose}
          style={{position:'fixed',inset:0,top:54,background:'rgba(0,0,0,0.55)',zIndex:498,backdropFilter:'blur(2px)'}}/>
      )}
      <div style={{position:'fixed',top:54,bottom:0,left:0,width:260,background:BASE.bg1,borderRight:`1px solid ${BASE.bd}`,overflowY:'auto',zIndex:499,transform:open?'translateX(0)':'translateX(-100%)',transition:'transform 240ms cubic-bezier(0.16,1,0.3,1)'}}>
        {navItems}
      </div>
    </>
  );
}

/* ── Tweaks Panel ──────────────────────────────────────────── */
function TweaksPanel({ tweaks, setTweaks, onClose }) {
  const set = (k, v) => {
    const next = {...tweaks, [k]:v};
    setTweaks(next);
    window.parent.postMessage({ type:'__edit_mode_set_keys', edits:{[k]:v} }, '*');
  };

  return (
    <div style={{position:'fixed',bottom:24,right:24,width:224,background:BASE.bg4,border:`1px solid ${BASE.bh}`,borderRadius:14,boxShadow:'0 8px 32px rgba(0,0,0,0.4)',zIndex:3000,padding:'16px'}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
        <span style={{fontSize:13,fontWeight:700,color:BASE.t1}}>Tweaks</span>
        <button onClick={onClose} style={{background:'none',border:'none',cursor:'pointer',display:'flex',padding:0}}>
          <Ico id="x" size={14} color={BASE.t3}/>
        </button>
      </div>

      <div style={{marginBottom:16}}>
        <div style={{fontSize:11,fontWeight:600,color:BASE.t3,textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:8}}>Accent Color</div>
        <div style={{display:'flex',gap:8}}>
          {ACCENTS.map(a=>(
            <div key={a} onClick={()=>set('accent',a)}
              style={{width:28,height:28,borderRadius:7,background:a,cursor:'pointer',border:`2px solid ${tweaks.accent===a?'#fff':'transparent'}`,boxShadow:tweaks.accent===a?`0 0 0 1px ${a}`:'none',flexShrink:0}}/>
          ))}
        </div>
      </div>

      <div>
        <div style={{fontSize:11,fontWeight:600,color:BASE.t3,textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:8}}>Density</div>
        <div style={{display:'flex',gap:6}}>
          {[['Spacious',false],['Compact',true]].map(([l,v])=>(
            <button key={l} onClick={()=>set('compact',v)}
              style={{flex:1,padding:'6px',borderRadius:7,border:`1px solid ${tweaks.compact===v?tweaks.accent+'44':BASE.bd}`,background:tweaks.compact===v?`${tweaks.accent}14`:'transparent',color:tweaks.compact===v?tweaks.accent:BASE.t2,fontSize:11,fontWeight:tweaks.compact===v?600:400,cursor:'pointer'}}>
              {l}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { CommandBar, Sidebar, TweaksPanel });
