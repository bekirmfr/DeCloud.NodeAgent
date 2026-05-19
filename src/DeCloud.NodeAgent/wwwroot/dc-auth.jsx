/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   dc-auth.jsx
   Auth state hook · MetaMask login · demo fallback
   Production: handleConnect → Reown AppKit (app.js)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const { useState } = React;
const { BASE, useTheme, Ico, inputSty, getOrchestratorUrl } = window;

/* ── Auth state ────────────────────────────────────────────── */
function useAuth() {
  const [state, setState] = useState(() => ({
    isLoggedIn : !!localStorage.getItem('authToken'),
    wallet     : localStorage.getItem('wallet') || '',
    balance    : 127.50,
    token      : localStorage.getItem('authToken') || '',
  }));

  const login = (wallet = '0x4a2f3c9e8b1d7a5f2c6e9b3d8a1f4c7e2b5d8a1f', token = null) => {
    const tok = token || 'demo_' + Date.now();
    localStorage.setItem('authToken', tok);
    localStorage.setItem('wallet', wallet);
    setState({ isLoggedIn:true, wallet, balance:127.50, token:tok });
  };

  const logout = () => {
    ['authToken','refreshToken','wallet','connectionType'].forEach(k => localStorage.removeItem(k));
    setState({ isLoggedIn:false, wallet:'', balance:0, token:'' });
  };

  const setBalance = b => setState(s => ({...s, balance:b}));

  return { ...state, login, logout, setBalance };
}

/* ── MetaMask authentication ───────────────────────────────── */
async function tryMetaMaskAuth(orchestratorUrl) {
  if (!window.ethereum) return null;

  // 1. Request wallet access
  const accounts = await window.ethereum
    .request({ method:'eth_requestAccounts' })
    .catch(() => null);
  if (!accounts?.length) return null;
  const walletAddress = accounts[0];

  // 2. Get message to sign from backend
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 6000);
  let msgData;
  try {
    const r = await fetch(
      `${orchestratorUrl}/api/auth/message?walletAddress=${walletAddress}`,
      { signal: ctrl.signal }
    );
    clearTimeout(timer);
    if (!r.ok) return null;          // backend unreachable → demo fallback
    msgData = await r.json();
    if (!msgData.success) return null;
  } catch { clearTimeout(timer); return null; }

  const { message, timestamp } = msgData.data;

  // 3. Sign message (MetaMask popup)
  const signature = await window.ethereum
    .request({ method:'personal_sign', params:[message, walletAddress] });

  // 4. Authenticate
  const authRes = await fetch(`${orchestratorUrl}/api/auth/wallet`, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ walletAddress, signature, message, timestamp }),
  }).catch(() => null);
  if (!authRes?.ok) return null;

  const authData = await authRes.json();
  if (!authData.success) return null;

  return {
    wallet : walletAddress,
    token  : authData.data.accessToken,
    refresh: authData.data.refreshToken,
  };
}

/* ── Login overlay ─────────────────────────────────────────── */
function LoginOverlay({ onLogin }) {
  const { accent } = useTheme();
  const [url,     setUrl]     = useState(localStorage.getItem('orchestratorUrl') || 'http://localhost:5000');
  const [loading, setLoading] = useState(false);
  const [status,  setStatus]  = useState('');
  const hasWallet = !!(typeof window !== 'undefined' && window.ethereum);

  const connect = async (forceDemo = false) => {
    localStorage.setItem('orchestratorUrl', url);
    setLoading(true);

    if (!forceDemo && hasWallet) {
      setStatus('Opening wallet…');
      try {
        const result = await tryMetaMaskAuth(url);
        if (result) {
          if (result.refresh) localStorage.setItem('refreshToken', result.refresh);
          setStatus('Authenticated ✓');
          setTimeout(() => onLogin(result.wallet, result.token), 300);
          return;
        }
        setStatus('Backend offline — switching to demo data');
        await new Promise(r => setTimeout(r, 700));
      } catch (e) {
        if (e.code === 4001 || e.message?.toLowerCase().includes('reject')) {
          setStatus('Signature rejected');
          setLoading(false);
          return;
        }
        setStatus('Wallet error — switching to demo');
        await new Promise(r => setTimeout(r, 700));
      }
    } else if (!forceDemo) {
      setStatus('No wallet — entering demo mode');
      await new Promise(r => setTimeout(r, 500));
    }

    onLogin();   // demo login
    setLoading(false);
  };

  return (
    <div style={{position:'fixed',inset:0,background:BASE.bg0,display:'flex',alignItems:'center',justifyContent:'center',zIndex:5000}}>
      {/* grid bg */}
      <div style={{position:'absolute',inset:0,backgroundImage:`linear-gradient(${BASE.bd} 1px,transparent 1px),linear-gradient(90deg,${BASE.bd} 1px,transparent 1px)`,backgroundSize:'32px 32px',opacity:0.5,pointerEvents:'none'}}/>
      {/* glow */}
      <div style={{position:'absolute',top:'10%',left:'15%',width:480,height:480,borderRadius:'50%',background:`radial-gradient(circle,${accent}0d 0%,transparent 70%)`,pointerEvents:'none'}}/>

      <div style={{position:'relative',background:BASE.bg2,border:`1px solid ${BASE.bh}`,borderRadius:20,padding:'clamp(24px,5vw,44px) clamp(20px,5vw,48px)',width:'min(460px,94vw)',boxShadow:'0 24px 64px rgba(0,0,0,0.5)'}}>

        {/* Logo */}
        <div style={{display:'flex',alignItems:'center',gap:12,justifyContent:'center',marginBottom:26}}>
          <div style={{width:38,height:38,borderRadius:10,background:`linear-gradient(135deg,${accent},${accent}cc)`,display:'flex',alignItems:'center',justifyContent:'center',fontSize:14,fontWeight:700,color:'#0a0b0f'}}>DC</div>
          <span style={{fontSize:22,fontWeight:700,color:BASE.t1,letterSpacing:'-0.03em'}}>DeCloud</span>
        </div>

        <h1 style={{fontSize:20,fontWeight:700,color:BASE.t1,textAlign:'center',letterSpacing:'-0.02em',marginBottom:8}}>Welcome to DeCloud</h1>
        <p style={{fontSize:13,color:BASE.t3,textAlign:'center',lineHeight:1.6,marginBottom:24}}>Enterprise-grade decentralised compute at your fingertips</p>

        {/* Orchestrator URL */}
        <div style={{marginBottom:20}}>
          <label style={{display:'block',fontSize:12,fontWeight:600,color:BASE.t2,marginBottom:6}}>Orchestrator URL</label>
          <input value={url} onChange={e=>setUrl(e.target.value)} style={inputSty} placeholder="http://localhost:5000"/>
          <p style={{fontSize:11,color:BASE.t3,marginTop:4}}>Point to your running backend, or leave default for local dev</p>
        </div>

        {/* Primary button */}
        <button onClick={()=>connect(false)} disabled={loading}
          style={{width:'100%',padding:'13px',background:`linear-gradient(135deg,${accent},${accent}cc)`,border:'none',borderRadius:10,color:'#0a0b0f',fontSize:14,fontWeight:700,cursor:loading?'default':'pointer',fontFamily:BASE.ff,display:'flex',alignItems:'center',justifyContent:'center',gap:9,marginBottom:10,boxShadow:`0 6px 24px ${accent}2a`,opacity:loading?0.85:1}}>
          <Ico id="wallet" size={16} color="#0a0b0f"/>
          {loading
            ? (status || 'Connecting…')
            : hasWallet ? 'Connect with MetaMask' : 'Connect Wallet'
          }
        </button>

        {/* Demo fallback */}
        {!loading && (
          <button onClick={()=>connect(true)}
            style={{width:'100%',padding:'9px',background:'transparent',border:`1px solid ${BASE.bd}`,borderRadius:10,color:BASE.t3,fontSize:12,cursor:'pointer',fontFamily:BASE.ff,marginBottom:20}}>
            {hasWallet ? 'Skip — continue in demo mode' : 'Continue in demo mode (no wallet)'}
          </button>
        )}
        {loading && status && (
          <p style={{textAlign:'center',fontSize:12,color:BASE.t3,marginBottom:16,lineHeight:1.5}}>{status}</p>
        )}

        {/* Security badges */}
        <div style={{display:'flex',gap:20,justifyContent:'center',marginBottom:18}}>
          {[['shield','End-to-end encrypted'],['check','Wallet-verified identity']].map(([icon,text])=>(
            <div key={text} style={{display:'flex',alignItems:'center',gap:5,fontSize:11,color:BASE.t3}}>
              <Ico id={icon} size={12} color={BASE.t3}/>{text}
            </div>
          ))}
        </div>

        <p style={{fontSize:10,color:BASE.t3,textAlign:'center',lineHeight:1.6,borderTop:`1px solid ${BASE.bd}`,paddingTop:14}}>
          {hasWallet
            ? 'MetaMask detected — real auth available. Backend must be running at the URL above.'
            : 'No wallet detected. Install MetaMask for full auth. Demo mode uses mock data.'
          }
        </p>
      </div>
    </div>
  );
}

Object.assign(window, { useAuth, LoginOverlay });
