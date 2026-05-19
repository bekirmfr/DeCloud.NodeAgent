/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   dc-atoms.jsx
   Shared UI primitives — icons, buttons, cards,
   modals, toasts, service-readiness dots, etc.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const { useContext } = React;
const { BASE, ThemeCtx, useTheme, useBreakpoint, IP } = window;

/* ── Icon ──────────────────────────────────────────────────── */
function Ico({ id, size=16, color='currentColor', sw=1.75 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
      <path d={IP[id]}/>
    </svg>
  );
}

/* ── Logo ──────────────────────────────────────────────────── */
function Logo() {
  const { accent } = useTheme();
  return (
    <div style={{display:'flex',alignItems:'center',gap:9}}>
      <div style={{width:28,height:28,borderRadius:7,background:`linear-gradient(135deg,${accent},${accent}cc)`,display:'flex',alignItems:'center',justifyContent:'center',fontSize:10,fontWeight:700,color:'#0a0b0f',letterSpacing:'-0.02em',flexShrink:0}}>DC</div>
      <span style={{fontSize:15,fontWeight:700,color:BASE.t1,letterSpacing:'-0.02em'}}>DeCloud</span>
    </div>
  );
}

/* ── Buttons ───────────────────────────────────────────────── */
function PBtn({ children, onClick, sm, disabled }) {
  const { accent } = useTheme();
  return (
    <button onClick={onClick} disabled={disabled} style={{display:'flex',alignItems:'center',gap:6,padding:sm?'6px 13px':'9px 18px',background:disabled?BASE.bg4:`linear-gradient(135deg,${accent},${accent}cc)`,border:'none',borderRadius:8,color:disabled?BASE.t3:'#0a0b0f',fontSize:sm?11:13,fontWeight:700,cursor:disabled?'not-allowed':'pointer',fontFamily:BASE.ff,boxShadow:disabled?'none':`0 4px 14px ${accent}28`,whiteSpace:'nowrap',flexShrink:0}}>
      {children}
    </button>
  );
}

function SBtn({ children, onClick, danger }) {
  return (
    <button onClick={onClick} style={{display:'flex',alignItems:'center',gap:6,padding:'7px 14px',background:danger?`${BASE.ad}14`:BASE.bg3,border:`1px solid ${danger?BASE.ad+'40':BASE.bd}`,borderRadius:8,color:danger?BASE.ad:BASE.t2,fontSize:12,cursor:'pointer',fontFamily:BASE.ff,whiteSpace:'nowrap',flexShrink:0}}>
      {children}
    </button>
  );
}

function IBtn({ id, title, danger, onClick }) {
  return (
    <button title={title} onClick={onClick} style={{width:30,height:30,display:'flex',alignItems:'center',justifyContent:'center',background:'transparent',border:`1px solid ${BASE.bd}`,borderRadius:7,cursor:'pointer',flexShrink:0}}>
      <Ico id={id} size={13} color={danger?BASE.ad:BASE.t3}/>
    </button>
  );
}

/* ── Badge ─────────────────────────────────────────────────── */
function Badge({ text, color }) {
  return (
    <span style={{fontSize:10,fontWeight:600,padding:'2px 8px',borderRadius:20,background:`${color}18`,color,border:`1px solid ${color}30`,whiteSpace:'nowrap'}}>
      {text}
    </span>
  );
}

/* ── Service readiness dots ────────────────────────────────── */
const SVC_COLORS = { Ready:'#00d4aa', Checking:'#00a8ff', Pending:'#6b7280', TimedOut:'#f59e0b', Failed:'#ef4444' };

function ServiceReadiness({ services }) {
  if (!services?.length) return null;
  return (
    <div style={{display:'flex',gap:5,marginTop:4}}>
      {services.map(s => {
        const c = SVC_COLORS[s.status] || '#6b7280';
        return (
          <div key={s.name} title={`${s.name}: ${s.status}`}
               style={{width:6,height:6,borderRadius:'50%',background:c,boxShadow:s.status==='Ready'?`0 0 4px ${c}`:'none',flexShrink:0}}/>
        );
      })}
    </div>
  );
}

/* ── Stat Card ─────────────────────────────────────────────── */
function StatCard({ label, val, sub, color, icon }) {
  return (
    <div style={{background:BASE.bg2,border:`1px solid ${BASE.bd}`,borderRadius:14,padding:'20px 22px',display:'flex',flexDirection:'column',gap:10}}>
      <div style={{width:34,height:34,borderRadius:9,background:`${color}14`,display:'flex',alignItems:'center',justifyContent:'center'}}>
        <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          <path d={icon}/>
        </svg>
      </div>
      <div style={{fontFamily:BASE.fm,fontSize:26,fontWeight:600,color:BASE.t1,lineHeight:1}}>{val}</div>
      <div>
        <div style={{fontSize:13,color:BASE.t2,fontWeight:500}}>{label}</div>
        <div style={{fontSize:11,color:BASE.t3,marginTop:2}}>{sub}</div>
      </div>
    </div>
  );
}

/* ── Page Header ───────────────────────────────────────────── */
function PageHeader({ title, subtitle, children }) {
  const { compact } = useTheme();
  return (
    <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:compact?16:26,flexShrink:0}}>
      <div>
        <h1 style={{fontSize:22,fontWeight:700,color:BASE.t1,letterSpacing:'-0.02em'}}>{title}</h1>
        {subtitle && <p style={{fontSize:13,color:BASE.t3,marginTop:5}}>{subtitle}</p>}
      </div>
      {children && <div style={{display:'flex',gap:8,alignItems:'center'}}>{children}</div>}
    </div>
  );
}

/* ── Table Head ────────────────────────────────────────────── */
function THead({ cols }) {
  return (
    <thead>
      <tr style={{borderBottom:`1px solid ${BASE.bd}`}}>
        {cols.map(c => (
          <th key={c} style={{padding:'9px 16px',textAlign:'left',fontSize:10,fontWeight:700,color:BASE.t3,textTransform:'uppercase',letterSpacing:'0.07em',whiteSpace:'nowrap'}}>
            {c}
          </th>
        ))}
      </tr>
    </thead>
  );
}

/* ── Cards ─────────────────────────────────────────────────── */
function Card({ children, style={} }) {
  return <div style={{background:BASE.bg2,border:`1px solid ${BASE.bd}`,borderRadius:14,overflow:'hidden',...style}}>{children}</div>;
}

function CardHead({ title, action }) {
  return (
    <div style={{padding:'14px 18px',borderBottom:`1px solid ${BASE.bd}`,display:'flex',justifyContent:'space-between',alignItems:'center'}}>
      <span style={{fontSize:14,fontWeight:600,color:BASE.t1}}>{title}</span>
      {action}
    </div>
  );
}

/* ── Modal wrapper ─────────────────────────────────────────── */
function Modal({ title, onClose, footer, children, wide=false }) {
  const bp = useBreakpoint();
  const maxW = wide ? (bp.mobile ? '96vw' : 680) : (bp.mobile ? '96vw' : 520);
  return (
    <div style={{position:'fixed',inset:0,background:'rgba(0,0,0,0.72)',display:'flex',alignItems:bp.mobile?'flex-end':'center',justifyContent:'center',zIndex:2000,backdropFilter:'blur(4px)'}}>
      <div style={{background:BASE.bg2,border:`1px solid ${BASE.bh}`,borderRadius:bp.mobile?'16px 16px 0 0':16,width:maxW,maxHeight:bp.mobile?'90vh':'88vh',display:'flex',flexDirection:'column',boxShadow:'0 20px 60px rgba(0,0,0,0.5)'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'18px 22px',borderBottom:`1px solid ${BASE.bd}`}}>
          <span style={{fontSize:16,fontWeight:700,color:BASE.t1}}>{title}</span>
          <button onClick={onClose} style={{background:'none',border:'none',cursor:'pointer',display:'flex',padding:0}}>
            <Ico id="x" size={18} color={BASE.t3}/>
          </button>
        </div>
        <div style={{flex:1,overflowY:'auto',padding:'20px 22px'}}>{children}</div>
        {footer && <div style={{padding:'14px 22px',borderTop:`1px solid ${BASE.bd}`,display:'flex',justifyContent:'flex-end',gap:8}}>{footer}</div>}
      </div>
    </div>
  );
}

/* ── Form helpers ──────────────────────────────────────────── */
function FG({ label, hint, children }) {
  return (
    <div style={{marginBottom:14}}>
      <label style={{display:'block',fontSize:12,fontWeight:600,color:BASE.t2,marginBottom:5}}>{label}</label>
      {children}
      {hint && <p style={{fontSize:11,color:BASE.t3,marginTop:3}}>{hint}</p>}
    </div>
  );
}

const inputSty = {
  width:'100%', padding:'10px 14px',
  background:BASE.bg3, border:`1px solid ${BASE.bd}`,
  borderRadius:8, color:BASE.t1, fontSize:13, outline:'none', fontFamily:'inherit',
};

/* ── Toast stack ───────────────────────────────────────────── */
function Toasts({ toasts, dismiss }) {
  return (
    <div style={{position:'fixed',bottom:24,right:24,display:'flex',flexDirection:'column',gap:8,zIndex:9000,pointerEvents:'none'}}>
      {toasts.map(t => (
        <div key={t.id} style={{display:'flex',alignItems:'center',gap:10,padding:'10px 16px',background:BASE.bg4,border:`1px solid ${t.type==='success'?'#00d4aa44':t.type==='error'?BASE.ad+'44':BASE.bd}`,borderRadius:10,boxShadow:'0 4px 24px rgba(0,0,0,0.4)',pointerEvents:'all',minWidth:260,maxWidth:360}}>
          <Ico id={t.type==='success'?'check':t.type==='error'?'x':'bell'} size={14} color={t.type==='success'?'#00d4aa':t.type==='error'?BASE.ad:BASE.a2}/>
          <span style={{fontSize:13,color:BASE.t1,flex:1}}>{t.msg}</span>
          <button onClick={()=>dismiss(t.id)} style={{background:'none',border:'none',cursor:'pointer',display:'flex',padding:0}}>
            <Ico id="x" size={12} color={BASE.t3}/>
          </button>
        </div>
      ))}
    </div>
  );
}

/* ── Empty state ───────────────────────────────────────────── */
function EmptyState({ icon, title, subtitle, action }) {
  return (
    <div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',padding:'80px 0',textAlign:'center'}}>
      <div style={{width:52,height:52,borderRadius:12,background:BASE.bg3,display:'flex',alignItems:'center',justifyContent:'center',marginBottom:16}}>
        <Ico id={icon} size={22} color={BASE.t3}/>
      </div>
      <div style={{fontSize:15,fontWeight:600,color:BASE.t2,marginBottom:6}}>{title}</div>
      {subtitle && <p style={{fontSize:13,color:BASE.t3,maxWidth:320,lineHeight:1.6}}>{subtitle}</p>}
      {action && <div style={{marginTop:20}}>{action}</div>}
    </div>
  );
}

Object.assign(window, {
  Ico, Logo, PBtn, SBtn, IBtn, Badge,
  ServiceReadiness, SVC_COLORS,
  StatCard, PageHeader, THead, Card, CardHead,
  Modal, FG, inputSty, Toasts, EmptyState,
});
