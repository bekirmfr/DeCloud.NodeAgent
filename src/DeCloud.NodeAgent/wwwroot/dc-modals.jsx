/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   dc-modals.jsx
   CreateVM · AddSSHKey · DirectAccess ·
   CustomDomains · Payment · VmEvents
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const { useState } = React;
const { BASE, useTheme, Ico, PBtn, SBtn, IBtn, Badge, Modal, FG, inputSty, THead } = window;

/* ── Create VM ─────────────────────────────────────────────── */
function CreateVMModal({ onClose, toast }) {
  const [name,setName]=useState('');
  const [os,setOs]=useState('ubuntu-24.04');
  const [cpu,setCpu]=useState('2');
  const [mem,setMem]=useState('2048');
  const [tier,setTier]=useState('1');

  const submit = () => {
    if (!name.trim()) return;
    toast(`VM "${name.trim()}" queued — provisioning shortly.`, 'success');
    onClose();
  };

  return (
    <Modal title="Create Virtual Machine" onClose={onClose}
      footer={<><SBtn onClick={onClose}>Cancel</SBtn><PBtn onClick={submit}><Ico id="plus" size={13} color="#0a0b0f"/> Create VM</PBtn></>}>
      <FG label="VM Name" hint="Lowercase letters, numbers, hyphens. A unique suffix is auto-appended.">
        <input value={name} onChange={e=>setName(e.target.value)} placeholder="my-vm" style={inputSty}/>
      </FG>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
        <FG label="CPU Cores">
          <select value={cpu} onChange={e=>setCpu(e.target.value)} style={inputSty}>
            {[1,2,4,8,16].map(n=><option key={n} value={n}>{n} cores</option>)}
          </select>
        </FG>
        <FG label="Memory">
          <select value={mem} onChange={e=>setMem(e.target.value)} style={inputSty}>
            {[512,1024,2048,4096,8192,16384].map(m=><option key={m} value={m}>{m>=1024?m/1024+' GB':m+' MB'}</option>)}
          </select>
        </FG>
      </div>
      <FG label="Operating System">
        <select value={os} onChange={e=>setOs(e.target.value)} style={inputSty}>
          <option value="ubuntu-24.04">Ubuntu 24.04 LTS</option>
          <option value="ubuntu-22.04">Ubuntu 22.04 LTS</option>
          <option value="debian-12">Debian 12</option>
        </select>
      </FG>
      <FG label="Quality Tier">
        <select value={tier} onChange={e=>setTier(e.target.value)} style={inputSty}>
          <option value="0">Guaranteed — dedicated 1:1 resources</option>
          <option value="1">Standard — 1.5:1 overcommit</option>
          <option value="2">Balanced — 2:1 overcommit</option>
          <option value="3">Burstable — 4:1, lowest cost</option>
        </select>
      </FG>
      <div style={{background:BASE.bg3,borderRadius:9,padding:'11px 14px',display:'flex',justifyContent:'space-between'}}>
        <span style={{fontSize:12,color:BASE.t3}}>Estimated cost</span>
        <span style={{fontFamily:BASE.fm,fontSize:13,color:BASE.t1,fontWeight:600}}>~$0.02 / hour</span>
      </div>
    </Modal>
  );
}

/* ── Add SSH Key ───────────────────────────────────────────── */
function AddSSHKeyModal({ onClose, toast }) {
  const [name,setName]=useState('');
  const [key,setKey]=useState('');

  const submit = () => {
    if (!name.trim() || !key.trim()) return;
    toast(`SSH key "${name.trim()}" added.`, 'success');
    onClose();
  };

  return (
    <Modal title="Add SSH Key" onClose={onClose}
      footer={<><SBtn onClick={onClose}>Cancel</SBtn><PBtn onClick={submit}><Ico id="plus" size={13} color="#0a0b0f"/> Add Key</PBtn></>}>
      <FG label="Key Name" hint="A friendly label for this key.">
        <input value={name} onChange={e=>setName(e.target.value)} placeholder="my-laptop" style={inputSty}/>
      </FG>
      <FG label="Public Key" hint="Paste the contents of ~/.ssh/id_rsa.pub or equivalent.">
        <textarea value={key} onChange={e=>setKey(e.target.value)} placeholder="ssh-rsa AAAAB3..." rows={5}
          style={{...inputSty,resize:'vertical',fontFamily:BASE.fm,fontSize:11}}/>
      </FG>
    </Modal>
  );
}

/* ── Direct Access (port forwarding) ───────────────────────── */
const INIT_PORTS = [
  { id:'pm1', vmPort:22,   publicPort:40022, proto:'TCP', label:'SSH' },
  { id:'pm2', vmPort:5432, publicPort:40432, proto:'TCP', label:'Postgres' },
];

function DirectAccessModal({ vm, onClose, toast }) {
  const [ports,setPorts]=useState(INIT_PORTS);
  const [vmPort,setVmPort]=useState('');
  const [proto,setProto]=useState('TCP');
  const [label,setLabel]=useState('');

  const addPort = () => {
    if (!vmPort) return;
    const pub = 40000 + Math.floor(Math.random()*20000);
    setPorts(ps=>[...ps,{id:'pm'+Date.now(),vmPort:+vmPort,publicPort:pub,proto,label:label||`Port ${vmPort}`}]);
    setVmPort(''); setLabel('');
    toast(`Port ${vmPort}→${pub} (${proto}) allocated.`, 'success');
  };

  const removePort = id => setPorts(ps=>ps.filter(p=>p.id!==id));

  return (
    <Modal title={`Direct Access — ${vm.name}`} onClose={onClose} wide footer={<SBtn onClick={onClose}>Close</SBtn>}>
      <p style={{fontSize:13,color:BASE.t3,marginBottom:16,lineHeight:1.6}}>
        Expose VM ports to the internet via iptables DNAT. CGNAT nodes route through their relay automatically.
      </p>

      {ports.length > 0 && (
        <div style={{marginBottom:20,background:BASE.bg3,borderRadius:10,overflow:'hidden',border:`1px solid ${BASE.bd}`}}>
          <table style={{width:'100%',borderCollapse:'collapse'}}>
            <THead cols={['VM Port','Public Port','Protocol','Label','']}/>
            <tbody>
              {ports.map((p,i)=>(
                <tr key={p.id} style={{borderBottom:i<ports.length-1?`1px solid ${BASE.bd}`:'none'}}>
                  <td style={{padding:'9px 14px',fontFamily:BASE.fm,fontSize:12,color:BASE.t2}}>{p.vmPort}</td>
                  <td style={{padding:'9px 14px',fontFamily:BASE.fm,fontSize:12,color:BASE.a2,fontWeight:600}}>{p.publicPort}</td>
                  <td style={{padding:'9px 14px',fontSize:12,color:BASE.t2}}>{p.proto}</td>
                  <td style={{padding:'9px 14px',fontSize:12,color:BASE.t3}}>{p.label}</td>
                  <td style={{padding:'9px 14px'}}><IBtn id="trash" title="Remove" danger onClick={()=>removePort(p.id)}/></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div style={{background:BASE.bg3,borderRadius:10,padding:'16px',border:`1px solid ${BASE.bd}`}}>
        <p style={{fontSize:12,fontWeight:600,color:BASE.t2,marginBottom:12}}>Add Port Mapping</p>
        <div style={{display:'grid',gridTemplateColumns:'1fr 110px 1fr auto',gap:8,alignItems:'end'}}>
          <FG label="VM Port">
            <input value={vmPort} onChange={e=>setVmPort(e.target.value)} type="number" placeholder="8080" style={inputSty}/>
          </FG>
          <FG label="Protocol">
            <select value={proto} onChange={e=>setProto(e.target.value)} style={inputSty}>
              <option>TCP</option><option>UDP</option><option>Both</option>
            </select>
          </FG>
          <FG label="Label">
            <input value={label} onChange={e=>setLabel(e.target.value)} placeholder="Web App" style={inputSty}/>
          </FG>
          <div style={{marginBottom:14}}>
            <PBtn onClick={addPort}><Ico id="plus" size={13} color="#0a0b0f"/> Add</PBtn>
          </div>
        </div>
      </div>
    </Modal>
  );
}

/* ── Custom Domains ────────────────────────────────────────── */
const INIT_DOMAINS = [
  { id:'d1', domain:'app.example.com', target:'vm-abc123.vms.stackfi.tech', active:true },
];

function CustomDomainsModal({ vm, onClose, toast }) {
  const [domains,setDomains]=useState(INIT_DOMAINS);
  const [input,setInput]=useState('');

  const addDomain = () => {
    if (!input.trim()) return;
    setDomains(ds=>[...ds,{id:'d'+Date.now(),domain:input.trim(),target:`${vm.id}.vms.decloud.tech`,active:false}]);
    toast(`Domain added. Point CNAME to activate.`,'info');
    setInput('');
  };

  return (
    <Modal title={`Custom Domains — ${vm.name}`} onClose={onClose} wide footer={<SBtn onClick={onClose}>Close</SBtn>}>
      <div style={{background:`${BASE.a2}10`,border:`1px solid ${BASE.a2}28`,borderRadius:10,padding:'12px 16px',marginBottom:20}}>
        <p style={{fontSize:11,color:BASE.a2,fontWeight:600,marginBottom:4}}>Default subdomain (always active)</p>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <code style={{fontFamily:BASE.fm,fontSize:12,color:BASE.t1,flex:1}}>{vm.id}.vms.decloud.tech</code>
          <IBtn id="copy" title="Copy" onClick={()=>navigator.clipboard?.writeText(`${vm.id}.vms.decloud.tech`)}/>
        </div>
      </div>

      {domains.map(d=>(
        <div key={d.id} style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',background:BASE.bg3,borderRadius:9,marginBottom:8,border:`1px solid ${BASE.bd}`}}>
          <Ico id="globe" size={14} color={d.active?'#00d4aa':BASE.t3}/>
          <div style={{flex:1}}>
            <div style={{fontSize:13,color:BASE.t1,fontWeight:500}}>{d.domain}</div>
            <div style={{fontSize:11,color:BASE.t3,fontFamily:BASE.fm,marginTop:2}}>CNAME → {d.target}</div>
          </div>
          <Badge text={d.active?'Active':'Pending DNS'} color={d.active?'#00d4aa':BASE.aw}/>
          <IBtn id="trash" title="Remove" danger onClick={()=>setDomains(ds=>ds.filter(x=>x.id!==d.id))}/>
        </div>
      ))}

      <div style={{display:'flex',gap:8,marginTop:16}}>
        <input value={input} onChange={e=>setInput(e.target.value)} placeholder="your-domain.com"
          style={{...inputSty,flex:1}} onKeyDown={e=>e.key==='Enter'&&addDomain()}/>
        <PBtn onClick={addDomain}><Ico id="plus" size={13} color="#0a0b0f"/> Add Domain</PBtn>
      </div>
      <p style={{fontSize:11,color:BASE.t3,marginTop:8}}>Create a CNAME record pointing to the target above. SSL is provisioned automatically via Caddy.</p>
    </Modal>
  );
}

/* ── Payment / Balance ─────────────────────────────────────── */
function PaymentModal({ balance, onClose, toast }) {
  const { accent } = useTheme();
  const [amount,setAmount]=useState('10');

  const deposit = () => {
    toast(`Deposit of $${amount} USDC initiated on Polygon.`,'info');
    onClose();
  };

  const txns = [
    { type:'Deposit',  amount:'+$50.00', date:'2026-05-15', color:'#00d4aa' },
    { type:'VM Charge',amount:'-$0.48',  date:'2026-05-15', color:BASE.t3 },
    { type:'VM Charge',amount:'-$1.20',  date:'2026-05-14', color:BASE.t3 },
  ];

  return (
    <Modal title="Balance & Payments" onClose={onClose} footer={<SBtn onClick={onClose}>Close</SBtn>}>
      <div style={{textAlign:'center',padding:'14px 0 22px'}}>
        <div style={{fontSize:11,color:BASE.t3,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.09em',marginBottom:8}}>Current Balance</div>
        <div style={{fontSize:44,fontWeight:700,color:BASE.t1,fontFamily:BASE.fm,letterSpacing:'-0.03em'}}>${(balance||0).toFixed(2)}</div>
        <div style={{fontSize:13,color:accent,marginTop:4}}>USDC on Polygon</div>
      </div>

      <div style={{background:BASE.bg3,borderRadius:12,padding:'18px',marginBottom:18}}>
        <p style={{fontSize:13,fontWeight:600,color:BASE.t2,marginBottom:12}}>Deposit USDC</p>
        <div style={{display:'flex',gap:8}}>
          <input value={amount} onChange={e=>setAmount(e.target.value)} type="number" min="1" placeholder="Amount" style={{...inputSty,flex:1}}/>
          <PBtn onClick={deposit}><Ico id="deposit" size={13} color="#0a0b0f"/> Deposit</PBtn>
        </div>
        <p style={{fontSize:11,color:BASE.t3,marginTop:8}}>Available instantly after 1 Polygon confirmation (~2 seconds).</p>
      </div>

      <p style={{fontSize:12,fontWeight:600,color:BASE.t2,marginBottom:10}}>Recent Transactions</p>
      {txns.map((t,i)=>(
        <div key={i} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'9px 0',borderBottom:i<txns.length-1?`1px solid ${BASE.bd}`:'none'}}>
          <span style={{fontSize:13,color:BASE.t2}}>{t.type}</span>
          <div style={{textAlign:'right'}}>
            <div style={{fontFamily:BASE.fm,fontSize:13,color:t.color,fontWeight:600}}>{t.amount}</div>
            <div style={{fontSize:10,color:BASE.t3}}>{t.date}</div>
          </div>
        </div>
      ))}
    </Modal>
  );
}

/* ── VM Events / Messages ──────────────────────────────────── */
const MOCK_EVENTS = [
  { ts:'05:14:23', lvl:'info', msg:'VM scheduled on us-east-node-1' },
  { ts:'05:14:25', lvl:'info', msg:'cloud-init started' },
  { ts:'05:14:58', lvl:'info', msg:'cloud-init completed successfully' },
  { ts:'05:15:02', lvl:'info', msg:'Service WebUI :7860 is Ready' },
  { ts:'06:01:44', lvl:'warn', msg:'High CPU usage detected (92%)' },
  { ts:'06:05:10', lvl:'info', msg:'CPU usage normalised (41%)' },
];

function VmEventsModal({ vm, onClose }) {
  const lvlColor = { info:BASE.a2, warn:BASE.aw, error:BASE.ad };
  return (
    <Modal title={`Events — ${vm.name}`} onClose={onClose} footer={<SBtn onClick={onClose}>Close</SBtn>}>
      <div style={{display:'flex',flexDirection:'column',gap:2}}>
        {MOCK_EVENTS.map((e,i)=>(
          <div key={i} style={{display:'flex',gap:12,padding:'9px 12px',borderRadius:8,background:i%2===0?BASE.bg3:'transparent',alignItems:'flex-start'}}>
            <div style={{width:5,height:5,borderRadius:'50%',background:lvlColor[e.lvl]||BASE.t3,marginTop:5,flexShrink:0}}/>
            <code style={{fontFamily:BASE.fm,fontSize:11,color:BASE.t3,flexShrink:0,minWidth:56,marginTop:1}}>{e.ts}</code>
            <span style={{fontSize:13,color:BASE.t1,lineHeight:1.5}}>{e.msg}</span>
          </div>
        ))}
      </div>
    </Modal>
  );
}

Object.assign(window, {
  CreateVMModal, AddSSHKeyModal,
  DirectAccessModal, CustomDomainsModal,
  PaymentModal, VmEventsModal,
});
