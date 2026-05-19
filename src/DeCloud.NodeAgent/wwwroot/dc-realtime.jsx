/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   dc-realtime.jsx — SignalR hub connection
   Requires window.signalR (@microsoft/signalr CDN).

   Expected hub events — update names to match your hub:
     VmStatusChanged(vmId: string, status: int)
     VmCreated(vm: object)
     VmDeleted(vmId: string)
     NodeStatusChanged(nodeId: string, online: bool)
     SystemStatsUpdated(stats: object)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const { useState, useEffect } = React;
const { getOrchestratorUrl } = window;

function useSignalR(token) {
  const [rtStatus, setRtStatus] = useState('idle');
  const [hub, setHub] = useState(null);

  useEffect(() => {
    if (!token || !window.signalR) {
      setRtStatus(token ? 'unavailable' : 'idle');
      return;
    }

    const conn = new window.signalR.HubConnectionBuilder()
      .withUrl(`${getOrchestratorUrl()}/hub/orchestrator`, {
        accessTokenFactory: () => localStorage.getItem('authToken') || '',
      })
      .withAutomaticReconnect([0, 2000, 5000, 10000, 30000])
      .configureLogging(window.signalR.LogLevel.Warning)
      .build();

    conn.onclose(() => { setHub(null); setRtStatus('error'); });
    conn.onreconnecting(() => setRtStatus('connecting'));
    conn.onreconnected(() => { setRtStatus('connected'); });

    setRtStatus('connecting');
    conn.start()
      .then(() => {
        setHub(conn);
        setRtStatus('connected');
        console.log('[SignalR] Connected to /hub/orchestrator');
      })
      .catch(e => {
        console.warn('[SignalR] Connection failed (backend may be offline):', e.message);
        setRtStatus('error');
      });

    return () => { conn.stop().catch(()=>{}); setHub(null); };
  }, [token]);

  return { rtStatus, hub };
}

Object.assign(window, { useSignalR });
