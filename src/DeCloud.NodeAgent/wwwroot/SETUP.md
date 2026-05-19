# DeCloud React Frontend — Drop-in Package

React 18 + Babel (CDN) replacement for the vanilla JS frontend.
**No build step. No npm install. Just copy files.**

---

## Quick Start

### 1 — Copy files to wwwroot

```
src/Orchestrator/wwwroot/
  app-v2.html        ← new entry point
  dc-tokens.jsx      ← tokens · mock data · API helper · icons
  dc-atoms.jsx       ← UI atoms: buttons, cards, modals, toasts
  dc-auth.jsx        ← login overlay + MetaMask auth
  dc-realtime.jsx    ← SignalR real-time hook
  dc-modals.jsx      ← CreateVM · DirectAccess · Domains · Payment · Events
  dc-pages.jsx       ← 7 pages with live API + service readiness
  dc-shell.jsx       ← CommandBar · Sidebar · TweaksPanel
```

### 2 — Start the orchestrator

```bash
cd src/Orchestrator
dotnet run
```

### 3 — Open the new UI

```
http://localhost:5000/app-v2.html
```

The existing `index.html` (vanilla JS) continues to work unchanged at `/`.

---

## Co-existence Strategy (Phased Migration)

Both UIs hit the same backend. Migrate one page at a time:

| Phase | Action |
|-------|--------|
| 1 — Now    | Both run side by side. New UI at `/app-v2.html`. |
| 2 — Verify | Confirm each page works with live data in new UI. |
| 3 — Cut over | Rename `app-v2.html` → `index.html`, archive old files. |

---

## Authentication

| Scenario | Behaviour |
|----------|-----------|
| MetaMask installed + backend running | Real wallet auth, JWT stored in `localStorage` |
| MetaMask installed, backend offline | Demo mode with mock data (fallback) |
| No wallet | Demo mode, user prompted to install MetaMask |

To wire up Reown AppKit (WalletConnect) instead of direct MetaMask:
replace `tryMetaMaskAuth()` in `dc-auth.jsx` with the AppKit flow from `src/app.js`.
The integration point is the `onLogin(wallet, token)` callback — pass the real
`accessToken` from the AppKit auth response.

---

## SignalR Event Names

The hub at `/hub/orchestrator` is connected automatically when logged in.
Update event names in `dc-pages.jsx` (VMsPage `useEffect`) to match your hub:

```csharp
// In your Hub class, look for calls like:
await Clients.All.SendAsync("YourEventName", payload);
```

Current expected names (update if different):

| Event | Payload | Effect |
|-------|---------|--------|
| `VmStatusChanged` | `(vmId: string, status: int)` | Updates VM status dot live |
| `VmCreated`       | `(vm: object)`                | Adds VM to list |
| `VmDeleted`       | `(vmId: string)`              | Removes VM from list |
| `NodeStatusChanged` | `(nodeId: string, online: bool)` | Future use |

---

## API Compatibility

All endpoints match the existing orchestrator exactly:

| Endpoint | Used by |
|----------|---------|
| `GET /api/vms` | VMs page, Dashboard |
| `GET /api/marketplace/nodes` | Nodes page |
| `GET /api/system/stats` | Dashboard stat cards |
| `GET /api/marketplace/templates` | Template marketplace |
| `GET /api/ssh-keys` | SSH Keys page |
| `GET /api/user/balance` | Balance pill (via Payment modal) |
| `POST /api/auth/message` | MetaMask auth step 1 |
| `POST /api/auth/wallet` | MetaMask auth step 2 |
| `/hub/orchestrator` | SignalR real-time |

---

## Tweaks Panel

Click the **Tweaks** toggle in the toolbar to expose:
- **Accent colour** — 4 presets (teal / blue / purple / amber)
- **Density** — Spacious / Compact

---

## File Sizes (reference)

| File | Lines |
|------|-------|
| `app-v2.html` | ~110 |
| `dc-tokens.jsx` | ~145 |
| `dc-atoms.jsx` | ~220 |
| `dc-auth.jsx` | ~150 |
| `dc-realtime.jsx` | ~55 |
| `dc-modals.jsx` | ~225 |
| `dc-pages.jsx` | ~390 |
| `dc-shell.jsx` | ~195 |

All files are plain browser JavaScript — readable and directly editable.
No transpilation, no source maps, no bundler needed in development.
