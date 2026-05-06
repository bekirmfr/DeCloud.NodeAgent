# DeCloud CLI Dashboard — Redesign

**Version:** 2.0
**Scope:** Complete redesign of `cli/dashboard/` (Python Textual TUI)
**Audience:** Node operators

---

## 1 · Why redesign

The previous CLI dashboard was a *mini-orchestrator UI*. Its top-level
screens were **Nodes**, **VMs**, **Networking**, **Ingress Routes**,
**Billing** — all of which already exist in the orchestrator's web UI
that the same operator can open in a browser. Meanwhile, the CLI runs
on a node host and connects primarily to the local Node Agent at
`localhost:5100`, where a richer dataset is available but barely
surfaced: `/api/node/snapshot`, `/api/node/resources`, `/api/dashboard/database`,
`/api/dashboard/obligations`, full GPU detail, system VM state data, etc.

Two design failures followed from this:

1. **Wrong system boundary.** A CLI run on a node should focus on
   *that node's operation*. Cross-fleet data is supplementary.
2. **Underused API surface.** The Node Agent exposes 10+ rich
   dashboard endpoints; the CLI consumed about 4 of them.

This redesign re-aligns the CLI to the natural system boundary
(*this node, accessed via the local agent*) and makes the Node Agent's
existing capabilities first-class.

---

## 2 · Design principles

| | Principle | Manifestation |
|---|---|---|
| 1 | **Node-first** | Local agent is the primary data source; orchestrator is optional. Token is *not* required for ~90 % of features. |
| 2 | **Map screens to operator mental models, not to API surface** | 9 screens, each answering one question an operator actually asks. |
| 3 | **Density done right** | Gauges + sparklines + status pills, not big rectangles of text. Inspired by `btop`, `k9s`, `lazygit`. |
| 4 | **Graceful degradation** | If a data source is unavailable, the screen shows what it would need — never an unexplained error. |
| 5 | **Keyboard-first** | 1–9 to switch screens; single-letter actions on each screen; permanent hint bar at the bottom. |
| 6 | **No fake controls** | If async refresh handles it, there is no "force refresh" button. If the orchestrator handles deletion of system VMs, the CLI doesn't pretend to. |
| 7 | **Consistent visual language** | Every colour comes from `theme.py`; every status pill from `widgets/statpill`; every state badge from `widgets/badges`. |

---

## 3 · Screen inventory (9 screens, ⇐ the existing 1–9 keymap)

| # | Screen | Replaces | Owns |
|---|---|---|---|
| 1 | **Overview** | Dashboard | Identity, gauges with sparklines, system obligations, earnings (when authenticated), recent events. Also drives the persistent top status strip. |
| 2 | **Hardware** | *(new — was a tile on Dashboard)* | CPU benchmark, memory, storage volumes, GPU detail (IOMMU / passthrough / proxy), KVM / WSL / container runtimes, NICs. |
| 3 | **Virtual Machines** | VMs **+** System VMs | Unified table; `SYS` / `USR` role chip distinguishes system from tenant. Per-row actions: start, stop, restart, delete (D pressed twice to confirm). |
| 4 | **Network** | Networking | Interfaces, WireGuard peers (handshake age, traffic), bridges with each tap-port mapped to its VM, routes — split into 4 tabs. |
| 5 | **Firewall** | *(new — was buried in Networking on the web)* | Listening TCP/UDP ports with process names, UFW status & rules, iptables INPUT / FORWARD / NAT POSTROUTING. |
| 6 | **Services** | *(new)* | systemd unit health. Critical units (`decloud-node-agent`, `libvirtd`) are flagged when down. |
| 7 | **Logs** | Live Logs | Filterable tail (level + free text). |
| 8 | **Diagnostics** | *(new)* | 6 health checks + one-click *Snapshot Export* — collects every dashboard endpoint into `~/.decloud/snapshots/decloud-snapshot-<ts>.json` (mode 0600) for support tickets. |
| 9 | **Settings** | Settings | Connection config, persisted to `~/.decloud/config` with mode 0600. |

**Dropped:** Nodes, Ingress Routes, Billing — these are orchestrator
concerns and belong in the orchestrator web UI. Per-VM ingress URLs are
shown inline in the VMs table; node-local earnings appear as a small
card on Overview when the orchestrator is configured.

---

## 4 · ASCII mockups

> Real rendering is in 256-truecolor with rounded borders and
> Unicode block characters (`▁▂▃▄▅▆▇█`). These ASCII sketches
> approximate the layout, not the polish.

### 4.1 Overview

```
┌──────────┬──────────────────────────────────────────────────────────────┐
│ ◆ DECLOUD│ DECLOUD nd_a3f12b…   host edge-eu-04 · linux · uptime 4d 6h  │
│  node ops│ ─────────────────────────────────────────────────────────── │
│          │ ●Orch 12s ago │ CPU 32% │ RAM 41% │ STOR 12% │ VMs 6/8 │ ERR0│
│ HEALTH   │ ─────────────────────────────────────────────────────────── │
│ 1 Overview│ ╭ Resources ──────────────────╮ ╭ System Obligations ───╮  │
│ 2 Hardware│ │ CPU     ████░░░░░░  32 % ▁▂▅│ │ ● DHT          Active │  │
│ WORKLOADS │ │ Memory  ██████░░░░  41 % ▂▃▆│ │ ● Block Store  Active │  │
│ 3 Virtual…│ │ Storage █░░░░░░░░░  12 % ▁▁▁│ │ ● Relay        Active │  │
│ CONNECT.. │ │ GPU     ─── n/a            │ ╰───────────────────────╯  │
│ 4 Network │ │ Network eth0   3.21 MB/s   │                              │
│ 5 Firewall│ ╰────────────────────────────╯ ╭ Earnings  24 h / 30 d ─╮  │
│ SYSTEM    │                                  │ $0.42  $9.81  6 active │ │
│ 6 Services│ ╭ Virtual Machines (8) ──────╮  ╰────────────────────────╯ │
│ 7 Logs    │ │ ▶ Running  blockstore-eu-… │  ╭ Recent Events ─────────╮ │
│ TOOLS     │ │ ▶ Running  dht-eu-…        │  │ 14:02 INF  vm up       │ │
│ 8 Diag…   │ │ ▶ Running  ml-trainer      │  │ 14:01 WRN  hb 35s ago  │ │
│ 9 Settings│ │ ■ Stopped  web-prod        │  │ 13:58 INF  obl Active  │ │
│          │ ╰────────────────────────────╯  ╰────────────────────────╯ │
│          │ ─────────────────────────────────────────────────────────── │
│          │ 1-9 switch  r refresh  ? help  q quit  v VMs  d Diag · 14:02:03 │
└──────────┴──────────────────────────────────────────────────────────────┘
```

The top **identity bar** and **status strip** are persistent — they
appear above every screen so the operator never loses situational
awareness while drilling into details.

### 4.2 Hardware

```
╭ CPU ────────────────────────────────╮  ╭ GPU ───────────────────────────╮
│ Model          AMD Ryzen 9 5950X     │  │ GPUs           1               │
│ Cores          32 logical / 16 phys… │  │ Model          NVIDIA RTX 3090 │
│ Freq           3400 MHz              │  │ VRAM           24.0 GB         │
│ Architecture   x86_64                │  │ Driver         535.86.05       │
│ Benchmark      24831                 │  │ IOMMU          yes             │
│ Features       avx, avx2, sse4_2 …   │  │ Passthrough    yes             │
╰──────────────────────────────────────╯  │ Proxy capable  yes             │
                                          ╰────────────────────────────────╯
╭ Memory ─────────────────────────────╮
│ Total          128.0 GB              │  ╭ Virtualization & Runtimes ────╮
│ Available      75.4 GB               │  │ KVM            available       │
│ Swap           8.0 GB                │  │ WSL2           no              │
╰──────────────────────────────────────╯  │ Container rt.  docker v24.0.7  │
                                          │ GPU containers yes             │
╭ Storage Volumes ────────────────────╮  ╰────────────────────────────────╯
│ /var/lib/decloud  ext4  812 GB / 2T │
│ /                 ext4  47 GB / 100… │  ╭ Network Interfaces ───────────╮
╰──────────────────────────────────────╯  │ eth0   …  10000 Mbps  UP       │
                                          │ wg0    …  N/A         UP       │
                                          ╰────────────────────────────────╯
```

### 4.3 Virtual Machines

```
[ search:  __________ ] [ Filter: All ▾ ]
8 shown (3 system / 5 tenant) · 6 running

┌────────┬──────┬───────────────────┬──────────┬─────┬───────┬───────┬────────┐
│ State  │ Role │ Name              │ Type     │ vCPU│ Mem   │ IP    │ Ingress│
├────────┼──────┼───────────────────┼──────────┼─────┼───────┼───────┼────────┤
│▶Runnin│ SYS  │ blockstore-eu-04  │BlockStore│  2  │ 4.0 GB│ 10.0..│ —      │
│▶Runnin│ SYS  │ dht-eu-04         │ DHT      │  1  │ 1.0 GB│ 10.0..│ —      │
│▶Runnin│ SYS  │ relay-eu-04       │ Relay    │  1  │ 1.0 GB│ 10.0..│ —      │
│▶Runnin│ USR  │ ml-trainer        │ GPU      │  8  │16.0 GB│192.16.│https…  │
│▶Runnin│ USR  │ web-prod          │ General  │  2  │ 2.0 GB│192.16.│https…  │
│■Stoppd│ USR  │ batch-job-7       │ Compute  │  4  │ 8.0 GB│ —     │ —      │
└────────┴──────┴───────────────────┴──────────┴─────┴───────┴───────┴────────┘

  s start    S stop    R restart    D delete (press twice to confirm)
```

### 4.4 Diagnostics

```
╭ Health Checks ──────────────────────────╮  ╭ Snapshot Export ────────────╮
│ ✓ Node agent reachable                   │  │ Collect a JSON snapshot of  │
│ ✓ KVM available (/dev/kvm present)       │  │ every node-agent endpoint.  │
│ ✓ Orchestrator heartbeat fresh (12s ago) │  │ Saved to ~/.decloud/        │
│ ✓ All critical services active           │  │ snapshots/ with mode 0600.  │
│ ✓ All 3 system obligations Active        │  │                             │
│ ✓ Storage headroom OK (12% used)         │  │ [ Collect & Save ]          │
╰──────────────────────────────────────────╯  │                             │
                                              │ Saved /home/op/.decloud/    │
                                              │ snapshots/decloud-snapsh…   │
                                              ╰─────────────────────────────╯
```

---

## 5 · File layout

```
cli/dashboard/
├── __main__.py                  CLI entry  (argparse, validate, run)
├── app.py                       Top-level Textual App + global theming
├── config.py                    Env + file config; chmod 0600 on save
├── theme.py                     Colour tokens, severity grading, glyphs
│
├── api/
│   ├── client.py                BaseClient: HTTPS warning, JWT in header,
│   │                            bounded timeouts, bounded retries
│   ├── node_agent.py            Full coverage of /api/dashboard/*,
│   │                            /api/node/*, /api/vms/*
│   └── orchestrator.py          Minimal — earnings + fleet glance only
│
├── widgets/
│   ├── card.py                  Bordered titled container
│   ├── header.py                IdentityBar + StatusStrip (persistent chrome)
│   ├── gauge.py                 Label · bar · % · sparkline (one row)
│   ├── sparkline.py             Block-char inline sparkline
│   ├── statpill.py              Coloured ●/✓ pill with label
│   ├── badges.py                VM state + obligation badges (Rich Text)
│   └── keyhints.py              Bottom hint bar with last-update time
│
├── screens/
│   ├── _base.py                 Sidebar + BaseScreen (chrome stays put)
│   ├── overview.py              ⓵
│   ├── hardware.py              ⓶
│   ├── vms.py                   ⓷
│   ├── network.py               ⓸
│   ├── firewall.py              ⓹
│   ├── services.py              ⓺
│   ├── logs.py                  ⓻
│   ├── diagnostics.py           ⓼
│   └── settings.py              ⓽
│
└── util/
    ├── format.py                fmt_bytes, fmt_age, fmt_pct, truncate, …
    └── history.py               Ring (bounded deque for sparklines)
```

---

## 6 · Security posture

* **HTTPS preferred.** `BaseClient` warns once (and only once) per
  base URL when connecting over plain HTTP to a non-localhost endpoint.
  It does not refuse — lab and dev setups need to work.
* **JWT in header only.** Tokens never appear in URLs, log lines, or
  rendered output. The Settings screen uses a password input.
* **`~/.decloud/config` is mode 0600.** A config file with looser
  permissions is rejected at load time with a printed warning.
* **Snapshot export is mode 0600.** Even on personal machines, a
  diagnostic dump that lands in someone's `~` should not be world-readable.
* **No mutating actions on system VMs proxied locally.** Start/Stop/
  Delete on a system VM go to the local agent, which the orchestrator
  reconciles. The CLI does not bypass the obligation reconciler.

---

## 7 · What changed (migration notes)

| Old | New |
|---|---|
| `screens/dashboard.py` (Dashboard) | `screens/overview.py` |
| `screens/nodes.py`                 | **dropped** (orchestrator concern) |
| `screens/vms.py` + `screens/system_vms.py` | merged into `screens/vms.py` with role chip |
| `screens/networking.py`            | `screens/network.py` (tabs: Interfaces / WG / Bridges / Routes) |
| `screens/ingress_routes.py`        | **dropped** — per-VM ingress is shown in the VMs table |
| `screens/billing.py`               | **dropped** — earnings card on Overview when authenticated |
| `screens/logs.py` (Live Logs)      | `screens/logs.py` |
| `screens/settings.py`              | `screens/settings.py` |
| —                                  | `screens/hardware.py` (new) |
| —                                  | `screens/firewall.py` (new) |
| —                                  | `screens/services.py` (new) |
| —                                  | `screens/diagnostics.py` (new — snapshot export) |

The 1–9 keymap is preserved (keys map to the new labels in order).

---

## 8 · Running

```sh
# Most common — node-only mode, no auth needed.
DECLOUD_NODE_URL=http://localhost:5100  python -m dashboard

# With orchestrator (enables Earnings card on Overview).
DECLOUD_URL=https://orch.example.com  DECLOUD_TOKEN=<jwt>  python -m dashboard

# CLI flags override env which overrides ~/.decloud/config.
python -m dashboard --node http://localhost:5100 --refresh 3 --node-only
```
