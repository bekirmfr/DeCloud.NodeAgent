# Installation

The dashboard runs on **Python 3.11+** and depends only on three pure-Python
packages: `textual`, `httpx`, `rich`.

Pick the path that matches what you want to do:

| | Use case | Time | What you get |
|---|---|---|---|
| **A** | Just see what it looks like | 30 s | The TUI runs from anywhere |
| **B** | Use it day-to-day on a node, in a venv | 1 min | Isolated install, no system pollution |
| **C** | Production install on a node | 2 min | `decloud-dashboard` and `decloud dashboard` system-wide |

---

## Path A — Quick try (no venv, no installer)

For evaluating the redesign on a dev machine.

```bash
# 1. Extract the archive
tar xzf cli-dashboard-v2.tar.gz
cd cli-dashboard

# 2. Install the three deps (uses your system Python)
pip install --user -r requirements.txt
#   On Ubuntu 24.04+ / Debian 12+ pip refuses by default; either
#     pip install --user --break-system-packages -r requirements.txt
#   or use Path B (venv) which is cleaner.

# 3. Run it — point it at any reachable Node Agent
python __main__.py --node http://localhost:5100 --node-only
```

That's it. No file goes outside the extracted directory.

---

## Path B — Per-user venv install

For ongoing personal use without touching system files.

```bash
tar xzf cli-dashboard-v2.tar.gz
cd cli-dashboard

# 1. Create a venv just for the dashboard
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies into the venv
pip install -r requirements.txt

# 3. Run
python __main__.py
```

Re-running later is just `source .venv/bin/activate && python __main__.py`.

---

## Path C — Production install on a node (replaces the existing dashboard)

This is for a real DeCloud node that already has the v1 dashboard installed
(or is having it installed for the first time). It puts the dashboard at the
expected path, runs the installer, and makes `decloud-dashboard` and
`decloud dashboard` available system-wide.

### Step 1 — Replace the `cli/dashboard/` directory

```bash
# At the root of your DeCloud node-agent checkout
cd /path/to/DeCloud.NodeAgent

# Back up the existing v1 dashboard (in case you want to roll back)
mv cli/dashboard cli/dashboard.v1.bak

# Extract the redesign in its place
tar xzf /path/to/cli-dashboard-v2.tar.gz -C cli/
mv cli/cli-dashboard cli/dashboard
```

### Step 2 — Run the installer

```bash
bash cli/dashboard/install-dashboard.sh
```

The installer will:

1. Verify Python 3.11+
2. Create / reuse a venv at `~/.decloud/dashboard-venv`
3. Install `requirements.txt` into the venv
4. Write `/usr/local/bin/decloud-dashboard` (uses sudo)
5. Patch `/usr/local/bin/decloud` to add the `dashboard` subcommand (uses sudo)
6. Seed `~/.decloud/config` with mode 0600 if it doesn't exist

### Step 3 — Run the dashboard

```bash
decloud-dashboard            # full path
decloud dashboard            # via the patched main CLI
decloud-dashboard --node-only  # if you don't have orchestrator credentials
```

### What changed vs the v1 installer

The v1 installer's launcher shim used `python -m dashboard`, which only
worked when invoked from a working directory that happened to contain a
`dashboard/` subdirectory — an unusual user CWD. The redesigned installer's
shim invokes `__main__.py` directly, so it works regardless of CWD. No other
behavior changed.

### Rollback

```bash
sudo rm /usr/local/bin/decloud-dashboard
rm -rf ~/.decloud/dashboard-venv
mv cli/dashboard.v1.bak cli/dashboard
# The 'dashboard' subcommand patch in /usr/local/bin/decloud is harmless to leave;
# if you want to remove it, edit /usr/local/bin/decloud and delete the block
# marked '# ── dashboard subcommand (added by install-dashboard.sh) ──'.
```

---

## Configuration

The dashboard reads config in this order (highest precedence first):

1. **CLI flags** — `--url`, `--token`, `--node`, `--refresh`, `--node-only`
2. **Environment** — `DECLOUD_URL`, `DECLOUD_TOKEN`, `DECLOUD_NODE_URL`,
   `DECLOUD_REFRESH`
3. **`~/.decloud/config`** — `KEY=VALUE` lines (must be mode 0600)

Minimal config for a node-only setup:

```sh
# ~/.decloud/config
DECLOUD_NODE_URL=http://localhost:5100
DECLOUD_REFRESH=5
```

With orchestrator (enables the Earnings card on Overview):

```sh
DECLOUD_URL=https://orchestrator.example.com
DECLOUD_TOKEN=<node JWT>
DECLOUD_NODE_URL=http://localhost:5100
DECLOUD_REFRESH=5
```

> **Security note.** Prefer setting `DECLOUD_TOKEN` via the config file (mode
> 0600) or as an environment variable. Passing `--token` on the command line
> exposes it in your shell history and to anyone running `ps`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Python 3.11+ not found` | `sudo apt install python3.11` (or 3.12) |
| `ModuleNotFoundError: No module named 'textual'` | You skipped the venv activate step. `source .venv/bin/activate` first, or use Path C. |
| Dashboard launches but every screen says "—" | The Node Agent isn't reachable on the URL you configured. Try `curl http://localhost:5100/api/dashboard/summary` to confirm. |
| "Connection failed" toasts | `DECLOUD_NODE_URL` points at a port that's closed, or the Node Agent is down. `systemctl status decloud-node-agent`. |
| The Earnings card stays "—" | You're in node-only mode (intended), or the orchestrator URL/token isn't configured. |
| `~/.decloud/config is readable by group/other; ignoring file` | The config file's permissions are too loose. `chmod 600 ~/.decloud/config`. |

To diagnose at a deeper level, the **Diagnostics** screen (key `8`) runs six
health checks and can export a full JSON snapshot of every node-agent endpoint
to `~/.decloud/snapshots/decloud-snapshot-<timestamp>.json` (mode 0600) —
attach that file to support tickets instead of pasting screenshots.
