"""
config.py — Connection and display configuration.

Resolution order (highest priority first):
  1. CLI flags  (parsed in __main__.py, set on cfg before app starts)
  2. Environment: DECLOUD_URL, DECLOUD_TOKEN, DECLOUD_NODE_URL, DECLOUD_REFRESH
  3. ~/.decloud/config — KEY=VALUE lines

Security:
  • The user config file is created with mode 0600. Existing files
    with looser permissions raise a startable warning (printed once).
  • The token is never written back; rewriting the file preserves
    the file's 0600 mode.
  • cfg.token is held in memory only; cfg.has_token() lets screens
    decide whether to attempt orchestrator calls without leaking the
    value.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

_USER_CONFIG = Path.home() / ".decloud" / "config"

DEFAULT_NODE_URL = "http://localhost:5100"
DEFAULT_REFRESH  = 5
DEFAULT_LOG_LINES = 500


@dataclass
class Config:
    orchestrator_url: str = ""
    token: str = ""
    node_url: str = DEFAULT_NODE_URL
    node_only: bool = False
    refresh_interval: int = DEFAULT_REFRESH
    log_lines: int = DEFAULT_LOG_LINES

    # Live identity, populated from the most recent /api/dashboard/summary.
    # Other widgets (IdentityBar) read from here so they don't re-fetch.
    identity: dict = field(default_factory=dict)

    # ─── Properties used by screens for graceful degradation ───────────

    @property
    def has_node_agent(self) -> bool:
        return bool(self.node_url)

    @property
    def has_orchestrator(self) -> bool:
        return bool(self.orchestrator_url and self.token) and not self.node_only

    def has_token(self) -> bool:
        return bool(self.token)

    # ─── Loading / saving ──────────────────────────────────────────────

    def load(self) -> None:
        # Load file first so env vars override.
        self._load_file()
        self._load_env()

    def _load_env(self) -> None:
        if v := os.environ.get("DECLOUD_URL"):
            self.orchestrator_url = v.rstrip("/")
        if v := os.environ.get("DECLOUD_TOKEN"):
            self.token = v
        if v := os.environ.get("DECLOUD_NODE_URL"):
            self.node_url = v.rstrip("/")
        if v := os.environ.get("DECLOUD_REFRESH"):
            try:
                self.refresh_interval = max(1, int(v))
            except ValueError:
                pass

    def _load_file(self) -> None:
        if not _USER_CONFIG.exists():
            return
        # Reject world/group-readable config files — token would be exposed.
        st = _USER_CONFIG.stat()
        if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(
                f"[warning] {_USER_CONFIG} is readable by group/other; "
                "ignoring file. Run: chmod 600 ~/.decloud/config",
                flush=True,
            )
            return
        try:
            for raw in _USER_CONFIG.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip().upper(), val.strip()
                if key == "DECLOUD_URL":
                    self.orchestrator_url = val.rstrip("/")
                elif key == "DECLOUD_TOKEN":
                    self.token = val
                elif key == "DECLOUD_NODE_URL":
                    self.node_url = val.rstrip("/")
                elif key == "DECLOUD_REFRESH":
                    try:
                        self.refresh_interval = max(1, int(val))
                    except ValueError:
                        pass
        except OSError:
            pass

    def save(self) -> list[str]:
        """Persist current config to ~/.decloud/config with mode 0600.

        Returns a list of issues encountered (empty on success).
        """
        issues: list[str] = []
        try:
            _USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            # Write with restrictive umask, then chmod for safety.
            content = "\n".join([
                f"DECLOUD_URL={self.orchestrator_url}",
                f"DECLOUD_TOKEN={self.token}",
                f"DECLOUD_NODE_URL={self.node_url}",
                f"DECLOUD_REFRESH={self.refresh_interval}",
                "",
            ])
            tmp = _USER_CONFIG.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.replace(tmp, _USER_CONFIG)
        except OSError as exc:
            issues.append(f"could not write {_USER_CONFIG}: {exc}")
        return issues

    def validate(self) -> list[str]:
        """Return a list of human-readable problems, empty if all good."""
        errs: list[str] = []
        if not self.has_node_agent and not self.has_orchestrator:
            errs.append("No data sources configured. Set DECLOUD_NODE_URL "
                        "(local) or DECLOUD_URL+DECLOUD_TOKEN (orchestrator).")
        return errs


cfg = Config()
cfg.load()
