"""
config.py — Credential and settings loader.

Priority order:
  1. Environment variables  (DECLOUD_URL, DECLOUD_TOKEN, …)
  2. ~/.decloud/config       (chmod 600, key=value format)
  3. .env file in cwd        (dev convenience, never production)

Secrets are never written to disk by this module.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Load sources (lowest priority first so env vars always win)
# ---------------------------------------------------------------------------

_USER_CONFIG = Path.home() / ".decloud" / "config"


def _load_user_config() -> None:
    """Load ~/.decloud/config if it exists and has safe permissions."""
    if not _USER_CONFIG.exists():
        return
    mode = _USER_CONFIG.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(
            f"[warn] {_USER_CONFIG} is world/group-readable — "
            "run: chmod 600 ~/.decloud/config"
        )
    load_dotenv(dotenv_path=_USER_CONFIG, override=False)


def _load_dotenv_cwd() -> None:
    """Load .env from cwd (dev only, does not override already-set vars)."""
    load_dotenv(override=False)


_load_user_config()
_load_dotenv_cwd()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Orchestrator
    orchestrator_url: str = field(default_factory=lambda: os.getenv("DECLOUD_URL", "").rstrip("/"))
    token: str = field(default_factory=lambda: os.getenv("DECLOUD_TOKEN", ""))

    # Node Agent (optional)
    node_url: str = field(default_factory=lambda: os.getenv("DECLOUD_NODE_URL", "").rstrip("/"))

    # Behaviour
    refresh_interval: int = field(
        default_factory=lambda: int(os.getenv("DECLOUD_REFRESH_INTERVAL", "5"))
    )
    log_lines: int = field(
        default_factory=lambda: int(os.getenv("DECLOUD_LOG_LINES", "200"))
    )
    default_region: str = field(
        default_factory=lambda: os.getenv("DECLOUD_DEFAULT_REGION", "")
    )

    # Runtime overrides (set by CLI flags, not env vars)
    node_only: bool = False

    @property
    def has_orchestrator(self) -> bool:
        return bool(self.orchestrator_url and self.token)

    @property
    def has_node_agent(self) -> bool:
        return bool(self.node_url)

    def validate(self) -> list[str]:
        """Return a list of human-readable errors; empty means config is valid."""
        errors: list[str] = []
        if not self.has_orchestrator and not self.node_only:
            errors.append("DECLOUD_URL and DECLOUD_TOKEN are required (or use --node-only)")
        if self.node_only and not self.has_node_agent:
            errors.append("DECLOUD_NODE_URL is required in --node-only mode")
        if self.refresh_interval < 1:
            errors.append("DECLOUD_REFRESH_INTERVAL must be >= 1 second")
        return errors

    def auth_header(self) -> dict[str, str]:
        """Return the Authorization header dict for orchestrator requests."""
        return {"Authorization": f"Bearer {self.token}"}


# Singleton — import and use directly
cfg = Config()
