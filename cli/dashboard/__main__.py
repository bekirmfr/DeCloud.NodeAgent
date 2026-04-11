"""
__main__.py — DeCloud Dashboard entry point.

Usage:
  python -m dashboard                           # full mode
  python -m dashboard --node-only               # node agent only
  python -m dashboard --url https://...         # override orchestrator URL
  python -m dashboard --token eyJ...            # override token (prefer env var)
  python -m dashboard --node http://localhost:5100
  python -m dashboard --refresh 10              # override refresh interval

Security note: passing --token on the CLI is convenient for dev but exposes
the token in shell history and ps output. Prefer DECLOUD_TOKEN env var.
"""

from __future__ import annotations

import argparse
import sys


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="decloud dashboard",
        description="DeCloud CLI Dashboard — terminal UI for the DeCloud platform",
    )
    p.add_argument("--url",         metavar="URL",  help="Orchestrator URL (overrides DECLOUD_URL)")
    p.add_argument("--token",       metavar="JWT",  help="API token (prefer DECLOUD_TOKEN env var)")
    p.add_argument("--node",        metavar="URL",  help="Node Agent URL (overrides DECLOUD_NODE_URL)")
    p.add_argument("--node-only",   action="store_true", help="Connect to node agent only")
    p.add_argument("--refresh",     metavar="SEC",  type=int, help="Refresh interval in seconds")
    p.add_argument("--version",     action="store_true", help="Print version and exit")
    return p.parse_args()


def main() -> None:
    args = _parse()

    if args.version:
        print("decloud-dashboard 0.1.0")
        sys.exit(0)

    # Apply CLI overrides to config before importing the app
    from config import cfg

    if args.url:
        cfg.orchestrator_url = args.url.rstrip("/")
    if args.token:
        cfg.token = args.token
    if args.node:
        cfg.node_url = args.node.rstrip("/")
    if args.node_only:
        cfg.node_only = True
    if args.refresh:
        cfg.refresh_interval = args.refresh

    # Validate
    errors = cfg.validate()
    if errors:
        for err in errors:
            print(f"[error] {err}", file=sys.stderr)
        print(
            "\nSet credentials via environment variables or ~/.decloud/config:\n"
            "  DECLOUD_URL=https://orchestrator.example.com\n"
            "  DECLOUD_TOKEN=eyJ...\n"
            "  DECLOUD_NODE_URL=http://localhost:5100  (optional)\n"
            "\nOr run:  decloud dashboard --url URL --token TOKEN",
            file=sys.stderr,
        )
        sys.exit(1)

    from app import DeCloudDashboard
    DeCloudDashboard().run()


if __name__ == "__main__":
    main()
