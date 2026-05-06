"""
__main__.py — CLI entry point.

Run with:  python -m dashboard
Or:        python dashboard/__main__.py

Flags merge into cfg with the precedence:
    CLI flag > environment > ~/.decloud/config

Validation is run before launching the TUI; if no data sources are
configured, the program exits with a helpful message instead of
opening an empty dashboard.
"""

from __future__ import annotations

import argparse
import sys

from config import cfg


__version__ = "2.0.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decloud-dashboard",
        description="DeCloud node operator dashboard",
    )
    parser.add_argument("--url",
        help="Orchestrator URL (overrides DECLOUD_URL)")
    parser.add_argument("--token",
        help="API token (overrides DECLOUD_TOKEN). "
             "Tip: prefer DECLOUD_TOKEN env var to keep it out of shell history.")
    parser.add_argument("--node",
        help="Node Agent URL (overrides DECLOUD_NODE_URL)")
    parser.add_argument("--node-only", action="store_true",
        help="Skip orchestrator entirely; node agent only.")
    parser.add_argument("--refresh", type=int, metavar="SECONDS",
        help="Auto-refresh interval in seconds (default: 5)")
    parser.add_argument("--version", action="version",
        version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    # Apply CLI overrides on top of file+env config.
    if args.url:      cfg.orchestrator_url = args.url.rstrip("/")
    if args.token:    cfg.token = args.token
    if args.node:     cfg.node_url = args.node.rstrip("/")
    if args.node_only: cfg.node_only = True
    if args.refresh:  cfg.refresh_interval = max(1, args.refresh)

    # Validate before opening the TUI — better UX than an empty dashboard.
    issues = cfg.validate()
    if issues:
        print("Configuration error:", file=sys.stderr)
        for i in issues:
            print(f"  • {i}", file=sys.stderr)
        print(
            "\nFix:\n"
            "  • Set DECLOUD_NODE_URL=http://localhost:5100 (the most common case),\n"
            "  • or DECLOUD_URL+DECLOUD_TOKEN for orchestrator access,\n"
            "  • or pass --node / --url / --token on the command line.",
            file=sys.stderr,
        )
        return 2

    from app import DeCloudDashboard
    DeCloudDashboard().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
