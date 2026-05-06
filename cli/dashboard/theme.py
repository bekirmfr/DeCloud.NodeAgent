"""
theme.py — Visual language for the dashboard.

Single source of truth for colors, symbols, and severity grading.
Every widget reads from here — no ad-hoc hex codes elsewhere.

Design intent: a calm, dense, professional terminal UI in the
spirit of btop, k9s, lazygit. Cyan accent matches the DeCloud
brand from the existing web dashboard (wwwroot/dashboard.css).

Glyph compatibility:
  Most monospace fonts ship with U+2588 FULL BLOCK ("█") and the
  shade characters U+2591–U+2593 ("░▒▓"), but a surprising number
  lack the partial eighth-blocks (U+2581–U+2587).  We default to
  the rich set; users on minimal fonts (e.g. Windows Terminal with
  Consolas) can set DECLOUD_GLYPHS=basic to fall back to a 4-level
  shade-based set that renders everywhere.
"""

from __future__ import annotations

import os


# ─── Severity gradient ────────────────────────────────────────────────────
# Used by gauges, status pills, threshold colouring.
# Thresholds are inclusive lower bounds.
SEV_OK_BELOW   = 70.0   # < 70 %  → ok
SEV_WARN_BELOW = 85.0   # 70–85 %  → warn
                        # ≥ 85 %    → crit


def severity(pct: float) -> str:
    """Return 'ok' | 'warn' | 'crit' for a 0–100 percentage."""
    if pct >= SEV_WARN_BELOW:
        return "crit"
    if pct >= SEV_OK_BELOW:
        return "warn"
    return "ok"


# ─── Colour tokens (Textual rich-style names) ────────────────────────────
# Use semantic names everywhere; map to concrete colours in CSS via $vars.
COLOR = {
    "ok":      "rgb(56,215,132)",     # green
    "warn":    "rgb(255,184,0)",      # amber
    "crit":    "rgb(244,92,92)",      # red
    "info":    "rgb(122,220,255)",    # cyan accent (brand)
    "muted":   "rgb(170,178,195)",    # secondary text — slightly lighter
    "dim":     "rgb(110,116,130)",    # dim text
    "title":   "rgb(255,255,255)",    # high-contrast titles
}


# ─── Symbols (Unicode, not emoji — universal terminal support) ──────────
# Pills and indicators — chosen for monospace alignment.
DOT_FILLED = "●"
DOT_OPEN   = "○"
CHECK      = "✓"
CROSS      = "✗"
WARN_TRI   = "▲"
ARROW_UP   = "↑"
ARROW_DOWN = "↓"
DASH       = "—"

# VM state glyphs
VM_STATE_GLYPH = {
    "Running":   "▶",
    "Starting":  "◐",
    "Pending":   "◌",
    "Creating":  "◐",
    "Stopping":  "◑",
    "Stopped":   "■",
    "Paused":    "‖",
    "Failed":    "✗",
    "NotFound":  "?",
    "Deleted":   "·",
    "Migrating": "↔",
}


# ─── Glyph sets ──────────────────────────────────────────────────────────
# Two histogram-cell variants. Auto-selected via DECLOUD_GLYPHS env var.
#   "rich"  (default) — uses partial eighth blocks for smooth height
#   "basic"           — uses only full block + shade chars (U+2591–U+2593,
#                       U+2588). These are present in essentially every
#                       monospace font including Consolas, Courier New,
#                       Lucida Console, SimSun.

_SPARK_RICH  = " ▁▂▃▄▅▆▇█"   # 9 levels — the canonical TUI sparkline set
_SPARK_BASIC = " ░▒▓█"        # 5 levels — shade chars, very widely supported

_glyph_mode = os.environ.get("DECLOUD_GLYPHS", "rich").lower()
if _glyph_mode not in ("rich", "basic"):
    _glyph_mode = "rich"

# Public API: callers index into SPARK_BLOCKS by ratio (0.0 – 1.0).
# Index 0 is "empty" (space); the last index is the maximum-height block.
SPARK_BLOCKS = _SPARK_RICH if _glyph_mode == "rich" else _SPARK_BASIC


# ─── Refresh history depth ────────────────────────────────────────────────
# How many samples to retain for sparklines.
# At 5 s default refresh, 30 samples ≈ 2.5 min of history.
HISTORY_LEN = 30

