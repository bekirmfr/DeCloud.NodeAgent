"""
theme.py — Visual language for the dashboard.

Single source of truth for colors, symbols, and severity grading.
Every widget reads from here — no ad-hoc hex codes elsewhere.

Design intent: a calm, dense, professional terminal UI in the
spirit of btop, k9s, lazygit. Cyan accent matches the DeCloud
brand from the existing web dashboard (wwwroot/dashboard.css).
"""

from __future__ import annotations


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
    "muted":   "rgb(140,148,165)",    # secondary text
    "dim":     "rgb(90,96,110)",      # dim text
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

# Box-drawing for sparklines (eighth-block heights)
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


# ─── Refresh history depth ────────────────────────────────────────────────
# How many samples to retain for sparklines.
# At 5 s default refresh, 30 samples ≈ 2.5 min of history.
HISTORY_LEN = 30
