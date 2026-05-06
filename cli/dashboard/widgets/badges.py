"""
widgets/badges.py — Inline state badges for VM and obligation states.

Pure rendering functions returning Rich Text — no Textual widget needed,
which keeps DataTable cells lightweight.
"""

from __future__ import annotations

from rich.text import Text

from theme import COLOR, VM_STATE_GLYPH, DOT_FILLED, DOT_OPEN


# Map VM state name → severity class
_VM_STATE_SEV = {
    "Running":   "ok",
    "Starting":  "warn",
    "Pending":   "warn",
    "Creating":  "warn",
    "Stopping":  "warn",
    "Migrating": "warn",
    "Stopped":   "crit",
    "Failed":    "crit",
    "NotFound":  "crit",
    "Paused":    "warn",
    "Deleted":   "unknown",
}

_SEV_COLOR = {
    "ok":      COLOR["ok"],
    "warn":    COLOR["warn"],
    "crit":    COLOR["crit"],
    "info":    COLOR["info"],
    "unknown": COLOR["dim"],
}


def vm_state_badge(state: int | str | None) -> Text:
    """Render a VM state as 'GLYPH State' coloured."""
    if state is None:
        return Text("? Unknown", style=f"{COLOR['dim']}")

    # Allow either the numeric state from /api/vms or the textual one.
    if isinstance(state, int):
        from theme import VM_STATE_GLYPH  # local: enum->name map below
        name = _STATE_INT_TO_NAME.get(state, f"State{state}")
    else:
        name = state

    sev = _VM_STATE_SEV.get(name, "unknown")
    glyph = VM_STATE_GLYPH.get(name, "·")
    color = _SEV_COLOR[sev]
    out = Text()
    out.append(glyph + " ", style=f"{color}")
    out.append(name, style=f"{color}")
    return out


# Mirrors the canonical VmState enum on the server side.
# 0=Pending, 1=Creating, 2=Starting, 3=Running, 4=Paused,
# 5=Stopping, 6=Stopped, 7=Failed, 8=NotFound, 9=Deleted, 10=Migrating
_STATE_INT_TO_NAME = {
    0: "Pending", 1: "Creating", 2: "Starting", 3: "Running",
    4: "Paused", 5: "Stopping", 6: "Stopped", 7: "Failed",
    8: "NotFound", 9: "Deleted", 10: "Migrating",
}


# SystemVmObligation status → severity (for the obligations list)
_OBL_STATUS = {
    "Active":      ("ok",      DOT_FILLED),
    "Healthy":     ("ok",      DOT_FILLED),
    "Deploying":   ("warn",    DOT_FILLED),
    "Provisioning":("warn",    DOT_FILLED),
    "Pending":     ("warn",    DOT_OPEN),
    "Failed":      ("crit",    DOT_FILLED),
    "Unhealthy":   ("crit",    DOT_FILLED),
    "Stopped":     ("crit",    DOT_FILLED),
    "Unknown":     ("unknown", DOT_OPEN),
}


def obligation_badge(status: str | None) -> Text:
    sev, glyph = _OBL_STATUS.get(status or "Unknown", ("unknown", DOT_OPEN))
    color = _SEV_COLOR[sev]
    out = Text()
    out.append(glyph + " ", style=f"{color}")
    out.append(status or "Unknown", style=f"{color}")
    return out
