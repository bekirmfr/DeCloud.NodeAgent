"""
widgets/nav_input.py — Input subclass that lets navigation keys reach app bindings.

Why this exists:
    Textual's Input widget consumes all printable characters (including digits)
    via check_consume_key — so when an Input has focus, the app's 1–9 screen-
    switch bindings never fire (the keys are typed into the input instead).
    This is correct default behavior for forms, but for our dashboard's
    1–9 = switch screen UX, those keys must always navigate.

    NavInput overrides check_consume_key to let a small allow-list of
    non-text-entry keys pass through to the binding chain. Everything else
    (letters, punctuation) still types normally into the input.

Use NavInput in place of Input for any free-text field on a dashboard screen.
"""

from __future__ import annotations

from textual.widgets import Input


# Keys that must always reach the app/screen bindings regardless of focus.
# Digits drive the 1-9 nav. We deliberately do NOT include 'r', '?' or 'q'
# — those need to remain typeable in a search box.
_NAV_KEYS = frozenset("1234567890")


class NavInput(Input):
    """Input that doesn't consume our nav digit keys."""

    def check_consume_key(self, key: str, character: str | None) -> bool:
        if character is not None and character in _NAV_KEYS:
            return False  # let the binding system handle it
        return super().check_consume_key(key, character)
