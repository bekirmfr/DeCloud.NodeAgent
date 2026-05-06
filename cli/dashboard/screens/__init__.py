"""Screen registry — label → screen instance."""

from __future__ import annotations

from textual.screen import Screen


def get_screen(label: str) -> Screen | None:
    # Local imports prevent circular-import storms.
    from screens.overview    import OverviewScreen
    from screens.hardware    import HardwareScreen
    from screens.vms         import VmsScreen
    from screens.network     import NetworkScreen
    from screens.firewall    import FirewallScreen
    from screens.services    import ServicesScreen
    from screens.logs        import LogsScreen
    from screens.diagnostics import DiagnosticsScreen
    from screens.settings    import SettingsScreen

    mapping = {
        "Overview":         OverviewScreen,
        "Hardware":         HardwareScreen,
        "Virtual Machines": VmsScreen,
        "Network":          NetworkScreen,
        "Firewall":         FirewallScreen,
        "Services":         ServicesScreen,
        "Logs":             LogsScreen,
        "Diagnostics":      DiagnosticsScreen,
        "Settings":         SettingsScreen,
    }
    cls = mapping.get(label)
    return cls() if cls else None
