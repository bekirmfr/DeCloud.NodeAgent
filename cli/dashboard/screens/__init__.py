"""Screen registry — maps label names to screen instances."""

from __future__ import annotations
from textual.screen import Screen


def get_screen(label: str) -> Screen | None:
    from screens.dashboard  import DashboardScreen
    from screens.nodes      import NodesScreen
    from screens.vms        import VmsScreen
    from screens.system_vms import SystemVmsScreen
    from screens.network    import NetworkScreen
    from screens.ingress    import IngressScreen
    from screens.billing    import BillingScreen
    from screens.logs       import LogsScreen
    from screens.settings   import SettingsScreen

    mapping = {
        "Dashboard":        DashboardScreen,
        "Nodes":            NodesScreen,
        "Virtual Machines": VmsScreen,
        "System VMs":       SystemVmsScreen,
        "Networking":       NetworkScreen,
        "Ingress Routes":   IngressScreen,
        "Billing":          BillingScreen,
        "Live Logs":        LogsScreen,
        "Settings":         SettingsScreen,
    }
    cls = mapping.get(label)
    return cls() if cls else None
