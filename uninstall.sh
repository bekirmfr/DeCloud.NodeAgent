#!/bin/bash
#
# DeCloud Node Agent Uninstall Script
#
# Reverses install.sh completely:
#   - Stops and disables systemd service
#   - Removes all VMs from libvirt + disk
#   - Removes WireGuard interfaces and configs
#   - Removes SSH CA configuration
#   - Removes firewall rules
#   - Removes installed binaries and helper scripts
#   - Removes application directories
#   - Notifies orchestrator of node departure (best-effort)
#   - Optionally deregisters from orchestrator
#
# Usage:
#   sudo bash uninstall.sh [--force] [--keep-vms] [--keep-data] [--keep-wg]
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $1"; }

# ── Paths (must match install.sh) ─────────────────────────────────────────
INSTALL_DIR="/opt/decloud"
DATA_DIR="/var/lib/decloud"
LOG_DIR="/var/log/decloud"
BACKUP_DIR="/var/backups/decloud"
CONFIG_DIR="/etc/decloud"
SERVICE_NAME="decloud-node-agent"
WG_INTERFACE="wg0"    # node agent WG interface
AGENT_PORT=5100

# ── Flags ─────────────────────────────────────────────────────────────────
FORCE=false
KEEP_VMS=false
KEEP_DATA=false
KEEP_WG=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)     FORCE=true;     shift ;;
            --keep-vms)  KEEP_VMS=true;  shift ;;
            --keep-data) KEEP_DATA=true; shift ;;
            --keep-wg)   KEEP_WG=true;   shift ;;
            --help|-h)   show_help; exit 0 ;;
            *) log_error "Unknown option: $1"; show_help; exit 1 ;;
        esac
    done
}

show_help() {
    cat << EOF
DeCloud Node Agent Uninstaller

Usage: sudo bash uninstall.sh [options]

Options:
  --force       Skip confirmation prompts
  --keep-vms    Do not destroy running VMs (useful for migration)
  --keep-data   Preserve /var/lib/decloud (VM disk images, databases)
  --keep-wg     Preserve WireGuard keys and config in /etc/wireguard
  --help        Show this help

Examples:
  sudo bash uninstall.sh                # Full interactive uninstall
  sudo bash uninstall.sh --force        # Non-interactive (CI/automation)
  sudo bash uninstall.sh --keep-vms --keep-data  # Remove agent, keep VMs+data
EOF
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Must run as root: sudo bash uninstall.sh"
        exit 1
    fi
}

confirm() {
    if [[ "$FORCE" == true ]]; then return 0; fi
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║       ⚠️  DESTRUCTIVE OPERATION — NODE AGENT REMOVAL     ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "This will permanently:"
    echo "  • Stop and remove the decloud-node-agent service"
    [[ "$KEEP_VMS"  == false ]] && echo "  • Destroy ALL virtual machines and delete their disk images"
    [[ "$KEEP_DATA" == false ]] && echo "  • Delete all data in $DATA_DIR"
    [[ "$KEEP_WG"   == false ]] && echo "  • Remove WireGuard interfaces and keys"
    echo "  • Remove SSH CA configuration"
    echo "  • Remove all DeCloud binaries and helper scripts"
    echo "  • Remove $INSTALL_DIR and $LOG_DIR"
    echo ""
    read -rp "Type 'REMOVE' to confirm: " input
    [[ "$input" == "REMOVE" ]] || { log_info "Cancelled."; exit 0; }
}

# ── Step 1: Notify orchestrator (best-effort) ─────────────────────────────
notify_orchestrator() {
    log_step "Notifying orchestrator of node departure..."
    local credentials="/etc/decloud/credentials"
    local orchestrator_url=""

    # Read orchestrator URL from config
    if [[ -f "$INSTALL_DIR/publish/appsettings.Production.json" ]]; then
        orchestrator_url=$(grep -oP '"BaseUrl"\s*:\s*"\K[^"]+' \
            "$INSTALL_DIR/publish/appsettings.Production.json" 2>/dev/null || true)
    fi

    if [[ -z "$orchestrator_url" ]]; then
        log_warn "Cannot determine orchestrator URL — skipping notification"
        return
    fi

    local node_id=""
    if [[ -f "$credentials" ]]; then
        node_id=$(grep -oP 'NODE_ID=\K.+' "$credentials" 2>/dev/null || true)
        local api_key
        api_key=$(grep -oP 'API_KEY=\K.+' "$credentials" 2>/dev/null || true)

        if [[ -n "$node_id" && -n "$api_key" ]]; then
            if curl -s -X POST \
                "$orchestrator_url/api/nodes/$node_id/deregister" \
                -H "Authorization: Bearer $api_key" \
                -H "Content-Type: application/json" \
                -d '{"reason":"manual_uninstall"}' \
                --max-time 10 > /dev/null 2>&1; then
                log_success "Orchestrator notified (node $node_id deregistered)"
            else
                log_warn "Could not reach orchestrator — node will timeout naturally"
            fi
        fi
    else
        log_warn "No credentials found — skipping deregistration"
    fi
}

# ── Step 2: Stop service ──────────────────────────────────────────────────
stop_service() {
    log_step "Stopping and disabling node agent service..."
    systemctl stop  "$SERVICE_NAME" 2>/dev/null || true
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
    systemctl daemon-reload
    log_success "Service removed"
}

# ── Step 3: Destroy VMs ───────────────────────────────────────────────────
destroy_vms() {
    if [[ "$KEEP_VMS" == true ]]; then
        log_warn "Skipping VM destruction (--keep-vms)"
        return
    fi

    log_step "Destroying all VMs..."
    if ! command -v virsh &>/dev/null; then
        log_warn "virsh not found — skipping VM cleanup"
        return
    fi

    local vms
    mapfile -t vms < <(virsh list --all --name 2>/dev/null | grep -v '^$' || true)

    if [[ ${#vms[@]} -eq 0 ]]; then
        log_info "No VMs found"
        return
    fi

    for vm in "${vms[@]}"; do
        log_info "Destroying VM: $vm"
        virsh destroy  "$vm" 2>/dev/null || true
        virsh undefine "$vm" --remove-all-storage 2>/dev/null || true
        log_success "  $vm removed"
    done
    log_success "${#vms[@]} VM(s) destroyed"
}

# ── Step 4: WireGuard cleanup ─────────────────────────────────────────────
remove_wireguard() {
    if [[ "$KEEP_WG" == true ]]; then
        log_warn "Skipping WireGuard removal (--keep-wg)"
        return
    fi

    log_step "Removing WireGuard interfaces..."

    # Only remove node-agent-managed interfaces
    local managed=("wg0" "wg-relay" "wg-relay-server" "wg-hub")
    for iface in "${managed[@]}"; do
        if wg show "$iface" &>/dev/null 2>&1; then
            systemctl stop  "wg-quick@${iface}" 2>/dev/null || true
            systemctl disable "wg-quick@${iface}" 2>/dev/null || true
            wg-quick down "$iface" 2>/dev/null || true
            log_success "  Removed interface: $iface"
        fi
        # Back up and remove config
        local conf="/etc/wireguard/${iface}.conf"
        if [[ -f "$conf" ]]; then
            cp "$conf" "${conf}.uninstall-$(date +%Y%m%d-%H%M%S)"
            rm -f "$conf"
        fi
    done

    # Remove node agent WG keys (NOT orchestrator keys)
    rm -f /etc/wireguard/private.key
    rm -f /etc/wireguard/public.key

    # NOTE: /etc/wireguard/orchestrator-*.key are NOT removed
    # — they belong to the orchestrator, not the node agent
    log_success "WireGuard cleaned up"
}

# ── Step 5: SSH CA cleanup ────────────────────────────────────────────────
remove_ssh_ca() {
    log_step "Removing SSH CA configuration..."

    # Remove SSH config additions
    local sshd_config="/etc/ssh/sshd_config"
    if grep -q "DeCloud" "$sshd_config" 2>/dev/null; then
        # Remove the decloud block
        sed -i '/# DeCloud user configuration/,/^$/d' "$sshd_config"
        sed -i '/TrustedUserCAKeys.*decloud/d' "$sshd_config"
        sed -i '/AuthorizedPrincipalsFile.*decloud/d' "$sshd_config"
        systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true
        log_success "SSH config cleaned"
    fi

    # Remove CA keys
    rm -f /etc/decloud/ssh_ca
    rm -f /etc/decloud/ssh_ca.pub
    rm -rf /etc/ssh/auth_principals

    # Remove decloud system user
    if id decloud &>/dev/null; then
        userdel -r decloud 2>/dev/null || userdel decloud 2>/dev/null || true
        log_success "decloud user removed"
    fi
}

# ── Step 6: Firewall rules ────────────────────────────────────────────────
remove_firewall_rules() {
    log_step "Removing firewall rules..."
    if command -v ufw &>/dev/null; then
        ufw delete allow "$AGENT_PORT/tcp" 2>/dev/null || true
        ufw delete allow "51820/udp"       2>/dev/null || true
        ufw delete allow "51821/udp"       2>/dev/null || true
        log_success "UFW rules removed"
    fi
}

# ── Step 7: Binaries and helpers ──────────────────────────────────────────
remove_binaries() {
    log_step "Removing binaries and helper scripts..."
    rm -f /usr/local/bin/decloud
    rm -f /usr/local/bin/decloud-wg
    rm -f /usr/local/bin/cli-decloud-node
    rm -f /usr/local/bin/decloud-relay-nat
    rm -rf /usr/local/share/doc/decloud
    rm -f /etc/profile.d/golang.sh
    log_success "Binaries removed"
}

# ── Step 8: Application directories ──────────────────────────────────────
remove_directories() {
    log_step "Removing application directories..."

    # Install dir (source + publish)
    rm -rf "$INSTALL_DIR/DeCloud.NodeAgent"
    rm -rf "$INSTALL_DIR/publish"

    # Data dir (VM images, SQLite DBs) — optional
    if [[ "$KEEP_DATA" == false ]]; then
        rm -rf "$DATA_DIR"
        log_success "Data directory removed: $DATA_DIR"
    else
        log_warn "Keeping data directory: $DATA_DIR (--keep-data)"
    fi

    # Logs
    rm -rf "$LOG_DIR"

    # Config (credentials, install-params, ssh_ca)
    # Back up install-params before removing in case of reinstall
    if [[ -f "/etc/decloud/install-params" ]]; then
        cp /etc/decloud/install-params /tmp/decloud-install-params.bak
        log_info "Install params backed up to /tmp/decloud-install-params.bak"
    fi
    rm -rf "$CONFIG_DIR"

    log_success "Directories removed"
}

# ── Summary ───────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         ✅ Node Agent Uninstalled Successfully           ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Removed:"
    echo "    • systemd service (decloud-node-agent)"
    [[ "$KEEP_VMS"  == false ]] && echo "    • All virtual machines"
    [[ "$KEEP_WG"   == false ]] && echo "    • WireGuard interfaces and node keys"
    echo "    • SSH CA configuration"
    echo "    • Firewall rules"
    echo "    • Binaries and helper scripts"
    echo "    • Application directories"
    echo ""
    if [[ "$KEEP_DATA" == true ]]; then
        echo -e "  ${YELLOW}Preserved: $DATA_DIR (--keep-data)${NC}"
    fi
    if [[ "$KEEP_WG" == true ]]; then
        echo -e "  ${YELLOW}Preserved: WireGuard config (--keep-wg)${NC}"
    fi
    echo ""
    echo "  Note: Orchestrator WireGuard keys (/etc/wireguard/orchestrator-*.key)"
    echo "        were NOT removed — they belong to the orchestrator."
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────
main() {
    check_root
    parse_args "$@"
    confirm

    notify_orchestrator
    stop_service
    destroy_vms
    remove_wireguard
    remove_ssh_ca
    remove_firewall_rules
    remove_binaries
    remove_directories

    print_summary
}

main "$@"