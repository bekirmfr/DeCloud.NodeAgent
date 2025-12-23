#!/bin/bash
#
# DeCloud Node Agent Installation Script v2.1.0
# 
# BREAKING CHANGE: Wallet address is now MANDATORY
# No fallback or auto-generation - operators must provide a valid wallet
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/.../install.sh | sudo bash -s -- \
#       --orchestrator https://decloud.stackfi.tech \
#       --wallet 0xYourWalletAddress
#

set -e

VERSION="2.1.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# ============================================================
# Configuration Defaults
# ============================================================

# Required - NO DEFAULTS!
ORCHESTRATOR_URL=""
NODE_WALLET=""  # ← MUST be provided by operator

# Node Identity
NODE_NAME=$(hostname)
NODE_REGION="default"
NODE_ZONE="default"

# Paths
INSTALL_DIR="/opt/decloud"
CONFIG_DIR="/etc/decloud"
DATA_DIR="/var/lib/decloud/vms"
LOG_DIR="/var/log/decloud"
REPO_URL="https://github.com/bekirmfr/DeCloud.NodeAgent.git"

# Ports
AGENT_PORT=5100
WIREGUARD_PORT=51820

# WireGuard
WIREGUARD_HUB_IP="10.10.0.1"
SKIP_WIREGUARD=false
ENABLE_WIREGUARD_HUB=true

# SSH CA
SSH_CA_KEY_PATH="/etc/decloud/ssh_ca"
SSH_CA_PUB_PATH="/etc/decloud/ssh_ca.pub"

# Other
SKIP_LIBVIRT=false
UPDATE_MODE=false

# ============================================================
# Wallet Validation
# ============================================================

validate_wallet_address() {
    local wallet="$1"
    
    # Check if empty
    if [ -z "$wallet" ]; then
        return 1
    fi
    
    # Check format: must start with 0x and be 42 characters total
    if [[ ! "$wallet" =~ ^0x[0-9a-fA-F]{40}$ ]]; then
        return 1
    fi
    
    # Check if null address
    if [ "$wallet" == "0x0000000000000000000000000000000000000000" ]; then
        return 1
    fi
    
    return 0
}

# ============================================================
# Argument Parsing
# ============================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --orchestrator)
                ORCHESTRATOR_URL="$2"
                shift 2
                ;;
            --wallet)
                NODE_WALLET="$2"
                shift 2
                ;;
            --name)
                NODE_NAME="$2"
                shift 2
                ;;
            --region)
                NODE_REGION="$2"
                shift 2
                ;;
            --zone)
                NODE_ZONE="$2"
                shift 2
                ;;
            --port)
                AGENT_PORT="$2"
                shift 2
                ;;
            --wg-port)
                WIREGUARD_PORT="$2"
                shift 2
                ;;
            --wg-ip)
                WIREGUARD_HUB_IP="$2"
                shift 2
                ;;
            --skip-wireguard)
                SKIP_WIREGUARD=true
                shift
                ;;
            --no-wireguard-hub)
                ENABLE_WIREGUARD_HUB=false
                shift
                ;;
            --skip-libvirt)
                SKIP_LIBVIRT=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
DeCloud Node Agent Installer v${VERSION}

Usage: $0 --orchestrator <url> --wallet <address> [options]

${RED}REQUIRED:${NC}
  --orchestrator <url>   Orchestrator URL (e.g., https://decloud.stackfi.tech)
  --wallet <address>     ${RED}MANDATORY${NC} Ethereum wallet address (0x...)
                         - Must be valid 42-character Ethereum address
                         - Cannot be null address (0x000...000)
                         - Used for node identity and billing

Node Identity:
  --name <name>          Node name (default: hostname)
  --region <region>      Region identifier (default: default)
  --zone <zone>          Zone identifier (default: default)

Network (all ports are configurable):
  --port <port>          Agent API port (default: 5100)
  --wg-port <port>       WireGuard listen port (default: 51820)
  --wg-ip <ip>           WireGuard hub IP (default: 10.10.0.1)
  --skip-wireguard       Skip WireGuard installation
  --no-wireguard-hub     Install WireGuard but don't configure hub

Other:
  --skip-libvirt         Skip libvirt installation (testing only)
  --help                 Show this help message

${YELLOW}WALLET ADDRESS REQUIREMENT:${NC}
  Starting from v2.1.0, wallet address is MANDATORY for node identity.
  - Node ID is deterministically generated from hardware + wallet
  - Same hardware + same wallet = always same node ID
  - Ensures stable node identity across restarts
  - No fallback or auto-generation available

${YELLOW}PORT REQUIREMENTS:${NC}
  This installer only needs TWO ports:
  - Agent API port (default 5100) - for orchestrator communication
  - WireGuard port (default 51820) - for overlay network

  Ports 80/443 are NOT required! Your existing web servers stay untouched.
  HTTP ingress is handled centrally by the Orchestrator.

${GREEN}Examples:${NC}
  # Basic installation
  $0 --orchestrator https://decloud.stackfi.tech \\
     --wallet 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb

  # Custom ports and region
  $0 --orchestrator https://decloud.stackfi.tech \\
     --wallet 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb \\
     --port 5200 --region us-east --name my-node-1

${CYAN}Get a wallet address:${NC}
  - MetaMask: https://metamask.io
  - Generate with OpenSSL: openssl rand -hex 20 | awk '{print "0x" \$1}'
  - Use your existing Ethereum wallet address
EOF
}

# ============================================================
# Requirement Checks
# ============================================================

check_required_params() {
    log_step "Validating required parameters..."
    
    # Check orchestrator URL
    if [ -z "$ORCHESTRATOR_URL" ]; then
        log_error "Orchestrator URL is required!"
        log_error "Use: --orchestrator https://your-orchestrator.com"
        echo ""
        show_help
        exit 1
    fi
    
    # Check wallet address
    if [ -z "$NODE_WALLET" ]; then
        log_error "❌ Wallet address is REQUIRED!"
        echo ""
        log_error "Starting from v2.1.0, wallet address is mandatory."
        log_error "No fallback or auto-generation available."
        echo ""
        log_info "Usage: $0 --orchestrator <url> --wallet 0xYourAddress"
        echo ""
        log_info "Get a wallet:"
        log_info "  • Use your existing Ethereum wallet"
        log_info "  • Create one at https://metamask.io"
        log_info "  • Generate: openssl rand -hex 20 | awk '{print \"0x\" \$1}'"
        echo ""
        exit 1
    fi
    
    # Validate wallet format
    if ! validate_wallet_address "$NODE_WALLET"; then
        log_error "❌ Invalid wallet address: $NODE_WALLET"
        echo ""
        log_error "Wallet address must:"
        log_error "  • Start with '0x'"
        log_error "  • Be exactly 42 characters (0x + 40 hex digits)"
        log_error "  • Not be the null address (0x000...000)"
        echo ""
        log_info "Example valid wallet: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
        echo ""
        exit 1
    fi
    
    log_success "Orchestrator: $ORCHESTRATOR_URL"
    log_success "Wallet:       $NODE_WALLET"
    log_success "Node Name:    $NODE_NAME"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# [Rest of the functions remain the same - check_os, check_architecture, etc.]
# ... (keeping the file manageable, use the original install.sh for these)

# ============================================================
# Application Setup - Updated Configuration
# ============================================================

create_configuration() {
    log_step "Creating configuration..."

    cat > "${CONFIG_DIR}/appsettings.Production.json" << EOF
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning",
      "DeCloud": "Debug"
    }
  },
  "Urls": "http://0.0.0.0:${AGENT_PORT}",
  "Orchestrator": {
    "BaseUrl": "${ORCHESTRATOR_URL}",
    "WalletAddress": "${NODE_WALLET}",
    "HeartbeatIntervalSeconds": 30,
    "CommandPollIntervalSeconds": 5
  },
  "Node": {
    "Name": "${NODE_NAME}",
    "Region": "${NODE_REGION}",
    "Zone": "${NODE_ZONE}"
  },
  "WireGuard": {
    "Interface": "wg0",
    "ConfigPath": "/etc/wireguard/wg0.conf",
    "ListenPort": ${WIREGUARD_PORT}
  },
  "Libvirt": {
    "Uri": "qemu:///system",
    "VmStoragePath": "${DATA_DIR}",
    "ImageCachePath": "${DATA_DIR}/images",
    "VncPortStart": 5900,
    "ReconcileOnStartup": true
  },
  "SshCa": {
    "PrivateKeyPath": "${SSH_CA_KEY_PATH}",
    "PublicKeyPath": "${SSH_CA_PUB_PATH}"
  }
}
EOF

    chmod 640 "${CONFIG_DIR}/appsettings.Production.json"
    log_success "Configuration created"
    
    # Display machine ID for reference
    if [ -f "/etc/machine-id" ]; then
        MACHINE_ID=$(cat /etc/machine-id)
        log_info "Machine ID: ${MACHINE_ID}"
        log_info "Node identity = SHA256(machine-id + wallet)"
    fi
}

# ============================================================
# Summary - Updated
# ============================================================
print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       DeCloud Node Agent Installation Complete!              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Node Agent:      http://localhost:${AGENT_PORT}"
    echo "  Orchestrator:    ${ORCHESTRATOR_URL}"
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Node Identity (Deterministic):"
    echo "  ─────────────────────────────────────────────────────────────"
    echo "    Wallet:        ${NODE_WALLET}"
    echo "    Machine ID:    $(cat /etc/machine-id 2>/dev/null || echo 'unknown')"
    echo "    Node Name:     ${NODE_NAME}"
    echo ""
    echo -e "    ${GREEN}Node ID is deterministically generated from:${NC}"
    echo "    SHA256(machine-id + wallet)"
    echo "    → Same hardware + same wallet = always same node ID"
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Ports Used:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo "    Agent API:     ${AGENT_PORT}/tcp"
    if [ "$SKIP_WIREGUARD" = false ]; then
        echo "    WireGuard:     ${WIREGUARD_PORT}/udp"
    fi
    echo ""
    echo -e "    ${GREEN}Ports 80/443: NOT USED${NC} - Your existing apps are safe!"
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Commands:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo "    Status:        sudo systemctl status decloud-node-agent"
    echo "    Logs:          sudo journalctl -u decloud-node-agent -f"
    echo "    Restart:       sudo systemctl restart decloud-node-agent"
    echo "    View Config:   sudo cat ${CONFIG_DIR}/appsettings.Production.json"
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Files:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo "    Configuration: ${CONFIG_DIR}/appsettings.Production.json"
    echo "    Data:          ${DATA_DIR}"
    echo "    Logs:          ${LOG_DIR}/nodeagent.log"
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Resources:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo "    CPU Cores:     ${CPU_CORES}"
    echo "    Memory:        ${MEMORY_MB}MB"
    echo "    Disk:          ${DISK_GB}GB"
    echo ""
    echo "  ${YELLOW}IMPORTANT:${NC}"
    echo "  • Node ID is deterministic (hardware + wallet)"
    echo "  • Changing wallet = new node ID = requires VM migration"
    echo "  • Keep your wallet address secure and documented"
    echo ""
}

# ============================================================
# Main - Updated
# ============================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       DeCloud Node Agent Installer v${VERSION}                    ║"
    echo "║       ${RED}Wallet Address Now REQUIRED${NC}                           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    
    parse_args "$@"
    
    # CRITICAL: Validate required parameters FIRST
    check_required_params
    
    # Other checks
    check_root
    check_os
    check_architecture
    check_virtualization
    check_resources
    check_ports
    check_network
    
    echo ""
    log_info "All requirements met. Starting installation..."
    echo ""
    
    # Install dependencies
    install_base_dependencies
    install_dotnet
    install_libvirt
    install_wireguard
    configure_wireguard_hub
    
    # SSH CA
    setup_ssh_ca
    setup_decloud_user
    configure_decloud_sshd
    
    # Application
    create_directories
    download_node_agent
    build_node_agent
    create_configuration
    create_systemd_service
    configure_firewall
    create_helper_scripts
    start_service
    
    # Done
    print_summary
}

main "$@"