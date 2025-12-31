#!/bin/bash
#
# DeCloud Node Agent Installation Script
# 
# Installs and configures the Node Agent with minimal dependencies:
# - .NET 8 Runtime
# - KVM/QEMU/libvirt for virtualization
# - WireGuard for overlay networking
# - SSH CA for certificate authentication
#
# PORTS REQUIRED:
# - Agent API (default 5100) - configurable
# - WireGuard (default 51820) - configurable
#
# PORTS NOT REQUIRED:
# - 80/443 - These stay with your existing apps!
#
# Version: 2.0.0
# Architecture: Central ingress via Orchestrator, not on nodes
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/.../install.sh | sudo bash -s -- \
#       --orchestrator https://decloud.stackfi.tech
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

# Required
ORCHESTRATOR_URL=""

# Node Identity
NODE_WALLET=""  # MANDATORY - must be provided
NODE_NAME=$(hostname)
NODE_REGION="default"
NODE_ZONE="default"

# Paths
INSTALL_DIR="/opt/decloud"
CONFIG_DIR="/opt/decloud/publish"
DATA_DIR="/var/lib/decloud/vms"
LOG_DIR="/var/log/decloud"
REPO_URL="https://github.com/bekirmfr/DeCloud.NodeAgent.git"

# Ports (configurable - work with your existing infrastructure)
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

# Update mode (detected if node agent already running)
UPDATE_MODE=false

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

Usage: $0 --orchestrator <url> [options]

Required (MANDATORY):
  --orchestrator <url>   Orchestrator URL (e.g., https://decloud.stackfi.tech)
  --wallet <address>     **MANDATORY** Ethereum wallet address (0x...)
                         Must be valid 42-char address, not null (0x000...000)

Node Identity:
  --wallet <address>     Node operator wallet address
  --name <name>          Node name (default: hostname)
  --region <region>      Region identifier (default: default)
  --zone <zone>          Zone identifier (default: default)

Network (all ports are configurable!):
  --port <port>          Agent API port (default: 5100)
  --wg-port <port>       WireGuard listen port (default: 51820)
  --wg-ip <ip>           WireGuard hub IP (default: 10.10.0.1)
  --skip-wireguard       Skip WireGuard installation
  --no-wireguard-hub     Install WireGuard but don't configure hub

Other:
  --skip-libvirt         Skip libvirt installation (testing only)
  --help                 Show this help message

PORT REQUIREMENTS:
  This installer only needs TWO ports:
  - Agent API port (default 5100) - for orchestrator communication
  - WireGuard port (default 51820) - for overlay network

  Ports 80/443 are NOT required! Your existing web servers stay untouched.
  HTTP ingress is handled centrally by the Orchestrator.

Examples:
  # Basic installation
  $0 --orchestrator https://decloud.stackfi.tech

  # Custom ports (if defaults conflict)
  $0 --orchestrator https://decloud.stackfi.tech --port 5200 --wg-port 51821

  # With wallet and region
  $0 --orchestrator https://decloud.stackfi.tech --wallet 0xYourWallet --region us-east
EOF
}

# ============================================================
# Requirement Checks

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

check_required_params() {
    log_step "Validating required parameters..."
    
    # Check orchestrator URL
    
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

# ============================================================
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_os() {
    log_step "Checking operating system..."
    
    if [ ! -f /etc/os-release ]; then
        log_error "Cannot detect OS. Only Ubuntu/Debian are supported."
        exit 1
    fi
    
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
    
    case $OS in
        ubuntu)
            if [[ "${VERSION_ID%%.*}" -lt 20 ]]; then
                log_error "Ubuntu 20.04 or later required. Found: $VERSION_ID"
                exit 1
            fi
            log_success "Ubuntu $VERSION_ID detected"
            ;;
        debian)
            if [[ "${VERSION_ID%%.*}" -lt 11 ]]; then
                log_error "Debian 11 or later required. Found: $VERSION_ID"
                exit 1
            fi
            log_success "Debian $VERSION_ID detected"
            ;;
        *)
            log_error "Unsupported OS: $OS. Only Ubuntu/Debian are supported."
            exit 1
            ;;
    esac
}

check_architecture() {
    log_step "Checking architecture..."
    
    ARCH=$(uname -m)
    case $ARCH in
        x86_64|amd64)
            log_success "x86_64 architecture detected"
            ;;
        aarch64|arm64)
            log_success "ARM64 architecture detected"
            ;;
        *)
            log_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
}

check_virtualization() {
    log_step "Checking virtualization support..."
    
    if [ "$SKIP_LIBVIRT" = true ]; then
        log_warn "Skipping virtualization check (--skip-libvirt)"
        return
    fi
    
    if grep -E 'vmx|svm' /proc/cpuinfo > /dev/null 2>&1; then
        log_success "Hardware virtualization supported"
    else
        log_warn "Hardware virtualization may not be available"
        log_warn "VMs may run with reduced performance"
    fi
}

check_resources() {
    log_step "Checking system resources..."
    
    CPU_CORES=$(nproc)
    MEMORY_MB=$(free -m | awk '/^Mem:/{print $2}')
    DISK_GB=$(df -BG "$INSTALL_DIR" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo "50")
    
    if [ "$CPU_CORES" -lt 2 ]; then
        log_warn "Only $CPU_CORES CPU cores available (2+ recommended)"
    else
        log_success "$CPU_CORES CPU cores available"
    fi
    
    if [ "$MEMORY_MB" -lt 2048 ]; then
        log_warn "Only ${MEMORY_MB}MB RAM available (4096MB+ recommended)"
    else
        log_success "${MEMORY_MB}MB RAM available"
    fi
    
    if [ "$DISK_GB" -lt 20 ]; then
        log_warn "Only ${DISK_GB}GB disk space available (50GB+ recommended)"
    else
        log_success "${DISK_GB}GB disk space available"
    fi
}

check_ports() {
    log_step "Checking required ports..."
    
    # Check Agent API port
    local agent_port_process=""
    agent_port_process=$(ss -tlnp 2>/dev/null | grep ":${AGENT_PORT} " | sed -n 's/.*users:(("\([^"]*\)".*/\1/p' | head -1)
    
    if [ -n "$agent_port_process" ]; then
        if [ "$agent_port_process" = "decloud-node" ] || [ "$agent_port_process" = "dotnet" ]; then
            log_warn "Node Agent already running on port $AGENT_PORT - will update in place"
            UPDATE_MODE=true
        else
            log_error "Port $AGENT_PORT is already in use by '$agent_port_process'"
            log_info "Use --port <number> to specify a different port"
            exit 1
        fi
    else
        log_success "Port $AGENT_PORT is available (Agent API)"
    fi
    
    # Check WireGuard port
    if [ "$SKIP_WIREGUARD" = false ]; then
        local wg_port_process=""
        wg_port_process=$(ss -ulnp 2>/dev/null | grep ":${WIREGUARD_PORT} " | sed -n 's/.*users:(("\([^"]*\)".*/\1/p' | head -1)
        
        if [ -n "$wg_port_process" ]; then
            # WireGuard already running is fine for updates
            if [ "$wg_port_process" = "wireguard" ] || [[ "$wg_port_process" == *"wg"* ]]; then
                log_success "WireGuard already running on port $WIREGUARD_PORT"
            else
                log_error "Port $WIREGUARD_PORT is already in use by '$wg_port_process'"
                log_info "Use --wg-port <number> to specify a different port"
                exit 1
            fi
        else
            log_success "Port $WIREGUARD_PORT is available (WireGuard)"
        fi
    fi
    
    # Inform about 80/443
    log_info "Ports 80/443 NOT required - ingress handled by Orchestrator"
}

check_network() {
    log_step "Checking network connectivity..."
    
    if curl -s --max-time 5 https://github.com > /dev/null 2>&1; then
        log_success "Internet connectivity OK"
    else
        log_error "Cannot reach github.com. Check internet connection."
        exit 1
    fi
    
    if curl -s --max-time 5 "$ORCHESTRATOR_URL/health" > /dev/null 2>&1; then
        log_success "Orchestrator reachable at $ORCHESTRATOR_URL"
    else
        log_warn "Cannot reach orchestrator at $ORCHESTRATOR_URL"
        log_warn "Make sure orchestrator is running before starting the agent"
    fi
    
    # RELAY ARCHITECTURE: Detect if node is behind CGNAT
    local public_ip=""
    local private_ip=""
    
    public_ip=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "")
    private_ip=$(hostname -I | awk '{print $1}')
    
    if [ -n "$public_ip" ]; then
        log_info "Public IP: $public_ip"
        log_info "Private IP: $private_ip"
        
        # Check if behind NAT/CGNAT
        if [[ "$public_ip" != "$private_ip" ]]; then
            if [[ "$public_ip" =~ ^100\. ]] || \
               [[ "$private_ip" =~ ^10\. ]] || \
               [[ "$private_ip" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] || \
               [[ "$private_ip" =~ ^192\.168\. ]]; then
                log_info "✓ Node appears to be behind NAT/CGNAT"
                log_info "  → Will be assigned to a relay node automatically"
                log_info "  → No public IP required for this node!"
            else
                log_success "Node has public IP - eligible to be a relay node"
                log_info "  → Can provide relay service for CGNAT nodes"
            fi
        else
            log_success "Node has direct public IP"
        fi
    fi
}

# ============================================================
# Installation Functions
# ============================================================
install_base_dependencies() {
    log_step "Installing base dependencies..."
    
    apt-get update -qq
    apt-get install -y -qq \
        curl wget git jq apt-transport-https ca-certificates \
        gnupg lsb-release software-properties-common > /dev/null 2>&1
    
    log_success "Base dependencies installed"
}

install_dotnet() {
    log_step "Installing .NET 8 SDK..."
    
    if command -v dotnet &> /dev/null; then
        DOTNET_VERSION=$(dotnet --version 2>/dev/null | head -1)
        if [[ "$DOTNET_VERSION" == 8.* ]]; then
            log_success ".NET 8 already installed: $DOTNET_VERSION"
            return
        fi
    fi
    
    # Add Microsoft repository
    wget -q https://packages.microsoft.com/config/$OS/$OS_VERSION/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb
    dpkg -i /tmp/packages-microsoft-prod.deb > /dev/null 2>&1
    rm /tmp/packages-microsoft-prod.deb
    
    apt-get update -qq
    apt-get install -y -qq dotnet-sdk-8.0 > /dev/null 2>&1
    
    log_success ".NET 8 SDK installed"
}

install_libvirt() {
    log_step "Installing libvirt/KVM and virtualization tools..."
    
    if [ "$SKIP_LIBVIRT" = true ]; then
        log_warn "Skipping libvirt installation (--skip-libvirt)"
        return
    fi
    
    PACKAGES="qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst"
    PACKAGES="$PACKAGES cloud-image-utils genisoimage qemu-utils"
    PACKAGES="$PACKAGES libguestfs-tools openssh-client"
    PACKAGES="$PACKAGES sysbench"  # ← ADD THIS LINE for CPU benchmarking
    
    apt-get install -y -qq $PACKAGES > /dev/null 2>&1
    
    # Load nbd module
    modprobe nbd max_part=8 2>/dev/null || true
    echo "nbd" >> /etc/modules-load.d/decloud.conf 2>/dev/null || true
    
    # Enable libvirtd
    systemctl enable libvirtd --quiet 2>/dev/null || true
    systemctl start libvirtd 2>/dev/null || true
    
    # Setup default network
    if ! virsh net-info default &>/dev/null; then
        virsh net-define /usr/share/libvirt/networks/default.xml 2>/dev/null || true
    fi
    virsh net-autostart default 2>/dev/null || true
    virsh net-start default 2>/dev/null || true
    
    log_success "libvirt installed and configured"
}

install_wireguard() {
    if [ "$SKIP_WIREGUARD" = true ]; then
        log_warn "Skipping WireGuard installation (--skip-wireguard)"
        return
    fi
    
    log_step "Installing WireGuard..."
    
    apt-get install -y -qq wireguard wireguard-tools > /dev/null 2>&1
    
    log_success "WireGuard installed"
}

configure_wireguard_hub() {
    if [ "$SKIP_WIREGUARD" = true ]; then
        log_info "Skipping WireGuard configuration (--skip-wireguard)"
        return
    fi
    
    log_step "Configuring WireGuard..."
    
    # Ensure config directory exists (for both hub and client modes)
    mkdir -p /etc/wireguard
    chmod 700 /etc/wireguard
    
    if [ "$ENABLE_WIREGUARD_HUB" = false ]; then
        log_info "WireGuard hub disabled (--no-wireguard-hub)"
        log_info "Node will operate in client mode only"
        log_info "→ Suitable for CGNAT nodes that will connect to relays"
        return
    fi
    
    # Generate keys if they don't exist
    if [ ! -f /etc/wireguard/wg0-private.key ]; then
        log_info "Generating WireGuard keypair..."
        wg genkey | tee /etc/wireguard/wg0-private.key | wg pubkey > /etc/wireguard/wg0-public.key
        chmod 600 /etc/wireguard/wg0-private.key
        chmod 644 /etc/wireguard/wg0-public.key
        log_success "WireGuard keys generated"
    fi
    
    WG_PRIVATE_KEY=$(cat /etc/wireguard/wg0-private.key)
    WG_PUBLIC_KEY=$(cat /etc/wireguard/wg0-public.key)
    
    # Create WireGuard hub configuration (for relay nodes)
    cat > /etc/wireguard/wg0.conf << EOFWG
[Interface]
PrivateKey = ${WG_PRIVATE_KEY}
Address = ${WIREGUARD_HUB_IP}/16
ListenPort = ${WIREGUARD_PORT}
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# Peers will be added dynamically via orchestrator commands
# This node can act as a relay for CGNAT nodes
EOFWG
    
    chmod 600 /etc/wireguard/wg0.conf
    
    # Enable IP forwarding (required for relay nodes)
    if ! grep -q "^net.ipv4.ip_forward=1" /etc/sysctl.conf; then
        echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
        sysctl -p > /dev/null 2>&1
    fi
    
    # Start WireGuard (if not already running)
    if ! systemctl is-active --quiet wg-quick@wg0; then
        systemctl enable wg-quick@wg0 > /dev/null 2>&1
        systemctl start wg-quick@wg0 > /dev/null 2>&1
        
        if systemctl is-active --quiet wg-quick@wg0; then
            log_success "WireGuard hub started on ${WIREGUARD_HUB_IP}:${WIREGUARD_PORT}"
            log_info "WireGuard Public Key: ${WG_PUBLIC_KEY}"
        else
            log_warn "WireGuard failed to start - check logs: journalctl -u wg-quick@wg0"
        fi
    else
        log_success "WireGuard already running"
    fi
}

# ============================================================
# SSH CA Setup
# ============================================================
setup_ssh_ca() {
    log_step "Setting up SSH Certificate Authority..."
    
    if [ -f "$SSH_CA_KEY_PATH" ] && [ -f "$SSH_CA_PUB_PATH" ]; then
        log_success "SSH CA already exists"
        return
    fi
    
    mkdir -p "$(dirname $SSH_CA_KEY_PATH)"
    
    ssh-keygen -t ed25519 -f "$SSH_CA_KEY_PATH" -N "" -C "DeCloud SSH CA" > /dev/null 2>&1
    chmod 600 "$SSH_CA_KEY_PATH"
    chmod 644 "$SSH_CA_PUB_PATH"
    
    log_success "SSH CA created"
}

setup_decloud_user() {
    log_step "Setting up decloud user for SSH jump host..."
    
    if id "decloud" &>/dev/null; then
        log_info "User 'decloud' already exists"
    else
        useradd -r -m -s /bin/bash -d /home/decloud decloud
        log_success "User 'decloud' created"
    fi
    
    mkdir -p /home/decloud/.ssh
    chmod 700 /home/decloud/.ssh
    chown -R decloud:decloud /home/decloud/.ssh
    
    # Set proper password hash to enable account
    usermod -p '*' decloud 2>/dev/null || true
}

configure_decloud_sshd() {
    log_step "Configuring SSH for certificate authentication..."
    
    local sshd_config="/etc/ssh/sshd_config"
    
    # Add TrustedUserCAKeys if not present
    if ! grep -q "TrustedUserCAKeys.*$SSH_CA_PUB_PATH" "$sshd_config" 2>/dev/null; then
        echo "" >> "$sshd_config"
        echo "# DeCloud SSH CA" >> "$sshd_config"
        echo "TrustedUserCAKeys $SSH_CA_PUB_PATH" >> "$sshd_config"
    fi
    
    # Add Match block for decloud user
    if ! grep -q "Match User decloud" "$sshd_config" 2>/dev/null; then
        cat >> "$sshd_config" << 'EOF'

# DeCloud user configuration
Match User decloud
    PasswordAuthentication no
    PubkeyAuthentication yes
    AuthorizedKeysFile none
EOF
    fi
    
    # Check AllowUsers directive
    if grep -q "^AllowUsers" "$sshd_config" 2>/dev/null; then
        if ! grep -q "AllowUsers.*decloud" "$sshd_config" 2>/dev/null; then
            sed -i 's/^AllowUsers.*/& decloud/' "$sshd_config"
            log_info "Added decloud to AllowUsers"
        fi
    fi
    
    # Reload SSH
    systemctl reload sshd 2>/dev/null || systemctl reload ssh 2>/dev/null || true
    
    log_success "SSH configured for certificate authentication"
}

# ============================================================
# Application Setup
# ============================================================
create_directories() {
    log_step "Creating directories..."
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p /var/lib/decloud
    
    log_success "Directories created"
}

download_node_agent() {
    log_step "Downloading Node Agent..."
    
    # Stop service if running (update mode)
    if [ "$UPDATE_MODE" = true ]; then
        log_info "Stopping existing node agent service..."
        systemctl stop decloud-node-agent 2>/dev/null || true
        sleep 2
    fi
    
    if [ -d "$INSTALL_DIR/DeCloud.NodeAgent" ]; then
        rm -rf "$INSTALL_DIR/DeCloud.NodeAgent"
    fi
    
    cd "$INSTALL_DIR"
    git clone --depth 1 "$REPO_URL" DeCloud.NodeAgent > /dev/null 2>&1
    
    cd DeCloud.NodeAgent
    COMMIT=$(git rev-parse --short HEAD)
    
    log_success "Code downloaded (commit: $COMMIT)"
}

build_node_agent() {
    log_step "Building Node Agent..."
    
    cd "$INSTALL_DIR/DeCloud.NodeAgent"
    
    dotnet build --configuration Release --verbosity quiet > /dev/null 2>&1
    dotnet publish src/DeCloud.NodeAgent/DeCloud.NodeAgent.csproj \
        --configuration Release \
        --output "$INSTALL_DIR/publish" \
        --verbosity quiet > /dev/null 2>&1
    
    log_success "Node Agent built"
}

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
  "OrchestratorClient": {
    "BaseUrl": "${ORCHESTRATOR_URL}",
    "WalletAddress": "${NODE_WALLET}",
    "Timeout": "00:00:30",
    "CommandPollInterval": "00:00:05"
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
    
    # Display machine ID for reference
    if [ -f "/etc/machine-id" ]; then
        MACHINE_ID=$(cat /etc/machine-id)
        log_info "Machine ID: ${MACHINE_ID}"
        log_info "Node identity = SHA256(machine-id + wallet)"
    fi
    log_success "Configuration created"
}

create_systemd_service() {
    log_step "Creating systemd service..."
    
    cat > /etc/systemd/system/decloud-node-agent.service << EOF
[Unit]
Description=DeCloud Node Agent
After=network.target libvirtd.service
Wants=libvirtd.service

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}/publish
ExecStart=/usr/bin/dotnet ${INSTALL_DIR}/publish/DeCloud.NodeAgent.dll
Restart=always
RestartSec=10
Environment=ASPNETCORE_ENVIRONMENT=Production
Environment=DOTNET_ENVIRONMENT=Production
StandardOutput=append:${LOG_DIR}/nodeagent.log
StandardError=append:${LOG_DIR}/nodeagent.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log_success "Systemd service created"
}

configure_firewall() {
    log_step "Configuring firewall..."
    
    if ! command -v ufw &> /dev/null; then
        log_info "UFW not installed, skipping firewall configuration"
        return
    fi
    
    if ! ufw status | grep -q "Status: active"; then
        log_info "UFW not active, skipping firewall configuration"
        return
    fi
    
    # Always allow Agent API
    ufw allow ${AGENT_PORT}/tcp comment "DeCloud Agent API" > /dev/null 2>&1 || true
    log_success "Firewall: Agent API port ${AGENT_PORT}/tcp"
    
    # WireGuard configuration
    if [ "$SKIP_WIREGUARD" = false ]; then
        # Allow incoming WireGuard connections
        ufw allow ${WIREGUARD_PORT}/udp comment "DeCloud WireGuard" > /dev/null 2>&1 || true
        log_success "Firewall: WireGuard port ${WIREGUARD_PORT}/udp"
        
        # If this is a relay node (has WireGuard hub), allow forwarding
        if [ "$ENABLE_WIREGUARD_HUB" = true ]; then
            # Allow forwarding for relay functionality
            if ! grep -q "^DEFAULT_FORWARD_POLICY=\"ACCEPT\"" /etc/default/ufw; then
                sed -i 's/DEFAULT_FORWARD_POLICY="DROP"/DEFAULT_FORWARD_POLICY="ACCEPT"/' /etc/default/ufw 2>/dev/null || true
                log_info "Firewall: Enabled forwarding (relay node capability)"
            fi
        fi
    fi
    
    log_success "Firewall configured"
}

create_helper_scripts() {
    log_step "Creating helper scripts..."
    
    # WireGuard helper
    cat > /usr/local/bin/decloud-wg << 'EOFWG'
#!/bin/bash
case "$1" in
    status) wg show ;;
    add) wg set wg0 peer "$2" allowed-ips "$3/32" ;;
    remove) wg set wg0 peer "$2" remove ;;
    *) echo "Usage: decloud-wg {status|add <pubkey> <ip>|remove <pubkey>}" ;;
esac
EOFWG
    chmod +x /usr/local/bin/decloud-wg
    
    log_success "Helper scripts created"
}

start_service() {
    log_step "Starting Node Agent service..."
    
    systemctl enable decloud-node-agent --quiet 2>/dev/null || true
    systemctl start decloud-node-agent 2>/dev/null || true
    
    sleep 3
    
    if systemctl is-active --quiet decloud-node-agent; then
        log_success "Node Agent service started"
    else
        log_error "Failed to start Node Agent"
        log_info "Check logs: journalctl -u decloud-node-agent -n 50"
    fi
}

# ============================================================
# Summary
# ============================================================
print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       DeCloud Node Agent Installation Complete!              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  ═══════════════════════════════════════════════════════════"
    echo "  Connection Information:"
    echo "  ═══════════════════════════════════════════════════════════"
    echo "    Orchestrator:  ${ORCHESTRATOR_URL}"
    echo "    Wallet:        ${NODE_WALLET}"
    echo "    Node Name:     ${NODE_NAME}"
    echo "    Region/Zone:   ${NODE_REGION}/${NODE_ZONE}"
    echo ""
    echo "  ═══════════════════════════════════════════════════════════"
    echo "  Network Configuration:"
    echo "  ═══════════════════════════════════════════════════════════"
    echo "    Agent API:     http://$(hostname -I | awk '{print $1}'):${AGENT_PORT}"
    
    if [ "$SKIP_WIREGUARD" = false ]; then
        echo "    WireGuard:     Port ${WIREGUARD_PORT}/udp"
        if [ "$ENABLE_WIREGUARD_HUB" = true ]; then
            local wg_pubkey=$(cat /etc/wireguard/wg0-public.key 2>/dev/null || echo "N/A")
            echo "    WireGuard Hub: ${WIREGUARD_HUB_IP}:${WIREGUARD_PORT}"
            echo "    Public Key:    ${wg_pubkey}"
        else
            echo "    Mode:          Client only (for CGNAT nodes)"
        fi
    fi
    
    echo ""
    echo "  ${CYAN}NOTE: HTTP ingress is handled centrally by the Orchestrator.${NC}"
    echo "    Ports 80/443 are NOT required on nodes."
    echo ""
    
    # ADD RELAY ARCHITECTURE INFO
    echo "  ═══════════════════════════════════════════════════════════"
    echo "  Relay Architecture (CGNAT Support):"
    echo "  ═══════════════════════════════════════════════════════════"
    
    local public_ip=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "unknown")
    local private_ip=$(hostname -I | awk '{print $1}')
    
    if [ "$public_ip" != "unknown" ]; then
        if [[ "$public_ip" != "$private_ip" ]]; then
            echo "    ${YELLOW}✓ CGNAT Node Detected${NC}"
            echo "    Your node is behind NAT/CGNAT"
            echo "    → Will auto-connect to a relay node"
            echo "    → No static IP required!"
            echo "    → Relay fees: ~\$0.001/hour (~\$0.72/month)"
        else
            echo "    ${GREEN}✓ Public IP Node Detected${NC}"
            echo "    Your node has a public IP"
            
            local cpu_cores=$(nproc)
            local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
            
            if [ "$cpu_cores" -ge 16 ] && [ "$memory_gb" -ge 32 ]; then
                echo "    → ${GREEN}Eligible to be a RELAY node!${NC}"
                echo "    → Can earn relay fees from CGNAT nodes"
                echo "    → Estimated capacity: $((cpu_cores / 4)) CGNAT nodes"
            else
                echo "    → Standard compute node"
                echo "    → (Relay requires: 16+ cores, 32GB+ RAM)"
            fi
        fi
    fi
    
    echo ""
    echo "  ═══════════════════════════════════════════════════════════"
    echo "  Commands:"
    echo "  ═══════════════════════════════════════════════════════════"
    echo "    Status:        sudo systemctl status decloud-node-agent"
    echo "    Logs:          sudo journalctl -u decloud-node-agent -f"
    echo "    Restart:       sudo systemctl restart decloud-node-agent"
    if [ "$SKIP_WIREGUARD" = false ]; then
        echo "    WireGuard:     sudo wg show"
        echo "                   sudo systemctl status wg-quick@wg0"
    fi
    echo ""
    echo "  ═══════════════════════════════════════════════════════════"
    echo "  Important Notes:"
    echo "  ═══════════════════════════════════════════════════════════"
    echo "  • Node ID is deterministic (hardware + wallet)"
    echo "  • Relay assignment is automatic (orchestrator decides)"
    echo "  • CGNAT nodes work seamlessly with relay infrastructure"
    echo "  • Keep your wallet address secure and documented"
    echo ""
}

# ============================================================
# Main
# ============================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       DeCloud Node Agent Installer v${VERSION}                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    
    parse_args "$@"
    
    # CRITICAL: Validate required parameters FIRST
    check_required_params
    
    
    # Checks
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