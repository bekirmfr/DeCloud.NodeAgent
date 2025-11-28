#!/bin/bash
#
# DeCloud Node Agent Installation Script
# 
# Installs and configures the Node Agent with all dependencies:
# - .NET 8 Runtime
# - KVM/QEMU/libvirt for virtualization
# - WireGuard for overlay networking (with hub auto-configuration)
# - cloud-init tools for VM provisioning
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/bekirmfr/DeCloud.NodeAgent/master/install.sh | sudo bash -s -- --orchestrator http://IP:5050
#

set -e

VERSION="1.2.0"

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

# Default configuration
ORCHESTRATOR_URL=""
INSTALL_DIR="/opt/decloud"
DATA_DIR="/var/lib/decloud"
CONFIG_DIR="/etc/decloud"
WALLET_ADDRESS="0x0000000000000000000000000000000000000000"
AGENT_PORT=5100
NODE_NAME=$(hostname)
REGION="default"
ZONE="default"

# WireGuard configuration
WIREGUARD_PORT=51820
WIREGUARD_INTERFACE="wg-decloud"
WIREGUARD_HUB_IP="10.10.0.1"
WIREGUARD_NETWORK="10.10.0.0/24"
VM_NETWORK="192.168.122.0/24"
SKIP_WIREGUARD=false
ENABLE_WIREGUARD_HUB=true

# Libvirt
SKIP_LIBVIRT=false

# Minimum requirements
MIN_CPU_CORES=2
MIN_MEMORY_MB=2048
MIN_DISK_GB=20

# Detected values
PUBLIC_IP=""
OS=""
VERSION_ID=""
CPU_CORES=""
MEMORY_MB=""
DISK_GB=""
WG_PRIVATE_KEY=""
WG_PUBLIC_KEY=""

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
                WALLET_ADDRESS="$2"
                shift 2
                ;;
            --name)
                NODE_NAME="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --zone)
                ZONE="$2"
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

Required:
  --orchestrator <url>   Orchestrator URL (e.g., http://142.234.200.108:5050)

Optional:
  --wallet <address>     Node operator wallet address (default: zero address)
  --name <name>          Node name (default: hostname)
  --region <region>      Region identifier (default: default)
  --zone <zone>          Zone identifier (default: default)
  --port <port>          Agent API port (default: 5100)

WireGuard Options:
  --wg-port <port>       WireGuard listen port (default: 51820)
  --wg-ip <ip>           WireGuard hub IP (default: 10.10.0.1)
  --skip-wireguard       Skip WireGuard installation entirely
  --no-wireguard-hub     Install WireGuard but don't configure hub

Other:
  --skip-libvirt         Skip libvirt installation (testing only)
  --help                 Show this help message

Examples:
  # Basic installation
  $0 --orchestrator http://142.234.200.108:5050

  # With custom wallet and region
  $0 --orchestrator http://142.234.200.108:5050 --wallet 0xYourWallet --region us-east

  # Without WireGuard
  $0 --orchestrator http://142.234.200.108:5050 --skip-wireguard
EOF
}

# ============================================================
# Requirement Checks
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
    VERSION_ID=$VERSION_ID
    
    case $OS in
        ubuntu)
            if [[ "${VERSION_ID%%.*}" -lt 20 ]]; then
                log_error "Ubuntu 20.04 or later required. Found: $VERSION_ID"
                exit 1
            fi
            ;;
        debian)
            if [[ "${VERSION_ID%%.*}" -lt 11 ]]; then
                log_error "Debian 11 or later required. Found: $VERSION_ID"
                exit 1
            fi
            ;;
        *)
            log_warn "Unsupported OS: $OS. Proceeding anyway, but issues may occur."
            ;;
    esac
    
    log_success "OS: $OS $VERSION_ID"
}

check_architecture() {
    log_step "Checking architecture..."
    
    ARCH=$(uname -m)
    if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
        log_error "Unsupported architecture: $ARCH. Only x86_64 and aarch64 are supported."
        exit 1
    fi
    
    log_success "Architecture: $ARCH"
}

check_virtualization() {
    log_step "Checking virtualization support..."
    
    if [ "$SKIP_LIBVIRT" = true ]; then
        log_warn "Skipping virtualization check (--skip-libvirt)"
        return
    fi
    
    # Check for KVM support
    if [ ! -e /dev/kvm ]; then
        # Try loading the module
        modprobe kvm 2>/dev/null || true
        modprobe kvm_intel 2>/dev/null || modprobe kvm_amd 2>/dev/null || true
        
        if [ ! -e /dev/kvm ]; then
            log_error "KVM not available. Enable virtualization in BIOS/UEFI."
            log_error "Check with: lscpu | grep Virtualization"
            exit 1
        fi
    fi
    
    # Check CPU flags
    if ! grep -qE '(vmx|svm)' /proc/cpuinfo; then
        log_error "CPU does not support hardware virtualization (VT-x/AMD-V)"
        exit 1
    fi
    
    log_success "KVM virtualization available"
}

check_resources() {
    log_step "Checking system resources..."
    
    # CPU cores
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -lt "$MIN_CPU_CORES" ]; then
        log_error "Minimum $MIN_CPU_CORES CPU cores required. Found: $CPU_CORES"
        exit 1
    fi
    log_success "CPU cores: $CPU_CORES"
    
    # Memory
    MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEMORY_MB=$((MEMORY_KB / 1024))
    if [ "$MEMORY_MB" -lt "$MIN_MEMORY_MB" ]; then
        log_error "Minimum ${MIN_MEMORY_MB}MB RAM required. Found: ${MEMORY_MB}MB"
        exit 1
    fi
    log_success "Memory: ${MEMORY_MB}MB"
    
    # Disk space
    DISK_KB=$(df /var | tail -1 | awk '{print $4}')
    DISK_GB=$((DISK_KB / 1024 / 1024))
    if [ "$DISK_GB" -lt "$MIN_DISK_GB" ]; then
        log_error "Minimum ${MIN_DISK_GB}GB free disk space required. Found: ${DISK_GB}GB"
        exit 1
    fi
    log_success "Free disk: ${DISK_GB}GB"
}

check_network() {
    log_step "Checking network connectivity..."
    
    # Check internet access
    if ! ping -c 1 -W 5 8.8.8.8 &> /dev/null; then
        log_error "No internet connectivity"
        exit 1
    fi
    log_success "Internet connectivity OK"
    
    # Check orchestrator reachability
    if [ -n "$ORCHESTRATOR_URL" ]; then
        if curl -s --max-time 10 "${ORCHESTRATOR_URL}/health" > /dev/null 2>&1; then
            log_success "Orchestrator reachable at $ORCHESTRATOR_URL"
        else
            log_warn "Cannot reach orchestrator at $ORCHESTRATOR_URL (may be OK if not started yet)"
        fi
    fi
    
    # Detect public IP
    PUBLIC_IP=$(curl -s --max-time 10 https://api.ipify.org 2>/dev/null || \
                curl -s --max-time 10 https://ifconfig.me 2>/dev/null || \
                hostname -I | awk '{print $1}')
    
    if [ -z "$PUBLIC_IP" ]; then
        log_warn "Could not detect public IP. Using hostname IP."
        PUBLIC_IP=$(hostname -I | awk '{print $1}')
    fi
    log_success "Public IP: $PUBLIC_IP"
}

# ============================================================
# Installation Functions
# ============================================================
install_base_dependencies() {
    log_step "Installing base dependencies..."
    
    apt-get update -qq
    
    PACKAGES="curl wget git jq apt-transport-https ca-certificates gnupg lsb-release"
    apt-get install -y -qq $PACKAGES > /dev/null 2>&1
    
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
    wget -q https://packages.microsoft.com/config/$OS/$VERSION_ID/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb
    dpkg -i /tmp/packages-microsoft-prod.deb > /dev/null 2>&1
    rm /tmp/packages-microsoft-prod.deb
    
    apt-get update -qq
    apt-get install -y -qq dotnet-sdk-8.0 > /dev/null 2>&1
    
    log_success ".NET 8 SDK installed"
}

install_libvirt() {
    log_step "Installing libvirt/KVM..."
    
    if [ "$SKIP_LIBVIRT" = true ]; then
        log_warn "Skipping libvirt installation (--skip-libvirt)"
        return
    fi
    
    PACKAGES="qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst"
    PACKAGES="$PACKAGES cloud-image-utils genisoimage qemu-utils"
    
    apt-get install -y -qq $PACKAGES > /dev/null 2>&1
    
    # Enable and start libvirtd
    systemctl enable libvirtd --quiet
    systemctl start libvirtd
    
    # Ensure default network exists and is active
    if ! virsh net-info default &> /dev/null; then
        virsh net-define /usr/share/libvirt/networks/default.xml > /dev/null 2>&1 || true
    fi
    virsh net-autostart default > /dev/null 2>&1 || true
    virsh net-start default > /dev/null 2>&1 || true
    
    log_success "libvirt/KVM installed and configured"
}

install_wireguard() {
    log_step "Installing WireGuard..."
    
    if [ "$SKIP_WIREGUARD" = true ]; then
        log_warn "Skipping WireGuard installation (--skip-wireguard)"
        return
    fi
    
    apt-get install -y -qq wireguard wireguard-tools > /dev/null 2>&1
    
    log_success "WireGuard installed"
}

check_wireguard_conflicts() {
    log_step "Checking for WireGuard conflicts..."
    
    # Check if our interface already exists
    if ip link show "$WIREGUARD_INTERFACE" &> /dev/null; then
        log_info "Interface $WIREGUARD_INTERFACE already exists"
        
        # Check if it's ours (has our config)
        if [ -f "$WG_DIR/$WIREGUARD_INTERFACE.conf" ]; then
            log_info "Found existing DeCloud WireGuard config, will update"
            wg-quick down "$WIREGUARD_INTERFACE" 2>/dev/null || true
        else
            log_warn "Interface $WIREGUARD_INTERFACE exists but no config found"
        fi
    fi
    
    # Check if port is in use by another WireGuard interface
    local port_in_use=false
    local existing_interfaces=$(wg show interfaces 2>/dev/null || echo "")
    
    for iface in $existing_interfaces; do
        local iface_port=$(wg show "$iface" listen-port 2>/dev/null || echo "")
        if [ "$iface_port" = "$WIREGUARD_PORT" ] && [ "$iface" != "$WIREGUARD_INTERFACE" ]; then
            port_in_use=true
            log_warn "Port $WIREGUARD_PORT already in use by interface: $iface"
            break
        fi
    done
    
    # Also check if port is in use by any other process
    if [ "$port_in_use" = false ]; then
        if ss -uln | grep -q ":${WIREGUARD_PORT} " 2>/dev/null; then
            # Double check it's not our own interface
            if ! wg show "$WIREGUARD_INTERFACE" listen-port 2>/dev/null | grep -q "$WIREGUARD_PORT"; then
                port_in_use=true
                log_warn "Port $WIREGUARD_PORT already in use by another process"
            fi
        fi
    fi
    
    # If port is in use, find an available one
    if [ "$port_in_use" = true ]; then
        log_info "Finding available port..."
        
        local new_port=$((WIREGUARD_PORT + 1))
        local max_port=$((WIREGUARD_PORT + 100))
        
        while [ $new_port -le $max_port ]; do
            local port_free=true
            
            # Check WireGuard interfaces
            for iface in $existing_interfaces; do
                local iface_port=$(wg show "$iface" listen-port 2>/dev/null || echo "")
                if [ "$iface_port" = "$new_port" ]; then
                    port_free=false
                    break
                fi
            done
            
            # Check other processes
            if [ "$port_free" = true ]; then
                if ss -uln | grep -q ":${new_port} " 2>/dev/null; then
                    port_free=false
                fi
            fi
            
            if [ "$port_free" = true ]; then
                WIREGUARD_PORT=$new_port
                log_success "Using available port: $WIREGUARD_PORT"
                break
            fi
            
            ((new_port++))
        done
        
        if [ $new_port -gt $max_port ]; then
            log_error "Could not find available port in range $((WIREGUARD_PORT - 100 + 1))-$max_port"
            log_error "Specify a port manually with --wg-port or use --skip-wireguard"
            return 1
        fi
    else
        log_success "Port $WIREGUARD_PORT is available"
    fi
    
    # Check for IP address conflicts
    local ip_in_use=false
    
    if ip addr show | grep -q "inet ${WIREGUARD_HUB_IP}/" 2>/dev/null; then
        # Check if it's on our interface
        if ! ip addr show "$WIREGUARD_INTERFACE" 2>/dev/null | grep -q "inet ${WIREGUARD_HUB_IP}/"; then
            ip_in_use=true
            log_warn "IP $WIREGUARD_HUB_IP already in use by another interface"
        fi
    fi
    
    if [ "$ip_in_use" = true ]; then
        log_info "Finding available IP in 10.10.x.1 range..."
        
        local subnet=0
        while [ $subnet -le 255 ]; do
            local test_ip="10.10.${subnet}.1"
            
            if ! ip addr show | grep -q "inet ${test_ip}/" 2>/dev/null; then
                WIREGUARD_HUB_IP="$test_ip"
                WIREGUARD_NETWORK="10.10.${subnet}.0/24"
                log_success "Using available IP: $WIREGUARD_HUB_IP"
                break
            fi
            
            ((subnet++))
        done
        
        if [ $subnet -gt 255 ]; then
            log_error "Could not find available IP in 10.10.x.1 range"
            log_error "Specify an IP manually with --wg-ip or use --skip-wireguard"
            return 1
        fi
    else
        log_success "IP $WIREGUARD_HUB_IP is available"
    fi
    
    return 0
}

configure_wireguard_hub() {
    log_step "Configuring WireGuard hub for external VM access..."
    
    if [ "$SKIP_WIREGUARD" = true ]; then
        return
    fi
    
    if [ "$ENABLE_WIREGUARD_HUB" = false ]; then
        log_warn "Skipping WireGuard hub configuration (--no-wireguard-hub)"
        return
    fi
    
    WG_DIR="/etc/wireguard"
    mkdir -p "$WG_DIR"
    
    # Check for conflicts and find available port/IP
    if ! check_wireguard_conflicts; then
        log_error "WireGuard configuration failed due to conflicts"
        log_warn "You can skip WireGuard with --skip-wireguard and configure manually later"
        return 1
    fi
    
    # Generate keys if they don't exist
    if [ ! -f "$WG_DIR/node_private.key" ]; then
        umask 077
        wg genkey > "$WG_DIR/node_private.key"
        cat "$WG_DIR/node_private.key" | wg pubkey > "$WG_DIR/node_public.key"
        log_info "Generated WireGuard keypair"
    fi
    
    WG_PRIVATE_KEY=$(cat "$WG_DIR/node_private.key")
    WG_PUBLIC_KEY=$(cat "$WG_DIR/node_public.key")
    
    # Create WireGuard hub configuration
    cat > "$WG_DIR/$WIREGUARD_INTERFACE.conf" << EOF
# DeCloud WireGuard Hub Configuration
# Generated by install.sh v${VERSION}
# 
# To add a client:
#   decloud-wg add <CLIENT_PUBKEY> <CLIENT_IP>
#
# Example:
#   decloud-wg add ABC123... ${WIREGUARD_HUB_IP%.*}.2

[Interface]
PrivateKey = $WG_PRIVATE_KEY
Address = $WIREGUARD_HUB_IP/24
ListenPort = $WIREGUARD_PORT
SaveConfig = false

# Enable IP forwarding and NAT for VM network access
PostUp = sysctl -w net.ipv4.ip_forward=1
PostUp = iptables -t nat -A POSTROUTING -s $WIREGUARD_NETWORK -d $VM_NETWORK -j MASQUERADE
PostUp = iptables -A FORWARD -i %i -o virbr0 -j ACCEPT
PostUp = iptables -A FORWARD -i virbr0 -o %i -m state --state RELATED,ESTABLISHED -j ACCEPT

PostDown = iptables -t nat -D POSTROUTING -s $WIREGUARD_NETWORK -d $VM_NETWORK -j MASQUERADE || true
PostDown = iptables -D FORWARD -i %i -o virbr0 -j ACCEPT || true
PostDown = iptables -D FORWARD -i virbr0 -o %i -m state --state RELATED,ESTABLISHED -j ACCEPT || true

# Clients are added dynamically using 'decloud-wg add' command
EOF
    
    chmod 600 "$WG_DIR/$WIREGUARD_INTERFACE.conf"
    
    # Open firewall port
    if command -v ufw &> /dev/null; then
        ufw allow $WIREGUARD_PORT/udp > /dev/null 2>&1 || true
    fi
    iptables -I INPUT -p udp --dport $WIREGUARD_PORT -j ACCEPT 2>/dev/null || true
    
    # Enable and start WireGuard
    systemctl enable wg-quick@$WIREGUARD_INTERFACE --quiet 2>/dev/null || true
    wg-quick down $WIREGUARD_INTERFACE 2>/dev/null || true
    wg-quick up $WIREGUARD_INTERFACE
    
    log_success "WireGuard hub configured"
    log_info "WireGuard public key: ${WG_PUBLIC_KEY:0:20}..."
}

# ============================================================
# Application Setup
# ============================================================
create_directories() {
    log_step "Creating directories..."
    
    mkdir -p $INSTALL_DIR
    mkdir -p $DATA_DIR/vms
    mkdir -p $DATA_DIR/images
    mkdir -p $CONFIG_DIR
    
    log_success "Directories created"
}

download_node_agent() {
    log_step "Downloading Node Agent..."
    
    cd $INSTALL_DIR
    
    if [ -d "DeCloud.NodeAgent" ]; then
        log_info "Updating existing installation..."
        cd DeCloud.NodeAgent
        git pull --quiet
    else
        git clone --quiet https://github.com/bekirmfr/DeCloud.NodeAgent.git
        cd DeCloud.NodeAgent
    fi
    
    log_success "Node Agent downloaded"
}

build_node_agent() {
    log_step "Building Node Agent..."
    
    cd $INSTALL_DIR/DeCloud.NodeAgent
    
    dotnet restore --verbosity quiet
    dotnet build -c Release --verbosity quiet
    
    log_success "Node Agent built"
}

create_configuration() {
    log_step "Creating configuration..."
    
    cat > $CONFIG_DIR/appsettings.Production.json << EOF
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning",
      "DeCloud": "Information"
    }
  },
  "AllowedHosts": "*",
  "Kestrel": {
    "Endpoints": {
      "Http": {
        "Url": "http://0.0.0.0:${AGENT_PORT}"
      }
    }
  },
  "Node": {
    "Name": "${NODE_NAME}",
    "Region": "${REGION}",
    "Zone": "${ZONE}",
    "PublicIp": "${PUBLIC_IP}"
  },
  "Libvirt": {
    "VmStoragePath": "${DATA_DIR}/vms",
    "ImageCachePath": "${DATA_DIR}/images",
    "LibvirtUri": "qemu:///system",
    "VncPortStart": 5900
  },
  "Images": {
    "CachePath": "${DATA_DIR}/images",
    "VmStoragePath": "${DATA_DIR}/vms",
    "DownloadTimeout": "00:30:00"
  },
  "WireGuard": {
    "InterfaceName": "${WIREGUARD_INTERFACE}",
    "ConfigPath": "/etc/wireguard",
    "ListenPort": ${WIREGUARD_PORT},
    "Address": "${WIREGUARD_HUB_IP}/24"
  },
  "Heartbeat": {
    "Interval": "00:00:15",
    "OrchestratorUrl": "${ORCHESTRATOR_URL}"
  },
  "CommandProcessor": {
    "PollInterval": "00:00:05"
  },
  "Orchestrator": {
    "BaseUrl": "${ORCHESTRATOR_URL}",
    "ApiKey": "",
    "Timeout": "00:00:30",
    "WalletAddress": "${WALLET_ADDRESS}"
  }
}
EOF

    # Symlink to project
    ln -sf $CONFIG_DIR/appsettings.Production.json \
        $INSTALL_DIR/DeCloud.NodeAgent/src/DeCloud.NodeAgent/appsettings.Production.json
    
    log_success "Configuration created"
}

create_systemd_service() {
    log_step "Creating systemd service..."
    
    cat > /etc/systemd/system/decloud-node-agent.service << EOF
[Unit]
Description=DeCloud Node Agent
Documentation=https://github.com/bekirmfr/DeCloud.NodeAgent
After=network.target libvirtd.service wg-quick@${WIREGUARD_INTERFACE}.service
Wants=libvirtd.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}/DeCloud.NodeAgent
ExecStart=/usr/bin/dotnet run --project src/DeCloud.NodeAgent -c Release --no-build --environment Production
Restart=always
RestartSec=10
Environment=DOTNET_ENVIRONMENT=Production
Environment=ASPNETCORE_ENVIRONMENT=Production

# Security
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=decloud-node-agent

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable decloud-node-agent --quiet
    
    log_success "Systemd service created"
}

configure_firewall() {
    log_step "Configuring firewall..."
    
    # Agent API port
    if command -v ufw &> /dev/null; then
        ufw allow $AGENT_PORT/tcp > /dev/null 2>&1 || true
    fi
    iptables -I INPUT -p tcp --dport $AGENT_PORT -j ACCEPT 2>/dev/null || true
    
    log_success "Firewall configured"
}

start_service() {
    log_step "Starting Node Agent..."
    
    systemctl start decloud-node-agent
    
    # Wait and check
    sleep 5
    
    if systemctl is-active --quiet decloud-node-agent; then
        log_success "Node Agent is running"
    else
        log_error "Node Agent failed to start"
        log_error "Check logs: journalctl -u decloud-node-agent -n 50"
        exit 1
    fi
}

create_client_helper_script() {
    log_step "Creating WireGuard client helper script..."
    
    cat > /usr/local/bin/decloud-wg << 'SCRIPT_EOF'
#!/bin/bash
#
# DeCloud WireGuard Client Management
#

WG_INTERFACE="wg-decloud"
WG_DIR="/etc/wireguard"

# Read config to get actual values
get_config_value() {
    local key="$1"
    local default="$2"
    local config_file="$WG_DIR/$WG_INTERFACE.conf"
    
    if [ -f "$config_file" ]; then
        local value=$(grep "^${key}" "$config_file" 2>/dev/null | awk '{print $3}' | head -1)
        if [ -n "$value" ]; then
            echo "$value"
            return
        fi
    fi
    echo "$default"
}

# Get hub IP from config (strip /24)
get_hub_ip() {
    local addr=$(get_config_value "Address" "10.10.0.1/24")
    echo "${addr%/*}"
}

# Get network from hub IP
get_network() {
    local hub_ip=$(get_hub_ip)
    local prefix="${hub_ip%.*}"
    echo "${prefix}.0/24"
}

# Suggest next client IP
suggest_client_ip() {
    local hub_ip=$(get_hub_ip)
    local prefix="${hub_ip%.*}"
    local suggested=2
    
    # Find next available IP by checking existing peers
    while wg show $WG_INTERFACE allowed-ips 2>/dev/null | grep -q "${prefix}.${suggested}/32"; do
        ((suggested++))
        if [ $suggested -gt 254 ]; then
            suggested=2
            break
        fi
    done
    
    echo "${prefix}.${suggested}"
}

case "${1:-help}" in
    add)
        PUBKEY="$2"
        CLIENT_IP="$3"
        if [ -z "$PUBKEY" ] || [ -z "$CLIENT_IP" ]; then
            SUGGESTED=$(suggest_client_ip)
            echo "Usage: decloud-wg add <public-key> <client-ip>"
            echo "Example: decloud-wg add ABC123...xyz $SUGGESTED"
            exit 1
        fi
        wg set $WG_INTERFACE peer "$PUBKEY" allowed-ips "$CLIENT_IP/32"
        echo "✓ Added client $CLIENT_IP"
        echo ""
        echo "Client can now connect to VMs via 192.168.122.x"
        ;;
        
    remove)
        PUBKEY="$2"
        if [ -z "$PUBKEY" ]; then
            echo "Usage: decloud-wg remove <public-key>"
            exit 1
        fi
        wg set $WG_INTERFACE peer "$PUBKEY" remove
        echo "✓ Removed client"
        ;;
        
    list)
        echo "WireGuard Peers:"
        echo ""
        wg show $WG_INTERFACE
        ;;
        
    client-config)
        CLIENT_IP="${2:-$(suggest_client_ip)}"
        HUB_PUBKEY=$(cat $WG_DIR/node_public.key 2>/dev/null)
        PUBLIC_IP=$(curl -s https://api.ipify.org 2>/dev/null)
        WG_PORT=$(get_config_value "ListenPort" "51820")
        WG_NETWORK=$(get_network)
        
        if [ -z "$HUB_PUBKEY" ]; then
            echo "Error: WireGuard not configured on this node"
            exit 1
        fi
        
        echo ""
        echo "# ============================================"
        echo "# DeCloud WireGuard Client Configuration"
        echo "# ============================================"
        echo "# "
        echo "# 1. Save this as: wg-decloud.conf"
        echo "# 2. Generate your private key: wg genkey"
        echo "# 3. Replace <YOUR_PRIVATE_KEY> below"
        echo "# 4. Import into WireGuard app"
        echo "# 5. Run on this server: decloud-wg add <YOUR_PUBKEY> $CLIENT_IP"
        echo "#"
        echo ""
        echo "[Interface]"
        echo "PrivateKey = <YOUR_PRIVATE_KEY>"
        echo "Address = $CLIENT_IP/24"
        echo ""
        echo "[Peer]"
        echo "PublicKey = $HUB_PUBKEY"
        echo "Endpoint = $PUBLIC_IP:$WG_PORT"
        echo "AllowedIPs = $WG_NETWORK, 192.168.122.0/24"
        echo "PersistentKeepalive = 25"
        echo ""
        ;;
        
    status)
        echo "WireGuard Status:"
        echo ""
        if wg show $WG_INTERFACE &>/dev/null; then
            wg show $WG_INTERFACE
            echo ""
            echo "Hub IP: $(get_hub_ip)"
            echo "Network: $(get_network)"
        else
            echo "Interface $WG_INTERFACE not running"
        fi
        echo ""
        if [ -f "$WG_DIR/node_public.key" ]; then
            echo "Hub Public Key: $(cat $WG_DIR/node_public.key)"
        fi
        ;;
        
    *)
        SUGGESTED=$(suggest_client_ip 2>/dev/null || echo "10.10.0.2")
        echo "DeCloud WireGuard Client Management"
        echo ""
        echo "Usage: decloud-wg <command> [args]"
        echo ""
        echo "Commands:"
        echo "  add <pubkey> <ip>    Add a client (e.g., add ABC... $SUGGESTED)"
        echo "  remove <pubkey>      Remove a client"
        echo "  list                 List all connected clients"
        echo "  client-config [ip]   Generate client config template"
        echo "  status               Show WireGuard status and hub public key"
        echo ""
        echo "Examples:"
        echo "  decloud-wg client-config $SUGGESTED"
        echo "  decloud-wg add aBcDeFg123... $SUGGESTED"
        echo "  decloud-wg list"
        echo ""
        ;;
esac
SCRIPT_EOF

    chmod +x /usr/local/bin/decloud-wg
    log_success "Helper script created: decloud-wg"
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
    log_success "Node Agent v${VERSION} installed successfully!"
    echo ""
    echo "  Node Name:       ${NODE_NAME}"
    echo "  Public IP:       ${PUBLIC_IP}"
    echo "  Agent API:       http://${PUBLIC_IP}:${AGENT_PORT}"
    echo "  Orchestrator:    ${ORCHESTRATOR_URL}"
    echo ""
    
    if [ "$SKIP_WIREGUARD" = false ] && [ "$ENABLE_WIREGUARD_HUB" = true ]; then
        WG_PUBLIC_KEY=$(cat /etc/wireguard/node_public.key 2>/dev/null || echo "")
        if [ -n "$WG_PUBLIC_KEY" ]; then
            echo "  ─────────────────────────────────────────────────────────────"
            echo "  WireGuard Hub (for external VM access):"
            echo "  ─────────────────────────────────────────────────────────────"
            echo "  Hub Overlay IP:  ${WIREGUARD_HUB_IP}"
            echo "  Hub Endpoint:    ${PUBLIC_IP}:${WIREGUARD_PORT}"
            echo "  Hub Public Key:  ${WG_PUBLIC_KEY}"
            echo ""
            echo "  To connect from your machine:"
            echo "    1. Generate client config:  decloud-wg client-config 10.10.0.2"
            echo "    2. Add client to hub:       decloud-wg add <YOUR_PUBKEY> 10.10.0.2"
            echo "    3. Connect and access VMs:  ssh ubuntu@192.168.122.x"
            echo ""
        fi
    fi
    
    echo "Useful commands:"
    echo "  Status:          sudo systemctl status decloud-node-agent"
    echo "  Logs:            sudo journalctl -u decloud-node-agent -f"
    echo "  Restart:         sudo systemctl restart decloud-node-agent"
    if [ "$SKIP_WIREGUARD" = false ]; then
        echo "  WireGuard:       sudo decloud-wg status"
    fi
    echo ""
    echo "Configuration:     ${CONFIG_DIR}/appsettings.Production.json"
    echo "Data directory:    ${DATA_DIR}"
    echo ""
    echo "System resources:"
    echo "  CPU Cores:       ${CPU_CORES}"
    echo "  Memory:          ${MEMORY_MB}MB"
    echo "  Free Disk:       ${DISK_GB}GB"
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
    
    # Validate required arguments
    if [ -z "$ORCHESTRATOR_URL" ]; then
        log_error "Orchestrator URL is required"
        echo ""
        show_help
        exit 1
    fi
    
    # Run checks
    check_root
    check_os
    check_architecture
    check_virtualization
    check_resources
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
    
    # Setup application
    create_directories
    download_node_agent
    build_node_agent
    create_configuration
    create_systemd_service
    configure_firewall
    create_client_helper_script
    start_service
    
    # Done
    print_summary
}

main "$@"