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
            --skip-wireguard)
                SKIP_WIREGUARD=true
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
  --skip-wireguard       Skip WireGuard installation

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

configure_wireguard_keys() {
    if [ "$SKIP_WIREGUARD" = true ]; then
        log_info "Skipping WireGuard setup (--skip-wireguard)"
        return
    fi
    
    log_step "Preparing WireGuard environment..."
    
    # Create config directory
    mkdir -p /etc/wireguard
    chmod 700 /etc/wireguard
    
    # Generate generic keypair if doesn't exist
    # Note: Keys are NOT interface-specific anymore
    if [ ! -f /etc/wireguard/private.key ]; then
        log_info "Generating WireGuard keypair..."
        wg genkey | tee /etc/wireguard/private.key | wg pubkey > /etc/wireguard/public.key
        chmod 600 /etc/wireguard/private.key
        chmod 644 /etc/wireguard/public.key
        log_success "WireGuard keys generated"
    else
        log_success "WireGuard keys already exist"
    fi
    
    # Enable IP forwarding (needed for potential relay role)
    if ! grep -q "^net.ipv4.ip_forward=1" /etc/sysctl.conf; then
        echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
        sysctl -p > /dev/null 2>&1
        log_info "IP forwarding enabled (for relay capability)"
    fi
    
    log_success "WireGuard environment ready"
    log_info "→ Interface configuration will be automatic based on node role"
    log_info "   CGNAT nodes → wg-relay tunnel"
    log_info "   Relay VMs → wg-relay-server"
    log_info "   Regular nodes → wg-hub (optional)"
}

# ============================================================
# WalletConnect CLI Installation
# ============================================================

# =============================================================================
# UPDATED INSTALL.SH FUNCTIONS FOR REAL WALLETCONNECT
# =============================================================================

install_python_dependencies() {
    log_step "Installing Python dependencies for authentication..."
    
    # =====================================================
    # STEP 1: Check Python installation
    # =====================================================
    if ! command -v python3 &> /dev/null; then
        log_info "Installing Python..."
        apt-get install -y -qq python3 python3-pip > /dev/null 2>&1
    fi
    
    local python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
    local major=$(echo "$python_version" | cut -d'.' -f1)
    local minor=$(echo "$python_version" | cut -d'.' -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 8 ]); then
        log_error "Python 3.8+ required, found $python_version"
        exit 1
    fi
    
    log_info "Python $python_version detected"
    
    # =====================================================
    # STEP 2: Ensure pip is installed
    # =====================================================
    log_info "Checking pip installation..."
    
    # Check if pip module exists
    if ! python3 -m pip --version &> /dev/null; then
        log_info "pip not found, installing..."
        
        # Method 1: Try apt-get (works on Ubuntu/Debian)
        if apt-get install -y python3-pip python3-dev > /dev/null 2>&1; then
            log_info "✓ pip installed via apt-get"
        else
            # Method 2: Try get-pip.py (fallback for systems without apt package)
            log_info "Trying alternative pip installation..."
            if curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py 2>/dev/null; then
                python3 /tmp/get-pip.py > /dev/null 2>&1
                rm -f /tmp/get-pip.py
                log_info "✓ pip installed via get-pip.py"
            else
                log_error "Failed to install pip"
                log_info "Manual installation required:"
                log_info "  sudo apt-get install -y python3-pip"
                return 1
            fi
        fi
    else
        log_info "✓ pip already installed"
    fi
    
    # =====================================================
    # STEP 3: Upgrade pip
    # =====================================================
    log_info "Upgrading pip..."
    # Try with environment variable for Ubuntu 24.04+ (PEP 668)
    if PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --upgrade pip --quiet 2>/dev/null; then
        log_info "✓ pip upgraded"
    elif python3 -m pip install --upgrade pip --quiet 2>/dev/null; then
        log_info "✓ pip upgraded"
    else
        log_info "⚠ pip upgrade skipped (not critical)"
    fi
    
    # =====================================================
    # STEP 4: Install dependencies (multiple methods)
    # =====================================================
    log_info "Installing wallet authentication libraries..."
    
    local PACKAGES="web3 eth-account requests qrcode pillow"
    local INSTALL_SUCCESS=false
    
    # Detect Ubuntu 24.04+ or Debian 12+ (PEP 668 enabled)
    local USE_BREAK_PACKAGES=false
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" && "${VERSION_ID}" > "23" ]] || \
           [[ "$ID" == "debian" && "${VERSION_ID}" > "11" ]]; then
            USE_BREAK_PACKAGES=true
            log_info "Detected PEP 668 system (Ubuntu 24.04+)"
        fi
    fi
    
    # Method 1: For Ubuntu 24.04+ - Install system-wide with environment variable
    if [ "$USE_BREAK_PACKAGES" = true ]; then
        log_info "Installing packages system-wide..."
        # Use environment variable to bypass PEP 668 AND force system-wide installation
        # Use --ignore-installed to avoid conflicts with system packages like typing-extensions
        if PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --ignore-installed $PACKAGES 2>/dev/null; then
            INSTALL_SUCCESS=true
            log_info "✓ Installed system-wide (PEP 668 bypass)"
        fi
    fi
    
    # Method 2: Try with --user flag (works on most older systems)
    if [ "$INSTALL_SUCCESS" = false ]; then
        if python3 -m pip install --user $PACKAGES --quiet 2>/dev/null; then
            INSTALL_SUCCESS=true
            log_info "✓ Installed with --user flag"
        fi
    fi
    
    # Method 3: Try without --user (for systems where --user doesn't work)
    if [ "$INSTALL_SUCCESS" = false ]; then
        log_info "Trying system-wide installation..."
        if python3 -m pip install $PACKAGES --quiet 2>/dev/null; then
            INSTALL_SUCCESS=true
            log_info "✓ Installed system-wide"
        fi
    fi
    
    # Method 4: Try with --break-system-packages flag
    if [ "$INSTALL_SUCCESS" = false ]; then
        log_info "Trying with --break-system-packages flag..."
        if PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --ignore-installed $PACKAGES --quiet 2>/dev/null; then
            INSTALL_SUCCESS=true
            log_info "✓ Installed with PIP_BREAK_SYSTEM_PACKAGES"
        fi
    fi
    
    # Method 5: Install what we can from apt, rest from pip with environment variable
    if [ "$INSTALL_SUCCESS" = false ]; then
        log_info "Trying hybrid apt + pip installation..."
        apt-get install -y python3-requests python3-pil > /dev/null 2>&1 || true
        if PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --ignore-installed web3 eth-account qrcode --quiet 2>/dev/null; then
            INSTALL_SUCCESS=true
            log_info "✓ Installed via apt + pip (system-wide)"
        elif python3 -m pip install --user web3 eth-account qrcode --quiet 2>/dev/null; then
            INSTALL_SUCCESS=true
            log_info "✓ Installed via apt + pip (--user)"
        else
            log_warn "Some packages may not have installed correctly"
        fi
    fi
    
    # =====================================================
    # STEP 5: Verify installation
    # =====================================================
    log_info "Verifying installation..."
    
    local VERIFY_FAILED=false
    
    # Check each package
    if python3 -c "import web3" 2>/dev/null; then
        log_info "✓ web3 installed"
    else
        log_warn "✗ web3 not found"
        VERIFY_FAILED=true
    fi
    
    if python3 -c "import eth_account" 2>/dev/null; then
        log_info "✓ eth-account installed"
    else
        log_warn "✗ eth-account not found"
        VERIFY_FAILED=true
    fi
    
    if python3 -c "import requests" 2>/dev/null; then
        log_info "✓ requests installed"
    else
        log_warn "✗ requests not found"
        VERIFY_FAILED=true
    fi
    
    if python3 -c "import qrcode" 2>/dev/null; then
        log_info "✓ qrcode installed"
    else
        log_warn "✗ qrcode not found"
        VERIFY_FAILED=true
    fi
    
    if python3 -c "from PIL import Image" 2>/dev/null; then
        log_info "✓ pillow installed"
    else
        log_warn "✗ pillow not found"
        VERIFY_FAILED=true
    fi
    
    if [ "$VERIFY_FAILED" = true ]; then
        log_error "Failed to install Python dependencies"
        echo ""
        log_info "Manual installation required. For Ubuntu 24.04+ use:"
        log_info "  sudo PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --ignore-installed web3 eth-account requests qrcode pillow"
        echo ""
        log_info "Or use the --break-system-packages flag:"
        log_info "  python3 -m pip install --break-system-packages --ignore-installed web3 eth-account requests qrcode pillow"
        echo ""
        log_info "For older Ubuntu/Debian use:"
        log_info "  python3 -m pip install --user web3 eth-account requests qrcode pillow"
        echo ""
        return 1
    fi
    
    log_success "✓ Wallet authentication libraries ready (with QR code support)"
    return 0
}

install_walletconnect_cli() {
    log_step "Installing DeCloud authentication CLI..."
    
    # The CLI is in the repository we just cloned
    local cli_source="$INSTALL_DIR/DeCloud.NodeAgent/cli/cli-decloud-node"
    local cli_dest="/usr/local/bin/cli-decloud-node"
    
    if [ ! -f "$cli_source" ]; then
        log_error "CLI script not found at $cli_source"
        log_error "Repository may be incomplete"
        exit 1
    fi
    
    # Copy CLI to /usr/local/bin
    cp "$cli_source" "$cli_dest"
    chmod +x "$cli_dest"
    
    # Verify CLI works
    if cli-decloud-node version &> /dev/null; then
        local cli_version=$(cli-decloud-node version 2>/dev/null | awk '{print $NF}')
        log_success "Authentication CLI installed"
        
        # Check which version (show feature)
        if cli-decloud-node version 2>/dev/null | grep -q "WalletConnect"; then
            log_info "CLI: WalletConnect Edition (${cli_version})"
        else
            log_info "CLI: $(basename $cli_source) (v${cli_version})"
        fi
    else
        log_error "CLI installation failed - testing returned error"
        return 1
    fi
    
    return 0
}

run_node_authentication() {
    log_step "Authenticating node with WalletConnect..."
    
    echo ""
    log_info "You will now authorize this node using your wallet."
    log_info "This process uses WalletConnect - you'll scan a QR code with your mobile wallet."
    echo ""
    
    # Run the CLI login
    if cli-decloud-node login --orchestrator "$ORCHESTRATOR_URL"; then
        log_success "Node authenticated successfully"
        return 0
    else
        log_error "Authentication failed"
        echo ""
        log_info "You can re-run authentication later with:"
        log_info "  sudo cli-decloud-node login"
        echo ""
        return 1
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

# Fixed build_node_agent function for install.sh
# Replace the existing function with this version

build_node_agent() {
    log_step "Building Node Agent..."
    
    # Verify we're in the right directory
    if [ ! -d "$INSTALL_DIR/DeCloud.NodeAgent" ]; then
        log_error "Repository directory not found: $INSTALL_DIR/DeCloud.NodeAgent"
        return 1
    fi
    
    cd "$INSTALL_DIR/DeCloud.NodeAgent"
    
    # Verify project structure
    if [ ! -f "src/DeCloud.NodeAgent/DeCloud.NodeAgent.csproj" ]; then
        log_error "Project file not found: src/DeCloud.NodeAgent/DeCloud.NodeAgent.csproj"
        log_info "Directory contents:"
        ls -la
        return 1
    fi
    
    # Clean previous build (show errors)
    log_info "Cleaning previous build..."
    if ! dotnet clean --configuration Release 2>&1 | grep -i "error" ; then
        log_info "Clean completed"
    fi
    
    # Build (capture and show errors)
    log_info "Building project..."
    BUILD_OUTPUT=$(dotnet build --configuration Release 2>&1)
    BUILD_EXIT=$?
    
    if [ $BUILD_EXIT -ne 0 ]; then
        log_error "Build failed!"
        echo "$BUILD_OUTPUT" | tail -30
        return 1
    fi
    
    log_info "Build succeeded"
    
    # Publish (capture and show errors)
    log_info "Publishing to $INSTALL_DIR/publish..."
    PUBLISH_OUTPUT=$(dotnet publish src/DeCloud.NodeAgent/DeCloud.NodeAgent.csproj \
        --configuration Release \
        --output "$INSTALL_DIR/publish" \
        --no-build \
        2>&1)
    PUBLISH_EXIT=$?
    
    if [ $PUBLISH_EXIT -ne 0 ]; then
        log_error "Publish failed!"
        echo "$PUBLISH_OUTPUT" | tail -30
        return 1
    fi
    
    # Verify publish output
    if [ ! -f "$INSTALL_DIR/publish/DeCloud.NodeAgent.dll" ]; then
        log_error "Published DLL not found!"
        log_info "Publish directory contents:"
        ls -la "$INSTALL_DIR/publish" 2>&1 || echo "Directory does not exist"
        return 1
    fi
    
    # Show publish statistics
    local file_count=$(find "$INSTALL_DIR/publish" -type f | wc -l)
    local dir_size=$(du -sh "$INSTALL_DIR/publish" | cut -f1)
    log_info "Published $file_count files ($dir_size)"
    
    # Verify CloudInit templates were copied
    if [ -d "$INSTALL_DIR/publish/CloudInit/Templates" ]; then
        local template_count=$(find "$INSTALL_DIR/publish/CloudInit/Templates" -name "*.yaml" | wc -l)
        if [ $template_count -gt 0 ]; then
            log_success "CloudInit templates included ($template_count templates)"
        else
            log_warn "No CloudInit templates found in publish output!"
        fi
    else
        log_warn "CloudInit/Templates directory not found in publish output"
    fi
    
    log_success "Node Agent built successfully"
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
        
        # Enable forwarding for potential relay role
        if ! grep -q "^DEFAULT_FORWARD_POLICY=\"ACCEPT\"" /etc/default/ufw; then
            sed -i 's/DEFAULT_FORWARD_POLICY="DROP"/DEFAULT_FORWARD_POLICY="ACCEPT"/' /etc/default/ufw 2>/dev/null || true
            log_info "Firewall: Forwarding enabled (for relay capability)"
        fi
    fi


    
    log_success "Firewall configured"
}

install_relay_nat_support() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Installing Relay NAT Support"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Install conntrack if not present
    if ! command -v conntrack &> /dev/null; then
        echo "→ Installing conntrack-tools..."
        apt-get install -y conntrack > /dev/null 2>&1 || {
            echo "⚠ Warning: Could not install conntrack (optional)"
        }
    fi
    
    # Copy relay NAT manager script from file
    echo "→ Installing relay NAT manager..."

    # The relay NAT manager is in the repository we just cloned
    local file_source="$INSTALL_DIR/DeCloud.NodeAgent/decloud-relay-nat"
    local file_dest="/usr/local/bin/decloud-relay-nat"
    
    if [ ! -f "file_source" ]; then
        log_error "Relay NAT manager script not found at $file_source"
        log_error "Repository may be incomplete"
        exit 1
    fi
    
    # Copy relay NAT manager script to /usr/local/bin
    cp "$file_source" "$file_dest"
    chmod +x "$file_dest"
    
    mkdir -p /var/log
    
    # Clean any existing old rules from previous installs
    echo "→ Cleaning old relay rules..."
    /usr/local/bin/decloud-relay-nat clean 2>/dev/null || true
    
    echo "✓ Relay NAT support installed"
}

create_helper_scripts() {
    log_step "Creating helper scripts..."
    
    # WireGuard helper - supports dynamic interface names
    cat > /usr/local/bin/decloud-wg << 'EOFWG'
#!/bin/bash
# DeCloud WireGuard Helper
# Interfaces are managed automatically by WireGuardConfigManager

show_help() {
    cat << EOF
DeCloud WireGuard Helper

Usage: decloud-wg <command> [args]

Commands:
  status              Show all WireGuard interfaces and their status
  interfaces          List active interface names
  show <interface>    Show detailed info for specific interface
  
Examples:
  decloud-wg status
  decloud-wg interfaces
  decloud-wg show wg-relay
  decloud-wg show wg-hub

Notes:
  • Interfaces are automatically created by the node agent
  • CGNAT nodes use 'wg-relay' tunnel to assigned relay
  • Relay VMs use 'wg-relay-server' to serve CGNAT clients
  • Regular nodes may use 'wg-hub' for peer-to-peer mesh
  • Interface selection is automatic based on orchestrator assignment
EOF
}

case "$1" in
    status)
        echo "═══════════════════════════════════════════════════════"
        echo "  DeCloud WireGuard Status"
        echo "═══════════════════════════════════════════════════════"
        
        if ! command -v wg &> /dev/null; then
            echo "Error: WireGuard not installed"
            exit 1
        fi
        
        interfaces=$(wg show interfaces 2>/dev/null)
        
        if [ -z "$interfaces" ]; then
            echo "No WireGuard interfaces active"
            echo ""
            echo "This is normal if:"
            echo "  • Node is newly installed (interfaces created on first heartbeat)"
            echo "  • WireGuard not needed for current node role"
            echo ""
            echo "Check node agent logs: journalctl -u decloud-node-agent -f | grep -i wireguard"
        else
            wg show
            echo ""
            echo "Active interfaces: $interfaces"
        fi
        ;;
        
    interfaces)
        wg show interfaces 2>/dev/null || echo "No interfaces"
        ;;
        
    show)
        if [ -z "$2" ]; then
            echo "Error: Interface name required"
            echo ""
            echo "Usage: decloud-wg show <interface>"
            echo "Example: decloud-wg show wg-relay"
            echo ""
            echo "Available interfaces:"
            wg show interfaces 2>/dev/null || echo "  (none)"
            exit 1
        fi
        
        if ! wg show "$2" &>/dev/null; then
            echo "Error: Interface '$2' not found"
            echo ""
            echo "Available interfaces:"
            wg show interfaces 2>/dev/null || echo "  (none)"
            exit 1
        fi
        
        wg show "$2"
        ;;
        
    help|-h|--help|"")
        show_help
        ;;
        
    *)
        echo "Error: Unknown command '$1'"
        echo ""
        show_help
        exit 1
        ;;
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
    echo "║       ✅ DeCloud Node Agent Installed Successfully!          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    
    public_ip=$(curl -s https://api.ipify.org || echo "unknown")
    
    echo "  Installation Summary:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo "    Node Agent:    Installed & Running"
    echo "    Status:        Awaiting Authentication"
    echo "    Public IP:     ${public_ip}"
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Next Steps:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo ""
    echo "    1. Authenticate your node:"
    echo "       sudo cli-decloud-node login"
    echo ""
    echo "    2. Check status:"
    echo "       sudo journalctl -u decloud-node-agent -f"
    echo ""
    echo "    3. The node agent will automatically complete registration"
    echo "       after authentication."
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo ""
    echo ""
    echo "  Relay NAT Manager: sudo decloud-relay-nat {add|clean|show}"
    echo ""
}

print_summary_unauthenticated() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                 Installation Complete!                       ║"
    echo "║              (Authentication Required)                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  ⚠  Node agent installed but NOT authenticated"
    echo ""
    echo "  To complete setup, run:"
    echo "    sudo cli-decloud-node login"
    echo ""
    echo "  This will:"
    echo "    1. Display a QR code"
    echo "    2. You scan it with your mobile wallet"
    echo "    3. Sign the message on your phone"
    echo "    4. Node gets authorized automatically"
    echo ""
    echo "  After authentication, the node agent will start automatically."
    echo ""
}

# ============================================================
# Main
# ============================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       DeCloud Node Agent Installer v${VERSION}               ║"
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
    configure_wireguard_keys
    
    # Install Python & WalletConnect CLI
    install_python_dependencies
    
    # SSH CA
    setup_ssh_ca
    setup_decloud_user
    configure_decloud_sshd
    
    # Application
    create_directories
    download_node_agent
    
    # Install CLI from downloaded repo
    install_walletconnect_cli
    install_relay_nat_support

    build_node_agent
    create_configuration
    create_systemd_service
    configure_firewall
    create_helper_scripts
    
    # Start service (only if authenticated)
    start_service

    # Authenticate node before starting
    # if ! run_node_authentication; then
    #     log_warn "Node agent installed but not authenticated"
    #     log_warn "The service will not start until you authenticate"
    #     log_warn "Run: sudo cli-decloud-node login"
    #     echo ""
    #     print_summary_unauthenticated
    #     exit 0
    # fi
    
    # Done
    print_summary
}

main "$@"