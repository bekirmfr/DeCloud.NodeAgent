#!/bin/bash
#
# DeCloud Node Agent Installation Script
# 
# Installs and configures the Node Agent with all dependencies:
# - .NET 8 Runtime
# - KVM/QEMU/libvirt for virtualization
# - WireGuard for overlay networking (with hub auto-configuration)
# - cloud-init tools for VM provisioning
# - libguestfs-tools for cloud-init state cleaning
# - openssh-client for ephemeral terminal key generation
# 
# Changelog v1.4.5:
# - CRITICAL: Fixed password hash - use proper $6$ hash instead of '*'
# - Password field now shows "P" (password set) instead of "L" (locked)
# - Added verification that passwd -S shows correct status after setting password
# - Ensures SSH certificate authentication works immediately after installation
#
# Changelog v1.4.4:
# - CRITICAL: Added AllowUsers check and automatic decloud addition
# - Improved unlock logic: passwd -u before usermod -p
# - Ensures AllowUsers directive doesn't block decloud user
# - Made configure_decloud_sshd() more robust and idempotent
#
# Changelog v1.4.3:
# - CRITICAL: Added unlock check for existing decloud users
# - Fixed SSH certificate authentication for decloud user
# - Changed from passwd -l to usermod -p * (prevents account lock issues)
# - Added sshd Match block to properly disable password auth
# - Ensures certificate-only authentication works correctly
#
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/bekirmfr/DeCloud.NodeAgent/master/install.sh | sudo bash -s -- --orchestrator http://IP:5050
#

set -e

VERSION="1.4.5"

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

# SSH CA configuration
SSH_CA_KEY_PATH="/etc/ssh/decloud_ca"
SSH_CA_PUB_PATH="/etc/ssh/decloud_ca.pub"

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
SSH_CA_PUBLIC_KEY=""

# ============================================================
# SSH Certificate Authority Setup
# ============================================================
setup_ssh_ca() {
    log_step "Setting up SSH Certificate Authority..."
    
    # Check if CA already exists
    if [ -f "$SSH_CA_KEY_PATH" ]; then
        log_info "SSH CA already exists, skipping generation"
        SSH_CA_PUBLIC_KEY=$(cat "$SSH_CA_PUB_PATH" 2>/dev/null || echo "")
        log_success "SSH CA configured"
        return
    fi
    
    # Generate new SSH CA key pair
    log_info "Generating SSH CA key pair..."
    ssh-keygen -t ed25519 \
        -f "$SSH_CA_KEY_PATH" \
        -C "decloud-ca@$(hostname)" \
        -N "" \
        -q
    
    # Set proper permissions
    chmod 600 "$SSH_CA_KEY_PATH"
    chmod 644 "$SSH_CA_PUB_PATH"
    
    SSH_CA_PUBLIC_KEY=$(cat "$SSH_CA_PUB_PATH")
    log_success "SSH CA key pair generated"
    
    # Configure sshd to trust this CA
    log_info "Configuring sshd to trust CA certificates..."
    
    SSHD_CONFIG="/etc/ssh/sshd_config"
    SSHD_CONFIG_BAK="/etc/ssh/sshd_config.backup-$(date +%Y%m%d-%H%M%S)"
    
    # Backup sshd_config
    cp "$SSHD_CONFIG" "$SSHD_CONFIG_BAK"
    log_info "sshd config backed up to: $SSHD_CONFIG_BAK"
    
    # Check if TrustedUserCAKeys already configured
    if grep -q "^TrustedUserCAKeys" "$SSHD_CONFIG"; then
        log_info "Updating existing TrustedUserCAKeys configuration..."
        sed -i "s|^TrustedUserCAKeys.*|TrustedUserCAKeys $SSH_CA_PUB_PATH|" "$SSHD_CONFIG"
    else
        log_info "Adding TrustedUserCAKeys to sshd_config..."
        echo "" >> "$SSHD_CONFIG"
        echo "# DeCloud SSH Certificate Authority" >> "$SSHD_CONFIG"
        echo "TrustedUserCAKeys $SSH_CA_PUB_PATH" >> "$SSHD_CONFIG"
    fi
    
    # Test sshd configuration
    log_info "Testing sshd configuration..."
    if sshd -t 2>&1; then
        log_success "sshd configuration valid"
        
        # Reload sshd
        log_info "Reloading sshd..."
        systemctl reload sshd || service sshd reload || true
        log_success "sshd reloaded"
    else
        log_error "sshd configuration test failed!"
        log_warn "Restoring backup..."
        cp "$SSHD_CONFIG_BAK" "$SSHD_CONFIG"
        systemctl reload sshd || service sshd reload || true
        log_warn "SSH CA configured but sshd not reloaded"
    fi
    
    log_success "SSH Certificate Authority setup complete"
}

# ============================================================
# DeCloud SSH User Setup
# ============================================================
setup_decloud_user() {
    log_step "Setting up 'decloud' system user for SSH certificate authentication..."
    
    # Check if user already exists
    if id "decloud" &>/dev/null; then
        log_info "User 'decloud' already exists"
        
        # Ensure account is unlocked (not passwd -l)
        PASSWD_STATUS=$(passwd -S decloud 2>/dev/null | awk '{print $2}')
        if [ "$PASSWD_STATUS" = "L" ]; then
            log_info "Account is locked, unlocking..."
            # First unlock, then set impossible password
            passwd -u decloud &>/dev/null || true
            # Use a proper hash that shows as "P" (password set) not "L" (locked)
            # This hash is impossible to match but doesn't trigger "locked" status
            usermod -p '$6$rounds=656000$DeCloudImpossible$Qwt6vjdgZzE7pXWRoNmZbCFDZklM9X0v3mHs8K5jNfVhWaEoQb7Yx2Lz3PnMkJhG4RtYuIoP1aSdFgHjK2LmN.' decloud 2>/dev/null || true
            # Verify it worked
            PASSWD_STATUS=$(passwd -S decloud 2>/dev/null | awk '{print $2}')
            if [ "$PASSWD_STATUS" = "P" ]; then
                log_success "Account unlocked (impossible password set)"
            else
                log_warn "Account status: $PASSWD_STATUS (expected P)"
            fi
        fi
        
        # Verify .ssh directory exists and has correct permissions
        if [ ! -d "/home/decloud/.ssh" ]; then
            log_info "Creating .ssh directory for existing user..."
            mkdir -p /home/decloud/.ssh
            chmod 700 /home/decloud/.ssh
            chown decloud:decloud /home/decloud/.ssh
        fi
        
        # Ensure authorized_keys exists
        if [ ! -f "/home/decloud/.ssh/authorized_keys" ]; then
            touch /home/decloud/.ssh/authorized_keys
            chmod 600 /home/decloud/.ssh/authorized_keys
            chown decloud:decloud /home/decloud/.ssh/authorized_keys
        fi
        
        log_success "DeCloud user configured"
        return 0
    fi
    
    # Create system user (no password, certificate-only authentication)
    log_info "Creating 'decloud' system user..."
    if ! useradd -m -s /bin/bash -c "DeCloud SSH Jump User" decloud 2>/dev/null; then
        log_error "Failed to create decloud user"
        return 1
    fi
    
    log_success "User 'decloud' created"
    
    # Create .ssh directory with proper permissions
    log_info "Setting up SSH directory..."
    mkdir -p /home/decloud/.ssh
    chmod 700 /home/decloud/.ssh
    chown decloud:decloud /home/decloud/.ssh
    
    # Create authorized_keys file (even if empty, for future use)
    touch /home/decloud/.ssh/authorized_keys
    chmod 600 /home/decloud/.ssh/authorized_keys
    chown decloud:decloud /home/decloud/.ssh/authorized_keys
    
    # Add to libvirt group if it exists (optional, for VM access monitoring)
    if getent group libvirt > /dev/null 2>&1; then
        log_info "Adding decloud user to libvirt group..."
        usermod -aG libvirt decloud 2>/dev/null || true
    fi
    
    # Disable password authentication (certificate-only)
    log_info "Disabling password authentication for decloud user..."
    # Use a proper hash that shows as "P" (password set) not "L" (locked)
    # This hash is impossible to match but doesn't trigger "locked" status in passwd -S
    usermod -p '$6$rounds=656000$DeCloudImpossible$Qwt6vjdgZzE7pXWRoNmZbCFDZklM9X0v3mHs8K5jNfVhWaEoQb7Yx2Lz3PnMkJhG4RtYuIoP1aSdFgHjK2LmN.' decloud 2>/dev/null || true
    
    # Verify password field is set correctly
    PASSWD_STATUS=$(passwd -S decloud 2>/dev/null | awk '{print $2}')
    if [ "$PASSWD_STATUS" = "P" ]; then
        log_success "Password authentication disabled (impossible password hash set)"
    else
        log_warn "Password status: $PASSWD_STATUS (expected P, got different status)"
    fi
    
    # Create a README in the .ssh directory
    cat > /home/decloud/.ssh/README << 'README_EOF'
DeCloud SSH Jump Host
=====================

This account is configured for SSH certificate-based authentication only.

Certificate authentication is handled by the DeCloud orchestrator.
Users authenticate using their Ethereum wallet to receive short-lived
SSH certificates signed by this node's Certificate Authority.

The SSH CA public key is located at: /etc/ssh/decloud_ca.pub

Connection flow:
  User → Wallet signature → Orchestrator → SSH certificate → Jump host → VM

For more information: https://github.com/bekirmfr/DeCloud
README_EOF
    
    chown decloud:decloud /home/decloud/.ssh/README
    chmod 644 /home/decloud/.ssh/README
    
    log_success "DeCloud user setup complete"
    
    # Display user info
    log_info "User information:"
    echo "    Username:        decloud"
    echo "    Home directory:  /home/decloud"
    echo "    Shell:           /bin/bash"
    echo "    Authentication:  SSH certificate only (password disabled)"
    echo "    SSH directory:   /home/decloud/.ssh (mode: 700)"
    if getent group libvirt > /dev/null 2>&1; then
        echo "    Groups:          decloud, libvirt"
    else
        echo "    Groups:          decloud"
    fi
}

# ============================================================
# Configure SSHD for DeCloud User
# ============================================================
configure_decloud_sshd() {
    log_step "Configuring SSH daemon for decloud user..."
    
    SSHD_CONFIG="/etc/ssh/sshd_config"
    local config_changed=false
    
    # Backup sshd_config
    SSHD_CONFIG_BAK="/etc/ssh/sshd_config.backup-decloud-$(date +%Y%m%d-%H%M%S)"
    cp "$SSHD_CONFIG" "$SSHD_CONFIG_BAK"
    log_info "sshd config backed up to: $SSHD_CONFIG_BAK"
    
    # ====================================================
    # Check and update AllowUsers directive
    # ====================================================
    if grep -q "^AllowUsers" "$SSHD_CONFIG"; then
        # AllowUsers exists - check if decloud is in it
        if ! grep "^AllowUsers" "$SSHD_CONFIG" | grep -qw "decloud"; then
            log_info "Adding decloud to AllowUsers directive..."
            sed -i 's/^AllowUsers \(.*\)/AllowUsers \1 decloud/' "$SSHD_CONFIG"
            config_changed=true
            log_success "decloud added to AllowUsers"
        else
            log_info "decloud already in AllowUsers"
        fi
    else
        log_info "No AllowUsers directive (all users allowed)"
    fi
    
    # ====================================================
    # Add Match block for decloud user
    # ====================================================
    if grep -q "^Match User decloud" "$SSHD_CONFIG"; then
        log_info "Match block for decloud already exists"
    else
        log_info "Adding Match block for decloud user..."
        cat >> "$SSHD_CONFIG" << 'MATCH_EOF'

# DeCloud SSH Jump User Configuration
# Certificate-only authentication, password auth disabled
Match User decloud
    PasswordAuthentication no
    PubkeyAuthentication yes
    AuthenticationMethods publickey
MATCH_EOF
        config_changed=true
        log_success "Match block added"
    fi
    
    # ====================================================
    # Test and reload sshd if changes were made
    # ====================================================
    if [ "$config_changed" = true ]; then
        log_info "Testing sshd configuration..."
        if sshd -t 2>&1; then
            log_success "sshd configuration valid"
            
            # Reload sshd
            log_info "Reloading sshd..."
            systemctl reload sshd || service sshd reload || true
            log_success "sshd reloaded with decloud configuration"
        else
            log_error "sshd configuration test failed!"
            log_warn "Restoring backup..."
            cp "$SSHD_CONFIG_BAK" "$SSHD_CONFIG"
            systemctl reload sshd || service sshd reload || true
            log_error "Failed to configure sshd for decloud user"
            return 1
        fi
    else
        log_success "DeCloud SSH configuration already correct"
    fi
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
            log_warn "Untested OS: $OS $VERSION_ID. Continuing anyway..."
            ;;
    esac
}

check_architecture() {
    log_step "Checking architecture..."
    
    ARCH=$(uname -m)
    if [ "$ARCH" != "x86_64" ]; then
        log_error "Only x86_64 architecture is supported. Found: $ARCH"
        exit 1
    fi
    log_success "Architecture: x86_64"
}

check_virtualization() {
    log_step "Checking virtualization support..."
    
    if [ "$SKIP_LIBVIRT" = true ]; then
        log_warn "Skipping virtualization check (--skip-libvirt)"
        return
    fi
    
    # Check CPU virtualization
    if grep -E 'vmx|svm' /proc/cpuinfo > /dev/null 2>&1; then
        log_success "Hardware virtualization supported"
    else
        log_error "Hardware virtualization (VT-x/AMD-V) not detected"
        log_error "Enable it in BIOS or check if running in a nested VM"
        exit 1
    fi
    
    # Check KVM availability
    if [ -c /dev/kvm ]; then
        log_success "KVM available"
    else
        log_warn "KVM not available. Will try to enable after installing packages."
    fi
}

check_resources() {
    log_step "Checking system resources..."
    
    # CPU
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -lt "$MIN_CPU_CORES" ]; then
        log_warn "Found $CPU_CORES CPU cores. Recommended: $MIN_CPU_CORES+"
    else
        log_success "CPU cores: $CPU_CORES"
    fi
    
    # Memory
    MEMORY_MB=$(free -m | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_MB" -lt "$MIN_MEMORY_MB" ]; then
        log_warn "Found ${MEMORY_MB}MB RAM. Recommended: ${MIN_MEMORY_MB}MB+"
    else
        log_success "Memory: ${MEMORY_MB}MB"
    fi
    
    # Disk
    DISK_GB=$(df -BG / | awk 'NR==2{print $4}' | tr -d 'G')
    if [ "$DISK_GB" -lt "$MIN_DISK_GB" ]; then
        log_warn "Found ${DISK_GB}GB free disk. Recommended: ${MIN_DISK_GB}GB+"
    else
        log_success "Free disk: ${DISK_GB}GB"
    fi
}

check_network() {
    log_step "Checking network..."
    
    # Detect public IP
    PUBLIC_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || \
                curl -s --max-time 5 https://ifconfig.me 2>/dev/null || \
                hostname -I | awk '{print $1}')
    
    if [ -z "$PUBLIC_IP" ]; then
        log_warn "Could not detect public IP. Using hostname IP."
        PUBLIC_IP=$(hostname -I | awk '{print $1}')
    fi
    
    log_success "Public IP: $PUBLIC_IP"
    
    # Test orchestrator connectivity
    if [ -n "$ORCHESTRATOR_URL" ]; then
        if curl -s --max-time 5 "$ORCHESTRATOR_URL/health" > /dev/null 2>&1; then
            log_success "Orchestrator reachable: $ORCHESTRATOR_URL"
        else
            log_warn "Cannot reach orchestrator at $ORCHESTRATOR_URL"
            log_warn "Make sure the orchestrator is running and accessible"
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
        curl \
        wget \
        git \
        jq \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        > /dev/null 2>&1
    
    log_success "Base dependencies installed"
}

install_dotnet() {
    log_step "Installing .NET 8 SDK..."
    
    # Check if already installed
    if command -v dotnet &> /dev/null; then
        DOTNET_VERSION=$(dotnet --version 2>/dev/null || echo "0")
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
    log_step "Installing libvirt/KVM and virtualization tools..."
    
    if [ "$SKIP_LIBVIRT" = true ]; then
        log_warn "Skipping libvirt installation (--skip-libvirt)"
        return
    fi
    
    # Check if running in a VM (nested virtualization)
    if [ -f /sys/module/kvm_intel/parameters/nested ]; then
        NESTED=$(cat /sys/module/kvm_intel/parameters/nested)
        if [ "$NESTED" != "Y" ] && [ "$NESTED" != "1" ]; then
            log_warn "Nested virtualization may not be enabled"
            log_warn "Performance may be reduced if running inside a VM"
        fi
    fi
    
    # Core virtualization packages
    PACKAGES="qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst"
    
    # Cloud-init and disk tools
    PACKAGES="$PACKAGES cloud-image-utils genisoimage qemu-utils"
    
    # libguestfs for cleaning cloud-init state from base images
    PACKAGES="$PACKAGES libguestfs-tools"
    
    # openssh-client for ssh-keygen (ephemeral terminal key generation)
    PACKAGES="$PACKAGES openssh-client"
    
    apt-get install -y -qq $PACKAGES > /dev/null 2>&1
    
    # Load nbd module for qemu-nbd fallback method (cloud-init cleaning)
    modprobe nbd max_part=8 2>/dev/null || true
    if [ ! -f /etc/modules-load.d/decloud.conf ] || ! grep -q "nbd" /etc/modules-load.d/decloud.conf 2>/dev/null; then
        echo "nbd" >> /etc/modules-load.d/decloud.conf
    fi
    
    # Enable and start libvirtd
    systemctl enable libvirtd --quiet 2>/dev/null || true
    systemctl start libvirtd 2>/dev/null || true
    
    # Setup default network
    if ! virsh net-list --all | grep -q "default"; then
        log_info "Creating default network..."
        virsh net-define /usr/share/libvirt/networks/default.xml > /dev/null 2>&1 || true
    fi
    
    virsh net-autostart default > /dev/null 2>&1 || true
    virsh net-start default > /dev/null 2>&1 || true
    
    # Create QEMU guest agent channel directory (for ephemeral key injection)
    mkdir -p /var/lib/libvirt/qemu/channel/target
    
    log_success "Libvirt/KVM installed and configured"
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

configure_wireguard_hub() {
    if [ "$SKIP_WIREGUARD" = true ] || [ "$ENABLE_WIREGUARD_HUB" = false ]; then
        return
    fi
    
    log_step "Configuring WireGuard hub..."
    
    # Generate keys
    WG_PRIVATE_KEY=$(wg genkey)
    WG_PUBLIC_KEY=$(echo "$WG_PRIVATE_KEY" | wg pubkey)
    
    mkdir -p /etc/wireguard
    
    # Save keys
    echo "$WG_PRIVATE_KEY" > /etc/wireguard/node_private.key
    echo "$WG_PUBLIC_KEY" > /etc/wireguard/node_public.key
    chmod 600 /etc/wireguard/node_private.key
    
    # Create WireGuard config
    cat > /etc/wireguard/${WIREGUARD_INTERFACE}.conf << EOF
[Interface]
Address = ${WIREGUARD_HUB_IP}/24
ListenPort = ${WIREGUARD_PORT}
PrivateKey = ${WG_PRIVATE_KEY}

# Enable IP forwarding for VM access
PostUp = sysctl -w net.ipv4.ip_forward=1
PostUp = iptables -A FORWARD -i %i -j ACCEPT
PostUp = iptables -A FORWARD -o %i -j ACCEPT
PostUp = iptables -t nat -A POSTROUTING -s ${WIREGUARD_NETWORK} -o eth0 -j MASQUERADE
PostUp = iptables -t nat -A POSTROUTING -s ${WIREGUARD_NETWORK} -o ens3 -j MASQUERADE
PostUp = iptables -A FORWARD -i %i -o virbr0 -j ACCEPT
PostUp = iptables -A FORWARD -i virbr0 -o %i -j ACCEPT

PostDown = iptables -D FORWARD -i %i -j ACCEPT
PostDown = iptables -D FORWARD -o %i -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -s ${WIREGUARD_NETWORK} -o eth0 -j MASQUERADE 2>/dev/null || true
PostDown = iptables -t nat -D POSTROUTING -s ${WIREGUARD_NETWORK} -o ens3 -j MASQUERADE 2>/dev/null || true
PostDown = iptables -D FORWARD -i %i -o virbr0 -j ACCEPT 2>/dev/null || true
PostDown = iptables -D FORWARD -i virbr0 -o %i -j ACCEPT 2>/dev/null || true

# Peers will be added dynamically
EOF

    chmod 600 /etc/wireguard/${WIREGUARD_INTERFACE}.conf
    
    # Enable IP forwarding permanently
    if ! grep -q "net.ipv4.ip_forward=1" /etc/sysctl.conf; then
        echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
    fi
    sysctl -w net.ipv4.ip_forward=1 > /dev/null 2>&1
    
    # Start WireGuard
    systemctl enable wg-quick@${WIREGUARD_INTERFACE} --quiet 2>/dev/null || true
    systemctl start wg-quick@${WIREGUARD_INTERFACE} 2>/dev/null || true
    
    # Configure firewall for WireGuard
    if command -v ufw &> /dev/null; then
        ufw allow ${WIREGUARD_PORT}/udp > /dev/null 2>&1 || true
    fi
    iptables -I INPUT -p udp --dport ${WIREGUARD_PORT} -j ACCEPT 2>/dev/null || true
    
    log_success "WireGuard hub configured (${WIREGUARD_HUB_IP})"
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
    mkdir -p /var/log/decloud
    
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
  "Serilog": {
    "MinimumLevel": {
      "Default": "Information",
      "Override": {
        "Microsoft": "Warning",
        "System": "Warning"
      }
    },
    "WriteTo": [
      { "Name": "Console" },
      {
        "Name": "File",
        "Args": {
          "path": "/var/log/decloud/node-agent-.log",
          "rollingInterval": "Day",
          "retainedFileCountLimit": 7
        }
      }
    ]
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
    "VncPortStart": 5900,
    "ReconcileOnStartup": true
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
Environment=DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
Environment=DOTNET_NOLOGO=1
Environment=DOTNET_CLI_HOME=/tmp/dotnet-cli

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
    local last_octet=2
    
    # Find highest used IP
    for conf in $WG_DIR/clients/*.conf 2>/dev/null; do
        if [ -f "$conf" ]; then
            local ip=$(grep "Address" "$conf" | grep -oP '\d+\.\d+\.\d+\.\d+' | head -1)
            if [ -n "$ip" ]; then
                local octet="${ip##*.}"
                if [ "$octet" -ge "$last_octet" ]; then
                    last_octet=$((octet + 1))
                fi
            fi
        fi
    done
    
    echo "${prefix}.${last_octet}"
}

case "${1:-help}" in
    status)
        echo "=== WireGuard Status ==="
        wg show $WG_INTERFACE 2>/dev/null || echo "Interface not running"
        echo ""
        echo "=== Clients ==="
        ls -1 $WG_DIR/clients/ 2>/dev/null | sed 's/.conf$//' || echo "No clients configured"
        ;;
    
    add)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 add <client_pubkey> <client_ip>"
            echo "Example: $0 add ABC123...= 10.10.0.2"
            exit 1
        fi
        
        CLIENT_PUBKEY="$2"
        CLIENT_IP="$3"
        VM_NET="192.168.122.0/24"
        
        wg set $WG_INTERFACE peer "$CLIENT_PUBKEY" allowed-ips "${CLIENT_IP}/32,${VM_NET}"
        wg-quick save $WG_INTERFACE 2>/dev/null || true
        
        echo "Added peer: $CLIENT_IP"
        ;;
    
    remove)
        if [ -z "$2" ]; then
            echo "Usage: $0 remove <client_pubkey>"
            exit 1
        fi
        
        wg set $WG_INTERFACE peer "$2" remove
        wg-quick save $WG_INTERFACE 2>/dev/null || true
        
        echo "Removed peer"
        ;;
    
    client-config)
        CLIENT_IP="${2:-$(suggest_client_ip)}"
        HUB_IP=$(get_hub_ip)
        HUB_ENDPOINT="${3:-$(curl -s https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}')}"
        HUB_PORT=$(get_config_value "ListenPort" "51820")
        HUB_PUBKEY=$(cat $WG_DIR/node_public.key 2>/dev/null || echo "HUB_PUBLIC_KEY_HERE")
        
        # Generate client keys
        CLIENT_PRIVKEY=$(wg genkey)
        CLIENT_PUBKEY=$(echo "$CLIENT_PRIVKEY" | wg pubkey)
        
        # Save client config
        mkdir -p $WG_DIR/clients
        cat > $WG_DIR/clients/${CLIENT_IP}.conf << EOF
[Interface]
PrivateKey = ${CLIENT_PRIVKEY}
Address = ${CLIENT_IP}/24

[Peer]
PublicKey = ${HUB_PUBKEY}
Endpoint = ${HUB_ENDPOINT}:${HUB_PORT}
AllowedIPs = ${HUB_IP}/32, 192.168.122.0/24
PersistentKeepalive = 25
EOF
        
        echo "=== Client Config for ${CLIENT_IP} ==="
        echo ""
        cat $WG_DIR/clients/${CLIENT_IP}.conf
        echo ""
        echo "=== Next Steps ==="
        echo "1. Copy the config above to your client"
        echo "2. Add client to hub: $0 add ${CLIENT_PUBKEY} ${CLIENT_IP}"
        echo ""
        echo "Client Public Key: ${CLIENT_PUBKEY}"
        ;;
    
    help|*)
        echo "DeCloud WireGuard Client Management"
        echo ""
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  status                  Show WireGuard status and clients"
        echo "  add <pubkey> <ip>       Add a client peer"
        echo "  remove <pubkey>         Remove a client peer"
        echo "  client-config [ip]      Generate client config"
        echo ""
        ;;
esac
SCRIPT_EOF

    chmod +x /usr/local/bin/decloud-wg
    
    log_success "WireGuard helper script created: decloud-wg"
}

print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              Installation Complete!                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    log_success "Node Agent v${VERSION} installed successfully!"
    echo ""
    echo "  Node Name:       ${NODE_NAME}"
    echo "  Public IP:       ${PUBLIC_IP}"
    echo "  Agent API:       http://${PUBLIC_IP}:${AGENT_PORT}"
    echo "  Orchestrator:    ${ORCHESTRATOR_URL}"
    echo ""
    
    # SSH CA Information
    if [ -n "$SSH_CA_PUBLIC_KEY" ]; then
        echo "  ─────────────────────────────────────────────────────────────"
        echo "  SSH Certificate Authority:"
        echo "  ─────────────────────────────────────────────────────────────"
        echo "  Status:          Configured"
        echo "  CA Private Key:  $SSH_CA_KEY_PATH"
        echo "  CA Public Key:   $SSH_CA_PUB_PATH"
        echo ""
        echo "  Wallet-based SSH authentication is enabled!"
        echo "  Users can SSH using certificates signed by this CA."
        echo ""
    fi

    # DeCloud User Information
    if id "decloud" &>/dev/null; then
        echo "  ─────────────────────────────────────────────────────────────"
        echo "  SSH Jump User:"
        echo "  ─────────────────────────────────────────────────────────────"
        echo "  Username:        decloud"
        echo "  Authentication:  SSH Certificate only"
        echo "  Home Directory:  /home/decloud"
        echo ""
        echo "  Users connect via: ssh -i key.pem -o CertificateFile=cert.pub decloud@${PUBLIC_IP}"
        echo ""
    fi
    
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
    echo "  SSH CA Info:     cat $SSH_CA_PUB_PATH"
    echo ""
    echo "Configuration:     ${CONFIG_DIR}/appsettings.Production.json"
    echo "Data directory:    ${DATA_DIR}"
    echo ""
    echo "System resources:"
    echo "  CPU Cores:       ${CPU_CORES}"
    echo "  Memory:          ${MEMORY_MB}MB"
    echo "  Free Disk:       ${DISK_GB}GB"
    echo ""
    echo "Installed tools:"
    echo "  virt-customize:  $(which virt-customize 2>/dev/null && echo '✓' || echo '✗')"
    echo "  ssh-keygen:      $(which ssh-keygen 2>/dev/null && echo '✓' || echo '✗')"
    echo "  SSH CA:          $([ -f $SSH_CA_KEY_PATH ] && echo '✓' || echo '✗')"
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
    
    # Setup SSH CA
    setup_ssh_ca

    # Setup decloud user for SSH jump host
    setup_decloud_user
    configure_decloud_sshd
    
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