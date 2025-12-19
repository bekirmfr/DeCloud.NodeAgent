#!/bin/bash
#
# DeCloud Node Agent Installation Script
# 
# Installs and configures the Node Agent with all dependencies:
# - .NET 8 Runtime
# - KVM/QEMU/libvirt for virtualization
# - WireGuard for overlay networking
# - Caddy for HTTP/HTTPS ingress with automatic TLS
# - fail2ban for DDoS protection
# - SSH CA for certificate authentication
#
# Version: 1.5.0
# Changelog:
# - Added Caddy ingress gateway with automatic Let's Encrypt TLS
# - Added fail2ban DDoS/abuse protection
# - Added security audit logging
# - Simplified architecture (HTTP-only ingress, no direct port forwarding)
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/.../install.sh | sudo bash -s -- \
#       --orchestrator http://IP:5050 --caddy-email admin@example.com
#

set -e

VERSION="1.5.0"

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
NODE_WALLET="0x0000000000000000000000000000000000000000"
NODE_NAME=$(hostname)
NODE_REGION="default"
NODE_ZONE="default"

# Paths
INSTALL_DIR="/opt/decloud"
CONFIG_DIR="/etc/decloud"
DATA_DIR="/var/lib/decloud/vms"
REPO_URL="https://github.com/bekirmfr/DeCloud.NodeAgent.git"

# Ports
AGENT_PORT=5100
WIREGUARD_PORT=51820

# WireGuard
WIREGUARD_HUB_IP="10.10.0.1"
SKIP_WIREGUARD=false
ENABLE_WIREGUARD_HUB=true

# Caddy Ingress
INSTALL_CADDY=${INSTALL_CADDY:-true}
CADDY_ACME_EMAIL=""
CADDY_ACME_STAGING=false
CADDY_DATA_DIR="/var/lib/caddy"
CADDY_LOG_DIR="/var/log/caddy"
CADDY_CONFIG_DIR="/etc/caddy"

# Security / fail2ban
INSTALL_FAIL2BAN=${INSTALL_FAIL2BAN:-true}
DECLOUD_LOG_DIR="/var/log/decloud"
DECLOUD_AUDIT_LOG="${DECLOUD_LOG_DIR}/audit.log"

# SSH CA
SSH_CA_KEY_PATH="/etc/decloud/ssh_ca"
SSH_CA_PUB_PATH="/etc/decloud/ssh_ca.pub"

# Other
SKIP_LIBVIRT=false

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
            --skip-caddy)
                INSTALL_CADDY=false
                shift
                ;;
            --caddy-email)
                CADDY_ACME_EMAIL="$2"
                shift 2
                ;;
            --caddy-staging)
                CADDY_ACME_STAGING=true
                shift
                ;;
            --skip-fail2ban)
                INSTALL_FAIL2BAN=false
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

Node Identity:
  --wallet <address>     Node operator wallet address
  --name <name>          Node name (default: hostname)
  --region <region>      Region identifier (default: default)
  --zone <zone>          Zone identifier (default: default)
  --port <port>          Agent API port (default: 5100)

WireGuard:
  --wg-port <port>       WireGuard listen port (default: 51820)
  --wg-ip <ip>           WireGuard hub IP (default: 10.10.0.1)
  --skip-wireguard       Skip WireGuard installation
  --no-wireguard-hub     Install WireGuard but don't configure hub

Ingress Gateway:
  --skip-caddy           Skip Caddy ingress gateway installation
  --caddy-email <email>  Email for Let's Encrypt TLS certificates (recommended)
  --caddy-staging        Use Let's Encrypt staging (for testing)

Security:
  --skip-fail2ban        Skip fail2ban DDoS protection

Other:
  --skip-libvirt         Skip libvirt installation (testing only)
  --help                 Show this help message

Examples:
  # Full installation with ingress
  $0 --orchestrator http://142.234.200.108:5050 --caddy-email admin@example.com

  # Without ingress gateway
  $0 --orchestrator http://142.234.200.108:5050 --skip-caddy --skip-fail2ban

  # Test environment with staging certificates
  $0 --orchestrator http://142.234.200.108:5050 --caddy-email test@example.com --caddy-staging
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
}

# ============================================================
# Base Installation
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
    if [ "$SKIP_WIREGUARD" = true ] || [ "$ENABLE_WIREGUARD_HUB" = false ]; then
        return
    fi
    
    log_step "Configuring WireGuard hub..."
    
    mkdir -p /etc/wireguard
    
    if [ ! -f /etc/wireguard/privatekey ]; then
        wg genkey | tee /etc/wireguard/privatekey | wg pubkey > /etc/wireguard/publickey
        chmod 600 /etc/wireguard/privatekey
    fi
    
    PRIVATE_KEY=$(cat /etc/wireguard/privatekey)
    PUBLIC_KEY=$(cat /etc/wireguard/publickey)
    
    cat > /etc/wireguard/wg0.conf << EOF
[Interface]
PrivateKey = $PRIVATE_KEY
Address = $WIREGUARD_HUB_IP/24
ListenPort = $WIREGUARD_PORT
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -A FORWARD -o wg0 -j ACCEPT
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -D FORWARD -o wg0 -j ACCEPT
EOF
    
    chmod 600 /etc/wireguard/wg0.conf
    
    systemctl enable wg-quick@wg0 --quiet 2>/dev/null || true
    systemctl start wg-quick@wg0 2>/dev/null || true
    
    log_success "WireGuard hub configured"
    log_info "Public key: $PUBLIC_KEY"
}

# ============================================================
# Caddy Ingress Gateway
# ============================================================
install_caddy() {
    if [ "$INSTALL_CADDY" = false ]; then
        log_warn "Skipping Caddy installation (--skip-caddy)"
        return 0
    fi

    log_step "Installing Caddy web server..."

    if command -v caddy &> /dev/null; then
        local version=$(caddy version 2>/dev/null | head -1 | awk '{print $1}')
        log_success "Caddy already installed: $version"
        return 0
    fi

    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl > /dev/null 2>&1

    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
        gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg 2>/dev/null

    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
        tee /etc/apt/sources.list.d/caddy-stable.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq caddy > /dev/null 2>&1

    if ! command -v caddy &> /dev/null; then
        log_error "Caddy installation failed"
        return 1
    fi

    local version=$(caddy version 2>/dev/null | head -1 | awk '{print $1}')
    log_success "Caddy installed: $version"
}

configure_caddy() {
    if [ "$INSTALL_CADDY" = false ]; then
        return 0
    fi

    if ! command -v caddy &> /dev/null; then
        log_warn "Caddy not installed, skipping configuration"
        return 0
    fi

    log_step "Configuring Caddy..."

    mkdir -p "$CADDY_DATA_DIR" "$CADDY_LOG_DIR" "$CADDY_CONFIG_DIR"
    chown caddy:caddy "$CADDY_DATA_DIR" "$CADDY_LOG_DIR"

    local acme_email_line=""
    local acme_ca_line=""

    if [ -n "$CADDY_ACME_EMAIL" ]; then
        acme_email_line="    email $CADDY_ACME_EMAIL"
    fi

    if [ "$CADDY_ACME_STAGING" = true ]; then
        acme_ca_line="    acme_ca https://acme-staging-v02.api.letsencrypt.org/directory"
    fi

    cat > "$CADDY_CONFIG_DIR/Caddyfile" << EOF
# DeCloud Ingress Gateway Configuration
# Routes managed dynamically via Admin API (localhost:2019)

{
    admin localhost:2019

    log {
        output file $CADDY_LOG_DIR/caddy.log {
            roll_size 100mb
            roll_keep 5
            roll_keep_for 720h
        }
        format json
    }

$acme_email_line
$acme_ca_line
}

:8080 {
    respond /health "OK" 200
    respond /ready "OK" 200
}
EOF

    mkdir -p /etc/systemd/system/caddy.service.d
    cat > /etc/systemd/system/caddy.service.d/decloud.conf << 'EOF'
[Service]
AmbientCapabilities=CAP_NET_BIND_SERVICE
LimitNOFILE=1048576
Restart=always
RestartSec=5
Environment="XDG_DATA_HOME=/var/lib/caddy"
Environment="XDG_CONFIG_HOME=/etc/caddy"
ProtectSystem=full
ProtectHome=true
PrivateTmp=true
NoNewPrivileges=true
EOF

    systemctl daemon-reload
    log_success "Caddy configured"
}

configure_caddy_firewall() {
    if [ "$INSTALL_CADDY" = false ]; then
        return 0
    fi

    if ! command -v caddy &> /dev/null; then
        return 0
    fi

    log_step "Configuring firewall for ingress (ports 80, 443)..."

    if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
        ufw allow 80/tcp comment "DeCloud Ingress HTTP" > /dev/null 2>&1 || true
        ufw allow 443/tcp comment "DeCloud Ingress HTTPS" > /dev/null 2>&1 || true
        log_success "UFW rules added for ports 80/443"
    fi

    if command -v firewall-cmd &> /dev/null && systemctl is-active --quiet firewalld; then
        firewall-cmd --permanent --add-service=http > /dev/null 2>&1 || true
        firewall-cmd --permanent --add-service=https > /dev/null 2>&1 || true
        firewall-cmd --reload > /dev/null 2>&1 || true
        log_success "Firewalld rules added for HTTP/HTTPS"
    fi

    if command -v iptables &> /dev/null; then
        iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || \
            iptables -I INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || true
        iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || \
            iptables -I INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || true
    fi

    log_success "Firewall configured for ingress"
}

start_caddy() {
    if [ "$INSTALL_CADDY" = false ]; then
        return 0
    fi

    if ! command -v caddy &> /dev/null; then
        return 0
    fi

    log_step "Starting Caddy service..."

    systemctl enable caddy --quiet 2>/dev/null || true
    systemctl start caddy 2>/dev/null || true

    sleep 2

    if systemctl is-active --quiet caddy; then
        log_success "Caddy service started"

        local admin_check=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:2019/config/ 2>/dev/null || echo "000")
        if [ "$admin_check" = "200" ]; then
            log_success "Caddy Admin API accessible (localhost:2019)"
        fi
    else
        log_error "Failed to start Caddy service"
        return 1
    fi
}

# ============================================================
# fail2ban DDoS Protection
# ============================================================
install_fail2ban() {
    if [ "$INSTALL_FAIL2BAN" = false ]; then
        log_warn "Skipping fail2ban installation (--skip-fail2ban)"
        return 0
    fi

    log_step "Installing fail2ban..."

    apt-get install -y -qq fail2ban > /dev/null 2>&1

    if ! command -v fail2ban-client &> /dev/null; then
        log_error "fail2ban installation failed"
        return 1
    fi

    log_success "fail2ban installed"
}

configure_fail2ban() {
    if [ "$INSTALL_FAIL2BAN" = false ]; then
        return 0
    fi

    if ! command -v fail2ban-client &> /dev/null; then
        return 0
    fi

    log_step "Configuring fail2ban filters and jails..."

    mkdir -p "$DECLOUD_LOG_DIR"
    touch "$DECLOUD_AUDIT_LOG"
    chmod 640 "$DECLOUD_AUDIT_LOG"

    # Caddy rate limiting filter
    cat > /etc/fail2ban/filter.d/caddy-ratelimit.conf << 'EOF'
[Definition]
failregex = ^.*"client_ip":"<HOST>".*"status":(429|503).*$
            ^.*"remote_ip":"<HOST>".*"status":(429|503).*$
ignoreregex =
datepattern = "ts":{EPOCH}
              %%Y-%%m-%%dT%%H:%%M:%%S
EOF

    # Caddy abuse filter
    cat > /etc/fail2ban/filter.d/caddy-abuse.conf << 'EOF'
[Definition]
failregex = ^.*"client_ip":"<HOST>".*"status":(400|401|403|404|405).*$
            ^.*"remote_ip":"<HOST>".*"status":(400|401|403|404|405).*$
ignoreregex = 
datepattern = "ts":{EPOCH}
              %%Y-%%m-%%dT%%H:%%M:%%S
EOF

    # DeCloud API filter
    cat > /etc/fail2ban/filter.d/decloud-api.conf << 'EOF'
[Definition]
failregex = ^.*"RemoteIpAddress":"<HOST>".*"StatusCode":(401|403).*$
            ^.*Unauthorized access attempt from <HOST>.*$
ignoreregex =
datepattern = %%Y-%%m-%%d %%H:%%M:%%S
EOF

    # Jail configuration
    cat > /etc/fail2ban/jail.d/decloud.conf << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 10
banaction = iptables-multiport
action = %(action_)s

[sshd]
enabled = true
port = ssh,22,2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 5
bantime = 86400

[caddy-ratelimit]
enabled = true
port = http,https
filter = caddy-ratelimit
logpath = ${CADDY_LOG_DIR}/access.log
maxretry = 50
findtime = 60
bantime = 1800

[caddy-abuse]
enabled = true
port = http,https
filter = caddy-abuse
logpath = ${CADDY_LOG_DIR}/access.log
maxretry = 30
findtime = 300
bantime = 3600

[decloud-api]
enabled = true
port = ${AGENT_PORT}
filter = decloud-api
logpath = ${DECLOUD_LOG_DIR}/nodeagent.log
maxretry = 10
findtime = 300
bantime = 3600

[recidive]
enabled = true
filter = recidive
logpath = /var/log/fail2ban.log
banaction = iptables-allports
bantime = 604800
findtime = 86400
maxretry = 3
EOF

    log_success "fail2ban configured"
}

start_fail2ban() {
    if [ "$INSTALL_FAIL2BAN" = false ]; then
        return 0
    fi

    if ! command -v fail2ban-client &> /dev/null; then
        return 0
    fi

    log_step "Starting fail2ban service..."

    if ! fail2ban-client -t > /dev/null 2>&1; then
        log_warn "fail2ban configuration has warnings"
    fi

    systemctl enable fail2ban --quiet 2>/dev/null || true
    systemctl restart fail2ban 2>/dev/null || true

    sleep 2

    if systemctl is-active --quiet fail2ban; then
        log_success "fail2ban service started"
        local jails=$(fail2ban-client status 2>/dev/null | grep "Jail list" | cut -d: -f2 | tr -d '[:space:]')
        log_info "Active jails: $jails"
    else
        log_warn "fail2ban may not be running correctly"
    fi
}

# ============================================================
# Security Logging
# ============================================================
setup_security_logging() {
    log_step "Setting up security logging..."

    mkdir -p "$DECLOUD_LOG_DIR"
    mkdir -p "$CADDY_LOG_DIR"

    touch "$DECLOUD_AUDIT_LOG"
    chmod 640 "$DECLOUD_AUDIT_LOG"

    cat > /etc/logrotate.d/decloud << EOF
${DECLOUD_LOG_DIR}/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 640 root root
    sharedscripts
    postrotate
        systemctl reload decloud-node-agent > /dev/null 2>&1 || true
    endscript
}
EOF

    log_success "Security logging configured"
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
    mkdir -p "$DECLOUD_LOG_DIR"
    mkdir -p /var/lib/decloud
    
    log_success "Directories created"
}

download_node_agent() {
    log_step "Downloading Node Agent..."
    
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

    local caddy_staging="false"
    [ "$CADDY_ACME_STAGING" = true ] && caddy_staging="true"

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
    "HeartbeatIntervalSeconds": 30,
    "CommandPollIntervalSeconds": 5
  },
  "Node": {
    "WalletAddress": "${NODE_WALLET}",
    "Name": "${NODE_NAME}",
    "Region": "${NODE_REGION}",
    "Zone": "${NODE_ZONE}"
  },
  "WireGuard": {
    "Interface": "wg0",
    "ConfigPath": "/etc/wireguard/wg0.conf"
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
  },
  "Caddy": {
    "AdminApiUrl": "http://localhost:2019",
    "ConfigPath": "/etc/caddy/Caddyfile",
    "AcmeEmail": "${CADDY_ACME_EMAIL:-}",
    "UseAcmeStaging": ${caddy_staging},
    "DataDir": "/var/lib/caddy",
    "EnableAccessLog": true,
    "AccessLogPath": "/var/log/caddy/access.log",
    "AutoHttpsRedirect": true
  },
  "PortSecurity": {
    "MinAllowedPort": 1,
    "MaxAllowedPort": 65535,
    "BlockedPorts": [22, 2222, 3306, 5432, 27017, 6379, 9200, 5100, 51820, 2019, 16509]
  },
  "AuditLog": {
    "Enabled": true,
    "LogPath": "/var/log/decloud/audit.log",
    "MaxFileSizeMb": 100,
    "MaxFiles": 10,
    "RetentionDays": 90
  }
}
EOF

    chmod 640 "${CONFIG_DIR}/appsettings.Production.json"
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
Type=notify
WorkingDirectory=${INSTALL_DIR}/publish
ExecStart=${INSTALL_DIR}/publish/DeCloud.NodeAgent
Restart=always
RestartSec=10
Environment=ASPNETCORE_ENVIRONMENT=Production
Environment=DOTNET_ENVIRONMENT=Production
StandardOutput=append:${DECLOUD_LOG_DIR}/nodeagent.log
StandardError=append:${DECLOUD_LOG_DIR}/nodeagent.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log_success "Systemd service created"
}

configure_firewall() {
    log_step "Configuring firewall..."
    
    if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
        ufw allow ${AGENT_PORT}/tcp comment "DeCloud Agent API" > /dev/null 2>&1 || true
        
        if [ "$SKIP_WIREGUARD" = false ]; then
            ufw allow ${WIREGUARD_PORT}/udp comment "DeCloud WireGuard" > /dev/null 2>&1 || true
        fi
        
        log_success "UFW rules added"
    fi
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
# Status Display
# ============================================================
print_caddy_status() {
    if [ "$INSTALL_CADDY" = false ]; then
        return 0
    fi

    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Caddy Ingress Gateway:"
    echo "  ─────────────────────────────────────────────────────────────"

    if command -v caddy &> /dev/null && systemctl is-active --quiet caddy; then
        echo -e "  Status:          ${GREEN}Running${NC}"
        echo "  Admin API:       http://localhost:2019"
        echo "  Health Check:    http://localhost:8080/health"
        echo "  Ports:           80 (HTTP), 443 (HTTPS)"
        
        if [ -n "$CADDY_ACME_EMAIL" ]; then
            echo "  ACME Email:      $CADDY_ACME_EMAIL"
        else
            echo -e "  ACME Email:      ${YELLOW}Not configured${NC}"
        fi
        
        if [ "$CADDY_ACME_STAGING" = true ]; then
            echo -e "  ACME Mode:       ${YELLOW}STAGING${NC}"
        else
            echo "  ACME Mode:       Production"
        fi
    else
        echo -e "  Status:          ${RED}Not Running${NC}"
    fi
}

print_fail2ban_status() {
    if [ "$INSTALL_FAIL2BAN" = false ]; then
        return 0
    fi

    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  fail2ban DDoS Protection:"
    echo "  ─────────────────────────────────────────────────────────────"

    if command -v fail2ban-client &> /dev/null && systemctl is-active --quiet fail2ban; then
        echo -e "  Status:          ${GREEN}Running${NC}"
        local jails=$(fail2ban-client status 2>/dev/null | grep "Jail list" | cut -d: -f2 | tr -d '[:space:]')
        echo "  Active Jails:    $jails"
    else
        echo -e "  Status:          ${RED}Not Running${NC}"
    fi
}

print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       DeCloud Node Agent Installation Complete!              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Node Agent:      http://localhost:${AGENT_PORT}"
    echo "  Orchestrator:    ${ORCHESTRATOR_URL}"
    echo "  Configuration:   ${CONFIG_DIR}/appsettings.Production.json"
    echo "  Data directory:  ${DATA_DIR}"
    echo ""
    echo "  Commands:"
    echo "    Status:        sudo systemctl status decloud-node-agent"
    echo "    Logs:          sudo journalctl -u decloud-node-agent -f"
    echo "    Restart:       sudo systemctl restart decloud-node-agent"
    
    if [ "$SKIP_WIREGUARD" = false ]; then
        echo "    WireGuard:     sudo wg show"
    fi
    
    print_caddy_status
    print_fail2ban_status
    
    echo ""
    echo "  System resources:"
    echo "    CPU Cores:     ${CPU_CORES}"
    echo "    Memory:        ${MEMORY_MB}MB"
    echo "    Free Disk:     ${DISK_GB}GB"
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
    
    if [ -z "$ORCHESTRATOR_URL" ]; then
        log_error "Orchestrator URL is required"
        echo ""
        show_help
        exit 1
    fi
    
    # Checks
    check_root
    check_os
    check_architecture
    check_virtualization
    check_resources
    check_network
    
    echo ""
    log_info "All requirements met. Starting installation..."
    echo ""
    
    # Base installation
    install_base_dependencies
    install_dotnet
    install_libvirt
    install_wireguard
    configure_wireguard_hub
    
    # Ingress & Security
    install_caddy
    configure_caddy
    configure_caddy_firewall
    install_fail2ban
    configure_fail2ban
    setup_security_logging
    
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
    start_service
    
    # Start security services
    start_caddy
    start_fail2ban
    
    # Done
    print_summary
}

main "$@"
