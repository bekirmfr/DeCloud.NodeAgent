#!/bin/bash
#
# DeCloud Node Agent Installation Script
# 
# Installs and configures the Node Agent from signed release artifacts:
# - ASP.NET Core 8 Runtime (NOT the SDK — no compiler on production nodes)
# - cosign (for verifying release signatures)
# - KVM/QEMU/libvirt for virtualization
# - Docker + NVIDIA Container Toolkit (auto-detected for GPU nodes)
# - GPU CUDA shim (downloaded pre-built from signed release)
# - WireGuard for overlay networking
# - SSH CA for certificate authentication
#
# RELEASE MODEL:
# - All binaries fetched from signed GitHub Releases (cosign keyless verified)
# - Atomic version swap via /opt/decloud/publish symlink (rollback-friendly)
# - Pin a version with --version vX.Y.Z, default is latest stable
# - dht-node and blockstore-node Go binaries are NOT installed by this script:
#   the orchestrator's TemplateArtifact mechanism delivers them to system VMs
#   directly from the DeCloud.Builds release pipeline.
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

VERSION="2.2.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
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
PUBLIC_IP=""

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
BACKUP_DIR="/var/backups/decloud"
# ── Release / CI distribution ────────────────────────────────────────────
# As of v2.2.0, install.sh fetches signed release artifacts instead of
# cloning the source repo and building on the node. The git clone path is
# kept only as a fallback for development (--from-source flag, future).
#
# RELEASE_BASE_URL points at the GitHub Releases download root for the
# DeCloud.NodeAgent repo. dht-node and blockstore-node binaries are NOT
# distributed through this URL — those are owned by the DeCloud.Builds
# release pipeline and consumed by the orchestrator's TemplateArtifact
# system, not by install.sh.
RELEASE_BASE_URL_DEFAULT="https://github.com/bekirmfr/DeCloud.NodeAgent/releases/download"
RELEASE_API_LATEST="https://api.github.com/repos/bekirmfr/DeCloud.NodeAgent/releases/latest"
RELEASE_BASE_URL=""
RELEASE_VERSION=""

# Cosign keyless identity allow-list. Only signatures produced by the
# release.yml workflow in the DeCloud.NodeAgent repo are accepted.
COSIGN_IDENTITY_REGEXP_DEFAULT='^https://github\.com/bekirmfr/DeCloud\.NodeAgent/\.github/workflows/release\.yml@'
COSIGN_OIDC_ISSUER='https://token.actions.githubusercontent.com'
COSIGN_VERSION="v2.4.1"

# Staging dirs for non-publish files extracted from releases.
STAGE_DIR="${INSTALL_DIR}/stage"
GPU_PROXY_DIR="${INSTALL_DIR}/gpu-proxy"
RELEASE_CACHE_DIR="${INSTALL_DIR}/releases"
CURRENT_VERSION_FILE="${INSTALL_DIR}/current-version"
PREVIOUS_VERSION_FILE="${INSTALL_DIR}/previous-version"

# Logging
INSTALL_LOG="$LOG_DIR/install.log"
ENABLE_LOGGING=true  # Can be overridden by --no-log

# Ports (configurable - work with your existing infrastructure)
AGENT_PORT=5100
WIREGUARD_PORT=51821

# WireGuard
WIREGUARD_HUB_IP="10.10.0.1"
SKIP_WIREGUARD=false

# Git Sync
SKIP_DOWNLOAD=false

# SSH CA
SSH_CA_KEY_PATH="/etc/ssh/decloud_ca"
SSH_CA_PUB_PATH="/etc/ssh/decloud_ca.pub"

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
            --no-log)
                ENABLE_LOGGING=false
                shift
                ;;
            --orchestrator)
                ORCHESTRATOR_URL="$2"
                shift 2
                ;;
            --wallet)
                NODE_WALLET="$2"
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
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --skip-wireguard)
                SKIP_WIREGUARD=true
                shift
                ;;
            --skip-libvirt)
                SKIP_LIBVIRT=true
                shift
                ;;
            --version)
                RELEASE_VERSION="$2"
                shift 2
                ;;
            --release-base-url)
                RELEASE_BASE_URL="$2"
                shift 2
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

# ============================================================
# Initialize Logging
# ============================================================
init_logging() {
    if [ "$ENABLE_LOGGING" = false ]; then
        return 0
    fi
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    touch "$INSTALL_LOG"
    chmod 644 "$INSTALL_LOG"
    
    # Redirect all output to both terminal and log file
    exec 1> >(tee -a "$INSTALL_LOG")
    exec 2>&1
    
    echo "==================================================================="
    echo "DeCloud Node Agent Installation"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Version: $VERSION"
    echo "Command: $0 $@"
    echo "Log: $INSTALL_LOG"
    echo "==================================================================="
    echo ""
}

show_help() {
    cat << EOF
DeCloud Node Agent Installer v${VERSION}

Usage: $0 --orchestrator <url> --wallet <address> [options]

Required:
  --orchestrator <url>   Orchestrator URL (default: https://decloud.stackfi.tech)
  --wallet <address>     Node operator wallet address (0x...)

Network (all ports are configurable!):
  --port <port>          Agent API port (default: 5100)
  --wg-port <port>       WireGuard listen port (default: 51820)
  --skip-wireguard       Skip WireGuard installation

Other:
  --skip-libvirt         Skip libvirt installation (testing only)
  --skip-download        Skip release download (use existing files on disk)
  --version <vX.Y.Z>     Pin a specific release version (default: latest stable)
  --release-base-url <u> Override the release host (default: GitHub Releases)
  --help                 Show this help message

PORT REQUIREMENTS:
  This installer only needs TWO ports:
  - Agent API port (default 5100) - for orchestrator communication
  - WireGuard port (default 51820) - for overlay network

  Ports 80/443 are NOT required! Your existing web servers stay untouched.
  HTTP ingress is handled centrally by the Orchestrator.

After install, configure your node:
  sudo decloud configure --country TR --region eu-east --name MyNode
  sudo decloud register
  sudo decloud login

Examples:
  # Basic installation
  $0 --orchestrator https://decloud.stackfi.tech --wallet 0xYourWallet

  # Custom ports (if defaults conflict)
  $0 --orchestrator https://decloud.stackfi.tech --wallet 0xYourWallet --port 5200 --wg-port 51821
EOF
}

# ============================================================
# Requirement Checks
# ============================================================

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

    local prompted=false

    # /dev/tty resolves to the controlling terminal even when stdin is a pipe
    # (the canonical "curl | bash" case). Reading from /dev/tty lets us prompt
    # the human while the script body streams in over the pipe.
    local tty_available=false
    [ -r /dev/tty ] && [ -w /dev/tty ] && tty_available=true

    # Orchestrator URL
    if [ -z "$ORCHESTRATOR_URL" ]; then
        if [ "$tty_available" = true ]; then
            local default_orch="https://decloud.stackfi.tech"
            echo ""
            read -r -p "Orchestrator URL [${default_orch}]: " ORCHESTRATOR_URL </dev/tty
            ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-$default_orch}"
            prompted=true
        else
            _missing_param_error "--orchestrator <url>"
        fi
    fi

    # Wallet
    if [ -z "$NODE_WALLET" ]; then
        if [ "$tty_available" = true ]; then
            echo ""
            log_info "Wallet address is required for node identity and rewards."
            log_info "  Use your existing Ethereum wallet, or create one at https://metamask.io"
            echo ""
            while true; do
                read -r -p "Wallet address (0x...): " NODE_WALLET </dev/tty
                if validate_wallet_address "$NODE_WALLET"; then
                    break
                fi
                log_warn "Invalid format. Wallet must start with 0x followed by 40 hex characters."
            done
            prompted=true
        else
            _missing_param_error "--wallet 0x..."
        fi
    fi

    # Validate wallet format whether prompted or provided.
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

    # If anything was prompted, the install-params file written at parse time
    # is missing those values. Re-save from current variable state so future
    # 'decloud update' invocations have everything they need.
    if [ "$prompted" = true ]; then
        save_settings_json
    fi

    log_success "Orchestrator: $ORCHESTRATOR_URL"
    log_success "Wallet:       $NODE_WALLET"
}

# Print a clear, copy-pasteable error for non-interactive invocations
# (curl | bash with no /dev/tty) when a required flag is missing.
_missing_param_error() {
    local param="$1"
    echo ""
    log_error "Missing required parameter: $param"
    echo ""
    log_error "No interactive terminal detected (running under curl | bash, CI, or similar)."
    log_error "All required parameters must be provided as flags in this mode:"
    echo ""
    log_error "  curl -fsSL https://github.com/bekirmfr/DeCloud.NodeAgent/releases/latest/download/install.sh \\"
    log_error "    | sudo bash -s -- \\"
    log_error "        --orchestrator https://decloud.stackfi.tech \\"
    log_error "        --wallet 0xYourWalletAddress"
    echo ""
    log_error "Or download install.sh first and run it interactively to be prompted:"
    log_error "  curl -fsSL https://github.com/bekirmfr/DeCloud.NodeAgent/releases/latest/download/install.sh -o install.sh"
    log_error "  sudo bash install.sh"
    echo ""
    exit 1
}

# Reconstruct /etc/decloud/install-params from the current variable state.
# Called after check_required_params if any value was prompted, so that
# 'decloud update' has a complete params file to re-run install non-interactively.
save_settings_json() {
    mkdir -p /etc/decloud
    if [ -f /etc/decloud/settings.json ]; then
        local tmp; tmp=$(mktemp)
        jq --arg o "$ORCHESTRATOR_URL" --arg w "$NODE_WALLET" \
           '.orchestrator_url = $o | .wallet = $w' \
           /etc/decloud/settings.json > "$tmp" \
           && mv "$tmp" /etc/decloud/settings.json
    else
        cat > /etc/decloud/settings.json <<SETTINGSEOF
{
  "version": 1,
  "orchestrator_url": "${ORCHESTRATOR_URL}",
  "wallet": "${NODE_WALLET}",
  "name": "$(hostname)",
  "region": "default",
  "zone": "default",
  "agent_port": ${AGENT_PORT},
  "wireguard_port": ${WIREGUARD_PORT}
}
SETTINGSEOF
    fi
    chmod 600 /etc/decloud/settings.json
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
    local private_ip=""
    
    private_ip=$(hostname -I | awk '{print $1}')
    
    if [ -n "$PUBLIC_IP" ]; then
        log_info "Public IP: $PUBLIC_IP"
        log_info "Private IP: $private_ip"
        
        # Check if behind NAT/CGNAT
        if [[ "$PUBLIC_IP" != "$private_ip" ]]; then
            if [[ "$PUBLIC_IP" =~ ^100\. ]] || \
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
        curl wget git jq tar xz-utils apt-transport-https ca-certificates \
        gnupg lsb-release software-properties-common > /dev/null 2>&1
    
    install_cosign

    log_success "Base dependencies installed"
}

# Cosign is the trust root for verifying release manifests.
# Static binary, single-file install, pinned version.
install_cosign() {
    if command -v cosign &>/dev/null; then
        # cosign 2.x output begins with multi-line ASCII art; the GitVersion
        # line is the reliable identifier across versions.
        local cosign_ver
        cosign_ver=$(cosign version 2>/dev/null | awk '/GitVersion/{print $NF; exit}')
        log_info "cosign already installed: ${cosign_ver:-installed}"
        return 0
    fi

    local cosign_arch
    case "$(uname -m)" in
        x86_64|amd64)  cosign_arch="amd64" ;;
        aarch64|arm64) cosign_arch="arm64" ;;
        *)
            log_error "Unsupported architecture for cosign: $(uname -m)"
            return 1
            ;;
    esac

    local url="https://github.com/sigstore/cosign/releases/download/${COSIGN_VERSION}/cosign-linux-${cosign_arch}"
    local dest="/usr/local/bin/cosign"

    log_info "Installing cosign ${COSIGN_VERSION} (${cosign_arch})..."
    if ! curl -fsSL --retry 3 --retry-delay 2 "$url" -o "${dest}.tmp"; then
        log_error "Failed to download cosign from $url"
        rm -f "${dest}.tmp"
        return 1
    fi
    chmod +x "${dest}.tmp"
    mv "${dest}.tmp" "$dest"

    if ! cosign version &>/dev/null; then
        log_error "cosign installation verification failed"
        rm -f "$dest"
        return 1
    fi
    log_success "cosign installed"
}

install_dotnet() {
    log_step "Installing ASP.NET Core 8 Runtime..."

    # The release pipeline ships framework-dependent publish output, so the
    # node only needs the ASP.NET Core runtime — not the SDK. Removing the
    # SDK eliminates ~700 MB of toolchain and a meaningful attack surface
    # (compilers, NuGet restore paths, build-time package downloads).

    if command -v dotnet &>/dev/null; then
        if dotnet --list-runtimes 2>/dev/null | grep -q '^Microsoft\.AspNetCore\.App 8\.'; then
            local rt
            rt=$(dotnet --list-runtimes 2>/dev/null | grep '^Microsoft\.AspNetCore\.App 8\.' | head -1)
            log_success "ASP.NET Core 8 runtime already installed: $rt"
            return 0
        fi
        log_warn "dotnet present but no ASP.NET Core 8 runtime — installing"
    fi

    # Add Microsoft repository
    log_info "Adding Microsoft package repository..."
    if ! wget -q "https://packages.microsoft.com/config/${OS}/${OS_VERSION}/packages-microsoft-prod.deb" \
            -O /tmp/packages-microsoft-prod.deb 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        log_error "Failed to download Microsoft repository package"
        log_error "Check network connectivity and logs: $LOG_DIR/install.log"
        return 1
    fi

    if ! dpkg -i /tmp/packages-microsoft-prod.deb 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        log_error "Failed to add Microsoft repository"
        log_error "Check logs: $LOG_DIR/install.log"
        rm -f /tmp/packages-microsoft-prod.deb
        return 1
    fi

    rm -f /tmp/packages-microsoft-prod.deb
    log_info "Microsoft repository added"

    log_info "Updating package cache..."
    apt-get update -qq 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null || true

    log_info "Installing aspnetcore-runtime-8.0..."
    if ! apt-get install -y aspnetcore-runtime-8.0 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        log_error "Failed to install aspnetcore-runtime-8.0"
        log_error "Check logs: $LOG_DIR/install.log"
        log_error "Try manually: sudo apt-get install -y aspnetcore-runtime-8.0"
        return 1
    fi

    if ! command -v dotnet &>/dev/null; then
        log_error "dotnet command not found after installation"
        return 1
    fi

    if ! dotnet --list-runtimes 2>/dev/null | grep -q '^Microsoft\.AspNetCore\.App 8\.'; then
        log_error "ASP.NET Core 8 runtime not present after install"
        return 1
    fi

    log_success "ASP.NET Core 8 runtime installed"
    return 0
}

# ============================================================
# Release fetch / verify helpers (used by download_node_agent
# and build_gpu_proxy)
# ============================================================

# Resolve the effective release base URL.
release_base_url() {
    echo "${RELEASE_BASE_URL:-$RELEASE_BASE_URL_DEFAULT}"
}

# Map node arch (uname -m) to the asset arch tag used in release filenames.
release_asset_arch() {
    case "$(uname -m)" in
        x86_64|amd64)  echo "linux-amd64" ;;
        aarch64|arm64) echo "linux-arm64" ;;
        *)
            log_error "Unsupported architecture: $(uname -m)"
            return 1
            ;;
    esac
}

# Resolve "latest stable" via the GitHub API when --version was not given.
# Fails loudly rather than silently installing HEAD-equivalent.
resolve_release_version() {
    if [ -n "$RELEASE_VERSION" ]; then
        echo "$RELEASE_VERSION"
        return 0
    fi

    log_info "No --version given, resolving latest stable release..." >&2
    local latest
    latest=$(curl -fsSL --retry 3 "$RELEASE_API_LATEST" 2>/dev/null \
        | jq -r '.tag_name // empty' 2>/dev/null)

    if [ -z "$latest" ] || [ "$latest" = "null" ]; then
        log_error "Could not resolve latest release from $RELEASE_API_LATEST"
        log_error "Pin a version explicitly with --version vX.Y.Z"
        return 1
    fi

    RELEASE_VERSION="$latest"
    log_info "Latest release: $RELEASE_VERSION" >&2
    echo "$RELEASE_VERSION"
}

# Download manifest.json + signature, verify cosign keyless signature.
# Cached under $RELEASE_CACHE_DIR/manifest.<version>.json after verification.
fetch_and_verify_manifest() {
    local version="$1"
    local base; base=$(release_base_url)

    mkdir -p "$RELEASE_CACHE_DIR"
    local manifest="${RELEASE_CACHE_DIR}/manifest.${version}.json"
    local sig="${manifest}.sig"
    local cert="${manifest}.pem"

    log_info "Fetching manifest for ${version}..."
    if ! curl -fsSL --retry 3 "${base}/${version}/manifest.json"     -o "${manifest}.tmp" \
       || ! curl -fsSL --retry 3 "${base}/${version}/manifest.json.sig" -o "${sig}.tmp" \
       || ! curl -fsSL --retry 3 "${base}/${version}/manifest.json.pem" -o "${cert}.tmp"; then
        log_error "Failed to download manifest artifacts from ${base}/${version}/"
        rm -f "${manifest}.tmp" "${sig}.tmp" "${cert}.tmp"
        return 1
    fi

    log_info "Verifying manifest signature (cosign keyless)..."
    local identity_regexp="${COSIGN_IDENTITY_REGEXP_OVERRIDE:-$COSIGN_IDENTITY_REGEXP_DEFAULT}"
    if ! cosign verify-blob \
            --certificate-identity-regexp "$identity_regexp" \
            --certificate-oidc-issuer     "$COSIGN_OIDC_ISSUER" \
            --signature   "${sig}.tmp" \
            --certificate "${cert}.tmp" \
            "${manifest}.tmp" >/dev/null 2>&1; then
        log_error "MANIFEST SIGNATURE VERIFICATION FAILED for ${version}"
        log_error "  Expected identity regex: $identity_regexp"
        log_error "  Expected OIDC issuer:    $COSIGN_OIDC_ISSUER"
        log_error "Refusing to install unverified release."
        rm -f "${manifest}.tmp" "${sig}.tmp" "${cert}.tmp"
        return 1
    fi

    mv "${manifest}.tmp" "$manifest"
    mv "${sig}.tmp"      "$sig"
    mv "${cert}.tmp"     "$cert"
    log_success "Manifest verified: $manifest"
}

# Download an artifact and verify its sha256 against the manifest.
# Args: <version> <artifact-name> <out-path>
download_artifact_verified() {
    local version="$1"
    local name="$2"
    local out="$3"
    local manifest="${RELEASE_CACHE_DIR}/manifest.${version}.json"
    local base; base=$(release_base_url)

    local expected
    expected=$(jq -r --arg n "$name" '.artifacts[$n].sha256 // empty' "$manifest")
    if [ -z "$expected" ]; then
        log_error "Artifact '$name' not present in manifest for $version"
        return 1
    fi

    log_info "Downloading $name..."
    if ! curl -fsSL --retry 3 "${base}/${version}/${name}" -o "${out}.tmp"; then
        log_error "Download failed: ${base}/${version}/${name}"
        return 1
    fi

    local actual
    actual=$(sha256sum "${out}.tmp" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        log_error "SHA256 mismatch for $name"
        log_error "  expected: $expected"
        log_error "  actual:   $actual"
        rm -f "${out}.tmp"
        return 1
    fi

    mv "${out}.tmp" "$out"
    log_success "Verified $name"
}

install_libvirt() {
    log_step "Installing libvirt/KVM and virtualization tools..."
    
    if [ "$SKIP_LIBVIRT" = true ]; then
        log_warn "Skipping libvirt installation (--skip-libvirt)"
        return 0
    fi
    
    # Check if already installed
    if command -v virsh &> /dev/null && systemctl is-active --quiet libvirtd; then
        local virsh_version=$(virsh --version)
        log_success "libvirt already installed and running (version $virsh_version)"
        return 0
    fi
    
    PACKAGES="qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst"
    PACKAGES="$PACKAGES cloud-image-utils genisoimage qemu-utils"
    PACKAGES="$PACKAGES libguestfs-tools openssh-client"
    PACKAGES="$PACKAGES sysbench"
    
    log_info "Installing virtualization packages..."
    log_info "This may take several minutes..."
    
    # Update package cache first
    if ! apt-get update -qq 2>&1 | grep -qi "err:"; then
        log_info "Package cache updated"
    fi
    
    # Install packages with visible errors
    if ! apt-get install -y $PACKAGES 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        log_error "Failed to install libvirt packages"
        log_error "Check logs: $LOG_DIR/install.log"
        log_error "Try manually: sudo apt-get install -y qemu-kvm libvirt-daemon-system"
        return 1
    fi
    
    # Load nbd module
    modprobe nbd max_part=8 2>/dev/null || true
    # vhost_vsock: required for GPU proxy vsock communication (bare metal only)
    if [ "$IS_WSL2" != true ]; then
        modprobe vhost_vsock 2>/dev/null || true
    fi
    echo "nbd" >> /etc/modules-load.d/decloud.conf 2>/dev/null || true
    if [ "$IS_WSL2" != true ]; then
        grep -q "vhost_vsock" /etc/modules-load.d/decloud.conf 2>/dev/null || \
            echo "vhost_vsock" >> /etc/modules-load.d/decloud.conf
    fi
    
    # Enable and start libvirtd
    systemctl enable libvirtd --quiet 2>/dev/null || true
    systemctl start libvirtd 2>/dev/null || true
    
    # Give libvirtd a moment to start
    sleep 2
    
    # Verify libvirtd is running
    if ! systemctl is-active --quiet libvirtd; then
        log_error "libvirtd failed to start"
        log_error "Check: sudo systemctl status libvirtd"
        log_error "Logs: sudo journalctl -u libvirtd -n 50"
        return 1
    fi
    
    # Setup default network
    if ! virsh net-info default &>/dev/null; then
        if [ -f /usr/share/libvirt/networks/default.xml ]; then
            virsh net-define /usr/share/libvirt/networks/default.xml 2>/dev/null || true
        fi
    fi
    virsh net-autostart default 2>/dev/null || true
    virsh net-start default 2>/dev/null || true
    
    # Verify virsh command works
    if ! command -v virsh &> /dev/null; then
        log_error "virsh command not found after installation"
        return 1
    fi
    
    log_success "libvirt installed and configured"
    log_info "libvirtd status: $(systemctl is-active libvirtd)"
    
    return 0
}

# ============================================================
# ARM64 QEMU/KVM Installation
# ============================================================
install_arm_virtualization() {
    log_step "Installing ARM64 virtualization support..."
    
    ARCH=$(uname -m)
    
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        log_info "Detected ARM64 architecture"
        
        # Install ARM64-specific QEMU and firmware
        apt-get install -y \
            qemu-system-arm \
            qemu-efi-aarch64 \
            qemu-efi \
            ipxe-qemu \
            || log_error "Failed to install ARM64 virtualization packages"
        
        log_success "ARM64 QEMU installed"
        
        # Verify AAVMF firmware for ARM64 VMs
        if [ ! -f "/usr/share/AAVMF/AAVMF_CODE.fd" ]; then
            log_warn "AAVMF firmware not found, some ARM64 VMs may not boot properly"
            log_info "Installing AAVMF firmware..."
            
            apt-get install -y qemu-efi-aarch64 || log_warn "Could not install AAVMF firmware"
        else
            log_success "AAVMF firmware verified"
        fi
        
        # Enable KVM for ARM64
        if [ -e /dev/kvm ]; then
            log_success "KVM device available for ARM64"
        else
            log_warn "KVM device not found - VMs will run without hardware acceleration"
            log_info "Attempting to load KVM module..."
            modprobe kvm || log_warn "Could not load KVM module"
        fi
        
    else
        log_info "x86_64 architecture detected - using standard QEMU"
    fi
}

# ============================================================
# Architecture-Specific Optimizations
# ============================================================
configure_architecture_optimizations() {
    log_step "Configuring architecture-specific optimizations..."
    
    ARCH=$(uname -m)
    
    # CPU Governor for ARM devices (performance vs power saving)
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        log_info "Setting CPU governor for ARM64..."
        
        # Install cpufrequtils for ARM CPU management
        apt-get install -y cpufrequtils || log_warn "Could not install cpufrequtils"
        
        # Set performance governor for compute nodes
        if command -v cpufreq-set &> /dev/null; then
            for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
                if [ -f "$cpu" ]; then
                    echo "performance" > "$cpu" 2>/dev/null || true
                fi
            done
            log_success "CPU governor set to performance mode"
        fi
    fi
}

# ============================================================
# ARM64 Image Cache Pre-warming (Optional)
# ============================================================
prewarm_arm_images() {
    log_step "Pre-warming ARM64 image cache (optional)..."
    
    ARCH=$(uname -m)
    
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        log_info "Downloading popular ARM64 cloud images..."
        
        IMAGE_CACHE_DIR="/var/lib/decloud/images"
        mkdir -p "$IMAGE_CACHE_DIR"
        
        # Ubuntu 22.04 ARM64 (most popular)
        if [ ! -f "$IMAGE_CACHE_DIR/ubuntu-22.04-arm64.img" ]; then
            log_info "Downloading Ubuntu 22.04 ARM64 (350MB)..."
            wget -q --show-progress \
                -O "$IMAGE_CACHE_DIR/ubuntu-22.04-arm64.img" \
                "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-arm64.img" \
                || log_warn "Could not download Ubuntu 22.04 ARM64 image"
        fi
        
        log_success "ARM64 image cache pre-warmed"
    fi
}

# ============================================================
# NVIDIA GPU Detection (supports WSL2 + bare-metal)
# ============================================================
NVIDIA_SMI_PATH=""

detect_nvidia_gpu() {
    # Already detected in this run
    if [ -n "$NVIDIA_SMI_PATH" ]; then
        return 0
    fi

    # 1. Standard PATH (bare-metal Linux, /usr/bin/nvidia-smi)
    if command -v nvidia-smi &> /dev/null; then
        NVIDIA_SMI_PATH="$(command -v nvidia-smi)"
        return 0
    fi

    # 2. WSL2-specific path (Windows GPU passthrough)
    if [ -x "/usr/lib/wsl/lib/nvidia-smi" ]; then
        NVIDIA_SMI_PATH="/usr/lib/wsl/lib/nvidia-smi"
        log_info "WSL2 GPU detected via ${NVIDIA_SMI_PATH}"
        return 0
    fi

    # 3. Fallback: check for WSL2 CUDA libraries (nvidia-smi may not exist but GPU is there)
    if ls /usr/lib/wsl/lib/libcuda.so* &> /dev/null; then
        NVIDIA_SMI_PATH="wsl-cuda-only"
        log_info "WSL2 CUDA libraries detected (GPU available)"
        return 0
    fi

    # 4. PCI bus scan (bare-metal only, doesn't work in WSL2)
    if lspci 2>/dev/null | grep -qi 'nvidia'; then
        NVIDIA_SMI_PATH="pci-detected"
        return 0
    fi

    return 1
}

# ============================================================
# GPU Mode Detection (passthrough vs proxy vs none)
# Mirrors ResourceDiscoveryService.cs logic
# ============================================================
GPU_MODE="none"
IS_WSL2=false

# ============================================================
# WSL2 Detection (runs independently of GPU detection)
# Sets IS_WSL2=true when running inside Windows Subsystem for Linux 2.
# Called early in main() so every downstream function can rely on it.
# ============================================================
detect_wsl2() {
    if grep -qi 'microsoft\|wsl' /proc/version 2>/dev/null || [ -e /dev/dxg ]; then
        IS_WSL2=true
        log_info "WSL2 environment detected"
    else
        IS_WSL2=false
    fi
}

detect_gpu_mode() {
    log_step "Detecting GPU sharing mode..."

    # 1. No GPU → nothing to do
    if ! detect_nvidia_gpu; then
        GPU_MODE="none"
        log_info "No NVIDIA GPU detected — GPU_MODE=none"
        return 0
    fi

    # 2. Check WSL2 (passthrough never available under WSL2)
    # IS_WSL2 is already set by detect_wsl2() called early in main().
    if [ "$IS_WSL2" = true ]; then
        GPU_MODE="proxy"
        log_info "WSL2 environment — GPU_MODE=proxy (no passthrough)"
        return 0
    fi

    # 3. Try to get the GPU PCI address via nvidia-smi
    local pci_addr=""
    if [ -n "$NVIDIA_SMI_PATH" ] && [ "$NVIDIA_SMI_PATH" != "pci-detected" ] && [ "$NVIDIA_SMI_PATH" != "wsl-cuda-only" ]; then
        pci_addr=$($NVIDIA_SMI_PATH --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
    fi

    # 4. Normalise PCI address (00000000:01:00.0 → 0000:01:00.0)
    if [ -n "$pci_addr" ]; then
        # nvidia-smi returns e.g. "00000000:01:00.0" — trim leading domain zeros
        pci_addr=$(echo "$pci_addr" | sed -E 's/^0+:/0000:/')
        # Ensure lowercase for sysfs lookup
        pci_addr=$(echo "$pci_addr" | tr '[:upper:]' '[:lower:]')
    fi

    # 5. Check IOMMU group for this device
    local has_iommu=false
    if [ -n "$pci_addr" ] && [ -e "/sys/bus/pci/devices/${pci_addr}/iommu_group" ]; then
        has_iommu=true
        local iommu_group
        iommu_group=$(basename "$(readlink "/sys/bus/pci/devices/${pci_addr}/iommu_group")")
        log_info "GPU PCI ${pci_addr} is in IOMMU group ${iommu_group}"
    fi

    # 6. Check vfio-pci kernel module
    local has_vfio=false
    if modinfo vfio-pci &>/dev/null; then
        has_vfio=true
    fi

    # 7. Decide mode
    if [ "$has_iommu" = true ] && [ "$has_vfio" = true ]; then
        GPU_MODE="passthrough"
        log_success "GPU_MODE=passthrough (IOMMU + vfio-pci available)"
    else
        GPU_MODE="proxy"
        if [ "$has_iommu" = false ]; then
            log_info "IOMMU not enabled for GPU — falling back to proxy mode"
        elif [ "$has_vfio" = false ]; then
            log_info "vfio-pci module not available — falling back to proxy mode"
        fi
        log_success "GPU_MODE=proxy (daemon + shim required)"
    fi

    return 0
}

# ============================================================
# CUDA Toolkit Detection / Installation (for daemon build)
# ============================================================
CUDA_HOME=""

install_cuda_toolkit() {
    log_step "Checking CUDA toolkit (needed to build GPU proxy daemon)..."

    # Only needed in proxy mode — passthrough uses VFIO, not the daemon
    if [ "$GPU_MODE" != "proxy" ]; then
        log_info "GPU_MODE=${GPU_MODE} — CUDA toolkit not needed (proxy daemon not required)"
        return 0
    fi

    # 1. Honour explicit CUDA_HOME
    if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/include" ] && [ -d "$CUDA_HOME/lib64" ]; then
        log_success "CUDA toolkit found at CUDA_HOME=$CUDA_HOME"
        return 0
    fi

    # 2. Standard path: /usr/local/cuda
    if [ -d "/usr/local/cuda/include" ] && [ -d "/usr/local/cuda/lib64" ]; then
        CUDA_HOME="/usr/local/cuda"
        log_success "CUDA toolkit found at $CUDA_HOME"
        return 0
    fi

    # 3. WSL2: CUDA libraries live under /usr/lib/wsl/lib but headers may be missing
    if [ "$IS_WSL2" = true ] && ls /usr/lib/wsl/lib/libcuda.so* &>/dev/null; then
        # Check if cuda headers exist anywhere
        if [ -f "/usr/include/cuda.h" ] || [ -f "/usr/local/cuda/include/cuda.h" ]; then
            CUDA_HOME="/usr/local/cuda"
            log_success "CUDA headers + WSL2 runtime libraries found"
            return 0
        fi
        log_info "WSL2 CUDA runtime found but headers missing — will install toolkit"
    fi

    # 4. Check if already installed via apt
    if dpkg -l cuda-toolkit-* 2>/dev/null | grep -q '^ii'; then
        # Find the path
        local cuda_ver
        cuda_ver=$(dpkg -l cuda-toolkit-* 2>/dev/null | grep '^ii' | head -1 | awk '{print $2}' | sed 's/cuda-toolkit-//')
        if [ -d "/usr/local/cuda-${cuda_ver}" ]; then
            CUDA_HOME="/usr/local/cuda-${cuda_ver}"
        elif [ -d "/usr/local/cuda" ]; then
            CUDA_HOME="/usr/local/cuda"
        fi
        if [ -n "$CUDA_HOME" ]; then
            log_success "CUDA toolkit already installed (apt): $CUDA_HOME"
            return 0
        fi
    fi

    # 5. Check nvidia-cuda-toolkit (lighter-weight package on Debian/Ubuntu)
    if dpkg -l nvidia-cuda-toolkit 2>/dev/null | grep -q '^ii'; then
        # This package puts headers in /usr/include and libs in /usr/lib
        if [ -f "/usr/include/cuda.h" ]; then
            # Create a pseudo CUDA_HOME that the Makefile can use
            CUDA_HOME="/usr"
            log_success "nvidia-cuda-toolkit found (headers in /usr/include)"
            return 0
        fi
    fi

    # 6. Install nvidia-cuda-toolkit (lightweight — headers + runtime stubs)
    log_info "CUDA toolkit not found — installing nvidia-cuda-toolkit..."

    apt-get update -qq 2>/dev/null
    if apt-get install -y nvidia-cuda-toolkit 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        if [ -f "/usr/include/cuda.h" ]; then
            CUDA_HOME="/usr"
            log_success "nvidia-cuda-toolkit installed (CUDA_HOME=/usr)"
            return 0
        elif [ -d "/usr/local/cuda" ]; then
            CUDA_HOME="/usr/local/cuda"
            log_success "nvidia-cuda-toolkit installed (CUDA_HOME=/usr/local/cuda)"
            return 0
        fi
    fi

    log_warn "Could not install CUDA toolkit"
    log_warn "GPU proxy daemon will NOT be built (shim will still be built)"
    log_info "To fix: install CUDA toolkit manually and re-run, or set CUDA_HOME"
    return 0
}

# ============================================================
# GPU Proxy Build (daemon + shim)
# ============================================================
build_gpu_proxy() {
    log_step "Setting up GPU proxy components (release-mode)..."

    if [ "$GPU_MODE" = "none" ]; then
        log_info "No GPU — skipping GPU proxy setup"
        return 0
    fi

    # The shim install location used by the 9p share for guest VMs.
    # CRITICAL: shims go ONLY here, never to /usr/local/lib/, because that
    # would poison the host's ldconfig cache and cause the daemon to load
    # our shim instead of the real CUDA runtime (circular dependency).
    local SHIM_DIR="/usr/local/lib/decloud-gpu-shim"

    # ── 1. Resolve version + arch ────────────────────────────────────────
    local version asset_arch
    if [ ! -f "$CURRENT_VERSION_FILE" ]; then
        log_error "current-version not recorded — download_node_agent must run first"
        return 1
    fi
    version=$(cat "$CURRENT_VERSION_FILE")
    asset_arch=$(release_asset_arch)

    # GPU shim is published for linux-amd64 only — NVIDIA's proprietary
    # stack targets x86_64 and the shim links against it. ARM64 nodes
    # can't run the shim, but they can still passthrough physical GPUs.
    if [ "$asset_arch" != "linux-amd64" ]; then
        log_warn "GPU proxy shim is published for linux-amd64 only — skipping on ${asset_arch}"
        return 0
    fi

    # ── 2. Verify the artifact exists in the manifest ────────────────────
    local manifest="${RELEASE_CACHE_DIR}/manifest.${version}.json"
    local shim_name="decloud-gpu-shim-${version}-${asset_arch}.tar.gz"
    if ! jq -e --arg n "$shim_name" '.artifacts[$n]' "$manifest" >/dev/null 2>&1; then
        log_warn "GPU shim artifact not in manifest for ${version} — skipping"
        log_warn "  (older releases may not include shims; rebuild from source if needed)"
        return 0
    fi

    # ── 3. Download + verify ─────────────────────────────────────────────
    local shim_tar="${RELEASE_CACHE_DIR}/${shim_name}"
    download_artifact_verified "$version" "$shim_name" "$shim_tar" || return 1

    # ── 4. Extract to staging ────────────────────────────────────────────
    rm -rf "${GPU_PROXY_DIR}/build"
    mkdir -p "${GPU_PROXY_DIR}/build"
    if ! tar -xzf "$shim_tar" -C "${GPU_PROXY_DIR}/build"; then
        log_error "Failed to extract GPU shim tarball"
        return 1
    fi
    log_success "GPU shims extracted to ${GPU_PROXY_DIR}/build"

    # ── 5. Install to SHIM_DIR ───────────────────────────────────────────
    # Mirrors the layout produced by 'make install-all-shims-compat'.
    # The compat-built runtime shim is delivered under TWO names because
    # different guest workloads dlopen different ones (Ollama looks up
    # libdecloud_cuda_shim.so, PyTorch dlopens libcudart.so.12).
    install -d "$SHIM_DIR"

    install -m 644 "${GPU_PROXY_DIR}/build/libdecloud_cuda_shim-compat.so" \
        "${SHIM_DIR}/libdecloud_cuda_shim.so"
    install -m 644 "${GPU_PROXY_DIR}/build/libdecloud_cuda_shim-compat.so" \
        "${SHIM_DIR}/libcudart.so.12"

    install -m 644 "${GPU_PROXY_DIR}/build/libcuda.so.1"            "$SHIM_DIR/"
    install -m 644 "${GPU_PROXY_DIR}/build/libnvidia-ml.so.1"       "$SHIM_DIR/"
    install -m 644 "${GPU_PROXY_DIR}/build/libcublas_stub.so"       "$SHIM_DIR/"
    install -m 644 "${GPU_PROXY_DIR}/build/libcublasLt_stub.so"     "$SHIM_DIR/"
    install -m 644 "${GPU_PROXY_DIR}/build/libcudnn_stub.so"        "$SHIM_DIR/"
    install -m 644 "${GPU_PROXY_DIR}/build/libcuda_pytorch_stubs.so" "$SHIM_DIR/"

    log_success "Shims installed to $SHIM_DIR/"

    # ── 6. Daemon: NOT shipped via release ──────────────────────────────
    # The daemon links against the host's exact CUDA runtime version, so
    # a single prebuilt binary is unsafe across CUDA versions. Document
    # the manual build path and continue. VM creation in proxy mode will
    # surface the missing-daemon error clearly when the user tries it.
    if [ "$GPU_MODE" = "proxy" ]; then
        if [ -x /usr/local/bin/gpu-proxy-daemon ]; then
            log_info "Existing gpu-proxy-daemon detected at /usr/local/bin/gpu-proxy-daemon"
            log_info "  Daemon was NOT replaced — release pipeline does not ship it."
            log_info "  Rebuild manually if needed (see docs/gpu-proxy-daemon.md)"
        else
            log_warn "GPU_MODE=proxy but no daemon installed."
            log_warn "  The daemon must be built locally against your CUDA toolkit."
            log_warn "  See: https://github.com/bekirmfr/DeCloud.NodeAgent/blob/${version}/docs/gpu-proxy-daemon.md"
        fi
    fi

    # ── 7. Verify symbol counts on installed stubs ──────────────────────
    # Same checks as before — catches a broken/stale upload as early as
    # possible, before VM creation fails inside a guest.
    local cublas_syms=0 cublaslt_syms=0 cudnn_syms=0
    [ -f "$SHIM_DIR/libcublas_stub.so"   ] && cublas_syms=$(objdump -T "$SHIM_DIR/libcublas_stub.so"   2>/dev/null | grep -c "libcublas.so.12" || true)
    [ -f "$SHIM_DIR/libcublasLt_stub.so" ] && cublaslt_syms=$(objdump -T "$SHIM_DIR/libcublasLt_stub.so" 2>/dev/null | grep -c "libcublasLt.so.12" || true)
    [ -f "$SHIM_DIR/libcudnn_stub.so"    ] && cudnn_syms=$(objdump -T "$SHIM_DIR/libcudnn_stub.so"    2>/dev/null | grep -c "libcudnn.so.8"   || true)

    if [ "$cublas_syms"   -lt 20 ]; then log_error "libcublas_stub.so   has only $cublas_syms versioned symbols (expected 20+)"; else log_success "libcublas_stub.so   OK ($cublas_syms versioned symbols)"; fi
    if [ "$cublaslt_syms" -lt 28 ]; then log_error "libcublasLt_stub.so has only $cublaslt_syms versioned symbols (expected 28)";  else log_success "libcublasLt_stub.so OK ($cublaslt_syms versioned symbols)"; fi
    if [ "$cudnn_syms"    -lt 84 ]; then log_error "libcudnn_stub.so    has only $cudnn_syms versioned symbols (expected 84+)"; else log_success "libcudnn_stub.so    OK ($cudnn_syms versioned symbols)"; fi

    # ── 8. Summary ───────────────────────────────────────────────────────
    echo ""
    log_info "┌──────────────────────────────────────────────────┐"
    log_info "│ GPU Proxy Setup Summary                           │"
    log_info "├──────────────────────────────────────────────────┤"
    log_info "│ GPU Mode:       ${GPU_MODE}"
    log_info "│ WSL2:           ${IS_WSL2}"
    log_info "│ Release ver:    ${version}"
    log_info "│ Runtime Shim:   $([ -f $SHIM_DIR/libdecloud_cuda_shim.so ] && echo 'installed' || echo 'missing')"
    log_info "│ Driver Shim:    $([ -f $SHIM_DIR/libcuda.so.1 ] && echo 'installed' || echo 'missing')"
    log_info "│ NVML Shim:      $([ -f $SHIM_DIR/libnvidia-ml.so.1 ] && echo 'installed' || echo 'missing')"
    log_info "│ cuBLAS Stub:    $([ -f $SHIM_DIR/libcublas_stub.so ] && echo 'installed' || echo 'missing')"
    log_info "│ cuBLAS Lt Stub: $([ -f $SHIM_DIR/libcublasLt_stub.so ] && echo 'installed' || echo 'missing')"
    log_info "│ cuDNN Stub:     $([ -f $SHIM_DIR/libcudnn_stub.so ] && echo 'installed' || echo 'missing')"
    log_info "│ PyTorch Stubs:  $([ -f $SHIM_DIR/libcuda_pytorch_stubs.so ] && echo 'installed' || echo 'missing')"
    log_info "│ Daemon:         $([ -x /usr/local/bin/gpu-proxy-daemon ] && echo 'installed (manual build)' || echo 'not installed')"
    log_info "└──────────────────────────────────────────────────┘"
    echo ""

    return 0
}

# ============================================================
# Docker Installation (for GPU container sharing)
# ============================================================
install_docker() {
    log_step "Checking Docker (for GPU container workloads)..."

    if ! detect_nvidia_gpu; then
        log_info "No NVIDIA GPU detected — skipping Docker installation"
        return 0
    fi

    log_info "NVIDIA GPU detected — Docker required for GPU container sharing"
    log_info "GPU detection method: ${NVIDIA_SMI_PATH}"

    # Check if already installed and running
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        local docker_version=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')
        log_success "Docker already installed and running (v${docker_version})"
        return 0
    fi

    # Check if installed but not running
    if command -v docker &> /dev/null; then
        log_info "Docker installed but not running — starting..."
        systemctl enable docker --quiet 2>/dev/null || true
        systemctl start docker 2>/dev/null || true
        sleep 2
        if docker info &> /dev/null; then
            log_success "Docker started successfully"
            return 0
        fi
        log_warn "Docker failed to start — reinstalling..."
    fi

    log_info "Installing Docker via official script..."

    # Use Docker's convenience script (supports Ubuntu/Debian)
    if ! curl -fsSL https://get.docker.com -o /tmp/get-docker.sh 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        log_error "Failed to download Docker install script"
        log_warn "GPU container support will not be available"
        return 0  # Non-fatal: node can still run VMs via libvirt
    fi

    if ! sh /tmp/get-docker.sh 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        log_error "Docker installation failed"
        log_warn "GPU container support will not be available"
        rm -f /tmp/get-docker.sh
        return 0
    fi

    rm -f /tmp/get-docker.sh

    # Enable and start Docker
    systemctl enable docker --quiet 2>/dev/null || true
    systemctl start docker 2>/dev/null || true
    sleep 2

    # Verify
    if ! docker info &> /dev/null; then
        log_error "Docker installed but failed to start"
        log_warn "GPU container support will not be available"
        log_info "Check: sudo systemctl status docker"
        return 0
    fi

    local docker_version=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')
    log_success "Docker installed and running (v${docker_version})"
    return 0
}

# ============================================================
# NVIDIA Container Toolkit (for --gpus all support)
# ============================================================
install_nvidia_container_toolkit() {
    log_step "Checking NVIDIA Container Toolkit..."

    if ! detect_nvidia_gpu; then
        log_info "No NVIDIA GPU detected — skipping NVIDIA Container Toolkit"
        return 0
    fi

    # Skip if Docker not available
    if ! command -v docker &> /dev/null || ! docker info &> /dev/null; then
        log_info "Docker not available — skipping NVIDIA Container Toolkit"
        return 0
    fi

    # Check if already working
    if docker info 2>/dev/null | grep -qi 'nvidia'; then
        log_success "NVIDIA Container Toolkit already configured"

        # Verify GPU access works (use detected path)
        if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            log_success "GPU container access verified"
        else
            log_info "NVIDIA runtime present but GPU test skipped (image may not be cached)"
        fi
        return 0
    fi

    log_info "Installing NVIDIA Container Toolkit..."

    # Add NVIDIA GPG key
    if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
        if ! curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null; then
            log_error "Failed to add NVIDIA GPG key"
            log_warn "GPU container support will not be available"
            return 0
        fi
    else
        log_info "NVIDIA GPG keyring already exists — skipping download"
    fi

    # Add NVIDIA repository
    # Use distro-agnostic stable/deb URL first (works on all Ubuntu versions including 24.04)
    # Fallback to distribution-specific URL for older setups
    local repo_added=false

    # Primary: distro-agnostic URL (recommended by NVIDIA, works on all Ubuntu/Debian)
    local list_content
    if list_content=$(curl -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" 2>/dev/null) \
        && [ -n "$list_content" ]; then
        echo "$list_content" \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            > /etc/apt/sources.list.d/nvidia-container-toolkit.list
        repo_added=true
    fi

    # Fallback: distribution-specific URL
    if [ "$repo_added" = false ]; then
        log_info "Trying distribution-specific repository format..."
        local distribution
        distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")

        if list_content=$(curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" 2>/dev/null) \
            && [ -n "$list_content" ]; then
            echo "$list_content" \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                > /etc/apt/sources.list.d/nvidia-container-toolkit.list
            repo_added=true
        fi
    fi

    if [ "$repo_added" = false ]; then
        log_error "Failed to add NVIDIA container toolkit repository"
        log_warn "GPU container support will not be available"
        return 0
    fi

    # Install the toolkit
    apt-get update -qq 2>/dev/null
    if ! apt-get install -y nvidia-container-toolkit 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        log_error "Failed to install nvidia-container-toolkit"
        log_warn "GPU container support will not be available"
        return 0
    fi

    # Configure Docker to use NVIDIA runtime
    if ! nvidia-ctk runtime configure --runtime=docker 2>&1 | tee -a "$LOG_DIR/install.log" > /dev/null; then
        log_error "Failed to configure NVIDIA runtime for Docker"
        log_warn "GPU container support may not work"
        return 0
    fi

    # Restart Docker to pick up new runtime
    log_info "Restarting Docker to enable NVIDIA runtime..."
    systemctl restart docker 2>/dev/null || true
    sleep 3

    # Verify NVIDIA runtime is registered
    if docker info 2>/dev/null | grep -qi 'nvidia'; then
        log_success "NVIDIA Container Toolkit installed and configured"
    else
        log_warn "NVIDIA runtime not detected in Docker — check configuration"
        log_info "Manual fix: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        return 0
    fi

    # Pull base CUDA image in background for faster first deployment
    log_info "Pre-pulling NVIDIA CUDA base image (background)..."
    docker pull nvidia/cuda:12.0.0-base-ubuntu22.04 > /dev/null 2>&1 &

    log_success "GPU container support ready (Docker + NVIDIA Container Toolkit)"
    return 0
}

install_wireguard() {
    if [ "$SKIP_WIREGUARD" = true ]; then
        log_warn "Skipping WireGuard installation (--skip-wireguard)"
        return 0
    fi
    
    log_step "Installing WireGuard..."
    
    # Check if already installed
    if command -v wg &> /dev/null; then
        local wg_version=$(wg --version 2>&1 | head -n1)
        log_success "WireGuard already installed: $wg_version"
        return 0
    fi
    
    # Determine OS version for optimization
    local os_version=$(lsb_release -rs)
    local os_version_major=$(echo "$os_version" | cut -d'.' -f1)
    
    log_info "Detected Ubuntu $os_version"
    
    # Update package cache (CRITICAL - prevents 404 errors)
    log_info "Updating package cache..."
    if ! apt-get update -qq 2>&1 | grep -qi "err:"; then
        log_info "Package cache updated successfully"
    else
        log_warn "Package cache update had warnings"
    fi
    
    # Ubuntu 24.04+ has WireGuard built into kernel
    if [ "$os_version_major" -ge 24 ]; then
        log_info "Ubuntu 24.04+ detected - installing userspace tools only"
        
        if ! apt-get install -y wireguard-tools 2>&1 | tee -a "$LOG_DIR/install.log"; then
            log_error "Failed to install WireGuard tools"
            log_error "Check logs: $LOG_DIR/install.log"
            log_error "Try manually: sudo apt-get install -y wireguard-tools"
            return 1
        fi
    else
        # Ubuntu < 24.04 needs kernel headers and full package
        log_info "Ubuntu $os_version - installing full WireGuard with DKMS"
        
        # Install kernel headers first
        local kernel_version=$(uname -r)
        log_info "Installing kernel headers for $kernel_version..."
        
        if apt-get install -y linux-headers-${kernel_version} 2>&1 | tee -a "$LOG_DIR/install.log"; then
            log_info "Kernel headers installed"
        else
            log_warn "Kernel headers installation had issues (might still work)"
        fi
        
        # Install WireGuard
        if ! apt-get install -y wireguard wireguard-tools 2>&1 | tee -a "$LOG_DIR/install.log"; then
            log_error "Failed to install WireGuard"
            log_error "Check logs: $LOG_DIR/install.log"
            log_error "Try manually: sudo apt-get install -y wireguard wireguard-tools"
            return 1
        fi
    fi
    
    # Verify installation
    if ! command -v wg &> /dev/null; then
        log_error "WireGuard command not found after installation"
        log_error "Installation may have failed silently"
        return 1
    fi
    
    # Check kernel module
    if modinfo wireguard &> /dev/null 2>&1; then
        log_info "WireGuard kernel module available"
    elif [ "$os_version_major" -ge 24 ]; then
        log_info "WireGuard module built into kernel (Ubuntu 24.04+)"
    else
        log_warn "WireGuard kernel module not detected (might still work)"
    fi
    
    local wg_version=$(wg --version 2>&1 | head -n1)
    log_success "WireGuard installed: $wg_version"
    
    return 0
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
    
    # Source from the verified CLI bundle extracted by download_node_agent.
    local cli_source="${STAGE_DIR}/cli/cli-decloud-node"
    local cli_dest="/usr/local/bin/cli-decloud-node"
    
    if [ ! -f "$cli_source" ]; then
        log_error "CLI script not found at $cli_source"
        log_error "CLI bundle may be incomplete — re-run install with a valid --version"
        exit 1
    fi
    
    # Copy CLI to /usr/local/bin
    cp "$cli_source" "$cli_dest"
    chmod +x "$cli_dest"
    
    # Verify CLI works
    if cli-decloud-node version &> /dev/null; then
        local cli_version=$(cli-decloud-node version 2>/dev/null | grep -oP 'v\K[\d.]+')
        log_success "Authentication CLI installed"

        # Check which version (show feature)
        if cli-decloud-node version 2>/dev/null | grep -q "WalletConnect"; then
            log_info "CLI: WalletConnect Edition (v${cli_version})"
        else
            log_info "CLI: $(basename $cli_source) (v${cli_version})"
        fi
    else
        log_error "CLI installation failed - testing returned error"
        return 1
    fi
    
    return 0
}

# ============================================================
# DeCloud CLI Installation
# ============================================================

install_decloud_cli() {
    log_step "Installing DeCloud unified CLI..."
    
    # Source from the verified CLI bundle extracted by download_node_agent.
    local cli_source="${STAGE_DIR}/cli/decloud"
    local cli_dest="/usr/local/bin/decloud"
    
    # Check if CLI source exists
    if [ ! -f "$cli_source" ]; then
        log_warn "DeCloud CLI not found at $cli_source"
        log_info "This is optional - installation will continue"
        return 0
    fi
    
    # Copy CLI to system path
    cp "$cli_source" "$cli_dest"
    chmod +x "$cli_dest"
    
    # Verify CLI works
    if decloud --version &> /dev/null; then
        local cli_version=$(decloud --version 2>/dev/null | grep -oP 'v\K[\d.]+')
        log_success "DeCloud CLI installed (v${cli_version})"
    else
        # Version check might fail on first install, that's OK
        log_success "DeCloud CLI installed"
    fi
    
    # Copy supporting scripts if they exist (vm-cleanup.sh ships at bundle root)
    local vm_cleanup_source="${STAGE_DIR}/vm-cleanup.sh"
    local vm_cleanup_dest="/usr/local/bin/vm-cleanup.sh"
    
    if [ -f "$vm_cleanup_source" ]; then
        cp "$vm_cleanup_source" "$vm_cleanup_dest"
        chmod +x "$vm_cleanup_dest"
        log_info "→ VM cleanup script installed"
    else
        log_warn "vm-cleanup.sh not found at $vm_cleanup_source — skipping"
        log_info "  (decloud vm cleanup works without it — virsh logic is built in)"
    fi

    # Refresh install.sh and uninstall.sh at the canonical persistent location.
    # cli/decloud's find_install_script and cmd_remove look here when running
    # 'decloud update' or 'decloud uninstall'. Doing this on every install run
    # makes the loop self-healing: each install places its own scripts at the
    # well-known path so the next invocation always has copies that match the
    # version it was last installed from.
    install -d /usr/local/share/decloud

    local install_sh_source="${STAGE_DIR}/install.sh"
    local install_sh_dest="/usr/local/share/decloud/install.sh"
    if [ -f "$install_sh_source" ]; then
        install -m 755 "$install_sh_source" "$install_sh_dest"
        log_info "→ install.sh installed to $install_sh_dest (used by 'decloud update')"
    else
        log_warn "install.sh not found at $install_sh_source"
        log_warn "  'decloud update' may fail to locate the installer until next release"
    fi

    local uninstall_sh_source="${STAGE_DIR}/uninstall.sh"
    local uninstall_sh_dest="/usr/local/share/decloud/uninstall.sh"
    if [ -f "$uninstall_sh_source" ]; then
        install -m 755 "$uninstall_sh_source" "$uninstall_sh_dest"
        log_info "→ uninstall.sh installed to $uninstall_sh_dest (used by 'decloud uninstall')"
    else
        log_warn "uninstall.sh not found at $uninstall_sh_source"
        log_warn "  'decloud uninstall' may fail to locate the uninstaller until next release"
    fi
    
    log_success "✓ DeCloud unified CLI ready"
}

install_decloud_docs() {
    local doc_dir="/usr/local/share/doc/decloud"
    local source_dir="${STAGE_DIR}/cli/docs"
    
    if [ -d "$source_dir" ]; then
        mkdir -p "$doc_dir"
        
        # Copy documentation files
        for doc in README.md QUICKREF.md DESIGN.md; do
            if [ -f "$source_dir/$doc" ]; then
                cp "$source_dir/$doc" "$doc_dir/"
                log_info "→ Installed $doc"
            fi
        done
        
        log_success "Documentation installed to $doc_dir"
    fi
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
    # NOTE: do NOT pre-create $CONFIG_DIR. Under release-mode install,
    # CONFIG_DIR=/opt/decloud/publish is a SYMLINK created by
    # download_node_agent pointing at the active publish.<version> directory.
    # Pre-creating it here as a real directory blocks the atomic symlink
    # swap (mv -T refuses to replace a directory with a non-directory).
    # The symlink itself provides the path; create_configuration writes
    # appsettings.Production.json into it later in main().
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p /var/lib/decloud
    
    log_success "Directories created"
}

# ── Release-mode install ─────────────────────────────────────────────────
# As of v2.2.0, install.sh fetches signed release artifacts from GitHub
# Releases instead of cloning source and building on the node.
#
# Layout produced under $INSTALL_DIR after a successful install:
#
#   /opt/decloud/
#   ├── publish              -> publish.<version>     (active symlink)
#   ├── publish.<previous>/                            (kept for rollback)
#   ├── publish.<version>/                             (current release)
#   │   ├── DeCloud.NodeAgent.dll
#   │   ├── appsettings.Production.json
#   │   └── CloudInit/Templates/...
#   ├── stage/                                         (extracted CLI bundle)
#   ├── gpu-proxy/build/                               (extracted GPU shims)
#   ├── releases/                                      (manifests + tarball cache)
#   └── current-version                                (active version string)
#
# The systemd unit's WorkingDirectory and ExecStart point at /opt/decloud/publish
# (the symlink), so version swaps are atomic from systemd's perspective.

download_node_agent() {
    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_warn "Skipping Node Agent download (--skip-download)"
        return
    fi
    log_step "Downloading Node Agent release..."
    
    # ── PRE: save system VM UUIDs while NodeAgent is still up ────────
    # NodeAgent maps friendly names (blockstore-region-nodeid) to UUIDs via its API.
    # After the service stops this mapping is unavailable, so we save it to temp files now.
    rm -f /tmp/decloud-blockstore-vm-id /tmp/decloud-dht-vm-id
    if [ "$UPDATE_MODE" = true ] && command -v curl &>/dev/null; then
        if systemctl is-active --quiet decloud-node-agent 2>/dev/null; then
            local vms_json
            vms_json=$(curl -sf --max-time 5 "http://127.0.0.1:${AGENT_PORT}/api/vms" 2>/dev/null || echo "")
            if [ -n "$vms_json" ] && command -v jq &>/dev/null; then
                local bs_id dht_id
                bs_id=$(echo "$vms_json" \
                    | jq -r '.[] | select((.name // .spec.name // "") | test("^blockstore-")) | (.id // .vmId // "")' \
                    2>/dev/null | head -1 | tr -d '[:space:]' || true)
                dht_id=$(echo "$vms_json" \
                    | jq -r '.[] | select((.name // .spec.name // "") | test("^dht-")) | (.id // .vmId // "")' \
                    2>/dev/null | head -1 | tr -d '[:space:]' || true)
                [ -n "$bs_id" ]  && echo "$bs_id"  > /tmp/decloud-blockstore-vm-id
                [ -n "$dht_id" ] && echo "$dht_id" > /tmp/decloud-dht-vm-id
                log_info "Saved system VM IDs for post-swap cleanup"
            fi
        fi
    fi

    # ── 1. Resolve version, fetch + verify manifest ──────────────────
    local version asset_arch
    version=$(resolve_release_version) || return 1
    asset_arch=$(release_asset_arch)   || return 1
    fetch_and_verify_manifest "$version" || return 1

    # ── 2. Stop service before swapping the publish symlink ──────────
    if [ "$UPDATE_MODE" = true ]; then
        log_info "Stopping existing node agent service..."
        systemctl stop decloud-node-agent 2>/dev/null || true
        sleep 2
    fi

    # ── 3. Download + extract NodeAgent tarball ──────────────────────
    local nodeagent_name="decloud-node-agent-${version}-${asset_arch}.tar.gz"
    local nodeagent_tar="${RELEASE_CACHE_DIR}/${nodeagent_name}"
    download_artifact_verified "$version" "$nodeagent_name" "$nodeagent_tar" || return 1

    local target_dir="${INSTALL_DIR}/publish.${version}"
    rm -rf "$target_dir"
    mkdir -p "$target_dir"
    if ! tar -xzf "$nodeagent_tar" -C "$target_dir"; then
        log_error "Failed to extract NodeAgent tarball"
        rm -rf "$target_dir"
        return 1
    fi

    if [ ! -f "${target_dir}/DeCloud.NodeAgent.dll" ]; then
        log_error "Extracted publish dir is missing DeCloud.NodeAgent.dll"
        rm -rf "$target_dir"
        return 1
    fi

    # ── 4. Download + extract CLI bundle to staging ──────────────────
    local cli_name="decloud-cli-${version}.tar.gz"
    local cli_tar="${RELEASE_CACHE_DIR}/${cli_name}"
    download_artifact_verified "$version" "$cli_name" "$cli_tar" || return 1

    rm -rf "$STAGE_DIR"
    mkdir -p "$STAGE_DIR"
    if ! tar -xzf "$cli_tar" -C "$STAGE_DIR"; then
        log_error "Failed to extract CLI bundle"
        return 1
    fi

    # ── 5. Atomic publish symlink swap ───────────────────────────────
    # Prerequisite: /opt/decloud/publish must NOT exist as a real directory.
    # It may be:
    #   (a) absent (fresh install, expected case)
    #   (b) a symlink from a previous release-mode install (will be replaced)
    #   (c) a real directory from a legacy non-release-mode install or a
    #       broken prior install attempt — we clean those up here.
    # mv -T refuses to overwrite a directory with a non-directory, so case
    # (c) requires explicit removal.
    if [ -d "${INSTALL_DIR}/publish" ] && [ ! -L "${INSTALL_DIR}/publish" ]; then
        log_info "Removing pre-existing /opt/decloud/publish directory (legacy or recovered state)"
        rm -rf "${INSTALL_DIR}/publish"
    fi

    local previous=""
    if [ -L "${INSTALL_DIR}/publish" ]; then
        previous=$(readlink "${INSTALL_DIR}/publish" | xargs basename | sed 's/^publish\.//')
    fi
    local tmp_link="${INSTALL_DIR}/publish.new"
    ln -sfn "$target_dir" "$tmp_link"
    mv -Tf  "$tmp_link"   "${INSTALL_DIR}/publish"

    echo "$version" > "$CURRENT_VERSION_FILE"
    if [ -n "$previous" ] && [ "$previous" != "$version" ]; then
        echo "$previous" > "$PREVIOUS_VERSION_FILE"
    fi

    # ── 6. Prune older publish.* dirs, keep current + 1 prior ────────
    find "$INSTALL_DIR" -maxdepth 1 -type d -name 'publish.*' \
        ! -name "publish.${version}" \
        ! -name "publish.${previous:-NEVER_MATCH_SENTINEL}" \
        -exec rm -rf {} + 2>/dev/null || true

    # ── 7. One-time migration: clean up old git-clone layout ─────────
    # If the node was previously installed via the git-clone flow,
    # /opt/decloud/DeCloud.NodeAgent and /opt/decloud/Decloud.Shared exist
    # but are no longer used. Remove to avoid confusion.
    if [ -d "${INSTALL_DIR}/DeCloud.NodeAgent/.git" ]; then
        log_info "Removing obsolete git clone at ${INSTALL_DIR}/DeCloud.NodeAgent"
        rm -rf "${INSTALL_DIR}/DeCloud.NodeAgent"
    fi
    if [ -d "${INSTALL_DIR}/Decloud.Shared/.git" ]; then
        log_info "Removing obsolete DeCloud.Shared clone"
        rm -rf "${INSTALL_DIR}/Decloud.Shared"
    fi

    log_success "Node Agent ${version} installed (active: ${INSTALL_DIR}/publish -> publish.${version})"
}

# NOTE: download_shared_library() has been removed.
# DeCloud.Shared is now consumed at CI build time as a transitive ProjectRef
# in the DeCloud.NodeAgent solution. The published NodeAgent tarball already
# contains the compiled Shared DLL — no separate download is needed on nodes.

build_node_agent() {
    log_step "Verifying installed Node Agent..."

    # Build happened in CI; the atomic symlink flip happened in download_node_agent.
    # This function exists in main()'s call order to preserve the existing flow,
    # but no compilation is performed on the node.

    if [ ! -L "${INSTALL_DIR}/publish" ]; then
        log_error "${INSTALL_DIR}/publish is not a symlink — release install is incomplete"
        log_error "  Did download_node_agent run successfully?"
        return 1
    fi

    if [ ! -f "${INSTALL_DIR}/publish/DeCloud.NodeAgent.dll" ]; then
        log_error "DeCloud.NodeAgent.dll missing under ${INSTALL_DIR}/publish"
        log_error "  Symlink target: $(readlink "${INSTALL_DIR}/publish" 2>/dev/null || echo unknown)"
        return 1
    fi

    # CloudInit templates ship inside the publish output.
    if [ -d "${INSTALL_DIR}/publish/CloudInit/Templates" ]; then
        local template_count
        template_count=$(find "${INSTALL_DIR}/publish/CloudInit/Templates" -name '*.yaml' | wc -l)
        if [ "$template_count" -gt 0 ]; then
            log_success "CloudInit templates present (${template_count})"
        else
            log_warn "CloudInit/Templates directory present but empty"
        fi
    else
        log_warn "CloudInit/Templates directory not present in publish output"
    fi

    local file_count dir_size active_version
    file_count=$(find -L "${INSTALL_DIR}/publish" -type f | wc -l)
    dir_size=$(du -shL "${INSTALL_DIR}/publish" 2>/dev/null | cut -f1)
    active_version=$(cat "$CURRENT_VERSION_FILE" 2>/dev/null || echo unknown)
    log_info "Active version: ${active_version} (${file_count} files, ${dir_size})"

    log_success "Node Agent verified"
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
Environment=PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
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

    # Install netfilter-persistent if not present
    if ! command -v netfilter-persistent &> /dev/null; then
        echo "→ Installing netfilter-persistent..."
        apt-get install -y netfilter-persistent > /dev/null 2>&1 || {
            echo "⚠ Warning: Could not install netfilter-persistent (optional)"
        }
    fi
    
    # Copy relay NAT manager script from file
    echo "→ Installing relay NAT manager..."

    # The relay NAT manager script ships in the verified CLI bundle.
    local file_source="${STAGE_DIR}/decloud-relay-nat"
    local file_dest="/usr/local/bin/decloud-relay-nat"
    
    if [ ! -f "$file_source" ]; then
        log_error "Relay NAT manager script not found at $file_source"
        log_error "CLI bundle may be incomplete"
        exit 1
    fi
    
    # Copy relay NAT manager script to /usr/local/bin
    cp "$file_source" "$file_dest"
    chmod +x "$file_dest"
    
    mkdir -p /var/log
    
    # NOTE: Do NOT pre-clean rules here. The 'decloud-relay-nat add' command
    # atomically adds new rules before removing stale ones, ensuring zero
    # downtime for CGNAT system VMs during node agent updates.
    # Explicit clean would flush FORWARD rules and kill active WG sessions.
    echo "✓ Relay NAT support installed (rules preserved during update)"
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
# WSL2 Windows Installer Helpers
# ============================================================

# Writes DeCloud-Node-Setup.bat to the Windows Desktop from embedded content.
# No download or external URL needed — always in sync with this installer.
# Echoes the Windows backslash path on success, empty string on failure.
stage_windows_installer() {
    local win_desktop wsl_desktop bat_wsl_path bat_win_path

    # Embedded installer — base64-encoded DeCloud-Node-Setup.bat
    local BAT_B64="QGVjaG8gb2ZmCnNldGxvY2FsCgo6OiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KOjogRGVDbG91ZCBOb2RlIEFnZW50IC0gV2luZG93cyBXU0wgU2V0dXAKOjogRG91YmxlLWNsaWNrIHRvIGluc3RhbGwuIFJ1biBhcyBBZG1pbmlzdHJhdG9yLgo6OiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KCjo6IFNlbGYtZWxldmF0ZSBpZiBub3QgYWxyZWFkeSBhZG1pbgpuZXQgc2Vzc2lvbiA+bnVsIDI+JjEKaWYgJWVycm9ybGV2ZWwlIGVxdSAwIGdvdG8gOkFMUkVBRFlfQURNSU4KCmVjaG8uCmVjaG8gIFtEZUNsb3VkXSBSZXF1ZXN0aW5nIGFkbWluaXN0cmF0b3IgcHJpdmlsZWdlcy4uLgplY2hvICBBIFVBQyBwcm9tcHQgd2lsbCBhcHBlYXIgLSBjbGljayBZZXMgdG8gY29udGludWUuCmVjaG8uCnBvd2Vyc2hlbGwgLU5vUHJvZmlsZSAtQ29tbWFuZCAiU3RhcnQtUHJvY2VzcyBjbWQgLUFyZ3VtZW50TGlzdCAnL2MgIiIlfmYwIiInIC1WZXJiIFJ1bkFzIC1XYWl0IgpleGl0IC9iICVlcnJvcmxldmVsJQoKOkFMUkVBRFlfQURNSU4KCjo6IEV4dHJhY3QgYW5kIHJ1biB0aGUgZW1iZWRkZWQgUG93ZXJTaGVsbCBpbnN0YWxsZXIuCjo6IFRoZSBwYXlsb2FkIGlzIGEgYmFzZTY0LWVuY29kZWQgUFMgc2NyaXB0IHN0b3JlZCBhZnRlciB0aGUgX19QQVlMT0FEX18gbWFya2VyLgo6OiBXZSByZWFkIGl0IGZyb20gdGhpcyBmaWxlLCBkZWNvZGUgaXQsIHdyaXRlIHRvIGEgdGVtcCAucHMxLCBhbmQgZXhlY3V0ZSBpdC4KcG93ZXJzaGVsbCAtTm9Qcm9maWxlIC1FeGVjdXRpb25Qb2xpY3kgQnlwYXNzIC1Db21tYW5kICIkZj1HZXQtQ29udGVudCAtTGl0ZXJhbFBhdGggJyV+ZjAnIC1FbmNvZGluZyBVVEY4OyAkbWs9JGZ8U2VsZWN0LVN0cmluZyAnXjo6X19QQVlMT0FEX18nfFNlbGVjdC1PYmplY3QgLUZpcnN0IDE7IGlmKC1ub3QgJG1rKXtXcml0ZS1Ib3N0ICdbRVJSXSBQYXlsb2FkIG5vdCBmb3VuZC4nIC1Gb3JlZ3JvdW5kQ29sb3IgUmVkO2V4aXQgMX07ICRiNjQ9KCRmWyRtay5MaW5lTnVtYmVyLi4oJGYuTGVuZ3RoLTEpXSAtam9pbiAnJykuVHJpbSgpOyAkYnl0ZXM9W0NvbnZlcnRdOjpGcm9tQmFzZTY0U3RyaW5nKCRiNjQpOyAkc3JjPVtTeXN0ZW0uVGV4dC5FbmNvZGluZ106OlVURjguR2V0U3RyaW5nKCRieXRlcyk7ICR0bXA9W0lPLlBhdGhdOjpHZXRUZW1wRmlsZU5hbWUoKSsnLnBzMSc7IFNldC1Db250ZW50IC1QYXRoICR0bXAgLVZhbHVlICRzcmMgLUVuY29kaW5nIFVURjg7IHRyeXsmIHBvd2Vyc2hlbGwgLU5vUHJvZmlsZSAtRXhlY3V0aW9uUG9saWN5IEJ5cGFzcyAtRmlsZSAkdG1wfWZpbmFsbHl7UmVtb3ZlLUl0ZW0gJHRtcCAtRm9yY2UgLUVycm9yQWN0aW9uIFNpbGVudGx5Q29udGludWV9IgpleGl0IC9iICVlcnJvcmxldmVsJQoKOjpfX1BBWUxPQURfXwpJMUpsY1hWcGNtVnpJQzFXWlhKemFXOXVJRFV1TVFwVFpYUXRVM1J5YVdOMFRXOWtaU0F0Vm1WeWMybHZiaUJNWVhSbGMzUUtKRVZ5Y205eVFXTjBhVzl1VUhKbFptVnlaVzVqWlNBOUlDZFRkRzl3SndvS0l5RGlsSURpbElBZ1EyOXVabWxuZFhKaGRHbHZiaURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElBS0pFUkpVMVJTVHlBZ0lDQWdJQ0E5SUNkVlluVnVkSFVuQ2lSVFJWSldTVU5GWDA1QlRVVWdQU0FuWkdWamJHOTFaQzF1YjJSbExXRm5aVzUwSndva1NVNVRWRUZNVEY5RVNWSWdJRDBnSWlSbGJuWTZVSEp2WjNKaGJVUmhkR0ZjUkdWRGJHOTFaQ0lLSkZSQlUwdGZUa0ZOUlNBZ0lDQTlJQ2RFWlVOc2IzVmtMVmR6YkZkaGRHTm9aRzluSndva1RFOUhYMFJKVWlBZ0lDQWdJRDBnSWlSbGJuWTZVSEp2WjNKaGJVUmhkR0ZjUkdWRGJHOTFaRnhNYjJkeklnb0tJeURpbElEaWxJQWdRMjl1YzI5c1pTQm9aV3h3WlhKeklPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ0FwbWRXNWpkR2x2YmlCWGNtbDBaUzFDWVc1dVpYSWdld29nSUNBZ1YzSnBkR1V0U0c5emRDQWlJZ29nSUNBZ1YzSnBkR1V0U0c5emRDQWlJQ0FyTFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTc2lJQzFHYjNKbFozSnZkVzVrUTI5c2IzSWdUV0ZuWlc1MFlRb2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlDQjhJQ0FnSUNBZ0lFUmxRMnh2ZFdRZ1RtOWtaU0JCWjJWdWRDQXRMU0JYYVc1a2IzZHpJRmRUVENCVFpYUjFjQ0FnSUNBZ0lDQWdJQ0FnSUh3aUlDMUdiM0psWjNKdmRXNWtRMjlzYjNJZ1RXRm5aVzUwWVFvZ0lDQWdWM0pwZEdVdFNHOXpkQ0FpSUNBckxTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMU3NpSUMxR2IzSmxaM0p2ZFc1a1EyOXNiM0lnVFdGblpXNTBZUW9nSUNBZ1YzSnBkR1V0U0c5emRDQWlJZ3A5Q21aMWJtTjBhVzl1SUZkeWFYUmxMVk4wWlhBZ2V5QndZWEpoYlNoYmMzUnlhVzVuWFNSdEtTQlhjbWwwWlMxSWIzTjBJQ0lnSUZzK1BsMGdKRzBpSUMxR2IzSmxaM0p2ZFc1a1EyOXNiM0lnUTNsaGJpQWdJSDBLWm5WdVkzUnBiMjRnVjNKcGRHVXRUMHNnSUNCN0lIQmhjbUZ0S0Z0emRISnBibWRkSkcwcElGZHlhWFJsTFVodmMzUWdJaUFnVzA5TFhTQWtiU0lnTFVadmNtVm5jbTkxYm1SRGIyeHZjaUJIY21WbGJpQWdmUXBtZFc1amRHbHZiaUJYY21sMFpTMVhZWEp1SUhzZ2NHRnlZVzBvVzNOMGNtbHVaMTBrYlNrZ1YzSnBkR1V0U0c5emRDQWlJQ0JiSVNGZElDUnRJaUF0Um05eVpXZHliM1Z1WkVOdmJHOXlJRmxsYkd4dmR5QjlDbVoxYm1OMGFXOXVJRmR5YVhSbExVWmhhV3dnZXlCd1lYSmhiU2hiYzNSeWFXNW5YU1J0S1NCWGNtbDBaUzFJYjNOMElDSWdJRnNoSVYwZ0pHMGlJQzFHYjNKbFozSnZkVzVrUTI5c2IzSWdVbVZrSUNBZ0lIMEtablZ1WTNScGIyNGdWM0pwZEdVdFNXNW1ieUI3SUhCaGNtRnRLRnR6ZEhKcGJtZGRKRzBwSUZkeWFYUmxMVWh2YzNRZ0lpQWdJQ0FnSUNBa2JTSWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb0tablZ1WTNScGIyNGdRMjl1Wm1seWJTMUJaRzFwYmlCN0NpQWdJQ0FrYVdRZ1BTQmJVMlZqZFhKcGRIa3VVSEpwYm1OcGNHRnNMbGRwYm1SdmQzTkpaR1Z1ZEdsMGVWMDZPa2RsZEVOMWNuSmxiblFvS1FvZ0lDQWdKSEFnSUQwZ1cxTmxZM1Z5YVhSNUxsQnlhVzVqYVhCaGJDNVhhVzVrYjNkelVISnBibU5wY0dGc1hTUnBaQW9nSUNBZ2FXWWdLQzF1YjNRZ0pIQXVTWE5KYmxKdmJHVW9XMU5sWTNWeWFYUjVMbEJ5YVc1amFYQmhiQzVYYVc1a2IzZHpRblZwYkhSSmJsSnZiR1ZkT2pwQlpHMXBibWx6ZEhKaGRHOXlLU2tnZXdvZ0lDQWdJQ0FnSUZkeWFYUmxMVVpoYVd3Z0lrNXZkQ0J5ZFc1dWFXNW5JR0Z6SUVGa2JXbHVhWE4wY21GMGIzSXVJZ29nSUNBZ0lDQWdJRmR5YVhSbExVbHVabThnSWxKcFoyaDBMV05zYVdOcklIUm9aU0JtYVd4bElHRnVaQ0JqYUc5dmMyVWdKMUoxYmlCaGN5QmhaRzFwYm1semRISmhkRzl5Snk0aUNpQWdJQ0FnSUNBZ1VtVmhaQzFJYjNOMElDQWlVSEpsYzNNZ1JXNTBaWElnZEc4Z1pYaHBkQ0lLSUNBZ0lDQWdJQ0JsZUdsMElERUtJQ0FnSUgwS2ZRb0tJeURpbElEaWxJQWdVSEpsY21WeGRXbHphWFJsSUdOb1pXTnJjeURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElBS1puVnVZM1JwYjI0Z1ZHVnpkQzFRY21WeVpYRjFhWE5wZEdWeklIc0tJQ0FnSUZkeWFYUmxMVk4wWlhBZ0lrTm9aV05yYVc1bklIQnlaWEpsY1hWcGMybDBaWE11TGk0aUNnb2dJQ0FnYVdZZ0tDMXViM1FnS0VkbGRDMURiMjF0WVc1a0lDZDNjMnd1WlhobEp5QXRSWEp5YjNKQlkzUnBiMjRnVTJsc1pXNTBiSGxEYjI1MGFXNTFaU2twSUhzS0lDQWdJQ0FnSUNCWGNtbDBaUzFHWVdsc0lDSlhVMHdnYVhNZ2JtOTBJR2x1YzNSaGJHeGxaQzRpQ2lBZ0lDQWdJQ0FnVjNKcGRHVXRTVzVtYnlBaVVuVnVJR2x1SUdGdUlHVnNaWFpoZEdWa0lGQnZkMlZ5VTJobGJHdzZJQ0IzYzJ3Z0xTMXBibk4wWVd4c0lDMWtJRlZpZFc1MGRTSUtJQ0FnSUNBZ0lDQlNaV0ZrTFVodmMzUWdJQ0pRY21WemN5QkZiblJsY2lCMGJ5QmxlR2wwSWpzZ1pYaHBkQ0F4Q2lBZ0lDQjlDaUFnSUNCWGNtbDBaUzFQU3lBaWQzTnNMbVY0WlNCbWIzVnVaQzRpQ2dvZ0lDQWdKSFpsY2t4cGJtVnpJRDBnSmlCM2Myd3VaWGhsSUMwdGRtVnljMmx2YmlBeVBpWXhDaUFnSUNBa2RtVnlUR2x1WlNBZ1BTQWtkbVZ5VEdsdVpYTWdmQ0JUWld4bFkzUXRVM1J5YVc1bklDZFhVMHdnZG1WeWMybHZiaWNnZkNCVFpXeGxZM1F0VDJKcVpXTjBJQzFHYVhKemRDQXhDaUFnSUNCcFppQW9KSFpsY2t4cGJtVXBJSHNnVjNKcGRHVXRUMHNnSWlSMlpYSk1hVzVsSWlCOUNnb2dJQ0FnSkhKaGR5QWdQU0FtSUhkemJDNWxlR1VnTFd3Z0xYWWdNajRtTVFvZ0lDQWdKSFJsZUhRZ1BTQmJVM2x6ZEdWdExsUmxlSFF1Ulc1amIyUnBibWRkT2pwQlUwTkpTUzVIWlhSVGRISnBibWNvQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JiVTNsemRHVnRMbFJsZUhRdVJXNWpiMlJwYm1kZE9qcFZibWxqYjJSbExrZGxkRUo1ZEdWektDUnlZWGNnTFdwdmFXNGdJbUJ1SWlrS0lDQWdJQ0FnSUNBZ0lDQWdLUzVTWlhCc1lXTmxLQ0pnTUNJc0lDY25LUW9LSUNBZ0lHbG1JQ2drZEdWNGRDQXRibTkwYldGMFkyZ2dXM0psWjJWNFhUbzZSWE5qWVhCbEtDUkVTVk5VVWs4cEtTQjdDaUFnSUNBZ0lDQWdWM0pwZEdVdFJtRnBiQ0FpVjFOTUlHUnBjM1J5YnlBbkpFUkpVMVJTVHljZ2JtOTBJR1p2ZFc1a0xpSUtJQ0FnSUNBZ0lDQlhjbWwwWlMxSmJtWnZJQ0pKYm5OMFlXeHNJSGRwZEdnNklDQjNjMndnTFMxcGJuTjBZV3hzSUMxa0lGVmlkVzUwZFNJS0lDQWdJQ0FnSUNCU1pXRmtMVWh2YzNRZ0lDSlFjbVZ6Y3lCRmJuUmxjaUIwYnlCbGVHbDBJanNnWlhocGRDQXhDaUFnSUNCOUNpQWdJQ0JYY21sMFpTMVBTeUFpVjFOTUlHUnBjM1J5YnlBbkpFUkpVMVJTVHljZ1ptOTFibVF1SWdwOUNnb2pJT0tVZ09LVWdDQlhZWFJqYUdSdlp5QnpZM0pwY0hRZ0tHVnRZbVZrWkdWa0tTRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJQUtablZ1WTNScGIyNGdSMlYwTFZkaGRHTm9aRzluVTJOeWFYQjBJSHNLSUNBZ0lDTWdUbTkwWlRvZ2MybHVaMnhsTFhGMWIzUmxjeUJwYm5OcFpHVWdkR2hwY3lCb1pYSmxMWE4wY21sdVp5QmhjbVVnYzJGbVpUc2dibThnYVc1MFpYSndiMnhoZEdsdmJpQnZZMk4xY25NdUNpQWdJQ0J5WlhSMWNtNGdRQ2NLSTFKbGNYVnBjbVZ6SUMxV1pYSnphVzl1SURVdU1RcFRaWFF0VTNSeWFXTjBUVzlrWlNBdFZtVnljMmx2YmlCTVlYUmxjM1FLSkVWeWNtOXlRV04wYVc5dVVISmxabVZ5Wlc1alpTQTlJQ2RUZEc5d0p3b0tKRU5QVGtaSlJ5QTlJRUI3Q2lBZ0lDQlhjMnhFYVhOMGNtOGdJQ0FnSUNBZ1BTQW5WV0oxYm5SMUp3b2dJQ0FnVTJWeWRtbGpaVTVoYldVZ0lDQWdJRDBnSjJSbFkyeHZkV1F0Ym05a1pTMWhaMlZ1ZENjS0lDQWdJRXh2WjBScGNpQWdJQ0FnSUNBZ0lDQTlJQ0lrWlc1Mk9sQnliMmR5WVcxRVlYUmhYRVJsUTJ4dmRXUmNURzluY3lJS0lDQWdJRXh2WjBacGJHVWdJQ0FnSUNBZ0lDQTlJQ0lrWlc1Mk9sQnliMmR5WVcxRVlYUmhYRVJsUTJ4dmRXUmNURzluYzF4M2Myd3RkMkYwWTJoa2IyY3ViRzluSWdvZ0lDQWdURzlqYTBacGJHVWdJQ0FnSUNBZ0lEMGdJaVJsYm5ZNlVISnZaM0poYlVSaGRHRmNSR1ZEYkc5MVpGeDNjMnd0ZDJGMFkyaGtiMmN1Ykc5amF5SUtJQ0FnSUUxaGVFeHZaMU5wZW1WQ2VYUmxjeUE5SURFd1RVSUtJQ0FnSUVobFlXeDBhRU5vWldOclUyVmpJQ0E5SURZd0NpQWdJQ0JYYzJ4VGRHRnlkSFZ3VTJWaklDQWdQU0F5TUFvZ0lDQWdUV0Y0VW1WemRHRnlkSE1nSUNBZ0lEMGdOUW9nSUNBZ1FtRmphMjltWmxObFl5QWdJQ0FnSUQwZ01USXdDbjBLQ21aMWJtTjBhVzl1SUVsdWFYUnBZV3hwZW1VdFRHOW5aMmx1WnlCN0NpQWdJQ0JwWmlBb0xXNXZkQ0FvVkdWemRDMVFZWFJvSUNSRFQwNUdTVWN1VEc5blJHbHlLU2tnZXdvZ0lDQWdJQ0FnSUU1bGR5MUpkR1Z0SUMxSmRHVnRWSGx3WlNCRWFYSmxZM1J2Y25rZ0xWQmhkR2dnSkVOUFRrWkpSeTVNYjJkRWFYSWdMVVp2Y21ObElId2dUM1YwTFU1MWJHd0tJQ0FnSUgwS0lDQWdJR2xtSUNoVVpYTjBMVkJoZEdnZ0pFTlBUa1pKUnk1TWIyZEdhV3hsS1NCN0NpQWdJQ0FnSUNBZ0pITnBlbVVnUFNBb1IyVjBMVWwwWlcwZ0pFTlBUa1pKUnk1TWIyZEdhV3hsS1M1TVpXNW5kR2dLSUNBZ0lDQWdJQ0JwWmlBb0pITnBlbVVnTFdkMElDUkRUMDVHU1VjdVRXRjRURzluVTJsNlpVSjVkR1Z6S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ1JoY21Ob2FYWmxJRDBnSWlRb0pFTlBUa1pKUnk1TWIyZEdhV3hsS1M0eElnb2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1ZHVnpkQzFRWVhSb0lDUmhjbU5vYVhabEtTQjdJRkpsYlc5MlpTMUpkR1Z0SUNSaGNtTm9hWFpsSUMxR2IzSmpaU0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lGSmxibUZ0WlMxSmRHVnRJQzFRWVhSb0lDUkRUMDVHU1VjdVRHOW5SbWxzWlNBdFRtVjNUbUZ0WlNBa1lYSmphR2wyWlNBdFJtOXlZMlVLSUNBZ0lDQWdJQ0I5Q2lBZ0lDQjlDbjBLQ21aMWJtTjBhVzl1SUZkeWFYUmxMVXh2WnlCN0NpQWdJQ0J3WVhKaGJTaGJjM1J5YVc1blhTUk1aWFpsYkN3Z1czTjBjbWx1WjEwa1RXVnpjMkZuWlNrS0lDQWdJQ1IwY3lBZ0lEMGdLRWRsZEMxRVlYUmxLUzVVYjFOMGNtbHVaeWduZVhsNWVTMU5UUzFrWkNCSVNEcHRiVHB6Y3ljcENpQWdJQ0FrYkdsdVpTQTlJQ0piSkhSelhTQmJKRXhsZG1Wc1hTQWtUV1Z6YzJGblpTSUtJQ0FnSUhSeWVTQjdJRUZrWkMxRGIyNTBaVzUwSUMxUVlYUm9JQ1JEVDA1R1NVY3VURzluUm1sc1pTQXRWbUZzZFdVZ0pHeHBibVVnTFVWdVkyOWthVzVuSUZWVVJqZ2dmU0JqWVhSamFDQjdmUW9nSUNBZ1YzSnBkR1V0U0c5emRDQWtiR2x1WlFwOUNncG1kVzVqZEdsdmJpQlhjbWwwWlMxSmJtWnZJSHNnY0dGeVlXMG9XM04wY21sdVoxMGtiU2tnVjNKcGRHVXRURzluSUNkSlRrWlBJQ2NnSkcwZ2ZRcG1kVzVqZEdsdmJpQlhjbWwwWlMxWFlYSnVJSHNnY0dGeVlXMG9XM04wY21sdVoxMGtiU2tnVjNKcGRHVXRURzluSUNkWFFWSk9JQ2NnSkcwZ2ZRcG1kVzVqZEdsdmJpQlhjbWwwWlMxRmNuSWdJSHNnY0dGeVlXMG9XM04wY21sdVoxMGtiU2tnVjNKcGRHVXRURzluSUNkRlVsSlBVaWNnSkcwZ2ZRb0tablZ1WTNScGIyNGdWR1Z6ZEMxQlkzRjFhWEpsVEc5amF5QjdDaUFnSUNBa2JHOWphMFJwY2lBOUlGTndiR2wwTFZCaGRHZ2dKRU5QVGtaSlJ5NU1iMk5yUm1sc1pRb2dJQ0FnYVdZZ0tDMXViM1FnS0ZSbGMzUXRVR0YwYUNBa2JHOWphMFJwY2lrcElIc0tJQ0FnSUNBZ0lDQk9aWGN0U1hSbGJTQXRTWFJsYlZSNWNHVWdSR2x5WldOMGIzSjVJQzFRWVhSb0lDUnNiMk5yUkdseUlDMUdiM0pqWlNCOElFOTFkQzFPZFd4c0NpQWdJQ0I5Q2lBZ0lDQnBaaUFvVkdWemRDMVFZWFJvSUNSRFQwNUdTVWN1VEc5amEwWnBiR1VwSUhzS0lDQWdJQ0FnSUNBa1pYaHBjM1JwYm1jZ1BTQkhaWFF0UTI5dWRHVnVkQ0FrUTA5T1JrbEhMa3h2WTJ0R2FXeGxJQzFGY25KdmNrRmpkR2x2YmlCVGFXeGxiblJzZVVOdmJuUnBiblZsQ2lBZ0lDQWdJQ0FnYVdZZ0tDUmxlR2x6ZEdsdVp5QXRiV0YwWTJnZ0oxNWNaQ3NrSnlrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FrWlhocGMzUnBibWRRYVdRZ1BTQmJhVzUwWFNSbGVHbHpkR2x1WndvZ0lDQWdJQ0FnSUNBZ0lDQWtjSEp2WXlBOUlFZGxkQzFRY205alpYTnpJQzFKWkNBa1pYaHBjM1JwYm1kUWFXUWdMVVZ5Y205eVFXTjBhVzl1SUZOcGJHVnVkR3g1UTI5dWRHbHVkV1VLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLQ1J3Y205aklDMWhibVFnSkhCeWIyTXVUbUZ0WlNBdGJXRjBZMmdnSjNCdmQyVnljMmhsYkd3bktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQlhjbWwwWlMxSmJtWnZJQ0pCYm05MGFHVnlJSGRoZEdOb1pHOW5JR2x6SUhKMWJtNXBibWNnS0ZCSlJDQWtaWGhwYzNScGJtZFFhV1FwTGlCRmVHbDBhVzVuTGlJS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUhKbGRIVnliaUFrWm1Gc2MyVUtJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNCWGNtbDBaUzFKYm1adklDSlNaVzF2ZG1sdVp5QnpkR0ZzWlNCc2IyTnJJQ2hRU1VRZ0pHVjRhWE4wYVc1bktTSUtJQ0FnSUgwS0lDQWdJRk5sZEMxRGIyNTBaVzUwSUMxUVlYUm9JQ1JEVDA1R1NVY3VURzlqYTBacGJHVWdMVlpoYkhWbElDUlFTVVFnTFVWdVkyOWthVzVuSUZWVVJqZ0tJQ0FnSUhKbGRIVnliaUFrZEhKMVpRcDlDZ3BtZFc1amRHbHZiaUJTWlcxdmRtVXRURzlqYXlCN0NpQWdJQ0JTWlcxdmRtVXRTWFJsYlNBa1EwOU9Sa2xITGt4dlkydEdhV3hsSUMxR2IzSmpaU0F0UlhKeWIzSkJZM1JwYjI0Z1UybHNaVzUwYkhsRGIyNTBhVzUxWlFwOUNncG1kVzVqZEdsdmJpQlVaWE4wTFZkemJFRjJZV2xzWVdKc1pTQjdDaUFnSUNCMGNua2dleUFrYm5Wc2JDQTlJQ1lnZDNOc0xtVjRaU0F0TFhOMFlYUjFjeUF5UGlZeE95QnlaWFIxY200Z0pFeEJVMVJGV0VsVVEwOUVSU0F0WlhFZ01DQjlDaUFnSUNCallYUmphQ0I3SUhKbGRIVnliaUFrWm1Gc2MyVWdmUXA5Q2dwbWRXNWpkR2x2YmlCSFpYUXRWM05zUkdsemRISnZVM1JoZEdVZ2V3b2dJQ0FnY0dGeVlXMG9XM04wY21sdVoxMGtSR2x6ZEhKdktRb2dJQ0FnZEhKNUlIc0tJQ0FnSUNBZ0lDQWtjbUYzSUNBOUlDWWdkM05zTG1WNFpTQXRiQ0F0ZGlBeVBpWXhDaUFnSUNBZ0lDQWdKSFJsZUhRZ1BTQmJVM2x6ZEdWdExsUmxlSFF1Ulc1amIyUnBibWRkT2pwQlUwTkpTUzVIWlhSVGRISnBibWNvQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1cxTjVjM1JsYlM1VVpYaDBMa1Z1WTI5a2FXNW5YVG82Vlc1cFkyOWtaUzVIWlhSQ2VYUmxjeWdrY21GM0lDMXFiMmx1SUNKZ2JpSXBDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQXBMbEpsY0d4aFkyVW9JbUF3SWl3Z0p5Y3BDaUFnSUNBZ0lDQWdhV1lnS0NSMFpYaDBJQzF0WVhSamFDQW9XM0psWjJWNFhUbzZSWE5qWVhCbEtDUkVhWE4wY204cElDc2dKMXh6S3loY2R5c3BKeWtwSUhzZ2NtVjBkWEp1SUNSTllYUmphR1Z6V3pGZElIMEtJQ0FnSUNBZ0lDQnlaWFIxY200Z0owNXZkRVp2ZFc1a0p3b2dJQ0FnZlNCallYUmphQ0I3SUhKbGRIVnliaUFuVlc1cmJtOTNiaWNnZlFwOUNncG1kVzVqZEdsdmJpQlRkR0Z5ZEMxWGMyeEVhWE4wY204Z2V3b2dJQ0FnY0dGeVlXMG9XM04wY21sdVoxMGtSR2x6ZEhKdktRb2dJQ0FnVjNKcGRHVXRTVzVtYnlBaVUzUmhjblJwYm1jZ1YxTk1JR1JwYzNSeWJ6b2dKRVJwYzNSeWJ5SUtJQ0FnSUNSdWRXeHNJRDBnSmlCM2Myd3VaWGhsSUMxa0lDUkVhWE4wY204Z0xTMWxlR1ZqSUM5aWFXNHZkSEoxWlNBeVBpWXhDbjBLQ21aMWJtTjBhVzl1SUVsdWRtOXJaUzFYYzJ4RGIyMXRZVzVrSUhzS0lDQWdJSEJoY21GdEtGdHpkSEpwYm1kZEpFUnBjM1J5Ynl3Z1czTjBjbWx1WjEwa1EyOXRiV0Z1WkNrS0lDQWdJQ1J2ZFhSd2RYUWdQU0FtSUhkemJDNWxlR1VnTFdRZ0pFUnBjM1J5YnlBdExXVjRaV01nTDJKcGJpOWlZWE5vSUMxaklDUkRiMjF0WVc1a0lESStKakVLSUNBZ0lISmxkSFZ5YmlCQWV5QkZlR2wwUTI5a1pTQTlJQ1JNUVZOVVJWaEpWRU5QUkVVN0lFOTFkSEIxZENBOUlDZ2tiM1YwY0hWMElDMXFiMmx1SUNKZ2JpSXBJSDBLZlFvS1puVnVZM1JwYjI0Z1ZHVnpkQzFUZVhOMFpXMWtSVzVoWW14bFpDQjdDaUFnSUNCd1lYSmhiU2hiYzNSeWFXNW5YU1JFYVhOMGNtOHBDaUFnSUNBa2NpQTlJRWx1ZG05clpTMVhjMnhEYjIxdFlXNWtJQ1JFYVhOMGNtOGdJbU5oZENBdmNISnZZeTh4TDJOdmJXMGdNajR2WkdWMkwyNTFiR3dpQ2lBZ0lDQnlaWFIxY200Z0tDUnlMazkxZEhCMWRDNVVjbWx0S0NrZ0xXVnhJQ2R6ZVhOMFpXMWtKeWtLZlFvS1puVnVZM1JwYjI0Z1JXNWhZbXhsTFZONWMzUmxiV1JKYmxkemJDQjdDaUFnSUNCd1lYSmhiU2hiYzNSeWFXNW5YU1JFYVhOMGNtOHBDaUFnSUNBa1kyaGxZMnNnUFNCSmJuWnZhMlV0VjNOc1EyOXRiV0Z1WkNBa1JHbHpkSEp2SUNKbmNtVndJQzF4SUNkemVYTjBaVzFrUFhSeWRXVW5JQzlsZEdNdmQzTnNMbU52Ym1ZZ01qNHZaR1YyTDI1MWJHd2dKaVlnWldOb2J5QjVaWE1nZkh3Z1pXTm9ieUJ1YnlJS0lDQWdJR2xtSUNna1kyaGxZMnN1VDNWMGNIVjBMbFJ5YVcwb0tTQXRaWEVnSjNsbGN5Y3BJSHNnY21WMGRYSnVJQ1IwY25WbElIMEtDaUFnSUNCWGNtbDBaUzFKYm1adklDSkZibUZpYkdsdVp5QnplWE4wWlcxa0lHbHVJRmRUVENBb0pFUnBjM1J5YnlrdUxpNGlDaUFnSUNBa1kyMWtJRDBnSjJkeVpYQWdMWEVnSWx0aWIyOTBYU0lnTDJWMFl5OTNjMnd1WTI5dVppQXlQaTlrWlhZdmJuVnNiQ0I4ZkNCd2NtbHVkR1lnSWx0aWIyOTBYVnh1YzNsemRHVnRaRDEwY25WbFhHNGlJRDQrSUM5bGRHTXZkM05zTG1OdmJtWTdJR2R5WlhBZ0xYRWdJbk41YzNSbGJXUTlkSEoxWlNJZ0wyVjBZeTkzYzJ3dVkyOXVaaUI4ZkNCelpXUWdMV2tnSWk5Y1cySnZiM1JjWFM5aElITjVjM1JsYldROWRISjFaU0lnTDJWMFl5OTNjMnd1WTI5dVppY0tJQ0FnSUNSeUlEMGdTVzUyYjJ0bExWZHpiRU52YlcxaGJtUWdKRVJwYzNSeWJ5QWtZMjFrQ2lBZ0lDQnBaaUFvSkhJdVJYaHBkRU52WkdVZ0xXNWxJREFwSUhzZ1YzSnBkR1V0UlhKeUlDSkdZV2xzWldRZ2RHOGdaVzVoWW14bElITjVjM1JsYldRNklDUW9KSEl1VDNWMGNIVjBLU0k3SUhKbGRIVnliaUFrWm1Gc2MyVWdmUW9LSUNBZ0lGZHlhWFJsTFVsdVptOGdJbk41YzNSbGJXUWdaVzVoWW14bFpDNGdVbVZ6ZEdGeWRHbHVaeUJYVTB3Z2RHOGdZWEJ3YkhrdUxpNGlDaUFnSUNBbUlIZHpiQzVsZUdVZ0xTMTBaWEp0YVc1aGRHVWdKRVJwYzNSeWJ5QXlQaVl4SUh3Z1QzVjBMVTUxYkd3S0lDQWdJRk4wWVhKMExWTnNaV1Z3SUMxVFpXTnZibVJ6SURVS0lDQWdJRk4wWVhKMExWZHpiRVJwYzNSeWJ5QWtSR2x6ZEhKdkNpQWdJQ0JUZEdGeWRDMVRiR1ZsY0NBdFUyVmpiMjVrY3lBa1EwOU9Sa2xITGxkemJGTjBZWEowZFhCVFpXTUtJQ0FnSUhKbGRIVnliaUFrZEhKMVpRcDlDZ3BtZFc1amRHbHZiaUJIWlhRdFUyVnlkbWxqWlZOMFlYUjFjeUI3Q2lBZ0lDQndZWEpoYlNoYmMzUnlhVzVuWFNSRWFYTjBjbThzSUZ0emRISnBibWRkSkZObGNuWnBZMlVwQ2lBZ0lDQWtjaUE5SUVsdWRtOXJaUzFYYzJ4RGIyMXRZVzVrSUNSRWFYTjBjbThnSW5ONWMzUmxiV04wYkNCcGN5MWhZM1JwZG1VZ0pGTmxjblpwWTJVZ01qNHZaR1YyTDI1MWJHd2lDaUFnSUNCeVpYUjFjbTRnSkhJdVQzVjBjSFYwTGxSeWFXMG9LUXA5Q2dwbWRXNWpkR2x2YmlCVGRHRnlkQzFCWjJWdWRGTmxjblpwWTJVZ2V3b2dJQ0FnY0dGeVlXMG9XM04wY21sdVoxMGtSR2x6ZEhKdkxDQmJjM1J5YVc1blhTUlRaWEoyYVdObEtRb2dJQ0FnVjNKcGRHVXRTVzVtYnlBaVUzUmhjblJwYm1jZ0pGTmxjblpwWTJVdUxpNGlDaUFnSUNBa2NpQTlJRWx1ZG05clpTMVhjMnhEYjIxdFlXNWtJQ1JFYVhOMGNtOGdJbk41YzNSbGJXTjBiQ0J6ZEdGeWRDQWtVMlZ5ZG1salpTQXlQaVl4SWdvZ0lDQWdhV1lnS0NSeUxrVjRhWFJEYjJSbElDMWxjU0F3S1NCN0lGZHlhWFJsTFVsdVptOGdJaVJUWlhKMmFXTmxJSE4wWVhKMFpXUWdUMHN1SWpzZ2NtVjBkWEp1SUNSMGNuVmxJSDBLSUNBZ0lGZHlhWFJsTFVWeWNpQWlSbUZwYkdWa0lIUnZJSE4wWVhKMElDUlRaWEoyYVdObElDaGxlR2wwSUNRb0pISXVSWGhwZEVOdlpHVXBLVG9nSkNna2NpNVBkWFJ3ZFhRcElnb2dJQ0FnY21WMGRYSnVJQ1JtWVd4elpRcDlDZ3BtZFc1amRHbHZiaUJTWlhObGRDMUdZV2xzWldSVFpYSjJhV05sSUhzS0lDQWdJSEJoY21GdEtGdHpkSEpwYm1kZEpFUnBjM1J5Ynl3Z1czTjBjbWx1WjEwa1UyVnlkbWxqWlNrS0lDQWdJQ1J1ZFd4c0lEMGdTVzUyYjJ0bExWZHpiRU52YlcxaGJtUWdKRVJwYzNSeWJ5QWljM2x6ZEdWdFkzUnNJSEpsYzJWMExXWmhhV3hsWkNBa1UyVnlkbWxqWlNBeVBpOWtaWFl2Ym5Wc2JDSUtmUW9LWm5WdVkzUnBiMjRnVTNSaGNuUXRWMkYwWTJoa2IyZE1iMjl3SUhzS0lDQWdJQ1JrYVhOMGNtOGdJRDBnSkVOUFRrWkpSeTVYYzJ4RWFYTjBjbThLSUNBZ0lDUnpaWEoyYVdObElEMGdKRU5QVGtaSlJ5NVRaWEoyYVdObFRtRnRaUW9nSUNBZ0pHTnZibk5sWTNWMGFYWmxVbVZ6ZEdGeWRITWdQU0F3Q2dvZ0lDQWdWM0pwZEdVdFNXNW1ieUFpUFQwOUlFUmxRMnh2ZFdRZ1YxTk1JRmRoZEdOb1pHOW5JSE4wWVhKMFpXUWdLRkJKUkNBa1VFbEVLU0E5UFQwaUNpQWdJQ0JYY21sMFpTMUpibVp2SUNKRWFYTjBjbTg2SUNSa2FYTjBjbThnZkNCVFpYSjJhV05sT2lBa2MyVnlkbWxqWlNJS0NpQWdJQ0FrYldGNFYyRnBkQ0E5SURFeU1Ec2dKSGRoYVhSbFpDQTlJREFLSUNBZ0lIZG9hV3hsSUNndGJtOTBJQ2hVWlhOMExWZHpiRUYyWVdsc1lXSnNaU2twSUhzS0lDQWdJQ0FnSUNCcFppQW9KSGRoYVhSbFpDQXRaMlVnSkcxaGVGZGhhWFFwSUhzZ1YzSnBkR1V0UlhKeUlDSlhVMHdnYm05MElHRjJZV2xzWVdKc1pTQmhablJsY2lBa2UyMWhlRmRoYVhSOWN5NGlPeUJ5WlhSMWNtNGdmUW9nSUNBZ0lDQWdJRmR5YVhSbExVbHVabThnSWxkaGFYUnBibWNnWm05eUlGZFRUQzR1TGlBb0pIZGhhWFJsWkM4a2UyMWhlRmRoYVhSOWN5a2lDaUFnSUNBZ0lDQWdVM1JoY25RdFUyeGxaWEFnTFZObFkyOXVaSE1nTVRBN0lDUjNZV2wwWldRZ0t6MGdNVEFLSUNBZ0lIMEtJQ0FnSUZkeWFYUmxMVWx1Wm04Z0lsZFRUQ0JwY3lCaGRtRnBiR0ZpYkdVdUlnb0tJQ0FnSUNSemRHRjBaU0E5SUVkbGRDMVhjMnhFYVhOMGNtOVRkR0YwWlNBa1pHbHpkSEp2Q2lBZ0lDQnBaaUFvSkhOMFlYUmxJQzFsY1NBblRtOTBSbTkxYm1RbktTQjdJRmR5YVhSbExVVnljaUFpUkdsemRISnZJQ2NrWkdsemRISnZKeUJ1YjNRZ1ptOTFibVF1SWpzZ2NtVjBkWEp1SUgwS0lDQWdJR2xtSUNna2MzUmhkR1VnTFc1bElDZFNkVzV1YVc1bkp5a2dleUJUZEdGeWRDMVhjMnhFYVhOMGNtOGdKR1JwYzNSeWJ6c2dVM1JoY25RdFUyeGxaWEFnTFZObFkyOXVaSE1nSkVOUFRrWkpSeTVYYzJ4VGRHRnlkSFZ3VTJWaklIMEtDaUFnSUNCcFppQW9MVzV2ZENBb1ZHVnpkQzFUZVhOMFpXMWtSVzVoWW14bFpDQWtaR2x6ZEhKdktTa2dld29nSUNBZ0lDQWdJR2xtSUNndGJtOTBJQ2hGYm1GaWJHVXRVM2x6ZEdWdFpFbHVWM05zSUNSa2FYTjBjbThwS1NCN0lGZHlhWFJsTFVWeWNpQWlRMkZ1Ym05MElHVnVZV0pzWlNCemVYTjBaVzFrTGlJN0lISmxkSFZ5YmlCOUNpQWdJQ0I5Q2dvZ0lDQWdkMmhwYkdVZ0tDUjBjblZsS1NCN0NpQWdJQ0FnSUNBZ2RISjVJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0pITjBZWFJsSUQwZ1IyVjBMVmR6YkVScGMzUnliMU4wWVhSbElDUmthWE4wY204S0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0NSemRHRjBaU0F0Ym1VZ0oxSjFibTVwYm1jbktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQlhjbWwwWlMxWFlYSnVJQ0pFYVhOMGNtOGdKeVJrYVhOMGNtOG5JR2x6SUNSemRHRjBaUzRnVW1WemRHRnlkR2x1Wnk0dUxpSUtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGTjBZWEowTFZkemJFUnBjM1J5YnlBa1pHbHpkSEp2Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JUZEdGeWRDMVRiR1ZsY0NBdFUyVmpiMjVrY3lBa1EwOU9Sa2xITGxkemJGTjBZWEowZFhCVFpXTUtJQ0FnSUNBZ0lDQWdJQ0FnZlFvS0lDQWdJQ0FnSUNBZ0lDQWdKSE4wWVhSMWN5QTlJRWRsZEMxVFpYSjJhV05sVTNSaGRIVnpJQ1JrYVhOMGNtOGdKSE5sY25acFkyVUtDaUFnSUNBZ0lDQWdJQ0FnSUdsbUlDZ2tjM1JoZEhWeklDMWxjU0FuWVdOMGFYWmxKeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnVjNKcGRHVXRTVzVtYnlBaVd5UnpaWEoyYVdObFhTQmhZM1JwZG1VdUlFOUxMaUlLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ1JqYjI1elpXTjFkR2wyWlZKbGMzUmhjblJ6SUQwZ01Bb2dJQ0FnSUNBZ0lDQWdJQ0I5SUdWc2MyVWdld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdWM0pwZEdVdFYyRnliaUFpV3lSelpYSjJhV05sWFNCcGN5QW5KSE4wWVhSMWN5Y3VJRkpsYzNSaGNuUnBibWN1TGk0aUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9KSE4wWVhSMWN5QXRaWEVnSjJaaGFXeGxaQ2NwSUhzZ1VtVnpaWFF0Um1GcGJHVmtVMlZ5ZG1salpTQWtaR2x6ZEhKdklDUnpaWEoyYVdObElIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hUZEdGeWRDMUJaMlZ1ZEZObGNuWnBZMlVnSkdScGMzUnlieUFrYzJWeWRtbGpaU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDUmpiMjV6WldOMWRHbDJaVkpsYzNSaGNuUnpJRDBnTUFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlNCbGJITmxJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBa1kyOXVjMlZqZFhScGRtVlNaWE4wWVhKMGN5c3JDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tDUmpiMjV6WldOMWRHbDJaVkpsYzNSaGNuUnpJQzFuWlNBa1EwOU9Sa2xITGsxaGVGSmxjM1JoY25SektTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGZHlhWFJsTFZkaGNtNGdJbFJ2YnlCdFlXNTVJR1poYVd4MWNtVnpMaUJDWVdOcmFXNW5JRzltWmlBa0tDUkRUMDVHU1VjdVFtRmphMjltWmxObFl5bHpMaTR1SWdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JUZEdGeWRDMVRiR1ZsY0NBdFUyVmpiMjVrY3lBa1EwOU9Sa2xITGtKaFkydHZabVpUWldNS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSkdOdmJuTmxZM1YwYVhabFVtVnpkR0Z5ZEhNZ1BTQXdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdmU0JqWVhSamFDQjdDaUFnSUNBZ0lDQWdJQ0FnSUZkeWFYUmxMVVZ5Y2lBaVYyRjBZMmhrYjJjZ2JHOXZjQ0JsY25KdmNqb2dKRjhpQ2lBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUZOMFlYSjBMVk5zWldWd0lDMVRaV052Ym1SeklDUkRUMDVHU1VjdVNHVmhiSFJvUTJobFkydFRaV01LSUNBZ0lIMEtmUW9LZEhKNUlIc0tJQ0FnSUVsdWFYUnBZV3hwZW1VdFRHOW5aMmx1WndvZ0lDQWdhV1lnS0MxdWIzUWdLRlJsYzNRdFFXTnhkV2x5WlV4dlkyc3BLU0I3SUdWNGFYUWdNQ0I5Q2lBZ0lDQjBjbmtnSUNBZ0lIc2dVM1JoY25RdFYyRjBZMmhrYjJkTWIyOXdJSDBLSUNBZ0lHWnBibUZzYkhrZ2V5QlNaVzF2ZG1VdFRHOWphenNnVjNKcGRHVXRTVzVtYnlBaVBUMDlJRVJsUTJ4dmRXUWdWMU5NSUZkaGRHTm9aRzluSUhOMGIzQndaV1FnUFQwOUlpQjlDbjBnWTJGMFkyZ2dld29nSUNBZ0pIUnpJRDBnS0VkbGRDMUVZWFJsS1M1VWIxTjBjbWx1WnlnbmVYbDVlUzFOVFMxa1pDQklTRHB0YlRwemN5Y3BDaUFnSUNCMGNua2dleUJCWkdRdFEyOXVkR1Z1ZENBdFVHRjBhQ0FrUTA5T1JrbEhMa3h2WjBacGJHVWdMVlpoYkhWbElDSmJKSFJ6WFNCYlJrRlVRVXhkSUNSZklpQXRSVzVqYjJScGJtY2dWVlJHT0NCOUlHTmhkR05vSUh0OUNpQWdJQ0JsZUdsMElERUtmUW9uUUFwOUNnb2pJT0tVZ09LVWdDQkpibk4wWVd4c0lIZGhkR05vWkc5bklITmpjbWx3ZENEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElBS1puVnVZM1JwYjI0Z1NXNXpkR0ZzYkMxWFlYUmphR1J2WjFOamNtbHdkQ0I3Q2lBZ0lDQlhjbWwwWlMxVGRHVndJQ0pKYm5OMFlXeHNhVzVuSUhkaGRHTm9aRzluSUhSdk9pQWtTVTVUVkVGTVRGOUVTVklpQ2dvZ0lDQWdabTl5WldGamFDQW9KR1JwY2lCcGJpQkFLQ1JKVGxOVVFVeE1YMFJKVWl3Z0pFeFBSMTlFU1ZJcEtTQjdDaUFnSUNBZ0lDQWdhV1lnS0MxdWIzUWdLRlJsYzNRdFVHRjBhQ0FrWkdseUtTa2dld29nSUNBZ0lDQWdJQ0FnSUNCT1pYY3RTWFJsYlNBdFNYUmxiVlI1Y0dVZ1JHbHlaV04wYjNKNUlDMVFZWFJvSUNSa2FYSWdMVVp2Y21ObElId2dUM1YwTFU1MWJHd0tJQ0FnSUNBZ0lDQjlDaUFnSUNCOUNnb2dJQ0FnSXlCTWIyTnJJR1JwY21WamRHOXllU0IwYnlCVFdWTlVSVTBnS3lCQlpHMXBibWx6ZEhKaGRHOXljeUJ2Ym14NUNpQWdJQ0FrWVdOc0lEMGdSMlYwTFVGamJDQWtTVTVUVkVGTVRGOUVTVklLSUNBZ0lDUmhZMnd1VTJWMFFXTmpaWE56VW5Wc1pWQnliM1JsWTNScGIyNG9KSFJ5ZFdVc0lDUm1ZV3h6WlNrS0lDQWdJRUFvQ2lBZ0lDQWdJQ0FnVzFObFkzVnlhWFI1TGtGalkyVnpjME52Ym5SeWIyd3VSbWxzWlZONWMzUmxiVUZqWTJWemMxSjFiR1ZkT2pwdVpYY29KMU5aVTFSRlRTY3NJQ0FnSUNBZ0lDQWdKMFoxYkd4RGIyNTBjbTlzSnl3Z0owTnZiblJoYVc1bGNrbHVhR1Z5YVhRc1QySnFaV04wU1c1b1pYSnBkQ2NzSUNkT2IyNWxKeXdnSjBGc2JHOTNKeWtzQ2lBZ0lDQWdJQ0FnVzFObFkzVnlhWFI1TGtGalkyVnpjME52Ym5SeWIyd3VSbWxzWlZONWMzUmxiVUZqWTJWemMxSjFiR1ZkT2pwdVpYY29KMEZrYldsdWFYTjBjbUYwYjNKekp5d2dKMFoxYkd4RGIyNTBjbTlzSnl3Z0owTnZiblJoYVc1bGNrbHVhR1Z5YVhRc1QySnFaV04wU1c1b1pYSnBkQ2NzSUNkT2IyNWxKeXdnSjBGc2JHOTNKeWtLSUNBZ0lDa2dmQ0JHYjNKRllXTm9MVTlpYW1WamRDQjdJQ1JoWTJ3dVFXUmtRV05qWlhOelVuVnNaU2drWHlrZ2ZRb2dJQ0FnVTJWMExVRmpiQ0F0VUdGMGFDQWtTVTVUVkVGTVRGOUVTVklnTFVGamJFOWlhbVZqZENBa1lXTnNDZ29nSUNBZ0pIZGhkR05vWkc5blVHRjBhQ0E5SUVwdmFXNHRVR0YwYUNBa1NVNVRWRUZNVEY5RVNWSWdKMFJsUTJ4dmRXUXRWM05zVjJGMFkyaGtiMmN1Y0hNeEp3b2dJQ0FnVTJWMExVTnZiblJsYm5RZ0xWQmhkR2dnSkhkaGRHTm9aRzluVUdGMGFDQXRWbUZzZFdVZ0tFZGxkQzFYWVhSamFHUnZaMU5qY21sd2RDa2dMVVZ1WTI5a2FXNW5JRlZVUmpnS0lDQWdJRmR5YVhSbExVOUxJQ0pYWVhSamFHUnZaeUJwYm5OMFlXeHNaV1FnZEc4NklDUjNZWFJqYUdSdloxQmhkR2dpQ2lBZ0lDQnlaWFIxY200Z0pIZGhkR05vWkc5blVHRjBhQXA5Q2dvaklPS1VnT0tVZ0NCU1pXZHBjM1JsY2lCVFkyaGxaSFZzWldRZ1ZHRnpheURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSUFLWm5WdVkzUnBiMjRnVW1WbmFYTjBaWEl0VjJGMFkyaGtiMmRVWVhOcklIc0tJQ0FnSUhCaGNtRnRLRnR6ZEhKcGJtZGRKRmRoZEdOb1pHOW5VR0YwYUNrS0lDQWdJRmR5YVhSbExWTjBaWEFnSWxKbFoybHpkR1Z5YVc1bklGTmphR1ZrZFd4bFpDQlVZWE5yT2lBbkpGUkJVMHRmVGtGTlJTY2lDZ29nSUNBZ2FXWWdLRWRsZEMxVFkyaGxaSFZzWldSVVlYTnJJQzFVWVhOclRtRnRaU0FrVkVGVFMxOU9RVTFGSUMxRmNuSnZja0ZqZEdsdmJpQlRhV3hsYm5Sc2VVTnZiblJwYm5WbEtTQjdDaUFnSUNBZ0lDQWdWM0pwZEdVdFYyRnliaUFpUlhocGMzUnBibWNnZEdGemF5Qm1iM1Z1WkNBdExTQnlaWEJzWVdOcGJtY3VJZ29nSUNBZ0lDQWdJRk4wYjNBdFUyTm9aV1IxYkdWa1ZHRnpheUFnSUNBZ0lDQXRWR0Z6YTA1aGJXVWdKRlJCVTB0ZlRrRk5SU0F0UlhKeWIzSkJZM1JwYjI0Z1UybHNaVzUwYkhsRGIyNTBhVzUxWlFvZ0lDQWdJQ0FnSUZWdWNtVm5hWE4wWlhJdFUyTm9aV1IxYkdWa1ZHRnpheUF0VkdGemEwNWhiV1VnSkZSQlUwdGZUa0ZOUlNBdFEyOXVabWx5YlRva1ptRnNjMlVLSUNBZ0lIMEtDaUFnSUNBa2NITkZlR1VnSUQwZ0lpUmxiblk2VTNsemRHVnRVbTl2ZEZ4VGVYTjBaVzB6TWx4WGFXNWtiM2R6VUc5M1pYSlRhR1ZzYkZ4Mk1TNHdYSEJ2ZDJWeWMyaGxiR3d1WlhobElnb2dJQ0FnSkhCelFYSm5jeUE5SUNJdFRtOXVTVzUwWlhKaFkzUnBkbVVnTFZkcGJtUnZkMU4wZVd4bElFaHBaR1JsYmlBdFJYaGxZM1YwYVc5dVVHOXNhV041SUVKNWNHRnpjeUF0Um1sc1pTQmdJaVJYWVhSamFHUnZaMUJoZEdoZ0lpSUtJQ0FnSUNSaFkzUnBiMjRnUFNCT1pYY3RVMk5vWldSMWJHVmtWR0Z6YTBGamRHbHZiaUF0UlhobFkzVjBaU0FrY0hORmVHVWdMVUZ5WjNWdFpXNTBJQ1J3YzBGeVozTUtDaUFnSUNBa2RISnBaMmRsY2tKdmIzUWdJQ0FnSUNBZ1BTQk9aWGN0VTJOb1pXUjFiR1ZrVkdGemExUnlhV2RuWlhJZ0xVRjBVM1JoY25SMWNBb2dJQ0FnSkhSeWFXZG5aWEpDYjI5MExrUmxiR0Y1SUQwZ0oxQlVNekJUSndvZ0lDQWdKSFJ5YVdkblpYSlNaWEJsWVhRZ0lDQWdJRDBnVG1WM0xWTmphR1ZrZFd4bFpGUmhjMnRVY21sbloyVnlJQzFTWlhCbGRHbDBhVzl1U1c1MFpYSjJZV3dnS0U1bGR5MVVhVzFsVTNCaGJpQXRUV2x1ZFhSbGN5QTFLU0F0VDI1alpTQXRRWFFnS0VkbGRDMUVZWFJsS1FvS0lDQWdJQ1J6WlhSMGFXNW5jeUE5SUU1bGR5MVRZMmhsWkhWc1pXUlVZWE5yVTJWMGRHbHVaM05UWlhRZ1lBb2dJQ0FnSUNBZ0lDMUZlR1ZqZFhScGIyNVVhVzFsVEdsdGFYUWdJQ0FnSUNBZ0lEQWdZQW9nSUNBZ0lDQWdJQzFTWlhOMFlYSjBRMjkxYm5RZ0lDQWdJQ0FnSUNBZ0lDQWdJREV3SUdBS0lDQWdJQ0FnSUNBdFVtVnpkR0Z5ZEVsdWRHVnlkbUZzSUNBZ0lDQWdJQ0FnSUNBb1RtVjNMVlJwYldWVGNHRnVJQzFOYVc1MWRHVnpJREVwSUdBS0lDQWdJQ0FnSUNBdFUzUmhjblJYYUdWdVFYWmhhV3hoWW14bElDQWdJQ0FnSUNCZ0NpQWdJQ0FnSUNBZ0xWSjFiazl1YkhsSlprNWxkSGR2Y210QmRtRnBiR0ZpYkdVNkpHWmhiSE5sSUdBS0lDQWdJQ0FnSUNBdFRYVnNkR2x3YkdWSmJuTjBZVzVqWlhNZ0lDQWdJQ0FnSUNCSloyNXZjbVZPWlhjS0NpQWdJQ0FrY0hKcGJtTnBjR0ZzSUQwZ1RtVjNMVk5qYUdWa2RXeGxaRlJoYzJ0UWNtbHVZMmx3WVd3Z1lBb2dJQ0FnSUNBZ0lDMVZjMlZ5U1dRZ0lDQWdKMU5aVTFSRlRTY2dZQW9nSUNBZ0lDQWdJQzFNYjJkdmJsUjVjR1VnVTJWeWRtbGpaVUZqWTI5MWJuUWdZQW9nSUNBZ0lDQWdJQzFTZFc1TVpYWmxiQ0FnU0dsbmFHVnpkQW9LSUNBZ0lGSmxaMmx6ZEdWeUxWTmphR1ZrZFd4bFpGUmhjMnNnWUFvZ0lDQWdJQ0FnSUMxVVlYTnJUbUZ0WlNBZ0lDQWtWRUZUUzE5T1FVMUZJR0FLSUNBZ0lDQWdJQ0F0UkdWelkzSnBjSFJwYjI0Z0owdGxaWEJ6SUhSb1pTQkVaVU5zYjNWa0lHNXZaR1VnWVdkbGJuUWdjblZ1Ym1sdVp5QnBibk5wWkdVZ1YxTk1JR0YwSUdGc2JDQjBhVzFsY3k0bklHQUtJQ0FnSUNBZ0lDQXRRV04wYVc5dUlDQWdJQ0FnSkdGamRHbHZiaUJnQ2lBZ0lDQWdJQ0FnTFZSeWFXZG5aWElnSUNBZ0lFQW9KSFJ5YVdkblpYSkNiMjkwTENBa2RISnBaMmRsY2xKbGNHVmhkQ2tnWUFvZ0lDQWdJQ0FnSUMxVFpYUjBhVzVuY3lBZ0lDQWtjMlYwZEdsdVozTWdZQW9nSUNBZ0lDQWdJQzFRY21sdVkybHdZV3dnSUNBa2NISnBibU5wY0dGc0lHQUtJQ0FnSUNBZ0lDQXRSbTl5WTJVZ2ZDQlBkWFF0VG5Wc2JBb0tJQ0FnSUZkeWFYUmxMVTlMSUNKVFkyaGxaSFZzWldRZ2RHRnpheUJ5WldkcGMzUmxjbVZrTGlJS2ZRb0tJeURpbElEaWxJQWdVM1JoY25RZ2QyRjBZMmhrYjJjZ2FXMXRaV1JwWVhSbGJIa2c0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0FDbVoxYm1OMGFXOXVJRk4wWVhKMExWZGhkR05vWkc5blRtOTNJSHNLSUNBZ0lGZHlhWFJsTFZOMFpYQWdJbE4wWVhKMGFXNW5JSGRoZEdOb1pHOW5MaTR1SWdvZ0lDQWdVM1JoY25RdFUyTm9aV1IxYkdWa1ZHRnpheUF0VkdGemEwNWhiV1VnSkZSQlUwdGZUa0ZOUlFvZ0lDQWdVM1JoY25RdFUyeGxaWEFnTFZObFkyOXVaSE1nTXdvZ0lDQWdKSE4wWVhSbElEMGdLRWRsZEMxVFkyaGxaSFZzWldSVVlYTnJJQzFVWVhOclRtRnRaU0FrVkVGVFMxOU9RVTFGS1M1VGRHRjBaUW9nSUNBZ1YzSnBkR1V0VDBzZ0lsZGhkR05vWkc5bklITjBZWFJsT2lBa2MzUmhkR1VpQ24wS0NpTWc0cFNBNHBTQUlGTjFiVzFoY25rZzRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0FDbVoxYm1OMGFXOXVJRmR5YVhSbExWTjFiVzFoY25rZ2V3b2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlnb2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlDQXJMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFNzaUlDMUdiM0psWjNKdmRXNWtRMjlzYjNJZ1IzSmxaVzRLSUNBZ0lGZHlhWFJsTFVodmMzUWdJaUFnZkNBZ0lDQWdJQ0FnSUNBZ0lDQWdTVzV6ZEdGc2JHRjBhVzl1SUVOdmJYQnNaWFJsSVNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjhJaUF0Um05eVpXZHliM1Z1WkVOdmJHOXlJRWR5WldWdUNpQWdJQ0JYY21sMFpTMUliM04wSUNJZ0lDc3RMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0S3lJZ0xVWnZjbVZuY205MWJtUkRiMnh2Y2lCSGNtVmxiZ29nSUNBZ1YzSnBkR1V0U0c5emRDQWlJZ29nSUNBZ1YzSnBkR1V0U0c5emRDQWlJQ0JYVTB3Z1JHbHpkSEp2SUNBZ09pQWtSRWxUVkZKUElnb2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlDQk9iMlJsSUZObGNuWnBZMlVnT2lBa1UwVlNWa2xEUlY5T1FVMUZJZ29nSUNBZ1YzSnBkR1V0U0c5emRDQWlJQ0JVWVhOcklFNWhiV1VnSUNBZ09pQWtWRUZUUzE5T1FVMUZJZ29nSUNBZ1YzSnBkR1V0U0c5emRDQWlJQ0JNYjJjZ1JtbHNaU0FnSUNBZ09pQWtURTlIWDBSSlVseDNjMnd0ZDJGMFkyaGtiMmN1Ykc5bklnb2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlnb2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlDQlVhR1VnZDJGMFkyaGtiMmNnZDJsc2JEb2lDaUFnSUNCWGNtbDBaUzFJYjNOMElDSWdJQ0FnS2lCQmRYUnZMWE4wWVhKMElHOXVJR1YyWlhKNUlGZHBibVJ2ZDNNZ1ltOXZkQ0FvTXpCeklHUmxiR0Y1S1NJS0lDQWdJRmR5YVhSbExVaHZjM1FnSWlBZ0lDQXFJRk5sYkdZdGFHVmhiQ0JsZG1WeWVTQTFJRzFwYm5WMFpYTWdhV1lnYVhRZ1kzSmhjMmhsY3lJS0lDQWdJRmR5YVhSbExVaHZjM1FnSWlBZ0lDQXFJRkpsYzNSaGNuUWdkR2hsSUc1dlpHVXRZV2RsYm5RZ2FXWWdhWFFnYzNSdmNITWlDaUFnSUNCWGNtbDBaUzFJYjNOMElDSWdJQ0FnS2lCU2RXNGdjMmxzWlc1MGJIa2dZWE1nVTFsVFZFVk5JQzB0SUc1dklHeHZaMmx1SUhKbGNYVnBjbVZrSWdvZ0lDQWdWM0pwZEdVdFNHOXpkQ0FpSWdvZ0lDQWdWM0pwZEdVdFNHOXpkQ0FpSUNCTmIyNXBkRzl5SUd4dlozTTZJZ29nSUNBZ1YzSnBkR1V0U0c5emRDQWlJQ0FnSUVkbGRDMURiMjUwWlc1MElHQWlKRXhQUjE5RVNWSmNkM05zTFhkaGRHTm9aRzluTG14dloyQWlJQzFVWVdsc0lEVXdJQzFYWVdsMElnb2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlnb2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlDQlZibWx1YzNSaGJHdzZJZ29nSUNBZ1YzSnBkR1V0U0c5emRDQWlJQ0FnSUZWdWNtVm5hWE4wWlhJdFUyTm9aV1IxYkdWa1ZHRnpheUF0VkdGemEwNWhiV1VnSnlSVVFWTkxYMDVCVFVVbklDMURiMjVtYVhKdE9tQWtabUZzYzJVaUNpQWdJQ0JYY21sMFpTMUliM04wSUNJZ0lDQWdVbVZ0YjNabExVbDBaVzBnTFZKbFkzVnljMlVnTFVadmNtTmxJQ2NrU1U1VFZFRk1URjlFU1ZJbklnb2dJQ0FnVjNKcGRHVXRTRzl6ZENBaUlncDlDZ29qSU9LVWdPS1VnQ0JOWVdsdUlPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnQXBYY21sMFpTMUNZVzV1WlhJS1EyOXVabWx5YlMxQlpHMXBiZ3BVWlhOMExWQnlaWEpsY1hWcGMybDBaWE1LSkhkaGRHTm9aRzluVUdGMGFDQTlJRWx1YzNSaGJHd3RWMkYwWTJoa2IyZFRZM0pwY0hRS1VtVm5hWE4wWlhJdFYyRjBZMmhrYjJkVVlYTnJJQzFYWVhSamFHUnZaMUJoZEdnZ0pIZGhkR05vWkc5blVHRjBhQXBUZEdGeWRDMVhZWFJqYUdSdlowNXZkd3BYY21sMFpTMVRkVzF0WVhKNUNsSmxZV1F0U0c5emRDQWlVSEpsYzNNZ1JXNTBaWElnZEc4Z1kyeHZjMlVpQ2c9PQo="

    # Resolve the Windows user's Desktop path through cmd.exe interop
    win_desktop=$(cmd.exe /c 'echo %USERPROFILE%\Desktop' 2>/dev/null | tr -d '\r\n')
    if [ -z "$win_desktop" ]; then
        log_warn "Could not resolve Windows Desktop path"
        return 1
    fi

    wsl_desktop=$(wslpath "$win_desktop" 2>/dev/null)
    if [ -z "$wsl_desktop" ] || [ ! -d "$wsl_desktop" ]; then
        log_warn "Could not convert Windows Desktop path to WSL path"
        return 1
    fi

    bat_wsl_path="${wsl_desktop}/DeCloud-Node-Setup.bat"

    log_info "Writing Windows installer to Desktop..."
    if printf '%s' "$BAT_B64" | base64 -d > "$bat_wsl_path" 2>/dev/null; then
        bat_win_path=$(wslpath -w "$bat_wsl_path" 2>/dev/null)
        log_success "Installer written to Desktop: ${bat_win_path}"
        echo "$bat_win_path"
    else
        log_warn "Could not write installer to Desktop"
        return 1
    fi
}

# Emits an OSC 8 clickable hyperlink — supported by Windows Terminal, iTerm2, etc.
# Usage: make_hyperlink "file:///C:/..." "link label"
make_hyperlink() {
    local url="$1" text="$2"
    printf '\e]8;;%s\e\\%s\e]8;;\e\\' "$url" "$text"
}

# ============================================================
# Summary
# ============================================================
print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           ✅ Installation Complete!                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Installation Summary:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo "    Node Agent:    Installed & Running"
    echo "    Status:        Awaiting Authentication"
    echo "    Public IP:     ${PUBLIC_IP}"

    # GPU summary
    if [ "$GPU_MODE" != "none" ]; then
        local gpu_name=""
        if [ -n "$NVIDIA_SMI_PATH" ] && [ "$NVIDIA_SMI_PATH" != "pci-detected" ] && [ "$NVIDIA_SMI_PATH" != "wsl-cuda-only" ]; then
            gpu_name=$($NVIDIA_SMI_PATH --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        fi
        echo "    GPU:           ${gpu_name:-detected}"
        echo "    GPU Mode:      ${GPU_MODE}"
        echo "    Proxy Daemon:  $([ -f /usr/local/bin/gpu-proxy-daemon ] && echo 'installed' || echo 'not built')"
        echo "    CUDA Shim:     $([ -f /usr/local/lib/decloud-gpu-shim/libdecloud_cuda_shim.so ] && echo 'installed' || echo 'not built')"
    else
        echo "    GPU:           not detected"
    fi
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Next Steps:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo ""
    echo "    1. Configure your node:"
    echo -e "       ${BOLD}sudo decloud configure --country TR --region eu-east --name MyNode${NC}"
    echo ""
    echo "    2. Register with wallet signature:"
    echo -e "       ${BOLD}sudo decloud register${NC}"
    echo ""
    echo "    3. Resume scheduling:"
    echo -e "       ${BOLD}sudo decloud login${NC}"
    echo ""
    echo "    4. Check node status:"
    echo -e "       ${BOLD}decloud status${NC}"
    echo ""
    echo "    5. Monitor logs:"
    echo -e "       ${BOLD}decloud logs -f${NC}"
    echo "       (or: sudo journalctl -u decloud-node-agent -f)"
    echo ""
    echo "    6. View all commands:"
    echo -e "       ${BOLD}decloud --help${NC}"
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Quick Reference:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo ""
    echo -e "    ${BOLD}decloud status${NC}           Show comprehensive node status"
    echo -e "    ${BOLD}decloud vm list${NC}          List all VMs on this node"
    echo -e "    ${BOLD}decloud diagnose${NC}         Run health diagnostics"
    echo -e "    ${BOLD}decloud resources${NC}        Show resource information"
    echo -e "    ${BOLD}decloud logs -f${NC}          Follow service logs"
    echo -e "    ${BOLD}decloud relay show${NC}       Show relay NAT rules"
    echo -e "    ${BOLD}decloud wg${NC}               WireGuard status"
    echo ""
    echo "  ─────────────────────────────────────────────────────────────"
    echo "  Documentation:"
    echo "  ─────────────────────────────────────────────────────────────"
    echo ""
    echo "    README:     cat /usr/local/share/doc/decloud/README.md"
    echo "    Quick Ref:  cat /usr/local/share/doc/decloud/QUICKREF.md"
    echo ""

    # ── WSL2: Windows auto-start notice ──────────────────────────────────────
    if [ "$IS_WSL2" = true ]; then
        echo "  ─────────────────────────────────────────────────────────────"
        echo -e "  ${YELLOW}⚠  Running inside WSL2 — Windows Auto-Start Required${NC}"
        echo "  ─────────────────────────────────────────────────────────────"
        echo ""
        echo "  The node agent is running NOW, but will stop if Windows"
        echo "  reboots or WSL is shut down. Set up auto-start once and"
        echo "  it will survive every reboot automatically."
        echo ""

        local bat_win_path
        bat_win_path=$(stage_windows_installer)

        if [ -n "$bat_win_path" ]; then
            # Build a file:// URL (backslashes → forward slashes)
            local bat_url
            bat_url="file:///$(echo "$bat_win_path" | sed 's|\\|/|g')"

            echo "  The installer has been placed on your Windows Desktop:"
            echo ""
            # OSC 8 clickable link — works in Windows Terminal
            echo -e "    $(make_hyperlink "$bat_url" "  ▶  Click here to run DeCloud-Node-Setup.bat  ")"
            echo ""
            echo "  Clicking opens cmd → UAC prompt → auto-start service"
            echo "  installed. One click, done."
            echo ""
            echo -e "  ${BOLD}Or run it from any Windows terminal (PowerShell / CMD):${NC}"
            echo "    start \"\" \"${bat_win_path}\""
        else
            echo -e "  ${BOLD}To set up auto-start on the Windows host:${NC}"
            echo "    1. Download  DeCloud-Node-Setup.bat"
            echo "    2. Double-click it and approve the UAC prompt"
        fi

        echo ""
        echo -e "  ${BOLD}To use the DeCloud CLI while the agent runs in the background:${NC}"
        echo "    Open Windows Terminal / PowerShell and type:  wsl"
        echo ""
    fi
}

# ============================================================
# Main
# ============================================================
main() {
    # Reset CWD to / immediately.
    # If invoked from a directory that no longer exists (e.g. the previous
    # /opt/decloud/DeCloud.NodeAgent clone that was just wiped by 'decloud update'),
    # every subshell bash spawns will try to getcwd() on the deleted inode and
    # print "getcwd: cannot access parent directories: No such file or directory"
    # for every single command.  Changing to / before anything else ensures all
    # subsequent cd commands and subprocess CWDs resolve cleanly.
    cd / 2>/dev/null || true

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       DeCloud Node Agent Installer v${VERSION}               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    
    parse_args "$@"

    # Load operational settings from settings.json if they exist.
    # On fresh install these default to hostname/default/default.
    # On update (decloud update) they were set by prior configure calls.
    if [ -f /etc/decloud/settings.json ]; then
        NODE_NAME=$(jq -r '.name // empty' /etc/decloud/settings.json)
        NODE_REGION=$(jq -r '.region // "default"' /etc/decloud/settings.json)
        NODE_ZONE=$(jq -r '.zone // "default"' /etc/decloud/settings.json)
    fi
    # Defaults if not set
    NODE_NAME="${NODE_NAME:-$(hostname)}"
    NODE_REGION="${NODE_REGION:-default}"
    NODE_ZONE="${NODE_ZONE:-default}"

    # ─────────────────────────────────────────────────────────────────────
    # Persist install parameters immediately after parsing.
    # 'decloud update' reads /etc/decloud/install-params to re-run this
    # script with the exact same arguments — no manual bookkeeping needed.
    # Write before init_logging so the file exists even if logging setup fails.
    # ─────────────────────────────────────────────────────────────────────
    mkdir -p /etc/decloud
    # Merge into existing settings.json if it exists (preserves
    # configure-set fields like country/region across updates),
    # or create fresh if this is a new install.
    if [ -f /etc/decloud/settings.json ]; then
        local tmp; tmp=$(mktemp)
        jq --arg o "$ORCHESTRATOR_URL" --arg w "$NODE_WALLET" \
           '.orchestrator_url = $o | .wallet = $w' \
           /etc/decloud/settings.json > "$tmp" \
           && mv "$tmp" /etc/decloud/settings.json
    else
        cat > /etc/decloud/settings.json <<SETTINGSEOF
{
  "version": 1,
  "orchestrator_url": "${ORCHESTRATOR_URL}",
  "wallet": "${NODE_WALLET}",
  "name": "$(hostname)",
  "region": "default",
  "zone": "default",
  "agent_port": ${AGENT_PORT},
  "wireguard_port": ${WIREGUARD_PORT}
}
SETTINGSEOF
    fi
    chmod 600 /etc/decloud/settings.json

    # Initialize logging
    init_logging

    # CRITICAL: Validate required parameters FIRST
    check_required_params
    
    # Checks
    check_root
    check_os
    detect_wsl2
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

    # GPU: detect mode, container runtime, and CUDA toolkit
    detect_gpu_mode
    install_docker
    install_nvidia_container_toolkit
    install_cuda_toolkit

    # ARM64 support
    install_arm_virtualization
    configure_architecture_optimizations

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
    
    # Install CLI from verified release bundle
    install_walletconnect_cli
    install_decloud_cli
    install_decloud_docs
    install_relay_nat_support

    build_gpu_proxy
    build_node_agent
    create_configuration
    create_systemd_service
    configure_firewall
    create_helper_scripts
    
    # Start service (only if authenticated)
    start_service
    
    # Done
    print_summary
}

main "$@"