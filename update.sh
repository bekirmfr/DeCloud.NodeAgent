#!/bin/bash
#
# DeCloud Node Agent Update Script
#
# Updates the Node Agent while preserving configuration.
# Safe to run multiple times.
#
# Version: 2.0.0
#
# Usage:
#   sudo ./update.sh              # Normal update
#   sudo ./update.sh --force      # Force rebuild even if no changes
#

set -e

VERSION="2.0.0"

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
# Configuration
# ============================================================

INSTALL_DIR="/opt/decloud"
REPO_DIR="$INSTALL_DIR/DeCloud.NodeAgent"
CONFIG_DIR="/etc/decloud"
LOG_DIR="/var/log/decloud"
SERVICE_NAME="decloud-node-agent"

# Flags
FORCE_REBUILD=false
DEPS_ONLY=false
CHANGES_DETECTED=false

# ============================================================
# Argument Parsing
# ============================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force|-f)
                FORCE_REBUILD=true
                shift
                ;;
            --deps-only|-d)
                DEPS_ONLY=true
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
DeCloud Node Agent Update Script v${VERSION}

Usage: $0 [options]

Options:
  --force, -f       Force rebuild even if no code changes detected
  --deps-only, -d   Only check dependencies, don't update code
  --help, -h        Show this help message

Examples:
  $0                # Normal update
  $0 --force        # Force rebuild
  $0 --deps-only    # Only check dependencies
EOF
}

# ============================================================
# Checks
# ============================================================
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_installation() {
    if [ ! -d "$REPO_DIR" ]; then
        log_error "Node Agent not installed at $REPO_DIR"
        log_error "Run install.sh first"
        exit 1
    fi
    
    if [ ! -f "$CONFIG_DIR/appsettings.Production.json" ]; then
        log_error "Configuration not found at $CONFIG_DIR/appsettings.Production.json"
        exit 1
    fi
    
    log_success "Existing installation found"
}

check_service_status() {
    log_step "Checking service status..."
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        log_success "Node Agent service is running"
    else
        log_warn "Node Agent service is not running"
    fi
}

check_wireguard() {
    log_step "Checking WireGuard status..."
    
    if systemctl is-active --quiet wg-quick@wg0; then
        local peers=$(wg show wg0 peers 2>/dev/null | wc -l)
        log_success "WireGuard running ($peers peers)"
    else
        log_warn "WireGuard not running"
    fi
}

# ============================================================
# Update Functions
# ============================================================
fetch_updates() {
    log_step "Fetching latest code..."
    
    cd "$REPO_DIR"
    
    # Store current commit
    OLD_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    
    # Fetch changes
    git fetch origin --quiet 2>/dev/null || {
        log_error "Failed to fetch updates from repository"
        exit 1
    }
    
    # Check for changes
    LOCAL=$(git rev-parse HEAD 2>/dev/null)
    REMOTE=$(git rev-parse origin/master 2>/dev/null || git rev-parse origin/main 2>/dev/null)
    
    if [ "$LOCAL" = "$REMOTE" ]; then
        log_success "Already up to date (commit: ${LOCAL:0:7})"
        CHANGES_DETECTED=false
    else
        log_info "Updates available: ${LOCAL:0:7} → ${REMOTE:0:7}"
        
        # Pull changes
        git reset --hard origin/master 2>/dev/null || git reset --hard origin/main 2>/dev/null
        
        NEW_COMMIT=$(git rev-parse HEAD)
        log_success "Updated to commit: ${NEW_COMMIT:0:7}"
        CHANGES_DETECTED=true
        
        # Show recent commits
        echo ""
        log_info "Recent changes:"
        git log --oneline -5 2>/dev/null | while read line; do
            echo "    $line"
        done
        echo ""
    fi
}

build_node_agent() {
    log_step "Building Node Agent..."
    
    cd "$REPO_DIR"
    
    # Clean previous build
    dotnet clean --verbosity quiet > /dev/null 2>&1 || true
    
    # Build
    dotnet build --configuration Release --verbosity quiet > /dev/null 2>&1
    
    # Publish
    dotnet publish src/DeCloud.NodeAgent/DeCloud.NodeAgent.csproj \
        --configuration Release \
        --output "$INSTALL_DIR/publish" \
        --verbosity quiet > /dev/null 2>&1
    
    log_success "Build complete"
}

restart_service() {
    log_step "Restarting Node Agent service..."
    
    systemctl restart $SERVICE_NAME 2>/dev/null || {
        log_error "Failed to restart service"
        return 1
    }
    
    sleep 3
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        log_success "Service restarted successfully"
    else
        log_error "Service failed to start"
        log_info "Check logs: journalctl -u $SERVICE_NAME -n 50"
        return 1
    fi
}

verify_health() {
    log_step "Verifying Node Agent health..."
    
    local max_attempts=10
    local attempt=1
    local port=$(grep -oP '"Urls":\s*"http://0.0.0.0:\K[0-9]+' "$CONFIG_DIR/appsettings.Production.json" 2>/dev/null || echo "5100")
    
    while [ $attempt -le $max_attempts ]; do
        local status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null || echo "000")
        
        if [ "$status" = "200" ]; then
            log_success "Node Agent healthy (port $port)"
            return 0
        fi
        
        sleep 1
        ((attempt++))
    done
    
    log_warn "Health check incomplete (service may still be starting)"
}

# ============================================================
# Status Display
# ============================================================
show_status() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "                  DeCloud Node Agent Status"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Service status
    echo "  Service:"
    if systemctl is-active --quiet $SERVICE_NAME; then
        local port=$(grep -oP '"Urls":\s*"http://0.0.0.0:\K[0-9]+' "$CONFIG_DIR/appsettings.Production.json" 2>/dev/null || echo "5100")
        echo -e "    Status:        ${GREEN}Running${NC}"
        echo "    API Port:      $port"
    else
        echo -e "    Status:        ${RED}Stopped${NC}"
    fi
    
    # Version
    if [ -d "$REPO_DIR/.git" ]; then
        local commit=$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        local branch=$(cd "$REPO_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        echo "    Version:       $branch @ $commit"
    fi
    
    # WireGuard
    echo ""
    echo "  WireGuard:"
    if systemctl is-active --quiet wg-quick@wg0; then
        local wg_port=$(grep -oP 'ListenPort\s*=\s*\K[0-9]+' /etc/wireguard/wg0.conf 2>/dev/null || echo "51820")
        local peers=$(wg show wg0 peers 2>/dev/null | wc -l)
        echo -e "    Status:        ${GREEN}Running${NC}"
        echo "    Port:          $wg_port/udp"
        echo "    Peers:         $peers"
    else
        echo -e "    Status:        ${YELLOW}Not running${NC}"
    fi
    
    # Resources
    echo ""
    echo "  Resources:"
    local cpu=$(nproc)
    local mem=$(free -m | awk '/^Mem:/{print $2}')
    local disk=$(df -BG /var/lib/decloud 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo "?")
    echo "    CPU Cores:     $cpu"
    echo "    Memory:        ${mem}MB"
    echo "    Disk Free:     ${disk}GB"
    
    # VM count
    local vm_count=$(find /var/lib/decloud/vms -maxdepth 1 -type d 2>/dev/null | wc -l)
    vm_count=$((vm_count - 1))
    if [ $vm_count -ge 0 ]; then
        echo "    VMs:           $vm_count"
    fi
    
    echo ""
    echo "  Commands:"
    echo "    Logs:          sudo journalctl -u $SERVICE_NAME -f"
    echo "    Restart:       sudo systemctl restart $SERVICE_NAME"
    echo "    WireGuard:     sudo decloud-wg status"
    echo ""
}

# ============================================================
# Main
# ============================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       DeCloud Node Agent Update Script v${VERSION}                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    
    parse_args "$@"
    
    # Checks
    check_root
    check_installation
    check_service_status
    check_wireguard
    
    if [ "$DEPS_ONLY" = true ]; then
        log_success "Dependency check complete"
        show_status
        exit 0
    fi
    
    # Fetch updates
    fetch_updates
    
    # Build and restart if changes detected or forced
    if [ "$CHANGES_DETECTED" = true ] || [ "$FORCE_REBUILD" = true ]; then
        if [ "$FORCE_REBUILD" = true ] && [ "$CHANGES_DETECTED" = false ]; then
            log_info "Forcing rebuild (--force)"
        fi
        
        build_node_agent
        restart_service
        verify_health
    else
        log_info "No changes detected, skipping rebuild"
        log_info "Use --force to rebuild anyway"
    fi
    
    # Show final status
    show_status
    
    log_success "Update complete!"
}

main "$@"