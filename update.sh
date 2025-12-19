#!/bin/bash
#
# DeCloud Node Agent Update Script
#
# Updates the Node Agent, Caddy, and fail2ban while preserving configuration.
# Safe to run multiple times.
#
# Version: 1.5.0
# Changelog:
# - Added Caddy ingress gateway update support
# - Added fail2ban update and health verification
# - Added security status reporting
#
# Usage:
#   sudo ./update.sh              # Normal update
#   sudo ./update.sh --force      # Force rebuild even if no changes
#   sudo ./update.sh --deps-only  # Only check/install dependencies
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
# Configuration
# ============================================================

INSTALL_DIR="/opt/decloud"
REPO_DIR="$INSTALL_DIR/DeCloud.NodeAgent"
CONFIG_DIR="/etc/decloud"
DECLOUD_LOG_DIR="/var/log/decloud"
SERVICE_NAME="decloud-node-agent"

# Flags
FORCE_REBUILD=false
DEPS_ONLY=false
CHANGES_DETECTED=false
SKIP_CADDY_UPDATE=false
SKIP_FAIL2BAN_UPDATE=false

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
            --skip-caddy-update)
                SKIP_CADDY_UPDATE=true
                shift
                ;;
            --skip-fail2ban-update)
                SKIP_FAIL2BAN_UPDATE=true
                shift
                ;;
            --skip-security-update)
                SKIP_CADDY_UPDATE=true
                SKIP_FAIL2BAN_UPDATE=true
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
  --force, -f              Force rebuild even if no code changes detected
  --deps-only, -d          Only check and install dependencies
  --skip-caddy-update      Skip Caddy update
  --skip-fail2ban-update   Skip fail2ban update
  --skip-security-update   Skip both Caddy and fail2ban updates
  --help, -h               Show this help message

Examples:
  $0                       # Normal update
  $0 --force               # Force rebuild
  $0 --deps-only           # Only check dependencies
  $0 --skip-security-update # Skip Caddy and fail2ban
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

# ============================================================
# Caddy Functions
# ============================================================
check_caddy() {
    if ! command -v caddy &> /dev/null; then
        log_info "Caddy not installed (ingress gateway disabled)"
        return 0
    fi

    log_step "Checking Caddy status..."

    if systemctl is-active --quiet caddy; then
        local version=$(caddy version 2>/dev/null | head -1 | awk '{print $1}')
        log_success "Caddy running: $version"
        
        local admin_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:2019/config/ 2>/dev/null || echo "000")
        if [ "$admin_status" = "200" ]; then
            log_success "Caddy Admin API accessible"
        else
            log_warn "Caddy Admin API not responding"
        fi
    else
        log_warn "Caddy service not running"
    fi
}

update_caddy() {
    if [ "$SKIP_CADDY_UPDATE" = true ]; then
        log_info "Skipping Caddy update (--skip-caddy-update)"
        return 0
    fi

    if ! command -v caddy &> /dev/null; then
        log_info "Caddy not installed, skipping update"
        return 0
    fi

    log_step "Checking for Caddy updates..."

    apt-get update -qq > /dev/null 2>&1

    local current_version=$(caddy version 2>/dev/null | head -1 | awk '{print $1}')
    local available_version=$(apt-cache policy caddy 2>/dev/null | grep Candidate | awk '{print $2}')

    if [ "$current_version" = "$available_version" ] || [ -z "$available_version" ]; then
        log_success "Caddy is up to date: $current_version"
        return 0
    fi

    log_info "Updating Caddy: $current_version → $available_version"

    systemctl stop caddy 2>/dev/null || true
    apt-get install -y -qq caddy > /dev/null 2>&1
    systemctl start caddy 2>/dev/null || true

    sleep 2
    if systemctl is-active --quiet caddy; then
        local new_version=$(caddy version 2>/dev/null | head -1 | awk '{print $1}')
        log_success "Caddy updated: $new_version"
    else
        log_error "Caddy failed to restart after update"
        return 1
    fi
}

verify_caddy_health() {
    if ! command -v caddy &> /dev/null; then
        return 0
    fi

    log_step "Verifying Caddy health..."

    local max_attempts=10
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        local health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null || echo "000")
        local admin=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:2019/config/ 2>/dev/null || echo "000")

        if [ "$health" = "200" ] && [ "$admin" = "200" ]; then
            log_success "Caddy healthy"
            return 0
        fi

        sleep 1
        ((attempt++))
    done

    log_warn "Caddy health check incomplete"
}

# ============================================================
# fail2ban Functions
# ============================================================
check_fail2ban() {
    if ! command -v fail2ban-client &> /dev/null; then
        log_info "fail2ban not installed"
        return 0
    fi

    log_step "Checking fail2ban status..."

    if systemctl is-active --quiet fail2ban; then
        local jails=$(fail2ban-client status 2>/dev/null | grep "Jail list" | cut -d: -f2 | tr -d '[:space:]')
        log_success "fail2ban running"
        log_info "Active jails: $jails"
        
        # Count total bans
        local total_banned=0
        for jail in $(echo "$jails" | tr ',' ' '); do
            local banned=$(fail2ban-client status "$jail" 2>/dev/null | grep "Currently banned" | awk '{print $NF}')
            total_banned=$((total_banned + banned))
        done
        
        if [ "$total_banned" -gt 0 ]; then
            log_info "Currently banned IPs: $total_banned"
        fi
    else
        log_warn "fail2ban service not running"
    fi
}

update_fail2ban() {
    if [ "$SKIP_FAIL2BAN_UPDATE" = true ]; then
        log_info "Skipping fail2ban update (--skip-fail2ban-update)"
        return 0
    fi

    if ! command -v fail2ban-client &> /dev/null; then
        log_info "fail2ban not installed, skipping update"
        return 0
    fi

    log_step "Checking for fail2ban updates..."

    local current_version=$(fail2ban-client --version 2>/dev/null | head -1 | awk '{print $NF}')
    local available_version=$(apt-cache policy fail2ban 2>/dev/null | grep Candidate | awk '{print $2}')

    if [ "$current_version" = "$available_version" ] || [ -z "$available_version" ]; then
        log_success "fail2ban is up to date: $current_version"
        return 0
    fi

    log_info "Updating fail2ban: $current_version → $available_version"

    apt-get install -y -qq fail2ban > /dev/null 2>&1
    systemctl restart fail2ban 2>/dev/null || true

    sleep 2
    if systemctl is-active --quiet fail2ban; then
        log_success "fail2ban updated and running"
    else
        log_error "fail2ban failed to restart after update"
        return 1
    fi
}

verify_fail2ban_health() {
    if ! command -v fail2ban-client &> /dev/null; then
        return 0
    fi

    log_step "Verifying fail2ban health..."

    if fail2ban-client ping > /dev/null 2>&1; then
        log_success "fail2ban responsive"
    else
        log_error "fail2ban not responding"
        return 1
    fi
}

# ============================================================
# Security Log Check
# ============================================================
check_security_logs() {
    log_step "Checking security logs..."

    if [ -f "$DECLOUD_LOG_DIR/audit.log" ]; then
        local log_size=$(du -sh "$DECLOUD_LOG_DIR/audit.log" 2>/dev/null | awk '{print $1}')
        local log_entries=$(wc -l < "$DECLOUD_LOG_DIR/audit.log" 2>/dev/null || echo "0")
        log_success "Audit log: $log_size ($log_entries entries)"
        
        # Check for recent security events
        local recent_violations=$(grep -c "SecurityViolation\|BlockedPortAttempt" "$DECLOUD_LOG_DIR/audit.log" 2>/dev/null || echo "0")
        if [ "$recent_violations" -gt 0 ]; then
            log_warn "Security events detected: $recent_violations violations in audit log"
        fi
    else
        log_info "Audit log not found (may not be enabled)"
    fi
}

# ============================================================
# Node Agent Update Functions
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
    echo "                    DeCloud Node Agent Status"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Node Agent
    echo "  Node Agent Service:"
    if systemctl is-active --quiet $SERVICE_NAME; then
        local port=$(grep -oP '"Urls":\s*"http://0.0.0.0:\K[0-9]+' "$CONFIG_DIR/appsettings.Production.json" 2>/dev/null || echo "5100")
        echo -e "    Status:     ${GREEN}Running${NC}"
        echo "    Port:       $port"
        echo "    Config:     $CONFIG_DIR/appsettings.Production.json"
    else
        echo -e "    Status:     ${RED}Stopped${NC}"
    fi
    
    # Code version
    if [ -d "$REPO_DIR/.git" ]; then
        local commit=$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        local branch=$(cd "$REPO_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        echo "    Version:    $branch @ $commit"
    fi
}

show_security_status() {
    echo ""
    echo "  Security Services:"
    
    # Caddy
    if command -v caddy &> /dev/null; then
        if systemctl is-active --quiet caddy; then
            local version=$(caddy version 2>/dev/null | head -1 | awk '{print $1}')
            echo -e "    Caddy:      ${GREEN}Running${NC} ($version)"
            
            # Count routes
            local route_count=$(curl -s http://localhost:2019/config/apps/http/servers/ingress/routes 2>/dev/null | jq 'length' 2>/dev/null || echo "0")
            echo "                $route_count active routes"
        else
            echo -e "    Caddy:      ${RED}Stopped${NC}"
        fi
    else
        echo "    Caddy:      Not installed"
    fi
    
    # fail2ban
    if command -v fail2ban-client &> /dev/null; then
        if systemctl is-active --quiet fail2ban; then
            local jails=$(fail2ban-client status 2>/dev/null | grep "Jail list" | cut -d: -f2 | tr ',' ' ' | wc -w)
            
            local total_banned=0
            for jail in $(fail2ban-client status 2>/dev/null | grep "Jail list" | cut -d: -f2 | tr ',' ' '); do
                local banned=$(fail2ban-client status "$jail" 2>/dev/null | grep "Currently banned" | awk '{print $NF}')
                total_banned=$((total_banned + banned))
            done
            
            echo -e "    fail2ban:   ${GREEN}Running${NC} ($jails jails, $total_banned banned)"
        else
            echo -e "    fail2ban:   ${RED}Stopped${NC}"
        fi
    else
        echo "    fail2ban:   Not installed"
    fi
    
    # Logs
    echo ""
    echo "  Logs:"
    echo "    Agent:      journalctl -u $SERVICE_NAME -f"
    if [ -f "$DECLOUD_LOG_DIR/audit.log" ]; then
        echo "    Audit:      tail -f $DECLOUD_LOG_DIR/audit.log"
    fi
    if [ -f "/var/log/caddy/caddy.log" ]; then
        echo "    Caddy:      tail -f /var/log/caddy/caddy.log"
    fi
    echo "    fail2ban:   tail -f /var/log/fail2ban.log"
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
    
    # Security checks
    check_caddy
    check_fail2ban
    check_security_logs
    
    if [ "$DEPS_ONLY" = true ]; then
        log_success "Dependency check complete"
        show_status
        show_security_status
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
        
        # Update security components
        update_caddy
        verify_caddy_health
        update_fail2ban
        verify_fail2ban_health
    else
        log_info "No changes detected, skipping rebuild"
        log_info "Use --force to rebuild anyway"
    fi
    
    # Show final status
    show_status
    show_security_status
    
    log_success "Update complete!"
}

main "$@"
