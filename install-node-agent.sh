#!/bin/bash
#
# DeCloud Node Agent Installation Script
# Usage: curl -sSL https://raw.githubusercontent.com/bekirmfr/DeCloud.NodeAgent/main/install.sh | sudo bash -s -- --orchestrator http://ORCHESTRATOR_IP:5050
#
# Or download and run:
#   chmod +x install.sh
#   sudo ./install.sh --orchestrator http://142.234.200.108:5050
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# Default values
ORCHESTRATOR_URL=""
INSTALL_DIR="/opt/decloud"
DATA_DIR="/var/lib/decloud"
CONFIG_DIR="/etc/decloud"
WALLET_ADDRESS="0x0000000000000000000000000000000000000000"
AGENT_PORT=5100
NODE_NAME=$(hostname)
REGION="default"
ZONE="default"

# Parse arguments
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
        --help)
            echo "DeCloud Node Agent Installer"
            echo ""
            echo "Usage: $0 --orchestrator <url> [options]"
            echo ""
            echo "Required:"
            echo "  --orchestrator <url>   Orchestrator URL (e.g., http://142.234.200.108:5050)"
            echo ""
            echo "Optional:"
            echo "  --wallet <address>     Node operator wallet address (default: zero address)"
            echo "  --name <name>          Node name (default: hostname)"
            echo "  --region <region>      Region identifier (default: default)"
            echo "  --zone <zone>          Zone identifier (default: default)"
            echo "  --port <port>          Agent API port (default: 5100)"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required args
if [ -z "$ORCHESTRATOR_URL" ]; then
    log_error "Orchestrator URL is required. Use --orchestrator <url>"
    echo "Example: $0 --orchestrator http://142.234.200.108:5050"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           DeCloud Node Agent Installer v1.0.0                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
log_info "Orchestrator: $ORCHESTRATOR_URL"
log_info "Node Name: $NODE_NAME"
log_info "Region/Zone: $REGION / $ZONE"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "Please run as root (sudo)"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    log_error "Cannot detect OS. Only Ubuntu/Debian are supported."
    exit 1
fi

log_info "Detected OS: $OS $VERSION"

# ============================================================
# Step 1: Install system dependencies
# ============================================================
log_info "Installing system dependencies..."

apt-get update -qq

# Install virtualization packages
apt-get install -y -qq \
    qemu-kvm \
    libvirt-daemon-system \
    libvirt-clients \
    bridge-utils \
    virtinst \
    cloud-image-utils \
    genisoimage \
    curl \
    wget \
    jq \
    > /dev/null 2>&1

log_success "Virtualization packages installed"

# Install .NET 8 if not present
if ! command -v dotnet &> /dev/null; then
    log_info "Installing .NET 8 SDK..."
    
    # Add Microsoft package repository
    wget -q https://packages.microsoft.com/config/$OS/$VERSION/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb
    dpkg -i /tmp/packages-microsoft-prod.deb > /dev/null 2>&1
    rm /tmp/packages-microsoft-prod.deb
    
    apt-get update -qq
    apt-get install -y -qq dotnet-sdk-8.0 > /dev/null 2>&1
    
    log_success ".NET 8 SDK installed"
else
    log_success ".NET already installed: $(dotnet --version)"
fi

# ============================================================
# Step 2: Configure libvirt
# ============================================================
log_info "Configuring libvirt..."

# Enable and start libvirtd
systemctl enable libvirtd --quiet
systemctl start libvirtd

# Ensure default network exists and is active
if ! virsh net-info default &> /dev/null; then
    log_info "Creating default network..."
    virsh net-define /usr/share/libvirt/networks/default.xml > /dev/null 2>&1 || true
fi

virsh net-autostart default > /dev/null 2>&1 || true
virsh net-start default > /dev/null 2>&1 || true

log_success "Libvirt configured"

# ============================================================
# Step 3: Create directories
# ============================================================
log_info "Creating directories..."

mkdir -p $INSTALL_DIR
mkdir -p $DATA_DIR/vms
mkdir -p $DATA_DIR/images
mkdir -p $CONFIG_DIR

log_success "Directories created"

# ============================================================
# Step 4: Download/Clone Node Agent
# ============================================================
log_info "Downloading Node Agent..."

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

# ============================================================
# Step 5: Build the Node Agent
# ============================================================
log_info "Building Node Agent..."

dotnet restore --verbosity quiet
dotnet build -c Release --verbosity quiet

log_success "Node Agent built"

# ============================================================
# Step 6: Create configuration
# ============================================================
log_info "Creating configuration..."

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
    "InterfaceName": "wg0",
    "ConfigPath": "/etc/wireguard",
    "ListenPort": 51820
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

# Create symlink to config
ln -sf $CONFIG_DIR/appsettings.Production.json $INSTALL_DIR/DeCloud.NodeAgent/src/DeCloud.NodeAgent/appsettings.Production.json

log_success "Configuration created"

# ============================================================
# Step 7: Create systemd service
# ============================================================
log_info "Creating systemd service..."

cat > /etc/systemd/system/decloud-node-agent.service << EOF
[Unit]
Description=DeCloud Node Agent
After=network.target libvirtd.service
Requires=libvirtd.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}/DeCloud.NodeAgent
ExecStart=/usr/bin/dotnet run --project src/DeCloud.NodeAgent -c Release --no-build --environment Production
Restart=always
RestartSec=10
Environment=DOTNET_ENVIRONMENT=Production
Environment=ASPNETCORE_ENVIRONMENT=Production

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

# ============================================================
# Step 8: Configure firewall
# ============================================================
log_info "Configuring firewall..."

# Allow agent port
if command -v ufw &> /dev/null; then
    ufw allow $AGENT_PORT/tcp > /dev/null 2>&1 || true
    log_success "UFW rule added for port $AGENT_PORT"
fi

# Also add iptables rule directly (for Docker environments)
iptables -I INPUT 1 -p tcp --dport $AGENT_PORT -j ACCEPT 2>/dev/null || true

# ============================================================
# Step 9: Start the service
# ============================================================
log_info "Starting Node Agent..."

systemctl start decloud-node-agent

# Wait for startup
sleep 5

# Check if running
if systemctl is-active --quiet decloud-node-agent; then
    log_success "Node Agent is running"
else
    log_error "Node Agent failed to start. Check: journalctl -u decloud-node-agent -f"
    exit 1
fi

# ============================================================
# Step 10: Verify registration
# ============================================================
log_info "Verifying registration with orchestrator..."

sleep 10  # Wait for registration

# Check health endpoint
if curl -s http://localhost:$AGENT_PORT/ | grep -q "running"; then
    log_success "Node Agent API is responding"
else
    log_warn "Node Agent API not responding yet (may still be starting)"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Installation Complete!                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
log_success "DeCloud Node Agent installed successfully!"
echo ""
echo "  Node Agent API:  http://$(hostname -I | awk '{print $1}'):${AGENT_PORT}"
echo "  Swagger UI:      http://$(hostname -I | awk '{print $1}'):${AGENT_PORT}/swagger"
echo "  Orchestrator:    ${ORCHESTRATOR_URL}"
echo ""
echo "Useful commands:"
echo "  Status:          sudo systemctl status decloud-node-agent"
echo "  Logs:            sudo journalctl -u decloud-node-agent -f"
echo "  Restart:         sudo systemctl restart decloud-node-agent"
echo "  Stop:            sudo systemctl stop decloud-node-agent"
echo ""
echo "Configuration:     ${CONFIG_DIR}/appsettings.Production.json"
echo "Data directory:    ${DATA_DIR}"
echo ""

# Show node resources
log_info "Node resources:"
echo "  CPU Cores: $(nproc)"
echo "  Memory:    $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Storage:   $(df -h / | awk 'NR==2 {print $4}') available"
echo ""
