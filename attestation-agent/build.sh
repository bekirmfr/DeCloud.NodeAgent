#!/bin/bash
# ============================================================
# DeCloud Attestation Agent Build Script
# ============================================================
#
# This script builds the Go attestation agent for multiple
# architectures and prepares it for inclusion in cloud-init.
#
# Usage:
#   ./build.sh              # Build for current architecture
#   ./build.sh all          # Build for all supported architectures
#   ./build.sh install      # Build and install to CloudInit/Templates
#
# Output:
#   bin/decloud-agent-amd64     # x86_64 binary
#   bin/decloud-agent-arm64     # ARM64 binary
#   bin/decloud-agent.b64       # Base64-encoded default binary
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Go is installed
check_go() {
    if ! command -v go &> /dev/null; then
        log_error "Go is not installed. Please install Go 1.21+ first."
        echo ""
        echo "Installation options:"
        echo "  Ubuntu/Debian: sudo apt install golang-go"
        echo "  macOS:         brew install go"
        echo "  Manual:        https://go.dev/dl/"
        exit 1
    fi
    
    GO_VERSION=$(go version | grep -oP 'go\K[0-9]+\.[0-9]+')
    log_info "Go version: $GO_VERSION"
}

# Initialize Go module if needed
init_module() {
    if [ ! -f "go.mod" ]; then
        log_info "Initializing Go module..."
        go mod init decloud-attestation-agent
    fi
}

# Build for a specific OS/ARCH
build_for() {
    local GOOS=$1
    local GOARCH=$2
    local OUTPUT=$3
    
    log_info "Building for ${GOOS}/${GOARCH}..."
    
    CGO_ENABLED=0 GOOS=$GOOS GOARCH=$GOARCH go build \
        -ldflags="-s -w -X main.Version=$(git describe --tags --always 2>/dev/null || echo 'dev')" \
        -trimpath \
        -o "$OUTPUT" \
        ./main.go
    
    # Get file size
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    log_info "  → $OUTPUT ($SIZE)"
}

# Build all architectures
build_all() {
    mkdir -p bin
    
    # Linux AMD64 (most common for servers)
    build_for linux amd64 bin/decloud-agent-amd64
    
    # Linux ARM64 (Raspberry Pi, ARM servers, Apple Silicon VMs)
    build_for linux arm64 bin/decloud-agent-arm64
    
    # Create symlink for default (amd64)
    ln -sf decloud-agent-amd64 bin/decloud-agent
    
    log_info "All builds complete!"
}

# Build for current architecture only
build_current() {
    mkdir -p bin
    
    local CURRENT_ARCH=$(uname -m)
    local GOARCH="amd64"
    
    case $CURRENT_ARCH in
        x86_64)  GOARCH="amd64" ;;
        aarch64) GOARCH="arm64" ;;
        arm64)   GOARCH="arm64" ;;
        *)       log_warn "Unknown architecture: $CURRENT_ARCH, defaulting to amd64" ;;
    esac
    
    build_for linux $GOARCH "bin/decloud-agent-${GOARCH}"
    ln -sf "decloud-agent-${GOARCH}" bin/decloud-agent
}

# Create base64-encoded version for cloud-init
create_base64() {
    local ARCH=${1:-amd64}
    local BINARY="bin/decloud-agent-${ARCH}"
    local OUTPUT="bin/decloud-agent-${ARCH}.b64"
    
    if [ ! -f "$BINARY" ]; then
        log_error "Binary not found: $BINARY"
        log_error "Run './build.sh' first"
        exit 1
    fi
    
    log_info "Creating base64-encoded version for ${ARCH}..."
    base64 -w 0 "$BINARY" > "$OUTPUT"
    
    # Also create a default one
    cp "$OUTPUT" "bin/decloud-agent.b64"
    
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    log_info "  → $OUTPUT ($SIZE)"
}

# Install to CloudInit/Templates directory
install_to_cloudinit() {
    local CLOUDINIT_DIR="../src/DeCloud.NodeAgent/CloudInit/Templates"
    
    if [ ! -d "$CLOUDINIT_DIR" ]; then
        log_error "CloudInit templates directory not found at: $CLOUDINIT_DIR"
        log_error "Run this script from the attestation-agent directory"
        exit 1
    fi
    
    log_info "Installing attestation agent to CloudInit/Templates..."
    
    # Copy base64 files for each architecture
    for arch in amd64 arm64; do
        if [ -f "bin/decloud-agent-${arch}.b64" ]; then
            cp "bin/decloud-agent-${arch}.b64" "$CLOUDINIT_DIR/"
            log_info "  → Copied decloud-agent-${arch}.b64"
        fi
    done
    
    # Copy default
    if [ -f "bin/decloud-agent.b64" ]; then
        cp "bin/decloud-agent.b64" "$CLOUDINIT_DIR/"
        log_info "  → Copied decloud-agent.b64 (default)"
    fi
    
    log_info "Installation complete!"
    echo ""
    log_info "The attestation agent will be included in the next Node Agent build."
}

# Show usage
show_usage() {
    echo "DeCloud Attestation Agent Build Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  (none)    Build for current architecture"
    echo "  all       Build for all supported architectures (amd64, arm64)"
    echo "  base64    Create base64-encoded binaries for cloud-init"
    echo "  install   Build, encode, and install to CloudInit/Templates"
    echo "  clean     Remove build artifacts"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Quick build for testing"
    echo "  $0 all          # Build all architectures"
    echo "  $0 install      # Full build and install to Node Agent"
}

# Clean build artifacts
clean() {
    log_info "Cleaning build artifacts..."
    rm -rf bin/
    log_info "Clean complete"
}

# Main
main() {
    echo "=============================================="
    echo "  DeCloud Attestation Agent Builder"
    echo "=============================================="
    echo ""
    
    check_go
    init_module
    
    case "${1:-}" in
        all)
            build_all
            create_base64 amd64
            create_base64 arm64
            ;;
        base64)
            create_base64 amd64
            create_base64 arm64
            ;;
        install)
            build_all
            create_base64 amd64
            create_base64 arm64
            install_to_cloudinit
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            build_current
            create_base64
            ;;
    esac
    
    echo ""
    log_info "Done!"
}

main "$@"
