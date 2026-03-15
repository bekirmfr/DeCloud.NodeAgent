#!/bin/bash
# GPU Proxy Diagnostic Script
# Run inside the VM to diagnose GPU proxy issues (gibberish output, etc.)
#
# Usage:
#   sudo bash diag-gpu-proxy.sh          # Enable diagnostics + show current state
#   sudo bash diag-gpu-proxy.sh collect   # Collect logs after reproducing the issue
#   sudo bash diag-gpu-proxy.sh disable   # Disable diagnostics
#
# Workflow:
#   1. Run: sudo bash diag-gpu-proxy.sh
#   2. Reproduce the issue (run Ollama inference)
#   3. Run: sudo bash diag-gpu-proxy.sh collect
#   4. Share /tmp/gpu-proxy-diag-report.txt

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DIAG_TRIGGER="/tmp/gpu-proxy-diag"
DIAG_LOG="/tmp/gpu-proxy-diag.log"
REPORT="/tmp/gpu-proxy-diag-report.txt"
ENV_FILE="/etc/decloud/gpu-proxy.env"
PROFILE_SCRIPT="/etc/profile.d/gpu-proxy.sh"

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

cmd_disable() {
    rm -f "$DIAG_TRIGGER"
    info "Diagnostics disabled. New processes won't log."
    info "Existing processes still log until restarted."
}

cmd_collect() {
    info "Collecting diagnostic report → $REPORT"
    {
        echo "=== GPU Proxy Diagnostic Report ==="
        echo "Date: $(date -Iseconds)"
        echo "Hostname: $(hostname)"
        echo "Kernel: $(uname -r)"
        echo ""

        echo "=== /etc/decloud/gpu-proxy.env ==="
        if [ -f "$ENV_FILE" ]; then
            # Redact token
            sed 's/\(DECLOUD_GPU_PROXY_TOKEN=\).*/\1<redacted>/' "$ENV_FILE"
        else
            echo "(file not found)"
        fi
        echo ""

        echo "=== /etc/profile.d/gpu-proxy.sh ==="
        if [ -f "$PROFILE_SCRIPT" ]; then
            cat "$PROFILE_SCRIPT"
        else
            echo "(file not found)"
        fi
        echo ""

        echo "=== Shim libraries ==="
        for lib in /usr/local/lib/libdecloud_cuda_shim.so \
                   /usr/local/lib/libcuda.so.1 \
                   /usr/local/lib/libnvidia-ml.so.1 \
                   /usr/local/lib/libcublas.so.12 \
                   /usr/local/lib/libcublasLt.so.12 \
                   /usr/local/lib/libcudnn.so.9 \
                   /usr/local/lib/libcuda_pytorch_stubs.so; do
            if [ -f "$lib" ]; then
                echo "  $lib: $(ls -la "$lib" | awk '{print $5, $6, $7, $8}') $(file "$lib" | grep -o 'ELF.*')"
            else
                echo "  $lib: NOT FOUND"
            fi
        done
        echo ""

        echo "=== LD_PRELOAD in running Ollama processes ==="
        for pid in $(pgrep -f ollama 2>/dev/null || true); do
            echo "  PID $pid ($(readlink /proc/$pid/exe 2>/dev/null || echo unknown)):"
            echo "    LD_PRELOAD=$(grep -z '^LD_PRELOAD=' /proc/$pid/environ 2>/dev/null | tr '\0' '\n' || echo '(cannot read)')"
            echo "    GGML_CUDA_DISABLE_GRAPHS=$(grep -z '^GGML_CUDA_DISABLE_GRAPHS=' /proc/$pid/environ 2>/dev/null | tr '\0' '\n' || echo '(not set)')"
            echo "    GGML_CUDA_FORCE_MMQ=$(grep -z '^GGML_CUDA_FORCE_MMQ=' /proc/$pid/environ 2>/dev/null | tr '\0' '\n' || echo '(not set)')"
            echo "    Loaded shims:"
            grep -l 'decloud\|cuda_shim\|libcuda\.so' /proc/$pid/maps 2>/dev/null | head -1 >/dev/null && \
                grep -E 'decloud|cuda_shim|libcuda\.so|libcublas|libcudart' /proc/$pid/maps 2>/dev/null | awk '{print "      "$NF}' | sort -u || \
                echo "      (cannot read maps)"
        done
        echo ""

        echo "=== Daemon status ==="
        if pgrep -f gpu-proxy-daemon >/dev/null 2>&1; then
            echo "  Running: $(pgrep -af gpu-proxy-daemon)"
        else
            echo "  NOT RUNNING"
        fi
        echo ""

        echo "=== Network connectivity ==="
        # Check vsock
        if [ -e /dev/vsock ]; then
            echo "  /dev/vsock: present"
        else
            echo "  /dev/vsock: NOT FOUND"
        fi
        # Check TCP to host
        timeout 2 bash -c 'echo >/dev/tcp/192.168.122.1/9999' 2>/dev/null && \
            echo "  TCP 192.168.122.1:9999: reachable" || \
            echo "  TCP 192.168.122.1:9999: unreachable"
        echo ""

        echo "=== Diagnostic log (last 200 lines) ==="
        if [ -f "$DIAG_LOG" ]; then
            echo "  Log size: $(wc -l < "$DIAG_LOG") lines, $(du -h "$DIAG_LOG" | cut -f1)"
            echo "---"
            tail -200 "$DIAG_LOG"
            echo "---"
        else
            echo "  (no diagnostic log found — did you enable diagnostics before reproducing?)"
        fi
        echo ""

        echo "=== Graph operation summary from diag log ==="
        if [ -f "$DIAG_LOG" ]; then
            echo "  BeginCapture calls:  $(grep -c 'cudaStreamBeginCapture' "$DIAG_LOG" 2>/dev/null || echo 0)"
            echo "  EndCapture calls:    $(grep -c 'cudaStreamEndCapture' "$DIAG_LOG" 2>/dev/null || echo 0)"
            echo "  GraphInstantiate:    $(grep -c 'cudaGraphInstantiate' "$DIAG_LOG" 2>/dev/null || echo 0)"
            echo "  GraphLaunch calls:   $(grep -c 'cudaGraphLaunch' "$DIAG_LOG" 2>/dev/null || echo 0)"
            echo "  GraphExecUpdate:     $(grep -c 'cudaGraphExecUpdate' "$DIAG_LOG" 2>/dev/null || echo 0)"
            echo "  graph-capture ops:   $(grep -c 'graph-capture: recorded op' "$DIAG_LOG" 2>/dev/null || echo 0)"
            echo "  RPC FAILED:          $(grep -c 'FAILED' "$DIAG_LOG" 2>/dev/null || echo 0)"
            echo "  UNKNOWN handle:      $(grep -c 'UNKNOWN handle' "$DIAG_LOG" 2>/dev/null || echo 0)"
            echo ""
            echo "  Destructor stats:"
            grep -A20 'DESTRUCTOR' "$DIAG_LOG" 2>/dev/null | tail -20 || echo "  (no destructor stats yet — process still running)"
        fi
        echo ""

        echo "=== Ollama server log (last 50 lines) ==="
        journalctl -u ollama --no-pager -n 50 2>/dev/null || echo "(journalctl not available or ollama not a systemd service)"
        echo ""

        echo "=== dmesg GPU-related (last 20 lines) ==="
        dmesg 2>/dev/null | grep -iE 'cuda|gpu|vsock|vhost' | tail -20 || echo "(no GPU-related kernel messages)"

    } > "$REPORT" 2>&1

    info "Report written to: $REPORT"
    info "Lines: $(wc -l < "$REPORT")"
    echo ""
    info "To view: cat $REPORT"
    info "To share: copy the file off the VM"
}

cmd_enable() {
    info "=== GPU Proxy Diagnostics ==="
    echo ""

    # Enable diagnostic trigger
    touch "$DIAG_TRIGGER"
    info "Diagnostic trigger created: $DIAG_TRIGGER"
    info "New processes will log to: $DIAG_LOG"
    echo ""

    # Clear old log
    if [ -f "$DIAG_LOG" ]; then
        mv "$DIAG_LOG" "${DIAG_LOG}.bak"
        info "Previous log backed up to ${DIAG_LOG}.bak"
    fi

    # Quick health check
    info "Quick health check:"

    # Check env file
    if [ -f "$ENV_FILE" ]; then
        info "  $ENV_FILE exists"
    else
        error "  $ENV_FILE MISSING — shim won't know how to connect!"
    fi

    # Check shim libraries
    if [ -f /usr/local/lib/libdecloud_cuda_shim.so ]; then
        info "  cuda_shim.so found"
    else
        error "  cuda_shim.so NOT FOUND"
    fi

    if [ -f /usr/local/lib/libcuda.so.1 ]; then
        info "  libcuda.so.1 (driver shim) found"
    else
        error "  libcuda.so.1 (driver shim) NOT FOUND"
    fi

    # Check Ollama
    if pgrep -f ollama >/dev/null 2>&1; then
        warn "  Ollama is already running. Restart it so the shim picks up diagnostics."
        warn "  Run: systemctl restart ollama  (or kill and restart manually)"
    else
        info "  Ollama not running (will pick up diagnostics on next start)"
    fi

    echo ""
    info "Next steps:"
    info "  1. Restart Ollama if it's already running"
    info "  2. Run a prompt that produces gibberish"
    info "  3. Run: sudo bash $0 collect"
}

case "${1:-}" in
    collect)  cmd_collect ;;
    disable)  cmd_disable ;;
    *)        cmd_enable ;;
esac
