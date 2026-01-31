#!/bin/bash
# NAT Rule Checking Optimization - Deployment Script

set -e

echo "=============================================="
echo "  NAT Check Optimization Deployment"
echo "=============================================="
echo ""
echo "Changes:"
echo "  ✓ Reduced NAT checking from every 1 min to every 10 min"
echo "  ✓ Per-VM caching to avoid redundant checks"
echo "  ✓ Use comprehensive check (all 3 rules)"
echo "  ✓ Reduced log noise (Debug/Trace instead of Warning)"
echo "  ✓ 90% reduction in iptables overhead"
echo ""

cd /opt/decloud/DeCloud.NodeAgent

echo "[1/4] Building NodeAgent..."
dotnet build -c Release

echo ""
echo "[2/4] Stopping service..."
sudo systemctl stop decloud-nodeagent

echo ""
echo "[3/4] Installing..."
sudo ./install.sh

echo ""
echo "[4/4] Starting service..."
sudo systemctl start decloud-nodeagent

echo ""
echo "=============================================="
echo "  ✓ Deployment Complete!"
echo "=============================================="
echo ""
echo "Monitor NAT checking (should see checks only every 10 min):"
echo "  sudo journalctl -u decloud-nodeagent -f | grep -i nat"
echo ""
echo "Verify reduced log noise:"
echo "  sudo journalctl -u decloud-nodeagent --since '10 minutes ago' | grep NAT | wc -l"
echo ""
echo "Before: ~10 NAT log lines per minute"
echo "After:  ~1 NAT log line per 10 minutes"
