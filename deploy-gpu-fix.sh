#!/bin/bash
# Quick deployment script for GPU detection fix

set -e

echo "=========================================="
echo "  GPU Detection Fix - Deployment Script"
echo "=========================================="
echo ""

cd /opt/decloud/DeCloud.NodeAgent

echo "[1/4] Building project..."
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
echo "=========================================="
echo "  ✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Monitor logs with:"
echo "  sudo journalctl -u decloud-nodeagent -f"
echo ""
echo "Or check GPU detection specifically:"
echo "  sudo cat /var/log/decloud/nodeagent.log | grep -i 'gpu\\|nvidia'"
