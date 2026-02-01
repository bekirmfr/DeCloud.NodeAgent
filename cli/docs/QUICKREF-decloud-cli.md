# DeCloud CLI - Quick Reference Guide

## Command Summary

### Authentication
| Command | Description | Requires Root |
|---------|-------------|---------------|
| `decloud login` | Authenticate with wallet | ✅ |
| `decloud logout` | Remove authentication | ✅ |

### Information
| Command | Description | Requires Root |
|---------|-------------|---------------|
| `decloud status` | Show comprehensive status | ❌ |
| `decloud info` | Show detailed node info | ❌ |
| `decloud resources` | Show resource details | ❌ |
| `decloud heartbeat` | Show last heartbeat | ❌ |

### VM Management
| Command | Description | Requires Root |
|---------|-------------|---------------|
| `decloud vm list` | List all VMs | ❌ |
| `decloud vm info <id>` | Show VM details | ❌ |
| `decloud vm cleanup <id>` | Clean up VM | ✅ |
| `decloud vm cleanup --all` | Clean up all VMs | ✅ |

### Service Control
| Command | Description | Requires Root |
|---------|-------------|---------------|
| `decloud start` | Start service | ✅ |
| `decloud stop` | Stop service | ✅ |
| `decloud restart` | Restart service | ✅ |
| `decloud logs` | View logs | ❌ |
| `decloud log clear` | Clear all logs | ✅ |
| `decloud log clear --before-last-start` | Clear old logs only | ✅ |

### Diagnostics
| Command | Description | Requires Root |
|---------|-------------|---------------|
| `decloud diagnose` | Run diagnostics | ❌ |
| `decloud test-api` | Test API endpoints | ❌ |

## Common Workflows

### Initial Setup

```bash
# 1. Install the CLI
sudo cp decloud /usr/local/bin/
sudo chmod +x /usr/local/bin/decloud

# 2. Verify installation
decloud --version

# 3. Authenticate
sudo decloud login
# Follow prompts, scan QR code with wallet

# 4. Verify authentication
decloud status

# 5. Check resources
decloud resources
```

### Daily Operations

```bash
# Morning health check
decloud status

# View active VMs
decloud vm list

# Monitor in real-time
decloud logs -f

# Check resource usage
decloud resources
```

### Troubleshooting

```bash
# Node not responding?
decloud diagnose

# Check recent errors
decloud logs -n 100

# Restart service
sudo decloud restart

# Wait a moment, then verify
sleep 5
decloud status

# Still issues? Deep dive
decloud test-api
journalctl -u decloud-node-agent -n 200
```

### VM Management

```bash
# List all VMs
decloud vm list

# Get details on specific VM
decloud vm info abc-123-def-456

# VM stuck? Clean it up
sudo decloud vm cleanup abc-123-def-456

# Dry run first to see what would happen
sudo decloud vm cleanup abc-123-def-456 --dry-run

# Force cleanup without confirmation
sudo decloud vm cleanup abc-123-def-456 --force
```

### Maintenance

```bash
# Stop service for maintenance
sudo decloud stop

# Clean up all VMs (DESTRUCTIVE!)
sudo decloud vm cleanup --all

# Clear old logs to free up space
sudo decloud log clear --before-last-start

# Restart service
sudo decloud start

# Verify everything is working
decloud status
decloud diagnose
```

### Re-authentication

```bash
# Something wrong with auth?
sudo decloud logout

# Re-authenticate
sudo decloud login

# Restart service to use new credentials
sudo decloud restart

# Verify
decloud status
```

## Practical Examples

### Example 1: New Node Setup

```bash
#!/bin/bash
# setup-node.sh - Complete node setup script

set -e

echo "Setting up DeCloud node..."

# Install CLI
sudo cp decloud /usr/local/bin/
sudo chmod +x /usr/local/bin/decloud

# Authenticate
sudo decloud login

# Start service
sudo decloud start

# Wait for service to stabilize
sleep 5

# Verify everything
decloud status
decloud diagnose

echo "✓ Node setup complete!"
```

### Example 2: Health Monitoring Script

```bash
#!/bin/bash
# monitor-node.sh - Continuous health monitoring

while true; do
    clear
    echo "════════════════════════════════════════"
    echo "  DeCloud Node Monitor"
    echo "  $(date)"
    echo "════════════════════════════════════════"
    echo ""
    
    decloud status
    
    echo ""
    echo "Press Ctrl+C to exit"
    
    sleep 30
done
```

### Example 3: Automated Cleanup

```bash
#!/bin/bash
# cleanup-failed-vms.sh - Clean up failed VMs

echo "Finding failed VMs..."

# Get VM list and filter for failed state
failed_vms=$(decloud vm list | grep -i "failed" | awk '{print $1}')

if [ -z "$failed_vms" ]; then
    echo "No failed VMs found"
    exit 0
fi

echo "Found failed VMs:"
echo "$failed_vms"
echo ""

read -p "Clean up these VMs? (y/N): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled"
    exit 0
fi

# Clean up each failed VM
for vm_id in $failed_vms; do
    echo "Cleaning up: $vm_id"
    sudo decloud vm cleanup "$vm_id" --force
done

echo "✓ Cleanup complete"
```

### Example 4: Daily Report

```bash
#!/bin/bash
# daily-report.sh - Generate daily node report

REPORT_FILE="/var/log/decloud/daily-report-$(date +%Y%m%d).txt"

{
    echo "════════════════════════════════════════"
    echo "  DeCloud Node Daily Report"
    echo "  $(date)"
    echo "════════════════════════════════════════"
    echo ""
    
    echo "STATUS:"
    decloud status
    echo ""
    
    echo "RESOURCES:"
    decloud resources
    echo ""
    
    echo "VMs:"
    decloud vm list
    echo ""
    
    echo "RECENT ERRORS:"
    journalctl -u decloud-node-agent --since yesterday -p err --no-pager
    echo ""
    
    echo "════════════════════════════════════════"
} > "$REPORT_FILE"

echo "Report saved to: $REPORT_FILE"
```

### Example 5: Pre-flight Check

```bash
#!/bin/bash
# preflight.sh - Run before important operations

echo "Running pre-flight checks..."

# 1. Check authentication
if ! decloud status | grep -q "Authenticated"; then
    echo "❌ Node not authenticated"
    exit 1
fi
echo "✓ Authentication OK"

# 2. Check service
if ! systemctl is-active --quiet decloud-node-agent; then
    echo "❌ Service not running"
    exit 1
fi
echo "✓ Service OK"

# 3. Check disk space
usage=$(df -h /var/lib/libvirt/decloud-vms | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$usage" -gt 90 ]; then
    echo "⚠️  Disk usage high: ${usage}%"
fi
echo "✓ Disk space OK (${usage}% used)"

# 4. Check connectivity
if ! decloud heartbeat | jq -e '.heartbeat.orchestratorReachable == true' >/dev/null 2>&1; then
    echo "❌ Cannot reach orchestrator"
    exit 1
fi
echo "✓ Orchestrator reachable"

# 5. Run diagnostics
echo ""
echo "Running diagnostics..."
decloud diagnose

echo ""
echo "✓ All pre-flight checks passed"
```

## Tips & Tricks

### 1. Use Aliases

Add to `~/.bashrc`:
```bash
alias ds='decloud status'
alias dl='decloud logs -f'
alias dv='decloud vm list'
alias dr='sudo decloud restart'
```

### 2. Watch Mode

Monitor status in real-time:
```bash
watch -n 5 decloud status
```

### 3. Quick Logs

Last 50 lines:
```bash
decloud logs
```

Follow logs:
```bash
decloud logs -f
```

Errors only:
```bash
journalctl -u decloud-node-agent -p err -f
```

Clear all logs:
```bash
sudo decloud log clear
```

Clear only old logs (keep recent):
```bash
sudo decloud log clear --before-last-start
```

### 4. JSON Processing

Get just the node ID:
```bash
decloud info | jq -r '.nodeId'
```

Count running VMs:
```bash
decloud vm list | jq -r '[.[] | select(.state == "Running")] | length'
```

### 5. Debug Mode

Enable debug output:
```bash
DEBUG=1 decloud status
```

### 6. Custom API URL

For testing:
```bash
NODE_AGENT_URL="http://test-server:5050" decloud status
```

### 7. Output to File

Save status:
```bash
decloud status > /tmp/node-status.txt
```

Save VM list as JSON:
```bash
decloud vm list > vms.json
```

### 8. Cron Jobs

Health check every 5 minutes:
```bash
*/5 * * * * /usr/local/bin/decloud status >> /var/log/decloud/health.log 2>&1
```

Daily report at 6 AM:
```bash
0 6 * * * /usr/local/bin/daily-report.sh
```

## Common Issues & Solutions

### Issue: Command not found

**Solution:**
```bash
sudo cp decloud /usr/local/bin/
sudo chmod +x /usr/local/bin/decloud
```

### Issue: Permission denied

**Solution:**
Most state-changing commands need root:
```bash
sudo decloud login
sudo decloud restart
sudo decloud vm cleanup <id>
```

### Issue: Service not running

**Solution:**
```bash
sudo decloud start
decloud logs
```

### Issue: API not responding

**Solution:**
```bash
sudo decloud restart
sleep 5
decloud test-api
```

### Issue: Authentication failed

**Solution:**
```bash
sudo decloud logout
sudo decloud login
sudo decloud restart
```

### Issue: VM stuck in weird state

**Solution:**
```bash
# Force cleanup
sudo decloud vm cleanup <vm-id> --force

# If that doesn't work, manual cleanup
sudo virsh destroy <vm-id>
sudo virsh undefine <vm-id> --remove-all-storage
```

## Keyboard Shortcuts

When using `decloud logs -f`:
- `Ctrl+C` - Stop following
- `q` - Quit (if using pager)

## Exit Codes Reference

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | Command completed successfully |
| 1 | Error | Service failed to start |
| 2 | Usage error | Missing required argument |
| 3 | Not authenticated | Credentials missing |
| 4 | API error | Node agent not responding |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `0` | Enable debug output (`DEBUG=1`) |
| `NODE_AGENT_URL` | `http://localhost:5050` | Node agent API URL |

## File Locations

| Path | Description |
|------|-------------|
| `/usr/local/bin/decloud` | CLI executable |
| `/etc/decloud/credentials` | Authentication credentials |
| `/etc/decloud/pending-auth` | Temporary auth file |
| `/var/lib/libvirt/decloud-vms/` | VM storage |
| `/var/log/decloud/` | Log directory |

## Getting Help

1. **Built-in help:**
   ```bash
   decloud --help
   decloud <command> --help
   ```

2. **Documentation:**
   - README: `/usr/local/share/doc/decloud/README.md`
   - Online: https://docs.decloud.io

3. **Logs:**
   ```bash
   decloud logs
   journalctl -u decloud-node-agent
   ```

4. **Diagnostics:**
   ```bash
   decloud diagnose
   ```

5. **Community:**
   - Discord: https://discord.gg/decloud
   - GitHub: https://github.com/decloud/issues

## Quick Command Reference Card

```
┌─────────────────────────────────────────────────────┐
│              DeCloud CLI Quick Reference            │
├─────────────────────────────────────────────────────┤
│ Authentication                                      │
│   sudo decloud login        - Authenticate          │
│   sudo decloud logout       - Remove auth           │
│                                                     │
│ Status                                              │
│   decloud status            - Show status           │
│   decloud info              - Node details          │
│   decloud resources         - Resource info         │
│                                                     │
│ VMs                                                 │
│   decloud vm list           - List VMs              │
│   decloud vm info <id>      - VM details            │
│   sudo decloud vm cleanup   - Clean up VM           │
│                                                     │
│ Service                                             │
│   sudo decloud restart      - Restart service       │
│   decloud logs -f           - Follow logs           │
│                                                     │
│ Diagnostics                                         │
│   decloud diagnose          - Health check          │
│   decloud test-api          - Test API              │
└─────────────────────────────────────────────────────┘
```

**Save this card:** Print or save to `~/decloud-quick-ref.txt` for easy access!
