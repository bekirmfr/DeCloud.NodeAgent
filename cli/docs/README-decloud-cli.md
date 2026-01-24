# DeCloud CLI - Node Agent Management Tool

A unified, production-grade command-line interface for managing DeCloud node agent operations.

## Features

- 🔐 **Secure Authentication** - Wallet-based authentication with WalletConnect
- 📊 **Comprehensive Status** - Real-time node health, resources, and VM information
- 🖥️ **VM Management** - List, inspect, and clean up virtual machines
- 🔧 **Service Control** - Start, stop, restart the node agent service
- 🩺 **Built-in Diagnostics** - Automated health checks and troubleshooting
- 📝 **Detailed Logging** - Access and follow service logs
- ✨ **CLI Best Practices** - Inspired by industry-standard tools (docker, kubectl, systemctl)

## Quick Start

```bash
# Install the CLI
sudo cp decloud /usr/local/bin/decloud
sudo chmod +x /usr/local/bin/decloud

# Authenticate your node
sudo decloud login

# Check status
decloud status

# View help
decloud --help
```

## Installation

### Prerequisites

- Linux system (Ubuntu 22.04+ recommended)
- DeCloud node agent installed
- Root access (for some operations)

### Standard Installation

```bash
# Download the CLI
curl -o decloud https://raw.githubusercontent.com/your-repo/decloud-cli/main/decloud

# Make executable
chmod +x decloud

# Move to system path
sudo mv decloud /usr/local/bin/

# Verify installation
decloud version
```

### From Source

```bash
# Clone repository
git clone https://github.com/your-repo/decloud-cli.git
cd decloud-cli

# Install
sudo make install

# Or manually
sudo cp decloud /usr/local/bin/decloud
sudo chmod +x /usr/local/bin/decloud
```

## Commands

### Authentication Commands

#### `decloud login`
Authenticate the node using wallet signature (WalletConnect).

**Requires:** root/sudo

```bash
sudo decloud login
```

This will:
1. Generate a signing message
2. Display QR code for mobile wallet
3. Save authentication credentials
4. Enable the node to communicate with orchestrator

#### `decloud logout`
Remove node authentication and stop the service.

**Requires:** root/sudo

```bash
sudo decloud logout

# Skip confirmation
sudo decloud logout --force
```

### Information Commands

#### `decloud status`
Display comprehensive node status including authentication, service health, VMs, and network connectivity.

```bash
decloud status
```

**Output:**
- Authentication status (node ID, wallet, API key)
- Service status (running, uptime, memory usage)
- Health check (API responding)
- VM count (total, running)
- Network status (internet, orchestrator reachability)

#### `decloud info`
Display detailed node information from the API.

```bash
decloud info
```

**Requires:** authenticated node, running service

**Output:** Full JSON response with node details

#### `decloud resources`
Display detailed resource information (CPU, memory, storage, network).

```bash
decloud resources
```

**Requires:** authenticated node, running service

**Output:** Detailed JSON for each resource type

#### `decloud heartbeat`
Display the most recent heartbeat data sent to orchestrator.

```bash
decloud heartbeat
```

**Requires:** authenticated node, running service

### VM Management

#### `decloud vm list`
List all virtual machines on this node.

```bash
decloud vm list
```

**Output:** Table with VM ID, state, CPU, and memory

#### `decloud vm info <vm-id>`
Display detailed information about a specific VM.

```bash
decloud vm info abc-123-def-456
```

**Output:** Full JSON with VM configuration and status

#### `decloud vm cleanup <vm-id>`
Clean up a specific VM (stop, undefine, delete files).

**Requires:** root/sudo

```bash
# Clean up specific VM
sudo decloud vm cleanup abc-123-def-456

# Dry run (show what would be done)
sudo decloud vm cleanup abc-123-def-456 --dry-run

# Skip confirmation
sudo decloud vm cleanup abc-123-def-456 --force
```

#### `decloud vm cleanup --all`
Clean up ALL VMs on the node (with confirmation).

**Requires:** root/sudo

**⚠️ WARNING:** This is a destructive operation!

```bash
# Clean up all VMs (will prompt for confirmation)
sudo decloud vm cleanup --all

# Skip confirmation
sudo decloud vm cleanup --all --force

# Dry run
sudo decloud vm cleanup --all --dry-run
```

### Service Management

#### `decloud start`
Start the DeCloud node agent service.

**Requires:** root/sudo

```bash
sudo decloud start
```

#### `decloud stop`
Stop the DeCloud node agent service.

**Requires:** root/sudo

```bash
sudo decloud stop
```

#### `decloud restart`
Restart the DeCloud node agent service.

**Requires:** root/sudo

```bash
sudo decloud restart
```

#### `decloud logs`
View service logs from journalctl.

```bash
# Show last 50 lines (default)
decloud logs

# Show last 100 lines
decloud logs -n 100

# Follow logs in real-time
decloud logs -f

# Alternative syntax
decloud logs --lines 100
decloud logs --follow
```

### Diagnostic Commands

#### `decloud diagnose`
Run comprehensive diagnostics to identify issues.

```bash
decloud diagnose
```

**Checks:**
1. Service status
2. Authentication
3. Libvirt/virtualization
4. WireGuard networking
5. Network connectivity (internet, orchestrator)
6. Disk space
7. Recent errors

#### `decloud test-api`
Test node agent API endpoints.

```bash
decloud test-api

# Verbose output
VERBOSE=1 decloud test-api
```

**Tests:**
- `/health`
- `/api/node`
- `/api/node/cpu`
- `/api/node/memory`
- `/api/node/storage`

### Utility Commands

#### `decloud version`
Display CLI version.

```bash
decloud version
```

#### `decloud --help`
Display comprehensive help.

```bash
decloud --help
decloud -h
```

## Environment Variables

### `DEBUG`
Enable debug output for troubleshooting.

```bash
DEBUG=1 decloud status
```

### `NODE_AGENT_URL`
Override the default node agent API URL.

```bash
NODE_AGENT_URL="http://192.168.1.100:5050" decloud status
```

**Default:** `http://localhost:5050`

## Configuration Files

### `/etc/decloud/credentials`
Stores node authentication credentials (node ID, API key, wallet address).

**⚠️ Security:** This file contains sensitive information. Permissions: `600` (root only)

**Format:**
```
NODE_ID=abc-123-def-456
API_KEY=your-api-key-here
WALLET_ADDRESS=0x1234...
AUTHORIZED_AT=2025-01-15T10:30:00Z
```

### `/etc/decloud/pending-auth`
Temporary file used during authentication process.

### `/var/lib/libvirt/decloud-vms/`
VM storage directory (disk images, configs, cloud-init ISOs).

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Usage error (invalid command or options)
- `3` - Not authenticated (credentials missing)
- `4` - API error (node agent not responding)

## Examples

### Daily Operations

```bash
# Morning check
decloud status

# View what's happening
decloud logs -f

# Check resources
decloud resources
```

### Troubleshooting

```bash
# Node not working?
decloud diagnose

# Check recent errors
decloud logs -n 100

# Test API connectivity
decloud test-api

# Restart service
sudo decloud restart
```

### VM Management

```bash
# List VMs
decloud vm list

# Get details on specific VM
decloud vm info abc-123-def

# Clean up failed VM
sudo decloud vm cleanup abc-123-def
```

### Fresh Start

```bash
# Stop service
sudo decloud stop

# Clean up all VMs
sudo decloud vm cleanup --all

# Logout
sudo decloud logout

# Re-authenticate
sudo decloud login

# Start service
sudo decloud start

# Verify
decloud status
```

## Best Practices

### Security

1. **Never share credentials file** - Contains sensitive API keys
2. **Use sudo only when required** - Most info commands don't need root
3. **Verify authentication** - Always check `decloud status` after login

### Operations

1. **Check status regularly** - Use `decloud status` for health checks
2. **Monitor logs** - Use `decloud logs -f` during operations
3. **Run diagnostics** - Use `decloud diagnose` when issues arise
4. **Test before production** - Use `--dry-run` flags when available

### Automation

```bash
# Example monitoring script
#!/bin/bash
if ! decloud status | grep -q "Authenticated"; then
    echo "Node not authenticated!"
    exit 1
fi

if ! systemctl is-active --quiet decloud-node-agent; then
    echo "Service not running!"
    sudo decloud start
fi
```

## Integration with Other Tools

### SystemD

```bash
# Service status
systemctl status decloud-node-agent

# Enable auto-start
sudo systemctl enable decloud-node-agent

# View detailed logs
journalctl -u decloud-node-agent -f
```

### Monitoring

```bash
# Prometheus node exporter metrics
curl http://localhost:5050/metrics

# Health check for monitoring systems
curl http://localhost:5050/health
```

### Automation

```bash
# Cron job for status checks
*/5 * * * * /usr/local/bin/decloud status > /var/log/decloud/status.log 2>&1
```

## Troubleshooting

### Command Not Found

```bash
# Check if installed
which decloud

# Re-install
sudo cp decloud /usr/local/bin/decloud
sudo chmod +x /usr/local/bin/decloud
```

### Permission Denied

```bash
# Most commands that modify state need root
sudo decloud login
sudo decloud start
sudo decloud vm cleanup abc-123

# Info commands don't need root
decloud status
decloud info
decloud logs
```

### Service Not Running

```bash
# Start service
sudo decloud start

# Check logs for errors
decloud logs -n 50

# Run diagnostics
decloud diagnose
```

### API Not Responding

```bash
# Check service
sudo systemctl status decloud-node-agent

# Restart
sudo decloud restart

# Test API
decloud test-api
```

### Authentication Failed

```bash
# Re-authenticate
sudo decloud logout
sudo decloud login

# Verify credentials
cat /etc/decloud/credentials
```

## Development

### Testing

```bash
# Enable debug mode
DEBUG=1 decloud status

# Test API without running service
NODE_AGENT_URL="http://test-server:5050" decloud status
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style

- Follow [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
- Use `shellcheck` for linting
- Add tests for new features
- Update documentation

## Support

- 📖 Documentation: https://docs.decloud.io
- 💬 Discord: https://discord.gg/decloud
- 🐛 Issues: https://github.com/your-repo/decloud-cli/issues

## License

MIT License - See LICENSE file for details

## Changelog

### v1.0.0 (2025-01-24)
- Initial release
- Authentication commands (login, logout)
- Status and info commands
- VM management (list, info, cleanup)
- Service control (start, stop, restart, logs)
- Diagnostic commands (diagnose, test-api)
- Comprehensive help and documentation

## Acknowledgments

- Inspired by Docker, Kubernetes, and systemd CLI design
- Built for the DeCloud decentralized cloud platform
- Community feedback and contributions
