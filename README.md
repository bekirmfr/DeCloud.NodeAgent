# DeCloud Node Agent

A **cross-platform** C# .NET 8 node agent for a decentralized cloud computing platform. This agent runs on hardware provider nodes, managing VM lifecycle, resource discovery, and communication with the orchestration layer.

## Platform Support

| Feature | Windows | Linux |
|---------|---------|-------|
| Resource Discovery | ✅ Full (via WMI/PowerShell) | ✅ Full (via /proc, lscpu, etc.) |
| GPU Detection | ✅ NVIDIA + Generic | ✅ NVIDIA + Generic |
| WireGuard | ⚠️ Manual (via GUI) | ✅ Full automation |
| VM Management | ❌ Not supported | ✅ Full (KVM/libvirt) |
| API & Swagger | ✅ Full | ✅ Full |

**Windows:** Great for development and testing the API. Resource discovery works fully.  
**Linux:** Required for production VM hosting (KVM/QEMU/libvirt).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DeCloud Node Agent                        │
├─────────────────────────────────────────────────────────────┤
│  API Layer (ASP.NET Core)                                   │
│  ├── /api/vms      - VM lifecycle management                │
│  ├── /api/node     - Resource & health info                 │
│  └── Swagger UI    - API documentation                      │
├─────────────────────────────────────────────────────────────┤
│  Background Services                                         │
│  ├── HeartbeatService     - Reports status to orchestrator  │
│  └── CommandProcessorService - Executes orchestrator cmds   │
├─────────────────────────────────────────────────────────────┤
│  Core Services                                               │
│  ├── ResourceDiscoveryService - CPU, RAM, GPU, storage      │
│  ├── LibvirtVmManager        - KVM/QEMU via virsh           │
│  ├── ImageManager            - Base image caching           │
│  └── WireGuardNetworkManager - Overlay networking           │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure                                              │
│  ├── libvirt/KVM/QEMU   - Virtualization                    │
│  └── WireGuard          - Encrypted overlay network         │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
DeCloud.NodeAgent/
├── src/
│   ├── DeCloud.NodeAgent/              # Main application
│   │   ├── Controllers/                # REST API endpoints
│   │   ├── Services/                   # Background services
│   │   ├── Program.cs                  # Entry point & DI setup
│   │   └── appsettings.json           # Configuration
│   │
│   ├── DeCloud.NodeAgent.Core/         # Interfaces & models
│   │   ├── Interfaces/                 # Service contracts
│   │   └── Models/                     # Domain models
│   │
│   └── DeCloud.NodeAgent.Infrastructure/  # Implementations
│       ├── Services/                   # Core service impls
│       ├── Libvirt/                    # VM management
│       └── Network/                    # WireGuard management
│
└── tests/
    └── DeCloud.NodeAgent.Tests/        # Unit tests
```

## Prerequisites

### Windows (Development/Testing)

- .NET 8 SDK
- (Optional) WireGuard from https://www.wireguard.com/install/
- (Optional) NVIDIA drivers for GPU detection

```powershell
# Install .NET 8 SDK from https://dotnet.microsoft.com/download

# Build and run
cd decloud-node-agent
dotnet build
dotnet run --project src/DeCloud.NodeAgent
```

### Linux (Production - Full VM Support)

- Linux (Ubuntu 22.04+ recommended)
- .NET 8 SDK
- KVM/QEMU with libvirt
- WireGuard
- Root or sudo access (for VM/network management)

### Install Dependencies (Ubuntu)

```bash
# .NET 8 SDK
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt update
sudo apt install -y dotnet-sdk-8.0

# Virtualization
sudo apt install -y qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst

# WireGuard
sudo apt install -y wireguard wireguard-tools

# Cloud-init tools (for VM initialization)
sudo apt install -y cloud-image-utils genisoimage

# Add your user to required groups
sudo usermod -aG libvirt,kvm $USER

# Enable and start libvirt
sudo systemctl enable --now libvirtd

# Verify KVM
kvm-ok
virsh list --all
```

### Verify Setup

```bash
# Check libvirt
virsh version

# Check WireGuard
wg --version

# Check .NET
dotnet --version
```

## Building

```bash
cd DeCloud.NodeAgent

# Restore packages
dotnet restore

# Build
dotnet build

# Run
sudo dotnet run --project src/DeCloud.NodeAgent
```

> **Note**: Running as root/sudo is required for libvirt and WireGuard operations.

## Configuration

Edit `src/DeCloud.NodeAgent/appsettings.json`:

```json
{
  "Libvirt": {
    "VmStoragePath": "/var/lib/decloud/vms",
    "ImageCachePath": "/var/lib/decloud/images"
  },
  "WireGuard": {
    "InterfaceName": "wg0",
    "ListenPort": 51820,
    "Address": "10.100.0.1/24"  // Assign unique IP per node
  },
  "Heartbeat": {
    "Interval": "00:00:15",
    "OrchestratorUrl": "https://your-orchestrator.com"
  },
  "Orchestrator": {
    "BaseUrl": "https://your-orchestrator.com",
    "ApiKey": "your-api-key"
  }
}
```

### Storage Setup

```bash
# Create required directories
sudo mkdir -p /var/lib/decloud/{vms,images}
sudo mkdir -p /etc/decloud
sudo chown -R $USER:$USER /var/lib/decloud
```

## API Usage

Once running, access the API at `http://localhost:5100`:

### Swagger UI
Open `http://localhost:5100/swagger` in a browser.

### Example API Calls

```bash
# Get node resources
curl http://localhost:5100/api/node/resources

# Get resource snapshot
curl http://localhost:5100/api/node/snapshot

# List VMs
curl http://localhost:5100/api/vms

# Create a VM
curl -X POST http://localhost:5100/api/vms \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-vm",
    "vCpus": 2,
    "memoryBytes": 2147483648,
    "diskBytes": 10737418240,
    "baseImageUrl": "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img",
    "baseImageHash": "sha256:...",
    "sshPublicKey": "ssh-rsa AAAA... user@host"
  }'

# Start a VM
curl -X POST http://localhost:5100/api/vms/{vmId}/start

# Stop a VM
curl -X POST http://localhost:5100/api/vms/{vmId}/stop

# Delete a VM
curl -X DELETE http://localhost:5100/api/vms/{vmId}
```

## Testing Locally

### Download a Test Image

```bash
# Ubuntu 22.04 cloud image (small, ~600MB)
wget -P /var/lib/decloud/images \
  https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img

# Convert to qcow2 if needed
qemu-img convert -O qcow2 \
  /var/lib/decloud/images/jammy-server-cloudimg-amd64.img \
  /var/lib/decloud/images/ubuntu-22.04.qcow2
```

### Create a Test VM via API

```bash
curl -X POST http://localhost:5100/api/vms \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-vm",
    "vCpus": 1,
    "memoryBytes": 1073741824,
    "diskBytes": 5368709120,
    "baseImageUrl": "file:///var/lib/decloud/images/ubuntu-22.04.qcow2",
    "baseImageHash": "",
    "sshPublicKey": "'"$(cat ~/.ssh/id_rsa.pub)"'"
  }'
```

### Connect to VM Console

```bash
# Get VNC port from VM info
curl http://localhost:5100/api/vms/{vmId}

# Connect via VNC viewer
vncviewer localhost:5900
```

## Next Steps

This is the **node agent** component. To complete the platform, you'll also need:

1. **Orchestrator Service** - Central coordination, scheduling, API gateway
2. **Smart Contracts** - Node registration, payments, staking
3. **Frontend** - User dashboard for deploying VMs
4. **Monitoring** - Prometheus/Grafana for metrics

### Roadmap for Node Agent

- [ ] Add GPU passthrough support
- [ ] Implement VM live migration
- [ ] Add storage benchmarking
- [ ] Integrate with real orchestrator API
- [ ] Add TEE attestation support
- [ ] Implement resource metering for billing

## Troubleshooting

### Permission Denied on virsh

```bash
# Add user to libvirt group
sudo usermod -aG libvirt $USER
# Log out and back in

# Or run as root
sudo dotnet run --project src/DeCloud.NodeAgent
```

### WireGuard Interface Not Found

```bash
# Check if module is loaded
lsmod | grep wireguard

# Load module
sudo modprobe wireguard

# Create interface manually
sudo ip link add wg0 type wireguard
```

### Cannot Create VMs

```bash
# Check KVM support
kvm-ok

# Check libvirt status
sudo systemctl status libvirtd

# Check default network
virsh net-list --all
virsh net-start default
```

## License

MIT
