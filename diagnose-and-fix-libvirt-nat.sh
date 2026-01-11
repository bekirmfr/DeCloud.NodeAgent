#!/bin/bash
#
# DeCloud Libvirt NAT Diagnostic & Fix Script
# Diagnoses and fixes VM internet connectivity issues
# NO CONSOLE ACCESS REQUIRED - All operations from host
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "DeCloud Libvirt NAT Diagnostic & Fix"
echo "========================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root (use sudo)${NC}"
    exit 1
fi

# =====================================================
# STEP 1: Check Current Network Configuration
# =====================================================
echo -e "${BLUE}📋 Step 1: Checking libvirt default network...${NC}"
echo "----------------------------------------"

if ! virsh net-info default &>/dev/null; then
    echo -e "${RED}❌ Default network doesn't exist${NC}"
    echo "   Creating it now..."
    virsh net-define /usr/share/libvirt/networks/default.xml || {
        echo -e "${RED}❌ Failed to define default network${NC}"
        exit 1
    }
fi

echo "Current network configuration:"
virsh net-dumpxml default

# Check if NAT is configured
if virsh net-dumpxml default | grep -q '<forward mode=.nat.>'; then
    echo -e "${GREEN}✓ NAT mode is configured${NC}"
    NAT_CONFIGURED=true
else
    echo -e "${RED}❌ NAT mode is NOT configured${NC}"
    NAT_CONFIGURED=false
fi

# Check if network is active
if virsh net-info default | grep -q "Active:.*yes"; then
    echo -e "${GREEN}✓ Network is active${NC}"
    NETWORK_ACTIVE=true
else
    echo -e "${RED}❌ Network is NOT active${NC}"
    NETWORK_ACTIVE=false
fi

# Check if network is set to autostart
if virsh net-info default | grep -q "Autostart:.*yes"; then
    echo -e "${GREEN}✓ Network autostart is enabled${NC}"
else
    echo -e "${YELLOW}⚠ Network autostart is disabled${NC}"
fi

echo ""

# =====================================================
# STEP 2: Check IP Forwarding
# =====================================================
echo -e "${BLUE}📋 Step 2: Checking IP forwarding...${NC}"
echo "----------------------------------------"

IPV4_FORWARD=$(sysctl -n net.ipv4.ip_forward)
if [ "$IPV4_FORWARD" = "1" ]; then
    echo -e "${GREEN}✓ IPv4 forwarding is enabled${NC}"
    IP_FORWARD_OK=true
else
    echo -e "${RED}❌ IPv4 forwarding is DISABLED${NC}"
    IP_FORWARD_OK=false
fi

echo ""

# =====================================================
# STEP 3: Check iptables NAT Rules
# =====================================================
echo -e "${BLUE}📋 Step 3: Checking iptables NAT rules...${NC}"
echo "----------------------------------------"

if iptables -t nat -L POSTROUTING -n | grep -q "MASQUERADE.*192.168.122.0/24"; then
    echo -e "${GREEN}✓ NAT MASQUERADE rule exists for VM network${NC}"
    NAT_RULE_OK=true
else
    echo -e "${RED}❌ NAT MASQUERADE rule is missing${NC}"
    NAT_RULE_OK=false
fi

echo ""
echo "Current POSTROUTING rules:"
iptables -t nat -L POSTROUTING -n -v | head -10

echo ""

# =====================================================
# STEP 4: Check Host Internet Connectivity
# =====================================================
echo -e "${BLUE}📋 Step 4: Checking host internet connectivity...${NC}"
echo "----------------------------------------"

if ping -c 2 8.8.8.8 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Host can reach internet (8.8.8.8)${NC}"
    HOST_INTERNET_OK=true
else
    echo -e "${RED}❌ Host cannot reach internet${NC}"
    echo "   Fix host internet connection first!"
    HOST_INTERNET_OK=false
fi

echo ""

# =====================================================
# STEP 5: Check virbr0 Interface
# =====================================================
echo -e "${BLUE}📋 Step 5: Checking virbr0 bridge interface...${NC}"
echo "----------------------------------------"

if ip link show virbr0 &>/dev/null; then
    echo -e "${GREEN}✓ virbr0 interface exists${NC}"
    VIRBR0_EXISTS=true
    
    # Check if it's up
    if ip link show virbr0 | grep -q "state UP"; then
        echo -e "${GREEN}✓ virbr0 is UP${NC}"
    else
        echo -e "${YELLOW}⚠ virbr0 is DOWN${NC}"
    fi
    
    # Show IP address
    echo "virbr0 IP address:"
    ip addr show virbr0 | grep "inet "
else
    echo -e "${RED}❌ virbr0 interface doesn't exist${NC}"
    VIRBR0_EXISTS=false
fi

echo ""

# =====================================================
# STEP 6: Test from Relay VM (if running)
# =====================================================
echo -e "${BLUE}📋 Step 6: Testing relay VM connectivity...${NC}"
echo "----------------------------------------"

RELAY_VM_ID=$(virsh list --all | grep relay | awk '{print $2}' | head -1)

if [ -n "$RELAY_VM_ID" ]; then
    echo "Found relay VM: $RELAY_VM_ID"
    
    # Get VM IP
    VM_IP=$(virsh domifaddr "$RELAY_VM_ID" | grep ipv4 | awk '{print $4}' | cut -d'/' -f1)
    
    if [ -n "$VM_IP" ]; then
        echo "Relay VM IP: $VM_IP"
        
        # Ping test
        echo -n "Testing ping to VM: "
        if ping -c 2 "$VM_IP" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Reachable${NC}"
        else
            echo -e "${RED}✗ Not reachable${NC}"
        fi
        
        # Try to test internet from VM (without console access)
        # We can check if DNS is working by querying port 53
        echo -n "Testing if VM can do DNS lookups: "
        if timeout 5 nc -zv "$VM_IP" 53 2>/dev/null | grep -q succeeded; then
            echo -e "${GREEN}✓ Port 53 accessible (DNS might work)${NC}"
        else
            echo -e "${YELLOW}⚠ Cannot verify (port 53 not responding)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Relay VM has no IP address yet${NC}"
    fi
else
    echo "No relay VM found (this is OK if you haven't created one yet)"
fi

echo ""

# =====================================================
# DIAGNOSIS SUMMARY
# =====================================================
echo "========================================"
echo -e "${BLUE}📊 DIAGNOSIS SUMMARY${NC}"
echo "========================================"
echo ""

ISSUES_FOUND=false

if [ "$NAT_CONFIGURED" != true ]; then
    echo -e "${RED}❌ NAT mode not configured in libvirt network${NC}"
    ISSUES_FOUND=true
fi

if [ "$NETWORK_ACTIVE" != true ]; then
    echo -e "${RED}❌ Network not active${NC}"
    ISSUES_FOUND=true
fi

if [ "$IP_FORWARD_OK" != true ]; then
    echo -e "${RED}❌ IP forwarding disabled${NC}"
    ISSUES_FOUND=true
fi

if [ "$NAT_RULE_OK" != true ]; then
    echo -e "${RED}❌ iptables NAT rules missing${NC}"
    ISSUES_FOUND=true
fi

if [ "$HOST_INTERNET_OK" != true ]; then
    echo -e "${RED}❌ Host has no internet connectivity${NC}"
    ISSUES_FOUND=true
fi

if [ "$VIRBR0_EXISTS" != true ]; then
    echo -e "${RED}❌ virbr0 bridge interface missing${NC}"
    ISSUES_FOUND=true
fi

if [ "$ISSUES_FOUND" != true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Your libvirt NAT configuration appears correct."
    echo "If VMs still can't access internet, the issue might be:"
    echo "  - Firewall rules blocking traffic"
    echo "  - DNS configuration in VMs"
    echo "  - Cloud-init timing issues"
    exit 0
fi

echo ""

# =====================================================
# OFFER TO FIX
# =====================================================
echo "========================================"
echo -e "${YELLOW}🔧 FIX AVAILABLE${NC}"
echo "========================================"
echo ""
echo "This script can automatically fix the issues found."
echo ""
read -p "Do you want to apply fixes now? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "Exiting without making changes."
    echo "To fix manually, run the commands shown above."
    exit 0
fi

# =====================================================
# APPLY FIXES
# =====================================================
echo "========================================"
echo -e "${GREEN}🔧 APPLYING FIXES${NC}"
echo "========================================"
echo ""

# Fix 1: Enable IP forwarding
if [ "$IP_FORWARD_OK" != true ]; then
    echo -e "${BLUE}Fix 1: Enabling IP forwarding...${NC}"
    sysctl -w net.ipv4.ip_forward=1
    if ! grep -q "^net.ipv4.ip_forward=1" /etc/sysctl.conf; then
        echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
    fi
    echo -e "${GREEN}✓ IP forwarding enabled${NC}"
    echo ""
fi

# Fix 2: Configure NAT in libvirt network
if [ "$NAT_CONFIGURED" != true ]; then
    echo -e "${BLUE}Fix 2: Reconfiguring libvirt network with NAT...${NC}"
    
    # Stop network
    virsh net-destroy default 2>/dev/null || true
    
    # Create proper NAT configuration
    cat > /tmp/default-network-nat.xml <<'EOF'
<network>
  <name>default</name>
  <bridge name='virbr0' stp='on' delay='0'/>
  <forward mode='nat'>
    <nat>
      <port start='1024' end='65535'/>
    </nat>
  </forward>
  <ip address='192.168.122.1' netmask='255.255.255.0'>
    <dhcp>
      <range start='192.168.122.2' end='192.168.122.254'/>
    </dhcp>
  </ip>
</network>
EOF
    
    # Undefine and redefine network
    virsh net-undefine default 2>/dev/null || true
    virsh net-define /tmp/default-network-nat.xml
    virsh net-autostart default
    virsh net-start default
    
    rm /tmp/default-network-nat.xml
    
    echo -e "${GREEN}✓ Network reconfigured with NAT${NC}"
    echo ""
fi

# Fix 3: Start network if not active
if [ "$NETWORK_ACTIVE" != true ]; then
    echo -e "${BLUE}Fix 3: Starting default network...${NC}"
    virsh net-start default || {
        echo -e "${YELLOW}⚠ Network might already be started${NC}"
    }
    echo -e "${GREEN}✓ Network started${NC}"
    echo ""
fi

# Fix 4: Add NAT iptables rule if missing
if [ "$NAT_RULE_OK" != true ]; then
    echo -e "${BLUE}Fix 4: Adding iptables NAT rules...${NC}"
    
    # Add MASQUERADE rule if it doesn't exist
    if ! iptables -t nat -C POSTROUTING -s 192.168.122.0/24 ! -d 192.168.122.0/24 -j MASQUERADE 2>/dev/null; then
        iptables -t nat -A POSTROUTING -s 192.168.122.0/24 ! -d 192.168.122.0/24 -j MASQUERADE
        echo -e "${GREEN}✓ NAT MASQUERADE rule added${NC}"
    else
        echo -e "${GREEN}✓ NAT rule already exists${NC}"
    fi
    
    # Try to make it persistent
    if command -v netfilter-persistent &> /dev/null; then
        netfilter-persistent save
        echo -e "${GREEN}✓ iptables rules saved (persistent)${NC}"
    elif command -v iptables-save &> /dev/null; then
        mkdir -p /etc/iptables
        iptables-save > /etc/iptables/rules.v4
        echo -e "${GREEN}✓ iptables rules saved to /etc/iptables/rules.v4${NC}"
    else
        echo -e "${YELLOW}⚠ Could not save iptables rules (will be lost on reboot)${NC}"
        echo "   Install iptables-persistent: apt-get install iptables-persistent"
    fi
    
    echo ""
fi

# =====================================================
# VERIFICATION
# =====================================================
echo "========================================"
echo -e "${BLUE}🔍 VERIFYING FIXES${NC}"
echo "========================================"
echo ""

sleep 2

echo "IP Forwarding: $(sysctl -n net.ipv4.ip_forward)"
echo ""

echo "NAT Mode:"
if virsh net-dumpxml default | grep -q '<forward mode=.nat.>'; then
    echo -e "${GREEN}✓ Configured${NC}"
else
    echo -e "${RED}✗ Still not configured${NC}"
fi
echo ""

echo "Network Active:"
if virsh net-info default | grep -q "Active:.*yes"; then
    echo -e "${GREEN}✓ Yes${NC}"
else
    echo -e "${RED}✗ No${NC}"
fi
echo ""

echo "NAT Rules:"
if iptables -t nat -L POSTROUTING -n | grep -q "MASQUERADE.*192.168.122.0/24"; then
    echo -e "${GREEN}✓ Present${NC}"
else
    echo -e "${RED}✗ Missing${NC}"
fi
echo ""

echo "virbr0 Interface:"
if ip link show virbr0 &>/dev/null; then
    echo -e "${GREEN}✓ Exists and $(ip link show virbr0 | grep -o 'state [A-Z]*')${NC}"
else
    echo -e "${RED}✗ Missing${NC}"
fi
echo ""

# =====================================================
# COMPLETION
# =====================================================
echo "========================================"
echo -e "${GREEN}✅ FIXES APPLIED${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Destroy and remove the failed relay VM:"
echo "   sudo virsh destroy f3c5a983-cc3a-4ccd-8594-421b5cfef159"
echo "   sudo virsh undefine f3c5a983-cc3a-4ccd-8594-421b5cfef159"
echo "   sudo rm -rf /var/lib/decloud/vms/f3c5a983-cc3a-4ccd-8594-421b5cfef159/"
echo ""
echo "2. Clean orchestrator database (on srv020184):"
echo "   mongosh"
echo "   use orchestrator"
echo "   db.vms.deleteOne({_id: 'f3c5a983-cc3a-4ccd-8594-421b5cfef159'})"
echo "   db.nodes.updateOne({_id: 'e9277b2c-614d-a8cb-6487-54395b6d7880'}, {\$set: {relayVmId: null}})"
echo "   exit"
echo ""
echo "3. Restart node agent to trigger new relay VM creation:"
echo "   sudo systemctl restart decloud-nodeagent"
echo ""
echo "4. Monitor the new relay VM creation:"
echo "   sudo tail -f /var/log/decloud/nodeagent.log | grep -E '(relay|CreateVm)'"
echo ""
echo "The new relay VM should now be able to access the internet!"
echo ""
