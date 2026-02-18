#!/bin/bash
#
# DeCloud WireGuard Mesh Enrollment (Building Block)
# Version: 1.0
#
# Any VM can use this script to join the relay's WireGuard mesh.
# It generates a keypair, registers with the relay, and starts
# the WireGuard interface.
#
# Required environment variables (set by cloud-init):
#   WG_RELAY_ENDPOINT  — Relay VM's WireGuard endpoint (ip:port)
#   WG_RELAY_PUBKEY    — Relay VM's WireGuard public key
#   WG_TUNNEL_IP       — This VM's assigned tunnel IP (10.20.x.y/24)
#   WG_RELAY_API       — Relay VM's API URL (http://ip:8080)
#
# Optional:
#   WG_INTERFACE       — Interface name (default: wg-mesh)
#   WG_DESCRIPTION     — Description for relay peer registry
#   WG_PEER_TYPE       — Peer classification: "system-vm" (default) or "cgnat-node"
#   WG_PARENT_NODE_ID  — Node ID that owns this VM (for grouping in dashboard)
#

set -euo pipefail

LOG_TAG="wg-mesh-enroll"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$LOG_TAG] $*" | tee -a /var/log/wg-mesh-enroll.log; }
log_err() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$LOG_TAG] ERROR: $*" | tee -a /var/log/wg-mesh-enroll.log >&2; }

# ==================== Source Environment ====================
# Load WG mesh config from env file (cloud-init writes this).
# The runcmd may not export vars to child processes, so we source here.
ENV_FILE="/etc/decloud/wg-mesh.env"
if [ -f "$ENV_FILE" ]; then
    set -a  # auto-export all sourced vars
    source "$ENV_FILE"
    set +a
    log "Loaded environment from $ENV_FILE"
else
    log_err "Environment file not found: $ENV_FILE"
    exit 1
fi

# ==================== Validate Config ====================
WG_INTERFACE="${WG_INTERFACE:-wg-mesh}"
WG_DESCRIPTION="${WG_DESCRIPTION:-vm-peer}"
WG_PEER_TYPE="${WG_PEER_TYPE:-system-vm}"
WG_PARENT_NODE_ID="${WG_PARENT_NODE_ID:-}"

for var in WG_RELAY_ENDPOINT WG_RELAY_PUBKEY WG_TUNNEL_IP WG_RELAY_API; do
    if [ -z "${!var:-}" ]; then
        log_err "Missing required env: $var"
        exit 1
    fi
done

log "Starting WireGuard mesh enrollment"
log "  Relay endpoint: $WG_RELAY_ENDPOINT"
log "  Relay pubkey:   ${WG_RELAY_PUBKEY:0:16}..."
log "  Tunnel IP:      $WG_TUNNEL_IP"
log "  Relay API:      $WG_RELAY_API"
log "  Interface:      $WG_INTERFACE"

# ==================== Idempotency Guard ====================
# If the WG interface is already up and has a peer with a handshake,
# skip re-enrollment to prevent duplicate peer registration on the relay.
if wg show "$WG_INTERFACE" 2>/dev/null | grep -q "latest handshake"; then
    log "WireGuard interface ${WG_INTERFACE} already active with handshake — skipping enrollment"
    exit 0
fi

# ==================== Generate Keypair ====================
PRIVATE_KEY=$(wg genkey)
PUBLIC_KEY=$(echo "$PRIVATE_KEY" | wg pubkey)

log "Generated WireGuard keypair (pubkey: ${PUBLIC_KEY:0:16}...)"

# ==================== Extract IP without CIDR ====================
# WG_TUNNEL_IP may be "10.20.1.253/24" or "10.20.1.253"
TUNNEL_IP_BARE="${WG_TUNNEL_IP%%/*}"

# ==================== Write WireGuard Config ====================
mkdir -p /etc/wireguard

cat > "/etc/wireguard/${WG_INTERFACE}.conf" <<EOF
[Interface]
PrivateKey = ${PRIVATE_KEY}
Address = ${WG_TUNNEL_IP}

[Peer]
PublicKey = ${WG_RELAY_PUBKEY}
Endpoint = ${WG_RELAY_ENDPOINT}
AllowedIPs = 10.20.0.0/16
PersistentKeepalive = 25
EOF

chmod 600 "/etc/wireguard/${WG_INTERFACE}.conf"
log "Wrote WireGuard config to /etc/wireguard/${WG_INTERFACE}.conf"

# ==================== Register with Relay API ====================
# The relay needs to know our public key and allowed IPs so it routes
# traffic to us through the mesh.
#
# The relay API (port 8080) runs inside the relay VM. From this VM,
# port 8080 on the relay's public IP may not be reachable (only WireGuard
# UDP/51820 is NAT-forwarded from the host). So we first try the NodeAgent
# proxy on the host (reachable via virbr0 default gateway on port 5100),
# then fall back to the direct relay API URL.

REGISTER_PAYLOAD=$(cat <<EOF
{
    "public_key": "${PUBLIC_KEY}",
    "allowed_ips": "${TUNNEL_IP_BARE}/32",
    "description": "${WG_DESCRIPTION}",
    "peer_type": "${WG_PEER_TYPE}",
    "parent_node_id": "${WG_PARENT_NODE_ID}"
}
EOF
)

# Discover NodeAgent on host via default gateway (virbr0)
GATEWAY=$(ip route | awk '/default/ {print $3; exit}')
NODEAGENT_API="http://${GATEWAY}:5100"

REGISTERED=false

# Strategy 1: Try NodeAgent proxy (host can reach relay VM on bridge network)
if [ -n "$GATEWAY" ]; then
    log "Trying NodeAgent proxy at ${NODEAGENT_API}/api/relay/wg-mesh-enroll..."

    for attempt in $(seq 1 3); do
        RESPONSE=$(curl -s -w "\n%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -d "${REGISTER_PAYLOAD}" \
            "${NODEAGENT_API}/api/relay/wg-mesh-enroll" \
            --connect-timeout 5 \
            --max-time 10 \
            2>/dev/null || true)

        HTTP_CODE=$(echo "$RESPONSE" | tail -1)
        BODY=$(echo "$RESPONSE" | head -n -1)

        if [ "$HTTP_CODE" = "200" ]; then
            log "Successfully registered via NodeAgent proxy (attempt $attempt)"
            REGISTERED=true
            break
        else
            log "NodeAgent proxy attempt $attempt failed (HTTP $HTTP_CODE)"
            sleep 2
        fi
    done
fi

# Strategy 2: Fall back to direct relay API
if [ "$REGISTERED" = "false" ]; then
    log "Trying direct relay API at ${WG_RELAY_API}/api/relay/add-peer..."

    MAX_RETRIES=10
    RETRY_DELAY=5

    for attempt in $(seq 1 $MAX_RETRIES); do
        RESPONSE=$(curl -s -w "\n%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -d "${REGISTER_PAYLOAD}" \
            "${WG_RELAY_API}/api/relay/add-peer" \
            --connect-timeout 5 \
            --max-time 10 \
            2>/dev/null || true)

        HTTP_CODE=$(echo "$RESPONSE" | tail -1)
        BODY=$(echo "$RESPONSE" | head -n -1)

        if [ "$HTTP_CODE" = "200" ]; then
            log "Successfully registered with relay directly (attempt $attempt)"
            REGISTERED=true
            break
        elif [ "$attempt" -eq "$MAX_RETRIES" ]; then
            log_err "Failed to register with relay after $MAX_RETRIES attempts (HTTP $HTTP_CODE)"
            log_err "Response: $BODY"
            log "Continuing anyway — WireGuard may still work if relay adds us manually"
        else
            log "Direct registration attempt $attempt failed (HTTP $HTTP_CODE), retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        fi
    done
fi

# ==================== Start WireGuard Interface ====================
log "Starting WireGuard interface ${WG_INTERFACE}..."

# Enable and start via wg-quick
systemctl enable "wg-quick@${WG_INTERFACE}.service" 2>/dev/null || true
wg-quick up "$WG_INTERFACE" 2>/dev/null || {
    log "wg-quick up failed (might already be running), trying restart..."
    wg-quick down "$WG_INTERFACE" 2>/dev/null || true
    wg-quick up "$WG_INTERFACE"
}

# ==================== Verify Connectivity ====================
sleep 2

# Extract relay gateway IP from endpoint (first IP in subnet, .254 is gateway)
RELAY_GW=$(echo "$WG_RELAY_ENDPOINT" | cut -d: -f1)

# Check interface is up
if wg show "$WG_INTERFACE" > /dev/null 2>&1; then
    log "✅ WireGuard interface ${WG_INTERFACE} is UP"
    log "  Local IP:  ${WG_TUNNEL_IP}"
    log "  Public key: ${PUBLIC_KEY}"

    # Try to ping relay gateway over tunnel
    TUNNEL_RELAY_IP=$(echo "$TUNNEL_IP_BARE" | sed 's/\.[0-9]*$/.254/')
    if ping -c 1 -W 3 "$TUNNEL_RELAY_IP" > /dev/null 2>&1; then
        log "✅ Relay gateway ${TUNNEL_RELAY_IP} reachable via tunnel"
    else
        log "⚠️  Relay gateway ${TUNNEL_RELAY_IP} not reachable yet (may take a few seconds)"
    fi
else
    log_err "WireGuard interface ${WG_INTERFACE} failed to come up"
    exit 1
fi

log "WireGuard mesh enrollment complete"
