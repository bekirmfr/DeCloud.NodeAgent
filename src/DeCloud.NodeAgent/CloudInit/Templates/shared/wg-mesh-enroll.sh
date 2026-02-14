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
#

set -euo pipefail

LOG_TAG="wg-mesh-enroll"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$LOG_TAG] $*" | tee -a /var/log/wg-mesh-enroll.log; }
log_err() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$LOG_TAG] ERROR: $*" | tee -a /var/log/wg-mesh-enroll.log >&2; }

# ==================== Validate Config ====================
WG_INTERFACE="${WG_INTERFACE:-wg-mesh}"
WG_DESCRIPTION="${WG_DESCRIPTION:-vm-peer}"

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

REGISTER_PAYLOAD=$(cat <<EOF
{
    "public_key": "${PUBLIC_KEY}",
    "allowed_ips": "${TUNNEL_IP_BARE}/32",
    "description": "${WG_DESCRIPTION}"
}
EOF
)

log "Registering with relay API at ${WG_RELAY_API}/api/relay/add-peer..."

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
        log "Successfully registered with relay (attempt $attempt)"
        break
    elif [ "$attempt" -eq "$MAX_RETRIES" ]; then
        log_err "Failed to register with relay after $MAX_RETRIES attempts (HTTP $HTTP_CODE)"
        log_err "Response: $BODY"
        log "Continuing anyway — WireGuard may still work if relay adds us manually"
    else
        log "Registration attempt $attempt failed (HTTP $HTTP_CODE), retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    fi
done

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
