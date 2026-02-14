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
#   WG_TUNNEL_IP       — This VM's assigned tunnel IP (10.20.x.y)
#   WG_RELAY_API       — Relay VM's API URL (http://ip:8080)
#
# Optional:
#   WG_INTERFACE       — Interface name (default: wg-mesh)
#   WG_DESCRIPTION     — Description for relay peer registry
#

set -e

LOG_TAG="wg-mesh-enroll"
WG_INTERFACE="${WG_INTERFACE:-wg-mesh}"
WG_DESCRIPTION="${WG_DESCRIPTION:-DeCloud VM}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$LOG_TAG] $*"
    logger -t "$LOG_TAG" "$*" 2>/dev/null || true
}

# =====================================================
# Validate required variables
# =====================================================
MISSING=""
[ -z "$WG_RELAY_ENDPOINT" ] && MISSING="$MISSING WG_RELAY_ENDPOINT"
[ -z "$WG_RELAY_PUBKEY" ]   && MISSING="$MISSING WG_RELAY_PUBKEY"
[ -z "$WG_TUNNEL_IP" ]      && MISSING="$MISSING WG_TUNNEL_IP"
[ -z "$WG_RELAY_API" ]      && MISSING="$MISSING WG_RELAY_API"

if [ -n "$MISSING" ]; then
    log "ERROR: Missing required variables:$MISSING"
    exit 1
fi

# =====================================================
# Check if already enrolled
# =====================================================
if ip link show "$WG_INTERFACE" &>/dev/null; then
    log "WireGuard interface $WG_INTERFACE already exists — skipping enrollment"
    exit 0
fi

log "Enrolling in WireGuard mesh: tunnel_ip=$WG_TUNNEL_IP relay=$WG_RELAY_ENDPOINT"

# =====================================================
# Generate WireGuard keypair
# =====================================================
PRIVATE_KEY=$(wg genkey)
PUBLIC_KEY=$(echo "$PRIVATE_KEY" | wg pubkey)
log "Generated WireGuard keypair (pubkey: ${PUBLIC_KEY:0:12}...)"

# =====================================================
# Write WireGuard config
# =====================================================
WG_CONF="/etc/wireguard/${WG_INTERFACE}.conf"

cat > "$WG_CONF" <<WGEOF
[Interface]
Address = ${WG_TUNNEL_IP}/24
PrivateKey = ${PRIVATE_KEY}

[Peer]
# Relay VM
PublicKey = ${WG_RELAY_PUBKEY}
Endpoint = ${WG_RELAY_ENDPOINT}
AllowedIPs = 10.20.0.0/16
PersistentKeepalive = 25
WGEOF

chmod 600 "$WG_CONF"
log "WireGuard config written to $WG_CONF"

# =====================================================
# Register with relay VM as a peer
# =====================================================
log "Registering with relay at $WG_RELAY_API..."

for i in {1..12}; do
    RESPONSE=$(curl -s -X POST "${WG_RELAY_API}/api/relay/add-peer" \
        -H "Content-Type: application/json" \
        -d "{
            \"public_key\": \"$PUBLIC_KEY\",
            \"allowed_ips\": \"${WG_TUNNEL_IP}/32\",
            \"persistent_keepalive\": 25,
            \"description\": \"$WG_DESCRIPTION\"
        }" \
        --max-time 5 \
        -w "\nHTTP_CODE:%{http_code}" \
        2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | grep -oP 'HTTP_CODE:\K\d+' || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        log "Registered with relay successfully"
        break
    fi

    if [ "$i" -eq 12 ]; then
        log "ERROR: Failed to register with relay after 12 attempts (HTTP $HTTP_CODE)"
        log "Response: $RESPONSE"
        exit 1
    fi

    log "Relay not ready yet (HTTP $HTTP_CODE), retrying in 5s... (attempt $i/12)"
    sleep 5
done

# =====================================================
# Start WireGuard interface
# =====================================================
wg-quick up "$WG_INTERFACE"
systemctl enable "wg-quick@${WG_INTERFACE}" 2>/dev/null || true

# Verify connectivity
sleep 2
RELAY_WG_IP=$(echo "$WG_RELAY_ENDPOINT" | cut -d: -f1)
if wg show "$WG_INTERFACE" &>/dev/null; then
    log "WireGuard interface $WG_INTERFACE is UP (tunnel IP: $WG_TUNNEL_IP)"
else
    log "WARNING: WireGuard interface may not be fully up yet"
fi

log "Mesh enrollment complete"
