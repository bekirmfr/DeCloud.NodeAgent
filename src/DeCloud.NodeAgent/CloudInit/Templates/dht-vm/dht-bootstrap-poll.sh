#!/bin/bash
#
# DeCloud DHT Bootstrap Peer Polling
# Version: 1.0
#
# Runs as a long-lived systemd service inside the DHT VM.
# Polls the orchestrator's /api/dht/join endpoint to:
#   1. Register this node's DHT peer ID (so the orchestrator knows us)
#   2. Receive bootstrap peers to connect to (so we join the network)
#
# Auth: HMAC-SHA256(auth_token, nodeId:vmId) — same pattern as relay's
#       notify-orchestrator.sh uses HMAC(wireguard_private_key, nodeId:vmId).
#
# This replaces the previous flow where PeerId propagated via:
#   dht-notify-ready.sh → NodeAgent → heartbeat → orchestrator
# Now the DHT VM talks to the orchestrator directly.
#

set -euo pipefail

LOG_FILE="/var/log/decloud-dht-bootstrap-poll.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# =====================================================
# Configuration (injected by cloud-init)
# =====================================================
ORCHESTRATOR_URL="__ORCHESTRATOR_URL__"
NODE_ID="__NODE_ID__"
VM_ID="__VM_ID__"
AUTH_TOKEN="__DHT_AUTH_TOKEN__"
API_PORT="__DHT_API_PORT__"
API_TOKEN="__DHT_API_TOKEN__"

POLL_INTERVAL_ISOLATED=15    # seconds between polls when no peers
POLL_INTERVAL_CONNECTED=300  # seconds between polls when connected (maintenance)
MAX_CONNECT_FAILURES=0       # track consecutive orchestrator failures

log "Starting DHT bootstrap poll service"
log "  Orchestrator: $ORCHESTRATOR_URL"
log "  Node ID:      $NODE_ID"
log "  VM ID:        $VM_ID"
log "  DHT API port: $API_PORT"

# =====================================================
# Wait for DHT binary to start
# =====================================================
log "Waiting for DHT binary to start..."
PEER_ID=""

for i in {1..60}; do
    HEALTH=$(curl -s --max-time 3 "http://127.0.0.1:${API_PORT}/health" 2>/dev/null) || true

    if [ -n "$HEALTH" ]; then
        PEER_ID=$(echo "$HEALTH" | jq -r '.peerId // empty' 2>/dev/null) || true

        if [ -n "$PEER_ID" ]; then
            log "DHT binary started - peer ID: $PEER_ID"
            break
        fi
    fi

    if [ $((i % 10)) -eq 0 ]; then
        log "  Still waiting for DHT binary... (attempt $i/60)"
    fi
    sleep 2
done

if [ -z "$PEER_ID" ]; then
    log "ERROR: Failed to get peer ID after 120 seconds"
    exit 1
fi

# =====================================================
# Compute authentication token (relay pattern)
# HMAC-SHA256(auth_token, nodeId:vmId)
# =====================================================
compute_token() {
    local message="${NODE_ID}:${VM_ID}"
    echo -n "$message" | openssl dgst -sha256 -hmac "$AUTH_TOKEN" -binary | base64
}

TOKEN=$(compute_token)

# =====================================================
# Main polling loop
# =====================================================
log "Entering bootstrap poll loop..."
CONSECUTIVE_FAILURES=0

while true; do
    # Check current peer count
    HEALTH=$(curl -s --max-time 3 "http://127.0.0.1:${API_PORT}/health" 2>/dev/null) || true
    CONNECTED=$(echo "$HEALTH" | jq -r '.connectedPeers // 0' 2>/dev/null) || CONNECTED=0

    if [ "$CONNECTED" -gt 0 ]; then
        # We have peers — back off to maintenance polling
        if [ "$CONSECUTIVE_FAILURES" -gt 0 ]; then
            log "Connected to $CONNECTED peer(s) — resuming maintenance polling"
            CONSECUTIVE_FAILURES=0
        fi
        sleep "$POLL_INTERVAL_CONNECTED"
        continue
    fi

    log "No connected peers — polling orchestrator for bootstrap peers..."

    # Call orchestrator /api/dht/join
    RESPONSE=$(curl -X POST "${ORCHESTRATOR_URL}/api/dht/join" \
        -H "Content-Type: application/json" \
        -H "X-DHT-Token: $TOKEN" \
        -d "{
            \"nodeId\": \"$NODE_ID\",
            \"vmId\": \"$VM_ID\",
            \"peerId\": \"$PEER_ID\"
        }" \
        --max-time 10 \
        -s \
        -w "\nHTTP_CODE:%{http_code}" \
        2>&1) || true

    HTTP_CODE=$(echo "$RESPONSE" | grep -oP 'HTTP_CODE:\K\d+' || echo "000")
    BODY=$(echo "$RESPONSE" | sed '/HTTP_CODE:/d')

    if [ "$HTTP_CODE" = "200" ]; then
        CONSECUTIVE_FAILURES=0

        # Extract bootstrap peers from response
        PEER_COUNT=$(echo "$BODY" | jq -r '.bootstrapPeers | length // 0' 2>/dev/null) || PEER_COUNT=0

        if [ "$PEER_COUNT" -gt 0 ]; then
            log "Received $PEER_COUNT bootstrap peer(s) from orchestrator"

            # Build JSON array for POST /connect
            PEERS_JSON=$(echo "$BODY" | jq -c '.bootstrapPeers' 2>/dev/null)

            # Inject peers via DHT binary's POST /connect endpoint
            CONNECT_RESULT=$(curl -X POST "http://127.0.0.1:${API_PORT}/connect" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $API_TOKEN" \
                -d "{\"peers\": $PEERS_JSON}" \
                --max-time 10 \
                -s \
                2>/dev/null) || true

            CONNECTED_COUNT=$(echo "$CONNECT_RESULT" | jq -r '.connected // 0' 2>/dev/null) || CONNECTED_COUNT=0
            log "Connected to $CONNECTED_COUNT/$PEER_COUNT bootstrap peer(s)"
        else
            log "Orchestrator returned 0 bootstrap peers (may be the first node)"
        fi
    elif [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "403" ]; then
        log "ERROR: Authentication failed (HTTP $HTTP_CODE) — check DHT_AUTH_TOKEN"
        # Don't spam on auth failures — back off significantly
        sleep 60
        continue
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        if [ "$CONSECUTIVE_FAILURES" -le 3 ] || [ $((CONSECUTIVE_FAILURES % 20)) -eq 0 ]; then
            log "WARNING: Orchestrator unreachable (HTTP ${HTTP_CODE:-timeout}, failures: $CONSECUTIVE_FAILURES)"
        fi
    fi

    sleep "$POLL_INTERVAL_ISOLATED"
done
