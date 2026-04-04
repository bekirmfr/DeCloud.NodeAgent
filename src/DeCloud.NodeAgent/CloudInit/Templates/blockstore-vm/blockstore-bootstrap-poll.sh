#!/bin/bash
#
# DeCloud Block Store Bootstrap Peer Polling
# Version: 1.0
#
# Polls the orchestrator for bootstrap peers while the block store
# node has no connected peers. Once connected, backs off to maintenance
# polling.
#
# Authentication: HMAC-SHA256(authToken, nodeId:vmId)
# Endpoint:       POST /api/blockstore/join  (Orchestrator)
#
# Flow:
#   1. Wait for block store binary to start and report its peer ID
#   2. Call /api/blockstore/join — register peerId, receive bootstrap peers
#   3. Connect binary to returned peers via POST /connect on local API
#   4. Loop: re-poll if peers drop to zero

set -euo pipefail

LOG_FILE="/var/log/decloud-blockstore-bootstrap.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ═══════════════════════════════════════════════════════════════════
# Configuration (baked in by cloud-init)
# ═══════════════════════════════════════════════════════════════════
ORCHESTRATOR_URL="__ORCHESTRATOR_URL__"
NODE_ID="__BLOCKSTORE_NODE_ID__"
VM_ID="__VM_ID__"
AUTH_TOKEN="__BLOCKSTORE_AUTH_TOKEN__"
API_PORT="__BLOCKSTORE_API_PORT__"

POLL_INTERVAL_ISOLATED=30    # seconds between polls when no peers
POLL_INTERVAL_CONNECTED=300  # seconds between polls when connected

# ═══════════════════════════════════════════════════════════════════
# Wait for block store binary to start
# ═══════════════════════════════════════════════════════════════════
log "Waiting for block store binary to start..."
PEER_ID=""

for i in $(seq 1 60); do
    # Try peer-id file first (written by binary at startup)
    if [ -f "/var/lib/decloud-blockstore/peer-id" ]; then
        PEER_ID=$(cat /var/lib/decloud-blockstore/peer-id 2>/dev/null | tr -d '\n')
    fi

    # Fallback: read from HTTP API
    if [ -z "$PEER_ID" ] || [ "$PEER_ID" = "null" ]; then
        HEALTH=$(curl -s --max-time 3 "http://127.0.0.1:${API_PORT}/health" 2>/dev/null) || true
        PEER_ID=$(echo "$HEALTH" | jq -r '.peerId // ""' 2>/dev/null) || true
    fi

    if [ -n "$PEER_ID" ] && [ "$PEER_ID" != "null" ]; then
        log "Block store binary started — peer ID: $PEER_ID"
        break
    fi

    if [ $((i % 10)) -eq 0 ]; then
        log "  Waiting for binary... (attempt $i/60)"
    fi
    sleep 2
done

if [ -z "$PEER_ID" ] || [ "$PEER_ID" = "null" ]; then
    log "Block store binary did not start within 120 seconds"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════
# Compute HMAC-SHA256 authentication token
# Token: HMAC-SHA256(authToken, nodeId:vmId)
# ═══════════════════════════════════════════════════════════════════
compute_token() {
    local message="${NODE_ID}:${VM_ID}"
    echo -n "$message" | openssl dgst -sha256 -hmac "$AUTH_TOKEN" -binary | base64
}

TOKEN=$(compute_token)

# Read advertise IP from blockstore env once — written by cloud-init runcmd
# after WireGuard tunnel IP override, so this reflects the WG mesh IP.
ADVERTISE_IP=$(grep -oP 'BLOCKSTORE_ADVERTISE_IP=\K\S+' \
    /etc/decloud-blockstore/blockstore.env 2>/dev/null || true)
log "Advertise IP: ${ADVERTISE_IP:-<not set>}"

# ═══════════════════════════════════════════════════════════════════
# Main polling loop
# ═══════════════════════════════════════════════════════════════════
log "Entering bootstrap poll loop..."
CONSECUTIVE_FAILURES=0
# Always poll orchestrator at least once on startup, regardless of peer count.
# This ensures blockstore-to-blockstore connections are established via the
# join response even when already connected to the local DHT VM.
# Without this, bitswap must rely on slow DHT FindProviders for every block fetch.
INITIAL_POLL_DONE=false

while true; do
    # Check current peer count
    HEALTH=$(curl -s --max-time 3 "http://127.0.0.1:${API_PORT}/health" 2>/dev/null) || true
    CONNECTED=$(echo "$HEALTH" | jq -r '.connectedPeers // 0' 2>/dev/null) || CONNECTED=0

    if [ "$CONNECTED" -gt 0 ] && [ "$INITIAL_POLL_DONE" = "true" ]; then
        if [ "$CONSECUTIVE_FAILURES" -gt 0 ]; then
            log "Connected to $CONNECTED peer(s) — resuming maintenance polling"
            CONSECUTIVE_FAILURES=0
        fi
        sleep "$POLL_INTERVAL_CONNECTED"
        continue
    fi

    if [ "$CONNECTED" -gt 0 ]; then
        log "Connected to $CONNECTED peer(s) — performing initial orchestrator poll to discover other blockstores..."
    else
        log "No connected peers — polling orchestrator for bootstrap peers..."
    fi
    INITIAL_POLL_DONE=true

    # Call orchestrator /api/blockstore/join
    RESPONSE=$(curl -X POST "${ORCHESTRATOR_URL}/api/blockstore/join" \
        -H "Content-Type: application/json" \
        -H "X-BlockStore-Token: $TOKEN" \
        -d "{
            \"nodeId\":     \"$NODE_ID\",
            \"vmId\":       \"$VM_ID\",
            \"peerId\":     \"$PEER_ID\",
            \"advertiseIp\": \"$ADVERTISE_IP\"
        }" \
        --max-time 10 \
        -s \
        -w "\nHTTP_CODE:%{http_code}" \
        2>&1) || true

    HTTP_CODE=$(echo "$RESPONSE" | grep -oE 'HTTP_CODE:[0-9]+' | cut -d: -f2 || echo "000")
    BODY=$(echo "$RESPONSE" | sed '/HTTP_CODE:/d')

    if [ "$HTTP_CODE" = "200" ]; then
        CONSECUTIVE_FAILURES=0
        BOOTSTRAP_PEERS=$(echo "$BODY" | jq -r '.bootstrapPeers[]?' 2>/dev/null | tr '\n' ',')
        BOOTSTRAP_PEERS="${BOOTSTRAP_PEERS%,}"  # trim trailing comma
        PEER_COUNT=$(echo "$BODY" | jq '.bootstrapPeers | length' 2>/dev/null || echo 0)

        log "Orchestrator returned $PEER_COUNT bootstrap peer(s)"

        if [ -n "$BOOTSTRAP_PEERS" ] && [ "$PEER_COUNT" -gt 0 ]; then
            # Feed bootstrap peers to the running binary via its /connect API
            PEERS_JSON=$(echo "$BODY" | jq '.bootstrapPeers')
            CONNECT_RESP=$(curl -s -X POST "http://127.0.0.1:${API_PORT}/connect" \
                -H "Content-Type: application/json" \
                -d "{\"peers\": $PEERS_JSON}" \
                --max-time 15 2>/dev/null) || true

            CONNECTED_COUNT=$(echo "$CONNECT_RESP" | jq -r '.connected // 0' 2>/dev/null) || CONNECTED_COUNT=0
            log "Connected to $CONNECTED_COUNT/$PEER_COUNT bootstrap peer(s)"
        else
            log "No bootstrap peers available yet — will retry in ${POLL_INTERVAL_ISOLATED}s"
        fi
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        log "Orchestrator join failed (HTTP ${HTTP_CODE:-timeout}), failures: $CONSECUTIVE_FAILURES"

        # Exponential backoff capped at 5 minutes
        BACKOFF=$((POLL_INTERVAL_ISOLATED * CONSECUTIVE_FAILURES))
        BACKOFF=$((BACKOFF > 300 ? 300 : BACKOFF))
        sleep "$BACKOFF"
        continue
    fi

    sleep "$POLL_INTERVAL_ISOLATED"
done