#!/usr/bin/env bash
# ============================================================================
# test-scan-node.sh — Phase 6 pass 2b, node-side half.
#
# Proves the two things the orchestrator script cannot see:
#   1. HandleScanVmAsync ran, invoked the scanner, and reported NotScanned
#      (the honest stub — NullCsamScanner, not Enabled).
#   2. The node independently refused a RUNNING VM (plan §4.5 step 3) — the
#      orchestrator's held+stopped assertion is not trusted; the node checks.
#
# This is observational: it reads the agent journal after a scan is ordered
# from the orchestrator (via test-scan-chain.sh or a manual scan-vm call). It
# does not issue commands itself — the node has no admin surface.
#
# Usage (on the node, as root), AFTER ordering a scan for VM_ID:
#   VM_ID=<the held VM that was scanned> ./test-scan-node.sh
#
# Optional: NODE_SERVICE (default decloud-node-agent), SINCE (default "-10 min")
# ============================================================================
set -u

VM_ID="${VM_ID:?Set VM_ID (the VM a scan was ordered for)}"
NODE_SERVICE="${NODE_SERVICE:-decloud-node-agent}"
SINCE="${SINCE:--10 min}"

PASS=0; FAIL=0; SKIP=0
ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }
skip() { echo "  – $1 (skipped)"; SKIP=$((SKIP+1)); }
say()  { echo; echo "── $1"; }

[ "$(id -u)" = "0" ] || { echo "Run as root (journalctl)."; exit 1; }
systemctl is-active --quiet "$NODE_SERVICE" \
  || { echo "$NODE_SERVICE not active. Set NODE_SERVICE=<unit>."; exit 1; }

log() { journalctl -u "$NODE_SERVICE" --since "$SINCE" --no-pager -q 2>/dev/null; }

say "did HandleScanVmAsync run for $VM_ID?"
# The handler logs "ScanVm: VM {VmId} → {Outcome} ({Matcher})" on success, or
# "ScanVm: refusing VM {VmId} — it is Running", or "ScanVm: failed for VM ...".
if log | grep -qF "ScanVm:" && log | grep -qF "$VM_ID"; then
  ok "ScanVm handler fired for $VM_ID"
else
  bad "no ScanVm log line for $VM_ID in the last window — was a scan ordered? widen SINCE?"
  echo; echo "PASS=$PASS FAIL=$FAIL"; exit 1
fi

say "outcome is NotScanned (the honest stub), never Clean"
if log | grep -F "ScanVm:" | grep -F "$VM_ID" | grep -q "NotScanned"; then
  ok "reported NotScanned — NullCsamScanner honest, matcher unwired"
elif log | grep -F "ScanVm:" | grep -F "$VM_ID" | grep -qi "refusing.*Running"; then
  skip "handler refused a RUNNING VM — expected if the VM wasn't stopped; see next check"
elif log | grep -F "ScanVm:" | grep -F "$VM_ID" | grep -q "Clean"; then
  bad "reported Clean — the honesty clamp FAILED; a matcher-less stub must never say Clean"
else
  bad "ScanVm ran but outcome unclear: $(log | grep -F "ScanVm:" | grep -F "$VM_ID" | tail -1)"
fi

say "independent running-VM refusal (only if the VM was running when scanned)"
# This is the safety property: even if the orchestrator's view was stale and it
# ordered a scan on a running VM, the node refuses. Can only be observed if such
# a scan actually happened — otherwise correctly skipped.
if log | grep -F "$VM_ID" | grep -qi "ScanVm: refusing.*Running"; then
  ok "node refused a running VM independently (safety check held)"
else
  skip "no running-VM scan observed — the precondition held upstream (VM was stopped)"
fi

say "the scanner was invoked, not bypassed"
# NullCsamScanner does no I/O and logs nothing itself unless ForceOutcome is set;
# the handler's own success line IS the evidence it called ScanAsync. If you want
# positive proof the stub ran, set Csam:Stub:ForceOutcome=Unscannable on this node
# and re-scan — the outcome should read Unscannable, proving the seam is live.
if log | grep -F "ScanVm:" | grep -F "$VM_ID" | grep -q "NotScanned\|Unscannable"; then
  ok "scanner seam is live (outcome came from ScanAsync)"
else
  skip "seam liveness inconclusive from logs alone — see the ForceOutcome note above"
fi

echo
echo "════════════════════════════════════════════════════════════"
echo "PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
cat <<'EOF'

To close the loop end-to-end:
  1. Order a scan (test-scan-chain.sh, or POST /api/admin/compliance/scan-vm).
  2. Run this script on the node → confirms the handler ran + NotScanned.
  3. Back on the orchestrator, GET /api/admin/abuse and confirm the report's
     scan record moved Ordered → Completed, outcome NotScanned, matcher
     NullCsamScanner. That proves the ack landed (finding (b)'s routing).

To prove a FAILED scan lands (not silently dropped):
  Order a scan while the VM is RUNNING (resume it first, without lifting the
  hold — or scan before stopping). The node refuses; the ack carries success
  = false; the report's record must read status=Failed, error="VM is running".
  If instead the record stays Ordered forever, the ack-routing (NodeService,
  edit §12) did not fire — investigate.
EOF
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
