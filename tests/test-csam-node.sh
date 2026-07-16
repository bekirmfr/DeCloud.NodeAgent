#!/usr/bin/env bash
# ============================================================================
# test-csam-node.sh — Phase 6 pass 1: node-side checks A–E, automated
#
# Runs ON THE NODE (root). Covers the manual checklist printed by
# test-csam-pass1.sh, plus the Clean-refusal honesty invariant:
#
#   A. Null scanner, RF>0 VM: cycle replicates as before; lazysync.json
#      records csamScan.outcome = NotScanned (never Clean).
#   B. RF=0 VM enrolls and scans (snapshot taken, state recorded) but does
#      NOT replicate (no push/manifest).
#   E. Merge-back invariant: no lazysync-overlay-*.qcow2 persists after a
#      cycle (covers the idle / changedChunks==0 path the pass fixed).
#   X. Forced ForceOutcome=Clean is REFUSED by the stub (startup warning;
#      state stays NotScanned). The core honesty invariant, checked cheaply.
#   D. ForceOutcome=Unscannable: cycle proceeds, state Unscannable,
#      replication unaffected.
#   C. ForceOutcome=Match (DESTRUCTIVE, LAST): match persisted + reported,
#      VM suspended (ComplianceHold), csam P0 item in the abuse queue,
#      push blocked. Then cleanup: config removed, VM resumed.
#
# Usage (on the node, as root):
#   ORCH_URL=https://orchestrator.example ADMIN_JWT=eyJ... \
#   RF1_VM_ID=<RF>0 tenant VM id> RF0_VM_ID=<RF=0 DISPOSABLE tenant VM id> \
#   CONFIRM_FORCE=yes ./test-csam-node.sh
#
# Optional env:
#   NODE_SERVICE   systemd unit of the node agent   (default: decloud-node-agent)
#   VM_STORAGE     VM storage root                   (default: /var/lib/decloud/vms)
#   CYCLE_WAIT     max seconds to wait for a cycle   (default: 420  = 7 min)
#   RESTART_WAIT   max wait after a service restart  (default: 780  = 13 min:
#                  5 min startup delay + 5 min first interval + margin)
#
# ⚠ BLAST RADIUS: Csam:Stub:ForceOutcome applies to EVERY tenant VM this
#   node scans. Phase C will report + SUSPEND every enrolled tenant VM on
#   this node, not just RF0_VM_ID. Run only on a node whose tenant VMs are
#   disposable. CONFIRM_FORCE=yes is required for phases X/D/C; without it
#   the script runs only the passive checks (A, B, E) and exits.
#
# ⚠ RUNTIME: passive checks ≤ ~8 min; full run ~30–40 min (three service
#   restarts, each followed by the 5-min startup delay before a cycle).
#
# What this script does NOT prove (honest limits):
#   - Whether the pre-fix changedChunks==0 overlay leak existed on the OLD
#     agent (check E verifies the fixed agent's invariant; run `ls
#     $VM_STORAGE/*/lazysync-overlay-*.qcow2` on a not-yet-updated node to
#     settle the historical question, and note the answer in the build log).
#   - Budget-overrun deferral (the stub is instant; needs the real matcher).
# ============================================================================
set -u

# ── Inputs ──────────────────────────────────────────────────────────────────
ORCH_URL="${ORCH_URL:?Set ORCH_URL}"
ADMIN_JWT="${ADMIN_JWT:?Set ADMIN_JWT}"
RF1_VM_ID="${RF1_VM_ID:?Set RF1_VM_ID (an RF>0 tenant VM on this node)}"
RF0_VM_ID="${RF0_VM_ID:?Set RF0_VM_ID (a DISPOSABLE RF=0 tenant VM on this node)}"
NODE_SERVICE="${NODE_SERVICE:-decloud-node-agent}"
VM_STORAGE="${VM_STORAGE:-/var/lib/decloud/vms}"
CYCLE_WAIT="${CYCLE_WAIT:-420}"
RESTART_WAIT="${RESTART_WAIT:-780}"
CONFIRM_FORCE="${CONFIRM_FORCE:-no}"

PASS=0; FAIL=0; SKIP=0
ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }
skip() { echo "  – $1 (skipped)"; SKIP=$((SKIP+1)); }
say()  { echo; echo "── $1"; }

# ── Preflight ────────────────────────────────────────────────────────────────
[ "$(id -u)" = "0" ] || { echo "Run as root (journalctl, systemd, VM dirs)."; exit 1; }
for dep in jq curl systemctl journalctl; do
  command -v "$dep" >/dev/null || { echo "Missing dependency: $dep"; exit 1; }
done
systemctl is-active --quiet "$NODE_SERVICE" \
  || { echo "Service $NODE_SERVICE is not active. Set NODE_SERVICE=<unit>."; exit 1; }
for id in "$RF1_VM_ID" "$RF0_VM_ID"; do
  [ -d "$VM_STORAGE/$id" ] \
    || { echo "No $VM_STORAGE/$id — wrong VM_STORAGE or VM not on this node."; exit 1; }
done

DROPIN_DIR="/etc/systemd/system/${NODE_SERVICE}.service.d"
DROPIN="$DROPIN_DIR/99-csam-test.conf"
FORCED_ACTIVE=0   # a forced-outcome drop-in is currently installed
C_SUSPENDED=0     # phase C ran; VMs on this node may be held

# ── Helpers ──────────────────────────────────────────────────────────────────
acurl() { curl -s -o /tmp/csamnode_body -w "%{http_code}" \
          -H "Authorization: Bearer $ADMIN_JWT" -H "Content-Type: application/json" "$@"; }

# wait_log <since> <timeout_s> <fixed-string...>  → 0 when ALL strings appear
wait_log() {
  local since="$1" timeout="$2"; shift 2
  local deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local all=1 s
    for s in "$@"; do
      journalctl -u "$NODE_SERVICE" --since "$since" --no-pager -q 2>/dev/null \
        | grep -qF "$s" || { all=0; break; }
    done
    [ "$all" = 1 ] && return 0
    sleep 15
  done
  return 1
}

log_has() { journalctl -u "$NODE_SERVICE" --since "$1" --no-pager -q 2>/dev/null | grep -qF "$2"; }

# state_outcome <vmId> → prints outcome as name, tolerating string or numeric
# enum serialization and either JSON property casing. NotScanned=0 Clean=1
# Match=2 Unscannable=3 (CsamOutcome member order in ICsamScanner.cs).
state_outcome() {
  local f="$VM_STORAGE/$1/lazysync.json"
  [ -f "$f" ] || { echo "NOFILE"; return; }
  jq -r '(.csamScan // .CsamScan // empty) | (.outcome // .Outcome // empty)
         | if . == 0 then "NotScanned" elif . == 1 then "Clean"
           elif . == 2 then "Match" elif . == 3 then "Unscannable"
           else tostring end' "$f" 2>/dev/null || echo "PARSE_ERROR"
}

state_field() {  # state_field <vmId> <camelName> <PascalName>
  local f="$VM_STORAGE/$1/lazysync.json"
  [ -f "$f" ] || { echo ""; return; }
  jq -r "(.csamScan // .CsamScan // empty) | (.$2 // .$3 // empty) | tostring" "$f" 2>/dev/null
}

set_force() {  # set_force <Match|Unscannable|Clean> — install drop-in + restart
  mkdir -p "$DROPIN_DIR"
  printf '[Service]\nEnvironment="Csam__Stub__ForceOutcome=%s"\n' "$1" > "$DROPIN"
  systemctl daemon-reload && systemctl restart "$NODE_SERVICE" || return 1
  FORCED_ACTIVE=1
}

clear_force() {  # remove drop-in + restart back to honest config
  [ -f "$DROPIN" ] || return 0
  rm -f "$DROPIN"; rmdir "$DROPIN_DIR" 2>/dev/null || true
  systemctl daemon-reload && systemctl restart "$NODE_SERVICE"
  FORCED_ACTIVE=0
}

cleanup() {
  echo
  echo "── Cleanup"
  if [ "$FORCED_ACTIVE" = 1 ]; then
    clear_force && echo "  · ForceOutcome drop-in removed; $NODE_SERVICE restarted honest." \
               || echo "  ! FAILED to remove $DROPIN — REMOVE IT MANUALLY and restart."
  fi
  if [ "$C_SUSPENDED" = 1 ]; then
    code=$(acurl -X POST "$ORCH_URL/api/admin/compliance/resume-vm" \
      -d "{\"vmId\":\"$RF0_VM_ID\",\"reason\":\"test-csam-node cleanup\"}")
    [ "$code" = "200" ] && echo "  · resume-vm $RF0_VM_ID → 200" \
                        || echo "  ! resume-vm $RF0_VM_ID → $code — resume manually."
    echo "  ! Phase C forces a Match for EVERY scanned tenant VM on this node."
    echo "    Check the admin queue/VM list for other held VMs and resume them."
    echo "    The csam report(s) stay Open in the queue BY DESIGN — a human"
    echo "    resolves them there; this script does not."
  fi
}
trap cleanup EXIT

# ════════════════════════════════════════════════════════════════════════════
say "Phase 1 — passive checks A, B, E (honest null scanner, no config change)"
# ════════════════════════════════════════════════════════════════════════════
T0=$(date '+%Y-%m-%d %H:%M:%S')
echo "  waiting up to ${CYCLE_WAIT}s for one lazysync cycle to touch both VMs…"

# A: RF>0 VM — replication still happens, and honest state is recorded.
if wait_log "$T0" "$CYCLE_WAIT" "VM $RF1_VM_ID: snapshot taken" "VM $RF1_VM_ID: lazysync v"; then
  ok "A: RF>0 VM $RF1_VM_ID scanned + replicated this cycle"
else
  # An idle RF>0 VM legitimately produces no 'lazysync v' line (0 changed
  # chunks) — the snapshot line alone still proves scan enrollment.
  if log_has "$T0" "VM $RF1_VM_ID: snapshot taken"; then
    ok "A: RF>0 VM $RF1_VM_ID scanned (idle cycle — no changed chunks, no push needed)"
  else
    bad "A: no snapshot line for RF>0 VM $RF1_VM_ID within ${CYCLE_WAIT}s"
  fi
fi
oc=$(state_outcome "$RF1_VM_ID")
case "$oc" in
  NotScanned) ok "A: $RF1_VM_ID csamScan.outcome=NotScanned (honest; not Clean)";;
  Clean)      bad "A: $RF1_VM_ID reads Clean under the null scanner — HONESTY INVARIANT VIOLATED";;
  *)          bad "A: $RF1_VM_ID unexpected outcome '$oc'";;
esac
lsat=$(state_field "$RF1_VM_ID" lastScanAt LastScanAt)
[ -n "$lsat" ] && ok "A: lastScanAt recorded ($lsat)" || bad "A: no lastScanAt in lazysync.json"

# B: RF=0 VM — enrolled + scanned, but no replication tail.
if log_has "$T0" "VM $RF0_VM_ID: snapshot taken"; then
  ok "B: RF=0 VM $RF0_VM_ID enrolled + snapshotted (Decision 15)"
else
  bad "B: no snapshot line for RF=0 VM $RF0_VM_ID — not enrolled?"
fi
if log_has "$T0" "VM $RF0_VM_ID: lazysync v"; then
  bad "B: RF=0 VM $RF0_VM_ID PUSHED a manifest — replication tail not gated!"
else
  ok "B: RF=0 VM $RF0_VM_ID did not replicate (no push/manifest line)"
fi
oc=$(state_outcome "$RF0_VM_ID")
[ "$oc" = "NotScanned" ] && ok "B: $RF0_VM_ID outcome=NotScanned" \
                         || bad "B: $RF0_VM_ID outcome '$oc' (expected NotScanned)"

# E: merge-back invariant — no overlay survives the cycle. Grace period
# covers a cycle currently in flight (overlay legitimately live mid-cycle).
sleep 90
leftover=$(find "$VM_STORAGE/$RF1_VM_ID" "$VM_STORAGE/$RF0_VM_ID" \
           -maxdepth 1 -name 'lazysync-overlay-*.qcow2' 2>/dev/null)
if [ -z "$leftover" ]; then
  ok "E: no lazysync-overlay-*.qcow2 persists after the cycle (single-exit merge holds)"
else
  bad "E: orphan overlay(s) found — merge-back invariant broken: $leftover"
fi

# ════════════════════════════════════════════════════════════════════════════
if [ "$CONFIRM_FORCE" != "yes" ]; then
  say "Phases X, D, C require CONFIRM_FORCE=yes (they restart $NODE_SERVICE and"
  echo "   phase C will SUSPEND every enrolled tenant VM on this node). Stopping"
  echo "   after the passive checks."
  skip "X: forced-Clean refusal"; skip "D: forced Unscannable"; skip "C: forced Match chain"
else

  # ══════════════════════════════════════════════════════════════════════════
  say "Phase 2 (X) — forced Clean must be REFUSED (honesty invariant, §8)"
  # ══════════════════════════════════════════════════════════════════════════
  TX=$(date '+%Y-%m-%d %H:%M:%S')
  if set_force "Clean"; then
    # NullCsamScanner logs the refusal at construction — no cycle wait needed.
    if wait_log "$TX" 120 "Csam:Stub:ForceOutcome" "ignored"; then
      ok "X: stub refused ForceOutcome=Clean at startup (warning logged)"
    else
      bad "X: no refusal warning for ForceOutcome=Clean within 120s — check stub wiring"
    fi
  else
    bad "X: could not install drop-in / restart $NODE_SERVICE"
  fi

  # ══════════════════════════════════════════════════════════════════════════
  say "Phase 3 (D) — forced Unscannable proceeds; replication unaffected"
  echo "   (restart + 5-min startup delay: waiting up to ${RESTART_WAIT}s)"
  # ══════════════════════════════════════════════════════════════════════════
  TD=$(date '+%Y-%m-%d %H:%M:%S')
  if set_force "Unscannable"; then
    if wait_log "$TD" "$RESTART_WAIT" "VM $RF1_VM_ID: snapshot taken"; then
      sleep 30   # let the state write land
      oc=$(state_outcome "$RF1_VM_ID")
      [ "$oc" = "Unscannable" ] && ok "D: $RF1_VM_ID outcome=Unscannable" \
                                || bad "D: $RF1_VM_ID outcome '$oc' (expected Unscannable)"
      if log_has "$TD" "VM $RF1_VM_ID: lazysync v"; then
        ok "D: replication proceeded despite Unscannable (only Match blocks)"
      else
        skip "D: no push this cycle (idle VM, 0 changed chunks) — gate not exercised; write a file in the guest and re-run to prove it"
      fi
    else
      bad "D: no cycle observed for $RF1_VM_ID within ${RESTART_WAIT}s of restart"
    fi
  else
    bad "D: could not install drop-in / restart $NODE_SERVICE"
  fi

  # ══════════════════════════════════════════════════════════════════════════
  say "Phase 4 (C) — forced Match: persist → report → suspend → P0 queue (DESTRUCTIVE)"
  echo "   (waiting up to ${RESTART_WAIT}s for the match report)"
  # ══════════════════════════════════════════════════════════════════════════
  TC=$(date '+%Y-%m-%d %H:%M:%S')
  if set_force "Match"; then
    C_SUSPENDED=1
    if wait_log "$TC" "$RESTART_WAIT" "VM $RF0_VM_ID: CSAM match reported to orchestrator"; then
      ok "C: match reported to orchestrator for $RF0_VM_ID"
      oc=$(state_outcome "$RF0_VM_ID")
      [ "$oc" = "Match" ] && ok "C: match persisted in lazysync.json (survives crash)" \
                          || bad "C: state outcome '$oc' (expected Match)"
      mr=$(state_field "$RF0_VM_ID" matchReported MatchReported)
      [ "$mr" = "true" ] && ok "C: matchReported=true (node stops retrying; queue is durable)" \
                         || bad "C: matchReported='$mr' (expected true after ack)"
      if log_has "$TC" "VM $RF0_VM_ID: lazysync v"; then
        bad "C: $RF0_VM_ID pushed a manifest AFTER matching — GATE BREACHED"
      else
        ok "C: no push for $RF0_VM_ID (Match blocks the tail)"
      fi
      # Orchestrator side: hold applied + P0 item filed.
      code=$(acurl "$ORCH_URL/api/vms/$RF0_VM_ID")
      if [ "$code" = "200" ] && grep -qiE '"complianceHold" *: *true' /tmp/csamnode_body; then
        ok "C: orchestrator shows ComplianceHold=true (protective suspend landed)"
      else
        bad "C: VM fetch → $code, complianceHold not true (host-fencing? NodeId mismatch?)"
      fi
      code=$(acurl "$ORCH_URL/api/admin/abuse")
      if [ "$code" = "200" ] && grep -qF "$RF0_VM_ID" /tmp/csamnode_body \
         && grep -qiF "csam" /tmp/csamnode_body; then
        ref=$(grep -oE 'ABU-[0-9]{4}-[0-9]+' /tmp/csamnode_body | head -1)
        ok "C: csam item for $RF0_VM_ID in the P0 queue (${ref:-reference not parsed})"
      else
        bad "C: abuse queue ($code) has no csam item referencing $RF0_VM_ID"
      fi
    else
      bad "C: no 'CSAM match reported' line for $RF0_VM_ID within ${RESTART_WAIT}s"
      log_has "$TC" "CSAM match report FAILED" \
        && echo "    (node logged report FAILURES — check orchestrator reachability/auth; the match is persisted and push stays blocked, so fail-closed held)"
    fi
  else
    bad "C: could not install drop-in / restart $NODE_SERVICE"
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════"
echo "PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
echo
echo "Note for the build log: check E proves the FIXED agent's merge-back"
echo "invariant. To settle whether the pre-fix changedChunks==0 overlay leak"
echo "was real, run on a NOT-yet-updated node:"
echo "    ls $VM_STORAGE/*/lazysync-overlay-*.qcow2"
echo "Accumulated overlays there = the bug was live; record either answer."
exit $([ "$FAIL" = 0 ] && echo 0 || echo 1)
