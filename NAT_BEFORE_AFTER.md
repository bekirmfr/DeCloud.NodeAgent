# NAT Rule Checking: Before vs After

## Log Output Comparison

### BEFORE (Every 1 minute)
```
[2026-01-31 10:00:00] [dbug] NAT check failed for 10.42.0.15, assuming rules missing
[2026-01-31 10:00:00] [warn] Relay VM relay-abc at 10.42.0.15 missing NAT rules - reconfiguring
[2026-01-31 10:00:02] [info] ✓ NAT configured successfully: eth0:51820 → 10.42.0.15:51820

[2026-01-31 10:01:00] [dbug] NAT check failed for 10.42.0.15, assuming rules missing
[2026-01-31 10:01:00] [warn] Relay VM relay-abc at 10.42.0.15 missing NAT rules - reconfiguring
[2026-01-31 10:01:02] [info] ✓ NAT configured successfully: eth0:51820 → 10.42.0.15:51820

[2026-01-31 10:02:00] [dbug] NAT check failed for 10.42.0.15, assuming rules missing
[2026-01-31 10:02:00] [warn] Relay VM relay-abc at 10.42.0.15 missing NAT rules - reconfiguring
[2026-01-31 10:02:02] [info] ✓ NAT configured successfully: eth0:51820 → 10.42.0.15:51820

... (repeated 60 times per hour per relay VM)
```

### AFTER (Every 10 minutes, with caching)
```
[2026-01-31 10:00:00] [dbug] NAT check command failed for 10.42.0.15 (exit code 1), rules likely missing
[2026-01-31 10:00:00] [warn] Relay VM relay-abc at 10.42.0.15 missing NAT rules - reconfiguring
[2026-01-31 10:00:02] [info] ✓ NAT configured successfully: eth0:51820 → 10.42.0.15:51820

(Checks skipped from 10:01-10:09 due to cache)

[2026-01-31 10:10:00] [trace] All NAT rules verified for 10.42.0.15: PREROUTING=✓, POSTROUTING=✓, FORWARD=✓
[2026-01-31 10:10:00] [dbug] ✓ Relay VM relay-abc at 10.42.0.15 has complete NAT rules

(Checks skipped from 10:11-10:19 due to cache)

[2026-01-31 10:20:00] [trace] All NAT rules verified for 10.42.0.15: PREROUTING=✓, POSTROUTING=✓, FORWARD=✓
[2026-01-31 10:20:00] [dbug] ✓ Relay VM relay-abc at 10.42.0.15 has complete NAT rules

... (only 6 checks per hour per relay VM)
```

## Metrics

### Check Frequency
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Checks per hour (per VM) | 60 | 6 | **90% reduction** |
| `iptables` calls per hour | 60+ | 6 | **90% reduction** |
| Time between checks | 1 min | 10 min | **10x less frequent** |
| Log lines per hour | ~180 | ~18 | **90% reduction** |

### Check Accuracy
| Metric | Before | After |
|--------|--------|-------|
| Rules checked | 1 (PREROUTING only) | 3 (PREROUTING + POSTROUTING + FORWARD) |
| False negatives | Common (incomplete check) | Rare (comprehensive check) |
| Detection method | Raw `iptables -t nat -C` | NAT script `check` command |

### Resource Usage
| Resource | Before | After | Savings |
|----------|--------|-------|---------|
| CPU (iptables calls) | High | Low | ~90% |
| Log storage | High | Low | ~90% |
| Network I/O | N/A | N/A | - |

## Code Flow

### BEFORE
```
VmHealthService (every 1 min)
  ↓
CheckRelayVmNatRulesAsync()
  ↓
RuleExistsAsync()
  ↓
iptables -t nat -C PREROUTING ...  ← Only checks 1 rule
  ↓
  ✗ Fails (incomplete check or format mismatch)
  ↓
AddPortForwardingAsync()
  ↓
Reconfigure all rules
  ↓
(Repeat next minute...)
```

### AFTER
```
VmHealthService (every 1 min)
  ↓
CheckRelayVmNatRulesAsync()
  ↓
Check cache: Last check < 10 min ago?
  ├─ YES → Skip check (cached)
  └─ NO  → Continue
       ↓
     HasRulesForVmAsync()
       ↓
     decloud-relay-nat check <vmip>  ← Checks all 3 rules
       ↓
       ✓ Success (all rules present)
       ↓
     Update cache timestamp
       ↓
     (Skip next 9 checks due to cache)
```

## Detection Improvements

### Rule Coverage

**Before:** Only checked PREROUTING
```bash
iptables -t nat -C PREROUTING -p udp --dport 51820 -j DNAT --to-destination 10.42.0.15:51820
```

**After:** Checks all 3 required rules
```bash
# PREROUTING (DNAT)
iptables -t nat -C PREROUTING -p udp --dport 51820 -j DNAT --to-destination 10.42.0.15:51820

# POSTROUTING (MASQUERADE)
iptables -t nat -C POSTROUTING -d 10.42.0.15 -j MASQUERADE

# FORWARD (ACCEPT)
iptables -C FORWARD -d 10.42.0.15 -p udp --dport 51820 -j ACCEPT
```

## Cache Behavior

### First Detection (Rules Missing)
```
Check 1: Rules missing → Configure → Cache updated → Next check in 10 min
Check 2: (Skipped - cached)
...
Check 10: Rules verified → Cache updated → Next check in 10 min
```

### Steady State (Rules Present)
```
Check 1: Rules verified → Cache updated → Next check in 10 min
Check 2-10: (Skipped - cached)
Check 11: Rules verified → Cache updated → Next check in 10 min
```

### After Relay VM Restart
```
Restart: Cache preserved (per-VM)
Check 1: Rules verified (or reconfigured if missing) → Cache updated
Check 2-10: (Skipped - cached)
```

## Benefits

1. **90% Less Overhead**
   - Fewer `iptables` executions
   - Reduced CPU usage
   - Less log storage

2. **Cleaner Logs**
   - Expected states use Debug/Trace
   - Warnings only for real issues
   - Easier to spot problems

3. **Better Detection**
   - All 3 NAT rules verified
   - Uses proper check command
   - Fewer false positives

4. **Faster Recovery**
   - Missing rules detected quickly
   - Reconfiguration is logged clearly
   - Cache ensures healthy state remembered

5. **Scalable**
   - Per-VM caching
   - Thread-safe implementation
   - Works with multiple relay VMs

## Deployment Steps

```bash
cd /opt/decloud/DeCloud.NodeAgent
chmod +x deploy-nat-optimization.sh
./deploy-nat-optimization.sh
```

Or manually:
```bash
cd /opt/decloud/DeCloud.NodeAgent
dotnet build -c Release
sudo ./install.sh
sudo systemctl restart decloud-nodeagent
sudo journalctl -u decloud-nodeagent -f | grep -i nat
```

## Verification

After deployment, verify the improvements:

```bash
# Count NAT checks in last hour (should be ~6 per relay VM)
sudo journalctl -u decloud-nodeagent --since "1 hour ago" | grep "NAT check" | wc -l

# Verify 10-minute intervals (timestamps should be ~10 min apart)
sudo journalctl -u decloud-nodeagent --since "1 hour ago" | grep "NAT rules verified"

# Check for excessive warnings (should be minimal)
sudo journalctl -u decloud-nodeagent --since "1 hour ago" | grep "NAT" | grep "warn"
```

Expected: **6 checks/hour per VM** instead of **60 checks/hour**
