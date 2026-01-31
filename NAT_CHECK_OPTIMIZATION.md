# NAT Rule Checking Optimization

## Problem Analysis

The logs showed repeated NAT rule checking failures every minute:

```
[dbug] NAT check failed for 10.42.0.X, assuming rules missing
[warn] Relay VM relay-xyz at 10.42.0.X missing NAT rules - reconfiguring
[info] ✓ NAT configured successfully
```

This cycle repeated continuously, causing:
- Excessive `iptables` command executions
- Log noise making real issues harder to spot
- Unnecessary NAT reconfiguration attempts
- Performance overhead from repeated checks

## Root Causes

### 1. Wrong Check Method
**VmHealthService.cs** was using `RuleExistsAsync()` which only checked the PREROUTING rule using raw `iptables -t nat -C`. This method:
- Only checked 1 of 3 required rules (PREROUTING, POSTROUTING, FORWARD)
- Used exact string matching which could fail on minor format differences
- Didn't leverage the NAT script's built-in `check` command

### 2. Excessive Check Frequency
Checks ran **every 1 minute** for every relay VM, even though NAT rules rarely change.

### 3. No Caching
Once rules were verified, the system kept checking them repeatedly without remembering the last successful verification.

## Solutions Implemented

### 1. Use Comprehensive Check Method
✅ Changed `VmHealthService` to use `HasRulesForVmAsync()` instead of `RuleExistsAsync()`
✅ `HasRulesForVmAsync()` uses the NAT script's `check` command which verifies all 3 rules
✅ `RuleExistsAsync()` now delegates to `HasRulesForVmAsync()` for consistency

### 2. Implement Check Interval
✅ Introduced `NatCheckInterval = TimeSpan.FromMinutes(10)`
✅ Health checks still run every 1 minute, but NAT checks only happen every 10 minutes
✅ Reduces NAT checking by **90%** (from 60/hour to 6/hour per VM)

### 3. Add Per-VM Check Caching
✅ Track last check time per VM in `_lastNatCheckByVm` dictionary
✅ Skip checks if recently verified (within 10 minutes)
✅ Update cache timestamp after successful verification or reconfiguration
✅ Thread-safe with `SemaphoreSlim`

### 4. Reduce Log Noise
✅ Changed "NAT check failed" from `LogWarning` to `LogDebug` (expected when rules missing)
✅ Changed "Incomplete NAT rules" from `LogWarning` to `LogDebug` (first detection)
✅ Added `LogTrace` for successful comprehensive checks
✅ Added NAT script existence check before attempting operations

## Code Changes

### VmHealthService.cs

**Added:**
- `NatCheckInterval = TimeSpan.FromMinutes(10)`
- `_lastNatCheckByVm` dictionary for per-VM caching
- `_natCheckLock` semaphore for thread safety

**Modified:**
- `CheckRelayVmNatRulesAsync()` to:
  - Check cache before running NAT verification
  - Use `HasRulesForVmAsync()` instead of `RuleExistsAsync()`
  - Update cache after successful checks
  - Pass cancellation token properly

### NatRuleManager.cs

**Modified:**
- `RuleExistsAsync()` now delegates to `HasRulesForVmAsync()`
- `HasRulesForVmAsync()` improved with:
  - NAT script existence check
  - Better error logging (Debug instead of Warning)
  - Trace logging for successful verifications
  - More defensive error handling

## Impact

### Before:
```
Minute 0: [dbug] NAT check failed → [warn] Missing NAT rules → [info] Configured
Minute 1: [dbug] NAT check failed → [warn] Missing NAT rules → [info] Configured
Minute 2: [dbug] NAT check failed → [warn] Missing NAT rules → [info] Configured
...
(Repeated 60 times per hour per relay VM)
```

### After:
```
Minute 0: [dbug] NAT check failed → [warn] Missing NAT rules → [info] Configured → [trace] Verified
Minute 1-9: (Skipped - within 10 minute cache window)
Minute 10: [trace] All NAT rules verified ✓
Minute 11-19: (Skipped)
Minute 20: [trace] All NAT rules verified ✓
...
(Only 6 checks per hour per relay VM, 90% reduction)
```

## Expected Results

1. **90% Reduction** in NAT rule checking operations
2. **Cleaner Logs** - Expected states use Debug/Trace instead of Warning
3. **Better Performance** - Less `iptables` command overhead
4. **Correct Verification** - All 3 NAT rules checked (PREROUTING, POSTROUTING, FORWARD)
5. **Faster Detection** - Cache ensures we remember healthy state

## Deployment

```bash
cd /opt/decloud/DeCloud.NodeAgent
dotnet build -c Release
sudo ./install.sh
sudo systemctl restart decloud-nodeagent

# Monitor logs - should see much less NAT checking
sudo journalctl -u decloud-nodeagent -f | grep -i nat
```

## Monitoring

After deployment, verify:

```bash
# Should see NAT checks only every 10 minutes (not every minute)
sudo journalctl -u decloud-nodeagent --since "10 minutes ago" | grep "NAT check"

# Should see mostly Debug/Trace instead of Warning
sudo journalctl -u decloud-nodeagent --since "10 minutes ago" | grep -E "NAT|rules" | grep -v dbug | grep -v trace

# Verify relay VMs are working
sudo virsh list | grep relay
```

## Backward Compatibility

✅ All public interface methods unchanged
✅ Existing callers continue to work
✅ `RuleExistsAsync()` still available (now delegates to better method)
✅ No breaking changes to NAT script interface
