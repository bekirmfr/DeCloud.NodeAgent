# DeCloud CLI - Design Document

## Overview

The DeCloud CLI is a unified command-line interface for managing DeCloud node agent operations. It consolidates functionality from multiple existing tools into a single, cohesive interface following industry-standard CLI design patterns.

## Design Principles

### 1. Security First

**Authentication & Authorization**
- Root/sudo required only for operations that modify system state
- Read-only operations (status, info, logs) accessible to non-root users
- Credentials stored in protected files (`/etc/decloud/credentials`) with `600` permissions
- API calls use secure token-based authentication
- Sensitive information (API keys) partially masked in output

**Validation**
- Input validation for all user-provided parameters
- VM ID format validation
- Safe handling of shell arguments (prevent injection)
- Proper error handling for all external commands

**Audit Trail**
- All operations logged via journalctl
- Command execution logged with user context
- Destructive operations require confirmation (unless `--force`)

### 2. KISS (Keep It Simple, Stupid)

**Single Binary**
- All functionality in one shell script
- No external dependencies beyond standard Linux tools
- Self-contained and portable

**Clear Command Structure**
```
decloud <command> [subcommand] [options]
```

**Consistent Patterns**
- Predictable command naming (list, info, cleanup)
- Standard option flags (-f, --force, -n, --lines)
- Uniform output formatting

**Minimal Configuration**
- Uses standard paths (`/etc/decloud/`, `/var/lib/libvirt/decloud-vms/`)
- Environment variables for overrides
- No complex config files

### 3. Industry Best Practices

**Inspired by Docker CLI**
- `decloud <resource> <action>` pattern (e.g., `decloud vm list`)
- Consistent subcommand structure
- JSON output for machine parsing
- Human-readable tables for interactive use

**Inspired by kubectl**
- Resource-oriented commands
- Short and long aliases (e.g., `ls` and `list`)
- Rich output formatting
- Diagnostic commands

**Inspired by systemctl**
- Service management (start, stop, restart)
- Log viewing with follow mode
- Status checks with detailed output

**Exit Codes**
```
0  - Success
1  - General error
2  - Usage error
3  - Authentication error
4  - API error
```

**Error Handling**
- Clear error messages
- Helpful suggestions for resolution
- Graceful degradation
- Proper exit codes

## Architecture

### Command Structure

```
decloud
├── Authentication
│   ├── login       -> delegates to cli-decloud-node
│   └── logout      -> delegates to cli-decloud-node
│
├── Information
│   ├── status      -> comprehensive status display
│   ├── info        -> node details from API
│   ├── resources   -> CPU/memory/storage/network
│   └── heartbeat   -> last heartbeat data
│
├── VM Management
│   ├── vm list     -> list all VMs
│   ├── vm info     -> VM details
│   └── vm cleanup  -> delegates to vm-cleanup.sh
│
├── Service
│   ├── start       -> systemctl start
│   ├── stop        -> systemctl stop
│   ├── restart     -> systemctl restart
│   └── logs        -> journalctl with options
│
└── Diagnostics
    ├── diagnose    -> comprehensive health check
    └── test-api    -> API endpoint testing
```

### Integration Points

**1. Python CLI (cli-decloud-node)**
- Used for: login, logout
- Why: Wallet authentication requires WalletConnect libraries
- How: `exec` delegation preserves user context

**2. vm-cleanup.sh**
- Used for: VM cleanup operations
- Why: Complex libvirt operations already implemented
- How: `exec` delegation with option pass-through

**3. Node Agent API**
- Used for: All runtime information
- Why: Single source of truth for node state
- How: `curl` with error handling

**4. systemd**
- Used for: Service management
- Why: Standard Linux service control
- How: `systemctl` commands

**5. journald**
- Used for: Log viewing
- Why: Centralized logging system
- How: `journalctl` with filters

### Function Organization

```bash
# Global Constants
VERSION, PATHS, COLORS, EXIT_CODES

# Logging
log_info(), log_success(), log_warn(), log_error(), log_debug()

# Utilities
check_root(), check_authenticated(), check_dependencies()
get_node_id(), get_machine_id()

# API Layer
api_call(), api_get(), api_post(), api_delete()

# Command Handlers
cmd_login(), cmd_logout()
cmd_status(), cmd_info(), cmd_resources(), cmd_heartbeat()
cmd_vm_list(), cmd_vm_info(), cmd_vm_cleanup()
cmd_start(), cmd_stop(), cmd_restart(), cmd_logs()
cmd_diagnose(), cmd_test_api()

# Main
main() - command router and option parsing
```

## Design Decisions

### 1. Why Bash?

**Pros:**
- Universal availability on Linux systems
- No compilation required
- Easy to modify and debug
- Direct system integration
- Standard tool for system administration

**Cons:**
- Not type-safe
- Less robust error handling than compiled languages
- Performance (not critical for CLI tool)

**Decision:** Bash is the right choice for a system administration CLI tool.

### 2. Why Delegate to Existing Tools?

**Authentication (cli-decloud-node):**
- Wallet signing requires Python libraries (web3, eth-account)
- QR code generation requires PIL/pillow
- Already implemented and tested
- Rewriting in Bash would be complex and error-prone

**VM Cleanup (vm-cleanup.sh):**
- Complex libvirt operations with multiple edge cases
- Backup functionality
- Already handles dry-run, force, verbose modes
- Well-tested in production

**Decision:** Use the best tool for each job. Don't reimplement working solutions.

### 3. Why JSON Output?

**Benefits:**
- Machine-parseable for automation
- Consistent structure
- Standard format
- Can be piped to `jq` for filtering

**Implementation:**
- API returns JSON natively
- Pass through to user
- Optionally format with `jq` if available
- Provide table views for common cases (VM list)

### 4. Why Two-Level Commands?

**Pattern:**
```bash
decloud <resource> <action>
```

**Examples:**
- `decloud vm list`
- `decloud vm cleanup`
- `decloud vm info`

**Benefits:**
- Clear resource organization
- Scalable (easy to add new resources)
- Matches user mental model
- Consistent with Docker/kubectl

### 5. Why Separate Status and Info?

**`status`** - Quick health check
- Is node authenticated?
- Is service running?
- Is API responding?
- Can reach orchestrator?
- How many VMs?

**`info`** - Detailed information
- Full node configuration
- Hardware details
- Network configuration
- Performance metrics

**Decision:** Different use cases require different levels of detail.

## Security Considerations

### Credential Storage

**File:** `/etc/decloud/credentials`

**Permissions:** `600` (owner read/write only)

**Contents:**
```
NODE_ID=uuid
API_KEY=secret
WALLET_ADDRESS=0x...
```

**Security Measures:**
1. Only root can read/write
2. Not stored in environment variables
3. Not logged
4. Partially masked in output
5. Deleted on logout

### Command Injection Prevention

**All user input validated:**
```bash
# VM ID validation
if [[ -z "$vm_id" ]]; then
    log_error "Invalid VM ID"
    exit 1
fi

# No direct eval or exec of user input
# All variables quoted: "$vm_id", not $vm_id
```

### Privilege Separation

**Root required for:**
- login/logout (writes `/etc/decloud/`)
- VM cleanup (libvirt operations)
- Service control (systemctl)

**No root required for:**
- status, info, resources
- VM list, info
- logs, diagnostics

**Implementation:**
```bash
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This command requires root privileges"
        exit 1
    fi
}
```

## Error Handling Strategy

### Levels of Error Handling

**1. Input Validation**
```bash
if [[ -z "$vm_id" ]]; then
    log_error "VM ID required"
    exit $EXIT_USAGE
fi
```

**2. Dependency Checks**
```bash
check_dependencies virsh jq curl
```

**3. State Verification**
```bash
check_authenticated
check_service_running
```

**4. API Error Handling**
```bash
if response=$(api_get "/endpoint"); then
    echo "$response"
else
    log_error "API call failed"
    exit $EXIT_API_ERROR
fi
```

**5. External Command Errors**
```bash
set -euo pipefail  # Exit on error, undefined variable, pipe failure

trap 'log_error "Command failed"' ERR
```

### User-Friendly Messages

**Bad:**
```
Error: 404
```

**Good:**
```
ERROR: Node not authenticated
Run: decloud login
```

**Best:**
```
ERROR: Node not authenticated

To authenticate your node, run:
  sudo decloud login

This will guide you through wallet-based authentication.
```

## Testing Strategy

### Unit Testing

**Approach:**
- Test individual functions in isolation
- Mock external dependencies (API, systemctl)
- Validate input/output

**Example:**
```bash
# Test log functions
test_log_info() {
    output=$(log_info "test message")
    assert_contains "$output" "[INFO]"
    assert_contains "$output" "test message"
}
```

### Integration Testing

**Approach:**
- Test command end-to-end
- Verify file creation/modification
- Check exit codes

**Example:**
```bash
# Test status command
test_status_not_authenticated() {
    rm -f /etc/decloud/credentials
    output=$(decloud status 2>&1)
    assert_contains "$output" "Not authenticated"
}
```

### Smoke Testing

**Quick validation:**
```bash
make test
```

**Tests:**
1. `--version` returns version
2. `--help` displays help
3. Invalid command shows error
4. Missing subcommand shows error

## Extensibility

### Adding New Commands

**1. Add command function:**
```bash
cmd_new_feature() {
    # Implementation
}
```

**2. Add to command router:**
```bash
case "$command" in
    new-feature)
        cmd_new_feature "$@"
        ;;
esac
```

**3. Update help text**

**4. Add tests**

### Adding New VM Subcommands

```bash
case "$vm_command" in
    new-action)
        cmd_vm_new_action "$@"
        ;;
esac
```

### Adding API Endpoints

**Wrapper pattern:**
```bash
cmd_new_info() {
    if data=$(api_get "/api/new/endpoint"); then
        echo "$data" | jq -C '.'
    else
        log_error "Failed to fetch data"
        exit $EXIT_API_ERROR
    fi
}
```

## Performance Considerations

### Startup Time

**Target:** < 100ms for simple commands

**Optimizations:**
1. No unnecessary sourcing of files
2. Lazy evaluation of expensive checks
3. Early exit on errors
4. Minimal external command invocations

### API Calls

**Strategy:**
- Single API call per command (when possible)
- Timeout on API calls (via curl)
- Cache responses for batch operations (future)

### Large Outputs

**Handling:**
- Pipe through `less` for large outputs (future)
- Pagination for long lists (future)
- Streaming for log following (already implemented)

## Future Enhancements

### Phase 2 Features

1. **Autocomplete**
   - Bash completion for commands and options
   - VM ID completion from API

2. **Configuration File**
   - `~/.decloud/config` for user preferences
   - Custom output formats
   - Default options

3. **Batch Operations**
   - `decloud vm cleanup --pattern "test-*"`
   - `decloud vm list --state running`

4. **Enhanced Output**
   - `--output json|yaml|table`
   - Color customization
   - Quiet mode (`-q`)

5. **Monitoring Integration**
   - Export metrics to Prometheus format
   - Health check endpoint
   - Alerting hooks

### Backward Compatibility

**Promise:**
- Semantic versioning (MAJOR.MINOR.PATCH)
- No breaking changes in minor/patch versions
- Deprecation warnings before removal
- Graceful degradation for old node agents

## Conclusion

The DeCloud CLI is designed to be:
- **Secure** - Authentication, validation, privilege separation
- **Simple** - Single binary, clear commands, KISS principle
- **Standard** - Follows industry best practices
- **Extensible** - Easy to add new features
- **Maintainable** - Well-organized, documented code

It consolidates multiple tools into a unified interface while preserving the strengths of each component. The design prioritizes security, usability, and operational excellence.
