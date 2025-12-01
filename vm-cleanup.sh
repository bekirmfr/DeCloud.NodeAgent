#!/bin/bash
#
# DeCloud Node Agent - VM Cleanup Script
# Version: 1.0.0
#
# Safely removes VMs from libvirt and deletes all associated files.
# Supports single VM cleanup or bulk cleanup of all VMs.
#
# Security Features:
# - Requires explicit confirmation for destructive operations
# - Dry-run mode to preview changes
# - Comprehensive logging
# - Validates VM existence before deletion
# - Graceful error handling
# - Backup option for VM configurations
#
# Usage:
#   ./vm-cleanup.sh --vm <vm-id>              # Clean up single VM
#   ./vm-cleanup.sh --all                      # Clean up all VMs
#   ./vm-cleanup.sh --vm <vm-id> --dry-run    # Preview without executing
#   ./vm-cleanup.sh --all --force              # Skip confirmation
#   ./vm-cleanup.sh --vm <vm-id> --backup     # Backup before deletion
#

set -euo pipefail

# ============================================================
# Configuration
# ============================================================
SCRIPT_VERSION="1.0.0"
VM_STORAGE_PATH="${VM_STORAGE_PATH:-/var/lib/decloud/vms}"
IMAGE_CACHE_PATH="${IMAGE_CACHE_PATH:-/var/lib/decloud/images}"
BACKUP_PATH="${BACKUP_PATH:-/var/lib/decloud/backups}"
LOG_FILE="${LOG_FILE:-/var/log/decloud/vm-cleanup.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Flags
DRY_RUN=false
FORCE=false
CREATE_BACKUP=false
VERBOSE=false
VM_ID=""
CLEANUP_ALL=false

# Statistics
TOTAL_VMS=0
SUCCESSFUL_CLEANUPS=0
FAILED_CLEANUPS=0

# ============================================================
# Logging Functions
# ============================================================
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Log to console with colors
    case "$level" in
        INFO)
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        SUCCESS)
            echo -e "${GREEN}[✓]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[!]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[✗]${NC} $message"
            ;;
        DEBUG)
            if [ "$VERBOSE" = true ]; then
                echo -e "${CYAN}[DEBUG]${NC} $message"
            fi
            ;;
        STEP)
            echo -e "${MAGENTA}[STEP]${NC} $message"
            ;;
    esac
}

# ============================================================
# Helper Functions
# ============================================================
print_banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         DeCloud VM Cleanup Script v${SCRIPT_VERSION}              ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --vm <vm-id>        Clean up a specific VM by ID
    --all               Clean up all VMs (use with caution!)
    --dry-run           Preview actions without executing them
    --force             Skip confirmation prompts (dangerous!)
    --backup            Create backup before deletion
    --verbose           Enable verbose logging
    --help              Show this help message

Examples:
    # Preview cleanup of a single VM
    $0 --vm abc123-def456 --dry-run

    # Clean up a single VM with backup
    $0 --vm abc123-def456 --backup

    # Clean up all VMs (with confirmation)
    $0 --all

    # Force cleanup all VMs without confirmation (dangerous!)
    $0 --all --force

Environment Variables:
    VM_STORAGE_PATH     Path to VM storage (default: /var/lib/decloud/vms)
    IMAGE_CACHE_PATH    Path to image cache (default: /var/lib/decloud/images)
    BACKUP_PATH         Path for backups (default: /var/lib/decloud/backups)
    LOG_FILE            Path to log file (default: /var/log/decloud/vm-cleanup.log)

EOF
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log ERROR "This script must be run as root or with sudo"
        echo ""
        echo "Please run with: sudo $0 $*"
        exit 1
    fi
}

check_dependencies() {
    local missing_deps=()
    
    local required_commands=("virsh" "rm" "find" "tar")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log ERROR "Missing required dependencies: ${missing_deps[*]}"
        log ERROR "Please install libvirt-clients and basic utilities"
        exit 1
    fi
    
    # Check if libvirtd is running
    if ! systemctl is-active --quiet libvirtd; then
        log ERROR "libvirtd service is not running"
        log ERROR "Start it with: sudo systemctl start libvirtd"
        exit 1
    fi
    
    log DEBUG "All dependencies satisfied"
}

confirm_action() {
    local message="$1"
    
    if [ "$FORCE" = true ]; then
        log WARN "Skipping confirmation (--force flag)"
        return 0
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log INFO "Dry-run mode - no confirmation needed"
        return 0
    fi
    
    echo ""
    echo -e "${YELLOW}⚠️  WARNING: This action is destructive!${NC}"
    echo -e "${YELLOW}$message${NC}"
    echo ""
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirmation
    
    if [ "$confirmation" != "yes" ]; then
        log WARN "Operation cancelled by user"
        exit 0
    fi
    
    log INFO "User confirmed action"
}

get_vm_state() {
    local vm_id="$1"
    
    # Check if VM exists in libvirt
    if ! virsh dominfo "$vm_id" &>/dev/null; then
        echo "not_found"
        return 0
    fi
    
    # Get VM state
    virsh domstate "$vm_id" 2>/dev/null | tr '[:upper:]' '[:lower:]' || echo "unknown"
}

list_all_vms() {
    # List all VMs from libvirt
    virsh list --all --uuid 2>/dev/null | grep -v '^$' || true
}

# ============================================================
# Backup Functions
# ============================================================
backup_vm() {
    local vm_id="$1"
    
    if [ "$CREATE_BACKUP" != true ]; then
        log DEBUG "Backup not requested, skipping"
        return 0
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log INFO "[DRY-RUN] Would create backup of VM: $vm_id"
        return 0
    fi
    
    log STEP "Creating backup of VM: $vm_id"
    
    local backup_dir="$BACKUP_PATH/vm-$vm_id-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup VM directory
    local vm_dir="$VM_STORAGE_PATH/$vm_id"
    if [ -d "$vm_dir" ]; then
        log INFO "Backing up VM directory..."
        tar -czf "$backup_dir/vm-files.tar.gz" -C "$VM_STORAGE_PATH" "$vm_id" 2>/dev/null || {
            log WARN "Failed to backup VM directory"
        }
    fi
    
    # Backup libvirt XML configuration
    if virsh dominfo "$vm_id" &>/dev/null; then
        log INFO "Backing up libvirt XML configuration..."
        virsh dumpxml "$vm_id" > "$backup_dir/domain.xml" 2>/dev/null || {
            log WARN "Failed to backup libvirt XML"
        }
    fi
    
    log SUCCESS "Backup created at: $backup_dir"
}

# ============================================================
# VM Cleanup Functions
# ============================================================
cleanup_vm() {
    local vm_id="$1"
    
    log STEP "Processing VM: $vm_id"
    
    # Validate VM ID format (should be a valid UUID or similar)
    if [ -z "$vm_id" ]; then
        log ERROR "Invalid VM ID (empty)"
        return 1
    fi
    
    # Get current VM state
    local vm_state=$(get_vm_state "$vm_id")
    log INFO "VM state: $vm_state"
    
    # Create backup if requested
    if [ "$CREATE_BACKUP" = true ]; then
        backup_vm "$vm_id"
    fi
    
    # Step 1: Stop VM if running
    if [ "$vm_state" = "running" ] || [ "$vm_state" = "paused" ]; then
        if [ "$DRY_RUN" = true ]; then
            log INFO "[DRY-RUN] Would forcefully stop (destroy) VM: $vm_id"
        else
            log INFO "Stopping VM..."
            if virsh destroy "$vm_id" 2>/dev/null; then
                log SUCCESS "VM stopped successfully"
                sleep 2  # Give libvirt time to clean up
            else
                log WARN "Failed to stop VM (may already be stopped)"
            fi
        fi
    fi
    
    # Step 2: Undefine VM from libvirt
    if [ "$vm_state" != "not_found" ]; then
        if [ "$DRY_RUN" = true ]; then
            log INFO "[DRY-RUN] Would undefine VM from libvirt with: virsh undefine $vm_id --remove-all-storage --nvram"
        else
            log INFO "Undefining VM from libvirt..."
            
            # Try with all options first
            if virsh undefine "$vm_id" --remove-all-storage --nvram 2>/dev/null; then
                log SUCCESS "VM undefined successfully (with storage removal)"
            elif virsh undefine "$vm_id" --remove-all-storage 2>/dev/null; then
                log SUCCESS "VM undefined successfully (with storage removal, no NVRAM)"
            elif virsh undefine "$vm_id" 2>/dev/null; then
                log SUCCESS "VM undefined successfully (manual storage cleanup needed)"
            else
                log WARN "Failed to undefine VM from libvirt (may not exist)"
            fi
        fi
    else
        log INFO "VM not found in libvirt, skipping undefine"
    fi
    
    # Step 3: Delete VM directory and all files
    local vm_dir="$VM_STORAGE_PATH/$vm_id"
    
    if [ -d "$vm_dir" ]; then
        if [ "$DRY_RUN" = true ]; then
            log INFO "[DRY-RUN] Would delete VM directory: $vm_dir"
            log INFO "[DRY-RUN] Files that would be deleted:"
            find "$vm_dir" -type f -exec ls -lh {} \; 2>/dev/null | sed 's/^/  /' || true
        else
            log INFO "Deleting VM directory: $vm_dir"
            
            # List files being deleted for logging
            if [ "$VERBOSE" = true ]; then
                log DEBUG "Files being deleted:"
                find "$vm_dir" -type f -exec ls -lh {} \; 2>/dev/null | sed 's/^/  /' || true
            fi
            
            # Delete the directory
            if rm -rf "$vm_dir" 2>/dev/null; then
                log SUCCESS "VM directory deleted successfully"
            else
                log ERROR "Failed to delete VM directory: $vm_dir"
                return 1
            fi
        fi
    else
        log INFO "VM directory not found: $vm_dir"
    fi
    
    # Step 4: Clean up any orphaned libvirt storage volumes
    if [ "$DRY_RUN" != true ]; then
        log DEBUG "Checking for orphaned storage volumes..."
        
        # List all storage volumes and check for this VM ID
        local orphaned_volumes=$(virsh vol-list default 2>/dev/null | grep "$vm_id" | awk '{print $1}' || true)
        
        if [ -n "$orphaned_volumes" ]; then
            log WARN "Found orphaned storage volumes:"
            echo "$orphaned_volumes" | while read -r vol; do
                log INFO "  - $vol"
                if virsh vol-delete "$vol" --pool default 2>/dev/null; then
                    log SUCCESS "  Deleted orphaned volume: $vol"
                else
                    log WARN "  Failed to delete orphaned volume: $vol"
                fi
            done
        fi
    fi
    
    log SUCCESS "VM cleanup completed: $vm_id"
    return 0
}

cleanup_all_vms() {
    log STEP "Cleaning up ALL VMs"
    
    # Get list of all VMs
    local all_vms=$(list_all_vms)
    
    if [ -z "$all_vms" ]; then
        log INFO "No VMs found in libvirt"
        TOTAL_VMS=0
    else
        TOTAL_VMS=$(echo "$all_vms" | wc -l)
        log INFO "Found $TOTAL_VMS VMs in libvirt"
    fi
    
    # Also check for VMs in the storage directory
    local vm_dirs=()
    if [ -d "$VM_STORAGE_PATH" ]; then
        while IFS= read -r dir; do
            local vm_id=$(basename "$dir")
            # Only add if not already in the libvirt list
            if ! echo "$all_vms" | grep -q "^$vm_id$"; then
                vm_dirs+=("$vm_id")
            fi
        done < <(find "$VM_STORAGE_PATH" -mindepth 1 -maxdepth 1 -type d)
        
        if [ ${#vm_dirs[@]} -gt 0 ]; then
            log INFO "Found ${#vm_dirs[@]} additional VM directories not in libvirt"
            TOTAL_VMS=$((TOTAL_VMS + ${#vm_dirs[@]}))
        fi
    fi
    
    if [ $TOTAL_VMS -eq 0 ]; then
        log INFO "No VMs to clean up"
        return 0
    fi
    
    # Confirm with user
    confirm_action "This will DELETE ALL $TOTAL_VMS VMs and their data permanently!"
    
    # Clean up VMs from libvirt
    if [ -n "$all_vms" ]; then
        echo "$all_vms" | while read -r vm_id; do
            if [ -n "$vm_id" ]; then
                if cleanup_vm "$vm_id"; then
                    SUCCESSFUL_CLEANUPS=$((SUCCESSFUL_CLEANUPS + 1))
                else
                    FAILED_CLEANUPS=$((FAILED_CLEANUPS + 1))
                fi
            fi
        done
    fi
    
    # Clean up orphaned VM directories
    if [ ${#vm_dirs[@]} -gt 0 ]; then
        for vm_id in "${vm_dirs[@]}"; do
            if cleanup_vm "$vm_id"; then
                SUCCESSFUL_CLEANUPS=$((SUCCESSFUL_CLEANUPS + 1))
            else
                FAILED_CLEANUPS=$((FAILED_CLEANUPS + 1))
            fi
        done
    fi
}

# ============================================================
# Argument Parsing
# ============================================================
parse_arguments() {
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --vm)
                VM_ID="$2"
                shift 2
                ;;
            --all)
                CLEANUP_ALL=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --backup)
                CREATE_BACKUP=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                log ERROR "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Validate arguments
    if [ "$CLEANUP_ALL" = false ] && [ -z "$VM_ID" ]; then
        log ERROR "Must specify either --vm <vm-id> or --all"
        print_usage
        exit 1
    fi
    
    if [ "$CLEANUP_ALL" = true ] && [ -n "$VM_ID" ]; then
        log ERROR "Cannot specify both --vm and --all"
        print_usage
        exit 1
    fi
}

# ============================================================
# Main Execution
# ============================================================
main() {
    print_banner
    
    parse_arguments "$@"
    
    log INFO "DeCloud VM Cleanup Script v$SCRIPT_VERSION"
    log INFO "Started at: $(date)"
    
    if [ "$DRY_RUN" = true ]; then
        log WARN "DRY-RUN MODE: No actual changes will be made"
    fi
    
    # Security checks
    check_root
    check_dependencies
    
    log INFO "VM Storage Path: $VM_STORAGE_PATH"
    log INFO "Backup Path: $BACKUP_PATH"
    log INFO "Log File: $LOG_FILE"
    echo ""
    
    # Execute cleanup
    if [ "$CLEANUP_ALL" = true ]; then
        cleanup_all_vms
    else
        TOTAL_VMS=1
        
        # Check if VM exists
        local vm_state=$(get_vm_state "$VM_ID")
        if [ "$vm_state" = "not_found" ] && [ ! -d "$VM_STORAGE_PATH/$VM_ID" ]; then
            log ERROR "VM not found: $VM_ID"
            log ERROR "VM does not exist in libvirt or storage directory"
            exit 1
        fi
        
        # Confirm with user
        confirm_action "This will DELETE VM '$VM_ID' and all its data permanently!"
        
        if cleanup_vm "$VM_ID"; then
            SUCCESSFUL_CLEANUPS=1
        else
            FAILED_CLEANUPS=1
        fi
    fi
    
    # Print summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    Cleanup Summary                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    log INFO "Total VMs processed: $TOTAL_VMS"
    log SUCCESS "Successful cleanups: $SUCCESSFUL_CLEANUPS"
    if [ $FAILED_CLEANUPS -gt 0 ]; then
        log ERROR "Failed cleanups: $FAILED_CLEANUPS"
    fi
    echo ""
    log INFO "Log file: $LOG_FILE"
    log INFO "Completed at: $(date)"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${CYAN}This was a dry-run. No actual changes were made.${NC}"
        echo -e "${CYAN}Run without --dry-run to execute the cleanup.${NC}"
        echo ""
    fi
    
    # Exit with error if any cleanups failed
    if [ $FAILED_CLEANUPS -gt 0 ]; then
        exit 1
    fi
}

# Run main function
main "$@"
