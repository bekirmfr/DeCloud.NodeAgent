#!/usr/bin/env python3
"""
DeCloud Relay VM Management API
Version: 2.1.0 - IMPROVED

Provides REST API for relay monitoring, CGNAT peer management,
and serves the dashboard interface.

IMPROVEMENTS v2.1.0:
- Added peer age tracking to prevent premature cleanup of new peers
- Unified stale threshold across background and manual cleanup
- Aligned with orchestrator 30-second grace period
- Better logging for peer lifecycle events
"""

import json
import subprocess
import os
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys

# ==================== Configuration ====================
RELAY_ID = "__VM_ID__"
RELAY_NAME = "__VM_NAME__"
RELAY_REGION = "__RELAY_REGION__"
RELAY_CAPACITY = int("__RELAY_CAPACITY__")
WIREGUARD_INTERFACE = "wg-relay-server"
STATIC_DIR = "/opt/decloud-relay/static"
LISTEN_PORT = 8080

# ==================== Periodic Cleanup Configuration ====================
CLEANUP_ENABLED = True                    # Enable/disable automatic cleanup
CLEANUP_INTERVAL_SECONDS = 300            # 5 minutes (300 seconds)

# Unified stale threshold for both background and manual cleanup
STALE_THRESHOLD_SECONDS = 300             # 5 minutes - peer with no handshake for this long is stale

# Grace period for newly-added peers
# Peers registered within this time window are never removed, even with no handshake
# This aligns with orchestrator's 30-second grace period and provides buffer
PEER_REGISTRATION_GRACE_PERIOD = 60       # 60 seconds - gives time for client to connect

ORCHESTRATOR_IP = "10.20.0.1/32"          # Never remove orchestrator peer

# ==================== Logging ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/decloud-relay-api.log')
    ]
)
logger = logging.getLogger('relay-api')

# ==================== Global State ====================
cleanup_stats = {
    'last_run': None,
    'total_runs': 0,
    'total_removed': 0,
    'last_removed_count': 0,
    'errors': 0
}

# Track when peers were registered (public_key -> timestamp)
peer_registration_times = {}
peer_registration_lock = threading.Lock()

# ==================== Helper Functions ====================

def record_peer_registration(public_key):
    """Record timestamp when peer is registered"""
    with peer_registration_lock:
        peer_registration_times[public_key] = time.time()
        logger.debug(f"📝 Recorded registration time for peer {public_key[:16]}...")

def get_peer_age(public_key, current_time):
    """Get age of peer in seconds since registration"""
    with peer_registration_lock:
        registration_time = peer_registration_times.get(public_key)
        
        if registration_time is None:
            # Peer was registered before we started tracking (or tracking failed)
            # Assume it's old enough to evaluate normally
            return float('inf')
        
        return current_time - registration_time

def cleanup_stale_peer_records():
    """Remove registration records for peers that no longer exist"""
    with peer_registration_lock:
        # Get current peers
        try:
            result = subprocess.run(
                ['wg', 'show', WIREGUARD_INTERFACE, 'dump'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return
            
            lines = result.stdout.strip().split('\n')
            current_peers = set()
            
            for line in lines[1:]:  # Skip interface line
                if not line.strip():
                    continue
                fields = line.split('\t')
                if len(fields) >= 1:
                    current_peers.add(fields[0])
            
            # Remove records for peers that no longer exist
            stale_records = [pk for pk in peer_registration_times.keys() if pk not in current_peers]
            
            for pk in stale_records:
                del peer_registration_times[pk]
                logger.debug(f"🧹 Cleaned up registration record for removed peer {pk[:16]}...")
                
        except Exception as e:
            logger.debug(f"Error cleaning up peer records: {e}")

# ==================== Background Cleanup Thread ====================
class PeerCleanupThread(threading.Thread):
    """Background daemon thread for periodic peer cleanup"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.name = "PeerCleanupThread"
    
    def run(self):
        """Main cleanup loop - runs in background"""
        logger.info("🧹 Peer cleanup thread started")
        logger.info(f"   Cleanup interval: {CLEANUP_INTERVAL_SECONDS}s ({CLEANUP_INTERVAL_SECONDS // 60} minutes)")
        logger.info(f"   Stale threshold: {STALE_THRESHOLD_SECONDS}s ({STALE_THRESHOLD_SECONDS // 60} minutes)")
        logger.info(f"   ✅ NEW: Grace period: {PEER_REGISTRATION_GRACE_PERIOD}s for newly-registered peers")
        
        # Wait 30 seconds before first run (let server stabilize)
        time.sleep(30)
        
        while not self.stop_event.is_set():
            try:
                self.cleanup_stale_peers()
                cleanup_stale_peer_records()  # Clean up registration tracking
                
                # Wait for next interval or until stopped
                self.stop_event.wait(CLEANUP_INTERVAL_SECONDS)
                
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}", exc_info=True)
                cleanup_stats['errors'] += 1
                # Wait a bit before retrying
                self.stop_event.wait(60)
    
    def stop(self):
        """Stop the cleanup thread gracefully"""
        logger.info("Stopping peer cleanup thread...")
        self.stop_event.set()
    
    def cleanup_stale_peers(self):
        """
        Remove peers with no handshake or stale handshake
        Respects grace period for newly-registered peers
        """
        try:
            current_time = int(time.time())
            
            # Get current peers from WireGuard
            result = subprocess.run(
                ['wg', 'show', WIREGUARD_INTERFACE, 'dump'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to get WireGuard peers: {result.stderr}")
                return
            
            lines = result.stdout.strip().split('\n')
            
            removed = []
            kept = []
            skipped_new = []  # Track peers skipped due to grace period
            
            # Skip interface line (first line), process peers
            for line in lines[1:]:
                if not line.strip():
                    continue
                
                fields = line.split('\t')
                if len(fields) < 7:
                    continue
                
                public_key = fields[0]
                allowed_ips = fields[3]
                latest_handshake = int(fields[4]) if fields[4] else 0
                
                # Never remove orchestrator peer
                if ORCHESTRATOR_IP in allowed_ips:
                    kept.append({
                        'public_key': public_key[:16] + '...',
                        'allowed_ips': allowed_ips,
                        'reason': 'orchestrator'
                    })
                    continue
                
                # Check if peer is within grace period
                peer_age = get_peer_age(public_key, current_time)
                
                if peer_age < PEER_REGISTRATION_GRACE_PERIOD:
                    skipped_new.append({
                        'public_key': public_key[:16] + '...',
                        'allowed_ips': allowed_ips,
                        'age_seconds': int(peer_age)
                    })
                    logger.debug(
                        f"⏱️  Skipping new peer {public_key[:16]}... "
                        f"(age: {peer_age:.0f}s < {PEER_REGISTRATION_GRACE_PERIOD}s grace period)"
                    )
                    continue
                
                # Calculate time since last handshake
                if latest_handshake == 0:
                    time_since_handshake = float('inf')
                    handshake_status = 'never'
                else:
                    time_since_handshake = current_time - latest_handshake
                    handshake_status = f'{time_since_handshake}s ago'
                
                # Check if peer is stale (and past grace period)
                if time_since_handshake > STALE_THRESHOLD_SECONDS:
                    # Remove stale peer
                    remove_result = subprocess.run(
                        ['wg', 'set', WIREGUARD_INTERFACE, 'peer', public_key, 'remove'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if remove_result.returncode == 0:
                        removed.append({
                            'public_key': public_key[:16] + '...',
                            'allowed_ips': allowed_ips,
                            'handshake': handshake_status,
                            'peer_age_seconds': int(peer_age) if peer_age != float('inf') else 'unknown'
                        })
                        logger.info(
                            f"🗑️  Removed stale peer: {public_key[:16]}... "
                            f"({allowed_ips}, handshake: {handshake_status}, peer age: {peer_age:.0f}s)"
                        )
                    else:
                        logger.error(
                            f"Failed to remove peer {public_key[:16]}...: "
                            f"{remove_result.stderr}"
                        )
                else:
                    # Peer is active
                    kept.append({
                        'public_key': public_key[:16] + '...',
                        'allowed_ips': allowed_ips,
                        'handshake': handshake_status
                    })
            
            # Save WireGuard configuration if we removed any peers
            if removed:
                try:
                    subprocess.run(
                        ['wg-quick', 'save', WIREGUARD_INTERFACE],
                        capture_output=True,
                        timeout=5
                    )
                    logger.info("💾 WireGuard configuration saved after cleanup")
                except Exception as e:
                    logger.warning(f"Failed to save WireGuard config: {e}")
            
            # Update statistics
            cleanup_stats['last_run'] = time.time()
            cleanup_stats['total_runs'] += 1
            cleanup_stats['total_removed'] += len(removed)
            cleanup_stats['last_removed_count'] = len(removed)
            
            # Log summary
            if removed or skipped_new:
                logger.info(
                    f"✅ Cleanup completed: removed {len(removed)} stale peers, "
                    f"kept {len(kept)} active peers, skipped {len(skipped_new)} new peers (grace period)"
                )
            else:
                logger.debug(
                    f"✅ Cleanup completed: no stale peers found, "
                    f"{len(kept)} active peers"
                )
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Cleanup timed out")
            cleanup_stats['errors'] += 1
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}", exc_info=True)
            cleanup_stats['errors'] += 1


# ==================== Request Handler ====================
class RelayAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for relay management API"""
    
    # Suppress default logging
    def log_message(self, format, *args):
        pass  # Disable default request logging
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Log request
        logger.debug(f"GET {path}")
        
        if path == '/health':
            self.health_check()
        elif path == '/api/relay/status':
            self.get_relay_status()
        elif path == '/api/relay/wireguard':
            status = self.get_wireguard_status()
            self.send_json_response(status)
        elif path == '/api/relay/cleanup/stats':
            self.get_cleanup_stats()
        elif path == '/' or path == '/index.html':
            self.serve_dashboard()
        elif path.startswith('/static/'):
            self.serve_static_file(path)
        else:
            self.send_error_response(404, 'Not Found', f'Path {path} not found')
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        logger.info(f"POST {path}")
        
        if path == '/api/relay/add-peer':
            self.add_cgnat_peer()
        elif path == '/api/relay/remove-peer':
            self.remove_cgnat_peer()
        elif path == '/api/relay/cleanup':
            self.manual_cleanup()
        else:
            self.send_error_response(404, 'Not Found', f'Path {path} not found')
    
    # ==================== API Endpoints ====================
    
    def health_check(self):
        """Simple health check"""
        self.send_json_response({
            'status': 'healthy',
            'relay_id': RELAY_ID,
            'timestamp': int(time.time())
        })
    
    def get_relay_status(self):
        """Get comprehensive relay status"""
        try:
            # Get system uptime
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = int(float(f.read().split()[0]))
            
            # Get WireGuard stats
            wg_status = self.get_wireguard_status()
            
            self.send_json_response({
                'relay_id': RELAY_ID,
                'relay_name': RELAY_NAME,
                'region': RELAY_REGION,
                'max_capacity': RELAY_CAPACITY,
                'current_load': wg_status.get('peer_count', 0),
                'available_slots': wg_status.get('available_slots', RELAY_CAPACITY),
                'uptime_seconds': uptime_seconds,
                'wireguard_interface': WIREGUARD_INTERFACE,
                'cleanup_config': {
                    'enabled': CLEANUP_ENABLED,
                    'interval_seconds': CLEANUP_INTERVAL_SECONDS,
                    'stale_threshold_seconds': STALE_THRESHOLD_SECONDS,
                    'grace_period_seconds': PEER_REGISTRATION_GRACE_PERIOD
                },
                'cleanup_stats': cleanup_stats,
                'peer_tracking': {  # Peer registration tracking stats
                    'tracked_peers': len(peer_registration_times),
                    'tracking_enabled': True
                }
            })
        except Exception as e:
            logger.error(f"Error getting relay status: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Error', str(e))
    
    def get_cleanup_stats(self):
        """Get cleanup statistics"""
        self.send_json_response({
            'cleanup_enabled': CLEANUP_ENABLED,
            'interval_seconds': CLEANUP_INTERVAL_SECONDS,
            'stale_threshold_seconds': STALE_THRESHOLD_SECONDS,
            'grace_period_seconds': PEER_REGISTRATION_GRACE_PERIOD,
            'stats': cleanup_stats,
            'tracked_peers_count': len(peer_registration_times)
        })
    
    def add_cgnat_peer(self):
        """
        ✅ IMPROVED: Add CGNAT node as WireGuard peer
        Now records registration timestamp for grace period tracking
        """
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error_response(400, 'Bad Request', 'Request body is empty')
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Validate required fields
            required_fields = ['public_key', 'allowed_ips']
            missing_fields = [f for f in required_fields if f not in data]
            
            if missing_fields:
                self.send_error_response(
                    400,
                    'Missing Required Fields',
                    f"Missing fields: {', '.join(missing_fields)}"
                )
                return
            
            public_key = data['public_key']
            
            # Build wg set command
            cmd = [
                'wg', 'set', WIREGUARD_INTERFACE,
                'peer', public_key,
                'allowed-ips', data['allowed_ips']
            ]
            
            # Add optional persistent keepalive
            if 'persistent_keepalive' in data:
                cmd.extend(['persistent-keepalive', str(data['persistent_keepalive'])])
            
            description = data.get('description', 'CGNAT Node')
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to add peer: {result.stderr}")
                self.send_error_response(
                    500,
                    'Failed to Add Peer',
                    result.stderr or 'Unknown error'
                )
                return
            
            # Save configuration
            try:
                subprocess.run(
                    ['wg-quick', 'save', WIREGUARD_INTERFACE],
                    capture_output=True,
                    timeout=5
                )
            except Exception as e:
                logger.warning(f"Failed to save WireGuard config: {e}")
            
            # Record peer registration timestamp
            record_peer_registration(public_key)
            
            logger.info(f"✅ Added CGNAT peer: {public_key[:16]}... ({description})")
            
            self.send_json_response({
                'success': True,
                'peer_public_key': public_key[:16] + '...',
                'allowed_ips': data['allowed_ips'],
                'description': description
            })
            
        except json.JSONDecodeError as e:
            self.send_error_response(400, 'Invalid JSON', str(e))
        except subprocess.TimeoutExpired:
            self.send_error_response(504, 'Command Timeout', 'WireGuard command timed out')
        except Exception as e:
            logger.error(f"Add peer error: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Error', str(e))
    
    def remove_cgnat_peer(self):
        """Remove CGNAT node peer"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error_response(400, 'Bad Request', 'Request body is empty')
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            if 'public_key' not in data:
                self.send_error_response(400, 'Missing Field', 'public_key is required')
                return
            
            public_key = data['public_key']
            
            # Remove peer
            result = subprocess.run(
                ['wg', 'set', WIREGUARD_INTERFACE, 'peer', public_key, 'remove'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to remove peer: {result.stderr}")
                self.send_error_response(
                    500,
                    'Failed to Remove Peer',
                    result.stderr or 'Unknown error'
                )
                return
            
            # Save configuration
            try:
                subprocess.run(
                    ['wg-quick', 'save', WIREGUARD_INTERFACE],
                    capture_output=True,
                    timeout=5
                )
            except Exception as e:
                logger.warning(f"Failed to save WireGuard config: {e}")
            
            # Clean up registration tracking
            with peer_registration_lock:
                if public_key in peer_registration_times:
                    del peer_registration_times[public_key]
            
            logger.info(f"✅ Removed CGNAT peer: {public_key[:16]}...")
            
            self.send_json_response({
                'success': True,
                'peer_public_key': public_key[:16] + '...'
            })
            
        except json.JSONDecodeError as e:
            self.send_error_response(400, 'Invalid JSON', str(e))
        except subprocess.TimeoutExpired:
            self.send_error_response(504, 'Command Timeout', 'WireGuard command timed out')
        except Exception as e:
            logger.error(f"Remove peer error: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Error', str(e))
    
    def manual_cleanup(self):
        """
        Manually trigger peer cleanup
        Uses consistent STALE_THRESHOLD_SECONDS and respects grace period
        """
        try:
            current_time = int(time.time())
            
            # Get current peers from WireGuard
            result = subprocess.run(
                ['wg', 'show', WIREGUARD_INTERFACE, 'dump'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                self.send_error_response(500, 'WireGuard Error', result.stderr)
                return
            
            lines = result.stdout.strip().split('\n')
            removed = []
            kept = []
            skipped_new = []
            
            # Skip interface line, process peers
            for line in lines[1:]:
                if not line.strip():
                    continue
                
                fields = line.split('\t')
                if len(fields) < 7:
                    continue
                
                public_key = fields[0]
                allowed_ips = fields[3]
                latest_handshake = int(fields[4]) if fields[4] else 0
                
                # Never remove orchestrator peer
                if ORCHESTRATOR_IP in allowed_ips:
                    kept.append({
                        'public_key': public_key[:16] + '...',
                        'allowed_ips': allowed_ips,
                        'reason': 'orchestrator'
                    })
                    continue
                
                # ✅ NEW: Check grace period
                peer_age = get_peer_age(public_key, current_time)
                
                if peer_age < PEER_REGISTRATION_GRACE_PERIOD:
                    skipped_new.append({
                        'public_key': public_key[:16] + '...',
                        'allowed_ips': allowed_ips,
                        'age_seconds': int(peer_age)
                    })
                    continue
                
                # Calculate handshake status
                if latest_handshake == 0:
                    time_since_handshake = float('inf')
                    handshake_status = 'never'
                else:
                    time_since_handshake = current_time - latest_handshake
                    handshake_status = f'{time_since_handshake}s ago'
                
                # Use consistent STALE_THRESHOLD_SECONDS
                if time_since_handshake > STALE_THRESHOLD_SECONDS:
                    # Remove stale peer
                    remove_result = subprocess.run(
                        ['wg', 'set', WIREGUARD_INTERFACE, 'peer', public_key, 'remove'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if remove_result.returncode == 0:
                        removed.append({
                            'public_key': public_key[:16] + '...',
                            'allowed_ips': allowed_ips,
                            'handshake': handshake_status
                        })
                        logger.info(f"🗑️  Removed stale peer: {public_key[:16]}... ({allowed_ips})")
                    else:
                        logger.error(f"Failed to remove peer {public_key[:16]}...: {remove_result.stderr}")
                else:
                    kept.append({
                        'public_key': public_key[:16] + '...',
                        'allowed_ips': allowed_ips,
                        'handshake': handshake_status
                    })
            
            # Save configuration if we removed any
            if removed:
                try:
                    subprocess.run(
                        ['wg-quick', 'save', WIREGUARD_INTERFACE],
                        capture_output=True,
                        timeout=5
                    )
                except Exception as e:
                    logger.warning(f"Failed to save WireGuard config: {e}")
            
            self.send_json_response({
                'success': True,
                'removed_count': len(removed),
                'kept_count': len(kept),
                'skipped_new_count': len(skipped_new),
                'removed': removed,
                'kept': kept,
                'skipped_new': skipped_new
            })
            
        except subprocess.TimeoutExpired:
            self.send_error_response(504, 'Command Timeout', 'WireGuard command timed out')
        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Error', str(e))
    
    def get_wireguard_status(self):
        """Get WireGuard interface status"""
        try:
            result = subprocess.run(
                ['wg', 'show', WIREGUARD_INTERFACE, 'dump'],
                capture_output=True,
                text=True,
                timeout=5
            )
        
            if result.returncode != 0:
                logger.error(f"wg show failed: {result.stderr}")
                return {
                    'interface': WIREGUARD_INTERFACE,
                    'relay_id': RELAY_ID,
                    'region': RELAY_REGION,
                    'peer_count': 0,
                    'max_capacity': RELAY_CAPACITY,
                    'available_slots': RELAY_CAPACITY,
                    'peers': []
                }
        
            lines = result.stdout.strip().split('\n')
            peers = []
        
            for line in lines[1:]:
                if not line.strip():
                    continue
                
                fields = line.split('\t')
                if len(fields) >= 7:
                    # Return FULL public keys for orchestrator matching
                    # Dashboard can truncate for display if needed
                    peers.append({
                        'public_key': fields[0],  # Full key, not truncated!
                        'endpoint': fields[2] if fields[2] != '(none)' else None,
                        'allowed_ips': fields[3],
                        'latest_handshake': int(fields[4]) if fields[4] and fields[4] != '0' else 0,
                        'rx_bytes': int(fields[5]) if fields[5] else 0,
                        'tx_bytes': int(fields[6]) if fields[6] else 0
                    })
        
            return {
                'interface': WIREGUARD_INTERFACE,
                'relay_id': RELAY_ID,
                'region': RELAY_REGION,
                'peer_count': len(peers),
                'max_capacity': RELAY_CAPACITY,
                'available_slots': max(0, RELAY_CAPACITY - len(peers)),
                'peers': peers
            }
        
        except Exception as e:
            logger.error(f"Error getting WireGuard status: {e}", exc_info=True)
            return {
                'interface': WIREGUARD_INTERFACE,
                'relay_id': RELAY_ID,
                'region': RELAY_REGION,
                'peer_count': 0,
                'max_capacity': RELAY_CAPACITY,
                'available_slots': RELAY_CAPACITY,
                'peers': [],
                'error': str(e)
            }
    
    def serve_dashboard(self):
        """Serve dashboard HTML"""
        try:
            dashboard_path = os.path.join(STATIC_DIR, 'dashboard.html')
            
            if not os.path.exists(dashboard_path):
                self.send_error_response(404, 'Not Found', 'Dashboard file not found')
                return
            
            with open(dashboard_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(content))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content)
            
        except Exception as e:
            logger.error(f"Error serving dashboard: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Error', str(e))
    
    def serve_static_file(self, path):
        """Serve static files (CSS, JS, etc.)"""
        try:
            # Remove /static/ prefix
            file_path = path.replace('/static/', '', 1)
            full_path = os.path.join(STATIC_DIR, file_path)
            
            # Security: prevent directory traversal
            if not os.path.abspath(full_path).startswith(os.path.abspath(STATIC_DIR)):
                self.send_error_response(403, 'Forbidden', 'Access denied')
                return
            
            if not os.path.exists(full_path):
                self.send_error_response(404, 'Not Found', f'File not found: {file_path}')
                return
            
            # Determine content type
            content_type = 'application/octet-stream'
            if full_path.endswith('.css'):
                content_type = 'text/css'
            elif full_path.endswith('.js'):
                content_type = 'application/javascript'
            elif full_path.endswith('.json'):
                content_type = 'application/json'
            
            with open(full_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.send_header('Cache-Control', 'public, max-age=3600')
            self.end_headers()
            self.wfile.write(content)
            
        except Exception as e:
            logger.error(f"Error serving static file: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Error', str(e))
    
    # ==================== Response Helpers ====================
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        json_data = json.dumps(data, indent=2)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(json_data.encode('utf-8')))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(json_data.encode('utf-8'))
    
    def send_error_response(self, status_code, error_type, message):
        """Send error response"""
        self.send_json_response({
            'error': error_type,
            'message': message,
            'status': status_code
        }, status_code=status_code)


# ==================== Server Startup ====================
def main():
    """Start the relay API server with background cleanup"""
    
    if not os.path.exists(STATIC_DIR):
        logger.error(f"Static directory not found: {STATIC_DIR}")
        logger.error("Dashboard files may not be accessible")
    
    # Start background cleanup thread if enabled
    cleanup_thread = None
    
    if CLEANUP_ENABLED:
        cleanup_thread = PeerCleanupThread()
        cleanup_thread.start()
        logger.info("✅ Background peer cleanup enabled")
    else:
        logger.warning("⚠️  Background peer cleanup disabled")
    
    # Start HTTP server
    server_address = ('0.0.0.0', LISTEN_PORT)
    httpd = HTTPServer(server_address, RelayAPIHandler)
    
    logger.info('=' * 60)
    logger.info('DeCloud Relay VM Management API')
    logger.info('=' * 60)
    logger.info(f'  Version:      2.1.0 (IMPROVED)')
    logger.info(f'  Relay ID:     {RELAY_ID}')
    logger.info(f'  Relay Name:   {RELAY_NAME}')
    logger.info(f'  Region:       {RELAY_REGION}')
    logger.info(f'  Capacity:     {RELAY_CAPACITY} nodes')
    logger.info(f'  Interface:    {WIREGUARD_INTERFACE}')
    logger.info(f'  Listen:       http://0.0.0.0:{LISTEN_PORT}')
    logger.info(f'  Dashboard:    http://0.0.0.0:{LISTEN_PORT}/')
    logger.info(f'  Static Files: {STATIC_DIR}')
    if CLEANUP_ENABLED:
        logger.info(f'  🧹 Cleanup:    Every {CLEANUP_INTERVAL_SECONDS}s ({CLEANUP_INTERVAL_SECONDS // 60} min)')
        logger.info(f'  ⏱️  Stale After: {STALE_THRESHOLD_SECONDS}s ({STALE_THRESHOLD_SECONDS // 60} min)')
        logger.info(f'  ✅ Grace Period: {PEER_REGISTRATION_GRACE_PERIOD}s for new peers')
    logger.info('=' * 60)
    logger.info('✅ Server started successfully')
    logger.info('')
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info('\n🛑 Shutting down server...')
        
        if cleanup_thread:
            cleanup_thread.stop()
            cleanup_thread.join(timeout=5)
        
        httpd.shutdown()
        logger.info('✅ Server stopped')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)