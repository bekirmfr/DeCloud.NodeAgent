#!/usr/bin/env python3
"""
DeCloud Relay VM Management API
Version: 2.0.0

Provides REST API for relay monitoring, CGNAT peer management,
and serves the dashboard interface.

NEW: Automatic periodic cleanup of stale peers via background thread
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
STALE_THRESHOLD_SECONDS = 180             # 3 minutes - peer with no handshake for this long is stale
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
        
        # Wait 30 seconds before first run (let server stabilize)
        time.sleep(30)
        
        while not self.stop_event.is_set():
            try:
                self.cleanup_stale_peers()
                
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
        """Remove peers with no handshake or stale handshake"""
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
                
                # Calculate time since last handshake
                if latest_handshake == 0:
                    # Never had a handshake
                    time_since_handshake = float('inf')
                    handshake_status = 'never'
                else:
                    time_since_handshake = current_time - latest_handshake
                    handshake_status = f'{time_since_handshake}s ago'
                
                # Check if peer is stale
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
                        logger.info(
                            f"🗑️  Removed stale peer: {public_key[:16]}... "
                            f"({allowed_ips}, handshake: {handshake_status})"
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
            if removed:
                logger.info(
                    f"✅ Cleanup completed: removed {len(removed)} stale peers, "
                    f"kept {len(kept)} active peers"
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
        """Override to use custom logging"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path
        
        try:
            # Route requests
            if path == '/':
                self.serve_dashboard()
            elif path.startswith('/static/'):
                self.serve_static_file(path)
            elif path == '/api/relay/wireguard':
                self.serve_wireguard_status()
            elif path == '/api/relay/system':
                self.serve_system_status()
            elif path == '/api/relay/debug':
                self.serve_debug_info()
            elif path == '/api/relay/cleanup-stats':  # NEW ENDPOINT
                self.serve_cleanup_stats()
            elif path == '/health':
                self.serve_health_check()
            else:
                self.send_error_response(404, 'Not Found', 'The requested resource was not found')
        
        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Server Error', str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        path = urlparse(self.path).path
    
        try:
            if path in ['/api/relay/peer', '/api/relay/add-peer']:
                self.add_cgnat_peer()
            elif path == '/api/relay/remove-peer':
                self.remove_cgnat_peer()
            elif path == '/api/relay/cleanup-stale':
                self.cleanup_stale_peers()
            else:
                self.send_error_response(404, 'Not Found', 'The requested resource was not found')
    
        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}", exc_info=True)
    
    # ==================== NEW: Cleanup Stats Endpoint ====================
    
    def serve_cleanup_stats(self):
        """Serve cleanup statistics"""
        import datetime
        
        stats = cleanup_stats.copy()
        
        # Format last_run timestamp
        if stats['last_run']:
            stats['last_run_formatted'] = datetime.datetime.fromtimestamp(
                stats['last_run']
            ).isoformat()
            stats['seconds_since_last_run'] = int(time.time() - stats['last_run'])
        else:
            stats['last_run_formatted'] = 'Never'
            stats['seconds_since_last_run'] = None
        
        # Add configuration info
        stats['config'] = {
            'enabled': CLEANUP_ENABLED,
            'interval_seconds': CLEANUP_INTERVAL_SECONDS,
            'stale_threshold_seconds': STALE_THRESHOLD_SECONDS,
            'orchestrator_protected_ip': ORCHESTRATOR_IP
        }
        
        self.send_json_response(stats)
    
    # ==================== Dashboard & Static Files ====================
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        self.serve_static_file('/static/dashboard.html')
    
    def serve_static_file(self, path):
        """Serve static file with intelligent caching"""
        # Extract filename from path
        if path.startswith('/static/'):
            filename = path[8:]  # Remove '/static/'
        else:
            filename = os.path.basename(path)
    
        file_path = os.path.join(STATIC_DIR, filename)
    
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                self.send_error_response(404, 'File Not Found', f'File {filename} not found')
                return
        
            # Get file stats for caching
            stat_info = os.stat(file_path)
            mtime = stat_info.st_mtime
            size = stat_info.st_size
        
            # Generate ETag
            etag = f'"{int(mtime)}-{size}"'
            last_modified = self.format_timestamp(mtime)
        
            # Check conditional requests
            if_none_match = self.headers.get('If-None-Match')
            if_modified_since = self.headers.get('If-Modified-Since')
        
            if if_none_match == etag or if_modified_since == last_modified:
                # Return 304 Not Modified
                self.send_response(304)
                self.send_header('ETag', etag)
                self.send_header('Last-Modified', last_modified)
                self.send_header('Cache-Control', self.get_cache_control(filename))
                self.end_headers()
                return
        
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
            # Replace template variables (only for HTML)
            if filename.endswith('.html'):
                content = self.replace_template_variables(content)
        
            # Determine content type
            content_type = self.get_content_type(filename)
        
            # Determine cache control based on file type
            cache_control = self.get_cache_control(filename)
        
            # Send response with caching headers
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content.encode('utf-8')))
            self.send_header('Cache-Control', cache_control)
            self.send_header('ETag', etag)
            self.send_header('Last-Modified', last_modified)
            self.send_header('X-Content-Type-Options', 'nosniff')
        
            # Add CORS headers if needed
            self.send_header('Access-Control-Allow-Origin', '*')
        
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        
        except Exception as e:
            logger.error(f"Error serving file {file_path}: {e}")
            self.send_error_response(500, 'Error Reading File', str(e))

    def get_cache_control(self, filename):
        """Get appropriate Cache-Control header based on file type"""
        ext = os.path.splitext(filename)[1].lower()
        cache_policies = {
            '.html': 'public, max-age=300, must-revalidate',
            '.css': 'public, max-age=86400, must-revalidate',
            '.js': 'public, max-age=86400, must-revalidate',
            '.png': 'public, max-age=2592000, immutable',
            '.jpg': 'public, max-age=2592000, immutable',
            '.jpeg': 'public, max-age=2592000, immutable',
            '.gif': 'public, max-age=2592000, immutable',
            '.svg': 'public, max-age=2592000, immutable',
            '.ico': 'public, max-age=2592000, immutable',
            '.woff': 'public, max-age=2592000, immutable',
            '.woff2': 'public, max-age=2592000, immutable',
            '.ttf': 'public, max-age=2592000, immutable',
        }
        return cache_policies.get(ext, 'public, max-age=3600, must-revalidate')

    def format_timestamp(self, timestamp):
        """Format Unix timestamp to HTTP date format"""
        from email.utils import formatdate
        return formatdate(timestamp, usegmt=True)
    
    def replace_template_variables(self, content):
        """Replace template variables in content"""
        replacements = {
            '__VM_ID__': RELAY_ID,
            '__VM_NAME__': RELAY_NAME,
            '__RELAY_REGION__': RELAY_REGION,
            '__RELAY_CAPACITY__': str(RELAY_CAPACITY)
        }
        
        for key, value in replacements.items():
            content = content.replace(key, value)
        
        return content
    
    def get_content_type(self, filename):
        """Determine content type based on file extension"""
        ext = os.path.splitext(filename)[1].lower()
        
        content_types = {
            '.html': 'text/html; charset=utf-8',
            '.css': 'text/css; charset=utf-8',
            '.js': 'application/javascript; charset=utf-8',
            '.json': 'application/json; charset=utf-8',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon',
            '.woff': 'font/woff',
            '.woff2': 'font/woff2',
            '.ttf': 'font/ttf',
        }
        
        return content_types.get(ext, 'application/octet-stream')
    
    # ==================== API Endpoints ====================
    
    def serve_health_check(self):
        """Enhanced health check with cleanup info"""
        self.send_json_response({
            'status': 'healthy',
            'relay_id': RELAY_ID,
            'timestamp': int(time.time()),
            'cleanup_enabled': CLEANUP_ENABLED,
            'last_cleanup': cleanup_stats.get('last_run'),
            'total_cleanups': cleanup_stats.get('total_runs', 0)
        })
    
    def serve_wireguard_status(self):
        """Serve WireGuard interface status"""
        status = self.get_wireguard_status()
        self.send_json_response(status)
    
    def serve_system_status(self):
        """Serve system status information"""
        try:
            # Get system uptime
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
            
            # Get load average
            with open('/proc/loadavg', 'r') as f:
                load = f.read().split()[:3]
            
            # Get memory info
            mem_info = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]
                        mem_info[key] = int(value)
            
            self.send_json_response({
                'uptime_seconds': int(uptime_seconds),
                'load_average': {
                    '1min': float(load[0]),
                    '5min': float(load[1]),
                    '15min': float(load[2])
                },
                'memory': {
                    'total_kb': mem_info.get('MemTotal', 0),
                    'available_kb': mem_info.get('MemAvailable', 0),
                    'free_kb': mem_info.get('MemFree', 0)
                }
            })
        
        except Exception as e:
            logger.error(f"Error getting system status: {e}", exc_info=True)
            self.send_error_response(500, 'System Status Error', str(e))
    
    def serve_debug_info(self):
        """Serve debug information"""
        self.send_json_response({
            'relay_id': RELAY_ID,
            'relay_name': RELAY_NAME,
            'region': RELAY_REGION,
            'capacity': RELAY_CAPACITY,
            'interface': WIREGUARD_INTERFACE,
            'cleanup_config': {
                'enabled': CLEANUP_ENABLED,
                'interval_seconds': CLEANUP_INTERVAL_SECONDS,
                'stale_threshold_seconds': STALE_THRESHOLD_SECONDS
            },
            'cleanup_stats': cleanup_stats
        })
    
    def add_cgnat_peer(self):
        """Add CGNAT node as WireGuard peer"""
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
            
            # Build wg set command
            cmd = [
                'wg', 'set', WIREGUARD_INTERFACE,
                'peer', data['public_key'],
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
            
            logger.info(f"✅ Added CGNAT peer: {data['public_key'][:16]}... ({description})")
            
            self.send_json_response({
                'success': True,
                'message': 'Peer added successfully',
                'peer': {
                    'public_key': data['public_key'][:16] + '...',
                    'allowed_ips': data['allowed_ips'],
                    'description': description
                }
            })
            
        except json.JSONDecodeError as e:
            self.send_error_response(400, 'Invalid JSON', str(e))
        except subprocess.TimeoutExpired:
            self.send_error_response(504, 'Command Timeout', 'WireGuard command timed out')
        except Exception as e:
            logger.error(f"Add peer error: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Error', str(e))
    
    def remove_cgnat_peer(self):
        """Remove CGNAT node from WireGuard peers"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error_response(400, 'Bad Request', 'Request body is empty')
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Require public_key
            if 'public_key' not in data:
                self.send_error_response(
                    400,
                    'Missing Required Fields',
                    'public_key field is required'
                )
                return
            
            public_key = data['public_key']
            node_id = data.get('node_id', 'unknown')
            
            # Remove peer
            cmd = [
                'wg', 'set', WIREGUARD_INTERFACE,
                'peer', public_key,
                'remove'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Success if peer was removed OR if peer didn't exist
            if result.returncode == 0 or 'Unable to remove' in result.stderr:
                # Save configuration
                try:
                    subprocess.run(
                        ['wg-quick', 'save', WIREGUARD_INTERFACE],
                        capture_output=True,
                        timeout=5
                    )
                except Exception as e:
                    logger.warning(f"Failed to save WireGuard config: {e}")
                
                logger.info(f"✅ Removed peer {public_key[:16]}... (node: {node_id})")
                
                self.send_json_response({
                    'success': True,
                    'message': 'Peer removed successfully',
                    'public_key': public_key[:16] + '...',
                    'node_id': node_id
                })
            else:
                logger.error(f"Failed to remove peer: {result.stderr}")
                self.send_error_response(
                    500,
                    'Failed to Remove Peer',
                    result.stderr or 'Unknown error'
                )
            
        except json.JSONDecodeError as e:
            self.send_error_response(400, 'Invalid JSON', str(e))
        except subprocess.TimeoutExpired:
            self.send_error_response(504, 'Command Timeout', 'WireGuard command timed out')
        except Exception as e:
            logger.error(f"Remove peer error: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Error', str(e))
    
    def cleanup_stale_peers(self):
        """Manual cleanup endpoint - calls same logic as background thread"""
        try:
            import time as time_module
            
            STALE_THRESHOLD = 300  # 5 minutes
            ORCHESTRATOR_IP_LOCAL = "10.20.0.1/32"
            
            # Get current peers
            result = subprocess.run(
                ['wg', 'show', WIREGUARD_INTERFACE, 'dump'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                self.send_error_response(
                    500,
                    'Failed to Get Peers',
                    result.stderr or 'Unknown error'
                )
                return
            
            current_time = int(time_module.time())
            lines = result.stdout.strip().split('\n')
            
            removed = []
            kept = []
            
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
                
                # Never remove orchestrator
                if ORCHESTRATOR_IP_LOCAL in allowed_ips:
                    kept.append({
                        'public_key': public_key[:16] + '...',
                        'allowed_ips': allowed_ips,
                        'reason': 'orchestrator'
                    })
                    continue
                
                # Check if stale
                if latest_handshake == 0:
                    time_since_handshake = float('inf')
                else:
                    time_since_handshake = current_time - latest_handshake
                
                if time_since_handshake > STALE_THRESHOLD:
                    # Remove stale peer
                    remove_cmd = [
                        'wg', 'set', WIREGUARD_INTERFACE,
                        'peer', public_key,
                        'remove'
                    ]
                    
                    remove_result = subprocess.run(
                        remove_cmd,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if remove_result.returncode == 0:
                        removed.append({
                            'public_key': public_key[:16] + '...',
                            'allowed_ips': allowed_ips,
                            'stale_seconds': time_since_handshake if time_since_handshake != float('inf') else 'never'
                        })
                        logger.info(f"🗑️  Removed stale peer: {public_key[:16]}... ({allowed_ips})")
                    else:
                        logger.error(f"Failed to remove peer {public_key[:16]}...: {remove_result.stderr}")
                else:
                    kept.append({
                        'public_key': public_key[:16] + '...',
                        'allowed_ips': allowed_ips,
                        'last_handshake_seconds': time_since_handshake
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
                'removed': removed,
                'kept': kept
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
                    peers.append({
                        'public_key': fields[0][:16] + '...' if len(fields[0]) > 16 else fields[0],
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
    logger.info(f'  Version:      2.0.0 (with auto-cleanup)')
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