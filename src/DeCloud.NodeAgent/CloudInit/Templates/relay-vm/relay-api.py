#!/usr/bin/env python3
"""
DeCloud Relay VM Management API
Version: 1.0.0

Provides REST API for relay monitoring, CGNAT peer management,
and serves the dashboard interface.
"""

import json
import subprocess
import os
import logging
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
            else:
                self.send_error_response(404, 'Not Found', 'The requested resource was not found')
        
        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}", exc_info=True)
            self.send_error_response(500, 'Internal Server Error', str(e))
    
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
        # Determine file extension
        ext = os.path.splitext(filename)[1].lower()
    
        # Cache policies by file type
        cache_policies = {
            # HTML files: short cache, must revalidate
            '.html': 'public, max-age=300, must-revalidate',  # 5 minutes
        
            # CSS and JS: medium cache with revalidation
            '.css': 'public, max-age=86400, must-revalidate',  # 1 day
            '.js': 'public, max-age=86400, must-revalidate',   # 1 day
        
            # Images: long cache
            '.png': 'public, max-age=2592000, immutable',  # 30 days
            '.jpg': 'public, max-age=2592000, immutable',
            '.jpeg': 'public, max-age=2592000, immutable',
            '.gif': 'public, max-age=2592000, immutable',
            '.svg': 'public, max-age=2592000, immutable',
            '.ico': 'public, max-age=2592000, immutable',
        
            # Fonts: long cache
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
        """Get MIME type for file"""
        ext = os.path.splitext(filename)[1].lower()
        
        mime_types = {
            '.html': 'text/html; charset=utf-8',
            '.css': 'text/css; charset=utf-8',
            '.js': 'application/javascript; charset=utf-8',
            '.json': 'application/json',
            '.txt': 'text/plain; charset=utf-8',
            '.svg': 'image/svg+xml',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif'
        }
        
        return mime_types.get(ext, 'application/octet-stream')
    
    # ==================== API Endpoints ====================
    
    def serve_wireguard_status(self):
        """Serve WireGuard status with short cache"""
        try:
            data = get_wireguard_status()
        
            # Generate ETag from data hash
            import hashlib
            data_str = json.dumps(data, sort_keys=True)
            etag = f'"{hashlib.md5(data_str.encode()).hexdigest()[:16]}"'
        
            # Check If-None-Match
            if_none_match = self.headers.get('If-None-Match')
            if if_none_match == etag:
                self.send_response(304)
                self.send_header('ETag', etag)
                self.send_header('Cache-Control', 'private, max-age=10, must-revalidate')
                self.end_headers()
                return
        
            # Send full response
            json_data = json.dumps(data, indent=2)
        
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', len(json_data.encode('utf-8')))
            self.send_header('Cache-Control', 'private, max-age=10, must-revalidate')
            self.send_header('ETag', etag)
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(json_data.encode('utf-8'))
        
        except Exception as e:
            logger.error(f"Error getting WireGuard status: {e}")
            self.send_error_response(500, 'Internal Error', str(e))
    
    def serve_system_status(self):
        """Get system resource status"""
        try:
            # Get uptime
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
            
            # Get memory info
            meminfo = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        meminfo[key.strip()] = value.strip()
            
            mem_total = int(meminfo.get('MemTotal', '0').split()[0]) * 1024
            mem_available = int(meminfo.get('MemAvailable', '0').split()[0]) * 1024
            mem_used = mem_total - mem_available
            mem_percent = int((mem_used / mem_total) * 100) if mem_total > 0 else 0
            
            # Get CPU load average
            with open('/proc/loadavg', 'r') as f:
                load_avg = f.read().split()[:3]
            
            # Get network interface info
            net_interface = 'eth0'  # Default
            net_ip = None
            try:
                ip_result = subprocess.run(
                    ['ip', '-4', 'addr', 'show', net_interface],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if ip_result.returncode == 0:
                    for line in ip_result.stdout.split('\n'):
                        if 'inet ' in line:
                            net_ip = line.strip().split()[1].split('/')[0]
                            break
            except:
                pass
            
            status = {
                'uptime_seconds': int(uptime_seconds),
                'memory_total': mem_total,
                'memory_used': mem_used,
                'memory_available': mem_available,
                'memory_percent': mem_percent,
                'load_average': [float(x) for x in load_avg],
                'network_interface': net_interface,
                'network_ip': net_ip,
                'relay_id': RELAY_ID,
                'relay_region': RELAY_REGION
            }
            
            self.send_json_response(status)
            
        except Exception as e:
            logger.error(f"System status error: {e}", exc_info=True)
            self.send_error_response(500, 'System Status Error', str(e))
    
    def serve_debug_info(self):
        """Get debug information"""
        debug_info = {
            'relay_id': RELAY_ID,
            'relay_name': RELAY_NAME,
            'region': RELAY_REGION,
            'capacity': RELAY_CAPACITY,
            'interface': WIREGUARD_INTERFACE,
            'api_version': '1.0.0',
            'api_port': LISTEN_PORT
        }
        self.send_json_response(debug_info)
    
    def serve_health_check(self):
        """Simple health check endpoint"""
        self.send_json_response({
            'status': 'healthy',
            'relay_id': RELAY_ID,
            'timestamp': int(subprocess.run(
                ['date', '+%s'],
                capture_output=True,
                text=True
            ).stdout.strip())
        })
    
    def add_cgnat_peer(self):
        """Add CGNAT node as WireGuard peer"""
        try:
            # Read request body
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
            
            # Add optional description (as comment)
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
                # Don't fail the request if save fails
            
            logger.info(f"Added CGNAT peer: {data['public_key'][:16]}... ({description})")
            
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
    
    # ==================== Helper Functions ====================

    def parse_wireguard_dump(self, dump_output):
        """Parse WireGuard dump output into structured data"""
        lines = dump_output.strip().split('\n')
        
        if len(lines) < 1:
            return {
                'interface': WIREGUARD_INTERFACE,
                'relay_id': RELAY_ID,
                'peer_count': 0,
                'max_capacity': RELAY_CAPACITY,
                'available_slots': RELAY_CAPACITY,
                'peers': []
            }
        
        peers = []
        
        # Skip interface line (first line) and parse peers
        for line in lines[1:]:
            fields = line.split('\t')
            if len(fields) >= 7:
                # Fields: public_key, preshared_key, endpoint, allowed_ips, 
                #         latest_handshake, rx_bytes, tx_bytes
                peers.append({
                    'public_key': fields[0][:16] + '...' if len(fields[0]) > 16 else fields[0],
                    'endpoint': fields[2] if fields[2] != '(none)' else None,
                    'allowed_ips': fields[3],
                    'latest_handshake': int(fields[4]) if fields[4] else 0,
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

    def get_wireguard_status():
        """Get WireGuard interface status"""
        try:
            # Run wg show dump
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
        
            # Parse dump output
            lines = result.stdout.strip().split('\n')
            peers = []
        
            # Skip interface line (first line) and parse peers
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
    """Start the relay API server"""
    
    # Verify static directory exists
    if not os.path.exists(STATIC_DIR):
        logger.error(f"Static directory not found: {STATIC_DIR}")
        logger.error("Dashboard files may not be accessible")
    
    # Start HTTP server
    server_address = ('0.0.0.0', LISTEN_PORT)
    httpd = HTTPServer(server_address, RelayAPIHandler)
    
    logger.info('=' * 60)
    logger.info('DeCloud Relay VM Management API')
    logger.info('=' * 60)
    logger.info(f'  Version:      1.0.0')
    logger.info(f'  Relay ID:     {RELAY_ID}')
    logger.info(f'  Relay Name:   {RELAY_NAME}')
    logger.info(f'  Region:       {RELAY_REGION}')
    logger.info(f'  Capacity:     {RELAY_CAPACITY} nodes')
    logger.info(f'  Interface:    {WIREGUARD_INTERFACE}')
    logger.info(f'  Listen:       http://0.0.0.0:{LISTEN_PORT}')
    logger.info(f'  Dashboard:    http://0.0.0.0:{LISTEN_PORT}/')
    logger.info(f'  Static Files: {STATIC_DIR}')
    logger.info('=' * 60)
    logger.info('✓ Server started successfully')
    logger.info('')
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info('Shutting down server...')
        httpd.shutdown()
        logger.info('✓ Server stopped')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)