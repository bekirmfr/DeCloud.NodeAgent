package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	multiaddr "github.com/multiformats/go-multiaddr"
)

const (
	protocolPrefix = "/decloud"
	keyFileName    = "identity.key"
)

// Config holds the DHT node configuration from environment variables.
type Config struct {
	ListenPort     string
	APIPort        string
	AdvertiseIP    string
	BootstrapPeers string
	DataDir        string
	NodeID         string
	Region         string
}

// NodeState tracks runtime state of the DHT node.
type NodeState struct {
	mu             sync.RWMutex
	host           host.Host
	dht            *dht.IpfsDHT
	pubsub         *pubsub.PubSub
	eventTopic     *pubsub.Topic
	startTime      time.Time
	connectedPeers int
	status         string
}

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmsgprefix)
	log.SetPrefix("[dht-node] ")

	cfg := loadConfig()
	log.Printf("Starting DeCloud DHT node (nodeId=%s, region=%s)", cfg.NodeID, cfg.Region)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Load or generate persistent identity
	privKey, err := loadOrCreateIdentity(cfg.DataDir)
	if err != nil {
		log.Fatalf("Failed to load identity: %v", err)
	}

	// Build libp2p host
	listenAddr := fmt.Sprintf("/ip4/0.0.0.0/tcp/%s", cfg.ListenPort)
	opts := []libp2p.Option{
		libp2p.Identity(privKey),
		libp2p.ListenAddrStrings(listenAddr),
	}

	if cfg.AdvertiseIP != "" {
		extAddr := fmt.Sprintf("/ip4/%s/tcp/%s", cfg.AdvertiseIP, cfg.ListenPort)
		extMA, err := multiaddr.NewMultiaddr(extAddr)
		if err == nil {
			opts = append(opts, libp2p.AddrsFactory(func(addrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
				return append(addrs, extMA)
			}))
		}
	}

	h, err := libp2p.New(opts...)
	if err != nil {
		log.Fatalf("Failed to create libp2p host: %v", err)
	}
	defer h.Close()

	log.Printf("Peer ID: %s", h.ID())
	for _, addr := range h.Addrs() {
		log.Printf("Listening on: %s/p2p/%s", addr, h.ID())
	}

	// Initialize Kademlia DHT in server mode
	kadDHT, err := dht.New(ctx, h,
		dht.Mode(dht.ModeServer),
		dht.ProtocolPrefix(protocolPrefix),
		dht.Datastore(nil), // in-memory datastore
	)
	if err != nil {
		log.Fatalf("Failed to create DHT: %v", err)
	}

	if err := kadDHT.Bootstrap(ctx); err != nil {
		log.Fatalf("Failed to bootstrap DHT: %v", err)
	}

	// Connect to bootstrap peers
	connectBootstrapPeers(ctx, h, cfg.BootstrapPeers)

	// Initialize GossipSub
	ps, err := pubsub.NewGossipSub(ctx, h)
	if err != nil {
		log.Fatalf("Failed to create GossipSub: %v", err)
	}

	// Join the DeCloud events topic
	topic, err := ps.Join(fmt.Sprintf("%s/events/%s", protocolPrefix, cfg.Region))
	if err != nil {
		log.Fatalf("Failed to join events topic: %v", err)
	}

	// Subscribe to receive events (required for topic participation)
	sub, err := topic.Subscribe()
	if err != nil {
		log.Fatalf("Failed to subscribe to events topic: %v", err)
	}
	go handleEvents(ctx, sub)

	// Start mDNS discovery for local peers
	mdnsService := mdns.NewMdnsService(h, protocolPrefix, &mdnsNotifee{h: h, ctx: ctx})
	if err := mdnsService.Start(); err != nil {
		log.Printf("mDNS discovery failed to start (non-fatal): %v", err)
	} else {
		defer mdnsService.Close()
	}

	state := &NodeState{
		host:      h,
		dht:       kadDHT,
		pubsub:    ps,
		eventTopic: topic,
		startTime: time.Now(),
		status:    "active",
	}

	// Start background peer counter
	go trackPeers(ctx, state)

	// Start HTTP API server
	go startAPIServer(cfg.APIPort, state)

	log.Printf("DHT node is ready (peer ID: %s)", h.ID())

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down DHT node...")
	state.mu.Lock()
	state.status = "shutting_down"
	state.mu.Unlock()

	cancel()
	kadDHT.Close()
}

func loadConfig() Config {
	return Config{
		ListenPort:     envOrDefault("DHT_LISTEN_PORT", "4001"),
		APIPort:        envOrDefault("DHT_API_PORT", "5080"),
		AdvertiseIP:    os.Getenv("DHT_ADVERTISE_IP"),
		BootstrapPeers: os.Getenv("DHT_BOOTSTRAP_PEERS"),
		DataDir:        envOrDefault("DHT_DATA_DIR", "/var/lib/decloud-dht"),
		NodeID:         os.Getenv("DECLOUD_NODE_ID"),
		Region:         envOrDefault("DECLOUD_REGION", "default"),
	}
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// loadOrCreateIdentity loads a persistent Ed25519 key or generates a new one.
func loadOrCreateIdentity(dataDir string) (crypto.PrivKey, error) {
	keyPath := filepath.Join(dataDir, keyFileName)

	data, err := os.ReadFile(keyPath)
	if err == nil {
		privKey, err := crypto.UnmarshalPrivateKey(data)
		if err == nil {
			log.Printf("Loaded existing identity from %s", keyPath)
			return privKey, nil
		}
		log.Printf("Failed to unmarshal existing key, generating new one: %v", err)
	}

	// Generate new Ed25519 key
	privKey, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate Ed25519 key: %w", err)
	}

	keyBytes, err := crypto.MarshalPrivateKey(privKey)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal key: %w", err)
	}

	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		return nil, fmt.Errorf("failed to create data dir: %w", err)
	}

	if err := os.WriteFile(keyPath, keyBytes, 0o600); err != nil {
		log.Printf("Warning: failed to persist identity key: %v", err)
	} else {
		log.Printf("Generated and saved new identity to %s", keyPath)
	}

	return privKey, nil
}

func connectBootstrapPeers(ctx context.Context, h host.Host, peersStr string) {
	if peersStr == "" {
		log.Println("No bootstrap peers configured (first node in network)")
		return
	}

	peers := strings.Split(peersStr, ",")
	var connected int
	for _, p := range peers {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}

		ma, err := multiaddr.NewMultiaddr(p)
		if err != nil {
			log.Printf("Invalid bootstrap peer address %q: %v", p, err)
			continue
		}

		pi, err := peer.AddrInfoFromP2pAddr(ma)
		if err != nil {
			log.Printf("Failed to parse peer info from %q: %v", p, err)
			continue
		}

		if err := h.Connect(ctx, *pi); err != nil {
			log.Printf("Failed to connect to bootstrap peer %s: %v", pi.ID.String()[:12], err)
		} else {
			log.Printf("Connected to bootstrap peer: %s", pi.ID.String()[:12])
			connected++
		}
	}
	log.Printf("Connected to %d/%d bootstrap peers", connected, len(peers))
}

func handleEvents(ctx context.Context, sub *pubsub.Subscription) {
	for {
		msg, err := sub.Next(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			log.Printf("Error reading from events subscription: %v", err)
			continue
		}
		// Log event receipt (could be extended to handle specific event types)
		log.Printf("Received event from %s (%d bytes)", msg.GetFrom().String()[:12], len(msg.Data))
	}
}

func trackPeers(ctx context.Context, state *NodeState) {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			state.mu.Lock()
			state.connectedPeers = len(state.host.Network().Peers())
			state.mu.Unlock()
		}
	}
}

// startAPIServer runs the HTTP health/status API and dashboard.
func startAPIServer(port string, state *NodeState) {
	mux := http.NewServeMux()

	// Serve dashboard at root
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, "/opt/decloud-dht/static/dashboard.html")
	})

	// Serve static assets
	mux.Handle("/static/", http.StripPrefix("/static/",
		http.FileServer(http.Dir("/opt/decloud-dht/static"))))

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		state.mu.RLock()
		defer state.mu.RUnlock()

		resp := map[string]interface{}{
			"status":         state.status,
			"peerId":         state.host.ID().String(),
			"connectedPeers": state.connectedPeers,
			"uptime":         time.Since(state.startTime).String(),
			"uptimeSeconds":  int(time.Since(state.startTime).Seconds()),
			"addresses":      formatAddresses(state.host),
			"routingTable":   state.dht.RoutingTable().Size(),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	mux.HandleFunc("/peers", func(w http.ResponseWriter, r *http.Request) {
		state.mu.RLock()
		defer state.mu.RUnlock()

		peers := state.host.Network().Peers()
		peerList := make([]string, len(peers))
		for i, p := range peers {
			peerList[i] = p.String()
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"count": len(peerList),
			"peers": peerList,
		})
	})

	mux.HandleFunc("/connect", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var payload struct {
			Peers []string `json:"peers"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}

		if len(payload.Peers) == 0 {
			http.Error(w, "no peers provided", http.StatusBadRequest)
			return
		}

		results := make([]map[string]interface{}, 0, len(payload.Peers))
		var connected int
		for _, p := range payload.Peers {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}

			ma, err := multiaddr.NewMultiaddr(p)
			if err != nil {
				results = append(results, map[string]interface{}{
					"addr":  p,
					"error": fmt.Sprintf("invalid multiaddr: %v", err),
				})
				continue
			}

			pi, err := peer.AddrInfoFromP2pAddr(ma)
			if err != nil {
				results = append(results, map[string]interface{}{
					"addr":  p,
					"error": fmt.Sprintf("invalid peer addr: %v", err),
				})
				continue
			}

			// Skip self
			if pi.ID == state.host.ID() {
				results = append(results, map[string]interface{}{
					"addr":    p,
					"skipped": "self",
				})
				continue
			}

			// Skip already-connected peers
			if state.host.Network().Connectedness(pi.ID) == 1 { // Connected
				results = append(results, map[string]interface{}{
					"addr":      p,
					"peerId":    pi.ID.String(),
					"connected": true,
					"skipped":   "already connected",
				})
				connected++
				continue
			}

			if err := state.host.Connect(context.Background(), *pi); err != nil {
				log.Printf("POST /connect: failed to connect to %s: %v", pi.ID.String()[:12], err)
				results = append(results, map[string]interface{}{
					"addr":      p,
					"peerId":    pi.ID.String(),
					"connected": false,
					"error":     err.Error(),
				})
			} else {
				log.Printf("POST /connect: connected to peer %s", pi.ID.String()[:12])
				results = append(results, map[string]interface{}{
					"addr":      p,
					"peerId":    pi.ID.String(),
					"connected": true,
				})
				connected++
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"results":   results,
			"connected": connected,
			"total":     len(payload.Peers),
		})
	})

	mux.HandleFunc("/publish", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var payload struct {
			Data string `json:"data"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}

		state.mu.RLock()
		topic := state.eventTopic
		state.mu.RUnlock()

		if err := topic.Publish(context.Background(), []byte(payload.Data)); err != nil {
			http.Error(w, fmt.Sprintf("publish failed: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "published"})
	})

	addr := fmt.Sprintf("0.0.0.0:%s", port)
	log.Printf("HTTP API listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("HTTP API server failed: %v", err)
	}
}

func formatAddresses(h host.Host) []string {
	addrs := h.Addrs()
	result := make([]string, len(addrs))
	for i, a := range addrs {
		result[i] = fmt.Sprintf("%s/p2p/%s", a, h.ID())
	}
	return result
}

// mdnsNotifee handles mDNS peer discovery.
type mdnsNotifee struct {
	h   host.Host
	ctx context.Context
}

func (n *mdnsNotifee) HandlePeerFound(pi peer.AddrInfo) {
	if pi.ID == n.h.ID() {
		return
	}
	log.Printf("mDNS: discovered peer %s", pi.ID.String()[:12])
	if err := n.h.Connect(n.ctx, pi); err != nil {
		log.Printf("mDNS: failed to connect to %s: %v", pi.ID.String()[:12], err)
	}
}
