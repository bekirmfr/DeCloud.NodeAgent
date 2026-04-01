// DeCloud Block Store Node
//
// A libp2p node providing content-addressed distributed storage
// for VM disk overlay replication (lazysync) and AI model shard distribution.
//
// Storage:  FlatFS (IPFS-compatible, content-addressed blocks)
// Exchange: Bitswap (block fetching from network peers)
// Routing:  Kademlia DHT client (provider record announce/lookup)
// Events:   GossipSub subscription for `decloud/blockstore/new-blocks`
// API:      HTTP on localhost:5090 (internal only)
//
// LRU eviction at 85% capacity, hard refuse writes at 95%.
// Adaptive XOR pull threshold based on local capacity utilization.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/big"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"net"

	bsnetwork "github.com/ipfs/boxo/bitswap/network"
	"github.com/ipfs/boxo/bitswap"
	blockstore "github.com/ipfs/boxo/blockstore"
	blocks "github.com/ipfs/go-block-format"
	"github.com/ipfs/go-cid"
	"github.com/ipfs/go-datastore"
	"github.com/ipfs/go-datastore/mount"
	flatfs "github.com/ipfs/go-ds-flatfs"
	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/multiformats/go-multiaddr"
	"github.com/multiformats/go-multihash"
)

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const (
	GossipSubTopic  = "decloud/blockstore/new-blocks"
	IdentityKeyFile = "identity.key"
	PeerIDFile      = "peer-id"
	BlocksSubdir    = "blocks"
	DagsSubdir      = "dags"   // ResourceManifest JSON files
	OwnersSubdir    = "owners" // Per-VM CID owner index files
	StorageDir      = "/var/lib/decloud-blockstore"

	// GC thresholds
	GCTriggerPercent = 85
	GCHardLimit      = 95

	// XOR proximity thresholds (fraction of keyspace, capacity-adaptive)
	XORThresholdFull   = 0.05
	XORThresholdMedium = 0.10
	XORThresholdLight  = 0.20

	// Bitswap fetch timeout
	BitswapTimeout = 30 * time.Second
)

// ═══════════════════════════════════════════════════════════════════
// Resource types — what kind of data a manifest describes
// ═══════════════════════════════════════════════════════════════════

type ResourceType string

const (
	ResourceTypeVMOverlay     ResourceType = "VMOverlay"     // VM disk overlay chunks (lazysync)
	ResourceTypeModelShard    ResourceType = "ModelShard"    // AI model weight shard
	ResourceTypeLoRAAdapter   ResourceType = "LoRAAdapter"   // LoRA fine-tune adapter weights
	ResourceTypeImageTemplate ResourceType = "ImageTemplate" // Base OS image template
	ResourceTypeUnknown       ResourceType = "Unknown"
)

// ShardMetadata describes an AI model shard for distributed inference routing.
// When a large model (e.g. Llama-3 70B) is split into shards, each shard covers
// a range of transformer layers. Inference VMs use this to build a pipeline:
// shard 0 (layers 0-15) → shard 1 (layers 16-31) → ... → output.
type ShardMetadata struct {
	ModelName      string `json:"modelName"`
	ModelVersion   string `json:"modelVersion"`
	ShardIndex     int    `json:"shardIndex"`
	TotalShards    int    `json:"totalShards"`
	LayerStart     int    `json:"layerStart"`
	LayerEnd       int    `json:"layerEnd"`
	ParameterCount int64  `json:"parameterCount"`
	QuantBits      int    `json:"quantBits"` // 4, 8, 16, 32
}

// ResourceManifest tracks a stored resource: its type, owner, chunk CIDs,
// and optional AI model shard metadata.
// Persisted to /var/lib/decloud-blockstore/dags/{rootCid}.json.
type ResourceManifest struct {
	RootCid       string        `json:"rootCid"`
	ResourceType  ResourceType  `json:"resourceType"`
	ResourceID    string        `json:"resourceId"`    // vmId, model name, template slug
	ResourceOwner string        `json:"resourceOwner"` // wallet address
	Version       int           `json:"version"`
	TotalBytes    int64         `json:"totalBytes"`
	ChunkCIDs     []string      `json:"chunkCids"`
	ShardMeta     *ShardMetadata `json:"shardMeta,omitempty"`
	RegisteredAt  time.Time     `json:"registeredAt"`
	UpdatedAt     time.Time     `json:"updatedAt"`
}

// ═══════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════

type Config struct {
	ListenPort      int
	APIPort         int
	AdvertiseIP     string
	StorageBytes    int64
	AuthToken       string
	NodeID          string
	VMID            string
	OrchestratorURL string
	BootstrapPeers  []string
}

func parseConfig() Config {
	return Config{
		ListenPort:      envInt("BLOCKSTORE_LISTEN_PORT", 5001),
		APIPort:         envInt("BLOCKSTORE_API_PORT", 5090),
		AdvertiseIP:     envStr("BLOCKSTORE_ADVERTISE_IP", ""),
		StorageBytes:    int64(envInt("BLOCKSTORE_STORAGE_BYTES", 10*1024*1024*1024)),
		AuthToken:       envStr("BLOCKSTORE_AUTH_TOKEN", ""),
		NodeID:          envStr("BLOCKSTORE_NODE_ID", ""),
		VMID:            envStr("BLOCKSTORE_VM_ID", ""),
		OrchestratorURL: envStr("ORCHESTRATOR_URL", ""),
		BootstrapPeers:  splitPeers(envStr("BLOCKSTORE_BOOTSTRAP_PEERS", "")),
	}
}

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func splitPeers(s string) []string {
	if s == "" {
		return nil
	}
	var out []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// isIPOnAnyInterface checks if ip is assigned to any local network interface.
func isIPOnAnyInterface(ip string) bool {
	ifaces, err := net.Interfaces()
	if err != nil {
		return false
	}
	for _, iface := range ifaces {
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, addr := range addrs {
			var ifIP net.IP
			switch v := addr.(type) {
			case *net.IPNet:
				ifIP = v.IP
			case *net.IPAddr:
				ifIP = v.IP
			}
			if ifIP != nil && ifIP.String() == ip {
				return true
			}
		}
	}
	return false
}

// ═══════════════════════════════════════════════════════════════════
// Identity
// ═══════════════════════════════════════════════════════════════════

func loadOrCreateIdentity(dir string) (crypto.PrivKey, error) {
	keyPath := filepath.Join(dir, IdentityKeyFile)
	if data, err := os.ReadFile(keyPath); err == nil {
		priv, err := crypto.UnmarshalPrivateKey(data)
		if err != nil {
			return nil, fmt.Errorf("unmarshal identity key: %w", err)
		}
		log.Printf("Loaded persistent identity from %s", keyPath)
		return priv, nil
	}

	priv, _, err := crypto.GenerateEd25519Key(nil)
	if err != nil {
		return nil, fmt.Errorf("generate identity key: %w", err)
	}
	data, err := crypto.MarshalPrivateKey(priv)
	if err != nil {
		return nil, fmt.Errorf("marshal identity key: %w", err)
	}
	if err := os.WriteFile(keyPath, data, 0600); err != nil {
		return nil, fmt.Errorf("write identity key: %w", err)
	}
	log.Printf("Generated new Ed25519 identity, saved to %s", keyPath)
	return priv, nil
}

// ═══════════════════════════════════════════════════════════════════
// BlockNode — main node state
// ═══════════════════════════════════════════════════════════════════

type BlockNode struct {
	cfg    Config
	host   host.Host
	dht    *dht.IpfsDHT
	pubsub *pubsub.PubSub
	bstore blockstore.Blockstore
	bsExch *bitswap.Bitswap

	// LRU tracking
	accessMu    sync.Mutex
	accessTimes map[string]time.Time
	blockSizes  map[string]int64

	// Manifest registry
	manifestsMu sync.RWMutex
	manifests   map[string]*ResourceManifest // rootCid → manifest

	// Counters
	mu              sync.RWMutex
	bitswapSent     uint64
	bitswapReceived uint64
}

// ═══════════════════════════════════════════════════════════════════
// Setup
// ═══════════════════════════════════════════════════════════════════

func setup(ctx context.Context, cfg Config) (*BlockNode, error) {
	for _, sub := range []string{"", BlocksSubdir, DagsSubdir, OwnersSubdir} {
		if err := os.MkdirAll(filepath.Join(StorageDir, sub), 0755); err != nil {
			return nil, fmt.Errorf("create dir %s: %w", sub, err)
		}
	}

	priv, err := loadOrCreateIdentity(StorageDir)
	if err != nil {
		return nil, fmt.Errorf("identity: %w", err)
	}

	blocksDir := filepath.Join(StorageDir, BlocksSubdir)
	shardFn, err := flatfs.ParseShardFunc("/repo/flatfs/shard/v1/next-to-last/2")
	if err != nil {
		return nil, fmt.Errorf("shard func: %w", err)
	}
	fds, err := flatfs.CreateOrOpen(blocksDir, shardFn, false)
	if err != nil {
		return nil, fmt.Errorf("open flatfs: %w", err)
	}
	// blockstore.NewBlockstore prepends "/blocks/" to all CID keys.
	// FlatFS only supports single-segment keys and rejects path-style keys.
	// Mount flatfs at "/blocks" so the prefix is stripped before reaching flatfs,
	// giving it single-segment keys like "/CIQB23..." it can handle.
	mds := mount.New([]mount.Mount{
		{
			Prefix:    datastore.NewKey("/blocks"),
			Datastore: fds,
		},
	})
	bs := blockstore.NewIdStore(blockstore.NewBlockstore(mds))

	// Wait up to 30s for the WG mesh interface to be assigned the advertise IP.
	// The binary starts concurrently with wg-quick; without this wait the
	// libp2p host logs only the libvirt bridge IP and the AddrsFactory below
	// advertises an IP that isn't yet on any interface.
	if cfg.AdvertiseIP != "" {
		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			if isIPOnAnyInterface(cfg.AdvertiseIP) {
				log.Printf("WG mesh interface ready — advertise IP %s is up", cfg.AdvertiseIP)
				break
			}
			log.Printf("Waiting for WG mesh interface (%s)...", cfg.AdvertiseIP)
			time.Sleep(2 * time.Second)
		}
		if !isIPOnAnyInterface(cfg.AdvertiseIP) {
			log.Printf("Warning: advertise IP %s not found on any interface after 30s — "+
				"peering via WG mesh will not work", cfg.AdvertiseIP)
		}
	}

	listenAddr := fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", cfg.ListenPort)
	opts := []libp2p.Option{
		libp2p.ListenAddrStrings(listenAddr),
		libp2p.Identity(priv),
	}

	// Only advertise the WG tunnel IP — don't expose unreachable
	// libvirt bridge or localhost addresses to other peers.
	if cfg.AdvertiseIP != "" {
		extAddr := fmt.Sprintf("/ip4/%s/tcp/%d", cfg.AdvertiseIP, cfg.ListenPort)
		extMA, maErr := multiaddr.NewMultiaddr(extAddr)
		if maErr == nil {
			opts = append(opts, libp2p.AddrsFactory(func(_ []multiaddr.Multiaddr) []multiaddr.Multiaddr {
				return []multiaddr.Multiaddr{extMA}
			}))
		}
	}

	h, err := libp2p.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("create libp2p host: %w", err)
	}

	log.Printf("libp2p peer ID: %s", h.ID())

	if err := os.WriteFile(filepath.Join(StorageDir, PeerIDFile),
		[]byte(h.ID().String()), 0644); err != nil {
		log.Printf("Warning: could not write peer-id file: %v", err)
	}

	kadDHT, err := dht.New(ctx, h,
		dht.Mode(dht.ModeClient),
		dht.ProtocolPrefix("/decloud"),
	)
	if err != nil {
		return nil, fmt.Errorf("create DHT: %w", err)
	}
	if err := kadDHT.Bootstrap(ctx); err != nil {
		log.Printf("Warning: DHT bootstrap: %v", err)
	}

	bsNet := bsnetwork.NewFromIpfsHost(h, kadDHT)
	bsExch := bitswap.New(ctx, bsNet, bs)

	ps, err := pubsub.NewGossipSub(ctx, h)
	if err != nil {
		return nil, fmt.Errorf("create pubsub: %w", err)
	}

	return &BlockNode{
		cfg:         cfg,
		host:        h,
		dht:         kadDHT,
		pubsub:      ps,
		bstore:      bs,
		bsExch:      bsExch,
		accessTimes: make(map[string]time.Time),
		blockSizes:  make(map[string]int64),
		manifests:   make(map[string]*ResourceManifest),
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
// Bootstrap peers
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) connectBootstrapPeers(ctx context.Context) {
	if len(n.cfg.BootstrapPeers) == 0 {
		log.Println("No bootstrap peers configured — running as genesis node")
		return
	}
	log.Printf("Connecting to %d bootstrap peer(s)...", len(n.cfg.BootstrapPeers))
	for _, addrStr := range n.cfg.BootstrapPeers {
		maddr, err := multiaddr.NewMultiaddr(addrStr)
		if err != nil {
			log.Printf("Invalid bootstrap peer addr %q: %v", addrStr, err)
			continue
		}
		pi, err := peer.AddrInfoFromP2pAddr(maddr)
		if err != nil {
			log.Printf("Parse bootstrap peer info %q: %v", addrStr, err)
			continue
		}
		connCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		if err := n.host.Connect(connCtx, *pi); err != nil {
			log.Printf("Connect to bootstrap peer %s: %v", pi.ID, err)
		} else {
			log.Printf("Connected to bootstrap peer %s", pi.ID)
		}
		cancel()
	}
}

// ═══════════════════════════════════════════════════════════════════
// Manifest registry
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) loadExistingManifests(ctx context.Context) {
	dagsDir := filepath.Join(StorageDir, DagsSubdir)
	entries, err := os.ReadDir(dagsDir)
	if err != nil {
		return
	}
	count := 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dagsDir, e.Name()))
		if err != nil {
			continue
		}
		var m ResourceManifest
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		n.manifests[m.RootCid] = &m
		count++
	}
	if count > 0 {
		log.Printf("Loaded %d manifests from disk", count)
	}
}

func (n *BlockNode) saveManifest(m *ResourceManifest) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(StorageDir, DagsSubdir, m.RootCid+".json")
	return os.WriteFile(path, data, 0644)
}

// ═══════════════════════════════════════════════════════════════════
// GossipSub subscription
// ═══════════════════════════════════════════════════════════════════

type NewBlockAnnouncement struct {
	CID          string `json:"cid"`
	Size         int64  `json:"size"`
	SourceNodeID string `json:"sourceNodeId"`
}

func (n *BlockNode) startGossipSubSubscription(ctx context.Context) error {
	topic, err := n.pubsub.Join(GossipSubTopic)
	if err != nil {
		return fmt.Errorf("join gossipsub topic: %w", err)
	}
	sub, err := topic.Subscribe()
	if err != nil {
		return fmt.Errorf("subscribe to topic: %w", err)
	}
	go func() {
		log.Printf("Subscribed to GossipSub topic: %s", GossipSubTopic)
		for {
			msg, err := sub.Next(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				log.Printf("GossipSub receive error: %v", err)
				continue
			}
			if msg.ReceivedFrom == n.host.ID() {
				continue
			}
			var ann NewBlockAnnouncement
			if err := json.Unmarshal(msg.Data, &ann); err != nil {
				continue
			}
			go n.handleNewBlockAnnouncement(ctx, ann)
		}
	}()
	return nil
}

func (n *BlockNode) handleNewBlockAnnouncement(ctx context.Context, ann NewBlockAnnouncement) {
	c, err := cid.Decode(ann.CID)
	if err != nil {
		return
	}
	has, _ := n.bstore.Has(ctx, c)
	if has {
		return
	}
	usedBytes, _ := n.usedBytes(ctx)
	usagePct := float64(usedBytes) / float64(n.cfg.StorageBytes) * 100
	if usagePct >= GCHardLimit {
		return
	}
	if !n.isWithinXORThreshold(c, usagePct) {
		return
	}
	pullCtx, cancel := context.WithTimeout(ctx, BitswapTimeout)
	defer cancel()
	blk, err := n.bsExch.GetBlock(pullCtx, c)
	if err != nil {
		return
	}
	if err := n.bstore.Put(ctx, blk); err != nil {
		return
	}
	n.touchBlock(c.String(), int64(len(blk.RawData())))
	n.mu.Lock()
	n.bitswapReceived++
	n.mu.Unlock()
	_ = n.dht.Provide(ctx, c, true)
	log.Printf("Pulled block %s (%d bytes) via GossipSub + bitswap", ann.CID[:12], len(blk.RawData()))
}

// ═══════════════════════════════════════════════════════════════════
// XOR proximity
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) isWithinXORThreshold(c cid.Cid, usagePct float64) bool {
	peerHash := sha256.Sum256([]byte(n.host.ID()))
	cidHash := sha256.Sum256(c.Bytes())
	xorBytes := make([]byte, 32)
	for i := range xorBytes {
		xorBytes[i] = peerHash[i] ^ cidHash[i]
	}
	xorInt := new(big.Int).SetBytes(xorBytes)
	maxInt := new(big.Int).Lsh(big.NewInt(1), 256)
	xorFraction, _ := new(big.Float).Quo(
		new(big.Float).SetInt(xorInt),
		new(big.Float).SetInt(maxInt),
	).Float64()

	var threshold float64
	switch {
	case usagePct >= 75:
		threshold = XORThresholdFull
	case usagePct >= 50:
		threshold = XORThresholdMedium
	default:
		threshold = XORThresholdLight
	}
	return xorFraction <= threshold
}

// ═══════════════════════════════════════════════════════════════════
// LRU eviction
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) touchBlock(cidStr string, size int64) {
	n.accessMu.Lock()
	defer n.accessMu.Unlock()
	n.accessTimes[cidStr] = time.Now()
	if size > 0 {
		n.blockSizes[cidStr] = size
	}
}

func (n *BlockNode) startGCLoop(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			usedBytes, _ := n.usedBytes(ctx)
			if float64(usedBytes)/float64(n.cfg.StorageBytes)*100 >= GCTriggerPercent {
				freed, err := n.runGC(ctx)
				if err != nil {
					log.Printf("GC error: %v", err)
				} else if freed > 0 {
					log.Printf("GC freed %d bytes", freed)
				}
			}
		}
	}
}

func (n *BlockNode) runGC(ctx context.Context) (int64, error) {
	n.accessMu.Lock()
	type entry struct {
		cid        string
		lastAccess time.Time
		size       int64
	}
	var entries []entry
	for c, t := range n.accessTimes {
		entries = append(entries, entry{c, t, n.blockSizes[c]})
	}
	n.accessMu.Unlock()

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].lastAccess.Before(entries[j].lastAccess)
	})

	var freed int64
	for _, e := range entries {
		usedBytes, _ := n.usedBytes(ctx)
		if float64(usedBytes)/float64(n.cfg.StorageBytes)*100 < GCTriggerPercent {
			break
		}
		c, err := cid.Decode(e.cid)
		if err != nil {
			continue
		}
		if err := n.bstore.DeleteBlock(ctx, c); err != nil {
			continue
		}
		freed += e.size
		n.accessMu.Lock()
		delete(n.accessTimes, e.cid)
		delete(n.blockSizes, e.cid)
		n.accessMu.Unlock()
	}
	return freed, nil
}

// ═══════════════════════════════════════════════════════════════════
// Storage stats
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) usedBytes(ctx context.Context) (int64, error) {
	n.accessMu.Lock()
	defer n.accessMu.Unlock()
	var total int64
	for _, size := range n.blockSizes {
		total += size
	}
	return total, nil
}

func (n *BlockNode) blockCount(ctx context.Context) (int, error) {
	ch, err := n.bstore.AllKeysChan(ctx)
	if err != nil {
		return 0, err
	}
	count := 0
	for range ch {
		count++
	}
	return count, nil
}

// ═══════════════════════════════════════════════════════════════════
// HTTP API
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) startHTTPServer(ctx context.Context) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", n.handleHealth)
	mux.HandleFunc("/blocks/", n.handleBlock)
	mux.HandleFunc("/blocks", n.handleBlocks)
	mux.HandleFunc("/stats", n.handleStats)
	mux.HandleFunc("/gc", n.handleGC)
	mux.HandleFunc("/connect", n.handleConnect)
	mux.HandleFunc("/manifests/", n.handleManifestByID)
	mux.HandleFunc("/manifests", n.handleManifests)
	mux.HandleFunc("/owners/", n.handleOwnerDelete)

	// Bind to all interfaces so the NodeAgent host can reach the API
	// via the virbr0 bridge (192.168.122.x). The API port is not
	// exposed via nginx and is not publicly reachable.
	addr := fmt.Sprintf("0.0.0.0:%d", n.cfg.APIPort)
	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
	}
	go func() {
		log.Printf("HTTP API listening on %s", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()
	go func() {
		<-ctx.Done()
		_ = srv.Shutdown(context.Background())
	}()
}

func (n *BlockNode) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	usedBytes, _ := n.usedBytes(ctx)
	count, _ := n.blockCount(ctx)
	n.mu.RLock()
	sent := n.bitswapSent
	recv := n.bitswapReceived
	n.mu.RUnlock()

	peerID := n.host.ID().String()
	if data, err := os.ReadFile(filepath.Join(StorageDir, PeerIDFile)); err == nil {
		peerID = strings.TrimSpace(string(data))
	}

	n.manifestsMu.RLock()
	manifestCount := len(n.manifests)
	n.manifestsMu.RUnlock()

	json.NewEncoder(w).Encode(map[string]any{
		"status":          "ok",
		"peerId":          peerID,
		"connectedPeers":  len(n.host.Network().Peers()),
		"capacityBytes":   n.cfg.StorageBytes,
		"usedBytes":       usedBytes,
		"usagePercent":    float64(usedBytes) / float64(n.cfg.StorageBytes) * 100,
		"blockCount":      count,
		"manifestCount":   manifestCount,
		"bitswapSent":     sent,
		"bitswapReceived": recv,
		"nodeId":          n.cfg.NodeID,
		"vmId":            n.cfg.VMID,
	})
}

func (n *BlockNode) handleBlock(w http.ResponseWriter, r *http.Request) {
	cidStr := strings.TrimPrefix(r.URL.Path, "/blocks/")
	if cidStr == "" {
		http.Error(w, "missing CID", http.StatusBadRequest)
		return
	}
	c, err := cid.Decode(cidStr)
	if err != nil {
		http.Error(w, "invalid CID", http.StatusBadRequest)
		return
	}
	switch r.Method {
	case http.MethodGet:
		n.getBlock(w, r, c)
	case http.MethodDelete:
		n.deleteBlock(w, r, c)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (n *BlockNode) getBlock(w http.ResponseWriter, r *http.Request, c cid.Cid) {
	ctx := r.Context()
	blk, err := n.bstore.Get(ctx, c)
	if err != nil {
		fetchCtx, cancel := context.WithTimeout(ctx, BitswapTimeout)
		defer cancel()
		blk, err = n.bsExch.GetBlock(fetchCtx, c)
		if err != nil {
			http.Error(w, "block not found", http.StatusNotFound)
			return
		}
		_ = n.bstore.Put(ctx, blk)
		n.touchBlock(c.String(), int64(len(blk.RawData())))
		n.mu.Lock()
		n.bitswapReceived++
		n.mu.Unlock()
	} else {
		n.touchBlock(c.String(), int64(len(blk.RawData())))
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Block-CID", c.String())
	w.Write(blk.RawData())
}

func (n *BlockNode) deleteBlock(w http.ResponseWriter, r *http.Request, c cid.Cid) {
	ctx := r.Context()
	if err := n.bstore.DeleteBlock(ctx, c); err != nil {
		http.Error(w, "delete failed", http.StatusInternalServerError)
		return
	}
	n.accessMu.Lock()
	delete(n.accessTimes, c.String())
	delete(n.blockSizes, c.String())
	n.accessMu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

func (n *BlockNode) handleBlocks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	ctx := r.Context()
	usedBytes, _ := n.usedBytes(ctx)
	if float64(usedBytes)/float64(n.cfg.StorageBytes)*100 >= GCHardLimit {
		http.Error(w, "storage full", http.StatusInsufficientStorage)
		return
	}
	data, err := io.ReadAll(io.LimitReader(r.Body, 64*1024*1024))
	if err != nil {
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}
	mhash, err := multihash.Sum(data, multihash.SHA2_256, -1)
	if err != nil {
		http.Error(w, "hash error", http.StatusInternalServerError)
		return
	}
	c := cid.NewCidV1(cid.Raw, mhash)
	blk, err := blocks.NewBlockWithCid(data, c)
	if err != nil {
		http.Error(w, "block error", http.StatusInternalServerError)
		return
	}
	if err := n.bstore.Put(ctx, blk); err != nil {
		log.Printf("bstore.Put error: %v", err)
		http.Error(w, fmt.Sprintf("store error: %v", err), http.StatusInternalServerError)
		return
	}
	n.touchBlock(c.String(), int64(len(data)))
 
	// Append CID to owner index if ?owner= query param is present.
	// Owner index: /var/lib/decloud-blockstore/owners/{vmId}.cids
	// One CID per line. Used by DELETE /owners/{vmId} for bulk cleanup on VM delete.
	if owner := r.URL.Query().Get("owner"); owner != "" {
		go func(ownerID, cidStr string) {
			ownerFile := filepath.Join(StorageDir, OwnersSubdir, ownerID+".cids")
			f, err := os.OpenFile(ownerFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				log.Printf("owner index: failed to open %s: %v", ownerFile, err)
				return
			}
			defer f.Close()
			if _, err := fmt.Fprintln(f, cidStr); err != nil {
				log.Printf("owner index: failed to write CID %s: %v", cidStr[:12], err)
			}
		}(owner, c.String())
	}
 
	go func() {
		announceCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = n.dht.Provide(announceCtx, c, true)
	}()
	go n.publishNewBlock(c, int64(len(data)))
 
	json.NewEncoder(w).Encode(map[string]string{"cid": c.String()})
}

// handleOwnerDelete: DELETE /owners/{vmId}
// Deletes all blocks associated with a VM owner, withdraws DHT provider
// records, and removes the owner index file. Called by NodeAgent when a
// VM is deleted — single HTTP call replaces N individual block deletes.
// Reference counting: a block is only evicted from FlatFS when all of its
// owner references are removed (rare — VM overlays produce unique blocks).
func (n *BlockNode) handleOwnerDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "DELETE only", http.StatusMethodNotAllowed)
		return
	}
	vmId := strings.TrimPrefix(r.URL.Path, "/owners/")
	vmId = strings.TrimSpace(vmId)
	if vmId == "" {
		http.Error(w, "missing vmId", http.StatusBadRequest)
		return
	}
 
	ownerFile := filepath.Join(StorageDir, OwnersSubdir, vmId+".cids")
	data, err := os.ReadFile(ownerFile)
	if err != nil {
		if os.IsNotExist(err) {
			// No owner file — nothing to delete, idempotent success
			w.WriteHeader(http.StatusNoContent)
			return
		}
		http.Error(w, fmt.Sprintf("read owner file: %v", err), http.StatusInternalServerError)
		return
	}
 
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	ctx := r.Context()
	deleted := 0
	skipped := 0
 
	for _, cidStr := range lines {
		cidStr = strings.TrimSpace(cidStr)
		if cidStr == "" {
			continue
		}
		c, err := cid.Decode(cidStr)
		if err != nil {
			log.Printf("owner delete: invalid CID %q: %v", cidStr, err)
			continue
		}
 
		// Reference counting: check if any other owner still references this CID.
		// Scan all other owner files for this CID before deleting from FlatFS.
		// In practice VM overlay blocks are unique so this is rarely needed.
		otherOwnerRefs, scanErr := n.cidHasOtherOwners(vmId, cidStr)
		if scanErr != nil {
			log.Printf("owner delete: ref-count scan failed for %s: %v", cidStr[:12], scanErr)
		}
		if otherOwnerRefs {
			skipped++
			continue
		}
 
		// Delete from FlatFS
		if err := n.bstore.DeleteBlock(ctx, c); err != nil {
			log.Printf("owner delete: FlatFS delete failed for %s: %v", cidStr[:12], err)
			continue
		}
 
		// Clean up LRU tracking
		n.accessMu.Lock()
		delete(n.accessTimes, cidStr)
		delete(n.blockSizes, cidStr)
		n.accessMu.Unlock()
 
		// Withdraw DHT provider record (best-effort, non-blocking)
		go func(c cid.Cid) {
			withdrawCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = n.dht.Provide(withdrawCtx, c, false)
		}(c)
 
		deleted++
	}
 
	// Remove owner index file
	if err := os.Remove(ownerFile); err != nil && !os.IsNotExist(err) {
		log.Printf("owner delete: failed to remove owner file %s: %v", ownerFile, err)
	}
 
	log.Printf("owner delete: VM %s — deleted %d blocks, skipped %d (other owners)", vmId, deleted, skipped)
	json.NewEncoder(w).Encode(map[string]any{
		"vmId":    vmId,
		"deleted": deleted,
		"skipped": skipped,
	})
}
 
// cidHasOtherOwners returns true if any owner file other than excludeOwner
// contains the given CID string. Used for reference counting on delete.
func (n *BlockNode) cidHasOtherOwners(excludeOwner, cidStr string) (bool, error) {
	ownersDir := filepath.Join(StorageDir, OwnersSubdir)
	entries, err := os.ReadDir(ownersDir)
	if err != nil {
		return false, err
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".cids") {
			continue
		}
		ownerID := strings.TrimSuffix(e.Name(), ".cids")
		if ownerID == excludeOwner {
			continue
		}
		data, err := os.ReadFile(filepath.Join(ownersDir, e.Name()))
		if err != nil {
			continue
		}
		if strings.Contains(string(data), cidStr) {
			return true, nil
		}
	}
	return false, nil
}

func (n *BlockNode) handleStats(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	usedBytes, _ := n.usedBytes(ctx)
	count, _ := n.blockCount(ctx)
	n.mu.RLock()
	sent := n.bitswapSent
	recv := n.bitswapReceived
	n.mu.RUnlock()
	json.NewEncoder(w).Encode(map[string]any{
		"capacityBytes":   n.cfg.StorageBytes,
		"usedBytes":       usedBytes,
		"usagePercent":    float64(usedBytes) / float64(n.cfg.StorageBytes) * 100,
		"blockCount":      count,
		"connectedPeers":  len(n.host.Network().Peers()),
		"bitswapSent":     sent,
		"bitswapReceived": recv,
	})
}

func (n *BlockNode) handleGC(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	freed, err := n.runGC(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	usedBytes, _ := n.usedBytes(r.Context())
	json.NewEncoder(w).Encode(map[string]any{
		"freedBytes":     freed,
		"remainingBytes": usedBytes,
	})
}

func (n *BlockNode) handleConnect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Peers []string `json:"peers"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	results := make([]map[string]any, 0, len(req.Peers))
	connected := 0
	for _, addrStr := range req.Peers {
		maddr, err := multiaddr.NewMultiaddr(addrStr)
		if err != nil {
			results = append(results, map[string]any{"addr": addrStr, "ok": false, "error": err.Error()})
			continue
		}
		pi, err := peer.AddrInfoFromP2pAddr(maddr)
		if err != nil {
			results = append(results, map[string]any{"addr": addrStr, "ok": false, "error": err.Error()})
			continue
		}
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		err = n.host.Connect(ctx, *pi)
		cancel()
		if err != nil {
			results = append(results, map[string]any{"addr": addrStr, "ok": false, "error": err.Error()})
		} else {
			results = append(results, map[string]any{"addr": addrStr, "ok": true})
			connected++
		}
	}
	json.NewEncoder(w).Encode(map[string]any{
		"results":   results,
		"connected": connected,
		"total":     len(req.Peers),
	})
}

// ── Manifest endpoints ────────────────────────────────────────────

// handleManifests: GET returns all manifests, POST registers a new one.
func (n *BlockNode) handleManifests(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		n.manifestsMu.RLock()
		list := make([]*ResourceManifest, 0, len(n.manifests))
		for _, m := range n.manifests {
			list = append(list, m)
		}
		n.manifestsMu.RUnlock()

		// Sort by UpdatedAt descending (most recent first)
		sort.Slice(list, func(i, j int) bool {
			return list[i].UpdatedAt.After(list[j].UpdatedAt)
		})

		json.NewEncoder(w).Encode(map[string]any{
			"manifests": list,
			"count":     len(list),
		})

	case http.MethodPost:
		var m ResourceManifest
		if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if m.RootCid == "" {
			http.Error(w, "rootCid required", http.StatusBadRequest)
			return
		}
		if m.ResourceType == "" {
			m.ResourceType = ResourceTypeUnknown
		}
		now := time.Now().UTC()
		n.manifestsMu.Lock()
		if existing, ok := n.manifests[m.RootCid]; ok {
			// Update existing — preserve RegisteredAt
			m.RegisteredAt = existing.RegisteredAt
		} else {
			m.RegisteredAt = now
		}
		m.UpdatedAt = now
		n.manifests[m.RootCid] = &m
		n.manifestsMu.Unlock()

		if err := n.saveManifest(&m); err != nil {
			log.Printf("Warning: failed to persist manifest %s: %v", m.RootCid[:12], err)
		}

		json.NewEncoder(w).Encode(map[string]any{
			"success": true,
			"rootCid": m.RootCid,
			"version": m.Version,
		})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleManifestByID: GET /manifests/{rootCid}
func (n *BlockNode) handleManifestByID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	rootCid := strings.TrimPrefix(r.URL.Path, "/manifests/")
	if rootCid == "" {
		http.Error(w, "rootCid required", http.StatusBadRequest)
		return
	}
	n.manifestsMu.RLock()
	m, ok := n.manifests[rootCid]
	n.manifestsMu.RUnlock()
	if !ok {
		http.Error(w, "manifest not found", http.StatusNotFound)
		return
	}
	json.NewEncoder(w).Encode(m)
}

// ═══════════════════════════════════════════════════════════════════
// GossipSub publisher
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) publishNewBlock(c cid.Cid, size int64) {
	topic, err := n.pubsub.Join(GossipSubTopic)
	if err != nil {
		return
	}
	ann := NewBlockAnnouncement{CID: c.String(), Size: size, SourceNodeID: n.cfg.NodeID}
	data, err := json.Marshal(ann)
	if err != nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = topic.Publish(ctx, data)
}

// ═══════════════════════════════════════════════════════════════════
// Bootstrap LRU from existing blocks
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) loadExistingAccessTimes(ctx context.Context) {
	ch, err := n.bstore.AllKeysChan(ctx)
	if err != nil {
		return
	}
	n.accessMu.Lock()
	defer n.accessMu.Unlock()
	for c := range ch {
		cidStr := c.String()
		if _, exists := n.accessTimes[cidStr]; !exists {
			n.accessTimes[cidStr] = time.Now()
			buf := new(bytes.Buffer)
			_ = binary.Write(buf, binary.BigEndian, int64(len(c.Bytes())))
			n.blockSizes[cidStr] = 1024
		}
	}
}

// reannounceExistingBlocks re-publishes DHT provider records for all locally
// stored blocks after startup. Called after connectBootstrapPeers so the
// routing table has time to populate before Provide() calls are issued.
// Handles: binary restart, DHT VM redeployment, 24h record expiry.
func (n *BlockNode) reannounceExistingBlocks(ctx context.Context) {
    time.Sleep(15 * time.Second) // wait for Kademlia routing table to populate

    ch, err := n.bstore.AllKeysChan(ctx)
    if err != nil {
        log.Printf("reannounce: failed to list blocks: %v", err)
        return
    }

    count := 0
    for c := range ch {
        if ctx.Err() != nil {
            return
        }
        announceCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
        _ = n.dht.Provide(announceCtx, c, true)
        cancel()
        count++
        if count%100 == 0 {
            log.Printf("reannounce: %d blocks announced to DHT", count)
        }
    }
    log.Printf("reannounce: complete — %d blocks re-announced to DHT", count)
}

// ═══════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Println("DeCloud Block Store Node starting...")

	cfg := parseConfig()
	log.Printf("Config: listenPort=%d apiPort=%d storageBytes=%d nodeID=%s vmID=%s",
		cfg.ListenPort, cfg.APIPort, cfg.StorageBytes, cfg.NodeID, cfg.VMID)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	node, err := setup(ctx, cfg)
	if err != nil {
		log.Fatalf("Setup failed: %v", err)
	}
	defer node.host.Close()

	node.loadExistingAccessTimes(ctx)
	node.loadExistingManifests(ctx)
	node.connectBootstrapPeers(ctx)
	go node.reannounceExistingBlocks(ctx)

	if err := node.startGossipSubSubscription(ctx); err != nil {
		log.Printf("Warning: GossipSub subscription failed: %v", err)
	}

	node.startHTTPServer(ctx)
	go node.startGCLoop(ctx)

	for _, addr := range node.host.Addrs() {
		log.Printf("Listening: %s/p2p/%s", addr, node.host.ID())
	}
	log.Printf("Block store node ready — peer ID: %s", node.host.ID())

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down block store node...")
	cancel()
	time.Sleep(500 * time.Millisecond)
}
