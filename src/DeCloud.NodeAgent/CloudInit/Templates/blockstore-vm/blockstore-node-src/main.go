// DeCloud Block Store Node
//
// A libp2p node providing content-addressed distributed storage
// for VM disk overlay replication (lazysync).
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

	bsnetwork "github.com/ipfs/boxo/bitswap/network"
	"github.com/ipfs/boxo/bitswap"
	blockstore "github.com/ipfs/boxo/blockstore"
	blocks "github.com/ipfs/go-block-format"
	"github.com/ipfs/go-cid"
	"github.com/ipfs/go-datastore"
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
	GossipSubTopic      = "decloud/blockstore/new-blocks"
	IdentityKeyFile     = "identity.key"
	PeerIDFile          = "peer-id"
	BlocksSubdir        = "blocks"
	StorageDir          = "/var/lib/decloud-blockstore"

	// GC thresholds
	GCTriggerPercent = 85
	GCHardLimit      = 95

	// XOR proximity thresholds (as fraction of keyspace, capacity-based)
	// Lower capacity → wider pull radius
	XORThresholdFull   = 0.05  // >75% full: pull top 5% closest
	XORThresholdMedium = 0.10  // 50-75%:    pull top 10% closest
	XORThresholdLight  = 0.20  // <50%:      pull top 20% closest

	// Bitswap fetch timeout
	BitswapTimeout = 30 * time.Second
)

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
	cfg := Config{
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
	return cfg
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

// ═══════════════════════════════════════════════════════════════════
// Identity
// ═══════════════════════════════════════════════════════════════════

// loadOrCreateIdentity loads the persistent Ed25519 key from disk,
// creating it on first boot. Identity survives VM restarts.
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

	// First boot: generate new Ed25519 key
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
	cfg     Config
	host    host.Host
	dht     *dht.IpfsDHT
	pubsub  *pubsub.PubSub
	bstore  blockstore.Blockstore
	bsExch  *bitswap.Bitswap

	// LRU tracking: CID string → last access time
	accessMu    sync.Mutex
	accessTimes map[string]time.Time
	blockSizes  map[string]int64

	// Counters
	mu              sync.RWMutex
	bitswapSent     uint64
	bitswapReceived uint64
}

// ═══════════════════════════════════════════════════════════════════
// Setup
// ═══════════════════════════════════════════════════════════════════

func setup(ctx context.Context, cfg Config) (*BlockNode, error) {
	if err := os.MkdirAll(StorageDir, 0755); err != nil {
		return nil, fmt.Errorf("create storage dir: %w", err)
	}

	// ── Identity ─────────────────────────────────────────────────
	priv, err := loadOrCreateIdentity(StorageDir)
	if err != nil {
		return nil, fmt.Errorf("identity: %w", err)
	}

	// ── FlatFS blockstore ─────────────────────────────────────────
	blocksDir := filepath.Join(StorageDir, BlocksSubdir)
	if err := os.MkdirAll(blocksDir, 0755); err != nil {
		return nil, fmt.Errorf("create blocks dir: %w", err)
	}

	shardFn, err := flatfs.ParseShardFunc("/repo/flatfs/shard/v1/next-to-last/2")
	if err != nil {
		return nil, fmt.Errorf("shard func: %w", err)
	}
	fds, err := flatfs.CreateOrOpen(blocksDir, shardFn, false)
	if err != nil {
		return nil, fmt.Errorf("open flatfs: %w", err)
	}

	bs := blockstore.NewBlockstore(datastore.Batching(fds))
	bs = blockstore.NewIdStore(bs) // handles inline (identity) CIDs

	// ── libp2p host ───────────────────────────────────────────────
	listenAddr := fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", cfg.ListenPort)
	h, err := libp2p.New(
		libp2p.ListenAddrStrings(listenAddr),
		libp2p.Identity(priv),
	)
	if err != nil {
		return nil, fmt.Errorf("create libp2p host: %w", err)
	}

	log.Printf("libp2p peer ID: %s", h.ID())
	log.Printf("Listening on: %s/p2p/%s", listenAddr, h.ID())

	// Write peer ID to file (for bootstrap-poll.sh and notify-ready.sh)
	peerIDPath := filepath.Join(StorageDir, PeerIDFile)
	if err := os.WriteFile(peerIDPath, []byte(h.ID().String()), 0644); err != nil {
		log.Printf("Warning: could not write peer-id file: %v", err)
	}

	// ── Kademlia DHT client ───────────────────────────────────────
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

	// ── Bitswap exchange ──────────────────────────────────────────
	bsNet := bsnetwork.NewFromIpfsHost(h, kadDHT)
	bsExch := bitswap.New(ctx, bsNet, bs)

	// ── GossipSub ─────────────────────────────────────────────────
	ps, err := pubsub.NewGossipSub(ctx, h)
	if err != nil {
		return nil, fmt.Errorf("create pubsub: %w", err)
	}

	node := &BlockNode{
		cfg:         cfg,
		host:        h,
		dht:         kadDHT,
		pubsub:      ps,
		bstore:      bs,
		bsExch:      bsExch,
		accessTimes: make(map[string]time.Time),
		blockSizes:  make(map[string]int64),
	}

	return node, nil
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

			// Don't process our own messages
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

// handleNewBlockAnnouncement decides whether to pull an announced block
// based on XOR proximity and current capacity utilization.
func (n *BlockNode) handleNewBlockAnnouncement(ctx context.Context, ann NewBlockAnnouncement) {
	c, err := cid.Decode(ann.CID)
	if err != nil {
		return
	}

	// Skip if we already have it
	has, _ := n.bstore.Has(ctx, c)
	if has {
		return
	}

	// Check capacity — don't pull if at hard limit
	usedBytes, _ := n.usedBytes(ctx)
	usagePercent := float64(usedBytes) / float64(n.cfg.StorageBytes) * 100
	if usagePercent >= GCHardLimit {
		return
	}

	// Check XOR proximity
	if !n.isWithinXORThreshold(c, usagePercent) {
		return
	}

	// Pull via bitswap
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

	// Announce we now hold this block to the DHT
	_ = n.dht.Provide(ctx, c, true)

	log.Printf("Pulled block %s (%d bytes) via GossipSub + bitswap", ann.CID[:12], len(blk.RawData()))
}

// ═══════════════════════════════════════════════════════════════════
// XOR proximity
// ═══════════════════════════════════════════════════════════════════

// isWithinXORThreshold returns true if this node is "close enough"
// to the given CID in Kademlia XOR space to warrant pulling the block.
// Threshold shrinks as capacity fills up (adaptive pull radius).
func (n *BlockNode) isWithinXORThreshold(c cid.Cid, usagePercent float64) bool {
	// Compute XOR distance between our peer ID hash and the CID
	peerHash := sha256.Sum256([]byte(n.host.ID()))
	cidHash := sha256.Sum256(c.Bytes())

	xorBytes := make([]byte, 32)
	for i := range xorBytes {
		xorBytes[i] = peerHash[i] ^ cidHash[i]
	}

	// Convert XOR distance to a fraction of the 256-bit keyspace
	xorInt := new(big.Int).SetBytes(xorBytes)
	maxInt := new(big.Int).Lsh(big.NewInt(1), 256)
	xorFraction, _ := new(big.Float).Quo(
		new(big.Float).SetInt(xorInt),
		new(big.Float).SetInt(maxInt),
	).Float64()

	// Adaptive threshold based on capacity
	var threshold float64
	switch {
	case usagePercent >= 75:
		threshold = XORThresholdFull
	case usagePercent >= 50:
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
			usagePercent := float64(usedBytes) / float64(n.cfg.StorageBytes) * 100

			if usagePercent >= GCTriggerPercent {
				freed, err := n.runGC(ctx)
				if err != nil {
					log.Printf("GC error: %v", err)
				} else if freed > 0 {
					log.Printf("GC freed %d bytes (was %.1f%% full)", freed, usagePercent)
				}
			}
		}
	}
}

// runGC evicts least-recently-used blocks until usage drops below GCTriggerPercent.
// Returns the number of bytes freed.
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

	// Sort by last access time (oldest first = evict first)
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

		// Withdraw provider record from DHT
		// (provider record TTL naturally expires when we stop re-announcing)
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

	addr := fmt.Sprintf("127.0.0.1:%d", n.cfg.APIPort)
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
	usagePct := float64(usedBytes) / float64(n.cfg.StorageBytes) * 100

	n.mu.RLock()
	sent := n.bitswapSent
	recv := n.bitswapReceived
	n.mu.RUnlock()

	peerID := n.host.ID().String()

	// Read peer ID file (most reliable source)
	if data, err := os.ReadFile(filepath.Join(StorageDir, PeerIDFile)); err == nil {
		peerID = strings.TrimSpace(string(data))
	}

	json.NewEncoder(w).Encode(map[string]any{
		"status":           "ok",
		"peerId":           peerID,
		"connectedPeers":   len(n.host.Network().Peers()),
		"capacityBytes":    n.cfg.StorageBytes,
		"usedBytes":        usedBytes,
		"usagePercent":     usagePct,
		"blockCount":       count,
		"bitswapSent":      sent,
		"bitswapReceived":  recv,
	})
}

// handleBlock routes GET /blocks/{cid} and DELETE /blocks/{cid}
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

	// Check local store first
	blk, err := n.bstore.Get(ctx, c)
	if err != nil {
		// Try to fetch from network via bitswap
		fetchCtx, cancel := context.WithTimeout(ctx, BitswapTimeout)
		defer cancel()

		blk, err = n.bsExch.GetBlock(fetchCtx, c)
		if err != nil {
			http.Error(w, "block not found", http.StatusNotFound)
			return
		}

		// Store locally after fetch
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

// handleBlocks routes POST /blocks (store raw bytes, return CID)
func (n *BlockNode) handleBlocks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	// Check capacity hard limit
	usedBytes, _ := n.usedBytes(ctx)
	if float64(usedBytes)/float64(n.cfg.StorageBytes)*100 >= GCHardLimit {
		http.Error(w, "storage full", http.StatusInsufficientStorage)
		return
	}

	data, err := io.ReadAll(io.LimitReader(r.Body, 64*1024*1024)) // 64 MB max
	if err != nil {
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}

	// CIDv1 with raw codec and SHA2-256
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
		http.Error(w, "store error", http.StatusInternalServerError)
		return
	}

	n.touchBlock(c.String(), int64(len(data)))

	// Announce to DHT that we hold this block
	go func() {
		announceCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = n.dht.Provide(announceCtx, c, true)
	}()

	// Publish to GossipSub so nearby nodes can pull
	go n.publishNewBlock(c, int64(len(data)))

	json.NewEncoder(w).Encode(map[string]string{"cid": c.String()})
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

// ═══════════════════════════════════════════════════════════════════
// GossipSub publisher
// ═══════════════════════════════════════════════════════════════════

func (n *BlockNode) publishNewBlock(c cid.Cid, size int64) {
	topic, err := n.pubsub.Join(GossipSubTopic)
	if err != nil {
		return
	}

	ann := NewBlockAnnouncement{
		CID:          c.String(),
		Size:         size,
		SourceNodeID: n.cfg.NodeID,
	}
	data, err := json.Marshal(ann)
	if err != nil {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = topic.Publish(ctx, data)
}

// ═══════════════════════════════════════════════════════════════════
// Helper: size hint from CID (approximate, for LRU bootstrap)
// ═══════════════════════════════════════════════════════════════════

func cidSizeHint(c cid.Cid) int64 {
	// Encode CID length as a simple size hint for LRU bookkeeping
	// when actual size isn't known. Replaced by real size on first access.
	buf := new(bytes.Buffer)
	_ = binary.Write(buf, binary.BigEndian, int64(len(c.Bytes())))
	return 1024 // 1 KB default hint
}

// ═══════════════════════════════════════════════════════════════════
// Bootstrap LRU tracking from existing blocks
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
			n.accessTimes[cidStr] = time.Now() // Default: treat as recently accessed
			n.blockSizes[cidStr] = cidSizeHint(c)
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Println("DeCloud Block Store Node starting...")

	cfg := parseConfig()
	log.Printf("Config: listenPort=%d apiPort=%d storageBytes=%d advertiseIP=%s nodeID=%s vmID=%s",
		cfg.ListenPort, cfg.APIPort, cfg.StorageBytes, cfg.AdvertiseIP, cfg.NodeID, cfg.VMID)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup
	node, err := setup(ctx, cfg)
	if err != nil {
		log.Fatalf("Setup failed: %v", err)
	}
	defer node.host.Close()

	// Load existing block access times (for LRU on restart)
	node.loadExistingAccessTimes(ctx)

	// Connect bootstrap peers
	node.connectBootstrapPeers(ctx)

	// Start GossipSub subscription
	if err := node.startGossipSubSubscription(ctx); err != nil {
		log.Printf("Warning: GossipSub subscription failed: %v", err)
	}

	// Start HTTP API
	node.startHTTPServer(ctx)

	// Start GC loop
	go node.startGCLoop(ctx)

	// Log node addresses
	for _, addr := range node.host.Addrs() {
		log.Printf("Listening: %s/p2p/%s", addr, node.host.ID())
	}

	log.Printf("Block store node ready — peer ID: %s", node.host.ID())

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down block store node...")
	cancel()
	time.Sleep(500 * time.Millisecond) // Allow graceful close
}
