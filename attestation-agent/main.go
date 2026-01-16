package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"
)

// ============================================================
// DeCloud Ephemeral Attestation Agent
// ============================================================
//
// SECURITY MODEL:
// ---------------
// This agent generates FRESH Ed25519 keypairs for EACH attestation
// challenge. The private key exists in memory for only ~20ms before
// being zeroed.
//
// WHY THIS WORKS:
// ---------------
// A malicious node operator COULD read VM memory, but this takes TIME:
//   - Pause VM:        ~1ms
//   - Dump memory:     100-500ms (for 1GB+)
//   - Search for key:  10-100ms
//   - Total:           >150ms
//
// The Orchestrator requires responses within 100ms.
// The node CANNOT complete the attack in time.
//
// By the time the node could extract the key, it's already zeroed.
//
// WHAT THIS PROVES:
// -----------------
// 1. A VM is actually running (responds in real-time)
// 2. The VM has the claimed CPU cores (from /proc/cpuinfo)
// 3. The VM has real RAM, not swap (memory touch test)
// 4. The response wasn't pre-computed (random nonce)
// 5. The response wasn't replayed (nonce must match)
//
// ============================================================

// Version is set at build time via ldflags
var Version = "dev"

// Challenge from Orchestrator
type Challenge struct {
	Nonce            string `json:"nonce"`
	Timestamp        int64  `json:"timestamp"`
	VmId             string `json:"vmId"`
	ExpectedCores    int    `json:"expectedCores"`
	ExpectedMemoryMb int64  `json:"expectedMemoryMb"`
}

// Response to Orchestrator
type Response struct {
	Nonce           string      `json:"nonce"`
	EphemeralPubKey string      `json:"ephemeralPubKey"`
	Metrics         Metrics     `json:"metrics"`
	MemoryTouch     MemoryTouch `json:"memoryTouch"`
	Timing          TimingInfo  `json:"timing"`
	Signature       string      `json:"signature"`
}

// Metrics collected from /proc/*
type Metrics struct {
	CpuCores      int     `json:"cpuCores"`
	MemoryKb      int64   `json:"memoryKb"`
	MemoryFreeKb  int64   `json:"memoryFreeKb"`
	LoadAvg1      float64 `json:"loadAvg1"`
	LoadAvg5      float64 `json:"loadAvg5"`
	LoadAvg15     float64 `json:"loadAvg15"`
	UptimeSeconds float64 `json:"uptimeSeconds"`
	BootId        string  `json:"bootId"`
	MachineId     string  `json:"machineId"`
}

// MemoryTouch proves real RAM exists (not swap)
type MemoryTouch struct {
	AllocatedKb  int     `json:"allocatedKb"`
	PagesTouched int     `json:"pagesTouched"`
	TotalMs      float64 `json:"totalMs"`
	MaxPageMs    float64 `json:"maxPageMs"`
	ContentHash  string  `json:"contentHash"`
}

// TimingInfo for performance analysis
type TimingInfo struct {
	KeyGenMs      float64 `json:"keyGenMs"`
	MetricsMs     float64 `json:"metricsMs"`
	MemoryTouchMs float64 `json:"memoryTouchMs"`
	SigningMs     float64 `json:"signingMs"`
	TotalMs       float64 `json:"totalMs"`
}

var vmId string

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)

	// Read VM ID from cloud-init injected file
	vmIdBytes, err := os.ReadFile("/etc/decloud/vm-id")
	if err != nil {
		log.Printf("Warning: Could not read VM ID from /etc/decloud/vm-id: %v", err)
		// Fallback to machine-id
		vmIdBytes, _ = os.ReadFile("/etc/machine-id")
	}
	vmId = strings.TrimSpace(string(vmIdBytes))

	log.Printf("╔════════════════════════════════════════════════════════════╗")
	log.Printf("║     DeCloud Ephemeral Attestation Agent v%s", padRight(Version, 14)+"║")
	log.Printf("╠════════════════════════════════════════════════════════════╣")
	log.Printf("║  VM ID: %-50s ║", truncate(vmId, 50))
	log.Printf("║  Port:  %-50s ║", "9999")
	log.Printf("╚════════════════════════════════════════════════════════════╝")

	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/challenge", challengeHandler)

	log.Printf("Listening on :9999")
	log.Fatal(http.ListenAndServe(":9999", nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "healthy",
		"vmId":    vmId,
		"agent":   "decloud-attestation-agent",
		"version": Version,
	})
}

func challengeHandler(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	// Only accept POST
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	// Parse challenge
	var challenge Challenge
	if err := json.NewDecoder(r.Body).Decode(&challenge); err != nil {
		log.Printf("Invalid challenge JSON: %v", err)
		http.Error(w, `{"error":"invalid challenge json"}`, http.StatusBadRequest)
		return
	}

	// Log challenge receipt (truncate nonce for readability)
	log.Printf("Challenge received: nonce=%s..., vmId=%s",
		truncate(challenge.Nonce, 8), truncate(challenge.VmId, 8))

	// =========================================
	// STEP 1: Generate EPHEMERAL Ed25519 keypair
	// This is the CORE security feature
	// =========================================
	keyGenStart := time.Now()

	pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		log.Printf("Key generation failed: %v", err)
		http.Error(w, `{"error":"key generation failed"}`, http.StatusInternalServerError)
		return
	}

	keyGenMs := msElapsed(keyGenStart)

	// =========================================
	// STEP 2: Collect system metrics
	// =========================================
	metricsStart := time.Now()
	metrics := collectMetrics()
	metricsMs := msElapsed(metricsStart)

	// =========================================
	// STEP 3: Perform memory touch test
	// =========================================
	memTouchStart := time.Now()
	memTouch := performMemoryTouch()
	memTouchMs := msElapsed(memTouchStart)

	// =========================================
	// STEP 4: Build canonical message and sign
	// =========================================
	signingStart := time.Now()

	// Canonical message format (MUST match Orchestrator exactly!)
	canonicalMsg := fmt.Sprintf(
		"%s|%d|%s|%d|%d|%d|%s|%s|%.3f|%s",
		challenge.Nonce,
		challenge.Timestamp,
		challenge.VmId,
		metrics.CpuCores,
		metrics.MemoryKb,
		memTouch.PagesTouched,
		memTouch.ContentHash,
		metrics.BootId,
		metrics.UptimeSeconds,
		hex.EncodeToString(pubKey),
	)

	// Sign with ephemeral private key
	signature := ed25519.Sign(privKey, []byte(canonicalMsg))

	signingMs := msElapsed(signingStart)

	// =========================================
	// STEP 5: Build response
	// =========================================
	totalMs := msElapsed(startTime)

	response := Response{
		Nonce:           challenge.Nonce,
		EphemeralPubKey: hex.EncodeToString(pubKey),
		Metrics:         metrics,
		MemoryTouch:     memTouch,
		Timing: TimingInfo{
			KeyGenMs:      keyGenMs,
			MetricsMs:     metricsMs,
			MemoryTouchMs: memTouchMs,
			SigningMs:     signingMs,
			TotalMs:       totalMs,
		},
		Signature: hex.EncodeToString(signature),
	}

	// =========================================
	// STEP 6: CRITICAL - Zero the private key!
	// =========================================
	zeroPrivateKey(privKey)

	// Force GC to clear any copies
	runtime.GC()

	// Send response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	log.Printf("Challenge responded: %.2fms total (keygen=%.2f, metrics=%.2f, memtouch=%.2f, sign=%.2f)",
		totalMs, keyGenMs, metricsMs, memTouchMs, signingMs)
}

// zeroPrivateKey securely zeros the private key memory
// This is CRITICAL for security
func zeroPrivateKey(key ed25519.PrivateKey) {
	for i := range key {
		key[i] = 0
	}
	runtime.KeepAlive(key)
}

func collectMetrics() Metrics {
	metrics := Metrics{}

	// CPU cores from /proc/cpuinfo
	if cpuinfo, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		content := string(cpuinfo)
		// Count "processor" lines
		metrics.CpuCores = strings.Count(content, "processor\t:")
		if metrics.CpuCores == 0 {
			metrics.CpuCores = strings.Count(content, "processor:")
		}
		if metrics.CpuCores == 0 {
			// ARM format
			metrics.CpuCores = strings.Count(content, "processor	:")
		}
	}
	if metrics.CpuCores == 0 {
		metrics.CpuCores = 1 // Fallback
	}

	// Memory from /proc/meminfo
	if meminfo, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, line := range strings.Split(string(meminfo), "\n") {
			if strings.HasPrefix(line, "MemTotal:") {
				fmt.Sscanf(line, "MemTotal: %d kB", &metrics.MemoryKb)
			} else if strings.HasPrefix(line, "MemFree:") {
				fmt.Sscanf(line, "MemFree: %d kB", &metrics.MemoryFreeKb)
			}
		}
	}

	// Load average from /proc/loadavg
	if loadavg, err := os.ReadFile("/proc/loadavg"); err == nil {
		fmt.Sscanf(string(loadavg), "%f %f %f",
			&metrics.LoadAvg1, &metrics.LoadAvg5, &metrics.LoadAvg15)
	}

	// Uptime from /proc/uptime
	if uptime, err := os.ReadFile("/proc/uptime"); err == nil {
		fmt.Sscanf(string(uptime), "%f", &metrics.UptimeSeconds)
	}

	// Boot ID (changes each boot - detects VM swaps)
	if bootId, err := os.ReadFile("/proc/sys/kernel/random/boot_id"); err == nil {
		metrics.BootId = strings.TrimSpace(string(bootId))
	}

	// Machine ID (persistent - detects VM replacement)
	if machineId, err := os.ReadFile("/etc/machine-id"); err == nil {
		metrics.MachineId = strings.TrimSpace(string(machineId))
	}

	return metrics
}

func performMemoryTouch() MemoryTouch {
	const allocSize = 16 * 1024 * 1024 // 16 MB
	const pageSize = 4096
	const pagesToTouch = 64

	result := MemoryTouch{
		AllocatedKb:  allocSize / 1024,
		PagesTouched: pagesToTouch,
	}

	// Allocate memory
	buffer := make([]byte, allocSize)

	startTime := time.Now()
	var maxPageTime float64

	hasher := sha256.New()

	for i := 0; i < pagesToTouch; i++ {
		// Deterministic but spread-out page selection
		pageOffset := ((i * 31337) % (allocSize / pageSize)) * pageSize

		pageStart := time.Now()

		// Write pattern to page (forces page fault if not in RAM)
		pattern := byte(i ^ 0xAA)
		for j := 0; j < pageSize; j++ {
			buffer[pageOffset+j] = pattern ^ byte(j&0xFF)
		}

		// Read and hash
		hasher.Write(buffer[pageOffset : pageOffset+pageSize])

		pageTimeMs := msElapsed(pageStart)
		if pageTimeMs > maxPageTime {
			maxPageTime = pageTimeMs
		}
	}

	result.TotalMs = msElapsed(startTime)
	result.MaxPageMs = maxPageTime
	result.ContentHash = hex.EncodeToString(hasher.Sum(nil))

	// Zero and release buffer
	for i := range buffer {
		buffer[i] = 0
	}
	buffer = nil

	return result
}

// Helper functions
func msElapsed(start time.Time) float64 {
	return float64(time.Since(start).Microseconds()) / 1000.0
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}

func padRight(s string, length int) string {
	if len(s) >= length {
		return s
	}
	return s + strings.Repeat(" ", length-len(s))
}
