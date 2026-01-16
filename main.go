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
// Security model:
// - Generates FRESH Ed25519 keypair for EACH challenge
// - Private key exists in memory for ~20ms
// - Private key is ZEROED immediately after signing
// - Response time is measured by Orchestrator (wall-clock)
//
// Why node can't cheat:
// - To forge a response, node would need the ephemeral private key
// - Extracting key from VM memory takes >100ms (pause + dump + search)
// - Orchestrator rejects responses that take >100ms
// - By the time node could extract the key, it's already zeroed
//
// ============================================================

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
	Nonce           string        `json:"nonce"`
	EphemeralPubKey string        `json:"ephemeralPubKey"`
	Metrics         Metrics       `json:"metrics"`
	MemoryTouch     MemoryTouch   `json:"memoryTouch"`
	Timing          TimingInfo    `json:"timing"`
	Signature       string        `json:"signature"`
}

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

type MemoryTouch struct {
	AllocatedKb  int     `json:"allocatedKb"`
	PagesTouched int     `json:"pagesTouched"`
	TotalMs      float64 `json:"totalMs"`
	MaxPageMs    float64 `json:"maxPageMs"`
	ContentHash  string  `json:"contentHash"`
}

type TimingInfo struct {
	KeyGenMs      float64 `json:"keyGenMs"`
	MetricsMs     float64 `json:"metricsMs"`
	MemoryTouchMs float64 `json:"memoryTouchMs"`
	SigningMs     float64 `json:"signingMs"`
	TotalMs       float64 `json:"totalMs"`
}

var vmId string

func main() {
	// Read VM ID from cloud-init injected file
	vmIdBytes, err := os.ReadFile("/etc/decloud/vm-id")
	if err != nil {
		log.Printf("Warning: Could not read VM ID from /etc/decloud/vm-id: %v", err)
		vmIdBytes, _ = os.ReadFile("/etc/machine-id")
	}
	vmId = strings.TrimSpace(string(vmIdBytes))

	log.Printf("DeCloud Ephemeral Attestation Agent starting")
	log.Printf("VM ID: %s", vmId)

	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/challenge", challengeHandler)

	log.Printf("Listening on :9999")
	log.Fatal(http.ListenAndServe(":9999", nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"status":"healthy","vmId":"%s","agent":"ephemeral-attestation","version":"1.0.0"}`, vmId)
}

func challengeHandler(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse challenge
	var challenge Challenge
	if err := json.NewDecoder(r.Body).Decode(&challenge); err != nil {
		http.Error(w, "Invalid challenge JSON", http.StatusBadRequest)
		return
	}

	// Verify this challenge is for us (optional but good practice)
	if challenge.VmId != "" && challenge.VmId != vmId {
		log.Printf("Warning: Challenge VM ID mismatch (got %s, expected %s)", challenge.VmId, vmId)
		// Continue anyway - Orchestrator might have different ID
	}

	// =========================================
	// STEP 1: Generate EPHEMERAL Ed25519 keypair
	// This is the KEY security feature - fresh key every time
	// =========================================
	keyGenStart := time.Now()

	pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		http.Error(w, "Key generation failed", http.StatusInternalServerError)
		return
	}

	keyGenMs := float64(time.Since(keyGenStart).Microseconds()) / 1000.0

	// =========================================
	// STEP 2: Collect system metrics from /proc
	// =========================================
	metricsStart := time.Now()
	metrics := collectMetrics()
	metricsMs := float64(time.Since(metricsStart).Microseconds()) / 1000.0

	// =========================================
	// STEP 3: Perform memory touch test
	// Proves we have real RAM (not swap)
	// =========================================
	memTouchStart := time.Now()
	memTouch := performMemoryTouch()
	memTouchMs := float64(time.Since(memTouchStart).Microseconds()) / 1000.0

	// =========================================
	// STEP 4: Build canonical message and sign
	// =========================================
	signingStart := time.Now()

	// Canonical message format (MUST match Orchestrator exactly)
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

	signingMs := float64(time.Since(signingStart).Microseconds()) / 1000.0

	// =========================================
	// STEP 5: Build response
	// =========================================
	totalMs := float64(time.Since(startTime).Microseconds()) / 1000.0

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
	// This is the security guarantee
	// =========================================
	zeroPrivateKey(privKey)

	// Force garbage collection to clear any copies
	runtime.GC()

	// Send response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	log.Printf("Challenge responded in %.2fms (keygen: %.2fms, metrics: %.2fms, memtouch: %.2fms, sign: %.2fms)",
		totalMs, keyGenMs, metricsMs, memTouchMs, signingMs)
}

// zeroPrivateKey securely zeros the private key memory
// This is CRITICAL for security - prevents key extraction
func zeroPrivateKey(key ed25519.PrivateKey) {
	for i := range key {
		key[i] = 0
	}
	// Attempt to prevent compiler optimization from removing the zeroing
	runtime.KeepAlive(key)
}

func collectMetrics() Metrics {
	metrics := Metrics{}

	// CPU cores from /proc/cpuinfo
	cpuinfo, _ := os.ReadFile("/proc/cpuinfo")
	metrics.CpuCores = strings.Count(string(cpuinfo), "processor\t:")
	if metrics.CpuCores == 0 {
		metrics.CpuCores = strings.Count(string(cpuinfo), "processor:")
	}
	if metrics.CpuCores == 0 {
		metrics.CpuCores = 1 // Fallback
	}

	// Memory from /proc/meminfo
	meminfo, _ := os.ReadFile("/proc/meminfo")
	for _, line := range strings.Split(string(meminfo), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			fmt.Sscanf(line, "MemTotal: %d kB", &metrics.MemoryKb)
		} else if strings.HasPrefix(line, "MemFree:") {
			fmt.Sscanf(line, "MemFree: %d kB", &metrics.MemoryFreeKb)
		}
	}

	// Load average from /proc/loadavg
	loadavg, _ := os.ReadFile("/proc/loadavg")
	fmt.Sscanf(string(loadavg), "%f %f %f",
		&metrics.LoadAvg1, &metrics.LoadAvg5, &metrics.LoadAvg15)

	// Uptime from /proc/uptime
	uptime, _ := os.ReadFile("/proc/uptime")
	fmt.Sscanf(string(uptime), "%f", &metrics.UptimeSeconds)

	// Boot ID (changes each boot)
	bootId, _ := os.ReadFile("/proc/sys/kernel/random/boot_id")
	metrics.BootId = strings.TrimSpace(string(bootId))

	// Machine ID (persistent)
	machineId, _ := os.ReadFile("/etc/machine-id")
	metrics.MachineId = strings.TrimSpace(string(machineId))

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

	// Touch random pages and measure timing
	startTime := time.Now()
	var maxPageTime float64

	hasher := sha256.New()

	for i := 0; i < pagesToTouch; i++ {
		// Random page offset (deterministic based on iteration for reproducibility)
		pageOffset := ((i * 31337) % (allocSize / pageSize)) * pageSize

		pageStart := time.Now()

		// Write pattern to page
		pattern := byte(i ^ 0xAA)
		for j := 0; j < pageSize; j++ {
			buffer[pageOffset+j] = pattern ^ byte(j&0xFF)
		}

		// Read and hash the page
		hasher.Write(buffer[pageOffset : pageOffset+pageSize])

		pageTimeMs := float64(time.Since(pageStart).Microseconds()) / 1000.0
		if pageTimeMs > maxPageTime {
			maxPageTime = pageTimeMs
		}
	}

	result.TotalMs = float64(time.Since(startTime).Microseconds()) / 1000.0
	result.MaxPageMs = maxPageTime
	result.ContentHash = hex.EncodeToString(hasher.Sum(nil))

	// Explicitly zero and release buffer to free memory
	for i := range buffer {
		buffer[i] = 0
	}
	buffer = nil

	return result
}
