# DeCloud GPU Proxy — Complete Debugging Journal

**Date Range:** 2026-02-27 through 2026-02-28
**Authors:** BMA + Claude AI assistant
**Status:** Paused — pivoting to generic GPU proxy approach

---

## Executive Summary

Over two days of intensive debugging, we attempted to make the DeCloud GPU proxy system work with Ollama v0.17 as a test application. While we successfully solved many individual problems (GPU detection, symbol versioning, transport connectivity, GLIBC compatibility, CUDA graph blocking), each fix revealed a deeper layer of NVIDIA's internal API complexity. The final blocker — `cuGetExportTable` returning internal function tables that cuBLAS requires — demonstrates that **a shim-only approach cannot generically proxy GPU workloads**. The CUDA API surface is too large, too version-dependent, and too internally coupled for function-level interception to work across arbitrary applications.

**Key conclusion:** The proxy architecture must operate at a lower level (binary RPC for actual GPU operations like malloc/memcpy/kernel-launch) rather than trying to impersonate the entire NVIDIA driver stack.

---

## 1. Architecture Context

### What We Built

```
┌─────────────────────────────────────────┐
│  VM (no physical GPU)                   │
│                                         │
│  Application (e.g., Ollama)             │
│      ↓ cuda*() calls                    │
│  libcudart.so.12  (cudart shim)         │
│      ↓ cu*() calls                      │
│  libcuda.so.1     (driver shim)         │
│      ↓ RPC over TCP/vsock               │
├─────────────────────────────────────────┤
│  Host (real GPU)                        │
│                                         │
│  gpu-proxy-daemon                       │
│      ↓ real CUDA calls                  │
│  Real NVIDIA Driver + GPU               │
└─────────────────────────────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| CUDA Runtime Shim | `shim/cuda_shim.c` | Intercepts `cuda*()` API calls, forwards via RPC |
| CUDA Driver Shim | `shim/cuda_driver_shim.c` | Intercepts `cu*()` API calls, device enumeration |
| NVML Shim | `shim/nvml_shim.c` | Fakes GPU management/monitoring info |
| GPU Proxy Daemon | `daemon/gpu_proxy_daemon.c` | Host-side daemon executing real CUDA operations |
| Transport Layer | `shim/transport.{c,h}` | TCP/vsock RPC communication |
| Protocol | `proto/gpu_proxy_proto.h` | Binary wire protocol definitions |

### Test Environment

- **Host:** MSI laptop, Ubuntu 24.04, NVIDIA RTX 4060 Laptop GPU (8GB, compute 8.9), GLIBC 2.39
- **VM:** ai-chatbot-d2c1 (192.168.122.56), Ubuntu 22.04, GLIBC 2.35, no physical GPU
- **Node Agent:** srv022010 running DeCloud.NodeAgent
- **Test App:** Ollama v0.17 with llama3.2:3b model

---

## 2. Chronological Problem Log

### Day 1 (2026-02-27): Foundation

#### Problem 1: Symbol Versioning Mismatch
- **Symptom:** `initial_count=0` — Ollama bootstrap found zero GPUs despite shim loading
- **Root Cause:** cudart shim built without `--version-script`, so symbols lacked `@@libcudart.so.12` version tags. libggml-cuda.so requires versioned symbols.
- **Fix:** Added `libcudart.version` linker script with version tags
- **Lesson:** ELF symbol versioning is mandatory for drop-in library replacement

#### Problem 2: Stack Buffer Overflow in Device Properties
- **Symptom:** Stack corruption / crash during `cudaGetDeviceProperties`
- **Root Cause:** ggml allocates only 1096 bytes for `cudaDeviceProp` on the stack. Our shim memset the full struct (848+ bytes) which overflowed
- **Fix:** Added `SAFE_PROP_MEMSET_SIZE` (768 bytes), fill only safe region
- **Lesson:** ggml uses hardcoded byte offsets (major@360, minor@364, etc.), not the full struct

#### Problem 3: Bootstrap Timeout
- **Symptom:** Ollama killed bootstrap subprocess before GPU detection completed
- **Root Cause:** RPC connection during `dlopen` added latency exceeding Ollama's 30-second bootstrap timeout
- **Fix:** Lazy initialization — zero RPC during library load, connect on first actual call
- **Lesson:** Shim initialization must be zero-cost at load time

### Day 2 Morning (2026-02-28): Deployment & Connectivity

#### Problem 4: Library Poisoning on Host
- **Symptom:** GPU proxy daemon crash loop on srv022010
- **Root Cause:** Shim `.so` files installed to host system paths (`/usr/local/lib/`) instead of VM-only paths, poisoning the daemon's own CUDA library resolution
- **Fix:** Isolated shims to VM-only deployment via 9p filesystem share at `/usr/local/lib/decloud-gpu-shim/`
- **Lesson:** Shims must NEVER be installed on the host — they're VM-only components

#### Problem 5: LD_PRELOAD Not Reaching Ollama
- **Symptom:** GPU detected manually but not through systemd service
- **Root Cause:** `LD_PRELOAD` environment variable not configured in systemd service override
- **Fix:** Added `gpu-proxy.conf` systemd drop-in with all required environment variables
- **Lesson:** Ollama's bootstrap subprocess strips non-OLLAMA prefixed env vars; shim must read config from `/etc/decloud/gpu-proxy.env` as fallback

#### Problem 6: TCP Firewall Blocking Proxy
- **Symptom:** Shim connected but received no data from daemon
- **Root Cause:** Missing `iptables` rule for port 9999 between VM network (192.168.122.0/24) and host
- **Fix:** Added `iptables -I INPUT -s 192.168.122.0/24 -p tcp --dport 9999 -j ACCEPT`
- **Lesson:** KVM bridge networking requires explicit firewall rules for VM↔host communication

#### Problem 7: Config File Permissions
- **Symptom:** Shim couldn't read proxy config as ollama user
- **Root Cause:** `/etc/decloud/gpu-proxy.env` had restrictive permissions (600)
- **Fix:** Changed to 644 (world-readable, since it contains only connection info, not secrets)
- **Lesson:** Config files read by service users need appropriate permissions

### Day 2 Afternoon: The CUDA Graph Saga

#### Problem 8: SIGSEGV at addr=0x10 During Compute Warmup
- **Symptom:** GPU detected, model loaded, KV cache allocated, then SIGSEGV during graph reservation
- **Root Cause (initial theory):** CUDA graph API stubs returning `cudaSuccess` with NULL handles
- **Partial Fix:** Changed cudart graph stubs to return error code 801 (`cudaErrorNotSupported`)

#### Problem 9: Driver API Graph Functions Bypassing cudart Stubs
- **Symptom:** SIGSEGV persisted even with cudart graph stubs fixed
- **Root Cause:** ggml-cuda calls driver API (`cu*`) graph functions through `cuGetProcAddress`, bypassing our cudart stubs entirely. The generic stub in `cuGetProcAddress` returned `CUDA_SUCCESS` for all unknown functions.
- **Fix:** Added `graph_not_supported_stub()` returning 801, intercepted `cuGraph*`, `cuStreamBeginCapture`, `cuStreamEndCapture` in `cuGetProcAddress`
- **Lesson:** CUDA has TWO parallel API layers (Runtime and Driver). Both must be handled.

#### Problem 10: GLIBC 2.39 vs 2.35 Incompatibility
- **Symptom:** `LD_PRELOAD` caused immediate crash: `GLIBC_2.38 not found`
- **Root Cause:** Shim compiled on MSI (Ubuntu 24.04, GLIBC 2.39) used `__isoc23_sscanf` (C23 function). VM runs Ubuntu 22.04 (GLIBC 2.35). LD_PRELOAD forces eager symbol resolution, exposing the mismatch.
- **Fix:** Installed gcc on VM, rebuilt shims natively with GLIBC 2.35
- **Lesson:** Cross-compilation between GLIBC versions requires explicit targeting. Shims should be compiled in the deployment environment or use static linking.

#### Problem 11: Capture-Query Functions Need Specific Stubs
- **Symptom:** After blocking graph creation, `cublasCreate_v2` failed with "library was not initialized"
- **Root Cause:** Blocking `cuStreamIsCapturing` and `cuStreamGetCaptureInfo` with error codes broke cuBLAS initialization, which legitimately queries capture state during setup
- **Fix:** Added specific stubs returning "not capturing" status instead of errors:
  - `cuStreamIsCapturing` → returns `CUDA_SUCCESS` with status=0
  - `cuStreamGetCaptureInfo` / `_v2` → returns `CUDA_SUCCESS` with status=0, id=0, NULL graph
  - `cuStreamUpdateCaptureDependencies` → returns `CUDA_SUCCESS` (no-op)
  - `cuThreadExchangeStreamCaptureMode` → returns `CUDA_SUCCESS` with mode=0
- **Lesson:** Must distinguish between "this operation is not supported" vs "this operation succeeds trivially"

#### Problem 12: cuGraphics* vs cuGraph* Name Collision
- **Symptom:** Graph blocking `strncmp(symbol, "cuGraph", 7)` also blocked OpenGL/EGL interop functions (`cuGraphicsMapResources`, `cuGraphicsSubResourceGetMappedArray`, etc.)
- **Fix:** Added exclusion: `strncmp(symbol, "cuGraphics", 10) != 0` before blocking
- **Lesson:** CUDA namespace has overlapping prefixes — filtering must be precise

### Day 2 Evening: The Final Wall

#### Problem 13: cuGetExportTable — The Unproxiable API ⭐
- **Symptom:** After fixing all graph and capture stubs, SIGSEGV persisted at addr=0x10
- **Root Cause:** `cuGetExportTable` hit the generic stub, returning `CUDA_SUCCESS` **without setting the output pointer**. cuBLAS calls this to get internal NVIDIA function tables (private driver APIs). With success but NULL pointer, cuBLAS dereferenced `NULL+0x10` → crash.
- **Fix attempted:** Return `CUDA_ERROR_NOT_FOUND` from `cuGetExportTable` to force cuBLAS fallback paths
- **Result:** cuBLAS then failed with "library was not initialized" — it **requires** the internal function tables and has no fallback
- **Fix attempted:** `GGML_CUDA_FORCE_MMQ=1` to bypass cuBLAS entirely
- **Result:** Environment variable not propagated to runner subprocess; ggml reported `FORCE_MMQ: no`

**This is the fundamental blocker.** cuBLAS requires internal driver function tables that are:
- Not part of any public API
- Contain host-local function pointers (can't be serialized/proxied)
- Version-specific and undocumented
- Essential for cuBLAS initialization (no graceful fallback exists)

---

## 3. Critical Finding: Manual vs Systemd Behavioral Divergence

This was a major time-sink and a key lesson for generic proxy design.

### The Observation

| Execution Method | GPU Detected? | Model Loaded to GPU? | Inference? |
|-----------------|--------------|---------------------|-----------|
| Manual: `sudo -u ollama` with env vars directly | ✅ Yes (CUDA0, 7096 MiB) | ✅ Yes (1918 MiB buffer) | ❌ SIGSEGV at compute |
| systemd service (no LD_PRELOAD) | ❌ No (`library=cpu`, `initial_count=0`) | N/A | CPU only |
| systemd + LD_PRELOAD (MSI-compiled shim) | ❌ Crash at startup | N/A | GLIBC_2.38 not found |
| systemd + LD_PRELOAD (VM-compiled shim) | ✅ Yes (found 1 CUDA device) | ✅ Yes | ❌ SIGSEGV at compute |

### Root Cause Chain

The strace analysis (PID 7911, bootstrap subprocess) proved:

1. **Bootstrap subprocess loads our cudart shim** via dlopen from `cuda_v12/` → succeeds (fd=9)
2. **Bootstrap NEVER calls `cudaGetDeviceCount`** — zero `connect()` syscalls, zero `openat` for `gpu-proxy.env`
3. **Bootstrap only checks library loadability** — it verifies the `.so` files are valid ELF, symbols resolve, then exits
4. **Main Go process does the actual GPU enumeration** — but it needs cudart/NVML symbols globally available
5. **Without LD_PRELOAD, main process has no access to our shim symbols** → reports 0 GPUs
6. **With LD_PRELOAD, shim symbols are globally available** → GPU detected, model loaded

### Why Manual Worked Differently

When running `sudo -u ollama OLLAMA_DEBUG=1 OLLAMA_LLM_LIBRARY=cuda_v12 ... ollama serve` with `DECLOUD_GPU_PROXY_*` env vars explicitly set, the manual invocation passed all environment variables directly to the process. The `LD_PRELOAD` was also set in the gpu-proxy.conf at that point, making cudart symbols available globally.

Under systemd, the environment propagation is more restrictive:
- Ollama strips non-`OLLAMA_` prefixed vars when spawning runner subprocesses
- The shim falls back to reading `/etc/decloud/gpu-proxy.env` for connection details
- But the bootstrap never invokes any CUDA functions, so the fallback config is never read
- The main Go binary never loads `libnvidia-ml.so.1` either (confirmed via strace)

### Implication for Generic Proxy Design

**Any application that uses a multi-process architecture with library compatibility checks will exhibit this same problem.** The shim approach requires `LD_PRELOAD` to work, which means:

1. The shim must be compiled for the target VM's GLIBC version (not the build host)
2. `LD_PRELOAD` forces eager symbol resolution — every symbol in the shim must resolve at load time
3. The shim must be a single file deployed consistently (same soname matching) to avoid dual-instance conflicts
4. Applications that gate GPU discovery on device nodes (e.g., `/dev/nvidiactl`) need those nodes present even without a real driver

This is fundamentally fragile for a generic proxy — every new application may have different discovery mechanisms, subprocess architectures, and library loading patterns.

---

## 4. Fundamental Technical Findings

### Why Function-Level Shimming Cannot Work Generically

1. **Internal APIs:** NVIDIA libraries (cuBLAS, cuDNN, cuFFT) use `cuGetExportTable` to access private driver functions. These are undocumented, version-specific, and contain host-side function pointers that cannot be proxied over a network.

2. **API Surface Explosion:** The CUDA Driver API has 600+ functions. Each NVIDIA library (cuBLAS, cuDNN, cuFFT, cuSPARSE, etc.) calls a different subset. Shimming every function for every version is unsustainable.

3. **Two-Layer API:** CUDA has both Runtime API (`cuda*`) and Driver API (`cu*`). Libraries freely mix both layers. cuBLAS primarily uses the Driver API internally, bypassing any Runtime API shims.

4. **Version Coupling:** `cuGetProcAddress` requests functions by name AND version number. Internal behaviors change between CUDA toolkit versions. A shim that works with CUDA 12.4 may break with 12.5.

5. **State Locality:** Many CUDA operations (graph capture, context state, stream ordering) are inherently host-local. They depend on the state of the actual GPU driver, not just RPC call/response semantics.

### What DOES Work Through the Proxy

The following operations successfully proxy over RPC:
- Device enumeration (`cudaGetDeviceCount`, `cudaGetDeviceProperties`)
- Memory allocation (`cudaMalloc`, `cudaFree`)
- Memory transfers (`cudaMemcpy` H2D/D2H/D2D)
- Stream creation and synchronization
- Event creation and timing
- Kernel launches (with serialized arguments)
- Device properties and capability queries

### What CANNOT Work Through Function-Level Shimming

- `cuGetExportTable` (internal function tables)
- cuBLAS/cuDNN/cuFFT initialization (depends on internal tables)
- CUDA graph capture and replay (host-local state machine)
- Peer-to-peer memory operations
- Unified Memory coherence
- Driver-level context manipulation

---

## 5. Path Forward: Generic GPU Proxy Architecture

### Option A: Binary-Level GPU Proxy (Recommended)

Instead of impersonating the NVIDIA driver, **run the NVIDIA libraries on the host and proxy at the application boundary:**

```
┌──────────────────────────────────┐
│  VM                              │
│  Application                     │
│      ↓                           │
│  Thin RPC Client Library         │
│  (NOT a CUDA replacement)        │
│      ↓ binary RPC                │
├──────────────────────────────────┤
│  Host                            │
│  GPU Worker Process              │
│      ↓                           │
│  Real CUDA + cuBLAS + cuDNN      │
│      ↓                           │
│  Real NVIDIA Driver + GPU        │
└──────────────────────────────────┘
```

The host-side worker loads the real NVIDIA stack and receives high-level operations (allocate, transfer, launch kernel) rather than trying to impersonate every CUDA function.

**Reference implementations:** rCUDA, GVirtuS, NVIDIA's own GPU sharing (MIG/MPS)

### Option B: VFIO GPU Passthrough

Pass the physical GPU directly to the VM using VFIO/IOMMU. No proxy needed — the VM gets native GPU access. Limitations: one GPU per VM, requires IOMMU hardware support.

### Option C: NVIDIA vGPU / MIG

Use NVIDIA's native virtualization (vGPU for consumer cards via driver hacks, MIG for datacenter GPUs). Limited hardware support but zero proxy overhead.

### Recommended Hybrid Approach

1. **VFIO passthrough** when hardware supports it (dedicated GPU per VM)
2. **Host-side worker process** for GPU sharing (multiple VMs, one GPU)
3. **Container-native GPU** when VM isolation isn't required

The current shim infrastructure is valuable for device enumeration and basic operations. The daemon's RPC protocol is solid. What needs to change is: **stop trying to replace libcuda.so.1 entirely** and instead provide a purpose-built GPU compute API that applications opt into.

---

## 6. Files Modified (Uncommitted)

### On MSI Host (`/opt/decloud/DeCloud.NodeAgent/src/gpu-proxy/`)

#### `shim/cuda_driver_shim.c` — Major changes:
1. **Added `graph_not_supported_stub()`** (line ~91): Returns 801 for graph operations
2. **Added graph interception in `cuGetProcAddress`** (line ~1707): Blocks `cuGraph*` (excluding `cuGraphics*`), `cuStreamBeginCapture`, `cuStreamEndCapture`
3. **Added capture-query stubs:**
   - `cuStreamIsCapturing()` — returns success with status=0 (not capturing)
   - `cuStreamGetCaptureInfo()` / `_v2()` — returns success with null graph info
   - `cuStreamUpdateCaptureDependencies()` — no-op success
   - `cuThreadExchangeStreamCaptureMode()` — no-op success
4. **Added `cuGetExportTable()`** (line ~652): Returns `CUDA_ERROR_NOT_FOUND` with NULL pointer (prevents crash but breaks cuBLAS)
5. **Added `cuGetExportTable` to dispatch table** (line ~1716)

#### `shim/cuda_shim.c` — Minor changes:
1. Graph-related stubs changed from `return cudaSuccess` to `return 801` (cudaErrorNotSupported):
   - `cudaGraphInstantiate`, `cudaGraphInstantiateWithFlags`, `cudaGraphInstantiateWithParams`
   - `cudaGraphLaunch`, `cudaGraphExecDestroy`, `cudaGraphDestroy`
   - `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaStreamIsCapturing`

### On VM (`ai-chatbot-d2c1`, 192.168.122.56)

#### `/etc/systemd/system/ollama.service.d/gpu-proxy.conf`:
```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libdecloud_cuda_shim.so"
Environment="GGML_CUDA_DISABLE_GRAPHS=1"
Environment="GGML_CUDA_FORCE_MMQ=1"
Environment="CUDA_LAUNCH_BLOCKING=1"
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"
Environment="OLLAMA_FLASH_ATTENTION=0"
Environment="DECLOUD_GPU_PROXY_TRANSPORT=tcp"
Environment="DECLOUD_GPU_PROXY_PORT=9999"
Environment="DECLOUD_GPU_PROXY_HOST=192.168.122.1"
Environment="DECLOUD_GPU_PROXY_TOKEN=<token>"
```

#### VM-compiled binaries (native GLIBC 2.35):
- `/usr/local/lib/libcuda.so.1` — driver shim (VM-compiled)
- `/usr/local/lib/libdecloud_cuda_shim.so` — cudart shim (VM-compiled)
- `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90` — cudart shim copy
- `/tmp/build/shim/` — build directory with source copies

---

## 7. Verified Working Capabilities

| Capability | Status | Notes |
|------------|--------|-------|
| GPU detection through proxy | ✅ Working | RTX 4060, 7096 MiB free |
| Device properties via RPC | ✅ Working | Name, compute capability, memory |
| Model loading to GPU VRAM | ✅ Working | 1918.35 MiB buffer allocated |
| KV cache allocation | ✅ Working | 448 MiB on CUDA0 |
| CUDA graph blocking | ✅ Working | Returns errors, ggml attempts fallback |
| cuBLAS initialization | ❌ Blocked | Requires cuGetExportTable internal tables |
| GPU inference end-to-end | ❌ Blocked | cuBLAS failure prevents compute |
| CPU inference with GPU detection | ⚠️ Partial | Falls back to CPU when GPU compute fails |

---

## 8. Key Technical Reference

### CUDA API Layering
```
Application
    ↓
cudart (Runtime API: cuda*)  ←  Our cuda_shim.c
    ↓ internally uses
cuBLAS / cuDNN / cuFFT       ←  Real NVIDIA libraries (can't be shimmed)
    ↓ internally uses
CUDA Driver API (cu*)        ←  Our cuda_driver_shim.c
    ↓ internally uses
cuGetExportTable             ←  PRIVATE internal API (our wall)
    ↓
NVIDIA kernel driver (/dev/nvidia*)
```

### Wire Protocol
- Magic: `0x44435544` ("DUCD")
- Version: 2
- Transport: TCP (port 9999) or vsock (CID=2, port=9999)
- Auth: Token-based (SHA-256)

### Build Commands (VM-native)
```bash
# cudart shim
gcc -shared -fPIC -I. -O2 -o libcudart_shim.so cuda_shim.c -ldl -lpthread \
    -Wl,-soname,libcudart.so.12 -Wl,--version-script=libcudart.version

# driver shim
gcc -shared -fPIC -I.. -O2 -o libcuda_shim.so cuda_driver_shim.c -lpthread -ldl
```

---

## 9. Recommendations

1. **Stop shimming the entire NVIDIA stack.** The internal API surface is unbounded and undocumented.
2. **Evaluate rCUDA/GVirtuS** as proven GPU virtualization solutions that handle the complexity.
3. **Consider host-side worker pattern** where the real NVIDIA libraries run on the host and VMs communicate via a high-level compute API.
4. **Preserve current shim work** for device enumeration, basic memory ops, and kernel launches — these work well and are genuinely useful.
5. **Target inference frameworks directly** (e.g., vLLM, TGI) if the primary use case is AI hosting — these can use custom backends that natively support remote GPU.
