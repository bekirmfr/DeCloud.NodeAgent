# DeCloud GPU Proxy — Architecture Review & Redesign

**Date:** 2026-02-28, updated 2026-03-01
**Status:** Active — True GPU Presence approach selected for near-term, ioctl proxy remains long-term option
**Context:** Three days of Ollama integration debugging proved API-level shimming works for ~95% of the CUDA surface. The remaining gap (kernel attribute queries) has a surgical fix.

---

## 1. What We Built (Current State)

### Architecture

```
VM (no physical GPU)                      Host (real GPU)
┌──────────────────────┐                  ┌──────────────────────┐
│ Application           │                  │                      │
│   ↓ cuda*()          │                  │ gpu-proxy-daemon     │
│ libcudart.so.12      │ ── TCP/vsock ──→ │   ↓                  │
│ (cuda_shim.c)        │                  │ Real libcudart.so.12 │
│   ↓ cu*()            │                  │ Real libcuda.so.1    │
│ libcuda.so.1         │ ── TCP/vsock ──→ │ Real NVIDIA driver   │
│ (cuda_driver_shim.c) │                  │   ↓                  │
│   ↓ nvml*()          │                  │ Physical GPU         │
│ libnvidia-ml.so.1    │                  │                      │
│ (nvml_shim.c)        │                  └──────────────────────┘
└──────────────────────┘
```

### Components Built (6,500+ lines of C)

| Component | Lines | Status |
|-----------|-------|--------|
| `cuda_shim.c` (Runtime API shim) | ~1,500 | Working — device detection, malloc, memcpy, kernel launch, cuBLAS stubs, graph no-ops |
| `cuda_driver_shim.c` (Driver API shim) | ~1,750 | Working — device enumeration, context, memory ops, graph blocking, cuGetExportTable |
| `nvml_shim.c` (NVML shim) | ~400 | Working — static device info for GPU management queries |
| `gpu_proxy_daemon.c` (Host daemon) | ~1,200 | Working — handles all RPC commands, per-connection isolation, module/function registration |
| `gpu_proxy_proto.h` (Wire protocol) | ~250 | Stable — 25 command types, memory quota, usage stats |
| `transport.{c,h}` (Shared transport) | ~300 | Working — vsock + TCP fallback, token auth, config file |
| `Makefile` + tests | ~200 | Working — Docker compat builds, cross-env deployment |

### What Works Today (Proven on Ollama v0.17 + RTX 4060)

- Device enumeration (count, properties, UUID, name, compute capability)
- Memory allocation/free/memcpy/memset (both Runtime and Driver API)
- Model loading to GPU VRAM (1252 MiB buffer, 17/17 layers offloaded)
- KV cache allocation (128 MiB on CUDA0)
- Kernel launch via Driver API (cuModuleLoadData → cuLaunchKernel)
- Kernel launch via Runtime API (__cudaRegisterFatBinary → cudaLaunchKernel)
- Stream and event management
- Per-connection CUDA context isolation
- Memory quota enforcement and usage tracking
- Token authentication over TCP, CID authentication over vsock
- cuBLAS stub interception (dummy handles, prevents crash)
- CUDA graph no-op passthrough (returns success, no state machine)
- Constructor-based env var injection (bypasses Ollama's env filtering)
- DT_NEEDED library replacement (intercepts co-located dependencies)
- Binary wire protocol with 64 MiB max transfer

### What's Blocked (One Remaining Gap)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `mmq_x_best=0` crash | `cudaFuncGetAttributes` returns fake values; kernel selector can't find valid MMQ variant | True GPU Presence: proxy attribute query to daemon where real GPU kernels are loaded |

---

## 2. Industry Approaches Compared (Updated)

### Approach A: VFIO Passthrough (Hardware-level)

**How it works:** PCIe device is directly assigned to VM via IOMMU. VM gets native driver access.

**Used by:** Kata Containers + NVIDIA GPU Operator, AWS EC2 GPU instances, Azure NC-series

**Compatibility:** 100% — real driver runs inside VM

**Performance:** Near-native (< 3% overhead from IOMMU address translation)

**DeCloud status:** Already implemented in current `DeploymentMode.VirtualMachine`. Works when node has IOMMU.

**Limitation for DeCloud:** 60-80% of target node operators (home users with gaming GPUs) don't have IOMMU enabled or have consumer motherboards without proper IOMMU group isolation. One GPU = one VM, no sharing.

### Approach B: ioctl-Level Proxy (Kernel interface level)

**How it works:** Virtual `/dev/nvidia*` device files intercept `ioctl()` syscalls and forward them to the real driver. Real NVIDIA user-space libraries (libcuda, libcudart, cuBLAS, etc.) run unmodified inside the sandbox.

**Used by:** gVisor nvproxy (Google, production in GKE Sandbox)

**Compatibility:** High (~98%) — real UMD libraries handle all cuBLAS/cuDNN/cuFFT initialization natively.

**Performance:** Near-native — ioctls are small control messages, data transfers go through mapped memory

**DeCloud applicability:** Ultimate solution for 100% compatibility. Solves cuBLAS/cuDNN compute natively. Higher implementation effort.

### Approach C: API-Level Shim with True GPU Presence (What we're building)

**How it works:** Replace NVIDIA shared libraries with shims that serialize API calls over a network. Fat binary registration is proxied to the daemon for real kernel loading. cuBLAS is stubbed out; ggml's native MMQ kernels provide the compute path.

**Used by:** DeCloud (novel approach combining shim + fat binary proxying)

**Compatibility:** High for ggml/Ollama (~95%) — covers all CUDA operations except cuBLAS compute. Moderate for general workloads (~70%) — cuBLAS/cuDNN compute still requires stubs.

**Performance:** Good for inference — RPC overhead amortized over large kernel executions. Memory transfers are the bottleneck, not RPC latency.

**Why this works for DeCloud's target market:** 90%+ of uncensored AI hosting uses Ollama/ggml, which has a native CUDA kernel path (MMQ) that completely bypasses cuBLAS. With real kernel attributes from the daemon, MMQ kernel selection works correctly.

---

## 3. Decision Matrix for DeCloud (Updated)

| Criterion | VFIO Passthrough | ioctl Proxy | API Shim + True GPU Presence |
|-----------|:---:|:---:|:---:|
| Works without IOMMU | ❌ | ✅ | ✅ |
| GPU sharing (multi-VM) | ❌ | ✅ | ✅ |
| cuBLAS/cuDNN compute | ✅ | ✅ | ❌ (stubbed, MMQ used instead) |
| CUDA graphs | ✅ | ✅ | ⚠️ (no-op stubs, disabled via env) |
| Zero app modification | ✅ | ✅ | ✅ |
| ggml/Ollama inference | ✅ | ✅ | ✅ (via MMQ path) |
| Driver version agnostic | ✅ | ❌ (needs ABI per version) | ✅ (fat binary format is stable) |
| Network remoting | N/A | Possible via vsock | ✅ (built and proven) |
| Implementation effort | Done | High (6-10 weeks) | Low (days — 2 new commands) |
| Maintenance burden | None | Medium (per driver) | Low (CUDA ABI is stable) |
| Production-proven | Yes (AWS, Azure) | Yes (GKE Sandbox) | In progress (DeCloud) |

---

## 4. Recommended Architecture: Phased Approach

### Phase 1: True GPU Presence (NOW — days of work)

Complete the API-level shim with real kernel attribute proxying.

**Changes required:**
1. Add `GPU_CMD_FUNC_GET_ATTRIBUTES` (0x54) to protocol
2. Add `GPU_CMD_OCCUPANCY_MAX_BLOCKS` (0x55) to protocol
3. `cudaFuncGetAttributes` triggers eager module upload, then RPCs for real attributes
4. `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` RPCs for real occupancy
5. Daemon handlers call `cuFuncGetAttribute` and `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`

**Expected result:** Ollama inference works end-to-end via ggml's MMQ kernels on proxied GPU.

**What stays:**
- cuBLAS stubs (init succeeds, compute returns NOT_SUPPORTED)
- CUDA graph no-ops (disabled via env, stubs as safety net)
- Constructor setenv (ensures GGML_CUDA_DISABLE_GRAPHS=1)
- DT_NEEDED library replacement in Ollama's cuda_v12 directory

**Covers:** Ollama, llama.cpp, any ggml-based application, any CUDA application that doesn't need cuBLAS/cuDNN compute.

### Phase 2: VFIO Passthrough (Already done)

```
Node has IOMMU + GPU → DeploymentMode.VirtualMachine
                        KVM VM with VFIO GPU passthrough
                        Real NVIDIA driver inside VM
                        100% compatibility, near-native performance
```

No changes needed. This is our best path and should remain the preferred option for nodes with IOMMU.

### Phase 3: ioctl Proxy (Future — if needed)

For workloads requiring real cuBLAS/cuDNN compute (PyTorch training, TensorFlow, etc.), implement ioctl-level proxy based on gVisor nvproxy patterns.

```
Inside VM:
  Application
    ↓ CUDA API calls
  Real NVIDIA UMD libraries (copied from host)
    ↓ ioctl(fd, NV_ESC_RM_CONTROL, &params)
  Virtual /dev/nvidia* (FUSE or kernel module)
    ↓ vsock
  ioctl Proxy Client

Outside VM:
  ioctl Proxy Daemon
    ↓ translated ioctl()
  Real /dev/nvidia*
    ↓
  Real nvidia.ko → Physical GPU
```

**Trigger for Phase 3:** Customer demand for PyTorch/TensorFlow training workloads on non-IOMMU nodes. Current evidence suggests 90%+ of DeCloud's target use case (uncensored AI inference) is covered by Phase 1.

### Phase 4: Container-only (Fallback)

```
Node without KVM → DeploymentMode.Container
                    Docker --gpus (weak isolation, monitoring only)
```

Acceptable for trusted workloads on nodes that can't run VMs.

---

## 5. What Happens to the Current Code

### Phase 1 (True GPU Presence) — Extend, Don't Replace

| Component | Disposition |
|-----------|-------------|
| `gpu_proxy_proto.h` | **Extend** — add 2 new command IDs and structs |
| `cuda_shim.c` | **Extend** — replace fake stubs with RPC calls for attributes/occupancy |
| `cuda_driver_shim.c` | **Keep** — graph blocking, cuGetExportTable handling all working |
| `nvml_shim.c` | **Keep** — GPU management queries working |
| `gpu_proxy_daemon.c` | **Extend** — add 2 new handlers |
| `transport.{c,h}` | **Keep** — proven transport layer |
| cuBLAS stub library | **Keep** — prevents DT_NEEDED crash |
| Makefile + compat build | **Keep** — Docker-based universal builds |

~95% of codebase is reused. Only 2 new handlers + 2 modified functions.

### Phase 3 (ioctl Proxy) — Reuse Foundation

| Component | Disposition |
|-----------|-------------|
| `gpu_proxy_proto.h` | Retire — replaced by ioctl forwarding protocol |
| `cuda_shim.c` | Retire — real libcudart runs in VM |
| `cuda_driver_shim.c` | Retire — real libcuda runs in VM |
| `nvml_shim.c` | Retire — real libnvidia-ml runs in VM |
| `gpu_proxy_daemon.c` | Retire — replaced by ioctl proxy daemon |
| `transport.{c,h}` | **Reuse** — vsock/TCP transport layer still valuable |
| Wire protocol framing | **Reuse** — magic/version/header structure is solid |
| Memory quota logic | **Reuse** — quota enforcement moves to ioctl proxy |
| Token authentication | **Reuse** — auth mechanism unchanged |

---

## 6. Risk Analysis

### Phase 1 (True GPU Presence) Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fat binary upload too large for RPC | Low | Max payload is 64 MiB, typical fatbin is 5-20 MiB. Already proven with deferred upload. |
| Kernel attribute values don't satisfy MMQ selector | Low | Values come from real GPU — same values that work in native CUDA. |
| Additional missing stubs discovered | Low | Pattern established: add stub, rebuild, deploy. Each fix is minutes. |
| ggml changes MMQ kernel selection algorithm | Low | Unlikely to remove `cudaFuncGetAttributes` — it's fundamental CUDA API. |
| Performance overhead from eager module upload | Medium | Upload happens once at init, not per-inference. 5-20 MiB over TCP is <1 second. |

### Phase 3 (ioctl Proxy) Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| NVIDIA driver ABI instability | Medium | Pin to LTS driver branches. gVisor maintains ABI tables. |
| mmap complexity | High | Start with synchronous memcpy. Add shared memory after correctness proven. |
| FD translation bugs | Medium | gVisor source provides reference implementation. |
| Performance overhead from FUSE | Medium | FUSE adds ~10-30μs per ioctl. Acceptable for compute. |

---

## 7. Comparison Summary (Updated)

```
                    API Shim +              ioctl Proxy
                    True GPU Presence       (future option)
                    ──────────────────      ───────────────
Interception:       600+ CUDA functions     ~100-200 ioctl commands
                    (most already done)

Real NVIDIA UMD:    Replaced by stubs       Runs unmodified in VM

cuBLAS compute:     ❌ Stubbed (MMQ used)   ✅ Works natively

CUDA graphs:        ⚠️ No-op stubs         ✅ Works natively

cudaFuncGetAttrs:   ✅ Real (proxied)       ✅ Real (native)

Kernel launch:      ✅ Real (proxied)       ✅ Real (native)

GPU discovery:      ✅ LD_PRELOAD + NVML    ✅ Real /dev/nvidia*
                    + DT_NEEDED replace

Implementation:     ~95% done, days left    6-10 weeks from scratch

Maintenance:        Low (CUDA ABI stable)   Medium (driver ABI per version)

Target workloads:   ggml/Ollama/llama.cpp   Everything (PyTorch, TF, etc.)
                    (~90% of DeCloud demand)

Production proof:   In progress (DeCloud)   gVisor (Google GKE)
```

---

## 8. Recommendation

**Implement True GPU Presence immediately.** It's a surgical extension to the existing codebase (2 new protocol commands, 2 modified shim functions, 2 new daemon handlers). The entire GPU proxy infrastructure is proven — device detection, memory management, model loading, kernel launch RPC all work. The only gap is `cudaFuncGetAttributes` returning fake values instead of querying the daemon for real kernel attributes.

**This is NOT a pivot from the API-level approach.** Day 3 proved that API-level shimming works far better than initially assessed. The DT_NEEDED library replacement technique, constructor-based env injection, and cuBLAS stubbing overcome the `cuGetExportTable` wall for ggml workloads. The remaining fix is an incremental improvement, not a redesign.

**Keep ioctl proxy as Phase 3.** If DeCloud's market expands beyond ggml/Ollama to PyTorch training workloads, the ioctl approach will be needed. But the current evidence strongly suggests that inference-focused AI hosting — DeCloud's core value proposition — is fully served by the API-level shim with True GPU Presence.

**Preserve all current code.** Every component is either directly used in Phase 1 or provides reusable infrastructure for Phase 3. Nothing needs to be retired.
