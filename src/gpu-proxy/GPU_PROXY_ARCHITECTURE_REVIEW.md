# DeCloud GPU Proxy — Architecture Review & Redesign

**Date:** 2026-02-28
**Status:** Active review — decision required before further implementation
**Context:** Two days of Ollama integration debugging exposed fundamental limitations in the current API-level shimming approach

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

### Components Built (5,200+ lines of C)

| Component | Lines | Status |
|-----------|-------|--------|
| `cuda_shim.c` (Runtime API shim) | ~1,340 | Working — device detection, malloc, memcpy, kernel launch |
| `cuda_driver_shim.c` (Driver API shim) | ~1,750 | Partial — device enumeration, context, memory ops. Graph blocking added. |
| `nvml_shim.c` (NVML shim) | ~400 | Working — static device info for GPU management queries |
| `gpu_proxy_daemon.c` (Host daemon) | ~1,200 | Working — handles all RPC commands, per-connection isolation |
| `gpu_proxy_proto.h` (Wire protocol) | ~250 | Stable — 25 command types, memory quota, usage stats |
| `transport.{c,h}` (Shared transport) | ~300 | Working — vsock + TCP fallback, token auth, config file |
| `Makefile` + tests | ~200 | Working — cross-compilation, compat builds |

### What Works Today

- Device enumeration (count, properties, UUID, name)
- Memory allocation/free/memcpy/memset (both Runtime and Driver API)
- Kernel launch via Driver API (cuModuleLoadData → cuLaunchKernel)
- Stream and event management
- Per-connection CUDA context isolation
- Memory quota enforcement and usage tracking
- Token authentication over TCP, CID authentication over vsock
- Binary wire protocol with 64MB max transfer

### What Broke (And Why It Matters)

| Blocker | Root Cause | Generic? |
|---------|-----------|----------|
| cuGetExportTable → SIGSEGV | cuBLAS requires internal NVIDIA function tables containing host-local pointers | Yes — any app using cuBLAS, cuDNN, cuFFT |
| CUDA graphs → SIGSEGV | ggml-cuda uses driver-API graph capture; stubs returning success → NULL handle deref | Yes — any app using CUDA graphs |
| Systemd vs manual execution | Bootstrap subprocess only checks library loadability, never calls cudaGetDeviceCount; LD_PRELOAD required for main process | Yes — any multi-process app with discovery phase |
| GLIBC version mismatch | Shim compiled on host (2.39) fails LD_PRELOAD on VM (2.35) due to eager resolution | Yes — any cross-environment deployment |
| cuGetProcAddress API surface | 600+ driver functions queried by name+version; generic stubs silently corrupt state | Yes — grows with every CUDA release |

**The fundamental problem:** NVIDIA's CUDA stack is not designed to be split across a network boundary at the API level. Internal libraries (cuBLAS, cuDNN, cuFFT) use private, undocumented driver interfaces (`cuGetExportTable`) that contain host-local function pointers. These cannot be serialized, proxied, or faked.

---

## 2. Industry Approaches Compared

There are exactly three proven approaches to GPU virtualization. Every production system uses one or a combination of these.

### Approach A: VFIO Passthrough (Hardware-level)

**How it works:** PCIe device is directly assigned to VM via IOMMU. VM gets native driver access.

**Used by:** Kata Containers + NVIDIA GPU Operator, AWS EC2 GPU instances, Azure NC-series

**Requirements:** IOMMU (VT-d/AMD-Vi), one physical GPU per VM, VFIO kernel driver

**Compatibility:** 100% — real driver runs inside VM, zero translation needed

**Performance:** Near-native (< 3% overhead from IOMMU address translation)

**DeCloud status:** Already implemented in current `DeploymentMode.VirtualMachine`. Works when node has IOMMU.

**Limitation for DeCloud:** 60-80% of target node operators (home users with gaming GPUs) don't have IOMMU enabled or have consumer motherboards without proper IOMMU group isolation. One GPU = one VM, no sharing.

### Approach B: ioctl-Level Proxy (Kernel interface level)

**How it works:** Virtual `/dev/nvidia*` device files intercept `ioctl()` syscalls and forward them to the real driver. Real NVIDIA user-space libraries (libcuda, libcudart, cuBLAS, etc.) run unmodified inside the sandbox.

**Used by:** gVisor nvproxy (Google, production in GKE Sandbox)

**Requirements:** Deep understanding of NVIDIA kernel driver ABI per driver version. FD and pointer translation for ioctl struct fields.

**Compatibility:** High (~98%) — real UMD libraries handle all cuBLAS/cuDNN/cuFFT initialization natively. Only limited by which ioctls are implemented.

**Performance:** Near-native — ioctls are small control messages, data transfers go through mapped memory

**Key insight from gVisor:** The ioctl interface is *simpler* than the CUDA API surface. There are ~100-200 distinct ioctl commands vs 600+ CUDA API functions. Most ioctls are "simple" (no pointers/FDs) and can be proxied as opaque byte blobs. Only ioctls containing pointers or FDs need manual translation.

**Key limitation:** NVIDIA's kernel ABI is unstable — struct layouts change between driver versions. nvproxy must be updated for each supported driver version. Google maintains a version table in `SupportedIoctls()`.

**DeCloud applicability:** This is the approach that solves our cuBLAS/cuDNN problem. The real NVIDIA libraries run inside the VM, so `cuGetExportTable` works natively — it returns real function tables from the real driver running on the host, proxied through the ioctl interface.

### Approach C: API-Level Remoting (What we built)

**How it works:** Replace NVIDIA shared libraries with shims that serialize API calls over a network to a host-side daemon.

**Used by:** rCUDA (academic), GVirtuS (academic), our current implementation

**Requirements:** Shim libraries matching every API version, handling of all function signatures

**Compatibility:** Low (~60-70% for basic CUDA, fails for cuBLAS/cuDNN) — internal NVIDIA APIs cannot be proxied

**Performance:** Variable — small operations have RPC overhead, large transfers are network-bound

**Why it fails for general workloads (proven by our debugging):**
1. `cuGetExportTable` returns host-local function pointers (cannot serialize)
2. cuBLAS/cuDNN/cuFFT initialization depends on these internal tables (no fallback)
3. API surface is 600+ functions, growing every CUDA release
4. Two parallel API layers (Runtime + Driver) that libraries freely mix
5. Applications discover GPUs through multiple mechanisms (NVML, cudart, driver API) — all must be shimmed consistently

**Note on rCUDA:** Academic project, last updated ~2020, never achieved production adoption. Works for simple CUDA programs but breaks on real-world ML frameworks for the same reasons we hit.

---

## 3. Decision Matrix for DeCloud

| Criterion | VFIO Passthrough | ioctl Proxy | API Shim (current) |
|-----------|:---:|:---:|:---:|
| Works without IOMMU | ❌ | ✅ | ✅ |
| GPU sharing (multi-VM) | ❌ | ✅ | ✅ |
| cuBLAS/cuDNN/cuFFT | ✅ | ✅ | ❌ |
| CUDA graphs | ✅ | ✅ | ❌ |
| Zero app modification | ✅ | ✅ | ✅ |
| Driver version agnostic | ✅ | ❌ (needs ABI per version) | ❌ (needs API per version) |
| Network remoting (VM ↔ host) | N/A | Possible via vsock | ✅ (built) |
| Implementation effort | Done | High | High (and incomplete) |
| Maintenance burden | None | Medium (per driver version) | Very high (per CUDA release) |
| Production-proven | Yes (AWS, Azure, GCE) | Yes (GKE Sandbox) | No |

---

## 4. Recommended Architecture: Hybrid Approach

### Strategy

Use VFIO passthrough as the primary path where hardware supports it (already done), and an ioctl-level proxy as the fallback for non-IOMMU nodes. Retire API-level shimming for general workloads.

### Tier 1: VFIO Passthrough (already implemented)

```
Node has IOMMU + GPU → DeploymentMode.VirtualMachine
                        KVM VM with VFIO GPU passthrough
                        Real NVIDIA driver inside VM
                        100% compatibility, near-native performance
```

No changes needed. This is our best path and should remain the preferred option.

### Tier 2: VM-in-Docker with ioctl Proxy (new — recommended for non-IOMMU)

```
Host (no IOMMU)
└─ Docker Container (--gpus all --device /dev/kvm)
     ├─ Real NVIDIA driver + /dev/nvidia* devices
     ├─ QEMU/KVM VM (tenant sandbox)
     │    ├─ Real NVIDIA UMD libraries (libcuda, libcudart, cuBLAS...)
     │    ├─ Virtual /dev/nvidia* → ioctl proxy over virtio-vsock
     │    └─ Tenant workload runs here (full compatibility)
     └─ ioctl Proxy Daemon
          ├─ Receives ioctl() calls from VM via vsock
          ├─ Translates FDs and pointers
          └─ Forwards to real /dev/nvidia* on host
```

**Why this works for general workloads:**

1. **cuBLAS/cuDNN/cuFFT:** Run unmodified inside the VM. When they call `cuGetExportTable`, it goes through the real `libcuda.so.1` → ioctl → proxy → host driver → real function tables returned. The function pointers are valid because they point into the VM's own copy of the real NVIDIA UMD.

2. **CUDA graphs:** Work natively — the real CUDA runtime manages graph state locally inside the VM. Graph capture/replay are local operations that don't cross the proxy boundary.

3. **No LD_PRELOAD needed:** The proxy operates at the `/dev/nvidia*` device file level, not at the shared library level. Applications load real NVIDIA libraries normally. No symbol versioning, no GLIBC compatibility issues.

4. **No API surface problem:** The ioctl interface is the narrow waist of the NVIDIA stack. All CUDA API calls eventually become ioctls. Proxy at this level and everything above works automatically.

### Tier 3: Container-only (fallback for nodes without KVM)

```
Node without KVM → DeploymentMode.Container
                    Docker --gpus (weak isolation, monitoring only)
```

Acceptable for trusted workloads on nodes that can't run VMs.

### What Happens to the Current Code

| Component | Disposition |
|-----------|-------------|
| `gpu_proxy_proto.h` | **Retire** — replaced by ioctl forwarding protocol |
| `cuda_shim.c` | **Retire** — no longer needed, real libcudart runs in VM |
| `cuda_driver_shim.c` | **Retire** — no longer needed, real libcuda runs in VM |
| `nvml_shim.c` | **Retire** — real libnvidia-ml runs in VM |
| `gpu_proxy_daemon.c` | **Retire** — replaced by ioctl proxy daemon |
| `transport.{c,h}` | **Reuse** — vsock/TCP transport layer still valuable |
| Wire protocol framing | **Reuse** — magic/version/header structure is solid |
| Memory quota logic | **Reuse** — quota enforcement moves to ioctl proxy |
| Token authentication | **Reuse** — auth mechanism unchanged |

Approximately 40% of the codebase (transport, auth, quota, framing) is reusable. The shims and daemon are replaced.

---

## 5. ioctl Proxy Implementation Plan

### How NVIDIA ioctls Work

```
Application
  ↓ CUDA API calls
Real libcudart.so.12 / libcuda.so.1 (NVIDIA's actual libraries)
  ↓ ioctl(fd, NV_ESC_RM_CONTROL, &params)
/dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm
  ↓ (kernel)
nvidia.ko kernel module
  ↓
Physical GPU
```

The ioctl proxy replaces the kernel interface, not the user-space libraries:

```
Inside VM:
  Application
    ↓ CUDA API calls
  Real NVIDIA UMD libraries (copied from host)
    ↓ ioctl(fd, NV_ESC_RM_CONTROL, &params)
  Virtual /dev/nvidia* (FUSE or kernel module)
    ↓ vsock
  ioctl Proxy Client

Outside VM (in Docker container):
  ioctl Proxy Daemon
    ↓ translated ioctl()
  Real /dev/nvidia* (from Docker --gpus)
    ↓
  Real nvidia.ko → Physical GPU
```

### Key ioctl Commands to Implement

Based on gVisor's nvproxy analysis, a minimal CUDA compute workload uses approximately 50-80 distinct ioctl commands. These fall into categories:

**Simple ioctls (majority — opaque forwarding):** No pointers or FDs in the struct. Copy bytes in, forward to host, copy result back. ~60% of all ioctls.

**FD-translating ioctls:** Struct contains file descriptors that must be mapped between VM and host FD spaces. ~15% of all ioctls.

**Pointer-translating ioctls:** Struct contains virtual addresses pointing into application memory. Must be copied to proxy memory, ioctl issued with proxy-side pointers, results copied back. ~20% of all ioctls.

**mmap operations:** Device memory mapping — the most complex part. GPU memory BARs mapped into the application's virtual address space. Requires shared memory between VM and host (virtio-pmem or equivalent).

### Reference: gVisor's ioctl Surface

From gVisor's `SupportedIoctls()`, a typical CUDA workload uses:

```
/dev/nvidiactl:
  NV_ESC_CARD_INFO, NV_ESC_CHECK_VERSION_STR, NV_ESC_RM_ALLOC,
  NV_ESC_RM_CONTROL, NV_ESC_RM_DUP_OBJECT, NV_ESC_RM_FREE,
  NV_ESC_RM_MAP_MEMORY, NV_ESC_RM_UNMAP_MEMORY, NV_ESC_RM_ALLOC_MEMORY,
  NV_ESC_REGISTER_FD, NV_ESC_SYS_PARAMS

/dev/nvidia-uvm:
  UVM_INITIALIZE, UVM_CREATE_RANGE_GROUP, UVM_REGISTER_GPU,
  UVM_REGISTER_GPU_VASPACE, UVM_MAP_EXTERNAL_ALLOCATION,
  UVM_MM_INITIALIZE, UVM_PAGEABLE_MEM_ACCESS, UVM_FREE
```

Each `NV_ESC_RM_CONTROL` contains a sub-command ID, making the effective surface larger. But the forwarding pattern is consistent.

### Implementation Phases

**Phase 1 — Proof of Concept (2-3 weeks)**
- Build ioctl proxy daemon (reuse transport layer from current code)
- Implement virtual `/dev/nvidiactl` and `/dev/nvidia0` via FUSE in VM
- Forward simple ioctls (no FD/pointer translation)
- Test with `nvidia-smi` inside VM (uses NVML → ioctls, no CUDA needed)
- Pin to one specific NVIDIA driver version (e.g., 550.x)

**Phase 2 — CUDA Compute (2-3 weeks)**
- Add FD translation table (VM FD ↔ host FD mapping)
- Add pointer translation for known ioctl structs
- Add `/dev/nvidia-uvm` support (Unified Virtual Memory)
- Handle mmap for GPU memory (virtio-pmem or shared memory region)
- Test with simple CUDA programs (vectorAdd, matmul)

**Phase 3 — cuBLAS/ML Frameworks (1-2 weeks)**
- Copy real NVIDIA UMD libraries into VM image
- Test with PyTorch, Ollama, vLLM
- Fix any missing ioctl commands (iterative, guided by error logs)
- Profile and optimize hot-path ioctls

**Phase 4 — Multi-version Support (ongoing)**
- Add ioctl struct definitions for additional driver versions
- Build version detection (query host driver version → select ABI)
- Use gVisor's `ioctl_sniffer` tool to validate new workloads

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Virtual device mechanism | FUSE (initial), kernel module (future) | FUSE is simpler to develop; kernel module reduces overhead |
| UMD library distribution | Copy from host into VM image | Ensures version match with host driver |
| mmap strategy | virtio-pmem shared memory region | Zero-copy for large GPU memory mappings |
| Driver version support | Pin to one version initially | Reduce complexity; expand incrementally |
| ioctl struct definitions | Derive from gVisor nvproxy source | Battle-tested, open source (Apache 2.0) |

---

## 6. Risk Analysis

### ioctl Approach Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| NVIDIA driver ABI instability | Medium | Pin to LTS driver branches (e.g., 550.x). NVIDIA breaks ABI rarely within minor versions. gVisor maintains ABI tables we can reference. |
| mmap complexity | High | Start with synchronous memcpy for data transfers. Add shared memory optimization after correctness proven. |
| FD translation bugs | Medium | Comprehensive FD lifecycle tracking. gVisor source provides reference implementation. |
| Performance overhead from FUSE | Medium | FUSE adds ~10-30μs per ioctl. Acceptable for compute (ioctls are control-plane, not data-plane). Can move to kernel module later. |
| Licensing | Low | gVisor is Apache 2.0. We reference their ioctl tables, not fork their code. Our implementation is independent. |

### What We Preserve from Current Work

The two days of debugging were not wasted. Key insights that directly inform the ioctl proxy design:

1. **Transport layer is proven** — TCP/vsock, auth, framing all work correctly
2. **Per-connection isolation model** — context/module/stream tracking per VM is the right pattern
3. **Memory quota enforcement** — tracking allocations at the proxy level is correct
4. **Lazy initialization is essential** — zero overhead at library load time
5. **VM-native compilation required** — shims must match target GLIBC (applies to FUSE client too)
6. **Device nodes required** — applications check `/dev/nvidia*` before attempting GPU init
7. **Single instance loading** — avoid dual-load conflicts from LD_PRELOAD + dlopen

---

## 7. Comparison Summary

```
                    API-level Shim           ioctl Proxy
                    (what we built)          (proposed)
                    ──────────────           ───────────
Interception:       600+ CUDA functions      ~100-200 ioctl commands
Real NVIDIA UMD:    Replaced by our shims    Runs unmodified in VM
cuBLAS/cuDNN:       ❌ Requires internal     ✅ Works natively —
                    function tables           real UMD handles it
cuGetExportTable:   ❌ Cannot proxy —         ✅ Non-issue — real
                    host-local pointers       driver returns real tables
CUDA graphs:        ❌ State machine cannot   ✅ Works natively —
                    be proxied over RPC       local CUDA runtime manages
GPU discovery:      ❌ LD_PRELOAD + NVML      ✅ Real /dev/nvidia* +
                    + systemd conflicts       real libnvidia-ml
Maintenance:        Every CUDA release adds   Driver ABI changes
                    new API functions          less frequently
Production proof:   None (rCUDA abandoned)    gVisor (Google GKE)
```

---

## 8. Recommendation

**Stop extending the API-level shims.** The cuGetExportTable wall is fundamental and affects all real-world GPU workloads (anything using cuBLAS, cuDNN, or cuFFT). Further investment in this approach has diminishing returns.

**Commit and archive the current code.** The debugging work documents critical knowledge about NVIDIA's internal architecture that directly informs the ioctl proxy design. The transport layer, auth, quota tracking, and wire framing are reusable.

**Build an ioctl-level proxy.** This is the only approach proven to handle general GPU workloads (Google runs it in production) that doesn't require IOMMU hardware. The VM-in-Docker deployment model with ioctl forwarding over virtio-vsock addresses every blocker we hit:

- cuBLAS works because the real NVIDIA UMD runs inside the VM
- No LD_PRELOAD because we operate at the device file level
- No GLIBC conflicts because we don't replace shared libraries
- No API surface explosion because ioctls are the narrow waist

**Use gVisor nvproxy as a reference** (Apache 2.0 licensed), not a fork. Their ioctl struct definitions and FD/pointer translation patterns are directly applicable. Our implementation runs as a standalone daemon over vsock, not inside the gVisor sentry.
