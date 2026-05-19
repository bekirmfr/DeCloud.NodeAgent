# Optimization: System VM Base Image Size

**Severity:** Medium — impacts first-install experience and node storage
**Component:** System VM images (relay, DHT, blockstore)
**Current state:** Each system VM downloads a 427 MB Debian 12 generic cloud image
**Goal:** Reduce download size and boot time for system VMs

---

## Problem

System VMs are lightweight infrastructure services — a single Go binary
(relay, DHT, blockstore) running on a minimal Linux kernel. Yet each is
built on the full Debian 12 generic cloud image:

```
https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2
```

**Download:** ~427 MB per image
**First install time:** ~7 minutes on a 1 Gbps connection (observed ~1 MB/10s)
**Disk footprint:** 427 MB cached base + 2-10 GB overlay per VM

The image contains packages the system VMs never use: man pages, docs,
locales, firmware blobs, apt cache, systemd units for services that
aren't running.

---

## Current Architecture

### Image Registry (Orchestrator)

`src/Orchestrator/Models/BaseImageUrlResolver.cs`:

```
["debian-12-relay"]      → debian-12-generic-amd64.qcow2  (427 MB)
["debian-12-dht"]        → debian-12-generic-amd64.qcow2  (427 MB)
["debian-12-blockstore"] → debian-12-generic-amd64.qcow2  (427 MB)
```

All three image IDs resolve to the **exact same URL**. The image cache
on the node agent keys by URL, so after the first download, the other
two system VMs should use the cached copy. But the initial download
still takes ~7 minutes.

### Image Seed Data (Orchestrator)

`src/Orchestrator/Persistence/DataStore.cs` — `InitialiseSeedData`:

```csharp
new VmImage { Id = "debian-12-relay",      SizeGb = 2, ... }
new VmImage { Id = "debian-12-dht",        SizeGb = 2, ... }
new VmImage { Id = "debian-12-blockstore", SizeGb = 2, ... }
```

Three separate image records, all pointing to the same physical image.

### VM Creation Flow

1. `SystemVmReconciler` decides to create a system VM
2. Calls `VmDeploymentPipeline` → `ImageManager.EnsureImageAvailableAsync`
3. Downloads if not cached, creates qcow2 overlay on top of base
4. Injects cloud-init (installs Go binary, configures service)
5. Boots VM — cloud-init runs on first boot

### What cloud-init does inside the VM

The system VM cloud-init template:
- Downloads a single Go binary (~10-20 MB) from artifact cache
- Configures systemd unit for the service
- Sets up networking (WireGuard for relay, libp2p for DHT/blockstore)
- No apt packages installed at boot time

---

## Why It's Slow

1. **Upstream image is generic** — includes firmware, locales, man
   pages, apt cache, full systemd suite. System VMs use <5% of it.

2. **Download from cloud.debian.org** — CDN but not optimized for
   our use case. No delta updates, no regional mirrors.

3. **qcow2 overlay allocation** — even though the VM only writes
   a few MB (the Go binary + config), the backing image must be
   fully downloaded before the overlay can be created.

---

## Options

### Option A: Switch to Alpine (minimal change)

Alpine Linux cloud image is ~50 MB. Already in the image registry:

```
["alpine-3.19"] → nocloud_alpine-3.19.1-x86_64-bios-cloudinit-r0.qcow2  (~50 MB)
```

**Changes needed:**
- Update `BaseImageUrlResolver.cs`: point system VM image IDs to Alpine
- Update cloud-init templates: Alpine uses `apk` not `apt`, `openrc`
  not `systemd`
- Test Go binary compatibility (static binaries work fine on Alpine)
- Update system VM specs if memory/disk requirements change

**Pro:** 9x smaller download. Proven cloud image. Cloud-init supported.
**Con:** Cloud-init on Alpine is less mature. OpenRC instead of systemd
requires template rewrites.

### Option B: Custom minimal image (purpose-built)

Build a custom qcow2 with only what system VMs need:
- Minimal kernel + initramfs
- cloud-init (or a simpler alternative like ignition)
- Network tools (ip, wg for relay)
- No package manager (everything pre-installed or injected via
  cloud-init artifacts)

**Target size:** 80-120 MB

**Build tooling:** `debootstrap` minimal + cleanup script, or
`cloud-image-utils` with package exclusion list.

**Pro:** Smallest possible size. Full control over contents.
**Con:** Build pipeline to maintain. Security updates require rebuilds.

### Option C: Shared base image with explicit deduplication

Keep Debian 12 but ensure the image cache correctly deduplicates:

1. Use a single image ID `debian-12-system` for all three roles
2. Verify `ImageManager` cache deduplication works across image IDs
   that resolve to the same URL
3. Pre-download during install (background task) instead of blocking
   first VM creation

**Pro:** No template changes. Minimal code change.
**Con:** Still 427 MB first download. Doesn't solve the size problem.

### Option D: Layered image with pre-baked system VM base

Build a custom "system-vm-base" image that is Debian 12 minimal with
pre-installed common dependencies (curl, ca-certificates, systemd).
Cloud-init only injects the Go binary and config.

**Target size:** 150-200 MB (stripped Debian)
**Build:** `virt-builder` or `virt-customize` with package removal list

**Pro:** Familiar Debian base. Smaller than generic. systemd templates
work unchanged.
**Con:** Still larger than Alpine. Custom build pipeline.

### Recommended: Option A (Alpine) for system VMs

System VMs run a single static Go binary. Alpine is the industry
standard for this workload (Docker uses it for the same reason).
The 50 MB image downloads in <30 seconds. Cloud-init support is
sufficient for the simple templates used by system VMs.

Keep Debian/Ubuntu for tenant VMs where users expect a full OS.

---

## Files to Examine

| File | Relevance |
| --- | --- |
| `src/Orchestrator/Models/BaseImageUrlResolver.cs` | Image ID → URL mapping. Change system VM image IDs here. |
| `src/Orchestrator/Persistence/DataStore.cs` | `InitialiseSeedData` — VmImage seed records for system VMs |
| `src/DeCloud.NodeAgent.Infrastructure/Services/ImageManager.cs` | Download, cache, and overlay creation. Verify cache dedup behavior. |
| `src/DeCloud.NodeAgent.Infrastructure/Services/SystemVm/SystemVmReconciler.cs` | Creates system VMs — reads template spec including BaseImageUrl |
| `src/Shared/Models/SystemVmTemplate.cs` | Template model carrying BaseImageUrl to the node agent |
| `src/Orchestrator/Services/SystemVm/SystemVmTemplateSeeder.cs` | Template definitions including RecommendedSpec with image ID |
| Cloud-init templates (relay, DHT, blockstore) | Would need adaptation if switching from Debian/systemd to Alpine/openrc |

---

## Measurements to Take

Before choosing an approach, measure:

```bash
# Current base image size
ls -lh /var/lib/decloud/images/

# Actual disk usage per system VM (overlay only, excluding base)
du -sh /var/lib/decloud/vms/*/disk.qcow2

# Alpine image size for comparison
wget -q --spider --server-response \
  https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/cloud/nocloud_alpine-3.19.1-x86_64-bios-cloudinit-r0.qcow2 \
  2>&1 | grep Content-Length

# Boot time comparison (if testing)
time virsh start <vm-id>  # from create to cloud-init complete
```

---

## Impact

| Metric | Current (Debian) | Option A (Alpine) | Option D (Slim Debian) |
| --- | --- | --- | --- |
| Base image download | 427 MB (~7 min) | ~50 MB (~30s) | ~180 MB (~2 min) |
| First node setup | ~10 min (3 VMs serial) | ~2 min | ~5 min |
| Disk cache footprint | 427 MB | 50 MB | 180 MB |
| Cloud-init template changes | None | Rewrite (apk/openrc) | None |
| Maintenance burden | None (upstream) | Low (upstream) | Medium (custom build) |
