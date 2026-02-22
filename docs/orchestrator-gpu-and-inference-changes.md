# Orchestrator Changes Required: GPU Scheduling & Inference Template

## Context

The node agent (`DeCloud.NodeAgent`) has been updated to treat `GpuMode` as an
**orchestrator-owned scheduling parameter**. The node agent no longer auto-detects
whether to use Passthrough or Proxied GPU mode — the orchestrator must decide
this during scheduling based on node capabilities.

Branch: `claude/review-docs-pull-latest-DqvD4`
Commit: `Remove RequiresGpu field; use GpuMode as orchestrator scheduling parameter`

---

## 1. Remove `requiresGpu` from CreateVm Payloads

The `requiresGpu` boolean field has been **removed** from the node agent's
`VmSpec`. The node agent will silently ignore it if sent, but it no longer has
any effect. Remove it from all CreateVm command payloads.

**Before (old):**
```json
{
  "vmId": "...",
  "gpuMode": 0,
  "requiresGpu": true,
  ...
}
```

**After (new):**
```json
{
  "vmId": "...",
  "gpuMode": 1,
  ...
}
```

---

## 2. Set `gpuMode` Explicitly During Scheduling

The orchestrator must now set `gpuMode` in every CreateVm payload based on the
template's GPU requirements and the target node's hardware capabilities.

### GpuMode Values

| Value | Name        | Meaning                                                        |
|-------|-------------|----------------------------------------------------------------|
| 0     | None        | No GPU access                                                  |
| 1     | Passthrough | VFIO passthrough — dedicated GPU, requires IOMMU on the node   |
| 2     | Proxied     | GPU proxy daemon — shared GPU over virtio-vsock, no IOMMU req  |

### Scheduling Logic

When a VM template requires GPU (e.g., Inference / AI Chatbot):

1. **User selects "dedicated GPU" (full performance):**
   - Schedule on a node that has IOMMU enabled and an available GPU
   - Send `gpuMode: 1` (Passthrough)
   - Optionally send `gpuPciAddress` to pin a specific GPU, or omit it to let
     the node agent auto-assign from its pool

2. **User selects "shared GPU" (cost-effective):**
   - Schedule on any node that has a GPU (IOMMU not required)
   - Send `gpuMode: 2` (Proxied)
   - Do NOT send `gpuPciAddress` (not used for proxied mode)

3. **No GPU needed:**
   - Send `gpuMode: 0` (None) or omit the field (defaults to 0)

### Node Capability Discovery

The orchestrator needs to know per-node:
- Whether the node has GPUs (and how many are available)
- Whether the node has IOMMU enabled (for Passthrough scheduling)

This information should come from the node's inventory/heartbeat reports. If
the orchestrator already tracks node capabilities, use those. Otherwise, add
fields like `hasGpu`, `gpuCount`, `iommuEnabled` to the node registration or
heartbeat payload.

### What the Node Agent Does

- **`gpuMode: 1` with `gpuPciAddress`** — Verifies that specific GPU isn't
  already assigned to another VM. If it is, the VM creation proceeds without
  GPU (logs a warning).
- **`gpuMode: 1` without `gpuPciAddress`** — Auto-assigns the first available
  passthrough-capable GPU from its local pool. If none available, falls back to
  no GPU (logs a warning).
- **`gpuMode: 2`** — Sets up virtio-vsock and GPU proxy daemon. No PCI address
  needed; multiple VMs can share one GPU.
- **`gpuMode: 0` with `gpuPciAddress` set** — Legacy fallback: infers
  Passthrough mode automatically.

---

## 3. Update the AI Chatbot / Inference Template

The current inference template in the node agent (`inference-vm-cloudinit.yaml`)
is a **stub** that references a non-existent `decloud/inference-api:latest`
Docker image. It needs to be replaced with a real Ollama + Open WebUI
deployment.

### Current Template Problems

1. Uses fictional image `decloud/inference-api:latest`
2. Hardcoded model list (`llama-2-7b`, `mistral-7b`, etc.) in a JSON config
   that nothing reads
3. All config values are hardcoded in the node agent's
   `PopulateInferenceVariablesAsync()` — the orchestrator has no way to
   customize them
4. Template always assumes GPU is present (`gpu_enabled: true`)

### What the Orchestrator Template Should Define

The orchestrator likely maintains template definitions (marketplace entries)
that map to CreateVm payloads. For the "AI Chatbot (Ollama + Open WebUI)"
template, it should produce a CreateVm payload like:

```json
{
  "vmId": "chatbot-<uuid>",
  "name": "ai-chatbot-<short-id>",
  "ownerId": "<wallet-address>",
  "vmType": 7,
  "gpuMode": 1,
  "qualityTier": 1,
  "virtualCpuCores": 4,
  "memoryBytes": 17179869184,
  "diskBytes": 107374182400,
  "baseImageUrl": "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img",
  "sshPublicKey": "<user-ssh-key>",
  "deploymentMode": 0,
  "labels": {
    "template": "ai-chatbot",
    "ollama-models": "llama3.2,mistral",
    "webui-port": "3000"
  },
  "services": [
    {
      "name": "Open WebUI",
      "port": 3000,
      "protocol": "tcp",
      "checkType": "HttpGet",
      "httpPath": "/",
      "timeoutSeconds": 600
    }
  ]
}
```

**Key fields:**
- `vmType: 7` (Inference)
- `gpuMode: 1` or `2` depending on user's GPU preference / node availability
- `labels` carry template-specific config that the cloud-init template reads
- `services` define readiness checks so the orchestrator knows when the chatbot
  is ready

### Node Agent Cloud-Init Template Update

The node agent's `inference-vm-cloudinit.yaml` should be updated to deploy
Ollama + Open WebUI. The docker-compose section should look something like:

```yaml
# Docker Compose for AI Chatbot (Ollama + Open WebUI)
- path: /opt/decloud-inference/docker-compose.yml
  permissions: '0644'
  owner: inference:inference
  content: |
    version: '3.8'

    services:
      ollama:
        image: ollama/ollama:latest
        container_name: ollama-__VM_ID__
        restart: unless-stopped
        ports:
          - "11434:11434"
        volumes:
          - ollama-data:/root/.ollama
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: all
                  capabilities: [gpu]
        networks:
          - inference-network

      open-webui:
        image: ghcr.io/open-webui/open-webui:main
        container_name: webui-__VM_ID__
        restart: unless-stopped
        ports:
          - "3000:8080"
        volumes:
          - webui-data:/app/backend/data
        environment:
          - OLLAMA_BASE_URL=http://ollama:11434
        depends_on:
          - ollama
        networks:
          - inference-network

    volumes:
      ollama-data:
        driver: local
      webui-data:
        driver: local

    networks:
      inference-network:
        driver: bridge
```

And the `runcmd` section should pull a default model after startup:

```yaml
runcmd:
  # ... (docker setup, nvidia-ctk, etc. — same as current) ...

  # Start the stack
  - systemctl daemon-reload
  - systemctl enable decloud-inference
  - systemctl start decloud-inference

  # Wait for Ollama to be ready, then pull default model(s)
  - |
    for i in $(seq 1 30); do
      curl -s http://localhost:11434/api/tags && break
      sleep 5
    done
  - docker exec ollama-__VM_ID__ ollama pull llama3.2 || true
```

### Decision: Single Template or Separate VmType?

**Option A — Reuse `VmType.Inference` (7):**
The orchestrator sends `vmType: 7` with labels indicating it's an AI Chatbot.
The node agent's inference template is updated to deploy Ollama + Open WebUI.
Simpler, but couples the inference template to one specific use case.

**Option B — Add `VmType.AiChatbot` (8):**
Add a new enum value in the node agent, create a separate
`ai-chatbot-vm-cloudinit.yaml`, and a new case in `CloudInitTemplateService`.
More flexible — inference template stays generic for future use.

This is your call. If Option B, the node agent needs a small PR to add the enum
value and template file.

---

## 4. Reference: Full CreateVm Payload Schema

All properties the node agent accepts (both `camelCase` and `PascalCase`
variants are supported):

| Property               | Type       | Default            | Notes                              |
|------------------------|------------|--------------------|------------------------------------|
| `vmId`                 | string     | generated UUID     | Unique VM identifier               |
| `name`                 | string     | `vm-{vmId[:8]}`    | Display name                       |
| `ownerId`              | string     | `"unknown"`        | Tenant/user ID                     |
| `password`             | string?    | null               | Optional (wallet-encrypted)        |
| `vmType`               | int        | 0                  | See VmType enum below              |
| `qualityTier`          | int        | 3                  | See QualityTier enum below         |
| `virtualCpuCores`      | int        | 1                  | vCPU count                         |
| `computePointCost`     | int        | 0                  | Billing cost                       |
| `memoryBytes`          | long       | 1024               | Memory in bytes                    |
| `diskBytes`            | long       | 10                 | Disk in bytes                      |
| `baseImageUrl`         | string?    | null               | Base image download URL            |
| `imageId`              | string?    | null               | Image identifier                   |
| `sshPublicKey`         | string?    | null               | SSH public key                     |
| `userData`             | string?    | null               | Custom cloud-init user data        |
| `gpuPciAddress`        | string?    | null               | Specific GPU PCI address           |
| `gpuMode`              | int        | 0                  | 0=None, 1=Passthrough, 2=Proxied  |
| `deploymentMode`       | int        | 0                  | 0=VM, 1=Container                  |
| `containerImage`       | string?    | null               | Docker image (container mode only) |
| `environmentVariables` | dict?      | null               | Env vars (container mode only)     |
| `labels`               | dict?      | null               | Arbitrary key-value metadata       |
| `services`             | array?     | system service     | Readiness check definitions        |

### Enums

**VmType:** General=0, Compute=1, Memory=2, Storage=3, Gpu=4, Relay=5, Dht=6, Inference=7

**GpuMode:** None=0, Passthrough=1, Proxied=2

**DeploymentMode:** VirtualMachine=0, Container=1

**QualityTier:** Guaranteed=0, Standard=1, Balanced=2, Burstable=3

---

## 5. Summary of Required Orchestrator Changes

1. **Stop sending `requiresGpu`** — field removed from node agent
2. **Send `gpuMode` explicitly** (1 or 2) for GPU workloads based on node
   capabilities during scheduling
3. **Track node GPU capabilities** (has GPU, IOMMU enabled, available GPU count)
   for scheduling decisions
4. **Update the AI Chatbot template** to produce CreateVm payloads with correct
   `gpuMode`, `labels`, and `services`
5. **Coordinate with node agent** on whether to reuse `VmType.Inference` or add
   a new `VmType.AiChatbot` — then update the node agent's cloud-init template
   to deploy Ollama + Open WebUI
