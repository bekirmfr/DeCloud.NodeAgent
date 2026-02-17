# DeCloud: The Minecraft Vision
## Building the World's First Emergent Compute Network

**Version:** 2.0
**Last Updated:** 2026-02-17
**Status:** Phase 1 COMPLETE, Phase 2 IN PROGRESS — See status markers below
**Philosophy:** Simple primitives → Complex outcomes → Community ownership

---

## 🎮 THE MINECRAFT ANALOGY

### What Makes Minecraft Work

```
MINECRAFT FORMULA:
1. Simple building blocks (64 block types)
2. Composable mechanics (redstone, crafting, farming)
3. Permissionless creation (anyone can build anything)
4. Emergent complexity (computers, cities, economies)
5. Multiplayer discovery (server lists, sharing worlds)
6. Community content (mods, texture packs, maps)
7. Economic systems (villagers, trading, markets)

RESULT: 200M+ players, $3B+ revenue, cultural phenomenon
```

### The DeCloud Translation

```
DECLOUD FORMULA:
1. Simple compute blocks (VMs, storage, network)
2. Composable infrastructure (relay, DHT, inference)
3. Permissionless deployment (wallet = identity)
4. Emergent applications (AI models, privacy tools, distributed services)
5. Discovery layer (node marketplace, image sharing)
6. Community templates (one-click deployments, infrastructure patterns)
7. Economic roles (node operators, relay providers, marketplace creators)

GOAL: World's largest decentralized compute network
```

---

## 📊 CURRENT STATE ASSESSMENT (Updated 2026-02-17)

### What's Built ✅ (~85% Minecraft-Ready)

#### 1. **Core Primitives** ✅
```
- General VMs (basic compute)
- Relay VMs (auto-deployed networking infrastructure)
- DHT VMs (libp2p decentralized coordination — production-verified)
- Inference VMs (AI workloads with GPU passthrough)
- Quality Tiers (Guaranteed/Standard/Balanced/Burstable)
- Bandwidth Tiers (Basic/Standard/Performance/Unmetered with libvirt QoS)
```

#### 2. **Self-Organizing Infrastructure** ✅
```
- Relay auto-deployment for CGNAT nodes (60-80% of users)
- SystemVm reconciliation (Kubernetes-style controller for relay, DHT, ingress VMs)
- DHT mesh auto-enrollment over WireGuard tunnels
- Smart port allocation with 3-hop DNAT forwarding
- CentralIngress-aware routing (HTTP via Caddy, TCP/UDP via iptables)
```

#### 3. **Permissionless Participation** ✅
```
- Wallet = identity, no approval needed
- Node operators: Install agent, authenticate with wallet
- Users: Connect wallet, deploy VMs
- No KYC, no whitelist, no gatekeeping
```

#### 4. **Economic Foundation** ✅
```
- USDC payments on Polygon with escrow smart contract
- Hybrid pricing: platform floor rates + node operator custom pricing
- Bandwidth-aware billing with tier multipliers
- Per-hour compute point pricing across quality tiers
```

#### 5. **Discovery Layer** ✅ (was missing in v1.0)
```
- Node marketplace with search, filtering, featured nodes
- One-click "Deploy Here" from marketplace cards
- Multi-criteria search (tags, region, GPU, price, uptime)
```

#### 6. **Marketplace & Templates** ✅ (was missing in v1.0)
```
- VM template marketplace (6 seed templates, community submissions)
- One-click deployments with cloud-init variable substitution
- Paid templates with 85/15 author/platform revenue split
- Draft → publish workflow for community templates
```

#### 7. **Reputation System** ✅ (was missing in v1.0)
```
- 30-day rolling uptime tracking via failed heartbeat detection
- Universal review system (templates + nodes, eligibility-verified)
- Denormalized rating aggregates
- TotalVmsHosted + SuccessfulVmCompletions counters
```

#### 8. **Per-Service VM Readiness** ✅
```
- Distinguishes "VM Running" from "VM Ready for Service"
- qemu-guest-agent probing (CloudInitDone, TcpPort, HttpGet, ExecCommand)
- Auto-inference from template protocols
- Orchestrator + NodeAgent both complete
```

### What's Still Missing ❌ (~15% To Build)

#### 1. **Collaboration Features** ❌
```
Current: Single-user VMs only
Needed: Shared VMs, team workspaces, infrastructure sharing
```

#### 2. **Visualization** ❌
```
Current: No network map
Needed: Live topology view, resource visualization
```

#### 3. **Advanced Economics** ❌
```
Current: Static pricing, no relay revenue sharing
Needed: Dynamic pricing, relay revenue splits, staking/slashing
```

#### 4. **Weighted Trust Score** ❌
```
Current: Raw uptime percentage only
Needed: Composite score (uptime 40% + performance 20% + ratings 25% + longevity 10%)
```

#### 5. **Frontend Polish** ❌
```
Current: Backend-complete features lacking frontend
Needed: Trust badges, review prompts after VM termination, node operator dashboard
```

---

## 🗺️ IMPLEMENTATION ROADMAP

### Phase 1: Discovery & Marketplace (Weeks 1-4) — ✅ COMPLETE

**Goal:** Let users discover and choose nodes like Minecraft server lists
**Status:** All features shipped and production-tested as of 2026-02-09.

#### Feature 1.1: Node Marketplace ✅ COMPLETE (2026-01-30)

**Implemented in:** `NodeMarketplaceService.cs` (260 lines), `MarketplaceController.cs` (821 lines)

**What was built:**
- `GET /api/marketplace/nodes` — Search with filters (tags, region, GPU, price, uptime, capacity)
- `GET /api/marketplace/nodes/featured` — Top 10 nodes (>95% uptime, online, with description)
- `GET /api/marketplace/nodes/{id}` — Node details with full hardware specs
- `PATCH /api/marketplace/nodes/{id}/profile` — Operator profile management
- Frontend: Rich marketplace UI with node cards, detail modals, multi-criteria search
- One-click "Deploy Here" button pre-populates VM creation modal with target node

#### Feature 1.2: VM Template Marketplace ✅ COMPLETE (2026-02-09)

**Implemented in:** `TemplateService.cs` (715 lines), `TemplateSeederService.cs` (1,828 lines)

**What was built:**
- Full template CRUD with community and platform-curated templates
- 5 seed categories: AI & ML, Databases, Dev Tools, Web Apps, Privacy & Security
- 6 seed templates: Stable Diffusion, PostgreSQL, VS Code Server, Private Browser (Neko), Shadowsocks, Web Proxy Browser (Ultraviolet)
- Cloud-init variable substitution (${DECLOUD_VM_ID}, ${DECLOUD_PASSWORD}, etc.)
- Paid templates with PerDeploy pricing (85/15 author/platform split via escrow)
- Draft → publish workflow for community submissions
- Security validation (fork bombs, rm -rf, untrusted curl|bash detection)
- Frontend: marketplace-templates.js, my-templates.js, template-detail.js

**What the vision proposed vs what was actually built:**

| Vision Proposed | Actually Built | Delta |
|---|---|---|
| `VmTemplate` model with tags, ratings, pricing | Full model + categories, slugs, specs, readiness checks | Exceeded |
| `SearchTemplatesAsync` interface | Full CRUD + search + filter + featured + deploy | Exceeded |
| 3 example templates (Whisper, Nextcloud, Minecraft) | 6 production templates (SD, PostgreSQL, VS Code, Neko, Shadowsocks, Ultraviolet) | Different set, same spirit |
| Simple review system | Universal ReviewService with eligibility proofs, denormalized aggregates | Exceeded |

---

### Phase 2: Reputation & Trust (Weeks 5-8) — ⚠️ 70% COMPLETE (IN PROGRESS)

**Goal:** Build trust signals so users can choose reliable nodes
**Status:** Core backend complete. Weighted trust score algorithm and frontend polish remaining.

#### Feature 2.1: Reputation System ✅ BACKEND COMPLETE (2026-01-30)

**Implemented in:** `NodeReputationService.cs` (352 lines), `NodeReputationMaintenanceService.cs`

**What was built:**
- 30-day rolling uptime tracking via failed heartbeat detection (15s precision)
- `FailedHeartbeatsByDay` dictionary with auto-cleanup (30-day window)
- `TotalVmsHosted` and `SuccessfulVmCompletions` lifetime counters
- Hourly maintenance service for recalculation and data cleanup
- Integrated with `NodeHealthMonitorService` (30s detection cycle)

**What's still missing from the vision:**

| Vision Proposed | Status |
|---|---|
| Uptime tracking (30-day rolling) | ✅ Implemented |
| `ConsecutiveDaysOnline` | ❌ Not tracked |
| `AverageResponseTime`, `AverageVmBootTime` | ❌ Not tracked |
| `FailedVmDeployments` counter | ❌ Not tracked |
| Weighted trust score (0-100) | ❌ Only raw uptime % exists |
| `SlashingHistory` / `TotalSlashed` | ❌ No slashing mechanism |
| Trust score algorithm (40% uptime + 20% perf + 25% ratings + 10% longevity) | ❌ Not implemented |

#### Feature 2.2: User Reviews ✅ BACKEND COMPLETE (2026-02-09)

**Implemented in:** `ReviewService.cs` (223 lines)

**What was built:**
- Universal review system for templates AND nodes
- 1-5 star ratings with eligibility verification (proof of deployment/usage)
- One review per user per resource enforcement
- Denormalized rating aggregates (AverageRating, TotalReviews, RatingDistribution)
- API endpoints: submit review, get reviews, check user's existing review

**What's still missing:**
- ❌ Frontend: review prompt modal after VM termination
- ❌ Frontend: node rating aggregates display in marketplace
- ❌ Frontend: trust badges ("99.9% uptime", "100+ VMs hosted")
- ❌ `ReviewService.UpdateReviewAsync` throws `NotImplementedException`

---

### Phase 3: Collaboration & Multiplayer (Weeks 9-12) — ❌ NOT STARTED

**Goal:** Enable teams to work together, like Minecraft multiplayer
**Status:** No implementation exists. All designs below are aspirational.

#### Feature 3.1: Shared VMs (Multi-Wallet Access)

```csharp
// VirtualMachine model update
public class VirtualMachine
{
    // ... existing fields ...
    
    // Multi-wallet access
    public string OwnerWalletAddress { get; set; } // Primary owner
    public List<string> CollaboratorWallets { get; set; } = new();
    public VmAccessPolicy AccessPolicy { get; set; }
}

public class VmAccessPolicy
{
    public bool AllowCollaboratorStart { get; set; }
    public bool AllowCollaboratorStop { get; set; }
    public bool AllowCollaboratorDelete { get; set; }
    public bool AllowCollaboratorSsh { get; set; }
}

// Service methods
public interface IVmCollaborationService
{
    Task<bool> AddCollaboratorAsync(string vmId, string collaboratorWallet, VmAccessPolicy policy);
    Task<bool> RemoveCollaboratorAsync(string vmId, string collaboratorWallet);
    Task<List<VirtualMachine>> GetSharedVmsAsync(string userWallet);
}
```

**Use Cases:**
- Dev teams sharing staging environments
- Research groups sharing AI training nodes
- Students collaborating on projects

#### Feature 3.2: Infrastructure Templates (Shared Configs)

```csharp
// Users share entire infrastructure setups
public class InfrastructureTemplate
{
    public string Id { get; set; }
    public string Name { get; set; }
    public string Description { get; set; }
    public string CreatorWallet { get; set; }
    
    // Multiple VMs + networking
    public List<VmSpec> VmSpecs { get; set; }
    public NetworkTopology? Network { get; set; }
    
    // Example: "3-tier web app" (frontend VM + backend VM + database VM)
}

public class NetworkTopology
{
    public List<VmConnection> Connections { get; set; }
}

public class VmConnection
{
    public string SourceVmName { get; set; }
    public string TargetVmName { get; set; }
    public List<int> AllowedPorts { get; set; }
}
```

**Example Templates:**

```yaml
# 3-Tier Web Application
name: "Scalable Web App Stack"
vms:
  - name: "frontend"
    type: "nginx-proxy"
    cpu: 2
    memory: 4GB
  - name: "backend"
    type: "node-api"
    cpu: 4
    memory: 8GB
  - name: "database"
    type: "postgres"
    cpu: 2
    memory: 16GB
network:
  - frontend -> backend:3000
  - backend -> database:5432

---

# Distributed AI Training
name: "Multi-GPU AI Training Cluster"
vms:
  - name: "coordinator"
    type: "training-coordinator"
    cpu: 8
    memory: 32GB
  - name: "worker-1"
    type: "gpu-worker"
    cpu: 16
    memory: 64GB
    gpu: "A100"
  - name: "worker-2"
    type: "gpu-worker"
    cpu: 16
    memory: 64GB
    gpu: "A100"
```

#### Feature 3.3: Team Workspaces

```csharp
public class Workspace
{
    public string Id { get; set; }
    public string Name { get; set; }
    public string OwnerWallet { get; set; }
    
    // Team members
    public List<WorkspaceMember> Members { get; set; }
    
    // Resources
    public List<string> VmIds { get; set; }
    public List<string> TemplateIds { get; set; }
    
    // Billing
    public string PaymentWallet { get; set; } // Centralized billing
    public decimal MonthlyBudget { get; set; }
    public decimal CurrentSpend { get; set; }
}

public class WorkspaceMember
{
    public string WalletAddress { get; set; }
    public WorkspaceRole Role { get; set; }
}

public enum WorkspaceRole
{
    Admin,   // Full access
    Editor,  // Create/modify VMs
    Viewer   // Read-only
}
```

---

### Phase 4: Visualization & Discovery (Weeks 13-16) — ❌ NOT STARTED

**Goal:** Make the network "visible" like Minecraft's F3 screen
**Status:** No implementation exists. Node details are available via text-based marketplace UI only.

#### Feature 4.1: Live Network Map

```javascript
// Real-time visualization using D3.js or Three.js
class NetworkMap {
    constructor() {
        this.nodes = [];
        this.vms = [];
        this.relayConnections = [];
    }
    
    async loadData() {
        const data = await fetch('/api/network/topology').then(r => r.json());
        
        // Render nodes as spheres
        data.nodes.forEach(node => {
            this.renderNode(node);
        });
        
        // Render VMs on nodes
        data.vms.forEach(vm => {
            this.renderVm(vm, vm.nodeId);
        });
        
        // Render relay connections
        data.relayConnections.forEach(conn => {
            this.renderConnection(conn.cgnatNode, conn.relayNode);
        });
    }
    
    renderNode(node) {
        // Visual representation:
        // - Size = total compute capacity
        // - Color = utilization (green = low, yellow = medium, red = high)
        // - Glow = relay node
        // - Position = geographic location
    }
}
```

**What Users Can See:**
- All active nodes (colored by region)
- Their own VMs (highlighted)
- Relay topology (CGNAT nodes connected to relay nodes)
- Resource utilization heatmap
- Live VM creation/termination animations

#### Feature 4.2: Node Explorer

```javascript
// Click on any node to see details
async function showNodeDetails(nodeId) {
    const node = await fetch(`/api/nodes/${nodeId}`).then(r => r.json());
    
    // Show modal with:
    // - Hardware specs
    // - Current VMs
    // - Reputation score
    // - Price
    // - Operator description
    // - "Deploy Here" button
}
```

---

### Phase 5: Advanced Economics (Weeks 17-20) — ⚠️ 15% COMPLETE

**Goal:** Create specialized roles and economic differentiation
**Status:** Static pricing and USDC escrow exist. Dynamic pricing, relay revenue sharing, and staking/slashing are not implemented.

#### Feature 5.1: Node Specialization Pricing

```csharp
public class NodePricingStrategy
{
    // Base price per compute point
    public decimal BasePrice { get; set; }
    
    // Premiums for specialization
    public decimal GpuPremium { get; set; }
    public decimal HighMemoryPremium { get; set; }
    public decimal NvmePremium { get; set; }
    public decimal LowLatencyPremium { get; set; }
    
    // Geographic premiums
    public Dictionary<string, decimal> RegionMultipliers { get; set; }
    
    // Reputation bonus
    public decimal HighReputationDiscount { get; set; } // Trusted nodes can charge less
}

// Dynamic pricing
public decimal CalculateVmCost(VmSpec spec, Node node)
{
    var baseCost = spec.ComputePointCost * node.Pricing.BasePrice;
    
    // Apply premiums
    if (spec.RequiresGpu && node.Capabilities.HasGpu)
    {
        baseCost += node.Pricing.GpuPremium;
    }
    
    if (node.Region == "US-East")
    {
        baseCost *= node.Pricing.RegionMultipliers["US-East"];
    }
    
    // High reputation nodes might charge less (they're in demand anyway)
    if (node.Reputation.TrustScore > 90)
    {
        baseCost *= (1 - node.Pricing.HighReputationDiscount);
    }
    
    return baseCost;
}
```

#### Feature 5.2: Relay Revenue Sharing

```csharp
// Relay operators earn from enabling CGNAT nodes
public class RelayEconomics
{
    public async Task<decimal> CalculateRelayEarningsAsync(Node relayNode)
    {
        var cgnatNodes = await GetCgnatNodesUsingRelay(relayNode.Id);
        
        decimal totalEarnings = 0;
        
        foreach (var cgnatNode in cgnatNodes)
        {
            var vms = await GetActiveVmsOnNode(cgnatNode.Id);
            var nodeRevenue = vms.Sum(vm => vm.TotalRevenue);
            
            // Relay gets 15% of revenue from VMs it enables
            totalEarnings += nodeRevenue * 0.15m;
        }
        
        return totalEarnings;
    }
}
```

#### Feature 5.3: Staking & Slashing

```csharp
public class NodeStaking
{
    // Node operators stake USDC to participate
    public decimal RequiredStake { get; set; } = 1000m; // $1000 minimum
    
    // Higher stake = higher trust score
    public decimal CalculateTrustBonus(decimal stakedAmount)
    {
        return Math.Min(stakedAmount / 10000m, 1.0m) * 10m; // Max 10 bonus points
    }
    
    // Slashing for misbehavior
    public async Task SlashNodeAsync(string nodeId, string reason, decimal amount)
    {
        var node = await GetNodeAsync(nodeId);
        
        node.StakedAmount -= amount;
        node.Reputation.TotalSlashed += amount;
        
        if (node.StakedAmount < RequiredStake)
        {
            // Node falls below minimum stake - suspend
            await SuspendNodeAsync(nodeId);
        }
        
        // Record slashing event
        await RecordSlashingAsync(nodeId, reason, amount);
    }
}
```

**Slashing Triggers:**
- Extended downtime (>24 hours)
- Failed attestations (fraud detection)
- User complaints (verified abuse)
- Terms of Service violations

---

## 🎨 USER EXPERIENCE EXAMPLES

### Example 1: New User Onboarding

```
User Journey: "I want to deploy a private AI chatbot"

1. Connect wallet (MetaMask)
2. Browse template marketplace
3. Find "Private ChatGPT Clone" template
4. See: "4-core, 16GB RAM, GPU-accelerated, $12/month"
5. Click "Deploy from Template"
6. Choose node (filtered to "GPU nodes in EU")
7. One-click deployment
8. Get SSH access + web URL
9. ChatGPT running in 90 seconds

Total steps: 6 clicks
Complexity hidden: VM provisioning, relay assignment, network configuration
```

### Example 2: Power User Building Complex Infrastructure

```
User Journey: "I want to build a 5-node Kubernetes cluster"

1. Create new workspace "Production K8s"
2. Invite team members (3 DevOps engineers)
3. Browse infrastructure templates
4. Find "HA Kubernetes Cluster" template
5. Customize: 5 nodes, mix of regions for HA
6. Deploy entire stack
7. Team collaborates via shared SSH access
8. Template auto-configures networking between nodes
9. Export configuration to share with community

Minecraft parallel: Building a complex redstone computer
Emergence: Template becomes popular, other users fork it, improve it
```

### Example 3: Node Operator Specialization

```
Node Operator Journey: "I have a spare server with 4x RTX 4090s"

1. Install node agent
2. Authenticate with wallet
3. Agent detects GPUs automatically
4. Set pricing: "GPU VMs: $2/hour (2x base price)"
5. Add description: "High-end consumer GPUs, perfect for Stable Diffusion, LLMs"
6. Tag node: "gpu", "ai", "stable-diffusion", "uncensored"
7. Node appears in marketplace
8. Users searching for "AI without filters" find this node
9. Operator earns premium for specialized hardware

Economic differentiation: Not competing on price, competing on capability
```

---

## 🧩 TECHNICAL IMPLEMENTATION PRIORITIES

### Priority 1: Foundation (Weeks 1-4) — ✅ COMPLETE
- [x] Node marketplace backend (search, filtering, advertisements) — `NodeMarketplaceService.cs`
- [x] Template marketplace backend (create, browse, deploy) — `TemplateService.cs`, `TemplateSeederService.cs`
- [x] Reputation calculation service — `NodeReputationService.cs`
- [x] Frontend: Node browser UI — `marketplace.js`
- [x] Frontend: Template marketplace UI — `marketplace-templates.js`, `template-detail.js`

### Priority 2: Trust (Weeks 5-8) — ⚠️ 70% COMPLETE
- [x] Uptime tracking system — `NodeReputationService.cs` (30-day rolling, 15s precision)
- [x] User review system (post-VM termination) — `ReviewService.cs` (backend only)
- [ ] Slashing mechanism — Deferred: no evidence of spam/quality issues yet
- [ ] Weighted trust score calculation — Only raw uptime % exists
- [ ] Frontend: Reputation dashboard — Not started
- [ ] Frontend: Trust badges — Not started
- [ ] Frontend: Review prompt after VM deletion — Not started
- [ ] Frontend: Node rating aggregates display — Not started

### Priority 3: Collaboration (Weeks 9-12) — ❌ NOT STARTED
- [ ] Shared VMs (multi-wallet access)
- [ ] Workspace system
- [ ] Infrastructure templates (multi-VM)
- [ ] Frontend: Team collaboration UI

### Priority 4: Discovery (Weeks 13-16) — ❌ NOT STARTED
- [ ] Network topology API
- [ ] Live map visualization (D3.js/Three.js)
- [ ] Node explorer (text-based detail exists in marketplace; no map)
- [ ] Resource heatmap

### Priority 5: Economics (Weeks 17-20) — ⚠️ 15% COMPLETE
- [ ] Dynamic pricing engine — Only static floor + operator pricing exists
- [ ] Relay revenue sharing — Documented but not coded
- [ ] Node staking system — Deferred until XDE token launch + 50 nodes + 3 months data
- [x] Settlement automation — `DeCloudEscrow.sol` + `OnChainSettlementService.cs`

---

## 🚀 QUICK WINS (Week 1) — ✅ ALL COMPLETE

All quick wins from the original vision have been implemented:

### 1. Node Tags & Filtering ✅
Implemented in `NodeMarketplaceService.cs`. Nodes have `Tags`, `Description`, and `BasePrice` fields. Multi-criteria search supports tags, region, GPU, price range, and uptime filtering.

### 2. Featured Templates ✅
6 production-ready seed templates via `TemplateSeederService.cs` with semantic versioning. Featured templates endpoint returns top-rated templates. Exceeded the vision's "hardcode 5" by building a full dynamic marketplace.

### 3. Simple Uptime Tracking ✅
`NodeReputationService.cs` tracks uptime via failed heartbeat detection (more accurate than the proposed heartbeat-counting approach). 30-day rolling window with auto-cleanup. Integrated with marketplace search filtering.

---

## 🎯 SUCCESS METRICS

### Network Health
- **Node count**: Target 1,000 nodes in 6 months
- **Geographic distribution**: Nodes in 20+ countries
- **Specialization**: 20% of nodes offering specialized hardware (GPU, high-memory, etc.)

### User Engagement
- **Template downloads**: 10,000 downloads/month
- **User-created templates**: 500+ templates in marketplace
- **Collaboration**: 30% of VMs are shared across multiple wallets

### Economic Activity
- **Total compute hours**: 1M hours/month
- **Average node earnings**: $200/month
- **Template creators earnings**: $50/month average for top 100 templates

### Community
- **GitHub contributors**: 50+ contributors
- **Discord members**: 5,000+ members
- **User-generated content**: 100+ blog posts, tutorials, infrastructure patterns

---

## 🌟 THE VISION IN ACTION

### Year 1: Foundation
- 1,000 nodes, 50 countries
- 10,000 active users
- 500 community templates
- Basic marketplace, reputation system

### Year 2: Emergence
- 10,000 nodes, specialized operators
- 100,000 users
- Complex multi-VM deployments common
- Advanced templates (AI training clusters, HA setups)
- Community-run relays

### Year 3: Ecosystem
- 50,000 nodes
- 1M users
- Secondary markets (template trading, node leasing)
- Integration with DeFi (staking derivatives, insurance pools)
- Minecraft-level network effects

---

## 💡 THE KEY INSIGHT

**Minecraft succeeded not because it was technically impressive, but because it enabled CREATION and COMMUNITY.**

Your relay architecture is already more sophisticated than anything in Minecraft. Now you need to:

1. **Make creation easy** (templates, one-click deploys)
2. **Make discovery possible** (marketplace, search, recommendations)
3. **Make trust visible** (reputation, reviews, transparency)
4. **Reward community content** (template marketplace, revenue sharing)
5. **Enable collaboration** (shared VMs, workspaces, teams)

**Do these things, and DeCloud becomes unstoppable.**

---

## 🏗️ BUILT BEYOND THE ORIGINAL VISION

These significant features were implemented but were NOT in the original Minecraft Vision roadmap. They strengthen the foundation and reflect organic platform evolution:

| Feature | Status | Description |
|---|---|---|
| **Relay Infrastructure** | Production | Auto-deployed WireGuard relay VMs for CGNAT bypass (~60-80% of nodes) |
| **DHT Infrastructure** | Production | libp2p DHT nodes over WireGuard mesh (2 nodes live, bootstrap polling) |
| **SystemVm Reconciliation** | Production | Kubernetes-style controller managing relay, DHT, and ingress system VMs |
| **VM Lifecycle State Machine** | Production | `VmLifecycleManager.cs` — centralized transitions, side effects, crash-safe |
| **Bandwidth QoS Tiers** | Production | 4-tier bandwidth limiting enforced at libvirt level (Basic/Standard/Performance/Unmetered) |
| **Hybrid Pricing (Model C)** | Production | Platform floor rates + operator custom pricing with server-side enforcement |
| **Smart Port Allocation** | Production | 3-hop DNAT forwarding for CGNAT, direct for public nodes, 40000-65535 range |
| **CentralIngress-Aware Ports** | Production | HTTP/WS ports skip iptables (handled by Caddy), only TCP/UDP gets DNAT |
| **Per-Service VM Readiness** | Production | qemu-guest-agent probing (CloudInitDone, TcpPort, HttpGet, ExecCommand) |
| **Web Proxy Browser (Ultraviolet)** | Production | Privacy browsing template with Cross-Origin Isolation fix |
| **ARM Architecture Support** | Production | Raspberry Pi support with dual-arch domain XML |
| **CLI Tool** | Production | `decloud` v1.3.0 — wallet auth, VM management, diagnostics |
| **USDC Escrow Contract** | Production | `DeCloudEscrow.sol` on Polygon — deposits, settlement, payouts |

---

## 🚦 NEXT IMMEDIATE STEPS (Updated 2026-02-17)

**Current Focus: Phase 2 — User Engagement & Retention**

**This Sprint:**
1. ~~Add node tags and description fields~~ ✅ Done
2. ~~Create featured templates~~ ✅ Done (6 seed templates)
3. ~~Build node marketplace UI~~ ✅ Done
4. Node operator dashboard (Priority 2.1) — earnings, uptime, relay stats
5. Frontend trust badges ("99.9% uptime", "100+ VMs hosted")
6. Review prompt modal after VM termination

**This Month:**
1. ~~Implement full template marketplace~~ ✅ Done
2. ~~Add basic reputation tracking~~ ✅ Done
3. Grow seed templates to 10-15 (then community to 50+)
4. Wire up node rating aggregates in marketplace display
5. Weighted trust score algorithm

**This Quarter:**
1. ~~Ship all Priority 1 features~~ ✅ Done
2. Complete remaining Priority 2 items (frontend polish, trust score)
3. Onboard 50+ active node operators
4. Evaluate Phase 3 (collaboration) based on user demand

**Deferred (awaiting prerequisites):**
- Staking/slashing: Needs XDE token launch, 50+ nodes, 3+ months reputation data
- Relay revenue sharing: Needs billing infrastructure maturity
- Dynamic pricing: Needs market data from operational marketplace

**This is how you build the Minecraft of compute.**
