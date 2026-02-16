# CLAUDE.md — DeCloud Node Agent

This file captures codebase conventions, architecture decisions, and development guidance for AI-assisted development.

---

## Project Overview

**DeCloud Node Agent** is a cross-platform .NET 8 agent that runs on hardware provider nodes in a decentralized cloud computing platform. It manages VM lifecycle, resource discovery, and communication with the orchestration layer.

- **Primary language**: C# (.NET 8 / ASP.NET Core)
- **Secondary languages**: Go (system VMs, attestation), Bash (deployment scripts), Python (VM dashboards)
- **Production target**: Linux (Ubuntu 22.04+) with KVM/QEMU/libvirt
- **Dev-friendly**: Windows (API & resource discovery only)

---

## Solution Structure

```
DeCloud.NodeAgent.sln
├── src/DeCloud.NodeAgent/              # ASP.NET Core web API (entry point)
│   ├── Controllers/                    # REST API endpoints (12 controllers)
│   ├── Services/                       # Background hosted services
│   ├── CloudInit/Templates/            # Cloud-init YAML + embedded Go/Python
│   ├── Program.cs                      # Entry point, DI registration
│   └── appsettings.json               # Configuration
├── src/DeCloud.NodeAgent.Core/         # Interfaces, models, constants
│   └── Interfaces/IServices.cs         # Core service contracts
├── src/DeCloud.NodeAgent.Infrastructure/ # Service implementations
│   ├── Services/                       # Business logic
│   ├── Libvirt/                        # VM management via virsh
│   ├── Network/                        # WireGuard overlay networking
│   └── Persistence/                    # SQLite repositories
├── src/Shared/                         # Shared models (linked into projects)
├── attestation-agent/                  # Go: Ephemeral key attestation
└── cli/                                # Bash: CLI management tools
```

## Build & Run

```bash
# .NET solution
dotnet restore
dotnet build DeCloud.NodeAgent.sln
dotnet run --project src/DeCloud.NodeAgent

# Go attestation agent
cd attestation-agent && ./build.sh all

# Go DHT node (built automatically during .NET publish)
cd src/DeCloud.NodeAgent/CloudInit/Templates/dht-vm/dht-node-src && ./build.sh
```

The project targets `net8.0`. All three C# projects share nullable reference types and implicit usings enabled.

## Architecture Patterns

- **Dependency Injection**: Constructor-based DI via `Microsoft.Extensions.DependencyInjection` (configured in `Program.cs`)
- **Repository Pattern**: `VmRepository` and `PortMappingRepository` (SQLite)
- **Background Services**: `IHostedService` implementations for heartbeat, command polling, health monitoring, port reconciliation
- **Named HttpClients**: `VmProxy` for proxying to VMs, typed clients for orchestrator
- **IOptions<T>**: Settings classes bound from `appsettings.json` sections
- **Cloud-Init Templates**: Parameterized YAML files rendered by `CloudInitTemplateService`

## Key Conventions

### Naming
- Controllers: `{Feature}Controller` under `DeCloud.NodeAgent.Controllers`
- Services: `{Feature}Service` under `DeCloud.NodeAgent.Infrastructure.Services`
- Interfaces: `I{Feature}` under `DeCloud.NodeAgent.Core.Interfaces`
- Settings: `{Feature}Options` under `DeCloud.NodeAgent.Core.Settings`
- Models: Under `DeCloud.NodeAgent.Core.Models` or `Orchestrator.Models` (shared)

### Coding Style
- File-scoped namespaces (`namespace Foo;` not `namespace Foo { }`)
- Primary constructors not used (standard constructor DI)
- Async/await throughout with `CancellationToken` propagation
- `ILogger<T>` injected in every class
- camelCase JSON serialization (`JsonNamingPolicy.CamelCase`)

### API Design
- Route convention: `api/[controller]` (attribute routing)
- All controllers inherit `ControllerBase`
- Return `ActionResult<T>` for typed responses
- WebSocket endpoints for SSH/SFTP proxying

### Cloud-Init Templates
- Located in `CloudInit/Templates/`
- VM types: `general-vm`, `relay-vm`, `dht-vm`, `inference-vm` (blockstore-vm planned)
- Go binaries embedded as base64 in YAML templates
- Shared component: `wg-mesh-enroll.sh` for WireGuard mesh enrollment
- Templates copied to output via `<CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>`

## Configuration

Settings are in `src/DeCloud.NodeAgent/appsettings.json`. Key sections:
- `Libvirt`: VM storage paths, libvirt URI
- `WireGuard`: Network overlay config
- `Heartbeat`: Orchestrator heartbeat interval & URL
- `OrchestratorClient`: Orchestrator connection
- `PortSecurity`: Blocked/allowed ports
- `ProxySettings`: Generic proxy configuration
- `AuditLog`: Security audit logging

## Dependencies

### .NET (NuGet)
- `Microsoft.Data.Sqlite` 10.0.0 — SQLite persistence
- `Nethereum.Web3` / `Nethereum.Signer` 5.0.0 — Ethereum wallet auth
- `SSH.NET` 2025.1.0 — SSH/SFTP client for VM proxying
- `Swashbuckle.AspNetCore` 6.5.0 — Swagger/OpenAPI
- `Microsoft.Extensions.Hosting` 8.0.0 — Background services

### Go
- Attestation agent: Go 1.21+, standard library only
- DHT node: Go 1.23+, libp2p, IPFS libraries

### System (Linux production)
- KVM/QEMU/libvirt, WireGuard, cloud-image-utils, genisoimage

## Important Notes

- **No test project exists yet** — the README references `tests/` but it's not implemented
- **No CI/CD pipeline** — no GitHub Actions workflows configured
- **No Dockerfile** — container builds not set up
- **Swagger is always enabled** — not gated behind `IsDevelopment()`
- **API has no authentication middleware** — relies on network-level protection (localhost/firewall)
- **Kestrel listens on `0.0.0.0:5100`** (HTTP, all interfaces)
- `VmManagerInitializationService` is defined inline in `Program.cs` (bottom of file)
- Shared models are linked via `<Compile Include="../Shared/**/*.cs" />` (not a project reference)
