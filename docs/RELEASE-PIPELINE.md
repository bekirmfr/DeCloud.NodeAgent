# Node Agent Release Pipeline

Reference for the CI/CD pipeline that builds, signs, and publishes the
DeCloud Node Agent. Lives in `.github/workflows/release.yml` in the
`bekirmfr/DeCloud.NodeAgent` repository.

> **Scope.** This document covers the NodeAgent release pipeline only.
> System-VM service binaries (`dht-node`, `blockstore-node`) are released
> by a separate pipeline in `bekirmfr/DeCloud.Builds`. The orchestrator
> consumes those via `TemplateArtifact.SourceUrl + Sha256` and the
> `IArtifactCacheService` runtime mechanism — install.sh on nodes never
> sees them.

---

## Table of Contents

1. [Design intent](#design-intent)
2. [What the pipeline produces](#what-the-pipeline-produces)
3. [Trust model](#trust-model)
4. [Workflow jobs](#workflow-jobs)
5. [Cutting a release](#cutting-a-release)
6. [RC versus stable](#rc-versus-stable)
7. [Verification (what consumers do)](#verification-what-consumers-do)
8. [Operational runbook](#operational-runbook)
9. [What is intentionally NOT here](#what-is-intentionally-not-here)

---

## Design intent

Each repository's release pipeline is responsible for artifacts built
**from that repository's source**, signed by **that repository's workflow
identity**. This keeps the cosign keyless provenance chain tight: a
signature attests to "produced by `<repo>/.github/workflows/release.yml`
at commit X", which is a tighter assertion than "produced by some central
build system that pulled this source from somewhere."

The pipeline is triggered only by **signed git tags** matching `v*.*.*`.
Tag signature is verified via the GitHub API before any build job runs
— see the `verify-tag` job. Operators with push access to the repo are
the trust root; nobody else can produce a valid release.

Releases are immutable once published. To fix a bad release, cut a new
patch version. Never delete and re-publish a tag.

---

## What the pipeline produces

A signed git tag pushes to `origin` triggers the workflow, which
publishes a GitHub Release with the following assets:

| Asset | Purpose |
| --- | --- |
| `decloud-node-agent-vX.Y.Z-linux-amd64.tar.gz` | ASP.NET Core publish output (framework-dependent), x86_64 |
| `decloud-node-agent-vX.Y.Z-linux-arm64.tar.gz` | Same, aarch64 |
| `decloud-cli-vX.Y.Z.tar.gz` | Shell scripts: `cli/decloud`, `cli/cli-decloud-node`, `decloud-relay-nat`, `vm-cleanup.sh`, `install.sh`, `uninstall.sh` |
| `decloud-gpu-shim-vX.Y.Z-linux-amd64.tar.gz` | Pre-built CUDA shims/stubs (compat-built against glibc 2.31, x86_64 only) |
| `install.sh` | Standalone bootstrap script — fetched by `curl \| bash` and by `decloud update` |
| `uninstall.sh` | Standalone uninstall script — fetched by `decloud uninstall` |
| `manifest.json` | List of all `*.tar.gz` artifacts plus their SHA-256 hashes, version, git SHA, build timestamp |
| `manifest.json.sig` | Cosign keyless signature over `manifest.json` |
| `manifest.json.pem` | Cosign signing certificate (Sigstore Fulcio) |
| `sbom.cdx.json` | CycloneDX Software Bill of Materials for the .NET app |

**Not produced here:**
- `gpu-proxy-daemon` (the host-side daemon; links against host CUDA
  version, must be built locally on each operator's machine)
- `dht-node`, `blockstore-node` Go binaries (DeCloud.Builds pipeline)
- `DeCloud.Shared.dll` as a separate artifact (consumed transitively at
  build time and embedded in the NodeAgent publish output)

---

## Trust model

Three layered claims, each with a different cryptographic anchor:

```
Operator
   │  TLS to github.com
   ▼
install.sh  (bootstrap, served from GitHub Releases)
   │  cosign keyless verify
   ▼
manifest.json  (signed by release.yml workflow OIDC identity)
   │  SHA-256 lookup
   ▼
*.tar.gz, *.so  (each artifact's hash matches manifest)
   │  extract
   ▼
running NodeAgent on disk
```

### TLS bootstrap

`install.sh` is fetched over HTTPS from GitHub. Trust at this layer is
GitHub's TLS certificate plus operator awareness that the URL points at
the correct repo. install.sh is small and auditable; operators who care
can `curl -o install.sh ...` first and `less` it before running.

### Cosign keyless on manifest

The `release` job signs `manifest.json` using cosign keyless via GitHub
OIDC. The certificate's `Subject Alternative Name` carries the workflow
identity:

```
https://github.com/bekirmfr/DeCloud.NodeAgent/.github/workflows/release.yml@refs/tags/vX.Y.Z
```

install.sh verifies this with `cosign verify-blob`, requiring:

```bash
--certificate-identity-regexp \
    '^https://github\.com/bekirmfr/DeCloud\.NodeAgent/\.github/workflows/release\.yml@'
--certificate-oidc-issuer 'https://token.actions.githubusercontent.com'
```

Anyone with push access to the NodeAgent repo can produce this
signature; nobody else can. Note that this does **not** attest that the
content is benign — only that it was produced by this workflow at this
commit. The integrity of what gets signed depends entirely on the
maintainer process: signed commits, branch protection, and code review.

### SHA-256 on individual artifacts

Once the manifest is verified, every `*.tar.gz` is downloaded and its
SHA-256 is checked against the `artifacts.<name>.sha256` field in the
verified manifest. A single-byte mismatch aborts the install.

`install.sh` and `uninstall.sh` are NOT in the manifest. Reason:
including them would create a verification loop (install.sh would have
to verify itself). They are TLS-trusted at the bootstrap step;
everything they fetch downstream is cryptographically verified.

---

## Workflow jobs

```
┌─────────────────────┐
│   verify-tag        │  Gate: tag signature verified via GitHub API
└──────────┬──────────┘
           │
   ┌───────┼─────────┬──────────────┐
   ▼       ▼         ▼              ▼
┌──────┐┌──────┐┌─────────┐┌─────────────┐
│ amd64││ arm64││  cli    ││  gpu-shims  │
│agent ││agent ││ bundle  ││  (amd64)    │
└──┬───┘└──┬───┘└────┬────┘└──────┬──────┘
   │       │         │            │
   └───────┴─────────┴────────────┘
                     │
                     ▼
           ┌──────────────────┐
           │     release      │
           │ - manifest.json  │
           │ - cosign sign    │
           │ - attest         │
           │ - publish to     │
           │   GitHub Release │
           └──────────────────┘
```

| Job | Purpose | Failure mode |
| --- | --- | --- |
| `verify-tag` | Confirms `verification.verified == true` via GitHub API. Refuses unsigned tags. | Build aborts before any artifact is built. Operator must sign the tag and re-push. |
| `build-nodeagent (linux-x64, linux-arm64)` | Clones `DeCloud.Shared` as sibling source (build-time dep), runs `dotnet publish` framework-dependent, packages tarball, generates SBOM (amd64 only). | Most likely cause: csproj reference broken or Shared incompatible. Inspect the publish step output. |
| `build-cli-bundle` | Verifies expected files exist, packages CLI scripts and `install.sh`/`uninstall.sh` into the tarball, also uploads `install.sh` and `uninstall.sh` as standalone artifacts. CRLF precheck rejects Windows-edited shell files. | Missing required file → `test -f` fails fast. CRLF detected → fails fast. |
| `build-gpu-shims` | Runs `make -C src/gpu-proxy all-shims-compat` inside an Ubuntu 20.04 container for glibc 2.31 universal compat. Skips if `src/gpu-proxy/` is absent. | Make error → log shows missing target or compile fail. |
| `release` | Aggregates all artifacts, generates `manifest.json` with SHA-256 of each, signs with cosign keyless, attests provenance, publishes GitHub Release. | OIDC token not granted (org policy) → cosign sign fails. Otherwise rare. |

Workflow file: [`.github/workflows/release.yml`](../.github/workflows/release.yml)

---

## Cutting a release

Prerequisites: SSH or GPG signing key registered on GitHub as a
**Signing Key** (Settings → SSH and GPG keys → New SSH key → Key type:
Signing Key). Local git config:

```bash
git config --global gpg.format ssh
git config --global user.signingkey "$HOME/.ssh/id_ed25519.pub"
git config --global tag.gpgsign true
git config --global commit.gpgsign true
```

Procedure:

```bash
# 1. Ensure master is up to date
git checkout master
git pull

# 2. Sign and push the tag
git tag -s vX.Y.Z -m "Release vX.Y.Z — short description"
git tag -v vX.Y.Z          # local sanity check
git push origin vX.Y.Z

# 3. Watch the workflow run
# https://github.com/bekirmfr/DeCloud.NodeAgent/actions
```

Roughly 5-8 minutes from push to release published. Watch the
`build-nodeagent (linux-arm64)` job in particular — it's the slowest
(QEMU emulation on x86 runners).

After the workflow completes, **verify the release manually before
trusting it**:

```bash
VERSION=vX.Y.Z
BASE=https://github.com/bekirmfr/DeCloud.NodeAgent/releases/download/${VERSION}

cd $(mktemp -d)
curl -fsSLO ${BASE}/manifest.json
curl -fsSLO ${BASE}/manifest.json.sig
curl -fsSLO ${BASE}/manifest.json.pem

cosign verify-blob \
  --certificate-identity-regexp \
    '^https://github\.com/bekirmfr/DeCloud\.NodeAgent/\.github/workflows/release\.yml@' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  --signature   manifest.json.sig \
  --certificate manifest.json.pem \
  manifest.json
# Expected: "Verified OK"

# Spot-check one artifact
ARTIFACT=decloud-cli-${VERSION}.tar.gz
curl -fsSLO ${BASE}/${ARTIFACT}
[ "$(jq -r --arg n "$ARTIFACT" '.artifacts[$n].sha256' manifest.json)" \
   = "$(sha256sum "$ARTIFACT" | awk '{print $1}')" ] \
  && echo "OK" || echo "MISMATCH"
```

If anything fails, **do not promote**. Investigate the failure, push a
fix, retag (using `vX.Y.Z+1`, never the same tag).

---

## RC versus stable

GitHub auto-classifies tags by name:

| Tag pattern | GitHub classification | Used by `releases/latest`? |
| --- | --- | --- |
| `vX.Y.Z` | Release | Yes |
| `vX.Y.Z-rc1`, `-beta`, `-alpha`, etc. | Pre-release | No (excluded) |

Implications for download URLs:

- **`releases/latest/download/install.sh`** redirects to the most
  recent stable release. Returns 404 if no stable release exists.
- **`releases/download/<tag>/install.sh`** always works for any
  specific tag, stable or pre-release.

`install.sh`'s `resolve_release_version` handles this with a two-step
strategy: first try `/releases/latest`, fall back to
`/releases?per_page=1` (which includes pre-releases). When the fallback
fires, a clear warning is logged.

**Promotion procedure** (RC → stable):

1. Cut and validate `vX.Y.Z-rc1`. Test on at least one node end-to-end:
   curl-pipe install, `decloud login`, `decloud status`, `decloud update`,
   `decloud uninstall`. Watch logs through `journalctl -u
   decloud-node-agent -f`.
2. If issues found, fix, push to master, retag a new RC (`-rc2`).
3. Once an RC passes, cut `vX.Y.Z` (no suffix). Same workflow runs.
4. The public landing page's `releases/latest/download/install.sh` URL
   now resolves to the new stable release.

Pre-releases stay in the release history forever. Don't delete them
even after promotion; they're a record of what was tested.

---

## Verification (what consumers do)

The flow is automated by `install.sh`. Documented here for understanding
and for operators who want to verify manually.

```
1. fetch manifest.json + .sig + .pem from release URL
2. cosign verify-blob with identity-regex pinned to this repo's workflow
   ↓ on success: trust manifest contents
3. for each artifact in manifest:
     - download from release URL
     - sha256sum
     - compare to manifest.<name>.sha256
   ↓ on any mismatch: abort install
4. extract verified tarballs to versioned directories
5. atomic symlink swap to make new version active
6. (unrelated: configure node, start service)
```

---

## Operational runbook

### A release fails partway through CI

The workflow stops at the first failed job. No release is published.
Diagnose from the Actions log, push a fix to master, then **delete the
old tag and re-create it on the new commit**:

```bash
git push --delete origin vX.Y.Z-rc1
git tag -d vX.Y.Z-rc1
git pull
git tag -s vX.Y.Z-rc1 -m "..."
git push origin vX.Y.Z-rc1
```

This is acceptable for **pre-release tags** that nobody has installed
yet. For published stable tags, never reuse — bump the patch number.

### A bad release was published and operators have installed it

The release is immutable; operators have already verified its
signature and pulled artifacts. Fix forward:

1. Push a fix to master.
2. Cut `vX.Y.Z+1` with a signed tag.
3. Operators run `decloud update` to pick it up.
4. Optionally mark the bad release on GitHub as "this release should not
   be used" via the release page's edit UI. This does not retract it
   cryptographically — only the visual indicator changes.

### The OIDC keyless signing fails

Most likely cause: the GitHub org has restricted OIDC token issuance.
Workflow log will show a token error in the `Sign manifest` step. The
workflow declares `id-token: write` correctly; the restriction is at
the org level. Resolve via org admin or fall back to cosign with a
managed key (separate setup, not currently configured).

### Cosign version pinned in workflow gets old

Pinned at `v2.4.1` via `sigstore/cosign-installer@v3` action input. To
bump:

1. Test the new version locally (`cosign verify-blob` against an
   existing release).
2. Update `cosign-release` field in `release.yml`.
3. Update `COSIGN_VERSION` in `install.sh` so node-side install fetches
   the matching version.
4. Cut a new release; both ends of the chain use the bumped version.

### Operator's tag signature stops verifying

GitHub's API check (`verification.verified`) depends on the signing
key being registered on the tagger's GitHub account as a
**Signing Key**. If a maintainer rotates their key, register the new
public key on GitHub before pushing the next signed tag.

---

## What is intentionally NOT here

- **DeCloud.Shared as a NuGet package**: today, `build-nodeagent`
  clones DeCloud.Shared as a sibling at master HEAD. This means two CI
  runs of the same NodeAgent tag may pick up different Shared
  revisions, breaking strict reproducibility. The clean fix is to
  publish DeCloud.Shared as a signed NuGet package and use
  `<PackageReference>` with a version pin. Tracked as separate work.
- **Daemon binary in releases**: `gpu-proxy-daemon` requires CUDA at
  the host's exact version. A single prebuilt binary is unsafe across
  CUDA versions. Operators with proxy-mode GPUs build the daemon
  locally per `docs/gpu-proxy-daemon.md`.
- **System-VM Go binaries**: `dht-node` and `blockstore-node` source
  lives in DeCloud.Builds with its own pipeline; the orchestrator
  consumes them via `TemplateArtifact`.
- **Self-hosted artifact mirror**: today operators fetch directly from
  `github.com`. If GitHub becomes a bottleneck or you need air-gapped
  installs, set up a CDN mirror and document the
  `--release-base-url` flag override.

---

## Files in this pipeline

| File | Owner |
| --- | --- |
| `.github/workflows/release.yml` | This pipeline definition |
| `install.sh` | Top-level bootstrap and release fetch logic |
| `uninstall.sh` | Top-level cleanup logic |
| `cli/decloud` | Operator CLI (cmd_update, cmd_remove use `fetch_release_script`) |
| `src/DeCloud.NodeAgent/DeCloud.NodeAgent.csproj` | Built by `build-nodeagent` |
| `src/gpu-proxy/Makefile` | Built by `build-gpu-shims` |
| `.gitattributes` | Enforces LF line endings on shell files |
