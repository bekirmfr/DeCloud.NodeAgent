// NEW FILE — src/Orchestrator/Controllers/CsamReportController.cs
// Phase 6 pass 1, build item 4 (handout §6). Node-facing CSAM match intake.
//
// ROUTE NOTE (deliberate deviation from the handout's "/api/admin/csam-report"):
// this endpoint is authenticated with the NODE role, not Admin — putting it
// under /api/admin would either mis-scope it to [Authorize(Roles="Admin")]
// (nodes could never call it) or put a node-auth endpoint on an admin route
// (confusing, and admin routes are where humans look for human actions).
// It lives at POST /api/compliance/csam-report with [Authorize(Roles="node")],
// matching how nodes are actually authenticated (NodeService issues node JWTs
// with ClaimTypes.Role = "node" and a "node_id" claim).
//
// TWO INTEGRATION POINTS TO VERIFY against the live Phase 4 code (which was
// built 2026-07-08 and is not in the repo snapshot this was grounded on):
//   [VERIFY-1] The abuse-report intake service: the build log describes a
//              self-contained store service ("mirroring WalletBlocklistService")
//              behind AbuseController. Wire `_abuse.FileReportAsync(...)` to
//              its actual create method — the one POST /api/abuse uses — so
//              the report gets a real ABU-YYYY-NNNNN reference and the
//              deterministic csam→P0/2h triage. Do NOT call the public HTTP
//              endpoint (it is anonymous-intake shaped and per-IP rate-limited).
//   [VERIFY-2] IEnforcementService.SuspendVmAsync gained an optional
//              `reference` parameter in Phase 4 ("reportReference rides
//              EnforcementAction.Reference"). If the live signature differs,
//              adjust the call below accordingly.

using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Orchestrator.Interfaces;
using Orchestrator.Persistence;

namespace Orchestrator.Controllers;

/// <summary>
/// Node-authenticated CSAM match intake (Phase 6, first pass).
///
/// A node's scanner found a known-CSAM hash match on a tenant VM's frozen
/// snapshot. This endpoint does exactly two things, both through EXISTING
/// seams — no parallel review surface, no parallel enforcement path:
///
///   1. Files the match into the Phase 4 abuse queue (category "csam",
///      deterministic P0 / 2h SLA). Human review, NCMEC reporting, and any
///      wallet blacklisting happen ONLY from that queue, by a human — no
///      automated path here does anything irreversible.
///   2. Applies the protective, reversible containment: SuspendVmAsync sets
///      the ComplianceHold (force-stop, owner cannot restart, overlay
///      preserved, deletion refused). Fenced to the VM's authoritative host —
///      a hostile or compromised node must not be able to suspend arbitrary
///      VMs by forging reports; a non-host report is still filed to the P0
///      queue for the human to judge.
/// </summary>
[ApiController]
[Route("api/compliance")]
[Authorize(Roles = "node")]
public class CsamReportController : ControllerBase
{
    private readonly IAbuseReportService _abuse;          // [VERIFY-1] actual Phase 4 service type
    private readonly IEnforcementService _enforcement;
    private readonly DataStore _dataStore;
    private readonly ILogger<CsamReportController> _logger;

    public CsamReportController(
        IAbuseReportService abuse,
        IEnforcementService enforcement,
        DataStore dataStore,
        ILogger<CsamReportController> logger)
    {
        _abuse = abuse;
        _enforcement = enforcement;
        _dataStore = dataStore;
        _logger = logger;
    }

    [HttpPost("csam-report")]
    public async Task<IActionResult> Report([FromBody] CsamMatchReportRequest req, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.VmId) || string.IsNullOrWhiteSpace(req.MatchedFileHash))
            return BadRequest(new { error = "vmId and matchedFileHash are required" });

        // Node identity comes from the TOKEN, never the body — a node cannot
        // speak for another node.
        var nodeId = User.FindFirstValue("node_id");
        if (string.IsNullOrEmpty(nodeId))
            return Forbid();

        var vm = await _dataStore.GetVmAsync(req.VmId);
        if (vm == null)
        {
            _logger.LogWarning("csam-report for unknown VM {VmId} from node {NodeId}", req.VmId, nodeId);
            return NotFound(new { error = "VM not found" });
        }

        // ── 1. File into the EXISTING Phase 4 queue (csam → P0/2h) ────────────
        string? reference = null;
        try
        {
            reference = await _abuse.FileReportAsync(          // [VERIFY-1]
                category: "csam",
                targetType: "vm",
                targetId: req.VmId,
                description:
                    $"Node-level CSAM scanner reported a known-hash match on VM {req.VmId}. " +
                    $"hash={req.MatchedFileHash}, db={req.DbSource ?? "unknown"}, " +
                    $"detectedAt={req.DetectedAt:O}, reportingNode={nodeId}. " +
                    "Automated protective suspend applied (if reporter is the authoritative host); " +
                    "human confirmation required before NCMEC report or wallet blacklist.",
                reporter: $"node:{nodeId}",
                ct: ct);
        }
        catch (Exception ex)
        {
            // Containment must not depend on queue availability — fall through
            // to the suspend and signal the node to retry the report.
            _logger.LogError(ex, "csam-report: failed to file abuse-queue item for VM {VmId}", req.VmId);
        }

        // ── 2. Protective suspend — authoritative host only ───────────────────
        var suspended = false;
        if (string.Equals(vm.NodeId, nodeId, StringComparison.OrdinalIgnoreCase))
        {
            if (vm.ComplianceHold)
            {
                suspended = true; // already held — idempotent, no duplicate audit row
            }
            else
            {
                var result = await _enforcement.SuspendVmAsync(
                    req.VmId,
                    reason: "CSAM scanner hash match — protective hold pending human review",
                    actor: $"node:{nodeId}",
                    reference: reference,                       // [VERIFY-2]
                    ct: ct);
                suspended = result.Success;
                if (!result.Success)
                    _logger.LogError(
                        "csam-report: protective suspend FAILED for VM {VmId}: {Error} — " +
                        "P0 queue item {Ref} is the remaining control (2h SLA)",
                        req.VmId, result.Error, reference ?? "(none)");
            }
        }
        else
        {
            _logger.LogWarning(
                "csam-report for VM {VmId} from NON-authoritative node {NodeId} (host={Host}) — " +
                "report filed, auto-suspend refused (forged-report fence)",
                req.VmId, nodeId, vm.NodeId);
        }

        // The durable, load-bearing guarantee is the queue item (human loop
        // engaged, P0/2h). If it could not be filed, tell the node to retry;
        // the node keeps its push blocked either way.
        if (reference == null)
            return StatusCode(500, new { error = "Report could not be filed — retry", suspended });

        return Ok(new { reference, suspended });
    }
}

/// <summary>Wire shape of a node's CSAM match report. Hashes only — never content.</summary>
public record CsamMatchReportRequest(
    string VmId,
    string MatchedFileHash,
    string? DbSource,
    DateTime DetectedAt);
