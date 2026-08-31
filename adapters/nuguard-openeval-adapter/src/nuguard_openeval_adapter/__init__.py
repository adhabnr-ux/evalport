"""nuguard <-> EvalPort adapter.

Standalone converter between nuguard's `Finding` / `ValidateRunResult`
models (https://github.com/NuGuardAI/nuguard) and the EvalPort interchange
format (https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside nuguard
itself: this follows the same playbook that already worked for CrewAI/AutoGen
(https://github.com/adhabnr-ux/evalport/tree/main/adapters/crewai-openeval-adapter):
it works against nuguard's public `Finding`/`ValidateRunResult` shapes
(pydantic model instances, `.model_dump()` dicts, or plain dicts) from the
outside, so you get EvalPort export today without needing anything merged
into nuguard's core.

Grounded in nuguard's real, current source (not just its README):

- `nuguard/models/finding.py` -- the `Finding` pydantic model. `severity` is
  a `Severity` str-enum (critical/high/medium/low/info); `verified` is
  `bool | None` (None = not run, True = reproduced, False = unconfirmed);
  `ngrs_score` is the underlying 0-100 risk score `severity` is banded from.
- `nuguard/models/validate.py` -- `ValidateRunResult.findings` is typed
  `list[dict]` (serialized `Finding` dicts), **not** `list[Finding]` -- see
  the correction posted to NuGuardAI/nuguard#355. `ValidateRunResult` also
  carries `capability_map` (tool/agent coverage) and `policy_records`
  (per-turn `TurnPolicyRecord`s) that have no clean 1:1 field in EvalPort's
  `ResultSet` today; both are carried through as `ResultSet.metadata.nuguard`
  rather than force-fit into `Result`/`GraderResult`.
- `nuguard/validate/runner.py` -- confirms the real `scan_outcome` vocabulary
  produced for validate runs: `"no_findings"`, `"findings"`,
  `"high_findings"`, `"critical_findings"` (this is the validate-mode
  vocabulary specifically; nuguard's other run modes use a larger, overlapping
  set including `"aborted_target_unavailable"` etc. that validate runs never
  emit) -- and that every `Finding.goal_type` on a validate-run finding is one
  of `ValidateFindingType`'s four values (`CAPABILITY_GAP`,
  `CAPABILITY_REGRESSION`, `POLICY_VIOLATION`, `BOUNDARY_FAILURE`).

See SPEC.md (next to this file) for the full problem/solution/mapping writeup.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "finding_to_result", "__version__"]
__version__ = "0.1.0"


# severity -> OpenEval [0.0, 1.0] score axis. 0.0 = worst (critical), 1.0 =
# best (info-only). This is the inverse orientation of nuguard's own NGRS
# (higher NGRS = worse), matching EvalPort's convention that a higher score
# means "closer to passing".
_SEVERITY_SCORE = {
    "critical": 0.0,
    "high": 0.25,
    "medium": 0.5,
    "low": 0.75,
    "info": 1.0,
}

# scan_outcome values validate runs actually produce (nuguard/validate/runner.py).
_VALIDATE_SCAN_OUTCOMES = frozenset({"no_findings", "findings", "high_findings", "critical_findings"})


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    A `Finding`/`ValidateRunResult` may arrive as a live pydantic model
    instance, a `.model_dump()` dict (what `ValidateRunResult.findings`
    actually holds per `nuguard/models/validate.py`), or hand-built plain
    dict/object test fixtures -- every accessor in this module goes through
    here rather than assuming one shape.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _severity_str(severity: Any) -> str:
    """Normalize a `Severity` enum member, its `.value`, or a plain string to lowercase text."""
    if severity is None:
        return "info"
    value = getattr(severity, "value", severity)
    return str(value).lower()


def _severity_score(severity: Any, ngrs_score: Any = None) -> float:
    """Map severity (falling back to `ngrs_score`/100 when severity is unrecognized) to [0.0, 1.0]."""
    sev = _severity_str(severity)
    if sev in _SEVERITY_SCORE:
        return _SEVERITY_SCORE[sev]
    if isinstance(ngrs_score, (int, float)):
        return max(0.0, min(1.0, 1.0 - (float(ngrs_score) / 100.0)))
    return 0.5


def _tags(finding: Any) -> List[str]:
    """Collect owasp/mitre/policy-clause references into a flat tag list."""
    tags: List[str] = []
    for key in ("owasp_llm_ref", "owasp_asi_ref", "mitre_atlas_technique"):
        value = _get(finding, key)
        if value:
            tags.append(str(value))
    for clause in _get(finding, "policy_clauses_violated") or []:
        tags.append(f"policy:{clause}")
    goal_type = _get(finding, "goal_type")
    if goal_type:
        tags.append(f"goal:{goal_type}")
    return tags


def finding_to_result(finding: Any, index: int = 0) -> Dict[str, Any]:
    """Convert a single nuguard `Finding` (model, `.model_dump()` dict, or plain dict) to an EvalPort `Result` dict.

    Mapping (per the plan posted to NuGuardAI/nuguard#355):

    - `test_case_id` <- `chain_id` if set (redteam multi-step chains), else
      `finding_id`, else a positional fallback.
    - one `GraderResult` per finding, `grader_id` derived from `goal_type`
      (falls back to `"gr_nuguard_finding"` when `goal_type` is unset, which
      happens for plain redteam findings that never go through validate
      mode's `ValidateFindingType` tagging).
    - `score` <- severity banded to [0.0, 1.0] (`_SEVERITY_SCORE`), falling
      back to `1 - ngrs_score/100` when severity doesn't parse.
    - `passed` <- `True` only when `verified is False` (nuguard's post-hoc
      probe explicitly could not reproduce the finding); every other case
      (unverified, reproduced, or verification not run at all) is `passed:
      False`, since a `Finding`'s existence *is* the failure signal.
    - `reason` <- `evidence_quote`, else `evidence`, else `description`.
    - `metadata` <- severity, ngrs_score/vector, authorization_decision,
      guardrail_control, verified, chain_id, container_image, and the raw
      finding dict under `metadata.nuguard_finding` for anything this mapping
      doesn't otherwise surface.
    """
    finding_id = _get(finding, "finding_id") or f"finding_{index}"
    chain_id = _get(finding, "chain_id")
    test_case_id = str(chain_id or finding_id)

    severity = _get(finding, "severity")
    ngrs_score = _get(finding, "ngrs_score")
    score = _severity_score(severity, ngrs_score)

    verified = _get(finding, "verified")
    passed = verified is False

    reason = _get(finding, "evidence_quote") or _get(finding, "evidence") or _get(finding, "description") or ""

    goal_type = _get(finding, "goal_type")
    grader_id = f"gr_{goal_type.lower()}" if goal_type else "gr_nuguard_finding"

    gr_metadata: Dict[str, Any] = {
        "severity": _severity_str(severity),
        "tags": _tags(finding),
    }
    for key in (
        "ngrs_score",
        "ngrs_vector",
        "authorization_decision",
        "guardrail_control",
        "affected_component",
        "container_image",
        "success_indicator",
        "log_correlation_status",
    ):
        value = _get(finding, key)
        if value not in (None, ""):
            gr_metadata[key] = value
    if verified is not None:
        gr_metadata["verified"] = verified
    if chain_id:
        gr_metadata["chain_id"] = chain_id

    grader_result: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": "custom",
        "score": score,
        "passed": passed,
        "reason": reason,
        "metadata": gr_metadata,
    }

    title = _get(finding, "title") or ""
    return {
        "test_case_id": test_case_id,
        "actual_output": title,
        "grader_results": [grader_result],
        "passed": passed,
        "metadata": {
            "nuguard": {
                "finding_id": finding_id,
                "goal_type": goal_type,
                "scenario_type": _get(finding, "scenario_type"),
            }
        },
    }


def _no_findings_result(run_id: str) -> Dict[str, Any]:
    """Synthesize a single passing Result when a run produced zero findings.

    EvalPort's `validate_result_set()` requires `results` to be a non-empty
    list (spec.SPEC.md ResultSet.results is `array of Result` / required),
    so a clean `scan_outcome == "no_findings"` run -- which has nothing to
    report per-finding -- still needs one Result to produce a valid
    ResultSet. This one records the scan itself as a passing check rather
    than being omitted or treated as invalid.
    """
    return {
        "test_case_id": f"{run_id}_scan",
        "actual_output": "no findings",
        "grader_results": [
            {
                "grader_id": "gr_nuguard_scan_outcome",
                "type": "custom",
                "score": 1.0,
                "passed": True,
                "reason": "validate run completed with no findings",
                "metadata": {"scan_outcome": "no_findings"},
            }
        ],
        "passed": True,
        "metadata": {"nuguard": {"synthetic": True}},
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    scores = [gr["score"] for r in results for gr in r.get("grader_results", []) if gr.get("score") is not None]
    by_grader: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for gr in r.get("grader_results", []):
            bucket = by_grader.setdefault(gr["grader_id"], {"passed": 0, "failed": 0, "_scores": []})
            if gr.get("passed"):
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
            if gr.get("score") is not None:
                bucket["_scores"].append(gr["score"])
    for bucket in by_grader.values():
        s = bucket.pop("_scores")
        bucket["avg_score"] = sum(s) / len(s) if s else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped": 0,
        "pass_rate": passed / total if total else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "by_grader": by_grader,
    }


def to_openeval(run_result: Any, suite_id: Optional[str] = None) -> Dict[str, Any]:
    """Export a nuguard `ValidateRunResult` (model, or shape-compatible object/dict) to an EvalPort `ResultSet` dict.

    `run_result` must expose `run_id`, `findings` (a `list[dict]` of
    serialized `Finding`s -- or `list[Finding]` model instances, also
    handled), `scan_outcome`, and optionally `capability_map` /
    `policy_records` / `scenarios_executed` / `effective_endpoint` /
    `target_endpoint_source`.

    One `Finding` becomes one `Result` (see `finding_to_result`). When there
    are zero findings, a single synthetic passing `Result` is emitted so the
    `ResultSet` still validates (see `_no_findings_result`).

    `ValidateRunResult` carries no run-level timestamp of its own, so
    `started_at` is taken from `capability_map.built_at` (the closest
    available timestamp) when present, else the current time -- this is
    documented as a known approximation in README.md/SPEC.md, not a nuguard
    field being renamed.

    `capability_map` and `policy_records` don't have a clean 1:1 mapping
    into EvalPort's `Result`/`GraderResult` (per the plan posted to
    NuGuardAI/nuguard#355), so they're carried through losslessly as
    `ResultSet.metadata.nuguard.capability_map` /
    `ResultSet.metadata.nuguard.policy_records_count` instead of being
    force-fit into individual results.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass it
    to `openeval.validate.validate_result_set()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` result-set file.
    """
    run_id = str(_get(run_result, "run_id") or "nuguard_run")
    findings = _get(run_result, "findings") or []

    results = [finding_to_result(f, i) for i, f in enumerate(findings)]
    if not results:
        results = [_no_findings_result(run_id)]

    capability_map = _get(run_result, "capability_map")
    started_at = None
    if capability_map is not None:
        built_at = _get(capability_map, "built_at")
        if built_at is not None:
            started_at = built_at.isoformat() if hasattr(built_at, "isoformat") else str(built_at)
    if started_at is None:
        started_at = datetime.now(timezone.utc).isoformat()

    scan_outcome = _get(run_result, "scan_outcome") or "no_findings"

    nuguard_meta: Dict[str, Any] = {
        "scan_outcome": scan_outcome,
        "scenarios_executed": _get(run_result, "scenarios_executed"),
        "effective_endpoint": _get(run_result, "effective_endpoint"),
        "target_endpoint_source": _get(run_result, "target_endpoint_source"),
    }
    if capability_map is not None:
        entries = _get(capability_map, "entries") or []
        nuguard_meta["capability_map"] = {
            "tools_total": len(entries),
            "tools_exercised": sum(1 for e in entries if _get(e, "exercised")),
            "tools_policy_noncompliant": [
                _get(e, "tool_name") for e in entries if not _get(e, "policy_compliant", True)
            ],
        }
    policy_records = _get(run_result, "policy_records")
    if policy_records is not None:
        nuguard_meta["policy_records_count"] = len(policy_records)

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id or f"nuguard_validate_{run_id}",
        "run_id": run_id,
        "started_at": started_at,
        "runner": {"name": "nuguard", "version": "unknown"},
        "results": results,
        "summary": _summarize(results),
        "metadata": {
            "openeval": {"source": "nuguard"},
            "nuguard": {k: v for k, v in nuguard_meta.items() if v is not None},
        },
    }


def from_openeval(result_set: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort `ResultSet` into a list of nuguard-`Finding`-shaped dicts.

    This is necessarily lossy and partial: a generic `ResultSet` has no
    concept of `capability_map`/`policy_records`/redteam-specific fields
    (goal_type, sbom_path, attack_steps, ...), so only the fields with an
    obvious inverse of `finding_to_result`'s mapping are reconstructed.
    Every non-passing `Result` becomes one Finding-shaped dict; a `Result`
    where every `GraderResult` passed is skipped (nothing to report), and
    the synthetic `_no_findings_result`'s `test_case_id` suffix
    (`"_scan"`) is recognized and skipped as well.

    Returns plain dicts with nuguard's `Finding` field names, suitable for
    `Finding(**d)` once `severity` and `finding_id` are present.
    """
    findings: List[Dict[str, Any]] = []
    for result in result_set.get("results", []):
        if result.get("passed"):
            continue
        test_case_id = result.get("test_case_id", "")
        if test_case_id.endswith("_scan") and not result.get("grader_results"):
            continue
        grader_results = result.get("grader_results", [])
        if not grader_results:
            continue
        primary = grader_results[0]
        score = primary.get("score")
        severity = "info"
        if isinstance(score, (int, float)):
            # invert the score->severity banding used by finding_to_result
            for sev, sev_score in sorted(_SEVERITY_SCORE.items(), key=lambda kv: kv[1]):
                if score <= sev_score:
                    severity = sev
                    break
        gr_meta = primary.get("metadata") or {}
        finding: Dict[str, Any] = {
            "finding_id": test_case_id,
            "title": result.get("actual_output") or primary.get("grader_id", ""),
            "severity": gr_meta.get("severity", severity),
            "description": primary.get("reason", ""),
            "evidence": primary.get("reason", ""),
        }
        if "chain_id" in gr_meta:
            finding["chain_id"] = gr_meta["chain_id"]
        if "verified" in gr_meta:
            finding["verified"] = gr_meta["verified"]
        for key in ("ngrs_score", "ngrs_vector", "authorization_decision", "guardrail_control", "affected_component"):
            if key in gr_meta:
                finding[key] = gr_meta[key]
        findings.append(finding)
    return findings
