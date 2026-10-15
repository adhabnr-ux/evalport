"""Argus <-> EvalPort adapter.

Standalone converter between Argus (https://github.com/gy15901580825/Argus)
redteam run results and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Argus's redteam runner produces a `run` (dict or object) carrying
`id`/`status`/`started_at`/`finished_at` plus a `findings` sequence of
`Finding` objects -- the same shape `api_service/redteam/reports.py`'s
`render_html`/`render_sarif`/`render_junit` renderers already take, per
`Runner`/`Finding` as read in
https://github.com/gy15901580825/Argus/issues/17. Each `Finding`
(`id`, `probe_id`, `verdict`, `severity`, `confidence`, `reasoning`,
`attack_prompt`, `target_response`, `target_latency_ms`, `probed_at`,
`judge_model`, `escalated_model`, `atlas_id`, `owasp_id`, `nist_id`,
`eu_ai_act_id`, `evidence` -- the real dataclass fields in
orchestrator/orchestrator/redteam/runner.py) maps onto an EvalPort `Result`
almost one-to-one: `verdict` as the pass/fail signal, `confidence` as the
grader score (for a judged verdict only -- see `_JUDGED_VERDICTS` below),
`severity` and the ATLAS/OWASP/NIST/EU-AI-Act ids as grader-result
metadata/labels, `reasoning` as the grader's `reason`,
`target_latency_ms`/`probed_at` as `Result.duration_ms`/`.completed_at`,
and the `attack_prompt`/`target_response` pair as the transcript.

This is a standalone adapter package, not a change to Argus itself: per
https://github.com/gy15901580825/Argus/issues/17#issuecomment-6185241349
the maintainer confirmed a fourth renderer alongside `render_html` /
`render_sarif` / `render_junit` is the right shape, and that the `run`
dict those renderers take (now duplicated byte-identically between
`api_service/redteam/reports.py` and `cli/argus_probe/reports.py` as of
Argus #21) is the contract to build on -- so this adapter is written
against that `run` dict/`Finding` contract from the outside, exactly like
`render_sarif()` is, and needs no change inside either copy of Argus's
`reports.py`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"

# Real verdict strings (orchestrator/orchestrator/redteam/runner.py's
# VERDICT_* constants, lower-cased before comparison here) split exactly the
# way Argus's own coverage.py already splits them
# (orchestrator/orchestrator/redteam/coverage.py's `_TESTED_VERDICTS` /
# `_UNTESTED_ONLY_VERDICTS`):
#
# - "pass" / "fail" / "warn": the judge actually ran and rendered a verdict
#   (test_redteam_judge.py shows the judge model literally emitting
#   {"verdict": "warn", ...}). These get a real GraderResult.score.
# - "skipped" / "blocked_by_target" / "aborted_cost" / "error": the probe
#   never reached a real judgment -- a target-class mismatch, the target's
#   own gateway blocking the request (HTTP 4xx), a cost cap, or the
#   judge/runner itself erroring. `Finding.confidence` on these is not a
#   pass/fail score: `Runner._skip()` hard-codes `confidence=1.0` for every
#   skip regardless of outcome, since it means "confident this probe didn't
#   apply," not "confident in a pass/fail judgment." This adapter never
#   surfaces it as `GraderResult.score` for these verdicts -- per SPEC.md
#   Validation Rule 6, a GraderResult with no real judgment gets
#   `score: null` and `passed: false`, not a borrowed, miscategorized
#   number. The raw verdict string is always preserved under
#   `metadata.argus.verdict` either way, so `from_openeval()` round-trips it
#   exactly rather than collapsing it to a bare "fail".
_JUDGED_VERDICTS = {"pass", "fail", "warn"}
_PASS_VERDICTS = {"pass"}
_LABEL_FIELDS = ("atlas_id", "owasp_id", "nist_id", "eu_ai_act_id")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    Argus's `Finding`/run objects (a dataclass per
    orchestrator/orchestrator/redteam/runner.py) and JSON-loaded run output
    both show up in the wild, so every accessor in this module goes through
    here rather than assuming one shape -- the same pattern
    autogen-openeval-adapter and freeplay-openeval-adapter in this repo use.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clamp_score(value: Optional[Union[bool, float, int]]) -> Optional[float]:
    """Clamp a raw `confidence` value into EvalPort's required [0, 1] range."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return None


def _to_rfc3339(value: Any) -> str:
    """Best-effort normalization of a timestamp into an RFC 3339 date-time
    with seconds and an offset, as EvalPort's `started_at`/`completed_at`
    require. Accepts a `datetime` (naive datetimes are assumed UTC) or a
    string; a string that already carries an offset or trailing "Z" is
    passed through unchanged, and a bare local-time string has "Z" appended.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
    s = str(value)
    if s.endswith("Z") or s.endswith("z"):
        return s
    if len(s) >= 6 and s[-6] in "+-" and s[-3] == ":":
        return s
    return s + "Z"


def _finding_to_result(finding: Any, index: int) -> Dict[str, Any]:
    """Normalize a single Argus `Finding` (dict or attribute-object) into an
    EvalPort `Result` dict."""
    probe_id = _get(finding, "probe_id") or f"finding_{index}"

    verdict_raw = _get(finding, "verdict")
    verdict = str(verdict_raw).strip().lower() if verdict_raw is not None else ""
    is_judged = verdict in _JUDGED_VERDICTS
    is_pass = verdict in _PASS_VERDICTS

    confidence = _get(finding, "confidence")
    score = _clamp_score(confidence) if is_judged else None
    if is_judged and score is None:
        # A judged verdict (pass/fail/warn) Argus is confident enough to
        # render, but with no numeric `confidence` attached -- treat the
        # verdict itself as full confidence rather than discarding the
        # signal as "unverified".
        score = 1.0 if is_pass else 0.0

    # SPEC.md Rule 6: a GraderResult with score null ("not verified") MUST
    # have passed false. Reachable here for skipped/blocked_by_target/
    # aborted_cost/error, where no real judgment happened.
    passed = is_pass if score is not None else False

    grader_metadata: Dict[str, Any] = {}
    severity = _get(finding, "severity")
    if severity is not None:
        grader_metadata["severity"] = severity
    labels = {f: _get(finding, f) for f in _LABEL_FIELDS if _get(finding, f)}
    if labels:
        grader_metadata["labels"] = labels
    judge_model = _get(finding, "judge_model")
    if judge_model:
        grader_metadata["judge_model"] = judge_model
    escalated_model = _get(finding, "escalated_model")
    if escalated_model:
        grader_metadata["escalated_model"] = escalated_model
    if not is_judged and confidence is not None:
        # Not a pass/fail score (see _JUDGED_VERDICTS above) -- preserved
        # under its own key so it's never mistaken for one, but not lost.
        grader_metadata["categorization_confidence"] = confidence

    grader_result: Dict[str, Any] = {
        "grader_id": "argus-judge",
        "type": "llm_judge" if judge_model else "custom",
        "score": score,
        "passed": bool(passed),
    }
    reasoning = _get(finding, "reasoning")
    if reasoning:
        grader_result["reason"] = str(reasoning)
    if grader_metadata:
        grader_result["metadata"] = grader_metadata

    argus_meta: Dict[str, Any] = {}
    if verdict_raw is not None:
        argus_meta["verdict"] = verdict_raw
    attack_prompt = _get(finding, "attack_prompt")
    if attack_prompt is not None:
        argus_meta["attack_prompt"] = attack_prompt
    finding_id = _get(finding, "id")
    if finding_id is not None:
        argus_meta["finding_id"] = str(finding_id)
    evidence = _get(finding, "evidence")
    if evidence:
        argus_meta["evidence"] = evidence

    result: Dict[str, Any] = {
        "test_case_id": str(probe_id),
        "passed": bool(passed),
        "grader_results": [grader_result],
    }
    target_response = _get(finding, "target_response")
    if target_response is not None:
        result["actual_output"] = str(target_response)
    target_latency_ms = _get(finding, "target_latency_ms")
    if target_latency_ms is not None:
        try:
            result["duration_ms"] = max(0, int(round(float(target_latency_ms))))
        except (TypeError, ValueError):
            pass
    probed_at = _get(finding, "probed_at")
    if probed_at is not None:
        result["completed_at"] = _to_rfc3339(probed_at)
    if argus_meta:
        result["metadata"] = {"argus": argus_meta}
    return result


def to_openeval(run: Any, suite_id: Optional[str] = None) -> Dict[str, Any]:
    """Export an Argus redteam `run` (dict or object) to an EvalPort-shaped
    ResultSet (dict).

    `run` may be any object or dict exposing a `findings` sequence of
    `Finding`s plus `id`/`status`/`started_at`/`finished_at` -- this matches
    both Argus's real `Runner` output and plain JSON-loaded run output (the
    same `run` dict `reports.render_sarif(run)` already consumes), so no
    direct Argus import is required.

    `suite_id` names the EvalPort suite this run's probes belong to.
    Argus's redteam runs aren't driven by an EvalPort `EvalSuite` file, so
    this defaults to the run's own `suite_id` if present, else the constant
    `"argus-redteam-probes"` -- pass an explicit value if you track probe
    sets under a more specific id.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass
    it to `openeval.validate.validate_result_set()` to confirm compliance,
    or `json.dump()` it directly to share as a `.openeval.json` result file.
    """
    findings = _get(run, "findings") or []
    if not findings:
        raise ValueError(
            "to_openeval() requires a run with at least one finding -- "
            "EvalPort's ResultSet.results must be non-empty"
        )

    run_id = _get(run, "id") or _get(run, "run_id") or str(uuid.uuid4())
    started_at = _get(run, "started_at") or datetime.now(timezone.utc)
    finished_at = _get(run, "finished_at") or _get(run, "completed_at")
    status = _get(run, "status")

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": str(suite_id or _get(run, "suite_id") or "argus-redteam-probes"),
        "run_id": str(run_id),
        "started_at": _to_rfc3339(started_at),
        "results": [_finding_to_result(f, i) for i, f in enumerate(findings)],
        "runner": {"name": "argus"},
    }
    if finished_at:
        result_set["completed_at"] = _to_rfc3339(finished_at)
    if status:
        result_set["metadata"] = {"argus": {"status": status}}
    return result_set


def from_openeval(result_set: Dict[str, Any]) -> Dict[str, Any]:
    """Import an EvalPort ResultSet into an Argus-shaped run dict.

    Returns a plain dict (`id`, `status`, `started_at`, `finished_at`,
    `findings`) rather than Argus's real `Runner`/`Finding` classes, since
    those aren't imported here (see module docstring) -- pass
    `findings` straight into `Finding(**f)` if the field names line up, or
    map them explicitly. Round-trips every field this adapter's
    `to_openeval()` produced, including the original `verdict` string even
    for a skipped/blocked_by_target/aborted_cost/error verdict that
    `to_openeval()` had no real judge confidence for.
    """
    run_metadata = (result_set.get("metadata") or {}).get("argus") or {}

    findings: List[Dict[str, Any]] = []
    for result in result_set.get("results", []):
        grader_results = result.get("grader_results") or []
        gr = grader_results[0] if grader_results else {}
        gr_metadata = gr.get("metadata") or {}
        labels = gr_metadata.get("labels") or {}
        result_metadata = (result.get("metadata") or {}).get("argus") or {}

        verdict = result_metadata.get("verdict")
        if verdict is None:
            verdict = "pass" if result.get("passed") else "fail"
        is_judged = str(verdict).strip().lower() in _JUDGED_VERDICTS
        confidence = gr.get("score") if is_judged else gr_metadata.get("categorization_confidence")

        findings.append(
            {
                "id": result_metadata.get("finding_id"),
                "probe_id": result.get("test_case_id"),
                "verdict": verdict,
                "severity": gr_metadata.get("severity"),
                "confidence": confidence,
                "reasoning": gr.get("reason"),
                "attack_prompt": result_metadata.get("attack_prompt"),
                "target_response": result.get("actual_output"),
                "target_latency_ms": result.get("duration_ms"),
                "probed_at": result.get("completed_at"),
                "judge_model": gr_metadata.get("judge_model"),
                "escalated_model": gr_metadata.get("escalated_model"),
                "atlas_id": labels.get("atlas_id"),
                "owasp_id": labels.get("owasp_id"),
                "nist_id": labels.get("nist_id"),
                "eu_ai_act_id": labels.get("eu_ai_act_id"),
                "evidence": result_metadata.get("evidence"),
            }
        )

    return {
        "id": result_set.get("run_id"),
        "status": run_metadata.get("status"),
        "started_at": result_set.get("started_at"),
        "finished_at": result_set.get("completed_at"),
        "findings": findings,
    }
