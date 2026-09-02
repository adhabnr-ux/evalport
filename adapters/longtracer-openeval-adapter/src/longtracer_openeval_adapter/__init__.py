"""LongTracer -> EvalPort adapter.

Standalone, one-way converter from LongTracer's `VerificationResult`
(https://github.com/ENDEVSOLS/LongTracer, `longtracer/guard/verifier.py`) into
the EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Background: proposed in https://github.com/adhabnr-ux/evalport (see
ENDEVSOLS/LongTracer#15). The maintainer's stated preference is for this to
eventually live in LongTracer itself as an optional
`longtracer/adapters/evalport.py` module with no hard EvalPort dependency for
LongTracer users -- this package is a drop-in for exactly that: everything
below is a single, dependency-light module (only `evalport-sdk` at runtime --
the package providing the `openeval.*` modules this file imports -- used
solely for the `OPENEVAL_VERSION` constant) that a maintainer can copy
into `longtracer/adapters/evalport.py` largely as-is, or that anyone can
`pip install longtracer-openeval-adapter` and use against their own
LongTracer results today.

Design constraints, per the maintainer's review comment on issue #15:
  * `to_openeval()` only for v1 -- no `from_openeval()` (there is no
    meaningful "import into LongTracer" direction: LongTracer produces
    verification results, it does not consume EvalPort suites).
  * Preserve response-level fields: `trust_score`, `verdict`, `summary`,
    `latency_stats`.
  * Preserve claim-level evidence: `supported`, `score`, `best_source`,
    `is_hallucination` (plus the rest of the claim dict, for completeness).
  * Keep unsupported claims distinct from confirmed hallucinations.
  * Never recalculate or change LongTracer's own scores -- every score in
    the output is a direct passthrough (clamped only where EvalPort's
    schema requires a [0, 1] range; the original value is preserved
    alongside it).
  * Support both a single `VerificationResult` and batches (as produced by
    `CitationVerifier.verify_batch()`).

`VerificationResult` (and each claim dict inside it) is accessed structurally
(attribute-or-key, via `_get()`) rather than imported from `longtracer`
directly, so this adapter has no hard dependency on the `longtracer` package
itself -- the same pattern used by every other *-openeval-adapter in this
repo (see `adapters/crewai-openeval-adapter`). The shape matched here is
`longtracer.guard.verifier.VerificationResult` as of
ENDEVSOLS/LongTracer@4136a70 (`longtracer/guard/verifier.py`) and its claim
dicts as produced by `longtracer.guard.nli_model.HybridVerificationModel`
as of ENDEVSOLS/LongTracer@bf1cc72 (`longtracer/guard/nli_model.py`).
"""
from __future__ import annotations

import collections.abc as _abc
import datetime as _dt
from importlib import metadata as _importlib_metadata
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk is a runtime dependency,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "__version__"]
__version__ = "0.1.0"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    `VerificationResult` is a dataclass in real usage, but callers may also
    hand this a plain dict (e.g. loaded back from JSON, or built by hand in
    tests) -- every accessor here goes through this helper so both work.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_batch(results: Any) -> bool:
    """Decide whether `results` (the `to_openeval()` argument) is a batch of
    `VerificationResult`s, rather than a single one.

    `to_openeval()`'s type hint is `Union[Any, Sequence[Any]]`, but the
    original implementation only ever treated `list`/`tuple` as a batch --
    any other `Sequence` (a `deque`, a generator-backed custom `Sequence`,
    etc, as `CitationVerifier.verify_batch()` could in principle return, or a
    caller could hand-build) was silently misread as a single result and
    then failed downstream once `_get()` tried to read `VerificationResult`
    attributes off it. A single `VerificationResult` is itself never a
    `collections.abc.Sequence` (it's a dataclass, or a dict when duck-typed
    -- see `_get()`), so testing for `Sequence` membership (excluding
    str/bytes/bytearray, which are technically Sequences but are never a
    valid `results` value here) correctly covers `list`/`tuple`/`deque`/any
    other real sequence type alike, without misclassifying a single result.
    """
    if isinstance(results, (str, bytes, bytearray)):
        return False
    return isinstance(results, _abc.Sequence)


def _clamp01(value: Any) -> Optional[float]:
    """Clamp a numeric score into EvalPort's required [0.0, 1.0] range.

    LongTracer's per-claim `score` is an average cosine similarity, which is
    mathematically unbounded to [-1.0, 1.0] (in practice it is effectively
    always >= 0 for the sentence-transformer models LongTracer uses, but
    EvalPort's `GraderResult.score` is a hard [0, 1] requirement -- see
    "Validation Rules > 5. Score Range" in spec/SPEC.md). This never changes
    what LongTracer computed; the unclamped original is always preserved
    alongside it in `metadata.openeval.raw_score`.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(1.0, f))


def _claim_status(claim: Dict[str, Any], supported: bool) -> str:
    """Categorize a claim as one of "supported" / "unsupported" / "hallucination".

    This is the mechanism that keeps unsupported claims distinct from
    confirmed hallucinations (per the maintainer's requirement) -- it is a
    pure relabeling of LongTracer's own `supported` / `is_hallucination`
    booleans, not a new judgment.
    """
    if claim.get("is_hallucination"):
        return "hallucination"
    if not supported:
        return "unsupported"
    return "supported"


def _claim_to_grader_result(claim: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Convert one LongTracer claim dict into an EvalPort GraderResult.

    `claim` is one entry of `VerificationResult.claims`, produced by
    `HybridVerificationModel.verify_claim()` / `verify_claims_batch()`. Its
    keys as of ENDEVSOLS/LongTracer@bf1cc72 are: claim, supported, score,
    best_score, sentence_results, contradiction_score, entailment_score,
    nli_ran, best_source, best_source_index, best_source_metadata,
    is_hallucination, is_meta_statement, has_hallucination_pattern.
    """
    supported = bool(claim.get("supported", False))
    is_hallucination = bool(claim.get("is_hallucination", False))
    raw_score = claim.get("score")
    score = _clamp01(raw_score)

    if is_hallucination:
        reason = "Flagged as a hallucination: NLI contradiction and/or an unsupported claim pattern with no matching source."
    elif not supported:
        reason = "Claim not supported by any provided source (below LongTracer's support threshold)."
    else:
        reason = "Claim supported by source evidence."

    return {
        "grader_id": f"lt_claim_{index}",
        "type": "custom",
        "score": score,
        "passed": supported,
        "reason": reason,
        "metadata": {
            "claim": claim.get("claim"),
            "best_source": claim.get("best_source"),
            "best_source_index": claim.get("best_source_index"),
            "best_source_metadata": claim.get("best_source_metadata"),
            "best_score": claim.get("best_score"),
            "contradiction_score": claim.get("contradiction_score"),
            "entailment_score": claim.get("entailment_score"),
            "nli_ran": claim.get("nli_ran"),
            "is_hallucination": is_hallucination,
            "is_meta_statement": bool(claim.get("is_meta_statement", False)),
            "has_hallucination_pattern": bool(claim.get("has_hallucination_pattern", False)),
            "sentence_results": claim.get("sentence_results"),
            "openeval": {
                "raw_score": raw_score,
                "claim_status": _claim_status(claim, supported),
            },
        },
    }


def _result_to_evalport_result(
    verification_result: Any,
    test_case_id: str,
    response_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one `VerificationResult` into a single EvalPort `Result` object."""
    claims = _get(verification_result, "claims", []) or []
    verdict = _get(verification_result, "verdict", "FAIL")
    trust_score = _get(verification_result, "trust_score")
    latency_stats = _get(verification_result, "latency_stats")

    result: Dict[str, Any] = {
        "test_case_id": test_case_id,
        # Direct passthrough of LongTracer's own pass/fail call -- verdict
        # "PASS" already means all_supported and hallucination_count == 0
        # (see VerificationResult.__post_init__), so this is not a
        # recalculation, just a rename to EvalPort's boolean field.
        "passed": verdict == "PASS",
        "grader_results": [
            _claim_to_grader_result(c, i) for i, c in enumerate(claims)
        ],
        "metadata": {
            "trust_score": trust_score,
            "verdict": verdict,
            "summary": _get(verification_result, "summary"),
            "all_supported": _get(verification_result, "all_supported"),
            "hallucination_count": _get(verification_result, "hallucination_count"),
            "flagged_claim_count": len(_get(verification_result, "flagged_claims", []) or []),
            "latency_stats": latency_stats,
            "openeval": {"source": "longtracer"},
        },
    }

    if response_text is not None:
        result["actual_output"] = response_text

    if isinstance(latency_stats, dict) and isinstance(latency_stats.get("total_ms"), (int, float)):
        result["duration_ms"] = int(latency_stats["total_ms"])

    return result


def to_openeval(
    results: Union[Any, Sequence[Any]],
    *,
    test_case_ids: Optional[Sequence[str]] = None,
    response_texts: Optional[Sequence[Optional[str]]] = None,
    run_id: Optional[str] = None,
    suite_id: str = "longtracer_citation_verification",
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Export one or many LongTracer `VerificationResult`s to an EvalPort `ResultSet`.

    `results` is either a single `VerificationResult` (from
    `CitationVerifier.verify()` / `.verify_parallel()`) or a sequence of them
    (from `CitationVerifier.verify_batch()`). Each result becomes one
    EvalPort `Result`, with each of its claims becoming a `GraderResult`
    (type `"custom"`) inside that `Result.grader_results`.

    `test_case_ids` optionally names each result (defaults to
    `"claim_verification_0"`, `"claim_verification_1"`, ...). `response_texts`
    optionally supplies the original LLM response text per result -- LongTracer's
    `VerificationResult` does not retain the response it verified, so
    `actual_output` is omitted unless a response text is supplied here.

    `run_id` / `suite_id` / `started_at` populate the required `ResultSet`
    fields LongTracer has no equivalent for; sane defaults are used when
    omitted (`started_at` defaults to the current UTC time at conversion,
    since LongTracer does not track a run start timestamp itself).

    No score is recalculated anywhere in this conversion: `trust_score`,
    `verdict`, `summary`, `latency_stats`, and every per-claim `score`,
    `supported`, `best_source`, and `is_hallucination` are direct
    passthroughs of what LongTracer already computed (only clamped into
    EvalPort's required [0, 1] score range where needed -- see `_clamp01`).

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass
    it to `openeval.validate.validate_result_set()` to confirm compliance,
    or `json.dump()` it directly to share as a `.json` result-set file.
    """
    single = not _is_batch(results)
    result_list: List[Any] = [results] if single else list(results)

    if test_case_ids is not None and len(test_case_ids) != len(result_list):
        raise ValueError(
            f"test_case_ids has {len(test_case_ids)} entries but {len(result_list)} "
            f"result(s) were given"
        )
    if response_texts is not None and len(response_texts) != len(result_list):
        raise ValueError(
            f"response_texts has {len(response_texts)} entries but {len(result_list)} "
            f"result(s) were given"
        )

    ids = list(test_case_ids) if test_case_ids is not None else [
        f"claim_verification_{i}" for i in range(len(result_list))
    ]
    texts: List[Optional[str]] = list(response_texts) if response_texts is not None else [None] * len(result_list)

    evalport_results = [
        _result_to_evalport_result(vr, tid, text)
        for vr, tid, text in zip(result_list, ids, texts)
    ]

    total = len(evalport_results)
    passed = sum(1 for r in evalport_results if r["passed"])
    trust_scores = [
        r["metadata"]["trust_score"]
        for r in evalport_results
        if isinstance(r["metadata"].get("trust_score"), (int, float))
    ]

    return {
        "$schema": "https://evalport.org/schema/resultset.json",
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id or "longtracer_run",
        "started_at": started_at or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": evalport_results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total) if total else 0.0,
            "avg_score": (sum(trust_scores) / len(trust_scores)) if trust_scores else None,
        },
        "runner": {"name": "longtracer", "version": _longtracer_version()},
        "metadata": {"openeval": {"source": "longtracer"}},
    }


def _longtracer_version() -> Optional[str]:
    """Best-effort read of the installed `longtracer` package version.

    This adapter does not depend on `longtracer` being installed at all
    (see module docstring), so this is purely a nice-to-have for the
    `ResultSet.runner.version` field when it happens to be available.

    Reads the version via `importlib.metadata` (installed-distribution
    metadata) rather than `import longtracer` -- the real `longtracer`
    package imports `sentence-transformers` + `torch` at import time (see
    module docstring / `tests/test_adapter.py`'s own note on why its stand-in
    dataclass avoids the real import), so actually importing it here just to
    read `__version__` would add multi-second load time and heavy-dependency
    side effects to every `to_openeval()` call whenever LongTracer happens to
    be installed alongside this adapter -- entirely avoidable, since package
    version metadata is available without running any of the package's code.
    """
    try:
        return _importlib_metadata.version("longtracer")
    except _importlib_metadata.PackageNotFoundError:
        return None
