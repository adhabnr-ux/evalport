"""Tests for the LongTracer -> EvalPort adapter.

`VerificationResult` and its claim dicts are hand-built here to exactly match
the real shapes in ENDEVSOLS/LongTracer's `longtracer/guard/verifier.py`
(`VerificationResult` dataclass, @4136a70) and `longtracer/guard/nli_model.py`
(claim dict, @bf1cc72), rather than depending on the real `longtracer`
package -- it pulls in `sentence-transformers` + `torch` at import time
purely to define that dataclass, which would make this adapter's test suite
depend on a multi-GB install for what is otherwise a pure data-shape
conversion. This mirrors how `adapters/crewai-openeval-adapter` avoids a
hard dependency on the real `crewai` package in its own tests.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openeval.validate import validate_result_set

from longtracer_openeval_adapter import to_openeval


# --- Stand-in for longtracer.guard.verifier.VerificationResult --------------
# Field-for-field match of the real dataclass (see verifier.py @4136a70),
# including its __post_init__-computed verdict/summary, so tests exercise the
# adapter against the exact shape LongTracer actually produces.

@dataclass
class FakeVerificationResult:
    trust_score: float
    claims: List[Dict]
    flagged_claims: List[Dict]
    hallucinations: List[Dict]
    all_supported: bool
    hallucination_count: int
    verdict: str = "PASS"
    summary: str = ""
    latency_stats: Optional[Dict] = None

    def __post_init__(self):
        self.verdict = "PASS" if (self.all_supported and self.hallucination_count == 0) else "FAIL"
        total = len(self.claims)
        supported = total - len(self.flagged_claims)
        if total == 0:
            self.summary = "No claims to verify."
        elif self.all_supported:
            self.summary = f"All {total} claim(s) supported."
        else:
            parts = [f"{supported}/{total} claims supported"]
            if self.hallucination_count > 0:
                parts.append(f"{self.hallucination_count} hallucination(s) detected")
            self.summary = ", ".join(parts) + "."


def _claim(
    claim="Paris is the capital of France.",
    supported=True,
    score=0.87,
    best_source="Paris is the capital and largest city of France.",
    is_hallucination=False,
    **overrides,
):
    """Build a claim dict matching HybridVerificationModel.verify_claim()'s output shape."""
    base = {
        "claim": claim,
        "supported": supported,
        "score": score,
        "best_score": score,
        "sentence_results": [{"claim_sentence": claim, "score": score, "matched_source": best_source[:100]}],
        "contradiction_score": 0.02,
        "entailment_score": 0.9 if supported else 0.1,
        "nli_ran": True,
        "best_source": best_source,
        "best_source_index": 0,
        "best_source_metadata": {"doc_id": "doc_1"},
        "is_hallucination": is_hallucination,
        "is_meta_statement": False,
        "has_hallucination_pattern": is_hallucination,
    }
    base.update(overrides)
    return base


def _all_supported_result():
    claims = [_claim(), _claim(claim="Water boils at 100C.", best_source="Water boils at 100 degrees Celsius.")]
    return FakeVerificationResult(
        trust_score=0.9,
        claims=claims,
        flagged_claims=[],
        hallucinations=[],
        all_supported=True,
        hallucination_count=0,
        latency_stats={"sts_calls": 2, "sts_avg_ms": 12.0, "nli_calls": 2, "nli_avg_ms": 45.0, "nli_skipped": 0, "total_ms": 114.0},
    )


def _mixed_result():
    supported_claim = _claim()
    unsupported_claim = _claim(
        claim="The Eiffel Tower is in Berlin.",
        supported=False,
        score=0.15,
        best_source="",
        is_hallucination=False,
    )
    hallucinated_claim = _claim(
        claim="Napoleon was born in 1600.",
        supported=False,
        score=0.05,
        contradiction_score=0.91,
        entailment_score=0.02,
        best_source="Napoleon was born in 1769.",
        is_hallucination=True,
    )
    claims = [supported_claim, unsupported_claim, hallucinated_claim]
    return FakeVerificationResult(
        trust_score=0.35,
        claims=claims,
        flagged_claims=[unsupported_claim, hallucinated_claim],
        hallucinations=[hallucinated_claim],
        all_supported=False,
        hallucination_count=1,
        latency_stats={"sts_calls": 3, "sts_avg_ms": 10.0, "nli_calls": 3, "nli_avg_ms": 40.0, "nli_skipped": 0, "total_ms": 150.0},
    )


def test_single_result_shape_and_validity():
    vr = _all_supported_result()
    rs = to_openeval(vr, run_id="run_1", started_at="2026-08-31T00:00:00Z")

    assert rs["run_id"] == "run_1"
    assert rs["suite_id"] == "longtracer_citation_verification"
    assert len(rs["results"]) == 1

    result = rs["results"][0]
    assert result["test_case_id"] == "claim_verification_0"
    assert result["passed"] is True
    assert result["duration_ms"] == 114
    assert result["metadata"]["trust_score"] == 0.9
    assert result["metadata"]["verdict"] == "PASS"
    assert len(result["grader_results"]) == 2

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_response_level_fields_preserved_exactly():
    """trust_score, verdict, summary, latency_stats must be exact passthroughs."""
    vr = _mixed_result()
    rs = to_openeval(vr)
    result = rs["results"][0]

    assert result["metadata"]["trust_score"] == vr.trust_score
    assert result["metadata"]["verdict"] == vr.verdict == "FAIL"
    assert result["metadata"]["summary"] == vr.summary
    assert result["metadata"]["latency_stats"] == vr.latency_stats
    assert result["metadata"]["hallucination_count"] == 1
    assert result["metadata"]["flagged_claim_count"] == 2
    # Result.passed is a direct rename of verdict == "PASS", not a fresh AND
    # over grader_results computed independently.
    assert result["passed"] == (vr.verdict == "PASS") == False


def test_claim_level_evidence_preserved():
    vr = _mixed_result()
    rs = to_openeval(vr)
    graders = rs["results"][0]["grader_results"]

    supported_gr, unsupported_gr, hallucination_gr = graders

    assert supported_gr["passed"] is True
    assert supported_gr["score"] == 0.87
    assert supported_gr["metadata"]["best_source"] == "Paris is the capital and largest city of France."
    assert supported_gr["metadata"]["is_hallucination"] is False

    assert unsupported_gr["passed"] is False
    assert unsupported_gr["metadata"]["is_hallucination"] is False

    assert hallucination_gr["passed"] is False
    assert hallucination_gr["metadata"]["is_hallucination"] is True


def test_unsupported_distinct_from_hallucination():
    """The core requirement from the maintainer's review: an unsupported claim
    that is NOT flagged as a hallucination must be distinguishable from one
    that IS -- both have passed=False, but only the latter has
    is_hallucination=True / claim_status="hallucination"."""
    vr = _mixed_result()
    rs = to_openeval(vr)
    _, unsupported_gr, hallucination_gr = rs["results"][0]["grader_results"]

    assert unsupported_gr["passed"] is False
    assert unsupported_gr["metadata"]["is_hallucination"] is False
    assert unsupported_gr["metadata"]["openeval"]["claim_status"] == "unsupported"

    assert hallucination_gr["passed"] is False
    assert hallucination_gr["metadata"]["is_hallucination"] is True
    assert hallucination_gr["metadata"]["openeval"]["claim_status"] == "hallucination"

    # And a genuinely supported claim is a third, distinct bucket.
    supported_gr = rs["results"][0]["grader_results"][0]
    assert supported_gr["metadata"]["openeval"]["claim_status"] == "supported"


def test_score_is_never_recalculated_only_clamped():
    """A claim score outside [0, 1] (cosine similarity is unbounded to
    [-1, 1]) is clamped for the schema-required `score` field, but the exact
    original LongTracer value is preserved in metadata.openeval.raw_score."""
    claim = _claim(score=-0.3)
    vr = FakeVerificationResult(
        trust_score=0.5, claims=[claim], flagged_claims=[claim],
        hallucinations=[], all_supported=False, hallucination_count=0,
    )
    rs = to_openeval(vr)
    gr = rs["results"][0]["grader_results"][0]

    assert gr["score"] == 0.0  # clamped
    assert gr["metadata"]["openeval"]["raw_score"] == -0.3  # untouched original

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_batch_of_results():
    vr1 = _all_supported_result()
    vr2 = _mixed_result()
    rs = to_openeval([vr1, vr2], run_id="batch_run")

    assert rs["run_id"] == "batch_run"
    assert len(rs["results"]) == 2
    assert rs["results"][0]["test_case_id"] == "claim_verification_0"
    assert rs["results"][1]["test_case_id"] == "claim_verification_1"
    assert rs["summary"]["total"] == 2
    assert rs["summary"]["passed"] == 1
    assert rs["summary"]["failed"] == 1
    assert rs["summary"]["avg_score"] == (vr1.trust_score + vr2.trust_score) / 2

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_batch_detection_works_for_non_list_tuple_sequences():
    """`to_openeval()` is annotated to accept `Sequence[Any]`, not just
    `list`/`tuple` -- a `deque` (or any other real Sequence) must also be
    treated as a batch, not misread as a single VerificationResult."""
    from collections import deque

    vr1 = _all_supported_result()
    vr2 = _mixed_result()

    rs_deque = to_openeval(deque([vr1, vr2]), run_id="deque_run")
    assert len(rs_deque["results"]) == 2
    assert rs_deque["results"][0]["test_case_id"] == "claim_verification_0"
    assert rs_deque["results"][1]["test_case_id"] == "claim_verification_1"
    assert rs_deque["summary"]["total"] == 2
    validation = validate_result_set(rs_deque)
    assert validation.valid, validation.errors

    rs_tuple = to_openeval((vr1, vr2), run_id="tuple_run")
    assert len(rs_tuple["results"]) == 2


def test_batch_with_explicit_test_case_ids_and_response_texts():
    vr1 = _all_supported_result()
    vr2 = _mixed_result()
    rs = to_openeval(
        [vr1, vr2],
        test_case_ids=["resp_a", "resp_b"],
        response_texts=["Paris is the capital of France. Water boils at 100C.", None],
    )

    assert rs["results"][0]["test_case_id"] == "resp_a"
    assert rs["results"][1]["test_case_id"] == "resp_b"
    assert rs["results"][0]["actual_output"] == "Paris is the capital of France. Water boils at 100C."
    assert "actual_output" not in rs["results"][1]


def test_mismatched_test_case_ids_length_raises():
    vr1 = _all_supported_result()
    try:
        to_openeval([vr1], test_case_ids=["a", "b"])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_empty_claims_result_still_valid():
    """VerificationResult._empty_result() shape: no claims, vacuous PASS."""
    vr = FakeVerificationResult(
        trust_score=1.0, claims=[], flagged_claims=[], hallucinations=[],
        all_supported=True, hallucination_count=0,
        latency_stats={"sts_calls": 0, "sts_avg_ms": 0, "nli_calls": 0, "nli_avg_ms": 0, "nli_skipped": 0, "total_ms": 0.0},
    )
    rs = to_openeval(vr)
    result = rs["results"][0]

    assert result["passed"] is True
    assert result["grader_results"] == []

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_dict_input_also_supported():
    """to_openeval() duck-types via attribute-or-key access, so a plain dict
    (e.g. a VerificationResult round-tripped through JSON) works too."""
    vr_dict = {
        "trust_score": 0.5,
        "verdict": "FAIL",
        "summary": "1/2 claims supported.",
        "claims": [_claim(supported=False, score=0.1, is_hallucination=False)],
        "flagged_claims": [_claim(supported=False, score=0.1, is_hallucination=False)],
        "hallucinations": [],
        "all_supported": False,
        "hallucination_count": 0,
        "latency_stats": None,
    }
    rs = to_openeval(vr_dict)
    assert rs["results"][0]["passed"] is False
    assert rs["results"][0]["metadata"]["trust_score"] == 0.5

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_runner_version_uses_importlib_metadata_not_a_real_import():
    """`_longtracer_version()` must resolve the installed `longtracer`
    distribution's version via `importlib.metadata`, never `import
    longtracer` -- the real package pulls in sentence-transformers + torch
    at import time (see module docstring), so an actual import would add
    multi-second load time and heavy side effects to every to_openeval()
    call whenever longtracer happens to be installed. `longtracer` is not
    installed in this test environment (see this file's own module
    docstring), so confirm to_openeval() doesn't error trying to import it,
    doesn't add `longtracer` to sys.modules as a side effect, and reports no
    version rather than raising."""
    import sys

    assert "longtracer" not in sys.modules

    vr = _all_supported_result()
    rs = to_openeval(vr)

    assert rs["runner"]["name"] == "longtracer"
    assert rs["runner"]["version"] is None
    assert "longtracer" not in sys.modules


def test_runner_version_reads_installed_distribution_metadata(monkeypatch):
    """When a `longtracer` distribution IS installed, its version comes from
    distribution metadata (importlib.metadata.version), not from importing
    the package and reading `__version__` off it."""
    import longtracer_openeval_adapter as adapter_module

    monkeypatch.setattr(
        adapter_module._importlib_metadata, "version", lambda name: "9.9.9" if name == "longtracer" else None
    )
    vr = _all_supported_result()
    rs = to_openeval(vr)
    assert rs["runner"]["version"] == "9.9.9"
