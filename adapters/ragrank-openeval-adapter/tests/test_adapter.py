"""Tests for ragrank_openeval_adapter.

Builds real ragrank objects (Dataset, DataNode, EvalResult, metrics,
FakeLLM) rather than hand-rolled stand-ins, and validates every emitted
document against the real `openeval.validate` module -- not a mock of it.
"""

from __future__ import annotations

import pytest
from openeval.validate import validate_result_set, validate_suite

from ragrank.dataset import Dataset
from ragrank.evaluation.outputs import EvalResult
from ragrank.evaluation.usage import TokenUsage
from ragrank.llm.fake import FakeLLM
from ragrank.metric import RecallAtK, exact_match, metric as metric_decorator

from ragrank_openeval_adapter import (
    build_result_set,
    build_suite,
    from_openeval,
    to_openeval,
)


# --------------------------------------------------------------------------
# Fixture: a small, realistic three-row ragrank EvalResult with three
# metrics chosen specifically to exercise the two things the adapter has
# to get right -- null scores and non-unit score_range normalization --
# plus one ordinary LLM-judged metric.
#
# The null score comes from RecallAtK on a row with an empty (not missing)
# `reference_ids` list -- "nothing relevant to find" is the one way to get
# a legitimate per-row None out of a ragrank metric without running into
# Dataset's whole-column-optional typing for `reference`/`reference_ids`
# (a Dataset's `reference` column, if present, is `list[str]`, not
# `list[str | None]` -- there is no native "some rows have it, some
# don't" shape at the Dataset level, only DataNode.context/DataNode's own
# per-row lists, which can legitimately be empty).
# --------------------------------------------------------------------------


@metric_decorator(
    name="Answer Length Quality",
    threshold=3.0,
    score_range=(1.0, 5.0),
)
def length_quality(response: str) -> float:
    """A deterministic stand-in for a Likert-style (1-5) custom metric."""
    words = len(response.split())
    return min(5.0, max(1.0, 1.0 + words))


def _build_result() -> EvalResult:
    dataset = Dataset(
        question=[
            "What is the capital of France?",
            "What is the capital of Wakanda?",
            "What is the capital of Germany?",
        ],
        context=[
            ["France is a country in Western Europe. Its capital is Paris."],
            [],
            ["Germany is a country in Central Europe. Its capital is Berlin."],
        ],
        response=[
            "Paris is the capital of France.",
            "No.",
            "The capital of Germany is Berlin.",
        ],
        reference=["Paris", "unknown", "Berlin"],
        retrieved_ids=[["d1", "d2"], ["d5", "d6"], ["d9"]],
        reference_ids=[["d1"], [], ["d9"]],  # row 1: nothing relevant -> null score
    )

    llm = FakeLLM(responses=["0.9", "0.4", "0.95"])
    faithfulness_metric = _faithfulness(llm, threshold=0.7)

    metrics = [RecallAtK(), length_quality, faithfulness_metric]
    rows = list(dataset)

    results_grid = [[m.score(row) for row in rows] for m in metrics]
    scores_grid = [[r.score for r in row] for row in results_grid]

    return EvalResult(
        llm=llm,
        metrics=metrics,
        dataset=dataset,
        scores=scores_grid,
        results=results_grid,
        response_time=1.23,
        usage=TokenUsage(prompt_tokens=120, response_tokens=45, calls=3),
    )


def _faithfulness(llm: FakeLLM, threshold: float):
    from ragrank.metric import LLMJudge

    # LLMJudge (not the claim-decomposing Faithfulness metric) so a single
    # scripted FakeLLM response per row is a direct, one-call score --
    # Faithfulness itself is a ClaimMetric that issues an LLM call to
    # extract claims and then one *more* per claim, which a fixed
    # response script can't drive deterministically. `rubric=None` makes
    # it parse the judge's raw numeric answer instead of a rubric label.
    return LLMJudge(
        judge_name="Faithfulness",
        instructions="Rate how faithful the response is to the context, from 0.0 to 1.0.",
        rubric=None,
        threshold=threshold,
        llm=llm,
    )


@pytest.fixture
def result() -> EvalResult:
    return _build_result()


# --------------------------------------------------------------------------
# Core mapping behaviour
# --------------------------------------------------------------------------


def test_to_openeval_builds_paired_suite_and_result_set(result):
    out = to_openeval(result)
    suite, result_set = out["suite"], out["result_set"]

    assert suite is not None
    assert len(suite["test_cases"]) == 3
    assert [tc["id"] for tc in suite["test_cases"]] == ["tc_0", "tc_1", "tc_2"]
    assert result_set["suite_id"] == suite["id"]
    assert [r["test_case_id"] for r in result_set["results"]] == [
        "tc_0",
        "tc_1",
        "tc_2",
    ]


def test_referential_integrity_holds_by_construction(result):
    out = to_openeval(result)
    suite_ids = {tc["id"] for tc in out["suite"]["test_cases"]}
    result_ids = {r["test_case_id"] for r in out["result_set"]["results"]}
    assert result_ids <= suite_ids


def test_documents_validate_against_real_openeval_spec(result):
    out = to_openeval(result)

    suite_validation = validate_suite(out["suite"])
    assert suite_validation.valid, suite_validation.errors

    result_set_validation = validate_result_set(out["result_set"])
    assert result_set_validation.valid, result_set_validation.errors


def test_null_score_maps_to_null_and_failed(result):
    out = to_openeval(result)
    row1 = out["result_set"]["results"][1]  # empty reference_ids -> RecallAtK is None

    recall_gr = next(
        gr for gr in row1["grader_results"] if gr["grader_id"] == "recall"
    )
    assert recall_gr["score"] is None
    assert recall_gr["passed"] is False
    # error, when the full per-row detail is available, is not dropped.
    assert recall_gr["reason"]


def test_null_score_excluded_from_grader_aggregate(result):
    out = to_openeval(result)
    by_grader = out["result_set"]["summary"]["by_grader"]["recall"]
    # Two scored rows (tc_0, tc_2), both 1.0 -- the null row must not be
    # counted in a denominator it was never part of (Rule 6: a null score
    # is "not verified", not a scored failure), and must not be silently
    # dropped either -- it shows up as `skipped`, not folded into `failed`.
    assert by_grader["avg_score"] == 1.0
    assert by_grader["passed"] == 2
    assert by_grader["failed"] == 0
    assert by_grader["skipped"] == 1


def test_score_range_normalization_and_raw_score_preserved(result):
    out = to_openeval(result)
    row0 = out["result_set"]["results"][0]

    length_gr = next(
        gr
        for gr in row0["grader_results"]
        if gr["grader_id"] == "answer_length_quality"
    )
    # "Paris is the capital of France." = 6 words -> raw = min(5, 1+6) = 5.0
    # normalized over score_range (1.0, 5.0): (5.0 - 1.0) / (5.0 - 1.0) = 1.0
    assert length_gr["metadata"]["openeval"]["raw_score"] == 5.0
    assert length_gr["score"] == 1.0
    assert 0.0 <= length_gr["score"] <= 1.0


def test_score_range_normalization_below_threshold(result):
    out = to_openeval(result)
    row1 = out["result_set"]["results"][1]

    length_gr = next(
        gr
        for gr in row1["grader_results"]
        if gr["grader_id"] == "answer_length_quality"
    )
    # "No." = 1 word -> raw = 1 + 1 = 2.0, normalized = (2-1)/4 = 0.25
    assert length_gr["metadata"]["openeval"]["raw_score"] == 2.0
    assert length_gr["score"] == 0.25
    # raw 2.0 < threshold 3.0 -> fails on ragrank's own (unnormalized) terms
    assert length_gr["passed"] is False


def test_unit_range_metric_has_no_raw_score_metadata(result):
    out = to_openeval(result)
    row0 = out["result_set"]["results"][0]
    faithfulness_gr = next(
        gr for gr in row0["grader_results"] if gr["grader_id"] == "faithfulness"
    )
    # score_range is already (0.0, 1.0): nothing to preserve.
    assert "metadata" not in faithfulness_gr or "openeval" not in faithfulness_gr.get(
        "metadata", {}
    )
    assert faithfulness_gr["score"] == 0.9
    assert faithfulness_gr["passed"] is True  # 0.9 >= threshold 0.7


def test_threshold_pass_fail_uses_raw_not_normalized_score(result):
    out = to_openeval(result)
    row1_faithfulness = next(
        gr
        for gr in out["result_set"]["results"][1]["grader_results"]
        if gr["grader_id"] == "faithfulness"
    )
    assert row1_faithfulness["score"] == 0.4
    assert row1_faithfulness["passed"] is False  # 0.4 < threshold 0.7


def test_row_passed_is_and_of_grader_results(result):
    out = to_openeval(result)
    row0 = out["result_set"]["results"][0]
    # recall has no threshold, so it is treated as passed=True whenever it
    # produces a score at all (see README's documented convention).
    grader_passed = {gr["grader_id"]: gr["passed"] for gr in row0["grader_results"]}
    assert grader_passed == {
        "recall": True,
        "answer_length_quality": True,
        "faithfulness": True,
    }
    assert row0["passed"] is True

    row1 = out["result_set"]["results"][1]
    grader_passed_1 = {gr["grader_id"]: gr["passed"] for gr in row1["grader_results"]}
    # recall null (nothing relevant to find) -> False; length quality below
    # threshold -> False; faithfulness below threshold -> False.
    assert grader_passed_1 == {
        "recall": False,
        "answer_length_quality": False,
        "faithfulness": False,
    }
    assert row1["passed"] is False


def test_no_threshold_metric_with_score_counts_as_passed():
    """A metric with no threshold that *did* produce a score should not be
    treated as a scored failure -- EvalPort's GraderResult.passed has no
    "n/a" value, so this adapter's documented convention is: produced a
    score, no criterion to fail against -> passed."""
    dataset = Dataset(
        question=["q"], context=[["c"]], response=["r"], reference=["r"]
    )
    result = EvalResult(
        llm=FakeLLM(),
        metrics=[exact_match],
        dataset=dataset,
        scores=[[1.0]],
        response_time=0.1,
    )
    out = to_openeval(result)
    gr = out["result_set"]["results"][0]["grader_results"][0]
    assert gr["score"] == 1.0
    assert gr["passed"] is True


# --------------------------------------------------------------------------
# Grader shape / custom-type validity
# --------------------------------------------------------------------------


def test_graders_are_custom_type_with_handler(result):
    out = to_openeval(result)
    for grader in out["suite"]["graders"]:
        assert grader["type"] == "custom"
        assert grader["params"]["handler"] == f"ragrank:{grader['id']}"


def test_grader_ids_are_stable_slugs(result):
    out = to_openeval(result)
    ids = {g["id"] for g in out["suite"]["graders"]}
    assert ids == {"recall", "answer_length_quality", "faithfulness"}


# --------------------------------------------------------------------------
# Explicit test_case_ids escape hatch (pairing against a real suite)
# --------------------------------------------------------------------------


def test_explicit_test_case_ids_skips_synthetic_suite(result):
    out = to_openeval(
        result,
        suite_id="real_suite_1",
        run_id="run_1",
        test_case_ids=["qa_paris", "qa_wakanda", "qa_berlin"],
    )
    assert out["suite"] is None
    assert out["result_set"]["suite_id"] == "real_suite_1"
    assert out["result_set"]["run_id"] == "run_1"
    ids = [r["test_case_id"] for r in out["result_set"]["results"]]
    assert ids == ["qa_paris", "qa_wakanda", "qa_berlin"]

    validation = validate_result_set(out["result_set"])
    assert validation.valid, validation.errors


def test_explicit_test_case_ids_requires_suite_id(result):
    with pytest.raises(ValueError):
        to_openeval(result, test_case_ids=["a", "b", "c"])


def test_explicit_test_case_ids_length_mismatch_raises(result):
    with pytest.raises(ValueError):
        build_result_set(result, suite_id="s1", test_case_ids=["only_one"])


# --------------------------------------------------------------------------
# Works without the full per-row `results` detail (scores-only EvalResult)
# --------------------------------------------------------------------------


def test_works_without_results_detail():
    dataset = Dataset(
        question=["q1", "q2"],
        context=[["c1"], ["c2"]],
        response=["r1", "r2"],
        reference=["r1", "different"],
    )
    result = EvalResult(
        llm=FakeLLM(),
        metrics=[exact_match],
        dataset=dataset,
        scores=[[1.0, 0.0]],
        response_time=0.5,
        # results=None (default) -- no MetricResult detail available.
    )
    out = to_openeval(result)
    validation = validate_result_set(out["result_set"])
    assert validation.valid, validation.errors

    grader_results = [
        r["grader_results"][0] for r in out["result_set"]["results"]
    ]
    assert grader_results[0]["score"] == 1.0 and grader_results[0]["passed"]
    assert grader_results[1]["score"] == 0.0
    # exact_match has no threshold, so a produced score (even 0.0) passes.
    assert grader_results[1]["passed"] is True
    # no `reason` to report without the detail rows.
    assert "reason" not in grader_results[0]


# --------------------------------------------------------------------------
# build_suite() / build_result_set() called independently
# --------------------------------------------------------------------------


def test_build_suite_is_reusable_independently(result):
    suite = build_suite(result, suite_id="fixed_id")
    assert suite["id"] == "fixed_id"
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_suite_retrieval_context_matches_ragrank_context(result):
    suite = build_suite(result, suite_id="s")
    tc0 = suite["test_cases"][0]
    assert tc0["retrieval_context"] == [
        "France is a country in Western Europe. Its capital is Paris."
    ]
    tc1 = suite["test_cases"][1]
    assert "retrieval_context" not in tc1  # empty context list -> omitted


# --------------------------------------------------------------------------
# from_openeval(): import direction
# --------------------------------------------------------------------------


def test_from_openeval_round_trip_without_result_set(result):
    out = to_openeval(result)
    dataset = from_openeval(out["suite"])

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 3
    assert dataset.question == [
        "What is the capital of France?",
        "What is the capital of Wakanda?",
        "What is the capital of Germany?",
    ]
    # No ResultSet supplied -> response left empty, ready for a fresh run.
    assert dataset.response == ["", "", ""]
    assert dataset.reference[0] == "Paris"


def test_from_openeval_with_result_set_fills_in_actual_output(result):
    out = to_openeval(result)
    dataset = from_openeval(out["suite"], out["result_set"])

    assert dataset.response == [
        "Paris is the capital of France.",
        "No.",
        "The capital of Germany is Berlin.",
    ]


def test_from_openeval_empty_suite():
    dataset = from_openeval({"version": "1.0.0", "id": "empty", "test_cases": []})
    assert len(dataset) == 0
