import json
from pathlib import Path

from openeval.validate import validate_result_set

from clawbench_openeval_adapter import (
    INTERCEPTION_GRADER_ID,
    JUDGE_GRADER_ID,
    run_to_result,
    to_openeval,
    from_openeval,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# run_to_result: one run-meta.json (+ optional judge verdict) -> one Result
# ---------------------------------------------------------------------------


def test_run_to_result_intercepted_and_matched():
    run_meta = load("run-meta-pass.json")
    judge = load("judge_llm-pass.json")

    result = run_to_result(run_meta, judge, rubric="lenient")

    assert result["test_case_id"] == "myrecipes/leave-review"
    assert result["passed"] is True
    assert result["duration_ms"] == 187000

    graders = {g["grader_id"]: g for g in result["grader_results"]}
    assert graders[INTERCEPTION_GRADER_ID]["passed"] is True
    assert graders[INTERCEPTION_GRADER_ID]["score"] == 1.0
    assert graders[JUDGE_GRADER_ID]["passed"] is True
    assert graders[JUDGE_GRADER_ID]["score"] == 1.0
    assert "pinch of salt" in graders[JUDGE_GRADER_ID]["reason"]

    assert result["metadata"]["result_category"] == "passed"
    assert result["metadata"]["model"] == "claude-sonnet-4-6"
    assert result["metadata"]["rubric"] == "lenient"
    assert "seasoning" in result["metadata"]["instruction"]


def test_run_to_result_not_intercepted_never_judged():
    run_meta = load("run-meta-fail.json")

    result = run_to_result(run_meta, judge=None)

    assert result["passed"] is False
    assert len(result["grader_results"]) == 1  # no judge grader when never judged
    interception = result["grader_results"][0]
    assert interception["grader_id"] == INTERCEPTION_GRADER_ID
    assert interception["passed"] is False
    assert result["metadata"]["failure_category"] == "time_limit_exceeded"


def test_run_to_result_intercepted_but_judge_mismatch():
    run_meta = load("run-meta-mismatch.json")
    judge = load("judge_llm-mismatch.json")

    result = run_to_result(run_meta, judge, rubric="lenient")

    assert result["passed"] is False  # intercepted AND judge_match -- judge said false
    graders = {g["grader_id"]: g for g in result["grader_results"]}
    assert graders[INTERCEPTION_GRADER_ID]["passed"] is True
    assert graders[JUDGE_GRADER_ID]["passed"] is False
    assert graders[JUDGE_GRADER_ID]["score"] == 0.0
    assert "paperback" in graders[JUDGE_GRADER_ID]["reason"]


def test_run_to_result_judge_could_not_decide():
    run_meta = load("run-meta-pass.json")
    judge = {"match": None, "reason": "judge returned non-JSON output"}

    result = run_to_result(run_meta, judge)

    # docs/scoring.md: match: null is "treated as false in aggregate".
    assert result["passed"] is False
    judge_gr = result["grader_results"][1]
    assert judge_gr["passed"] is False
    assert judge_gr["score"] is None  # null score, not fabricated 0.0


def test_run_to_result_requires_test_case_or_task_id():
    try:
        run_to_result({"intercepted": True})
    except ValueError as e:
        assert "test_case" in str(e)
    else:
        raise AssertionError("expected ValueError for a run_meta with no test_case/task_id")


def test_run_to_result_task_id_fallback_strips_run_suffix():
    # ClawBench's real task_id carries a "#<run suffix>" (see run-meta-pass.json's
    # task_id "myrecipes/leave-review#0001" vs. its test_case
    # "myrecipes/leave-review") -- when test_case is absent, the task_id fallback
    # must strip that suffix so test_case_id stays consistent/aggregatable with
    # what test_case-derived IDs from other runs of the same task look like.
    run_meta = {"task_id": "myrecipes/leave-review#0001", "intercepted": True}
    result = run_to_result(run_meta)
    assert result["test_case_id"] == "myrecipes/leave-review"


def test_run_to_result_task_id_fallback_without_run_suffix_unchanged():
    # A task_id with no "#" suffix at all should pass through unchanged.
    run_meta = {"task_id": "myrecipes/leave-review", "intercepted": True}
    result = run_to_result(run_meta)
    assert result["test_case_id"] == "myrecipes/leave-review"


# ---------------------------------------------------------------------------
# to_openeval: rescore-summary.json (+ optional run_metas) -> ResultSet
# ---------------------------------------------------------------------------


def test_to_openeval_without_run_metas_is_spec_valid():
    rescore_summary = load("rescore-summary.json")

    result_set = to_openeval(
        rescore_summary,
        run_id="batch-20260830-140000",
        started_at="2026-08-30T14:00:00Z",
        completed_at="2026-08-30T14:32:07Z",
    )

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors
    assert result_set["suite_id"] == "clawbench_batch-20260830-140000"
    assert len(result_set["results"]) == 3
    assert result_set["summary"]["total"] == 3
    assert result_set["summary"]["passed"] == 1  # only myrecipes/leave-review fully passed
    assert result_set["summary"]["failed"] == 2
    assert result_set["metadata"]["clawbench_n_intercepted"] == 2

    # No run_metas given -> results still valid, just without instruction/model enrichment.
    by_id = {r["test_case_id"]: r for r in result_set["results"]}
    assert "instruction" not in by_id["myrecipes/leave-review"].get("metadata", {})


def test_to_openeval_with_run_metas_enriches_results():
    rescore_summary = load("rescore-summary.json")
    run_metas = {
        "myrecipes/leave-review": load("run-meta-pass.json"),
        "shopmart/checkout-flow": load("run-meta-fail.json"),
        "bookhaven/gift-wrap": load("run-meta-mismatch.json"),
    }

    result_set = to_openeval(
        rescore_summary,
        run_metas,
        run_id="batch-20260830-140000",
        started_at="2026-08-30T14:00:00Z",
    )

    assert validate_result_set(result_set).valid

    by_id = {r["test_case_id"]: r for r in result_set["results"]}

    passed_result = by_id["myrecipes/leave-review"]
    assert passed_result["passed"] is True
    assert passed_result["metadata"]["model"] == "claude-sonnet-4-6"
    assert "seasoning" in passed_result["metadata"]["instruction"]

    not_intercepted = by_id["shopmart/checkout-flow"]
    assert not_intercepted["passed"] is False
    assert len(not_intercepted["grader_results"]) == 1
    assert not_intercepted["metadata"]["failure_category"] == "time_limit_exceeded"

    mismatched = by_id["bookhaven/gift-wrap"]
    assert mismatched["passed"] is False
    assert mismatched["grader_results"][0]["passed"] is True  # was intercepted
    assert mismatched["grader_results"][1]["passed"] is False  # judge said no match


def test_to_openeval_requires_run_id_and_started_at():
    rescore_summary = load("rescore-summary.json")
    try:
        to_openeval(rescore_summary)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("run_id/started_at should be required keyword args")


def test_to_openeval_rejects_rubric_not_in_rescore_summary():
    # This fixture's rescore-summary.json only ran the "lenient" rubric
    # (rubrics: ["lenient"]) -- asking for "strict" must raise, not silently
    # produce a ResultSet where every task_row's match_strict lookup misses
    # and every run comes out looking never-judged/failed.
    rescore_summary = load("rescore-summary.json")
    try:
        to_openeval(
            rescore_summary,
            run_id="r1",
            started_at="2026-08-30T14:00:00Z",
            rubric="strict",
        )
    except ValueError as e:
        assert "strict" in str(e)
        assert "lenient" in str(e)
    else:
        raise AssertionError("expected ValueError for a rubric not in rescore_summary['rubrics']")


def test_to_openeval_accepts_rubric_that_is_in_rescore_summary():
    rescore_summary = load("rescore-summary.json")
    result_set = to_openeval(
        rescore_summary,
        run_id="r1",
        started_at="2026-08-30T14:00:00Z",
        rubric="lenient",
    )
    assert result_set["metadata"]["clawbench_rubric"] == "lenient"
    assert result_set["summary"]["passed"] == 1


def test_to_openeval_empty_tasks_still_valid_shape():
    result_set = to_openeval(
        {"batch_dir": "/tmp/empty-batch", "tasks": [], "rubrics": ["lenient"]},
        run_id="empty_run",
        started_at="2026-08-30T14:00:00Z",
    )
    assert result_set["results"] == []
    assert result_set["summary"]["total"] == 0
    assert result_set["summary"]["pass_rate"] == 0.0
    # validate_result_set requires a non-empty `results` list, so an empty
    # ResultSet is a structurally valid dict but not spec-valid on its own --
    # documenting that boundary rather than silently validating it.
    assert not validate_result_set(result_set).valid


# ---------------------------------------------------------------------------
# from_openeval: reverse direction (lossy, best-effort)
# ---------------------------------------------------------------------------


def test_from_openeval_round_trips_what_it_can():
    rescore_summary = load("rescore-summary.json")
    run_metas = {
        "myrecipes/leave-review": load("run-meta-pass.json"),
        "bookhaven/gift-wrap": load("run-meta-mismatch.json"),
    }
    del rescore_summary["tasks"][1]  # drop the not-intercepted row for this check

    result_set = to_openeval(
        rescore_summary,
        run_metas,
        run_id="r1",
        started_at="2026-08-30T14:00:00Z",
    )

    rows = from_openeval(result_set)
    by_test_case = {r["test_case"]: r for r in rows}

    passed_row = by_test_case["myrecipes/leave-review"]
    assert passed_row["intercepted"] is True
    assert passed_row["match"] is True
    assert passed_row["result_category"] == "passed"

    mismatched_row = by_test_case["bookhaven/gift-wrap"]
    assert mismatched_row["intercepted"] is True
    assert mismatched_row["match"] is False
    assert mismatched_row["failure_category"] == "wrong_payload"


def test_from_openeval_empty_result_set():
    assert from_openeval({"results": []}) == []
