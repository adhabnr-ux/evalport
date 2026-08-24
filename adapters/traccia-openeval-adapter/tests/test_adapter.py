"""Tests for traccia-openeval-adapter.

These call the REAL `traccia.eval.evaluate()` (installed from PyPI,
`traccia==0.1.28`) and validate the converted output against the REAL
`openeval.validate.validate_result_set()` (installed from `evalport-sdk`).
No mocking of either package's public behavior -- only traccia's own
network/tracing side effects are irrelevant here since every call below
uses `persist=False`.
"""
from __future__ import annotations

from openeval.validate import validate_result_set

from traccia_openeval_adapter import (
    clamp_score,
    results_to_openeval,
    row_to_result,
    score_to_grader_result,
)


def _run_real_evaluate(**overrides):
    from traccia.eval import evaluate

    def task(row):
        q = row["input"]["q"]
        return {"2+2": "4", "3+3": "7", "10-1": "9"}.get(q, "unknown")

    kwargs = dict(
        name="adapter-test-experiment",
        data=[
            {"input": {"q": "2+2"}, "expected": "4"},
            {"input": {"q": "3+3"}, "expected": "6"},  # deliberately wrong -> fails
            {"input": {"q": "10-1"}, "expected": "9"},
        ],
        task=task,
        scorers=["exact_match"],
        persist=False,
        progress=False,
    )
    kwargs.update(overrides)
    return evaluate(**kwargs)


def test_real_evaluate_shape_sanity():
    """Guard against traccia changing EvaluateResult's shape out from under
    this adapter -- if this fails, the mapping in __init__.py needs a look
    before anything else here is trusted."""
    result = _run_real_evaluate()
    assert result.name == "adapter-test-experiment"
    assert len(result.rows) == 3
    row0 = result.rows[0]
    assert set(["item_id", "input", "expected_output", "panels"]).issubset(row0.keys())
    assert len(row0["panels"]) == 1
    panel0 = row0["panels"][0]
    assert set(["output", "scores", "passed", "latency_ms"]).issubset(panel0.keys())
    assert panel0["scores"][0]["name"] == "exact_match"
    assert result.experiment_id is None  # persist=False never allocates one


def test_results_to_openeval_validates_against_real_openeval_sdk():
    result = _run_real_evaluate()
    result_set = results_to_openeval(result, suite_id="my-experiment")

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_results_to_openeval_field_mapping():
    result = _run_real_evaluate()
    result_set = results_to_openeval(result, suite_id="my-experiment")

    assert result_set["suite_id"] == "my-experiment"
    assert result_set["version"]
    assert result_set["started_at"]
    assert len(result_set["results"]) == 3

    passing = next(r for r in result_set["results"] if r["actual_output"] == "4")
    assert passing["passed"] is True
    assert passing["grader_results"][0]["grader_id"] == "exact_match"
    assert passing["grader_results"][0]["type"] == "exact_match"
    assert passing["grader_results"][0]["score"] == 1.0
    assert passing["grader_results"][0]["passed"] is True

    failing = next(r for r in result_set["results"] if r["actual_output"] == "7")
    assert failing["passed"] is False
    assert failing["grader_results"][0]["passed"] is False
    assert failing["grader_results"][0]["score"] == 0.0
    assert failing["grader_results"][0]["reason"] == "mismatch"

    assert result_set["summary"]["total"] == 3
    assert result_set["summary"]["passed"] == 2
    assert result_set["summary"]["failed"] == 1
    assert result_set["metadata"]["traccia_aggregates"]["scored_count"] == 3


def test_run_id_is_minted_when_not_persisted():
    """persist=False -> traccia's own experiment_id is None; the adapter
    must not leave EvalPort's required run_id field empty or None."""
    result = _run_real_evaluate()
    assert result.experiment_id is None
    result_set = results_to_openeval(result, suite_id="s1")
    assert result_set["run_id"]
    assert result_set["run_id"].startswith("traccia-local-")


def test_explicit_run_id_and_started_at_are_respected():
    result = _run_real_evaluate()
    result_set = results_to_openeval(
        result, suite_id="s1", run_id="exp-42", started_at="2026-01-01T00:00:00+00:00"
    )
    assert result_set["run_id"] == "exp-42"
    assert result_set["started_at"] == "2026-01-01T00:00:00+00:00"


def test_no_scorers_item_treated_as_passed():
    """A panel with no scorers has panel['passed'] is None in real traccia
    (not False) -- confirm the adapter doesn't misrepresent 'unscored' as
    'failed'."""
    result = _run_real_evaluate(scorers=[])
    for panel_row in result.rows:
        assert panel_row["panels"][0]["passed"] is None  # confirms real traccia behavior
    result_set = results_to_openeval(result, suite_id="s1")
    for r in result_set["results"]:
        assert r["grader_results"] == []
        assert r["passed"] is True
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_multi_panel_row_raises_clear_error():
    """traccia's evaluate() only ever emits one panel per row today; this
    locks in that assumption so a future multi-panel release fails loudly
    here instead of silently dropping panels[1:]."""
    row = {
        "item_id": "x1",
        "input": {"q": "hi"},
        "expected_output": "hi",
        "panels": [
            {"output": "hi", "scores": [], "passed": True, "latency_ms": 1.0},
            {"output": "hi", "scores": [], "passed": True, "latency_ms": 1.0},
        ],
    }
    try:
        row_to_result(row)
        assert False, "expected ValueError for multi-panel row"
    except ValueError as exc:
        assert "panels" in str(exc)


def test_score_to_grader_result_preserves_platform_scorer_fields():
    """Fields with no EvalPort GraderResult slot (scorer_id, model,
    latency_ms, cost_usd, usage, config) are preserved under metadata
    rather than dropped."""
    score = {
        "scorer_id": "sc_123",
        "scorer_name": "llm-judge-v2",
        "name": "llm-judge-v2",
        "type": "llm_judge",
        "passed": True,
        "score": 0.87,
        "reason": "matches intent",
        "model": "gpt-4o",
        "latency_ms": 812.3,
        "cost_usd": 0.0021,
        "usage": {"prompt_tokens": 120, "completion_tokens": 40},
        "config": {"threshold": 0.8},
    }
    gr = score_to_grader_result(score)
    assert gr["grader_id"] == "sc_123"
    assert gr["type"] == "llm_judge"
    assert gr["score"] == 0.87
    assert gr["passed"] is True
    assert gr["reason"] == "matches intent"
    assert gr["metadata"]["model"] == "gpt-4o"
    assert gr["metadata"]["cost_usd"] == 0.0021
    assert gr["metadata"]["usage"]["prompt_tokens"] == 120


def test_clamp_score_out_of_range_and_non_numeric():
    assert clamp_score(1.4) == 1.0
    assert clamp_score(-0.2) == 0.0
    assert clamp_score(0.5) == 0.5
    assert clamp_score(True) == 1.0
    assert clamp_score(False) == 0.0
    assert clamp_score(None) is None
    assert clamp_score("n/a") is None


def test_empty_suite_id_raises():
    result = _run_real_evaluate()
    try:
        results_to_openeval(result, suite_id="")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_error_row_is_carried_through():
    def flaky_task(row):
        if row["input"]["q"] == "3+3":
            raise RuntimeError("boom")
        return {"2+2": "4", "10-1": "9"}.get(row["input"]["q"], "unknown")

    result = _run_real_evaluate(task=flaky_task)
    assert len(result.errors) == 1
    result_set = results_to_openeval(result, suite_id="s1")
    errored = next(r for r in result_set["results"] if r.get("error") is not None)
    assert "boom" in errored["error"]["message"]
    assert errored["passed"] is False
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors
