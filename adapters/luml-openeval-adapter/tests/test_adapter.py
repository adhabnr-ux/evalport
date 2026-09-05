from dataclasses import dataclass, field
from typing import Any

import pytest

from openeval.validate import validate_result_set, validate_suite

from luml_openeval_adapter import (
    eval_item_from_test_case,
    eval_item_to_test_case,
    from_openeval,
    from_openeval_suite,
    to_openeval,
    to_openeval_suite,
)


# ---------------------------------------------------------------------------
# Stand-ins for luml's real dataclasses (sdk/python/sdk/luml/experiments/
# evaluation/types.py), reproduced field-for-field from the real source so
# these tests exercise the exact shape the real EvalItem/EvalResult/
# EvalResults would present -- not a guess. luml_sdk itself is not
# installable here (requires Python >=3.12, not published to PyPI), so this
# adapter is tested against these stand-ins via its dict-or-attribute _get()
# duck typing, the same approach every other adapter in this repo uses for a
# target library that isn't a hard dependency.
# ---------------------------------------------------------------------------


@dataclass
class FakeEvalItem:
    id: str
    inputs: dict[str, Any]
    expected_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeEvalResult:
    eval_item: FakeEvalItem
    model_response: Any
    scores: dict[str, Any]
    trace_id: str


@dataclass
class FakeEvalResults:
    results: list[FakeEvalResult]
    aggregated_scores: dict[str, Any]
    dataset_id: str


REASONING_SUFFIX = "_reasoning"  # luml.experiments.evaluation.types.REASONING_SUFFIX


# ---------------------------------------------------------------------------
# eval_item_to_test_case / eval_item_from_test_case
# ---------------------------------------------------------------------------


def test_eval_item_to_test_case_single_key_input_flattens_to_plain_string():
    item = FakeEvalItem(id="case_1", inputs={"question": "What is 2+2?"}, expected_output="4")
    tc = eval_item_to_test_case(item)

    assert tc["id"] == "case_1"
    assert tc["input"] == "What is 2+2?"
    assert tc["expected_output"] == "4"
    assert tc["graders"] == [
        {
            "id": "gr_luml_scorer",
            "type": "custom",
            "description": tc["graders"][0]["description"],
            "params": {"handler": "luml:scorer"},
        }
    ]
    assert tc["metadata"]["luml"]["inputs"] == {"question": "What is 2+2?"}


def test_eval_item_to_test_case_multi_key_input_flattens_to_json():
    item = FakeEvalItem(id="case_2", inputs={"question": "Summarize", "context": "long doc..."})
    tc = eval_item_to_test_case(item)

    import json

    assert json.loads(tc["input"]) == {"context": "long doc...", "question": "Summarize"}
    assert tc["metadata"]["luml"]["inputs"] == {"question": "Summarize", "context": "long doc..."}


def test_eval_item_to_test_case_non_string_expected_output_preserved_in_metadata():
    item = FakeEvalItem(id="case_3", inputs={"q": "pick a number"}, expected_output=42)
    tc = eval_item_to_test_case(item)

    assert tc["expected_output"] == "42"
    assert tc["metadata"]["luml"]["expected_output_type"] == "int"
    assert tc["metadata"]["luml"]["expected_output_raw"] == 42


def test_eval_item_to_test_case_works_with_plain_dict_too():
    item = {"id": "case_4", "inputs": {"q": "hi"}, "expected_output": None, "metadata": {"source": "unit_test"}}
    tc = eval_item_to_test_case(item)

    assert tc["id"] == "case_4"
    assert tc["input"] == "hi"
    assert "expected_output" not in tc
    assert tc["metadata"]["luml"]["item_metadata"] == {"source": "unit_test"}


def test_eval_item_to_test_case_llm_judge_grader_type_with_params():
    item = FakeEvalItem(id="case_5", inputs={"q": "hi"})
    tc = eval_item_to_test_case(
        item,
        grader_type="llm_judge",
        grader_params={"model": "gpt-4o", "prompt": "Score {output} against {expected}"},
    )
    assert tc["graders"] == [
        {
            "id": "gr_luml_scorer",
            "type": "llm_judge",
            "params": {"model": "gpt-4o", "prompt": "Score {output} against {expected}"},
        }
    ]


def test_eval_item_to_test_case_requires_id():
    with pytest.raises(ValueError):
        eval_item_to_test_case(FakeEvalItem(id="", inputs={"q": "hi"}))


def test_eval_item_from_test_case_round_trip():
    item = FakeEvalItem(
        id="case_6",
        inputs={"question": "Summarize", "context": "long doc..."},
        expected_output={"summary": "short doc"},
        metadata={"difficulty": "hard"},
    )
    tc = eval_item_to_test_case(item)
    reconstructed = eval_item_from_test_case(tc)

    assert reconstructed["id"] == "case_6"
    assert reconstructed["inputs"] == {"question": "Summarize", "context": "long doc..."}
    assert reconstructed["expected_output"] == {"summary": "short doc"}
    assert reconstructed["metadata"] == {"difficulty": "hard"}


def test_eval_item_from_test_case_handles_foreign_test_case():
    tc = {"id": "tc1", "input": "hello", "expected_output": "world", "graders": ["g1"]}
    reconstructed = eval_item_from_test_case(tc)
    assert reconstructed == {
        "id": "tc1",
        "inputs": {"input": "hello"},
        "expected_output": "world",
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# to_openeval_suite / from_openeval_suite
# ---------------------------------------------------------------------------


def test_to_openeval_suite_builds_valid_suite():
    items = [
        FakeEvalItem(id="c1", inputs={"q": "2+2?"}, expected_output="4"),
        FakeEvalItem(id="c2", inputs={"q": "3+3?"}, expected_output="6"),
    ]
    suite = to_openeval_suite(items, dataset_id="arithmetic")

    assert suite["id"] == "luml_arithmetic"
    assert len(suite["test_cases"]) == 2
    assert suite["metadata"]["luml"] == {"dataset_id": "arithmetic"}
    assert [g["id"] for g in suite["graders"]] == ["gr_luml_scorer"]

    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_to_openeval_suite_empty_items_raises():
    with pytest.raises(ValueError):
        to_openeval_suite([])


def test_from_openeval_suite_round_trip():
    items = [FakeEvalItem(id="c1", inputs={"q": "hi"}, expected_output="hello")]
    suite = to_openeval_suite(items, dataset_id="ds1")
    reconstructed = from_openeval_suite(suite)

    assert reconstructed == [
        {"id": "c1", "inputs": {"q": "hi"}, "expected_output": "hello", "metadata": {}}
    ]


# ---------------------------------------------------------------------------
# to_openeval (EvalResults -> ResultSet)
# ---------------------------------------------------------------------------

STARTED_AT = "2026-09-05T12:00:00Z"


def _basic_eval_results():
    return FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "Is Paris the capital of France?"}, expected_output="yes"),
                model_response="Yes, Paris is the capital of France.",
                scores={"correctness": 0.95, "correctness_reasoning": "The answer is accurate and complete."},
                trace_id="abc123",
            ),
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c2", inputs={"q": "Is Berlin the capital of France?"}, expected_output="no"),
                model_response="No, Berlin is the capital of Germany.",
                scores={"correctness": 1.0, "correctness_reasoning": "Correctly identifies the mistake."},
                trace_id="def456",
            ),
        ],
        aggregated_scores={"correctness_mean": 0.975, "correctness_min": 0.95, "correctness_max": 1.0, "correctness_count": 2, "total_items": 2, "successful_items": 2},
        dataset_id="capitals",
    )


def test_to_openeval_builds_valid_result_set_with_llm_judge_reasoning():
    result_set = to_openeval(_basic_eval_results(), started_at=STARTED_AT)

    assert result_set["suite_id"] == "capitals"
    assert result_set["run_id"] == "luml_capitals"
    assert result_set["started_at"] == STARTED_AT
    assert len(result_set["results"]) == 2

    r1 = result_set["results"][0]
    assert r1["test_case_id"] == "c1"
    assert r1["passed"] is True
    assert r1["actual_output"] == "Yes, Paris is the capital of France."
    assert r1["grader_results"] == [
        {
            "grader_id": "correctness",
            "type": "luml_llm_judge",
            "score": 0.95,
            "passed": True,
            "reason": "The answer is accurate and complete.",
        }
    ]

    assert result_set["summary"] == {
        "total": 2,
        "passed": 2,
        "failed": 0,
        "pass_rate": 1.0,
        "avg_score": pytest.approx(0.975),
    }
    assert result_set["metadata"]["luml"]["dataset_id"] == "capitals"
    assert result_set["metadata"]["luml"]["aggregated_scores"] == _basic_eval_results().aggregated_scores

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_to_openeval_plain_numeric_scorer_has_no_reasoning():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "hi"}),
                model_response="hello",
                scores={"exact_match": 1},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    result_set = to_openeval(eval_results, started_at=STARTED_AT)
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["type"] == "luml_scorer"
    assert "reason" not in gr
    assert gr["score"] == 1.0
    assert gr["passed"] is True


def test_to_openeval_whole_item_error_becomes_result_error_not_zero_score():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "hi"}),
                model_response=None,
                scores={"error": "inference_fn raised: connection timed out"},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    result_set = to_openeval(eval_results, started_at=STARTED_AT)
    result = result_set["results"][0]

    assert result["passed"] is False
    assert result["grader_results"] == []
    assert result["error"] == {
        "type": "runner_error",
        "message": "inference_fn raised: connection timed out",
    }
    assert "actual_output" not in result

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_to_openeval_per_scorer_error_preserved_alongside_real_scores():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "hi"}),
                model_response="hello",
                scores={"relevancy": 0.8, "__error__toxicity": "judge model returned invalid JSON"},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    result_set = to_openeval(eval_results, started_at=STARTED_AT)
    grader_results = result_set["results"][0]["grader_results"]

    by_id = {gr["grader_id"]: gr for gr in grader_results}
    assert by_id["relevancy"]["score"] == 0.8
    assert by_id["toxicity"] == {
        "grader_id": "toxicity",
        "type": "luml_scorer_error",
        "score": None,
        "passed": False,
        "reason": "judge model returned invalid JSON",
    }
    # An item with one real passing score and one scorer error should still
    # be considered failed overall, since not every grader passed.
    assert result_set["results"][0]["passed"] is False

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_to_openeval_clamps_out_of_range_scores_and_preserves_raw():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "hi"}),
                model_response="hello",
                scores={"custom_metric": 1.5},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    result_set = to_openeval(eval_results, started_at=STARTED_AT)
    gr = result_set["results"][0]["grader_results"][0]

    assert gr["score"] == 1.0
    assert gr["metadata"]["luml_raw_score"] == 1.5

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_to_openeval_unrecognized_score_type_preserved_not_dropped():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "hi"}),
                model_response="hello",
                scores={"weird_scorer": {"nested": "value"}},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    result_set = to_openeval(eval_results, started_at=STARTED_AT)
    result = result_set["results"][0]

    assert result["grader_results"] == []
    assert result["passed"] is False
    assert result["metadata"]["luml"]["unrecognized_scores"] == {"weird_scorer": {"nested": "value"}}

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_to_openeval_custom_threshold():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "hi"}),
                model_response="hello",
                scores={"quality": 0.6},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    default_result = to_openeval(eval_results, started_at=STARTED_AT)
    strict_result = to_openeval(eval_results, started_at=STARTED_AT, threshold=0.7)

    assert default_result["results"][0]["grader_results"][0]["passed"] is True
    assert strict_result["results"][0]["grader_results"][0]["passed"] is False


def test_to_openeval_empty_results_raises():
    eval_results = FakeEvalResults(results=[], aggregated_scores={}, dataset_id="ds1")
    with pytest.raises(ValueError):
        to_openeval(eval_results, started_at=STARTED_AT)


def test_to_openeval_missing_eval_item_id_raises():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="", inputs={}),
                model_response="hi",
                scores={"x": 1.0},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    with pytest.raises(ValueError):
        to_openeval(eval_results, started_at=STARTED_AT)


def test_to_openeval_explicit_run_id_and_completed_at():
    result_set = to_openeval(
        _basic_eval_results(),
        started_at=STARTED_AT,
        run_id="ci_build_42",
        completed_at="2026-09-05T12:05:00Z",
    )
    assert result_set["run_id"] == "ci_build_42"
    assert result_set["completed_at"] == "2026-09-05T12:05:00Z"


def test_to_openeval_works_with_plain_dicts_too():
    eval_results = {
        "results": [
            {
                "eval_item": {"id": "c1", "inputs": {"q": "hi"}},
                "model_response": "hello",
                "scores": {"exact_match": True},
                "trace_id": "t1",
            }
        ],
        "aggregated_scores": {},
        "dataset_id": "ds1",
    }
    result_set = to_openeval(eval_results, started_at=STARTED_AT)
    assert result_set["results"][0]["grader_results"][0]["score"] == 1.0

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


# ---------------------------------------------------------------------------
# from_openeval (ResultSet -> EvalResults kwargs)
# ---------------------------------------------------------------------------


def test_from_openeval_round_trip_of_scores_and_reasoning():
    result_set = to_openeval(_basic_eval_results(), started_at=STARTED_AT)
    reconstructed = from_openeval(result_set)

    assert reconstructed["dataset_id"] == "capitals"
    assert reconstructed["aggregated_scores"] == _basic_eval_results().aggregated_scores
    assert len(reconstructed["results"]) == 2

    r1 = reconstructed["results"][0]
    assert r1["eval_item"] == {"id": "c1", "inputs": {}, "expected_output": None, "metadata": {}}
    assert r1["model_response"] == "Yes, Paris is the capital of France."
    assert r1["scores"] == {"correctness": 0.95, "correctness_reasoning": "The answer is accurate and complete."}
    assert r1["trace_id"] == "abc123"


def test_from_openeval_reconstructs_whole_item_error():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "hi"}),
                model_response=None,
                scores={"error": "boom"},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    result_set = to_openeval(eval_results, started_at=STARTED_AT)
    reconstructed = from_openeval(result_set)
    assert reconstructed["results"][0]["scores"] == {"error": "boom"}


def test_from_openeval_reconstructs_per_scorer_error():
    eval_results = FakeEvalResults(
        results=[
            FakeEvalResult(
                eval_item=FakeEvalItem(id="c1", inputs={"q": "hi"}),
                model_response="hello",
                scores={"relevancy": 0.8, "__error__toxicity": "judge model returned invalid JSON"},
                trace_id="t1",
            )
        ],
        aggregated_scores={},
        dataset_id="ds1",
    )
    result_set = to_openeval(eval_results, started_at=STARTED_AT)
    reconstructed = from_openeval(result_set)
    assert reconstructed["results"][0]["scores"] == {
        "relevancy": 0.8,
        "__error__toxicity": "judge model returned invalid JSON",
    }


def test_from_openeval_handles_foreign_result_set():
    result_set = {
        "version": "1.0.0",
        "suite_id": "s1",
        "run_id": "r1",
        "started_at": STARTED_AT,
        "results": [
            {
                "test_case_id": "tc1",
                "passed": True,
                "actual_output": "hi",
                "grader_results": [
                    {"grader_id": "g1", "type": "exact_match", "score": 1.0, "passed": True}
                ],
            }
        ],
    }
    reconstructed = from_openeval(result_set)
    assert reconstructed == {
        "results": [
            {
                "eval_item": {"id": "tc1", "inputs": {}, "expected_output": None, "metadata": {}},
                "model_response": "hi",
                "scores": {"g1": 1.0},
                "trace_id": "",
            }
        ],
        "aggregated_scores": {},
        "dataset_id": "s1",
    }
