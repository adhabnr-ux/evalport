"""Tests for flashrag_openeval_adapter.

These exercise the real `flashrag.dataset.dataset.Item`/`Dataset` and
`flashrag.evaluator.evaluator.Evaluator` classes end to end (not a
hand-rolled stand-in), and validate adapter output against EvalPort's real
`openeval.validate.validate_suite()`/`validate_result_set()` -- not a mock
of either side.
"""
import importlib.util

import pytest

# This module imports real flashrag classes at module import time, so
# nothing in it can run without flashrag installed (CI's min-mode
# adapter-tests job installs evalport-sdk only -- see
# .github/workflows/ci.yml and .github/CONTRIBUTING.md). Skip the whole
# module when it is absent. Checked with find_spec rather than
# pytest.importorskip so that, with flashrag installed, a broken install or
# upstream API drift still fails loudly instead of being swallowed as a
# skip -- the same convention dspy-openeval-adapter and
# haystack-openeval-adapter use.
if importlib.util.find_spec("flashrag") is None:
    pytest.skip("flashrag not installed", allow_module_level=True)

from openeval.validate import validate_suite, validate_result_set
from openeval.types import OPENEVAL_VERSION

from flashrag.dataset.dataset import Dataset
from flashrag.evaluator.evaluator import Evaluator

from flashrag_openeval_adapter import (
    to_openeval_suite,
    to_openeval_resultset,
    from_openeval,
)


def _build_dataset():
    return Dataset(
        config={"dataset_name": "nq_test"},
        data=[
            {"id": "q1", "question": "What is 2+2?", "golden_answers": ["4"]},
            {"id": "q2", "question": "What is the capital of France?", "golden_answers": ["Paris", "paris"]},
            {"id": "q3", "question": "What is the capital of Japan?", "golden_answers": ["Tokyo"]},
        ],
    )


def _run_real_evaluator(dataset, preds, metrics):
    """Run FlashRAG's actual Evaluator (not a mock) and return its aggregate dict."""
    dataset.update_output("pred", preds)
    config = {
        "save_dir": "/tmp",
        "save_metric_score": False,
        "save_intermediate_data": False,
        "metrics": metrics,
        "dataset_name": "nq_test",
    }
    evaluator = Evaluator(config)
    return evaluator.evaluate(dataset)


# ---------------------------------------------------------------------------
# to_openeval_suite
# ---------------------------------------------------------------------------

def test_to_openeval_suite_basic_shape():
    dataset = _build_dataset()
    suite = to_openeval_suite(dataset, metric_names=["em", "f1"], suite_id="nq_test_suite")

    assert suite["id"] == "nq_test_suite"
    assert suite["version"] == OPENEVAL_VERSION
    assert len(suite["test_cases"]) == 3
    assert suite["test_cases"][0]["id"] == "q1"
    assert suite["test_cases"][0]["input"] == "What is 2+2?"
    assert suite["test_cases"][0]["expected_output"] == "4"
    # Multiple golden answers preserved under metadata, not collapsed/lost.
    assert suite["test_cases"][1]["metadata"]["flashrag_golden_answers"] == ["Paris", "paris"]
    assert suite["test_cases"][1]["expected_output"] == "Paris"

    assert [g["id"] for g in suite["graders"]] == ["gr_flashrag_em", "gr_flashrag_f1"]
    assert suite["graders"][0]["type"] == "custom"
    assert suite["graders"][0]["params"]["handler"] == "flashrag:em"
    for tc in suite["test_cases"]:
        assert tc["graders"] == ["gr_flashrag_em", "gr_flashrag_f1"]


def test_to_openeval_suite_validates_against_evalport_spec():
    dataset = _build_dataset()
    suite = to_openeval_suite(dataset, metric_names=["em"])
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_to_openeval_suite_multichoice_resolves_via_choices():
    # Mirrors flashrag.evaluator.metrics.BaseMetric.get_dataset_answer: when
    # `choices` is set, `golden_answers` holds indices into it, not literal text.
    dataset = Dataset(
        config={"dataset_name": "gaokao_test"},
        data=[
            {
                "id": "mc1",
                "question": "Pick the correct option.",
                "choices": ["Apple", "Banana", "Cherry"],
                "golden_answers": [2],
            }
        ],
    )
    suite = to_openeval_suite(dataset, metric_names=["acc"])
    tc = suite["test_cases"][0]
    assert tc["expected_output"] == "Cherry"
    assert tc["metadata"]["flashrag_choices"] == ["Apple", "Banana", "Cherry"]
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_to_openeval_suite_retrieval_context_from_retrieval_result():
    dataset = Dataset(
        config={"dataset_name": "rag_test"},
        data=[{"id": "r1", "question": "Q", "golden_answers": ["A"]}],
    )
    dataset.update_output(
        "retrieval_result",
        [[{"contents": "doc one text"}, {"contents": "doc two text"}]],
    )
    suite = to_openeval_suite(dataset, metric_names=["retrieval_recall"])
    assert suite["test_cases"][0]["retrieval_context"] == ["doc one text", "doc two text"]
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


# ---------------------------------------------------------------------------
# to_openeval_resultset -- against FlashRAG's REAL Evaluator, not a mock
# ---------------------------------------------------------------------------

def test_to_openeval_resultset_from_real_evaluator_em_and_f1():
    dataset = _build_dataset()
    aggregate = _run_real_evaluator(
        dataset,
        preds=["4", "London", "Tokyo"],  # q1 exact, q2 wrong, q3 exact
        metrics=["em", "f1"],
    )
    # Sanity: this really ran FlashRAG's own metric code, not a stand-in.
    assert aggregate["em"] == 2 / 3

    resultset = to_openeval_resultset(dataset, metric_names=["em", "f1"], run_id="run1", aggregate=aggregate)

    validation = validate_result_set(resultset)
    assert validation.valid, validation.errors

    by_id = {r["test_case_id"]: r for r in resultset["results"]}
    assert by_id["q1"]["passed"] is True
    assert by_id["q1"]["actual_output"] == "4"
    em_result = next(g for g in by_id["q1"]["grader_results"] if g["grader_id"] == "gr_flashrag_em")
    assert em_result["score"] == 1.0
    assert em_result["passed"] is True

    # q2's prediction ("London") is wrong -> em score 0.0, f1 0.0 too (no token overlap).
    assert by_id["q2"]["passed"] is False
    em_result_q2 = next(g for g in by_id["q2"]["grader_results"] if g["grader_id"] == "gr_flashrag_em")
    assert em_result_q2["score"] == 0.0
    assert em_result_q2["passed"] is False

    assert resultset["metadata"]["flashrag_aggregate"] == aggregate
    assert resultset["summary"]["total"] == 3


def test_to_openeval_resultset_partial_f1_with_lower_threshold():
    # f1 can be a fractional score (partial token overlap) -- confirm the
    # passed_threshold knob behaves as documented for continuous metrics.
    dataset = Dataset(
        config={"dataset_name": "f1_test"},
        data=[{"id": "p1", "question": "Name the president", "golden_answers": ["John Smith"]}],
    )
    _run_real_evaluator(dataset, preds=["John"], metrics=["f1"])
    f1_score = dataset[0].output["metric_score"]["f1"]
    assert 0.0 < f1_score < 1.0  # partial overlap, not exact

    strict = to_openeval_resultset(dataset, metric_names=["f1"], passed_threshold=1.0)
    assert strict["results"][0]["grader_results"][0]["passed"] is False

    lenient = to_openeval_resultset(dataset, metric_names=["f1"], passed_threshold=0.5)
    assert lenient["results"][0]["grader_results"][0]["score"] == f1_score
    assert lenient["results"][0]["grader_results"][0]["passed"] is True
    assert validate_result_set(lenient).valid


def test_to_openeval_resultset_clamps_out_of_range_score():
    # Regression guard for the pre-fix LLMJudge behaviour (score/10 + 1, up
    # to 2.0) flagged in https://github.com/RUC-NLPIR/FlashRAG/issues/246 --
    # confirmed fixed on current main (commit d98e892) by reading
    # flashrag/evaluator/metrics.py directly, but this adapter clamps
    # regardless since EvalPort's schema requires score in [0, 1] or null.
    dataset = Dataset(
        config={"dataset_name": "judge_test"},
        data=[{"id": "j1", "question": "Q", "golden_answers": ["A"]}],
    )
    dataset.update_output("pred", ["A"])
    dataset[0].update_evaluation_score("llm_judge", 2.0)  # out-of-range raw score

    resultset = to_openeval_resultset(dataset, metric_names=["llm_judge"])
    gr = resultset["results"][0]["grader_results"][0]
    assert gr["score"] == 1.0  # clamped
    assert gr["metadata"]["flashrag_raw_score"] == 2.0  # original value preserved, not silently lost
    assert validate_result_set(resultset).valid


def test_to_openeval_resultset_missing_metric_is_visible_not_dropped():
    dataset = Dataset(
        config={"dataset_name": "missing_metric_test"},
        data=[{"id": "m1", "question": "Q", "golden_answers": ["A"]}],
    )
    dataset.update_output("pred", ["A"])
    # No metric_score set at all (e.g. Evaluator.evaluate caught an exception
    # for this metric and only logged it -- see Evaluator.evaluate).
    resultset = to_openeval_resultset(dataset, metric_names=["em"])
    gr = resultset["results"][0]["grader_results"][0]
    assert gr["score"] is None
    assert gr["passed"] is False
    assert validate_result_set(resultset).valid


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------

def test_from_openeval_round_trip_into_real_flashrag_item():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {
                "id": "tc1",
                "input": "What is 2+2?",
                "expected_output": "4",
                "graders": ["g1"],
                "metadata": {"flashrag_golden_answers": ["4", "four"]},
            },
        ],
    }
    item_dicts = from_openeval(suite)
    assert item_dicts == [
        {"id": "tc1", "question": "What is 2+2?", "golden_answers": ["4", "four"]}
    ]

    # Feed straight into a real FlashRAG Dataset, confirming the shape fits
    # the actual Item constructor, not just a dict our own test invented.
    dataset = Dataset(config={"dataset_name": "roundtrip_test"}, data=item_dicts)
    assert dataset.question == ["What is 2+2?"]
    assert dataset.golden_answers == [["4", "four"]]
    assert dataset.id == ["tc1"]


def test_from_openeval_single_expected_output_without_metadata():
    suite = {
        "version": "1.0.0",
        "id": "s2",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {"id": "tc2", "input": "hi", "expected_output": "hello", "graders": ["g1"]},
        ],
    }
    item_dicts = from_openeval(suite)
    assert item_dicts == [{"id": "tc2", "question": "hi", "golden_answers": ["hello"]}]


def test_from_openeval_retrieval_context_round_trips_to_retrieval_result():
    suite = {
        "version": "1.0.0",
        "id": "s3",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {
                "id": "tc3",
                "input": "hi",
                "graders": ["g1"],
                "retrieval_context": ["doc a", "doc b"],
            },
        ],
    }
    item_dicts = from_openeval(suite)
    assert item_dicts[0]["output"]["retrieval_result"] == [
        {"contents": "doc a"},
        {"contents": "doc b"},
    ]


# ---------------------------------------------------------------------------
# Full round trip: FlashRAG -> EvalPort suite+resultset -> back to FlashRAG
# ---------------------------------------------------------------------------

def test_full_round_trip_preserves_questions_and_answers():
    original = _build_dataset()
    suite = to_openeval_suite(original, metric_names=["em"])
    assert validate_suite(suite).valid

    item_dicts = from_openeval(suite)
    rebuilt = Dataset(config={"dataset_name": "nq_test"}, data=item_dicts)

    assert rebuilt.question == original.question
    assert rebuilt.golden_answers == original.golden_answers
    assert rebuilt.id == original.id
