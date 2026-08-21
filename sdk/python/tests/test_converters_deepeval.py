"""Tests for openeval.converters_deepeval.from_deepeval(), verified against a real
`deepeval` install (4.1.9) and the real openeval.validate.validate_suite().

from_deepeval() takes a plain dict shape: {"test_cases": [{"id", "input", "metrics",
"expected_output", "context", "retrieval_context", "metadata", "expected_tools"}]}.
deepeval's own `LLMTestCase` (deepeval.test_case) is a pydantic model with no `id` or
`metrics` field of its own -- metrics are supplied separately to `evaluate(dataset,
metrics=[...])`, not stored on the test case. So the dict shape this converter expects
isn't literally `LLMTestCase.model_dump()`; it's `LLMTestCase.model_dump()` plus an
`id` and metric names layered on top, which is exactly how a caller wiring up an
EvalPort export would assemble it. Confirmed real field names (`input`,
`actual_output`, `expected_output`, `context`, `retrieval_context`, `metadata`,
`expected_tools`) via `LLMTestCase(...).model_dump()` against the actual installed
package, not assumed from the docs.

Metric-name matching is verified against the real class names deepeval ships --
`FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`,
`ContextualRecallMetric`, `HallucinationMetric`, `ToxicityMetric` -- via
`Metric.__name__`, confirmed importable from `deepeval.metrics` in the installed
package. `from_deepeval()`'s matching is case-insensitive substring matching against
whatever string it's handed, so a caller passes `metric.__name__` (or `type(metric
_instance).__name__`) directly, no re-derivation needed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("deepeval", reason="requires a real deepeval install")

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase  # noqa: E402

from openeval.converters_deepeval import from_deepeval  # noqa: E402
from openeval.validate import validate_suite  # noqa: E402


def _real_test_case_dict(tc_id: str, metrics, **overrides) -> dict:
    """Build the dict from_deepeval() expects, using LLMTestCase's real field names
    and a real .model_dump() as the base so nothing here is a guessed shape."""
    llm_tc = LLMTestCase(
        input=overrides.pop("input", "What is the capital of France?"),
        actual_output=overrides.pop("actual_output", "Paris"),
        expected_output=overrides.pop("expected_output", "Paris"),
        context=overrides.pop("context", ["France is a country in Europe."]),
        retrieval_context=overrides.pop("retrieval_context", ["Paris is the capital of France."]),
    )
    dumped = llm_tc.model_dump()
    result = {
        "id": tc_id,
        "input": dumped["input"],
        "metrics": [m.__name__ for m in metrics],
        "expected_output": dumped["expected_output"],
        "context": dumped["context"],
        "retrieval_context": dumped["retrieval_context"],
    }
    result.update(overrides)
    return result


def test_faithfulness_metric_maps_to_llm_judge_with_valid_params():
    de = {"test_cases": [_real_test_case_dict("tc1", [FaithfulnessMetric])]}
    suite = from_deepeval(de)
    grader = suite["graders"][0]
    assert grader["type"] == "llm_judge"
    assert grader["params"]["model"]
    assert "{output}" in grader["params"]["prompt"]
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_answer_relevancy_metric_maps_to_semantic_similarity():
    de = {"test_cases": [_real_test_case_dict("tc1", [AnswerRelevancyMetric])]}
    suite = from_deepeval(de)
    grader = suite["graders"][0]
    assert grader["type"] == "semantic_similarity"
    assert 0 <= grader["params"]["threshold"] <= 1
    result = validate_suite(suite)
    assert result.valid, result.errors


@pytest.mark.parametrize(
    "metric_cls",
    [ContextualPrecisionMetric, ContextualRecallMetric, HallucinationMetric, ToxicityMetric],
)
def test_real_deepeval_judge_metrics_all_produce_valid_llm_judge_graders(metric_cls):
    de = {"test_cases": [_real_test_case_dict("tc1", [metric_cls])]}
    suite = from_deepeval(de)
    grader = suite["graders"][0]
    assert grader["type"] == "llm_judge"
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_unknown_custom_metric_maps_to_custom_grader_with_handler():
    """A user-defined GEval or custom metric subclass has no built-in mapping and
    should fall back to `custom`, the same convention every adapter uses."""
    de = {"test_cases": [_real_test_case_dict("tc1", metrics=[])]}
    de["test_cases"][0]["metrics"] = ["MyCustomBusinessLogicMetric"]
    suite = from_deepeval(de)
    grader = suite["graders"][0]
    assert grader["type"] == "custom"
    assert grader["params"]["handler"] == "deepeval:MyCustomBusinessLogicMetric"
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_multiple_metrics_on_one_test_case_produce_one_grader_each():
    de = {"test_cases": [_real_test_case_dict("tc1", [FaithfulnessMetric, AnswerRelevancyMetric])]}
    suite = from_deepeval(de)
    assert len(suite["graders"]) == 2
    assert len(suite["test_cases"][0]["graders"]) == 2
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_test_case_with_no_metrics_gets_default_grader_not_empty_list():
    """graders.minItems:1 -- an EvalPort TestCase can never have zero graders,
    even when the source deepeval test case wasn't run against any metric."""
    de = {"test_cases": [_real_test_case_dict("tc1", metrics=[])]}
    suite = from_deepeval(de)
    assert suite["test_cases"][0]["graders"] == ["gr_default"]
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_context_and_retrieval_context_preserved_from_real_llm_test_case():
    de = {"test_cases": [_real_test_case_dict("tc1", [FaithfulnessMetric])]}
    suite = from_deepeval(de)
    tc = suite["test_cases"][0]
    assert tc["context"] == ["France is a country in Europe."]
    assert tc["retrieval_context"] == ["Paris is the capital of France."]


def test_expected_tools_preserved_when_present():
    de = {"test_cases": [_real_test_case_dict("tc1", [FaithfulnessMetric], expected_tools=["search_tool"])]}
    suite = from_deepeval(de)
    assert suite["test_cases"][0]["expected_tools"] == ["search_tool"]
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_multi_test_case_dataset_produces_spec_valid_suite_end_to_end():
    """Mirrors a real deepeval.dataset.EvaluationDataset with several LLMTestCases
    run against several metrics -- the realistic shape a real export would produce."""
    de = {
        "test_cases": [
            _real_test_case_dict(
                "geo-1",
                [FaithfulnessMetric, AnswerRelevancyMetric],
                input="What is the capital of France?",
                actual_output="Paris",
                expected_output="Paris",
            ),
            _real_test_case_dict(
                "geo-2",
                [ContextualPrecisionMetric],
                input="What is the capital of Japan?",
                actual_output="Tokyo",
                expected_output="Tokyo",
                context=["Japan is an island country in East Asia."],
                retrieval_context=["Tokyo is the capital of Japan."],
            ),
        ]
    }
    suite = from_deepeval(de)
    result = validate_suite(suite)
    assert result.valid, result.errors
    assert len(suite["test_cases"]) == 2
    assert len(suite["graders"]) == 3
    assert suite["metadata"]["openeval"]["source"] == "deepeval"


def test_missing_test_cases_key_produces_empty_but_still_valid_shape():
    de = {}
    suite = from_deepeval(de)
    assert suite["test_cases"] == []
    assert suite["graders"] == [{"id": "gr_default", "type": "exact_match"}]
