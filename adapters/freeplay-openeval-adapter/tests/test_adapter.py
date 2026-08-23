"""
Tests for freeplay-openeval-adapter.

Uses the real, installed `freeplay` package (DatasetTestCase, Dataset,
CompletionTestCase, TraceTestCase, UserMessage, AssistantMessage) and
asserts every produced document against the real `evalport-sdk`
validators (openeval.validate.validate_suite / validate_result_set) --
never mocks.
"""
import pytest

from freeplay.freeplay import Freeplay
from freeplay.resources.test_cases import Dataset, DatasetResults, DatasetTestCase
from freeplay.resources.test_suites import CompletionTestCase, TestSuites, TraceTestCase
from freeplay.model import AssistantMessage, UserMessage
from openeval.validate import validate_suite, validate_result_set

from freeplay_openeval_adapter import (
    clamp_score,
    flatten_inputs,
    from_openeval,
    results_to_openeval,
    to_openeval,
)


# --- flatten_inputs ---


def test_flatten_inputs_prefers_known_key():
    assert flatten_inputs({"question": "What is 2+2?", "context": "math"}) == "What is 2+2?"


def test_flatten_inputs_falls_back_to_first_string_value():
    assert flatten_inputs({"topic": "geometry", "difficulty": 3}) == "geometry"


def test_flatten_inputs_json_fallback_when_no_string_values():
    result = flatten_inputs({"count": 3, "enabled": True})
    assert "count" in result and "enabled" in result
    import json
    parsed = json.loads(result)
    assert parsed == {"count": 3, "enabled": True}


def test_flatten_inputs_empty_raises():
    with pytest.raises(ValueError):
        flatten_inputs({})


# --- clamp_score ---


def test_clamp_score_bool_true():
    assert clamp_score(True) == 1.0


def test_clamp_score_bool_false():
    assert clamp_score(False) == 0.0


def test_clamp_score_float_in_range():
    assert clamp_score(0.73) == 0.73


def test_clamp_score_float_above_range_clamps():
    assert clamp_score(8.5) == 1.0


def test_clamp_score_float_below_range_clamps():
    assert clamp_score(-3.0) == 0.0


def test_clamp_score_none_passthrough():
    assert clamp_score(None) is None


def test_clamp_score_invalid_type_raises():
    with pytest.raises(TypeError):
        clamp_score("not a number")


# --- to_openeval: real DatasetTestCase / Dataset objects ---


def _real_dataset():
    return Dataset(
        dataset_id="ds-123",
        test_cases=[
            DatasetTestCase(
                id="tc-1",
                inputs={"question": "What is the capital of France?"},
                output="Paris",
                metadata={"source": "manual"},
            ),
            DatasetTestCase(
                id="tc-2",
                inputs={"a": 2, "b": 3},
                output="5",
                history=[
                    UserMessage(content="What is 2+3?"),
                    AssistantMessage(content="5"),
                ],
            ),
        ],
    )


def test_to_openeval_real_dataset_validates():
    suite = to_openeval(_real_dataset())
    result = validate_suite(suite)
    assert result.valid, result.errors
    assert suite["id"] == "ds-123"
    assert len(suite["test_cases"]) == 2


def test_to_openeval_flattens_named_variable_inputs():
    suite = to_openeval(_real_dataset())
    tc1, tc2 = suite["test_cases"]
    assert tc1["input"] == "What is the capital of France?"
    # tc2 has no preferred key and no string value among {"a": 2, "b": 3} --
    # falls back to the JSON dump.
    import json
    assert json.loads(tc2["input"]) == {"a": 2, "b": 3}


def test_to_openeval_preserves_original_inputs_in_metadata():
    suite = to_openeval(_real_dataset())
    tc1 = suite["test_cases"][0]
    assert tc1["metadata"]["freeplay"]["original_inputs"] == {
        "question": "What is the capital of France?"
    }
    # user-supplied metadata is preserved alongside the freeplay namespace
    assert tc1["metadata"]["source"] == "manual"


def test_to_openeval_preserves_history():
    suite = to_openeval(_real_dataset())
    tc2 = suite["test_cases"][1]
    history = tc2["metadata"]["freeplay"]["history"]
    assert history == [
        {"role": "user", "content": "What is 2+3?"},
        {"role": "assistant", "content": "5"},
    ]


def test_to_openeval_default_grader_is_llm_judge():
    suite = to_openeval(_real_dataset())
    assert suite["graders"][0]["type"] == "llm_judge"
    assert suite["test_cases"][0]["graders"] == ["grader-1"]


def test_to_openeval_grader_type_override():
    suite = to_openeval(_real_dataset(), grader_type="exact_match")
    assert suite["graders"][0]["type"] == "exact_match"
    assert validate_suite(suite).valid


def test_to_openeval_accepts_plain_dict_dataset():
    dataset = {
        "dataset_id": "ds-dict",
        "test_cases": [
            {"id": "tc-a", "inputs": {"input": "hello"}, "output": "world"},
        ],
    }
    suite = to_openeval(dataset)
    assert validate_suite(suite).valid
    assert suite["id"] == "ds-dict"


def test_to_openeval_accepts_dataset_results():
    # fp.test_cases.get(...) returns DatasetResults, not Dataset -- same
    # shape (dataset_id, test_cases), but a distinct real class.
    dataset_results = DatasetResults(
        dataset_id="ds-results",
        test_cases=[DatasetTestCase(id="t1", inputs={"input": "hi"}, output="ok")],
    )
    suite = to_openeval(dataset_results)
    assert validate_suite(suite).valid
    assert suite["id"] == "ds-results"


def test_to_openeval_empty_dataset_raises():
    with pytest.raises(ValueError):
        to_openeval(Dataset(dataset_id="empty", test_cases=[]))


def test_to_openeval_missing_output_omits_expected_output():
    dataset = Dataset(
        dataset_id="ds-no-output",
        test_cases=[DatasetTestCase(id="tc-x", inputs={"input": "hi"}, output=None)],
    )
    suite = to_openeval(dataset)
    assert "expected_output" not in suite["test_cases"][0]
    assert validate_suite(suite).valid


# --- from_openeval ---


def test_from_openeval_restores_original_inputs():
    suite = to_openeval(_real_dataset())
    items = from_openeval(suite)
    assert items[0]["inputs"] == {"question": "What is the capital of France?"}
    assert items[0]["output"] == "Paris"
    assert items[0]["id"] == "tc-1"


def test_from_openeval_restores_history():
    suite = to_openeval(_real_dataset())
    items = from_openeval(suite)
    assert items[1]["history"] == [
        {"role": "user", "content": "What is 2+3?"},
        {"role": "assistant", "content": "5"},
    ]


def test_from_openeval_fallback_for_foreign_test_case():
    # A TestCase this adapter didn't produce -- no metadata.freeplay at all.
    suite = {
        "version": "1.0.0",
        "id": "external-suite",
        "test_cases": [
            {"id": "ext-1", "input": "plain string input", "graders": ["g1"]}
        ],
        "graders": [{"id": "g1", "type": "exact_match"}],
    }
    items = from_openeval(suite)
    assert items[0]["inputs"] == {"input": "plain string input"}


# --- TestSuites client-wiring gap (documented in README's Design notes) ---


def test_test_suites_not_wired_onto_client_but_constructible():
    """Confirms the README's claim: fp.test_suites doesn't exist on the
    real Freeplay client as of 0.6.0, but TestSuites(fp.call_support,
    fp.recordings) works, using the client's own public attributes."""
    fp = Freeplay(freeplay_api_key="test-key", api_base="https://example.invalid/api")
    assert not hasattr(fp, "test_suites")
    test_suites = TestSuites(fp.call_support, fp.recordings)
    assert isinstance(test_suites, TestSuites)


# --- results_to_openeval: CompletionTestCase (prompt-type suite) ---


def test_results_to_openeval_completion_test_cases_validates():
    tc = CompletionTestCase(
        test_case_id="tc-1",
        variables={"question": "2+2?"},
        output="4",
        history=None,
        custom_metadata=None,
    )
    recorded = [
        {
            "test_case_id": tc.id,
            "eval_results": {"exact_match": True, "helpfulness": 0.9},
            "output": "4",
        }
    ]
    result_set = results_to_openeval("suite-1", "run-1", recorded)
    result = validate_result_set(result_set)
    assert result.valid, result.errors
    assert result_set["results"][0]["passed"] is True


def test_results_to_openeval_trace_test_cases_validates():
    # Agent-type suites use TraceTestCase instead of CompletionTestCase --
    # results_to_openeval() is agnostic to which one produced the id, since
    # it only needs the test_case_id string, not the test case object.
    trace_tc = TraceTestCase(
        test_case_id="trace-1", input="do the thing", output=None, custom_metadata=None
    )
    recorded = [
        {"test_case_id": trace_tc.id, "eval_results": {"completed": False}},
    ]
    result_set = results_to_openeval("suite-2", "run-2", recorded)
    assert validate_result_set(result_set).valid
    assert result_set["results"][0]["passed"] is False


def test_results_to_openeval_clamps_and_preserves_raw_score():
    recorded = [{"test_case_id": "tc-1", "eval_results": {"quality": 8.5}}]
    result_set = results_to_openeval("suite-3", "run-3", recorded)
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["score"] == 1.0
    assert gr["metadata"]["openeval"]["raw_score"] == 8.5
    assert validate_result_set(result_set).valid


def test_results_to_openeval_bool_and_float_mixed():
    recorded = [
        {
            "test_case_id": "tc-mixed",
            "eval_results": {"correct": True, "similarity": 0.3},
        }
    ]
    result_set = results_to_openeval("suite-4", "run-4", recorded)
    # 0.3 similarity < 0.5 pass threshold, so overall passed should be False
    # since not every grader passed (all() semantics).
    assert result_set["results"][0]["passed"] is False
    assert validate_result_set(result_set).valid


def test_results_to_openeval_passed_override():
    recorded = [
        {"test_case_id": "tc-1", "eval_results": {"quality": 0.2}, "passed": True}
    ]
    result_set = results_to_openeval("suite-5", "run-5", recorded)
    assert result_set["results"][0]["passed"] is True


def test_results_to_openeval_empty_raises():
    with pytest.raises(ValueError):
        results_to_openeval("suite-6", "run-6", [])


def test_results_to_openeval_missing_test_case_id_raises():
    with pytest.raises(ValueError):
        results_to_openeval("suite-7", "run-7", [{"eval_results": {"x": True}}])


def test_results_to_openeval_invalid_eval_value_raises():
    with pytest.raises(TypeError):
        results_to_openeval(
            "suite-8", "run-8", [{"test_case_id": "tc-1", "eval_results": {"x": "bad"}}]
        )


def test_results_to_openeval_multiple_test_cases_and_started_at():
    recorded = [
        {"test_case_id": "tc-1", "eval_results": {"x": True}},
        {"test_case_id": "tc-2", "eval_results": {"x": False}},
    ]
    result_set = results_to_openeval(
        "suite-9", "run-9", recorded, started_at="2026-01-01T00:00:00Z"
    )
    assert result_set["started_at"] == "2026-01-01T00:00:00Z"
    assert len(result_set["results"]) == 2
    assert validate_result_set(result_set).valid
