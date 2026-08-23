"""Tests for vellum-openeval-adapter.

Organized around the two wrinkles from issue #16 (named typed-variable
flattening on the test-case side, typed metric-output union on the results
side), then the public API, with round trips validated against the REAL
`openeval.validate.validate_suite()` / `validate_result_set()` -- not
internal self-consistency. Uses the real `vellum.types` pydantic models
(`TestCaseStringVariableValue`, `TestSuiteRunExecutionMetricResult`, ...)
wherever practical, matching the convention set by
`adapters/literalai-openeval-adapter`.
"""

import pytest

import vellum.types as vt
from openeval.validate import validate_suite, validate_result_set

from vellum_openeval_adapter import (
    from_openeval,
    map_metric_output,
    results_to_openeval,
    stringify_variable_value,
    to_openeval,
    variables_to_input,
)


def make_test_case(id, input_values, evaluation_values=None, external_id=None, label=None):
    """Build a real `vellum.types.TestSuiteTestCase`, not a mock."""
    return vt.TestSuiteTestCase(
        id=id,
        external_id=external_id,
        label=label,
        input_values=input_values,
        evaluation_values=evaluation_values or [],
    )


def string_var(name, value):
    return vt.TestCaseStringVariableValue(variable_id=f"var_{name}", name=name, type="STRING", value=value)


def number_var(name, value):
    return vt.TestCaseNumberVariableValue(variable_id=f"var_{name}", name=name, type="NUMBER", value=value)


class FakeExecutions:
    """Plain stand-in for `PaginatedTestSuiteRunExecutionList` (has a
    `.results` attribute, like the real SDK response)."""

    def __init__(self, results):
        self.results = results


# ============================================================
# 1. Typed, named variable-value flattening (variables_to_input)
# ============================================================

class TestStringifyVariableValue:
    def test_string_type_passthrough(self):
        assert stringify_variable_value(string_var("q", "What is 2+2?")) == "What is 2+2?"

    def test_string_type_none_becomes_empty(self):
        assert stringify_variable_value(string_var("q", None)) == ""

    def test_number_type(self):
        assert stringify_variable_value(number_var("n", 42.5)) == "42.5"

    def test_chat_history_type(self):
        msg1 = vt.ChatMessage(role="SYSTEM", text="You are helpful.")
        msg2 = vt.ChatMessage(role="USER", text="Hi")
        var = vt.TestCaseChatHistoryVariableValue(
            variable_id="var_h", name="history", type="CHAT_HISTORY", value=[msg1, msg2]
        )
        result = stringify_variable_value(var)
        assert result == "SYSTEM: You are helpful.\nUSER: Hi"

    def test_json_type_serializes(self):
        var = vt.TestCaseJsonVariableValue(
            variable_id="var_j", name="ctx", type="JSON", value={"b": 2, "a": 1}
        )
        result = stringify_variable_value(var)
        assert result == '{"a": 1, "b": 2}'


class TestVariablesToInput:
    def test_single_string_variable_returns_plain_string(self):
        """★ Core case from the issue: one STRING variable -> plain string, not a 1-item array."""
        result = variables_to_input([string_var("input", "What is 2+2?")])
        assert result == "What is 2+2?"

    def test_single_number_variable_returns_labeled_array(self):
        # NUMBER is typed as Optional[float] on the real model, so pydantic
        # coerces 3 -> 3.0 -- asserting against that real coercion rather
        # than the int literal we passed in.
        result = variables_to_input([number_var("count", 3)])
        assert result == ["count: 3.0"]

    def test_multiple_variables_all_preserved_as_labeled_array(self):
        """★ Core wrinkle: a named, typed variable system -- every variable
        kept, not just the first/preferred one."""
        result = variables_to_input([string_var("system", "Be terse."), string_var("question", "2+2?")])
        assert result == ["system: Be terse.", "question: 2+2?"]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            variables_to_input([])


# ============================================================
# 2. Typed metric-output union (map_metric_output)
# ============================================================

class TestMapMetricOutput:
    def test_number_output_becomes_real_score(self):
        """★ Core case: a NUMBER metric output is a genuine score."""
        mr = vt.TestSuiteRunExecutionMetricResult(
            metric_id="m1",
            outputs=[vt.TestSuiteRunMetricNumberOutput(name="score", type="NUMBER", value=0.8)],
            metric_label="Accuracy",
        )
        gr = map_metric_output(mr)
        assert gr["score"] == 0.8
        assert gr["passed"] is True
        assert gr["grader_id"] == "Accuracy"
        assert gr["type"] == "custom"
        assert gr["metadata"]["openeval"]["raw_score"] == 0.8

    def test_number_output_is_clamped(self):
        mr = vt.TestSuiteRunExecutionMetricResult(
            metric_id="m1", outputs=[vt.TestSuiteRunMetricNumberOutput(name="score", type="NUMBER", value=8.5)]
        )
        gr = map_metric_output(mr)
        assert gr["score"] == 1.0
        assert gr["metadata"]["openeval"]["raw_score"] == 8.5

    def test_number_output_below_threshold_fails(self):
        mr = vt.TestSuiteRunExecutionMetricResult(
            metric_id="m1", outputs=[vt.TestSuiteRunMetricNumberOutput(name="score", type="NUMBER", value=0.3)]
        )
        gr = map_metric_output(mr, pass_threshold=0.5)
        assert gr["passed"] is False

    def test_string_output_has_null_score_and_passes_by_default(self):
        """★ Honest degradation: non-numeric metric output -> score: null,
        not a fabricated number."""
        mr = vt.TestSuiteRunExecutionMetricResult(
            metric_id="m1",
            outputs=[vt.TestSuiteRunMetricStringOutput(name="verdict", type="STRING", value="looks good")],
        )
        gr = map_metric_output(mr)
        assert gr["score"] is None
        assert gr["passed"] is True
        assert gr["metadata"]["vellum"]["outputs"][0]["value"] == "looks good"

    def test_error_output_fails_with_null_score(self):
        mr = vt.TestSuiteRunExecutionMetricResult(
            metric_id="m1",
            outputs=[
                vt.TestSuiteRunMetricErrorOutput(
                    name="err", type="ERROR", value=vt.VellumError(code="INTERNAL_SERVER_ERROR", message="boom")
                )
            ],
        )
        gr = map_metric_output(mr)
        assert gr["score"] is None
        assert gr["passed"] is False

    def test_falls_back_to_metric_id_when_no_label(self):
        mr = vt.TestSuiteRunExecutionMetricResult(
            metric_id="m_raw_id",
            outputs=[vt.TestSuiteRunMetricNumberOutput(name="score", type="NUMBER", value=1.0)],
        )
        gr = map_metric_output(mr)
        assert gr["grader_id"] == "m_raw_id"


# ============================================================
# 3. Public API: to_openeval / from_openeval / results_to_openeval
# ============================================================

class TestToOpenEval:
    def test_single_test_case_validates_against_real_schema(self):
        tc = make_test_case("tc1", [string_var("input", "What is 2+2?")], [string_var("expected", "4")])
        suite = to_openeval([tc], id="my_suite")
        validation = validate_suite(suite)
        assert validation.valid, validation.errors
        assert suite["test_cases"][0]["input"] == "What is 2+2?"
        assert suite["test_cases"][0]["expected_output"] == "4"
        assert suite["test_cases"][0]["graders"] == ["gr_vellum_default"]

    def test_multi_variable_test_case_validates(self):
        tc = make_test_case(
            "tc1",
            [string_var("system", "Be terse."), string_var("question", "2+2?")],
        )
        suite = to_openeval([tc])
        validation = validate_suite(suite)
        assert validation.valid, validation.errors
        assert suite["test_cases"][0]["input"] == ["system: Be terse.", "question: 2+2?"]

    def test_exact_match_grader_option_also_validates(self):
        tc = make_test_case("tc1", [string_var("input", "hi")])
        suite = to_openeval([tc], grader_type="exact_match")
        assert validate_suite(suite).valid
        assert suite["graders"][0]["type"] == "exact_match"

    def test_no_evaluation_values_omits_expected_output(self):
        tc = make_test_case("tc1", [string_var("input", "hi")])
        suite = to_openeval([tc])
        assert "expected_output" not in suite["test_cases"][0]

    def test_empty_test_cases_raises(self):
        with pytest.raises(ValueError):
            to_openeval([])

    def test_external_id_used_when_id_missing(self):
        tc = vt.TestSuiteTestCase(
            id=None, external_id="ext-1", label=None,
            input_values=[string_var("input", "hi")], evaluation_values=[],
        )
        suite = to_openeval([tc])
        assert suite["test_cases"][0]["id"] == "ext-1"


class TestFromOpenEval:
    def test_round_trip_recovers_original_named_variables(self):
        tc = make_test_case(
            "tc1",
            [string_var("system", "Be terse."), number_var("n", 5)],
            [string_var("expected", "done")],
        )
        suite = to_openeval([tc])
        items = from_openeval(suite)
        assert len(items) == 1
        input_values = items[0]["input_values"]
        assert len(input_values) == 2
        assert input_values[0]["name"] == "system"
        assert input_values[0]["value"] == "Be terse."
        assert input_values[1]["name"] == "n"
        assert input_values[1]["value"] == 5
        assert items[0]["evaluation_values"][0]["value"] == "done"

    def test_from_openeval_on_a_suite_not_from_this_adapter(self):
        suite = {
            "version": "1.0.0",
            "id": "s1",
            "test_cases": [
                {"id": "tc1", "input": "plain string input", "graders": ["g1"], "expected_output": "answer"}
            ],
        }
        items = from_openeval(suite)
        assert items[0]["input_values"] == [{"name": "input", "type": "STRING", "value": "plain string input"}]
        assert items[0]["evaluation_values"] == [{"name": "expected_output", "type": "STRING", "value": "answer"}]


class TestResultsToOpenEval:
    def test_full_results_conversion_validates_against_real_schema(self):
        execution = vt.TestSuiteRunExecution(
            id="exec1",
            test_case_id="tc1",
            outputs=[vt.TestSuiteRunExecutionStringOutput(
                name="output", type="STRING", value="4", output_variable_id="ov1"
            )],
            metric_results=[
                vt.TestSuiteRunExecutionMetricResult(
                    metric_id="m1",
                    outputs=[vt.TestSuiteRunMetricNumberOutput(name="score", type="NUMBER", value=1.0)],
                    metric_label="exact_match",
                )
            ],
        )
        result_set = results_to_openeval(
            [execution], suite_id="s1", run_id="r1", started_at="2026-01-01T00:00:00Z"
        )
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        assert result_set["results"][0]["actual_output"] == "4"
        assert result_set["results"][0]["passed"] is True

    def test_accepts_paginated_response_with_results_attribute(self):
        execution = vt.TestSuiteRunExecution(id="e1", test_case_id="tc1", outputs=[], metric_results=[])
        result_set = results_to_openeval(
            FakeExecutions([execution]), suite_id="s1", run_id="r1", started_at="2026-01-01T00:00:00Z"
        )
        assert validate_result_set(result_set).valid
        assert result_set["results"][0]["passed"] is False  # no graders -> unpassed, per convention

    def test_multiple_metrics_each_mapped_independently(self):
        execution = vt.TestSuiteRunExecution(
            id="e1", test_case_id="tc1", outputs=[],
            metric_results=[
                vt.TestSuiteRunExecutionMetricResult(
                    metric_id="m1", outputs=[vt.TestSuiteRunMetricNumberOutput(name="s", type="NUMBER", value=0.9)]
                ),
                vt.TestSuiteRunExecutionMetricResult(
                    metric_id="m2", outputs=[vt.TestSuiteRunMetricNumberOutput(name="s", type="NUMBER", value=0.2)]
                ),
            ],
        )
        result_set = results_to_openeval(
            [execution], suite_id="s1", run_id="r1", started_at="2026-01-01T00:00:00Z"
        )
        assert validate_result_set(result_set).valid
        assert len(result_set["results"][0]["grader_results"]) == 2
        assert result_set["results"][0]["passed"] is False  # second metric fails threshold

    def test_started_at_defaults_when_omitted(self):
        execution = vt.TestSuiteRunExecution(id="e1", test_case_id="tc1", outputs=[], metric_results=[])
        result_set = results_to_openeval([execution], suite_id="s1", run_id="r1")
        assert "started_at" in result_set and result_set["started_at"]
        assert validate_result_set(result_set).valid

    def test_multiple_outputs_joined_with_labels(self):
        execution = vt.TestSuiteRunExecution(
            id="e1", test_case_id="tc1",
            outputs=[
                vt.TestSuiteRunExecutionStringOutput(name="answer", type="STRING", value="4", output_variable_id="o1"),
                vt.TestSuiteRunExecutionStringOutput(name="rationale", type="STRING", value="basic math", output_variable_id="o2"),
            ],
            metric_results=[],
        )
        result_set = results_to_openeval([execution], suite_id="s1", run_id="r1", started_at="2026-01-01T00:00:00Z")
        assert result_set["results"][0]["actual_output"] == "answer: 4\nrationale: basic math"
