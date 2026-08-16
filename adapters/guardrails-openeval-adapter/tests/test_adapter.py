"""Tests for guardrails-openeval-adapter.

Every test here runs against the real, installed ``guardrails-ai`` package
(not a mock) and the real ``openeval.validate`` validators from
``evalport-sdk``. The validators used are small, purely local custom
validators registered with ``@register_validator`` -- exactly the supported,
documented way to write a Guardrails validator -- so nothing here depends on
network access or the Guardrails Hub (the official hub validators require a
network fetch to install, which this sandbox doesn't have; see the README
for why that's a deliberate choice, not a limitation of the adapter).
"""
from typing import Any, Dict, Optional

import pytest
from guardrails import Guard, OnFailAction
from guardrails.validator_base import FailResult, PassResult, Validator, register_validator

from openeval.validate import validate_result_set, validate_suite

from guardrails_openeval_adapter import (
    evaluation_result_to_openeval,
    from_openeval,
    to_openeval,
)


# ---------------------------------------------------------------------------
# Local, purely-computational test validators (no network, no Hub install).
# ---------------------------------------------------------------------------


@register_validator(name="openeval-test/valid-length", data_type="string")
class ValidLength(Validator):
    def __init__(self, min: int = 0, max: int = 1000, on_fail: Optional[Any] = None, **kwargs):
        super().__init__(on_fail=on_fail, min=min, max=max, **kwargs)
        self._min = min
        self._max = max

    def validate(self, value: Any, metadata: Dict[str, Any]) -> Any:
        length = len(value)
        if self._min <= length <= self._max:
            return PassResult()
        return FailResult(error_message=f"length {length} not in [{self._min},{self._max}]")


@register_validator(name="openeval-test/contains-substring", data_type="string")
class ContainsSubstring(Validator):
    def __init__(self, substring: str = "", on_fail: Optional[Any] = None, **kwargs):
        super().__init__(on_fail=on_fail, substring=substring, **kwargs)
        self._substring = substring

    def validate(self, value: Any, metadata: Dict[str, Any]) -> Any:
        if self._substring in value:
            return PassResult()
        return FailResult(error_message=f"{self._substring!r} not found in value")


@register_validator(name="openeval-test/no-digits", data_type="string")
class NoDigits(Validator):
    def __init__(self, on_fail: Optional[Any] = None, **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)

    def validate(self, value: Any, metadata: Dict[str, Any]) -> Any:
        if any(ch.isdigit() for ch in value):
            return FailResult(error_message="value contains a digit")
        return PassResult()


def make_guard(*validators):
    """Attach every validator in a single .use() call -- see the adapter
    module docstring's warning about .use(a).use(b) silently replacing
    rather than accumulating."""
    return Guard().use(*validators)


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpenEval:
    def test_basic_suite_is_spec_valid(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["hello world", "another value"])
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_test_case_count_matches_values(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["a", "b", "c"])
        assert len(suite["test_cases"]) == 3

    def test_input_is_the_raw_string(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["hello world"])
        assert suite["test_cases"][0]["input"] == "hello world"

    def test_grader_derived_from_attached_validator(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["hello"])
        assert len(suite["graders"]) == 1
        grader = suite["graders"][0]
        assert grader["id"] == "openeval-test/valid-length"
        assert grader["type"] == "custom"
        assert grader["params"]["handler"] == "ValidLength"
        assert grader["params"]["min"] == 1
        assert grader["params"]["max"] == 50

    def test_multiple_validators_all_become_graders(self):
        guard = make_guard(
            ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP),
            ContainsSubstring(substring="hi", on_fail=OnFailAction.NOOP),
        )
        suite = to_openeval(guard, ["hi there"])
        grader_ids = {g["id"] for g in suite["graders"]}
        assert grader_ids == {"openeval-test/valid-length", "openeval-test/contains-substring"}
        assert set(suite["test_cases"][0]["graders"]) == grader_ids

    def test_explicit_ids_are_used(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["a", "b"], ids=["tc_a", "tc_b"])
        assert [tc["id"] for tc in suite["test_cases"]] == ["tc_a", "tc_b"]

    def test_auto_generated_ids(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["a", "b"])
        assert [tc["id"] for tc in suite["test_cases"]] == [
            "guardrails_tc_0",
            "guardrails_tc_1",
        ]

    def test_mismatched_ids_length_raises(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        with pytest.raises(ValueError):
            to_openeval(guard, ["a", "b"], ids=["only_one"])

    def test_empty_values_raises(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        with pytest.raises(ValueError):
            to_openeval(guard, [])

    def test_guard_with_no_validators_raises(self):
        guard = Guard()
        with pytest.raises(ValueError):
            to_openeval(guard, ["a"])

    def test_duplicate_validator_class_raises(self):
        guard = make_guard(
            ContainsSubstring(substring="x", on_fail=OnFailAction.NOOP),
            ContainsSubstring(substring="y", on_fail=OnFailAction.NOOP),
        )
        with pytest.raises(ValueError, match="more than one validator"):
            to_openeval(guard, ["a"])

    def test_custom_suite_id_and_description(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["a"], suite_id="my_suite", description="desc")
        assert suite["id"] == "my_suite"
        assert suite["description"] == "desc"

    def test_default_suite_id(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["a"])
        assert suite["id"] == "guardrails_suite"


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


class TestFromOpenEval:
    def test_round_trips_string_values(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        suite = to_openeval(guard, ["hello", "world"], ids=["a", "b"])
        items = from_openeval(suite)
        assert items == [{"id": "a", "value": "hello"}, {"id": "b", "value": "world"}]

    def test_empty_suite_raises(self):
        with pytest.raises(ValueError):
            from_openeval({"version": "1.0.0", "id": "s", "graders": [], "test_cases": []})

    def test_array_input_is_joined_with_newline(self):
        suite = {
            "version": "1.0.0",
            "id": "s",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "x"}}],
            "test_cases": [
                {"id": "tc1", "input": ["line one", "line two"], "graders": ["g1"]}
            ],
        }
        items = from_openeval(suite)
        assert items == [{"id": "tc1", "value": "line one\nline two"}]

    def test_hand_authored_suite_with_string_input(self):
        suite = {
            "version": "1.0.0",
            "id": "s",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "x"}}],
            "test_cases": [{"id": "tc1", "input": "plain string", "graders": ["g1"]}],
        }
        items = from_openeval(suite)
        assert items == [{"id": "tc1", "value": "plain string"}]


# ---------------------------------------------------------------------------
# evaluation_result_to_openeval
# ---------------------------------------------------------------------------


class TestEvaluationResultToOpenEval:
    def test_basic_result_set_is_spec_valid(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        values = ["ok value", "also fine"]
        outcomes = [guard.validate(v) for v in values]
        ids = ["tc0", "tc1"]
        result_set = evaluation_result_to_openeval(guard, outcomes, ids)
        result = validate_result_set(result_set)
        assert result.valid, result.errors

    def test_passing_validator_has_no_reason(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        outcome = guard.validate("short")
        result_set = evaluation_result_to_openeval(guard, [outcome], ["tc0"])
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["passed"] is True
        assert gr["score"] == 1.0
        assert "reason" not in gr

    def test_failing_validator_has_reason(self):
        guard = make_guard(ValidLength(min=1, max=5, on_fail=OnFailAction.NOOP))
        outcome = guard.validate("this value is way too long")
        result_set = evaluation_result_to_openeval(guard, [outcome], ["tc0"])
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["passed"] is False
        assert gr["score"] == 0.0
        assert "not in" in gr["reason"]

    def test_missing_validators_treated_as_pass_when_others_fail(self):
        # Regression test for the documented "only failures are reported"
        # Guardrails behavior: with two DIFFERENT validators attached where
        # only one fails, the passing one must still show up as passed=True
        # in this adapter's output, not be silently dropped.
        guard = make_guard(
            ValidLength(min=1, max=100, on_fail=OnFailAction.NOOP),
            ContainsSubstring(substring="zzz_absent", on_fail=OnFailAction.NOOP),
        )
        outcome = guard.validate("a normal short value")
        result_set = evaluation_result_to_openeval(guard, [outcome], ["tc0"])
        grader_results = {
            gr["grader_id"]: gr for gr in result_set["results"][0]["grader_results"]
        }
        assert grader_results["openeval-test/valid-length"]["passed"] is True
        assert grader_results["openeval-test/contains-substring"]["passed"] is False
        assert result_set["results"][0]["passed"] is False

    def test_overall_passed_is_and_of_all_graders(self):
        guard = make_guard(
            ValidLength(min=1, max=100, on_fail=OnFailAction.NOOP),
            NoDigits(on_fail=OnFailAction.NOOP),
        )
        outcome_all_pass = guard.validate("no numbers here")
        outcome_one_fail = guard.validate("has a 1 digit")
        result_set = evaluation_result_to_openeval(
            guard, [outcome_all_pass, outcome_one_fail], ["tc0", "tc1"]
        )
        assert result_set["results"][0]["passed"] is True
        assert result_set["results"][1]["passed"] is False

    def test_actual_output_is_preserved(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        outcome = guard.validate("hello world")
        result_set = evaluation_result_to_openeval(guard, [outcome], ["tc0"])
        assert result_set["results"][0]["actual_output"] == "hello world"

    def test_summary_counts(self):
        guard = make_guard(ValidLength(min=1, max=5, on_fail=OnFailAction.NOOP))
        outcomes = [guard.validate(v) for v in ["ok", "this is far too long to pass"]]
        result_set = evaluation_result_to_openeval(guard, outcomes, ["tc0", "tc1"])
        assert result_set["summary"]["total"] == 2
        assert result_set["summary"]["passed"] == 1
        assert result_set["summary"]["failed"] == 1
        assert result_set["summary"]["pass_rate"] == 0.5

    def test_mismatched_outcomes_and_ids_length_raises(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        outcomes = [guard.validate("a"), guard.validate("b")]
        with pytest.raises(ValueError):
            evaluation_result_to_openeval(guard, outcomes, ["only_one_id"])

    def test_empty_outcomes_raises(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        with pytest.raises(ValueError):
            evaluation_result_to_openeval(guard, [], [])

    def test_duplicate_validator_class_raises(self):
        guard = make_guard(
            ContainsSubstring(substring="x", on_fail=OnFailAction.NOOP),
            ContainsSubstring(substring="y", on_fail=OnFailAction.NOOP),
        )
        outcome = guard.validate("x present")
        with pytest.raises(ValueError, match="more than one validator"):
            evaluation_result_to_openeval(guard, [outcome], ["tc0"])

    def test_explicit_run_id_and_timestamps_are_used(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        outcome = guard.validate("short")
        result_set = evaluation_result_to_openeval(
            guard,
            [outcome],
            ["tc0"],
            run_id="my_run",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:01:00Z",
        )
        assert result_set["run_id"] == "my_run"
        assert result_set["started_at"] == "2026-01-01T00:00:00Z"
        assert result_set["completed_at"] == "2026-01-01T00:01:00Z"

    def test_auto_generated_run_id_has_expected_prefix(self):
        guard = make_guard(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))
        outcome = guard.validate("short")
        result_set = evaluation_result_to_openeval(guard, [outcome], ["tc0"])
        assert result_set["run_id"].startswith("guardrails_run_")


# ---------------------------------------------------------------------------
# Full loop: to_openeval -> from_openeval -> guard.validate -> back to
# openeval, validated against the real spec end to end.
# ---------------------------------------------------------------------------


class TestFullLoop:
    def test_full_round_trip_validates_against_real_spec(self):
        guard = make_guard(
            ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP),
            ContainsSubstring(substring="hi", on_fail=OnFailAction.NOOP),
        )
        values = ["hi there", "no greeting here", "this value here is honestly just way too long for the length validator"]
        suite = to_openeval(guard, values, ids=["a", "b", "c"], suite_id="full_loop_suite")
        suite_result = validate_suite(suite)
        assert suite_result.valid, suite_result.errors

        items = from_openeval(suite)
        outcomes = []
        ids = []
        for item in items:
            outcomes.append(guard.validate(item["value"]))
            ids.append(item["id"])

        result_set = evaluation_result_to_openeval(
            guard, outcomes, ids, suite_id=suite["id"]
        )
        rs_result = validate_result_set(result_set)
        assert rs_result.valid, rs_result.errors

        # "hi there" passes both validators.
        r_a = next(r for r in result_set["results"] if r["test_case_id"] == "a")
        assert r_a["passed"] is True

        # "no greeting here" fails the contains-substring check only.
        r_b = next(r for r in result_set["results"] if r["test_case_id"] == "b")
        assert r_b["passed"] is False
        gr_b = {g["grader_id"]: g for g in r_b["grader_results"]}
        assert gr_b["openeval-test/valid-length"]["passed"] is True
        assert gr_b["openeval-test/contains-substring"]["passed"] is False

        # The too-long value fails the length check (and also the substring
        # check, since it doesn't contain "hi").
        r_c = next(r for r in result_set["results"] if r["test_case_id"] == "c")
        assert r_c["passed"] is False
        gr_c = {g["grader_id"]: g for g in r_c["grader_results"]}
        assert gr_c["openeval-test/valid-length"]["passed"] is False

        # test_case_ids in the ResultSet line up exactly with the suite's
        # test case ids.
        suite_ids = {tc["id"] for tc in suite["test_cases"]}
        result_ids = {r["test_case_id"] for r in result_set["results"]}
        assert suite_ids == result_ids
