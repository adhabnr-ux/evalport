"""Tests for openeval.convert: from_promptfoo(), compute_summary(), create_result_set().

None of these three functions had any test coverage before this file, despite being
shipped in the SDK and referenced from the project README/tracking issue as one of the
four "core converters" already supported. These tests close that gap, verified against
the real openeval.validate.validate_suite()/validate_result_set() -- not shape
assertions, actual spec validation.

The promptfoo `tests`/`assert`/`providers` shape mirrors Promptfoo's real, documented
YAML/JS test-case format (https://www.promptfoo.dev/docs/configuration/test-cases/):
a top-level `tests` list, each with `vars` (template variables fed to the prompt) and
`assert` (a list of `{type, value}` assertion objects), plus a top-level `providers`
list of `{id, model, ...}` provider configs referenced by the eval config.
"""
from __future__ import annotations

from openeval.convert import compute_summary, create_result_set, from_promptfoo
from openeval.types import OPENEVAL_VERSION
from openeval.validate import validate_result_set, validate_suite


# ---------------------------------------------------------------------------
# from_promptfoo()
# ---------------------------------------------------------------------------


def test_from_promptfoo_produces_spec_valid_suite():
    pf = {
        "tests": [
            {
                "vars": {"query": "What is the capital of France?"},
                "assert": [{"type": "equals", "value": "Paris"}],
            },
            {
                "vars": {"query": "Name a fruit."},
                "assert": [{"type": "contains", "value": "apple"}],
            },
        ],
        "providers": [{"id": "openai:gpt-4o-mini", "model": "gpt-4o-mini"}],
    }
    suite = from_promptfoo(pf)
    result = validate_suite(suite)
    assert result.valid, result.errors
    # Asserted against the live constant, not a hardcoded literal: a hardcoded
    # "1.0.0-rc.1" here is exactly the kind of silent-drift bug that let
    # OPENEVAL_VERSION itself go stale for a full spec revision undetected
    # (fixed in 1.0.0-rc.3 -- see openeval/types.py's comment on the constant).
    assert suite["version"] == OPENEVAL_VERSION
    assert len(suite["test_cases"]) == 2


def test_from_promptfoo_maps_equals_to_exact_match():
    pf = {"tests": [{"vars": {"query": "2+2?"}, "assert": [{"type": "equals", "value": "4"}]}]}
    suite = from_promptfoo(pf)
    assert suite["graders"][0]["type"] == "exact_match"
    assert suite["test_cases"][0]["graders"] == [suite["graders"][0]["id"]]


def test_from_promptfoo_maps_contains_with_substring_param():
    pf = {"tests": [{"vars": {"query": "list a color"}, "assert": [{"type": "contains", "value": "blue"}]}]}
    suite = from_promptfoo(pf)
    grader = suite["graders"][0]
    assert grader["type"] == "contains"
    assert grader["params"]["substring"] == "blue"
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_from_promptfoo_maps_unknown_assert_type_to_custom_with_handler():
    pf = {
        "tests": [
            {"vars": {"query": "translate hello"}, "assert": [{"type": "llm-rubric", "value": "is a translation"}]}
        ]
    }
    suite = from_promptfoo(pf)
    grader = suite["graders"][0]
    assert grader["type"] == "custom"
    assert grader["params"]["handler"] == "promptfoo:llm-rubric"
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_from_promptfoo_preserves_expected_output_from_vars():
    pf = {
        "tests": [
            {
                "vars": {"query": "2+2?", "expected": "4"},
                "assert": [{"type": "equals", "value": "4"}],
            }
        ]
    }
    suite = from_promptfoo(pf)
    assert suite["test_cases"][0]["expected_output"] == "4"


def test_from_promptfoo_with_no_asserts_gets_default_grader():
    """A promptfoo test case with no `assert` list still has to satisfy EvalPort's
    graders.minItems:1 requirement -- from_promptfoo() falls back to a shared
    gr_default exact_match grader rather than producing an invalid TestCase."""
    pf = {"tests": [{"vars": {"query": "just checking it runs"}}]}
    suite = from_promptfoo(pf)
    assert suite["test_cases"][0]["graders"] == ["gr_default"]
    assert suite["graders"] == [{"id": "gr_default", "type": "exact_match"}]
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_from_promptfoo_falls_back_to_prompt_var_when_no_query():
    pf = {"tests": [{"vars": {"prompt": "Summarize this document."}, "assert": [{"type": "equals", "value": "ok"}]}]}
    suite = from_promptfoo(pf)
    assert suite["test_cases"][0]["input"] == "Summarize this document."


def test_from_promptfoo_extracts_provider_model_into_config():
    pf = {
        "tests": [{"vars": {"query": "hi"}, "assert": [{"type": "equals", "value": "hello"}]}],
        "providers": [{"id": "anthropic:claude-3-5-sonnet", "model": "claude-3-5-sonnet-20241022"}],
    }
    suite = from_promptfoo(pf)
    assert suite["config"]["provider"]["model"] == "claude-3-5-sonnet-20241022"


def test_from_promptfoo_handles_multiple_asserts_per_test():
    pf = {
        "tests": [
            {
                "vars": {"query": "describe a dog"},
                "assert": [
                    {"type": "contains", "value": "animal"},
                    {"type": "equals", "value": "A dog is a domesticated animal."},
                ],
            }
        ]
    }
    suite = from_promptfoo(pf)
    assert len(suite["graders"]) == 2
    assert len(suite["test_cases"][0]["graders"]) == 2
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_from_promptfoo_empty_test_list_still_gets_default_grader():
    pf = {"tests": []}
    suite = from_promptfoo(pf)
    assert suite["graders"] == [{"id": "gr_default", "type": "exact_match"}]
    assert suite["test_cases"] == []


# ---------------------------------------------------------------------------
# compute_summary()
# ---------------------------------------------------------------------------


def test_compute_summary_counts_pass_fail_and_averages_scores():
    results = [
        {"passed": True, "grader_results": [{"score": 1.0}, {"score": 0.8}]},
        {"passed": False, "grader_results": [{"score": 0.0}]},
    ]
    summary = compute_summary(results)
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["pass_rate"] == 0.5
    assert abs(summary["avg_score"] - (1.0 + 0.8 + 0.0) / 3) < 1e-9


def test_compute_summary_ignores_none_scores_in_average():
    results = [
        {"passed": True, "grader_results": [{"score": None}, {"score": 1.0}]},
    ]
    summary = compute_summary(results)
    assert summary["avg_score"] == 1.0


def test_compute_summary_empty_results_does_not_divide_by_zero():
    summary = compute_summary([])
    assert summary["total"] == 0
    assert summary["pass_rate"] == 0
    assert summary["avg_score"] == 0


# ---------------------------------------------------------------------------
# create_result_set()
# ---------------------------------------------------------------------------


def test_create_result_set_produces_spec_valid_result_set():
    pf = {"tests": [{"vars": {"query": "2+2?"}, "assert": [{"type": "equals", "value": "4"}]}]}
    suite = from_promptfoo(pf)
    tc = suite["test_cases"][0]
    grader_id = tc["graders"][0]
    results = [
        {
            "test_case_id": tc["id"],
            "passed": True,
            "grader_results": [{"grader_id": grader_id, "type": "exact_match", "score": 1.0, "passed": True}],
        }
    ]
    rs = create_result_set(suite, results, run_id="run-1")
    validation = validate_result_set(rs)
    assert validation.valid, validation.errors
    assert rs["suite_id"] == suite["id"]
    assert rs["run_id"] == "run-1"
    assert rs["summary"]["total"] == 1
    assert rs["summary"]["passed"] == 1


def test_create_result_set_carries_provider_from_suite_config():
    suite = {
        "version": "1.0.0-rc.1",
        "id": "s1",
        "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}],
        "graders": [{"id": "g1", "type": "exact_match"}],
        "config": {"provider": {"model": "gpt-4o-mini"}},
    }
    results = [
        {
            "test_case_id": "tc1",
            "passed": True,
            "grader_results": [{"grader_id": "g1", "type": "exact_match", "score": 1.0, "passed": True}],
        }
    ]
    rs = create_result_set(suite, results, run_id="run-2")
    assert rs["provider"] == {"model": "gpt-4o-mini"}
    assert validate_result_set(rs).valid


def test_full_promptfoo_round_trip_suite_and_result_set_both_validate():
    """End-to-end: a realistic multi-test-case promptfoo export -> EvalPort Suite
    -> simulated grading -> EvalPort ResultSet, both validated against the real
    spec validators, matching the pattern every adapter's own end-to-end test uses."""
    pf = {
        "tests": [
            {"vars": {"query": "2+2?"}, "assert": [{"type": "equals", "value": "4"}]},
            {"vars": {"query": "list a fruit"}, "assert": [{"type": "contains", "value": "apple"}]},
        ],
        "providers": [{"id": "openai:gpt-4o-mini", "model": "gpt-4o-mini"}],
    }
    suite = from_promptfoo(pf)
    assert validate_suite(suite).valid

    results = []
    for i, tc in enumerate(suite["test_cases"]):
        grader_id = tc["graders"][0]
        grader = next(g for g in suite["graders"] if g["id"] == grader_id)
        passed = i == 0
        results.append(
            {
                "test_case_id": tc["id"],
                "passed": passed,
                "grader_results": [
                    {"grader_id": grader_id, "type": grader["type"], "score": 1.0 if passed else 0.0, "passed": passed}
                ],
            }
        )
    rs = create_result_set(suite, results, run_id="run-e2e")
    validation = validate_result_set(rs)
    assert validation.valid, validation.errors
    assert rs["summary"]["passed"] == 1
    assert rs["summary"]["failed"] == 1
    assert rs["summary"]["pass_rate"] == 0.5
