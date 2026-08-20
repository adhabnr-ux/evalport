"""Tests for huggingface_evaluate_openeval_adapter, run against the real
`evaluate` package (real EvaluationModule.compute() calls, not mocks) and
the real EvalPort validator (`openeval.validate.validate_suite`/
`validate_result_set`).

Requires `evaluate` and `scikit-learn` (for the `accuracy`/`f1` metrics
exercised here) -- see README.md "Running the tests" for the install
command. Network access is required: `evaluate.load()` fetches each
metric's builder script from the Hugging Face Hub the first time it's used,
even for metrics with no other runtime dependency.
"""

from __future__ import annotations

import pytest
from openeval.validate import validate_result_set, validate_suite

pytest.importorskip("evaluate", reason="evaluate is required for these tests; see README.md for the install command.")

from huggingface_evaluate_openeval_adapter import (  # noqa: E402
    compute_per_example,
    from_openeval,
    metric_result_to_openeval,
    to_openeval,
)


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpenevalExactMatch:
    def test_basic_suite_is_valid(self):
        inputs = ["What is the capital of France?", "What is 2+2?"]
        references = ["Paris", "4"]
        suite = to_openeval(inputs, references, "exact_match", suite_id="geo_and_math")

        assert suite["id"] == "geo_and_math"
        assert suite["version"] == "1.0.0"
        assert len(suite["test_cases"]) == 2
        tc = suite["test_cases"][0]
        assert tc["id"] == "case_0"
        assert tc["input"] == "What is the capital of France?"
        assert tc["expected_output"] == "Paris"
        assert tc["graders"] == [{"id": "exact_match", "type": "exact_match"}]

        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_default_suite_id_includes_metric_name(self):
        suite = to_openeval(["hi"], ["hey"], "exact_match")
        assert suite["id"] == "huggingface_evaluate_exact_match"

    def test_description_is_passed_through(self):
        suite = to_openeval(["hi"], ["hey"], "exact_match", description="A tiny suite.")
        assert suite["description"] == "A tiny suite."

    def test_custom_test_case_ids_are_used(self):
        suite = to_openeval(["hi", "bye"], ["hey", "later"], "exact_match", test_case_ids=["greet", "farewell"])
        ids = [tc["id"] for tc in suite["test_cases"]]
        assert ids == ["greet", "farewell"]

    def test_tags_are_applied_to_every_case(self):
        suite = to_openeval(["hi"], ["hey"], "exact_match", tags=["smoke"])
        assert suite["test_cases"][0]["tags"] == ["smoke"]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            to_openeval(["hi", "bye"], ["hey"], "exact_match")

    def test_mismatched_test_case_ids_length_raises(self):
        with pytest.raises(ValueError, match="test_case_ids"):
            to_openeval(["hi"], ["hey"], "exact_match", test_case_ids=["a", "b"])


class TestToOpenevalCustomMetric:
    def test_non_exact_match_maps_to_custom(self):
        suite = to_openeval([" "], [1], "accuracy", metric_kwargs={"normalize": True})
        grader = suite["test_cases"][0]["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"]["handler"] == "huggingface_evaluate:accuracy"
        assert grader["params"]["metric_kwargs"] == {"normalize": True}
        assert validate_suite(suite).valid

    def test_non_string_reference_is_stringified_and_type_preserved(self):
        suite = to_openeval(["predict the label"], [1], "accuracy")
        tc = suite["test_cases"][0]
        assert tc["expected_output"] == "1"
        assert tc["metadata"]["huggingface_evaluate"]["reference_type"] == "int"
        assert validate_suite(suite).valid


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


class TestFromOpeneval:
    def test_round_trips_exact_match_suite(self):
        suite = to_openeval(
            ["What is the capital of France?", "What is 2+2?"],
            ["Paris", "4"],
            "exact_match",
            test_case_ids=["geo", "math"],
        )
        groups = from_openeval(suite)
        assert len(groups) == 1
        group = groups[0]
        assert group["metric_name"] == "exact_match"
        assert group["test_case_ids"] == ["geo", "math"]
        assert group["inputs"] == ["What is the capital of France?", "What is 2+2?"]
        assert group["references"] == ["Paris", "4"]

    def test_round_trips_custom_metric_suite_with_kwargs_and_int_reference(self):
        suite = to_openeval(
            ["predict the label"], [1], "accuracy", metric_kwargs={"normalize": True}, test_case_ids=["c1"]
        )
        groups = from_openeval(suite)
        assert len(groups) == 1
        group = groups[0]
        assert group["metric_name"] == "accuracy"
        assert group["references"] == [1]  # cast back to int, not left as the string "1"
        assert group["metric_kwargs"] == {"normalize": True}

    def test_multi_metric_suite_groups_separately(self):
        em_suite = to_openeval(["hi"], ["hey"], "exact_match", test_case_ids=["em1"])
        acc_suite = to_openeval(["predict"], [0], "accuracy", test_case_ids=["acc1"])
        merged = {
            "version": "1.0.0",
            "id": "mixed",
            "test_cases": em_suite["test_cases"] + acc_suite["test_cases"],
        }
        assert validate_suite(merged).valid

        groups = from_openeval(merged)
        assert {g["metric_name"] for g in groups} == {"exact_match", "accuracy"}

    def test_multi_turn_input_array_is_joined_for_reconstruction(self):
        suite = to_openeval(["hi"], ["hey"], "exact_match", test_case_ids=["c1"])
        suite["test_cases"][0]["input"] = ["Hello", "how are you?"]
        groups = from_openeval(suite)
        assert groups[0]["inputs"] == ["Hello how are you?"]

    def test_bare_grader_id_string_is_clean_skipped(self):
        suite = {
            "version": "1.0.0",
            "id": "s",
            "test_cases": [{"id": "c1", "input": "hi", "graders": ["some_shared_grader_id"]}],
        }
        assert from_openeval(suite) == []

    def test_unrecognized_custom_handler_is_clean_skipped(self):
        suite = {
            "version": "1.0.0",
            "id": "s",
            "test_cases": [
                {
                    "id": "c1",
                    "input": "hi",
                    "graders": [{"id": "g1", "type": "custom", "params": {"handler": "some_other_tool:thing"}}],
                }
            ],
        }
        assert from_openeval(suite) == []


# ---------------------------------------------------------------------------
# compute_per_example
# ---------------------------------------------------------------------------


class TestComputePerExample:
    def test_exact_match_per_example_scores_are_real_and_binary(self):
        predictions = ["Paris", "5", "Berlin"]
        references = ["Paris", "4", "Berlin"]
        item_scores, aggregate = compute_per_example("exact_match", predictions, references)

        assert item_scores == [1.0, 0.0, 1.0]
        assert aggregate["exact_match"] == pytest.approx(2 / 3)

    def test_accuracy_per_example_scores_are_real(self):
        predictions = [1, 0, 1, 1]
        references = [1, 1, 1, 0]
        item_scores, aggregate = compute_per_example("accuracy", predictions, references)

        assert item_scores == [1.0, 0.0, 1.0, 0.0]
        assert aggregate["accuracy"] == pytest.approx(0.5)

    def test_f1_per_example_scores_are_real(self):
        predictions = [1, 0, 1, 1]
        references = [1, 1, 1, 0]
        item_scores, aggregate = compute_per_example("f1", predictions, references)

        # f1's per-example score for a single (pred, ref) pair is degenerate
        # (0.0 or 1.0 depending on agreement) -- still a real number from the
        # real metric, exercised here to prove the multi-output-key
        # extraction path works the same way for f1 as for accuracy.
        assert len(item_scores) == 4
        assert all(s in (0.0, 1.0) for s in item_scores)
        assert "f1" in aggregate

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            compute_per_example("exact_match", ["a", "b"], ["a"])


# ---------------------------------------------------------------------------
# metric_result_to_openeval
# ---------------------------------------------------------------------------


class TestMetricResultToOpeneval:
    def test_real_exact_match_run_produces_valid_result_set(self):
        predictions = ["Paris", "5", "Berlin"]
        references = ["Paris", "4", "Berlin"]
        item_scores, aggregate = compute_per_example("exact_match", predictions, references)

        result_set = metric_result_to_openeval(
            predictions,
            references,
            "exact_match",
            item_scores,
            suite_id="geo_and_math",
            run_id="run-1",
            started_at="2026-08-20T00:00:00Z",
            completed_at="2026-08-20T00:00:01Z",
            aggregate=aggregate,
            test_case_ids=["geo", "math", "geo2"],
        )

        assert result_set["suite_id"] == "geo_and_math"
        assert result_set["summary"]["total"] == 3
        assert result_set["summary"]["passed"] == 2
        assert result_set["summary"]["failed"] == 1
        assert result_set["summary"]["pass_rate"] == pytest.approx(2 / 3)
        assert result_set["metadata"]["huggingface_evaluate"]["aggregate"]["exact_match"] == pytest.approx(2 / 3)

        by_id = {r["test_case_id"]: r for r in result_set["results"]}
        assert by_id["geo"]["passed"] is True
        assert by_id["geo"]["grader_results"][0]["score"] == 1.0
        assert by_id["math"]["passed"] is False
        assert by_id["math"]["grader_results"][0]["score"] == 0.0
        assert by_id["math"]["actual_output"] == "5"

        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors

    def test_real_accuracy_run_maps_to_custom_grader_type(self):
        predictions = [1, 0, 1, 1]
        references = [1, 1, 1, 0]
        item_scores, aggregate = compute_per_example("accuracy", predictions, references)

        result_set = metric_result_to_openeval(
            predictions,
            references,
            "accuracy",
            item_scores,
            suite_id="labels",
            run_id="run-2",
            started_at="2026-08-20T00:00:00Z",
            aggregate=aggregate,
        )

        assert result_set["results"][0]["grader_results"][0]["type"] == "custom"
        assert result_set["summary"]["avg_score"] == pytest.approx(0.5)
        assert validate_result_set(result_set).valid

    def test_score_outside_unit_range_is_clamped_and_raw_preserved(self):
        result_set = metric_result_to_openeval(
            ["x"],
            ["y"],
            "some_unbounded_measurement",
            [1.7],
            suite_id="s",
            run_id="r",
            started_at="2026-08-20T00:00:00Z",
        )
        grader_result = result_set["results"][0]["grader_results"][0]
        assert grader_result["score"] == 1.0
        assert grader_result["metadata"]["huggingface_evaluate"]["raw_score"] == 1.7
        assert validate_result_set(result_set).valid

    def test_custom_threshold_changes_pass_fail(self):
        result_set = metric_result_to_openeval(
            ["x"], ["y"], "some_metric", [0.6], suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z", threshold=0.9
        )
        assert result_set["results"][0]["passed"] is False

        result_set_lenient = metric_result_to_openeval(
            ["x"], ["y"], "some_metric", [0.6], suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z", threshold=0.5
        )
        assert result_set_lenient["results"][0]["passed"] is True

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            metric_result_to_openeval(
                ["a", "b"], ["a"], "exact_match", [1.0], suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z"
            )

    def test_empty_results_still_validate(self):
        result_set = metric_result_to_openeval(
            [], [], "exact_match", [], suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z"
        )
        assert result_set["summary"]["total"] == 0
        assert result_set["summary"]["pass_rate"] == 0.0
        # An empty results[] array is intentionally NOT asserted valid here --
        # spec/schemas/resultset.json requires results to be non-empty
        # (minItems: 1); this test documents that this adapter doesn't
        # silently paper over an empty batch, the caller sees the same
        # validation error a hand-built empty ResultSet would produce.
        validation = validate_result_set(result_set)
        assert not validation.valid
