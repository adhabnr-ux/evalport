"""Tests for vertexai_openeval_adapter.

These exercise to_openeval()/from_openeval()/batch_eval_result_to_openeval()
against the real `vertexai.evaluation` classes (PointwiseMetric,
PairwiseMetric, CustomMetric, PointwiseMetricPromptTemplate) -- not mocks of
this adapter's own dependencies -- and against the real
openeval.validate.validate_suite()/validate_result_set(), the same real
EvalPort validator every other adapter in this ecosystem tests against.

PointwiseMetric/PairwiseMetric are constructed (real objects, real rendered
prompt template text via str()) but never run through EvalTask.evaluate(),
since that calls the live Vertex AI Evaluation Service and needs GCP
credentials + a project. CustomMetric is different: per Vertex's own
docstring, it is "computed on the client-side using the user-defined metric
function in SDK only, not by the Vertex Gen AI Evaluation Service" -- so
TestEndToEndWithRealCustomMetric actually calls a real CustomMetric's
metric_function() directly, exactly the way vertexai/evaluation/_evaluation.py
itself does internally (verified by reading that source file), with no
network access and no mocking.
"""

import math

import pandas as pd
import pytest
from openeval.validate import validate_result_set, validate_suite
from vertexai.evaluation import (
    CustomMetric,
    PairwiseMetric,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
)

from vertexai_openeval_adapter import (
    batch_eval_result_to_openeval,
    from_openeval,
    to_openeval,
)


def _exact_match_metric():
    def _fn(instance):
        ok = (instance.get("response") or "").strip() == (instance.get("reference") or "").strip()
        return {"exact_match_custom": 1.0 if ok else 0.0}

    return CustomMetric(name="exact_match_custom", metric_function=_fn)


def _fluency_metric():
    template = PointwiseMetricPromptTemplate(
        criteria={"fluency": "The response is grammatically correct and clear."},
        rating_rubric={"1": "fluent", "0": "not fluent"},
        input_variables=["response"],
    )
    return PointwiseMetric(metric="fluency", metric_prompt_template=template)


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpeneval:
    def test_basic_shape_with_custom_metric_validates_against_real_spec(self):
        suite = to_openeval(
            instances=[
                {"prompt": "What is the capital of France?", "reference": "Paris"},
                {"prompt": "What is 2+2?", "reference": "4"},
            ],
            metrics=[_exact_match_metric()],
            suite_id="geo_and_math",
        )
        assert suite["id"] == "geo_and_math"
        assert len(suite["test_cases"]) == 2
        assert suite["test_cases"][0]["id"] == "vertex_tc_0"
        assert suite["test_cases"][0]["input"] == "What is the capital of France?"
        assert suite["test_cases"][0]["expected_output"] == "Paris"
        assert suite["test_cases"][0]["graders"] == ["exact_match_custom"]

        grader = suite["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"]["handler"] == "exact_match_custom"

        validation = validate_suite(suite)
        assert validation.valid, validation.errors

    def test_pointwise_metric_maps_to_llm_judge_with_real_prompt_and_required_tokens(self):
        metric = _fluency_metric()
        suite = to_openeval(
            instances=[{"prompt": "Tell me about Paris."}],
            metrics=[metric],
            suite_id="s1",
        )
        grader = suite["graders"][0]
        assert grader["id"] == "fluency"
        assert grader["type"] == "llm_judge"
        assert grader["params"]["model"] == "vertex-hosted-judge"
        prompt = grader["params"]["prompt"]
        # The real, literal Vertex prompt text must be present verbatim.
        assert "fluency: The response is grammatically correct and clear." in prompt
        assert "Rating Rubric" in prompt
        # And EvalPort's required tokens must also be present.
        assert "{input}" in prompt
        assert "{output}" in prompt
        assert "{expected}" in prompt
        assert grader["metadata"]["vertexai"]["metric_prompt_template"] == str(
            metric.metric_prompt_template
        )

        validation = validate_suite(suite)
        assert validation.valid, validation.errors

    def test_pairwise_metric_maps_to_custom_export_only(self):
        pw = PairwiseMetric(
            metric="pairwise_fluency",
            metric_prompt_template="Compare {response} and {baseline_response}",
        )
        suite = to_openeval(
            instances=[{"prompt": "Tell me about Paris."}],
            metrics=[pw],
        )
        grader = suite["graders"][0]
        assert grader["id"] == "pairwise_fluency"
        assert grader["type"] == "custom"
        assert grader["metadata"]["vertexai"]["class"] == "PairwiseMetric"
        validation = validate_suite(suite)
        assert validation.valid, validation.errors

    def test_multiple_metrics_each_test_case_gets_all_graders(self):
        suite = to_openeval(
            instances=[{"prompt": "hi"}],
            metrics=[_exact_match_metric(), _fluency_metric()],
        )
        assert set(suite["test_cases"][0]["graders"]) == {"exact_match_custom", "fluency"}
        assert len(suite["graders"]) == 2

    def test_extra_instance_fields_preserved_and_response_excluded(self):
        suite = to_openeval(
            instances=[
                {
                    "prompt": "hi",
                    "reference": "hello",
                    "response": "should not appear in the suite",
                    "context": "some retrieved context",
                    "instruction": "be polite",
                }
            ],
            metrics=[_exact_match_metric()],
        )
        tc = suite["test_cases"][0]
        assert "response" not in str(tc)  # the actual response text never leaks in
        extra = tc["metadata"]["vertexai"]["extra_instance_fields"]
        assert extra == {"context": "some retrieved context", "instruction": "be polite"}

    def test_explicit_ids_are_respected(self):
        suite = to_openeval(
            instances=[{"prompt": "a"}, {"prompt": "b"}],
            metrics=[_exact_match_metric()],
            ids=["tc_a", "tc_b"],
        )
        assert [tc["id"] for tc in suite["test_cases"]] == ["tc_a", "tc_b"]

    def test_missing_prompt_key_raises(self):
        with pytest.raises(ValueError, match="missing the required 'prompt'"):
            to_openeval(instances=[{"reference": "x"}], metrics=[_exact_match_metric()])

    def test_empty_instances_raises(self):
        with pytest.raises(ValueError, match="instances is empty"):
            to_openeval(instances=[], metrics=[_exact_match_metric()])

    def test_empty_metrics_raises(self):
        with pytest.raises(ValueError, match="metrics is empty"):
            to_openeval(instances=[{"prompt": "a"}], metrics=[])

    def test_mismatched_ids_length_raises(self):
        with pytest.raises(ValueError, match="ids has length"):
            to_openeval(
                instances=[{"prompt": "a"}, {"prompt": "b"}],
                metrics=[_exact_match_metric()],
                ids=["only_one"],
            )

    def test_unsupported_raw_string_metric_raises_type_error(self):
        with pytest.raises(TypeError, match="unsupported metric type"):
            to_openeval(instances=[{"prompt": "a"}], metrics=["rouge_1"])


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


class TestFromOpeneval:
    def test_reconstructs_pointwise_metric_from_llm_judge_grader(self):
        original = _fluency_metric()
        suite = to_openeval(instances=[{"prompt": "hi"}], metrics=[original])
        rebuilt = from_openeval(suite)
        assert len(rebuilt["metrics"]) == 1
        reconstructed = rebuilt["metrics"][0]
        assert isinstance(reconstructed, PointwiseMetric)
        assert reconstructed.metric_name == "fluency"
        assert str(reconstructed.metric_prompt_template) == str(original.metric_prompt_template)

    def test_does_not_reconstruct_custom_metric_grader(self):
        suite = to_openeval(instances=[{"prompt": "hi"}], metrics=[_exact_match_metric()])
        rebuilt = from_openeval(suite)
        # A real bug this test suite is written to catch: a "custom" grader's
        # class name is still visible in metadata, but there's no safe
        # generic way to instantiate an arbitrary CustomMetric function from
        # that -- it must NOT show up reconstructed.
        assert rebuilt["metrics"] == []

    def test_does_not_reconstruct_pairwise_metric_grader(self):
        pw = PairwiseMetric(metric="pw", metric_prompt_template="compare {response} {baseline_response}")
        suite = to_openeval(instances=[{"prompt": "hi"}], metrics=[pw])
        rebuilt = from_openeval(suite)
        assert rebuilt["metrics"] == []

    def test_hand_authored_llm_judge_grader_with_no_vertexai_metadata_is_skipped(self):
        suite = {
            "version": "1.0.0",
            "id": "s1",
            "graders": [{"id": "g1", "type": "llm_judge", "params": {"prompt": "{output} {input} {expected}", "model": "gpt-4o"}}],
            "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}],
        }
        rebuilt = from_openeval(suite)
        assert rebuilt["metrics"] == []

    def test_returns_instances_with_reference_and_extra_fields_and_ids_in_order(self):
        suite = to_openeval(
            instances=[
                {"prompt": "q1", "reference": "a1", "context": "ctx1"},
                {"prompt": "q2"},
            ],
            metrics=[_exact_match_metric()],
            ids=["tc1", "tc2"],
        )
        rebuilt = from_openeval(suite)
        assert rebuilt["ids"] == ["tc1", "tc2"]
        assert rebuilt["instances"][0] == {"prompt": "q1", "reference": "a1", "context": "ctx1"}
        assert rebuilt["instances"][1] == {"prompt": "q2"}

    def test_empty_suite_raises(self):
        with pytest.raises(ValueError, match="no test_cases"):
            from_openeval({"test_cases": []})

    def test_multiturn_array_input_raises(self):
        suite = {
            "version": "1.0.0",
            "id": "s1",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "x"}}],
            "test_cases": [{"id": "tc1", "input": ["turn one", "turn two"], "graders": ["g1"]}],
        }
        with pytest.raises(ValueError, match="tc1"):
            from_openeval(suite)


# ---------------------------------------------------------------------------
# batch_eval_result_to_openeval
# ---------------------------------------------------------------------------


class TestBatchEvalResultToOpeneval:
    def test_basic_conversion_validates_against_real_spec(self):
        metric = _exact_match_metric()
        metrics_table = pd.DataFrame(
            [
                {"prompt": "q1", "response": "Paris", "exact_match_custom/score": 1.0},
                {"prompt": "q2", "response": "5", "exact_match_custom/score": 0.0},
            ]
        )
        result_set = batch_eval_result_to_openeval(
            metrics_table,
            test_case_ids=["tc1", "tc2"],
            metrics=[metric],
            suite_id="geo_and_math",
            run_id="run1",
            started_at="2026-08-15T00:00:00Z",
        )
        assert result_set["suite_id"] == "geo_and_math"
        assert result_set["run_id"] == "run1"
        assert len(result_set["results"]) == 2

        r1 = result_set["results"][0]
        assert r1["test_case_id"] == "tc1"
        assert r1["passed"] is True
        assert r1["grader_results"][0]["type"] == "custom"
        assert r1["grader_results"][0]["score"] == 1.0
        assert r1["actual_output"] == "Paris"

        r2 = result_set["results"][1]
        assert r2["passed"] is False

        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors

    def test_pointwise_metric_produces_llm_judge_grader_type(self):
        metric = _fluency_metric()
        metrics_table = pd.DataFrame([{"fluency/score": 1.0, "fluency/explanation": "clear and correct"}])
        result_set = batch_eval_result_to_openeval(
            metrics_table, test_case_ids=["tc1"], metrics=[metric]
        )
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["type"] == "llm_judge"
        assert gr["metadata"]["vertexai"]["explanation"] == "clear and correct"

    def test_missing_score_column_produces_no_grader_result(self):
        metric = _exact_match_metric()
        metrics_table = pd.DataFrame([{"prompt": "hi"}])  # no score column at all
        result_set = batch_eval_result_to_openeval(
            metrics_table, test_case_ids=["tc1"], metrics=[metric]
        )
        assert result_set["results"][0]["grader_results"] == []
        assert result_set["results"][0]["passed"] is False

    def test_nan_score_produces_no_grader_result(self):
        metric = _exact_match_metric()
        metrics_table = pd.DataFrame([{"exact_match_custom/score": math.nan}])
        result_set = batch_eval_result_to_openeval(
            metrics_table, test_case_ids=["tc1"], metrics=[metric]
        )
        assert result_set["results"][0]["grader_results"] == []
        assert result_set["results"][0]["passed"] is False

    def test_score_clamped_to_valid_range(self):
        metric = _exact_match_metric()
        metrics_table = pd.DataFrame([{"exact_match_custom/score": 1.5}])
        result_set = batch_eval_result_to_openeval(
            metrics_table, test_case_ids=["tc1"], metrics=[metric]
        )
        assert result_set["results"][0]["grader_results"][0]["score"] == 1.0
        assert validate_result_set(result_set).valid

    def test_run_id_auto_generated_when_omitted(self):
        metric = _exact_match_metric()
        metrics_table = pd.DataFrame([{"exact_match_custom/score": 1.0}])
        result_set = batch_eval_result_to_openeval(
            metrics_table, test_case_ids=["tc1"], metrics=[metric]
        )
        assert result_set["run_id"].startswith("vertex_run_")

    def test_empty_metrics_table_raises(self):
        with pytest.raises(ValueError, match="metrics_table is empty"):
            batch_eval_result_to_openeval(pd.DataFrame(), test_case_ids=["tc1"], metrics=[])

    def test_mismatched_row_count_raises(self):
        metric = _exact_match_metric()
        metrics_table = pd.DataFrame([{"exact_match_custom/score": 1.0}])
        with pytest.raises(ValueError, match="rows, expected"):
            batch_eval_result_to_openeval(
                metrics_table, test_case_ids=["tc1", "tc2"], metrics=[metric]
            )

    def test_summary_matches_actual_pass_fail_counts(self):
        metric = _exact_match_metric()
        metrics_table = pd.DataFrame(
            [
                {"exact_match_custom/score": 1.0},
                {"exact_match_custom/score": 0.0},
                {"exact_match_custom/score": 1.0},
            ]
        )
        result_set = batch_eval_result_to_openeval(
            metrics_table, test_case_ids=["tc1", "tc2", "tc3"], metrics=[metric]
        )
        assert result_set["summary"]["total"] == 3
        assert result_set["summary"]["passed"] == 2
        assert result_set["summary"]["failed"] == 1
        assert result_set["summary"]["pass_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# End-to-end: suite -> real CustomMetric.metric_function() run -> results,
# both validated against the real spec. CustomMetric is genuinely local
# (Vertex's own docstring: "computed on the client-side... not by the
# Vertex Gen AI Evaluation Service"), so this actually runs it -- no
# mocking, no network.
# ---------------------------------------------------------------------------


class TestEndToEndWithRealCustomMetric:
    def test_full_round_trip_suite_to_run_to_resultset(self):
        metric = _exact_match_metric()
        suite = to_openeval(
            instances=[
                {"prompt": "What is the capital of France?", "reference": "Paris"},
                {"prompt": "What is 2+2?", "reference": "4"},
            ],
            metrics=[metric],
            ids=["geo1", "math1"],
            suite_id="e2e_suite",
        )
        assert validate_suite(suite).valid

        rebuilt = from_openeval(suite)
        assert [inst["prompt"] for inst in rebuilt["instances"]] == [
            "What is the capital of France?",
            "What is 2+2?",
        ]

        # Simulate the system under test, then actually run the real
        # CustomMetric.metric_function -- exactly the way
        # vertexai/evaluation/_evaluation.py itself invokes it internally
        # (`custom_metric.metric_function(row_dict)`) -- no network, no mocking.
        my_app_outputs = ["Paris", "5"]  # second answer is wrong on purpose
        rows = []
        for i, inst in enumerate(rebuilt["instances"]):
            row = dict(inst)
            row["response"] = my_app_outputs[i]
            metric_output = metric.metric_function(row)
            row[f"{metric.name}/score"] = metric_output[metric.name]
            rows.append(row)

        metrics_table = pd.DataFrame(rows)

        result_set = batch_eval_result_to_openeval(
            metrics_table,
            test_case_ids=rebuilt["ids"],
            metrics=[metric],
            suite_id=suite["id"],
            started_at="2026-08-15T00:00:00Z",
        )
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        assert result_set["summary"]["passed"] == 1
        assert result_set["summary"]["failed"] == 1

        suite_ids = {tc["id"] for tc in suite["test_cases"]}
        assert all(r["test_case_id"] in suite_ids for r in result_set["results"])
