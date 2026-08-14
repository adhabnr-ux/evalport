"""Tests for langfuse_openeval_adapter.

Run against the real ``langfuse`` package's ``DatasetItem`` pydantic model
and ``langfuse.experiment`` classes (``Evaluation``, ``ExperimentItemResult``,
``ExperimentResult``) -- no mocks -- and against the real EvalPort
validators (openeval.validate). ``Langfuse().run_experiment()`` itself
requires a live Langfuse project and network access, so these tests build
the exact object shapes it produces (constructed via the real classes
imported from the installed package, not hand-rolled dicts) rather than
calling it.
"""

from __future__ import annotations

import datetime

import pytest
from langfuse.api.commons.types.dataset_item import DatasetItem
from langfuse.api.commons.types.dataset_status import DatasetStatus
from langfuse.experiment import Evaluation, ExperimentItemResult, ExperimentResult
from openeval.validate import validate_result_set, validate_suite

from langfuse_openeval_adapter import (
    experiment_result_to_openeval,
    from_openeval,
    to_openeval,
)


def _dataset_item(item_id, input_, expected_output, metadata=None):
    return DatasetItem(
        id=item_id,
        status=DatasetStatus.ACTIVE,
        input=input_,
        expected_output=expected_output,
        metadata=metadata or {},
        dataset_id="ds_1",
        dataset_name="geo_science_eval",
        created_at=datetime.datetime(2026, 1, 1),
        updated_at=datetime.datetime(2026, 1, 1),
        media_references=[],
    )


def _dataset_items():
    return [
        _dataset_item("item_0", {"question": "What is the capital of France?"}, "Paris"),
        _dataset_item("item_1", {"question": "What is the boiling point of water?"}, "100 degrees Celsius"),
    ]


# --- to_openeval ---------------------------------------------------------


class TestToOpenEval:
    def test_converts_real_dataset_items(self):
        suite = to_openeval(_dataset_items(), suite_id="geo_science_eval")
        assert validate_suite(suite).valid
        assert len(suite["test_cases"]) == 2

    def test_accepts_local_experiment_item_dicts(self):
        items = [
            {"input": "2+2?", "expected_output": "4", "metadata": {"topic": "math"}},
            {"input": "capital of Japan?", "expected_output": "Tokyo"},
        ]
        suite = to_openeval(items)
        assert validate_suite(suite).valid
        assert suite["test_cases"][0]["input"] == "2+2?"
        assert suite["test_cases"][0]["expected_output"] == "4"

    def test_uses_dataset_item_id_as_test_case_id(self):
        suite = to_openeval(_dataset_items())
        assert [tc["id"] for tc in suite["test_cases"]] == ["item_0", "item_1"]

    def test_falls_back_to_positional_id_for_local_items_without_id(self):
        items = [{"input": "q", "expected_output": "a"}]
        suite = to_openeval(items)
        assert suite["test_cases"][0]["id"] == "row_0"

    def test_dict_input_is_stringified_as_json(self):
        suite = to_openeval(_dataset_items())
        tc = suite["test_cases"][0]
        assert "France" in tc["input"]

    def test_preserves_full_raw_item_including_langfuse_only_fields(self):
        suite = to_openeval(_dataset_items())
        tc = suite["test_cases"][0]
        item = tc["metadata"]["langfuse"]["item"]
        assert item["id"] == "item_0"
        assert item["dataset_name"] == "geo_science_eval"
        assert item["status"] == "ACTIVE"

    def test_item_without_expected_output_has_no_expected_output_field(self):
        items = [{"input": "q"}]
        suite = to_openeval(items)
        assert "expected_output" not in suite["test_cases"][0]

    def test_default_grader_is_llm_judge_with_required_params(self):
        suite = to_openeval(_dataset_items())
        assert suite["graders"][0]["type"] == "llm_judge"
        assert "model" in suite["graders"][0]["params"]
        assert "prompt" in suite["graders"][0]["params"]

    def test_exact_match_grader_type_option(self):
        suite = to_openeval(_dataset_items(), grader_type="exact_match")
        assert suite["graders"][0]["type"] == "exact_match"
        assert validate_suite(suite).valid


# --- from_openeval ---------------------------------------------------------


class TestFromOpenEval:
    def test_round_trip_preserves_original_dataset_item_fields(self):
        suite = to_openeval(_dataset_items(), suite_id="geo_science_eval")
        rows = from_openeval(suite)
        assert rows[0]["id"] == "item_0"
        assert rows[0]["dataset_name"] == "geo_science_eval"
        assert rows[0]["expected_output"] == "Paris"

    def test_round_tripped_rows_are_usable_as_run_experiment_data(self):
        suite = to_openeval(_dataset_items())
        rows = from_openeval(suite)
        # LocalExperimentItem / DatasetItem-shaped: must have input + expected_output.
        for row in rows:
            assert "input" in row
            assert "expected_output" in row

    def test_builds_fresh_local_experiment_item_when_no_langfuse_metadata_present(self):
        suite = {
            "version": "1.0.0",
            "id": "manual_suite",
            "test_cases": [
                {
                    "id": "tc1",
                    "input": "What's 2+2?",
                    "expected_output": "4",
                    "graders": ["g1"],
                }
            ],
        }
        rows = from_openeval(suite)
        assert rows == [{"input": "What's 2+2?", "expected_output": "4"}]

    def test_fresh_row_carries_context_into_metadata(self):
        suite = {
            "version": "1.0.0",
            "id": "manual_suite",
            "test_cases": [
                {
                    "id": "tc1",
                    "input": "q",
                    "context": ["some background"],
                    "graders": ["g1"],
                }
            ],
        }
        rows = from_openeval(suite)
        assert rows[0]["metadata"]["context"] == ["some background"]


# --- experiment_result_to_openeval -----------------------------------------


class TestExperimentResultToOpenEval:
    def _experiment_result(self):
        items = _dataset_items()
        ev_high = Evaluation(name="correctness", value=0.95, comment="Matches expected answer.", data_type="NUMERIC")
        ev_low = Evaluation(name="correctness", value=0.1, comment="Does not match.", data_type="NUMERIC")
        ev_bool = Evaluation(name="is_concise", value=True, data_type="BOOLEAN")
        ev_cat = Evaluation(name="rubric_grade", value="good", data_type="CATEGORICAL")

        item_results = [
            ExperimentItemResult(
                item=items[0],
                output="Paris is the capital of France.",
                evaluations=[ev_high, ev_bool],
                trace_id="trace_0",
                dataset_run_id="run_abc",
            ),
            ExperimentItemResult(
                item=items[1],
                output="I don't know.",
                evaluations=[ev_low, ev_cat],
                trace_id="trace_1",
                dataset_run_id="run_abc",
            ),
        ]
        run_eval = Evaluation(name="overall_run_quality", value=0.8, data_type="NUMERIC")
        return ExperimentResult(
            name="geo_science_eval",
            run_name="run_abc",
            description="A test run",
            item_results=item_results,
            run_evaluations=[run_eval],
            experiment_id="exp_123",
            dataset_run_id="run_abc",
        )

    def test_converts_real_experiment_result_shape(self):
        rs = experiment_result_to_openeval(
            self._experiment_result(), suite_id="geo_science_eval", started_at="2026-08-14T00:00:00Z"
        )
        assert validate_result_set(rs).valid
        assert len(rs["results"]) == 2
        assert len(rs["results"][0]["grader_results"]) == 2

    def test_run_id_defaults_to_run_name(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        assert rs["run_id"] == "run_abc"

    def test_numeric_evaluation_clamped_and_passes_at_threshold(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        gr = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "langfuse_correctness")
        assert gr["score"] == 0.95
        assert gr["passed"] is True
        gr_low = next(g for g in rs["results"][1]["grader_results"] if g["grader_id"] == "langfuse_correctness")
        assert gr_low["score"] == 0.1
        assert gr_low["passed"] is False

    def test_boolean_evaluation_maps_to_one_or_zero(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        gr = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "langfuse_is_concise")
        assert gr["score"] == 1.0
        assert gr["passed"] is True

    def test_categorical_evaluation_has_null_score_and_preserves_raw_value(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        gr = next(g for g in rs["results"][1]["grader_results"] if g["grader_id"] == "langfuse_rubric_grade")
        assert gr["score"] is None
        assert gr["passed"] is True  # "good" is in the affirmative label set
        assert gr["metadata"]["value"] == "good"

    def test_comment_becomes_reason(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        gr = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "langfuse_correctness")
        assert gr["reason"] == "Matches expected answer."

    def test_test_case_id_comes_from_dataset_item_id(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        assert [r["test_case_id"] for r in rs["results"]] == ["item_0", "item_1"]

    def test_actual_output_comes_from_item_result_output(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        assert rs["results"][0]["actual_output"] == "Paris is the capital of France."

    def test_trace_id_and_dataset_run_id_preserved_in_result_metadata(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        assert rs["results"][0]["metadata"]["trace_id"] == "trace_0"
        assert rs["results"][0]["metadata"]["dataset_run_id"] == "run_abc"

    def test_run_evaluations_preserved_under_top_level_metadata(self):
        rs = experiment_result_to_openeval(self._experiment_result(), started_at="2026-08-14T00:00:00Z")
        run_evals = rs["metadata"]["langfuse"]["run_evaluations"]
        assert run_evals[0]["name"] == "overall_run_quality"
        assert run_evals[0]["value"] == 0.8

    def test_accepts_bare_list_of_item_results_without_experiment_result_wrapper(self):
        er = self._experiment_result()
        rs = experiment_result_to_openeval(er.item_results, started_at="2026-08-14T00:00:00Z")
        assert validate_result_set(rs).valid
        assert len(rs["results"]) == 2
        # No run_evaluations available from a bare list -> no top-level metadata key for it.
        assert "metadata" not in rs or "run_evaluations" not in rs.get("metadata", {}).get("langfuse", {})

    def test_item_result_with_no_evaluations_produces_no_grader_results_and_fails(self):
        items = _dataset_items()
        item_result = ExperimentItemResult(
            item=items[0], output="some output", evaluations=[], trace_id="t0", dataset_run_id=None
        )
        rs = experiment_result_to_openeval([item_result], started_at="2026-08-14T00:00:00Z")
        assert rs["results"][0]["grader_results"] == []
        assert rs["results"][0]["passed"] is False
        assert validate_result_set(rs).valid

    def test_full_pipeline_suite_to_result_set(self):
        suite = to_openeval(_dataset_items(), suite_id="geo_science_eval")
        assert validate_suite(suite).valid

        rows = from_openeval(suite)
        items = _dataset_items()
        item_results = [
            ExperimentItemResult(
                item=items[i],
                output=f"simulated output {i}",
                evaluations=[Evaluation(name="quality", value=score, data_type="NUMERIC")],
                trace_id=f"trace_{i}",
                dataset_run_id="run_full",
            )
            for i, score in enumerate([1.0, 0.2])
        ]
        result_set = experiment_result_to_openeval(
            item_results, suite_id=suite["id"], run_id="full_run",
            started_at="2026-08-14T00:00:00Z", completed_at="2026-08-14T00:00:05Z",
        )
        assert validate_result_set(result_set).valid
        assert [r["passed"] for r in result_set["results"]] == [True, False]
        assert len(rows) == 2
