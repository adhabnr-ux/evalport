"""Tests for uptrain_openeval_adapter.

Run against the real `uptrain` package's `Evals` enum and `DataSchema`
defaults (to confirm field names aren't guessed), and against the real
EvalPort validators (openeval.validate) -- no mocks. `EvalLLM.evaluate()`
itself requires a live OpenAI key and network access, so these tests build
the exact result shape it produces (read directly from its source, see the
adapter module docstring) rather than calling it.
"""

from __future__ import annotations

import pandas as pd
from openeval.validate import validate_result_set, validate_suite
from uptrain import Evals
from uptrain.framework.remote import DataSchema

from uptrain_openeval_adapter import from_openeval, results_to_openeval, to_openeval


def test_uptrain_default_schema_field_names_match_adapter_constants():
    # Guards against the adapter silently drifting from UpTrain's real
    # DataSchema defaults if a future uptrain release renames a field.
    schema = DataSchema().model_dump()
    assert schema["question"] == "question"
    assert schema["response"] == "response"
    assert schema["context"] == "context"
    assert schema["ground_truth"] == "ground_truth"


def test_uptrain_context_relevance_check_value_matches_adapter_expectation():
    assert Evals.CONTEXT_RELEVANCE.value == "context_relevance"


def _dataset():
    return [
        {
            "id": "0",
            "question": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "context": "France is a country in Europe. Its capital is Paris.",
            "ground_truth": "Paris",
        },
        {
            "id": "1",
            "question": "What is the boiling point of water?",
            "response": "Water boils at 100 degrees Celsius at sea level.",
            "context": "Water's boiling point is 100C at standard atmospheric pressure.",
            "ground_truth": "100 degrees Celsius",
        },
    ]


# --- to_openeval -------------------------------------------------------


class TestToOpenEval:
    def test_converts_list_of_dicts(self):
        suite = to_openeval(_dataset(), suite_id="geo_science_eval")
        assert validate_suite(suite).valid

    def test_accepts_pandas_dataframe(self):
        df = pd.DataFrame(_dataset())
        suite = to_openeval(df, suite_id="geo_science_eval")
        assert validate_suite(suite).valid
        assert len(suite["test_cases"]) == 2

    def test_maps_question_context_ground_truth_onto_test_case_fields(self):
        suite = to_openeval(_dataset())
        tc = suite["test_cases"][0]
        assert tc["input"] == "What is the capital of France?"
        assert tc["expected_output"] == "Paris"
        assert tc["context"] == ["France is a country in Europe. Its capital is Paris."]

    def test_preserves_full_raw_row_including_response_under_metadata(self):
        suite = to_openeval(_dataset())
        tc = suite["test_cases"][0]
        row = tc["metadata"]["uptrain"]["row"]
        assert row["response"] == "Paris is the capital of France."
        assert row["question"] == "What is the capital of France?"

    def test_uses_row_id_as_test_case_id(self):
        suite = to_openeval(_dataset())
        assert [tc["id"] for tc in suite["test_cases"]] == ["0", "1"]

    def test_falls_back_to_positional_id_when_row_has_no_id(self):
        rows = [{"question": "2+2?", "ground_truth": "4"}]
        suite = to_openeval(rows)
        assert suite["test_cases"][0]["id"] == "row_0"

    def test_row_without_ground_truth_or_context_has_neither_field(self):
        rows = [{"id": "0", "question": "2+2?", "response": "4"}]
        suite = to_openeval(rows)
        tc = suite["test_cases"][0]
        assert "expected_output" not in tc
        assert "context" not in tc

    def test_default_grader_is_llm_judge_with_required_params(self):
        suite = to_openeval(_dataset())
        assert suite["graders"][0]["type"] == "llm_judge"
        assert "model" in suite["graders"][0]["params"]
        assert "prompt" in suite["graders"][0]["params"]

    def test_exact_match_grader_type_option(self):
        suite = to_openeval(_dataset(), grader_type="exact_match")
        assert suite["graders"][0]["type"] == "exact_match"
        assert validate_suite(suite).valid


# --- from_openeval -------------------------------------------------------


class TestFromOpenEval:
    def test_round_trip_preserves_original_row_including_response(self):
        suite = to_openeval(_dataset(), suite_id="geo_science_eval")
        rows = from_openeval(suite)
        assert rows[0]["response"] == "Paris is the capital of France."
        assert rows[0]["question"] == "What is the capital of France?"
        assert rows[0]["ground_truth"] == "Paris"
        # Round-tripped rows must be directly usable as EvalLLM.evaluate() input.
        assert set(rows[0].keys()) >= {"question", "response", "context", "ground_truth"}

    def test_builds_fresh_row_when_no_uptrain_metadata_present(self):
        suite = {
            "version": "1.0.0",
            "id": "manual_suite",
            "test_cases": [
                {
                    "id": "tc1",
                    "input": "What's 2+2?",
                    "expected_output": "4",
                    "context": ["basic arithmetic"],
                    "graders": ["g1"],
                }
            ],
        }
        rows = from_openeval(suite)
        assert rows == [
            {
                "id": "tc1",
                "question": "What's 2+2?",
                "ground_truth": "4",
                "context": ["basic arithmetic"],
            }
        ]


# --- results_to_openeval -----------------------------------------------


class TestResultsToOpenEval:
    def _evaluated_rows(self):
        # This is the exact shape EvalLLM.evaluate() returns: original row
        # fields plus score_<check>/explanation_<check> per check.
        return [
            {
                "id": "0",
                "question": "What is the capital of France?",
                "response": "Paris is the capital of France.",
                "context": "France's capital is Paris.",
                "ground_truth": "Paris",
                "score_context_relevance": 1.0,
                "explanation_context_relevance": "The context directly answers the question.",
                "score_response_relevance": 0.95,
                "explanation_response_relevance": "The response directly answers the question.",
            },
            {
                "id": "1",
                "question": "What is the boiling point of water?",
                "response": "I don't know.",
                "context": "Water boils at 100C at sea level.",
                "ground_truth": "100 degrees Celsius",
                "score_context_relevance": 1.0,
                "explanation_context_relevance": "Context is relevant.",
                "score_response_relevance": 0.1,
                "explanation_response_relevance": "The response does not answer the question.",
            },
        ]

    def test_converts_real_evaluate_output_shape(self):
        rs = results_to_openeval(
            self._evaluated_rows(), suite_id="geo_science_eval", run_id="run1",
            started_at="2026-08-14T00:00:00Z",
        )
        assert validate_result_set(rs).valid
        assert len(rs["results"]) == 2
        assert len(rs["results"][0]["grader_results"]) == 2

    def test_high_scores_pass_low_scores_fail(self):
        rs = results_to_openeval(self._evaluated_rows(), started_at="2026-08-14T00:00:00Z")
        # Row 0: both checks score high -> passes overall.
        assert rs["results"][0]["passed"] is True
        # Row 1: response_relevance scores 0.1 -> fails overall.
        assert rs["results"][1]["passed"] is False

    def test_preserves_explanation_as_reason_and_metadata(self):
        rs = results_to_openeval(self._evaluated_rows(), started_at="2026-08-14T00:00:00Z")
        gr = next(
            g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "uptrain_context_relevance"
        )
        assert gr["reason"] == "The context directly answers the question."
        assert gr["metadata"]["uptrain_check"] == "context_relevance"

    def test_actual_output_comes_from_response_field(self):
        rs = results_to_openeval(self._evaluated_rows(), started_at="2026-08-14T00:00:00Z")
        assert rs["results"][0]["actual_output"] == "Paris is the capital of France."

    def test_score_confidence_keys_are_not_treated_as_separate_graders(self):
        rows = [
            {
                "id": "0",
                "question": "q",
                "response": "r",
                "score_valid_response": 1.0,
                "score_confidence_valid_response": 0.8,
            }
        ]
        rs = results_to_openeval(rows, started_at="2026-08-14T00:00:00Z")
        assert len(rs["results"][0]["grader_results"]) == 1
        gr = rs["results"][0]["grader_results"][0]
        assert gr["grader_id"] == "uptrain_valid_response"
        assert gr["metadata"]["confidence"] == 0.8

    def test_none_score_treated_as_unscored_not_a_crash(self):
        rows = [{"id": "0", "question": "q", "response": "r", "score_valid_response": None}]
        rs = results_to_openeval(rows, started_at="2026-08-14T00:00:00Z")
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] is None
        assert gr["passed"] is False
        assert validate_result_set(rs).valid

    def test_row_with_no_score_keys_produces_no_grader_results_and_fails(self):
        rows = [{"id": "0", "question": "q", "response": "r"}]
        rs = results_to_openeval(rows, started_at="2026-08-14T00:00:00Z")
        assert rs["results"][0]["grader_results"] == []
        assert rs["results"][0]["passed"] is False
        assert validate_result_set(rs).valid

    def test_full_pipeline_suite_to_result_set(self):
        suite = to_openeval(_dataset(), suite_id="geo_science_eval")
        assert validate_suite(suite).valid

        # Simulate what EvalLLM.evaluate() would have produced for the rows
        # from_openeval() reconstructs from this exact suite.
        rows = from_openeval(suite)
        evaluated = []
        for row, score in zip(rows, [1.0, 0.4]):
            row = dict(row)
            row["score_response_relevance"] = score
            row["explanation_response_relevance"] = "simulated"
            evaluated.append(row)

        result_set = results_to_openeval(
            evaluated, suite_id=suite["id"], run_id="full_run",
            started_at="2026-08-14T00:00:00Z", completed_at="2026-08-14T00:00:05Z",
        )
        assert validate_result_set(result_set).valid
        assert [r["passed"] for r in result_set["results"]] == [True, False]
