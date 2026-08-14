"""Tests for weave_openeval_adapter, run against the real `weave` package and
the real EvalPort validators (openeval.validate) -- no mocks.
"""

from __future__ import annotations

import weave
from openeval.validate import validate_result_set, validate_suite

from weave_openeval_adapter import evaluation_to_openeval, from_openeval, to_openeval


def _dataset():
    return weave.Dataset(
        name="grammar",
        rows=[
            {
                "id": "0",
                "question": "What is the capital of France?",
                "expected": "Paris",
            },
            {
                "id": "1",
                "question": "Who wrote 'To Kill a Mockingbird'?",
                "expected": "Harper Lee",
            },
            {
                "id": "2",
                "question": "What is the square root of 64?",
                "expected": "8",
            },
        ],
    )


# --- to_openeval -------------------------------------------------------


class TestToOpenEval:
    def test_converts_real_weave_dataset(self):
        suite = to_openeval(_dataset(), suite_id="grammar_suite")
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_accepts_plain_row_iterable_not_just_weave_dataset(self):
        rows = [
            {"id": "a", "prompt": "2+2?", "answer": "4"},
            {"id": "b", "prompt": "3+3?", "answer": "6"},
        ]
        suite = to_openeval(rows, suite_id="plain_rows")
        assert validate_suite(suite).valid
        assert len(suite["test_cases"]) == 2

    def test_detects_input_and_expected_output_by_common_names(self):
        suite = to_openeval(_dataset())
        tc = suite["test_cases"][0]
        assert tc["input"] == "What is the capital of France?"
        assert tc["expected_output"] == "Paris"

    def test_preserves_full_raw_row_under_metadata(self):
        suite = to_openeval(_dataset())
        tc = suite["test_cases"][0]
        assert tc["metadata"]["weave"]["row"]["question"] == "What is the capital of France?"
        assert tc["metadata"]["weave"]["row"]["expected"] == "Paris"
        assert tc["metadata"]["weave"]["row"]["id"] == "0"

    def test_uses_row_id_as_test_case_id_when_present(self):
        suite = to_openeval(_dataset())
        ids = [tc["id"] for tc in suite["test_cases"]]
        assert ids == ["0", "1", "2"]

    def test_falls_back_to_positional_id_when_row_has_no_id(self):
        rows = [{"sentence": "He no likes ice cream.", "correction": "He doesn't like ice cream."}]
        suite = to_openeval(rows)
        assert suite["test_cases"][0]["id"] == "row_0"

    def test_input_key_override(self):
        rows = [{"id": "0", "my_prompt": "hello", "my_target": "world"}]
        suite = to_openeval(rows, input_key="my_prompt", expected_output_key="my_target")
        tc = suite["test_cases"][0]
        assert tc["input"] == "hello"
        assert tc["expected_output"] == "world"

    def test_row_with_no_recognizable_keys_falls_back_to_single_remaining_key(self):
        rows = [{"id": "0", "sentence": "She goed to the store."}]
        suite = to_openeval(rows)
        tc = suite["test_cases"][0]
        # "sentence" isn't a known input key and isn't a known output key
        # either; with only one candidate column left it becomes the input.
        assert tc["input"] == "She goed to the store."
        assert "expected_output" not in tc

    def test_row_with_multiple_unrecognized_keys_falls_back_to_json_dump(self):
        rows = [{"id": "0", "foo": "bar", "baz": "qux"}]
        suite = to_openeval(rows)
        tc = suite["test_cases"][0]
        assert tc["input"].startswith("{")
        assert "bar" in tc["input"] and "qux" in tc["input"]

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
    def test_round_trip_through_weave_dataset_preserves_original_row(self):
        suite = to_openeval(_dataset(), suite_id="grammar_suite")
        rows = from_openeval(suite)
        # Reconstructing a real weave.Dataset from the round-tripped rows
        # must not raise -- this is the actual downstream consumer.
        ds = weave.Dataset(name="round_tripped", rows=rows)
        assert len(ds) == 3
        assert rows[0]["question"] == "What is the capital of France?"
        assert rows[0]["expected"] == "Paris"
        assert rows[0]["id"] == "0"

    def test_builds_fresh_rows_when_no_weave_metadata_present(self):
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
        assert rows == [{"id": "tc1", "input": "What's 2+2?", "expected": "4"}]
        ds = weave.Dataset(name="from_manual", rows=rows)
        assert len(ds) == 1

    def test_handles_non_string_input_by_json_encoding(self):
        suite = {
            "version": "1.0.0",
            "id": "s",
            "test_cases": [
                {"id": "tc1", "input": ["turn one", "turn two"], "graders": ["g1"]}
            ],
        }
        rows = from_openeval(suite)
        assert "turn one" in rows[0]["input"]


# --- evaluation_to_openeval -----------------------------------------------


class TestEvaluationToOpenEval:
    def _rows(self):
        return [
            {"id": "0", "question": "capital of France?", "expected": "Paris"},
            {"id": "1", "question": "2+2?", "expected": "4"},
            {"id": "2", "question": "3+3?", "expected": "6"},
        ]

    def test_converts_bool_scorer_results(self):
        rows = self._rows()
        eval_results = [
            {"output": "Paris", "scores": {"exact_match": True}, "model_latency": 0.42},
            {"output": "4", "scores": {"exact_match": True}, "model_latency": 0.11},
            {"output": "5", "scores": {"exact_match": False}, "model_latency": 0.13},
        ]
        rs = evaluation_to_openeval(
            rows, eval_results, suite_id="grammar_suite", run_id="run1",
            started_at="2026-08-14T00:00:00Z",
        )
        assert validate_result_set(rs).valid
        assert rs["results"][0]["passed"] is True
        assert rs["results"][2]["passed"] is False
        assert rs["results"][2]["grader_results"][0]["score"] == 0.0

    def test_converts_numeric_scorer_results_and_clamps_to_unit_range(self):
        rows = self._rows()[:1]
        eval_results = [{"output": "Paris", "scores": {"similarity": 1.4}, "model_latency": 0.2}]
        rs = evaluation_to_openeval(rows, eval_results, started_at="2026-08-14T00:00:00Z")
        assert validate_result_set(rs).valid
        assert rs["results"][0]["grader_results"][0]["score"] == 1.0

    def test_converts_dict_scorer_results_with_passed_key(self):
        rows = self._rows()[:1]
        eval_results = [
            {
                "output": "Paris",
                "scores": {"correctness": {"passed": True, "reasoning": "matches exactly"}},
                "model_latency": 0.2,
            }
        ]
        rs = evaluation_to_openeval(rows, eval_results, started_at="2026-08-14T00:00:00Z")
        assert validate_result_set(rs).valid
        gr = rs["results"][0]["grader_results"][0]
        assert gr["passed"] is True
        assert gr["metadata"]["raw"]["reasoning"] == "matches exactly"

    def test_converts_dict_scorer_results_with_score_key(self):
        rows = self._rows()[:1]
        eval_results = [
            {"output": "Paris", "scores": {"judge": {"score": 0.9, "explanation": "close"}}, "model_latency": 0.2}
        ]
        rs = evaluation_to_openeval(rows, eval_results, started_at="2026-08-14T00:00:00Z")
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] == 0.9
        assert gr["passed"] is True

    def test_handles_none_scorer_result_without_crashing(self):
        rows = self._rows()[:1]
        eval_results = [{"output": None, "scores": {"exact_match": None}, "model_latency": 0.0}]
        rs = evaluation_to_openeval(rows, eval_results, started_at="2026-08-14T00:00:00Z")
        assert validate_result_set(rs).valid
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] is None
        assert gr["passed"] is False

    def test_multiple_scorers_per_row_all_must_pass_for_overall_passed(self):
        rows = self._rows()[:1]
        eval_results = [
            {
                "output": "Paris",
                "scores": {"exact_match": True, "toxicity_free": False},
                "model_latency": 0.2,
            }
        ]
        rs = evaluation_to_openeval(rows, eval_results, started_at="2026-08-14T00:00:00Z")
        assert len(rs["results"][0]["grader_results"]) == 2
        assert rs["results"][0]["passed"] is False

    def test_duration_ms_derived_from_model_latency_seconds(self):
        rows = self._rows()[:1]
        eval_results = [{"output": "Paris", "scores": {"exact_match": True}, "model_latency": 0.42}]
        rs = evaluation_to_openeval(rows, eval_results, started_at="2026-08-14T00:00:00Z")
        assert rs["results"][0]["duration_ms"] == 420

    def test_mismatched_lengths_raises(self):
        rows = self._rows()
        eval_results = [{"output": "x", "scores": {}, "model_latency": 0.0}]
        try:
            evaluation_to_openeval(rows, eval_results, started_at="2026-08-14T00:00:00Z")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_full_pipeline_dataset_to_result_set_round_trip(self):
        dataset = _dataset()
        suite = to_openeval(dataset, suite_id="grammar_suite", grader_type="exact_match")
        assert validate_suite(suite).valid

        # Simulate what Evaluation.evaluate() would have produced for this
        # dataset, using the real predict_and_score output shape.
        eval_results = [
            {"output": "Paris", "scores": {"default_exact_match": True}, "model_latency": 0.1},
            {"output": "Harper Lee", "scores": {"default_exact_match": True}, "model_latency": 0.12},
            {"output": "9", "scores": {"default_exact_match": False}, "model_latency": 0.09},
        ]
        result_set = evaluation_to_openeval(
            list(dataset), eval_results, suite_id=suite["id"], run_id="full_run",
            started_at="2026-08-14T00:00:00Z", completed_at="2026-08-14T00:00:05Z",
        )
        assert validate_result_set(result_set).valid
        assert result_set["suite_id"] == "grammar_suite"
        assert [r["passed"] for r in result_set["results"]] == [True, True, False]
