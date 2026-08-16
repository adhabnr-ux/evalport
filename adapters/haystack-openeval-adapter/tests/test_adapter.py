"""Tests for haystack_openeval_adapter.

Every test runs against the real ``haystack-ai`` package (real evaluator
components, real ``EvaluationRunResult``) and the real ``evalport-sdk``
validators (``openeval.validate.validate_suite`` /
``validate_result_set``) -- nothing here is mocked.
"""
from __future__ import annotations

import pytest
from haystack.components.evaluators import AnswerExactMatchEvaluator, DocumentMRREvaluator
from haystack.dataclasses import Document
from haystack.evaluation import EvaluationRunResult
from openeval.validate import validate_result_set, validate_suite

from haystack_openeval_adapter import (
    evaluation_result_to_openeval,
    from_openeval,
    to_openeval,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def inputs():
    return {
        "questions": [
            "What is the capital of France?",
            "What is the capital of Japan?",
            "What is the capital of Germany?",
        ],
        "ground_truth_answers": ["Paris", "Tokyo", "Berlin"],
        "predicted_answers": ["Paris", "Tokyo", "Munich"],
    }


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpenEval:
    def test_basic_shape_and_validates(self, inputs):
        suite = to_openeval(
            inputs,
            input_keys=["questions"],
            expected_key="ground_truth_answers",
            graders=["answer_exact_match"],
        )
        result = validate_suite(suite)
        assert result.valid, result.errors
        assert len(suite["test_cases"]) == 3
        assert suite["id"] == "haystack_suite"

    def test_default_suite_id_and_placeholder_grader(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions"])
        assert suite["graders"] == [
            {
                "id": "haystack_metric",
                "type": "custom",
                "params": {"handler": "haystack_metric"},
                "description": (
                    "Placeholder for the Haystack 'haystack_metric' evaluator "
                    "(e.g. a *Evaluator component's .run(), or any custom "
                    "metric function) -- the caller must run the actual "
                    "evaluator and supply its score via "
                    "evaluation_result_to_openeval(), rather than this module "
                    "fabricating a fake implementation."
                ),
            }
        ]
        assert validate_suite(suite).valid

    def test_custom_suite_id_and_description(self, inputs):
        suite = to_openeval(
            inputs, input_keys=["questions"], suite_id="my_rag_suite", description="RAG QA eval"
        )
        assert suite["id"] == "my_rag_suite"
        assert suite["description"] == "RAG QA eval"

    def test_custom_ids(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions"], ids=["fr", "jp", "de"])
        assert [tc["id"] for tc in suite["test_cases"]] == ["fr", "jp", "de"]
        assert validate_suite(suite).valid

    def test_multiple_input_keys_flattened(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions", "predicted_answers"])
        tc0 = suite["test_cases"][0]
        assert tc0["input"] == [
            "questions: What is the capital of France?",
            "predicted_answers: Paris",
        ]
        assert validate_suite(suite).valid

    def test_expected_key_becomes_expected_output(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions"], expected_key="ground_truth_answers")
        assert suite["test_cases"][0]["expected_output"] == "Paris"
        assert suite["test_cases"][2]["expected_output"] == "Berlin"

    def test_missing_expected_key_omits_field(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions"], expected_key="nonexistent_column")
        assert "expected_output" not in suite["test_cases"][0]

    def test_metadata_preserves_full_row(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions"])
        row_meta = suite["test_cases"][1]["metadata"]["haystack"]["columns"]
        assert row_meta == {
            "questions": "What is the capital of Japan?",
            "ground_truth_answers": "Tokyo",
            "predicted_answers": "Tokyo",
        }

    def test_graders_list_infers_exact_match_type(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions"], graders=["answer_exact_match"])
        grader = suite["graders"][0]
        assert grader["id"] == "answer_exact_match"
        assert grader["type"] == "exact_match"
        assert "params" not in grader
        for tc in suite["test_cases"]:
            assert tc["graders"] == ["answer_exact_match"]
        assert validate_suite(suite).valid

    def test_graders_list_falls_back_to_custom_for_unknown_metric(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions"], graders=["faithfulness"])
        grader = suite["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"] == {"handler": "faithfulness"}
        assert validate_suite(suite).valid

    def test_multiple_graders(self, inputs):
        suite = to_openeval(
            inputs, input_keys=["questions"], graders=["answer_exact_match", "faithfulness"]
        )
        grader_ids = [g["id"] for g in suite["graders"]]
        assert grader_ids == ["answer_exact_match", "faithfulness"]
        for tc in suite["test_cases"]:
            assert tc["graders"] == ["answer_exact_match", "faithfulness"]
        assert validate_suite(suite).valid

    def test_empty_inputs_raises(self):
        with pytest.raises(ValueError, match="inputs is empty"):
            to_openeval({}, input_keys=["x"])

    def test_empty_input_keys_raises(self, inputs):
        with pytest.raises(ValueError, match="input_keys is empty"):
            to_openeval(inputs, input_keys=[])

    def test_missing_input_key_raises(self, inputs):
        with pytest.raises(ValueError, match="not present in inputs"):
            to_openeval(inputs, input_keys=["nonexistent_column"])

    def test_mismatched_column_lengths_raises(self, inputs):
        bad_inputs = dict(inputs)
        bad_inputs["questions"] = inputs["questions"][:2]
        with pytest.raises(ValueError, match="same length"):
            to_openeval(bad_inputs, input_keys=["questions"])

    def test_zero_row_columns_raises(self):
        with pytest.raises(ValueError, match="empty -- nothing to convert"):
            to_openeval({"questions": []}, input_keys=["questions"])

    def test_mismatched_ids_length_raises(self, inputs):
        with pytest.raises(ValueError, match="ids has length"):
            to_openeval(inputs, input_keys=["questions"], ids=["only_one"])


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


class TestFromOpenEval:
    def test_lossless_round_trip(self, inputs):
        suite = to_openeval(
            inputs,
            input_keys=["questions"],
            expected_key="ground_truth_answers",
            graders=["answer_exact_match"],
        )
        cols = from_openeval(suite)
        # Every original column survives exactly; the adapter additionally
        # surfaces an "id" column (see the module docstring) which the
        # original `inputs` fixture never had.
        assert {k: v for k, v in cols.items() if k != "id"} == inputs
        assert cols["id"] == ["haystack_tc_0", "haystack_tc_1", "haystack_tc_2"]

    def test_lossless_round_trip_preserves_row_order(self, inputs):
        suite = to_openeval(inputs, input_keys=["questions"], ids=["a", "b", "c"])
        cols = from_openeval(suite)
        assert cols["questions"] == inputs["questions"]
        assert cols["id"] == ["a", "b", "c"]

    def test_id_column_not_overwritten_when_already_real_data(self):
        inputs_with_real_id = {"questions": ["q1", "q2"], "id": ["real-id-1", "real-id-2"]}
        suite = to_openeval(inputs_with_real_id, input_keys=["questions"], ids=["tc_a", "tc_b"])
        cols = from_openeval(suite)
        # The genuine "id" column data survives untouched -- the test case
        # id ("tc_a"/"tc_b") is NOT injected over it.
        assert cols["id"] == ["real-id-1", "real-id-2"]

    def test_foreign_suite_positional_naming(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [
                {"id": "tc1", "input": ["hello", "world"], "graders": ["g1"], "expected_output": "hi"},
                {"id": "tc2", "input": ["foo", "bar"], "graders": ["g1"], "expected_output": "baz"},
            ],
        }
        cols = from_openeval(suite)
        assert cols == {
            "input_1": ["hello", "foo"],
            "input_2": ["world", "bar"],
            "expected_output": ["hi", "baz"],
            "id": ["tc1", "tc2"],
        }

    def test_foreign_suite_explicit_input_keys(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [
                {"id": "tc1", "input": ["hello"], "graders": ["g1"]},
                {"id": "tc2", "input": ["foo"], "graders": ["g1"]},
            ],
        }
        cols = from_openeval(suite, input_keys=["question"])
        assert cols == {"question": ["hello", "foo"], "id": ["tc1", "tc2"]}

    def test_foreign_suite_string_input(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [{"id": "tc1", "input": "just a string", "graders": ["g1"]}],
        }
        cols = from_openeval(suite)
        assert cols == {"input_1": ["just a string"], "id": ["tc1"]}

    def test_mismatched_input_keys_length_raises(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [{"id": "tc1", "input": ["a", "b"], "graders": ["g1"]}],
        }
        with pytest.raises(ValueError, match="input entries but input_keys has"):
            from_openeval(suite, input_keys=["only_one_name"])

    def test_empty_test_cases_raises(self):
        with pytest.raises(ValueError, match="no test_cases"):
            from_openeval({"version": "1.0.0", "id": "s", "graders": [], "test_cases": []})

    def test_inconsistent_columns_raises(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [
                {
                    "id": "tc1",
                    "input": ["x"],
                    "graders": ["g1"],
                    "metadata": {"haystack": {"columns": {"a": 1, "b": 2}}},
                },
                {
                    "id": "tc2",
                    "input": ["y"],
                    "graders": ["g1"],
                    "metadata": {"haystack": {"columns": {"a": 3, "c": 4}}},
                },
            ],
        }
        with pytest.raises(ValueError, match="inconsistent column names"):
            from_openeval(suite)


# ---------------------------------------------------------------------------
# evaluation_result_to_openeval
# ---------------------------------------------------------------------------


class TestEvaluationResultToOpenEval:
    def test_real_answer_exact_match_evaluator_round_trip(self, inputs):
        suite = to_openeval(
            inputs,
            input_keys=["questions"],
            expected_key="ground_truth_answers",
            graders=["answer_exact_match"],
            ids=["fr", "jp", "de"],
        )
        cols = from_openeval(suite)

        eval_output = AnswerExactMatchEvaluator().run(
            ground_truth_answers=cols["ground_truth_answers"],
            predicted_answers=cols["predicted_answers"],
        )
        assert eval_output["individual_scores"] == [1, 1, 0]  # real computation: Munich != Berlin

        run_result = EvaluationRunResult(
            run_name="exact_match_run",
            inputs=cols,
            results={"answer_exact_match": eval_output},
        )
        result_set = evaluation_result_to_openeval(
            run_result, suite_id=suite["id"], output_column="predicted_answers"
        )

        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        assert [r["test_case_id"] for r in result_set["results"]] == ["fr", "jp", "de"]
        assert [r["passed"] for r in result_set["results"]] == [True, True, False]
        assert result_set["results"][2]["actual_output"] == "Munich"
        assert result_set["summary"] == {"total": 3, "passed": 2, "failed": 1, "pass_rate": pytest.approx(2 / 3)}

    def test_real_document_mrr_evaluator(self):
        gt_docs = [[Document(content="Paris is the capital of France.")]]
        retrieved_docs = [
            [
                Document(content="Irrelevant document."),
                Document(content="Paris is the capital of France."),
            ]
        ]
        eval_output = DocumentMRREvaluator().run(
            ground_truth_documents=gt_docs, retrieved_documents=retrieved_docs
        )
        assert eval_output["individual_scores"] == [0.5]  # real computation: rank 2 -> 1/2

        run_result = EvaluationRunResult(
            run_name="mrr_run",
            inputs={"questions": ["q1"]},
            results={"document_mrr": eval_output},
        )
        result_set = evaluation_result_to_openeval(run_result, suite_id="retrieval_suite")
        assert validate_result_set(result_set).valid
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["grader_id"] == "document_mrr"
        assert gr["type"] == "custom"
        assert gr["score"] == 0.5
        assert gr["passed"] is True  # 0.5 >= default threshold 0.5

    def test_exact_match_metric_gets_exact_match_grader_type(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["type"] == "exact_match"
        assert validate_result_set(result_set).valid

    def test_multiple_metrics_and_row_level_passed(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {
                "answer_exact_match": {"score": 2 / 3, "individual_scores": [1, 1, 0]},
                "faithfulness": {"score": 0.8, "individual_scores": [0.9, 0.9, 0.9]},
            },
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s", pass_threshold=0.5)
        # row 2 fails answer_exact_match (0) even though faithfulness passes (0.9) -> overall fail
        assert [r["passed"] for r in result_set["results"]] == [True, True, False]
        assert len(result_set["results"][0]["grader_results"]) == 2
        assert validate_result_set(result_set).valid

    def test_score_clamping_preserves_raw_score(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"custom_metric": {"score": 1.5, "individual_scores": [1.5, -0.2, 0.6]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        grs = [r["grader_results"][0] for r in result_set["results"]]
        assert [gr["score"] for gr in grs] == [1.0, 0.0, 0.6]
        assert grs[0]["metadata"]["haystack"]["raw_score"] == 1.5
        assert grs[1]["metadata"]["haystack"]["raw_score"] == -0.2
        assert "raw_score" not in grs[2]["metadata"]["haystack"]
        assert validate_result_set(result_set).valid

    def test_aggregate_score_preserved_in_metadata(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"answer_exact_match": {"score": 0.75, "individual_scores": [1, 1, 0]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        for r in result_set["results"]:
            assert r["grader_results"][0]["metadata"]["haystack"]["aggregate_score"] == 0.75

    def test_id_column_used_when_present(self, inputs):
        inputs_with_id = dict(inputs)
        inputs_with_id["id"] = ["fr", "jp", "de"]
        run_result = {
            "run_name": "r",
            "inputs": inputs_with_id,
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        assert [r["test_case_id"] for r in result_set["results"]] == ["fr", "jp", "de"]

    def test_explicit_id_column_overrides_default(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s", id_column="questions")
        assert result_set["results"][0]["test_case_id"] == "What is the capital of France?"

    def test_default_generated_ids_when_no_id_column(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        assert [r["test_case_id"] for r in result_set["results"]] == [
            "haystack_tc_0",
            "haystack_tc_1",
            "haystack_tc_2",
        ]

    def test_output_column_auto_detection(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        assert result_set["results"][0]["actual_output"] == "Paris"

    def test_no_output_column_omits_actual_output(self):
        run_result = {
            "run_name": "r",
            "inputs": {"questions": ["q1"]},
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        assert "actual_output" not in result_set["results"][0]

    def test_explicit_run_id_and_timestamps(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        }
        result_set = evaluation_result_to_openeval(
            run_result,
            suite_id="s",
            run_id="fixed_run_id",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:05:00Z",
        )
        assert result_set["run_id"] == "fixed_run_id"
        assert result_set["started_at"] == "2026-01-01T00:00:00Z"
        assert result_set["completed_at"] == "2026-01-01T00:05:00Z"

    def test_default_run_id_and_started_at_generated(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        }
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        assert result_set["run_id"].startswith("haystack_run_")
        assert result_set["started_at"]
        assert "completed_at" not in result_set

    def test_empty_inputs_raises(self):
        with pytest.raises(ValueError, match="inputs is empty"):
            evaluation_result_to_openeval({"run_name": "r", "inputs": {}, "results": {}}, suite_id="s")

    def test_no_metrics_raises(self, inputs):
        with pytest.raises(ValueError, match="no metrics to convert"):
            evaluation_result_to_openeval(
                {"run_name": "r", "inputs": inputs, "results": {}}, suite_id="s"
            )

    def test_mismatched_individual_scores_length_raises(self, inputs):
        run_result = {
            "run_name": "r",
            "inputs": inputs,
            "results": {"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1]}},
        }
        with pytest.raises(ValueError, match="individual_scores, expected 3"):
            evaluation_result_to_openeval(run_result, suite_id="s")

    def test_mismatched_input_column_lengths_raises(self):
        run_result = {
            "run_name": "r",
            "inputs": {"a": [1, 2, 3], "b": [1, 2]},
            "results": {"m": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        }
        with pytest.raises(ValueError, match="mismatched lengths"):
            evaluation_result_to_openeval(run_result, suite_id="s")

    def test_accepts_real_evaluation_run_result_object_not_just_dict(self, inputs):
        run_result = EvaluationRunResult(
            run_name="typed_run",
            inputs=inputs,
            results={"answer_exact_match": {"score": 1.0, "individual_scores": [1, 1, 1]}},
        )
        result_set = evaluation_result_to_openeval(run_result, suite_id="s")
        assert validate_result_set(result_set).valid
        assert len(result_set["results"]) == 3


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestFullLoop:
    def test_full_loop_id_preservation_and_validation(self, inputs):
        suite = to_openeval(
            inputs,
            input_keys=["questions"],
            expected_key="ground_truth_answers",
            graders=["answer_exact_match"],
            ids=["fr", "jp", "de"],
            suite_id="capitals_suite",
        )
        assert validate_suite(suite).valid

        cols = from_openeval(suite)
        eval_output = AnswerExactMatchEvaluator().run(
            ground_truth_answers=cols["ground_truth_answers"],
            predicted_answers=cols["predicted_answers"],
        )
        run_result = EvaluationRunResult(
            run_name="capitals_run", inputs=cols, results={"answer_exact_match": eval_output}
        )
        result_set = evaluation_result_to_openeval(
            run_result, suite_id=suite["id"], output_column="predicted_answers"
        )
        assert validate_result_set(result_set).valid

        suite_ids = [tc["id"] for tc in suite["test_cases"]]
        result_ids = [r["test_case_id"] for r in result_set["results"]]
        assert suite_ids == result_ids == ["fr", "jp", "de"]
