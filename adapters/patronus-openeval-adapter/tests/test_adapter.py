"""Tests for patronus_openeval_adapter.

These exercise to_openeval()/from_openeval()/batch_eval_result_to_openeval()
against the real `patronus` package (patronus.evals.Evaluator,
patronus.evals.RemoteEvaluator, patronus.evals.EvaluationResult) -- not
mocks of this adapter's own dependencies -- and against the real
openeval.validate.validate_suite()/validate_result_set(), the same real
EvalPort validator every other adapter in this ecosystem tests against.

RemoteEvaluator is instantiated (its constructor makes no network call --
verified by reading patronus/evals/evaluators.py directly) but never
.evaluate()'d, since that requires a live Patronus API key. The local,
deterministic ExactMatchEvaluator subclass below is a real Evaluator
subclass that *is* actually run end to end, the same "use the target
library's own local/offline-capable evaluator machinery" pattern the
LlamaIndex adapter uses with MockLLM/MockEmbedding and the Giskard adapter
uses with its own local checks.
"""

import pytest
from openeval.validate import validate_result_set, validate_suite
from patronus.evals import Evaluator, EvaluationResult, RemoteEvaluator

from patronus_openeval_adapter import (
    batch_eval_result_to_openeval,
    from_openeval,
    to_openeval,
)


class ExactMatchEvaluator(Evaluator):
    """A real, local, deterministic Evaluator subclass -- no network, no
    LLM -- used to exercise the full adapter pipeline end to end."""

    evaluator_id = "exact_match"

    def evaluate(self, *, task_output=None, gold_answer=None, **kwargs) -> EvaluationResult:
        ok = (task_output or "").strip() == (gold_answer or "").strip()
        return EvaluationResult(
            score=1.0 if ok else 0.0,
            pass_=ok,
            text_output=task_output,
            explanation="exact string match" if ok else "strings differ",
        )


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpeneval:
    def test_basic_shape_with_local_evaluator_validates_against_real_spec(self):
        suite = to_openeval(
            inputs=["What is the capital of France?", "What is 2+2?"],
            evaluators={"exact_match": ExactMatchEvaluator()},
            expected_outputs=["Paris", "4"],
            suite_id="geo_and_math",
        )
        assert suite["id"] == "geo_and_math"
        assert len(suite["test_cases"]) == 2
        assert suite["test_cases"][0]["id"] == "patronus_tc_0"
        assert suite["test_cases"][0]["expected_output"] == "Paris"
        assert suite["test_cases"][0]["graders"] == ["exact_match"]

        grader = suite["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"]["handler"]

        validation = validate_suite(suite)
        assert validation.valid, validation.errors

    def test_remote_evaluator_maps_to_llm_judge_with_valid_prompt_tokens(self):
        suite = to_openeval(
            inputs=["Tell me about Paris."],
            evaluators={"judge": RemoteEvaluator("judge", criteria="conciseness")},
            suite_id="s1",
        )
        grader = suite["graders"][0]
        assert grader["type"] == "llm_judge"
        assert grader["params"]["model"] == "patronus-hosted-judge"
        prompt = grader["params"]["prompt"]
        assert "{input}" in prompt
        assert "{output}" in prompt
        assert "{expected}" in prompt
        assert grader["metadata"]["patronus"]["evaluator_id_or_alias"] == "judge"
        assert grader["metadata"]["patronus"]["criteria"] == "conciseness"

        validation = validate_suite(suite)
        assert validation.valid, validation.errors

    def test_multiple_evaluators_each_test_case_gets_all_graders(self):
        suite = to_openeval(
            inputs=["hi"],
            evaluators={
                "exact_match": ExactMatchEvaluator(),
                "judge": RemoteEvaluator("judge"),
            },
        )
        assert set(suite["test_cases"][0]["graders"]) == {"exact_match", "judge"}
        assert len(suite["graders"]) == 2

    def test_contexts_list_is_carried_onto_context_field(self):
        suite = to_openeval(
            inputs=["What is the capital of France?"],
            evaluators={"exact_match": ExactMatchEvaluator()},
            contexts_list=[["Paris is the capital of France."]],
        )
        assert suite["test_cases"][0]["context"] == ["Paris is the capital of France."]

    def test_explicit_ids_are_respected(self):
        suite = to_openeval(
            inputs=["a", "b"],
            evaluators={"exact_match": ExactMatchEvaluator()},
            ids=["tc_a", "tc_b"],
        )
        assert [tc["id"] for tc in suite["test_cases"]] == ["tc_a", "tc_b"]

    def test_description_is_optional_and_carried_through(self):
        suite = to_openeval(
            inputs=["a"],
            evaluators={"exact_match": ExactMatchEvaluator()},
            description="A tiny suite.",
        )
        assert suite["description"] == "A tiny suite."

    def test_empty_inputs_raises(self):
        with pytest.raises(ValueError, match="inputs is empty"):
            to_openeval(inputs=[], evaluators={"exact_match": ExactMatchEvaluator()})

    def test_empty_evaluators_raises(self):
        with pytest.raises(ValueError, match="evaluators is empty"):
            to_openeval(inputs=["a"], evaluators={})

    def test_mismatched_expected_outputs_length_raises(self):
        with pytest.raises(ValueError, match="expected_outputs"):
            to_openeval(
                inputs=["a", "b"],
                evaluators={"exact_match": ExactMatchEvaluator()},
                expected_outputs=["only_one"],
            )

    def test_mismatched_contexts_list_length_raises(self):
        with pytest.raises(ValueError, match="contexts_list"):
            to_openeval(
                inputs=["a", "b"],
                evaluators={"exact_match": ExactMatchEvaluator()},
                contexts_list=[["only_one"]],
            )

    def test_mismatched_ids_length_raises(self):
        with pytest.raises(ValueError, match="ids"):
            to_openeval(
                inputs=["a", "b"],
                evaluators={"exact_match": ExactMatchEvaluator()},
                ids=["only_one"],
            )


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


class TestFromOpeneval:
    def test_reconstructs_remote_evaluator_from_llm_judge_grader(self):
        suite = to_openeval(
            inputs=["hi"],
            evaluators={"judge": RemoteEvaluator("lynx", criteria="hallucination")},
        )
        rebuilt = from_openeval(suite)
        assert "judge" in rebuilt["evaluators"]
        reconstructed = rebuilt["evaluators"]["judge"]
        assert isinstance(reconstructed, RemoteEvaluator)
        assert reconstructed.evaluator_id_or_alias == "lynx"
        assert reconstructed.criteria == "hallucination"

    def test_does_not_reconstruct_custom_grader(self):
        suite = to_openeval(
            inputs=["hi"], evaluators={"exact_match": ExactMatchEvaluator()}
        )
        rebuilt = from_openeval(suite)
        # A real bug this test suite is written to catch: a "custom"
        # grader's evaluator class name is still visible in metadata, but
        # there's no safe generic way to instantiate arbitrary evaluator
        # code from that -- it must NOT show up reconstructed.
        assert "exact_match" not in rebuilt["evaluators"]

    def test_hand_authored_llm_judge_grader_with_no_patronus_metadata_is_skipped(self):
        suite = {
            "version": "1.0.0",
            "id": "s1",
            "graders": [{"id": "g1", "type": "llm_judge", "params": {"prompt": "{output} {input} {expected}", "model": "gpt-4o"}}],
            "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}],
        }
        rebuilt = from_openeval(suite)
        assert "g1" not in rebuilt["evaluators"]

    def test_returns_inputs_expected_outputs_contexts_and_ids_in_order(self):
        suite = to_openeval(
            inputs=["q1", "q2"],
            evaluators={"exact_match": ExactMatchEvaluator()},
            expected_outputs=["a1", None],
            contexts_list=[["ctx1"], None],
            ids=["tc1", "tc2"],
        )
        rebuilt = from_openeval(suite)
        assert rebuilt["inputs"] == ["q1", "q2"]
        assert rebuilt["expected_outputs"] == ["a1", None]
        assert rebuilt["contexts_list"] == [["ctx1"], None]
        assert rebuilt["ids"] == ["tc1", "tc2"]

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
        evaluators = {"exact_match": ExactMatchEvaluator()}
        eval_results = {
            "exact_match": [
                EvaluationResult(score=1.0, pass_=True, text_output="Paris"),
                EvaluationResult(score=0.0, pass_=False, text_output="5"),
            ]
        }
        result_set = batch_eval_result_to_openeval(
            eval_results,
            test_case_ids=["tc1", "tc2"],
            evaluators=evaluators,
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

    def test_remote_evaluator_produces_llm_judge_grader_type(self):
        evaluators = {"judge": RemoteEvaluator("judge")}
        eval_results = {"judge": [EvaluationResult(score=0.9, pass_=True)]}
        result_set = batch_eval_result_to_openeval(
            eval_results, test_case_ids=["tc1"], evaluators=evaluators
        )
        assert result_set["results"][0]["grader_results"][0]["type"] == "llm_judge"

    def test_score_only_result_derives_passed(self):
        eval_results = {"m": [EvaluationResult(score=0.9), EvaluationResult(score=0.1)]}
        result_set = batch_eval_result_to_openeval(
            eval_results, test_case_ids=["tc1", "tc2"], evaluators={"m": ExactMatchEvaluator()}
        )
        assert result_set["results"][0]["grader_results"][0]["passed"] is True
        assert result_set["results"][1]["grader_results"][0]["passed"] is False

    def test_pass_only_result_derives_score(self):
        eval_results = {"m": [EvaluationResult(pass_=True), EvaluationResult(pass_=False)]}
        result_set = batch_eval_result_to_openeval(
            eval_results, test_case_ids=["tc1", "tc2"], evaluators={"m": ExactMatchEvaluator()}
        )
        assert result_set["results"][0]["grader_results"][0]["score"] == 1.0
        assert result_set["results"][1]["grader_results"][0]["score"] == 0.0

    def test_neither_score_nor_pass_is_honestly_not_passed(self):
        eval_results = {"m": [EvaluationResult()]}
        result_set = batch_eval_result_to_openeval(
            eval_results, test_case_ids=["tc1"], evaluators={"m": ExactMatchEvaluator()}
        )
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["score"] is None
        assert gr["passed"] is False

    def test_score_clamped_to_valid_range(self):
        eval_results = {"m": [EvaluationResult(score=1.5)]}
        result_set = batch_eval_result_to_openeval(
            eval_results, test_case_ids=["tc1"], evaluators={"m": ExactMatchEvaluator()}
        )
        assert result_set["results"][0]["grader_results"][0]["score"] == 1.0
        assert validate_result_set(result_set).valid

    def test_none_result_produces_no_grader_result_for_that_case(self):
        eval_results = {"m": [None, EvaluationResult(score=1.0, pass_=True)]}
        result_set = batch_eval_result_to_openeval(
            eval_results, test_case_ids=["tc1", "tc2"], evaluators={"m": ExactMatchEvaluator()}
        )
        assert result_set["results"][0]["grader_results"] == []
        assert result_set["results"][0]["passed"] is False
        assert result_set["results"][1]["passed"] is True

    def test_patronus_specific_fields_preserved_in_metadata(self):
        eval_results = {
            "m": [
                EvaluationResult(
                    score=1.0,
                    pass_=True,
                    explanation="matched exactly",
                    tags={"env": "test"},
                    dataset_id="ds1",
                    dataset_sample_id="s1",
                )
            ]
        }
        result_set = batch_eval_result_to_openeval(
            eval_results, test_case_ids=["tc1"], evaluators={"m": ExactMatchEvaluator()}
        )
        meta = result_set["results"][0]["grader_results"][0]["metadata"]["patronus"]
        assert meta["explanation"] == "matched exactly"
        assert meta["tags"] == {"env": "test"}
        assert meta["dataset_id"] == "ds1"
        assert meta["dataset_sample_id"] == "s1"

    def test_run_id_auto_generated_when_omitted(self):
        eval_results = {"m": [EvaluationResult(score=1.0, pass_=True)]}
        result_set = batch_eval_result_to_openeval(
            eval_results, test_case_ids=["tc1"], evaluators={"m": ExactMatchEvaluator()}
        )
        assert result_set["run_id"].startswith("patronus_run_")

    def test_empty_eval_results_raises(self):
        with pytest.raises(ValueError, match="eval_results is empty"):
            batch_eval_result_to_openeval({}, test_case_ids=["tc1"], evaluators={})

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            batch_eval_result_to_openeval(
                {"m": [EvaluationResult(score=1.0)]},
                test_case_ids=["tc1", "tc2"],
                evaluators={"m": ExactMatchEvaluator()},
            )

    def test_summary_matches_actual_pass_fail_counts(self):
        eval_results = {
            "m": [
                EvaluationResult(score=1.0, pass_=True),
                EvaluationResult(score=0.0, pass_=False),
                EvaluationResult(score=1.0, pass_=True),
            ]
        }
        result_set = batch_eval_result_to_openeval(
            eval_results,
            test_case_ids=["tc1", "tc2", "tc3"],
            evaluators={"m": ExactMatchEvaluator()},
        )
        assert result_set["summary"]["total"] == 3
        assert result_set["summary"]["passed"] == 2
        assert result_set["summary"]["failed"] == 1
        assert result_set["summary"]["pass_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# End-to-end: suite -> real local evaluator run -> results, both validated
# against the real spec
# ---------------------------------------------------------------------------


class TestEndToEndWithRealEvaluator:
    def test_full_round_trip_suite_to_run_to_resultset(self):
        evaluator_obj = ExactMatchEvaluator()
        suite = to_openeval(
            inputs=["What is the capital of France?", "What is 2+2?"],
            evaluators={"exact_match": evaluator_obj},
            expected_outputs=["Paris", "4"],
            ids=["geo1", "math1"],
            suite_id="e2e_suite",
        )
        assert validate_suite(suite).valid

        rebuilt = from_openeval(suite)
        assert rebuilt["inputs"] == ["What is the capital of France?", "What is 2+2?"]

        # Simulate the system under test, then actually run the real,
        # local ExactMatchEvaluator (no network, no mocking) against it.
        my_app_outputs = ["Paris", "5"]  # second answer is wrong on purpose
        results = [
            evaluator_obj.evaluate(
                task_input=rebuilt["inputs"][i],
                task_output=my_app_outputs[i],
                gold_answer=rebuilt["expected_outputs"][i],
            )
            for i in range(len(rebuilt["inputs"]))
        ]

        result_set = batch_eval_result_to_openeval(
            {"exact_match": results},
            test_case_ids=rebuilt["ids"],
            evaluators={"exact_match": evaluator_obj},
            suite_id=suite["id"],
            started_at="2026-08-15T00:00:00Z",
        )
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        assert result_set["summary"]["passed"] == 1
        assert result_set["summary"]["failed"] == 1

        suite_ids = {tc["id"] for tc in suite["test_cases"]}
        assert all(r["test_case_id"] in suite_ids for r in result_set["results"])
