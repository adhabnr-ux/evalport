"""
Tests for llamaindex-openeval-adapter.

Every test runs against the real `llama_index.core.evaluation` classes and
the real `openeval.validate.validate_suite()` / `validate_result_set()` --
never mocks of *this adapter's* dependencies. Where a live LLM/embedding
call would otherwise be required (constructing an evaluator, or actually
running `BatchEvalRunner`), we configure llama_index's own first-party test
doubles (`MockLLM`, `MockEmbedding` from `llama_index.core.llms` /
`llama_index.core.embeddings`) via the real `Settings` singleton -- the same
mechanism llama_index's own test suite and its users use to run evaluators
offline, not a mock of anything this adapter itself does.
"""

from __future__ import annotations

import asyncio

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    BatchEvalRunner,
    CorrectnessEvaluator,
    EvaluationResult,
    FaithfulnessEvaluator,
    GuidelineEvaluator,
    PairwiseComparisonEvaluator,
    RelevancyEvaluator,
    SemanticSimilarityEvaluator,
)
from llama_index.core.llms import MockLLM

from openeval.validate import validate_result_set, validate_suite

from llamaindex_openeval_adapter import (
    batch_eval_result_to_openeval,
    from_openeval,
    to_openeval,
)


@pytest.fixture(autouse=True)
def _configure_settings():
    """Every evaluator in llama_index.core.evaluation resolves its LLM /
    embedding model from `Settings` when not given one explicitly -- point
    both at llama_index's own offline mock implementations so nothing in
    this test suite makes a network call or needs an API key."""
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=8)
    yield
    Settings.llm = None
    Settings.embed_model = None


def _mock_llm():
    return MockLLM()


def _mock_embed():
    return MockEmbedding(embed_dim=8)


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpeneval:
    def test_faithfulness_evaluator_maps_to_llm_judge(self):
        suite = to_openeval(
            queries=["What is the capital of France?"],
            evaluators={"faithful": FaithfulnessEvaluator(llm=_mock_llm())},
            contexts_list=[["Paris is the capital of France."]],
            suite_id="li_suite",
        )
        result = validate_suite(suite)
        assert result.valid, result.errors

        grader = suite["graders"][0]
        assert grader["id"] == "faithful"
        assert grader["type"] == "llm_judge"
        assert "{output}" in grader["params"]["prompt"]
        assert "{input}" in grader["params"]["prompt"]
        assert grader["metadata"]["llama_index"]["class"] == "FaithfulnessEvaluator"

        tc = suite["test_cases"][0]
        assert tc["id"] == "tc_0"
        assert tc["input"] == "What is the capital of France?"
        assert tc["graders"] == ["faithful"]
        assert tc["context"] == ["Paris is the capital of France."]

    def test_semantic_similarity_evaluator_maps_with_threshold(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={
                "sim": SemanticSimilarityEvaluator(
                    embed_model=_mock_embed(), similarity_threshold=0.9
                )
            },
            references=["reference answer"],
            suite_id="li_suite",
        )
        assert validate_suite(suite).valid

        grader = suite["graders"][0]
        assert grader["type"] == "semantic_similarity"
        assert grader["params"]["threshold"] == 0.9
        assert suite["test_cases"][0]["expected_output"] == "reference answer"

    def test_correctness_evaluator_prompt_includes_expected_token(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={"correct": CorrectnessEvaluator(llm=_mock_llm())},
            references=["reference answer"],
        )
        assert validate_suite(suite).valid
        prompt = suite["graders"][0]["params"]["prompt"]
        assert "{expected}" in prompt
        assert "{output}" in prompt
        assert "1 (worst) to 5 (best)" in prompt

    def test_guideline_evaluator_uses_its_own_guidelines_as_rubric(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={
                "guideline": GuidelineEvaluator(
                    llm=_mock_llm(), guidelines="The response must be in French."
                )
            },
        )
        assert validate_suite(suite).valid
        grader = suite["graders"][0]
        assert grader["type"] == "llm_judge"
        assert "The response must be in French." in grader["params"]["prompt"]
        assert grader["metadata"]["llama_index"]["guidelines"] == (
            "The response must be in French."
        )

    def test_relevancy_and_answer_relevancy_map_to_llm_judge(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={
                "relevancy": RelevancyEvaluator(llm=_mock_llm()),
                "answer_relevancy": AnswerRelevancyEvaluator(llm=_mock_llm()),
            },
        )
        assert validate_suite(suite).valid
        types = {g["id"]: g["type"] for g in suite["graders"]}
        assert types == {"relevancy": "llm_judge", "answer_relevancy": "llm_judge"}

    def test_pairwise_comparison_evaluator_maps_to_custom_without_dropping_config(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={"pairwise": PairwiseComparisonEvaluator(llm=_mock_llm())},
        )
        assert validate_suite(suite).valid
        grader = suite["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"]["handler"] == "llama_index.core.evaluation.PairwiseComparisonEvaluator"
        assert grader["metadata"]["llama_index"]["class"] == "PairwiseComparisonEvaluator"

    def test_multiple_evaluators_all_attached_to_every_test_case(self):
        suite = to_openeval(
            queries=["q1", "q2"],
            evaluators={
                "faithful": FaithfulnessEvaluator(llm=_mock_llm()),
                "sim": SemanticSimilarityEvaluator(embed_model=_mock_embed()),
            },
            references=["r1", "r2"],
        )
        assert validate_suite(suite).valid
        assert len(suite["graders"]) == 2
        for tc in suite["test_cases"]:
            assert set(tc["graders"]) == {"faithful", "sim"}

    def test_custom_ids_are_used_verbatim(self):
        suite = to_openeval(
            queries=["q1", "q2"],
            evaluators={"faithful": FaithfulnessEvaluator(llm=_mock_llm())},
            ids=["case-a", "case-b"],
        )
        assert [tc["id"] for tc in suite["test_cases"]] == ["case-a", "case-b"]

    def test_description_is_passed_through(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={"faithful": FaithfulnessEvaluator(llm=_mock_llm())},
            description="RAG regression suite",
        )
        assert suite["description"] == "RAG regression suite"

    def test_context_omitted_when_not_provided(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={"faithful": FaithfulnessEvaluator(llm=_mock_llm())},
        )
        assert "context" not in suite["test_cases"][0]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"queries": []},
            {"evaluators": {}},
            {"references": ["only one"]},
            {"contexts_list": [["only one"]]},
            {"ids": ["only-one"]},
        ],
    )
    def test_invalid_inputs_raise_value_error(self, kwargs):
        base = {
            "queries": ["q1", "q2"],
            "evaluators": {"faithful": FaithfulnessEvaluator(llm=_mock_llm())},
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            to_openeval(**base)


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


class TestFromOpeneval:
    def test_round_trips_faithfulness_evaluator(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={"faithful": FaithfulnessEvaluator(llm=_mock_llm())},
            contexts_list=[["ctx"]],
        )
        rebuilt = from_openeval(suite)
        assert rebuilt["ids"] == ["tc_0"]
        assert rebuilt["queries"] == ["q"]
        assert rebuilt["contexts_list"] == [["ctx"]]
        assert isinstance(rebuilt["evaluators"]["faithful"], FaithfulnessEvaluator)

    def test_round_trips_correctness_evaluator_with_score_threshold(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={
                "correct": CorrectnessEvaluator(llm=_mock_llm(), score_threshold=3.5)
            },
            references=["ref"],
        )
        rebuilt = from_openeval(suite)
        ev = rebuilt["evaluators"]["correct"]
        assert isinstance(ev, CorrectnessEvaluator)
        assert ev._score_threshold == 3.5
        assert rebuilt["references"] == ["ref"]

    def test_round_trips_semantic_similarity_threshold(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={
                "sim": SemanticSimilarityEvaluator(
                    embed_model=_mock_embed(), similarity_threshold=0.95
                )
            },
        )
        rebuilt = from_openeval(suite)
        ev = rebuilt["evaluators"]["sim"]
        assert isinstance(ev, SemanticSimilarityEvaluator)
        assert ev._similarity_threshold == 0.95

    def test_round_trips_guideline_evaluator_text(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={
                "guideline": GuidelineEvaluator(
                    llm=_mock_llm(), guidelines="Answer in Spanish."
                )
            },
        )
        rebuilt = from_openeval(suite)
        ev = rebuilt["evaluators"]["guideline"]
        assert isinstance(ev, GuidelineEvaluator)
        assert ev._guidelines == "Answer in Spanish."

    def test_pairwise_comparison_custom_grader_is_clean_skipped(self):
        suite = to_openeval(
            queries=["q"],
            evaluators={"pairwise": PairwiseComparisonEvaluator(llm=_mock_llm())},
        )
        rebuilt = from_openeval(suite)
        assert "pairwise" not in rebuilt["evaluators"]
        assert rebuilt["queries"] == ["q"]  # queries/ids still returned in full

    def test_bare_grader_id_string_with_no_inline_definition_is_skipped(self):
        suite = {
            "version": "1.0.0",
            "id": "hand_authored",
            "graders": [],  # no inline definition available anywhere
            "test_cases": [{"id": "tc_0", "input": "q", "graders": ["ghost_grader"]}],
        }
        rebuilt = from_openeval(suite)
        assert rebuilt["evaluators"] == {}
        assert rebuilt["queries"] == ["q"]

    def test_unrecognized_grader_type_is_clean_skipped(self):
        suite = {
            "version": "1.0.0",
            "id": "hand_authored",
            "graders": [{"id": "human_review", "type": "human"}],
            "test_cases": [
                {"id": "tc_0", "input": "q", "graders": ["human_review"]}
            ],
        }
        rebuilt = from_openeval(suite)
        assert rebuilt["evaluators"] == {}

    def test_generic_fallback_reconstructs_hand_authored_llm_judge_grader(self):
        """A grader authored outside this adapter (no llama_index metadata)
        still gets a working, real evaluator back -- via GuidelineEvaluator,
        the most generic "judge against a text rubric" evaluator."""
        suite = {
            "version": "1.0.0",
            "id": "hand_authored",
            "graders": [
                {
                    "id": "quality",
                    "type": "llm_judge",
                    "params": {
                        "model": "gpt-4o-mini",
                        "prompt": "Is {output} a good answer to {input}?",
                    },
                }
            ],
            "test_cases": [{"id": "tc_0", "input": "q", "graders": ["quality"]}],
        }
        rebuilt = from_openeval(suite)
        ev = rebuilt["evaluators"]["quality"]
        assert isinstance(ev, GuidelineEvaluator)
        assert "good answer" in ev._guidelines

    def test_generic_fallback_reconstructs_hand_authored_semantic_similarity_grader(self):
        suite = {
            "version": "1.0.0",
            "id": "hand_authored",
            "graders": [
                {"id": "sim", "type": "semantic_similarity", "params": {"threshold": 0.7}}
            ],
            "test_cases": [{"id": "tc_0", "input": "q", "graders": ["sim"]}],
        }
        rebuilt = from_openeval(suite)
        ev = rebuilt["evaluators"]["sim"]
        assert isinstance(ev, SemanticSimilarityEvaluator)
        assert ev._similarity_threshold == 0.7

    def test_multi_turn_array_input_is_joined_into_one_query_string(self):
        suite = {
            "version": "1.0.0",
            "id": "multi_turn",
            "graders": [],
            "test_cases": [
                {
                    "id": "tc_0",
                    "input": ["Hi there.", "What's the weather?"],
                    "graders": [],
                }
            ],
        }
        rebuilt = from_openeval(suite)
        assert rebuilt["queries"] == ["Hi there. What's the weather?"]

    def test_references_and_contexts_are_none_when_entirely_absent(self):
        suite = to_openeval(
            queries=["q1", "q2"],
            evaluators={"faithful": FaithfulnessEvaluator(llm=_mock_llm())},
        )
        rebuilt = from_openeval(suite)
        assert rebuilt["references"] is None
        assert rebuilt["contexts_list"] is None


# ---------------------------------------------------------------------------
# batch_eval_result_to_openeval
# ---------------------------------------------------------------------------


class TestBatchEvalResultToOpeneval:
    def test_faithfulness_style_binary_result(self):
        evaluators = {"faithful": FaithfulnessEvaluator(llm=_mock_llm())}
        results = {
            "faithful": [
                EvaluationResult(passing=True, score=1.0, feedback="YES"),
                EvaluationResult(passing=False, score=0.0, feedback="NO"),
            ]
        }
        result_set = batch_eval_result_to_openeval(
            results,
            test_case_ids=["tc_0", "tc_1"],
            evaluators=evaluators,
            response_strs=["Paris.", "The moon is made of cheese."],
            suite_id="li_suite",
            run_id="run-1",
            started_at="2026-08-15T00:00:00Z",
        )
        assert validate_result_set(result_set).valid

        rows = {r["test_case_id"]: r for r in result_set["results"]}
        assert rows["tc_0"]["passed"] is True
        assert rows["tc_0"]["grader_results"][0]["score"] == 1.0
        assert rows["tc_0"]["grader_results"][0]["type"] == "llm_judge"
        assert rows["tc_0"]["actual_output"] == "Paris."
        assert rows["tc_1"]["passed"] is False
        assert rows["tc_1"]["grader_results"][0]["score"] == 0.0

    def test_correctness_score_is_rescaled_from_1_5_to_0_1(self):
        evaluators = {"correct": CorrectnessEvaluator(llm=_mock_llm())}
        # score=4.0 is CorrectnessEvaluator's default passing threshold on its
        # native 1-5 scale -> should rescale to (4.0-1.0)/4.0 = 0.75.
        results = {
            "correct": [EvaluationResult(passing=True, score=4.0, feedback="Good")]
        }
        result_set = batch_eval_result_to_openeval(
            results,
            test_case_ids=["tc_0"],
            evaluators=evaluators,
            suite_id="s",
            run_id="r",
            started_at="2026-08-15T00:00:00Z",
        )
        assert validate_result_set(result_set).valid
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["score"] == pytest.approx(0.75)
        assert gr["metadata"]["llama_index"]["raw_score"] == 4.0

    def test_semantic_similarity_type_and_score_pass_through(self):
        evaluators = {
            "sim": SemanticSimilarityEvaluator(embed_model=_mock_embed())
        }
        results = {
            "sim": [EvaluationResult(passing=True, score=0.93, feedback="Similarity score: 0.93")]
        }
        result_set = batch_eval_result_to_openeval(
            results,
            test_case_ids=["tc_0"],
            evaluators=evaluators,
            suite_id="s",
            run_id="r",
            started_at="2026-08-15T00:00:00Z",
        )
        assert validate_result_set(result_set).valid
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["type"] == "semantic_similarity"
        assert gr["score"] == pytest.approx(0.93)

    def test_invalid_result_maps_to_null_score_and_failed(self):
        evaluators = {"faithful": FaithfulnessEvaluator(llm=_mock_llm())}
        results = {
            "faithful": [
                EvaluationResult(
                    invalid_result=True,
                    invalid_reason="could not parse judge output",
                    score=None,
                    passing=None,
                )
            ]
        }
        result_set = batch_eval_result_to_openeval(
            results,
            test_case_ids=["tc_0"],
            evaluators=evaluators,
            suite_id="s",
            run_id="r",
            started_at="2026-08-15T00:00:00Z",
        )
        assert validate_result_set(result_set).valid
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["score"] is None
        assert gr["passed"] is False
        assert "invalid_result" in gr["reason"]
        assert result_set["results"][0]["passed"] is False

    def test_pairwise_comparison_maps_to_custom_type(self):
        evaluators = {"pairwise": PairwiseComparisonEvaluator(llm=_mock_llm())}
        results = {
            "pairwise": [
                EvaluationResult(passing=True, score=1.0, pairwise_source="original")
            ]
        }
        result_set = batch_eval_result_to_openeval(
            results,
            test_case_ids=["tc_0"],
            evaluators=evaluators,
            suite_id="s",
            run_id="r",
            started_at="2026-08-15T00:00:00Z",
        )
        assert validate_result_set(result_set).valid
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["type"] == "custom"
        assert gr["metadata"]["llama_index"]["pairwise_source"] == "original"

    def test_evaluator_missing_from_evaluators_mapping_raises(self):
        results = {"faithful": [EvaluationResult(passing=True, score=1.0)]}
        with pytest.raises(ValueError):
            batch_eval_result_to_openeval(
                results,
                test_case_ids=["tc_0"],
                evaluators={},  # missing "faithful"
                suite_id="s",
                run_id="r",
                started_at="2026-08-15T00:00:00Z",
            )

    def test_mismatched_result_length_raises(self):
        evaluators = {"faithful": FaithfulnessEvaluator(llm=_mock_llm())}
        results = {"faithful": [EvaluationResult(passing=True, score=1.0)]}
        with pytest.raises(ValueError):
            batch_eval_result_to_openeval(
                results,
                test_case_ids=["tc_0", "tc_1"],  # 2 ids, only 1 result
                evaluators=evaluators,
                suite_id="s",
                run_id="r",
                started_at="2026-08-15T00:00:00Z",
            )

    def test_mismatched_response_strs_length_raises(self):
        evaluators = {"faithful": FaithfulnessEvaluator(llm=_mock_llm())}
        results = {"faithful": [EvaluationResult(passing=True, score=1.0)]}
        with pytest.raises(ValueError):
            batch_eval_result_to_openeval(
                results,
                test_case_ids=["tc_0"],
                evaluators=evaluators,
                response_strs=["a", "b"],
                suite_id="s",
                run_id="r",
                started_at="2026-08-15T00:00:00Z",
            )

    def test_completed_at_and_overall_passed_aggregate_across_graders(self):
        evaluators = {
            "faithful": FaithfulnessEvaluator(llm=_mock_llm()),
            "sim": SemanticSimilarityEvaluator(embed_model=_mock_embed()),
        }
        results = {
            "faithful": [EvaluationResult(passing=True, score=1.0)],
            "sim": [EvaluationResult(passing=False, score=0.4)],
        }
        result_set = batch_eval_result_to_openeval(
            results,
            test_case_ids=["tc_0"],
            evaluators=evaluators,
            suite_id="s",
            run_id="r",
            started_at="2026-08-15T00:00:00Z",
            completed_at="2026-08-15T00:05:00Z",
        )
        assert validate_result_set(result_set).valid
        assert result_set["completed_at"] == "2026-08-15T00:05:00Z"
        # one grader failed -> overall test case result is failed
        assert result_set["results"][0]["passed"] is False


# ---------------------------------------------------------------------------
# End-to-end: real BatchEvalRunner execution through the full round trip
# ---------------------------------------------------------------------------


class TestEndToEndWithRealBatchEvalRunner:
    def test_full_round_trip_through_a_real_batch_eval_runner_run(self):
        """to_openeval() -> from_openeval() -> a real, live
        BatchEvalRunner.aevaluate_response_strs() call (using llama_index's
        own MockEmbedding via Settings, no network) ->
        batch_eval_result_to_openeval() -> real validate_result_set()."""
        suite = to_openeval(
            queries=["What is the capital of France?", "What is 2+2?"],
            evaluators={
                "sim": SemanticSimilarityEvaluator(
                    embed_model=_mock_embed(), similarity_threshold=0.5
                )
            },
            references=["Paris is the capital of France.", "4"],
            suite_id="li_e2e_suite",
        )
        assert validate_suite(suite).valid

        rebuilt = from_openeval(suite)
        evaluators = rebuilt["evaluators"]
        assert set(evaluators.keys()) == {"sim"}

        runner = BatchEvalRunner(evaluators=evaluators, workers=2)
        response_strs = ["Paris is the capital of France.", "4"]

        results = asyncio.run(
            runner.aevaluate_response_strs(
                queries=rebuilt["queries"],
                response_strs=response_strs,
                reference=rebuilt["references"],
            )
        )
        assert set(results.keys()) == {"sim"}
        assert len(results["sim"]) == 2

        result_set = batch_eval_result_to_openeval(
            results,
            test_case_ids=rebuilt["ids"],
            evaluators=evaluators,
            response_strs=response_strs,
            suite_id="li_e2e_suite",
            run_id="run-e2e-1",
            started_at="2026-08-15T00:00:00Z",
        )
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        assert len(result_set["results"]) == 2
        for row in result_set["results"]:
            assert row["grader_results"][0]["type"] == "semantic_similarity"
            assert 0.0 <= row["grader_results"][0]["score"] <= 1.0
