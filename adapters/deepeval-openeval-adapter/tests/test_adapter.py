"""Tests for deepeval_openeval_adapter.

Runs against the real, installed `deepeval` package (LLMTestCase, ToolCall,
RetrievedContextData, TestResult, MetricData, EvaluationResult -- no mocks,
no reinvented stand-ins) and the real `openeval.validate` validators.
"""
import pytest

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.test_case.llm_test_case import RetrievedContextData, ToolCallType
from deepeval.evaluate.types import TestResult, EvaluationResult
from deepeval.test_run.api import MetricData

from openeval.validate import validate_suite, validate_result_set

from deepeval_openeval_adapter import to_openeval, from_openeval, test_results_to_openeval

# pytest collects any top-level `test_*` callable in this module by name --
# including this imported function itself, which is not a test. Tell pytest
# to leave it alone so collection doesn't try to call it as a fixture-less
# test (it takes required arguments and isn't one).
test_results_to_openeval.__test__ = False


# ---------------------------------------------------------------------------
# Fixtures / builders using the real deepeval classes
# ---------------------------------------------------------------------------

def _basic_test_case(**overrides):
    defaults = dict(input="What is the capital of France?", actual_output="Paris is the capital of France.")
    defaults.update(overrides)
    return LLMTestCase(**defaults)


def _full_test_case():
    return LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        expected_output="Paris",
        context=["France is a country in Western Europe."],
        retrieval_context=[
            "Paris is the capital and most populous city of France.",
            RetrievedContextData(context="France's capital is Paris.", source="wiki:France"),
        ],
        tools_called=[ToolCall(name="search", input_parameters={"q": "capital of France"}, output="Paris")],
        expected_tools=[ToolCall(name="search")],
        tags=["geography", "factual"],
        name="geo_fact_1",
        comments="A basic factual QA case.",
        token_cost=0.0012,
        completion_time=1.4,
        flaky=True,
        multimodal=False,
        metadata={"source_dataset": "geo_v1", "difficulty": "easy"},
    )


# MetricData fields whose *constructor* keyword is a camelCase alias, not
# the snake_case attribute name -- verified against the real class (no
# `populate_by_name` in its model_config, so only the alias is accepted on
# construction even though the resulting attribute is snake_case).
_METRIC_DATA_ALIASES = {
    "strict_mode": "strictMode",
    "evaluation_model": "evaluationModel",
    "evaluation_cost": "evaluationCost",
    "input_tokens": "inputTokenCount",
    "output_tokens": "outputTokenCount",
    "verbose_logs": "verboseLogs",
}


def _metric_data(name="Answer Relevancy", score=0.92, success=True, **overrides):
    defaults = dict(name=name, score=score, success=success, reason="The output directly answers the query.")
    defaults.update(overrides)
    kwargs = {_METRIC_DATA_ALIASES.get(k, k): v for k, v in defaults.items()}
    return MetricData(**kwargs)


_UNSET = object()


def _test_result(name=None, index=0, metrics_data=_UNSET, success=True, **overrides):
    if metrics_data is _UNSET:
        metrics_data = [_metric_data()]
    defaults = dict(
        name=name,
        success=success,
        metrics_data=metrics_data,
        conversational=False,
        index=index,
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
    )
    defaults.update(overrides)
    return TestResult(**defaults)


# ---------------------------------------------------------------------------
# to_openeval()
# ---------------------------------------------------------------------------

class TestToOpeneval:
    def test_basic_conversion_validates(self):
        suite = to_openeval([_basic_test_case()], suite_id="s1")
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_basic_fields(self):
        suite = to_openeval([_basic_test_case()], suite_id="s1")
        tc = suite["test_cases"][0]
        assert tc["input"] == "What is the capital of France?"
        assert tc["id"] == "tc_0"
        assert tc["graders"] == ["gr_deepeval_metrics"]

    def test_full_test_case_validates_and_maps_natively(self):
        suite = to_openeval([_full_test_case()], suite_id="s1")
        result = validate_suite(suite)
        assert result.valid, result.errors

        tc = suite["test_cases"][0]
        assert tc["id"] == "geo_fact_1"  # uses LLMTestCase.name
        assert tc["expected_output"] == "Paris"
        assert tc["context"] == ["France is a country in Western Europe."]
        # RetrievedContextData stringified as "source: context", matching its own serializer
        assert tc["retrieval_context"] == [
            "Paris is the capital and most populous city of France.",
            "wiki:France: France's capital is Paris.",
        ]
        assert tc["tools_called"] == ["search"]
        assert tc["expected_tools"] == ["search"]
        assert tc["tags"] == ["geography", "factual"]

    def test_user_metadata_preserved_alongside_deepeval_namespace(self):
        suite = to_openeval([_full_test_case()], suite_id="s1")
        meta = suite["test_cases"][0]["metadata"]
        assert meta["source_dataset"] == "geo_v1"
        assert meta["difficulty"] == "easy"
        assert meta["deepeval"]["actual_output"] == "Paris is the capital of France."
        assert meta["deepeval"]["comments"] == "A basic factual QA case."
        assert meta["deepeval"]["token_cost"] == 0.0012
        assert meta["deepeval"]["completion_time"] == 1.4
        assert meta["deepeval"]["flaky"] is True
        assert meta["deepeval"]["name"] == "geo_fact_1"
        assert "identifier" in meta["deepeval"]

    def test_full_tool_call_detail_preserved_in_metadata(self):
        suite = to_openeval([_full_test_case()], suite_id="s1")
        meta = suite["test_cases"][0]["metadata"]["deepeval"]
        full = meta["tools_called_full"][0]
        assert full["name"] == "search"
        assert full["input_parameters"] == {"q": "capital of France"}
        assert full["output"] == "Paris"
        assert full["type"] == "FUNCTION"  # ToolCallType enum serialized to plain string

    def test_explicit_ids_override_name_and_index(self):
        suite = to_openeval(
            [_basic_test_case(), _full_test_case()],
            suite_id="s1",
            ids=["custom_a", "custom_b"],
        )
        ids = [tc["id"] for tc in suite["test_cases"]]
        assert ids == ["custom_a", "custom_b"]

    def test_auto_id_falls_back_to_index_when_no_name(self):
        suite = to_openeval([_basic_test_case(), _basic_test_case()], suite_id="s1")
        ids = [tc["id"] for tc in suite["test_cases"]]
        assert ids == ["tc_0", "tc_1"]

    def test_missing_input_raises(self):
        # LLMTestCase.input is a required str, but nothing stops a caller
        # from passing a falsy/empty one via **overrides.
        with pytest.raises(ValueError, match="no `input`"):
            to_openeval([_basic_test_case(input="")], suite_id="s1")

    def test_custom_grader_id_and_handler(self):
        suite = to_openeval(
            [_basic_test_case()], suite_id="s1",
            grader_id="gr_custom", grader_handler="deepeval:my_metrics",
        )
        assert suite["test_cases"][0]["graders"] == ["gr_custom"]
        assert suite["graders"][0]["id"] == "gr_custom"
        assert suite["graders"][0]["params"]["handler"] == "deepeval:my_metrics"

    def test_multiple_test_cases_share_one_grader_definition(self):
        suite = to_openeval([_basic_test_case(), _full_test_case()], suite_id="s1")
        assert len(suite["graders"]) == 1
        assert all(tc["graders"] == ["gr_deepeval_metrics"] for tc in suite["test_cases"])

    def test_suite_name_defaults_and_can_be_overridden(self):
        default_suite = to_openeval([_basic_test_case()], suite_id="s1")
        assert "s1" in default_suite["name"]
        named_suite = to_openeval([_basic_test_case()], suite_id="s1", name="My Suite")
        assert named_suite["name"] == "My Suite"

    def test_no_context_or_tools_omits_those_keys(self):
        suite = to_openeval([_basic_test_case()], suite_id="s1")
        tc = suite["test_cases"][0]
        assert "context" not in tc
        assert "retrieval_context" not in tc
        assert "tools_called" not in tc
        assert "expected_tools" not in tc
        assert "tags" not in tc


# ---------------------------------------------------------------------------
# from_openeval()
# ---------------------------------------------------------------------------

class TestFromOpeneval:
    def test_round_trip_basic(self):
        suite = to_openeval([_basic_test_case()], suite_id="s1")
        items = from_openeval(suite)
        assert len(items) == 1
        # every returned dict must be a valid LLMTestCase(**item) construction
        tc = LLMTestCase(**items[0])
        assert tc.input == "What is the capital of France?"

    def test_round_trip_full_reconstructs_a_real_llmtestcase(self):
        suite = to_openeval([_full_test_case()], suite_id="s1")
        items = from_openeval(suite)
        item = items[0]
        assert item["expected_output"] == "Paris"
        assert item["context"] == ["France is a country in Western Europe."]
        assert item["tools_called"] == ["search"]  # names only, as documented
        assert item["tags"] == ["geography", "factual"]
        assert item["name"] == "geo_fact_1"
        assert item["comments"] == "A basic factual QA case."
        assert item["token_cost"] == 0.0012
        assert item["flaky"] is True

        # tools_called comes back as plain name strings, not ToolCall objects,
        # so constructing a real LLMTestCase from it needs that conversion --
        # exactly as documented in from_openeval()'s docstring.
        item = dict(item)
        item["tools_called"] = [ToolCall(name=n) for n in item["tools_called"]]
        item["expected_tools"] = [ToolCall(name=n) for n in item["expected_tools"]]
        reconstructed = LLMTestCase(**item)
        assert reconstructed.tools_called[0].name == "search"

    def test_name_falls_back_to_suite_id_when_not_from_this_adapter(self):
        suite = {
            "version": "1.0.0", "id": "s1", "test_cases": [
                {"id": "row_42", "input": "hello", "graders": ["g1"]}
            ],
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "x"}}],
        }
        items = from_openeval(suite)
        assert items[0]["name"] == "row_42"

    def test_multiturn_input_rejected(self):
        suite = {
            "version": "1.0.0", "id": "s1", "test_cases": [
                {"id": "t1", "input": ["turn 1", "turn 2"], "graders": ["g1"]}
            ],
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "x"}}],
        }
        with pytest.raises(ValueError, match="multi-turn"):
            from_openeval(suite)

    def test_empty_suite(self):
        assert from_openeval({"test_cases": []}) == []


# ---------------------------------------------------------------------------
# test_results_to_openeval()
# ---------------------------------------------------------------------------

class TestResultsToOpeneval:
    def test_basic_conversion_validates(self):
        result_set = test_results_to_openeval(
            [_test_result()], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        result = validate_result_set(result_set)
        assert result.valid, result.errors

    def test_grader_result_fields(self):
        result_set = test_results_to_openeval(
            [_test_result()], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["grader_id"] == "answer_relevancy"
        assert gr["score"] == 0.92
        assert gr["passed"] is True
        assert gr["reason"] == "The output directly answers the query."

    def test_score_clamped_above_one(self):
        tr = _test_result(metrics_data=[_metric_data(score=1.5)])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["grader_results"][0]["score"] == 1.0
        assert validate_result_set(result_set).valid

    def test_score_clamped_below_zero(self):
        tr = _test_result(metrics_data=[_metric_data(score=-0.3)])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["grader_results"][0]["score"] == 0.0

    def test_none_score_preserved_as_null(self):
        tr = _test_result(metrics_data=[_metric_data(score=None, success=False)])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["grader_results"][0]["score"] is None
        assert validate_result_set(result_set).valid

    def test_success_none_falls_back_to_score_threshold(self):
        md = _metric_data(score=0.7, success=None)
        tr = _test_result(metrics_data=[md])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["grader_results"][0]["passed"] is True

        md_low = _metric_data(score=0.2, success=None)
        tr_low = _test_result(metrics_data=[md_low])
        result_set_low = test_results_to_openeval(
            [tr_low], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set_low["results"][0]["grader_results"][0]["passed"] is False

    def test_metric_metadata_preserved(self):
        md = _metric_data(
            threshold=0.5, strict_mode=True, evaluation_model="gpt-4o",
            evaluation_cost=0.002, input_tokens=120, output_tokens=45,
        )
        tr = _test_result(metrics_data=[md])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        meta = result_set["results"][0]["grader_results"][0]["metadata"]
        assert meta["threshold"] == 0.5
        assert meta["strict_mode"] is True
        assert meta["evaluation_model"] == "gpt-4o"
        assert meta["evaluation_cost"] == 0.002
        assert meta["input_tokens"] == 120
        assert meta["output_tokens"] == 45

    def test_metric_error_preserved(self):
        md = _metric_data(score=None, success=False, error="LLM judge call timed out")
        tr = _test_result(metrics_data=[md])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["grader_results"][0]["metadata"]["error"] == "LLM judge call timed out"

    def test_multiple_metrics_per_result(self):
        tr = _test_result(metrics_data=[
            _metric_data(name="Answer Relevancy", score=0.9, success=True),
            _metric_data(name="Faithfulness", score=0.4, success=False),
        ])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        grader_ids = [g["grader_id"] for g in result_set["results"][0]["grader_results"]]
        assert grader_ids == ["answer_relevancy", "faithfulness"]
        # overall passed follows TestResult.success (explicit), not a
        # recompute from the individual grader passes
        assert result_set["results"][0]["passed"] is True

    def test_overall_passed_recomputed_when_test_result_success_is_none(self):
        tr = _test_result(
            success=None,
            metrics_data=[
                _metric_data(name="Answer Relevancy", score=0.9, success=True),
                _metric_data(name="Faithfulness", score=0.4, success=False),
            ],
        )
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["passed"] is False

    def test_empty_metrics_data_becomes_runner_error(self):
        tr = _test_result(metrics_data=[])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        r = result_set["results"][0]
        assert r["passed"] is False
        assert r["grader_results"] == []
        assert r["error"]["type"] == "runner_error"
        assert validate_result_set(result_set).valid

    def test_none_metrics_data_becomes_runner_error(self):
        tr = _test_result(metrics_data=None)
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["error"]["type"] == "runner_error"

    def test_accepts_evaluation_result_wrapper(self):
        eval_result = EvaluationResult(
            test_results=[_test_result()], confident_link=None, test_run_id="tr-1",
        )
        result_set = test_results_to_openeval(
            eval_result, suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert len(result_set["results"]) == 1
        assert validate_result_set(result_set).valid

    def test_explicit_ids_correlate_with_to_openeval(self):
        test_cases = [_basic_test_case(), _full_test_case()]
        ids = ["row_a", "row_b"]
        suite = to_openeval(test_cases, suite_id="s1", ids=ids)
        results = [_test_result(index=0), _test_result(index=1, name="geo_fact_1")]
        result_set = test_results_to_openeval(
            results, suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z", ids=ids,
        )
        suite_ids = {tc["id"] for tc in suite["test_cases"]}
        result_ids = {r["test_case_id"] for r in result_set["results"]}
        assert suite_ids == result_ids == {"row_a", "row_b"}

    def test_auto_id_correlation_via_name_matches_to_openeval_default(self):
        # to_openeval() defaults an unnamed test case's id to tc_{i}; without
        # explicit ids, test_results_to_openeval() falls back to the same
        # tc_{i} pattern when TestResult.name is also unset -- so the
        # defaults line up automatically for the common no-name case.
        suite = to_openeval([_basic_test_case(), _basic_test_case()], suite_id="s1")
        results = [_test_result(index=0, name=None), _test_result(index=1, name=None)]
        result_set = test_results_to_openeval(
            results, suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        suite_ids = [tc["id"] for tc in suite["test_cases"]]
        result_ids = [r["test_case_id"] for r in result_set["results"]]
        assert suite_ids == result_ids == ["tc_0", "tc_1"]

    def test_actual_output_string_preserved(self):
        tr = _test_result(actual_output="Paris.")
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["actual_output"] == "Paris."

    def test_multimodal_actual_output_list_is_joined(self):
        # TestResult.actual_output: Union[Optional[str], List[Union[str, MLLMImage]]]
        # -- a real, documented shape for multimodal DeepEval test cases.
        tr = _test_result(actual_output=["The capital is ", "Paris", "."])
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["results"][0]["actual_output"] == "The capital is Paris."

    def test_summary_counts_and_avg_score(self):
        # TestResult.success is the overall record verdict, tracked
        # independently of each individual MetricData.success -- set both
        # explicitly here so the test isn't relying on the builder default.
        results = [
            _test_result(index=0, success=True, metrics_data=[_metric_data(score=1.0, success=True)]),
            _test_result(index=1, success=False, metrics_data=[_metric_data(score=0.0, success=False)]),
        ]
        result_set = test_results_to_openeval(
            results, suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        summary = result_set["summary"]
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["avg_score"] == 0.5

    def test_started_at_defaults_when_omitted(self):
        result_set = test_results_to_openeval([_test_result()], suite_id="s1", run_id="run-1")
        assert result_set["started_at"]
        assert validate_result_set(result_set).valid

    def test_completed_at_defaults_to_started_at(self):
        result_set = test_results_to_openeval(
            [_test_result()], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        assert result_set["completed_at"] == "2026-08-22T00:00:00Z"

    def test_runner_info(self):
        result_set = test_results_to_openeval(
            [_test_result()], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
            runner_version="4.1.10",
        )
        assert result_set["runner"] == {"name": "deepeval", "version": "4.1.10"}

    def test_tr_metadata_preserved(self):
        tr = _test_result(metadata={"trace_id": "abc123"})
        result_set = test_results_to_openeval(
            [tr], suite_id="s1", run_id="run-1", started_at="2026-08-22T00:00:00Z",
        )
        deepeval_meta = result_set["results"][0]["metadata"]["deepeval"]
        assert deepeval_meta["user_metadata"] == {"trace_id": "abc123"}
        assert deepeval_meta["index"] == 0
        assert deepeval_meta["conversational"] is False


# ---------------------------------------------------------------------------
# End-to-end: suite -> (simulated) run -> result set, fully validated
# ---------------------------------------------------------------------------

def test_end_to_end_suite_to_results_round_trip():
    test_cases = [_basic_test_case(), _full_test_case()]
    ids = ["case_1", "case_2"]

    suite = to_openeval(test_cases, suite_id="e2e_suite", ids=ids)
    assert validate_suite(suite).valid

    # Simulate what deepeval.evaluate() would hand back for these two cases.
    results = [
        _test_result(index=0, metrics_data=[_metric_data(name="Answer Relevancy", score=0.95, success=True)]),
        _test_result(index=1, metrics_data=[
            _metric_data(name="Answer Relevancy", score=0.88, success=True),
            _metric_data(name="Faithfulness", score=0.6, success=True),
        ]),
    ]
    result_set = test_results_to_openeval(
        results, suite_id="e2e_suite", run_id="e2e_run", started_at="2026-08-22T00:00:00Z", ids=ids,
    )
    assert validate_result_set(result_set).valid
    assert result_set["suite_id"] == suite["id"]
    assert {r["test_case_id"] for r in result_set["results"]} == {tc["id"] for tc in suite["test_cases"]}
