import json
import math

import pytest
from openeval.validate import validate_suite, validate_result_set

from eval_ai_library_openeval_adapter import (
    from_openeval,
    metric_to_grader,
    results_to_openeval,
    to_openeval,
)
from eval_ai_library_openeval_adapter import test_case_id as tc_id_for  # aliased: pytest would collect a `test_*` name

try:  # the real upstream models, used only by the tests marked below
    from eval_lib.evaluation_schema import MetricResult
    from eval_lib.evaluation_schema import TestCaseResult as TCResult  # aliased so pytest does not try to collect it
    from eval_lib.testcases_schema import EvalTestCase
    from eval_lib import ExactMatchMetric, RegexMatchMetric, ContainsMetric, AnswerRelevancyMetric

    HAVE_EVAL_LIB = True
except Exception:  # pragma: no cover - exercised only when eval-ai-library is absent
    HAVE_EVAL_LIB = False

needs_eval_lib = pytest.mark.skipif(not HAVE_EVAL_LIB, reason="eval-ai-library not installed")


# --------------------------------------------------------------------------- #
# Plain-dict fixtures shaped exactly like eval_lib's own objects
# --------------------------------------------------------------------------- #
TC_RAG = {
    "input": "What is the capital of France?",
    "actual_output": "The capital of France is Paris.",
    "expected_output": "Paris",
    "retrieval_context": ["Paris is the capital and largest city of France."],
    "name": "capital_of_france",
}

TC_AGENT = {
    "input": "Book a table for two at 7pm.",
    "actual_output": "Booked a table for two at 7pm.",
    "expected_output": None,
    "retrieval_context": None,
    "tools_called": ["search_restaurants", "book_table"],
    "expected_tools": ["book_table"],
    "extra_fields": {"tags": ["agent", "booking"]},
}


def _mr(name, score, threshold, success, reason="ok", model="gpt-4o", cost=0.0001, log=None):
    return {
        "name": name,
        "score": score,
        "threshold": threshold,
        "success": success,
        "evaluation_cost": cost,
        "reason": reason,
        "evaluation_model": model,
        "evaluation_log": log,
    }


RESULT_RAG = {
    "input": TC_RAG["input"],
    "actual_output": TC_RAG["actual_output"],
    "expected_output": TC_RAG["expected_output"],
    "retrieval_context": TC_RAG["retrieval_context"],
    "success": True,
    "metrics_data": [
        _mr("answerRelevancyMetric", 0.92, 0.7, True, "All statements relevant."),
        _mr("faithfulnessMetric", 1.0, 0.8, True, "Grounded in context."),
        _mr("exactMatchMetric", 0.0, 0.5, False, "Output does not match expected.", model=None, cost=0.0),
    ],
}
# eval_lib's own `overall = all(d.success for d in mdata)` would make this
# False; the fixture deliberately keeps whatever the library recorded.
RESULT_RAG["success"] = False

RESULT_AGENT = {
    "input": TC_AGENT["input"],
    "actual_output": TC_AGENT["actual_output"],
    "expected_output": None,
    "retrieval_context": None,
    "tools_called": TC_AGENT["tools_called"],
    "expected_tools": TC_AGENT["expected_tools"],
    "success": True,
    "metrics_data": [
        _mr("toolCorrectnessMetric", 1.0, 0.5, True, "Expected tool was called."),
    ],
}

STARTED = "2026-09-27T10:00:00Z"


# --------------------------------------------------------------------------- #
# Suite direction
# --------------------------------------------------------------------------- #
def test_to_openeval_from_test_case_dicts_with_metric_names():
    suite = to_openeval(
        [TC_RAG, TC_AGENT],
        metrics=["answerRelevancyMetric", {"name": "exactMatchMetric", "threshold": 0.5}],
        suite_id="rag_smoke",
        name="RAG smoke",
    )
    assert suite["id"] == "rag_smoke"
    assert suite["name"] == "RAG smoke"

    graders = {g["id"]: g for g in suite["graders"]}
    assert set(graders) == {"gr_answer_relevancy_metric", "gr_exact_match_metric"}
    assert graders["gr_answer_relevancy_metric"]["type"] == "custom"
    assert graders["gr_answer_relevancy_metric"]["params"]["handler"] == "eval_ai_library:answerRelevancyMetric"
    assert graders["gr_exact_match_metric"]["type"] == "exact_match"
    assert graders["gr_exact_match_metric"]["params"]["eval_ai_library"]["threshold"] == 0.5

    tc1, tc2 = suite["test_cases"]
    assert tc1["id"] == "capital_of_france"  # EvalTestCase.name wins
    assert tc2["id"] == "tc_0002"  # positional fallback
    assert tc1["input"] == TC_RAG["input"]
    assert tc1["expected_output"] == "Paris"
    assert tc1["retrieval_context"] == TC_RAG["retrieval_context"]
    assert tc1["metadata"]["eval_ai_library"]["actual_output"] == TC_RAG["actual_output"]
    assert tc1["graders"] == ["gr_answer_relevancy_metric", "gr_exact_match_metric"]
    assert tc2["tools_called"] == ["search_restaurants", "book_table"]
    assert tc2["expected_tools"] == ["book_table"]
    assert tc2["metadata"]["eval_ai_library"]["extra_fields"] == {"tags": ["agent", "booking"]}
    assert "expected_output" not in tc2

    validation = validate_suite(suite)
    print(validation.errors)
    assert validation.valid, validation.errors


def test_to_openeval_derives_graders_from_test_case_results():
    suite = to_openeval([RESULT_RAG, RESULT_AGENT])
    ids = {g["id"] for g in suite["graders"]}
    assert ids == {
        "gr_answer_relevancy_metric",
        "gr_faithfulness_metric",
        "gr_exact_match_metric",
        "gr_tool_correctness_metric",
    }
    validation = validate_suite(suite)
    print(validation.errors)
    assert validation.valid, validation.errors


def test_to_openeval_accepts_evaluate_tuple_shape():
    # eval_lib.evaluate() returns List[Tuple[None, List[TestCaseResult]]]
    suite = to_openeval([(None, [RESULT_RAG]), (None, [RESULT_AGENT])])
    assert [tc["id"] for tc in suite["test_cases"]] == ["tc_0001", "tc_0002"]
    assert validate_suite(suite).valid


def test_to_openeval_placeholder_grader_when_nothing_known():
    suite = to_openeval([TC_RAG])
    assert suite["graders"][0]["id"] == "gr_eval_ai_library"
    assert suite["graders"][0]["type"] == "custom"
    assert suite["graders"][0]["params"]["handler"] == "eval_ai_library:evaluate"
    assert validate_suite(suite).valid


def test_to_openeval_rejects_empty_and_empty_input():
    with pytest.raises(ValueError):
        to_openeval([])
    with pytest.raises(ValueError):
        to_openeval([dict(TC_RAG, input="")], metrics=["exactMatchMetric"])
    with pytest.raises(ValueError):
        to_openeval([TC_RAG], metrics=[])


def test_to_openeval_rejects_conversational_results():
    conv = {"dialogue": [{"role": "user", "content": "hi"}], "success": True, "metrics_data": []}
    with pytest.raises(ValueError):
        to_openeval([conv])


def test_explicit_ids_and_duplicate_names_are_disambiguated():
    suite = to_openeval([TC_RAG, TC_RAG], metrics=["exactMatchMetric"])
    assert [tc["id"] for tc in suite["test_cases"]] == ["capital_of_france", "capital_of_france__2"]
    suite = to_openeval([TC_RAG, TC_AGENT], metrics=["exactMatchMetric"], ids=["a", "b"])
    assert [tc["id"] for tc in suite["test_cases"]] == ["a", "b"]
    with pytest.raises(ValueError):
        to_openeval([TC_RAG], metrics=["exactMatchMetric"], ids=["a", "b"])


def test_from_openeval_round_trip():
    suite = to_openeval([TC_RAG, TC_AGENT], metrics=["answerRelevancyMetric"])
    recovered = from_openeval(suite)
    assert len(recovered) == 2
    r1, r2 = recovered
    for field in ("input", "actual_output", "expected_output", "retrieval_context", "name"):
        assert r1[field] == TC_RAG[field]
    assert r1["tools_called"] is None and r1["expected_tools"] is None
    assert r2["input"] == TC_AGENT["input"]
    assert r2["actual_output"] == TC_AGENT["actual_output"]
    assert r2["tools_called"] == TC_AGENT["tools_called"]
    assert r2["expected_tools"] == TC_AGENT["expected_tools"]
    assert r2["extra_fields"] == TC_AGENT["extra_fields"]
    assert r2["name"] == "tc_0002"

    # And back again: a second to_openeval() of the recovered records is the
    # same suite (ids included, since name now carries the id).
    again = to_openeval(recovered, metrics=["answerRelevancyMetric"])
    assert [tc["id"] for tc in again["test_cases"]] == [tc["id"] for tc in suite["test_cases"]]
    assert again["test_cases"] == suite["test_cases"]


def test_from_openeval_handles_foreign_suite():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc1", "input": ["turn one", "turn two"], "expected_output": "x", "graders": ["g1"]}],
    }
    assert validate_suite(suite).valid
    (rec,) = from_openeval(suite)
    assert rec["input"] == "turn one\nturn two"
    assert rec["actual_output"] == ""  # eval_lib requires the field; nothing to recover
    assert rec["expected_output"] == "x"
    assert rec["name"] == "tc1"


# --------------------------------------------------------------------------- #
# Grader mapping
# --------------------------------------------------------------------------- #
def test_metric_to_grader_typed_mappings_from_metric_like_dicts():
    assert metric_to_grader("exactMatchMetric")["type"] == "exact_match"

    sem = metric_to_grader({"name": "semanticSimilarityMetric", "threshold": 0.7})
    assert sem["type"] == "semantic_similarity"
    assert sem["params"]["threshold"] == 0.7

    rx = metric_to_grader({"name": "regexMatchMetric", "threshold": 0.5, "pattern": r"\d+", "full_match": True})
    assert rx["type"] == "regex"
    assert rx["params"]["pattern"] == r"\d+"
    assert rx["params"]["eval_ai_library"]["full_match"] is True

    js = metric_to_grader({"name": "jsonSchemaMetric", "threshold": 0.5, "schema": {"type": "object"}})
    assert js["type"] == "json_schema"
    assert js["params"]["schema"] == {"type": "object"}

    one_kw = metric_to_grader({"name": "containsMetric", "threshold": 0.5, "keywords": ["Paris"], "mode": "any", "case_sensitive": False})
    assert one_kw["type"] == "contains"
    assert one_kw["params"] == {"substring": "Paris", "ignore_case": True, "eval_ai_library": {"metric": "containsMetric", "threshold": 0.5, "mode": "any"}}

    # Multi-keyword and mode="none" have no honest `contains` equivalent.
    many = metric_to_grader({"name": "containsMetric", "threshold": 0.5, "keywords": ["a", "b"], "mode": "all"})
    assert many["type"] == "custom"
    assert many["params"]["handler"] == "eval_ai_library:containsMetric"
    assert many["params"]["eval_ai_library"]["keywords"] == ["a", "b"]
    absent = metric_to_grader({"name": "containsMetric", "threshold": 0.5, "keywords": ["a"], "mode": "none"})
    assert absent["type"] == "custom"

    # Without the pattern/schema the typed mapping is not honest -> custom.
    assert metric_to_grader("regexMatchMetric")["type"] == "custom"
    assert metric_to_grader("jsonSchemaMetric")["type"] == "custom"
    # LLM-judge metrics are always custom (llm_judge needs a prompt we don't have).
    g = metric_to_grader({"name": "gEval", "threshold": 0.6, "model": "gpt-4o"})
    assert g["type"] == "custom"
    assert g["params"]["eval_ai_library"]["evaluation_model"] == "gpt-4o"

    # Every one of them validates as a Grader inside a suite (the two extra
    # containsMetric variants share one_kw's id, so they go in a second suite).
    for extra in (many, absent):
        assert validate_suite({"version": "1.0.0", "id": "s", "graders": [extra],
                               "test_cases": [{"id": "t", "input": "x", "graders": [extra["id"]]}]}).valid
    suite = {
        "version": "1.0.0",
        "id": "s",
        "graders": [sem, rx, js, one_kw, g, metric_to_grader("exactMatchMetric")],
        "test_cases": [{"id": "t", "input": "x", "graders": [sem["id"]]}],
    }
    validation = validate_suite(suite)
    print(validation.errors)
    assert validation.valid, validation.errors


def test_grader_id_slug_is_stable_and_safe():
    assert metric_to_grader("answerRelevancyMetric")["id"] == "gr_answer_relevancy_metric"
    assert metric_to_grader("gEval")["id"] == "gr_g_eval"
    assert metric_to_grader("piiLeakage")["id"] == "gr_pii_leakage"
    assert metric_to_grader("My Custom/Metric v2")["id"] == "gr_my_custom_metric_v2"
    with pytest.raises(ValueError):
        metric_to_grader({"threshold": 0.5})


# --------------------------------------------------------------------------- #
# Result direction
# --------------------------------------------------------------------------- #
def test_results_to_openeval_basic_mapping():
    rs = results_to_openeval(
        [(None, [RESULT_RAG]), (None, [RESULT_AGENT])],
        suite_id="rag_smoke",
        run_id="run_1",
        started_at=STARTED,
        completed_at="2026-09-27T10:01:00Z",
        provider={"model": "gpt-4o", "bogus": 1},
    )
    assert rs["suite_id"] == "rag_smoke"
    assert rs["run_id"] == "run_1"
    assert rs["completed_at"] == "2026-09-27T10:01:00Z"
    assert rs["provider"] == {"model": "gpt-4o"}
    assert rs["runner"] == {"name": "eval_ai_library"}

    r1, r2 = rs["results"]
    assert r1["test_case_id"] == "tc_0001"
    assert r1["actual_output"] == RESULT_RAG["actual_output"]
    assert r1["passed"] is False
    assert r1["metadata"]["eval_ai_library"]["input"] == RESULT_RAG["input"]
    assert r1["metadata"]["eval_ai_library"]["retrieval_context"] == RESULT_RAG["retrieval_context"]

    by_id = {g["grader_id"]: g for g in r1["grader_results"]}
    ar = by_id["gr_answer_relevancy_metric"]
    assert ar["type"] == "custom"
    assert ar["score"] == 0.92
    assert ar["passed"] is True
    assert ar["reason"] == "All statements relevant."
    ns = ar["metadata"]["eval_ai_library"]
    assert ns["threshold"] == 0.7
    assert ns["score"] == 0.92
    assert ns["evaluation_cost"] == 0.0001
    assert ns["evaluation_model"] == "gpt-4o"

    em = by_id["gr_exact_match_metric"]
    assert em["type"] == "exact_match"
    assert em["score"] == 0.0 and em["passed"] is False
    assert "evaluation_model" not in em["metadata"]["eval_ai_library"]  # was None

    assert r2["passed"] is True
    assert r2["grader_results"][0]["grader_id"] == "gr_tool_correctness_metric"

    s = rs["summary"]
    assert s["total"] == 2 and s["passed"] == 1 and s["failed"] == 1 and s["pass_rate"] == 0.5
    assert math.isclose(s["avg_score"], (0.92 + 1.0 + 0.0 + 1.0) / 4)
    assert s["by_grader"]["gr_exact_match_metric"] == {"passed": 0, "failed": 1, "avg_score": 0.0}
    assert math.isclose(rs["metadata"]["eval_ai_library"]["total_evaluation_cost"], 0.0003)

    validation = validate_result_set(rs)
    print(validation.errors)
    assert validation.valid, validation.errors
    json.dumps(rs)  # everything must be serializable


def test_results_to_openeval_null_score_and_skipped_metric():
    skipped_log = {"skipped": True, "skip_reason": "empty_actual_output", "final_score": 0.0}
    res = {
        "input": "q",
        "actual_output": "",
        "expected_output": None,
        "retrieval_context": None,
        "success": False,
        "metrics_data": [
            _mr("answerRelevancyMetric", 0.0, 0.7, False, "Assistant returned an empty response", log=skipped_log),
            _mr("faithfulnessMetric", None, 0.8, False, "judge call failed"),
            _mr("biasMetric", float("nan"), 0.5, False, None),
            _mr("toxicityMetric", 1.7, 0.5, True, "out-of-range from a third-party subclass"),
        ],
    }
    rs = results_to_openeval([res], run_id="r", started_at=STARTED)
    (r,) = rs["results"]
    by_id = {g["grader_id"]: g for g in r["grader_results"]}

    # eval_lib's empty-output guard reports 0.0 explicitly: kept, but flagged.
    sk = by_id["gr_answer_relevancy_metric"]
    assert sk["score"] == 0.0
    assert sk["metadata"]["eval_ai_library"]["skipped"] is True
    assert sk["metadata"]["eval_ai_library"]["skip_reason"] == "empty_actual_output"

    # None / NaN -> null per the spec ("Null if skipped or errored").
    assert by_id["gr_faithfulness_metric"]["score"] is None
    assert by_id["gr_faithfulness_metric"]["reason"] == "judge call failed"
    assert by_id["gr_bias_metric"]["score"] is None
    assert "reason" not in by_id["gr_bias_metric"]

    # Out of range -> clamped, raw preserved.
    assert by_id["gr_toxicity_metric"]["score"] == 1.0
    assert by_id["gr_toxicity_metric"]["metadata"]["eval_ai_library"]["score"] == 1.7

    assert rs["summary"]["avg_score"] == 0.5  # mean over the two non-null scores
    assert r["actual_output"] == ""

    validation = validate_result_set(rs)
    print(validation.errors)
    assert validation.valid, validation.errors


def test_results_to_openeval_error_path():
    class LiteLLMAPIConnectionError(Exception):
        pass

    rs = results_to_openeval(
        [RESULT_AGENT],
        run_id="r",
        started_at=STARTED,
        errors={
            1: LiteLLMAPIConnectionError("upstream 502"),
            2: TimeoutError("judge timed out"),
            3: {"type": "runner_error", "message": "metric raised", "code": "E1", "retryable": False},
            4: "plain string failure",
        },
        durations_ms={0: 1234.6, "tc_0002": 50},
    )
    ids = [r["test_case_id"] for r in rs["results"]]
    assert ids == ["tc_0001", "tc_0002", "tc_0003", "tc_0004", "tc_0005"]
    ok, e1, e2, e3, e4 = rs["results"]
    assert ok["passed"] is True and "error" not in ok and ok["duration_ms"] == 1235
    assert e1["error"] == {"type": "provider_error", "message": "LiteLLMAPIConnectionError: upstream 502"}
    assert e1["passed"] is False and e1["grader_results"] == [] and e1["duration_ms"] == 50
    assert e2["error"]["type"] == "timeout"
    assert e3["error"] == {"type": "runner_error", "message": "metric raised", "code": "E1", "retryable": False}
    assert e4["error"] == {"type": "runner_error", "message": "plain string failure"}
    assert rs["summary"] == {
        "total": 5, "passed": 1, "failed": 4, "pass_rate": 0.2, "avg_score": 1.0,
        "by_grader": {"gr_tool_correctness_metric": {"passed": 1, "failed": 0, "avg_score": 1.0}},
    }
    validation = validate_result_set(rs)
    print(validation.errors)
    assert validation.valid, validation.errors

    # An error keyed on an existing result marks that result failed, keeps its graders.
    rs2 = results_to_openeval([RESULT_AGENT], run_id="r", started_at=STARTED, errors={"tc_0001": RuntimeError("late failure")})
    (r,) = rs2["results"]
    assert r["passed"] is False and r["error"]["type"] == "runner_error" and len(r["grader_results"]) == 1
    assert validate_result_set(rs2).valid

    with pytest.raises(ValueError):
        results_to_openeval([], run_id="r", started_at=STARTED)


def test_suite_and_results_share_ids_and_grader_types():
    metrics = [
        {"name": "regexMatchMetric", "threshold": 0.5, "pattern": "Paris"},
        "answerRelevancyMetric",
    ]
    results = [
        dict(RESULT_RAG, metrics_data=[
            _mr("regexMatchMetric", 1.0, 0.5, True, "Pattern matched: Paris", model=None, cost=0.0),
            _mr("answerRelevancyMetric", 0.9, 0.7, True),
        ], success=True),
        RESULT_AGENT,
    ]
    suite = to_openeval(results, metrics, suite_id="s")
    rs = results_to_openeval(results, metrics, suite_id="s", run_id="r", started_at=STARTED)

    assert {tc["id"] for tc in suite["test_cases"]} == {r["test_case_id"] for r in rs["results"]}
    suite_types = {g["id"]: g["type"] for g in suite["graders"]}
    assert suite_types["gr_regex_match_metric"] == "regex"
    for r in rs["results"]:
        for gr in r["grader_results"]:
            if gr["grader_id"] in suite_types:
                assert gr["type"] == suite_types[gr["grader_id"]]
    # toolCorrectnessMetric was not in `metrics`, so it is not a suite grader;
    # its GraderResult still names a stable id/type of its own.
    tool = rs["results"][1]["grader_results"][0]
    assert tool["grader_id"] == "gr_tool_correctness_metric" and tool["type"] == "custom"

    # Without the metric objects, the same regex metric can only be `custom`
    # in both documents -- consistently.
    suite_plain = to_openeval(results, suite_id="s")
    rs_plain = results_to_openeval(results, suite_id="s", run_id="r", started_at=STARTED)
    assert {g["id"]: g["type"] for g in suite_plain["graders"]}["gr_regex_match_metric"] == "custom"
    assert rs_plain["results"][0]["grader_results"][0]["type"] == "custom"

    assert validate_suite(suite).valid
    assert validate_result_set(rs).valid


def test_test_case_id_helper():
    assert tc_id_for({"name": " named "}, 0) == "named"
    assert tc_id_for({"name": ""}, 0) == "tc_0001"
    assert tc_id_for({}, 41, id_prefix="case-") == "case-0042"


# --------------------------------------------------------------------------- #
# Real eval_lib models (skipped cleanly when eval-ai-library is not installed)
# --------------------------------------------------------------------------- #
@needs_eval_lib
def test_real_eval_lib_models_round_trip():
    tcs = [
        EvalTestCase(**TC_RAG),
        EvalTestCase(**{k: v for k, v in TC_AGENT.items() if v is not None}),
    ]
    metrics = [
        ExactMatchMetric(threshold=0.5, case_sensitive=False),
        RegexMatchMetric(pattern="Paris", threshold=0.5),
        ContainsMetric(keywords=["Paris"], threshold=0.5),
        AnswerRelevancyMetric(model="gpt-4o", threshold=0.7),
    ]
    suite = to_openeval(tcs, metrics, suite_id="real")
    types = {g["id"]: g["type"] for g in suite["graders"]}
    assert types == {
        "gr_exact_match_metric": "exact_match",
        "gr_regex_match_metric": "regex",
        "gr_contains_metric": "contains",
        "gr_answer_relevancy_metric": "custom",
    }
    graders = {g["id"]: g for g in suite["graders"]}
    assert graders["gr_regex_match_metric"]["params"]["pattern"] == "Paris"
    assert graders["gr_contains_metric"]["params"]["substring"] == "Paris"
    assert graders["gr_contains_metric"]["params"]["ignore_case"] is True
    assert graders["gr_exact_match_metric"]["params"]["case_sensitive"] is False
    assert graders["gr_answer_relevancy_metric"]["params"]["eval_ai_library"]["evaluation_model"] == "gpt-4o"
    validation = validate_suite(suite)
    print(validation.errors)
    assert validation.valid, validation.errors

    models = from_openeval(suite, as_models=True)
    assert all(isinstance(m, EvalTestCase) for m in models)
    assert models[0].model_dump() == tcs[0].model_dump()
    assert models[1].input == tcs[1].input
    assert models[1].extra_fields == TC_AGENT["extra_fields"]

    # Build the same shape eval_lib.evaluate() returns, with real dataclasses,
    # mirroring evaluate.py's own construction (no LLM call needed).
    mdata = [
        MetricResult(name="exactMatchMetric", score=0.0, threshold=0.5, success=False, evaluation_cost=0.0,
                     reason="Output does not match expected.", evaluation_model=None,
                     evaluation_log={"matched": False}),
        MetricResult(name="regexMatchMetric", score=1.0, threshold=0.5, success=True, evaluation_cost=0.0,
                     reason="Pattern matched: Paris", evaluation_model=None, evaluation_log={"matched": True}),
        MetricResult(name="containsMetric", score=1.0, threshold=0.5, success=True, evaluation_cost=0.0,
                     reason="Found 1/1 keywords (mode=any)", evaluation_model=None, evaluation_log=None),
        MetricResult(name="answerRelevancyMetric", score=0.95, threshold=0.7, success=True, evaluation_cost=0.00021,
                     reason="Relevant.", evaluation_model="gpt-4o", evaluation_log={"final_score": 0.95}),
    ]
    evaluate_output = [
        (None, [TCResult(
            input=tcs[0].input, actual_output=tcs[0].actual_output, expected_output=tcs[0].expected_output,
            retrieval_context=tcs[0].retrieval_context, tools_called=tcs[0].tools_called,
            expected_tools=tcs[0].expected_tools, success=all(m.success for m in mdata), metrics_data=mdata,
        )]),
        (None, [TCResult(
            input=tcs[1].input, actual_output=tcs[1].actual_output, expected_output=None,
            retrieval_context=None, tools_called=tcs[1].tools_called, expected_tools=tcs[1].expected_tools,
            success=True, metrics_data=[mdata[1]],
        )]),
    ]
    # Without `test_cases=` results only have positional ids, since
    # TestCaseResult carries no name...
    positional = results_to_openeval(evaluate_output, metrics, suite_id="real", run_id="run", started_at=STARTED)
    assert [r["test_case_id"] for r in positional["results"]] == ["tc_0001", "tc_0002"]
    # ...so hand the original EvalTestCase list back in to get the suite's ids.
    rs = results_to_openeval(evaluate_output, metrics, suite_id="real", run_id="run", started_at=STARTED, test_cases=tcs)
    validation = validate_result_set(rs)
    print(validation.errors)
    assert validation.valid, validation.errors
    r1, r2 = rs["results"]
    assert r1["passed"] is False and r2["passed"] is True
    by_id = {g["grader_id"]: g for g in r1["grader_results"]}
    assert by_id["gr_regex_match_metric"]["type"] == "regex"
    assert by_id["gr_contains_metric"]["type"] == "contains"
    assert by_id["gr_answer_relevancy_metric"]["score"] == 0.95
    assert by_id["gr_answer_relevancy_metric"]["metadata"]["eval_ai_library"]["evaluation_log"] == {"final_score": 0.95}
    # Suite (from EvalTestCase list) and ResultSet (from TestCaseResult list) join on id.
    assert {tc["id"] for tc in suite["test_cases"]} == {r["test_case_id"] for r in rs["results"]} == {"capital_of_france", "tc_0002"}
    json.dumps(rs)


@needs_eval_lib
def test_real_eval_lib_asdict_dumps_behave_like_models():
    import dataclasses

    mr = MetricResult(name="gEval", score=0.6, threshold=0.5, success=True, evaluation_cost=None,
                      reason=None, evaluation_model="gpt-4o-mini")
    tcr = TCResult(input="q", actual_output="a", expected_output=None, retrieval_context=None,
                   success=True, metrics_data=[mr])
    from_models = results_to_openeval([tcr], run_id="r", started_at=STARTED)
    from_dicts = results_to_openeval([dataclasses.asdict(tcr)], run_id="r", started_at=STARTED)
    assert from_models == from_dicts
    gr = from_models["results"][0]["grader_results"][0]
    assert gr["grader_id"] == "gr_g_eval" and "reason" not in gr
    assert "evaluation_cost" not in gr["metadata"]["eval_ai_library"]
    assert validate_result_set(from_models).valid
