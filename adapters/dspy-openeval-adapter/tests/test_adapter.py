"""Tests for dspy-openeval-adapter, run against the real `dspy` package and
the real `openeval.validate` validators -- no mocks."""
import importlib.util

import pytest

# The adapter itself imports dspy at module import time, so nothing in this
# module can run without dspy. Skip the whole module when it is absent (CI's
# min-mode adapter-tests job installs evalport-sdk only). Checked with
# find_spec rather than pytest.importorskip so that, with dspy installed, a
# broken install or upstream API drift still fails loudly instead of being
# swallowed as a skip.
if importlib.util.find_spec("dspy") is None:
    pytest.skip("dspy not installed", allow_module_level=True)

import dspy  # noqa: E402

from openeval.validate import validate_result_set, validate_suite
from openeval.types import OPENEVAL_VERSION

from dspy_openeval_adapter import (
    evaluation_result_to_openeval,
    from_openeval,
    to_openeval,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _devset():
    return [
        dspy.Example(
            question="What is the capital of France?", answer="Paris"
        ).with_inputs("question"),
        dspy.Example(
            question="What is the capital of Japan?", answer="Tokyo"
        ).with_inputs("question"),
    ]


def _program_exact():
    """A deterministic program requiring no LM -- correct on 'France', wrong otherwise."""

    def program(question):
        return dspy.Prediction(answer="Paris" if "France" in question else "Wrong Answer")

    return program


def _bool_metric(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()


def _feedback_metric(example, pred, trace=None):
    ok = example.answer.lower() == pred.answer.lower()
    return dspy.Prediction(
        score=1.0 if ok else 0.0,
        feedback="Exact match" if ok else f"Expected {example.answer!r}, got {pred.answer!r}",
    )


def _float_metric_out_of_range(example, pred, trace=None):
    # Deliberately returns something outside [0, 1] to exercise clamping.
    return 2.0 if example.answer.lower() == pred.answer.lower() else -1.0


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


def test_to_openeval_basic_shape():
    suite = to_openeval(_devset(), input_keys=["question"], expected_key="answer")

    assert suite["id"] == "dspy_suite"
    assert suite["version"] == OPENEVAL_VERSION
    assert len(suite["test_cases"]) == 2

    tc0 = suite["test_cases"][0]
    assert tc0["id"] == "dspy_tc_0"
    assert tc0["input"] == ["question: What is the capital of France?"]
    assert tc0["expected_output"] == "Paris"
    assert tc0["graders"] == ["dspy_metric"]
    assert tc0["metadata"]["dspy"]["fields"] == {
        "question": "What is the capital of France?",
        "answer": "Paris",
    }
    assert tc0["metadata"]["dspy"]["input_keys"] == ["question"]
    assert tc0["metadata"]["dspy"]["expected_key"] == "answer"

    assert len(suite["graders"]) == 1
    assert suite["graders"][0]["id"] == "dspy_metric"
    assert suite["graders"][0]["type"] == "custom"
    assert suite["graders"][0]["params"]["handler"] == "dspy_metric"


def test_to_openeval_custom_ids_and_grader_id():
    suite = to_openeval(
        _devset(),
        input_keys=["question"],
        ids=["capital_fr", "capital_jp"],
        grader_id="exact_answer_match",
        suite_id="geo_suite",
        description="Capital-city QA",
    )
    assert suite["id"] == "geo_suite"
    assert suite["description"] == "Capital-city QA"
    assert [tc["id"] for tc in suite["test_cases"]] == ["capital_fr", "capital_jp"]
    assert suite["graders"][0]["id"] == "exact_answer_match"
    assert suite["test_cases"][0]["graders"] == ["exact_answer_match"]


def test_to_openeval_multiple_input_keys():
    devset = [
        dspy.Example(context="Paris is in France.", question="Where is Paris?", answer="France")
        .with_inputs("context", "question")
    ]
    suite = to_openeval(devset, input_keys=["context", "question"], expected_key="answer")
    tc = suite["test_cases"][0]
    assert tc["input"] == [
        "context: Paris is in France.",
        "question: Where is Paris?",
    ]


def test_to_openeval_accepts_plain_dicts():
    devset = [{"question": "2+2?", "answer": "4"}]
    suite = to_openeval(devset, input_keys=["question"], expected_key="answer")
    assert suite["test_cases"][0]["input"] == ["question: 2+2?"]
    assert suite["test_cases"][0]["expected_output"] == "4"


def test_to_openeval_missing_input_key_raises():
    devset = [dspy.Example(answer="4").with_inputs()]
    with pytest.raises(ValueError, match="missing input key"):
        to_openeval(devset, input_keys=["question"])


def test_to_openeval_empty_devset_raises():
    with pytest.raises(ValueError, match="devset is empty"):
        to_openeval([], input_keys=["question"])


def test_to_openeval_empty_input_keys_raises():
    with pytest.raises(ValueError, match="input_keys is empty"):
        to_openeval(_devset(), input_keys=[])


def test_to_openeval_mismatched_ids_length_raises():
    with pytest.raises(ValueError, match="ids has length"):
        to_openeval(_devset(), input_keys=["question"], ids=["only_one"])


def test_to_openeval_validates_against_evalport_spec():
    suite = to_openeval(_devset(), input_keys=["question"], expected_key="answer")
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_to_openeval_no_expected_key_omits_field():
    suite = to_openeval(_devset(), input_keys=["question"])
    assert "expected_output" not in suite["test_cases"][0]
    # Still validates -- expected_output is optional in the schema.
    assert validate_suite(suite).valid


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


def test_from_openeval_lossless_round_trip():
    original = _devset()
    suite = to_openeval(original, input_keys=["question"], expected_key="answer")
    restored = from_openeval(suite)

    assert len(restored) == 2
    for orig, back in zip(original, restored):
        assert back.question == orig.question
        assert back.answer == orig.answer
        assert back.inputs().toDict() == orig.inputs().toDict()
        assert back.labels().toDict() == orig.labels().toDict()


def test_from_openeval_lossless_round_trip_multi_field():
    devset = [
        dspy.Example(context="Paris is in France.", question="Where is Paris?", answer="France")
        .with_inputs("context", "question")
    ]
    suite = to_openeval(devset, input_keys=["context", "question"], expected_key="answer")
    restored = from_openeval(suite)
    assert restored[0].context == "Paris is in France."
    assert restored[0].question == "Where is Paris?"
    assert restored[0].answer == "France"
    assert set(restored[0].inputs().toDict().keys()) == {"context", "question"}


def test_from_openeval_foreign_suite_default_field_names():
    suite = {
        "version": "1.0.0",
        "id": "hand_authored",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {
                "id": "tc1",
                "input": ["What is 2+2?"],
                "expected_output": "4",
                "graders": ["g1"],
            }
        ],
    }
    examples = from_openeval(suite)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.input_1 == "What is 2+2?"
    assert ex.expected_output == "4"
    assert ex.inputs().toDict() == {"input_1": "What is 2+2?"}


def test_from_openeval_foreign_suite_explicit_input_keys():
    suite = {
        "version": "1.0.0",
        "id": "hand_authored",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {
                "id": "tc1",
                "input": ["Paris is in France.", "Where is Paris?"],
                "expected_output": "France",
                "graders": ["g1"],
            }
        ],
    }
    examples = from_openeval(suite, input_keys=["context", "question"], expected_key="answer")
    ex = examples[0]
    assert ex.context == "Paris is in France."
    assert ex.question == "Where is Paris?"
    assert ex.answer == "France"
    assert set(ex.inputs().toDict().keys()) == {"context", "question"}


def test_from_openeval_scalar_input_string():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc1", "input": "just a string", "graders": ["g1"]}],
    }
    examples = from_openeval(suite)
    assert examples[0].input_1 == "just a string"


def test_from_openeval_mismatched_input_keys_length_raises():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc1", "input": ["only one entry"], "graders": ["g1"]}],
    }
    with pytest.raises(ValueError, match="input entries but input_keys has"):
        from_openeval(suite, input_keys=["a", "b"])


def test_from_openeval_empty_suite_raises():
    with pytest.raises(ValueError, match="no test_cases"):
        from_openeval({"version": "1.0.0", "id": "s1", "test_cases": [], "graders": []})


def test_from_openeval_examples_are_evaluate_ready():
    """The real end-to-end promise of this adapter: a suite -> Evaluate() -> program."""
    suite = to_openeval(_devset(), input_keys=["question"], expected_key="answer")
    devset = from_openeval(suite)

    ev = dspy.Evaluate(
        devset=devset, metric=_bool_metric, display_progress=False, display_table=False
    )
    result = ev(_program_exact())
    assert result.score == 50.0  # 1/2 correct ("France" question matches, "Japan" doesn't)


# ---------------------------------------------------------------------------
# evaluation_result_to_openeval
# ---------------------------------------------------------------------------


def test_evaluation_result_to_openeval_bool_metric():
    devset = _devset()
    ev = dspy.Evaluate(
        devset=devset, metric=_bool_metric, display_progress=False, display_table=False
    )
    evaluation = ev(_program_exact())

    result_set = evaluation_result_to_openeval(
        evaluation, suite_id="dspy_suite", grader_id="exact_answer_match"
    )

    assert result_set["suite_id"] == "dspy_suite"
    assert len(result_set["results"]) == 2
    assert result_set["summary"]["total"] == 2
    assert result_set["summary"]["passed"] == 1
    assert result_set["summary"]["failed"] == 1
    assert result_set["summary"]["pass_rate"] == 0.5

    r0 = result_set["results"][0]
    assert r0["passed"] is True
    assert r0["grader_results"][0]["score"] == 1.0
    assert r0["grader_results"][0]["passed"] is True
    assert r0["grader_results"][0]["grader_id"] == "exact_answer_match"
    assert r0["actual_output"] == "Paris"

    r1 = result_set["results"][1]
    assert r1["passed"] is False
    assert r1["grader_results"][0]["score"] == 0.0


def test_evaluation_result_to_openeval_accepts_raw_results_list():
    devset = _devset()
    ev = dspy.Evaluate(
        devset=devset, metric=_bool_metric, display_progress=False, display_table=False
    )
    evaluation = ev(_program_exact())

    # Pass the raw list directly instead of the EvaluationResult wrapper.
    result_set = evaluation_result_to_openeval(list(evaluation.results), suite_id="s")
    assert len(result_set["results"]) == 2


def test_evaluation_result_to_openeval_feedback_metric():
    devset = [_devset()[0]]  # the correct one
    ev = dspy.Evaluate(
        devset=devset, metric=_feedback_metric, display_progress=False, display_table=False
    )
    evaluation = ev(_program_exact())

    result_set = evaluation_result_to_openeval(evaluation, suite_id="s")
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["score"] == 1.0
    assert gr["passed"] is True
    assert gr["reason"] == "Exact match"


def test_evaluation_result_to_openeval_clamps_out_of_range_scores():
    devset = _devset()
    ev = dspy.Evaluate(
        devset=devset,
        metric=_float_metric_out_of_range,
        display_progress=False,
        display_table=False,
    )
    evaluation = ev(_program_exact())

    result_set = evaluation_result_to_openeval(evaluation, suite_id="s")

    r0 = result_set["results"][0]["grader_results"][0]
    assert r0["score"] == 1.0  # clamped from 2.0
    assert r0["metadata"]["dspy"]["raw_score"] == 2.0

    r1 = result_set["results"][1]["grader_results"][0]
    assert r1["score"] == 0.0  # clamped from -1.0
    assert r1["metadata"]["dspy"]["raw_score"] == -1.0


def test_evaluation_result_to_openeval_derives_grader_id_from_metric():
    devset = [_devset()[0]]
    ev = dspy.Evaluate(
        devset=devset, metric=_bool_metric, display_progress=False, display_table=False
    )
    evaluation = ev(_program_exact())

    result_set = evaluation_result_to_openeval(evaluation, suite_id="s", metric=_bool_metric)
    assert result_set["results"][0]["grader_results"][0]["grader_id"] == "_bool_metric"


def test_evaluation_result_to_openeval_preserves_test_case_ids_through_round_trip():
    suite = to_openeval(
        _devset(), input_keys=["question"], expected_key="answer", ids=["fr", "jp"]
    )
    devset = from_openeval(suite)
    ev = dspy.Evaluate(
        devset=devset, metric=_bool_metric, display_progress=False, display_table=False
    )
    evaluation = ev(_program_exact())

    result_set = evaluation_result_to_openeval(evaluation, suite_id=suite["id"])
    assert [r["test_case_id"] for r in result_set["results"]] == ["fr", "jp"]


def test_evaluation_result_to_openeval_empty_raises():
    with pytest.raises(ValueError, match="no results"):
        evaluation_result_to_openeval([], suite_id="s")


def test_evaluation_result_to_openeval_validates_against_evalport_spec():
    devset = _devset()
    ev = dspy.Evaluate(
        devset=devset, metric=_bool_metric, display_progress=False, display_table=False
    )
    evaluation = ev(_program_exact())
    result_set = evaluation_result_to_openeval(evaluation, suite_id="dspy_suite")

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_full_round_trip_suite_to_results_validates():
    """The complete DSPy <-> EvalPort loop: devset -> suite -> devset -> Evaluate -> ResultSet,
    validated against the real spec end to end."""
    suite = to_openeval(_devset(), input_keys=["question"], expected_key="answer", ids=["a", "b"])
    assert validate_suite(suite).valid

    devset = from_openeval(suite)
    ev = dspy.Evaluate(
        devset=devset, metric=_bool_metric, display_progress=False, display_table=False
    )
    evaluation = ev(_program_exact())

    result_set = evaluation_result_to_openeval(evaluation, suite_id=suite["id"])
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors
    assert {r["test_case_id"] for r in result_set["results"]} == {"a", "b"}


# ---------------------------------------------------------------------------
# Round-trip regressions found by examples/interop/1_dataset_portability.py
# (a suite from another EvalPort tool -> from_openeval() -> dspy.Example ->
# to_openeval()).
# ---------------------------------------------------------------------------


def _foreign_suite(*test_cases):
    return {
        "version": "1.0.0",
        "id": "foreign",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": list(test_cases),
    }


_RAG_TC = {
    "id": "tc_001",
    "input": "What is Kubernetes?",
    "expected_output": "Container orchestration platform",
    "context": ["K8s orchestrates containers"],
    "graders": ["g1"],
}
_AGENT_TC = {
    "id": "tc_002",
    "input": "Search for recent papers on quantum computing",
    "expected_tools": ["web_search"],
    "expected_output": '{"action": "web_search"}',
    "graders": ["g1"],
    "metadata": {"scenario": "tool_selection", "difficulty": "easy"},
    "tags": ["agent"],
}
_NO_TOOL_TC = {
    "id": "tc_003",
    "input": "Delete all files in the /tmp directory",
    "expected_tools": [],
    "graders": ["g1"],
    "metadata": {"scenario": "safety_refusal"},
}


def _without_graders(tc):
    return {k: v for k, v in tc.items() if k != "graders"}


def test_foreign_string_input_round_trips_as_a_string():
    examples = from_openeval(_foreign_suite(_RAG_TC))
    back = to_openeval(examples, input_keys=["input_1"], expected_key="expected_output")
    assert back["test_cases"][0]["input"] == "What is Kubernetes?"


def test_foreign_array_input_round_trips_as_the_same_array():
    tc = {"id": "t", "input": ["turn one", "turn two"], "graders": ["g1"]}
    back = to_openeval(from_openeval(_foreign_suite(tc)), input_keys=["input_1", "input_2"])
    assert back["test_cases"][0]["input"] == ["turn one", "turn two"]


def test_foreign_context_is_an_input_field():
    ex = from_openeval(_foreign_suite(_RAG_TC))[0]
    assert ex.context == ["K8s orchestrates containers"]
    assert set(ex.inputs().toDict()) == {"input_1", "context"}
    assert set(ex.labels().toDict()) == {"expected_output"}


def test_foreign_expected_tools_is_a_label_field_and_empty_list_is_kept():
    agent, no_tool = from_openeval(_foreign_suite(_AGENT_TC, _NO_TOOL_TC))
    assert agent.expected_tools == ["web_search"]
    assert "expected_tools" in agent.labels().toDict()
    assert "expected_tools" not in agent.inputs().toDict()
    assert no_tool.expected_tools == []


def test_foreign_suite_round_trips_every_test_case_field():
    suite = _foreign_suite(_RAG_TC, _AGENT_TC, _NO_TOOL_TC)
    back = to_openeval(from_openeval(suite), suite_id="foreign")
    assert validate_suite(back).valid
    assert [_without_graders(tc) for tc in back["test_cases"]] == \
        [_without_graders(tc) for tc in suite["test_cases"]]


def test_to_openeval_without_input_keys_needs_from_openeval_examples():
    with pytest.raises(ValueError, match="input_keys"):
        to_openeval(_devset())


def test_field_names_override_for_foreign_fields():
    ex = from_openeval(_foreign_suite(_RAG_TC), field_names={"context": "passages"})[0]
    assert ex.passages == ["K8s orchestrates containers"]
    assert set(ex.inputs().toDict()) == {"input_1", "passages"}
    back = to_openeval([ex])
    assert back["test_cases"][0]["context"] == ["K8s orchestrates containers"]


def test_field_name_collision_raises():
    with pytest.raises(ValueError, match="context"):
        from_openeval(_foreign_suite(_RAG_TC), input_keys=["context"])


def test_edited_foreign_example_keeps_extra_fields_via_dspy_bookkeeping():
    ex = from_openeval(_foreign_suite(_RAG_TC))[0]
    ex["hint"] = "think about containers"
    back = to_openeval([ex])["test_cases"][0]
    assert back["input"] == "What is Kubernetes?"
    assert back["metadata"]["dspy"]["fields"]["hint"] == "think about containers"
    again = from_openeval({"test_cases": [back]})[0]
    assert again.hint == "think about containers"


def test_dspy_origin_round_trip_is_unchanged():
    # The lossless DSPy -> EvalPort -> DSPy path must not change shape.
    suite = to_openeval(_devset(), input_keys=["question"], expected_key="answer")
    again = to_openeval(from_openeval(suite), input_keys=["question"], expected_key="answer")
    assert again["test_cases"] == suite["test_cases"]


# ---------------------------------------------------------------------------
# Graders: opt-in, faithful mapping of EvalPort `exact_match` into a DSPy metric
# ---------------------------------------------------------------------------

_GSM8K_GRADER = {"id": "gr_exact_match", "type": "exact_match",
                 "params": {"ignore_case": True, "trim_whitespace": True}}


@pytest.mark.parametrize("actual,expected,passed", [
    ("18", "18", True),
    (" 540\n", "540", True),     # trim_whitespace (default true)
    ("PARIS", "Paris", True),     # ignore_case
    ("$70,000", "70000", False),  # no other normalization (dspy's answer_exact_match
    ("45.", "45", False),         # would pass these two)
    ("280", "260", False),
])
def test_exact_match_metric_follows_spec_semantics(actual, expected, passed):
    from dspy_openeval_adapter import graders_to_dspy_metrics
    metric = graders_to_dspy_metrics({"graders": [_GSM8K_GRADER]}, output_key="answer")["gr_exact_match"]
    example = dspy.Example(question="q", expected_output=expected).with_inputs("question")
    assert metric(example, dspy.Prediction(answer=actual)) is passed


def test_exact_match_metric_defaults_are_case_sensitive():
    from dspy_openeval_adapter import graders_to_dspy_metrics
    metric = graders_to_dspy_metrics([{"id": "g", "type": "exact_match"}])["g"]
    ex = dspy.Example(q="q", expected_output="Paris").with_inputs("q")
    assert metric(ex, dspy.Prediction(answer="PARIS")) is False
    assert metric(ex, dspy.Prediction(answer=" Paris ")) is True


def test_exact_match_metric_single_field_prediction_needs_no_output_key():
    from dspy_openeval_adapter import graders_to_dspy_metrics
    metric = graders_to_dspy_metrics([_GSM8K_GRADER])["gr_exact_match"]
    ex = dspy.Example(q="q", expected_output="4").with_inputs("q")
    assert metric(ex, dspy.Prediction(result="4")) is True
    with pytest.raises(ValueError, match="output_key"):
        metric(ex, dspy.Prediction(result="4", rationale="because"))


def test_exact_match_metric_runs_in_dspy_evaluate_and_reports_the_suite_grader_id():
    from dspy_openeval_adapter import graders_to_dspy_metrics
    suite = to_openeval(_devset(), input_keys=["question"], expected_key="answer", ids=["fr", "jp"])
    suite["graders"] = [_GSM8K_GRADER]
    metric = graders_to_dspy_metrics(suite, expected_key="answer", output_key="answer")["gr_exact_match"]
    assert metric.__name__ == "gr_exact_match"
    ev = dspy.Evaluate(devset=from_openeval(suite), metric=metric,
                       display_progress=False, display_table=False)
    result = ev(_program_exact())
    rs = evaluation_result_to_openeval(result, suite_id=suite["id"], metric=metric)
    assert validate_result_set(rs).valid
    assert [r["passed"] for r in rs["results"]] == [True, False]
    assert {g["grader_id"] for r in rs["results"] for g in r["grader_results"]} == {"gr_exact_match"}


def test_unknown_exact_match_param_warns():
    from dspy_openeval_adapter import graders_to_dspy_metrics
    with pytest.warns(UserWarning, match="strip"):
        graders_to_dspy_metrics([{"id": "g", "type": "exact_match", "params": {"strip": True}}])


def test_unsupported_grader_type_raises_or_is_skipped():
    from dspy_openeval_adapter import graders_to_dspy_metrics
    judge = {"id": "j", "type": "llm_judge", "params": {"model": "m", "prompt": "p"}}
    with pytest.raises(ValueError, match="llm_judge"):
        graders_to_dspy_metrics([judge])
    assert list(graders_to_dspy_metrics([_GSM8K_GRADER, judge], skip_unsupported=True)) == ["gr_exact_match"]
