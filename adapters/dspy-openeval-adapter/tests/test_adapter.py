"""Tests for dspy-openeval-adapter, run against the real `dspy` package and
the real `openeval.validate` validators -- no mocks."""
import dspy
import pytest
from openeval.validate import validate_result_set, validate_suite

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
    assert suite["version"] == "1.0.0"
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
