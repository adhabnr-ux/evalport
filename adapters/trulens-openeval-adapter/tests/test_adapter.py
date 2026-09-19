from datetime import datetime, timezone

import pytest

from openeval.validate import validate_result_set

from trulens_openeval_adapter import to_openeval, from_openeval


class FakeRecord:
    """Stand-in for trulens.core.schema.record.Record (attribute-based).

    Field names (record_id, app_id, main_input, main_output, main_error, ts)
    match the real class -- see the cross-check test at the bottom of this
    file, which constructs the actual TruLens class when it's installed.
    """

    def __init__(self, record_id, app_id="app1", main_input=None, main_output=None, main_error=None, ts=None, tags=None):
        self.record_id = record_id
        self.app_id = app_id
        self.main_input = main_input
        self.main_output = main_output
        self.main_error = main_error
        self.ts = ts or datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
        self.tags = tags


class FakeFeedbackResult:
    """Stand-in for trulens.core.schema.feedback.FeedbackResult."""

    def __init__(self, record_id, name, result=None, error=None, status="done"):
        self.record_id = record_id
        self.name = name
        self.result = result
        self.error = error
        self.status = status


def test_to_openeval_basic_record_with_feedback():
    records = [FakeRecord("r1", main_input="What is 2+2?", main_output="4")]
    feedback = [FakeFeedbackResult("r1", "correctness", result=1.0)]

    rs = to_openeval(records, feedback)

    assert rs["results"][0]["test_case_id"] == "r1"
    assert rs["results"][0]["actual_output"] == "4"
    assert rs["results"][0]["passed"] is True
    assert rs["results"][0]["grader_results"][0] == {
        "grader_id": "correctness",
        "type": "trulens_feedback",
        "score": 1.0,
        "passed": True,
        "metadata": {"status": "done"},
    }
    assert rs["results"][0]["metadata"]["main_input"] == "What is 2+2?"


def test_to_openeval_below_threshold_fails():
    records = [FakeRecord("r1")]
    feedback = [FakeFeedbackResult("r1", "groundedness", result=0.2)]
    rs = to_openeval(records, feedback)
    assert rs["results"][0]["grader_results"][0]["passed"] is False
    assert rs["results"][0]["passed"] is False


def test_to_openeval_custom_pass_threshold():
    records = [FakeRecord("r1")]
    feedback = [FakeFeedbackResult("r1", "groundedness", result=0.4)]
    rs = to_openeval(records, feedback, pass_threshold=0.3)
    assert rs["results"][0]["grader_results"][0]["passed"] is True


def test_to_openeval_multiple_feedback_results_one_record():
    records = [FakeRecord("r1")]
    feedback = [
        FakeFeedbackResult("r1", "groundedness", result=0.9),
        FakeFeedbackResult("r1", "relevance", result=0.4),
    ]
    rs = to_openeval(records, feedback)
    names = {g["grader_id"] for g in rs["results"][0]["grader_results"]}
    assert names == {"groundedness", "relevance"}
    # One feedback below threshold -> whole result fails.
    assert rs["results"][0]["passed"] is False


def test_to_openeval_main_error_becomes_runner_error():
    records = [FakeRecord("r1", main_error="RuntimeError: tool call failed")]
    feedback = [FakeFeedbackResult("r1", "groundedness", result=0.9)]
    rs = to_openeval(records, feedback)
    result = rs["results"][0]
    assert result["error"] == {"type": "runner_error", "message": "RuntimeError: tool call failed"}
    assert result["passed"] is False
    # Feedback results are not scored for an errored record.
    assert result["grader_results"] == []


def test_to_openeval_feedback_error_produces_failed_grader_result():
    records = [FakeRecord("r1")]
    feedback = [FakeFeedbackResult("r1", "groundedness", result=None, error="provider timeout")]
    rs = to_openeval(records, feedback)
    gr = rs["results"][0]["grader_results"][0]
    assert gr["score"] is None
    assert gr["passed"] is False
    assert "provider timeout" in gr["reason"]


def test_to_openeval_out_of_range_score_preserved_not_clipped():
    records = [FakeRecord("r1")]
    feedback = [FakeFeedbackResult("r1", "custom_metric", result=1.7)]
    rs = to_openeval(records, feedback)
    gr = rs["results"][0]["grader_results"][0]
    assert gr["score"] is None
    assert gr["metadata"]["raw_result"] == 1.7
    assert gr["passed"] is False


def test_to_openeval_non_string_output_is_json_serialized():
    records = [FakeRecord("r1", main_output={"answer": 4, "confidence": 0.9})]
    rs = to_openeval(records, [])
    assert rs["results"][0]["actual_output"] == '{"answer": 4, "confidence": 0.9}'


def test_to_openeval_record_with_no_feedback_still_valid_and_passes():
    records = [FakeRecord("r1")]
    rs = to_openeval(records, [])
    assert rs["results"][0]["grader_results"] == []
    assert rs["results"][0]["passed"] is True


def test_to_openeval_accepts_dicts_not_just_objects():
    records = [{"record_id": "r1", "app_id": "app1", "main_output": "hi", "ts": "2026-09-19T12:00:00+00:00"}]
    feedback = [{"record_id": "r1", "name": "relevance", "result": 0.8}]
    rs = to_openeval(records, feedback)
    assert rs["results"][0]["actual_output"] == "hi"
    assert rs["results"][0]["grader_results"][0]["score"] == 0.8


def test_to_openeval_validates_against_evalport_spec():
    records = [
        FakeRecord("r1", main_input="q1", main_output="a1"),
        FakeRecord("r2", main_error="boom"),
    ]
    feedback = [FakeFeedbackResult("r1", "correctness", result=0.95)]
    rs = to_openeval(records, feedback, suite_id="my_trulens_app", run_id="run_2026_09_19")
    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_to_openeval_empty_records_produces_empty_results_list():
    """An empty `records` iterable produces a structurally-sane dict with an
    empty `results` list -- but EvalPort's own schema requires `results` to
    be non-empty (`minItems: 1`), so this deliberately does NOT assert
    schema validity, unlike every other conversion test in this file. A
    caller with zero records has nothing to report and should not call
    `validate_result_set()` on the output; this just confirms the adapter
    fails predictably (a clear REQUIRED error on `$.results`) rather than
    raising or fabricating a placeholder result.
    """
    rs = to_openeval([], [])
    assert rs["results"] == []
    validation = validate_result_set(rs)
    assert not validation.valid
    assert any(e["path"] == "$.results" for e in validation.errors)


def test_from_openeval_produces_ground_truth_agreement_golden_set():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {"id": "tc1", "input": "who invented the lightbulb?", "expected_output": "Thomas Edison", "graders": ["g1"]},
            {"id": "tc2", "input": "no expected answer here", "graders": ["g1"]},
        ],
    }
    golden_set = from_openeval(suite)
    assert golden_set == [{"query": "who invented the lightbulb?", "expected_response": "Thomas Edison"}]


def test_round_trip_shapes_are_consistent():
    records = [FakeRecord("r1", main_input="2+2?", main_output="4")]
    feedback = [FakeFeedbackResult("r1", "correctness", result=1.0)]
    rs = to_openeval(records, feedback)

    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "r1", "input": "2+2?", "expected_output": rs["results"][0]["actual_output"], "graders": ["g1"]}],
    }
    golden_set = from_openeval(suite)
    assert golden_set[0]["expected_response"] == rs["results"][0]["actual_output"]


def test_field_names_match_real_trulens_classes():
    """Cross-check against TruLens's actual current classes, not just the
    FakeRecord/FakeFeedbackResult stand-ins above. Skipped unless the
    `test-full` extra (trulens-core) is installed -- the adapter itself never
    imports trulens, only this one verification test does.
    """
    trulens_core = pytest.importorskip("trulens.core")
    from trulens.core.schema.record import Record
    from trulens.core.schema.feedback import FeedbackResult, FeedbackResultStatus

    record = Record(
        record_id="r1",
        calls=[],  # trulens.core.schema.record.Record.__init__ passes `calls`
        # straight to its pydantic base with no None->[] coalescing, so the
        # field's own `List[RecordAppCall] = []` default only applies when
        # the constructor arg is omitted entirely at the *call site of
        # __init__ itself* -- passing nothing here still resolves to the
        # `calls: Optional[...] = None` keyword default and fails pydantic
        # validation. Passing [] explicitly is the correct real-world usage
        # (this is what `with tru_app as recording: ...` produces internally
        # too), not a workaround for anything wrong in this adapter.
        app_id="app1",
        main_input="2+2?",
        main_output="4",
        ts=datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc),
    )
    feedback_result = FeedbackResult(
        record_id="r1",
        name="correctness",
        result=1.0,
        status=FeedbackResultStatus.DONE,
    )

    rs = to_openeval([record], [feedback_result])
    assert rs["results"][0]["test_case_id"] == "r1"
    assert rs["results"][0]["actual_output"] == "4"
    assert rs["results"][0]["grader_results"][0]["score"] == 1.0
    validation = validate_result_set(rs)
    assert validation.valid, validation.errors
