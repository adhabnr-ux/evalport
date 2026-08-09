from openeval.validate import validate_suite

from langsmith_openeval_adapter import to_openeval, from_openeval


class FakeRun:
    """Stand-in for langsmith.schemas.Run (attribute-based)."""

    def __init__(self, id, inputs=None, outputs=None, reference_output=None):
        self.id = id
        self.inputs = inputs or {}
        self.outputs = outputs
        self.reference_output = reference_output


class FakeFeedback:
    """Stand-in for langsmith.schemas.Feedback (attribute-based)."""

    def __init__(self, run_id, key, score, comment=None):
        self.run_id = run_id
        self.key = key
        self.score = score
        self.comment = comment


def test_to_openeval_from_objects_with_separate_feedback():
    runs = [
        FakeRun("r1", inputs={"question": "What is 2+2?"}, outputs={"answer": "4"}, reference_output={"answer": "4"}),
        FakeRun("r2", inputs={"question": "Capital of Japan?"}, outputs={"answer": "Tokyo"}),
    ]
    feedback = [
        FakeFeedback("r1", "correctness", 1.0),
        FakeFeedback("r1", "helpfulness", 0.9),
        FakeFeedback("r2", "correctness", 1.0),
    ]
    suite = to_openeval(runs, feedback=feedback, run_id="exp1")

    assert suite["id"] == "langsmith_eval_exp1"
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["id"] == "r1"
    assert tc1["input"] == "What is 2+2?"
    assert tc1["expected_output"] == "4"
    assert tc1["metadata"]["langsmith_feedback"] == {"correctness": 1.0, "helpfulness": 0.9}
    assert tc1["metadata"]["langsmith_actual_output"] == "4"
    assert set(tc1["graders"]) == {"gr_correctness", "gr_helpfulness"}

    grader_ids = {g["id"] for g in suite["graders"]}
    assert grader_ids == {"gr_correctness", "gr_helpfulness"}
    assert suite["metadata"]["langsmith_feedback_keys"] == ["correctness", "helpfulness"]


def test_to_openeval_from_dicts_with_attached_feedback():
    runs = [
        {
            "id": "r9",
            "inputs": {"question": "hi"},
            "outputs": {"answer": "hello"},
            "feedback": [{"key": "tone", "score": 0.8}],
        }
    ]
    suite = to_openeval(runs, run_id="exp2")
    tc = suite["test_cases"][0]
    assert tc["metadata"]["langsmith_feedback"] == {"tone": 0.8}
    assert tc["graders"] == ["gr_tone"]


def test_multi_field_inputs_serialize_to_json():
    runs = [FakeRun("r1", inputs={"a": "x", "b": "y"}, outputs={"c": "z", "d": "w"})]
    suite = to_openeval(runs, run_id="exp3")
    tc = suite["test_cases"][0]
    assert tc["input"] in ('{"a": "x", "b": "y"}',)
    assert tc["metadata"]["langsmith_actual_output"] in ('{"c": "z", "d": "w"}',)


def test_to_openeval_validates_against_evalport_spec():
    runs = [FakeRun("r1", inputs={"question": "q"}, outputs={"answer": "a"}, reference_output={"answer": "a"})]
    feedback = [FakeFeedback("r1", "correctness", 1.0)]
    suite = to_openeval(runs, feedback=feedback)
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_no_feedback_still_produces_valid_suite():
    runs = [FakeRun("r1", inputs={"question": "q"}, outputs={"answer": "a"})]
    suite = to_openeval(runs)
    assert suite["graders"] == [
        {"id": "gr_langsmith_feedback", "type": "custom", "params": {"handler": "langsmith:feedback"}}
    ]
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_from_openeval_round_trip():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "custom"}],
        "test_cases": [
            {"id": "tc1", "input": "hi", "expected_output": "hello", "graders": ["g1"]},
        ],
    }
    examples = from_openeval(suite)
    assert examples == [
        {"inputs": {"input": "hi"}, "outputs": {"output": "hello"}},
    ]


def test_empty_runs_still_produces_well_formed_suite():
    suite = to_openeval([], run_id="empty")
    assert suite["test_cases"] == []
    assert suite["version"] == "1.0.0"
    validation = validate_suite(suite)
    assert not validation.valid
    assert any(e["code"] == "MIN_ITEMS" for e in validation.errors)
