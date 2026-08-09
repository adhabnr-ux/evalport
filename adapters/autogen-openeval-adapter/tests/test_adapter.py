from openeval.validate import validate_suite

from autogen_openeval_adapter import to_openeval, from_openeval


class FakeTask:
    """Stand-in for AutoGen's eval task/result class (attribute-based)."""

    def __init__(self, task_id, task_description, expected_output=None, expected_tools=None, metadata=None):
        self.task_id = task_id
        self.task_description = task_description
        self.expected_output = expected_output
        self.expected_tools = expected_tools
        self.metadata = metadata


class FakeEvalResult:
    def __init__(self, run_id, results):
        self.run_id = run_id
        self.results = results


def test_to_openeval_from_objects():
    result = FakeEvalResult(
        run_id="run1",
        results=[
            FakeTask("t1", "Book a flight to SF", expected_output="Flight booked", expected_tools=["book_flight"]),
            FakeTask("t2", "Summarize the document"),
        ],
    )
    suite = to_openeval(result)

    assert suite["id"] == "autogen_eval_run1"
    assert len(suite["test_cases"]) == 2
    assert suite["test_cases"][0]["expected_tools"] == ["book_flight"]
    assert suite["test_cases"][0]["expected_output"] == "Flight booked"
    assert "expected_output" not in suite["test_cases"][1]
    # No "expected_outputs" (plural) field per spec compliance.
    assert "expected_outputs" not in suite["test_cases"][0]


def test_to_openeval_from_dicts():
    result = {
        "run_id": "run2",
        "results": [{"task_id": "t1", "task_description": "hi", "expected_output": "hello"}],
    }
    suite = to_openeval(result)
    assert suite["test_cases"][0]["expected_output"] == "hello"
    assert suite["test_cases"][0]["input"] == "hi"


def test_to_openeval_validates_against_evalport_spec():
    result = FakeEvalResult(run_id="run3", results=[FakeTask("t1", "task", expected_output="out")])
    suite = to_openeval(result)
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_from_openeval_round_trip():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {"id": "tc1", "input": "hi", "expected_output": "hello", "graders": ["g1"]},
        ],
    }
    tasks = from_openeval(suite)
    assert tasks == [
        {
            "task_id": "tc1",
            "description": "hi",
            "expected_output": "hello",
            "expected_tools": [],
            "metadata": {},
        }
    ]


def test_llm_judge_grader_option():
    result = FakeEvalResult(run_id="run4", results=[])
    suite = to_openeval(result, grader_type="llm_judge")
    assert suite["graders"][0]["type"] == "llm_judge"
    assert "{output}" in suite["graders"][0]["params"]["prompt"]


def test_empty_results_still_valid_shape():
    result = FakeEvalResult(run_id="run5", results=[])
    suite = to_openeval(result)
    assert suite["test_cases"] == []
    assert suite["version"] == "1.0.0"
