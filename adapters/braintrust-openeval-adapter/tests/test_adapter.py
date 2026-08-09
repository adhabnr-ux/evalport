from openeval.validate import validate_suite

from braintrust_openeval_adapter import to_openeval, from_openeval


class FakeCase:
    """Stand-in for a Braintrust EvalCase result (attribute-based)."""

    def __init__(self, id, input, expected=None, output=None, scores=None):
        self.id = id
        self.input = input
        self.expected = expected
        self.output = output
        self.scores = scores or {}


class FakeEvalResultWithSummary:
    """Stand-in for braintrust.Eval()'s EvalResultWithSummary."""

    def __init__(self, results, experimentName="exp1"):
        self.results = results
        self.experimentName = experimentName


def test_to_openeval_from_objects():
    result = FakeEvalResultWithSummary(
        results=[
            FakeCase(
                "c1",
                input="What is 2+2?",
                expected="4",
                output="4",
                scores={"Factuality": 1.0, "ExactMatch": 1.0},
            ),
            FakeCase("c2", input="Summarize this", expected="a summary", output="a summary-ish thing", scores={"Factuality": 0.7}),
        ],
        experimentName="my-eval-run",
    )
    suite = to_openeval(result)

    assert suite["id"] == "braintrust_eval_my-eval-run"
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["id"] == "c1"
    assert tc1["input"] == "What is 2+2?"
    assert tc1["expected_output"] == "4"
    assert tc1["metadata"]["braintrust_scores"] == {"Factuality": 1.0, "ExactMatch": 1.0}
    assert tc1["metadata"]["braintrust_actual_output"] == "4"
    assert set(tc1["graders"]) == {"gr_ExactMatch", "gr_Factuality"}

    grader_ids = {g["id"] for g in suite["graders"]}
    assert grader_ids == {"gr_ExactMatch", "gr_Factuality"}
    assert suite["metadata"]["braintrust_scorers"] == ["ExactMatch", "Factuality"]


def test_to_openeval_from_dicts():
    result = {
        "results": [{"id": "c1", "input": "hi", "expected": "hello", "output": "hello", "scores": {"Match": 1.0}}],
        "id": "run2",
    }
    suite = to_openeval(result)
    assert suite["id"] == "braintrust_eval_run2"
    assert suite["test_cases"][0]["expected_output"] == "hello"


def test_to_openeval_from_plain_list():
    cases = [{"id": "c1", "input": "hi", "expected": "hello", "scores": {"Match": 1.0}}]
    suite = to_openeval(cases, run_id="run3")
    assert suite["id"] == "braintrust_eval_run3"


def test_to_openeval_validates_against_evalport_spec():
    result = FakeEvalResultWithSummary(
        results=[FakeCase("c1", input="hi", expected="hello", output="hello", scores={"Factuality": 0.9})]
    )
    suite = to_openeval(result)
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_no_scores_still_produces_valid_suite():
    result = FakeEvalResultWithSummary(results=[FakeCase("c1", input="hi")])
    suite = to_openeval(result)
    assert suite["graders"] == [
        {"id": "gr_braintrust_score", "type": "custom", "params": {"handler": "braintrust:score"}}
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
    cases = from_openeval(suite)
    assert cases == [{"input": "hi", "expected": "hello"}]


def test_empty_results_still_produces_well_formed_suite():
    result = FakeEvalResultWithSummary(results=[], experimentName="empty")
    suite = to_openeval(result)
    assert suite["test_cases"] == []
    assert suite["version"] == "1.0.0"
    validation = validate_suite(suite)
    assert not validation.valid
    assert any(e["code"] == "MIN_ITEMS" for e in validation.errors)
