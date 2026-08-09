from openeval.validate import validate_suite

from ragas_openeval_adapter import to_openeval, from_openeval


class FakeDataFrame:
    """Minimal pandas.DataFrame stand-in exposing just .iterrows(), matching
    what Ragas's EvaluationResult.to_pandas() returns for our purposes."""

    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, FakeSeries(row)


class FakeSeries(dict):
    """dict subclass so `in` / `[]` work like a pandas Series, plus .to_dict()."""

    def to_dict(self):
        return dict(self)


class FakeEvaluationResult:
    """Stand-in for ragas.evaluate()'s EvaluationResult."""

    def __init__(self, rows, run_id="run1"):
        self._rows = rows
        self.run_id = run_id

    def to_pandas(self):
        return FakeDataFrame(self._rows)


def _sample_rows():
    return [
        {
            "user_input": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "reference": "Paris",
            "retrieved_contexts": ["Paris is the capital and most populous city of France."],
            "faithfulness": 1.0,
            "answer_relevancy": 0.95,
            "context_precision": 0.88,
        },
        {
            "user_input": "What is the capital of Germany?",
            "response": "Berlin.",
            "reference": "Berlin",
            "retrieved_contexts": ["Berlin is the capital of Germany."],
            "faithfulness": 1.0,
            "answer_relevancy": 0.99,
            "context_precision": 0.91,
        },
    ]


def test_to_openeval_from_evaluation_result():
    result = FakeEvaluationResult(_sample_rows(), run_id="exp42")
    suite = to_openeval(result)

    assert suite["id"] == "ragas_eval_exp42"
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["input"] == "What is the capital of France?"
    assert tc1["expected_output"] == "Paris"
    assert tc1["context"] == ["Paris is the capital and most populous city of France."]
    assert tc1["metadata"]["ragas_scores"]["faithfulness"] == 1.0
    assert tc1["metadata"]["ragas_actual_output"] == "Paris is the capital of France."
    assert set(tc1["graders"]) == {"gr_answer_relevancy", "gr_context_precision", "gr_faithfulness"}

    grader_ids = {g["id"] for g in suite["graders"]}
    assert grader_ids == {"gr_answer_relevancy", "gr_context_precision", "gr_faithfulness"}
    assert suite["metadata"]["ragas_metrics"] == ["answer_relevancy", "context_precision", "faithfulness"]


def test_to_openeval_from_plain_list():
    rows = [{"user_input": "hi", "response": "hello", "reference": "hello", "faithfulness": 0.5}]
    suite = to_openeval(rows, run_id="run_x")
    assert suite["id"] == "ragas_eval_run_x"
    assert suite["test_cases"][0]["expected_output"] == "hello"


def test_to_openeval_validates_against_evalport_spec():
    result = FakeEvaluationResult(_sample_rows())
    suite = to_openeval(result)
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_no_metrics_still_produces_valid_suite():
    rows = [{"user_input": "hi", "response": "hello"}]
    suite = to_openeval(rows, run_id="no_metrics")
    assert suite["graders"] == [
        {"id": "gr_ragas_score", "type": "custom", "params": {"handler": "ragas:score"}}
    ]
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_from_openeval_round_trip():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "custom"}],
        "test_cases": [
            {
                "id": "tc1",
                "input": "hi",
                "expected_output": "hello",
                "context": ["ctx1"],
                "graders": ["g1"],
            },
        ],
    }
    samples = from_openeval(suite)
    assert samples == [
        {"user_input": "hi", "reference": "hello", "retrieved_contexts": ["ctx1"]},
    ]


def test_empty_results_still_produces_well_formed_suite():
    # The EvalPort spec requires at least one test case (MIN_ITEMS on
    # test_cases), so an empty Ragas result won't validate as a suite —
    # but the shape returned should still be well-formed rather than
    # raising, so callers can detect and handle "nothing to convert" here.
    result = FakeEvaluationResult([], run_id="empty")
    suite = to_openeval(result)
    assert suite["test_cases"] == []
    assert suite["version"] == "1.0.0"
    validation = validate_suite(suite)
    assert not validation.valid
    assert any(e["code"] == "MIN_ITEMS" for e in validation.errors)
