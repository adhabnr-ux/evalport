from openeval.validate import validate_suite

from mlflow_openeval_adapter import to_openeval, from_openeval


class FakeDataFrame:
    """Minimal pandas.DataFrame stand-in exposing just .to_dict(orient="records"),
    matching what mlflow's EvaluationResult.tables["eval_results_table"] returns
    for our purposes."""

    def __init__(self, rows):
        self._rows = rows

    def to_dict(self, orient="records"):
        assert orient == "records"
        return list(self._rows)


class FakeEvaluationResult:
    """Stand-in for mlflow.evaluate()'s EvaluationResult."""

    def __init__(self, rows, metrics=None):
        self.tables = {"eval_results_table": FakeDataFrame(rows)}
        self.metrics = metrics or {}


def _sample_rows():
    return [
        {
            "inputs": "What is the capital of France?",
            "outputs": "Paris is the capital of France.",
            "targets": "Paris",
            "exact_match/v1/score": 0.0,
            "toxicity/v1/score": 0.02,
        },
        {
            "inputs": "What is the capital of Germany?",
            "outputs": "Berlin",
            "targets": "Berlin",
            "exact_match/v1/score": 1.0,
            "toxicity/v1/score": 0.01,
        },
    ]


def test_to_openeval_from_evaluation_result():
    result = FakeEvaluationResult(_sample_rows(), metrics={"exact_match/v1/mean": 0.5, "toxicity/v1/mean": 0.015})
    suite = to_openeval(result, run_id="exp42")

    assert suite["id"] == "mlflow_eval_exp42"
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["input"] == "What is the capital of France?"
    assert tc1["expected_output"] == "Paris"
    assert tc1["metadata"]["mlflow_actual_output"] == "Paris is the capital of France."
    assert tc1["metadata"]["mlflow_scores"] == {"exact_match/v1/score": 0.0, "toxicity/v1/score": 0.02}
    assert set(tc1["graders"]) == {"gr_exact_match_v1", "gr_toxicity_v1"}

    grader_ids = {g["id"] for g in suite["graders"]}
    assert grader_ids == {"gr_exact_match_v1", "gr_toxicity_v1"}
    assert suite["metadata"]["mlflow_metrics"] == {"exact_match/v1/mean": 0.5, "toxicity/v1/mean": 0.015}


def test_to_openeval_from_plain_list():
    rows = [{"inputs": "hi", "outputs": "hello", "targets": "hello", "custom_metric/score": 0.9}]
    suite = to_openeval(rows, run_id="run_x")
    assert suite["id"] == "mlflow_eval_run_x"
    assert suite["test_cases"][0]["expected_output"] == "hello"
    assert suite["test_cases"][0]["graders"] == ["gr_custom_metric"]


def test_to_openeval_validates_against_evalport_spec():
    result = FakeEvaluationResult(_sample_rows())
    suite = to_openeval(result)
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_no_metrics_still_produces_valid_suite():
    rows = [{"inputs": "hi", "outputs": "hello"}]
    suite = to_openeval(rows, run_id="no_metrics")
    assert suite["graders"] == [
        {"id": "gr_mlflow_score", "type": "custom", "params": {"handler": "mlflow:score"}}
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
    rows = from_openeval(suite)
    assert rows == [{"inputs": "hi", "targets": "hello"}]


def test_empty_results_still_produces_well_formed_suite():
    result = FakeEvaluationResult([])
    suite = to_openeval(result, run_id="empty")
    assert suite["test_cases"] == []
    assert suite["version"] == "1.0.0"
    validation = validate_suite(suite)
    assert not validation.valid
    assert any(e["code"] == "MIN_ITEMS" for e in validation.errors)
