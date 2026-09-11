import pytest
from openeval.validate import validate_result_set, validate_suite

from unitxt_openeval_adapter import evaluation_to_openeval, from_openeval, to_openeval

# ---------------------------------------------------------------------------
# to_openeval / from_openeval: plain raw task_data rows (no unitxt import
# needed for these -- the adapter only deals in plain dicts, matching the
# shape you'd pass as `test_set=` to `unitxt.api.create_dataset`).
# ---------------------------------------------------------------------------


def test_to_openeval_qa_task_shape():
    rows = [
        {"question": "What is the capital of France?", "answers": ["Paris"]},
        {"question": "Who wrote Hamlet?", "answers": ["Shakespeare"]},
    ]
    suite = to_openeval(rows, suite_id="unitxt_qa")
    assert validate_suite(suite).valid
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["input"] == "What is the capital of France?"
    assert tc1["expected_output"] == "Paris"
    # Both raw fields were fully consumed into input/expected_output, so
    # metadata.unitxt is empty (and therefore omitted) rather than carrying
    # a redundant copy of "question"/"answers".
    assert tc1.get("metadata", {}).get("unitxt", {}) == {}


def test_to_openeval_multi_reference_answers_preserved_in_metadata():
    rows = [{"question": "Name a primary color.", "answers": ["red", "blue", "yellow"]}]
    suite = to_openeval(rows)
    tc = suite["test_cases"][0]
    # First reference used for EvalPort's single-string expected_output...
    assert tc["expected_output"] == "red"
    # ...but the full list of acceptable answers is never lost.
    assert tc["metadata"]["unitxt"]["expected_output_full"] == ["red", "blue", "yellow"]


def test_to_openeval_classification_task_shape():
    rows = [{"text": "This movie was great!", "class": "positive"}]
    suite = to_openeval(rows)
    tc = suite["test_cases"][0]
    assert tc["input"] == "This movie was great!"
    assert tc["expected_output"] == "positive"


def test_to_openeval_explicit_key_override():
    rows = [{"custom_in": "ignored by heuristic", "custom_out": "also ignored"}]
    suite = to_openeval(rows, input_key="custom_in", expected_output_key="custom_out")
    tc = suite["test_cases"][0]
    assert tc["input"] == "ignored by heuristic"
    assert tc["expected_output"] == "also ignored"


def test_to_openeval_no_matching_keys_preserves_full_payload():
    rows = [{"weird_field_1": "a", "weird_field_2": "b"}]
    suite = to_openeval(rows)
    tc = suite["test_cases"][0]
    assert "weird_field_1" in tc["input"]
    assert tc["metadata"]["unitxt"]["weird_field_1"] == "a"


def test_to_openeval_no_expected_output_gets_custom_placeholder_grader():
    rows = [{"question": "no answer key here"}]
    suite = to_openeval(rows)
    assert validate_suite(suite).valid
    assert suite["graders"][0]["type"] == "custom"


def test_to_openeval_exact_match_grader_type():
    rows = [{"question": "Q", "answers": ["A"]}]
    suite = to_openeval(rows, grader_type="exact_match")
    assert suite["graders"][0]["type"] == "exact_match"


def test_from_openeval_round_trip_single_reference():
    rows = [{"question": "What is the capital of France?", "answers": ["Paris"]}]
    suite = to_openeval(rows)
    restored = from_openeval(suite)
    assert restored[0]["input"] == "What is the capital of France?"
    assert restored[0]["expected_output"] == "Paris"


def test_from_openeval_restores_multi_reference_answers():
    rows = [{"question": "Name a primary color.", "answers": ["red", "blue", "yellow"]}]
    suite = to_openeval(rows)
    restored = from_openeval(suite)
    assert restored[0]["answers"] == ["red", "blue", "yellow"]


# ---------------------------------------------------------------------------
# evaluation_to_openeval: exercised against a REAL unitxt task run, end to
# end -- not a hand-typed fixture. Runs unitxt.api.create_dataset +
# unitxt.api.evaluate against a real installed `unitxt`, then feeds the real
# EvaluationResults through the adapter. Skipped (not failed) if `unitxt`
# isn't installed, since it's an optional extra.
# ---------------------------------------------------------------------------

unitxt = pytest.importorskip("unitxt", reason="unitxt is an optional test dependency; install via `.[unitxt]`")


@pytest.fixture(scope="module")
def real_unitxt_results():
    """Runs a real unitxt QA task evaluation and returns the real
    EvaluationResults it produced, plus the rows used to build the dataset.
    """
    from unitxt.api import create_dataset, evaluate

    rows = [
        {"question": "What is the capital of France?", "answers": ["Paris"]},
        {"question": "Who wrote Hamlet?", "answers": ["Shakespeare"]},
        {"question": "What is 2+2?", "answers": ["4"]},
    ]
    dataset = create_dataset(
        task="tasks.qa.open",
        test_set=rows,
        split="test",
        format="formats.chat_api",
        metrics=["metrics.accuracy"],
    )
    predictions = ["Paris", "It was written by Christopher Marlowe.", "4"]
    results = evaluate(predictions=predictions, data=dataset)
    return rows, results


def test_evaluation_to_openeval_from_real_unitxt_run(real_unitxt_results):
    rows, results = real_unitxt_results
    # Confirm the real shape this adapter is built against.
    assert "score" in results[0]
    assert "instance" in results[0]["score"]
    assert "accuracy" in results[0]["score"]["instance"]

    result_set = evaluation_to_openeval(results, suite_id="unitxt_qa", run_id="run-1")
    assert validate_result_set(result_set).valid
    assert result_set["summary"]["total"] == 3

    r0 = result_set["results"][0]
    assert r0["test_case_id"] == "tc_0"
    assert r0["actual_output"] == "Paris"
    assert r0["passed"] is True
    gr = next(g for g in r0["grader_results"] if g["grader_id"] == "accuracy")
    assert gr["score"] == 1.0

    r1 = result_set["results"][1]
    assert r1["passed"] is False
    gr1 = next(g for g in r1["grader_results"] if g["grader_id"] == "accuracy")
    assert gr1["score"] == 0.0

    # score_name is metadata about the instance dict, not itself a grader.
    grader_ids = {g["grader_id"] for r in result_set["results"] for g in r["grader_results"]}
    assert "score_name" not in grader_ids

    # Run-level global scores preserved once, not repeated per result.
    assert "global_scores" in result_set["metadata"]["unitxt"]
    assert result_set["metadata"]["unitxt"]["global_scores"]["num_of_instances"] == 3


def test_evaluation_to_openeval_test_case_ids_match_to_openeval_order(real_unitxt_results):
    rows, results = real_unitxt_results
    suite = to_openeval(rows, suite_id="unitxt_qa")
    result_set = evaluation_to_openeval(results, suite_id="unitxt_qa", run_id="run-2")
    suite_ids = [tc["id"] for tc in suite["test_cases"]]
    result_ids = [r["test_case_id"] for r in result_set["results"]]
    assert suite_ids == result_ids == ["tc_0", "tc_1", "tc_2"]


def test_evaluation_to_openeval_multi_metric_task_produces_multiple_grader_results():
    # Hand-shaped instance matching the real score["instance"] structure for
    # a task with multiple attached metrics (e.g. metrics.rouge), verified
    # against unitxt's real Rouge metric output keys (rouge1/rouge2/rougeL/rougeLsum).
    fake_results = [{
        "task_data": {"document": "...", "summary": "..."},
        "prediction": "a short summary",
        "score": {
            "instance": {"rouge1": 0.8, "rouge2": 0.5, "rougeL": 0.7, "rougeLsum": 0.7, "score_name": "rougeL", "score": 0.7},
            "global": {"rouge1": 0.8, "score_name": "rougeL", "score": 0.7, "num_of_instances": 1},
        },
    }]
    result_set = evaluation_to_openeval(fake_results, suite_id="s", run_id="r")
    assert validate_result_set(result_set).valid
    grader_ids = {g["grader_id"] for g in result_set["results"][0]["grader_results"]}
    assert grader_ids == {"rouge1", "rouge2", "rougeL", "rougeLsum"}
