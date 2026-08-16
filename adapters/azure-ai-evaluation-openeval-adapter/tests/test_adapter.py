import json
import tempfile

import pytest
from azure.ai.evaluation import (
    BleuScoreEvaluator,
    F1ScoreEvaluator,
    RougeScoreEvaluator,
    RougeType,
    evaluate,
)
from openeval.validate import validate_result_set, validate_suite

from azure_ai_evaluation_openeval_adapter import (
    evaluation_result_to_openeval,
    from_openeval,
    to_openeval,
)

ROWS = [
    {
        "id": "row-0",
        "query": "What is the capital of Japan?",
        "response": "Tokyo is the capital of Japan.",
        "ground_truth": "The capital of Japan is Tokyo.",
    },
    {
        "id": "row-1",
        "query": "What is 2+2?",
        "response": "4",
        "ground_truth": "4",
    },
]


def _rows_to_jsonl(rows) -> str:
    """The real installed azure-ai-evaluation requires evaluate()'s `data`
    to be a path/PathLike, not a raw list of dicts (confirmed directly
    against the installed package -- contradicts what some docs examples
    imply). Every call to the real evaluate() in these tests goes through
    this helper for that reason; the adapter's own to_openeval()/
    from_openeval() still accept plain lists for convenience."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for row in rows:
        f.write(json.dumps(row) + "\n")
    f.close()
    return f.name


def _run_real_evaluate(evaluators, rows=None):
    return evaluate(data=_rows_to_jsonl(rows if rows is not None else ROWS), evaluators=evaluators)


# ---------------------------------------------------------------------------
# to_openeval()
# ---------------------------------------------------------------------------


def test_to_openeval_produces_valid_suite_with_f1():
    suite = to_openeval(ROWS, evaluators={"f1": F1ScoreEvaluator()}, suite_id="test-suite")
    validate_suite(suite)  # raises on invalid -- real validator, not a mock

    assert suite["id"] == "test-suite"
    assert len(suite["test_cases"]) == 2
    assert suite["graders"][0]["id"] == "f1"
    assert suite["graders"][0]["type"] == "custom"
    assert suite["graders"][0]["params"]["handler"] == "F1ScoreEvaluator"
    assert "no model/network required" in suite["graders"][0]["description"]


def test_to_openeval_maps_query_and_ground_truth():
    suite = to_openeval(ROWS, evaluators={"f1": F1ScoreEvaluator()})
    tc0 = suite["test_cases"][0]
    assert tc0["input"] == "What is the capital of Japan?"
    assert tc0["expected_output"] == "The capital of Japan is Tokyo."
    assert tc0["graders"] == ["f1"]
    # Full raw row preserved losslessly for from_openeval() round trip.
    assert tc0["metadata"]["azure_ai_evaluation"]["row"] == ROWS[0]


def test_to_openeval_multiple_evaluators_all_referenced():
    suite = to_openeval(
        ROWS,
        evaluators={"f1": F1ScoreEvaluator(), "bleu": BleuScoreEvaluator()},
    )
    validate_suite(suite)
    assert set(suite["test_cases"][0]["graders"]) == {"f1", "bleu"}
    grader_ids = {g["id"] for g in suite["graders"]}
    assert grader_ids == {"f1", "bleu"}


def test_to_openeval_rouge_captures_rouge_type():
    suite = to_openeval(ROWS, evaluators={"rouge": RougeScoreEvaluator(rouge_type=RougeType.ROUGE_L)})
    validate_suite(suite)
    grader = suite["graders"][0]
    assert grader["params"]["handler"] == "RougeScoreEvaluator"
    assert grader["params"]["rouge_type"] == str(RougeType.ROUGE_L)


def test_to_openeval_custom_function_evaluator_maps_by_name():
    def response_length(*, response: str, **kwargs):
        return {"score": len(response)}

    suite = to_openeval(ROWS, evaluators={"length": response_length})
    validate_suite(suite)
    grader = suite["graders"][0]
    assert grader["type"] == "custom"
    assert grader["params"]["handler"] == "response_length"
    assert "Custom function-based evaluator" in grader["description"]


def test_to_openeval_custom_class_evaluator_maps_by_class_name():
    class BlocklistEvaluator:
        def __init__(self, blocklist):
            self._blocklist = blocklist

        def __call__(self, *, response: str, **kwargs):
            return {"score": any(w in response for w in self._blocklist)}

    suite = to_openeval(ROWS, evaluators={"blocklist": BlocklistEvaluator(blocklist=["bad"])})
    validate_suite(suite)
    assert suite["graders"][0]["params"]["handler"] == "BlocklistEvaluator"


def test_to_openeval_accepts_jsonl_path(tmp_path):
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        for row in ROWS:
            f.write(json.dumps(row) + "\n")

    suite = to_openeval(str(jsonl_path), evaluators={"f1": F1ScoreEvaluator()})
    validate_suite(suite)
    assert len(suite["test_cases"]) == 2


def test_to_openeval_evaluator_config_preserved_in_metadata():
    evaluator_config = {"f1": {"column_mapping": {"response": "${data.response}"}}}
    suite = to_openeval(ROWS, evaluators={"f1": F1ScoreEvaluator()}, evaluator_config=evaluator_config)
    validate_suite(suite)
    assert suite["metadata"]["azure_ai_evaluation"]["evaluator_config"] == evaluator_config


def test_to_openeval_context_column_becomes_context_list():
    rows = [{"query": "q", "response": "r", "ground_truth": "g", "context": "some doc"}]
    suite = to_openeval(rows, evaluators={"f1": F1ScoreEvaluator()})
    validate_suite(suite)
    assert suite["test_cases"][0]["context"] == ["some doc"]


# ---------------------------------------------------------------------------
# from_openeval()
# ---------------------------------------------------------------------------


def test_from_openeval_round_trip_restores_original_row_exactly():
    suite = to_openeval(ROWS, evaluators={"f1": F1ScoreEvaluator()})
    restored = from_openeval(suite)
    assert restored == ROWS


def test_from_openeval_can_feed_real_evaluate_call():
    suite = to_openeval(ROWS, evaluators={"f1": F1ScoreEvaluator()})
    restored = from_openeval(suite)
    # Prove the round-tripped rows are genuinely usable by the real SDK, not
    # just structurally similar.
    result = evaluate(data=_rows_to_jsonl(restored), evaluators={"f1": F1ScoreEvaluator()})
    assert result["metrics"]["f1.f1_score"] == 1.0


def test_from_openeval_heuristic_fallback_without_saved_row():
    suite = {
        "version": "1.0.0-rc.2",
        "id": "hand-authored",
        "test_cases": [
            {
                "id": "tc-1",
                "input": "What is the capital of France?",
                "expected_output": "Paris",
                "graders": ["exact"],
            }
        ],
    }
    rows = from_openeval(suite)
    assert rows == [{"query": "What is the capital of France?", "ground_truth": "Paris", "id": "tc-1"}]


def test_from_openeval_multiturn_input_list_joined():
    suite = {
        "version": "1.0.0-rc.2",
        "id": "s",
        "test_cases": [{"id": "tc-1", "input": ["turn one", "turn two"], "graders": ["g"]}],
    }
    rows = from_openeval(suite)
    assert rows[0]["query"] == "turn one turn two"


# ---------------------------------------------------------------------------
# evaluation_result_to_openeval()
# ---------------------------------------------------------------------------


def test_evaluation_result_to_openeval_valid_result_set_single_evaluator():
    result = _run_real_evaluate({"f1": F1ScoreEvaluator()})
    result_set = evaluation_result_to_openeval(result, suite_id="test-suite", run_id="run-1")
    validate_result_set(result_set)  # real validator

    assert result_set["suite_id"] == "test-suite"
    assert result_set["run_id"] == "run-1"
    assert len(result_set["results"]) == 2


def test_evaluation_result_to_openeval_correct_scores_and_pass():
    result = _run_real_evaluate({"f1": F1ScoreEvaluator()})
    result_set = evaluation_result_to_openeval(result)
    validate_result_set(result_set)

    row0 = result_set["results"][0]
    assert row0["grader_results"][0]["grader_id"] == "f1"
    assert row0["grader_results"][0]["score"] == 1.0
    assert row0["grader_results"][0]["passed"] is True
    assert row0["passed"] is True


def test_evaluation_result_to_openeval_detects_failure():
    rows = [{"query": "q", "response": "completely unrelated text", "ground_truth": "The capital of Japan is Tokyo."}]
    result = evaluate(data=_rows_to_jsonl(rows), evaluators={"f1": F1ScoreEvaluator()})
    result_set = evaluation_result_to_openeval(result)
    validate_result_set(result_set)

    gr = result_set["results"][0]["grader_results"][0]
    assert gr["score"] < 0.5
    assert gr["passed"] is False
    assert result_set["results"][0]["passed"] is False


def test_evaluation_result_to_openeval_multiple_evaluators_per_row():
    # F1ScoreEvaluator + a plain custom evaluator, not BleuScoreEvaluator --
    # BLEU/GLEU/METEOR/ROUGE need NLTK's punkt_tab/wordnet corpora, which
    # this sandbox's SSRF-protection proxy blocks downloading (unrelated to
    # this adapter; test_to_openeval_multiple_evaluators_all_referenced
    # above already covers BLEU showing up correctly as a *grader
    # definition*, which needs no network -- just not an actual scored run).
    def length_at_least_one(*, response: str, **kwargs):
        return {"score": len(response) > 0}

    result = _run_real_evaluate({"f1": F1ScoreEvaluator(), "nonempty": length_at_least_one})
    result_set = evaluation_result_to_openeval(result)
    validate_result_set(result_set)

    grader_ids = {gr["grader_id"] for gr in result_set["results"][0]["grader_results"]}
    assert grader_ids == {"f1", "nonempty"}


def test_evaluation_result_to_openeval_summary_matches_real_counts():
    rows = [
        {"query": "q1", "response": "The capital of Japan is Tokyo.", "ground_truth": "The capital of Japan is Tokyo."},
        {"query": "q2", "response": "totally wrong", "ground_truth": "The capital of Japan is Tokyo."},
    ]
    result = evaluate(data=_rows_to_jsonl(rows), evaluators={"f1": F1ScoreEvaluator()})
    result_set = evaluation_result_to_openeval(result)
    validate_result_set(result_set)

    assert result_set["summary"]["total"] == 2
    assert result_set["summary"]["passed"] == sum(1 for r in result_set["results"] if r["passed"])
    assert result_set["summary"]["failed"] == 2 - result_set["summary"]["passed"]


def test_evaluation_result_to_openeval_preserves_raw_metrics_in_metadata():
    result = _run_real_evaluate({"f1": F1ScoreEvaluator()})
    result_set = evaluation_result_to_openeval(result)
    assert result_set["metadata"]["azure_ai_evaluation"]["metrics"] == result["metrics"]


def test_evaluation_result_to_openeval_actual_output_from_response_column():
    result = _run_real_evaluate({"f1": F1ScoreEvaluator()})
    result_set = evaluation_result_to_openeval(result)
    assert result_set["results"][0]["actual_output"] == ROWS[0]["response"]


# ---------------------------------------------------------------------------
# Full round trip
# ---------------------------------------------------------------------------


def test_full_round_trip_suite_to_rows_to_real_evaluate_to_resultset():
    suite = to_openeval(ROWS, evaluators={"f1": F1ScoreEvaluator()}, suite_id="round-trip-suite")
    validate_suite(suite)

    rows = from_openeval(suite)
    result = evaluate(data=_rows_to_jsonl(rows), evaluators={"f1": F1ScoreEvaluator()})

    result_set = evaluation_result_to_openeval(result, suite_id=suite["id"])
    validate_result_set(result_set)

    assert result_set["suite_id"] == "round-trip-suite"
    assert all(r["passed"] for r in result_set["results"])
