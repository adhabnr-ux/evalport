import pytest
from parea.schemas import TestCase, TestCaseCollection, ExperimentStatsSchema
from parea.schemas.models import TraceStatsSchema, EvaluationResultSchema, TraceLog
from openeval.validate import validate_suite, validate_result_set
from parea_openeval_adapter import to_openeval, from_openeval, experiment_to_openeval


def test_test_case_collection_conversion():
    # Create Parea test cases using real classes (id must be int)
    tc1 = TestCase(id=1, test_case_collection_id=42, inputs={"question": "What is 2+2?"}, target="4", tags=["math", "easy"])
    tc2 = TestCase(id=2, test_case_collection_id=42, inputs={"query": "Write hello world in python"}, target="print('Hello World')", tags=["code"])
    tc3 = TestCase(id=3, test_case_collection_id=42, inputs={"custom_input_key": "some value"}, target=None, tags=[])

    collection = TestCaseCollection(
        id=42,
        name="My Test Collection",
        created_at="2026-08-23T00:00:00Z",
        last_updated_at="2026-08-23T00:00:00Z",
        column_names=["inputs", "target", "tags"],
        test_cases={1: tc1, 2: tc2, 3: tc3}  # Parea expects a dict
    )

    # Convert to EvalPort suite
    suite = to_openeval(collection)

    # Validate against EvalPort SDK validator
    validation = validate_suite(suite)
    assert validation.valid, f"Validation failed: {validation.errors}"

    # Verify fields
    assert suite["id"] == "parea_suite_42"
    assert suite["name"] == "Parea collection: My Test Collection"
    assert len(suite["test_cases"]) == 3

    assert suite["test_cases"][0]["id"] == "1"
    assert suite["test_cases"][0]["input"] == "What is 2+2?"
    assert suite["test_cases"][0]["expected_output"] == "4"
    assert suite["test_cases"][0]["tags"] == ["math", "easy"]

    # Check input fallback logic
    assert suite["test_cases"][2]["id"] == "3"
    assert suite["test_cases"][2]["input"] == "some value"

    # Convert back to Parea-compatible dicts
    rebuilt = from_openeval(suite)
    assert len(rebuilt) == 3
    assert rebuilt[0]["id"] == "1"
    assert rebuilt[0]["inputs"] == {"question": "What is 2+2?"}
    assert rebuilt[0]["target"] == "4"
    assert rebuilt[0]["tags"] == ["math", "easy"]
    assert rebuilt[0]["test_case_collection_id"] == 42


def test_experiment_stats_conversion():
    # Create evaluation result schema (scores)
    score1 = EvaluationResultSchema(name="exact_match", score=1.0, reason="Perfect match")
    score2 = EvaluationResultSchema(name="semantic_similarity", score=0.4, reason="Somewhat similar")

    # Create trace stats schema
    stat1 = TraceStatsSchema(
        trace_id="trace_1",
        latency=0.5,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cost=0.002,
        scores=[score1]
    )
    stat2 = TraceStatsSchema(
        trace_id="trace_2",
        latency=0.8,
        input_tokens=12,
        output_tokens=8,
        total_tokens=20,
        cost=0.003,
        scores=[score2]
    )

    experiment_stats = ExperimentStatsSchema(parent_trace_stats=[stat1, stat2])

    # Convert to EvalPort ResultSet
    result_set = experiment_to_openeval(
        experiment_stats,
        suite_id="my_suite",
        run_id="my_run",
        started_at="2026-08-23T00:00:00Z"
    )

    # Validate against EvalPort SDK validator
    validation = validate_result_set(result_set)
    assert validation.valid, f"Validation failed: {validation.errors}"

    # Verify metadata and summary
    assert result_set["suite_id"] == "my_suite"
    assert result_set["run_id"] == "my_run"
    assert result_set["started_at"] == "2026-08-23T00:00:00Z"
    assert len(result_set["results"]) == 2

    # Check first result (score=1.0 -> passed=True)
    res1 = result_set["results"][0]
    assert res1["test_case_id"] == "trace_1"
    assert res1["passed"] is True
    assert len(res1["grader_results"]) == 1
    assert res1["grader_results"][0]["grader_id"] == "exact_match"
    assert res1["grader_results"][0]["score"] == 1.0
    assert res1["grader_results"][0]["passed"] is True
    assert res1["grader_results"][0]["reason"] == "Perfect match"
    assert res1["metadata"]["latency"] == 0.5
    assert res1["metadata"]["cost"] == 0.002

    # Check second result (score=0.4 -> passed=False)
    res2 = result_set["results"][1]
    assert res2["test_case_id"] == "trace_2"
    assert res2["passed"] is False
    assert res2["grader_results"][0]["passed"] is False


def test_trace_log_conversion_and_merging():
    # Create trace logs with actual outputs (parent_trace_id, root_trace_id, and start_timestamp are required to be strings by Parea validators)
    log1 = TraceLog(
        trace_id="trace_1",
        parent_trace_id="trace_1",
        root_trace_id="trace_1",
        start_timestamp="2026-08-23T00:00:00Z",
        inputs={"question": "What is 2+2?"},
        output="4",
        target="4",
        latency=0.5,
        end_timestamp="2026-08-23T00:00:01Z",
        scores=[EvaluationResultSchema(name="exact_match", score=1.0)]
    )

    log2 = TraceLog(
        trace_id="trace_2",
        parent_trace_id="trace_2",
        root_trace_id="trace_2",
        start_timestamp="2026-08-23T00:00:02Z",
        inputs={"question": "What is 3+3?"},
        output="5",
        target="6",
        latency=0.6,
        end_timestamp="2026-08-23T00:00:03Z",
        scores=[EvaluationResultSchema(name="exact_match", score=0.0)]
    )

    # Convert using trace logs directly
    result_set = experiment_to_openeval(
        [log1, log2],
        suite_id="my_log_suite",
        run_id="my_log_run"
    )

    # Validate against EvalPort SDK validator
    validation = validate_result_set(result_set)
    assert validation.valid, f"Validation failed: {validation.errors}"

    # Verify actual output mapping
    assert result_set["results"][0]["actual_output"] == "4"
    assert result_set["results"][1]["actual_output"] == "5"

    # Verify start and end times resolved from logs
    assert result_set["started_at"] == "2026-08-23T00:00:00Z"
    assert result_set["completed_at"] == "2026-08-23T00:00:03Z"
