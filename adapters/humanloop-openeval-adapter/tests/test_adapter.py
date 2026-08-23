import pytest
from humanloop.types import DatapointResponse, ChatMessage
from openeval.validate import validate_suite, validate_result_set
from humanloop_openeval_adapter import to_openeval, from_openeval, result_to_openeval


def test_datapoints_suite_conversion():
    # Chat message turn list
    messages = [
        ChatMessage(role="user", content="Tell me a joke."),
        ChatMessage(role="assistant", content="Why did the chicken cross the road? To get to the other side.")
    ]

    dp1 = DatapointResponse(
        id="dp_1",
        inputs={"question": "What is 2+2?"},
        target={"target": "4"}
    )
    dp2 = DatapointResponse(
        id="dp_2",
        messages=messages,
        target=None
    )
    dp3 = DatapointResponse(
        id="dp_3",
        inputs={"custom_key": "raw text input"},
        target=None
    )

    # Convert to EvalPort suite
    suite = to_openeval([dp1, dp2, dp3], suite_id="my_hl_dataset", suite_name="My HL Dataset")

    # Validate against EvalPort SDK schema validator
    validation = validate_suite(suite)
    assert validation.valid, f"Validation failed: {validation.errors}"

    # Verify conversions
    assert suite["id"] == "my_hl_dataset"
    assert suite["name"] == "My HL Dataset"
    assert len(suite["test_cases"]) == 3

    assert suite["test_cases"][0]["id"] == "dp_1"
    assert suite["test_cases"][0]["input"] == "What is 2+2?"
    assert suite["test_cases"][0]["expected_output"] == "4"

    # Chat messages conversation flattening check
    assert suite["test_cases"][1]["id"] == "dp_2"
    assert "user: Tell me a joke." in suite["test_cases"][1]["input"]
    assert "assistant: Why did the chicken cross the road?" in suite["test_cases"][1]["input"]

    # Reconstruct back with from_openeval
    reconstructed = from_openeval(suite)
    assert len(reconstructed) == 3
    assert reconstructed[0]["id"] == "dp_1"
    assert reconstructed[0]["inputs"] == {"question": "What is 2+2?"}
    assert reconstructed[0]["target"] == {"target": "4"}

    # Rebuild a real DatapointResponse object using reconstructed dict to prove round-trip compatibility
    rebuilt_dp = DatapointResponse(**reconstructed[0])
    assert rebuilt_dp.id == "dp_1"
    assert rebuilt_dp.inputs == {"question": "What is 2+2?"}


def test_test_case_empty_input_error():
    # Empty inputs/messages raising ValueError
    dp = DatapointResponse(id="dp_empty", inputs={}, messages=None, target=None)
    with pytest.raises(ValueError, match="has no non-empty input"):
        to_openeval([dp])


def test_evaluation_run_results_conversion():
    # We use dictionary-shaped objects representing Humanloop's EvaluationResponse and EvaluatorLogResponse
    evaluation = {
        "id": "eval_123",
        "name": "My Humanloop Evaluation",
        "evaluators": [
            {
                "id": "ev_bool",
                "name": "Exact Match",
                "spec": {"return_type": "boolean"}
            },
            {
                "id": "ev_num",
                "name": "Similarity",
                "spec": {"return_type": "number"}
            },
            {
                "id": "ev_text",
                "name": "Feedback",
                "spec": {"return_type": "text"}
            }
        ]
    }

    logs = [
        # Log 1: Boolean evaluator, pass
        {
            "id": "log_bool_dp1",
            "source_datapoint_id": "dp_1",
            "evaluator": {"id": "ev_bool", "name": "Exact Match"},
            "judgment": True,
            "start_time": "2026-08-23T21:40:00Z",
            "end_time": "2026-08-23T21:40:01Z",
            "provider_latency": 0.5,
            "output": "4"
        },
        # Log 2: Number evaluator, score 0.8
        {
            "id": "log_num_dp1",
            "source_datapoint_id": "dp_1",
            "evaluator": {"id": "ev_num", "name": "Similarity"},
            "judgment": 0.8,
            "start_time": "2026-08-23T21:40:00Z",
            "end_time": "2026-08-23T21:40:02Z",
            "provider_latency": 0.8,
            "output": "4"
        },
        # Log 3: Number evaluator out-of-range clamping check (clamped to 1.0)
        {
            "id": "log_num_dp2",
            "source_datapoint_id": "dp_2",
            "evaluator": {"id": "ev_num", "name": "Similarity"},
            "judgment": 1.5,
            "start_time": "2026-08-23T21:40:03Z",
            "end_time": "2026-08-23T21:40:04Z",
            "provider_latency": 0.6,
            "output": "5"
        },
        # Log 4: Text evaluator, non-numeric judgment (score should be None, passed=False)
        {
            "id": "log_text_dp2",
            "source_datapoint_id": "dp_2",
            "evaluator": {"id": "ev_text", "name": "Feedback"},
            "judgment": "Somewhat correct but needs expansion.",
            "start_time": "2026-08-23T21:40:03Z",
            "end_time": "2026-08-23T21:40:05Z",
            "provider_latency": 1.2,
            "output": "5"
        }
    ]

    # Convert to EvalPort ResultSet
    result_set = result_to_openeval(
        evaluation=evaluation,
        logs=logs,
        started_at="2026-08-23T21:40:00Z"
    )

    # Validate against EvalPort SDK schema validator
    validation = validate_result_set(result_set)
    assert validation.valid, f"Validation failed: {validation.errors}"

    # Verify metadata and summary
    assert result_set["suite_id"] == "humanloop_suite_eval_123"
    assert result_set["run_id"] == "humanloop_run_eval_123"
    assert len(result_set["results"]) == 2

    # Verify first test case results (dp_1, both tests passed)
    res1 = [r for r in result_set["results"] if r["test_case_id"] == "dp_1"][0]
    assert res1["passed"] is True
    assert res1["actual_output"] == "4"
    assert len(res1["grader_results"]) == 2

    # Exact Match check (Boolean)
    gr_bool = [g for g in res1["grader_results"] if g["grader_id"] == "exact_match"][0]
    assert gr_bool["score"] == 1.0
    assert gr_bool["passed"] is True
    assert gr_bool["metadata"]["humanloop.judgment"] is True

    # Similarity check (Number)
    gr_num1 = [g for g in res1["grader_results"] if g["grader_id"] == "similarity"][0]
    assert gr_num1["score"] == 0.8
    assert gr_num1["passed"] is True

    # Verify second test case results (dp_2)
    res2 = [r for r in result_set["results"] if r["test_case_id"] == "dp_2"][0]
    assert res2["passed"] is False  # Because the text feedback judgment score is None and passed=False
    assert res2["actual_output"] == "5"

    # Similarity check (Number clamping 1.5 -> 1.0)
    gr_num2 = [g for g in res2["grader_results"] if g["grader_id"] == "similarity"][0]
    assert gr_num2["score"] == 1.0
    assert gr_num2["passed"] is True

    # Feedback check (Text judgment -> score=None, passed=False)
    gr_text = [g for g in res2["grader_results"] if g["grader_id"] == "feedback"][0]
    assert gr_text["score"] is None
    assert gr_text["passed"] is False
    assert gr_text["metadata"]["humanloop.judgment"] == "Somewhat correct but needs expansion."
