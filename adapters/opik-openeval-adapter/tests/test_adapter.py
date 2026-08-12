from openeval.validate import validate_suite, validate_result_set

from opik_openeval_adapter import to_openeval, from_openeval, experiment_to_openeval

# These tests exercise the adapter against Opik's *real* SDK classes
# (opik.api_objects.dataset.dataset_item.DatasetItem,
#  opik.api_objects.experiment.experiment_item.ExperimentItemContent,
#  opik.types.FeedbackScoreDict) rather than hand-rolled fakes, so a shape
# drift in the real `opik` package would be caught here, not just in a
# fake that mirrors last year's SDK.
from opik.api_objects.dataset.dataset_item import DatasetItem
from opik.api_objects.experiment.experiment_item import ExperimentItemContent


def _dataset_item(**data):
    return DatasetItem(**data)


# ---------------------------------------------------------------------------
# to_openeval: real DatasetItem objects
# ---------------------------------------------------------------------------

def test_to_openeval_from_real_dataset_items():
    items = [
        _dataset_item(question="What is the capital of France?", expected_answer="Paris", category="geography"),
        _dataset_item(question="What is 2+2?", expected_answer="4", category="math"),
    ]
    suite = to_openeval(items, suite_id="opik_geo_math")

    assert suite["id"] == "opik_geo_math"
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["input"] == "What is the capital of France?"
    assert tc1["expected_output"] == "Paris"
    assert tc1["metadata"]["category"] == "geography"
    assert tc1["metadata"]["opik"]["dataset_item_id"] == tc1["id"]
    # input/expected_output keys are consumed, not duplicated into metadata
    assert "question" not in tc1["metadata"]
    assert "expected_answer" not in tc1["metadata"]


def test_to_openeval_auto_detects_input_and_expected_output_key_variants():
    items = [_dataset_item(user_input="hi", reference="hello")]
    suite = to_openeval(items)
    tc = suite["test_cases"][0]
    assert tc["input"] == "hi"
    assert tc["expected_output"] == "hello"


def test_to_openeval_explicit_key_override():
    items = [_dataset_item(prompt_text="ignored by heuristic", target="also ignored")]
    suite = to_openeval(items, input_key="prompt_text", expected_output_key="target")
    tc = suite["test_cases"][0]
    assert tc["input"] == "ignored by heuristic"
    assert tc["expected_output"] == "also ignored"


def test_to_openeval_from_plain_dicts_matches_dataset_get_items_shape():
    # This is exactly the shape opik's Dataset.get_items() returns: a plain
    # dict with 'id' plus the arbitrary data fields flattened to top level.
    items = [{"id": "abc123", "question": "hi", "expected_answer": "hello", "difficulty": "easy"}]
    suite = to_openeval(items)
    tc = suite["test_cases"][0]
    assert tc["id"] == "abc123"
    assert tc["input"] == "hi"
    assert tc["expected_output"] == "hello"
    assert tc["metadata"]["difficulty"] == "easy"


def test_to_openeval_no_matching_keys_preserves_full_payload_in_metadata():
    items = [_dataset_item(weird_field_1="a", weird_field_2="b")]
    suite = to_openeval(items)
    tc = suite["test_cases"][0]
    # Nothing recognized as input/expected_output -> falls back to the full
    # data dict as input, but every field is still present in metadata too.
    assert "weird_field_1" in tc["input"]
    assert tc["metadata"]["weird_field_1"] == "a"
    assert tc["metadata"]["weird_field_2"] == "b"


def test_to_openeval_validates_against_real_evalport_spec():
    items = [_dataset_item(question="q1", expected_answer="a1")]
    suite = to_openeval(items)
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_to_openeval_exact_match_grader_option():
    items = [_dataset_item(question="2+2", expected_answer="4")]
    suite = to_openeval(items, grader_type="exact_match")
    assert suite["graders"][0]["type"] == "exact_match"


def test_to_openeval_default_grader_is_llm_judge():
    items = [_dataset_item(question="q", expected_answer="a")]
    suite = to_openeval(items)
    assert suite["graders"][0]["type"] == "llm_judge"
    assert "{output}" in suite["graders"][0]["params"]["prompt"]


def test_to_openeval_empty_dataset_still_valid_shape():
    suite = to_openeval([])
    assert suite["test_cases"] == []
    assert suite["version"] == "1.0.0"
    assert suite["graders"] == []


def test_to_openeval_item_with_evaluators_carries_them_in_metadata():
    item = _dataset_item(
        question="q",
        expected_answer="a",
        evaluators=[{"name": "faithfulness", "type": "llm_as_judge", "config": {"model": "gpt-4o"}}],
    )
    suite = to_openeval([item])
    evaluators = suite["test_cases"][0]["metadata"]["opik"]["evaluators"]
    # Serialized to a plain JSON-safe dict, not left as a pydantic object,
    # so the suite is still writable with a plain json.dump().
    assert evaluators == [{"name": "faithfulness", "type": "llm_as_judge", "config": {"model": "gpt-4o"}}]
    import json
    json.dumps(suite)  # must not raise — full suite is JSON-serializable


# ---------------------------------------------------------------------------
# from_openeval: round-trip
# ---------------------------------------------------------------------------

def test_from_openeval_round_trip_matches_dataset_insert_shape():
    items = [_dataset_item(question="What is 2+2?", expected_answer="4", category="math")]
    suite = to_openeval(items)

    restored = from_openeval(suite)
    assert len(restored) == 1
    r = restored[0]
    assert r["input"] == "What is 2+2?"
    assert r["expected_output"] == "4"
    assert r["category"] == "math"
    assert "opik" not in r  # internal bookkeeping key is stripped, not re-inserted


def test_from_openeval_handles_hand_authored_suite_without_opik_metadata():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc1", "input": "hi", "expected_output": "hello", "graders": ["g1"]}],
    }
    restored = from_openeval(suite)
    assert restored == [{"id": "tc1", "input": "hi", "expected_output": "hello"}]


# ---------------------------------------------------------------------------
# experiment_to_openeval: real ExperimentItemContent objects
# ---------------------------------------------------------------------------

def _experiment_item(dataset_item_id, task_output, feedback_scores):
    return ExperimentItemContent(
        id=f"exp_{dataset_item_id}",
        dataset_item_id=dataset_item_id,
        trace_id=f"trace_{dataset_item_id}",
        dataset_item_data={},
        evaluation_task_output=task_output,
        feedback_scores=feedback_scores,
    )


def test_experiment_to_openeval_from_real_experiment_item_content():
    items = [
        _experiment_item(
            "di_1",
            {"output": "Paris"},
            [{"name": "correctness", "value": 1.0, "reason": "exact match"}],
        ),
        _experiment_item(
            "di_2",
            {"output": "5"},
            [{"name": "correctness", "value": 0.0, "reason": "wrong, expected 4"}],
        ),
    ]
    rs = experiment_to_openeval(items, suite_id="opik_geo_math", run_id="run_20260812")

    assert rs["suite_id"] == "opik_geo_math"
    assert rs["run_id"] == "run_20260812"
    assert len(rs["results"]) == 2

    r1 = rs["results"][0]
    assert r1["test_case_id"] == "di_1"
    assert r1["passed"] is True
    assert r1["grader_results"][0]["grader_id"] == "correctness"
    assert r1["grader_results"][0]["score"] == 1.0
    assert r1["metadata"]["opik"]["evaluation_task_output"] == {"output": "Paris"}

    r2 = rs["results"][1]
    assert r2["passed"] is False
    assert r2["grader_results"][0]["passed"] is False

    assert rs["summary"]["total"] == 2
    assert rs["summary"]["passed"] == 1
    assert rs["summary"]["pass_rate"] == 0.5


def test_experiment_to_openeval_multiple_feedback_scores_all_must_pass():
    items = [
        _experiment_item(
            "di_1",
            {"output": "x"},
            [
                {"name": "correctness", "value": 1.0},
                {"name": "toxicity", "value": 0.0},  # below default threshold -> fails
            ],
        )
    ]
    rs = experiment_to_openeval(items, suite_id="s", run_id="r")
    assert len(rs["results"][0]["grader_results"]) == 2
    # one grader failed -> overall result fails, mirroring evalport run's own
    # "every grader must pass" convention (cli/src/run/runner.ts)
    assert rs["results"][0]["passed"] is False


def test_experiment_to_openeval_custom_pass_threshold():
    items = [_experiment_item("di_1", {"output": "x"}, [{"name": "score", "value": 0.6}])]
    rs_default = experiment_to_openeval(items, suite_id="s", run_id="r")
    assert rs_default["results"][0]["passed"] is True  # 0.6 >= default 0.5

    rs_strict = experiment_to_openeval(items, suite_id="s", run_id="r", pass_threshold=0.9)
    assert rs_strict["results"][0]["passed"] is False  # 0.6 < 0.9


def test_experiment_to_openeval_no_feedback_scores_fails_cleanly():
    items = [_experiment_item("di_1", {"output": "x"}, [])]
    rs = experiment_to_openeval(items, suite_id="s", run_id="r")
    assert rs["results"][0]["passed"] is False
    assert rs["results"][0]["grader_results"] == []


def test_experiment_to_openeval_from_plain_dicts():
    items = [
        {
            "dataset_item_id": "di_1",
            "evaluation_task_output": {"output": "y"},
            "feedback_scores": [{"name": "correctness", "value": 1.0}],
        }
    ]
    rs = experiment_to_openeval(items, suite_id="s", run_id="r")
    assert rs["results"][0]["test_case_id"] == "di_1"
    assert rs["results"][0]["passed"] is True


def test_experiment_to_openeval_validates_against_real_evalport_spec():
    items = [_experiment_item("di_1", {"output": "x"}, [{"name": "correctness", "value": 1.0}])]
    rs = experiment_to_openeval(items, suite_id="s", run_id="r")
    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_experiment_to_openeval_explicit_started_at_is_respected():
    items = [_experiment_item("di_1", {"output": "x"}, [{"name": "c", "value": 1.0}])]
    rs = experiment_to_openeval(items, suite_id="s", run_id="r", started_at="2026-01-15T10:30:00Z")
    assert rs["started_at"] == "2026-01-15T10:30:00Z"
    assert rs["completed_at"] == "2026-01-15T10:30:00Z"


# ---------------------------------------------------------------------------
# End-to-end: dataset -> suite -> (simulated run) -> results, both validated
# ---------------------------------------------------------------------------

def test_end_to_end_dataset_and_experiment_round_trip_both_validate():
    dataset_items = [
        _dataset_item(question="What is the capital of France?", expected_answer="Paris"),
        _dataset_item(question="What is 2+2?", expected_answer="4"),
    ]
    suite = to_openeval(dataset_items, suite_id="opik_e2e")
    assert validate_suite(suite).valid

    experiment_items = [
        _experiment_item(suite["test_cases"][0]["id"], {"output": "Paris"}, [{"name": "correctness", "value": 1.0}]),
        _experiment_item(suite["test_cases"][1]["id"], {"output": "4"}, [{"name": "correctness", "value": 1.0}]),
    ]
    result_set = experiment_to_openeval(experiment_items, suite_id=suite["id"], run_id="run_e2e")
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors
    assert result_set["summary"]["pass_rate"] == 1.0

    # Every result's test_case_id actually resolves to a real test case in the suite
    suite_ids = {tc["id"] for tc in suite["test_cases"]}
    assert all(r["test_case_id"] in suite_ids for r in result_set["results"])
