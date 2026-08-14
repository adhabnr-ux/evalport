from openeval.validate import validate_result_set, validate_suite

from phoenix_openeval_adapter import (
    experiment_to_openeval,
    from_openeval,
    to_openeval,
)

# These tests exercise the adapter against Phoenix's *real* generated types
# (phoenix.client.__generated__.v1.DatasetExample, v1.ExperimentRun,
# phoenix.client.resources.experiments.types.ExperimentEvaluationRun) rather
# than hand-rolled fakes, so a shape drift in the real `arize-phoenix-client`
# package would be caught here, not just in a fake that mirrors last year's
# SDK.
from phoenix.client.resources.experiments.types import ExperimentEvaluationRun


def _example(id, input, output, metadata=None):
    # v1.DatasetExample is a TypedDict -- a plain dict literally satisfies
    # its type contract, this is exactly what real client code receives
    # back from Dataset.examples.
    return {
        "id": id,
        "node_id": id,
        "input": input,
        "output": output,
        "metadata": metadata or {},
        "updated_at": "2026-08-14T00:00:00Z",
        "source": "app",
    }


def _task_run(run_id, example_id, output, error=None):
    return {
        "dataset_example_id": example_id,
        "output": output,
        "repetition_number": 1,
        "start_time": "2026-08-14T00:00:00Z",
        "end_time": "2026-08-14T00:00:01.500Z",
        "id": run_id,
        "experiment_id": "exp_1",
        "error": error,
    }


def _eval_run(experiment_run_id, name, result, error=None):
    return ExperimentEvaluationRun(
        experiment_run_id=experiment_run_id,
        start_time=__import__("datetime").datetime(2026, 8, 14),
        end_time=__import__("datetime").datetime(2026, 8, 14),
        name=name,
        annotator_kind="CODE",
        error=error,
        result=result,
    )


# ---------------------------------------------------------------------------
# to_openeval: real v1.DatasetExample dicts
# ---------------------------------------------------------------------------


def test_to_openeval_from_real_dataset_examples():
    examples = [
        _example("ex_1", {"question": "What is the capital of France?"}, {"answer": "Paris"}, {"category": "geography"}),
        _example("ex_2", {"question": "What is 2+2?"}, {"answer": "4"}, {"category": "math"}),
    ]
    suite = to_openeval(examples, suite_id="phoenix_geo_math")

    assert suite["id"] == "phoenix_geo_math"
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["id"] == "ex_1"
    assert tc1["input"] == "What is the capital of France?"
    assert tc1["expected_output"] == "Paris"
    assert tc1["graders"] == ["gr_output_match"]
    # full raw input/output preserved even though a single key was picked
    assert tc1["metadata"]["phoenix"]["raw_input"] == {"question": "What is the capital of France?"}
    assert tc1["metadata"]["phoenix"]["raw_output"] == {"answer": "Paris"}
    assert tc1["metadata"]["phoenix"]["raw_metadata"] == {"category": "geography"}


def test_to_openeval_multi_key_input_falls_back_to_json_dump():
    examples = [_example("ex_1", {"system": "be terse", "user": "hi"}, {"answer": "hello"})]
    suite = to_openeval(examples)
    tc = suite["test_cases"][0]
    # neither key matches a known name and there's more than one key ->
    # nothing is silently dropped, the whole mapping is preserved as JSON
    assert "system" in tc["input"] and "user" in tc["input"]


def test_to_openeval_single_unrecognized_key_is_unwrapped():
    examples = [_example("ex_1", {"weird_field": "hello there"}, {"weird_out": "hi"})]
    suite = to_openeval(examples)
    tc = suite["test_cases"][0]
    assert tc["input"] == "hello there"
    assert tc["expected_output"] == "hi"


def test_to_openeval_explicit_key_override():
    examples = [_example("ex_1", {"a": "ignored", "prompt_text": "used"}, {"b": "ignored", "target": "used_out"})]
    suite = to_openeval(examples, input_key="prompt_text", expected_output_key="target")
    tc = suite["test_cases"][0]
    assert tc["input"] == "used"
    assert tc["expected_output"] == "used_out"


def test_to_openeval_no_output_omits_expected_output():
    examples = [_example("ex_1", {"question": "hi"}, {})]
    suite = to_openeval(examples)
    assert "expected_output" not in suite["test_cases"][0]


def test_to_openeval_exact_match_grader_option():
    examples = [_example("ex_1", {"question": "2+2"}, {"answer": "4"})]
    suite = to_openeval(examples, grader_type="exact_match")
    assert suite["graders"][0]["type"] == "exact_match"


def test_to_openeval_default_grader_is_llm_judge():
    examples = [_example("ex_1", {"question": "q"}, {"answer": "a"})]
    suite = to_openeval(examples)
    assert suite["graders"][0]["type"] == "llm_judge"
    assert "{output}" in suite["graders"][0]["params"]["prompt"]


def test_to_openeval_empty_examples_still_valid_shape():
    suite = to_openeval([])
    assert suite["test_cases"] == []
    assert suite["version"] == "1.0.0"
    assert suite["graders"] == []


def test_to_openeval_validates_against_real_evalport_spec():
    examples = [_example("ex_1", {"question": "q1"}, {"answer": "a1"})]
    suite = to_openeval(examples)
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_to_openeval_accepts_object_style_examples_not_just_dicts():
    class FakeExampleProxy:
        """Mimics phoenix.client.resources.experiments.types.ExampleProxy's
        attribute-style access without depending on constructing a real one
        (which requires a full v1.DatasetExample dict anyway)."""

        def __init__(self, id, input, output, metadata):
            self.id = id
            self.input = input
            self.output = output
            self.metadata = metadata

    proxy = FakeExampleProxy("ex_1", {"question": "hi"}, {"answer": "hello"}, {})
    suite = to_openeval([proxy])
    assert suite["test_cases"][0]["id"] == "ex_1"
    assert suite["test_cases"][0]["input"] == "hi"


# ---------------------------------------------------------------------------
# from_openeval: round-trip
# ---------------------------------------------------------------------------


def test_from_openeval_round_trip_matches_phoenix_upload_shape():
    examples = [_example("ex_1", {"question": "What is 2+2?"}, {"answer": "4"})]
    suite = to_openeval(examples)

    phoenix_examples = from_openeval(suite)
    assert len(phoenix_examples) == 1
    pe = phoenix_examples[0]
    assert pe["id"] == "ex_1"
    assert pe["input"] == {"input": "What is 2+2?"}
    assert pe["output"] == {"expected_output": "4"}

    # round trip: feeding these straight back through to_openeval recovers
    # the exact text, because "input"/"expected_output" are first in the
    # default key-detection lists.
    suite2 = to_openeval(phoenix_examples)
    assert suite2["test_cases"][0]["input"] == "What is 2+2?"
    assert suite2["test_cases"][0]["expected_output"] == "4"


def test_from_openeval_no_expected_output_produces_empty_output_mapping():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}],
    }
    phoenix_examples = from_openeval(suite)
    assert phoenix_examples[0]["output"] == {}


# ---------------------------------------------------------------------------
# experiment_to_openeval: real ExperimentEvaluationRun dataclass instances
# ---------------------------------------------------------------------------


def test_experiment_to_openeval_from_real_ran_experiment():
    ran_experiment = {
        "experiment_id": "exp_1",
        "dataset_id": "ds_1",
        "dataset_version_id": "dsv_1",
        "task_runs": [
            _task_run("run_1", "ex_1", "Paris"),
            _task_run("run_2", "ex_2", "5"),
        ],
        "evaluation_runs": [
            _eval_run("run_1", "correctness", {"score": 1.0, "label": "correct"}),
            _eval_run("run_2", "correctness", {"score": 0.0, "label": "incorrect", "explanation": "wrong, expected 4"}),
        ],
        "experiment_metadata": {},
        "project_name": None,
    }

    rs = experiment_to_openeval(ran_experiment, suite_id="phoenix_geo_math", run_id="exp_1")

    assert rs["suite_id"] == "phoenix_geo_math"
    assert rs["run_id"] == "exp_1"
    assert len(rs["results"]) == 2

    r1 = rs["results"][0]
    assert r1["test_case_id"] == "ex_1"
    assert r1["actual_output"] == "Paris"
    assert r1["passed"] is True
    assert r1["grader_results"][0]["grader_id"] == "correctness"
    assert r1["grader_results"][0]["score"] == 1.0
    assert r1["duration_ms"] == 1500

    r2 = rs["results"][1]
    assert r2["passed"] is False
    assert r2["grader_results"][0]["reason"] == "wrong, expected 4"

    assert rs["summary"]["total"] == 2
    assert rs["summary"]["passed"] == 1
    assert rs["summary"]["pass_rate"] == 0.5

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_experiment_to_openeval_label_only_evaluator_no_score():
    # Phoenix code evaluators commonly return only a label (e.g. a
    # pass/fail heuristic) with no numeric score.
    ran_experiment = {
        "experiment_id": "exp_2",
        "dataset_id": "ds_1",
        "task_runs": [_task_run("run_1", "ex_1", "some output")],
        "evaluation_runs": [_eval_run("run_1", "format_check", {"label": "PASS"})],
    }
    rs = experiment_to_openeval(ran_experiment)
    assert rs["results"][0]["grader_results"][0]["passed"] is True
    assert rs["results"][0]["grader_results"][0]["score"] is None


def test_experiment_to_openeval_multiple_evaluators_all_must_pass():
    ran_experiment = {
        "experiment_id": "exp_3",
        "dataset_id": "ds_1",
        "task_runs": [_task_run("run_1", "ex_1", "x")],
        "evaluation_runs": [
            _eval_run("run_1", "correctness", {"score": 1.0}),
            _eval_run("run_1", "toxicity", {"score": 0.0}),  # below threshold -> fails
        ],
    }
    rs = experiment_to_openeval(ran_experiment)
    assert len(rs["results"][0]["grader_results"]) == 2
    assert rs["results"][0]["passed"] is False


def test_experiment_to_openeval_evaluator_error_marks_grader_failed():
    ran_experiment = {
        "experiment_id": "exp_4",
        "dataset_id": "ds_1",
        "task_runs": [_task_run("run_1", "ex_1", "x")],
        "evaluation_runs": [_eval_run("run_1", "correctness", {"score": 1.0}, error="evaluator crashed")],
    }
    rs = experiment_to_openeval(ran_experiment)
    assert rs["results"][0]["grader_results"][0]["passed"] is False


def test_experiment_to_openeval_task_run_error_recorded():
    ran_experiment = {
        "experiment_id": "exp_5",
        "dataset_id": "ds_1",
        "task_runs": [_task_run("run_1", "ex_1", None, error="task failed")],
        "evaluation_runs": [],
    }
    rs = experiment_to_openeval(ran_experiment)
    assert rs["results"][0]["error"]["message"] == "task failed"
    assert rs["results"][0]["passed"] is False


def test_experiment_to_openeval_no_evaluations_fails_cleanly():
    ran_experiment = {
        "experiment_id": "exp_6",
        "dataset_id": "ds_1",
        "task_runs": [_task_run("run_1", "ex_1", "x")],
        "evaluation_runs": [],
    }
    rs = experiment_to_openeval(ran_experiment)
    assert rs["results"][0]["passed"] is False
    assert rs["results"][0]["grader_results"] == []


def test_experiment_to_openeval_custom_pass_threshold():
    ran_experiment = {
        "experiment_id": "exp_7",
        "dataset_id": "ds_1",
        "task_runs": [_task_run("run_1", "ex_1", "x")],
        "evaluation_runs": [_eval_run("run_1", "score", {"score": 0.6})],
    }
    rs_default = experiment_to_openeval(ran_experiment)
    assert rs_default["results"][0]["passed"] is True  # 0.6 >= default 0.5

    rs_strict = experiment_to_openeval(ran_experiment, pass_threshold=0.9)
    assert rs_strict["results"][0]["passed"] is False  # 0.6 < 0.9


def test_experiment_to_openeval_score_clamped_to_valid_range():
    ran_experiment = {
        "experiment_id": "exp_8",
        "dataset_id": "ds_1",
        "task_runs": [_task_run("run_1", "ex_1", "x")],
        "evaluation_runs": [_eval_run("run_1", "score", {"score": 1.5})],
    }
    rs = experiment_to_openeval(ran_experiment)
    assert rs["results"][0]["grader_results"][0]["score"] == 1.0
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# End-to-end: dataset -> suite -> (simulated experiment) -> results, both
# validated against the real EvalPort spec
# ---------------------------------------------------------------------------


def test_end_to_end_dataset_and_experiment_round_trip_both_validate():
    examples = [
        _example("ex_1", {"question": "What is the capital of France?"}, {"answer": "Paris"}),
        _example("ex_2", {"question": "What is 2+2?"}, {"answer": "4"}),
    ]
    suite = to_openeval(examples, suite_id="phoenix_e2e")
    assert validate_suite(suite).valid

    ran_experiment = {
        "experiment_id": "exp_e2e",
        "dataset_id": "phoenix_e2e",
        "task_runs": [
            _task_run("run_1", "ex_1", "Paris"),
            _task_run("run_2", "ex_2", "4"),
        ],
        "evaluation_runs": [
            _eval_run("run_1", "correctness", {"score": 1.0}),
            _eval_run("run_2", "correctness", {"score": 1.0}),
        ],
    }
    result_set = experiment_to_openeval(ran_experiment, suite_id=suite["id"], run_id="exp_e2e")
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors
    assert result_set["summary"]["pass_rate"] == 1.0

    # Every result's test_case_id actually resolves to a real test case
    suite_ids = {tc["id"] for tc in suite["test_cases"]}
    assert all(r["test_case_id"] in suite_ids for r in result_set["results"])
