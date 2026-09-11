import json
import os

import pytest
from openeval.validate import validate_result_set, validate_suite

from fmeval_openeval_adapter import eval_output_to_openeval, from_openeval, to_openeval

# ---------------------------------------------------------------------------
# to_openeval / from_openeval: plain fmeval-shaped rows (no fmeval import
# needed for these -- the adapter only deals in plain dicts here, matching
# the shape a caller would write to a JSON Lines file for an fmeval
# DataConfig).
# ---------------------------------------------------------------------------


def test_to_openeval_canonical_fmeval_columns():
    rows = [
        {"model_input": "What is the capital of France?", "target_output": "Paris", "category": "geography"},
        {"model_input": "What is 2+2?", "target_output": "4", "category": "math"},
    ]
    suite = to_openeval(rows, suite_id="fmeval_geo_math")

    assert validate_suite(suite).valid
    assert suite["id"] == "fmeval_geo_math"
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["input"] == "What is the capital of France?"
    assert tc1["expected_output"] == "Paris"
    assert tc1["metadata"]["fmeval"]["category"] == "geography"
    # input/expected_output keys are consumed, not duplicated into metadata
    assert "model_input" not in tc1["metadata"]["fmeval"]
    assert "target_output" not in tc1["metadata"]["fmeval"]


def test_to_openeval_auto_detects_raw_key_variants():
    rows = [{"question": "hi", "reference": "hello"}]
    suite = to_openeval(rows)
    tc = suite["test_cases"][0]
    assert tc["input"] == "hi"
    assert tc["expected_output"] == "hello"
    assert validate_suite(suite).valid


def test_to_openeval_explicit_key_override():
    rows = [{"prompt_text": "ignored by heuristic", "target": "also ignored"}]
    suite = to_openeval(rows, input_key="prompt_text", expected_output_key="target")
    tc = suite["test_cases"][0]
    assert tc["input"] == "ignored by heuristic"
    assert tc["expected_output"] == "also ignored"


def test_to_openeval_context_maps_to_test_case_context():
    rows = [{"model_input": "Summarize this.", "context": "The quick brown fox jumps over the lazy dog."}]
    suite = to_openeval(rows)
    tc = suite["test_cases"][0]
    assert tc["context"] == ["The quick brown fox jumps over the lazy dog."]


def test_to_openeval_context_list_passthrough():
    rows = [{"model_input": "Q", "context": ["doc one", "doc two"]}]
    suite = to_openeval(rows)
    assert suite["test_cases"][0]["context"] == ["doc one", "doc two"]


def test_to_openeval_no_matching_keys_preserves_full_payload():
    rows = [{"weird_field_1": "a", "weird_field_2": "b"}]
    suite = to_openeval(rows)
    tc = suite["test_cases"][0]
    assert "weird_field_1" in tc["input"]
    assert tc["metadata"]["fmeval"]["weird_field_1"] == "a"
    assert tc["metadata"]["fmeval"]["weird_field_2"] == "b"


def test_to_openeval_no_expected_output_still_validates():
    # Every EvalPort test case needs >=1 grader ref to validate, but with no
    # expected_output anywhere there's nothing for an llm_judge/exact_match
    # grader to compare against -- a "custom" placeholder grader is attached
    # instead, rather than silently claiming a comparison that can't happen.
    rows = [{"model_input": "no target here"}]
    suite = to_openeval(rows)
    assert validate_suite(suite).valid
    assert suite["graders"][0]["type"] == "custom"
    assert "expected_output" not in suite["test_cases"][0]


def test_to_openeval_with_expected_output_gets_llm_judge_grader_by_default():
    rows = [{"model_input": "Q", "target_output": "A"}]
    suite = to_openeval(rows)
    assert suite["graders"][0]["type"] == "llm_judge"


def test_to_openeval_exact_match_grader_type():
    rows = [{"model_input": "Q", "target_output": "A"}]
    suite = to_openeval(rows, grader_type="exact_match")
    assert suite["graders"][0]["type"] == "exact_match"


def test_from_openeval_round_trip():
    rows = [{"model_input": "What is the capital of France?", "target_output": "Paris", "category": "geography"}]
    suite = to_openeval(rows, suite_id="rt")
    restored = from_openeval(suite)
    assert restored == [{"model_input": "What is the capital of France?", "target_output": "Paris", "category": "geography"}]


def test_from_openeval_restores_context():
    rows = [{"model_input": "Q", "context": ["doc one"]}]
    suite = to_openeval(rows)
    restored = from_openeval(suite)
    assert restored[0]["context"] == ["doc one"]


# ---------------------------------------------------------------------------
# eval_output_to_openeval: exercised against a REAL fmeval algorithm run,
# end to end -- not a hand-typed fixture. This runs
# fmeval.eval_algorithms.factual_knowledge.FactualKnowledge().evaluate(...)
# against a real installed `fmeval`, reads the real JSON Lines file it
# writes, and feeds that real output through the adapter. Skipped (not
# failed) if `fmeval` isn't installed, since it's an optional extra.
# ---------------------------------------------------------------------------

fmeval = pytest.importorskip("fmeval", reason="fmeval is an optional test dependency; install via `.[fmeval]`")


@pytest.fixture(scope="module")
def real_fmeval_output(tmp_path_factory):
    """Runs a real fmeval FactualKnowledge evaluation and returns the parsed
    per-record JSON Lines it wrote, plus the dataset rows used to produce it.
    """
    from fmeval.data_loaders.data_config import DataConfig
    from fmeval.eval_algorithms.factual_knowledge import FactualKnowledge, FactualKnowledgeConfig
    from fmeval.constants import MIME_TYPE_JSONLINES

    tmp_path = tmp_path_factory.mktemp("fmeval_data")
    rows = [
        {"question": "What is the capital of France?", "answer": "Paris", "model_output": "The capital of France is Paris."},
        {"question": "Who wrote Hamlet?", "answer": "Shakespeare", "model_output": "It was written by Christopher Marlowe."},
    ]
    ds_path = tmp_path / "dataset.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    config = DataConfig(
        dataset_name="capitals",
        dataset_uri=str(ds_path),
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answer",
        model_output_location="model_output",
    )
    algo = FactualKnowledge(FactualKnowledgeConfig(target_output_delimiter="<OR>"))
    outputs = algo.evaluate(model=None, dataset_config=config, save=True, num_records=10)
    assert len(outputs) == 1
    output_path = outputs[0].output_path
    assert output_path and os.path.exists(output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records, output_path


def test_eval_output_to_openeval_from_real_fmeval_run(real_fmeval_output):
    records, _ = real_fmeval_output
    assert len(records) == 2
    # Confirm the real shape this adapter is built against.
    assert "scores" in records[0]
    assert records[0]["scores"][0]["name"] == "factual_knowledge"

    result_set = eval_output_to_openeval(records=records, suite_id="fmeval_geo_math", run_id="run-1")
    assert validate_result_set(result_set).valid
    assert result_set["summary"]["total"] == 2

    paris_result = next(r for r in result_set["results"] if "Paris" in (r.get("actual_output") or ""))
    assert paris_result["passed"] is True
    gr = paris_result["grader_results"][0]
    assert gr["grader_id"] == "factual_knowledge"
    assert gr["score"] == 1.0

    marlowe_result = next(r for r in result_set["results"] if "Marlowe" in (r.get("actual_output") or ""))
    assert marlowe_result["passed"] is False
    assert marlowe_result["grader_results"][0]["score"] == 0.0


def test_eval_output_to_openeval_from_output_path(real_fmeval_output):
    _, output_path = real_fmeval_output
    result_set = eval_output_to_openeval(output_path=output_path, suite_id="fmeval_geo_math", run_id="run-2")
    assert validate_result_set(result_set).valid
    assert result_set["summary"]["total"] == 2


def test_eval_output_to_openeval_matches_test_case_ids_by_content(real_fmeval_output):
    records, _ = real_fmeval_output
    rows = [
        {"model_input": "What is the capital of France?", "target_output": "Paris"},
        {"model_input": "Who wrote Hamlet?", "target_output": "Shakespeare"},
    ]
    suite = to_openeval(rows, suite_id="fmeval_geo_math")
    result_set = eval_output_to_openeval(
        records=records, suite_id="fmeval_geo_math", run_id="run-3", test_cases=suite["test_cases"]
    )
    ids = {r["test_case_id"] for r in result_set["results"]}
    assert ids == {"tc_0", "tc_1"}


def test_eval_output_to_openeval_requires_exactly_one_source():
    with pytest.raises(ValueError):
        eval_output_to_openeval(suite_id="s", run_id="r")
    with pytest.raises(ValueError):
        eval_output_to_openeval(records=[], output_path="/nonexistent.jsonl", suite_id="s", run_id="r")


def test_eval_output_to_openeval_error_score_maps_to_failed_grader_result():
    records = [{"model_input": "x", "model_output": "y", "target_output": "z", "scores": [{"name": "rouge", "error": "boom"}]}]
    result_set = eval_output_to_openeval(records=records, suite_id="s", run_id="r")
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["passed"] is False
    assert gr["score"] is None
    assert gr["reason"] == "boom"
    assert validate_result_set(result_set).valid


def test_eval_output_to_openeval_lower_is_better_toxicity():
    # Verified against fmeval's real TOXIGEN_SCORE_NAME convention: 0.05 = mostly non-toxic -> should pass.
    records = [{"model_input": "x", "model_output": "y", "scores": [{"name": "toxicity", "value": 0.05}]}]
    result_set = eval_output_to_openeval(records=records, suite_id="s", run_id="r")
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["passed"] is True
    assert gr["metadata"]["raw_value"] == 0.05


def test_eval_output_to_openeval_lower_is_better_toxicity_high_value_fails():
    records = [{"model_input": "x", "model_output": "y", "scores": [{"name": "toxicity", "value": 0.9}]}]
    result_set = eval_output_to_openeval(records=records, suite_id="s", run_id="r")
    assert result_set["results"][0]["grader_results"][0]["passed"] is False


def test_eval_output_to_openeval_clamps_unbounded_score_and_preserves_raw():
    # log_probability_difference (prompt stereotyping) is a signed, unbounded value.
    records = [{"model_input": "x", "model_output": "y", "scores": [{"name": "log_probability_difference", "value": -3.7}]}]
    result_set = eval_output_to_openeval(records=records, suite_id="s", run_id="r")
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["score"] == 0.0  # clamped into [0, 1]
    assert gr["metadata"]["raw_value"] == -3.7  # true value preserved
    assert validate_result_set(result_set).valid
