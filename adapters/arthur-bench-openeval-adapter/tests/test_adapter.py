import pytest
from arthur_bench.client.local.client import LocalBenchClient
from arthur_bench.run.testsuite import TestSuite as ArthurTestSuite
from arthur_bench.scoring.exact_match import ExactMatch
from openeval.validate import validate_result_set, validate_suite

from arthur_bench_openeval_adapter import (
    from_openeval,
    run_to_openeval,
    to_openeval,
)


@pytest.fixture
def bench_client(tmp_path):
    """Every test gets its own LocalBenchClient root dir -- Arthur Bench's
    local client writes suite.json/run files to disk immediately on
    TestSuite construction (confirmed by reading
    arthur_bench/client/local/client.py), and looks up existing suites by
    name, so sharing a directory (or the process default) across tests would
    let one test's suite collide with another's."""
    return LocalBenchClient(root_dir=str(tmp_path))


INPUTS = ["What is the capital of Japan?", "What is 2+2?"]
REFERENCES = ["Tokyo", "4"]


def _make_suite(bench_client, name="test-suite", scoring_method="exact_match", references=REFERENCES):
    return ArthurTestSuite(
        name=name,
        scoring_method=scoring_method,
        input_text_list=INPUTS,
        reference_output_list=references,
        client=bench_client,
    )


# ---------------------------------------------------------------------------
# to_openeval()
# ---------------------------------------------------------------------------


def test_to_openeval_produces_valid_suite(bench_client):
    suite = _make_suite(bench_client)
    result = to_openeval(suite, suite_id="my-suite")
    validate_suite(result)  # raises on invalid -- real validator, not a mock

    assert result["id"] == "my-suite"
    assert len(result["test_cases"]) == 2
    assert result["graders"][0]["id"] == "exact_match"
    assert result["graders"][0]["type"] == "custom"
    assert result["graders"][0]["params"]["handler"] == "exact_match"
    assert "case-INsensitive" in result["graders"][0]["description"]


def test_to_openeval_maps_input_and_expected_output(bench_client):
    suite = _make_suite(bench_client)
    result = to_openeval(suite)
    tc0 = result["test_cases"][0]
    assert tc0["input"] == "What is the capital of Japan?"
    assert tc0["expected_output"] == "Tokyo"
    assert tc0["graders"] == ["exact_match"]
    assert tc0["metadata"]["arthur_bench"]["input"] == "What is the capital of Japan?"
    assert tc0["metadata"]["arthur_bench"]["reference_output"] == "Tokyo"


def test_to_openeval_word_count_match_captures_scorer_config(bench_client):
    suite = _make_suite(bench_client, name="wc-suite", scoring_method="word_count_match")
    result = to_openeval(suite)
    validate_suite(result)
    grader = result["graders"][0]
    assert grader["id"] == "word_count_match"
    assert "local, no live API required" in grader["description"]


def test_to_openeval_exact_match_case_sensitive_false_captures_config(bench_client):
    scorer = ExactMatch(case_sensitive=False)
    suite = ArthurTestSuite(
        name="cs-suite",
        scoring_method=scorer,
        input_text_list=INPUTS,
        reference_output_list=REFERENCES,
        client=bench_client,
    )
    result = to_openeval(suite)
    validate_suite(result)
    assert result["graders"][0]["params"]["config"] == {"case_sensitive": False}


def test_to_openeval_description_preserved_in_metadata(bench_client):
    suite = ArthurTestSuite(
        name="described-suite",
        scoring_method="exact_match",
        description="a test suite about capitals and arithmetic",
        input_text_list=INPUTS,
        reference_output_list=REFERENCES,
        client=bench_client,
    )
    result = to_openeval(suite)
    validate_suite(result)
    assert result["metadata"]["arthur_bench"]["description"] == "a test suite about capitals and arithmetic"


# ---------------------------------------------------------------------------
# from_openeval()
# ---------------------------------------------------------------------------


def test_from_openeval_reconstructs_built_in_scorer(bench_client, tmp_path):
    original = _make_suite(bench_client)
    suite_dict = to_openeval(original, suite_id="roundtrip-suite")

    restored_client = LocalBenchClient(root_dir=str(tmp_path / "restored"))
    restored = from_openeval(suite_dict, client=restored_client, test_suite_name="restored-suite")

    assert isinstance(restored, ArthurTestSuite)
    assert restored.scorer.name() == "exact_match"
    assert restored.input_texts == INPUTS
    assert restored.reference_outputs == REFERENCES


def test_from_openeval_can_feed_a_real_run(bench_client, tmp_path):
    original = _make_suite(bench_client)
    suite_dict = to_openeval(original, suite_id="roundtrip-suite-2")

    restored_client = LocalBenchClient(root_dir=str(tmp_path / "restored2"))
    restored = from_openeval(suite_dict, client=restored_client, test_suite_name="restored-suite-2")

    run = restored.run(run_name="restored-run", candidate_output_list=["Tokyo", "4"], save=False)
    assert all(tc.score == 1.0 for tc in run.test_cases)


def test_from_openeval_requires_scorer_override_for_custom_grader(bench_client):
    suite_dict = {
        "version": "1.0.0-rc.2",
        "id": "custom-suite",
        "graders": [{"id": "my_custom_metric", "type": "custom", "params": {"handler": "my_custom_metric"}}],
        "test_cases": [{"id": "tc-1", "input": "hi", "graders": ["my_custom_metric"]}],
    }
    with pytest.raises(ValueError, match="not a built-in Arthur Bench scorer"):
        from_openeval(suite_dict)


def test_from_openeval_rejects_multi_grader_suite():
    suite_dict = {
        "version": "1.0.0-rc.2",
        "id": "s",
        "graders": [
            {"id": "a", "type": "custom", "params": {"handler": "exact_match"}},
            {"id": "b", "type": "custom", "params": {"handler": "word_count_match"}},
        ],
        "test_cases": [{"id": "tc-1", "input": "hi", "graders": ["a", "b"]}],
    }
    with pytest.raises(ValueError, match="exactly one grader"):
        from_openeval(suite_dict)


def test_from_openeval_heuristic_fallback_without_saved_metadata(bench_client, tmp_path):
    suite_dict = {
        "version": "1.0.0-rc.2",
        "id": "hand-authored",
        "graders": [{"id": "exact_match", "type": "custom", "params": {"handler": "exact_match"}}],
        "test_cases": [
            {"id": "tc-1", "input": "What is the capital of France?", "expected_output": "Paris", "graders": ["exact_match"]}
        ],
    }
    restored_client = LocalBenchClient(root_dir=str(tmp_path / "hand"))
    restored = from_openeval(suite_dict, client=restored_client, test_suite_name="hand-suite")
    assert restored.input_texts == ["What is the capital of France?"]
    assert restored.reference_outputs == ["Paris"]


# ---------------------------------------------------------------------------
# run_to_openeval()
# ---------------------------------------------------------------------------


def test_test_run_to_openeval_valid_result_set(bench_client):
    suite = _make_suite(bench_client)
    run = suite.run(run_name="run-1", candidate_output_list=["Tokyo", "4"], save=False)
    result_set = run_to_openeval(run, suite=suite, suite_id="my-suite", run_id="run-1")
    validate_result_set(result_set)  # real validator

    assert result_set["suite_id"] == "my-suite"
    assert result_set["run_id"] == "run-1"
    assert len(result_set["results"]) == 2


def test_test_run_to_openeval_correct_scores_and_pass(bench_client):
    suite = _make_suite(bench_client)
    run = suite.run(run_name="run-2", candidate_output_list=["Tokyo", "4"], save=False)
    result_set = run_to_openeval(run, suite=suite)
    validate_result_set(result_set)

    for r in result_set["results"]:
        gr = r["grader_results"][0]
        assert gr["grader_id"] == "exact_match"
        assert gr["score"] == 1.0
        assert gr["passed"] is True
        assert r["passed"] is True
        assert "match:" in gr["reason"]


def test_test_run_to_openeval_detects_failure_via_category_not_just_score(bench_client):
    suite = _make_suite(bench_client)
    run = suite.run(run_name="run-3", candidate_output_list=["Tokyo", "wrong answer"], save=False)
    result_set = run_to_openeval(run, suite=suite)
    validate_result_set(result_set)

    passed_flags = [r["passed"] for r in result_set["results"]]
    assert passed_flags == [True, False]
    assert result_set["summary"]["total"] == 2
    assert result_set["summary"]["passed"] == 1
    assert result_set["summary"]["failed"] == 1


def test_test_run_to_openeval_word_count_match_uses_numeric_threshold(bench_client):
    suite = _make_suite(bench_client, name="wc-run-suite", scoring_method="word_count_match")
    # candidate has 4 words vs 1-word reference "Tokyo" -> low ratio, should fail;
    # second candidate exactly matches reference word count -> should pass.
    run = suite.run(run_name="wc-run", candidate_output_list=["this has four words", "4"], save=False)
    result_set = run_to_openeval(run, suite=suite)
    validate_result_set(result_set)

    gr0 = result_set["results"][0]["grader_results"][0]
    gr1 = result_set["results"][1]["grader_results"][0]
    assert gr0["grader_id"] == "word_count_match"
    assert gr0["passed"] is False
    assert gr1["passed"] is True


def test_test_run_to_openeval_scorer_name_param_without_suite(bench_client):
    suite = _make_suite(bench_client, name="scorer-name-suite")
    run = suite.run(run_name="run-4", candidate_output_list=["Tokyo", "4"], save=False)
    # Same result whether the caller passes `suite` or just the scorer's name.
    via_suite = run_to_openeval(run, suite=suite)
    via_name = run_to_openeval(run, scorer_name="exact_match")
    assert via_suite["results"][0]["grader_results"][0]["grader_id"] == "exact_match"
    assert via_name["results"][0]["grader_results"][0]["grader_id"] == "exact_match"
    assert via_suite["results"][0]["passed"] == via_name["results"][0]["passed"]


def test_test_run_to_openeval_preserves_run_metadata(bench_client):
    suite = _make_suite(bench_client, name="meta-suite")
    run = suite.run(
        run_name="run-5",
        candidate_output_list=["Tokyo", "4"],
        model_name="gpt-4o-mini",
        save=False,
    )
    result_set = run_to_openeval(run, suite=suite)
    assert result_set["metadata"]["arthur_bench"]["model_name"] == "gpt-4o-mini"


def test_test_run_to_openeval_actual_output_recorded(bench_client):
    suite = _make_suite(bench_client, name="output-suite")
    run = suite.run(run_name="run-6", candidate_output_list=["Tokyo", "4"], save=False)
    result_set = run_to_openeval(run, suite=suite)
    assert result_set["results"][0]["actual_output"] == "Tokyo"


# ---------------------------------------------------------------------------
# Full round trip
# ---------------------------------------------------------------------------


def test_full_round_trip_suite_to_rows_to_real_run_to_resultset(bench_client, tmp_path):
    suite = _make_suite(bench_client, name="e2e-suite")
    suite_dict = to_openeval(suite, suite_id="e2e-suite")
    validate_suite(suite_dict)

    restored_client = LocalBenchClient(root_dir=str(tmp_path / "e2e"))
    restored = from_openeval(suite_dict, client=restored_client, test_suite_name="e2e-restored")
    run = restored.run(run_name="e2e-run", candidate_output_list=["Tokyo", "4"], save=False)

    result_set = run_to_openeval(run, suite=restored, suite_id=suite_dict["id"])
    validate_result_set(result_set)

    assert result_set["suite_id"] == "e2e-suite"
    assert all(r["passed"] for r in result_set["results"])
