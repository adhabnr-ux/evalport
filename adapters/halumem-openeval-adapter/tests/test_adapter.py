"""Tests for halumem-openeval-adapter.

tests/fixtures/synthetic_eval_results.json is entirely hand-written (NOT downloaded
or derived from the real HaluMem dataset, which is CC BY-NC-ND 4.0) -- shaped to
match the real record fields documented in eval/evaluation.py and eval/eval_tools.py
on MemTensor/HaluMem's `main` branch, per the plan in MemTensor/HaluMem#12. Its
"overall_score" block was computed by hand, following the exact same arithmetic
`eval/evaluation.py`'s `aggregate_eval_results()` uses (recall/precision/F1 and
per-category ratios) -- see the docstring of
`test_recomputed_aggregates_match_real_halumem_formula` for the worked numbers.
"""

from __future__ import annotations

import json
import os

import pytest

from halumem_openeval_adapter import (
    OPERATIONS,
    from_openeval,
    load_eval_results,
    result_to_openeval,
    to_openeval,
)
from openeval.validate import validate_result_set, validate_suite

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
EVAL_RESULTS_PATH = os.path.join(FIXTURES, "synthetic_eval_results.json")


@pytest.fixture
def eval_results():
    return load_eval_results(EVAL_RESULTS_PATH)


@pytest.fixture(autouse=True)
def _judge_model_env(monkeypatch):
    # Confirms judge_model resolution really does read OPENAI_MODEL (point 1) --
    # every test in this file relies on this fixture unless it explicitly overrides
    # or unsets the env var itself.
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    yield


# --------------------------------------------------------------------------
# load_eval_results
# --------------------------------------------------------------------------


def test_load_eval_results_reads_real_fixture_file(eval_results):
    assert set(eval_results.keys()) >= {
        "question_answering_records",
        "memory_integrity_records",
        "memory_accuracy_records",
        "memory_update_records",
        "overall_score",
    }
    assert len(eval_results["question_answering_records"]) == 4


# --------------------------------------------------------------------------
# judge_model resolution (point 1)
# --------------------------------------------------------------------------


def test_judge_model_read_from_openai_model_env_var(eval_results):
    suite = to_openeval(eval_results, "qa")
    assert suite["graders"][0]["params"]["model"] == "gpt-4o-mini-2024-07-18"


def test_judge_model_explicit_arg_overrides_env(eval_results):
    suite = to_openeval(eval_results, "qa", judge_model="claude-sonnet-5")
    assert suite["graders"][0]["params"]["model"] == "claude-sonnet-5"


def test_judge_model_raises_when_neither_env_nor_arg_given(monkeypatch, eval_results):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        to_openeval(eval_results, "qa")


def test_no_hardcoded_default_model_name_leaks_through(eval_results):
    # The original proposal sketch in MemTensor/HaluMem#12 hardcoded "gpt-4o" as a
    # default; the maintainer flagged this. Confirm that string never appears
    # anywhere the adapter didn't get it from explicitly.
    suite = to_openeval(eval_results, "qa", judge_model="my-actual-judge")
    assert suite["graders"][0]["params"]["model"] == "my-actual-judge"
    assert "gpt-4o" not in json.dumps(suite)


# --------------------------------------------------------------------------
# to_openeval — structural correctness, per operation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("operation", OPERATIONS)
def test_to_openeval_output_passes_real_validate_suite(eval_results, operation):
    suite = to_openeval(eval_results, operation)
    result = validate_suite(suite)
    assert result.valid, result.errors


@pytest.mark.parametrize("operation", OPERATIONS)
def test_to_openeval_accepts_bare_record_list_too(eval_results, operation):
    from halumem_openeval_adapter import _RECORD_KEY  # noqa: PLC0415 (internal, test-only)

    records = eval_results[_RECORD_KEY[operation]]
    suite_from_dict = to_openeval(eval_results, operation)
    suite_from_list = to_openeval(records, operation)
    assert suite_from_dict["test_cases"] == suite_from_list["test_cases"]


def test_to_openeval_qa_test_case_fields(eval_results):
    suite = to_openeval(eval_results, "qa")
    tc0 = suite["test_cases"][0]
    record0 = eval_results["question_answering_records"][0]
    assert tc0["input"] == record0["question"]
    assert tc0["expected_output"] == record0["answer"]
    assert tc0["context"] == [e["memory_content"] for e in record0["evidence"]]
    assert tc0["metadata"]["halumem.question_type"] == "single-hop"
    assert tc0["metadata"]["halumem.difficulty"] == "easy"
    assert tc0["metadata"]["halumem.evidence"] == record0["evidence"]
    assert tc0["graders"] == suite["test_cases"][0]["graders"] == ["halumem_qa_judge"]


def test_to_openeval_qa_grader_is_llm_judge_with_required_tokens(eval_results):
    suite = to_openeval(eval_results, "qa")
    grader = suite["graders"][0]
    assert grader["type"] == "llm_judge"
    for token in ("{input}", "{expected}", "{output}"):
        assert token in grader["params"]["prompt"]


def test_to_openeval_memory_integrity_preserves_source_and_importance(eval_results):
    suite = to_openeval(eval_results, "memory_integrity")
    interference_tc = next(
        tc for tc in suite["test_cases"] if tc["metadata"]["halumem.memory_source"] == "interference"
    )
    assert interference_tc["metadata"]["halumem.importance"] == 1
    assert interference_tc["input"] == interference_tc["expected_output"]


def test_to_openeval_memory_update_context_is_original_memories(eval_results):
    suite = to_openeval(eval_results, "memory_update")
    tc0 = suite["test_cases"][0]
    record0 = eval_results["memory_update_records"][0]
    assert tc0["context"] == record0["original_memories"]
    assert tc0["expected_output"] == record0["memory_content"]


def test_to_openeval_test_case_ids_are_stable_digests_not_builtin_hash(eval_results):
    # Point 4: same input twice -> same ID, computed twice independently (not cached).
    suite_a = to_openeval(eval_results, "qa")
    suite_b = to_openeval(eval_results, "qa")
    ids_a = [tc["id"] for tc in suite_a["test_cases"]]
    ids_b = [tc["id"] for tc in suite_b["test_cases"]]
    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a)  # all unique
    assert all(tc_id.startswith("halumem_qa_") for tc_id in ids_a)


def test_stable_id_matches_across_fresh_interpreter_process(eval_results, tmp_path):
    # The real defect being guarded against: Python's built-in hash() on strings is
    # randomized per-process (PYTHONHASHSEED) unless explicitly fixed, so an ID built
    # from hash() (as the original issue-proposal sketch did) would NOT reproduce
    # across two separate runs of the same conversion. Prove this adapter's IDs do,
    # by running the ID computation in a brand-new subprocess with hash randomization
    # left at Python's default (i.e. NOT disabled) and comparing to this process's ID.
    import subprocess
    import sys

    fixture_path = os.path.join(FIXTURES, "synthetic_eval_results.json")
    script = (
        "import json, sys; "
        "sys.path.insert(0, sys.argv[2]); "
        "from halumem_openeval_adapter import to_openeval, load_eval_results; "
        "er = load_eval_results(sys.argv[1]); "
        "suite = to_openeval(er, 'qa', judge_model='m'); "
        "print(suite['test_cases'][0]['id'])"
    )
    src_dir = os.path.join(os.path.dirname(FIXTURES), "..", "src")
    proc = subprocess.run(
        [sys.executable, "-c", script, fixture_path, os.path.abspath(src_dir)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "random"},
    )
    subprocess_id = proc.stdout.strip()

    this_process_suite = to_openeval(eval_results, "qa", judge_model="m")
    assert subprocess_id == this_process_suite["test_cases"][0]["id"]


def test_stable_ids_match_between_suite_and_resultset(eval_results):
    # A Result.test_case_id MUST reference a real TestCase.id in the same suite --
    # validate_suite() checks dangling grader refs, but the test_case_id <-> suite
    # linkage across to_openeval()/result_to_openeval() is this adapter's own
    # responsibility to get right, since suite and result set are built by two
    # separate function calls here (unlike a single-call runner).
    suite = to_openeval(eval_results, "memory_update")
    rs = result_to_openeval(eval_results, "memory_update", suite_id=suite["id"], run_id="run_1")
    suite_ids = {tc["id"] for tc in suite["test_cases"]}
    result_ids = {r["test_case_id"] for r in rs["results"]}
    assert result_ids == suite_ids


# --------------------------------------------------------------------------
# result_to_openeval — structural correctness, per operation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("operation", OPERATIONS)
def test_result_to_openeval_output_passes_real_validate_result_set(eval_results, operation):
    rs = result_to_openeval(eval_results, operation, suite_id=f"s_{operation}", run_id="r1")
    result = validate_result_set(rs)
    assert result.valid, result.errors


@pytest.mark.parametrize("operation", OPERATIONS)
def test_result_to_openeval_one_result_per_record(eval_results, operation):
    from halumem_openeval_adapter import _RECORD_KEY  # noqa: PLC0415

    records = eval_results[_RECORD_KEY[operation]]
    rs = result_to_openeval(eval_results, operation, suite_id="s", run_id="r1")
    assert len(rs["results"]) == len(records)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_result_to_openeval_grader_type_is_always_llm_judge_never_human(eval_results, operation):
    # Point 1, the core assertion: HaluMem's verdicts come from an LLM judge, not a
    # human, for every one of the four operations -- not just QA.
    rs = result_to_openeval(eval_results, operation, suite_id="s", run_id="r1")
    for result in rs["results"]:
        for gr in result["grader_results"]:
            assert gr["type"] == "llm_judge"
            assert gr["type"] != "human"


# --------------------------------------------------------------------------
# result_to_openeval — QA: categorical outcomes not collapsed (point 2)
# --------------------------------------------------------------------------


def test_qa_result_type_real_distribution_is_carried_through(eval_results):
    # This fixture's 4 QA records deliberately contain Correct, Hallucination,
    # Omission, and one unrecognized value ("Unscored") -- not all one outcome.
    result_types = {r["result_type"] for r in eval_results["question_answering_records"]}
    assert result_types == {"Correct", "Hallucination", "Omission", "Unscored"}

    rs = result_to_openeval(eval_results, "qa", suite_id="s", run_id="r1")
    by_reason = {r["grader_results"][0]["reason"]: r for r in rs["results"]}

    correct = by_reason["Correct"]
    assert correct["passed"] is True
    assert correct["grader_results"][0]["score"] == 1.0
    assert correct["grader_results"][0]["metadata"]["halumem.result_type"] == "Correct"

    halluc = by_reason["Hallucination"]
    assert halluc["passed"] is False
    assert halluc["grader_results"][0]["score"] == 0.0
    assert halluc["grader_results"][0]["metadata"]["halumem.result_type"] == "Hallucination"

    omission = by_reason["Omission"]
    assert omission["passed"] is False
    assert omission["grader_results"][0]["score"] == 0.0
    assert omission["grader_results"][0]["metadata"]["halumem.result_type"] == "Omission"

    # Hallucination and Omission both land at score=0.0/passed=False, but they must
    # remain machine-distinguishable via metadata -- not collapsed into an
    # indistinguishable "failed" bucket (this is the exact point 2 requirement).
    assert (
        halluc["grader_results"][0]["metadata"]["halumem.result_type"]
        != omission["grader_results"][0]["metadata"]["halumem.result_type"]
    )


def test_qa_unrecognized_result_type_gets_null_score_not_silently_zero(eval_results):
    rs = result_to_openeval(eval_results, "qa", suite_id="s", run_id="r1")
    unscored = next(r for r in rs["results"] if r["grader_results"][0]["reason"] == "Unscored")
    assert unscored["grader_results"][0]["score"] is None
    assert unscored["passed"] is False
    assert unscored["grader_results"][0]["metadata"]["halumem.unrecognized_result_type"] is True
    # Real EvalPort validator explicitly allows score: null (spec Validation Rule 5).
    assert validate_result_set(rs).valid


def test_qa_actual_output_is_the_real_system_response(eval_results):
    rs = result_to_openeval(eval_results, "qa", suite_id="s", run_id="r1")
    for record, result in zip(eval_results["question_answering_records"], rs["results"]):
        assert result["actual_output"] == record["system_response"]


# --------------------------------------------------------------------------
# result_to_openeval — memory_update: Other stays distinct from Hallucination/Omission (point 2)
# --------------------------------------------------------------------------


def test_memory_update_all_four_outcomes_distinguishable_in_metadata(eval_results):
    rs = result_to_openeval(eval_results, "memory_update", suite_id="s", run_id="r1")
    outcomes = {r["grader_results"][0]["metadata"]["halumem.memory_update_type"] for r in rs["results"]}
    assert outcomes == {"Correct", "Hallucination", "Omission", "Other"}

    by_outcome = {r["grader_results"][0]["metadata"]["halumem.memory_update_type"]: r for r in rs["results"]}
    # Hallucination, Omission, and Other all score 0.0/fail identically at the
    # score/passed level (HaluMem only rewards "Correct")...
    for outcome in ("Hallucination", "Omission", "Other"):
        assert by_outcome[outcome]["grader_results"][0]["score"] == 0.0
        assert by_outcome[outcome]["passed"] is False
    # ...but metadata.halumem.memory_update_type keeps them three genuinely
    # distinct, machine-readable strings, "Other" included -- the update task's
    # extra category beyond what the QA task has.
    reasons = {by_outcome[o]["grader_results"][0]["reason"] for o in ("Hallucination", "Omission", "Other")}
    assert reasons == {"Hallucination", "Omission", "Other"}


def test_memory_update_actual_output_uses_memories_from_system_when_present(eval_results):
    rs = result_to_openeval(eval_results, "memory_update", suite_id="s", run_id="r1")
    omission_result = next(r for r in rs["results"] if r["grader_results"][0]["reason"] == "Omission")
    # The Omission record's memories_from_system is [] (empty) in the fixture --
    # confirm the adapter doesn't fabricate an actual_output for it.
    assert "actual_output" not in omission_result

    correct_result = next(r for r in rs["results"] if r["grader_results"][0]["reason"] == "Correct")
    assert correct_result["actual_output"] == "Alex relocated from Portland to Seattle."


# --------------------------------------------------------------------------
# result_to_openeval — extraction: original scoring semantics preserved verbatim (point 3)
# --------------------------------------------------------------------------


def test_memory_integrity_score_carried_verbatim_and_normalized(eval_results):
    rs = result_to_openeval(eval_results, "memory_integrity", suite_id="s", run_id="r1")
    for record, result in zip(eval_results["memory_integrity_records"], rs["results"]):
        gr = result["grader_results"][0]
        assert gr["metadata"]["halumem.memory_integrity_score"] == record["memory_integrity_score"]
        assert gr["metadata"]["openeval.raw_score"] == record["memory_integrity_score"]
        # Per Validation Rule 5: normalized to [0,1] from the real 0/1/2 scale.
        assert gr["score"] == record["memory_integrity_score"] / 2.0


def test_memory_integrity_interference_pass_condition_is_inverted(eval_results):
    # A "dialogue"-sourced golden memory passes at score==2 (fully recalled); an
    # "interference"/distractor memory passes at score==0 (correctly NOT recalled).
    # This is the real asymmetry aggregate_eval_results() encodes with its separate
    # interference_memory_scores counter -- confirm the adapter reproduces it.
    rs = result_to_openeval(eval_results, "memory_integrity", suite_id="s", run_id="r1")
    by_content = {}
    for record, result in zip(eval_results["memory_integrity_records"], rs["results"]):
        by_content[record["memory_content"]] = (record, result)

    # "Alex once considered moving to Canada." — interference, score 0 -> should PASS
    _, interference_pass = by_content["Alex once considered moving to Canada."]
    assert interference_pass["passed"] is True

    # "Alex dislikes cold weather." — interference, score 2 -> should FAIL
    # (the system incorrectly incorporated a distractor memory)
    _, interference_fail = by_content["Alex dislikes cold weather."]
    assert interference_fail["passed"] is False

    # "Alex moved to Portland in March." — real golden memory, score 2 -> PASS
    _, golden_pass = by_content["Alex moved to Portland in March."]
    assert golden_pass["passed"] is True

    # "Alex's manager Priya approved the request." — real golden memory, score 0 -> FAIL
    _, golden_fail = by_content["Alex's manager Priya approved the request."]
    assert golden_fail["passed"] is False


def test_memory_accuracy_is_included_in_golden_memories_kept_as_literal_string(eval_results):
    rs = result_to_openeval(eval_results, "memory_accuracy", suite_id="s", run_id="r1")
    for record, result in zip(eval_results["memory_accuracy_records"], rs["results"]):
        gr = result["grader_results"][0]
        # Real field kept exactly as-is ("true"/"false" strings, per point 3), not
        # silently coerced to a bare Python bool and the original string discarded.
        assert gr["metadata"]["halumem.is_included_in_golden_memories"] == record["is_included_in_golden_memories"]
        assert isinstance(gr["metadata"]["halumem.is_included_in_golden_memories"], str)
        assert gr["metadata"]["halumem.memory_accuracy_score"] == record["memory_accuracy_score"]


# --------------------------------------------------------------------------
# from_openeval — round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("operation", OPERATIONS)
def test_round_trip_preserves_identifying_fields(eval_results, operation):
    from halumem_openeval_adapter import _RECORD_KEY  # noqa: PLC0415

    records = eval_results[_RECORD_KEY[operation]]
    suite = to_openeval(eval_results, operation)
    rebuilt = from_openeval(suite, operation)
    assert len(rebuilt) == len(records)
    for original, back in zip(records, rebuilt):
        assert back["uuid"] == original["uuid"]
        assert back["ssession_id"] == original["ssession_id"]


def test_round_trip_qa_preserves_question_answer_and_evidence(eval_results):
    suite = to_openeval(eval_results, "qa")
    rebuilt = from_openeval(suite, "qa")
    for original, back in zip(eval_results["question_answering_records"], rebuilt):
        assert back["question"] == original["question"]
        assert back["answer"] == original["answer"]
        assert back["evidence"] == original["evidence"]
        assert back["question_type"] == original["question_type"]
        assert back["difficulty"] == original["difficulty"]


def test_round_trip_memory_update_preserves_original_memories(eval_results):
    suite = to_openeval(eval_results, "memory_update")
    rebuilt = from_openeval(suite, "memory_update")
    for original, back in zip(eval_results["memory_update_records"], rebuilt):
        assert back["memory_content"] == original["memory_content"]
        assert back["original_memories"] == original["original_memories"]
        assert back["importance"] == original["importance"]


def test_from_openeval_handles_suite_not_produced_by_this_adapter():
    suite = {
        "version": "1.0.0",
        "id": "hand_written_suite",
        "test_cases": [{"id": "tc_1", "input": "What is 2+2?", "expected_output": "4", "graders": ["gr_1"]}],
        "graders": [{"id": "gr_1", "type": "exact_match"}],
    }
    rebuilt = from_openeval(suite, "qa")
    assert rebuilt == [{"question": "What is 2+2?", "answer": "4"}]


def test_from_openeval_rejects_unknown_operation():
    with pytest.raises(ValueError, match="Unknown operation"):
        from_openeval({"test_cases": []}, "not_a_real_operation")


def test_to_openeval_rejects_unknown_operation(eval_results):
    with pytest.raises(ValueError, match="Unknown operation"):
        to_openeval(eval_results, "not_a_real_operation")


# --------------------------------------------------------------------------
# End-to-end: recomputing HaluMem's own aggregate metrics from converted
# ResultSets alone (point 3's stated validation criterion)
# --------------------------------------------------------------------------


def test_recomputed_aggregates_match_real_halumem_formula(eval_results):
    """The maintainer's stated acceptance criterion (MemTensor/HaluMem#12 comment):
    'converting HaluMem evaluation records to EvalPort preserves all information
    required to reproduce the official HaluMem aggregate metrics.'

    This test recomputes memory_integrity.recall(all)/weighted_recall(all)/
    interference_accuracy(all), memory_accuracy.target_accuracy(all)/
    weighted_accuracy(all), memory_extraction_f1, memory_update's four ratios(all),
    and question_answering's three ratios(all)/(valid) -- using ONLY the fields
    available on the converted EvalPort ResultSets (metadata.halumem.* and
    metadata.openeval.raw_score) -- and diffs them against
    tests/fixtures/synthetic_eval_results.json's "overall_score" block, which was
    computed by hand using the exact same formula eval/evaluation.py's
    aggregate_eval_results() uses. A mismatch here would mean the conversion lost
    information HaluMem's own aggregate metrics depend on.
    """
    expected = eval_results["overall_score"]

    # --- memory_integrity ---
    integrity_rs = result_to_openeval(eval_results, "memory_integrity", suite_id="s", run_id="r1")
    golden_items, interference_items = [], []
    for r in integrity_rs["results"]:
        gr = r["grader_results"][0]
        raw = gr["metadata"]["halumem.memory_integrity_score"]
        importance = gr["metadata"].get("halumem.importance")
        is_interference = gr["metadata"]["halumem.memory_source"] == "interference"
        (interference_items if is_interference else golden_items).append((raw, importance))

    recall_all = sum(1 for score, _ in golden_items if score == 2) / len(golden_items)
    weighted_recall_all = sum(0.5 * score * imp for score, imp in golden_items) / sum(
        imp for _, imp in golden_items
    )
    interference_accuracy_all = sum(1 for score, _ in interference_items if score == 0) / len(interference_items)

    assert recall_all == pytest.approx(expected["memory_integrity"]["recall(all)"])
    assert weighted_recall_all == pytest.approx(expected["memory_integrity"]["weighted_recall(all)"])
    assert interference_accuracy_all == pytest.approx(expected["memory_integrity"]["interference_accuracy(all)"])

    # --- memory_accuracy ---
    accuracy_rs = result_to_openeval(eval_results, "memory_accuracy", suite_id="s", run_id="r1")
    target_scores, target_count, all_scores = [], 0, []
    for r in accuracy_rs["results"]:
        gr = r["grader_results"][0]
        raw = gr["metadata"]["halumem.memory_accuracy_score"]
        included = gr["metadata"]["halumem.is_included_in_golden_memories"] in ("true", "True")
        all_scores.append(raw)
        if included:
            target_scores.append(raw)
            target_count += 1

    target_accuracy_all = sum(0.5 * s for s in target_scores) / target_count
    weighted_accuracy_all = sum(0.5 * s for s in all_scores) / len(all_scores)

    assert target_accuracy_all == pytest.approx(expected["memory_accuracy"]["target_accuracy(all)"])
    assert weighted_accuracy_all == pytest.approx(expected["memory_accuracy"]["weighted_accuracy(all)"])

    # --- memory_extraction_f1 (harmonic mean of precision=target_accuracy(all), recall=recall(all)) ---
    precision, recall = target_accuracy_all, recall_all
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    assert f1 == pytest.approx(expected["memory_extraction_f1"])

    # --- memory_update ---
    update_rs = result_to_openeval(eval_results, "memory_update", suite_id="s", run_id="r1")
    update_types = [r["grader_results"][0]["metadata"]["halumem.memory_update_type"] for r in update_rs["results"]]
    n = len(update_types)
    for label, key in (
        ("Correct", "correct_update_memory_ratio(all)"),
        ("Hallucination", "hallucination_update_memory_ratio(all)"),
        ("Omission", "omission_update_memory_ratio(all)"),
        ("Other", "other_update_memory_ratio(all)"),
    ):
        assert update_types.count(label) / n == pytest.approx(expected["memory_update"][key])

    # --- question_answering ---
    qa_rs = result_to_openeval(eval_results, "qa", suite_id="s", run_id="r1")
    result_types = [r["grader_results"][0]["metadata"]["halumem.result_type"] for r in qa_rs["results"]]
    valid_types = [t for t in result_types if t in {"Correct", "Hallucination", "Omission"}]
    n_all, n_valid = len(result_types), len(valid_types)
    for label, key_all, key_valid in (
        ("Correct", "correct_qa_ratio(all)", "correct_qa_ratio(valid)"),
        ("Hallucination", "hallucination_qa_ratio(all)", "hallucination_qa_ratio(valid)"),
        ("Omission", "omission_qa_ratio(all)", "omission_qa_ratio(valid)"),
    ):
        assert result_types.count(label) / n_all == pytest.approx(expected["question_answering"][key_all])
        assert valid_types.count(label) / n_valid == pytest.approx(expected["question_answering"][key_valid])
