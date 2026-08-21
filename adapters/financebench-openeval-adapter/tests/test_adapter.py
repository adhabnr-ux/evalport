"""Tests for financebench-openeval-adapter.

Every fixture under tests/fixtures/ is real data trimmed from the real
FinanceBench repo (https://github.com/patronus-ai/financebench, `main` branch,
downloaded 2026-08-21) -- not synthetic/fabricated rows. `financebench_id`s in
the trimmed open-source and results fixtures were chosen to genuinely overlap,
and `doc_name`s in the trimmed open-source and document-info fixtures were
chosen to genuinely overlap, so the join/round-trip tests exercise real
matching behavior, not coincidence.
"""

from __future__ import annotations

import json
import os

import pytest

from financebench_openeval_adapter import (
    from_openeval,
    load_jsonl,
    result_to_openeval,
    to_openeval,
)
from openeval.validate import validate_result_set, validate_suite

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
OPEN_SOURCE_PATH = os.path.join(FIXTURES, "financebench_open_source_sample.jsonl")
DOC_INFO_PATH = os.path.join(FIXTURES, "financebench_document_information_sample.jsonl")
RESULTS_PATH = os.path.join(FIXTURES, "gpt-4_sharedStore_sample.jsonl")


@pytest.fixture
def question_rows():
    return load_jsonl(OPEN_SOURCE_PATH)


@pytest.fixture
def doc_info_rows():
    return load_jsonl(DOC_INFO_PATH)


@pytest.fixture
def result_rows():
    return load_jsonl(RESULTS_PATH)


# --------------------------------------------------------------------------
# load_jsonl
# --------------------------------------------------------------------------


def test_load_jsonl_reads_real_fixture_file(question_rows):
    assert len(question_rows) == 8
    assert all(isinstance(r, dict) for r in question_rows)
    assert question_rows[0]["financebench_id"] == "financebench_id_01858"


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    rows = load_jsonl(str(p))
    assert rows == [{"a": 1}, {"a": 2}]


# --------------------------------------------------------------------------
# to_openeval — structural correctness
# --------------------------------------------------------------------------


def test_to_openeval_produces_one_test_case_per_row(question_rows):
    suite = to_openeval(question_rows)
    assert len(suite["test_cases"]) == len(question_rows) == 8


def test_to_openeval_test_case_ids_match_financebench_ids(question_rows):
    suite = to_openeval(question_rows)
    got_ids = {tc["id"] for tc in suite["test_cases"]}
    want_ids = {r["financebench_id"] for r in question_rows}
    assert got_ids == want_ids


def test_to_openeval_input_is_the_real_question_text(question_rows):
    suite = to_openeval(question_rows)
    tc0 = suite["test_cases"][0]
    assert tc0["input"] == question_rows[0]["question"]
    assert "dividend" in tc0["input"]


def test_to_openeval_expected_output_is_the_gold_answer(question_rows):
    suite = to_openeval(question_rows)
    tc0 = suite["test_cases"][0]
    assert tc0["expected_output"] == question_rows[0]["answer"]
    assert "65 years" in tc0["expected_output"]


def test_to_openeval_context_comes_from_real_evidence_text(question_rows):
    suite = to_openeval(question_rows)
    tc0 = suite["test_cases"][0]
    assert "context" in tc0
    assert len(tc0["context"]) == len(question_rows[0]["evidence"])
    assert tc0["context"][0] == question_rows[0]["evidence"][0]["evidence_text"]
    # sanity: the real evidence text actually contains the fact backing the answer
    assert "65th consecutive" in tc0["context"][0]


def test_to_openeval_metadata_carries_financebench_fields(question_rows):
    suite = to_openeval(question_rows)
    tc0 = suite["test_cases"][0]
    md = tc0["metadata"]
    assert md["financebench.company"] == "3M"
    assert md["financebench.doc_name"] == "3M_2023Q2_10Q"
    assert md["financebench.question_type"] == "novel-generated"
    assert md["financebench.dataset_subset_label"] == "OPEN_SOURCE"
    assert md["financebench.evidence"] == question_rows[0]["evidence"]


def test_to_openeval_row_with_no_evidence_omits_context_key():
    rows = [
        {
            "financebench_id": "fb_test_no_evidence",
            "question": "Q?",
            "answer": "A.",
            "evidence": [],
            "company": "TestCo",
            "doc_name": "TestCo_2020_10K",
        }
    ]
    suite = to_openeval(rows)
    assert "context" not in suite["test_cases"][0]


def test_to_openeval_emits_single_llm_judge_grader(question_rows):
    suite = to_openeval(question_rows)
    assert len(suite["graders"]) == 1
    grader = suite["graders"][0]
    assert grader["type"] == "llm_judge"
    assert grader["params"]["model"] == "gpt-4o"
    for token in ("{input}", "{expected}", "{output}"):
        assert token in grader["params"]["prompt"]
    assert all(tc["graders"] == [grader["id"]] for tc in suite["test_cases"])


def test_to_openeval_judge_model_and_prompt_are_overridable(question_rows):
    suite = to_openeval(
        question_rows,
        judge_model="claude-sonnet-5",
        judge_prompt="Score {output} vs {expected} for {input}.",
    )
    grader = suite["graders"][0]
    assert grader["params"]["model"] == "claude-sonnet-5"
    assert grader["params"]["prompt"] == "Score {output} vs {expected} for {input}."


# --------------------------------------------------------------------------
# to_openeval — the document-info join
# --------------------------------------------------------------------------


def test_to_openeval_joins_document_info_by_doc_name(question_rows, doc_info_rows):
    suite = to_openeval(question_rows, document_info_rows=doc_info_rows)
    tc0 = suite["test_cases"][0]
    md = tc0["metadata"]
    # Real, independently-verifiable fact about 3M_2023Q2_10Q in the real fixture file.
    doc_row = next(d for d in doc_info_rows if d["doc_name"] == "3M_2023Q2_10Q")
    assert md["financebench.doc_type"] == doc_row["doc_type"]
    assert md["financebench.doc_period"] == doc_row["doc_period"]
    assert md["financebench.doc_link"] == doc_row["doc_link"]


def test_to_openeval_reads_real_gics_sector_field_not_readme_typo(doc_info_rows):
    # FinanceBench's README documents "comany_sector_gics"; the real file on
    # `main` (this fixture) actually uses "gics_sector". Confirm the adapter
    # reads the real one.
    assert "gics_sector" in doc_info_rows[0]
    assert "comany_sector_gics" not in doc_info_rows[0]

    rows = [
        {
            "financebench_id": "fb_sector_test",
            "question": "Q?",
            "answer": "A.",
            "evidence": [],
            "company": doc_info_rows[0]["company"],
            "doc_name": doc_info_rows[0]["doc_name"],
        }
    ]
    suite = to_openeval(rows, document_info_rows=doc_info_rows)
    assert suite["test_cases"][0]["metadata"]["financebench.gics_sector"] == doc_info_rows[0]["gics_sector"]


def test_to_openeval_falls_back_to_documented_field_name_if_present():
    doc_info = [{"doc_name": "X_2020_10K", "comany_sector_gics": "Health Care"}]
    rows = [
        {
            "financebench_id": "fb_fallback_test",
            "question": "Q?",
            "answer": "A.",
            "evidence": [],
            "doc_name": "X_2020_10K",
        }
    ]
    suite = to_openeval(rows, document_info_rows=doc_info)
    assert suite["test_cases"][0]["metadata"]["financebench.gics_sector"] == "Health Care"


def test_to_openeval_row_with_no_matching_doc_info_still_converts(question_rows):
    # doc_info_rows deliberately omitted -> no financebench.doc_* metadata, no crash.
    suite = to_openeval(question_rows)
    tc0 = suite["test_cases"][0]
    assert "financebench.doc_type" not in tc0["metadata"]
    assert "financebench.doc_link" not in tc0["metadata"]
    # but the row's own fields are still there
    assert tc0["metadata"]["financebench.company"] == "3M"


def test_to_openeval_unmatched_doc_name_does_not_pull_in_wrong_metadata(question_rows, doc_info_rows):
    # The fixture doc_info file has 2 "extra" docs not referenced by any
    # trimmed question row -- confirm they never leak into any test case.
    extra_doc_names = {d["doc_name"] for d in doc_info_rows} - {
        r["doc_name"] for r in question_rows
    }
    assert extra_doc_names, "fixture setup invariant: there should be unmatched extra docs"
    suite = to_openeval(question_rows, document_info_rows=doc_info_rows)
    used_doc_names = {tc["metadata"]["financebench.doc_name"] for tc in suite["test_cases"]}
    assert used_doc_names.isdisjoint(extra_doc_names)


# --------------------------------------------------------------------------
# to_openeval — real EvalPort validation
# --------------------------------------------------------------------------


def test_to_openeval_output_passes_real_validate_suite(question_rows, doc_info_rows):
    suite = to_openeval(question_rows, document_info_rows=doc_info_rows)
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_to_openeval_without_doc_info_still_passes_real_validate_suite(question_rows):
    suite = to_openeval(question_rows)
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_to_openeval_empty_input_still_produces_valid_shape_but_fails_min_items():
    # EvalSuite.test_cases has minItems: 1 per spec/schemas/testcase.json's
    # sibling schema -- confirm the real validator actually enforces that
    # against this adapter's output for an empty row list, rather than this
    # adapter silently producing something the spec doesn't allow unflagged.
    suite = to_openeval([])
    result = validate_suite(suite)
    assert not result.valid
    assert any(e["path"] == "$.test_cases" for e in result.errors)


# --------------------------------------------------------------------------
# from_openeval — round trip
# --------------------------------------------------------------------------


def test_round_trip_preserves_core_fields(question_rows):
    suite = to_openeval(question_rows)
    rebuilt = from_openeval(suite)
    assert len(rebuilt) == len(question_rows)
    for original, back in zip(question_rows, rebuilt):
        assert back["financebench_id"] == original["financebench_id"]
        assert back["question"] == original["question"]
        assert back["answer"] == original["answer"]
        assert back["company"] == original["company"]
        assert back["doc_name"] == original["doc_name"]
        assert back["question_type"] == original["question_type"]
        assert back["evidence"] == original["evidence"]


def test_round_trip_excludes_doc_info_only_fields_from_open_source_shape(question_rows, doc_info_rows):
    # doc_type/doc_period/doc_link/gics_sector belong to the *separate*
    # financebench_document_information.jsonl file, not to
    # financebench_open_source.jsonl rows -- from_openeval() should reproduce
    # the open_source shape, not a merged one, even though to_openeval() joined
    # the two for richer metadata.
    suite = to_openeval(question_rows, document_info_rows=doc_info_rows)
    rebuilt = from_openeval(suite)
    for row in rebuilt:
        assert "doc_type" not in row
        assert "doc_period" not in row
        assert "doc_link" not in row
        assert "gics_sector" not in row


def test_from_openeval_handles_suite_not_produced_by_this_adapter():
    suite = {
        "version": "1.0.0",
        "id": "hand_written_suite",
        "test_cases": [
            {"id": "tc_1", "input": "What is 2+2?", "expected_output": "4", "graders": ["gr_1"]}
        ],
        "graders": [{"id": "gr_1", "type": "exact_match"}],
    }
    rebuilt = from_openeval(suite)
    assert rebuilt == [{"financebench_id": "tc_1", "question": "What is 2+2?", "answer": "4"}]


def test_from_openeval_empty_suite_returns_empty_list():
    assert from_openeval({"test_cases": []}) == []
    assert from_openeval({}) == []


# --------------------------------------------------------------------------
# result_to_openeval
# --------------------------------------------------------------------------


def test_result_to_openeval_one_result_per_row(result_rows):
    rs = result_to_openeval(result_rows)
    assert len(rs["results"]) == len(result_rows)


def test_result_to_openeval_real_label_distribution_is_carried_through(result_rows):
    # Confirmed by inspecting the real downloaded file: this exact fixture's
    # 8 rows contain a mix of labels, not all-pass or all-fail -- a stronger
    # check than "output is non-empty".
    labels = {r["label"] for r in result_rows}
    assert labels, "fixture setup invariant: fixture must contain real labels"
    assert labels <= {"Correct Answer", "Incorrect Answer", "Refusal"}

    rs = result_to_openeval(result_rows)
    for row, result_entry in zip(result_rows, rs["results"]):
        expected_passed = row["label"] == "Correct Answer"
        assert result_entry["passed"] == expected_passed
        gr = result_entry["grader_results"][0]
        assert gr["passed"] == expected_passed
        assert gr["score"] == (1.0 if expected_passed else 0.0)
        assert gr["reason"] == row["label"]
        assert gr["type"] == "human"


def test_result_to_openeval_actual_output_is_the_real_model_answer(result_rows):
    rs = result_to_openeval(result_rows)
    for row, result_entry in zip(result_rows, rs["results"]):
        assert result_entry["actual_output"] == row["model_answer"]


def test_result_to_openeval_metadata_carries_run_context(result_rows):
    rs = result_to_openeval(result_rows)
    for row, result_entry in zip(result_rows, rs["results"]):
        md = result_entry["metadata"]
        assert md["financebench.model_name"] == row["model_name"]
        assert md["financebench.eval_mode"] == row["eval_mode"]
        assert md["financebench.temp"] == row["temp"]
        assert md["financebench.gold_answer"] == row["gold_answer"]


def test_result_to_openeval_unrecognized_label_flagged_not_silently_dropped():
    rows = [
        {
            "financebench_id": "fb_weird_label",
            "model_name": "test-model",
            "eval_mode": "test",
            "temp": 0.0,
            "question": "Q?",
            "gold_answer": "A",
            "model_answer": "B",
            "label": "Partially Correct",  # not one of the 3 observed real values
        }
    ]
    rs = result_to_openeval(rows)
    entry = rs["results"][0]
    assert entry["passed"] is False  # safe default: anything not exactly "Correct Answer" fails
    assert entry["metadata"]["financebench.unrecognized_label"] is True


def test_result_to_openeval_output_passes_real_validate_result_set(result_rows):
    rs = result_to_openeval(result_rows)
    result = validate_result_set(rs)
    assert result.valid, result.errors


def test_result_to_openeval_custom_suite_id_run_id_and_started_at_are_used(result_rows):
    rs = result_to_openeval(
        result_rows,
        suite_id="my_suite",
        run_id="run_42",
        started_at="2026-08-21T00:00:00Z",
    )
    assert rs["suite_id"] == "my_suite"
    assert rs["run_id"] == "run_42"
    assert rs["started_at"] == "2026-08-21T00:00:00Z"


def test_result_to_openeval_empty_rows_produces_empty_results_and_fails_real_validation():
    rs = result_to_openeval([])
    assert rs["results"] == []
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.results" for e in result.errors)


# --------------------------------------------------------------------------
# End-to-end: suite id / result set suite_id consistency
# --------------------------------------------------------------------------


def test_suite_id_and_result_set_suite_id_can_be_kept_in_sync(question_rows, result_rows):
    suite = to_openeval(question_rows, suite_id="financebench_smoke_test")
    rs = result_to_openeval(result_rows, suite_id=suite["id"], run_id="run_1")
    assert rs["suite_id"] == suite["id"]
    assert validate_suite(suite).valid
    assert validate_result_set(rs).valid
    # every result's test_case_id should reference a real test case in the suite
    suite_ids = {tc["id"] for tc in suite["test_cases"]}
    result_ids = {r["test_case_id"] for r in rs["results"]}
    assert result_ids <= suite_ids
