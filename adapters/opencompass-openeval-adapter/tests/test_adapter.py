"""Tests for opencompass_openeval_adapter.

Every test that touches OpenCompass calls the real, installed
`opencompass` package directly -- `opencompass.datasets.custom.CustomDataset`,
`opencompass.datasets.custom.OptionSimAccEvaluator`, and
`opencompass.openicl.icl_evaluator.AccEvaluator` -- not mocks. Every
produced Suite/ResultSet is validated against the real
`openeval.validate.validate_suite()` / `validate_result_set()`.
"""

from __future__ import annotations

import json
import tempfile
import os

import pytest

opencompass = pytest.importorskip(
    "opencompass", reason="install with: pip install -e '.[opencompass]' or '.[test]'"
)

from opencompass.datasets.custom import CustomDataset, OptionSimAccEvaluator  # noqa: E402
from opencompass.openicl.icl_evaluator import AccEvaluator  # noqa: E402

from openeval.validate import validate_result_set, validate_suite  # noqa: E402

from opencompass_openeval_adapter import (  # noqa: E402
    from_openeval,
    result_to_openeval,
    to_openeval,
)


MCQ_ROWS = [
    {
        "question": "What is the capital of France?",
        "A": "Paris",
        "B": "Berlin",
        "C": "Madrid",
        "D": "Rome",
        "answer": "A",
    },
    {
        "question": "What is 7 * 6?",
        "A": "41",
        "B": "42",
        "C": "43",
        "D": "40",
        "answer": "B",
    },
    {
        "question": "What is the largest planet in the solar system?",
        "A": "Earth",
        "B": "Mars",
        "C": "Jupiter",
        "D": "Venus",
        "answer": "C",
    },
]

QA_ROWS = [
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
]


@pytest.fixture(scope="module")
def mcq_dataset_from_real_loader():
    """Real CustomDataset.load() against a real .jsonl file on disk --
    exercises the actual OpenCompass loading path, not a hand-built dict."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for row in MCQ_ROWS:
            f.write(json.dumps(row) + "\n")
        path = f.name
    try:
        ds = CustomDataset.load(path=path)
        yield ds
    finally:
        os.unlink(path)


@pytest.fixture(scope="module")
def qa_dataset_from_real_loader():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for row in QA_ROWS:
            f.write(json.dumps(row) + "\n")
        path = f.name
    try:
        ds = CustomDataset.load(path=path)
        yield ds
    finally:
        os.unlink(path)


class TestToOpenevalMCQ:
    def test_produces_valid_suite(self, mcq_dataset_from_real_loader):
        suite = to_openeval(
            mcq_dataset_from_real_loader, options=["A", "B", "C", "D"], suite_id="mcq_smoke"
        )
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_test_case_count_matches_row_count(self, mcq_dataset_from_real_loader):
        suite = to_openeval(mcq_dataset_from_real_loader, options=["A", "B", "C", "D"])
        assert len(suite["test_cases"]) == len(MCQ_ROWS)

    def test_input_contains_question_and_options(self, mcq_dataset_from_real_loader):
        suite = to_openeval(mcq_dataset_from_real_loader, options=["A", "B", "C", "D"])
        tc0 = suite["test_cases"][0]
        assert "What is the capital of France?" in tc0["input"]
        assert "Paris" in tc0["input"]
        assert "Berlin" in tc0["input"]

    def test_expected_output_is_real_answer_letter(self, mcq_dataset_from_real_loader):
        suite = to_openeval(mcq_dataset_from_real_loader, options=["A", "B", "C", "D"])
        expected = [row["answer"] for row in MCQ_ROWS]
        actual = [tc["expected_output"] for tc in suite["test_cases"]]
        assert actual == expected

    def test_grader_is_custom_option_sim_acc(self, mcq_dataset_from_real_loader):
        suite = to_openeval(mcq_dataset_from_real_loader, options=["A", "B", "C", "D"])
        tc0 = suite["test_cases"][0]
        assert len(tc0["graders"]) == 1
        grader = tc0["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"]["handler"] == "opencompass:OptionSimAccEvaluator"
        assert grader["params"]["options"] == ["A", "B", "C", "D"]

    def test_rejects_lowercase_options(self, mcq_dataset_from_real_loader):
        with pytest.raises(ValueError, match="single uppercase letter"):
            to_openeval(mcq_dataset_from_real_loader, options=["a", "b"])

    def test_rejects_missing_output_column(self, mcq_dataset_from_real_loader):
        with pytest.raises(ValueError, match="answer"):
            to_openeval(
                mcq_dataset_from_real_loader,
                options=["A", "B", "C", "D"],
                output_column="nonexistent",
            )

    def test_rejects_missing_option_column(self, mcq_dataset_from_real_loader):
        with pytest.raises(ValueError, match="option column"):
            to_openeval(mcq_dataset_from_real_loader, options=["A", "B", "C", "D", "E"])


class TestToOpenevalQA:
    def test_produces_valid_suite(self, qa_dataset_from_real_loader):
        suite = to_openeval(qa_dataset_from_real_loader, suite_id="qa_smoke")
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_grader_is_exact_match(self, qa_dataset_from_real_loader):
        suite = to_openeval(qa_dataset_from_real_loader)
        tc0 = suite["test_cases"][0]
        assert tc0["graders"] == [{"id": "opencompass_acc", "type": "exact_match"}]

    def test_expected_output_is_real_answer(self, qa_dataset_from_real_loader):
        suite = to_openeval(qa_dataset_from_real_loader)
        expected = [row["answer"] for row in QA_ROWS]
        actual = [tc["expected_output"] for tc in suite["test_cases"]]
        assert actual == expected

    def test_custom_input_columns(self, qa_dataset_from_real_loader):
        suite = to_openeval(qa_dataset_from_real_loader, input_columns=["question"])
        assert suite["test_cases"][0]["input"] == f"question: {QA_ROWS[0]['question']}"


class TestFromOpenevalRoundTrip:
    def test_mcq_round_trip_restores_original_rows(self, mcq_dataset_from_real_loader):
        suite = to_openeval(mcq_dataset_from_real_loader, options=["A", "B", "C", "D"])
        rows = from_openeval(suite)
        assert rows == list(mcq_dataset_from_real_loader)

    def test_qa_round_trip_restores_original_rows(self, qa_dataset_from_real_loader):
        suite = to_openeval(qa_dataset_from_real_loader)
        rows = from_openeval(suite)
        assert rows == list(qa_dataset_from_real_loader)

    def test_round_tripped_rows_reload_via_real_custom_dataset(
        self, mcq_dataset_from_real_loader, tmp_path
    ):
        """The full loop: real loader -> to_openeval -> from_openeval ->
        write back to a real .jsonl file -> real CustomDataset.load() again."""
        suite = to_openeval(mcq_dataset_from_real_loader, options=["A", "B", "C", "D"])
        rows = from_openeval(suite)
        out_path = tmp_path / "roundtrip.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        reloaded = CustomDataset.load(path=str(out_path))
        assert list(reloaded) == list(mcq_dataset_from_real_loader)

    def test_foreign_suite_uses_heuristic_fallback(self):
        suite = {
            "version": "1.0.0",
            "id": "third_party",
            "test_cases": [
                {
                    "id": "tc1",
                    "input": "What is 2+2?",
                    "expected_output": "4",
                    "graders": ["exact_match"],
                }
            ],
        }
        rows = from_openeval(suite)
        assert rows == [{"question": "What is 2+2?", "answer": "4"}]


class TestResultToOpenevalMCQ:
    def test_produces_valid_result_set(self, mcq_dataset_from_real_loader):
        predictions = ["A", "B", "B", ]  # A correct, B correct, B wrong (gold C)
        result_set = result_to_openeval(
            mcq_dataset_from_real_loader,
            predictions,
            options=["A", "B", "C", "D"],
            suite_id="mcq_smoke",
            run_id="run1",
            started_at="2026-08-21T00:00:00Z",
        )
        result = validate_result_set(result_set)
        assert result.valid, result.errors

    def test_uses_real_option_sim_acc_evaluator_details(self, mcq_dataset_from_real_loader):
        predictions = ["A", "B", "B"]
        references = [row["answer"] for row in MCQ_ROWS]
        real = OptionSimAccEvaluator(options=["A", "B", "C", "D"]).score(
            predictions, references, list(mcq_dataset_from_real_loader)
        )
        result_set = result_to_openeval(
            mcq_dataset_from_real_loader,
            predictions,
            options=["A", "B", "C", "D"],
            suite_id="s",
            run_id="r",
            started_at="2026-08-21T00:00:00Z",
        )
        for i, res in enumerate(result_set["results"]):
            d = real["details"][str(i)]
            gr = res["grader_results"][0]
            assert gr["passed"] == d["correct"]
            assert gr["score"] == (1.0 if d["correct"] else 0.0)
            assert gr["metadata"]["opencompass"]["parsed"] == d["parsed"]

    def test_aggregate_accuracy_preserved_and_matches_real_evaluator(
        self, mcq_dataset_from_real_loader
    ):
        predictions = ["A", "B", "B"]
        references = [row["answer"] for row in MCQ_ROWS]
        real = OptionSimAccEvaluator(options=["A", "B", "C", "D"]).score(
            predictions, references, list(mcq_dataset_from_real_loader)
        )
        result_set = result_to_openeval(
            mcq_dataset_from_real_loader,
            predictions,
            options=["A", "B", "C", "D"],
            suite_id="s",
            run_id="r",
            started_at="2026-08-21T00:00:00Z",
        )
        assert result_set["metadata"]["opencompass"]["aggregate_accuracy"] == real["accuracy"]
        # 2 out of 3 correct
        assert result_set["summary"]["pass_rate"] == pytest.approx(2 / 3)


class TestResultToOpenevalQA:
    def test_produces_valid_result_set(self, qa_dataset_from_real_loader):
        predictions = ["Tokyo", "Au", "Marlowe"]  # first two correct, third wrong
        result_set = result_to_openeval(
            qa_dataset_from_real_loader,
            predictions,
            suite_id="qa_smoke",
            run_id="run1",
            started_at="2026-08-21T00:00:00Z",
        )
        result = validate_result_set(result_set)
        assert result.valid, result.errors

    def test_per_item_correctness_reproduces_real_acc_evaluator_aggregate(
        self, qa_dataset_from_real_loader
    ):
        """The core honesty check for the QA path: per-item str(pred) ==
        str(ref), summed and divided by n, must equal AccEvaluator's own
        real aggregate accuracy (as a fraction) -- proving the per-item
        breakdown genuinely reconstructs what the real evaluator computes,
        not a plausible-looking guess at it."""
        predictions = ["Tokyo", "Au", "Marlowe"]
        references = [row["answer"] for row in QA_ROWS]
        real = AccEvaluator().score(predictions=predictions, references=references)

        result_set = result_to_openeval(
            qa_dataset_from_real_loader,
            predictions,
            suite_id="s",
            run_id="r",
            started_at="2026-08-21T00:00:00Z",
        )
        n_correct = sum(1 for res in result_set["results"] if res["passed"])
        assert n_correct / len(QA_ROWS) == pytest.approx(real["accuracy"] / 100)
        assert result_set["metadata"]["opencompass"]["aggregate_accuracy"] == real["accuracy"]

    def test_wrong_prediction_marked_failed(self, qa_dataset_from_real_loader):
        predictions = ["Tokyo", "Au", "Marlowe"]
        result_set = result_to_openeval(
            qa_dataset_from_real_loader,
            predictions,
            suite_id="s",
            run_id="r",
            started_at="2026-08-21T00:00:00Z",
        )
        assert result_set["results"][0]["passed"] is True
        assert result_set["results"][1]["passed"] is True
        assert result_set["results"][2]["passed"] is False
        assert result_set["results"][2]["grader_results"][0]["type"] == "exact_match"


class TestFullLoop:
    def test_full_loop_mcq(self, mcq_dataset_from_real_loader):
        suite = to_openeval(
            mcq_dataset_from_real_loader, options=["A", "B", "C", "D"], suite_id="full_loop_mcq"
        )
        assert validate_suite(suite).valid

        rows = from_openeval(suite)
        predictions = [row["answer"] for row in rows]  # simulate a perfect model
        result_set = result_to_openeval(
            rows,
            predictions,
            options=["A", "B", "C", "D"],
            suite_id=suite["id"],
            run_id="perfect_run",
            started_at="2026-08-21T00:00:00Z",
            completed_at="2026-08-21T00:05:00Z",
        )
        assert validate_result_set(result_set).valid
        assert result_set["summary"]["pass_rate"] == 1.0
        assert all(
            res["test_case_id"] == tc["id"]
            for res, tc in zip(result_set["results"], suite["test_cases"])
        )

    def test_full_loop_qa(self, qa_dataset_from_real_loader):
        suite = to_openeval(qa_dataset_from_real_loader, suite_id="full_loop_qa")
        rows = from_openeval(suite)
        predictions = [row["answer"] for row in rows]
        result_set = result_to_openeval(
            rows,
            predictions,
            suite_id=suite["id"],
            run_id="perfect_run",
            started_at="2026-08-21T00:00:00Z",
        )
        assert validate_result_set(result_set).valid
        assert result_set["summary"]["pass_rate"] == 1.0


class TestInputValidation:
    def test_empty_rows_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            to_openeval([])

    def test_predictions_length_mismatch_rejected(self, qa_dataset_from_real_loader):
        with pytest.raises(ValueError, match="predictions has"):
            result_to_openeval(
                qa_dataset_from_real_loader,
                ["only one"],
                suite_id="s",
                run_id="r",
                started_at="2026-08-21T00:00:00Z",
            )

    def test_ids_length_mismatch_rejected(self, qa_dataset_from_real_loader):
        with pytest.raises(ValueError, match="ids has"):
            to_openeval(qa_dataset_from_real_loader, ids=["only_one"])

    def test_plain_list_of_dicts_works_without_real_loader(self):
        """rows need not come from CustomDataset.load() -- any list of dict
        rows works, e.g. hand-authored or loaded some other way."""
        suite = to_openeval(list(QA_ROWS), suite_id="plain_dicts")
        assert validate_suite(suite).valid
