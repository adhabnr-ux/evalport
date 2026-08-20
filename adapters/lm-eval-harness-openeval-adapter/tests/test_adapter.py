"""Tests for lm_eval_harness_openeval_adapter.

Every test that touches lm-eval runs a real `simple_evaluate(model="dummy",
..., log_samples=True)` call -- lm-eval's own built-in stub model, no real
weights, but a genuine evaluation loop against real tasks/documents pulled
live from the Hugging Face Hub. Every produced Suite/ResultSet is validated
against the real `openeval.validate.validate_suite()` /
`validate_result_set()` -- not mocks, not hand-rolled assertions about
shape alone.
"""

from __future__ import annotations

import pytest

lm_eval = pytest.importorskip(
    "lm_eval", reason="install with: pip install -e '.[lm-eval]' or '.[test]'"
)

from lm_eval import simple_evaluate  # noqa: E402

from openeval.validate import validate_result_set, validate_suite  # noqa: E402

from lm_eval_harness_openeval_adapter import (  # noqa: E402
    from_openeval,
    result_to_openeval,
    to_openeval,
)


@pytest.fixture(scope="module")
def copa_eval():
    """Real evaluation run: dummy model, real 'copa' task/documents, 3
    examples, log_samples=True. Loglikelihood-based multiple choice."""
    return simple_evaluate(
        model="dummy", tasks=["copa"], limit=3, log_samples=True, bootstrap_iters=0
    )


@pytest.fixture(scope="module")
def boolq_eval():
    """Real evaluation run: dummy model, real 'boolq' task, 3 examples."""
    return simple_evaluate(
        model="dummy", tasks=["boolq"], limit=3, log_samples=True, bootstrap_iters=0
    )


@pytest.fixture(scope="module")
def gsm8k_eval():
    """Real evaluation run: dummy model, real 'gsm8k' task, 2 examples.
    Generation task with two filters (strict-match / flexible-extract),
    metric exact_match -- exercises the multi-filter grader path and the
    exact_match-native-grader mapping."""
    return simple_evaluate(
        model="dummy", tasks=["gsm8k"], limit=2, log_samples=True, bootstrap_iters=0
    )


class TestToOpenevalMultipleChoice:
    def test_copa_produces_valid_suite(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        suite = to_openeval("copa", samples, suite_id="copa_smoke")
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_copa_test_case_count_matches_doc_count(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        suite = to_openeval("copa", samples)
        doc_ids = {s["doc_id"] for s in samples}
        assert len(suite["test_cases"]) == len(doc_ids)

    def test_copa_grader_is_custom_for_acc(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        suite = to_openeval("copa", samples)
        tc = suite["test_cases"][0]
        assert len(tc["graders"]) == 1
        grader = tc["graders"][0]
        assert grader["type"] == "custom"
        assert grader["id"] == "acc"
        assert grader["params"]["handler"] == "lm-evaluation-harness:acc"

    def test_copa_input_is_real_prompt_text(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        suite = to_openeval("copa", samples)
        for tc in suite["test_cases"]:
            assert isinstance(tc["input"], str) and tc["input"] != ""
            # the real context lm-eval sent the model, not a fabricated string
            assert tc["input"] == samples[0]["arguments"][0][0] or any(
                tc["input"] == s["arguments"][0][0]
                for s in samples
                if s["doc_id"] == int(tc["id"].rsplit("_", 1)[1])
            )

    def test_copa_expected_output_is_real_target(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        suite = to_openeval("copa", samples)
        targets = {s["doc_id"]: s["target"] for s in samples}
        for tc in suite["test_cases"]:
            doc_id = int(tc["id"].rsplit("_", 1)[1])
            assert tc["expected_output"] == targets[doc_id]

    def test_copa_metadata_preserves_real_doc_and_hashes(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        suite = to_openeval("copa", samples)
        tc = suite["test_cases"][0]
        meta = tc["metadata"]["lm_eval"]
        assert meta["task_name"] == "copa"
        assert meta["doc"] is not None
        assert meta["doc_hash"]
        assert meta["prompt_hash"]
        assert meta["target_hash"]

    def test_boolq_also_produces_valid_suite(self, boolq_eval):
        samples = boolq_eval["samples"]["boolq"]
        suite = to_openeval("boolq", samples, suite_id="boolq_smoke")
        result = validate_suite(suite)
        assert result.valid, result.errors


class TestToOpenevalGeneration:
    def test_gsm8k_produces_valid_suite(self, gsm8k_eval):
        samples = gsm8k_eval["samples"]["gsm8k"]
        suite = to_openeval("gsm8k", samples, suite_id="gsm8k_smoke")
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_gsm8k_dedups_across_filters(self, gsm8k_eval):
        # 2 docs x 2 filters = 4 sample entries, but only 2 unique documents
        samples = gsm8k_eval["samples"]["gsm8k"]
        assert len(samples) == 4
        suite = to_openeval("gsm8k", samples)
        assert len(suite["test_cases"]) == 2

    def test_gsm8k_has_one_grader_per_filter(self, gsm8k_eval):
        samples = gsm8k_eval["samples"]["gsm8k"]
        suite = to_openeval("gsm8k", samples)
        tc = suite["test_cases"][0]
        grader_ids = {g["id"] for g in tc["graders"]}
        assert grader_ids == {
            "exact_match__strict-match",
            "exact_match__flexible-extract",
        }

    def test_gsm8k_exact_match_maps_to_native_grader_type(self, gsm8k_eval):
        samples = gsm8k_eval["samples"]["gsm8k"]
        suite = to_openeval("gsm8k", samples)
        for grader in suite["test_cases"][0]["graders"]:
            assert grader["type"] == "exact_match"
            # exact_match has zero required params per spec/schemas/grader.json
            assert "params" not in grader


class TestFromOpeneval:
    def test_round_trips_prompt_and_target(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        suite = to_openeval("copa", samples)
        recovered = from_openeval(suite)
        assert len(recovered) == len(suite["test_cases"])
        by_doc_id = {r["doc_id"]: r for r in recovered}
        for s in samples:
            r = by_doc_id[s["doc_id"]]
            assert r["prompt"] == s["arguments"][0][0]
            assert r["target"] == s["target"]
            assert r["task_name"] == "copa"
            assert r["doc"] == s["doc"]

    def test_clean_skips_test_cases_without_lm_eval_metadata(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "test_cases": [
                {
                    "id": "tc1",
                    "input": "hi",
                    "graders": [{"id": "g1", "type": "exact_match"}],
                }
            ],
        }
        assert from_openeval(suite) == []


class TestResultToOpeneval:
    def test_copa_result_set_is_valid(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        rs = result_to_openeval(
            "copa",
            samples,
            suite_id="copa_smoke",
            run_id="run-1",
            started_at="2026-08-20T00:00:00Z",
            aggregate=copa_eval["results"]["copa"],
        )
        result = validate_result_set(rs)
        assert result.valid, result.errors

    def test_copa_result_count_matches_doc_count(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        rs = result_to_openeval(
            "copa",
            samples,
            suite_id="copa_smoke",
            run_id="run-1",
            started_at="2026-08-20T00:00:00Z",
        )
        doc_ids = {s["doc_id"] for s in samples}
        assert len(rs["results"]) == len(doc_ids)

    def test_copa_scores_are_real_not_fabricated(self, copa_eval):
        samples = copa_eval["samples"]["copa"]
        rs = result_to_openeval(
            "copa",
            samples,
            suite_id="copa_smoke",
            run_id="run-1",
            started_at="2026-08-20T00:00:00Z",
        )
        real_scores = {s["doc_id"]: float(s["acc"]) for s in samples}
        for result in rs["results"]:
            doc_id = int(result["test_case_id"].rsplit("_", 1)[1])
            gr = result["grader_results"][0]
            assert gr["score"] == real_scores[doc_id]

    def test_aggregate_preserved_verbatim_in_metadata(self, copa_eval):
        real_aggregate = copa_eval["results"]["copa"]
        samples = copa_eval["samples"]["copa"]
        rs = result_to_openeval(
            "copa",
            samples,
            suite_id="copa_smoke",
            run_id="run-1",
            started_at="2026-08-20T00:00:00Z",
            aggregate=real_aggregate,
        )
        assert rs["metadata"]["lm_eval"]["aggregate"] == real_aggregate

    def test_gsm8k_result_set_is_valid_with_multi_filter_graders(self, gsm8k_eval):
        samples = gsm8k_eval["samples"]["gsm8k"]
        rs = result_to_openeval(
            "gsm8k",
            samples,
            suite_id="gsm8k_smoke",
            run_id="run-1",
            started_at="2026-08-20T00:00:00Z",
            aggregate=gsm8k_eval["results"]["gsm8k"],
        )
        result = validate_result_set(rs)
        assert result.valid, result.errors
        # 2 docs, each with 2 filter-scoped grader_results
        assert len(rs["results"]) == 2
        for r in rs["results"]:
            assert len(r["grader_results"]) == 2
            grader_ids = {gr["grader_id"] for gr in r["grader_results"]}
            assert grader_ids == {
                "exact_match__strict-match",
                "exact_match__flexible-extract",
            }

    def test_scores_are_clamped_into_unit_range(self):
        fake_samples = [
            {
                "doc_id": 0,
                "target": "x",
                "arguments": [["prompt", "cont"]],
                "filter": "none",
                "metrics": ["weird_metric"],
                "weird_metric": 4.2,  # out-of-range on purpose
            }
        ]
        rs = result_to_openeval(
            "fake_task",
            fake_samples,
            suite_id="s",
            run_id="r",
            started_at="2026-08-20T00:00:00Z",
        )
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] == 1.0
        assert gr["metadata"]["lm_eval"]["raw_score"] == 4.2

    def test_empty_samples_raises_rather_than_producing_invalid_result_set(self):
        with pytest.raises(ValueError):
            result_to_openeval(
                "t", [], suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z"
            )


class TestEndToEnd:
    def test_full_round_trip_copa(self, copa_eval):
        """to_openeval -> validate -> from_openeval -> result_to_openeval -> validate,
        all against real data from one real evaluation run."""
        samples = copa_eval["samples"]["copa"]

        suite = to_openeval("copa", samples, suite_id="copa_e2e")
        assert validate_suite(suite).valid

        recovered = from_openeval(suite)
        assert {r["doc_id"] for r in recovered} == {s["doc_id"] for s in samples}

        rs = result_to_openeval(
            "copa",
            samples,
            suite_id=suite["id"],
            run_id="run-e2e",
            started_at="2026-08-20T00:00:00Z",
            aggregate=copa_eval["results"]["copa"],
            completed_at="2026-08-20T00:01:00Z",
        )
        rs_result = validate_result_set(rs)
        assert rs_result.valid, rs_result.errors
        assert rs["suite_id"] == suite["id"]
