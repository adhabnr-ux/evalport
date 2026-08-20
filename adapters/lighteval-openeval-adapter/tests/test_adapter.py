"""Tests for lighteval_openeval_adapter.

Every test that touches lighteval runs a real `Pipeline.evaluate()` call
against lighteval's own built-in `dummy` model (no real model weights, but a
genuine evaluation loop against real tasks/documents pulled live from the
Hugging Face Hub). Every produced Suite/ResultSet is validated against the
real `openeval.validate.validate_suite()` / `validate_result_set()` -- not
mocks, not hand-rolled shape assertions alone.

Requires `xxhash<4.0` (see this package's README and module docstring for
why -- a real bug in lighteval's own `DetailsLogger.log()`, unrelated to
this adapter's code, that otherwise crashes every `Pipeline.evaluate()`
call on a fresh `pip install lighteval` today).
"""

from __future__ import annotations

import pytest

lighteval = pytest.importorskip(
    "lighteval", reason="install with: pip install -e '.[lighteval]' or '.[test]' (needs xxhash<4.0, see README)"
)

from openeval.validate import validate_result_set, validate_suite  # noqa: E402

from lighteval_openeval_adapter import (  # noqa: E402
    from_openeval,
    result_to_openeval,
    to_openeval,
)


def _run_pipeline(task, max_samples, out_dir):
    """Real, minimal end-to-end lighteval run: dummy model, a real task
    pulled live from the Hub, log_details implicit via get_details()."""
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    evaluation_tracker = EvaluationTracker(
        output_dir=str(out_dir),
        save_details=False,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=max_samples,
        bootstrap_iters=2,
    )
    model = DummyModel(DummyModelConfig())
    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=model,
    )
    pipeline.evaluate()
    details = pipeline.get_details()[task]
    aggregate = pipeline.get_results()["results"].get(task.replace("|", ":"))
    return details, aggregate


@pytest.fixture(scope="module")
def hellaswag_run(tmp_path_factory):
    """Real run: dummy model, real 'hellaswag' task, 3 documents.
    Exercises the exact_match ('em') native-grader path, generative
    scoring (model_response.text) on a Doc.choices/gold_index-shaped task."""
    out = tmp_path_factory.mktemp("hellaswag_out")
    return _run_pipeline("hellaswag|0", 3, out)


@pytest.fixture(scope="module")
def gsm8k_run(tmp_path_factory):
    """Real run: dummy model, real 'gsm8k' task, 2 documents. Exercises the
    custom-grader path (metric_name='extractive_match', no EvalPort-native
    equivalent) and lighteval's inspect_ai-solver-backed task path."""
    out = tmp_path_factory.mktemp("gsm8k_out")
    return _run_pipeline("gsm8k|0", 2, out)


class TestToOpenevalMultipleChoice:
    def test_hellaswag_produces_valid_suite(self, hellaswag_run):
        details, _ = hellaswag_run
        suite = to_openeval("hellaswag|0", details, suite_id="hellaswag_smoke")
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_hellaswag_test_case_count_matches_doc_count(self, hellaswag_run):
        details, _ = hellaswag_run
        suite = to_openeval("hellaswag|0", details)
        assert len(suite["test_cases"]) == len(details)

    def test_hellaswag_grader_is_native_exact_match(self, hellaswag_run):
        details, _ = hellaswag_run
        suite = to_openeval("hellaswag|0", details)
        graders = {g["id"]: g for g in suite["graders"]}
        assert "em" in graders
        assert graders["em"]["type"] == "exact_match"
        # exact_match has zero required params per spec/schemas/grader.json
        assert "params" not in graders["em"]

    def test_hellaswag_input_is_real_prompt_text(self, hellaswag_run):
        details, _ = hellaswag_run
        suite = to_openeval("hellaswag|0", details)
        real_queries = {d.doc.id: d.doc.query for d in details}
        for tc in suite["test_cases"]:
            doc_id = tc["metadata"]["lighteval"]["doc_id"]
            assert tc["input"] == real_queries[doc_id]
            assert tc["input"] != ""

    def test_hellaswag_expected_output_is_real_gold_choice(self, hellaswag_run):
        details, _ = hellaswag_run
        suite = to_openeval("hellaswag|0", details)
        by_doc_id = {d.doc.id: d.doc for d in details}
        for tc in suite["test_cases"]:
            doc_id = tc["metadata"]["lighteval"]["doc_id"]
            doc = by_doc_id[doc_id]
            if isinstance(doc.gold_index, int) and doc.gold_index >= 0:
                assert tc["expected_output"] == doc.choices[doc.gold_index]

    def test_hellaswag_metadata_preserves_real_doc_fields(self, hellaswag_run):
        details, _ = hellaswag_run
        suite = to_openeval("hellaswag|0", details)
        tc = suite["test_cases"][0]
        meta = tc["metadata"]["lighteval"]
        assert meta["task_name"] == "hellaswag|0"
        assert meta["choices"] == details[0].doc.choices
        assert meta["gold_index"] == details[0].doc.gold_index
        assert "instruction" in meta


class TestToOpenevalGenerationCustomGrader:
    def test_gsm8k_produces_valid_suite(self, gsm8k_run):
        details, _ = gsm8k_run
        suite = to_openeval("gsm8k|0", details, suite_id="gsm8k_smoke")
        result = validate_suite(suite)
        assert result.valid, result.errors

    def test_gsm8k_grader_is_custom_with_real_metric_name(self, gsm8k_run):
        details, _ = gsm8k_run
        suite = to_openeval("gsm8k|0", details)
        graders = {g["id"]: g for g in suite["graders"]}
        assert "extractive_match" in graders
        grader = graders["extractive_match"]
        assert grader["type"] == "custom"
        assert grader["params"]["handler"] == "lighteval:extractive_match"


class TestFromOpeneval:
    def test_round_trips_query_and_doc_id(self, hellaswag_run):
        details, _ = hellaswag_run
        suite = to_openeval("hellaswag|0", details)
        recovered = from_openeval(suite)
        assert len(recovered) == len(details)
        by_doc_id = {r["doc_id"]: r for r in recovered}
        for d in details:
            r = by_doc_id[d.doc.id]
            assert r["query"] == d.doc.query
            assert r["task_name"] == "hellaswag|0"
            assert r["choices"] == d.doc.choices

    def test_clean_skips_test_cases_without_lighteval_metadata(self):
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
    def test_hellaswag_result_set_is_valid(self, hellaswag_run):
        details, aggregate = hellaswag_run
        rs = result_to_openeval(
            "hellaswag|0",
            details,
            suite_id="hellaswag_smoke",
            run_id="run-1",
            started_at="2026-08-20T00:00:00Z",
            aggregate=aggregate,
        )
        result = validate_result_set(rs)
        assert result.valid, result.errors

    def test_hellaswag_result_count_matches_doc_count(self, hellaswag_run):
        details, _ = hellaswag_run
        rs = result_to_openeval(
            "hellaswag|0", details, suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z",
        )
        assert len(rs["results"]) == len(details)

    def test_hellaswag_scores_are_real_not_fabricated(self, hellaswag_run):
        details, _ = hellaswag_run
        rs = result_to_openeval(
            "hellaswag|0", details, suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z",
        )
        real_scores = {d.doc.id: float(d.metric["em"]) for d in details}
        by_tc = {r["test_case_id"]: r for r in rs["results"]}
        suite = to_openeval("hellaswag|0", details)
        for tc in suite["test_cases"]:
            doc_id = tc["metadata"]["lighteval"]["doc_id"]
            result = by_tc[tc["id"]]
            gr = next(g for g in result["grader_results"] if g["grader_id"] == "em")
            assert gr["score"] == real_scores[doc_id]

    def test_aggregate_preserved_verbatim_in_metadata(self, hellaswag_run):
        details, aggregate = hellaswag_run
        assert aggregate is not None
        rs = result_to_openeval(
            "hellaswag|0", details, suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z",
            aggregate=aggregate,
        )
        expected = {k: float(v) for k, v in aggregate.items()}
        assert rs["metadata"]["lighteval"]["aggregate"] == expected

    def test_gsm8k_result_set_is_valid_with_custom_grader(self, gsm8k_run):
        details, aggregate = gsm8k_run
        rs = result_to_openeval(
            "gsm8k|0", details, suite_id="gsm8k_smoke", run_id="run-1",
            started_at="2026-08-20T00:00:00Z", aggregate=aggregate,
        )
        result = validate_result_set(rs)
        assert result.valid, result.errors
        assert len(rs["results"]) == len(details)

    def test_scores_are_clamped_into_unit_range(self):
        class _FakeDoc:
            id = 0

        class _FakeDetail:
            doc = _FakeDoc()
            metric = {"weird_metric": 4.2}  # out-of-range on purpose

        rs = result_to_openeval(
            "fake_task", [_FakeDetail()], suite_id="s", run_id="r",
            started_at="2026-08-20T00:00:00Z",
        )
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] == 1.0
        assert gr["metadata"]["lighteval"]["raw_score"] == 4.2

    def test_empty_details_raises_rather_than_producing_invalid_result_set(self):
        with pytest.raises(ValueError):
            result_to_openeval(
                "t", [], suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z",
            )

    def test_duplicate_doc_ids_get_distinct_ids_not_collapsed(self):
        class _FakeDoc:
            def __init__(self, doc_id):
                self.id = doc_id

        class _FakeDetail:
            def __init__(self, doc_id, score):
                self.doc = _FakeDoc(doc_id)
                self.metric = {"em": score}

        details = [_FakeDetail(1, 1), _FakeDetail(1, 0)]  # same doc, two seeds
        suite = to_openeval("dup_task", details)
        ids = [tc["id"] for tc in suite["test_cases"]]
        assert len(set(ids)) == 2
        rs = result_to_openeval(
            "dup_task", details, suite_id="s", run_id="r", started_at="2026-08-20T00:00:00Z",
        )
        assert {r["test_case_id"] for r in rs["results"]} == set(ids)


class TestEndToEnd:
    def test_full_round_trip_hellaswag(self, hellaswag_run):
        """to_openeval -> validate -> from_openeval -> result_to_openeval ->
        validate, all against real data from one real evaluation run."""
        details, aggregate = hellaswag_run

        suite = to_openeval("hellaswag|0", details, suite_id="hellaswag_e2e")
        assert validate_suite(suite).valid

        recovered = from_openeval(suite)
        assert {r["doc_id"] for r in recovered} == {d.doc.id for d in details}

        rs = result_to_openeval(
            "hellaswag|0", details, suite_id=suite["id"], run_id="run-e2e",
            started_at="2026-08-20T00:00:00Z", aggregate=aggregate,
            completed_at="2026-08-20T00:01:00Z",
        )
        rs_result = validate_result_set(rs)
        assert rs_result.valid, rs_result.errors
        assert rs["suite_id"] == suite["id"]
