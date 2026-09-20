"""Tests for pyserini_openeval_adapter.

This adapter has zero import dependency on pyserini itself (see the module
docstring for why: ``trec_eval()`` shells out to a bundled Java binary via
the JVM, so it can't run in a normal test sandbox). What's tested instead is
the actual, documented output *shape* of
``pyserini.eval.trec_eval.trec_eval(..., return_per_query_results=True)``,
read directly from pyserini's real ``pyserini/eval/trec_eval.py`` source
(``lines[line.split("\\t")[1]] = float(line.split("\\t")[2])`` -- a dict
keyed by query id, plus the aggregate ``"all"`` key) and from the exact
example given in the upstream adapter proposal
(castorini/pyserini#2651): ``{"301": 0.4231, "302": 0.5510, ..., "all":
0.4823}``. Every test here validates its output against the real
``evalport-sdk`` (``openeval.validate.validate_result_set``), not a mock.
"""
from __future__ import annotations

import pytest
from openeval.validate import validate_result_set

from pyserini_openeval_adapter import from_openeval, results_to_openeval


# ---------------------------------------------------------------------------
# Fixtures -- shaped exactly like real trec_eval(return_per_query_results=True)
# output, per the upstream proposal and pyserini's own source.
# ---------------------------------------------------------------------------


@pytest.fixture
def ndcg_per_query():
    # Verbatim from castorini/pyserini#2651's own worked example.
    return {"301": 0.4231, "302": 0.5510, "303": 0.0000, "all": 0.3247}


@pytest.fixture
def map_per_query():
    return {"301": 0.3011, "302": 0.6102, "303": 0.1000, "all": 0.3371}


class TestSingleMetric:
    def test_basic_shape_and_validates(self, ndcg_per_query):
        result_set = results_to_openeval(
            ndcg_per_query, metric="ndcg_cut_10", suite_id="beir-arguana-test"
        )
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        assert result_set["suite_id"] == "beir-arguana-test"
        assert len(result_set["results"]) == 3  # "all" excluded from results
        assert {r["test_case_id"] for r in result_set["results"]} == {"301", "302", "303"}

    def test_grader_id_is_metric_name(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        for r in result_set["results"]:
            assert len(r["grader_results"]) == 1
            gr = r["grader_results"][0]
            assert gr["grader_id"] == "ndcg_cut_10"
            assert gr["type"] == "custom"

    def test_scores_preserved_exactly(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        by_id = {r["test_case_id"]: r["grader_results"][0]["score"] for r in result_set["results"]}
        assert by_id == {"301": 0.4231, "302": 0.5510, "303": 0.0}

    def test_pass_threshold_default_is_nonzero(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        by_id = {r["test_case_id"]: r["passed"] for r in result_set["results"]}
        # default pass_threshold=0.0: any nonzero relevance signal passes
        assert by_id == {"301": True, "302": True, "303": False}

    def test_custom_pass_threshold(self, ndcg_per_query):
        result_set = results_to_openeval(
            ndcg_per_query, metric="ndcg_cut_10", suite_id="s", pass_threshold=0.5
        )
        by_id = {r["test_case_id"]: r["passed"] for r in result_set["results"]}
        assert by_id == {"301": False, "302": True, "303": False}

    def test_summary_computed_correctly(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        assert result_set["summary"]["total"] == 3
        assert result_set["summary"]["passed"] == 2
        assert result_set["summary"]["failed"] == 1
        assert result_set["summary"]["pass_rate"] == pytest.approx(2 / 3)

    def test_trec_eval_aggregate_preserved_in_summary_metadata(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        assert result_set["summary"]["metadata"]["pyserini"]["trec_eval_aggregate"] == {
            "ndcg_cut_10": 0.3247
        }

    def test_explicit_run_id_and_timestamps(self, ndcg_per_query):
        result_set = results_to_openeval(
            ndcg_per_query,
            metric="ndcg_cut_10",
            suite_id="s",
            run_id="fixed_run",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:05:00Z",
        )
        assert result_set["run_id"] == "fixed_run"
        assert result_set["started_at"] == "2026-01-01T00:00:00Z"
        assert result_set["completed_at"] == "2026-01-01T00:05:00Z"

    def test_default_run_id_generated(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        assert result_set["run_id"].startswith("pyserini_run_")
        assert "completed_at" not in result_set

    def test_missing_metric_raises(self, ndcg_per_query):
        with pytest.raises(ValueError, match="both per_query and metric are required"):
            results_to_openeval(ndcg_per_query, suite_id="s")

    def test_no_real_queries_raises(self):
        with pytest.raises(ValueError, match="no real per-query entries"):
            results_to_openeval({"all": 0.5}, metric="map", suite_id="s")

    def test_both_forms_at_once_raises(self, ndcg_per_query):
        with pytest.raises(ValueError, match="not both"):
            results_to_openeval(
                ndcg_per_query, metric="ndcg_cut_10", metrics={"map": ndcg_per_query}, suite_id="s"
            )


class TestMultipleMetrics:
    """The FIXME case: trec_eval() is called once per metric and results
    are merged, rather than losing every metric but the last one -- see the
    module docstring's explanation of the real trec_eval.py bug this design
    works around."""

    def test_multi_metric_produces_one_grader_result_per_metric(
        self, ndcg_per_query, map_per_query
    ):
        result_set = results_to_openeval(
            None,
            metrics={"ndcg_cut_10": ndcg_per_query, "map": map_per_query},
            suite_id="beir-arguana-test",
        )
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        for r in result_set["results"]:
            assert len(r["grader_results"]) == 2
            grader_ids = {gr["grader_id"] for gr in r["grader_results"]}
            assert grader_ids == {"ndcg_cut_10", "map"}

    def test_multi_metric_scores_are_independent(self, ndcg_per_query, map_per_query):
        result_set = results_to_openeval(
            None, metrics={"ndcg_cut_10": ndcg_per_query, "map": map_per_query}, suite_id="s"
        )
        r301 = next(r for r in result_set["results"] if r["test_case_id"] == "301")
        scores = {gr["grader_id"]: gr["score"] for gr in r301["grader_results"]}
        assert scores == {"ndcg_cut_10": 0.4231, "map": 0.3011}

    def test_multi_metric_passed_is_and_of_all_metrics(self, ndcg_per_query, map_per_query):
        result_set = results_to_openeval(
            None,
            metrics={"ndcg_cut_10": ndcg_per_query, "map": map_per_query},
            suite_id="s",
            pass_threshold=0.4,
        )
        # 301: ndcg=0.4231 (pass), map=0.3011 (fail) -> overall fail
        # 302: ndcg=0.5510 (pass), map=0.6102 (pass) -> overall pass
        by_id = {r["test_case_id"]: r["passed"] for r in result_set["results"]}
        assert by_id["301"] is False
        assert by_id["302"] is True

    def test_empty_metrics_dict_raises(self):
        with pytest.raises(ValueError, match="metrics is empty"):
            results_to_openeval(None, metrics={}, suite_id="s")

    def test_disagreeing_query_ids_raises(self, ndcg_per_query):
        mismatched = {"301": 0.1, "304": 0.2, "all": 0.15}
        with pytest.raises(ValueError, match="disagree on which query ids"):
            results_to_openeval(
                None, metrics={"ndcg_cut_10": ndcg_per_query, "map": mismatched}, suite_id="s"
            )


class TestRawScoreClamping:
    def test_count_metric_exceeding_one_is_clamped_with_raw_preserved(self):
        # num_rel_ret is a raw count, not a [0,1] fraction -- a real trec_eval
        # pseudo-metric this adapter must not silently misrepresent.
        per_query = {"301": 15.0, "302": 3.0, "all": 9.0}
        result_set = results_to_openeval(per_query, metric="num_rel_ret", suite_id="s")
        assert validate_result_set(result_set).valid
        gr301 = result_set["results"][0]["grader_results"][0]
        assert gr301["score"] == 1.0  # clamped
        assert gr301["metadata"]["pyserini"]["raw_score"] == 15.0

    def test_in_range_score_has_no_raw_score_metadata(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        for r in result_set["results"]:
            assert "raw_score" not in r["grader_results"][0]["metadata"]["pyserini"]


class TestFromOpenEval:
    def test_round_trip_single_metric(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        recovered = from_openeval(result_set)
        assert recovered["301"] == 0.4231
        assert recovered["302"] == 0.5510
        assert recovered["303"] == 0.0
        assert recovered["all"] == pytest.approx((0.4231 + 0.5510 + 0.0) / 3)

    def test_round_trip_multi_metric_requires_explicit_metric(
        self, ndcg_per_query, map_per_query
    ):
        result_set = results_to_openeval(
            None, metrics={"ndcg_cut_10": ndcg_per_query, "map": map_per_query}, suite_id="s"
        )
        with pytest.raises(ValueError, match="pass metric=... to disambiguate"):
            from_openeval(result_set)

    def test_round_trip_multi_metric_with_explicit_metric(self, ndcg_per_query, map_per_query):
        result_set = results_to_openeval(
            None, metrics={"ndcg_cut_10": ndcg_per_query, "map": map_per_query}, suite_id="s"
        )
        recovered = from_openeval(result_set, metric="map")
        assert recovered["301"] == 0.3011
        assert recovered["302"] == 0.6102

    def test_unknown_metric_raises(self, ndcg_per_query):
        result_set = results_to_openeval(ndcg_per_query, metric="ndcg_cut_10", suite_id="s")
        with pytest.raises(ValueError, match="not found"):
            from_openeval(result_set, metric="nonexistent")

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="no results"):
            from_openeval({"version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "t", "results": []})
