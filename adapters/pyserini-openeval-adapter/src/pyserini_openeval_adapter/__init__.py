"""Convert pyserini (https://github.com/castorini/pyserini) IR evaluation
results to EvalPort (https://github.com/adhabnr-ux/evalport) ResultSets.

Grounded in pyserini's actual ``pyserini.eval.trec_eval.trec_eval()`` source
(``pyserini/eval/trec_eval.py``), not guessed. That function shells out to a
bundled ``trec_eval`` Java binary via ``jnius``/the JVM, so it cannot run
inside this adapter's own test suite (or most CI environments) without a
full pyserini + JDK install -- this package therefore has **zero import
dependency on pyserini itself**, the same design already used by
``autogen-openeval-adapter`` for a framework whose objects it doesn't want
to require. It works entirely from the plain ``dict`` that
``trec_eval(..., return_per_query_results=True)`` returns.

That return shape, read directly from the real source: ``trec_eval`` prints
lines shaped ``<metric>\\t<qid>\\t<value>``, and the function parses them into
``lines[line.split("\\t")[1]] = float(line.split("\\t")[2])`` -- i.e. the
dict is keyed by **query id** (plus the aggregate pseudo-query id
``"all"``), not by metric name. That's exactly what the adapter proposal
(castorini/pyserini#2651) showed: ``{"301": 0.4231, "302": 0.5510, ...,
"all": 0.4823}`` for one metric.

The real source also documents a genuine footgun this adapter is built
around rather than papering over (see ``trec_eval.py``'s own ``# TODO:
FIXME`` block): because the returned dict is keyed by query id and not by
metric, requesting multiple ``-m`` metrics in one call makes each later
metric's per-query values silently overwrite the earlier ones for the same
query id -- only the last metric's numbers survive. The documented, correct
usage is therefore to call ``trec_eval()`` once per metric and hand each
metric's per-query dict to this adapter separately, which is exactly what
``results_to_openeval()``'s multi-metric form (``metrics={...}``) expects.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["results_to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"

# The aggregate pseudo-query-id trec_eval always emits alongside real query
# ids (see trec_eval.py: the "all" row from trec_eval's own stdout). Never a
# real per-query result, so it becomes ResultSet.summary rather than a
# Result entry.
_AGGREGATE_KEY = "all"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_score(value: float) -> "tuple[float, Optional[float]]":
    """Clamp a trec_eval metric value into EvalPort's required [0, 1] range.

    Most trec_eval measures used per-query (ndcg_cut.*, map, recall.*,
    P.*, mrr, ...) are already fractions in [0, 1]. A few pseudo-metrics
    trec_eval also reports (e.g. ``num_ret``, ``num_rel``, ``num_rel_ret``)
    are raw counts that can exceed 1 -- clamping those without a trace would
    silently misrepresent them, so the unclamped raw value is returned
    alongside whenever clamping actually changed anything, the same
    documented convention ``haystack-openeval-adapter`` uses for its own
    score normalization.
    """
    raw = float(value)
    clamped = max(0.0, min(1.0, raw))
    return (clamped, raw if raw != clamped else None)


def results_to_openeval(
    per_query: Any,
    metric: Optional[str] = None,
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    suite_id: str = "pyserini_suite",
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    pass_threshold: float = 0.0,
    version: str = OPENEVAL_VERSION,
) -> Dict[str, Any]:
    """Convert one or more ``trec_eval(..., return_per_query_results=True)``
    outputs into an EvalPort ResultSet.

    Two calling conventions, matching the two situations the real
    ``trec_eval()`` FIXME above actually produces:

    Single metric (the common case, and the exact shape proposed in
    castorini/pyserini#2651)::

        per_query = trec_eval(['-m', 'ndcg_cut.10', qrels, run],
                               return_per_query_results=True)
        # per_query == {"301": 0.4231, "302": 0.5510, ..., "all": 0.4823}
        result_set = results_to_openeval(per_query, metric="ndcg_cut_10",
                                          suite_id="beir-arguana-test")

    Multiple metrics -- call ``trec_eval()`` once per metric (as its own
    FIXME recommends) and pass every per-query dict together, so each
    query's Result carries one GraderResult per metric instead of only the
    last metric's value overwriting the others::

        per_query_ndcg = trec_eval(['-m', 'ndcg_cut.10', qrels, run],
                                    return_per_query_results=True)
        per_query_map = trec_eval(['-m', 'map', qrels, run],
                                   return_per_query_results=True)
        result_set = results_to_openeval(
            None,
            metrics={"ndcg_cut_10": per_query_ndcg, "map": per_query_map},
            suite_id="beir-arguana-test",
        )

    Args:
        per_query: A ``{query_id: value, "all": aggregate}`` dict for a
            single metric (used together with ``metric``). Pass ``None``
            when using the multi-metric ``metrics=`` form instead.
        metric: The trec_eval measure name ``per_query`` holds values for
            (e.g. ``"ndcg_cut_10"``, ``"map"``, ``"recall_100"``). Required
            when ``per_query`` is given; ignored when ``metrics`` is given.
        metrics: ``{metric_name: {query_id: value, "all": aggregate}, ...}``
            for the multi-metric form. Mutually exclusive with
            ``per_query``/``metric``.
        suite_id: The EvalPort suite/dataset this ResultSet's query ids
            belong to (e.g. a BEIR dataset name).
        run_id: EvalPort ``ResultSet.run_id``; a random one is generated if
            omitted.
        started_at, completed_at: ISO-8601 timestamps. ``started_at``
            defaults to now if omitted (required by the EvalPort schema;
            trec_eval's own output carries no run-level timestamp).
        pass_threshold: A (query, metric) pair "passes" when its value is
            ``> pass_threshold`` (default ``0.0``, i.e. any non-zero
            relevance signal). IR metrics have no universal "good" value --
            override this with a threshold meaningful for your metric and
            collection rather than trusting the default for anything but a
            basic non-zero check.
        version: EvalPort schema version.

    Returns:
        A dict matching EvalPort's ResultSet schema (validate with
        ``openeval.validate.validate_result_set``).

    Raises:
        ValueError: if neither/both of ``per_query``/``metric`` and
            ``metrics`` are given, if a dict has no real per-query entries
            (only ``"all"``), or if the metric dicts in ``metrics`` disagree
            on which query ids are present.
    """
    if metrics is not None:
        if per_query is not None or metric is not None:
            raise ValueError(
                "results_to_openeval: pass either (per_query, metric) for a "
                "single metric or metrics=... for multiple metrics, not both."
            )
        if not metrics:
            raise ValueError("results_to_openeval: metrics is empty.")
        metric_dicts = dict(metrics)
    else:
        if per_query is None or metric is None:
            raise ValueError(
                "results_to_openeval: both per_query and metric are required "
                "when not using the metrics=... multi-metric form."
            )
        metric_dicts = {metric: dict(per_query)}

    query_id_sets = []
    for name, values in metric_dicts.items():
        qids = {k for k in values.keys() if k != _AGGREGATE_KEY}
        if not qids:
            raise ValueError(
                f"results_to_openeval: metric {name!r} has no real per-query "
                f"entries (only {_AGGREGATE_KEY!r}). Was -q passed to trec_eval?"
            )
        query_id_sets.append((name, qids))

    all_qids = query_id_sets[0][1]
    for name, qids in query_id_sets[1:]:
        if qids != all_qids:
            missing = all_qids ^ qids
            raise ValueError(
                "results_to_openeval: metrics disagree on which query ids are "
                f"present (symmetric difference: {sorted(missing)}). Every "
                "metric's trec_eval call must have run against the same "
                "qrels/run pair."
            )

    ordered_qids = sorted(all_qids)
    results_out: List[Dict[str, Any]] = []
    for qid in ordered_qids:
        grader_results: List[Dict[str, Any]] = []
        for name, values in metric_dicts.items():
            normalized, raw = _normalize_score(values[qid])
            gr: Dict[str, Any] = {
                "grader_id": name,
                "type": "custom",
                "score": normalized,
                "passed": normalized > pass_threshold,
                "metadata": {"pyserini": {"metric": name}},
            }
            if raw is not None:
                gr["metadata"]["pyserini"]["raw_score"] = raw
            grader_results.append(gr)

        results_out.append(
            {
                "test_case_id": qid,
                "grader_results": grader_results,
                "passed": all(gr["passed"] for gr in grader_results),
            }
        )

    total = len(results_out)
    passed_count = sum(1 for r in results_out if r["passed"])

    summary: Dict[str, Any] = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": (passed_count / total) if total else 0.0,
    }
    # Preserve trec_eval's own aggregate ("all") value per metric -- this is
    # the number trec_eval itself reports as the collection-level score
    # (e.g. mean nDCG@10 over the whole run), distinct from EvalPort's own
    # pass_rate computed above from the per-threshold per-query "passed".
    aggregates = {
        name: values.get(_AGGREGATE_KEY)
        for name, values in metric_dicts.items()
        if _AGGREGATE_KEY in values
    }
    if aggregates:
        summary["metadata"] = {"pyserini": {"trec_eval_aggregate": aggregates}}

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"pyserini_run_{uuid.uuid4().hex[:12]}",
        "started_at": started_at or _now_iso(),
        "results": results_out,
        "summary": summary,
    }
    if completed_at:
        result_set["completed_at"] = completed_at
    return result_set


def from_openeval(result_set: Dict[str, Any], metric: Optional[str] = None) -> Dict[str, float]:
    """Convert an EvalPort ResultSet back into a trec_eval-shaped per-query
    dict: ``{query_id: value, "all": aggregate}``.

    This is the inverse of the single-metric form of ``results_to_openeval``
    -- useful for handing a ResultSet another tool produced back to code
    that expects ``trec_eval()``'s own return shape (e.g. a pyserini-based
    reporting script).

    Args:
        result_set: An EvalPort ResultSet dict.
        metric: Which grader_id's score to extract per query, when a
            Result carries more than one (the multi-metric case). Required
            whenever any Result has more than one grader_result; optional
            (and inferred) when every Result has exactly one.

    Returns:
        ``{query_id: score, "all": mean_of_per_query_scores}`` -- the "all"
        aggregate is recomputed as the mean of the extracted per-query
        scores (EvalPort's ResultSet doesn't require the original
        trec_eval-reported aggregate to be preserved verbatim; if it was,
        prefer ``result_set["summary"]["metadata"]["pyserini"]
        ["trec_eval_aggregate"][metric]`` for the exact original value).

    Raises:
        ValueError: if results is empty, a Result has no grader_results, or
            ``metric`` is required (multiple grader_results per Result) but
            not given / not found.
    """
    results = result_set.get("results") or []
    if not results:
        raise ValueError("from_openeval: result_set has no results to convert.")

    per_query: Dict[str, float] = {}
    for r in results:
        grs = r.get("grader_results") or []
        if not grs:
            raise ValueError(
                f"from_openeval: result {r.get('test_case_id')!r} has no "
                "grader_results."
            )
        if metric is not None:
            matches = [gr for gr in grs if gr.get("grader_id") == metric]
            if not matches:
                raise ValueError(
                    f"from_openeval: metric {metric!r} not found in result "
                    f"{r.get('test_case_id')!r}'s grader_results."
                )
            chosen = matches[0]
        elif len(grs) == 1:
            chosen = grs[0]
        else:
            raise ValueError(
                f"from_openeval: result {r.get('test_case_id')!r} has "
                f"{len(grs)} grader_results; pass metric=... to disambiguate."
            )
        per_query[r["test_case_id"]] = chosen["score"]

    per_query[_AGGREGATE_KEY] = sum(per_query.values()) / len(per_query) if per_query else 0.0
    return per_query
