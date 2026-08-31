"""ragrank <-> EvalPort adapter.

Standalone converter between ragrank evaluation results
(`ragrank.evaluation.outputs.EvalResult`) and the EvalPort interchange
format (https://github.com/adhabnr-ux/evalport).

Background: izam-mohammed/ragrank#63. The maintainer confirmed this should
live as a separate package rather than in ragrank core, since the roadmap
direction there is a framework-agnostic JSONL interchange
(`Dataset.to_records()`/`to_json()`/`to_jsonl()`) that an EvalPort adapter
would sit on top of, not inside.

Two design points called out in that thread, resolved here:

1. Null scores. `EvalResult.scores[m][i]` is `float | None` -- `None` marks
   a row a metric could not score. This lands on a slot EvalPort's
   `ResultSet` schema already reserves: Validation Rule 6 says a
   score-less `GraderResult` is `score: null, passed: false`, explicitly
   excluded from pass-rate/aggregate denominators. `MetricResult.error`
   (a string, when `results` detail is available) is carried into
   `GraderResult.reason` rather than dropped.

2. Per-metric score_range normalization. `BaseMetric.score_range` is
   configurable per metric (default `(0.0, 1.0)`, but not fixed there --
   a Likert-style custom metric might use `(1.0, 5.0)`). EvalPort's
   `GraderResult.score` is hard-clamped to `[0.0, 1.0]` (Rule 5), with no
   `score_range` extension: every source scale must be normalized before
   it is a valid document. This adapter linearly rescales
   `(raw - low) / (high - low)` and preserves the original value in the
   reserved `metadata.openeval.raw_score` key (Appendix B) whenever a
   metric's `score_range` isn't already the unit interval.

3. test_case_id referential integrity (open question at the time of the
   last update to that issue). EvalPort Rule 2 requires every
   `Result.test_case_id` in a `ResultSet` to reference a real `TestCase.id`
   in a paired `Suite` -- ragrank's `Dataset`/`DataNode` has no native
   `id` concept, so nothing in `EvalResult` naturally supplies one.
   Resolution: `to_openeval()` defaults to *also* emitting a minimal
   synthetic `Suite` alongside the `ResultSet`, using positional ids
   (`tc_0`, `tc_1`, ...) generated identically in both documents -- so
   referential integrity holds by construction, not by convention. A
   caller who already has a real suite with its own ids can pass
   `test_case_ids=[...]` explicitly; in that case no synthetic suite is
   built and `to_openeval()` returns `suite: None`, and the caller is
   responsible for pairing the returned `result_set` with their real
   suite. See README.md for the full reasoning.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at
    # runtime, but keep a sane fallback for static analysis / partial
    # installs, same pattern as the other adapters in this repo.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    ragrank's own objects (`EvalResult`, `DataNode`, `BaseMetric`,
    `MetricResult`) are pydantic models with plain attributes, but every
    accessor in this module goes through here anyway -- so a hand-built
    stand-in (a `SimpleNamespace`, a dict with the same keys) works for
    testing or for a caller who does not want a hard `ragrank` import.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    """Turn a metric's display name into a stable, machine-safe grader id."""
    slug = _SLUG_RE.sub("_", name.strip().lower()).strip("_")
    return slug or "metric"


def _normalize_score(raw: float, score_range: Tuple[float, float]) -> float:
    """Linearly rescale `raw` from `score_range` into `[0.0, 1.0]`.

    A degenerate range (`low == high`) can't be divided into -- treated as
    "at or above the single valid value counts as a pass", which is the
    only sane reading of a one-point range.
    """
    low, high = score_range
    if high == low:
        return 1.0 if raw >= high else 0.0
    value = (raw - low) / (high - low)
    return max(0.0, min(1.0, value))


def _metric_grader(metric: Any) -> Dict[str, Any]:
    """Build the EvalPort `Grader` dict for one ragrank metric.

    ragrank metrics don't map onto EvalPort's well-known grader types
    (`exact_match`, `llm_judge`, ...) in any generic, reliable way -- a
    `DeterministicMetric` subclass or an `LLMJudge` with an arbitrary
    rubric could be almost anything. So every ragrank metric becomes a
    `custom` grader, identified by a `ragrank:<slug>` handler, per the
    spec's type-openness rule (custom/unrecognized types require
    `params.handler` so a runner that doesn't know ragrank can skip the
    grader gracefully instead of guessing at its semantics).
    """
    name = _get(metric, "name")
    score_range = tuple(_get(metric, "score_range", (0.0, 1.0)))
    threshold = _get(metric, "threshold")
    metric_type = _get(metric, "metric_type")
    metric_type_name = getattr(metric_type, "value", metric_type)

    parts = [f"ragrank {metric_type_name or 'metric'}"]
    if score_range != (0.0, 1.0):
        parts.append(
            f"native score_range {list(score_range)}, "
            "normalized to [0.0, 1.0]"
        )
    if threshold is not None:
        parts.append(f"native threshold {threshold}")

    return {
        "id": _slug(name),
        "type": "custom",
        "params": {"handler": f"ragrank:{_slug(name)}"},
        "description": "; ".join(parts),
    }


def _test_case(node: Any, index: int) -> Dict[str, Any]:
    """Build the EvalPort `TestCase` dict for one ragrank `DataNode`."""
    tc: Dict[str, Any] = {
        "id": f"tc_{index}",
        "input": _get(node, "question", ""),
        "graders": [],  # filled in by the caller, which knows the metrics
    }
    context = _get(node, "context")
    if context:
        # ragrank's `context` *is* retrieved-document context (this is a
        # RAG evaluation library) -- `retrieval_context` is the more
        # precise EvalPort field for it than the generic `context`.
        tc["retrieval_context"] = list(context)
    reference = _get(node, "reference")
    if reference is not None:
        tc["expected_output"] = reference

    metadata: Dict[str, Any] = {}
    retrieved_ids = _get(node, "retrieved_ids")
    if retrieved_ids is not None:
        metadata["retrieved_ids"] = list(retrieved_ids)
    reference_ids = _get(node, "reference_ids")
    if reference_ids is not None:
        metadata["reference_ids"] = list(reference_ids)
    if metadata:
        tc["metadata"] = metadata

    return tc


def _dataset_rows(dataset: Any) -> List[Any]:
    """Iterate a ragrank `Dataset` (or anything else that iterates DataNodes)."""
    return list(dataset)


def build_suite(
    result: Any,
    *,
    suite_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the minimal synthetic `Suite` paired with `result`'s `ResultSet`.

    Exposed separately from `to_openeval()` in case a caller wants the
    suite (e.g. to write it to `suite.json`) without recomputing scores.
    Test case ids are positional (`tc_0`, `tc_1`, ...), matching the ids
    `to_openeval()` uses for the paired `ResultSet.results[].test_case_id`.
    """
    dataset = _get(result, "dataset")
    metrics = _get(result, "metrics", [])
    rows = _dataset_rows(dataset)

    graders = [_metric_grader(metric) for metric in metrics]
    grader_ids = [g["id"] for g in graders]

    test_cases = []
    for i, node in enumerate(rows):
        tc = _test_case(node, i)
        tc["graders"] = list(grader_ids)
        test_cases.append(tc)

    llm = _get(result, "llm")
    llm_name = _get(llm, "name") if llm is not None else None
    resolved_id = suite_id or f"ragrank_suite_{uuid.uuid4().hex[:12]}"

    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": resolved_id,
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "ragrank"}},
    }
    if name:
        suite["name"] = name
    elif llm_name:
        suite["name"] = f"ragrank evaluation ({llm_name})"
    return suite


def _grader_result(metric: Any, raw: Any, detail: Any) -> Dict[str, Any]:
    """Build one `GraderResult` for one (metric, row) pair."""
    grader_id = _slug(_get(metric, "name"))
    score_range = tuple(_get(metric, "score_range", (0.0, 1.0)))
    threshold = _get(metric, "threshold")

    metadata: Dict[str, Any] = {}
    detail_metadata = _get(detail, "metadata") if detail is not None else None
    if detail_metadata:
        metadata["ragrank"] = dict(detail_metadata)

    if raw is None:
        # Rule 6: a score-less grader result is score: null, passed: false,
        # not counted as a scored failure. `error` (when the full
        # per-row `results` detail is available) explains why.
        reason = _get(detail, "error") if detail is not None else None
        return _drop_empty(
            {
                "grader_id": grader_id,
                "type": "custom",
                "score": None,
                "passed": False,
                "reason": reason,
                "metadata": metadata,
            },
            keep=("score",),
        )

    score = _normalize_score(float(raw), score_range)
    if score_range != (0.0, 1.0):
        metadata.setdefault("openeval", {})["raw_score"] = raw

    if threshold is not None:
        passed = bool(raw >= threshold)
    else:
        # No threshold means ragrank never fails this metric on its own
        # terms -- a produced score is treated as a pass. Documented in
        # README.md: this is the one place this adapter has to invent a
        # convention ragrank itself doesn't state, since EvalPort's
        # `GraderResult.passed` is a required boolean with no "n/a" value.
        passed = True

    reason = _get(detail, "reason") if detail is not None else None

    return _drop_empty({
        "grader_id": grader_id,
        "type": "custom",
        "score": score,
        "passed": passed,
        "reason": reason,
        "metadata": metadata,
    })


def _drop_empty(d: Dict[str, Any], *, keep: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Drop `None`/empty-dict optional values so output stays minimal.

    `keep` names required keys that must survive even when their value is
    `None` -- notably `score`, where `None` is a meaningful, required
    value (Rule 6), not an absent optional field.
    """
    return {
        k: v
        for k, v in d.items()
        if k in keep or not (v is None or (isinstance(v, dict) and not v))
    }


def build_result_set(
    result: Any,
    *,
    suite_id: str,
    run_id: Optional[str] = None,
    test_case_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the EvalPort `ResultSet` for a ragrank `EvalResult`.

    `suite_id` must be the id of whatever `Suite` these results are meant
    to be paired with (Rule 2) -- either one built by `build_suite()` /
    `to_openeval()`, or a real pre-existing suite the caller supplies.
    """
    dataset = _get(result, "dataset")
    metrics = _get(result, "metrics", [])
    scores = _get(result, "scores", [])
    detail_rows = _get(result, "results")  # list[list[MetricResult]] | None
    rows = _dataset_rows(dataset)

    ids = test_case_ids or [f"tc_{i}" for i in range(len(rows))]
    if len(ids) != len(rows):
        raise ValueError(
            f"test_case_ids has {len(ids)} entries but the dataset has "
            f"{len(rows)} rows; they must match 1:1."
        )

    results: List[Dict[str, Any]] = []
    scored_count = 0
    passed_count = 0
    score_total = 0.0
    duration_total_ms = 0
    by_grader: Dict[str, Dict[str, Any]] = {}

    for i, node in enumerate(rows):
        grader_results = []
        row_duration_ms = 0
        for m, metric in enumerate(metrics):
            raw = scores[m][i] if m < len(scores) and i < len(scores[m]) else None
            detail = (
                detail_rows[m][i]
                if detail_rows is not None
                and m < len(detail_rows)
                and i < len(detail_rows[m])
                else None
            )
            gr = _grader_result(metric, raw, detail)
            grader_results.append(gr)

            gid = gr["grader_id"]
            stats = by_grader.setdefault(
                gid,
                {
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "_score_sum": 0.0,
                    "_scored": 0,
                },
            )
            # Rule 6: a null-score GraderResult is "not verified", not a
            # scored failure -- it must not be folded into passed/failed
            # counts or the avg_score denominator, here or below.
            if gr["score"] is not None:
                stats["_score_sum"] += gr["score"]
                stats["_scored"] += 1
                score_total += gr["score"]
                scored_count += 1
                if gr["passed"]:
                    stats["passed"] += 1
                    passed_count += 1
                else:
                    stats["failed"] += 1
            else:
                stats["skipped"] += 1

            process_time = _get(detail, "process_time") if detail is not None else None
            if process_time is not None:
                row_duration_ms += int(process_time * 1000)

        row_passed = all(gr["passed"] for gr in grader_results)
        all_null = all(gr["score"] is None for gr in grader_results)

        row: Dict[str, Any] = {
            "test_case_id": ids[i],
            "actual_output": _get(node, "response", ""),
            "grader_results": grader_results,
            "passed": row_passed,
        }
        if row_duration_ms:
            row["duration_ms"] = row_duration_ms
            duration_total_ms += row_duration_ms
        if all_null:
            row["metadata"] = {"openeval": {"aggregation_status": "unscored"}}
        results.append(row)

    by_grader_summary = {
        gid: {
            "passed": stats["passed"],
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "avg_score": (
                stats["_score_sum"] / stats["_scored"]
                if stats["_scored"]
                else None
            ),
        }
        for gid, stats in by_grader.items()
    }

    total = len(results)
    summary = {
        "total": total,
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "skipped": 0,
        "pass_rate": (passed_count / scored_count) if scored_count else None,
        "avg_score": (score_total / scored_count) if scored_count else None,
        "duration_ms": duration_total_ms or None,
        "by_grader": by_grader_summary,
    }

    response_time = _get(result, "response_time")
    started_at = datetime.now(timezone.utc)
    completed_at = (
        started_at + timedelta(seconds=response_time)
        if isinstance(response_time, (int, float))
        else started_at
    )

    llm = _get(result, "llm")
    provider = {"model": _get(llm, "name")} if llm is not None else None

    usage = _get(result, "usage")
    ragrank_meta: Dict[str, Any] = {"source": "ragrank"}
    if response_time is not None:
        ragrank_meta["response_time"] = response_time
    if usage is not None:
        ragrank_meta["usage"] = {
            "prompt_tokens": _get(usage, "prompt_tokens"),
            "response_tokens": _get(usage, "response_tokens"),
            "calls": _get(usage, "calls"),
        }

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id or f"ragrank_run_{uuid.uuid4().hex[:12]}",
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "results": results,
        "summary": _drop_empty(dict(summary)),
        "runner": {"name": "ragrank"},
        "metadata": {"ragrank": ragrank_meta},
    }
    if provider and provider["model"]:
        result_set["provider"] = provider
    return result_set


def to_openeval(
    result: Any,
    *,
    suite_id: Optional[str] = None,
    run_id: Optional[str] = None,
    test_case_ids: Optional[List[str]] = None,
    suite_name: Optional[str] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Convert a ragrank `EvalResult` to EvalPort documents.

    Returns `{"suite": ..., "result_set": ...}`.

    By default (`test_case_ids=None`) this also builds a minimal synthetic
    `Suite` (see `build_suite()`) so the returned `result_set`'s
    `test_case_id`s are guaranteed to satisfy EvalPort's referential
    integrity rule (Rule 2) against the returned `suite`, with no action
    required from the caller.

    Pass `test_case_ids` (and, ordinarily, `suite_id`) when you already
    have a real EvalPort suite you want these results paired with instead
    -- in that case `suite` in the return value is `None`, and pairing
    the `result_set` with your real suite (matching ids, matching
    `suite_id`) is the caller's responsibility.
    """
    if test_case_ids is not None:
        if suite_id is None:
            raise ValueError(
                "suite_id is required when test_case_ids is supplied "
                "explicitly -- there is no synthetic suite to take an id "
                "from."
            )
        result_set = build_result_set(
            result,
            suite_id=suite_id,
            run_id=run_id,
            test_case_ids=test_case_ids,
        )
        return {"suite": None, "result_set": result_set}

    suite = build_suite(result, suite_id=suite_id, name=suite_name)
    result_set = build_result_set(
        result, suite_id=suite["id"], run_id=run_id
    )
    return {"suite": suite, "result_set": result_set}


def from_openeval(suite: Dict[str, Any], result_set: Optional[Dict[str, Any]] = None) -> Any:
    """Convert an EvalPort `Suite` (optionally paired with a `ResultSet`)
    into a ragrank `Dataset`.

    Requires `ragrank` to be installed (see the `ragrank` extra) -- unlike
    `to_openeval()`, which only duck-types over its input, this direction
    has to construct real ragrank objects.

    Each `TestCase.input` becomes `DataNode.question` (the last turn, if
    `input` is a conversation array), `retrieval_context` (falling back to
    `context`) becomes `DataNode.context`, and `expected_output` becomes
    `DataNode.reference`.

    `DataNode.response` has no `TestCase` equivalent -- a suite is inputs
    and grading criteria, not model outputs. Two ways to get it filled in:
    pass `result_set` (an EvalPort `ResultSet` produced by running this
    suite through some *other* tool) and each row's `actual_output` is
    used, matched by `test_case_id`; otherwise `response` is left as the
    empty string, which is the right shape for handing the returned
    `Dataset` to your own ragrank evaluation run to populate.
    """
    from ragrank.dataset import Dataset, DataNode  # noqa: PLC0415

    actual_by_id: Dict[str, str] = {}
    if result_set is not None:
        for r in result_set.get("results", []):
            tcid = r.get("test_case_id")
            if tcid is not None:
                actual_by_id[tcid] = r.get("actual_output") or ""

    nodes = []
    for tc in suite.get("test_cases", []):
        question = tc.get("input", "")
        if isinstance(question, list):
            question = question[-1] if question else ""

        context = tc.get("retrieval_context") or tc.get("context") or []
        node_kwargs: Dict[str, Any] = {
            "question": question or "",
            "context": list(context),
            "response": actual_by_id.get(tc.get("id"), ""),
        }
        reference = tc.get("expected_output")
        if reference is not None:
            node_kwargs["reference"] = reference

        metadata = tc.get("metadata") or {}
        if "retrieved_ids" in metadata:
            node_kwargs["retrieved_ids"] = metadata["retrieved_ids"]
        if "reference_ids" in metadata:
            node_kwargs["reference_ids"] = metadata["reference_ids"]

        nodes.append(DataNode(**node_kwargs))

    if not nodes:
        return Dataset(question=[], context=[], response=[])

    dataset = nodes[0].to_dataset()
    for node in nodes[1:]:
        dataset.append(node)
    return dataset
