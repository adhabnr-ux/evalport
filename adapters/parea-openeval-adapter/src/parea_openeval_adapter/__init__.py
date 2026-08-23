"""Parea AI <-> EvalPort adapter.

Standalone converter between Parea AI's TestCase/TestCaseCollection/Experiment/ExperimentStatsSchema
objects and the EvalPort open evaluation format (https://github.com/adhabnr-ux/evalport).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "experiment_to_openeval", "__version__"]
__version__ = "0.1.0"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _slugify(name: str, fallback_index: int = 0) -> str:
    """Normalize metric name into a grader_id."""
    slug = "_".join(name.strip().lower().replace("-", " ").split())
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    if not slug:
        return f"metric_{fallback_index}"
    return slug


def _clamp01(value: Any) -> Optional[float]:
    """Clamp a score into EvalPort's required [0, 1] range."""
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))



def _get_input_string(inputs: Any) -> str:
    """Extract a string representation of the input from Parea's dict-based inputs."""
    if not inputs:
        return ""
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, dict):
        # Check standard input keys
        for key in ("input", "question", "query", "prompt"):
            if key in inputs and inputs[key] is not None:
                return str(inputs[key])
        # If there's only one key, return its value
        if len(inputs) == 1:
            return str(next(iter(inputs.values())))
        # Otherwise, fall back to json representation
        return json.dumps(inputs)
    return str(inputs)


def _test_case_to_openeval(case: Any, index: int, grader_id: str) -> Dict[str, Any]:
    """Convert a single Parea TestCase to an EvalPort TestCase."""
    case_id = _get(case, "id")
    if case_id is None:
        case_id = f"tc_{index}"
    else:
        case_id = str(case_id)

    inputs = _get(case, "inputs")
    input_str = _get_input_string(inputs)

    if not input_str:
        raise ValueError(
            f"TestCase at index {index} (id={case_id!r}) has no non-empty input. "
            "EvalPort requires a non-empty string for `input`."
        )

    tc: Dict[str, Any] = {
        "id": case_id,
        "input": input_str,
        "graders": [grader_id],
    }

    target = _get(case, "target")
    if target is not None:
        tc["expected_output"] = str(target)

    tags = _get(case, "tags")
    if tags:
        tc["tags"] = list(tags)

    # Preserve Parea specific attributes under metadata
    parea_meta: Dict[str, Any] = {}
    if isinstance(inputs, dict):
        parea_meta["inputs"] = inputs

    collection_id = _get(case, "test_case_collection_id")
    if collection_id is not None:
        parea_meta["test_case_collection_id"] = collection_id

    if parea_meta:
        tc["metadata"] = {"parea": parea_meta}

    return tc


def to_openeval(
    test_cases_or_collection: Any,
    *,
    suite_id: Optional[str] = None,
    suite_name: Optional[str] = None,
    grader_id: str = "gr_parea_evals",
    grader_handler: str = "parea:evals",
) -> Dict[str, Any]:
    """Export Parea `TestCaseCollection` or a sequence of `TestCase`s to an EvalPort suite.

    `test_cases_or_collection` can be a Parea `TestCaseCollection` object, or a list of Parea `TestCase` objects,
    or a list of equivalent dictionaries.

    Returns a plain dict conforming to the EvalPort EvalSuite schema.
    """
    test_cases: Sequence[Any] = []
    metadata: Dict[str, Any] = {"openeval": {"source": "parea"}}

    # If it is a TestCaseCollection object or dict representing it
    is_collection = False
    collection = None
    if isinstance(test_cases_or_collection, dict) and "test_cases" in test_cases_or_collection:
        is_collection = True
        collection = test_cases_or_collection
        test_cases = collection.get("test_cases") or []
    elif hasattr(test_cases_or_collection, "test_cases") and _get(test_cases_or_collection, "test_cases") is not None:
        is_collection = True
        collection = test_cases_or_collection
        test_cases = _get(collection, "test_cases") or []

    if is_collection:
        if isinstance(test_cases, dict):
            test_cases = list(test_cases.values())
        col_id = _get(collection, "id")
        col_name = _get(collection, "name")
        col_created = _get(collection, "created_at")
        col_updated = _get(collection, "last_updated_at")
        col_cols = _get(collection, "column_names")

        if suite_id is None and col_id is not None:
            suite_id = f"parea_suite_{col_id}"
        if suite_name is None and col_name is not None:
            suite_name = f"Parea collection: {col_name}"

        parea_suite_meta: Dict[str, Any] = {}
        if col_id is not None:
            parea_suite_meta["id"] = col_id
        if col_name is not None:
            parea_suite_meta["name"] = col_name
        if col_created is not None:
            parea_suite_meta["created_at"] = str(col_created)
        if col_updated is not None:
            parea_suite_meta["last_updated_at"] = str(col_updated)
        if col_cols:
            parea_suite_meta["column_names"] = list(col_cols)

        if parea_suite_meta:
            metadata["parea"] = parea_suite_meta
    else:
        if isinstance(test_cases_or_collection, dict):
            test_cases = list(test_cases_or_collection.values())
        else:
            test_cases = test_cases_or_collection

    if suite_id is None:
        suite_id = "parea_suite"
    if suite_name is None:
        suite_name = "Parea test cases"

    test_case_dicts = [_test_case_to_openeval(case, i, grader_id) for i, case in enumerate(test_cases)]

    if not test_case_dicts:
        raise ValueError("test_cases must contain at least one entry (EvalSuite requires minItems: 1)")

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": suite_name,
        "test_cases": test_case_dicts,
        "graders": [{
            "id": grader_id,
            "type": "custom",
            "params": {"handler": grader_handler},
            "description": "Parea eval metrics mapped to custom EvalPort handler.",
        }],
        "metadata": metadata,
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of Parea TestCase-compatible dictionaries.

    Each returned dict has keys matching `TestCase` fields (`id`, `inputs`, `target`, `tags`).
    """
    cases: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        tc_id = tc.get("id")
        try:
            if tc_id is not None and str(tc_id).isdigit():
                tc_id = int(tc_id)
        except Exception:
            pass
        input_val = tc.get("input")
        target_val = tc.get("expected_output")
        tags_val = tc.get("tags")

        # Attempt to reconstruct the original inputs dictionary from metadata
        metadata = tc.get("metadata") or {}
        parea_meta = metadata.get("parea") or {}
        inputs_dict = parea_meta.get("inputs")

        if not isinstance(inputs_dict, dict):
            # If not present in metadata, default to keying by standard "input"
            inputs_dict = {"input": input_val}

        item: Dict[str, Any] = {
            "id": tc_id,
            "inputs": inputs_dict,
        }
        if target_val is not None:
            item["target"] = target_val
        if tags_val is not None:
            item["tags"] = list(tags_val)

        col_id = parea_meta.get("test_case_collection_id")
        if col_id is not None:
            item["test_case_collection_id"] = col_id

        cases.append(item)
    return cases


def experiment_to_openeval(
    experiment_or_stats: Any,
    trace_logs: Optional[Sequence[Any]] = None,
    *,
    suite_id: Optional[str] = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    runner_name: str = "parea",
    runner_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a Parea `Experiment`, `ExperimentStatsSchema` or a list of `TraceLog`s to an EvalPort ResultSet.

    Returns a plain dict conforming to the EvalPort ResultSet schema.
    """
    # 1. Resolve inputs
    stats: Any = None
    experiment_obj: Any = None
    logs: List[Any] = []

    if trace_logs is not None:
        logs = list(trace_logs)

    # Check if a list of logs was passed as the first argument
    if isinstance(experiment_or_stats, list):
        logs = list(experiment_or_stats)
    elif hasattr(experiment_or_stats, "experiment_stats"):
        # Parea Experiment object
        experiment_obj = experiment_or_stats
        stats = _get(experiment_obj, "experiment_stats")
        if suite_id is None:
            suite_id = _get(experiment_obj, "experiment_name")
        if run_id is None:
            run_id = _get(experiment_obj, "run_name")
        meta = _get(experiment_obj, "metadata")
        if meta and isinstance(meta, dict) and "Dataset" in meta:
            # Try to get suite_id from dataset metadata if not set
            if not suite_id:
                suite_id = meta["Dataset"]
    else:
        # ExperimentStatsSchema directly
        stats = experiment_or_stats

    # Defaults
    if not suite_id:
        suite_id = _get(stats, "experiment_uuid") or "parea_experiment"
    if not run_id:
        run_id = _get(stats, "experiment_uuid") or "parea_run"

    # 2. Correlate parent trace stats and logs
    parent_trace_stats = _get(stats, "parent_trace_stats") or []

    # Map trace_id -> TraceLog/Dict
    log_map: Dict[str, Any] = {str(_get(log, "trace_id")): log for log in logs if _get(log, "trace_id") is not None}

    # Gather start/end times if logs are available
    timestamps: List[str] = []
    for log in logs:
        start_ts = _get(log, "start_timestamp")
        end_ts = _get(log, "end_timestamp")
        if start_ts:
            timestamps.append(str(start_ts))
        if end_ts:
            timestamps.append(str(end_ts))

    if timestamps:
        timestamps.sort()
        if started_at is None:
            started_at = timestamps[0]
        if completed_at is None:
            completed_at = timestamps[-1]

    if started_at is None:
        from datetime import datetime, timezone
        started_at = datetime.now(timezone.utc).isoformat()
    if completed_at is None:
        completed_at = started_at

    results: List[Dict[str, Any]] = []

    # Helper to build result from a trace log (when logs are passed directly or matched)
    def _build_result_from_log(trace_log: Any, tc_id: str) -> Dict[str, Any]:
        output = _get(trace_log, "output")
        scores = _get(trace_log, "scores") or []

        grader_results: List[Dict[str, Any]] = []
        for j, s in enumerate(scores):
            score_val = _get(s, "score")
            score_name = _get(s, "name") or "unknown"
            reason = _get(s, "reason")

            clamped_score = _clamp01(score_val)
            gr = {
                "grader_id": _slugify(score_name, j),
                "type": "custom",
                "score": clamped_score,
                "passed": clamped_score >= 0.5 if clamped_score is not None else False,
            }
            if reason:
                gr["reason"] = str(reason)
            gr["metadata"] = {"metric_name": score_name}
            grader_results.append(gr)

        overall_passed = all(g["passed"] for g in grader_results) if grader_results else True

        res: Dict[str, Any] = {
            "test_case_id": tc_id,
            "passed": overall_passed,
            "grader_results": grader_results,
        }
        if output is not None:
            res["actual_output"] = str(output)

        latency = _get(trace_log, "latency")
        input_tokens = _get(trace_log, "input_tokens")
        output_tokens = _get(trace_log, "output_tokens")
        total_tokens = _get(trace_log, "total_tokens")
        cost = _get(trace_log, "cost")

        meta: Dict[str, Any] = {}
        if latency is not None:
            meta["latency"] = float(latency)
        if input_tokens is not None:
            meta["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            meta["output_tokens"] = int(output_tokens)
        if total_tokens is not None:
            meta["total_tokens"] = int(total_tokens)
        if cost is not None:
            meta["cost"] = float(cost)

        parea_meta = {}
        inputs = _get(trace_log, "inputs")
        if isinstance(inputs, dict):
            parea_meta["inputs"] = inputs
        tags = _get(trace_log, "tags")
        if tags:
            parea_meta["tags"] = list(tags)
        trace_id = _get(trace_log, "trace_id")
        if trace_id is not None:
            parea_meta["trace_id"] = str(trace_id)

        if parea_meta:
            meta["parea"] = parea_meta
        if meta:
            res["metadata"] = meta

        return res

    # Helper to build result from trace stats schema (when only stats are available)
    def _build_result_from_stats(trace_stat: Any, tc_id: str) -> Dict[str, Any]:
        scores = _get(trace_stat, "scores") or []

        grader_results: List[Dict[str, Any]] = []
        for j, s in enumerate(scores):
            score_val = _get(s, "score")
            score_name = _get(s, "name") or "unknown"
            reason = _get(s, "reason")

            clamped_score = _clamp01(score_val)
            gr = {
                "grader_id": _slugify(score_name, j),
                "type": "custom",
                "score": clamped_score,
                "passed": clamped_score >= 0.5 if clamped_score is not None else False,
            }
            if reason:
                gr["reason"] = str(reason)
            gr["metadata"] = {"metric_name": score_name}
            grader_results.append(gr)

        overall_passed = all(g["passed"] for g in grader_results) if grader_results else True

        res: Dict[str, Any] = {
            "test_case_id": tc_id,
            "passed": overall_passed,
            "grader_results": grader_results,
        }

        latency = _get(trace_stat, "latency")
        input_tokens = _get(trace_stat, "input_tokens")
        output_tokens = _get(trace_stat, "output_tokens")
        total_tokens = _get(trace_stat, "total_tokens")
        cost = _get(trace_stat, "cost")

        meta: Dict[str, Any] = {}
        if latency is not None:
            meta["latency"] = float(latency)
        if input_tokens is not None:
            meta["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            meta["output_tokens"] = int(output_tokens)
        if total_tokens is not None:
            meta["total_tokens"] = int(total_tokens)
        if cost is not None:
            meta["cost"] = float(cost)

        trace_id = _get(trace_stat, "trace_id")
        if trace_id is not None:
            meta["parea"] = {"trace_id": str(trace_id)}

        if meta:
            res["metadata"] = meta

        return res

    # Process logs directly if no stats provided
    if not stats and logs:
        for i, log in enumerate(logs):
            trace_id = _get(log, "trace_id")
            tc_id_resolved = str(trace_id) if trace_id is not None else f"tc_{i}"
            results.append(_build_result_from_log(log, tc_id_resolved))
    else:
        for i, stat in enumerate(parent_trace_stats):
            trace_id = str(_get(stat, "trace_id"))
            tc_id = f"tc_{i}"
            tc_id_resolved = trace_id if trace_id else tc_id

            if trace_id in log_map:
                results.append(_build_result_from_log(log_map[trace_id], tc_id_resolved))
            else:
                results.append(_build_result_from_stats(stat, tc_id_resolved))

    # Calculate summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    scores = [
        g["score"]
        for r in results
        for g in r.get("grader_results", [])
        if g.get("score") is not None
    ]
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped": 0,
        "pass_rate": (passed / total) if total else 0.0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
    }

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "runner": {"name": runner_name, "version": runner_version or __version__},
        "results": results,
        "summary": summary,
        "metadata": {"openeval": {"source": "parea"}},
    }
