"""Humanloop AI <-> EvalPort adapter.

Standalone converter between Humanloop AI's DatapointResponse/EvaluationResponse/EvaluatorLogResponse
objects and the EvalPort open evaluation format (https://github.com/adhabnr-ux/evalport).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "result_to_openeval", "evaluation_to_openeval", "__version__"]
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
    try:
        return max(0.0, min(1.0, float(value)))
    except (ValueError, TypeError):
        return None


def _to_dict(obj: Any) -> Any:
    """Convert an object (e.g. Pydantic model) to a serializable dictionary."""
    if hasattr(obj, "dict"):  # Pydantic v1
        try:
            return obj.dict(exclude_none=True)
        except Exception:
            pass
    if hasattr(obj, "model_dump"):  # Pydantic v2
        try:
            return obj.model_dump(exclude_none=True)
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    return obj


def _extract_text(value: Any) -> Optional[str]:
    """Convert an input or messages list/dict to a string representation."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            role = _get(item, "role")
            content = _get(item, "content")
            # content can be a string or a list of content blocks
            if isinstance(content, list):
                content_strs = []
                for block in content:
                    block_text = _get(block, "text")
                    if block_text is not None:
                        content_strs.append(block_text)
                    else:
                        content_strs.append(str(block))
                content_str = " ".join(content_strs)
            else:
                content_str = content

            role_str = getattr(role, "value", role)
            if content_str is not None:
                lines.append(f"{role_str}: {content_str}" if role_str else str(content_str))
            else:
                lines.append(str(item))
        return "\n".join(lines)

    # If it is a dict or single object representing a message
    content = _get(value, "content")
    if content is not None:
        if isinstance(content, list):
            content_strs = []
            for block in content:
                block_text = _get(block, "text")
                if block_text is not None:
                    content_strs.append(block_text)
                else:
                    content_strs.append(str(block))
            return " ".join(content_strs)
        return content if isinstance(content, str) else str(content)
    return str(value)


def _get_input_string(inputs: Any) -> Optional[str]:
    """Extract a string representation of the input from Humanloop's dict-based inputs."""
    if not inputs:
        return None
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, dict):
        for key in ("input", "question", "query", "prompt"):
            if key in inputs and inputs[key] is not None:
                return str(inputs[key])
        if len(inputs) == 1:
            return str(next(iter(inputs.values())))
        return json.dumps(inputs)
    return str(inputs)


def _extract_target_string(target: Any) -> Optional[str]:
    """Extract a string representation of the expected output/target."""
    if not target:
        return None
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        for key in ("target", "output", "expected_output", "value", "response"):
            if key in target and target[key] is not None:
                return str(target[key])
        if len(target) == 1:
            return str(next(iter(target.values())))
        return json.dumps(target)
    return str(target)


def to_openeval(
    datapoints: Any,
    *,
    suite_id: Optional[str] = None,
    suite_name: Optional[str] = None,
    grader_id: str = "gr_humanloop_evals",
    grader_handler: str = "humanloop:evals",
) -> Dict[str, Any]:
    """Export Humanloop Datapoints to an EvalPort suite.

    `datapoints` can be a list of DatapointResponse objects or dicts.
    """
    if not datapoints:
        raise ValueError("datapoints must contain at least one entry (EvalSuite.test_cases requires minItems: 1)")

    test_cases: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {"openeval": {"source": "humanloop"}}

    # Normalize inputs to a list
    if isinstance(datapoints, dict):
        # If it's a dict representing a dataset
        items = datapoints.get("datapoints") or datapoints.get("rows") or list(datapoints.values())
    else:
        items = datapoints

    for i, item in enumerate(items):
        item_id = _get(item, "id")
        tc_id = str(item_id) if item_id is not None else f"tc_{i}"

        inputs = _get(item, "inputs")
        messages = _get(item, "messages")
        target = _get(item, "target")

        # Resolve inputs and messages
        input_str = None
        if inputs:
            input_str = _get_input_string(inputs)
        if not input_str and messages:
            input_str = _extract_text(messages)

        if not input_str:
            raise ValueError(f"Datapoint {tc_id} has no non-empty input; humanloop-openeval-adapter cannot synthesize one.")

        tc: Dict[str, Any] = {
            "id": tc_id,
            "input": input_str,
            "graders": [grader_id],
        }

        expected_output = _extract_target_string(target)
        if expected_output is not None:
            tc["expected_output"] = expected_output

        # Preserve Humanloop specific attributes under metadata
        hl_meta: Dict[str, Any] = {}
        if inputs is not None:
            hl_meta["inputs"] = _to_dict(inputs)
        if messages is not None:
            hl_meta["messages"] = _to_dict(messages)
        if target is not None:
            hl_meta["target"] = _to_dict(target)

        if hl_meta:
            tc["metadata"] = {"humanloop": hl_meta}

        test_cases.append(tc)

    if suite_id is None:
        suite_id = "humanloop_suite"
    if suite_name is None:
        suite_name = "Humanloop Datasets"

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": suite_name,
        "graders": [{
            "id": grader_id,
            "type": "custom",
            "params": {"handler": grader_handler},
            "description": "Humanloop evaluation metrics mapped to custom EvalPort handler.",
        }],
        "test_cases": test_cases,
        "metadata": metadata,
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of Humanloop datapoint-ready dictionaries."""
    cases: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        tc_id = tc.get("id")
        input_val = tc.get("input")
        expected_output = tc.get("expected_output")

        metadata = tc.get("metadata") or {}
        hl_meta = metadata.get("humanloop") or {}

        # Reconstruct inputs, messages, target
        inputs = hl_meta.get("inputs")
        messages = hl_meta.get("messages")
        target = hl_meta.get("target")

        # Fallbacks
        if inputs is None and messages is None:
            inputs = {"input": input_val}

        if target is None and expected_output is not None:
            target = {"target": expected_output}

        item: Dict[str, Any] = {"id": tc_id}
        if inputs is not None:
            item["inputs"] = inputs
        if messages is not None:
            item["messages"] = messages
        if target is not None:
            item["target"] = target

        cases.append(item)

    return cases


def result_to_openeval(
    evaluation: Any,
    logs: Sequence[Any],
    *,
    suite_id: Optional[str] = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    runner_name: str = "humanloop",
    runner_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a Humanloop evaluation and logs to an EvalPort ResultSet.

    - `evaluation` carries setup info (e.g. name, ID, list of evaluators).
    - `logs` contains judgments per datapoint log.
    """
    eval_id = _get(evaluation, "id")
    eval_name = _get(evaluation, "name")

    if suite_id is None:
        suite_id = f"humanloop_suite_{eval_id}" if eval_id else "humanloop_suite"
    if run_id is None:
        run_id = f"humanloop_run_{eval_id}" if eval_id else "humanloop_run"

    # Map evaluators and return types
    evaluators_list = _get(evaluation, "evaluators") or []
    eval_return_types: Dict[str, str] = {}
    for ev in evaluators_list:
        ev_id = _get(ev, "id")
        ev_spec = _get(ev, "spec")
        ev_type = _get(ev_spec, "return_type") if ev_spec else None
        if ev_id and ev_type:
            eval_return_types[str(ev_id)] = str(ev_type)

    # Group log entries by source_datapoint_id
    datapoint_logs: Dict[str, List[Any]] = {}
    for log in logs:
        dp_id = _get(log, "source_datapoint_id")
        # Fallbacks for datapoint id
        if dp_id is None:
            dp_id = _get(log, "id") or _get(log, "trace_id") or "unknown_dp"
        dp_id = str(dp_id)
        if dp_id not in datapoint_logs:
            datapoint_logs[dp_id] = []
        datapoint_logs[dp_id].append(log)

    results: List[Dict[str, Any]] = []

    # Start and completed time resolution
    min_start: Optional[datetime] = None
    max_end: Optional[datetime] = None

    for dp_id, logs_for_dp in datapoint_logs.items():
        grader_results: List[Dict[str, Any]] = []
        actual_output = None

        for j, log in enumerate(logs_for_dp):
            # Resolve actual output
            output_val = _get(log, "output") or _get(log, "output_message")
            if output_val is not None:
                actual_output = _extract_text(output_val)

            # Resolve timestamps
            start_str = _get(log, "start_time") or _get(log, "created_at")
            end_str = _get(log, "end_time") or start_str
            for t_str, is_start in [(start_str, True), (end_str, False)]:
                if t_str:
                    try:
                        # Clean up trailing Z or offset if needed
                        dt = datetime.fromisoformat(str(t_str).replace("Z", "+00:00"))
                        if is_start:
                            if min_start is None or dt < min_start:
                                min_start = dt
                        else:
                            if max_end is None or dt > max_end:
                                max_end = dt
                    except Exception:
                        pass

            # Grader judgment
            judgment_val = _get(log, "judgment")
            evaluator_info = _get(log, "evaluator")
            ev_id = _get(evaluator_info, "id")
            ev_name = _get(evaluator_info, "name") or "evaluator"
            grader_id = _slugify(ev_name, j)

            # Retrieve return type
            return_type = eval_return_types.get(str(ev_id)) if ev_id else None
            if return_type is None and evaluator_info:
                # Try from spec
                ev_spec = _get(evaluator_info, "spec")
                return_type = _get(ev_spec, "return_type")

            # Infer return type if still None
            if return_type is None and judgment_val is not None:
                if isinstance(judgment_val, bool):
                    return_type = "boolean"
                elif isinstance(judgment_val, (int, float)):
                    return_type = "number"
                else:
                    return_type = "text"

            score: Optional[float] = None
            passed = False

            if judgment_val is not None:
                if return_type == "boolean":
                    score = 1.0 if judgment_val is True else 0.0
                    passed = bool(judgment_val)
                elif return_type == "number":
                    score = _clamp01(judgment_val)
                    passed = score >= 0.5 if score is not None else False
                else:
                    # 'select', 'multi_select', 'text' -> non-numeric
                    score = None
                    passed = False

            gr: Dict[str, Any] = {
                "grader_id": grader_id,
                "type": "custom",
                "score": score,
                "passed": passed,
                "metadata": {
                    "humanloop.return_type": return_type,
                    "humanloop.judgment": _to_dict(judgment_val) if judgment_val is not None else None,
                },
            }

            error = _get(log, "error")
            if error:
                gr["metadata"]["humanloop.error"] = str(error)
                gr["passed"] = False

            grader_results.append(gr)

        overall_passed = all(g["passed"] for g in grader_results) if grader_results else False

        res: Dict[str, Any] = {
            "test_case_id": dp_id,
            "passed": overall_passed,
            "grader_results": grader_results,
        }
        if actual_output is not None:
            res["actual_output"] = actual_output

        # Extract metadata from logs if available
        first_log = logs_for_dp[0]
        latency = _get(first_log, "provider_latency")
        meta: Dict[str, Any] = {}
        if latency is not None:
            meta["latency"] = float(latency)

        hl_res_meta = {}
        log_id = _get(first_log, "id") or _get(first_log, "log_id")
        if log_id is not None:
            hl_res_meta["log_id"] = str(log_id)
        inputs_val = _get(first_log, "inputs")
        if inputs_val is not None:
            hl_res_meta["inputs"] = _to_dict(inputs_val)

        if hl_res_meta:
            meta["humanloop"] = hl_res_meta

        if meta:
            res["metadata"] = meta

        results.append(res)

    # Resolve overall timestamps
    if started_at is None and min_start is not None:
        started_at = min_start.isoformat().replace("+00:00", "Z")
    if completed_at is None and max_end is not None:
        completed_at = max_end.isoformat().replace("+00:00", "Z")

    if started_at is None:
        started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if completed_at is None:
        completed_at = started_at

    # Calculate summary
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    scores_list = [
        g["score"]
        for r in results
        for g in r.get("grader_results", [])
        if g.get("score") is not None
    ]
    summary = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "skipped": 0,
        "pass_rate": (passed_count / total) if total else 0.0,
        "avg_score": (sum(scores_list) / len(scores_list)) if scores_list else 0.0,
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
        "metadata": {"openeval": {"source": "humanloop"}},
    }


# Alias for compatibility with other naming conventions
evaluation_to_openeval = result_to_openeval
