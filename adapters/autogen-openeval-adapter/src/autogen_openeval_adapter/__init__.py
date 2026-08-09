"""AutoGen <-> EvalPort adapter.

Standalone converter between AutoGen agent evaluation tasks/results and the
EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside AutoGen
itself: as of August 2026, microsoft/autogen is in maintenance mode and its
README states contributions are limited to bug fixes, security patches, and
documentation improvements — new-feature PRs (like a format adapter) are out
of scope for the core repo going forward. This package lets AutoGen users get
EvalPort import/export today without waiting on that to change, by depending
on AutoGen's public task/result shapes rather than modifying AutoGen itself.

Design credit: the to_openeval()/from_openeval() surface mirrors the mapping
originally proposed by the EvalPort spec author in
https://github.com/microsoft/autogen/issues/8005, and was validated against a
draft implementation contributed by @DresdenGman in
https://github.com/microsoft/autogen/pull/8009.
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    AutoGen's eval task/result classes and JSON-loaded eval output both show
    up in the wild, so every accessor in this module goes through here rather
    than assuming one shape.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _task_payload(task: Any, index: int) -> Dict[str, Any]:
    """Normalize a single AutoGen eval task/result into an EvalPort TestCase dict."""
    task_id = _get(task, "task_id") or _get(task, "id") or f"tc_{index}"
    description = (
        _get(task, "task_description")
        or _get(task, "description")
        or _get(task, "input")
        or ""
    )
    expected_output = _get(task, "expected_output")
    expected_tools = _get(task, "expected_tools") or []
    metadata = dict(_get(task, "metadata") or {})

    tc: Dict[str, Any] = {
        "id": str(task_id),
        "input": description,
        "graders": ["gr_output_match"],
    }
    # Only `expected_output` (singular) is standard per the EvalPort spec —
    # deliberately not emitting an `expected_outputs` (plural) field here,
    # per review feedback on the original AutoGen PR.
    if expected_output is not None:
        tc["expected_output"] = str(expected_output)
    if expected_tools:
        tc["expected_tools"] = list(expected_tools)
    if metadata:
        tc["metadata"] = metadata
    return tc


def to_openeval(autogen_eval_result: Any, grader_type: str = "exact_match") -> Dict[str, Any]:
    """Export an AutoGen eval result to an EvalPort-shaped suite (dict).

    `autogen_eval_result` may be any object or dict exposing a `results`
    sequence of tasks and a `run_id` (or `id`) — this matches both AutoGen's
    EvalResult class and plain JSON-loaded eval output, so no direct AutoGen
    import is required.

    `grader_type` selects the generated grader: "exact_match" (default, with
    ignore_case=True) or "llm_judge" for agent evals where exact string
    matching is too strict — the EvalPort spec author flagged this as a
    worthwhile option in the original AutoGen PR review.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass it
    to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    results = _get(autogen_eval_result, "results", []) or []
    run_id = _get(autogen_eval_result, "run_id") or _get(autogen_eval_result, "id") or "autogen_run"

    test_cases = [_task_payload(r, i) for i, r in enumerate(results)]

    if grader_type == "llm_judge":
        grader: Dict[str, Any] = {
            "id": "gr_output_match",
            "type": "llm_judge",
            "params": {
                "model": "gpt-4o",
                "prompt": (
                    "Does {output} correctly accomplish the task described in "
                    '{input}? Expected: {expected}. Return JSON: {"score": 0.0-1.0}.'
                ),
            },
        }
    else:
        grader = {"id": "gr_output_match", "type": "exact_match", "params": {"ignore_case": True}}

    return {
        "version": OPENEVAL_VERSION,
        "id": f"autogen_eval_{run_id}",
        "name": f"AutoGen eval run {run_id}",
        "test_cases": test_cases,
        "graders": [grader],
        "metadata": {"openeval": {"source": "autogen"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of AutoGen-shaped eval task dicts.

    Returns plain dicts (task_id, description, expected_output,
    expected_tools, metadata) rather than a specific AutoGen class, since
    AutoGen's own eval-task constructor can vary by version. Pass these
    straight into your task class if the field names line up
    (`EvalTask(**task)`), or map them explicitly.
    """
    tasks: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        tasks.append(
            {
                "task_id": tc.get("id"),
                "description": tc.get("input"),
                "expected_output": tc.get("expected_output"),
                "expected_tools": tc.get("expected_tools") or [],
                "metadata": tc.get("metadata") or {},
            }
        )
    return tasks
