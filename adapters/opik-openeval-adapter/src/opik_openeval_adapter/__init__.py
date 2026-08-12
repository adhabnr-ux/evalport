"""Opik <-> EvalPort adapter.

Standalone converter between Comet Opik dataset items / experiment results
and the EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside Opik
itself: see the maintainer-agent ("Scout") triage on
https://github.com/comet-ml/opik/issues/7798, which recommended exactly
this shape as the lowest-friction starting point — a package that needs
zero changes to Opik core, built against the public `Dataset` /
`DatasetItem` / `Experiment` / `ExperimentItemContent` surface Opik already
exposes, following the same playbook as the AutoGen and CrewAI adapters
(https://github.com/adhabnr-ux/evalport/tree/main/adapters).

Two independent conversions are provided, mirroring Opik's own two-layer
model (a dataset layer and a separate experiment/results layer):

- `to_openeval()` / `from_openeval()` — Opik `Dataset` items <-> an EvalPort
  `EvalSuite` (the test cases themselves).
- `experiment_to_openeval()` — Opik `Experiment` items (an evaluation run's
  outputs and feedback scores) -> an EvalPort `ResultSet`.

Mapping (per the Scout triage on opik#7798):

| Opik concept                                    | EvalPort concept              |
|--------------------------------------------------|-------------------------------|
| `DatasetItem` data fields (`model_extra`)         | TestCase `input`/`expected_output` + `metadata` |
| `ExperimentItemContent.evaluation_task_output`    | Result actual output (via `metadata`) |
| `FeedbackScoreDict` (`name`, `value`, `reason`)   | GraderResult per test case    |
| `DatasetItem.evaluators`                          | EvalPort grader definitions   |
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "experiment_to_openeval", "__version__"]
__version__ = "0.1.0"

# Fields opik.api_objects.dataset.dataset_item.DatasetItem always carries,
# whether or not they're populated. Everything else on an item (or every
# other key in a plain dict from Dataset.get_items()) is the item's actual
# "data" payload — these are excluded from that payload.
_DATASET_ITEM_CORE_FIELDS = {"id", "trace_id", "span_id", "source", "description", "evaluators", "execution_policy"}

# Candidate keys checked (case-insensitively, in order) to auto-detect which
# data field is the model input / expected output, since Opik dataset items
# are schema-less by design. An explicit input_key/expected_output_key
# argument always wins over this heuristic.
_INPUT_KEY_CANDIDATES = ("input", "question", "user_input", "prompt", "query", "messages")
_EXPECTED_OUTPUT_KEY_CANDIDATES = ("expected_output", "expected_answer", "answer", "reference", "reference_answer", "ground_truth")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    `Dataset.get_items()` returns plain dicts; `DatasetItem` / `ExperimentItemContent`
    are Pydantic-style objects. Every accessor here goes through this so callers
    can pass either shape interchangeably.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any, known_fields: Any) -> Dict[str, Any]:
    """Normalize a dataset item (dict, or a pydantic-style object with model_dump/model_extra) to a plain dict."""
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump())
    return {f: getattr(obj, f) for f in known_fields if hasattr(obj, f)}


def _item_data(item: Any) -> Dict[str, Any]:
    """The item's "data" payload: every key that isn't one of DatasetItem's fixed core fields."""
    d = _as_dict(item, _DATASET_ITEM_CORE_FIELDS)
    return {k: v for k, v in d.items() if k not in _DATASET_ITEM_CORE_FIELDS and v is not None}


def _pick_key(data: Dict[str, Any], explicit: Optional[str], candidates: tuple) -> Optional[str]:
    if explicit is not None:
        return explicit if explicit in data else None
    lower_map = {k.lower(): k for k in data.keys()}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    import json as _json
    return _json.dumps(value, default=str)


def to_openeval(
    dataset_items: Any,
    *,
    suite_id: str = "opik_dataset_import",
    name: Optional[str] = None,
    input_key: Optional[str] = None,
    expected_output_key: Optional[str] = None,
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """Export Opik dataset items to an EvalPort-shaped suite (dict).

    `dataset_items` is any iterable of Opik dataset items — the list of dicts
    returned by `Dataset.get_items()`, a list of `DatasetItem` objects from
    `__internal_api__stream_items_as_dataclasses__()`, or plain JSON-loaded
    dicts with the same shape. No direct Opik import is required.

    Because Opik dataset items are schema-less (arbitrary keys per dataset),
    `input_key`/`expected_output_key` let you name the fields explicitly;
    if omitted, common conventions (`input`/`question`/`user_input`/...,
    `expected_output`/`answer`/`reference`/...) are auto-detected per item.
    Every other data field is preserved under the test case's `metadata`, so
    nothing in the original item is silently dropped even when the input/
    expected-output guess is wrong — you always get the full payload back.

    `grader_type` selects the default grader attached when a test case has
    an expected output: "llm_judge" (default, sane for free-text Opik
    datasets) or "exact_match".

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance.
    """
    test_cases: List[Dict[str, Any]] = []
    has_expected_output = False

    for i, item in enumerate(dataset_items):
        item_id = _get(item, "id") or f"tc_{i}"
        data = _item_data(item)

        in_key = _pick_key(data, input_key, _INPUT_KEY_CANDIDATES)
        out_key = _pick_key(data, expected_output_key, _EXPECTED_OUTPUT_KEY_CANDIDATES)

        if in_key is not None:
            input_value = _stringify(data[in_key])
        else:
            description = _get(item, "description")
            input_value = description if description else _stringify(data) if data else ""

        tc: Dict[str, Any] = {"id": str(item_id), "input": input_value, "graders": ["gr_output_match"]}

        if out_key is not None:
            tc["expected_output"] = _stringify(data[out_key])
            has_expected_output = True

        metadata = {k: v for k, v in data.items() if k not in (in_key, out_key)}
        metadata["opik"] = {"dataset_item_id": str(item_id)}
        tc["metadata"] = metadata

        evaluators = _get(item, "evaluators")
        if evaluators:
            # Serialize to plain JSON-safe dicts (evaluators may be pydantic
            # EvaluatorItem objects, not already-plain dicts) so the suite
            # stays writable with a plain json.dump() and stays comparable
            # by validate_suite(), which expects plain dict/list/str/number.
            tc["metadata"]["opik"]["evaluators"] = [
                e.model_dump() if hasattr(e, "model_dump") else e for e in evaluators
            ]

        test_cases.append(tc)

    graders: List[Dict[str, Any]] = []
    if any("gr_output_match" in tc["graders"] for tc in test_cases):
        if grader_type == "exact_match":
            graders.append({"id": "gr_output_match", "type": "exact_match", "params": {"ignore_case": True}})
        else:
            graders.append({
                "id": "gr_output_match",
                "type": "llm_judge",
                "params": {
                    "model": "gpt-4o",
                    "prompt": (
                        "Expected output: {expected}\nActual output: {output}\n"
                        "Does the actual output satisfy the expected output? "
                        'Return JSON: {"score": 0.0-1.0, "reason": "..."}'
                    ),
                },
            })
    if test_cases and not has_expected_output and not graders:
        # No expected_output anywhere: still attach a grader so the suite
        # validates (EvalPort requires >=1 grader ref per test case), but
        # make it clearly a placeholder rather than a silent no-op.
        graders.append({
            "id": "gr_output_match",
            "type": "custom",
            "params": {"handler": "opik:no_expected_output"},
        })

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name or f"Opik dataset import ({suite_id})",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "opik"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of Opik-insertable dataset item dicts.

    Returns plain dicts shaped for `Dataset.insert(items)` — each item's
    EvalPort `input` becomes the `input` data field, `expected_output`
    becomes `expected_output`, and every key that was round-tripped through
    `metadata` (excluding the `opik.*` bookkeeping key this adapter writes)
    is restored to the top level, so a suite exported by `to_openeval()` and
    re-imported here reconstructs the original Opik item shape.
    """
    items: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        metadata = dict(tc.get("metadata") or {})
        metadata.pop("opik", None)

        item: Dict[str, Any] = {"id": tc.get("id"), "input": tc.get("input")}
        if "expected_output" in tc:
            item["expected_output"] = tc["expected_output"]
        item.update(metadata)
        items.append(item)
    return items


def experiment_to_openeval(
    experiment_items: Any,
    *,
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    runner_name: str = "opik",
    runner_version: Optional[str] = None,
    pass_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Export Opik experiment items (an evaluation run's results) to an EvalPort ResultSet.

    `experiment_items` is any iterable of Opik experiment items —
    `ExperimentItemContent` objects (`dataset_item_id`, `evaluation_task_output`,
    `feedback_scores`) or equivalent dicts. Each Opik `FeedbackScoreDict`
    (`name`, `value`, `category_name`, `reason`) becomes one EvalPort
    `GraderResult`; a result's `passed` follows EvalPort's own convention
    (all of its grader results must individually pass) exactly the way
    `evalport run`'s own runner computes it.

    `pass_threshold` sets the score at/above which a feedback score counts
    as a pass (default 0.5) — Opik feedback scores don't carry their own
    pass/fail boolean, only a numeric value, so this threshold is where that
    boolean is decided. Pass a different threshold if your scores use a
    different scale/convention.

    `started_at` defaults to the current UTC time in ISO 8601 if omitted.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass
    it to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    if started_at is None:
        from datetime import datetime, timezone
        started_at = datetime.now(timezone.utc).isoformat()

    results: List[Dict[str, Any]] = []
    for item in experiment_items:
        dataset_item_id = _get(item, "dataset_item_id") or _get(item, "id")
        task_output = _get(item, "evaluation_task_output")
        feedback_scores = _get(item, "feedback_scores") or []

        grader_results: List[Dict[str, Any]] = []
        for fs in feedback_scores:
            name = _get(fs, "name")
            value = _get(fs, "value")
            reason = _get(fs, "reason")
            gr: Dict[str, Any] = {
                "grader_id": str(name) if name is not None else "gr_unknown",
                "type": "custom",
                "score": value,
                "passed": bool(value is not None and value >= pass_threshold),
            }
            if reason:
                gr["reason"] = reason
            category = _get(fs, "category_name")
            if category:
                gr["metadata"] = {"category_name": category}
            grader_results.append(gr)

        result: Dict[str, Any] = {
            "test_case_id": str(dataset_item_id),
            "passed": len(grader_results) > 0 and all(g["passed"] for g in grader_results),
            "grader_results": grader_results,
        }
        if task_output is not None:
            result["metadata"] = {"opik": {"evaluation_task_output": task_output}}
        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    scores = [g["score"] for r in results for g in r["grader_results"] if g.get("score") is not None]
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped": 0,
        "pass_rate": (passed / total) if total else 0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0,
    }

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": started_at,
        "runner": {"name": runner_name, "version": runner_version or __version__},
        "results": results,
        "summary": summary,
        "metadata": {"openeval": {"source": "opik"}},
    }
