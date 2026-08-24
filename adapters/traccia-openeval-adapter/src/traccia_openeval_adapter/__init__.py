"""
traccia-openeval-adapter

Convert traccia (https://github.com/traccia-ai/traccia-py) `evaluate()`
results to EvalPort's (https://github.com/adhabnr-ux/evalport) portable
`ResultSet` interchange format.

Verified against the real installed `traccia` package (0.1.28, PyPI) --
`traccia.eval.evaluate()` / `traccia.eval.evaluate.EvaluateResult` -- not
docs or the proposal sketch in traccia-ai/traccia-py#35. See README.md's
"Design notes" for the two real wrinkles this handles: `run_id` and
`started_at` have no source in `EvaluateResult` when `persist=False` (the
common case for local/offline runs, and the one shown in #35's own usage
example), so this module mints both at conversion time rather than
inventing fake provenance.

Zero footprint on traccia-py itself: this package works from the outside
against `EvaluateResult`'s public dataclass fields, the same "standalone
adapter" pattern already used by the other 36 packages under adapters/ in
this repo (see adapters/autogen-openeval-adapter, the reference
implementation, and issue #6 "Adapters wanted").
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk is a hard dependency at
    # runtime (see pyproject.toml); keep a sane fallback for static analysis.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "results_to_openeval",
    "row_to_result",
    "score_to_grader_result",
    "clamp_score",
]
__version__ = "0.1.0"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` off either a dict or an attribute-holding object.

    `EvaluateResult` and its `rows`/`panels`/`scores` entries are always
    plain dicts in the real traccia SDK (dataclasses.asdict is never
    called; `evaluate()` builds them as dicts directly -- see
    eval/evaluate.py's `_process()`), but this accessor stays duck-typed
    so a hand-built or future dataclass-shaped row still works.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def clamp_score(value: Any) -> Optional[float]:
    """Coerce a traccia scorer's `score` into EvalPort's required shape:
    a float in [0, 1], or None.

    traccia's three current builtin scorers (`exact_match`, `contains`,
    `json_valid` -- see eval/builtins.py) always emit 0.0 or 1.0, so this
    is a no-op clamp for them today. It exists because `evaluate()` also
    accepts custom callables and platform/remote scorers (`_run_scorer` in
    eval/evaluate.py), whose `score` is whatever the scorer returns --
    EvalPort's `validate_result_set()` rejects anything outside [0, 1] or
    non-numeric, so this keeps a scorer bug from producing a ResultSet
    that fails validation for a reason unrelated to the conversion itself.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        f = float(value)
        if f != f:  # NaN
            return None
        return max(0.0, min(1.0, f))
    return None


def _stringify_output(output: Any) -> Optional[str]:
    """EvalPort's `Result.actual_output` is `Optional[str]`; traccia's panel
    `output` is whatever the task returned (str, dict, list, number, ...) --
    `evaluate()` already ran it through its own `_jsonable()` so it is
    JSON-safe, but it is not necessarily a string yet."""
    if output is None:
        return None
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(output)


def score_to_grader_result(score: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one traccia score dict (a `panels[i]["scores"][j]` entry) to
    an EvalPort `GraderResult` dict.

    traccia score dicts always carry `name`/`scorer_name` (the same value,
    normalized by `_normalize_score()`), `passed`, `score`, `reason`, and
    for builtins a `type` equal to the scorer name (`_run_builtin_scorer`
    stamps `out["type"] = name`). Platform/remote scorers additionally
    carry `scorer_id`, `config`, `model`, `latency_ms`, `cost_usd`, `usage`
    -- none of those have a slot on EvalPort's `GraderResult`, so they are
    preserved under `metadata` rather than silently dropped.
    """
    grader_id = str(
        _get(score, "scorer_id") or _get(score, "scorer_name") or _get(score, "name") or "unknown_scorer"
    )
    grader_type = str(_get(score, "type") or _get(score, "scorer_name") or _get(score, "name") or "custom")
    passed = bool(_get(score, "passed"))

    extra_meta: Dict[str, Any] = {}
    for key in ("scorer_id", "scorer_name", "model", "latency_ms", "cost_usd", "usage", "config"):
        v = _get(score, key)
        if v is not None:
            extra_meta[key] = v

    grader_result: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": grader_type,
        "score": clamp_score(_get(score, "score")),
        "passed": passed,
    }
    reason = _get(score, "reason")
    if reason is not None:
        grader_result["reason"] = str(reason)
    if extra_meta:
        grader_result["metadata"] = extra_meta
    return grader_result


def row_to_result(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one traccia `EvaluateResult.rows[i]` entry to an EvalPort
    `Result` dict.

    A row's `item_id`/`input`/`expected_output` map directly. A row's
    `panels` list is always exactly one entry today -- `evaluate()` only
    ever appends a single "Task" cell per item (`aggregates["panel_count"]`
    is hardcoded to 1 in eval/evaluate.py; multi-panel/prompt-comparison
    support does not exist yet in this version of traccia, 0.1.28). This
    function uses `panels[0]` and raises a clear error rather than
    silently dropping data if that ever changes, so a future traccia
    release that adds real multi-panel rows fails loudly here instead of
    quietly losing panels[1:].
    """
    panels = _get(row, "panels") or []
    if len(panels) != 1:
        raise ValueError(
            f"row {_get(row, 'item_id')!r} has {len(panels)} panels; "
            "results_to_openeval() currently assumes traccia's single-panel "
            "evaluate() shape (panel_count == 1) and does not yet know how "
            "to map multi-panel rows onto EvalPort's one-Result-per-test-case "
            "model. See traccia-ai/traccia-py#35 and this package's README."
        )
    panel = panels[0]

    scores = _get(panel, "scores") or []
    grader_results = [score_to_grader_result(s) for s in scores]

    # `panel["passed"]` is `None` when the item had no scorers attached at
    # all (`cell["passed"] = None` in eval/evaluate.py's `_process()` --
    # distinct from a real, scored failure). EvalPort's `Result.passed` is
    # a required bool, so an unscored item is treated as passed (there is
    # nothing that failed it) rather than coerced to False, which would
    # misrepresent "not evaluated" as "evaluated and failed".
    raw_passed = _get(panel, "passed")
    if raw_passed is None:
        passed = all(gr["passed"] for gr in grader_results) if grader_results else True
    else:
        passed = bool(raw_passed)

    result: Dict[str, Any] = {
        "test_case_id": str(_get(row, "item_id")),
        "passed": passed,
        "grader_results": grader_results,
        "actual_output": _stringify_output(_get(panel, "output")),
    }

    latency_ms = _get(panel, "latency_ms")
    if isinstance(latency_ms, (int, float)):
        result["duration_ms"] = int(round(latency_ms))

    error = _get(panel, "error")
    if error is not None:
        result["error"] = {"message": str(error)}

    metadata: Dict[str, Any] = {}
    for key in ("label", "source", "trace_id", "model", "cost_usd", "prompt_version_id"):
        v = _get(panel, key)
        if v is not None:
            metadata[key] = v
    input_val = _get(row, "input")
    if input_val is not None:
        metadata["input"] = input_val
    expected = _get(row, "expected_output")
    if expected is not None:
        metadata["expected_output"] = expected
    if metadata:
        result["metadata"] = metadata

    return result


def results_to_openeval(
    result: Any,
    *,
    suite_id: str,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a traccia `EvaluateResult` (from `traccia.eval.evaluate()`)
    into an EvalPort `ResultSet` dict.

        from traccia.eval import evaluate
        from traccia_openeval_adapter import results_to_openeval
        from openeval.validate import validate_result_set

        result = evaluate(
            "my-experiment",
            data=[{"input": {"q": "2+2"}, "expected": "4"}],
            task=lambda row: my_agent(row["input"]),
            scorers=["exact_match"],
            persist=False,
        )
        result_set = results_to_openeval(result, suite_id="my-experiment")
        assert validate_result_set(result_set).valid

    Args:
        result: a real `traccia.eval.evaluate.EvaluateResult` (or any
            object/dict exposing the same `rows`/`aggregates`/`name`/
            `experiment_id`/`errors` fields).
        suite_id: EvalPort `ResultSet.suite_id` -- required by the spec,
            and not something `EvaluateResult` carries on its own (traccia
            experiments aren't defined against a versioned EvalPort suite),
            so the caller supplies it. Matches the usage shown in
            traccia-ai/traccia-py#35.
        run_id: EvalPort `ResultSet.run_id`. Defaults to
            `result.experiment_id` when traccia persisted the run
            (`persist=True`) and one was assigned; when `persist=False`
            (traccia never allocates an `experiment_id` in that path --
            see `evaluate()`'s `experiment_id = str(uuid.uuid4()) if
            persist else None`), a local id is minted instead so the
            required field is never silently fabricated as a copy of
            something else.
        started_at: EvalPort `ResultSet.started_at` (ISO 8601). traccia's
            `EvaluateResult` does not currently record a run start
            timestamp anywhere (only per-item `latency_ms`), so this
            defaults to conversion time, not run start time. Pass an
            explicit value if you captured the real start time yourself
            (e.g. `datetime.now(timezone.utc).isoformat()` before calling
            `evaluate()`).

    Returns:
        A plain dict conforming to EvalPort's `ResultSet` schema. Pass it
        to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    if not suite_id or not str(suite_id).strip():
        raise ValueError("suite_id is required")

    rows = _get(result, "rows") or []
    results = [row_to_result(r) for r in rows]

    resolved_run_id = run_id or _get(result, "experiment_id") or f"traccia-local-{uuid.uuid4().hex[:12]}"
    resolved_started_at = started_at or datetime.now(timezone.utc).isoformat()

    passed_count = sum(1 for r in results if r["passed"])
    all_scores = [
        gr["score"] for r in results for gr in r["grader_results"] if gr.get("score") is not None
    ]
    summary: Dict[str, Any] = {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": (passed_count / len(results)) if results else 0.0,
        "avg_score": (sum(all_scores) / len(all_scores)) if all_scores else None,
    }

    metadata: Dict[str, Any] = {"openeval": {"source": "traccia"}}
    traccia_agg = _get(result, "aggregates")
    if traccia_agg:
        metadata["traccia_aggregates"] = traccia_agg
    name = _get(result, "name")
    if name is not None:
        metadata["traccia_experiment_name"] = name
    dataset_id = _get(result, "dataset_id")
    if dataset_id is not None:
        metadata["traccia_dataset_id"] = dataset_id
    errors = _get(result, "errors")
    if errors:
        metadata["traccia_errors"] = errors

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": str(suite_id),
        "run_id": str(resolved_run_id),
        "started_at": resolved_started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "runner": {"name": "traccia-openeval-adapter", "version": __version__},
        "summary": summary,
        "metadata": metadata,
    }
    return result_set
