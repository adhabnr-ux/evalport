"""Convert between Langfuse datasets/experiments and EvalPort.

EvalPort (https://github.com/adhabnr-ux/evalport) is an open interchange
format (Apache 2.0) for portable LLM evaluation datasets: test cases,
graders, suites, and results as plain JSON, shared across evaluation tools.

Langfuse (https://github.com/langfuse/langfuse, ``pip install langfuse``,
tested against 4.14.4) organizes evaluation around three real, verified
shapes -- read directly from the installed package's source, not guessed:

* ``langfuse.api.commons.types.dataset_item.DatasetItem`` -- the pydantic
  model returned by ``Langfuse().get_dataset(name).items`` and by
  ``Langfuse().create_dataset_item(...)``. Fields (confirmed via
  ``DatasetItem.model_fields``): ``id``, ``status``, ``input``,
  ``expected_output``, ``metadata``, ``source_trace_id``,
  ``source_observation_id``, ``dataset_id``, ``dataset_name``,
  ``created_at``, ``updated_at``, ``media_references``.
* ``langfuse.experiment.LocalExperimentItem`` -- a plain dict with keys
  ``input``, ``expected_output``, ``metadata``, used as ad-hoc experiment
  data not tied to a stored Langfuse dataset (confirmed via its
  ``__annotations__``).
* ``langfuse.experiment.ExperimentResult`` -- what
  ``Langfuse().run_experiment(...)`` returns: ``name``, ``run_name``,
  ``description``, ``item_results`` (a list of ``ExperimentItemResult``,
  each with ``item``, ``output``, ``evaluations``, ``trace_id``,
  ``dataset_run_id``), ``run_evaluations`` (evaluations scored against the
  whole run rather than one item), ``experiment_id``, ``dataset_run_id``,
  ``dataset_run_url`` (all confirmed via ``inspect.signature`` on the real
  classes in the installed package).
* ``langfuse.experiment.Evaluation`` -- one scorer result: ``name``,
  ``value`` (``int | float | str | bool``), ``comment``, ``metadata``,
  ``data_type`` (``"NUMERIC" | "CATEGORICAL" | "BOOLEAN"`` or ``None``),
  ``config_id``.

Three entry points, matching the shape used by every other adapter in the
EvalPort ecosystem:

    to_openeval(items, ...)
        A list of Langfuse ``DatasetItem`` objects (e.g. from
        ``get_dataset(name).items``) -- or plain ``LocalExperimentItem``-shaped
        dicts -- -> an EvalPort suite.

    from_openeval(suite, ...)
        An EvalPort suite's test cases -> a list of ``LocalExperimentItem``-shaped
        dicts, directly usable as the ``data`` argument to
        ``Langfuse().run_experiment(data=...)`` or, unpacked, as kwargs to
        ``Langfuse().create_dataset_item(dataset_name=..., **item)``.

    experiment_result_to_openeval(experiment_result, ...)
        The ``ExperimentResult`` (or a bare list of ``ExperimentItemResult``)
        that ``run_experiment()`` returns -> an EvalPort ResultSet, one
        GraderResult per ``Evaluation`` scored against each item.

Langfuse's ``input``/``expected_output``/``metadata`` map directly onto
EvalPort's ``TestCase.input``/``expected_output``/``metadata``. The full
original item -- including Langfuse-only fields like ``id``, ``status``,
``dataset_id``, ``source_trace_id`` -- is always preserved under
``metadata["langfuse"]["item"]`` on export, so a round trip through this
adapter never silently drops a field Langfuse itself would need (the same
lossless-preservation pattern used by every other adapter in this
repository, e.g. ``uptrain-openeval-adapter`` and
``weave-openeval-adapter``).
"""

from __future__ import annotations

import datetime
import enum
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

try:
    from openeval.version import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "experiment_result_to_openeval"]

# Labels that count as an affirmative / passing result for Langfuse
# CATEGORICAL evaluations, whose ``value`` is an arbitrary string with no
# guaranteed vocabulary. Anything not in this set is treated as not-passed
# rather than guessed at, matching the conservative default every other
# adapter in this repository uses for string-valued scores.
_AFFIRMATIVE_LABELS = {"true", "pass", "passed", "correct", "good", "yes", "1"}


def _stringify(value: Any) -> str:
    """Render an arbitrary value as the string EvalPort's schema requires."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def _json_safe(value: Any) -> Any:
    """Recursively convert datetimes/enums/nested pydantic models into
    plain JSON-safe values, preserving dicts/lists/primitives as-is.
    """
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return value.value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump) and not isinstance(value, (Mapping, str, bytes)):
        return _json_safe(model_dump())
    if isinstance(value, Mapping):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _item_to_dict(item: Any) -> Dict[str, Any]:
    """Normalize a Langfuse ``DatasetItem`` (pydantic) or a plain
    ``LocalExperimentItem``-shaped dict into a plain, snake_case,
    JSON-safe dict.

    ``DatasetItem.model_dump()`` is deliberately *not* used here: the
    installed package's Fern-generated model serializes with its own
    camelCase wire aliases (``expectedOutput``, ``datasetName``, ...) even
    when ``by_alias=False`` is passed (confirmed directly against the
    installed ``langfuse`` 4.14.4 package -- every field's declared
    ``alias``/``serialization_alias`` is ``None``, so the camelCase output
    comes from custom serialization logic, not a standard alias config).
    Round-tripping that camelCase dict back through ``from_openeval()``
    would produce keys that don't match what ``DatasetItem(...)`` or
    ``create_dataset_item(...)`` actually accept as keyword arguments, so
    fields are read directly off the model's declared ``model_fields``
    instead, which always match its real snake_case constructor kwargs.
    """
    model_fields = getattr(type(item), "model_fields", None)
    if model_fields is not None and not isinstance(item, Mapping):
        return {name: _json_safe(getattr(item, name, None)) for name in model_fields}
    if isinstance(item, Mapping):
        return dict(item)
    raise TypeError(
        f"Expected a Langfuse DatasetItem or a mapping with input/expected_output/"
        f"metadata keys, got {type(item)!r}"
    )


def to_openeval(
    items: Iterable[Any],
    suite_id: str = "langfuse_dataset",
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """Convert Langfuse dataset items into an EvalPort suite.

    ``items`` is whatever you'd get from ``Langfuse().get_dataset(name).items``
    (a list of real ``DatasetItem`` objects), or a list of plain
    ``LocalExperimentItem``-shaped dicts (``{"input": ..., "expected_output":
    ..., "metadata": ...}``) if you haven't created a stored dataset yet.
    """
    default_grader = (
        {
            "id": "default_exact_match",
            "type": "exact_match",
        }
        if grader_type == "exact_match"
        else {
            "id": "default_llm_judge",
            "type": "llm_judge",
            "params": {
                "model": "gpt-4o",
                "prompt": "Does the actual output '{output}' match the expected output '{expected}'? Answer yes or no.",
            },
        }
    )

    test_cases: List[Dict[str, Any]] = []
    for idx, raw_item in enumerate(items):
        row = _item_to_dict(raw_item)

        row_id = row.get("id")
        tc_id = _stringify(row_id) if row_id is not None else f"row_{idx}"

        input_value = row.get("input")
        input_text = _stringify(input_value) if input_value is not None else _stringify(row)

        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": input_text,
            "graders": [default_grader["id"]],
            "metadata": {"langfuse": {"item": row}},
        }

        expected_output = row.get("expected_output")
        if expected_output is not None:
            test_case["expected_output"] = _stringify(expected_output)

        test_cases.append(test_case)

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": suite_id,
        "graders": [default_grader],
        "test_cases": test_cases,
    }


def from_openeval(suite: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite's test cases into Langfuse-shaped rows.

    Returns a list of ``LocalExperimentItem``-shaped dicts, directly usable
    as ``Langfuse().run_experiment(data=...)``'s ``data`` argument. If a
    test case's ``metadata["langfuse"]["item"]`` is present (i.e. it
    round-trips a suite that came from ``to_openeval()`` above), that
    original item -- including Langfuse-only fields like ``id`` and
    ``status`` -- is reconstructed verbatim (it also works fine unpacked as
    ``create_dataset_item(dataset_name=..., **item)``, which accepts those
    same field names as keyword arguments); otherwise a fresh
    ``input``/``expected_output``/``metadata`` row is built from the test
    case directly.
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        lf_meta = (tc.get("metadata") or {}).get("langfuse")
        if isinstance(lf_meta, Mapping) and "item" in lf_meta:
            row = dict(lf_meta["item"])
        else:
            row: Dict[str, Any] = {"input": tc.get("input")}
            if "expected_output" in tc:
                row["expected_output"] = tc["expected_output"]
            meta: Dict[str, Any] = {}
            if tc.get("context"):
                meta["context"] = list(tc["context"])
            if meta:
                row["metadata"] = meta
        rows.append(row)
    return rows


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _evaluation_to_grader_result(evaluation: Any) -> Dict[str, Any]:
    """Build one EvalPort GraderResult from one Langfuse ``Evaluation``.

    Langfuse's ``Evaluation.value`` can be ``bool``, a raw ``int``/``float``
    (no guaranteed range), or an arbitrary ``str`` (for CATEGORICAL
    evaluators, e.g. a rubric label). EvalPort requires ``score`` to be
    ``null`` or a number in ``[0, 1]``, so each shape is handled explicitly
    rather than force-fit into a single rule:

    * ``bool`` (or ``data_type == "BOOLEAN"``) -> ``score`` is 1.0/0.0,
      ``passed`` matches the boolean directly.
    * numeric (or ``data_type == "NUMERIC"``) -> clamped into ``[0, 1]``;
      the un-clamped original is preserved under ``metadata["raw_value"]``
      whenever clamping actually changed it, so nothing is silently lost.
      Passes at >= 0.5, the same convention every other adapter here uses
      when a tool doesn't supply an explicit pass/fail flag.
    * anything else (``str`` / ``data_type == "CATEGORICAL"``) -> EvalPort
      has no numeric score to report, so ``score`` is ``null``; ``passed``
      is true only for a small, explicit set of affirmative labels, and the
      raw label is always preserved under ``metadata["value"]`` so nothing
      is lost even when the heuristic guesses wrong.
    """
    name = getattr(evaluation, "name", None)
    value = getattr(evaluation, "value", None)
    comment = getattr(evaluation, "comment", None)
    data_type = getattr(evaluation, "data_type", None)
    config_id = getattr(evaluation, "config_id", None)
    eval_metadata = getattr(evaluation, "metadata", None)

    grader_id = f"langfuse_{name}" if name else "langfuse_evaluation"

    metadata: Dict[str, Any] = {}
    if data_type is not None:
        metadata["data_type"] = data_type
    if config_id is not None:
        metadata["config_id"] = config_id
    if eval_metadata:
        metadata["langfuse_metadata"] = eval_metadata

    if isinstance(value, bool) or data_type == "BOOLEAN":
        score: Optional[float] = 1.0 if value else 0.0
        passed = bool(value)
    elif isinstance(value, (int, float)) or data_type == "NUMERIC":
        numeric = float(value)
        clamped = _clamp01(numeric)
        if clamped != numeric:
            metadata["raw_value"] = numeric
        score = clamped
        passed = clamped >= 0.5
    else:
        score = None
        metadata["value"] = value
        passed = isinstance(value, str) and value.strip().lower() in _AFFIRMATIVE_LABELS

    result: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": "custom",
        "score": score,
        "passed": passed,
        "metadata": metadata,
    }
    if comment:
        result["reason"] = comment
    return result


def _item_result_test_case_id(item_result: Any, idx: int) -> str:
    item = getattr(item_result, "item", None)
    item_id = None
    if item is not None:
        item_id = getattr(item, "id", None)
        if item_id is None and isinstance(item, Mapping):
            item_id = item.get("id")
    if item_id is not None:
        return _stringify(item_id)
    trace_id = getattr(item_result, "trace_id", None)
    if trace_id:
        return _stringify(trace_id)
    return f"row_{idx}"


def experiment_result_to_openeval(
    experiment_result: Union[Any, Iterable[Any]],
    suite_id: str = "langfuse_dataset",
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a Langfuse ``ExperimentResult`` into an EvalPort ResultSet.

    ``experiment_result`` is exactly what ``Langfuse().run_experiment(...)``
    hands back (an ``ExperimentResult`` with an ``item_results`` list), or,
    for callers who already have the per-item list, a bare iterable of
    ``ExperimentItemResult`` objects -- both are accepted via duck typing.

    Every ``Evaluation`` scored against an item becomes one EvalPort
    ``GraderResult`` (see ``_evaluation_to_grader_result`` for exactly how
    each value shape maps). Evaluations scored against the whole run
    (``ExperimentResult.run_evaluations``, when present) have no single
    test case to attach to, so they're preserved under this ResultSet's
    top-level ``metadata["langfuse"]["run_evaluations"]`` rather than
    dropped.

    ``started_at`` defaults to the current UTC time if not supplied (the
    schema requires it); pass it explicitly for reproducible output, as the
    tests here do.
    """
    if started_at is None:
        import datetime

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    item_results = getattr(experiment_result, "item_results", None)
    if item_results is None:
        item_results = list(experiment_result)  # bare iterable of ExperimentItemResult

    run_evaluations = getattr(experiment_result, "run_evaluations", None) or []
    experiment_id = getattr(experiment_result, "experiment_id", None)
    result_run_name = getattr(experiment_result, "run_name", None)
    dataset_run_id = getattr(experiment_result, "dataset_run_id", None)

    if run_id is None:
        run_id = result_run_name or experiment_id or "langfuse_run"

    out_results: List[Dict[str, Any]] = []
    for idx, item_result in enumerate(item_results):
        test_case_id = _item_result_test_case_id(item_result, idx)
        output = getattr(item_result, "output", None)
        evaluations = getattr(item_result, "evaluations", None) or []

        grader_results = [_evaluation_to_grader_result(ev) for ev in evaluations]

        result_metadata: Dict[str, Any] = {}
        item_trace_id = getattr(item_result, "trace_id", None)
        item_dataset_run_id = getattr(item_result, "dataset_run_id", None)
        if item_trace_id:
            result_metadata["trace_id"] = item_trace_id
        if item_dataset_run_id:
            result_metadata["dataset_run_id"] = item_dataset_run_id

        result: Dict[str, Any] = {
            "test_case_id": test_case_id,
            "grader_results": grader_results,
            "passed": all(gr["passed"] for gr in grader_results) if grader_results else False,
        }
        if output is not None:
            result["actual_output"] = _stringify(output)
        if result_metadata:
            result["metadata"] = result_metadata

        out_results.append(result)

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": out_results,
        "runner": {"name": "langfuse-openeval-adapter"},
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at

    top_metadata: Dict[str, Any] = {}
    if run_evaluations:
        top_metadata["run_evaluations"] = [
            {
                "name": getattr(ev, "name", None),
                "value": getattr(ev, "value", None),
                "comment": getattr(ev, "comment", None),
                "data_type": getattr(ev, "data_type", None),
            }
            for ev in run_evaluations
        ]
    if experiment_id:
        top_metadata["experiment_id"] = experiment_id
    if dataset_run_id:
        top_metadata["dataset_run_id"] = dataset_run_id
    if top_metadata:
        result_set["metadata"] = {"langfuse": top_metadata}

    return result_set
