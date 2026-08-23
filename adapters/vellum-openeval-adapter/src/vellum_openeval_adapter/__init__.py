"""Vellum <-> EvalPort adapter.

Standalone converter between Vellum (`pip install vellum-ai`, tested against
1.14.7) Test Suite data and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport). Built for issue #16.

Vellum's own vocabulary is "Test Suite" / "Test Case" / "Test Suite Run" --
about as literal a naming match to EvalPort as any framework in this
ecosystem gets. Confirmed by importing the real installed package and
reading `model_fields` directly, not the docs:

  TestSuiteTestCase                 ['id', 'external_id', 'label',
                                      'input_values', 'evaluation_values']
  TestSuiteRunExecution              ['id', 'test_case_id', 'outputs',
                                      'metric_results']
  TestSuiteRunExecutionMetricResult  ['metric_id', 'outputs', 'metric_label',
                                      'metric_definition']

One real wrinkle this adapter solves (see issue #16): `input_values` and
`evaluation_values` are each a `List[TestCase*VariableValue]` -- a typed,
*named* variable system (`TestCaseStringVariableValue`,
`TestCaseNumberVariableValue`, `TestCaseJsonVariableValue`,
`TestCaseChatHistoryVariableValue`, `TestCaseSearchResultsVariableValue`,
`TestCaseErrorVariableValue`, `TestCaseFunctionCallVariableValue`,
`TestCaseArrayVariableValue`, `TestCaseAudioVariableValue`,
`TestCaseImageVariableValue`, `TestCaseVideoVariableValue`,
`TestCaseDocumentVariableValue`), not a flat dict or a single string.
`variables_to_input()` converts the whole named list into EvalPort's
`input: string | string[]`, honestly, per variable, rather than picking one
variable and dropping the rest.

A second wrinkle on the results side: `TestSuiteRunExecutionMetricResult
.outputs` is also a typed union (`TestSuiteRunMetricStringOutput`/
`NumberOutput`/`JsonOutput`/`ErrorOutput`/`ArrayOutput`). Only the
`NUMBER` variant is a real EvalPort `GraderResult.score` (which the spec
requires to be `null` or in `[0, 1]`); every other variant becomes
`score: null` with the raw value preserved in `metadata`, per the same
convention every other adapter in this repo uses for a non-numeric grader
result (see `map_metric_output()`).

This is a pydantic-v2-model SDK (not a plain dataclass), so every accessor
below reads Vellum objects via `getattr`/`dict.get` (handled by `_get()`)
so callers can pass either the real SDK objects or plain dicts of the same
shape.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "stringify_variable_value",
    "variables_to_input",
    "map_metric_output",
    "to_openeval",
    "from_openeval",
    "results_to_openeval",
    "__version__",
]
__version__ = "0.1.0"

_DEFAULT_GRADER_ID = "gr_vellum_default"

# Preferred single-variable names, in priority order, for the single-variable
# fast path in variables_to_input() -- mirrors the preferred-key convention
# used by the dict-flattening adapters (e.g. literalai-openeval-adapter).
_PREFERRED_VARIABLE_NAMES = ("input", "question", "prompt", "text", "query")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _model_dump(obj: Any) -> Any:
    """Best-effort JSON-safe dump of a Vellum pydantic model, a plain dict,
    or a scalar -- used both to stringify a value and to preserve it
    losslessly in metadata for round-tripping."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _model_dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_model_dump(v) for v in obj]
    return str(obj)


# --- Wrinkle 1: typed, named variable-value system --------------------------

def stringify_variable_value(variable: Any) -> str:
    """Convert one `TestCase*VariableValue` (or `TestSuiteRunExecution*Output`
    / `TestSuiteRunMetric*Output` -- all share the same {name, type, value}
    shape) into a plain string.

    - STRING: the value itself (empty string if `None`).
    - NUMBER: `str(value)`.
    - CHAT_HISTORY: each `ChatMessage` rendered as "role: text", one per line
      (falling back to a JSON dump of non-text content, e.g. function calls
      or images, since `ChatMessage.text` is `None` for those).
    - Everything else (JSON, ARRAY, SEARCH_RESULTS, ERROR, FUNCTION_CALL,
      AUDIO, IMAGE, VIDEO, DOCUMENT): JSON-serialized via `_model_dump()`,
      since none of these have an honest single-string representation and
      guessing one would be lossy in a way JSON isn't.
    """
    vtype = _get(variable, "type")
    value = _get(variable, "value")

    if vtype == "STRING":
        return value if isinstance(value, str) else ("" if value is None else str(value))
    if vtype == "NUMBER":
        return "" if value is None else str(value)
    if vtype == "CHAT_HISTORY":
        if not value:
            return ""
        lines = []
        for msg in value:
            role = _get(msg, "role", "USER")
            text = _get(msg, "text")
            if text is None:
                content = _get(msg, "content")
                text = json.dumps(_model_dump(content), sort_keys=True) if content is not None else ""
            lines.append(f"{role}: {text}")
        return "\n".join(lines)

    # JSON, ARRAY, SEARCH_RESULTS, ERROR, FUNCTION_CALL, AUDIO, IMAGE, VIDEO,
    # DOCUMENT, and any future variant this adapter doesn't special-case yet.
    return json.dumps(_model_dump(value), sort_keys=True, default=str)


def variables_to_input(variables: Sequence[Any]) -> str | List[str]:
    """Convert a `List[TestCase*VariableValue]` into EvalPort's
    `input: string | string[]`.

    Single-variable fast path: if there's exactly one variable and it's a
    `STRING` type, return its value directly (the common case -- a single
    prompt variable named e.g. "input" or "question" -- reads as a plain
    string rather than a one-element array or a "name: value" prefix).

    Otherwise (multiple variables, or a single non-string variable): each
    variable becomes its own `"{name}: {stringified value}"` entry in an
    array, in the same order Vellum returned them -- preserving every named
    variable rather than picking one and silently dropping the rest, per
    the honest-flattening convention this repo's adapters follow for
    schema-free/multi-field source data.
    """
    if not variables:
        raise ValueError("cannot convert an empty variable list to input")

    if len(variables) == 1:
        v = variables[0]
        vtype = _get(v, "type")
        if vtype == "STRING":
            value = _get(v, "value")
            return value if isinstance(value, str) else ("" if value is None else str(value))

    return [f"{_get(v, 'name', '?')}: {stringify_variable_value(v)}" for v in variables]


# --- Wrinkle 2: typed metric-output union -----------------------------------

def map_metric_output(
    metric_result: Any,
    pass_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Convert one `TestSuiteRunExecutionMetricResult` into an EvalPort
    `GraderResult` dict (spec/schemas/resultset.json's `grader_results[]`).

    `metric_result.outputs` is itself a list (a metric can report more than
    one named output, e.g. a raw score plus a rationale string) -- this
    looks for the first `NUMBER`-typed output as the real `score`, clamped
    into EvalPort's required `[0, 1]` range with the raw value preserved
    under `metadata.openeval.raw_score` (the spec's own reserved metadata
    key for exactly this purpose). If no `NUMBER` output exists, `score`
    stays honestly `null` rather than a fabricated number.

    `passed` has no universal semantic Vellum hands you directly, so it's
    derived with a documented, overridable heuristic: `score >= pass_threshold`
    when a numeric score exists; otherwise `False` if any output is `ERROR`
    type, else `True` (a non-error, non-numeric output -- e.g. a STRING
    rationale or a JSON verdict -- is treated as "ran without failing",
    which is the most honest default when nothing tells you otherwise). The
    full raw output list is always preserved under `metadata.vellum.outputs`
    so a caller who needs different pass/fail semantics can re-derive them.

    `type` (required by the schema on every `GraderResult`) is set to
    `"custom"` -- Vellum's `TestSuiteRunExecutionMetricDefinition` doesn't
    expose a grader-type vocabulary EvalPort's well-known types (exact_match,
    semantic_similarity, llm_judge, ...) could be mapped onto, so claiming
    one would be a guess. `metric_label`/`metric_definition` (when present)
    are preserved in `metadata.vellum` instead.
    """
    outputs = _get(metric_result, "outputs") or []
    metric_id = _get(metric_result, "metric_id") or "unknown_metric"
    metric_label = _get(metric_result, "metric_label")
    metric_definition = _get(metric_result, "metric_definition")

    score: Optional[float] = None
    raw_score: Any = None
    has_error = False
    for out in outputs:
        otype = _get(out, "type")
        if otype == "NUMBER" and score is None:
            raw_value = _get(out, "value")
            if raw_value is not None:
                raw_score = raw_value
                score = max(0.0, min(1.0, float(raw_value)))
        if otype == "ERROR":
            has_error = True

    if score is not None:
        passed = score >= pass_threshold
    else:
        passed = not has_error

    metadata: Dict[str, Any] = {
        "vellum": {
            "outputs": [_model_dump(out) for out in outputs],
        }
    }
    if metric_label is not None:
        metadata["vellum"]["metric_label"] = metric_label
    if metric_definition is not None:
        metadata["vellum"]["metric_definition"] = _model_dump(metric_definition)
    if raw_score is not None:
        metadata["openeval"] = {"raw_score": raw_score}

    grader_result: Dict[str, Any] = {
        "grader_id": str(metric_label or metric_id),
        "type": "custom",
        "score": score,
        "passed": passed,
        "metadata": metadata,
    }
    return grader_result


# --- Public API: to_openeval / from_openeval / results_to_openeval ---------

def to_openeval(
    test_cases: Sequence[Any],
    grader_id: str = _DEFAULT_GRADER_ID,
    grader_type: str = "llm_judge",
    grader_params: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a list of Vellum `TestSuiteTestCase` (or plain dicts with the
    same shape) into an EvalPort suite (dict). Conforms to
    `spec/schemas/suite.json` -- pass the result to
    `openeval.validate.validate_suite()` to confirm.

    Vellum test cases don't carry a grader/metric definition of their own
    (metrics are configured on the Test Suite separately from its test
    cases, and aren't visible on `TestSuiteTestCase`), so `graders` must be
    required elsewhere: this generates a single default grader
    (`grader_id`/`grader_type`/`grader_params`, `llm_judge` by default,
    matching the convention `literalai-openeval-adapter` uses for the same
    reason) that every test case references, rather than guessing a
    per-case grader from data that isn't there.
    """
    if not test_cases:
        raise ValueError("test_cases must contain at least one test case")

    if grader_type == "llm_judge":
        grader: Dict[str, Any] = {
            "id": grader_id,
            "type": "llm_judge",
            "params": grader_params
            or {
                "model": "gpt-4o",
                "prompt": (
                    'Does {output} correctly satisfy {input}? Expected: {expected}. '
                    'Return JSON: {"score": 0.0-1.0}.'
                ),
            },
        }
    elif grader_type == "exact_match":
        grader = {"id": grader_id, "type": "exact_match", "params": grader_params or {"ignore_case": True}}
    else:
        grader = {"id": grader_id, "type": grader_type, "params": grader_params or {"handler": grader_type}}

    out_test_cases: List[Dict[str, Any]] = []
    for idx, tc in enumerate(test_cases):
        tc_id = _get(tc, "id") or _get(tc, "external_id") or f"tc_{idx}"
        input_values = _get(tc, "input_values") or []
        evaluation_values = _get(tc, "evaluation_values") or []

        entry: Dict[str, Any] = {
            "id": str(tc_id),
            "input": variables_to_input(input_values) if input_values else "",
            "graders": [grader_id],
            "metadata": {
                "vellum": {
                    "original_input_values": [_model_dump(v) for v in input_values],
                    "original_evaluation_values": [_model_dump(v) for v in evaluation_values],
                }
            },
        }
        label = _get(tc, "label")
        if label:
            entry["metadata"]["vellum"]["label"] = label

        if evaluation_values:
            entry["expected_output"] = variables_to_input(evaluation_values)
            if isinstance(entry["expected_output"], list):
                # expected_output must be a plain string per spec/schemas/testcase.json;
                # only `input` allows an array. Join multi-variable expected values.
                entry["expected_output"] = "\n".join(entry["expected_output"])

        out_test_cases.append(entry)

    return {
        "version": OPENEVAL_VERSION,
        "id": id or "vellum_suite",
        "name": name or id or "Vellum test suite",
        "graders": [grader],
        "test_cases": out_test_cases,
        "metadata": {"openeval": {"source": "vellum"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite's test cases into Vellum
    `upsert_test_suite_test_case()`-ready dicts:
    `{"input_values": [...], "evaluation_values": [...], "label": Optional[str]}`,
    each a plain dict shaped like `NamedTestCase*VariableValueRequest`.

    If a test case's `metadata["vellum"]["original_input_values"]` /
    `["original_evaluation_values"]` are present (i.e. it round-trips a
    suite produced by `to_openeval()` above), the original named-variable
    list is restored verbatim rather than re-wrapping the flattened string
    -- this is what makes the round trip lossless for suites that came from
    this adapter in the first place. Otherwise, the test case's `input` /
    `expected_output` strings are wrapped as a single `STRING` variable
    named `"input"` / `"expected_output"`, the same honest fallback every
    adapter in this repo uses for data it didn't produce itself.
    """
    out: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        metadata = dict(tc.get("metadata") or {})
        vellum_meta = metadata.get("vellum") or {}

        original_input = vellum_meta.get("original_input_values")
        if isinstance(original_input, list) and original_input:
            input_values = original_input
        else:
            input_values = [{"name": "input", "type": "STRING", "value": tc.get("input")}]

        original_eval = vellum_meta.get("original_evaluation_values")
        if isinstance(original_eval, list) and original_eval:
            evaluation_values = original_eval
        elif "expected_output" in tc:
            evaluation_values = [
                {"name": "expected_output", "type": "STRING", "value": tc["expected_output"]}
            ]
        else:
            evaluation_values = []

        entry: Dict[str, Any] = {
            "input_values": input_values,
            "evaluation_values": evaluation_values,
        }
        label = vellum_meta.get("label")
        if label:
            entry["label"] = label
        out.append(entry)
    return out


def results_to_openeval(
    executions: Any,
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    pass_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Convert a Vellum Test Suite Run's executions into an EvalPort
    ResultSet. Conforms to `spec/schemas/resultset.json` -- pass the result
    to `openeval.validate.validate_result_set()` to confirm.

    `executions` accepts a `PaginatedTestSuiteRunExecutionList` (the real
    return type of `client.test_suite_runs.list_executions(id=run_id)`,
    read via its `.results` attribute), a bare list of
    `TestSuiteRunExecution`-shaped objects/dicts, or anything else with a
    `.results`/`["results"]` list.

    `started_at` defaults to the current UTC time if not supplied (the
    schema requires it); pass it explicitly for reproducible output.
    """
    if started_at is None:
        import datetime

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    items = _get(executions, "results", None)
    if items is None:
        items = list(executions)  # bare iterable of TestSuiteRunExecution

    results: List[Dict[str, Any]] = []
    for idx, execution in enumerate(items):
        test_case_id = _get(execution, "test_case_id") or f"tc_{idx}"
        outputs = _get(execution, "outputs") or []
        metric_results = _get(execution, "metric_results") or []

        grader_results = [map_metric_output(mr, pass_threshold=pass_threshold) for mr in metric_results]

        result: Dict[str, Any] = {
            "test_case_id": str(test_case_id),
            "grader_results": grader_results,
            "passed": all(gr["passed"] for gr in grader_results) if grader_results else False,
        }
        if outputs:
            result["actual_output"] = "\n".join(
                f"{_get(o, 'name', '?')}: {stringify_variable_value(o)}" for o in outputs
            ) if len(outputs) > 1 else stringify_variable_value(outputs[0])
            result["metadata"] = {"vellum": {"outputs": [_model_dump(o) for o in outputs]}}

        results.append(result)

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "runner": {"name": "vellum-openeval-adapter"},
    }
