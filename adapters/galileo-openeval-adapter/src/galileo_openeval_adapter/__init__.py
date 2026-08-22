"""Galileo <-> EvalPort adapter.

Standalone converter between the Galileo platform's `Dataset` content rows
and `Trace`/`Span` objects (the current `rungalileo/galileo-python` package,
`pip install galileo` -- NOT the older, deprecated `promptquality`
distribution, which is a different client for the same platform with an
entirely different, flat-table `PromptRow` data model) and the EvalPort
interchange format (https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than an in-repo Galileo
change: `rungalileo/galileo-python` has no CONTRIBUTING.md stating a
preferred integration path, and Galileo's SDK is fundamentally a client for
a hosted SaaS product (dataset storage, experiment execution, and most
built-in scorers all require a live API call) -- the same "standalone
package, zero footprint on the target framework" shape used by the AutoGen,
CrewAI, Giskard, and Guardrails adapters in this ecosystem, and the more
conservative choice given this adapter can only exercise the parts of the
SDK that work fully offline (see "What this adapter does NOT cover" below).

Two independent conversions are provided, mirroring the two things Galileo
itself treats as separate concerns (the dataset you define, and the traces/
scores a run of your application against that dataset produces):

- `to_openeval()` / `from_openeval()` -- Galileo `Dataset` *content* (a
  `list[dict[str, Any]]`, exactly the shape `galileo.Dataset(content=...)`
  accepts -- verified against `galileo/dataset.py`'s real constructor, not
  guessed) <-> an EvalPort `EvalSuite`.
- `spans_to_openeval()` -- real `galileo.Trace`/`galileo.LlmSpan` (or any
  other `Span` subtype) objects, scored by real `galileo.LocalMetric`
  instances, -> an EvalPort `ResultSet`.

Mapping, verified against the real, installed `galileo==2.6.0` source (not
the docs -- the docs largely describe the hosted product, not the local
Python object shapes):

| Galileo (`Dataset` content row, a plain dict) | EvalPort `TestCase` field |
|---|---|
| `input_key` (default `"input"`)               | `input`                    |
| `expected_output_key` (default `"output"`)     | `expected_output`          |
| every other key in the row                     | `metadata["galileo"]["row"]` (the full original row, for a lossless round trip) |

Galileo's `Dataset.__init__` places **no constraint at all** on row keys --
`content: list[dict[str, Any]]`, arbitrary keys, arbitrary values (verified
by reading the constructor directly: no validation beyond `name` being
required). The `"input"`/`"output"` convention comes from Galileo's own
docstring examples, not an enforced schema, hence the configurable
`input_key`/`expected_output_key` rather than hardcoding it.

| Galileo (`LocalMetric.scorer_fn(trace_or_span)` return value) | EvalPort `GraderResult` field |
|---|---|
| the metric's `name`               | `grader_id` (slug-normalized) |
| numeric half of the return value  | `score` (clamped to `[0, 1]`) |
| non-numeric return value (`str`/`list`/`dict`) | `score: null`, raw value preserved in `metadata["raw_value"]` |
| the `(score, metadata)` tuple's second element, if returned | `metadata["scorer_metadata"]` |

## What this adapter does NOT cover, and why

Galileo's SDK exposes four metric classes (`LocalMetric`, `GalileoMetric`,
`LlmMetric`, `CodeMetric`), but only `LocalMetric` is scored by a plain
Python callable this adapter can call directly and offline. The other
three -- built-in Galileo scorers (`Metric.metrics.correctness`,
`.completeness`, ...), custom LLM-judge metrics, and code metrics --
are executed server-side against Galileo's hosted scoring service once a
`LogStream`/`Experiment` actually runs; there is no local method that
produces a real score for them without an authenticated network call this
adapter cannot make (and, per this project's own hard rule against
fabricating results, will not pretend to). `spans_to_openeval()` raises
`ValueError` if a non-`LocalMetric` is passed, rather than silently
skipping it or inventing a placeholder score.

Likewise, `Dataset.get_content()` and `Experiment` run results both require
a live, authenticated call to Galileo's API to produce real data --
`to_openeval()`/`from_openeval()` work purely on the `content` list you
already have in hand (whether freshly authored or already fetched), and
`spans_to_openeval()` works on `Trace`/`Span` objects you construct or
already have, not on a live experiment run this adapter triggers itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "spans_to_openeval", "__version__"]
__version__ = "0.1.0"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clamp01(value: Optional[float]) -> Optional[float]:
    """Clamp a numeric score into EvalPort's required [0, 1] range.

    `LocalMetric.scorer_fn` is a completely arbitrary user-written Python
    function (verified by reading `galileo/metric.py`'s `MetricValueType =
    Union[float, int, str, None, List[...], Dict[str, ...]]`) -- nothing
    about Galileo's contract bounds a numeric return value to [0, 1], so
    this is a real, documented lossy step for any scorer that returns
    outside that range, not an assumption that all of them do.
    """
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _slugify(name: str) -> str:
    """Normalize a Galileo metric name into a grader_id, matching the
    normalization every other adapter in this ecosystem applies to
    human-readable metric/feedback names."""
    return "_".join(name.strip().lower().replace("-", " ").split())


def _extract_text(value: Any) -> Optional[str]:
    """Flatten a Galileo `Trace`/`Span` `.input`/`.output` value to plain
    text. Verified against `galileo.Message`/`galileo.MessageRole` and the
    real `Trace`/`LlmSpan` field types (`str | list[Message] | dict |
    None`, read from `galileo/dataset.py` and the constructed instances
    directly, not guessed): a plain string is returned as-is; a single
    `Message` (or a `{"role": ..., "content": ...}` dict) yields its
    `content`; a list of messages (multi-turn chat) is flattened to
    `"role: content"` lines, joined with newlines, so the ordering and
    speaker of a real conversation isn't collapsed into an ambiguous single
    string.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            role = _get(item, "role")
            content = _get(item, "content")
            role_str = getattr(role, "value", role)
            if content is not None:
                lines.append(f"{role_str}: {content}" if role_str else str(content))
            else:
                lines.append(str(item))
        return "\n".join(lines)
    content = _get(value, "content", None)
    if content is not None:
        return content if isinstance(content, str) else str(content)
    return str(value)


def to_openeval(
    content: Sequence[Dict[str, Any]],
    *,
    suite_id: str,
    name: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    input_key: str = "input",
    expected_output_key: str = "output",
    grader_id: str = "gr_galileo_metrics",
    grader_handler: str = "galileo:metrics",
) -> Dict[str, Any]:
    """Export Galileo `Dataset` content rows to an EvalPort-shaped suite (dict).

    `content` is any sequence of plain dicts -- exactly what you would pass
    as `galileo.Dataset(name=..., content=content)`, or what
    `Dataset.get_content()` would give you after flattening its
    `DatasetRow.values_dict`. Each row's `input_key` value becomes the test
    case's `input` (required -- raises `ValueError` if missing or falsy);
    `expected_output_key`, if present, becomes `expected_output`. Every
    other key on the row -- and the row in full, for exact reconstruction --
    is preserved under `metadata["galileo"]["row"]`, since Galileo's content
    rows are genuinely schema-free (verified by reading `Dataset.__init__`:
    no key names are required or reserved) and this adapter has no way to
    know in advance which extra columns matter to your dataset.

    `input`/`expected_output` are only accepted as `str`, or a `list[str]`
    for `input` (EvalPort's multi-turn shape) -- matching
    `spec/schemas/testcase.json`'s `input` type exactly (`string` or
    `array of string, minItems 1`). A row whose `input_key` value is some
    other type (a number, a nested object, a list of chat-message dicts)
    raises `ValueError` rather than silently stringifying it, since
    EvalPort's schema genuinely doesn't have a slot for that shape and
    guessing a coercion isn't this adapter's call to make -- flatten it
    yourself first if that's the right move for your data.

    Test case IDs: an explicit `ids[i]` if you pass one, else `row["id"]` /
    `row["row_id"]` if the row carries one (as a `DatasetRow`-derived dict
    would), else an auto `tc_{i}`.

    Galileo doesn't attach specific metrics to a dataset row up front --
    which `LocalMetric`/`GalileoMetric`/... instances score a run is chosen
    separately, when you call `log_stream.set_metrics([...])` or configure
    an `Experiment`. So every test case in the exported suite references
    one placeholder `custom` grader (`grader_id`/`grader_handler`,
    overridable) rather than guessing which metrics you intend to run; the
    real, per-metric grading shows up honestly in `spans_to_openeval()`'s
    output instead, once metrics have actually executed against real spans.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance.
    """
    rows = list(content)
    resolved_ids: List[str] = []
    for i, row in enumerate(rows):
        if ids is not None:
            resolved_ids.append(ids[i])
        else:
            row_id = _get(row, "id") or _get(row, "row_id")
            resolved_ids.append(str(row_id) if row_id else f"tc_{i}")

    test_case_dicts: List[Dict[str, Any]] = []
    for i, (row, tc_id) in enumerate(zip(rows, resolved_ids)):
        if input_key not in row or row[input_key] in (None, "", []):
            raise ValueError(
                f"Row at index {i} (id={tc_id!r}) has no {input_key!r} key (or it's empty) -- "
                "EvalPort's TestCase.input is required and non-empty. Pass a different "
                "input_key= if your dataset uses a different column name."
            )
        input_value = row[input_key]
        if isinstance(input_value, list):
            if not input_value or not all(isinstance(v, str) for v in input_value):
                raise ValueError(
                    f"Row at index {i} (id={tc_id!r}): {input_key!r} is a list, but EvalPort's "
                    "multi-turn `input` requires a non-empty list of plain strings."
                )
        elif not isinstance(input_value, str):
            raise ValueError(
                f"Row at index {i} (id={tc_id!r}): {input_key!r} is a {type(input_value).__name__}, "
                "not a str or list[str] -- EvalPort's TestCase.input schema only accepts those two "
                "shapes (spec/schemas/testcase.json). Flatten this value to text before calling "
                "to_openeval() if it's meant to be the model input."
            )

        ec: Dict[str, Any] = {"id": tc_id, "input": input_value, "graders": [grader_id]}

        expected_output = row.get(expected_output_key)
        if expected_output not in (None, ""):
            ec["expected_output"] = (
                expected_output if isinstance(expected_output, str) else str(expected_output)
            )

        ec["metadata"] = {"galileo": {"row": dict(row)}}

        test_case_dicts.append(ec)

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name or f"Galileo dataset ({suite_id})",
        "test_cases": test_case_dicts,
        "graders": [{
            "id": grader_id,
            "type": "custom",
            "params": {"handler": grader_handler},
            "description": (
                "Placeholder: Galileo metrics are attached to a LogStream/Experiment at "
                "run time, not to a dataset row up front. See spans_to_openeval() for the "
                "real, per-metric grader results once LocalMetrics have actually scored a run."
            ),
        }],
        "metadata": {"openeval": {"source": "galileo"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of Galileo `Dataset`-ready rows.

    Each returned dict is a plain `{str: Any}` row suitable for
    `galileo.Dataset(name=..., content=<this list>)` directly.

    If the test case's `metadata["galileo"]["row"]` is present (i.e. this
    suite round-trips one produced by `to_openeval()` above), the *exact*
    original row is restored byte-for-byte, including every extra column
    `to_openeval()` didn't otherwise touch -- a genuinely lossless round
    trip for this adapter's own suites. Otherwise (a suite authored by hand,
    or by a different EvalPort-speaking tool), a fresh minimal row is built
    from `input`/`expected_output` alone: `{"input": ..., "output": ...}`,
    matching Galileo's own documented column-naming convention.

    Multi-turn `input` (a list of strings) is passed through as-is --
    Galileo's `Dataset` content rows have no type constraint on values, so
    unlike DeepEval's single-turn-only `LLMTestCase`, there's nothing here
    that rejects it; it's simply stored under the row's `input_key` value
    as a list, exactly as EvalPort represented it.
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        metadata = tc.get("metadata") or {}
        galileo_meta = metadata.get("galileo") or {}
        if "row" in galileo_meta:
            rows.append(dict(galileo_meta["row"]))
            continue

        row: Dict[str, Any] = {"input": tc["input"]}
        if "expected_output" in tc:
            row["output"] = tc["expected_output"]
        rows.append(row)
    return rows


def spans_to_openeval(
    spans: Sequence[Any],
    *,
    metrics: Sequence[Any],
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    runner_name: str = "galileo",
    runner_version: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    pass_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Export real, locally-scored Galileo `Trace`/`Span` objects to an
    EvalPort `ResultSet`.

    `spans` is any sequence of `galileo.Trace` / `galileo.LlmSpan` / other
    `Span` subtype instances -- one per test case, representing what your
    application actually produced for that test case's input. These are
    genuine pydantic objects you construct directly (or that Galileo's own
    logger/decorator produces); this function does not fabricate them or
    call the Galileo API to get them.

    `metrics` must be a non-empty sequence of `galileo.LocalMetric`
    instances. Each metric's real `scorer_fn` is called directly against
    each span -- `metric.scorer_fn(span)` -- exactly matching Galileo's own
    documented scoring contract (`LocalMetric.scorer_fn: Callable[[Trace |
    Span], MetricValueType | tuple[MetricValueType, dict]]`, read from
    `galileo/metric.py`). This is real scoring, not a re-implementation:
    the same function object Galileo would call when a `LogStream` runs
    with this metric attached is what runs here.

    Passing a `GalileoMetric` (a built-in hosted scorer, e.g.
    `Metric.metrics.correctness`), `LlmMetric`, or `CodeMetric` raises
    `ValueError` immediately -- none of those have a local `scorer_fn`;
    computing a real score for them requires an authenticated call to
    Galileo's hosted scoring service, which this adapter cannot make and
    will not fabricate a placeholder for.

    A `MetricValueType` return value (`float | int | str | None |
    List[...] | Dict[str, ...]` -- Galileo's own type, not narrowed here)
    that isn't numeric becomes `score: null` with the raw value preserved
    verbatim in the grader result's `metadata["raw_value"]`, since EvalPort
    requires `score` to be a number in `[0, 1]` or `null`
    (`spec/schemas/resultset.json`) -- a categorical or structured scorer
    result is real data, not something to force into a fabricated number.
    A numeric value is clamped into `[0, 1]` (see `_clamp01`) and compared
    against `pass_threshold` for `passed`; a non-numeric value's `passed`
    falls back to "did the scorer return a truthy value at all", which is
    honestly a weaker signal than a real threshold comparison -- documented
    here rather than silently treated as equivalent.

    `test_case_id` correlation: pass the *same* `ids` list you gave
    `to_openeval()` (matched positionally against `spans`) to keep results
    correlated to the suite that produced them; if omitted, falls back to
    `tc_{index}`, matching `to_openeval()`'s own default when it wasn't
    given explicit ids either.

    `started_at` defaults to the current UTC time in ISO 8601 if omitted.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass
    it to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    if not metrics:
        raise ValueError(
            "`metrics` must contain at least one galileo.LocalMetric -- Galileo has no "
            "metrics implicitly attached to a span the way, e.g., a MetricData list "
            "already attached to a finished DeepEval TestResult would."
        )
    for m in metrics:
        if not callable(_get(m, "scorer_fn")):
            raise ValueError(
                f"Metric {_get(m, 'name')!r} has no callable local `scorer_fn` -- only "
                "galileo.LocalMetric is supported here. Built-in GalileoMetric scorers, "
                "LlmMetric, and CodeMetric all require a live call to Galileo's hosted "
                "scoring service to produce a real score, which this offline adapter "
                "cannot make (and will not fabricate a placeholder for)."
            )

    if started_at is None:
        from datetime import datetime, timezone
        started_at = datetime.now(timezone.utc).isoformat()

    span_list = list(spans)
    if not span_list:
        raise ValueError("`spans` must contain at least one span -- EvalPort's ResultSet.results requires minItems: 1.")

    results: List[Dict[str, Any]] = []
    for i, span in enumerate(span_list):
        test_case_id = str(ids[i]) if ids is not None else f"tc_{i}"

        result: Dict[str, Any] = {"test_case_id": test_case_id}
        actual_output = _extract_text(_get(span, "output"))
        if actual_output is not None:
            result["actual_output"] = actual_output

        grader_results: List[Dict[str, Any]] = []
        for m in metrics:
            metric_name = _get(m, "name") or "unknown_metric"
            raw = m.scorer_fn(span)
            if isinstance(raw, tuple) and len(raw) == 2:
                raw_value, scorer_metadata = raw
            else:
                raw_value, scorer_metadata = raw, None

            is_numeric = isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
            score = _clamp01(raw_value) if is_numeric else None

            gr: Dict[str, Any] = {
                "grader_id": _slugify(metric_name),
                "type": "custom",
                "score": score,
                "passed": (score >= pass_threshold) if score is not None else bool(raw_value),
            }

            gr_metadata: Dict[str, Any] = {"metric_name": metric_name}
            if not is_numeric and raw_value is not None:
                gr_metadata["raw_value"] = raw_value
            if scorer_metadata:
                gr_metadata["scorer_metadata"] = dict(scorer_metadata)
            gr["metadata"] = gr_metadata

            grader_results.append(gr)

        result["grader_results"] = grader_results
        result["passed"] = all(g["passed"] for g in grader_results)

        span_metadata: Dict[str, Any] = {}
        user_metadata = _get(span, "user_metadata")
        if user_metadata:
            span_metadata["user_metadata"] = dict(user_metadata)
        dataset_metadata = _get(span, "dataset_metadata")
        if dataset_metadata:
            span_metadata["dataset_metadata"] = dict(dataset_metadata)
        span_id = _get(span, "id")
        if span_id:
            span_metadata["span_id"] = span_id
        if span_metadata:
            result["metadata"] = {"galileo": span_metadata}

        results.append(result)

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
        "pass_rate": (passed / total) if total else 0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0,
    }

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at or started_at,
        "runner": {"name": runner_name, "version": runner_version or __version__},
        "results": results,
        "summary": summary,
        "metadata": {"openeval": {"source": "galileo"}},
    }
