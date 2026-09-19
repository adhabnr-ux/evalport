"""TruLens <-> EvalPort adapter.

Standalone converter between TruLens (https://github.com/truera/trulens,
now published as the `trulens` / `trulens-core` packages by Snowflake) and
the EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Why this maps to a ResultSet rather than an EvalSuite, unlike most of this
repo's other adapters: TruLens is observability-first. It instruments an
already-built app and produces `Record`s (one per app call, already executed)
plus `FeedbackResult`s (scores computed after the fact) -- there is no
TruLens-native concept of an unexecuted "suite of test cases" to convert, the
way AutoGen's `EvalTask`/`EvalResult` or Ragas's dataset rows are. So
`to_openeval()` here produces an EvalPort `ResultSet` (already-run results),
and `from_openeval()` goes the other way: an EvalPort `EvalSuite`'s test
cases become a TruLens ground-truth golden set, ready to drive
`trulens.feedback.GroundTruthAgreement` -- the actual, documented shape
TruLens itself expects for that purpose (see this module's `from_openeval`
docstring for the exact source verified).

Field names below (`Record.record_id/app_id/main_input/main_output/
main_error/ts`, `FeedbackResult.record_id/name/result/error/status`,
`GroundTruthAgreement`'s `[{"query": ..., "expected_response": ...}]` golden
set shape) were verified by reading the real, installed `trulens-core==2.14.0`
source directly (`trulens.core.schema.record.Record`,
`trulens.core.schema.feedback.FeedbackResult`,
`trulens.feedback.GroundTruthAgreement.agreement_measure`'s own docstring
example) -- not guessed from memory or from older `trulens_eval`-era docs.
Both classes are duck-typed here (attribute-or-dict access via `_get()`, the
same pattern AutoGen's adapter uses) rather than imported, so this package has
no hard dependency on TruLens itself; see pyproject.toml's `test-full` extra
for the one test that checks these names against the real classes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    A real `trulens.core.schema.record.Record` / `FeedbackResult` is a
    pydantic model (attribute access), but a caller may equally hand in a
    plain dict (e.g. loaded back from a database export), so every accessor
    here goes through this rather than assuming one shape -- the same
    approach adapters/autogen-openeval-adapter uses.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iso(ts: Any) -> str:
    """Coerce a TruLens `Record.ts` (a `datetime.datetime`, per real source)
    into the ISO-8601 string `ResultSet.started_at`/`completed_at` require.

    Falls back to `datetime.now(timezone.utc)` for a record with no
    timestamp at all, so a still-valid (if approximate) ResultSet can always
    be produced rather than failing the whole conversion over one field.
    """
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    if isinstance(ts, str) and ts:
        return ts
    return datetime.now(timezone.utc).isoformat()


def _stringify_output(value: Any) -> Optional[str]:
    """`Result.actual_output` is a plain string per the EvalPort schema, but
    `Record.main_output` is typed `Optional[JSON]` in TruLens -- a dict, a
    list, a number, or a string all show up in the wild depending on the
    wrapped app's return type. Non-string JSON is serialized with
    `json.dumps` (stable, round-trippable) rather than Python's `str()`
    (which produces single-quoted, non-JSON, non-round-trippable output for
    dicts/lists) so a consumer can `json.loads()` it back if the original
    shape matters.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _grader_result_from_feedback(feedback_result: Any, pass_threshold: float) -> Dict[str, Any]:
    """One TruLens `FeedbackResult` -> one EvalPort `GraderResult`.

    `FeedbackResult.result` (real field name, verified against
    `trulens.core.schema.feedback.FeedbackResult`) is `Optional[float]`,
    typically but not contractually 0.0-1.0 -- TruLens feedback functions are
    free to return outside that range (e.g. an unnormalized similarity
    metric), while EvalPort's `GraderResult.score` schema enforces
    `minimum: 0, maximum: 1`. Rather than silently clip or reject an
    out-of-range score, it is preserved in `metadata.raw_result` and `score`
    is left `null` for that one grader result, with `reason` explaining why --
    the same "don't lose information, don't fabricate a score" stance
    `Result.error` takes for a genuinely failed row.
    """
    name = _get(feedback_result, "name") or _get(feedback_result, "feedback_definition_id") or "trulens_feedback"
    error = _get(feedback_result, "error")
    status = _get(feedback_result, "status")
    raw_result = _get(feedback_result, "result")

    metadata: Dict[str, Any] = {}
    status_value = getattr(status, "value", status)
    if status_value is not None:
        metadata["status"] = str(status_value)

    if error:
        return {
            "grader_id": str(name),
            "type": "trulens_feedback",
            "score": None,
            "passed": False,
            "reason": str(error),
            "metadata": metadata,
        }

    if raw_result is None:
        return {
            "grader_id": str(name),
            "type": "trulens_feedback",
            "score": None,
            "passed": False,
            "reason": "TruLens FeedbackResult.result was None (feedback did not complete).",
            "metadata": metadata,
        }

    score = float(raw_result)
    if not (0.0 <= score <= 1.0):
        metadata["raw_result"] = score
        return {
            "grader_id": str(name),
            "type": "trulens_feedback",
            "score": None,
            "passed": False,
            "reason": (
                f"TruLens feedback result {score!r} is outside EvalPort's "
                "GraderResult.score range [0, 1]; preserved in metadata.raw_result."
            ),
            "metadata": metadata,
        }

    return {
        "grader_id": str(name),
        "type": "trulens_feedback",
        "score": score,
        "passed": score >= pass_threshold,
        "metadata": metadata,
    }


def to_openeval(
    records: Iterable[Any],
    feedback_results: Iterable[Any] = (),
    *,
    suite_id: str = "trulens_app",
    run_id: Optional[str] = None,
    pass_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Export TruLens `Record`s (+ their `FeedbackResult`s) to an EvalPort ResultSet.

    `records`: an iterable of TruLens `Record`-like objects or dicts, each
    exposing `record_id`, `app_id`, `main_input`, `main_output`, `main_error`
    and `ts` -- the real fields on `trulens.core.schema.record.Record` (a
    `Record` produced by `with tru_app as recording: ...` /
    `recording.records`, or read back via
    `TruSession().get_records_and_feedback()` and reconstructed per-row).

    `feedback_results`: an iterable of TruLens `FeedbackResult`-like objects
    or dicts, each exposing `record_id`, `name`, `result`, `error` and
    `status` -- the real fields on
    `trulens.core.schema.feedback.FeedbackResult`. Matched to `records` by
    `record_id`; a record with no matching feedback results still produces a
    valid `Result` with an empty `grader_results` list.

    `pass_threshold`: TruLens feedback functions return a continuous score
    with no universal pass/fail cutoff of their own (unlike, say, AutoGen's
    exact-match grader) -- 0.5 is a reasonable, commonly-used default for
    "higher is better" feedback (groundedness, relevance, ...), but is
    exposed here rather than hardcoded since some feedback functions are
    calibrated differently.

    A record whose `main_error` is set is reported as `Result.error` (type
    `"runner_error"`, since a `main_error` means the wrapped *app* itself
    raised -- an execution/harness failure, not a partial quality signal --
    the same category `error` is documented for) with `passed: false` and an
    empty `grader_results`, mirroring how the schema already excludes errored
    rows from being scored at all.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass it
    to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    feedback_by_record: Dict[str, List[Any]] = {}
    for fr in feedback_results:
        rid = _get(fr, "record_id")
        feedback_by_record.setdefault(rid, []).append(fr)

    results: List[Dict[str, Any]] = []
    timestamps: List[str] = []
    app_id: Optional[str] = None

    for record in records:
        record_id = _get(record, "record_id")
        app_id = app_id or _get(record, "app_id")
        main_error = _get(record, "main_error")
        ts = _iso(_get(record, "ts"))
        timestamps.append(ts)

        result: Dict[str, Any] = {
            "test_case_id": str(record_id),
            "completed_at": ts,
        }

        actual_output = _stringify_output(_get(record, "main_output"))
        if actual_output is not None:
            result["actual_output"] = actual_output

        if main_error:
            result["error"] = {"type": "runner_error", "message": str(main_error)}
            result["grader_results"] = []
            result["passed"] = False
        else:
            grader_results = [
                _grader_result_from_feedback(fr, pass_threshold)
                for fr in feedback_by_record.get(record_id, [])
            ]
            result["grader_results"] = grader_results
            result["passed"] = all(g["passed"] for g in grader_results) if grader_results else True

        meta: Dict[str, Any] = {}
        if _get(record, "app_id"):
            meta["app_id"] = _get(record, "app_id")
        main_input = _get(record, "main_input")
        if main_input is not None:
            meta["main_input"] = _stringify_output(main_input)
        tags = _get(record, "tags")
        if tags:
            meta["tags"] = tags
        if meta:
            result["metadata"] = meta

        results.append(result)

    started_at = min(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id or f"trulens_{app_id or 'run'}",
        "started_at": started_at,
        "results": results,
        "metadata": {"openeval": {"source": "trulens"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, str]]:
    """Import an EvalPort EvalSuite's test cases into a TruLens ground-truth golden set.

    Returns a list of `{"query": ..., "expected_response": ...}` dicts --
    exactly the shape `trulens.feedback.GroundTruthAgreement` documents for
    its own `ground_truth` constructor argument, verified against that
    class's real `agreement_measure()` docstring example
    (`golden_set = [{"query": "...", "expected_response": "..."}]`), not
    guessed. Pass the result straight in:

        from trulens.feedback import GroundTruthAgreement
        golden_set = from_openeval(my_suite)
        ground_truth = GroundTruthAgreement(golden_set, provider=...)

    Test cases with no `expected_output` are skipped (an empty
    `expected_response` would silently make every real response "disagree"),
    consistent with how `GroundTruthAgreement` has nothing meaningful to
    compare against for those.
    """
    golden_set: List[Dict[str, str]] = []
    for tc in suite.get("test_cases", []):
        expected = tc.get("expected_output")
        if expected is None:
            continue
        golden_set.append({"query": tc.get("input", ""), "expected_response": expected})
    return golden_set
