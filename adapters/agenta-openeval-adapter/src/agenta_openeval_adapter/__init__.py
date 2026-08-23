"""Agenta <-> EvalPort adapter.

Standalone converter between Agenta (https://github.com/Agenta-AI/agenta)
testsets/evaluator results and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Built at the explicit request of the Agenta maintainer @mmabrouk on
https://github.com/Agenta-AI/agenta/issues/6222 ("we'd prefer the first
option"): a standalone `agenta-openeval-adapter` package living in the
EvalPort repo, depending on the `agenta` PyPI package as a normal
dependency, rather than EvalPort support living inside Agenta's own core.

Two distinct Agenta object families are bridged here, matching Agenta's own
split between *data* and *evaluation*:

* ``agenta.testsets`` -- a testset is a ``TestsetRevision`` wrapping a
  ``TestsetRevisionData`` of ``Testcase`` rows (each row's actual content is
  a free-form ``data`` dict -- Agenta testsets have no fixed column schema).
  ``agenta_testset_to_suite()`` converts one of these into an EvalPort ``EvalSuite``.

* ``agenta.evaluator`` -- an Agenta *evaluator* is invoked once per test
  case: it is handed a ``WorkflowInvokeRequest`` (whose
  ``data.testcase``/``data.outputs`` carry the test case and the actual
  output being graded) and returns a ``WorkflowBatchResponse`` (or, for a
  streamed call, a ``WorkflowStreamingResponse``) carrying the grading
  verdict in ``data.outputs``. There is no single Agenta object that holds
  "a whole run's results" -- a runner collects one such
  (request, response) pair per (test case, grader) combination.
  ``invocations_to_resultset()`` takes that collected list and produces an
  EvalPort ``ResultSet``, grouping multiple graders per test case into one
  ``Result`` with several ``GraderResult`` entries, matching
  ``ResultSet.results[].grader_results[]`` in the spec.

``to_openeval()`` is a convenience dispatcher over both directions; the two
underlying functions (``agenta_testset_to_suite`` / ``invocations_to_resultset``)
are also exported directly since they take different keyword arguments and
most callers will know up front which one they mean. ``from_openeval()``
reverses ``agenta_testset_to_suite`` -- there is no meaningful reverse of
"evaluator results" (EvalPort results describe what already happened; they
are not something you replay back into Agenta's evaluator-invocation API).

NOTE on the shapes actually used here (verified against the real, installed
``agenta`` package -- see the adapter's README for details on where this
confirms vs. corrects the issue's own description): ``agenta.evaluator`` at
the top level is a *decorator* used to define your own evaluator function,
not an object with a callable ``.invoke()``. The request/response types
(``WorkflowInvokeRequest``, ``WorkflowBatchResponse``,
``WorkflowStreamingResponse``) do exist exactly as the issue described, but
live under ``agenta.sdk.models.workflows`` rather than being re-exported at
the top of the ``agenta`` package.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "agenta_testset_to_suite",
    "invocations_to_resultset",
    "__version__",
]
__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    Agenta's models (``TestsetRevision``, ``Testcase``, ``WorkflowInvokeRequest``,
    ``WorkflowBatchResponse``, ...) are pydantic ``BaseModel`` instances, but a
    caller may equally well hand this a plain dict pulled from a JSON API
    response (e.g. ``response.json()`` from Agenta's REST API rather than the
    SDK's own typed client), so every accessor in this module goes through
    here rather than assuming one shape.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    return value


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _coerce_uuid(raw_id: Any) -> Optional[str]:
    """Turn an EvalPort test_case id into a UUID string suitable for Agenta's
    ``Testcase.id`` (``Optional[UUID]``).

    If `raw_id` already looks like a UUID (as it will for any test case that
    started life as an Agenta testcase and round-tripped through
    `agenta_testset_to_suite`), it's returned unchanged. Otherwise a UUID is minted
    deterministically via uuid5 so that converting the same suite twice
    produces the same id (rather than a fresh random one each call, which
    would make the import non-reproducible and duplicate testcases on
    re-import).
    """
    if not raw_id:
        return None
    try:
        return str(uuid.UUID(str(raw_id)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://evalport.org/test-case/{raw_id}"))


# ---------------------------------------------------------------------------
# agenta.testsets.TestsetRevision  ->  EvalPort EvalSuite
# ---------------------------------------------------------------------------

# Candidate column names checked (in order) against each testcase's free-form
# `data` dict. Agenta testsets have no fixed schema -- these are simply the
# conventional column names seen in Agenta's own docs/templates and in the
# wild; anything not matched is preserved verbatim in `metadata.agenta_testcase`
# (and restored by `from_openeval`) rather than silently dropped.
_INPUT_KEYS = ("input", "query", "question", "prompt", "message")
_EXPECTED_KEYS = ("expected_output", "expected", "ground_truth", "correct_answer", "reference", "answer")
_CONTEXT_KEYS = ("context", "contexts", "retrieval_context")


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] is not None:
            return k
    return None


def _testcase_row(tc: Any) -> Dict[str, Any]:
    """Extract the free-form `data` dict off one Agenta `Testcase`."""
    row = _get(tc, "data", None)
    if row is None:
        row = tc if isinstance(tc, dict) else {}
    if not isinstance(row, dict):
        row = {"input": row}
    return row


def agenta_testset_to_suite(
    testset: Any,
    *,
    suite_id: Optional[str] = None,
    name: Optional[str] = None,
    grader_type: str = "llm_judge",
    input_key: Optional[str] = None,
    expected_output_key: Optional[str] = None,
    context_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Export an Agenta testset to an EvalPort-shaped suite (dict).

    `testset` may be:

    * an `agenta.testsets.TestsetRevision` (or dict of the same shape) --
      its `.data` (a `TestsetRevisionData`) is unwrapped automatically; or
    * a bare `TestsetRevisionData` (or dict) exposing `testcases` directly.

    Each Agenta `Testcase.data` is a free-form dict (Agenta testsets have no
    fixed column schema). `input_key` / `expected_output_key` / `context_key`
    let you name the columns explicitly; when omitted, common conventional
    names are auto-detected per-row (see `_INPUT_KEYS` / `_EXPECTED_KEYS` /
    `_CONTEXT_KEYS`). Any column not claimed by one of those three roles is
    preserved under `test_case.metadata.agenta_testcase` so `from_openeval`
    can restore it losslessly. If no input-like column is found at all, the
    whole (unclaimed) row is JSON-encoded into `input` so nothing is dropped
    silently.

    `grader_type` selects the output-quality grader attached to every test
    case: `"llm_judge"` (default) or `"exact_match"`.

    Returns a plain dict conforming to the EvalPort `EvalSuite` schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    revision_data = _get(testset, "data", None)
    if revision_data is None and (
        _get(testset, "testcases", None) is not None or _get(testset, "testcase_ids", None) is not None
    ):
        # `testset` is already a bare TestsetRevisionData (or equivalent dict).
        revision_data = testset
    raw_testcases = _get(revision_data, "testcases", None) or []

    tsid = _get(testset, "testset_id", None) or _get(testset, "id", None)
    slug = _get(testset, "testset_slug", None) or _get(testset, "slug", None)
    default_suite_id = f"agenta_testset_{tsid or slug or 'suite'}"
    suite_id = suite_id or default_suite_id
    name = name or _get(testset, "name", None) or (f"Agenta testset {slug}" if slug else "Agenta testset")

    test_cases: List[Dict[str, Any]] = []
    for i, tc in enumerate(raw_testcases):
        tc_id = _get(tc, "id", None)
        tc_id = str(tc_id) if tc_id else f"tc_{i}"
        row = _testcase_row(tc)

        ik = input_key or _first_present(row, _INPUT_KEYS)
        ek = expected_output_key or _first_present(row, _EXPECTED_KEYS)
        ck = context_key or _first_present(row, _CONTEXT_KEYS)
        claimed = {k for k in (ik, ek, ck) if k}

        input_val = row.get(ik) if ik else None
        if input_val is None:
            remainder = {k: v for k, v in row.items() if k not in claimed}
            input_val = json.dumps(remainder, sort_keys=True, default=str) if remainder else ""
        if isinstance(input_val, list):
            input_val = [str(x) for x in input_val]
        elif not isinstance(input_val, str):
            input_val = str(input_val)

        entry: Dict[str, Any] = {"id": tc_id, "input": input_val, "graders": ["gr_output_match"]}

        expected_val = row.get(ek) if ek else None
        if expected_val is not None:
            entry["expected_output"] = expected_val if isinstance(expected_val, str) else str(expected_val)

        context_val = row.get(ck) if ck else None
        if context_val is not None:
            entry["context"] = [str(x) for x in context_val] if isinstance(context_val, list) else [str(context_val)]

        extra = {k: v for k, v in row.items() if k not in claimed}
        if extra:
            entry["metadata"] = {"agenta_testcase": extra}

        test_cases.append(entry)

    graders: List[Dict[str, Any]] = []
    if test_cases:
        if grader_type == "exact_match":
            graders.append({"id": "gr_output_match", "type": "exact_match", "params": {"ignore_case": True}})
        else:
            graders.append(
                {
                    "id": "gr_output_match",
                    "type": "llm_judge",
                    "params": {
                        "model": "gpt-4o",
                        "prompt": (
                            "Expected outcome: {expected}\nActual output: {output}\n"
                            "Does the output satisfy the expected outcome? "
                            'Return JSON: {"score": 0.0-1.0, "reason": "..."}'
                        ),
                    },
                }
            )

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name,
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "agenta"}},
    }


# ---------------------------------------------------------------------------
# agenta evaluator invocations (WorkflowInvokeRequest/WorkflowBatchResponse)
#   ->  EvalPort ResultSet
# ---------------------------------------------------------------------------


def _parse_evaluator_outputs(outputs: Any) -> Tuple[Optional[float], bool, Optional[str]]:
    """Best-effort parse of an evaluator's `data.outputs` payload into
    `(score, passed, reason)`. Agenta evaluators are user-defined code, so
    `outputs` can legitimately be almost anything; this covers the shapes
    Agenta's own built-in evaluator templates produce (a dict with
    `score`/`passed`/`success`/`reason`, or a bare bool/number verdict).
    """
    if outputs is None:
        return None, False, None
    if isinstance(outputs, bool):
        return (1.0 if outputs else 0.0), outputs, None
    if isinstance(outputs, (int, float)):
        score = _clamp01(float(outputs))
        return score, score >= 0.5, None
    if isinstance(outputs, dict):
        reason = outputs.get("reason") or outputs.get("message")
        score = outputs.get("score")
        if score is not None:
            try:
                score = _clamp01(float(score))
            except (TypeError, ValueError):
                score = None
        passed = outputs.get("passed")
        if passed is None:
            passed = outputs.get("success")
        if passed is None:
            passed = bool(score is not None and score >= 0.5)
        return score, bool(passed), reason
    # Any other JSON-ish value (str, list, ...): can't infer pass/fail from it
    # alone, so surface it as the failure reason rather than guessing.
    return None, False, str(outputs)


def _invocation_to_grader_result(
    response: Any, *, grader_id: str
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Convert one evaluator `WorkflowBatchResponse` into an EvalPort
    `GraderResult` dict, plus an optional `error` dict when the invocation
    itself failed (HTTP-style `status.code >= 400`) as distinct from the
    evaluator running fine but grading the test case as failing.
    """
    if _get(response, "data", "__missing__") == "__missing__" and _get(response, "generator", None) is not None:
        raise TypeError(
            "invocations_to_resultset() needs a WorkflowBatchResponse (or dict of the same "
            "shape) per invocation, not a WorkflowStreamingResponse -- collect the stream into "
            "its final `outputs` first (e.g. via `await response.iterator()`)."
        )

    status = _get(response, "status", None) or {}
    code = _get(status, "code", 200)
    message = _get(status, "message", None)
    data = _get(response, "data", None) or {}
    outputs = _get(data, "outputs", None)

    error = None
    if isinstance(code, int) and code >= 400:
        error = {"message": message or f"evaluator invocation failed (status {code})"}

    score, passed, reason = _parse_evaluator_outputs(outputs)
    if error is not None:
        passed = False
        reason = reason or message

    grader_result: Dict[str, Any] = {"grader_id": grader_id, "type": "custom", "score": score, "passed": passed}
    if reason:
        grader_result["reason"] = str(reason)
    return grader_result, error


def invocations_to_resultset(
    invocations: Sequence[Any],
    *,
    suite_id: str,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    runner_name: str = "agenta-evaluator",
) -> Dict[str, Any]:
    """Export a collection of Agenta evaluator invocations to an
    EvalPort-shaped result set (dict).

    Agenta invokes an evaluator once per (test case, grader) pair -- there is
    no single Agenta object holding "a whole run's results" the way
    EvalPort's `ResultSet` does, so this takes the list your own runner/CI
    script assembled while looping over test cases and graders.

    Each item of `invocations` is a dict (or attribute-bearing object) with:

    * `response` (required) -- the evaluator's `WorkflowBatchResponse` (or
      dict of the same shape: `{"status": {...}, "data": {"outputs": ...}}`).
      `data.outputs` carries the grading verdict -- a bool, a number, or a
      dict such as `{"score": 0.8, "passed": true, "reason": "..."}`
      (Agenta evaluators are user-defined code, so the exact shape of
      `outputs` is whatever that evaluator returns).
    * `request` (optional) -- the `WorkflowInvokeRequest` that produced it.
      When given, `data.outputs` on the *request* is read as the actual
      output being graded (this is exactly what Agenta hands an evaluator to
      score) and used as the EvalPort `Result.actual_output`, and
      `data.testcase.id`/`data.testcase["id"]` is used as a fallback
      `test_case_id` if not given explicitly.
    * `test_case_id` (optional if derivable from `request`) -- the EvalPort
      test case this invocation scored.
    * `grader_id` (optional, default `"gr_agenta_eval"`) -- which grader this
      invocation represents; multiple invocations sharing a `test_case_id`
      are grouped into one `Result` with one `GraderResult` per grader.

    A `status.code >= 400` on the response (the evaluator invocation itself
    erroring out, as opposed to running fine and grading the test case as
    failing) is surfaced as `Result.error` and forces that result's `passed`
    to `False`.

    Returns a plain dict conforming to the EvalPort `ResultSet` schema. Pass
    it to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for i, inv in enumerate(invocations):
        response = _get(inv, "response", None)
        request = _get(inv, "request", None)
        grader_id = _get(inv, "grader_id", None) or "gr_agenta_eval"

        tcid = _get(inv, "test_case_id", None)
        actual_output = None
        if request is not None:
            req_data = _get(request, "data", None) or {}
            if not tcid:
                testcase = _get(req_data, "testcase", None) or {}
                tcid = _get(testcase, "id", None)
            actual_output = _get(req_data, "outputs", None)
        tcid = str(tcid) if tcid else f"tc_{i}"

        grader_result, error = _invocation_to_grader_result(response, grader_id=grader_id)

        if tcid not in grouped:
            grouped[tcid] = {"grader_results": [], "actual_output": None, "error": None}
            order.append(tcid)
        bucket = grouped[tcid]
        bucket["grader_results"].append(grader_result)
        if actual_output is not None:
            bucket["actual_output"] = actual_output
        if error is not None:
            bucket["error"] = error

    results: List[Dict[str, Any]] = []
    for tcid in order:
        bucket = grouped[tcid]
        passed = bucket["error"] is None and all(gr["passed"] for gr in bucket["grader_results"])
        result: Dict[str, Any] = {
            "test_case_id": tcid,
            "grader_results": bucket["grader_results"],
            "passed": passed,
        }
        if bucket["actual_output"] is not None:
            ao = bucket["actual_output"]
            result["actual_output"] = ao if isinstance(ao, str) else json.dumps(ao, sort_keys=True, default=str)
        if bucket["error"] is not None:
            result["error"] = bucket["error"]
        results.append(result)

    now = _now_iso()
    resultset: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id or f"agenta_run_{uuid.uuid4().hex[:8]}",
        "started_at": started_at or now,
        "completed_at": completed_at or now,
        "runner": {"name": runner_name},
        "results": results,
    }

    if results:
        total = len(results)
        passed_n = sum(1 for r in results if r["passed"])
        scores = [gr["score"] for r in results for gr in r["grader_results"] if gr.get("score") is not None]
        resultset["summary"] = {
            "total": total,
            "passed": passed_n,
            "failed": total - passed_n,
            "pass_rate": passed_n / total,
            "avg_score": (sum(scores) / len(scores)) if scores else None,
        }

    return resultset


# ---------------------------------------------------------------------------
# dispatcher + reverse direction
# ---------------------------------------------------------------------------


def _looks_like_testset(obj: Any) -> bool:
    data = _get(obj, "data", None)
    if _get(data, "testcases", None) is not None or _get(data, "testcase_ids", None) is not None:
        return True
    if _get(obj, "testcases", None) is not None or _get(obj, "testcase_ids", None) is not None:
        return True
    return False


def to_openeval(obj: Any, **kwargs: Any) -> Dict[str, Any]:
    """Dispatching convenience wrapper: exports either Agenta object family
    to its EvalPort counterpart.

    * A `TestsetRevision` / `TestsetRevisionData` (or dict of either shape)
      is routed to `agenta_testset_to_suite(obj, **kwargs)`.
    * A list/tuple of evaluator invocations (see `invocations_to_resultset`
      for the expected item shape) is routed to
      `invocations_to_resultset(obj, **kwargs)` -- this branch requires the
      `suite_id` keyword, same as calling `invocations_to_resultset` directly.

    Prefer calling `agenta_testset_to_suite` / `invocations_to_resultset` directly
    when you already know which direction you mean -- they take different
    keyword arguments, so this dispatcher exists mainly for parity with
    other EvalPort adapters' single-entry-point `to_openeval()`.
    """
    if isinstance(obj, (list, tuple)):
        return invocations_to_resultset(obj, **kwargs)
    if _looks_like_testset(obj):
        return agenta_testset_to_suite(obj, **kwargs)
    raise TypeError(
        "to_openeval() could not tell whether this is an Agenta testset (TestsetRevision / "
        "TestsetRevisionData) or a list of evaluator invocations -- call agenta_testset_to_suite() or "
        "invocations_to_resultset() directly."
    )


def from_openeval(suite: Dict[str, Any]) -> Dict[str, Any]:
    """Import an EvalPort suite into an Agenta-shaped testset revision data dict.

    Returns a plain dict shaped like `agenta.testsets.TestsetRevisionData`
    (`{"testcase_ids": None, "testcases": [...]}`), where each testcase is a
    plain dict shaped like `agenta.testsets.Testcase`
    (`{"id": <uuid str or None>, "data": {...}}`) -- feed these straight into
    `Testcase(**tc)` / `TestsetRevisionData(**data)` for real Agenta model
    instances, or into `agenta.testsets.acreate()` to commit a new testset.
    Plain dicts are returned rather than committed Agenta objects since doing
    that also requires calling the Agenta SDK's manager functions against a
    live project, which this offline converter has no way to do.

    `test_case.input` / `expected_output` / `context` map back onto `data`
    under those same column names; anything EvalPort-suite-only, non-Agenta
    columns previously round-tripped through
    `test_case.metadata.agenta_testcase` (see `agenta_testset_to_suite`) are merged
    back in, so `agenta -> openeval -> agenta` is lossless for the fields
    this adapter understands. Each Agenta testcase id is reused as-is when it
    is already a UUID string; otherwise one is minted deterministically (see
    `_coerce_uuid`) so repeated imports of the same suite are idempotent.
    """
    testcases: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        data: Dict[str, Any] = {}
        if tc.get("input") is not None:
            data["input"] = tc["input"]
        if tc.get("expected_output") is not None:
            data["expected_output"] = tc["expected_output"]
        if tc.get("context") is not None:
            data["context"] = tc["context"]

        extra = (tc.get("metadata") or {}).get("agenta_testcase")
        if isinstance(extra, dict):
            data.update(extra)

        testcases.append({"id": _coerce_uuid(tc.get("id")), "data": data})

    return {"testcase_ids": None, "testcases": testcases}
