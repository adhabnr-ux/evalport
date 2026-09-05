"""Convert pytector's ``GuardDecision`` into EvalPort ``ResultSet`` JSON.

`pytector <https://github.com/MaxMLang/pytector>`_ is a prompt-injection /
PII / toxicity screening toolkit. Its ``ToolOutputGuard.scan_text()`` /
``scan_tool_result()`` return a ``GuardDecision`` (defined in
``src/pytector/guard.py``, read in full while building this package) that
records whether a piece of untrusted text was allowed, redacted, or blocked,
plus the detector's confidence and reasons.

Built in direct response to `MaxMLang/pytector#1
<https://github.com/MaxMLang/pytector/issues/1>`_ ("Proposal: an
EvalPort/OpenEval adapter for GuardDecision"), where maintainer @MaxMLang
gave an explicit go-ahead: "feel free to go ahead ... Feel free to add a
link here if you got something ready".

What this package does, precisely
----------------------------------
A ``GuardDecision`` is the outcome of screening ONE piece of text. To turn
that into an EvalPort ``Result`` you need a second piece of information
``GuardDecision`` itself cannot supply: whether that text was *actually* an
injection attempt (i.e. the label a labeled eval dataset would carry). This
package never guesses that label -- every entry point requires the caller
to pass ``expected_injection`` explicitly, alongside the real
``GuardDecision`` pytector produced. The exported ``Result.passed`` is then
the real, observable fact "did pytector's classification match the
caller-supplied expected label", never a fabricated grade.

Two entry points:

* :func:`guard_decision_to_result` -- converts one already-computed
  ``GuardDecision`` (from any pytector version/backend: local HF model,
  GGUF, or Groq) plus its expected label into one EvalPort ``Result`` dict.
* :func:`to_openeval` -- batches many such (test_case_id, expected_injection,
  decision) triples into one EvalPort ``ResultSet`` document, ready for
  ``openeval.validate.validate_result_set``.
* :func:`run_and_convert` -- convenience wrapper that actually calls a real,
  caller-supplied ``pytector.ToolOutputGuard`` (or any duck-typed object
  exposing ``scan_text(text, tool_name=...)``) for each case and then does
  the same conversion -- the true end-to-end path from raw text to an
  EvalPort ``ResultSet``.

No hard dependency on pytector
-------------------------------
pytector's core `PromptInjectionDetector` unconditionally loads either a
Hugging Face model (``torch`` + ``transformers``), a GGUF model
(``llama-cpp-python``), or a Groq client (``groq``) -- there is no
"lightweight" construction path (verified by reading
``src/pytector/detector.py`` in full: ``__init__`` always calls
``_load_hf_model``, ``_load_gguf_model``, or sets up ``groq_client``, with
no branch that skips this). Requiring those heavy, largely GPU/ML-oriented
dependencies just to convert an already-computed ``GuardDecision`` into JSON
would be exactly backwards, so this package only depends on
``evalport-sdk``. Every function here works with anything shaped like the
real ``GuardDecision`` dataclass (attribute *or* dict access, via the same
duck-typing helper this repo's other adapters use) -- including the real
class itself, if pytector happens to be installed.

A privacy note
---------------
``GuardDecision.original_content`` / ``.content`` can contain the actual
scanned text -- which, for a prompt-injection guard, may itself be a
malicious payload, a leaked secret, or otherwise sensitive. Neither field is
copied into the emitted ``Result`` by default. Pass ``include_text=True`` to
opt in explicitly when you want it (e.g. for local debugging of a private
eval run you control end to end).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

__version__ = "0.1.0"

# Real action values from pytector's `src/pytector/guard.py`
# (ACTION_ALLOW / ACTION_REDACT / ACTION_BLOCK module constants).
ACTION_ALLOW = "allow"
ACTION_REDACT = "redact"
ACTION_BLOCK = "block"

DEFAULT_GRADER_ID = "pytector.guard_decision"
DEFAULT_GRADER_TYPE = "pytector_guard_decision"
DEFAULT_SUITE_ID = "pytector-guard-decisions"

# Tracks openeval.types.OPENEVAL_VERSION at the time this package was built.
# Only used as a last-resort fallback if the real installed evalport-sdk
# constant can't be imported (e.g. a stale/partial install) -- the real
# value is always preferred. See niceeval-openeval-exporter's identical
# pattern in this repo for the precedent.
OPENEVAL_VERSION_FALLBACK = "1.0.0-rc.5"

__all__ = [
    "ACTION_ALLOW",
    "ACTION_REDACT",
    "ACTION_BLOCK",
    "DEFAULT_GRADER_ID",
    "DEFAULT_GRADER_TYPE",
    "DEFAULT_SUITE_ID",
    "OPENEVAL_VERSION_FALLBACK",
    "guard_decision_to_result",
    "to_openeval",
    "run_and_convert",
]


def _openeval_version() -> str:
    try:
        from openeval.types import OPENEVAL_VERSION

        if isinstance(OPENEVAL_VERSION, str) and OPENEVAL_VERSION:
            return OPENEVAL_VERSION
    except Exception:
        pass
    return OPENEVAL_VERSION_FALLBACK


def _pytector_version() -> Optional[str]:
    """Best-effort, purely informational: the installed pytector's own
    ``__version__``, when pytector actually happens to be installed. Never
    required -- this package has no hard dependency on pytector."""
    try:
        import pytector  # type: ignore

        version = getattr(pytector, "__version__", None)
        return version if isinstance(version, str) else None
    except Exception:
        return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Duck-typed read: works for a dict, a dataclass instance (like the
    real ``GuardDecision``), or any other attribute-bearing object."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clamp01(value: Any) -> Optional[float]:
    """Clamp a detector confidence into [0, 1]; ``None`` for anything that
    isn't honestly a number (including ``None`` itself, which pytector's
    Groq and GGUF backends both return for ``score`` -- never invent a
    number those backends didn't produce)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _reason_text(reasons: Any) -> Optional[str]:
    if not reasons:
        return None
    try:
        parts = [str(r) for r in reasons if r]
    except TypeError:
        return None
    if not parts:
        return None
    return "; ".join(parts)


def _is_backend_error(metadata: Mapping[str, Any]) -> bool:
    """``ToolOutputGuard._run_detection`` sets ``metadata["api_error"] =
    True`` when the Groq backend call itself failed -- in that case
    ``is_injection`` reflects ``block_on_api_error`` (a policy default), not
    a real classification, so this must never be scored as a genuine
    pass/fail."""
    return bool(metadata.get("api_error"))


def guard_decision_to_result(
    decision: Any,
    *,
    expected_injection: bool,
    test_case_id: str,
    attempt: Optional[int] = None,
    duration_ms: Optional[int] = None,
    grader_id: str = DEFAULT_GRADER_ID,
    include_text: bool = False,
) -> Dict[str, Any]:
    """Build one EvalPort ``Result`` dict from a real pytector
    ``GuardDecision`` plus the expected (ground-truth) label.

    ``decision`` must be shaped like ``pytector.guard.GuardDecision``:
    real attributes (or dict keys) ``action``, ``is_injection``, ``score``,
    ``tool_name``, ``original_content``, ``content``, ``reasons``,
    ``metadata`` -- see ``src/pytector/guard.py`` in the pytector repo.
    """
    if not isinstance(test_case_id, str) or not test_case_id:
        raise ValueError("test_case_id must be a non-empty string.")

    is_injection = _get(decision, "is_injection")
    if not isinstance(is_injection, bool):
        raise TypeError(
            "decision does not look like a real pytector GuardDecision: "
            f"expected a boolean 'is_injection' attribute/key, got {is_injection!r}. "
            "Pass the GuardDecision returned by ToolOutputGuard.scan_text() / "
            "scan_tool_result(), not a raw score or string."
        )

    action = _get(decision, "action")
    metadata = dict(_get(decision, "metadata", {}) or {})
    reasons = list(_get(decision, "reasons", []) or [])
    score = _clamp01(_get(decision, "score"))
    tool_name = _get(decision, "tool_name")

    result: Dict[str, Any] = {"test_case_id": test_case_id}
    if attempt is not None:
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ValueError(f"attempt must be an integer >= 1, got {attempt!r}.")
        result["attempt"] = attempt
    if duration_ms is not None:
        result["duration_ms"] = duration_ms

    if _is_backend_error(metadata):
        # Never invent a pass/fail out of a backend failure: no
        # classification actually happened, so there is nothing to grade.
        message = _reason_text(reasons) or (
            "pytector could not determine a classification for this input "
            "due to a backend (Groq API) error."
        )
        result["passed"] = False
        result["grader_results"] = []
        result["error"] = {
            "type": "detector_error",
            "message": message,
            "backend": metadata.get("backend"),
        }
        result["metadata"] = {
            "pytector": {
                "action": action,
                "tool_name": tool_name,
                "backend": metadata.get("backend"),
                "api_error": True,
            }
        }
        return result

    expected_bool = bool(expected_injection)
    matched = is_injection == expected_bool

    reason = _reason_text(reasons) or (
        "pytector classified this input as {actual}; expected {expected}.".format(
            actual="an injection" if is_injection else "benign",
            expected="an injection" if expected_bool else "benign",
        )
    )

    grader_result: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": DEFAULT_GRADER_TYPE,
        "score": 1.0 if matched else 0.0,
        "passed": matched,
        "reason": reason,
        "metadata": {
            "pytector": {
                "action": action,
                "is_injection": is_injection,
                "expected_injection": expected_bool,
                "detector_score": score,
                "tool_name": tool_name,
                "backend": metadata.get("backend"),
                "threshold": metadata.get("threshold"),
                "sanitizer_modified": metadata.get("sanitizer_modified"),
                "sanitizer_changes": metadata.get("sanitizer_changes"),
                "reasons": reasons,
            }
        },
    }

    result["passed"] = matched
    result["grader_results"] = [grader_result]
    result["metadata"] = {
        "pytector": {
            "action": action,
            "was_allowed": action == ACTION_ALLOW,
            "was_redacted": action == ACTION_REDACT,
            "was_blocked": action == ACTION_BLOCK,
        }
    }

    if include_text:
        original_content = _get(decision, "original_content")
        content = _get(decision, "content")
        if isinstance(original_content, str):
            result["metadata"]["pytector"]["original_content"] = original_content
        if isinstance(content, str):
            result["metadata"]["pytector"]["content"] = content

    return result


CaseInput = Union[Mapping[str, Any], Sequence[Any]]


def _normalize_case(case: CaseInput) -> Dict[str, Any]:
    if isinstance(case, Mapping):
        if "test_case_id" not in case:
            raise ValueError("Each case dict must include 'test_case_id'.")
        if "expected_injection" not in case:
            raise ValueError(
                f"Case {case.get('test_case_id')!r} is missing 'expected_injection': "
                "this package never guesses the ground-truth label."
            )
        if "decision" not in case:
            raise ValueError(f"Case {case.get('test_case_id')!r} is missing 'decision'.")
        return dict(case)
    # 3-tuple/list convenience form: (test_case_id, expected_injection, decision)
    if isinstance(case, Sequence) and not isinstance(case, (str, bytes)):
        if len(case) != 3:
            raise ValueError(
                "A tuple/list case must have exactly 3 items: "
                "(test_case_id, expected_injection, decision)."
            )
        test_case_id, expected_injection, decision = case
        return {
            "test_case_id": test_case_id,
            "expected_injection": expected_injection,
            "decision": decision,
        }
    raise TypeError(
        f"Each case must be a dict or a 3-item tuple/list, got {type(case)!r}."
    )


def to_openeval(
    cases: Iterable[CaseInput],
    *,
    run_id: Optional[str] = None,
    suite_id: str = DEFAULT_SUITE_ID,
    suite_version: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    version: Optional[str] = None,
    grader_id: str = DEFAULT_GRADER_ID,
    include_text: bool = False,
) -> Dict[str, Any]:
    """Build one EvalPort ``ResultSet`` document from many pytector
    ``GuardDecision`` results.

    Each item in ``cases`` is either a dict with keys ``test_case_id``,
    ``expected_injection``, ``decision``, and optionally ``attempt`` /
    ``duration_ms`` / ``grader_id`` -- or a 3-tuple
    ``(test_case_id, expected_injection, decision)``.

    pytector has no concept of a "run": a ``GuardDecision`` is the outcome
    of one text screening call, not part of any run/session the library
    tracks. So, unlike the adapters in this repo that read a real run id out
    of the source system (e.g. NiceEval's ``RunDocument.runId``), ``run_id``
    here is generated (``uuid4``) when not supplied -- it is an opaque
    identifier for *this ResultSet document*, not a fabricated claim about
    pytector itself. Pass your own (e.g. your CI run id) when you have one.
    Likewise ``started_at`` defaults to the real current UTC time (this
    function's own invocation time), not anything read from pytector.
    """
    prepared = [_normalize_case(case) for case in cases]
    if not prepared:
        raise ValueError("cases must be non-empty.")

    counts: Dict[str, int] = {}
    for case in prepared:
        tcid = case["test_case_id"]
        counts[tcid] = counts.get(tcid, 0) + 1

    running: Dict[str, int] = {}
    results: List[Dict[str, Any]] = []
    error_count = 0
    blocked_count = 0
    redacted_count = 0
    allowed_count = 0

    for case in prepared:
        tcid = case["test_case_id"]
        attempt = case.get("attempt")
        if attempt is None and counts[tcid] > 1:
            running[tcid] = running.get(tcid, 0) + 1
            attempt = running[tcid]

        result = guard_decision_to_result(
            case["decision"],
            expected_injection=case["expected_injection"],
            test_case_id=tcid,
            attempt=attempt,
            duration_ms=case.get("duration_ms"),
            grader_id=case.get("grader_id", grader_id),
            include_text=include_text,
        )
        results.append(result)

        if result.get("error") is not None:
            error_count += 1
        pytector_meta = result.get("metadata", {}).get("pytector", {})
        if pytector_meta.get("was_blocked"):
            blocked_count += 1
        elif pytector_meta.get("was_redacted"):
            redacted_count += 1
        elif pytector_meta.get("was_allowed"):
            allowed_count += 1

    total = len(results)
    passed_count = sum(1 for r in results if r.get("passed"))

    resolved_started_at = started_at or datetime.now(timezone.utc).isoformat()
    resolved_run_id = run_id or str(uuid.uuid4())

    result_set: Dict[str, Any] = {
        "version": version or _openeval_version(),
        "suite_id": suite_id,
        "run_id": resolved_run_id,
        "started_at": resolved_started_at,
        "results": results,
        "runner": {"name": "pytector-openeval-adapter", "version": __version__},
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else None,
        },
        "metadata": {
            "pytector": {
                "detector_errors": error_count,
                "blocked": blocked_count,
                "redacted": redacted_count,
                "allowed": allowed_count,
            }
        },
    }
    if suite_version is not None:
        result_set["suite_version"] = suite_version
    if completed_at is not None:
        result_set["completed_at"] = completed_at

    installed_pytector_version = _pytector_version()
    if installed_pytector_version is not None:
        result_set["metadata"]["pytector"]["pytector_version"] = installed_pytector_version

    return result_set


def run_and_convert(
    guard: Any,
    cases: Iterable[Mapping[str, Any]],
    *,
    tool_name: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """The true end-to-end path: actually screen each case's text through a
    real (or duck-typed) ``pytector.ToolOutputGuard``-shaped ``guard``, then
    convert the resulting real ``GuardDecision`` objects into a ``ResultSet``.

    ``guard`` must expose ``scan_text(text, *, tool_name=None) ->
    GuardDecision`` -- exactly the real ``ToolOutputGuard.scan_text``
    signature. Passing the real ``pytector.ToolOutputGuard`` instance you
    already configured (with your chosen model/backend/threshold) works
    unmodified.

    Each item in ``cases`` is a mapping with ``test_case_id``, ``text``,
    ``expected_injection``, and optionally ``tool_name`` (overriding the
    call-level default) / ``attempt`` / ``duration_ms``.
    """
    prepared: List[Dict[str, Any]] = []
    for case in cases:
        if "text" not in case:
            raise ValueError(
                f"Case {case.get('test_case_id')!r} is missing 'text' to scan."
            )
        decision = guard.scan_text(
            case["text"], tool_name=case.get("tool_name", tool_name)
        )
        prepared_case: Dict[str, Any] = {
            "test_case_id": case["test_case_id"],
            "expected_injection": case["expected_injection"],
            "decision": decision,
        }
        if "attempt" in case:
            prepared_case["attempt"] = case["attempt"]
        if "duration_ms" in case:
            prepared_case["duration_ms"] = case["duration_ms"]
        if "grader_id" in case:
            prepared_case["grader_id"] = case["grader_id"]
        prepared.append(prepared_case)

    return to_openeval(prepared, **kwargs)
