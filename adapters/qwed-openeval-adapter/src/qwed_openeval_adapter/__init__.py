"""QWED Verification Context <-> EvalPort adapter.

Standalone converter between QWED's versioned **Verification Context v1.0**
document (https://github.com/QWED-AI/qwed-verification, `spec/v1.0/`,
`$id: https://qwed.dev/schemas/verification-context/v1.0`) and the EvalPort
interchange format (https://github.com/adhabnr-ux/evalport).

Built following the design agreed with the QWED maintainer in
https://github.com/QWED-AI/qwed-verification/issues/325 ("Design approved.
The two-grader model is correct: `qwed.verdict` and `qwed.admission` as
independent grader results, with `result.passed` computed from both rather
than from the verdict alone").

Dependency boundary
-------------------
This module depends only on the Verification Context contract: it reads the
plain JSON document shape (`spec_version`, `object`, `context`, `verdict`)
that QWED's supported surfaces emit -- `QWEDClient.create_verification_
context_from_diagnostic()` (HTTP `POST /verification-context/from-diagnostic`),
the `qwed context from-diagnostic` CLI, and `VerificationContextDocument
.to_dict()` from `qwed_sdk`. It never imports `qwed_new.*`. `qwed` itself is
optional: when the `qwed_sdk` package is importable, documents are also
checked with the SDK's own exported `qwed_sdk.validate_document()` (JSON
Schema plus the `proof_ref` commitment); without it, only the structural
invariants below are checked, and the output records which check ran under
`metadata.qwed.vc_validation`.

Mapping (one Verification Context document -> one EvalPort `Result`)
--------------------------------------------------------------------

Two independent `GraderResult`s, never merged:

| grader_id | type | QWED field | score | passed |
|---|---|---|---|---|
| `qwed.verdict` | `qwed_verification_context` | `verdict` = `VERIFIED` | `1.0` | `true` |
| `qwed.verdict` | `qwed_verification_context` | `verdict` = `UNVERIFIABLE` / `BLOCKED` | `null` | `false` |
| `qwed.admission` | `qwed_admission_decision` | `context.decision.admission` = `ADMIT` | `1.0` | `true` |
| `qwed.admission` | `qwed_admission_decision` | `context.decision.admission` = `DENY` | `0.0` | `false` |

`GraderResult.score` is a number in [0, 1] or `null` (EvalPort Validation
Rule 5). QWED's verdict and admission strings are never placed in `score`;
they are carried verbatim in `GraderResult.metadata.qwed.verdict` /
`.admission` and in `Result.metadata.qwed`.

* Verdict grader `passed` means: the claim was definitively established by
  QWED (`VERIFIED`). `UNVERIFIABLE` and `BLOCKED` are fail-closed
  "not proven" outcomes, so they get `score: null` -- EvalPort Validation
  Rule 6: "`score: null` means 'not verified'", and consumers "MUST NOT
  treat a `null`-score result as equivalent to a scored failure". Rule 6
  also requires `passed: false` alongside `score: null`. The Verification
  Context v1.0 `Verdict` enum has no "refuted" value, so this grader never
  emits `0.0`.
* Admission grader `passed` means: the result is admissible (`ADMIT`).
  `DENY` is a definitive decision (including QWED's fail-closed DENY for
  `UNVERIFIABLE`/`BLOCKED`), so it is a scored `0.0`.
* `Result.passed` = `verdict == VERIFIED and admission == ADMIT`. This equals
  EvalPort's default `all` aggregation over the non-null-scored graders, so
  `VERIFIED + DENY` (ADR-003's proven-unsafe case) is `passed: false` while
  its verdict grader stays `passed: true`.

The complete Verification Context document is embedded verbatim (deep copy)
under the verdict grader's `metadata.qwed.verification_context`, so
`proof_ref` stays resolvable against its bound payload. `from_openeval()`
returns those documents unchanged.
"""
from __future__ import annotations

import copy
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "result_to_openeval",
    "suite_to_openeval",
    "from_openeval",
    "result_from_openeval",
    "check_document",
    "VERDICT_GRADER_ID",
    "VERDICT_GRADER_TYPE",
    "ADMISSION_GRADER_ID",
    "ADMISSION_GRADER_TYPE",
    "VC_SPEC_VERSION",
    "__version__",
]
__version__ = "0.1.0"

#: Verification Context spec version this adapter reads (`spec_version` const
#: in QWED's `spec/v1.0/schemas/verification-context.schema.json`).
VC_SPEC_VERSION = "1.0"

VERDICT_GRADER_ID = "qwed.verdict"
VERDICT_GRADER_TYPE = "qwed_verification_context"
ADMISSION_GRADER_ID = "qwed.admission"
ADMISSION_GRADER_TYPE = "qwed_admission_decision"

# Inlined from the Verification Context v1.0 schema ($defs.Verdict,
# $defs.Admission, $defs.ProofRef) so this module needs no QWED import.
_VERDICTS = ("VERIFIED", "UNVERIFIABLE", "BLOCKED")
_ADMISSIONS = ("ADMIT", "DENY")
_PROOF_REF_RE = re.compile(r"^sha256:[a-f0-9]{64}$")

_VERDICT_MEANING = {
    "VERIFIED": "the claim was definitively established by QWED",
    "UNVERIFIABLE": "not proven (fail-closed); not a scored failure",
    "BLOCKED": "verification could not be attempted or completed (fail-closed); not a scored failure",
}
_ADMISSION_MEANING = {
    "ADMIT": "the result is admissible",
    "DENY": "the result is not admissible",
}


# ---------------------------------------------------------------------------
# Verification Context input handling
# ---------------------------------------------------------------------------


def _as_document(obj: Any) -> Dict[str, Any]:
    """Return a deep-copied plain-dict Verification Context document.

    Accepts a mapping (the JSON returned by the HTTP route / CLI) or any
    object with a ``to_dict()`` returning one (``qwed_sdk
    .VerificationContextDocument``). Anything else -- notably the legacy
    ``qwed_sdk.VerificationResult`` returned by the ``verify_*`` convenience
    methods, which carries no verdict/admission/proof_ref -- is rejected.
    """
    if isinstance(obj, Mapping):
        return copy.deepcopy(dict(obj))
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return copy.deepcopy(dict(data))
    raise TypeError(
        f"expected a QWED Verification Context v1.0 document (a mapping, or an "
        f"object with to_dict() such as qwed_sdk.VerificationContextDocument), "
        f"got {type(obj).__name__}. Legacy verify_* results "
        f"(qwed_sdk.VerificationResult) do not carry the verdict/admission/"
        f"proof_ref contract; convert a DiagnosticResult with "
        f"QWEDClient.create_verification_context_from_diagnostic() first."
    )


def check_document(document: Mapping[str, Any]) -> None:
    """Check the Verification Context v1.0 invariants this mapping relies on.

    Raises ``ValueError`` on: wrong ``spec_version``; missing or empty
    ``object.formal_statement``; unknown ``verdict`` or ``admission``;
    ``VERIFIED`` without a ``sha256:<64-hex>`` ``proof_ref``;
    ``UNVERIFIABLE``/``BLOCKED`` with a non-null ``proof_ref`` or with
    ``ADMIT``. These are the schema's own invariants (spec §4, §5). This does
    NOT re-derive the ``proof_ref`` commitment (RFC 8785 canonical encoding);
    ``qwed_sdk.validate_document()`` does that and is used by
    ``to_openeval()`` when ``qwed_sdk`` is importable.
    """
    if not isinstance(document, Mapping):
        raise ValueError("Verification Context document must be a JSON object")
    if document.get("spec_version") != VC_SPEC_VERSION:
        raise ValueError(
            f"spec_version must be {VC_SPEC_VERSION!r}, got {document.get('spec_version')!r}"
        )
    obj = document.get("object")
    if not isinstance(obj, Mapping):
        raise ValueError("object is required")
    statement = obj.get("formal_statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("object.formal_statement must be a non-empty string")
    context = document.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("context is required")
    for layer in ("interpretation", "proof", "evidence", "decision"):
        if not isinstance(context.get(layer), Mapping):
            raise ValueError(f"context.{layer} is required")
    proof = context["proof"]
    for key in ("verifier", "verifier_version"):
        if not isinstance(proof.get(key), str) or not proof[key].strip():
            raise ValueError(f"context.proof.{key} must be a non-empty string")
    verdict = document.get("verdict")
    if verdict not in _VERDICTS:
        raise ValueError(f"verdict must be one of {_VERDICTS}, got {verdict!r}")
    admission = context["decision"].get("admission")
    if admission not in _ADMISSIONS:
        raise ValueError(
            f"context.decision.admission must be one of {_ADMISSIONS}, got {admission!r}"
        )
    evidence = context["evidence"]
    if "evidence" not in evidence or not isinstance(evidence["evidence"], Mapping):
        raise ValueError("context.evidence.evidence must be an object")
    proof_ref = evidence.get("proof_ref")
    if verdict == "VERIFIED":
        if not isinstance(proof_ref, str) or not _PROOF_REF_RE.match(proof_ref):
            raise ValueError("VERIFIED requires context.evidence.proof_ref = sha256:<64-hex>")
    else:
        if "proof_ref" not in evidence or proof_ref is not None:
            raise ValueError(f"{verdict} requires context.evidence.proof_ref to be null")
        if admission != "DENY":
            raise ValueError(f"{verdict} requires admission DENY (fail-closed)")


def _qwed_validate():
    """Return ``qwed_sdk.validate_document`` or ``None`` if qwed isn't installed.

    Imported lazily from the public ``qwed_sdk`` package (its re-export),
    never from ``qwed_new``.
    """
    try:
        from qwed_sdk import validate_document  # type: ignore
    except ImportError:
        return None
    return validate_document


def _validated(document: Dict[str, Any], validate: Optional[bool]) -> str:
    """Run the checks; return the ``vc_validation`` label recorded in output."""
    check_document(document)
    if validate is False:
        return "structural"
    validator = _qwed_validate()
    if validator is None:
        if validate:
            raise ImportError(
                "validate=True needs the `qwed` package (qwed_sdk.validate_document); "
                "install qwed-openeval-adapter[qwed]"
            )
        return "structural"
    try:
        validator(document)
    except ValueError as exc:  # VerificationContextValidationError subclasses ValueError
        raise ValueError(f"Verification Context rejected by qwed_sdk.validate_document: {exc}") from exc
    return "qwed_sdk.validate_document"


# ---------------------------------------------------------------------------
# QWED -> EvalPort
# ---------------------------------------------------------------------------


def _reason(verdict: str, admission: str, document: Mapping[str, Any], which: str) -> str:
    proof = document["context"]["proof"]
    engine = f"{proof['verifier']} {proof['verifier_version']}"
    if which == "verdict":
        text = f"QWED verdict {verdict} ({engine}): {_VERDICT_MEANING[verdict]}."
        message = document["context"]["evidence"]["evidence"].get("agent_message")
        if isinstance(message, str) and message:
            text += f" {message}"
        return text
    return f"QWED admission {admission} ({engine}): {_ADMISSION_MEANING[admission]}."


def result_to_openeval(
    document: Any,
    *,
    test_case_id: str,
    actual_output: Optional[str] = None,
    validate: Optional[bool] = None,
) -> Dict[str, Any]:
    """Convert one Verification Context document to one EvalPort ``Result``.

    ``actual_output`` defaults to ``object.formal_statement`` (the object of
    verification, ADR-001). ``validate``: ``None`` (default) runs
    ``qwed_sdk.validate_document`` when qwed is installed and structural
    checks otherwise; ``True`` requires qwed; ``False`` runs structural
    checks only.
    """
    if not isinstance(test_case_id, str) or not test_case_id:
        raise ValueError("test_case_id must be a non-empty string")
    doc = _as_document(document)
    vc_validation = _validated(doc, validate)

    verdict = doc["verdict"]
    admission = doc["context"]["decision"]["admission"]
    proof = doc["context"]["proof"]
    proof_ref = doc["context"]["evidence"].get("proof_ref")
    verified = verdict == "VERIFIED"
    admitted = admission == "ADMIT"

    verdict_grader = {
        "grader_id": VERDICT_GRADER_ID,
        "type": VERDICT_GRADER_TYPE,
        "score": 1.0 if verified else None,
        "passed": verified,
        "reason": _reason(verdict, admission, doc, "verdict"),
        "metadata": {
            "qwed": {
                "verdict": verdict,
                "spec_version": doc["spec_version"],
                "proof_ref": proof_ref,
                "verification_context": copy.deepcopy(doc),
            }
        },
    }
    admission_grader = {
        "grader_id": ADMISSION_GRADER_ID,
        "type": ADMISSION_GRADER_TYPE,
        "score": 1.0 if admitted else 0.0,
        "passed": admitted,
        "reason": _reason(verdict, admission, doc, "admission"),
        "metadata": {"qwed": {"admission": admission}},
    }
    return {
        "test_case_id": test_case_id,
        "actual_output": doc["object"]["formal_statement"] if actual_output is None else actual_output,
        "grader_results": [verdict_grader, admission_grader],
        "passed": verified and admitted,
        "metadata": {
            "qwed": {
                "verdict": verdict,
                "admission": admission,
                "verifier": proof["verifier"],
                "verifier_version": proof["verifier_version"],
                "proof_ref": proof_ref,
                "spec_version": doc["spec_version"],
                "vc_validation": vc_validation,
            }
        },
    }


def _ids(n: int, test_case_ids: Optional[Sequence[str]]) -> List[str]:
    if test_case_ids is None:
        return [f"qwed-{i + 1}" for i in range(n)]
    ids = list(test_case_ids)
    if len(ids) != n:
        raise ValueError(f"got {len(ids)} test_case_ids for {n} documents")
    if len(set(ids)) != len(ids):
        raise ValueError("test_case_ids must be unique")
    return ids


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_openeval(
    documents: Iterable[Any],
    *,
    suite_id: str = "qwed-verification",
    run_id: Optional[str] = None,
    test_case_ids: Optional[Sequence[str]] = None,
    actual_outputs: Optional[Sequence[Optional[str]]] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    validate: Optional[bool] = None,
) -> Dict[str, Any]:
    """Convert Verification Context documents to an EvalPort ``ResultSet``.

    One ``Result`` per document, in order. ``test_case_ids`` default to
    ``qwed-1``, ``qwed-2``, ...; pass the ids of the matching EvalPort test
    cases when you have them. See the module docstring for the mapping.
    """
    docs = list(documents)
    if not docs:
        raise ValueError("at least one Verification Context document is required")
    ids = _ids(len(docs), test_case_ids)
    outputs = list(actual_outputs) if actual_outputs is not None else [None] * len(docs)
    if len(outputs) != len(docs):
        raise ValueError(f"got {len(outputs)} actual_outputs for {len(docs)} documents")

    results = [
        result_to_openeval(doc, test_case_id=tc_id, actual_output=out, validate=validate)
        for doc, tc_id, out in zip(docs, ids, outputs)
    ]
    verdict_counts = {v: 0 for v in _VERDICTS}
    admission_counts = {a: 0 for a in _ADMISSIONS}
    for r in results:
        verdict_counts[r["metadata"]["qwed"]["verdict"]] += 1
        admission_counts[r["metadata"]["qwed"]["admission"]] += 1
    n_passed = sum(1 for r in results if r["passed"])

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id or f"qwed-{uuid.uuid4().hex[:12]}",
        "started_at": started_at or _now(),
        "runner": {"name": "qwed-openeval-adapter", "version": __version__},
        "results": results,
        "summary": {
            "total": len(results),
            "passed": n_passed,
            "failed": len(results) - n_passed,
            "pass_rate": n_passed / len(results),
        },
        "metadata": {
            "qwed": {
                "vc_spec_version": VC_SPEC_VERSION,
                "verdict_counts": verdict_counts,
                "admission_counts": admission_counts,
                "passed_semantics": {
                    VERDICT_GRADER_ID: "the claim was definitively established by QWED (VERIFIED); "
                    "UNVERIFIABLE/BLOCKED are score null (not verified), not scored failures",
                    ADMISSION_GRADER_ID: "the result is admissible (ADMIT)",
                    "result": "VERIFIED and ADMIT",
                },
            }
        },
    }
    if completed_at:
        result_set["completed_at"] = completed_at
    return result_set


def suite_to_openeval(
    documents: Iterable[Any],
    *,
    suite_id: str = "qwed-verification",
    name: Optional[str] = None,
    test_case_ids: Optional[Sequence[str]] = None,
    validate: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build an EvalPort ``EvalSuite`` describing what QWED verified.

    One ``TestCase`` per document: ``input`` is
    ``object.formalization.source_query`` when present, else
    ``object.formal_statement``; both graders are referenced by id and
    defined once at suite level (non-standard types, so ``params.handler``
    is set per EvalPort's custom-type rule).
    """
    docs = [_as_document(d) for d in documents]
    if not docs:
        raise ValueError("at least one Verification Context document is required")
    ids = _ids(len(docs), test_case_ids)
    test_cases = []
    for doc, tc_id in zip(docs, ids):
        _validated(doc, validate)
        obj = doc["object"]
        formalization = obj.get("formalization") or {}
        source_query = formalization.get("source_query")
        test_cases.append(
            {
                "id": tc_id,
                "input": source_query if isinstance(source_query, str) and source_query else obj["formal_statement"],
                "graders": [VERDICT_GRADER_ID, ADMISSION_GRADER_ID],
                "metadata": {
                    "qwed": {
                        "formal_statement": obj["formal_statement"],
                        "verifier": doc["context"]["proof"]["verifier"],
                        "interpretation": copy.deepcopy(doc["context"]["interpretation"]),
                    }
                },
            }
        )
    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "graders": [
            {
                "id": VERDICT_GRADER_ID,
                "type": VERDICT_GRADER_TYPE,
                "params": {"handler": "qwed_openeval_adapter.verdict", "vc_spec_version": VC_SPEC_VERSION},
                "description": "QWED truth verdict. passed = the claim was definitively established (VERIFIED); "
                "UNVERIFIABLE/BLOCKED produce score null (not verified).",
            },
            {
                "id": ADMISSION_GRADER_ID,
                "type": ADMISSION_GRADER_TYPE,
                "params": {"handler": "qwed_openeval_adapter.admission", "vc_spec_version": VC_SPEC_VERSION},
                "description": "QWED admission decision, independent of the verdict. passed = the result is admissible (ADMIT).",
            },
        ],
        "test_cases": test_cases,
        "metadata": {"qwed": {"vc_spec_version": VC_SPEC_VERSION}},
    }
    if name:
        suite["name"] = name
    return suite


# ---------------------------------------------------------------------------
# EvalPort -> QWED
# ---------------------------------------------------------------------------


def _grader(result: Mapping[str, Any], grader_id: str) -> Optional[Mapping[str, Any]]:
    for gr in result.get("grader_results") or []:
        if isinstance(gr, Mapping) and gr.get("grader_id") == grader_id:
            return gr
    return None


def result_from_openeval(result: Mapping[str, Any], *, validate: Optional[bool] = None) -> Dict[str, Any]:
    """Recover the Verification Context document from one EvalPort ``Result``.

    Returns the embedded document unchanged (deep copy). Raises
    ``ValueError`` if the ``Result`` was not produced by this mapping, or if
    its grader scores/passed disagree with the embedded document (e.g. an
    edited score), since that would make the two representations diverge.
    """
    verdict_gr = _grader(result, VERDICT_GRADER_ID)
    if verdict_gr is None:
        raise ValueError(f"result {result.get('test_case_id')!r} has no {VERDICT_GRADER_ID!r} grader result")
    embedded = ((verdict_gr.get("metadata") or {}).get("qwed") or {}).get("verification_context")
    if not isinstance(embedded, Mapping):
        raise ValueError(
            f"result {result.get('test_case_id')!r}: {VERDICT_GRADER_ID} grader carries no "
            f"metadata.qwed.verification_context"
        )
    doc = copy.deepcopy(dict(embedded))
    _validated(doc, validate)

    verified = doc["verdict"] == "VERIFIED"
    admitted = doc["context"]["decision"]["admission"] == "ADMIT"
    admission_gr = _grader(result, ADMISSION_GRADER_ID)
    expected = [
        (verdict_gr, 1.0 if verified else None, verified),
    ]
    if admission_gr is None:
        raise ValueError(f"result {result.get('test_case_id')!r} has no {ADMISSION_GRADER_ID!r} grader result")
    expected.append((admission_gr, 1.0 if admitted else 0.0, admitted))
    for gr, score, passed in expected:
        if gr.get("score") != score or gr.get("passed") is not passed:
            raise ValueError(
                f"result {result.get('test_case_id')!r}: {gr.get('grader_id')} has "
                f"score={gr.get('score')!r}, passed={gr.get('passed')!r}, but the embedded "
                f"Verification Context implies score={score!r}, passed={passed!r}"
            )
    if result.get("passed") is not (verified and admitted):
        raise ValueError(
            f"result {result.get('test_case_id')!r}: passed={result.get('passed')!r} disagrees with "
            f"verdict={doc['verdict']}, admission={doc['context']['decision']['admission']}"
        )
    return doc


def from_openeval(result_set: Mapping[str, Any], *, validate: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Recover Verification Context documents from an EvalPort ``ResultSet``.

    Inverse of ``to_openeval()``: returns one document per ``Result``, in
    order, byte-for-byte equal (as JSON) to the documents passed in.
    """
    results = result_set.get("results") if isinstance(result_set, Mapping) else None
    if not isinstance(results, list):
        raise ValueError("result_set.results must be a list")
    return [result_from_openeval(r, validate=validate) for r in results]
