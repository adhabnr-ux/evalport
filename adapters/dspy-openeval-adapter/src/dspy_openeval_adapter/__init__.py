"""Convert between DSPy (https://github.com/stanfordnlp/dspy) datasets/eval
results and EvalPort (https://github.com/adhabnr-ux/evalport) suites and
result sets.

EvalPort is an open interchange format (Apache 2.0) for portable LLM
evaluation datasets: test cases, graders, suites, and results as plain JSON,
shared across evaluation tools (DeepEval, Promptfoo, Inspect AI, AutoGen,
CrewAI, Ragas, LangSmith, Braintrust, MLflow, Opik, Arize Phoenix, Weights &
Biases Weave, UpTrain, Langfuse, Giskard, LlamaIndex, Patronus AI, Vertex AI,
and now DSPy).

This module has three entry points, matching the "dataset in, results out"
split DSPy itself uses (``dspy.Example`` for a devset, ``dspy.Evaluate`` for
running it):

    to_openeval(devset, input_keys, expected_key=None, ...)
        Converts a ``list[dspy.Example]`` -- a DSPy devset/trainset/testset --
        into an EvalPort suite (test cases only; no results yet).

    from_openeval(suite, input_keys=None, expected_key=None)
        Converts an EvalPort suite back into a ``list[dspy.Example]``, ready
        to hand straight to ``dspy.Evaluate(devset=..., metric=...)``.

    evaluation_result_to_openeval(evaluation, suite_id, ...)
        Converts a ``dspy.EvaluationResult`` (or the raw
        ``list[(example, prediction, score)]`` its ``.results`` attribute
        holds -- both are accepted, since ``dspy.Evaluate.__call__`` is the
        only thing that actually produces one) into an EvalPort ResultSet.

Why field mapping needs a decision, honestly
----------------------------------------------

A ``dspy.Example`` is an arbitrary named-field record (whatever a user's
``dspy.Signature`` calls its inputs/outputs -- ``question``/``answer``,
``context``/``query``/``response``, anything). EvalPort's ``TestCase.input``
is a single string-or-array-of-strings concept, not a named-field dict. This
module does not guess which of a user's fields is "the" input: the caller
names them explicitly via ``input_keys`` (and, optionally, ``expected_key``
for the one field that should become ``expected_output``).

Multiple ``input_keys`` are flattened into EvalPort's array-of-strings input
form as ``f"{key}: {value}"`` per key -- the same "one string per named
field" idiom this ecosystem already uses at the OpenAI chat-message boundary
(see ``openai-python``'s ``OpenEvalItem`` conversion this same session
proposed). On a round trip through *this* adapter specifically, nothing is
lost: the original field values are additionally preserved verbatim under
``test_case.metadata.dspy.fields``, so ``from_openeval()`` reconstructs the
exact original ``Example`` (including any field that isn't an input or the
expected-output field) rather than re-deriving it from the flattened
strings. A suite built by a *different* EvalPort-speaking tool (no
``metadata.dspy.fields``) instead gets one field per array entry, named
positionally (``input_1``, ``input_2``, ...) -- documented explicitly rather
than silently mis-mapped.

What round-trips losslessly, and what doesn't
-----------------------------------------------

DSPy → EvalPort → DSPy (via this adapter both ways): lossless. Every
``Example`` field, plus which fields were marked as inputs via
``with_inputs()``, survives exactly.

DSPy → EvalPort → some other tool: the flattened ``f"{key}: {value}"``
strings and the single ``expected_key`` value are readable by any EvalPort
consumer, but a different tool has no way to know DSPy's own semantics
(which field was the "input" to a *signature*, versus free-form context) --
same tradeoff every adapter in this ecosystem takes for framework-specific
structure that doesn't have a native EvalPort field.

Score normalization for ``evaluation_result_to_openeval()``
--------------------------------------------------------------

A DSPy metric function is arbitrary user code -- it may return ``bool``
(the common case), a plain ``int``/``float`` (rarely outside ``[0, 1]``,
e.g. a raw F1 or exact-match count), or a ``dspy.Prediction`` carrying a
``score`` field plus free-text ``feedback`` (the GEPA-style
feedback-augmented metric shape). All three are handled explicitly:

- ``bool`` → ``score`` 1.0/0.0, ``passed`` the bool itself.
- ``dspy.Prediction`` with a ``score`` field → ``score`` is
  ``float(prediction)`` (DSPy's own ``Prediction.__float__``), and
  ``prediction.get("feedback")`` becomes the grader result's ``reason`` when
  present.
- any other numeric → clamped into EvalPort's required ``[0, 1]`` range for
  the ``score`` field; the *unclamped* raw value is preserved in
  ``grader_result.metadata.dspy.raw_score`` whenever clamping changed it, so
  nothing is silently rewritten without a trace.

``passed`` (for non-bool scores) is ``normalized_score >= pass_threshold``
(default ``0.5``, overridable).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import dspy
except ImportError as e:  # pragma: no cover - exercised by the packaging itself
    raise ImportError(
        "dspy-openeval-adapter requires the 'dspy' package. "
        "Install it with: pip install dspy"
    ) from e

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "evaluation_result_to_openeval",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _example_to_dict(example: Any) -> Dict[str, Any]:
    """Read every genuine data field off a dspy.Example/Prediction.

    ``Example.toDict()`` dumps every key in the example's internal
    ``_store`` -- which is exactly the genuine dataset fields, since this
    module's own round-trip bookkeeping (see ``_DSPY_TC_ID_ATTR`` below) is
    deliberately kept *out* of ``_store`` in the first place.
    """
    if hasattr(example, "toDict"):
        return dict(example.toDict())
    if isinstance(example, dict):
        return dict(example)
    raise TypeError(
        f"Expected a dspy.Example/Prediction or dict, got {type(example).__name__}"
    )


# Instance attribute (not a `_store` field!) used to round-trip a TestCase
# id through a reconstructed dspy.Example. `Example.__setattr__` special-
# cases any key starting with "_": it's set as a real Python instance
# attribute via `object.__setattr__`, bypassing `_store` entirely -- the
# same mechanism DSPy itself uses for `_input_keys`/`_demos`. That matters
# because `_store` is exactly what `toDict()`, `inputs()`, and, critically,
# `labels()` read from: a `_store` field (even one named with a "dspy_"
# prefix -- that prefix is *not* special beyond `__len__`) would leak into
# `example.labels()` as a spurious extra label, corrupting exactly the
# comparison a metric function makes against the model's real output. An
# underscore-prefixed instance attribute is invisible to all of those.
_DSPY_TC_ID_ATTR = "_openeval_test_case_id"


def to_openeval(
    devset: Sequence[Any],
    input_keys: Sequence[str],
    expected_key: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    grader_id: str = "dspy_metric",
    version: str = OPENEVAL_VERSION,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an EvalPort suite from a DSPy devset (``list[dspy.Example]``).

    Args:
        devset: A DSPy devset/trainset/testset -- ``dspy.Example`` instances,
            or plain dicts with the same fields.
        input_keys: Which example field(s) are the model's input, in the
            order they should appear in ``TestCase.input`` (an array of
            ``f"{key}: {value}"`` strings, one per key -- see the module
            docstring for why this isn't force-collapsed into one string).
            Must be non-empty and every key must exist on every example.
        expected_key: Optional field name to use as ``expected_output``
            (e.g. ``"answer"``). Omitted from the suite when ``None`` or
            when a given example doesn't have that field.
        ids: Optional explicit test case ids; auto-generated
            (``dspy_tc_<n>``) if omitted.
        suite_id: EvalPort ``Suite.id``; defaults to ``"dspy_suite"``.
        grader_id: The id of the single placeholder grader this suite
            references. DSPy's actual scoring logic is a Python callable
            passed to ``dspy.Evaluate(metric=...)`` at run time, not
            something with a portable, re-executable representation --
            EvalPort has no way to serialize an arbitrary Python closure,
            so (matching how ``CustomMetric``/``code``/``human`` graders are
            handled across this whole ecosystem) this emits one ``custom``
            grader that documents the metric must be supplied by whoever
            runs the suite, rather than fabricating a fake grader
            implementation.
        version, description: EvalPort Suite-level fields.

    Returns:
        A dict matching EvalPort's Suite schema
        (validate with ``openeval.validate.validate_suite``).

    Raises:
        ValueError: if ``devset``/``input_keys`` is empty, ``ids`` has a
            mismatched length, or an example is missing one of
            ``input_keys``.
    """
    if not devset:
        raise ValueError("to_openeval: devset is empty -- nothing to convert.")
    if not input_keys:
        raise ValueError(
            "to_openeval: input_keys is empty -- specify which example "
            "field(s) are the model input."
        )
    if ids is not None and len(ids) != len(devset):
        raise ValueError(
            f"to_openeval: ids has length {len(ids)}, expected {len(devset)} "
            "(one entry per example)."
        )

    test_cases: List[Dict[str, Any]] = []
    for i, example in enumerate(devset):
        fields = _example_to_dict(example)
        # No need to strip this adapter's own round-trip bookkeeping here:
        # it's kept as an instance attribute (_DSPY_TC_ID_ATTR), never a
        # `_store` field, so it never appears in `_example_to_dict()`'s
        # output even if `devset` itself came from a prior
        # `from_openeval()` call.
        missing = [k for k in input_keys if k not in fields]
        if missing:
            raise ValueError(
                f"to_openeval: example {i} is missing input key(s) {missing} "
                f"(has fields: {sorted(fields.keys())})."
            )

        tc_id = ids[i] if ids else f"dspy_tc_{i}"
        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": [f"{k}: {fields[k]}" for k in input_keys],
            "graders": [grader_id],
            "metadata": {"dspy": {"fields": fields, "input_keys": list(input_keys)}},
        }
        if expected_key is not None and expected_key in fields:
            test_case["expected_output"] = str(fields[expected_key])
            test_case["metadata"]["dspy"]["expected_key"] = expected_key

        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or "dspy_suite",
        "graders": [
            {
                "id": grader_id,
                "type": "custom",
                "params": {"handler": grader_id},
                "description": (
                    "Placeholder for a DSPy metric function passed to "
                    "dspy.Evaluate(metric=...) at run time -- DSPy metrics "
                    "are arbitrary Python callables with no portable, "
                    "re-executable representation, so this grader documents "
                    "that the caller must supply one, rather than "
                    "fabricating a fake implementation."
                ),
            }
        ],
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(
    suite: Dict[str, Any],
    input_keys: Optional[Sequence[str]] = None,
    expected_key: str = "expected_output",
) -> List[Any]:
    """Convert an EvalPort suite back into a ``list[dspy.Example]``.

    Ready to hand straight to ``dspy.Evaluate(devset=..., metric=...)``.

    For a test case carrying this adapter's own
    ``metadata.dspy.fields`` (i.e. one this adapter itself exported via
    ``to_openeval()``), the *exact* original example fields and input-key
    marking are restored -- a lossless DSPy → EvalPort → DSPy round trip.

    For any other test case (hand-authored, or produced by a different
    EvalPort-speaking tool), a best-effort ``Example`` is built instead:
    each entry of the array-form ``input`` becomes its own field, named
    positionally (``input_1``, ``input_2``, ...) unless ``input_keys`` is
    given explicitly (in which case those names are used, one per input
    entry -- length must match). A string-form ``input`` becomes a single
    ``input_1`` field. ``expected_output``, when present, becomes a field
    named by ``expected_key`` (default ``"expected_output"``). All input
    field(s) are marked via ``.with_inputs(...)``; the expected-output field
    is deliberately left unmarked, since it's the label, not something a
    DSPy program should receive as input.

    Args:
        suite: An EvalPort suite dict.
        input_keys: Field names to use for a non-``dspy``-sourced test
            case's array-form input entries, positionally. Ignored for test
            cases carrying this adapter's own ``metadata.dspy.fields``
            (those always restore their real original field names).
        expected_key: Field name for a non-``dspy``-sourced test case's
            ``expected_output``, when present.

    Returns:
        A list of ``dspy.Example`` instances, each with the appropriate
        input fields marked via ``with_inputs()``.

    Raises:
        ValueError: if the suite has no test cases.
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")

    examples: List[Any] = []
    for tc in test_cases:
        dspy_meta = ((tc.get("metadata") or {}).get("dspy")) or {}

        if dspy_meta.get("fields") is not None:
            # Lossless path: this adapter's own export.
            fields = dict(dspy_meta["fields"])
            marked_inputs = list(dspy_meta.get("input_keys") or [])
            example = dspy.Example(**fields)
            if marked_inputs:
                example = example.with_inputs(*marked_inputs)
        else:
            # Best-effort path: hand-authored or foreign-tool suite.
            raw_input = tc.get("input")
            entries = raw_input if isinstance(raw_input, list) else [raw_input]
            if input_keys is not None:
                if len(input_keys) != len(entries):
                    raise ValueError(
                        f"from_openeval: test case {tc.get('id')!r} has "
                        f"{len(entries)} input entries but input_keys has "
                        f"{len(input_keys)} names."
                    )
                names = list(input_keys)
            else:
                names = [f"input_{j + 1}" for j in range(len(entries))]

            fields = dict(zip(names, entries))
            if tc.get("expected_output") is not None:
                fields[expected_key] = tc["expected_output"]

            example = dspy.Example(**fields).with_inputs(*names)

        # setattr (not `example[...] = ...`, which always writes through to
        # `_store` regardless of key name) -- this routes through
        # `Example.__setattr__`'s underscore special case, landing as a
        # real instance attribute instead of a `_store` field.
        setattr(example, _DSPY_TC_ID_ATTR, tc.get("id"))
        examples.append(example)

    return examples


def _normalize_score(
    score: Any, pass_threshold: float
) -> Tuple[float, bool, Optional[str], Optional[float]]:
    """Return (normalized_score, passed, reason, raw_score_if_clamped)."""
    if isinstance(score, bool):
        return (1.0 if score else 0.0, score, None, None)

    if isinstance(score, dspy.Prediction):
        raw = float(score)
        reason = None
        if "feedback" in score:
            fb = score["feedback"]
            reason = str(fb) if fb is not None else None
        clamped = max(0.0, min(1.0, raw))
        return (clamped, clamped >= pass_threshold, reason, raw if raw != clamped else None)

    raw = float(score)
    clamped = max(0.0, min(1.0, raw))
    return (clamped, clamped >= pass_threshold, None, raw if raw != clamped else None)


def _prediction_to_text(prediction: Any) -> Optional[str]:
    """Render a dspy.Prediction (or plain program output) as a display string.

    ``prediction`` here is the model's *output* (e.g. ``Prediction(answer=...)``)
    -- a separate object from the *score* (which may itself be a
    ``Prediction(score=..., feedback=...)`` for GEPA-style metrics, handled
    in ``_normalize_score`` instead). Every field is included verbatim.
    """
    if prediction is None:
        return None
    if hasattr(prediction, "toDict"):
        fields = prediction.toDict()
        if not fields:
            return None
        if len(fields) == 1:
            return str(next(iter(fields.values())))
        return "; ".join(f"{k}: {v}" for k, v in fields.items())
    return str(prediction)


def evaluation_result_to_openeval(
    evaluation: Any,
    suite_id: str = "dspy_suite",
    grader_id: Optional[str] = None,
    metric: Any = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    pass_threshold: float = 0.5,
    version: str = OPENEVAL_VERSION,
) -> Dict[str, Any]:
    """Convert DSPy evaluation results into an EvalPort ResultSet.

    Args:
        evaluation: Either a ``dspy.EvaluationResult`` (what
            ``dspy.Evaluate(...)(program)`` returns), or the raw
            ``list[(example, prediction, score)]`` its ``.results``
            attribute holds -- both are accepted, duck-typed on the
            presence of a ``.results`` attribute.
        suite_id: The EvalPort suite this ResultSet's ``test_case_id``s
            refer to.
        grader_id: The grader id to attach each score to. Defaults to
            ``metric.__name__``/``metric.__class__.__name__`` when
            ``metric`` is given (mirroring ``dspy.Evaluate``'s own naming
            convention for its results table), else ``"dspy_metric"``.
        metric: Optional -- the metric callable/object passed to
            ``dspy.Evaluate(metric=...)``. Only used to derive a readable
            default ``grader_id``; never called.
        run_id: EvalPort ``ResultSet.run_id``; a random one is generated if
            omitted.
        started_at, completed_at: ISO-8601 timestamps. ``started_at``
            defaults to now if omitted (required by the EvalPort schema;
            ``dspy.EvaluationResult`` doesn't expose a run-level start
            time itself).
        pass_threshold: For non-bool scores, a case passes when its
            normalized score is ``>= pass_threshold``. Ignored for ``bool``
            metric results, which use the bool directly.
        version: EvalPort schema version.

    Returns:
        A dict matching EvalPort's ResultSet schema
        (validate with ``openeval.validate.validate_result_set``).

    Raises:
        ValueError: if there are no results to convert.
    """
    results = getattr(evaluation, "results", evaluation)
    results = list(results)
    if not results:
        raise ValueError(
            "evaluation_result_to_openeval: no results to convert -- "
            "evaluation.results (or the list passed directly) is empty."
        )

    if grader_id is None:
        if metric is not None:
            grader_id = getattr(metric, "__name__", None) or type(metric).__name__
        else:
            grader_id = "dspy_metric"

    results_out: List[Dict[str, Any]] = []
    for i, (example, prediction, score) in enumerate(results):
        tc_id = getattr(example, _DSPY_TC_ID_ATTR, None)
        if not tc_id:
            tc_id = f"dspy_tc_{i}"

        normalized_score, passed, reason, raw_score = _normalize_score(
            score, pass_threshold
        )

        grader_result: Dict[str, Any] = {
            "grader_id": grader_id,
            "type": "custom",
            "score": normalized_score,
            "passed": passed,
        }
        if reason is not None:
            grader_result["reason"] = reason
        if raw_score is not None:
            grader_result["metadata"] = {"dspy": {"raw_score": raw_score}}

        result_entry: Dict[str, Any] = {
            "test_case_id": str(tc_id),
            "grader_results": [grader_result],
            "passed": passed,
        }
        actual_output = _prediction_to_text(prediction)
        if actual_output is not None:
            result_entry["actual_output"] = actual_output

        results_out.append(result_entry)

    total = len(results_out)
    passed_count = sum(1 for r in results_out if r["passed"])

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"dspy_run_{uuid.uuid4().hex[:12]}",
        "started_at": started_at or _now_iso(),
        "results": results_out,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else 0.0,
        },
    }
    if completed_at:
        result_set["completed_at"] = completed_at
    return result_set
