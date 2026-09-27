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
running it), plus an opt-in grader mapping:

    to_openeval(devset, input_keys=None, expected_key=None, ...)
        Converts a ``list[dspy.Example]`` -- a DSPy devset/trainset/testset --
        into an EvalPort suite (test cases only; no results yet).

    from_openeval(suite, input_keys=None, expected_key="expected_output",
                  field_names=None)
        Converts an EvalPort suite back into a ``list[dspy.Example]``, ready
        to hand straight to ``dspy.Evaluate(devset=..., metric=...)``.

    evaluation_result_to_openeval(evaluation, suite_id, ...)
        Converts a ``dspy.EvaluationResult`` (or the raw
        ``list[(example, prediction, score)]`` its ``.results`` attribute
        holds -- both are accepted, since ``dspy.Evaluate.__call__`` is the
        only thing that actually produces one) into an EvalPort ResultSet.

    graders_to_dspy_metrics(suite, expected_key=..., output_key=...)
        Turns a suite's ``exact_match`` graders into DSPy metric functions
        with EvalPort's semantics (other grader types have no faithful DSPy
        equivalent and are refused with the reason).

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
``metadata.dspy.fields``) instead gets one field per input entry, named
positionally (``input_1``, ``input_2``, ...), plus ``context`` /
``retrieval_context`` (inputs) and ``expected_tools`` (a label) fields --
documented explicitly rather than silently mis-mapped. Such an example
remembers that layout, so ``to_openeval()`` writes every value back to its
native TestCase field, with a string ``input`` still a string.

What round-trips losslessly, and what doesn't
-----------------------------------------------

DSPy → EvalPort → DSPy (via this adapter both ways): lossless. Every
``Example`` field, plus which fields were marked as inputs via
``with_inputs()``, survives exactly.

EvalPort → DSPy → EvalPort (a foreign suite through ``from_openeval()`` then
``to_openeval()``): every TestCase data field (``id``, ``input`` in its
original string/array form, ``expected_output``, ``context``,
``retrieval_context``, ``tools_called``, ``expected_tools``, ``tags``,
``metadata``) comes back byte-identical. ``graders`` do not: a list of
``dspy.Example`` has no grader slot, so ``to_openeval()`` emits its
placeholder. ``graders_to_dspy_metrics()`` runs an ``exact_match`` grader
faithfully inside ``dspy.Evaluate`` instead.

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

import copy
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    "graders_to_dspy_metrics",
    "exact_match",
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

# Same mechanism, for how a test case from a *foreign* suite (one this
# adapter didn't export) was laid out as Example fields by from_openeval():
# which fields hold the input entries and in what form (string or array),
# which field holds expected_output, which fields hold context /
# retrieval_context / expected_tools, plus the TestCase fields that are not
# Example data at all (metadata, tags, tools_called). to_openeval() reads it
# to put every value back into its native EvalPort field. It is an instance
# attribute, so -- like the id above -- it never shows up in inputs()/labels(),
# and it does not survive Example.copy()/with_inputs() (DSPy only copies
# `_store`/`_input_keys`); without it, to_openeval() falls back to the plain
# `input_keys` flattening.
_OPENEVAL_SOURCE_ATTR = "_openeval_source"

# EvalPort TestCase fields that from_openeval() turns into Example fields for a
# foreign suite (default field name = the EvalPort field name), and whether
# each one is a program input (with_inputs) or a label.
_NATIVE_FIELDS = {"context": True, "retrieval_context": True, "expected_tools": False}
# TestCase fields carried alongside the Example (not as Example fields).
_EXTRA_FIELDS = ("tags", "tools_called", "metadata")


def _extras(tc: Dict[str, Any], drop_dspy: bool) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    for key in _EXTRA_FIELDS:
        if key not in tc:
            continue
        value = copy.deepcopy(tc[key])
        if key == "metadata" and drop_dspy and isinstance(value, dict):
            value.pop("dspy", None)
            if not value:
                continue
        extra[key] = value
    return extra


def _restore_foreign_test_case(
    example: Any,
    fields: Dict[str, Any],
    source: Dict[str, Any],
    tc_id: str,
    expected_key: Optional[str],
    grader_id: str,
) -> Dict[str, Any]:
    """Rebuild a foreign suite's TestCase from an Example that
    ``from_openeval()`` laid out (see ``_OPENEVAL_SOURCE_ATTR``)."""
    names = list(source["input_keys"])
    values = [fields[k] for k in names]
    if source["input_form"] == "string" and len(values) == 1 \
            and isinstance(values[0], str) and values[0]:
        input_value: Any = values[0]
    elif source["input_form"] == "list" and values and all(isinstance(v, str) for v in values):
        input_value = list(values)
    else:  # the example was edited into something the original form can't hold
        input_value = [f"{k}: {fields[k]}" for k in names]

    test_case: Dict[str, Any] = {"id": tc_id, "input": input_value, "graders": [grader_id]}
    ek = expected_key if expected_key is not None else source.get("expected_key")
    if ek is not None and ek in fields:
        test_case["expected_output"] = str(fields[ek])
    mapped = dict(source.get("fields") or {})
    for native, field_name in mapped.items():
        if field_name in fields:
            test_case[native] = list(fields[field_name])
    extra = copy.deepcopy(source.get("extra") or {})
    for key in ("tags", "tools_called"):
        if key in extra:
            test_case[key] = extra[key]
    metadata = extra.get("metadata")

    # Only fields/markings from_openeval() accounted for? Then the TestCase
    # above already says everything; otherwise keep the full Example too.
    accounted = set(names) | set(mapped.values()) | ({ek} if ek is not None else set())
    expected_inputs = set(names) | {f for n, f in mapped.items() if _NATIVE_FIELDS.get(n)}
    marked = getattr(example, "_input_keys", None)
    if set(fields) - accounted or (marked is not None and set(marked) != expected_inputs):
        marked_inputs = [k for k in fields if k in marked] if marked is not None \
            else [k for k in fields if k in expected_inputs]
        dspy_meta: Dict[str, Any] = {
            "fields": fields,
            "input_keys": marked_inputs,
            "source": {k: source[k] for k in ("input_keys", "input_form", "expected_key", "fields")},
        }
        if ek is not None and ek in fields:
            dspy_meta["expected_key"] = ek
        metadata = dict(metadata or {})
        metadata["dspy"] = dspy_meta
    if metadata is not None:
        test_case["metadata"] = metadata
    return test_case


def to_openeval(
    devset: Sequence[Any],
    input_keys: Optional[Sequence[str]] = None,
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
            May be omitted (``None``) only for examples that came from
            ``from_openeval()`` of a foreign suite, which remember their
            own layout (see "Foreign-suite examples" below).
        expected_key: Optional field name to use as ``expected_output``
            (e.g. ``"answer"``). Omitted from the suite when ``None`` or
            when a given example doesn't have that field.
        ids: Optional explicit test case ids. If omitted, an example that
            came from ``from_openeval()`` keeps its original test case id;
            any other example gets ``dspy_tc_<n>``.
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

    Foreign-suite examples:
        An example that ``from_openeval()`` built from a foreign suite (one
        this adapter didn't export) remembers how the test case was laid
        out. When ``input_keys`` is omitted or equals the field names
        ``from_openeval()`` chose, every value goes back into its native
        EvalPort field: ``input`` in its original form (a string stays a
        string, an array stays an array -- no ``"key: value"`` prefixes),
        ``context`` / ``retrieval_context`` / ``expected_tools`` from their
        Example fields (an empty list stays empty), and ``metadata`` /
        ``tags`` / ``tools_called`` as they were. No ``metadata.dspy``
        bookkeeping is added unless the example gained fields or a
        different input marking since, in which case it is added (merged
        with the original metadata) so nothing is lost.

    Returns:
        A dict matching EvalPort's Suite schema
        (validate with ``openeval.validate.validate_suite``).

    Raises:
        ValueError: if ``devset``/``input_keys`` is empty, ``input_keys``
            is omitted for an example that didn't come from
            ``from_openeval()``, ``ids`` has a mismatched length, or an
            example is missing one of ``input_keys``.
    """
    if not devset:
        raise ValueError("to_openeval: devset is empty -- nothing to convert.")
    if input_keys is not None and not input_keys:
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
        # it's kept as instance attributes (_DSPY_TC_ID_ATTR,
        # _OPENEVAL_SOURCE_ATTR), never `_store` fields, so it never appears
        # in `_example_to_dict()`'s output even if `devset` itself came from
        # a prior `from_openeval()` call.
        source = None if isinstance(example, dict) else getattr(example, _OPENEVAL_SOURCE_ATTR, None)
        if ids:
            tc_id = ids[i]
        else:
            original_id = None if isinstance(example, dict) else getattr(example, _DSPY_TC_ID_ATTR, None)
            tc_id = original_id or f"dspy_tc_{i}"

        if source is not None and (input_keys is None or list(input_keys) == source["input_keys"]) \
                and all(k in fields for k in source["input_keys"]):
            test_cases.append(_restore_foreign_test_case(
                example, fields, source, tc_id, expected_key, grader_id))
            continue

        if input_keys is None:
            raise ValueError(
                f"to_openeval: input_keys not given, and example {i} was not produced by "
                "from_openeval() from a foreign suite -- specify which example field(s) "
                "are the model input."
            )
        missing = [k for k in input_keys if k not in fields]
        if missing:
            raise ValueError(
                f"to_openeval: example {i} is missing input key(s) {missing} "
                f"(has fields: {sorted(fields.keys())})."
            )

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
    field_names: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """Convert an EvalPort suite back into a ``list[dspy.Example]``.

    Ready to hand straight to ``dspy.Evaluate(devset=..., metric=...)``.

    For a test case carrying this adapter's own
    ``metadata.dspy.fields`` (i.e. one this adapter itself exported via
    ``to_openeval()``), the *exact* original example fields and input-key
    marking are restored -- a lossless DSPy → EvalPort → DSPy round trip.

    For any other test case (hand-authored, or produced by a different
    EvalPort-speaking tool), an ``Example`` is built from the TestCase
    fields:

    - ``input``: each entry of the array form becomes its own field, named
      positionally (``input_1``, ``input_2``, ...) unless ``input_keys`` is
      given explicitly (one name per entry -- length must match). A string
      ``input`` becomes a single ``input_1`` field. Marked as inputs.
    - ``expected_output`` becomes a field named by ``expected_key``
      (default ``"expected_output"``). A label, not an input.
    - ``context`` and ``retrieval_context`` become fields of the same name,
      marked as inputs (a RAG program reads them).
    - ``expected_tools`` becomes a field of the same name -- a label, since
      it describes the expected behaviour. An empty list stays an empty
      list ("no tool should be called").
    - ``metadata``, ``tags`` and ``tools_called`` are not Example data;
      they are kept on the example (outside ``inputs()``/``labels()``) so
      ``to_openeval()`` can write them back unchanged.

    ``to_openeval()`` on such an example puts every value back into its
    native field, with ``input`` in its original string/array form, so
    EvalPort → DSPy → EvalPort is lossless for all TestCase data fields.

    Args:
        suite: An EvalPort suite dict.
        input_keys: Field names to use for a non-``dspy``-sourced test
            case's input entries, positionally. Ignored for test
            cases carrying this adapter's own ``metadata.dspy.fields``
            (those always restore their real original field names).
        expected_key: Field name for a non-``dspy``-sourced test case's
            ``expected_output``, when present.
        field_names: Optional renames for the ``context``,
            ``retrieval_context`` and ``expected_tools`` Example fields,
            e.g. ``{"context": "passages"}`` to match a signature.

    Returns:
        A list of ``dspy.Example`` instances, each with the appropriate
        input fields marked via ``with_inputs()``.

    Raises:
        ValueError: if the suite has no test cases, ``input_keys`` doesn't
            match the number of input entries, ``field_names`` names an
            unsupported field, or two TestCase fields would land on the
            same Example field name.
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")
    unknown = sorted(set(field_names or {}) - set(_NATIVE_FIELDS))
    if unknown:
        raise ValueError(
            f"from_openeval: field_names can rename {sorted(_NATIVE_FIELDS)}, not {unknown}."
        )
    native_names = {name: name for name in _NATIVE_FIELDS}
    native_names.update(field_names or {})

    examples: List[Any] = []
    for tc in test_cases:
        dspy_meta = ((tc.get("metadata") or {}).get("dspy")) or {}
        source: Optional[Dict[str, Any]] = None

        if dspy_meta.get("fields") is not None:
            # Lossless path: this adapter's own export.
            fields = dict(dspy_meta["fields"])
            marked_inputs = list(dspy_meta.get("input_keys") or [])
            example = dspy.Example(**fields)
            if marked_inputs:
                example = example.with_inputs(*marked_inputs)
            if dspy_meta.get("source"):
                # An edited foreign-suite example (see to_openeval()): keep
                # its layout so the next to_openeval() restores native fields.
                source = dict(dspy_meta["source"])
                source["extra"] = _extras(tc, drop_dspy=True)
        else:
            # Foreign path: hand-authored or another tool's suite.
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
            has_expected = tc.get("expected_output") is not None
            if has_expected:
                fields[expected_key] = tc["expected_output"]

            mapped: Dict[str, str] = {}
            for native in _NATIVE_FIELDS:
                if native not in tc:
                    continue
                name = native_names[native]
                if name in fields:
                    raise ValueError(
                        f"from_openeval: test case {tc.get('id')!r}: `{native}` would be stored "
                        f"as Example field {name!r}, which is already used (input_keys / "
                        "expected_key). Rename it with field_names={...}."
                    )
                fields[name] = list(tc[native])
                mapped[native] = name

            inputs = names + [mapped[n] for n in mapped if _NATIVE_FIELDS[n]]
            example = dspy.Example(**fields).with_inputs(*inputs)
            source = {
                "input_keys": names,
                "input_form": "list" if isinstance(raw_input, list) else "string",
                "expected_key": expected_key if has_expected else None,
                "fields": mapped,
                "extra": _extras(tc, drop_dspy=False),
            }

        # setattr (not `example[...] = ...`, which always writes through to
        # `_store` regardless of key name) -- this routes through
        # `Example.__setattr__`'s underscore special case, landing as a
        # real instance attribute instead of a `_store` field.
        setattr(example, _DSPY_TC_ID_ATTR, tc.get("id"))
        if source is not None:
            setattr(example, _OPENEVAL_SOURCE_ATTR, source)
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


# ---------------------------------------------------------------------------
# Graders: opt-in, faithful EvalPort -> DSPy metric mapping
# ---------------------------------------------------------------------------

# Params spec/SPEC.md defines for `exact_match` (both optional).
_EXACT_MATCH_PARAMS = ("ignore_case", "trim_whitespace")

_UNSUPPORTED_GRADER_REASONS = {
    "semantic_similarity": "the score depends on the embedding model the runner picks; a DSPy "
                           "metric would have to pick one, which the grader doesn't pin down",
    "llm_judge": "running the grader's prompt needs a judge LM; dspy.evaluate's built-in "
                 "LM metrics (SemanticF1, CompleteAndGrounded) use their own prompts",
    "model graded": "alias of llm_judge (same reason)",
}


def exact_match(
    actual_output: Optional[str],
    expected_output: Optional[str],
    *,
    ignore_case: bool = False,
    trim_whitespace: bool = True,
) -> bool:
    """EvalPort `exact_match` semantics (spec/SPEC.md grader table).

    Mirrors the reference runner (``cli/src/run/graders/tier1.ts``,
    ``gradeExactMatch``) step for step: trim both sides when
    ``trim_whitespace`` (default true), lowercase both sides when
    ``ignore_case`` (default false), then compare with ``==``. No other
    normalization -- unlike ``dspy.evaluate.answer_exact_match``, which also
    drops punctuation and articles. A missing ``expected_output`` compares
    as ``""``, as in the reference.

    One documented difference: Python's ``str.strip()``/``str.lower()`` are
    used where the TypeScript reference uses ``String.trim()``/
    ``toLowerCase()``. Their whitespace sets differ on exactly six code
    points: U+FEFF is trimmed by JS only; U+001C-U+001F and U+0085 are
    stripped by Python only. Lowercasing follows each language's Unicode
    case tables.
    """
    a = "" if actual_output is None else str(actual_output)
    e = "" if expected_output is None else str(expected_output)
    if trim_whitespace:
        a, e = a.strip(), e.strip()
    if ignore_case:
        a, e = a.lower(), e.lower()
    return a == e


def _exact_match_params(grader: Dict[str, Any]) -> Dict[str, bool]:
    params = dict(grader.get("params") or {})
    unknown = sorted(k for k in params if k not in _EXACT_MATCH_PARAMS)
    if unknown:
        warnings.warn(
            f"grader {grader.get('id')!r}: exact_match param(s) {unknown} are not defined by the "
            f"EvalPort spec (which defines {list(_EXACT_MATCH_PARAMS)}) and are ignored, as the "
            "reference runner ignores them",
            UserWarning,
            stacklevel=3,
        )
    return {
        "ignore_case": params.get("ignore_case") is True,
        "trim_whitespace": params.get("trim_whitespace") is not False,
    }


def _prediction_output(pred: Any, output_key: Optional[str]) -> Any:
    if isinstance(pred, str) or pred is None:
        return pred
    if output_key is not None:
        return pred[output_key]
    fields = pred.toDict() if hasattr(pred, "toDict") else dict(pred)
    if len(fields) != 1:
        raise ValueError(
            f"exact_match metric: the prediction has fields {sorted(fields)}; pass "
            "output_key=... to graders_to_dspy_metrics() to say which one is the output."
        )
    return next(iter(fields.values()))


def _make_exact_match_metric(
    grader_id: str, expected_key: str, output_key: Optional[str],
    ignore_case: bool, trim_whitespace: bool,
) -> Callable[..., bool]:
    def metric(example: Any, pred: Any, trace: Any = None) -> bool:
        expected = example.get(expected_key) if hasattr(example, "get") else example[expected_key]
        return exact_match(_prediction_output(pred, output_key), expected,
                           ignore_case=ignore_case, trim_whitespace=trim_whitespace)

    # evaluation_result_to_openeval(metric=...) and dspy.Evaluate's results
    # table both name a metric by __name__ -- use the EvalPort grader id so
    # the ResultSet's grader_id matches the suite's.
    metric.__name__ = grader_id
    metric.__qualname__ = grader_id
    metric.__doc__ = (f"EvalPort exact_match grader {grader_id!r} (ignore_case={ignore_case}, "
                      f"trim_whitespace={trim_whitespace}) as a DSPy metric.")
    return metric


def graders_to_dspy_metrics(
    suite_or_graders: Any,
    *,
    expected_key: str = "expected_output",
    output_key: Optional[str] = None,
    skip_unsupported: bool = False,
) -> Dict[str, Callable[..., bool]]:
    """Opt-in: map an EvalPort suite's grader definitions to DSPy metric functions.

    Returns ``{grader_id: metric}`` for every grader with a faithful DSPy
    equivalent. Each metric has DSPy's signature
    ``metric(example, pred, trace=None) -> bool`` and ``__name__`` set to the
    grader id, so it plugs into ``dspy.Evaluate(metric=...)`` and
    ``evaluation_result_to_openeval(..., metric=metric)`` reports results
    under the suite's own grader id. Pass the suite dict (its ``graders``
    list is read) or a list of grader dicts.

    Args:
        expected_key: Example field holding the reference answer --
            ``from_openeval()``'s default ``"expected_output"``, or whatever
            ``expected_key`` you passed it.
        output_key: Prediction field holding the program's output. If
            omitted, a prediction with exactly one field uses that field
            (a plain string prediction is used as is); otherwise the metric
            raises ``ValueError``.
        skip_unsupported: Leave grader types with no faithful mapping out of
            the result instead of raising ``ValueError``.

    Faithful mappings:

    - ``exact_match`` -> a metric computing ``exact_match()`` with the
      grader's ``ignore_case`` / ``trim_whitespace``. DSPy's own
      ``answer_exact_match`` is *not* used: it lowercases and strips
      punctuation and articles regardless of params, so it passes
      ``"45."`` against ``"45"``, which EvalPort's ``exact_match`` fails.

    Unknown ``exact_match`` params (e.g. ``strip``, not a spec param) raise
    a ``UserWarning`` and are ignored, as the reference runner ignores them.
    """
    graders = suite_or_graders.get("graders", []) if isinstance(suite_or_graders, dict) \
        else list(suite_or_graders)
    metrics: Dict[str, Callable[..., bool]] = {}
    for grader in graders:
        gtype = grader.get("type")
        if gtype == "exact_match":
            metrics[grader["id"]] = _make_exact_match_metric(
                grader["id"], expected_key, output_key, **_exact_match_params(grader))
            continue
        if skip_unsupported:
            continue
        reason = _UNSUPPORTED_GRADER_REASONS.get(
            gtype, "this adapter maps only exact_match, and no dspy.evaluate built-in metric "
                   "computes exactly this grader type")
        raise ValueError(
            f"grader {grader.get('id')!r} (type {gtype!r}) has no faithful DSPy metric: "
            f"{reason}. Pass skip_unsupported=True to map only the supported graders."
        )
    return metrics
