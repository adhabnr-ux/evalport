"""Argilla <-> EvalPort adapter.

Standalone converter between `Argilla <https://github.com/argilla-io/argilla>`_
``Record`` objects (2.x SDK: ``fields`` + ``suggestions`` + ``responses``) and
the EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Why Argilla is a genuinely good fit for EvalPort: an Argilla ``Record`` *is*
an evaluation-dataset row -- ``fields`` are the inputs shown to an annotator,
``suggestions`` are pre-computed candidate judgments (e.g. from an LLM judge
run before human review), and ``responses`` are the real, completed human
judgments. That maps almost directly onto EvalPort's ``TestCase`` (input +
graders) and ``ResultSet`` (grader results), which is exactly the "portable
LLM evaluation dataset" this project exists to standardize.

Why this is a standalone package rather than living inside the Argilla SDK
itself, following the same playbook that already worked for AutoGen,
CrewAI, LangSmith, and Guardrails (see the sibling ``*-openeval-adapter``
packages in this repo): it works against Argilla's public, documented
``Record``/``Suggestion``/``Response`` classes from the outside, so you get
EvalPort import/export today without anything needing to merge into
``argilla`` itself.

A real constraint this adapter works *around* deliberately, not by
accident: constructing a live ``argilla.Settings`` / ``Field`` / ``Question``
object -- or an ``argilla.Dataset`` -- requires a connected ``Argilla``
client, and that client validates the connection **eagerly**, at
construction time (``Argilla.__init__`` calls ``GET /api/v1/me`` before it
returns -- confirmed directly against ``argilla`` 2.8.0; it raises
``httpx.ConnectError`` with no server reachable). A conversion library has
no business requiring a live server just to translate data, so this adapter
only ever touches the parts of the Argilla object model that are
constructible fully offline: ``Record``, ``Suggestion``, and ``Response``.
All three were confirmed instantiable and serializable with zero network
calls. Dataset *settings* (field/question type definitions) are therefore
represented here as plain dicts under suite/test-case metadata, not as live
Argilla objects -- you build the real ``rg.Settings`` yourself, once you
have a connected client, from the field names this adapter preserves.

Tracked as https://github.com/adhabnr-ux/evalport/issues/6.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "responses_to_openeval",
    "__version__",
]
__version__ = "0.1.0"

_ARGILLA_META_KEY = "argilla"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    Accepts either a live ``argilla.Record``/``Suggestion``/``Response``
    instance or the plain-dict shape produced by ``record.to_dict()`` (or
    loaded straight from JSON), so callers who already serialized their
    records don't need to reconstruct live Argilla objects just to hand
    them to this adapter.
    """
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _fields_of(record: Any) -> Dict[str, Any]:
    fields = _get(record, "fields", {}) or {}
    if isinstance(fields, Mapping):
        return dict(fields)
    # Live ``RecordFields`` supports Mapping-like access without being a
    # ``Mapping`` subclass on some argilla versions; fall back to dict().
    return dict(fields)


def _suggestions_of(record: Any) -> Dict[str, Any]:
    """Return `{question_name: {"value", "score", "agent", "type"}}`.

    A live ``argilla.Record``'s ``.suggestions`` is a ``RecordSuggestions``
    collection: it exposes no ``.items()`` -- it's iterable, yielding
    ``Suggestion`` objects that each carry their own ``question_name``
    (confirmed directly against argilla 2.8.0). A plain dict (matching
    ``Record.to_dict()``'s shape) is already grouped as
    ``{question_name: {...}}``. Both are handled here.
    """
    raw = _get(record, "suggestions", {}) or {}
    out: Dict[str, Any] = {}
    if isinstance(raw, Mapping):
        for name, sug in raw.items():
            out[name] = {
                "value": sug.get("value"),
                "score": sug.get("score"),
                "agent": sug.get("agent"),
                "type": sug.get("type"),
            }
    else:
        for sug in raw:
            out[sug.question_name] = {
                "value": getattr(sug, "value", None),
                "score": getattr(sug, "score", None),
                "agent": getattr(sug, "agent", None),
                "type": getattr(sug, "type", None),
            }
    return out


def _responses_of(record: Any) -> Dict[str, List[Any]]:
    """Return `{question_name: [Response-like, ...]}` (one list per question;
    Argilla allows more than one annotator response per question).

    Like ``.suggestions``, a live ``Record``'s ``.responses`` is a
    ``RecordResponses`` collection with no ``.items()`` -- it's iterable,
    yielding flat ``Response`` objects each carrying their own
    ``question_name``, which this function groups itself. A plain dict
    (matching ``Record.to_dict()``'s shape) is already grouped as
    ``{question_name: [{...}, ...]}``.
    """
    raw = _get(record, "responses", {}) or {}
    out: Dict[str, List[Any]] = {}
    if isinstance(raw, Mapping):
        for name, vals in raw.items():
            out[name] = list(vals)
    else:
        for resp in raw:
            out.setdefault(resp.question_name, []).append(resp)
    return out


def _record_id(record: Any, fallback: str) -> str:
    rid = _get(record, "id", None)
    return str(rid) if rid is not None else fallback


def to_openeval(
    records: Sequence[Any],
    input_fields: Optional[Sequence[str]] = None,
    expected_output_field: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    version: str = OPENEVAL_VERSION,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert Argilla ``Record`` objects into an EvalPort suite.

    Each record's ``fields`` become the test case's ``input`` -- a single
    string when there's exactly one input field, an ordered array of
    strings when there are several (with the original field names preserved
    losslessly in ``metadata.argilla.field_names`` so ``from_openeval`` can
    reconstruct the original dict keys).

    Every test case is graded by a single ``"human"`` grader
    (``{"id": "human", "type": "human"}``) -- the one EvalPort grader type
    with zero required ``params`` (see ``spec/schemas/grader.json``), and
    the honest choice for a platform whose entire purpose is capturing
    human judgment. Any pre-existing ``suggestions`` on a record (candidate
    judgments produced *before* human review, e.g. by an LLM-judge pass)
    are preserved verbatim under ``metadata.argilla.suggestions`` rather
    than promoted to executed grader results -- a suggestion is a proposal,
    not yet a validated result, and this adapter never fabricates the
    latter from the former.

    Args:
        records: Argilla ``Record`` instances, or the plain-dict shape
            produced by ``record.to_dict()``.
        input_fields: which field names (in order) make up the input. If
            omitted, every field on the first record except
            ``expected_output_field`` is used, in the order Python's
            (order-preserving) dict returns them.
        expected_output_field: optional field name whose value becomes
            ``TestCase.expected_output``.
        ids: optional explicit test case ids, positional with ``records``.
            Defaults to each record's own ``.id`` (stringified), falling
            back to ``"record-<index>"`` when a record has no id.
        suite_id: EvalPort suite id. Defaults to ``"argilla_suite"``.
        version: EvalPort spec version to stamp the suite with.
        description: optional human-readable suite description.

    Returns:
        A dict matching ``spec/schemas/suite.json``.
    """
    if not records:
        raise ValueError("to_openeval() requires at least one record")

    if input_fields is None:
        first_fields = _fields_of(records[0])
        derived = [name for name in first_fields.keys() if name != expected_output_field]
        if not derived:
            raise ValueError(
                "could not derive input_fields from the first record's fields "
                "(it has no fields other than expected_output_field); pass "
                "input_fields explicitly"
            )
        input_fields = derived

    test_cases: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        fields = _fields_of(record)
        tc_id = str(ids[i]) if ids is not None else _record_id(record, f"record-{i}")

        values = [fields.get(name, "") for name in input_fields]
        values = ["" if v is None else str(v) for v in values]
        tc_input: Any = values[0] if len(values) == 1 else values

        tc: Dict[str, Any] = {
            "id": tc_id,
            "input": tc_input,
            "graders": ["human"],
        }

        if expected_output_field is not None and fields.get(expected_output_field) is not None:
            tc["expected_output"] = str(fields[expected_output_field])

        argilla_meta: Dict[str, Any] = {"field_names": list(input_fields)}
        record_meta = _get(record, "metadata", {}) or {}
        if record_meta:
            argilla_meta["record_metadata"] = dict(record_meta)
        suggestions = _suggestions_of(record)
        if suggestions:
            argilla_meta["suggestions"] = suggestions
        raw_id = _get(record, "id", None)
        if raw_id is not None:
            argilla_meta["record_id"] = str(raw_id)

        tc["metadata"] = {_ARGILLA_META_KEY: argilla_meta}
        test_cases.append(tc)

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or "argilla_suite",
        "test_cases": test_cases,
        "graders": [{"id": "human", "type": "human", "description": "A human annotator's judgment, captured in Argilla."}],
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite into ready-to-log Argilla record specs.

    Returns one plain dict per test case, shaped exactly like
    ``argilla.Record.to_dict()`` output -- pass each one through
    ``argilla.Record.from_dict(spec)`` to get a live ``Record`` (this
    adapter does not construct ``Record`` objects directly so it never
    silently depends on a particular installed argilla version's
    constructor signature; ``from_dict`` is Argilla's own stable
    reconstruction path).

    Field names are restored from ``metadata.argilla.field_names`` when
    present (round-tripping a suite this adapter produced). For a suite
    from elsewhere (no Argilla-specific metadata), fields are named
    ``field_0``, ``field_1``, ... in input order -- an honest, clearly
    synthetic fallback, not a guess at real names this adapter has no way
    to know.

    Pre-existing suggestions captured by ``to_openeval`` under
    ``metadata.argilla.suggestions`` are restored onto the record so a
    round trip through this adapter doesn't silently drop them.
    """
    specs: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        meta = tc.get("metadata", {}) or {}
        argilla_meta = meta.get(_ARGILLA_META_KEY, {}) or {}

        raw_input = tc["input"]
        values = raw_input if isinstance(raw_input, list) else [raw_input]
        field_names = argilla_meta.get("field_names")
        if not field_names or len(field_names) != len(values):
            field_names = [f"field_{i}" for i in range(len(values))]

        fields = dict(zip(field_names, values))
        if "expected_output" in tc:
            fields.setdefault("expected_output", tc["expected_output"])

        spec: Dict[str, Any] = {
            "id": argilla_meta.get("record_id", tc["id"]),
            "fields": fields,
            "metadata": dict(argilla_meta.get("record_metadata", {})),
        }

        suggestions = argilla_meta.get("suggestions")
        if suggestions:
            spec["suggestions"] = {
                name: {k: v for k, v in sug.items() if v is not None}
                for name, sug in suggestions.items()
            }
        specs.append(spec)
    return specs


def _normalize_score(
    value: Any, question_name: str, rating_ranges: Optional[Mapping[str, Tuple[float, float]]]
) -> Optional[float]:
    """Return a 0..1 score for a response value, or None when it can't be
    computed honestly (no fabricated numbers for non-numeric judgments)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if rating_ranges and question_name in rating_ranges:
            lo, hi = rating_ranges[question_name]
            if hi > lo:
                normalized = (float(value) - lo) / (hi - lo)
                return max(0.0, min(1.0, normalized))
        if 0.0 <= float(value) <= 1.0:
            return float(value)
        return None
    return None


def responses_to_openeval(
    records: Sequence[Any],
    ids: Optional[Sequence[str]] = None,
    suite_id: str = "argilla_suite",
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    version: str = OPENEVAL_VERSION,
    rating_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
    passing_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Convert Argilla records that carry completed human ``responses``
    into an EvalPort ``ResultSet``.

    Every question a record has at least one ``Response`` for becomes one
    ``GraderResult`` of ``type: "human"`` (the zero-required-params
    grader type -- this is real, completed human judgment, never a
    fabrication). Argilla allows more than one annotator to answer the same
    question on the same record; when a question has exactly one response
    its ``grader_id`` is the question name, and when it has several each
    gets its own ``grader_id`` of ``"<question_name>[<index>]"`` so no
    annotator's judgment is silently dropped or averaged away. The
    responding annotator's real ``user_id`` and ``status`` are preserved in
    each grader result's ``metadata``.

    Scoring: a numeric or boolean response value produces a 0..1 ``score``
    (booleans map to 1.0/0.0; numbers already in [0, 1] are used directly;
    numbers outside that range are normalized via ``rating_ranges`` -- e.g.
    ``{"quality": (1, 5)}`` for a 1-5 ``RatingQuestion`` -- when supplied,
    else left as ``score: null`` rather than guessing a scale). A label,
    multi-label, ranking, span, or free-text response value always has
    ``score: null``: there's no honest way to turn "the annotator picked
    label X" into a number without knowing what counts as correct, so this
    adapter doesn't invent one. ``passed`` is ``score >= passing_threshold``
    when a score was computed, and ``True`` otherwise -- a completed
    response with no computable score still represents a human having
    reviewed and judged the item, which this adapter treats as a pass by
    default; override ``passing_threshold`` (default ``0.5``) to change the
    cutoff for the numeric case, or post-process ``passed`` yourself for
    stricter label-matching semantics.

    Records with no responses at all (not yet annotated) are skipped --
    there is nothing real to report yet.

    Raises:
        ValueError: if no record in ``records`` has any responses.
    """
    results: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        responses = _responses_of(record)
        if not responses:
            continue

        tc_id = str(ids[i]) if ids is not None else _record_id(record, f"record-{i}")
        grader_results: List[Dict[str, Any]] = []
        for question_name, resp_list in responses.items():
            multi = len(resp_list) > 1
            for idx, resp in enumerate(resp_list):
                grader_id = f"{question_name}[{idx}]" if multi else question_name
                value = _get(resp, "value")
                score = _normalize_score(value, question_name, rating_ranges)
                passed = (score is None) or (score >= passing_threshold)
                user_id = _get(resp, "user_id")
                status = _get(resp, "status")
                grader_results.append(
                    {
                        "grader_id": grader_id,
                        "type": "human",
                        "score": score,
                        "passed": passed,
                        "reason": f"Human response: {value!r}",
                        "metadata": {
                            "user_id": str(user_id) if user_id is not None else None,
                            "status": str(status) if status is not None else None,
                        },
                    }
                )

        results.append(
            {
                "test_case_id": tc_id,
                "grader_results": grader_results,
                "passed": all(gr["passed"] for gr in grader_results),
            }
        )

    if not results:
        raise ValueError(
            "responses_to_openeval() found no records with any responses -- "
            "nothing has been human-annotated yet"
        )

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"argilla_run_{uuid.uuid4().hex[:12]}",
        "started_at": started_at or _now_iso(),
        "results": results,
    }
    if completed_at:
        result_set["completed_at"] = completed_at
    return result_set
