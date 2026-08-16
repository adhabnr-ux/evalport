"""Convert between Haystack (https://github.com/deepset-ai/haystack) evaluation
data/results and EvalPort (https://github.com/adhabnr-ux/evalport) suites and
result sets.

EvalPort is an open interchange format (Apache 2.0) for portable LLM
evaluation datasets: test cases, graders, suites, and results as plain JSON,
shared across evaluation tools (DeepEval, Promptfoo, Inspect AI, AutoGen,
CrewAI, Ragas, LangSmith, Braintrust, MLflow, Opik, Arize Phoenix, Weights &
Biases Weave, UpTrain, Langfuse, Giskard, LlamaIndex, Patronus AI, Vertex AI,
DSPy, and now Haystack).

Haystack's evaluation surface is columnar, not row-based: a plain
``dict[str, list[Any]]`` of named input columns (e.g.
``{"questions": [...], "contexts": [...], "predicted_answers": [...]}``,
exactly what ``haystack.components.evaluators.*`` and
``haystack.evaluation.EvaluationRunResult`` both consume/produce), plus a
``haystack.evaluation.EvaluationRunResult`` -- ``run_name`` + that same
``inputs`` dict + a ``results`` dict keyed by metric name, each holding an
aggregate ``score`` and a per-row ``individual_scores`` list. This module has
three entry points that mirror that shape directly:

    to_openeval(inputs, input_keys, expected_key=None, ...)
        Converts a Haystack-shaped ``dict[str, list[Any]]`` of input columns
        into an EvalPort suite (test cases only; no results yet).

    from_openeval(suite, input_keys=None, expected_key="expected_output")
        Converts an EvalPort suite back into a Haystack-shaped
        ``dict[str, list[Any]]``, ready to unpack into pipeline calls or
        evaluator components (e.g. ``AnswerExactMatchEvaluator().run(
        ground_truth_answers=cols["expected_output"],
        predicted_answers=cols["predicted_answers"])``).

    evaluation_result_to_openeval(run_result, suite_id, ...)
        Converts a ``haystack.evaluation.EvaluationRunResult`` (or any
        duck-typed object/dict exposing the same ``inputs``/``results``
        shape) into an EvalPort ResultSet -- one grader result per metric,
        per row.

Why field mapping needs a decision, honestly
----------------------------------------------

Haystack's ``inputs`` is an arbitrary named-column table (whatever the
caller's pipeline/evaluators need -- ``questions``/``contexts``/
``predicted_answers``, anything). EvalPort's ``TestCase.input`` is a single
string-or-array-of-strings concept, not a named-column table. This module
does not guess which column is "the" input: the caller names them explicitly
via ``input_keys`` (and, optionally, ``expected_key`` for the one column that
should become ``expected_output``).

Multiple ``input_keys`` are flattened into EvalPort's array-of-strings input
form as ``f"{key}: {value}"`` per key, the same "one string per named field"
idiom this ecosystem already uses for other named-field sources (see the
``dspy-openeval-adapter`` for the identical convention with DSPy's
``Example`` fields). On a round trip through *this* adapter specifically,
nothing is lost: every column's value for that row is additionally preserved
verbatim under ``test_case.metadata.haystack.columns``, so
``from_openeval()`` reconstructs the exact original columns rather than
re-deriving them from the flattened strings. A suite built by a *different*
EvalPort-speaking tool (no ``metadata.haystack.columns``) instead gets one
column per array entry, named positionally (``input_1``, ``input_2``, ...)
unless ``input_keys`` is passed explicitly -- documented rather than
silently mis-mapped.

What round-trips losslessly, and what doesn't
-----------------------------------------------

Haystack → EvalPort → Haystack (via this adapter both ways): lossless. Every
input column's value for every row survives exactly, including columns that
are neither ``input_keys`` nor ``expected_key``.

Haystack → EvalPort → some other tool: the flattened ``f"{key}: {value}"``
strings and the single ``expected_key`` value are readable by any EvalPort
consumer, but a different tool has no way to know which Haystack column was
the actual pipeline input versus retrieved context versus free-form
metadata -- the same tradeoff every adapter in this ecosystem takes for
framework-specific structure with no native EvalPort field.

Grader type inference
------------------------

Unlike a DSPy metric (an arbitrary Python callable with no portable
representation), several Haystack evaluators *are* identifiable by name and
map cleanly onto one of EvalPort's standard grader types with zero
fabricated parameters:

- ``"answer_exact_match"`` (``AnswerExactMatchEvaluator``'s metric name) ->
  ``"exact_match"`` -- this grader type has no required ``params`` per the
  EvalPort spec, so the mapping is exact and nothing is invented.

Every other evaluator name (``"faithfulness"``, ``"context_relevance"``,
``"sas_evaluator"``, ``"document_map"``, ``"document_mrr"``,
``"document_ndcg"``, ``"document_recall"``, ``"llm_evaluator"``, or any
custom/unrecognized name) maps to EvalPort's ``"custom"`` grader type with
``params.handler`` set to the metric name. This is a deliberate choice, not
a limitation worth silently working around: EvalPort's ``"llm_judge"`` type
*requires* ``params.model`` and ``params.prompt`` (with a template token),
and ``"semantic_similarity"`` *requires* ``params.threshold`` -- none of
which this module has an honest value for without the caller's actual
Haystack evaluator configuration (a live ``chat_generator``, an actual
prompt template, a chosen similarity threshold). Fabricating placeholder
values for required grader params would produce a suite that *looks*
correctly typed but silently misrepresents what will actually run --
exactly the kind of workaround this ecosystem's adapters avoid. Pass an
explicit ``Grader``-shaped dict in ``graders`` (see ``to_openeval``) instead
if you want one of these typed correctly with your real configuration.

Why ``from_openeval()`` always adds an ``"id"`` column
-----------------------------------------------------------

A ``dspy.Example`` can carry this adapter's round-trip bookkeeping as a
hidden instance attribute (see ``dspy-openeval-adapter``), invisible to
DSPy's own field-reading methods. A plain ``dict[str, list[Any]]`` has no
such hiding place -- any key is a real, visible column that would be handed
straight to whatever evaluator the caller runs. So instead of hiding the id,
``from_openeval()`` surfaces it plainly: every returned columns dict
includes an ``"id"`` column (each row's ``TestCase.id``), *unless* the row's
own original columns already used the name ``"id"`` for real data, in which
case that data is left untouched rather than silently overwritten. This is
what lets ``evaluation_result_to_openeval()`` recover each row's test case
id automatically (it looks for an ``"id"`` column by default) after the
columns have been round-tripped through a real Haystack evaluator run,
without the caller having to wire anything extra through by hand.

Score normalization for ``evaluation_result_to_openeval()``
--------------------------------------------------------------

Every Haystack evaluator's ``individual_scores`` entries are already plain
numbers (``AnswerExactMatchEvaluator`` and ``DocumentMAPEvaluator`` and
friends emit ``0``/``1`` ints; ``FaithfulnessEvaluator``, ``SASEvaluator``,
``ContextRelevanceEvaluator`` emit floats), so there is no wrapper object to
unwrap (contrast with DSPy's ``bool``/``dspy.Prediction``/arbitrary-numeric
metric return shapes). Each value is clamped into EvalPort's required
``[0, 1]`` range; the *unclamped* raw value is preserved in
``grader_result.metadata.haystack.raw_score`` whenever clamping changed it,
so nothing is silently rewritten without a trace. ``passed`` is
``normalized_score >= pass_threshold`` (default ``0.5``, overridable). The
metric's own aggregate ``score`` (the same value for every row of that
metric -- an artifact of Haystack's columnar shape, not a per-row
computation) is additionally preserved in
``grader_result.metadata.haystack.aggregate_score`` for context.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import haystack  # noqa: F401  (import-time dependency check only)
except ImportError as e:  # pragma: no cover - exercised by the packaging itself
    raise ImportError(
        "haystack-openeval-adapter requires the 'haystack-ai' package. "
        "Install it with: pip install haystack-ai"
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


# Haystack evaluator metric names (the keys EvaluationRunResult.results and
# the *Evaluator components' `run()` methods return / expect) that map onto
# an EvalPort standard grader type with zero fabricated required params.
# See the module docstring's "Grader type inference" section for why every
# other name falls back to "custom" instead of being force-typed.
_EXACT_MATCH_METRIC_NAMES = {"answer_exact_match"}


def _infer_grader_type(name: str) -> str:
    """Map a Haystack evaluator metric name to an EvalPort grader type.

    Only ``"exact_match"`` is inferred automatically (it has no required
    ``params``, so the mapping can't misrepresent anything). Everything else
    -- including real Haystack evaluators like ``"faithfulness"`` or
    ``"sas_evaluator"`` that conceptually map to ``"llm_judge"``/
    ``"semantic_similarity"`` -- falls back to ``"custom"`` rather than
    guessing at required params (``model``, ``prompt``, ``threshold``) this
    module has no honest value for. See the module docstring.
    """
    return "exact_match" if name in _EXACT_MATCH_METRIC_NAMES else "custom"


def _build_grader_dict(name: str) -> Dict[str, Any]:
    grader_type = _infer_grader_type(name)
    grader: Dict[str, Any] = {"id": name, "type": grader_type}
    if grader_type == "custom":
        grader["params"] = {"handler": name}
        grader["description"] = (
            f"Placeholder for the Haystack '{name}' evaluator (e.g. a "
            "*Evaluator component's .run(), or any custom metric function) "
            "-- the caller must run the actual evaluator and supply its "
            "score via evaluation_result_to_openeval(), rather than this "
            "module fabricating a fake implementation."
        )
    else:
        grader["description"] = (
            f"Haystack's '{name}' evaluator, inferred as EvalPort's "
            f"'{grader_type}' grader type (no additional params required)."
        )
    return grader


def _get(obj: Any, name: str) -> Any:
    """Read an attribute/key from either a real EvaluationRunResult or a
    plain dict shaped the same way (``{"run_name": ..., "inputs": ...,
    "results": ...}``) -- duck-typed so callers aren't forced to construct
    a real Haystack object just to hand this function pre-computed data."""
    if isinstance(obj, dict):
        if name not in obj:
            raise KeyError(
                f"evaluation_result_to_openeval: missing required key {name!r}."
            )
        return obj[name]
    return getattr(obj, name)


def to_openeval(
    inputs: Dict[str, Sequence[Any]],
    input_keys: Sequence[str],
    expected_key: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    graders: Optional[Sequence[str]] = None,
    version: str = OPENEVAL_VERSION,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an EvalPort suite from a Haystack-shaped input-columns dict.

    Args:
        inputs: A ``dict[str, list[Any]]`` of named columns, all the same
            length -- exactly the shape ``haystack.evaluation
            .EvaluationRunResult(inputs=...)`` and every ``*Evaluator``
            component's ``run()`` consume/produce (e.g.
            ``{"questions": [...], "contexts": [...]}``).
        input_keys: Which column(s) are the model's input, in the order
            they should appear in ``TestCase.input`` (an array of
            ``f"{key}: {value}"`` strings, one per key -- see the module
            docstring for why this isn't force-collapsed into one string).
            Must be non-empty and every key must exist in ``inputs``.
        expected_key: Optional column name to use as ``expected_output``
            (e.g. ``"ground_truth_answers"``). Omitted from the suite when
            ``None`` or when it isn't a key in ``inputs``.
        ids: Optional explicit test case ids; auto-generated
            (``haystack_tc_<n>``) if omitted.
        suite_id: EvalPort ``Suite.id``; defaults to ``"haystack_suite"``.
        graders: Names of the metric(s)/evaluator(s) this suite's test cases
            reference (e.g. ``["answer_exact_match", "faithfulness"]`` --
            matching the metric names ``evaluation_result_to_openeval()``
            will later see as ``EvaluationRunResult.results`` keys). Each
            name becomes one EvalPort grader, typed via ``_infer_grader_type``
            (see the module docstring). Defaults to a single placeholder
            ``"haystack_metric"`` grader when omitted, documenting that no
            specific evaluator has been chosen yet.
        version, description: EvalPort Suite-level fields.

    Returns:
        A dict matching EvalPort's Suite schema
        (validate with ``openeval.validate.validate_suite``).

    Raises:
        ValueError: if ``inputs``/``input_keys`` is empty, the columns in
            ``inputs`` have mismatched lengths, or ``ids`` has a mismatched
            length.
    """
    if not inputs:
        raise ValueError("to_openeval: inputs is empty -- nothing to convert.")
    if not input_keys:
        raise ValueError(
            "to_openeval: input_keys is empty -- specify which input "
            "column(s) are the model input."
        )
    missing_keys = [k for k in input_keys if k not in inputs]
    if missing_keys:
        raise ValueError(
            f"to_openeval: input_keys {missing_keys} not present in inputs "
            f"(has columns: {sorted(inputs.keys())})."
        )

    lengths = {len(v) for v in inputs.values()}
    if len(lengths) != 1:
        raise ValueError(
            "to_openeval: all columns in inputs must have the same length, "
            f"got lengths {sorted(lengths)} across columns "
            f"{sorted(inputs.keys())}."
        )
    num_rows = next(iter(lengths))
    if num_rows == 0:
        raise ValueError("to_openeval: inputs columns are empty -- nothing to convert.")

    if ids is not None and len(ids) != num_rows:
        raise ValueError(
            f"to_openeval: ids has length {len(ids)}, expected {num_rows} "
            "(one entry per row)."
        )

    grader_names = list(graders) if graders else ["haystack_metric"]
    grader_dicts = [_build_grader_dict(name) for name in grader_names]

    test_cases: List[Dict[str, Any]] = []
    for i in range(num_rows):
        row = {k: v[i] for k, v in inputs.items()}
        tc_id = ids[i] if ids else f"haystack_tc_{i}"
        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": [f"{k}: {row[k]}" for k in input_keys],
            "graders": list(grader_names),
            "metadata": {"haystack": {"columns": row, "input_keys": list(input_keys)}},
        }
        if expected_key is not None and expected_key in row:
            test_case["expected_output"] = str(row[expected_key])
            test_case["metadata"]["haystack"]["expected_key"] = expected_key
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or "haystack_suite",
        "graders": grader_dicts,
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(
    suite: Dict[str, Any],
    input_keys: Optional[Sequence[str]] = None,
    expected_key: str = "expected_output",
) -> Dict[str, List[Any]]:
    """Convert an EvalPort suite back into a Haystack-shaped input-columns
    dict (``dict[str, list[Any]]``).

    For a test case carrying this adapter's own
    ``metadata.haystack.columns`` (i.e. one this adapter itself exported via
    ``to_openeval()``), the *exact* original columns are restored -- a
    lossless Haystack → EvalPort → Haystack round trip.

    For any other test case (hand-authored, or produced by a different
    EvalPort-speaking tool), columns are reconstructed positionally: each
    entry of the array-form ``input`` becomes its own column, named
    ``input_1``, ``input_2``, ... unless ``input_keys`` is given explicitly
    (in which case those names are used, one per input entry -- length must
    match). A string-form ``input`` becomes a single ``input_1`` column.
    ``expected_output``, when present, becomes a column named by
    ``expected_key``.

    The returned dict also always includes an ``"id"`` column (each row's
    ``TestCase.id``) unless a column already legitimately named ``"id"``
    exists in the row's own data -- see the module docstring's "Why
    from_openeval() always adds an 'id' column" section. This is what lets
    ``evaluation_result_to_openeval()`` recover each row's test case id
    automatically after the columns pass through a real evaluator run.

    Every test case in the suite must resolve to the same set of column
    names (true by construction for a suite this adapter produced; for a
    foreign suite, pass ``input_keys`` explicitly to guarantee it) -- this
    is a hard requirement of the columnar ``dict[str, list]`` shape itself,
    not an artifact of this adapter.

    Args:
        suite: An EvalPort suite dict.
        input_keys: Column names to use for a non-``haystack``-sourced test
            case's array-form input entries, positionally. Ignored for test
            cases carrying this adapter's own ``metadata.haystack.columns``
            (those always restore their real original column names).
        expected_key: Column name for a non-``haystack``-sourced test
            case's ``expected_output``, when present.

    Returns:
        A ``dict[str, list[Any]]`` -- one list per column, each the same
        length as ``suite["test_cases"]``.

    Raises:
        ValueError: if the suite has no test cases, or test cases resolve
            to inconsistent column names (making a rectangular
            ``dict[str, list]`` impossible to build).
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")

    rows: List[Dict[str, Any]] = []
    for tc in test_cases:
        haystack_meta = ((tc.get("metadata") or {}).get("haystack")) or {}

        if haystack_meta.get("columns") is not None:
            # Lossless path: this adapter's own export.
            row = dict(haystack_meta["columns"])
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

            row = dict(zip(names, entries))
            if tc.get("expected_output") is not None:
                row[expected_key] = tc["expected_output"]

        rows.append(row)

    # Surface each row's TestCase.id as a real "id" column -- see the module
    # docstring's "Why from_openeval() always adds an 'id' column" section.
    # Only when the row's own columns don't already claim that name, so
    # genuine caller data is never silently overwritten.
    if "id" not in rows[0]:
        for row, tc in zip(rows, test_cases):
            row["id"] = tc.get("id")

    column_names = set(rows[0].keys())
    for i, row in enumerate(rows):
        if set(row.keys()) != column_names:
            raise ValueError(
                "from_openeval: test cases resolve to inconsistent column "
                f"names -- row 0 has {sorted(column_names)}, row {i} has "
                f"{sorted(row.keys())}. A rectangular Haystack inputs dict "
                "requires every row to have the same columns; pass "
                "input_keys explicitly to normalize a foreign suite."
            )

    return {col: [row[col] for row in rows] for col in column_names}


def _normalize_score(value: Any, pass_threshold: float) -> Tuple[float, bool, Optional[float]]:
    """Return (normalized_score, passed, raw_score_if_clamped)."""
    raw = float(value)
    clamped = max(0.0, min(1.0, raw))
    return (clamped, clamped >= pass_threshold, raw if raw != clamped else None)


def evaluation_result_to_openeval(
    run_result: Any,
    suite_id: str = "haystack_suite",
    id_column: Optional[str] = None,
    output_column: Optional[str] = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    pass_threshold: float = 0.5,
    version: str = OPENEVAL_VERSION,
) -> Dict[str, Any]:
    """Convert a Haystack evaluation result into an EvalPort ResultSet.

    Args:
        run_result: A ``haystack.evaluation.EvaluationRunResult``, or any
            duck-typed object/dict exposing the same ``inputs`` (a
            ``dict[str, list[Any]]``) and ``results`` (a
            ``dict[str, {"score": float, "individual_scores": list}]``)
            shape.
        suite_id: The EvalPort suite this ResultSet's ``test_case_id``s
            refer to.
        id_column: Name of an ``inputs`` column to use as each test case's
            id. Falls back to an ``"id"`` column if present, else generates
            ``haystack_tc_<n>``.
        output_column: Name of an ``inputs`` column to use as each result's
            ``actual_output``. Falls back to the first of
            ``"predicted_answers"``, ``"replies"``, ``"responses"``,
            ``"answers"`` present in ``inputs``, else omitted.
        run_id: EvalPort ``ResultSet.run_id``; a random one is generated if
            omitted.
        started_at, completed_at: ISO-8601 timestamps. ``started_at``
            defaults to now if omitted (required by the EvalPort schema;
            ``EvaluationRunResult`` doesn't expose a run-level start time
            itself).
        pass_threshold: A (test case, metric) pair passes when its
            normalized score is ``>= pass_threshold``. A test case's overall
            ``passed`` is the AND of every metric's ``passed`` for that row.
        version: EvalPort schema version.

    Returns:
        A dict matching EvalPort's ResultSet schema
        (validate with ``openeval.validate.validate_result_set``).

    Raises:
        ValueError: if ``run_result.inputs`` is empty, a metric's
            ``individual_scores`` length doesn't match the number of rows,
            or there are no metrics in ``run_result.results``.
    """
    inputs = dict(_get(run_result, "inputs"))
    results = dict(_get(run_result, "results"))

    if not inputs:
        raise ValueError("evaluation_result_to_openeval: run_result.inputs is empty.")
    if not results:
        raise ValueError(
            "evaluation_result_to_openeval: run_result.results has no "
            "metrics to convert."
        )

    lengths = {len(v) for v in inputs.values()}
    if len(lengths) != 1:
        raise ValueError(
            "evaluation_result_to_openeval: run_result.inputs columns have "
            f"mismatched lengths: {sorted(lengths)}."
        )
    num_rows = next(iter(lengths))

    for metric_name, metric_data in results.items():
        individual = metric_data.get("individual_scores")
        if individual is None or len(individual) != num_rows:
            raise ValueError(
                f"evaluation_result_to_openeval: metric {metric_name!r} has "
                f"{len(individual) if individual is not None else 0} "
                f"individual_scores, expected {num_rows} (one per row)."
            )

    if id_column is None and "id" in inputs:
        id_column = "id"

    if output_column is None:
        for candidate in ("predicted_answers", "replies", "responses", "answers"):
            if candidate in inputs:
                output_column = candidate
                break

    results_out: List[Dict[str, Any]] = []
    for i in range(num_rows):
        tc_id = str(inputs[id_column][i]) if id_column else f"haystack_tc_{i}"

        grader_results: List[Dict[str, Any]] = []
        for metric_name, metric_data in results.items():
            normalized_score, passed, raw_score = _normalize_score(
                metric_data["individual_scores"][i], pass_threshold
            )
            grader_result: Dict[str, Any] = {
                "grader_id": metric_name,
                "type": _infer_grader_type(metric_name),
                "score": normalized_score,
                "passed": passed,
                "metadata": {"haystack": {"aggregate_score": metric_data.get("score")}},
            }
            if raw_score is not None:
                grader_result["metadata"]["haystack"]["raw_score"] = raw_score
            grader_results.append(grader_result)

        row_passed = all(gr["passed"] for gr in grader_results)
        result_entry: Dict[str, Any] = {
            "test_case_id": tc_id,
            "grader_results": grader_results,
            "passed": row_passed,
        }
        if output_column is not None:
            result_entry["actual_output"] = str(inputs[output_column][i])

        results_out.append(result_entry)

    total = len(results_out)
    passed_count = sum(1 for r in results_out if r["passed"])

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"haystack_run_{uuid.uuid4().hex[:12]}",
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
