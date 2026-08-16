"""Convert between Evidently (https://github.com/evidentlyai/evidently)
evaluation DataFrames/Datasets and EvalPort
(https://github.com/adhabnr-ux/evalport) suites and result sets.

EvalPort is an open interchange format (Apache 2.0) for portable LLM
evaluation datasets: test cases, graders, suites, and results as plain JSON,
shared across evaluation tools (DeepEval, Promptfoo, Inspect AI, AutoGen,
CrewAI, Ragas, LangSmith, Braintrust, MLflow, Opik, Arize Phoenix, Weights &
Biases Weave, UpTrain, Langfuse, Giskard, LlamaIndex, Patronus AI, Vertex AI,
DSPy, Haystack, and now Evidently).

Evidently's native evaluation surface is a ``pandas.DataFrame``: a row per
test case, arbitrary named columns, and an ``evidently.Dataset`` built from
that frame plus a list of *descriptors* (``evidently.descriptors.*`` --
``ExactMatch``, ``Contains``, ``TextLength``, ``SemanticSimilarity``,
``LLMJudge``, dozens more), each of which appends one new column of
per-row scores. This module has three entry points that mirror that shape
directly:

    to_openeval(df, input_columns, expected_column=None, ...)
        Converts a ``pandas.DataFrame`` of test-case rows into an EvalPort
        suite (test cases only; no results yet).

    from_openeval(suite, input_columns=None, expected_column="expected_output")
        Converts an EvalPort suite back into a ``pandas.DataFrame``, ready to
        hand straight to ``evidently.Dataset.from_pandas(df, descriptors=...)``.

    evaluation_result_to_openeval(evaluated, descriptor_columns, suite_id, ...)
        Converts an evaluated ``evidently.Dataset`` (or any object/DataFrame
        exposing an ``.as_dataframe()``-equivalent shape) into an EvalPort
        ResultSet -- one grader result per named descriptor column, per row.

Why field mapping needs a decision, honestly
----------------------------------------------

A DataFrame's columns are whatever the caller's evaluation needs
(``question``/``context``/``answer``, anything). EvalPort's ``TestCase.input``
is a single string-or-array-of-strings concept, not a named-column table.
This module does not guess which column is "the" input: the caller names
them explicitly via ``input_columns`` (and, optionally, ``expected_column``
for the one column that should become ``expected_output``).

Multiple ``input_columns`` are flattened into EvalPort's array-of-strings
input form as ``f"{column}: {value}"`` per column, the same "one string per
named field" idiom this ecosystem already uses for other named-field
sources (see ``dspy-openeval-adapter`` and ``haystack-openeval-adapter``).
On a round trip through *this* adapter specifically, nothing is lost: every
column's value for that row is additionally preserved verbatim under
``test_case.metadata.evidently.columns``, so ``from_openeval()``
reconstructs the exact original row rather than re-deriving it from the
flattened strings. A suite built by a *different* EvalPort-speaking tool (no
``metadata.evidently.columns``) instead gets one column per array entry,
named positionally (``input_1``, ``input_2``, ...) unless ``input_columns``
is passed explicitly -- documented rather than silently mis-mapped.

Why a descriptor's EvalPort grader type usually can't be inferred, unlike Haystack
--------------------------------------------------------------------------------------

``haystack-openeval-adapter`` can infer a grader type from
``AnswerExactMatchEvaluator``'s fixed, framework-defined output column name
(``"answer_exact_match"``). Evidently's equivalent -- a descriptor's output
*column* -- has no fixed name: it's whatever ``alias`` the caller passed to
``ExactMatch(..., alias=...)``, ``Contains(..., alias=...)``, etc. This
module therefore cannot pattern-match a column name to a descriptor type the
way the Haystack adapter matches a metric name. Instead, both
``to_openeval()`` and ``evaluation_result_to_openeval()`` accept an optional
``descriptor_types`` mapping (``{column_alias: EvidentlyDescriptorClassName}``,
e.g. ``{"exact_match": "ExactMatch"}``) that the *caller* supplies, since only
the caller actually knows which descriptor class produced which column. Only
``"ExactMatch"`` is mapped to EvalPort's ``"exact_match"`` grader type (zero
required ``params``, so nothing is fabricated); every other class name (or no
entry at all) falls back to ``"custom"`` -- the same reasoning
``haystack-openeval-adapter`` documents for why ``"llm_judge"``'s required
``model``/``prompt`` and ``"semantic_similarity"``'s required ``threshold``
are never guessed at.

Why non-boolean, non-numeric descriptor values get ``score: null``
-----------------------------------------------------------------------

Descriptor output columns are not uniformly typed. ``ExactMatch``/
``Contains``/``BeginsWith`` and friends produce ``bool``; ``TextLength``/
``WordCount`` produce an unbounded ``int``; ``Sentiment`` produces a
``float`` in ``[-1, 1]``; but classification-style descriptors
(``MulticlassClassificationLLMEval`` and similar) produce an arbitrary
*label string* with no inherent numeric value. EvalPort's
``GraderResult.score`` is `Optional[float]` specifically for this case:
a non-boolean, non-numeric value becomes ``score: null`` with the raw label
preserved verbatim in ``grader_result.reason``, rather than this module
inventing a fake numeric encoding for an unordered category. ``passed`` for
that row is then determined by an optional caller-supplied ``pass_values``
allowlist (``{column: {"acceptable_label_1", ...}}``); without one, a
non-numeric result defaults to ``passed=False`` -- an unclassified label is
not silently treated as a pass.

Score normalization for numeric/boolean descriptor values
--------------------------------------------------------------

``bool`` (including numpy's ``bool_``) maps directly: ``score`` 1.0/0.0,
``passed`` the bool itself. Any other numeric value (``int``, ``float``, and
their numpy equivalents) is clamped into EvalPort's required ``[0, 1]``
range; the *unclamped* raw value is preserved in
``grader_result.metadata.evidently.raw_score`` whenever clamping changed it
(e.g. ``TextLength``'s raw character count), so nothing is silently
rewritten without a trace. ``passed`` is ``normalized_score >= pass_threshold``
(default ``0.5``, overridable) for numeric values.

Why ``from_openeval()`` always adds an ``"id"`` column
-----------------------------------------------------------

A ``pandas.DataFrame`` has no hidden slot to carry this adapter's round-trip
bookkeeping invisibly (the same constraint ``haystack-openeval-adapter``
documents for its plain ``dict[str, list]`` shape, versus
``dspy-openeval-adapter``'s ability to hide it on a ``dspy.Example`` instance
attribute). So ``from_openeval()`` surfaces it plainly: the returned
DataFrame always includes an ``"id"`` column (each row's ``TestCase.id``),
*unless* the row's own original columns already used that name for real
data, in which case that data is left untouched. This is what lets
``evaluation_result_to_openeval()`` recover each row's test case id
automatically after the DataFrame has passed through
``evidently.Dataset.from_pandas()`` and back out via ``.as_dataframe()``.

What round-trips losslessly, and what doesn't
-----------------------------------------------

Evidently → EvalPort → Evidently (via this adapter both ways): lossless.
Every column's value for every row survives exactly, restored from
``metadata.evidently.columns``.

Evidently → EvalPort → some other tool: the flattened ``f"{column}: {value}"``
strings and the single ``expected_output`` value are readable by any
EvalPort consumer, but a different tool has no way to know which column was
the actual model input versus retrieved context versus free-form metadata --
the same tradeoff every adapter in this ecosystem takes for framework-specific
structure with no native EvalPort field.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover - exercised by the packaging itself
    raise ImportError(
        "evidently-openeval-adapter requires the 'pandas' package. "
        "Install it with: pip install pandas"
    ) from e

try:
    import evidently  # noqa: F401  (import-time dependency check only)
except ImportError as e:  # pragma: no cover - exercised by the packaging itself
    raise ImportError(
        "evidently-openeval-adapter requires the 'evidently' package. "
        "Install it with: pip install evidently"
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


# Evidently descriptor class names (as passed via `descriptor_types`) that
# map onto an EvalPort standard grader type with zero fabricated required
# params. See the module docstring's "Why a descriptor's EvalPort grader
# type usually can't be inferred" section for why every other class name
# falls back to "custom" instead of being force-typed.
_EXACT_MATCH_DESCRIPTOR_CLASSES = {"ExactMatch"}


def _infer_grader_type(descriptor_class: Optional[str]) -> str:
    """Map an Evidently descriptor class name to an EvalPort grader type.

    Only ``"ExactMatch"`` is mapped automatically (EvalPort's
    ``"exact_match"`` type has no required ``params``, so the mapping can't
    misrepresent anything). Every other class name -- or no entry at all --
    falls back to ``"custom"`` rather than guessing at required params
    (``model``, ``prompt``, ``threshold``, ``pattern``, ``substring``) this
    module has no honest value for. See the module docstring.
    """
    return "exact_match" if descriptor_class in _EXACT_MATCH_DESCRIPTOR_CLASSES else "custom"


def _build_grader_dict(alias: str, descriptor_class: Optional[str]) -> Dict[str, Any]:
    grader_type = _infer_grader_type(descriptor_class)
    grader: Dict[str, Any] = {"id": alias, "type": grader_type}
    if grader_type == "custom":
        grader["params"] = {"handler": alias}
        grader["description"] = (
            f"Placeholder for the Evidently descriptor aliased '{alias}'"
            + (f" ({descriptor_class})" if descriptor_class else "")
            + " -- the caller must run the actual evidently.Dataset "
            "evaluation and supply its column values via "
            "evaluation_result_to_openeval(), rather than this module "
            "fabricating a fake implementation."
        )
    else:
        grader["description"] = (
            f"Evidently's '{descriptor_class}' descriptor (aliased '{alias}'), "
            f"inferred as EvalPort's '{grader_type}' grader type (no "
            "additional params required)."
        )
    return grader


def _to_native(value: Any) -> Any:
    """Convert a pandas/numpy scalar to a plain JSON-serializable Python
    value. EvalPort suites/result sets must be plain JSON; a raw
    ``numpy.int64``/``numpy.bool_`` surviving into ``metadata`` would break
    that guarantee for a consumer that actually calls ``json.dumps()``."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # pd.isna chokes on some container types; not a scalar to null out
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def to_openeval(
    df: "pd.DataFrame",
    input_columns: Sequence[str],
    expected_column: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    graders: Optional[Sequence[str]] = None,
    descriptor_types: Optional[Dict[str, str]] = None,
    version: str = OPENEVAL_VERSION,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an EvalPort suite from an Evidently-shaped test-case DataFrame.

    Args:
        df: A ``pandas.DataFrame``, one row per test case -- exactly what
            ``evidently.Dataset.from_pandas(df, ...)`` consumes.
        input_columns: Which column(s) are the model's input, in the order
            they should appear in ``TestCase.input`` (an array of
            ``f"{column}: {value}"`` strings, one per column -- see the
            module docstring for why this isn't force-collapsed into one
            string). Must be non-empty and every column must exist in
            ``df``.
        expected_column: Optional column name to use as ``expected_output``
            (e.g. ``"expected"``). Omitted from the suite when ``None`` or
            when it isn't a column in ``df``.
        ids: Optional explicit test case ids; auto-generated
            (``evidently_tc_<n>``) if omitted.
        suite_id: EvalPort ``Suite.id``; defaults to ``"evidently_suite"``.
        graders: Aliases of the descriptor(s) this suite's test cases
            reference (e.g. ``["exact_match", "answer_length"]`` -- matching
            the ``alias=...`` you'll later pass to each
            ``evidently.descriptors.*`` object). Each alias becomes one
            EvalPort grader, typed via ``descriptor_types`` (see below).
            Defaults to a single placeholder ``"evidently_descriptor"``
            grader when omitted, documenting that no specific descriptor has
            been chosen yet.
        descriptor_types: Optional ``{alias: EvidentlyDescriptorClassName}``
            mapping (e.g. ``{"exact_match": "ExactMatch"}``) used to type
            each grader in ``graders``. See the module docstring's "Why a
            descriptor's EvalPort grader type usually can't be inferred"
            section for why only ``"ExactMatch"`` is mapped to anything
            other than ``"custom"``.
        version, description: EvalPort Suite-level fields.

    Returns:
        A dict matching EvalPort's Suite schema
        (validate with ``openeval.validate.validate_suite``).

    Raises:
        ValueError: if ``df``/``input_columns`` is empty, ``input_columns``
            references a column not in ``df``, or ``ids`` has a mismatched
            length.
    """
    if df is None or len(df) == 0:
        raise ValueError("to_openeval: df is empty -- nothing to convert.")
    if not input_columns:
        raise ValueError(
            "to_openeval: input_columns is empty -- specify which column(s) "
            "are the model input."
        )
    missing_columns = [c for c in input_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"to_openeval: input_columns {missing_columns} not present in df "
            f"(has columns: {sorted(df.columns)})."
        )

    num_rows = len(df)
    if ids is not None and len(ids) != num_rows:
        raise ValueError(
            f"to_openeval: ids has length {len(ids)}, expected {num_rows} "
            "(one entry per row)."
        )

    descriptor_types = descriptor_types or {}
    grader_names = list(graders) if graders else ["evidently_descriptor"]
    grader_dicts = [
        _build_grader_dict(name, descriptor_types.get(name)) for name in grader_names
    ]

    rows = df.to_dict(orient="records")
    test_cases: List[Dict[str, Any]] = []
    for i, raw_row in enumerate(rows):
        row = {k: _to_native(v) for k, v in raw_row.items()}
        tc_id = ids[i] if ids else f"evidently_tc_{i}"
        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": [f"{c}: {row[c]}" for c in input_columns],
            "graders": list(grader_names),
            "metadata": {"evidently": {"columns": row, "input_columns": list(input_columns)}},
        }
        if expected_column is not None and expected_column in row and row[expected_column] is not None:
            test_case["expected_output"] = str(row[expected_column])
            test_case["metadata"]["evidently"]["expected_column"] = expected_column
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or "evidently_suite",
        "graders": grader_dicts,
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(
    suite: Dict[str, Any],
    input_columns: Optional[Sequence[str]] = None,
    expected_column: str = "expected_output",
) -> "pd.DataFrame":
    """Convert an EvalPort suite back into an Evidently-shaped test-case
    ``pandas.DataFrame``.

    For a test case carrying this adapter's own
    ``metadata.evidently.columns`` (i.e. one this adapter itself exported
    via ``to_openeval()``), the *exact* original row is restored -- a
    lossless Evidently → EvalPort → Evidently round trip.

    For any other test case (hand-authored, or produced by a different
    EvalPort-speaking tool), columns are reconstructed positionally: each
    entry of the array-form ``input`` becomes its own column, named
    ``input_1``, ``input_2``, ... unless ``input_columns`` is given
    explicitly (in which case those names are used, one per input entry --
    length must match). A string-form ``input`` becomes a single ``input_1``
    column. ``expected_output``, when present, becomes a column named by
    ``expected_column``.

    The returned frame also always includes an ``"id"`` column (each row's
    ``TestCase.id``) unless a column already legitimately named ``"id"``
    exists in the row's own data -- see the module docstring's "Why
    from_openeval() always adds an 'id' column" section. This is what lets
    ``evaluation_result_to_openeval()`` recover each row's test case id
    automatically after the frame passes through
    ``evidently.Dataset.from_pandas()``.

    Every test case in the suite must resolve to the same set of column
    names (true by construction for a suite this adapter produced; for a
    foreign suite, pass ``input_columns`` explicitly to guarantee it).

    Args:
        suite: An EvalPort suite dict.
        input_columns: Column names to use for a non-``evidently``-sourced
            test case's array-form input entries, positionally. Ignored for
            test cases carrying this adapter's own
            ``metadata.evidently.columns`` (those always restore their real
            original column names).
        expected_column: Column name for a non-``evidently``-sourced test
            case's ``expected_output``, when present.

    Returns:
        A ``pandas.DataFrame`` with one row per test case, in suite order.

    Raises:
        ValueError: if the suite has no test cases, or test cases resolve
            to inconsistent column names (making a rectangular DataFrame
            impossible to build cleanly).
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")

    rows: List[Dict[str, Any]] = []
    for tc in test_cases:
        evidently_meta = ((tc.get("metadata") or {}).get("evidently")) or {}

        if evidently_meta.get("columns") is not None:
            # Lossless path: this adapter's own export.
            row = dict(evidently_meta["columns"])
        else:
            # Best-effort path: hand-authored or foreign-tool suite.
            raw_input = tc.get("input")
            entries = raw_input if isinstance(raw_input, list) else [raw_input]
            if input_columns is not None:
                if len(input_columns) != len(entries):
                    raise ValueError(
                        f"from_openeval: test case {tc.get('id')!r} has "
                        f"{len(entries)} input entries but input_columns has "
                        f"{len(input_columns)} names."
                    )
                names = list(input_columns)
            else:
                names = [f"input_{j + 1}" for j in range(len(entries))]

            row = dict(zip(names, entries))
            if tc.get("expected_output") is not None:
                row[expected_column] = tc["expected_output"]

        rows.append(row)

    # Surface each row's TestCase.id as a real "id" column -- see the module
    # docstring's "Why from_openeval() always adds an 'id' column" section.
    if "id" not in rows[0]:
        for row, tc in zip(rows, test_cases):
            row["id"] = tc.get("id")

    column_names = set(rows[0].keys())
    for i, row in enumerate(rows):
        if set(row.keys()) != column_names:
            raise ValueError(
                "from_openeval: test cases resolve to inconsistent column "
                f"names -- row 0 has {sorted(column_names)}, row {i} has "
                f"{sorted(row.keys())}. A rectangular DataFrame requires "
                "every row to have the same columns; pass input_columns "
                "explicitly to normalize a foreign suite."
            )

    return pd.DataFrame(rows, columns=sorted(column_names))


def _normalize_numeric(value: Any, pass_threshold: float) -> Tuple[float, bool, Optional[float]]:
    """Return (normalized_score, passed, raw_score_if_clamped) for a
    numeric (non-boolean) descriptor value."""
    raw = float(value)
    clamped = max(0.0, min(1.0, raw))
    return (clamped, clamped >= pass_threshold, raw if raw != clamped else None)


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _is_numeric(value: Any) -> bool:
    if _is_bool_like(value):
        return False
    return isinstance(value, (int, float)) or type(value).__name__ in (
        "int64",
        "int32",
        "float64",
        "float32",
    )


def evaluation_result_to_openeval(
    evaluated: Any,
    descriptor_columns: Sequence[str],
    suite_id: str = "evidently_suite",
    descriptor_types: Optional[Dict[str, str]] = None,
    pass_values: Optional[Dict[str, Iterable[Any]]] = None,
    id_column: Optional[str] = None,
    output_column: Optional[str] = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    pass_threshold: float = 0.5,
    version: str = OPENEVAL_VERSION,
) -> Dict[str, Any]:
    """Convert an evaluated Evidently dataset into an EvalPort ResultSet.

    Args:
        evaluated: An ``evidently.Dataset`` (its ``.as_dataframe()`` is
            called automatically), or a plain ``pandas.DataFrame`` that
            already contains the descriptor result columns (e.g. one you
            built yourself, or the output of ``.as_dataframe()`` already
            called).
        descriptor_columns: REQUIRED -- which columns of the (as-)dataframe
            are descriptor/grader outputs to convert into grader results.
            There's no reliable way to distinguish a descriptor's output
            column from an original data column in an arbitrary DataFrame,
            so this module doesn't guess.
        suite_id: The EvalPort suite this ResultSet's ``test_case_id``s
            refer to.
        descriptor_types: Optional ``{column: EvidentlyDescriptorClassName}``
            mapping used to type each grader result (see the module
            docstring's "Why a descriptor's EvalPort grader type usually
            can't be inferred" section).
        pass_values: Optional ``{column: set_of_passing_values}`` used to
            decide ``passed`` for a non-boolean, non-numeric descriptor
            value (e.g. a classification label). Without an entry for a
            given column, a non-numeric value defaults to ``passed=False``
            (see the module docstring's "Why non-boolean, non-numeric
            descriptor values get score: null" section).
        id_column: Name of a column to use as each test case's id. Falls
            back to an ``"id"`` column if present, else generates
            ``evidently_tc_<n>``.
        output_column: Name of a column to use as each result's
            ``actual_output``. Falls back to the first of ``"answer"``,
            ``"response"``, ``"output"``, ``"prediction"`` present, else
            omitted.
        run_id: EvalPort ``ResultSet.run_id``; a random one is generated if
            omitted.
        started_at, completed_at: ISO-8601 timestamps. ``started_at``
            defaults to now if omitted.
        pass_threshold: A (test case, descriptor) pair with a numeric score
            passes when its normalized score is ``>= pass_threshold``. A
            test case's overall ``passed`` is the AND of every descriptor's
            ``passed`` for that row.
        version: EvalPort schema version.

    Returns:
        A dict matching EvalPort's ResultSet schema
        (validate with ``openeval.validate.validate_result_set``).

    Raises:
        ValueError: if ``descriptor_columns`` is empty, the resolved
            DataFrame is empty, or a named column is missing.
    """
    if not descriptor_columns:
        raise ValueError(
            "evaluation_result_to_openeval: descriptor_columns is empty -- "
            "specify which column(s) are descriptor/grader outputs."
        )

    df = evaluated.as_dataframe() if hasattr(evaluated, "as_dataframe") else evaluated
    if df is None or len(df) == 0:
        raise ValueError("evaluation_result_to_openeval: evaluated data is empty.")

    missing = [c for c in descriptor_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"evaluation_result_to_openeval: descriptor_columns {missing} "
            f"not present (has columns: {sorted(df.columns)})."
        )

    descriptor_types = descriptor_types or {}
    pass_values = pass_values or {}

    if id_column is None and "id" in df.columns:
        id_column = "id"

    if output_column is None:
        for candidate in ("answer", "response", "output", "prediction"):
            if candidate in df.columns:
                output_column = candidate
                break

    rows = df.to_dict(orient="records")
    results_out: List[Dict[str, Any]] = []
    for i, raw_row in enumerate(rows):
        row = {k: _to_native(v) for k, v in raw_row.items()}
        tc_id = str(row[id_column]) if id_column else f"evidently_tc_{i}"

        grader_results: List[Dict[str, Any]] = []
        for col in descriptor_columns:
            value = row[col]
            grader_type = _infer_grader_type(descriptor_types.get(col))

            if value is None:
                score: Optional[float] = None
                passed = False
                reason: Optional[str] = None
                raw_score = None
            elif _is_bool_like(value):
                score = 1.0 if value else 0.0
                passed = bool(value)
                reason = None
                raw_score = None
            elif _is_numeric(value):
                score, passed, raw_score = _normalize_numeric(value, pass_threshold)
                reason = None
            else:
                score = None
                reason = str(value)
                allowed = pass_values.get(col)
                passed = value in allowed if allowed is not None else False
                raw_score = None

            grader_result: Dict[str, Any] = {
                "grader_id": col,
                "type": grader_type,
                "score": score,
                "passed": passed,
            }
            if reason is not None:
                grader_result["reason"] = reason
            if raw_score is not None:
                grader_result["metadata"] = {"evidently": {"raw_score": raw_score}}
            grader_results.append(grader_result)

        row_passed = all(gr["passed"] for gr in grader_results)
        result_entry: Dict[str, Any] = {
            "test_case_id": tc_id,
            "grader_results": grader_results,
            "passed": row_passed,
        }
        if output_column is not None and row.get(output_column) is not None:
            result_entry["actual_output"] = str(row[output_column])

        results_out.append(result_entry)

    total = len(results_out)
    passed_count = sum(1 for r in results_out if r["passed"])

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"evidently_run_{uuid.uuid4().hex[:12]}",
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
