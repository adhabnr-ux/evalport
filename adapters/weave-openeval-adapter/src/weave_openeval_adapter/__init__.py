"""Convert between Weights & Biases Weave datasets/evaluations and EvalPort.

EvalPort (https://github.com/adhabnr-ux/evalport) is an open interchange
format (Apache 2.0) for portable LLM evaluation datasets: test cases,
graders, suites, and results as plain JSON, shared across evaluation tools.

Weave (https://github.com/wandb/weave, ``pip install weave``) represents an
evaluation dataset as a ``weave.Dataset`` -- an object wrapping ``rows``, a
plain list of flat dicts with arbitrary, user-defined column names (e.g.
``{"question": "...", "expected": "..."}``). There is no fixed input/output
schema, unlike EvalPort's ``TestCase.input``/``expected_output``.

A completed ``Evaluation.evaluate(model)`` call scores each dataset row with
every configured scorer and produces, per row, a dict of this exact shape
(this is the real internal shape produced by
``weave.evaluation.eval.Evaluation.predict_and_score``, not a guess):

    {"output": <model output, or None on error>,
     "scores": {"<scorer_name>": <bool | float | dict | None>, ...},
     "model_latency": <float seconds>}

Three entry points, matching the shape used by every other adapter in the
EvalPort ecosystem:

    to_openeval(dataset, ...)
        A ``weave.Dataset`` (or any iterable of row dicts / ``weave.Dataset
        rows``) -> an EvalPort suite.

    from_openeval(suite, ...)
        An EvalPort suite's test cases -> a list of row dicts shaped for
        ``weave.Dataset(name=..., rows=...)``.

    evaluation_to_openeval(rows, eval_results, ...)
        The original dataset rows, paired positionally with the per-row
        ``predict_and_score``-shaped results Weave produces during
        ``Evaluation.evaluate()`` (e.g. collected via ``EvaluationLogger`` or
        by iterating ``EvaluationResults.rows``) -> an EvalPort ResultSet,
        one GraderResult per scorer.

Because Weave dataset rows are schema-less flat dicts, ``to_openeval()``
auto-detects which column is the model input and which is the expected
output using the same heuristic ``opik-openeval-adapter`` and
``phoenix-openeval-adapter`` use for their own schema-less formats: check
common key names, fall back to the row's only remaining key if exactly one
is left after removing the detected input key, and fall back to a JSON dump
of the whole row as a last resort so nothing is ever silently dropped. The
full raw row is always preserved under ``metadata["weave"]["row"]``
regardless of which keys were picked.

Weave scorer results can be a bare bool, a bare number, or a dict with
arbitrary keys (Weave's own convention favors a ``dict`` with fields such as
``"passed"``, but plenty of user scorers just return a float or bool
directly -- see ``weave.flow.scorer.auto_summarize``, which handles exactly
these three shapes when Weave itself summarizes evaluation results).
``evaluation_to_openeval()`` mirrors that same tri-way handling when turning
one scorer result into one EvalPort ``GraderResult``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from openeval.version import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "evaluation_to_openeval"]

DEFAULT_INPUT_KEYS = ("input", "question", "query", "prompt", "user_input", "text")
DEFAULT_OUTPUT_KEYS = (
    "expected_output",
    "expected",
    "output",
    "answer",
    "reference",
    "ground_truth",
    "label",
    "correction",
)

_AFFIRMATIVE_LABELS = {"true", "pass", "passed", "correct", "yes", "good", "1"}


def _stringify(value: Any) -> str:
    """Render an arbitrary value as the string EvalPort's schema requires."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


_ALWAYS_EXCLUDED_KEYS = ("id",)


def _extract_field(
    row: Mapping[str, Any],
    preferred_keys: Sequence[str],
    exclude: Optional[str] = None,
) -> Optional[tuple]:
    """Pick a ``(key, stringified_value)`` pair out of a flat row dict.

    Checks ``preferred_keys`` in order (case-sensitive first, then
    case-insensitive). If none match, and exactly one candidate key remains
    (after excluding ``exclude`` -- used to avoid picking the same column
    twice -- and the row's own ``id`` field, which is structural bookkeeping
    rather than eval content), that lone key/value is used. Returns ``None``
    if nothing usable is found -- callers fall back to a full JSON dump of
    the row. The returned key lets a caller exclude it from a second
    extraction over the same row (e.g. so the expected-output detector never
    re-picks the column already used for the input).
    """
    for key in preferred_keys:
        if key in row:
            return key, _stringify(row[key])
    lowered = {str(k).lower(): k for k in row.keys()}
    for key in preferred_keys:
        if key in lowered:
            actual_key = lowered[key]
            return actual_key, _stringify(row[actual_key])

    excluded = set(_ALWAYS_EXCLUDED_KEYS)
    if exclude is not None:
        excluded.add(exclude)
    remaining = [k for k in row.keys() if k not in excluded]
    if len(remaining) == 1:
        return remaining[0], _stringify(row[remaining[0]])
    return None


def to_openeval(
    dataset: Any,
    suite_id: str = "weave_dataset",
    input_key: Optional[str] = None,
    expected_output_key: Optional[str] = None,
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """Convert a Weave dataset (or any iterable of row dicts) to an EvalPort suite.

    ``dataset`` may be a ``weave.Dataset`` instance (iterated via its
    ``rows``), or any iterable of plain dicts -- e.g. the list you'd pass to
    ``weave.Dataset(rows=...)`` in the first place.

    ``input_key``/``expected_output_key`` override the auto-detected column
    names when the heuristic guesses wrong; every row's full original
    content is preserved under ``metadata["weave"]["row"]`` either way, so
    an override never loses data that was already exported.
    """
    rows = list(dataset)

    default_grader = (
        {
            "id": "default_exact_match",
            "type": "exact_match",
        }
        if grader_type == "exact_match"
        else {
            "id": "default_llm_judge",
            "type": "llm_judge",
            "params": {
                "model": "gpt-4o",
                "prompt": "Does the actual output '{output}' match the expected output '{expected}'? Answer yes or no.",
            },
        }
    )

    test_cases: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            row = dict(row)

        row_id = row.get("id")
        tc_id = _stringify(row_id) if row_id is not None else f"row_{idx}"

        detected_input_key: Optional[str] = None
        if input_key is not None and input_key in row:
            detected_input_key = input_key
            input_text = _stringify(row[input_key])
        else:
            found = _extract_field(row, DEFAULT_INPUT_KEYS)
            if found is not None:
                detected_input_key, input_text = found
            else:
                input_text = None
        if input_text is None:
            input_text = _stringify(row)

        if expected_output_key is not None and expected_output_key in row:
            output_text = _stringify(row[expected_output_key])
        else:
            found = _extract_field(row, DEFAULT_OUTPUT_KEYS, exclude=detected_input_key)
            output_text = found[1] if found is not None else None

        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": input_text,
            "graders": [default_grader["id"]],
            "metadata": {"weave": {"row": row}},
        }
        if output_text is not None:
            test_case["expected_output"] = output_text
        test_cases.append(test_case)

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": suite_id,
        "graders": [default_grader],
        "test_cases": test_cases,
    }


def from_openeval(suite: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite's test cases into Weave dataset rows.

    Returns a plain list of dicts shaped for ``weave.Dataset(name=...,
    rows=...)``. If a test case's ``metadata["weave"]["row"]`` is present
    (i.e. it round-trips a suite that came from ``to_openeval()`` above),
    that original row is reconstructed verbatim; otherwise a fresh row is
    built from the test case's ``input``/``expected_output``/``id``.
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        weave_meta = (tc.get("metadata") or {}).get("weave")
        if isinstance(weave_meta, Mapping) and "row" in weave_meta:
            row = dict(weave_meta["row"])
        else:
            input_value = tc.get("input")
            row = {
                "id": tc.get("id"),
                "input": input_value if isinstance(input_value, str) else json.dumps(input_value),
            }
            if "expected_output" in tc:
                row["expected"] = tc["expected_output"]
        rows.append(row)
    return rows


def _score_to_grader_result(grader_id: str, scorer_name: str, result: Any) -> Dict[str, Any]:
    """Turn one Weave scorer result (bool | number | dict | None) into a GraderResult.

    Mirrors the three shapes ``weave.flow.scorer.auto_summarize`` itself
    special-cases when Weave summarizes a completed evaluation: a bare
    bool, a bare number, or a dict of sub-scores. A dict result is searched
    for a conventional pass/score field (``passed``/``correct``/``score``);
    if none is found, the first boolean or numeric value inside it is used,
    and the full dict is always preserved under ``metadata``.
    """
    metadata: Dict[str, Any] = {"weave_scorer": scorer_name}

    if result is None:
        return {
            "grader_id": grader_id,
            "type": "custom",
            "score": None,
            "passed": False,
            "metadata": metadata,
        }

    if isinstance(result, bool):
        return {
            "grader_id": grader_id,
            "type": "custom",
            "score": 1.0 if result else 0.0,
            "passed": result,
            "metadata": metadata,
        }

    if isinstance(result, (int, float)):
        score = max(0.0, min(1.0, float(result)))
        return {
            "grader_id": grader_id,
            "type": "custom",
            "score": score,
            "passed": score >= 0.5,
            "metadata": metadata,
        }

    if isinstance(result, Mapping):
        metadata["raw"] = dict(result)
        for key in ("passed", "correct", "is_correct"):
            if key in result and isinstance(result[key], bool):
                passed = result[key]
                return {
                    "grader_id": grader_id,
                    "type": "custom",
                    "score": 1.0 if passed else 0.0,
                    "passed": passed,
                    "metadata": metadata,
                }
        for key in ("score", "value", "similarity", "relevance"):
            val = result.get(key)
            if isinstance(val, bool):
                return {
                    "grader_id": grader_id,
                    "type": "custom",
                    "score": 1.0 if val else 0.0,
                    "passed": val,
                    "metadata": metadata,
                }
            if isinstance(val, (int, float)):
                score = max(0.0, min(1.0, float(val)))
                return {
                    "grader_id": grader_id,
                    "type": "custom",
                    "score": score,
                    "passed": score >= 0.5,
                    "metadata": metadata,
                }
        # Nothing conventional found -- fall back to the first bool/number
        # value anywhere in the dict, in insertion order.
        for val in result.values():
            if isinstance(val, bool):
                return {
                    "grader_id": grader_id,
                    "type": "custom",
                    "score": 1.0 if val else 0.0,
                    "passed": val,
                    "metadata": metadata,
                }
            if isinstance(val, (int, float)):
                score = max(0.0, min(1.0, float(val)))
                return {
                    "grader_id": grader_id,
                    "type": "custom",
                    "score": score,
                    "passed": score >= 0.5,
                    "metadata": metadata,
                }
        # Dict had no interpretable value at all -- record it, but we can't
        # honestly claim pass/fail, so score is null and passed is False
        # (EvalPort requires `passed` to be a bool, never null).
        return {
            "grader_id": grader_id,
            "type": "custom",
            "score": None,
            "passed": False,
            "metadata": metadata,
        }

    # Anything else (e.g. a bare string like "yes"/"no")
    text = str(result).strip().lower()
    passed = text in _AFFIRMATIVE_LABELS
    return {
        "grader_id": grader_id,
        "type": "custom",
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "metadata": metadata,
    }


def evaluation_to_openeval(
    rows: Iterable[Mapping[str, Any]],
    eval_results: Iterable[Mapping[str, Any]],
    suite_id: str = "weave_dataset",
    run_id: str = "weave_run",
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a completed Weave evaluation into an EvalPort ResultSet.

    ``rows`` is the original dataset (same rows passed to ``to_openeval()``,
    so test case ids line up), and ``eval_results`` is the list of per-row
    dicts Weave's ``Evaluation.predict_and_score`` produces --
    ``{"output": ..., "scores": {scorer_name: result}, "model_latency":
    ...}`` -- one per row, in the same order. This is exactly what you get
    back from an ``EvaluationLogger`` run, or by iterating
    ``EvaluationResults.rows`` after ``evaluation.get_eval_results(model)``.

    ``started_at`` defaults to the current UTC time if not supplied (the
    schema requires it); pass it explicitly for reproducible output, as the
    tests here do.
    """
    if started_at is None:
        import datetime

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    rows = list(rows)
    eval_results = list(eval_results)
    if len(rows) != len(eval_results):
        raise ValueError(
            f"rows ({len(rows)}) and eval_results ({len(eval_results)}) must be the same length "
            "and in the same order -- evaluation_to_openeval() pairs them positionally."
        )

    results: List[Dict[str, Any]] = []
    for idx, (row, eval_row) in enumerate(zip(rows, eval_results)):
        row_id = row.get("id") if isinstance(row, Mapping) else None
        test_case_id = _stringify(row_id) if row_id is not None else f"row_{idx}"

        scores = eval_row.get("scores") or {}
        grader_results = [
            _score_to_grader_result(f"weave_{scorer_name}", scorer_name, result)
            for scorer_name, result in scores.items()
        ]

        result: Dict[str, Any] = {
            "test_case_id": test_case_id,
            "grader_results": grader_results,
            "passed": all(gr["passed"] for gr in grader_results) if grader_results else False,
        }

        output = eval_row.get("output", eval_row.get("model_output"))
        if output is not None:
            result["actual_output"] = _stringify(output)

        latency = eval_row.get("model_latency")
        if isinstance(latency, (int, float)):
            result["duration_ms"] = int(latency * 1000)

        results.append(result)

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "runner": {"name": "weave-openeval-adapter"},
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    return result_set
