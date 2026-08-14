"""Convert between UpTrain evaluation datasets/results and EvalPort.

EvalPort (https://github.com/adhabnr-ux/evalport) is an open interchange
format (Apache 2.0) for portable LLM evaluation datasets: test cases,
graders, suites, and results as plain JSON, shared across evaluation tools.

UpTrain (https://github.com/uptrain-ai/uptrain, ``pip install uptrain``)
evaluates a list of row dicts (or a pandas/polars DataFrame) against a set
of built-in checks via ``EvalLLM.evaluate(data, checks)``. Rows use
UpTrain's default ``DataSchema`` field names -- ``question``, ``response``,
``context``, ``ground_truth`` (verified directly against
``uptrain.framework.remote.DataSchema()``, not guessed) -- and
``evaluate()`` returns that same list of rows with two new keys added per
check: ``score_<check>`` (a float, typically 0.0-1.0) and
``explanation_<check>`` (the LLM judge's reasoning), confirmed directly from
the ``EvalLLM.evaluate`` source (see the ``sink_data`` / ``status_score_*``
handling in ``uptrain/framework/evalllm.py``).

Three entry points, matching the shape used by every other adapter in the
EvalPort ecosystem:

    to_openeval(data, ...)
        A list of UpTrain-shaped row dicts (or a pandas/polars DataFrame) --
        i.e. what you'd pass as ``EvalLLM.evaluate(data=...)`` -- -> an
        EvalPort suite.

    from_openeval(suite, ...)
        An EvalPort suite's test cases -> a list of row dicts shaped for
        ``EvalLLM.evaluate(data=...)``.

    results_to_openeval(results, ...)
        The list of row dicts ``EvalLLM.evaluate()`` returns (original
        fields plus ``score_<check>``/``explanation_<check>`` per check) ->
        an EvalPort ResultSet, one GraderResult per check that scored each
        row.

UpTrain's ``question``/``ground_truth``/``context`` map directly onto
EvalPort's ``TestCase.input``/``expected_output``/``context``. UpTrain's
``response`` is the answer *already generated* elsewhere that UpTrain is
scoring -- not a task input -- so, matching how ``ragas-openeval-adapter``
handles the identical situation for Ragas's own ``answer`` field,
``to_openeval()`` keeps it under ``metadata["uptrain"]["row"]`` rather than
losing it, and ``results_to_openeval()`` surfaces it as each result's
``actual_output``. The full original row is always preserved under
``metadata["uptrain"]["row"]`` on export, so a round trip through this
adapter never silently drops a field UpTrain would otherwise need.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from openeval.version import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "results_to_openeval"]

# UpTrain's default DataSchema field names (uptrain.framework.remote.DataSchema()).
QUESTION_KEY = "question"
RESPONSE_KEY = "response"
CONTEXT_KEY = "context"
GROUND_TRUTH_KEY = "ground_truth"
ID_KEY = "id"

_SCORE_PREFIX = "score_"
_EXPLANATION_PREFIX = "explanation_"
_CONFIDENCE_INFIX = "confidence"


def _stringify(value: Any) -> str:
    """Render an arbitrary value as the string EvalPort's schema requires."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def _rows_from(data: Any) -> List[Dict[str, Any]]:
    """Normalize ``data`` (list of dicts, pandas DataFrame, or polars DataFrame) to a list of dicts."""
    to_dict_records = getattr(data, "to_dict", None)
    if callable(to_dict_records) and not isinstance(data, Mapping):
        try:
            return list(data.to_dict(orient="records"))  # pandas
        except TypeError:
            pass
    to_dicts = getattr(data, "to_dicts", None)
    if callable(to_dicts):
        return list(data.to_dicts())  # polars
    return [dict(row) for row in data]


def to_openeval(
    data: Any,
    suite_id: str = "uptrain_eval",
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """Convert an UpTrain-shaped dataset into an EvalPort suite.

    ``data`` is whatever you'd pass to ``EvalLLM.evaluate(data=...)``: a
    list of dicts, or a pandas/polars DataFrame, with rows using UpTrain's
    default field names (``question``, ``response``, ``context``,
    ``ground_truth``).
    """
    rows = _rows_from(data)

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
        row_id = row.get(ID_KEY)
        tc_id = _stringify(row_id) if row_id is not None else f"row_{idx}"

        question = row.get(QUESTION_KEY)
        input_text = _stringify(question) if question is not None else _stringify(row)

        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": input_text,
            "graders": [default_grader["id"]],
            "metadata": {"uptrain": {"row": row}},
        }

        ground_truth = row.get(GROUND_TRUTH_KEY)
        if ground_truth is not None:
            test_case["expected_output"] = _stringify(ground_truth)

        context = row.get(CONTEXT_KEY)
        if context is not None:
            test_case["context"] = list(context) if isinstance(context, (list, tuple)) else [_stringify(context)]

        test_cases.append(test_case)

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": suite_id,
        "graders": [default_grader],
        "test_cases": test_cases,
    }


def from_openeval(suite: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite's test cases into UpTrain-shaped rows.

    Returns a plain list of dicts shaped for ``EvalLLM.evaluate(data=...)``.
    If a test case's ``metadata["uptrain"]["row"]`` is present (i.e. it
    round-trips a suite that came from ``to_openeval()`` above), that
    original row -- including ``response``, which has no home on an EvalPort
    ``TestCase`` -- is reconstructed verbatim; otherwise a fresh row is
    built from the test case's ``input``/``expected_output``/``context``.
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        uptrain_meta = (tc.get("metadata") or {}).get("uptrain")
        if isinstance(uptrain_meta, Mapping) and "row" in uptrain_meta:
            row = dict(uptrain_meta["row"])
        else:
            input_value = tc.get("input")
            row = {
                ID_KEY: tc.get("id"),
                QUESTION_KEY: input_value if isinstance(input_value, str) else json.dumps(input_value),
            }
            if "expected_output" in tc:
                row[GROUND_TRUTH_KEY] = tc["expected_output"]
            if tc.get("context"):
                row[CONTEXT_KEY] = list(tc["context"])
        rows.append(row)
    return rows


def _grader_result_from_score(
    grader_id: str,
    check_name: str,
    score: Any,
    explanation: Optional[str],
    confidence: Optional[Any],
) -> Dict[str, Any]:
    """Build one EvalPort GraderResult from one UpTrain ``score_<check>`` value.

    UpTrain scores are typically floats in [0, 1] (occasionally ``None``
    when a check couldn't be scored for a row); clamped defensively into
    EvalPort's required [0, 1] range regardless. A score >= 0.5 counts as a
    pass, matching the convention every other adapter in this repository
    uses when a tool doesn't supply its own explicit pass/fail flag.
    """
    metadata: Dict[str, Any] = {"uptrain_check": check_name}
    if explanation is not None:
        metadata["explanation"] = explanation
    if confidence is not None:
        metadata["confidence"] = confidence

    if score is None:
        return {
            "grader_id": grader_id,
            "type": "custom",
            "score": None,
            "passed": False,
            "metadata": metadata,
        }

    clamped = max(0.0, min(1.0, float(score)))
    result: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": "custom",
        "score": clamped,
        "passed": clamped >= 0.5,
        "metadata": metadata,
    }
    if explanation:
        result["reason"] = explanation
    return result


def results_to_openeval(
    results: Iterable[Mapping[str, Any]],
    suite_id: str = "uptrain_eval",
    run_id: str = "uptrain_run",
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert the list ``EvalLLM.evaluate()`` returns into an EvalPort ResultSet.

    ``results`` is exactly what ``EvalLLM.evaluate(data, checks)`` hands
    back: each row from ``data``, with ``score_<check>`` and
    ``explanation_<check>`` keys added per check that was run (this is the
    real shape, read directly from ``EvalLLM.evaluate``'s source -- not a
    guess). Every key of the form ``score_<name>`` (other than
    ``score_confidence_<name>``, which UpTrain adds separately as the
    judge's confidence in its own score) becomes one EvalPort
    ``GraderResult``.

    ``started_at`` defaults to the current UTC time if not supplied (the
    schema requires it); pass it explicitly for reproducible output, as the
    tests here do.
    """
    if started_at is None:
        import datetime

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    results = list(results)
    out_results: List[Dict[str, Any]] = []

    for idx, row in enumerate(results):
        row_id = row.get(ID_KEY)
        test_case_id = _stringify(row_id) if row_id is not None else f"row_{idx}"

        score_keys = [
            k
            for k in row.keys()
            if k.startswith(_SCORE_PREFIX) and _CONFIDENCE_INFIX not in k
        ]

        grader_results = []
        for score_key in score_keys:
            check_name = score_key[len(_SCORE_PREFIX):]
            explanation = row.get(f"{_EXPLANATION_PREFIX}{check_name}")
            confidence = row.get(f"{_SCORE_PREFIX}{_CONFIDENCE_INFIX}_{check_name}")
            grader_results.append(
                _grader_result_from_score(
                    f"uptrain_{check_name}", check_name, row[score_key], explanation, confidence
                )
            )

        result: Dict[str, Any] = {
            "test_case_id": test_case_id,
            "grader_results": grader_results,
            "passed": all(gr["passed"] for gr in grader_results) if grader_results else False,
        }

        response = row.get(RESPONSE_KEY)
        if response is not None:
            result["actual_output"] = _stringify(response)

        out_results.append(result)

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": out_results,
        "runner": {"name": "uptrain-openeval-adapter"},
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    return result_set
