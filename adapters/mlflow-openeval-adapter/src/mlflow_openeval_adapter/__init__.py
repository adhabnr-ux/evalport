"""MLflow <-> EvalPort adapter.

Standalone converter between MLflow (https://mlflow.org) `mlflow.evaluate()`
results and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside the
MLflow SDK itself: it follows the same playbook that already worked for
AutoGen, CrewAI, Ragas, LangSmith, and Braintrust (see
../autogen-openeval-adapter, ../crewai-openeval-adapter,
../ragas-openeval-adapter, ../langsmith-openeval-adapter,
../braintrust-openeval-adapter) — it works against MLflow's public
`EvaluationResult` shape (its `.metrics` aggregate dict and
`.tables["eval_results_table"]` per-row DataFrame) from the outside, so you
get EvalPort import/export today without needing anything merged into the
`mlflow` package.

Tracked as https://github.com/adhabnr-ux/evalport/issues/4.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"

# Column names in MLflow's eval_results_table that map to standard EvalPort
# TestCase fields rather than being treated as a per-row metric score.
_INPUT_COLUMNS = ("inputs", "input", "question")
_EXPECTED_COLUMNS = ("targets", "target", "ground_truth", "expected")
_OUTPUT_COLUMNS = ("outputs", "output", "prediction")
_RESERVED_COLUMNS = set(_INPUT_COLUMNS) | set(_EXPECTED_COLUMNS) | set(_OUTPUT_COLUMNS)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first(row: Dict[str, Any], columns: tuple, default: Any = None) -> Any:
    for col in columns:
        if col in row and row[col] is not None:
            return row[col]
    return default


def _metric_grader_id(column: str) -> str:
    """Turn an mlflow metric column name (e.g. "exact_match/v1/score") into
    a stable EvalPort grader id (e.g. "gr_exact_match_v1").

    Only the trailing "/score" suffix mlflow appends to per-row metric
    columns is stripped; any remaining path segments (like a metric
    version, e.g. "v1") are preserved so distinct metric versions don't
    collide on the same grader id.
    """
    name = column[: -len("/score")] if column.endswith("/score") else column
    name = name.replace("/", "_").replace(".", "_").replace(" ", "_")
    return f"gr_{name}"


def _rows_from_result(mlflow_result: Any) -> List[Dict[str, Any]]:
    """Normalize an mlflow.evaluate() result (or dict/list stand-in) into a list of per-row dicts.

    Prefers `.tables["eval_results_table"]` (the documented way to get
    per-row predictions/scores out of an `EvaluationResult`) — accepting
    either a real pandas DataFrame (via `.to_dict(orient="records")`) or
    anything already shaped as a list of row dicts. Falls back to a plain
    list of row dicts directly, or a dict with a `rows` key, for
    tests / JSON-loaded output.
    """
    if isinstance(mlflow_result, list):
        return list(mlflow_result)

    tables = _get(mlflow_result, "tables", None)
    if tables is not None:
        table = _get(tables, "eval_results_table", None)
        if table is not None:
            to_dict = getattr(table, "to_dict", None)
            if callable(to_dict):
                return list(to_dict(orient="records"))
            if isinstance(table, list):
                return list(table)

    rows = _get(mlflow_result, "rows", None)
    return list(rows) if rows is not None else []


def _row_payload(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Normalize a single mlflow eval_results_table row into an EvalPort TestCase dict."""
    input_value = _first(row, _INPUT_COLUMNS, "")
    expected = _first(row, _EXPECTED_COLUMNS, None)
    output = _first(row, _OUTPUT_COLUMNS, None)

    scores = {
        col: val
        for col, val in row.items()
        if col not in _RESERVED_COLUMNS and isinstance(val, (int, float)) and not isinstance(val, bool)
    }
    graders = [_metric_grader_id(col) for col in sorted(scores.keys())] or ["gr_mlflow_score"]

    tc: Dict[str, Any] = {
        "id": f"tc_{index}",
        "input": input_value if isinstance(input_value, (str, list)) else str(input_value),
        "graders": graders,
    }
    if expected is not None:
        tc["expected_output"] = expected if isinstance(expected, str) else str(expected)

    metadata: Dict[str, Any] = {}
    if scores:
        metadata["mlflow_scores"] = scores
    if output is not None:
        # The row's actual model output belongs on a Result, not a
        # TestCase — kept as metadata so round-tripping doesn't lose it.
        metadata["mlflow_actual_output"] = output if isinstance(output, str) else str(output)
    if metadata:
        tc["metadata"] = metadata
    return tc


def to_openeval(mlflow_result: Any, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Export an `mlflow.evaluate()` result to an EvalPort-shaped suite (dict).

    `mlflow_result` may be a real MLflow `EvaluationResult` (uses its
    `.tables["eval_results_table"]` DataFrame), a plain dict with a `rows`
    key, or a bare list of row dicts — no direct `mlflow` import is
    required. Each row is expected to expose an input column
    (`inputs`/`input`/`question`), optionally a target/ground-truth column
    (`targets`/`target`/`ground_truth`/`expected`), an output column
    (`outputs`/`output`/`prediction`), and any number of per-row numeric
    metric-score columns (as `mlflow.evaluate()` adds for each metric
    passed to `extra_metrics=`, typically named like `"<metric>/score"` or
    `"<metric>/v1/score"`).

    Every metric-score column found becomes its own EvalPort grader
    (`gr_<metric>`, type "custom", handler `mlflow:<metric>`) so a
    downstream EvalPort runner can re-score with the same metric set. The
    scores MLflow already computed are preserved per test case under
    `metadata.mlflow_scores`, and the run's aggregate metrics (`.metrics`)
    are preserved at the suite level under `metadata.mlflow_metrics` — an
    `evaluate()` run is already-scored data, not just a task definition,
    so nothing is thrown away.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    rows = _rows_from_result(mlflow_result)
    aggregate_metrics = dict(_get(mlflow_result, "metrics", None) or {})
    resolved_run_id = run_id or "mlflow_run"

    test_cases = [_row_payload(r, i) for i, r in enumerate(rows)]

    score_columns = sorted(
        {col for r in rows for col in r.keys() if col not in _RESERVED_COLUMNS and isinstance(r[col], (int, float)) and not isinstance(r[col], bool)}
    )
    graders = [
        {
            "id": _metric_grader_id(col),
            "type": "custom",
            "description": f"MLflow '{col}' metric",
            "params": {"handler": f"mlflow:{col}"},
        }
        for col in score_columns
    ]
    if not graders:
        graders = [{"id": "gr_mlflow_score", "type": "custom", "params": {"handler": "mlflow:score"}}]

    return {
        "version": OPENEVAL_VERSION,
        "id": f"mlflow_eval_{resolved_run_id}",
        "name": f"MLflow eval run {resolved_run_id}",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "mlflow"}, "mlflow_metrics": aggregate_metrics},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of MLflow-shaped eval-data row dicts.

    Returns plain dicts with `inputs`/`targets` keys, ready to build the
    `data` argument of a fresh `mlflow.evaluate()` call:

        import mlflow
        import pandas as pd
        from mlflow_openeval_adapter import from_openeval

        rows = from_openeval(suite)
        mlflow.evaluate(model=my_model, data=pd.DataFrame(rows), targets="targets", ...)
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        row: Dict[str, Any] = {"inputs": tc.get("input")}
        if tc.get("expected_output") is not None:
            row["targets"] = tc.get("expected_output")
        rows.append(row)
    return rows
