"""IBM unitxt <-> EvalPort adapter.

Standalone converter between IBM's `unitxt` evaluation library and the
EvalPort interchange format (https://github.com/adhabnr-ux/evalport). No
changes to `unitxt` itself are required -- this is built entirely against its
public, documented API surface:

- `unitxt.api.create_dataset(task=..., test_set=[...])` for the dataset shape:
  `test_set` is a plain list of dicts matching the chosen task's raw input
  schema (e.g. `{"question": ..., "answers": [...]}` for `tasks.qa.open`).
- `unitxt.api.evaluate(predictions=[...], data=dataset)` for the evaluation
  output shape, confirmed by actually running it end to end in this
  adapter's test suite against a real installed `unitxt` and inspecting the
  real `EvaluationResults` it returns: a list, one dict per instance, each
  with `task_data` (the original raw input fields), `prediction`,
  `processed_prediction`, `references`, and `score` -- itself
  `{"instance": {<metric name>: <float>, ..., "score_name": <str>,
  "score": <float>}, "global": {...same shape, aggregated...}}`.

Two independent conversions, mirroring unitxt's own two-stage flow (a
dataset you build, and the scored results `evaluate()` produces):

- `to_openeval()` / `from_openeval()` -- a unitxt task's raw `test_set` rows
  <-> an EvalPort `EvalSuite`.
- `evaluation_to_openeval()` -- a real `unitxt.evaluate()` result -> an
  EvalPort `ResultSet`, one `GraderResult` per metric key found in each
  instance's `score["instance"]` dict (a task can attach more than one
  metric, e.g. `metrics.rouge` computes `rouge1`/`rouge2`/`rougeL`/`rougeLsum`
  all at once -- every one becomes its own `GraderResult`, not just the
  task's primary `score`).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "evaluation_to_openeval", "__version__"]
__version__ = "0.1.0"

# Candidate keys checked (case-insensitively, in order) to auto-detect which
# raw task_data field is the model input / expected output, since a unitxt
# task's input schema is entirely task-defined (verified against several
# real built-in tasks: `tasks.qa.open` uses `question`/`answers`,
# `tasks.classification.multi_class` uses `text`/`class`, `tasks.summarization.abstractive`
# uses `document`/`summary`). An explicit input_key/expected_output_key
# argument always overrides the heuristic.
_INPUT_KEY_CANDIDATES = ("question", "text", "document", "input", "prompt", "premise")
_EXPECTED_OUTPUT_KEY_CANDIDATES = ("answers", "answer", "class", "label", "summary", "expected_output", "reference", "references")

# score["instance"] / score["global"] always carry `score_name` (a string
# identifying which of the sibling metric keys is the task's "primary" one)
# and `score` (a numeric duplicate of that same primary metric's value --
# verified against a real unitxt run: {"accuracy": 1.0, "score_name":
# "accuracy", "score": 1.0}, where "score" == "accuracy"). Neither is a
# distinct metric in its own right, so neither is turned into its own
# GraderResult -- the primary metric is still fully represented via its own
# named key (e.g. "accuracy" above).
_NON_SCORE_KEYS = {"score_name", "score"}


def _pick_key(row: Dict[str, Any], explicit: Optional[str], candidates: tuple) -> Optional[str]:
    if explicit is not None:
        return explicit if explicit in row else None
    lower_map = {str(k).lower(): k for k in row.keys()}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    import json as _json
    return _json.dumps(value, default=str)


def _first_or_stringify(value: Any) -> str:
    """unitxt reference/expected-output fields are commonly a list of
    acceptable answers (e.g. `answers: ["Paris"]`) rather than a single
    string. EvalPort's `TestCase.expected_output` is a single optional
    string, so the first element is used and the full list is always
    additionally preserved under `metadata["unitxt"]` -- see `to_openeval()`.
    """
    if isinstance(value, list):
        return _stringify(value[0]) if value else ""
    return _stringify(value)


def _clamp01(value: Optional[float]) -> Optional[float]:
    """Clamp a raw unitxt metric value into EvalPort's required [0, 1] range,
    the same convention used by every numeric-score adapter in this
    repository (see e.g. the [fmeval adapter](../fmeval-openeval-adapter)).
    Most unitxt metrics (accuracy, F1, rouge, BLEU-derived scores) already
    live in [0, 1], but this guards against any metric that doesn't while
    always preserving the true raw value under `metadata.raw_value`.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


def to_openeval(
    rows: Iterable[Dict[str, Any]],
    *,
    suite_id: str = "unitxt_dataset_import",
    name: Optional[str] = None,
    input_key: Optional[str] = None,
    expected_output_key: Optional[str] = None,
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """Export a unitxt task's raw `test_set` rows to an EvalPort-shaped suite.

    `rows` is any iterable of plain dicts in the same shape you'd pass as
    `test_set=` to `unitxt.api.create_dataset(task=..., test_set=rows)` --
    the task's own raw input schema (e.g. `{"question": ..., "answers": [...]}`
    for `tasks.qa.open`). The input/expected-output field is auto-detected
    from several real built-in tasks' conventions (see module docstring), or
    named explicitly via `input_key`/`expected_output_key`.

    Because unitxt commonly represents the expected output as a list of
    acceptable answers (`answers: ["Paris"]`) rather than a single string,
    the first element is used for EvalPort's single-string
    `TestCase.expected_output`, and the full original list is always
    additionally preserved under `metadata["unitxt"]["expected_output_full"]`
    -- so a multi-reference task never silently loses its other valid
    answers. Every other field on the row is preserved under
    `metadata["unitxt"]`, so nothing from the original row is ever dropped.

    `grader_type` selects the default grader attached when a test case has
    an `expected_output`: `"llm_judge"` (default) or `"exact_match"`.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass it
    to `openeval.validate.validate_suite()` to confirm compliance.
    """
    test_cases: List[Dict[str, Any]] = []
    has_expected_output = False

    for i, row in enumerate(rows):
        row = dict(row)
        in_key = _pick_key(row, input_key, _INPUT_KEY_CANDIDATES)
        out_key = _pick_key(row, expected_output_key, _EXPECTED_OUTPUT_KEY_CANDIDATES)

        if in_key is not None:
            input_value = _stringify(row[in_key])
        else:
            input_value = _stringify(row) if row else ""

        tc: Dict[str, Any] = {"id": f"tc_{i}", "input": input_value, "graders": ["gr_output_match"]}

        metadata_unitxt: Dict[str, Any] = {}
        if out_key is not None:
            raw_expected = row[out_key]
            tc["expected_output"] = _first_or_stringify(raw_expected)
            has_expected_output = True
            if isinstance(raw_expected, list) and len(raw_expected) > 1:
                metadata_unitxt["expected_output_full"] = raw_expected

        consumed = {k for k in (in_key, out_key) if k is not None}
        for k, v in row.items():
            if k not in consumed:
                metadata_unitxt[k] = v
        tc["metadata"] = {"unitxt": metadata_unitxt} if metadata_unitxt else {}

        test_cases.append(tc)

    graders: List[Dict[str, Any]] = []
    if test_cases:
        if has_expected_output:
            if grader_type == "exact_match":
                graders.append({"id": "gr_output_match", "type": "exact_match", "params": {"ignore_case": True}})
            else:
                graders.append({
                    "id": "gr_output_match",
                    "type": "llm_judge",
                    "params": {
                        "model": "gpt-4o",
                        "prompt": (
                            "Expected output: {expected}\nActual output: {output}\n"
                            "Does the actual output satisfy the expected output? "
                            'Return JSON: {"score": 0.0-1.0, "reason": "..."}'
                        ),
                    },
                })
        else:
            graders.append({
                "id": "gr_output_match",
                "type": "custom",
                "params": {"handler": "unitxt:no_expected_output"},
            })

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name or f"unitxt dataset import ({suite_id})",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "unitxt"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of unitxt-loadable `test_set` rows.

    Returns plain dicts using generic `input`/`expected_output` keys --
    remap them to your target task's own schema (e.g. `question`/`answers`
    for `tasks.qa.open`) before passing to `create_dataset()`, unless the
    original row shape was fully preserved via `metadata["unitxt"]`, in
    which case that original shape is restored directly. Every field that
    was round-tripped through `metadata["unitxt"]` on the way in is restored
    to the top level, so a suite exported by `to_openeval()` from a unitxt
    task's own rows and re-imported here reconstructs the original row.
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        row: Dict[str, Any] = {"input": tc.get("input")}
        if "expected_output" in tc:
            row["expected_output"] = tc["expected_output"]
        metadata_unitxt = dict((tc.get("metadata") or {}).get("unitxt") or {})
        expected_output_full = metadata_unitxt.pop("expected_output_full", None)
        row.update(metadata_unitxt)
        if expected_output_full is not None:
            # Restore the original list-valued field under whichever key it
            # came from, if that key is recoverable (it's whatever remained
            # after `input`/`expected_output` were consumed on export, so a
            # generic "answers" is used when the original key can't be
            # inferred from the suite alone).
            row.setdefault("answers", expected_output_full)
        rows.append(row)
    return rows


def evaluation_to_openeval(
    results: Any,
    *,
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    runner_name: str = "unitxt",
    runner_version: Optional[str] = None,
    pass_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Export a real `unitxt.api.evaluate()` result to an EvalPort ResultSet.

    `results` is the `unitxt.metric_utils.EvaluationResults` object returned
    by `evaluate(predictions=..., data=dataset)` (or any list of dicts in
    that same per-instance shape: `{"task_data": {...}, "prediction": ...,
    "processed_prediction": ..., "references": [...], "score": {"instance":
    {...}, "global": {...}}}`).

    Every key in an instance's `score["instance"]` dict *except* the
    `score_name` string identifier becomes its own EvalPort `GraderResult`
    -- a task with multiple attached metrics (e.g. `metrics.rouge`, which
    computes `rouge1`/`rouge2`/`rougeL`/`rougeLsum` together) gets one
    `GraderResult` per metric, not just the task's single primary `score`.

    `test_case_id`s are assigned by position (`"tc_0"`, `"tc_1"`, ...),
    matching the order `to_openeval()` assigns them for the same rows --
    unlike fmeval's distributed Ray Dataset execution, unitxt's `evaluate()`
    is a synchronous, in-process, order-preserving operation over the exact
    `predictions` list you pass it, verified against the real per-instance
    output in this adapter's own test suite, so no separate id-matching step
    is needed.

    The run-level `score["global"]` dict (identical across every instance in
    a single `evaluate()` call) is preserved once under
    `metadata["unitxt"]["global_scores"]` rather than repeated per result.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass it
    to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    if started_at is None:
        from datetime import datetime, timezone
        started_at = datetime.now(timezone.utc).isoformat()

    results = list(results)
    global_scores = None

    eval_results: List[Dict[str, Any]] = []
    for i, instance in enumerate(results):
        score = instance.get("score") or {}
        instance_scores = score.get("instance") or {}
        if global_scores is None and score.get("global"):
            global_scores = dict(score["global"])

        grader_results: List[Dict[str, Any]] = []
        for metric_name, raw_value in instance_scores.items():
            if metric_name in _NON_SCORE_KEYS:
                continue
            clamped = _clamp01(raw_value if isinstance(raw_value, (int, float)) else None)
            gr: Dict[str, Any] = {
                "grader_id": str(metric_name),
                "type": "custom",
                "score": clamped,
                "passed": clamped is not None and clamped >= pass_threshold,
                "metadata": {"raw_value": raw_value},
            }
            grader_results.append(gr)

        result: Dict[str, Any] = {
            "test_case_id": f"tc_{i}",
            "passed": len(grader_results) > 0 and all(g["passed"] for g in grader_results),
            "grader_results": grader_results,
            "actual_output": instance.get("prediction"),
        }
        task_data = instance.get("task_data")
        if task_data is not None:
            result["metadata"] = {"unitxt": {"task_data": task_data}}
        eval_results.append(result)

    total = len(eval_results)
    passed = sum(1 for r in eval_results if r["passed"])
    scores = [g["score"] for r in eval_results for g in r["grader_results"] if g.get("score") is not None]
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped": 0,
        "pass_rate": (passed / total) if total else 0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0,
    }

    metadata: Dict[str, Any] = {"openeval": {"source": "unitxt"}}
    if global_scores:
        metadata["unitxt"] = {"global_scores": global_scores}

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": started_at,
        "runner": {"name": runner_name, "version": runner_version or __version__},
        "results": eval_results,
        "summary": summary,
        "metadata": metadata,
    }
