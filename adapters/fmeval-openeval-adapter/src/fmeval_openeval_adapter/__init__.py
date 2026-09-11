"""AWS fmeval <-> EvalPort adapter.

Standalone converter between the AWS `fmeval` (Foundation Model Evaluations)
library's datasets and evaluation output, and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport). No changes to `fmeval` itself are
required -- this is built entirely against its public, documented surface:

- `fmeval.data_loaders.data_config.DataConfig` for the dataset shape (verified
  against fmeval's real source: `model_input_location` / `target_output_location`
  / `category_location` / `context_location`, per `DataConfig.__doc__`).
- `fmeval.eval_algorithms.util.EvalOutputRecord.to_dict()` for the per-record
  output shape written to the JSON Lines file at `EvalOutput.output_path` when
  an algorithm's `.evaluate(..., save=True)` is called. Confirmed by actually
  running `FactualKnowledge().evaluate(...)` end to end in this adapter's test
  suite against a real installed `fmeval` and reading the real file it writes:
  each line is `{"model_input": ..., "model_output": ..., "target_output": ...,
  ["category": ...], ["context": ...], "scores": [{"name": ..., "value": <float>}
  | {"name": ..., "error": <str>}, ...]}`.

Two independent conversions, mirroring fmeval's own two-stage flow (a dataset
you evaluate, and the per-record scores that evaluation produces):

- `to_openeval()` / `from_openeval()` -- an fmeval-shaped dataset (a list of row
  dicts, using fmeval's own canonical column names or your own, auto-detected)
  <-> an EvalPort `EvalSuite`.
- `eval_output_to_openeval()` -- the real per-record JSON Lines fmeval writes to
  an algorithm's `output_path` (or an already-parsed list of dicts in that same
  shape) -> an EvalPort `ResultSet`, one `GraderResult` per fmeval `EvalScore`.

Score direction: every fmeval score is "higher is better" *except* two
verified-by-source exceptions -- `toxicity` (`TOXIGEN_SCORE_NAME` in
`fmeval.eval_algorithms.helper_models.helper_model`, 0..1, 1.0 = fully toxic)
and `word_error_rate` (`WER_SCORE` in `fmeval.eval_algorithms.general_semantic_robustness`,
0..1+, 0 = a perfect match). `eval_output_to_openeval()` accounts for this via
`_DEFAULT_LOWER_IS_BETTER` so a low toxicity/WER score reports `passed=True`
instead of being silently scored backwards -- extend it via the
`lower_is_better` parameter for any other fmeval score with this convention.
"""
from __future__ import annotations

import json as _json
from typing import Any, Dict, Iterable, List, Optional, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "eval_output_to_openeval", "__version__"]
__version__ = "0.1.0"

# Candidate keys checked (case-insensitively, in order) to auto-detect which
# raw dataset field is the model input / expected output / RAG context /
# category, since a caller's own JSON Lines dataset is free to use any field
# names before it's pointed at via an fmeval DataConfig's `*_location`
# JMESPath queries. fmeval's own canonical names (verified against
# `fmeval.constants.DatasetColumns`) are checked first.
_INPUT_KEY_CANDIDATES = ("model_input", "input", "question", "prompt", "query", "text")
_EXPECTED_OUTPUT_KEY_CANDIDATES = ("target_output", "expected_output", "answer", "reference", "ground_truth", "label")
_CONTEXT_KEY_CANDIDATES = ("context",)
_CATEGORY_KEY_CANDIDATES = ("category",)

# fmeval score names where a LOWER raw value means a BETTER result -- see the
# module docstring for how each was verified against fmeval's real source.
_DEFAULT_LOWER_IS_BETTER = {"toxicity", "word_error_rate"}


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
    return _json.dumps(value, default=str)


def _clamp01(value: Optional[float]) -> Optional[float]:
    """Clamp a raw fmeval score into EvalPort's required [0, 1] range.

    EvalPort's schema (`spec/schemas/resultset.json`, enforced by
    `openeval.validate.validate_result_set`) requires every `GraderResult.score`
    to be `null` or within `[0, 1]`. Most fmeval scores already live in that
    range (rouge, bertscore, factual_knowledge, classification accuracy,
    toxicity), but a few genuinely don't -- `log_probability_difference`
    (prompt stereotyping bias) is a signed, unbounded log-probability delta,
    and `word_error_rate` can exceed 1.0 for a very bad transcription. Rather
    than reject those or silently pass an invalid document, this clamps for
    the EvalPort `score` field while always preserving the true raw value
    under `metadata.raw_value` -- so nothing is lost, and the document stays
    schema-valid either way.
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
    suite_id: str = "fmeval_dataset_import",
    name: Optional[str] = None,
    input_key: Optional[str] = None,
    expected_output_key: Optional[str] = None,
    context_key: Optional[str] = None,
    category_key: Optional[str] = None,
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """Export an fmeval-shaped dataset to an EvalPort-shaped suite (dict).

    `rows` is any iterable of plain dicts -- the same rows you would write to
    a JSON Lines file and point an fmeval `DataConfig` at, either already
    using fmeval's own canonical column names (`model_input`, `target_output`,
    `category`, `context`) or your own raw field names, which are then
    auto-detected the same way every other schema-less adapter in this
    repository handles its source format (see e.g. the
    [Opik adapter](../opik-openeval-adapter)). An explicit `input_key`/
    `expected_output_key`/`context_key`/`category_key` always overrides the
    heuristic.

    fmeval's `context` column (used by RAG-oriented algorithms) maps directly
    onto EvalPort's own `TestCase.context` field, which is also a list of
    strings -- if the source value is a single string it's wrapped in a
    one-element list; if it's already a list, it's used as-is.

    Every other field on the row is preserved under the test case's
    `metadata["fmeval"]`, so nothing from the original row is ever silently
    dropped even when the input/expected-output/context/category guess picks
    (or misses) differently than you'd expect.

    `grader_type` selects the default grader attached when a test case has an
    `expected_output`: `"llm_judge"` (default) or `"exact_match"`.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass it
    to `openeval.validate.validate_suite()` to confirm compliance.
    """
    test_cases: List[Dict[str, Any]] = []
    has_expected_output = False

    for i, row in enumerate(rows):
        row = dict(row)
        in_key = _pick_key(row, input_key, _INPUT_KEY_CANDIDATES)
        out_key = _pick_key(row, expected_output_key, _EXPECTED_OUTPUT_KEY_CANDIDATES)
        ctx_key = _pick_key(row, context_key, _CONTEXT_KEY_CANDIDATES)
        cat_key = _pick_key(row, category_key, _CATEGORY_KEY_CANDIDATES)

        if in_key is not None:
            input_value = _stringify(row[in_key])
        else:
            input_value = _stringify(row) if row else ""

        tc: Dict[str, Any] = {"id": f"tc_{i}", "input": input_value, "graders": ["gr_output_match"]}

        if out_key is not None:
            tc["expected_output"] = _stringify(row[out_key])
            has_expected_output = True

        if ctx_key is not None:
            ctx_value = row[ctx_key]
            tc["context"] = ctx_value if isinstance(ctx_value, list) else [_stringify(ctx_value)]

        consumed = {k for k in (in_key, out_key, ctx_key) if k is not None}
        metadata_fmeval = {k: v for k, v in row.items() if k not in consumed}
        if cat_key is not None:
            metadata_fmeval.setdefault("category", row[cat_key])
        tc["metadata"] = {"fmeval": metadata_fmeval} if metadata_fmeval else {}

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
            # No expected_output anywhere: still attach a grader so the suite
            # validates (EvalPort requires >=1 grader ref per test case), but
            # make it clearly a placeholder rather than silently claiming an
            # llm_judge/exact_match comparison that has nothing to compare against.
            graders.append({
                "id": "gr_output_match",
                "type": "custom",
                "params": {"handler": "fmeval:no_expected_output"},
            })

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name or f"fmeval dataset import ({suite_id})",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "fmeval"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of fmeval-loadable dataset rows.

    Returns plain dicts using fmeval's own canonical column names
    (`model_input`, `target_output`, `context`) -- write each dict as one line
    of a JSON Lines file and point an `fmeval.data_loaders.data_config.DataConfig`
    at it (`model_input_location="model_input"`, etc.) to load the suite as an
    fmeval dataset. Every field that was round-tripped through
    `metadata["fmeval"]` on the way in is restored to the top level, so a
    suite exported by `to_openeval()` and re-imported here reconstructs the
    original row shape.
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        row: Dict[str, Any] = {"model_input": tc.get("input")}
        if "expected_output" in tc:
            row["target_output"] = tc["expected_output"]
        if tc.get("context"):
            row["context"] = tc["context"]
        metadata_fmeval = dict((tc.get("metadata") or {}).get("fmeval") or {})
        row.update(metadata_fmeval)
        rows.append(row)
    return rows


def eval_output_to_openeval(
    records: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    output_path: Optional[str] = None,
    suite_id: str,
    run_id: str,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    started_at: Optional[str] = None,
    runner_name: str = "fmeval",
    runner_version: Optional[str] = None,
    pass_threshold: float = 0.5,
    lower_is_better: Optional[set] = None,
) -> Dict[str, Any]:
    """Export fmeval's per-record evaluation output to an EvalPort ResultSet.

    Pass exactly one of:

    - `records`: an already-parsed iterable of dicts in the real
      `EvalOutputRecord.to_dict()` shape fmeval writes -- `{"model_input": ...,
      "model_output": ..., "target_output": ..., ["category": ...],
      ["context": ...], "scores": [{"name": ..., "value": <float>} |
      {"name": ..., "error": <str>}, ...]}`.
    - `output_path`: the path to the `.jsonl` file an algorithm's
      `EvalOutput.output_path` points at after calling `.evaluate(..., save=True)`
      -- this reads and parses it for you.

    Each fmeval `EvalScore` becomes one EvalPort `GraderResult` (`grader_id`
    is the score name, e.g. `"factual_knowledge"` or `"rouge"`). A score with
    an `error` instead of a `value` (fmeval's own convention for a failed
    per-record computation) becomes a `GraderResult` with `score=None`,
    `passed=False`, and the error text as `reason`. Every raw score value is
    additionally preserved verbatim under `grader_results[].metadata.raw_value`
    even when it had to be clamped into EvalPort's required `[0, 1]` range
    (see `_clamp01`) -- so nothing from the real fmeval output is ever lost.

    `lower_is_better` (default: `{"toxicity", "word_error_rate"}`, verified
    against fmeval's real source -- see the module docstring) names the score
    types where a *lower* raw value should count as a pass; every other score
    passes at `>= pass_threshold` (default `0.5`) after clamping. Extend the
    default set for any other fmeval score with this same convention.

    `test_cases` (optional): the same list of EvalPort TestCase dicts you got
    back from `to_openeval()` (or `suite["test_cases"]`), used to recover the
    correct `test_case_id` for each record by matching on
    `(input, expected_output)`. Recommended whenever you also built the input
    suite with `to_openeval()`, since fmeval evaluates via a distributed Ray
    Dataset internally and does not guarantee output row order matches input
    row order under parallel execution. If omitted, records are assigned
    sequential ids (`"tc_0"`, `"tc_1"`, ...) in the order given -- exactly
    correct for a single, unparallelized local run (the case this adapter's
    own tests exercise end to end against a real `fmeval` install), but not
    guaranteed for a distributed run.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass it
    to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    if (records is None) == (output_path is None):
        raise ValueError("Pass exactly one of `records` or `output_path`.")

    if output_path is not None:
        with open(output_path, "r", encoding="utf-8") as f:
            records = [_json.loads(line) for line in f if line.strip()]

    if started_at is None:
        from datetime import datetime, timezone
        started_at = datetime.now(timezone.utc).isoformat()

    lower_is_better = _DEFAULT_LOWER_IS_BETTER if lower_is_better is None else set(lower_is_better)

    id_lookup: Dict[tuple, str] = {}
    if test_cases:
        for tc in test_cases:
            id_lookup[(tc.get("input"), tc.get("expected_output"))] = tc.get("id")

    results: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        model_input = record.get("model_input")
        target_output = record.get("target_output")
        test_case_id = id_lookup.get((model_input, target_output), f"tc_{i}")

        grader_results: List[Dict[str, Any]] = []
        for score in record.get("scores", []):
            score_name = score.get("name", "gr_unknown")
            error = score.get("error")
            raw_value = score.get("value")
            clamped = _clamp01(raw_value)

            gr: Dict[str, Any] = {
                "grader_id": str(score_name),
                "type": "custom",
                "score": clamped,
                "metadata": {"raw_value": raw_value},
            }
            if error is not None:
                gr["passed"] = False
                gr["reason"] = str(error)
            elif clamped is None:
                gr["passed"] = False
            elif score_name in lower_is_better:
                gr["passed"] = clamped <= (1.0 - pass_threshold)
            else:
                gr["passed"] = clamped >= pass_threshold
            grader_results.append(gr)

        result: Dict[str, Any] = {
            "test_case_id": str(test_case_id),
            "passed": len(grader_results) > 0 and all(g["passed"] for g in grader_results),
            "grader_results": grader_results,
            "actual_output": record.get("model_output"),
        }
        fmeval_extra = {k: v for k, v in record.items() if k not in ("model_input", "model_output", "target_output", "scores")}
        if fmeval_extra:
            result["metadata"] = {"fmeval": fmeval_extra}
        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    scores = [g["score"] for r in results for g in r["grader_results"] if g.get("score") is not None]
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped": 0,
        "pass_rate": (passed / total) if total else 0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0,
    }

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": started_at,
        "runner": {"name": runner_name, "version": runner_version or __version__},
        "results": results,
        "summary": summary,
        "metadata": {"openeval": {"source": "fmeval"}},
    }
