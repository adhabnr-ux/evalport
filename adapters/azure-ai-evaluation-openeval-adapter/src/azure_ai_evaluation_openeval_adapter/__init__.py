"""
azure_ai_evaluation_openeval_adapter

Converts Azure AI Evaluation SDK (``azure-ai-evaluation``) data rows,
evaluators, and ``EvaluationResult``s to/from EvalPort
(https://github.com/adhabnr-ux/evalport), the open interchange format for
portable LLM evaluation test cases, graders, suites, and results.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

__all__ = ["to_openeval", "from_openeval", "evaluation_result_to_openeval"]

SPEC_VERSION = "1.0.0-rc.2"

# azure-ai-evaluation's own local, offline-computable NLP-metric evaluator
# classes. Even these are still mapped to EvalPort's "custom" grader type
# below, never "semantic_similarity" -- that type's schema implies an
# embedding/model-based method (params.model, params.provider), and these
# are lexical n-gram/token-overlap metrics (BLEU/ROUGE/METEOR/GLEU/F1), not
# semantic ones. Mapping them to "semantic_similarity" would misrepresent
# how they actually compute a score. This set only controls the grader's
# `description` text below, not its `type`.
_KNOWN_NLP_EVALUATOR_CLASSES = {
    "F1ScoreEvaluator",
    "BleuScoreEvaluator",
    "GleuScoreEvaluator",
    "MeteorScoreEvaluator",
    "RougeScoreEvaluator",
}

# Fields azure-ai-evaluation appends to a metric's base name, both in an
# evaluator's own __call__() return dict and in evaluate()'s per-row
# "outputs.<key>.*" columns -- confirmed against the real installed package
# (not assumed from docs; see this adapter's tests).
_METRIC_SUFFIXES = (
    "_score",
    "_passed",
    "_result",
    "_reason",
    "_status",
    "_threshold",
    "_properties",
    "_higher_is_better",
)


def _grader_for_evaluator(key: str, evaluator: Any) -> Dict[str, Any]:
    """Build an EvalPort grader dict for one (name, evaluator) pair from an
    ``evaluators={...}`` mapping, exactly as passed to
    ``azure.ai.evaluation.evaluate()``.

    Every evaluator maps to EvalPort's "custom" grader type. This is
    deliberate, not a shortcut: azure-ai-evaluation's evaluators span local
    NLP metrics (no fabrication risk either way), AI-assisted quality
    evaluators that need a live ``model_config`` just to construct, and
    safety evaluators that need a live Azure AI Foundry project -- none of
    which can be reconstructed from the outside without live credentials
    this adapter doesn't have. "custom" with ``params.handler`` set to the
    evaluator's real class (or function) name is the only honest mapping
    across all three categories; nothing here is guessed or fabricated.
    """
    cls_name = type(evaluator).__name__
    is_function = cls_name == "function"
    handler = getattr(evaluator, "__name__", key) if is_function else cls_name

    params: Dict[str, Any] = {"handler": handler}

    # RougeScoreEvaluator's rouge_type is real, offline-readable config --
    # capture it when present so a round trip doesn't lose it.
    rouge_type = getattr(evaluator, "_rouge_type", None)
    if rouge_type is not None:
        params["rouge_type"] = str(rouge_type)

    if cls_name in _KNOWN_NLP_EVALUATOR_CLASSES:
        description = f"azure-ai-evaluation {cls_name} (local NLP metric, no model/network required)"
    elif is_function:
        description = f"Custom function-based evaluator: {handler}"
    elif "model_config" in getattr(evaluator, "__dict__", {}) or hasattr(evaluator, "_prompty_file"):
        description = f"azure-ai-evaluation {cls_name} (AI-assisted, requires a live model_config to run)"
    else:
        description = f"azure-ai-evaluation {cls_name}"

    return {
        "id": key,
        "type": "custom",
        "params": params,
        "description": description,
    }


def _row_input_text(row: Dict[str, Any]) -> Union[str, List[str]]:
    for key in ("query", "input", "conversation", "prompt"):
        if key in row and row[key] not in (None, ""):
            value = row[key]
            return value if isinstance(value, str) else json.dumps(value, default=str)
    # No recognizable input column -- fall back to the whole row so nothing
    # is silently dropped, rather than raising on a shape this adapter
    # didn't anticipate.
    return json.dumps(row, default=str)


def to_openeval(
    data: Union[str, List[Dict[str, Any]]],
    evaluators: Dict[str, Any],
    evaluator_config: Optional[Dict[str, Any]] = None,
    suite_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert ``azure.ai.evaluation.evaluate()``'s own ``data``/``evaluators``
    arguments into a spec-valid EvalPort Suite.

    Args:
        data: Same shape ``evaluate(data=...)`` accepts -- a path to a
            ``.jsonl`` file, or an already-loaded list of row dicts.
        evaluators: Same shape ``evaluate(evaluators=...)`` accepts -- a
            dict mapping an evaluator name to an evaluator instance
            (built-in or a custom class/function).
        evaluator_config: Optional, same shape ``evaluate(evaluator_config=...)``
            accepts. Stored verbatim in the suite's metadata for a lossless
            round trip; not required for validity.
        suite_id: Suite id. Defaults to a generated uuid4.
        name: Suite name.

    Returns:
        A dict matching ``spec/schemas/suite.json``.
    """
    if isinstance(data, str):
        rows: List[Dict[str, Any]] = []
        with open(data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        rows = list(data)

    grader_ids = list(evaluators.keys())
    graders = [_grader_for_evaluator(k, v) for k, v in evaluators.items()]

    test_cases = []
    for i, row in enumerate(rows):
        row_id = str(row.get("id", i))
        expected = row.get("ground_truth")
        test_case: Dict[str, Any] = {
            "id": row_id,
            "input": _row_input_text(row),
            "graders": list(grader_ids),
            "metadata": {"azure_ai_evaluation": {"row": row}},
        }
        if expected is not None:
            test_case["expected_output"] = str(expected)
        if row.get("context") is not None:
            ctx = row["context"]
            test_case["context"] = ctx if isinstance(ctx, list) else [str(ctx)]
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": SPEC_VERSION,
        "id": suite_id or f"azure-ai-evaluation-{uuid.uuid4()}",
        "test_cases": test_cases,
        "graders": graders,
    }
    if name is not None:
        suite["name"] = name
    if evaluator_config is not None:
        suite["metadata"] = {"azure_ai_evaluation": {"evaluator_config": evaluator_config}}
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort Suite back into row dicts shaped for
    ``azure.ai.evaluation.evaluate(data=...)``.

    When a test case carries ``metadata["azure_ai_evaluation"]["row"]`` (set
    by ``to_openeval()`` above), that original row is restored byte-for-byte
    -- including any columns (``context``, custom columns, etc.) this
    adapter doesn't otherwise interpret. For a suite built elsewhere (no
    prior EvalPort round trip), each test case is mapped heuristically:
    ``input`` -> ``query``, ``expected_output`` -> ``ground_truth``.
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        metadata = tc.get("metadata") or {}
        azure_meta = metadata.get("azure_ai_evaluation") if isinstance(metadata, dict) else None
        saved_row = azure_meta.get("row") if isinstance(azure_meta, dict) else None
        if isinstance(saved_row, dict):
            rows.append(dict(saved_row))
            continue

        row: Dict[str, Any] = {}
        input_value = tc.get("input")
        row["query"] = " ".join(input_value) if isinstance(input_value, list) else input_value
        if "expected_output" in tc:
            row["ground_truth"] = tc["expected_output"]
        if "context" in tc:
            row["context"] = tc["context"]
        row["id"] = tc.get("id")
        rows.append(row)
    return rows


def _split_metric_groups(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group a result row's ``outputs.<evaluator_key>.<field>`` columns by
    evaluator_key -> {field: value}."""
    groups: Dict[str, Dict[str, Any]] = {}
    for full_key, value in row.items():
        if not full_key.startswith("outputs."):
            continue
        remainder = full_key[len("outputs."):]
        if "." not in remainder:
            continue
        evaluator_key, field = remainder.split(".", 1)
        groups.setdefault(evaluator_key, {})[field] = value
    return groups


def _base_metric_name(fields: Dict[str, Any]) -> Optional[str]:
    """Recover a metric's base name from a group of ``<base><suffix>``
    fields, e.g. {"f1_score": 1.0, "f1_score_score": 1.0,
    "f1_score_passed": True, ...} -> "f1_score". Returns None if no
    consistent base is found (falls back to a bare "score" field, handled
    by the caller)."""
    candidates: Dict[str, int] = {}
    for field in fields:
        base = field
        for suffix in _METRIC_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base and base in fields:  # base itself must be a real, bare field
            candidates[base] = candidates.get(base, 0) + 1
    if not candidates:
        return None
    return max(candidates.items(), key=lambda kv: kv[1])[0]


def _grader_result_for_group(evaluator_key: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    base = _base_metric_name(fields)
    score: Any = None
    passed: Optional[bool] = None
    reason: Optional[str] = None

    if base is not None:
        score = fields.get(base)
        passed = fields.get(f"{base}_passed")
        if passed is None:
            result_str = fields.get(f"{base}_result")
            if isinstance(result_str, str):
                passed = result_str.lower() == "pass"
        reason = fields.get(f"{base}_reason")

    if score is None:
        # No recognizable "<base>_score" shape (e.g. a plain custom
        # evaluator that just returned {"score": True}). Fall back to a
        # bare "score" field if present.
        score = fields.get("score")

    if isinstance(score, bool):
        numeric_score = 1.0 if score else 0.0
        if passed is None:
            passed = score
    elif isinstance(score, (int, float)):
        numeric_score = float(score)
    else:
        # Non-numeric result (e.g. a classification label). EvalPort
        # requires grader_results[].score to be a number -- derived from
        # `passed` if we have it, else conservatively 0.0, with the raw
        # value preserved in `reason` so nothing is silently lost.
        numeric_score = 1.0 if passed else 0.0
        if reason is None and score is not None:
            reason = str(score)

    if passed is None:
        passed = numeric_score >= 0.5

    grader_result: Dict[str, Any] = {
        "grader_id": evaluator_key,
        "type": "custom",
        "score": max(0.0, min(1.0, numeric_score)),
        "passed": bool(passed),
        "metadata": {"azure_ai_evaluation": fields},
    }
    if reason:
        grader_result["reason"] = str(reason)
    return grader_result


def evaluation_result_to_openeval(
    result: Dict[str, Any],
    suite_id: Optional[str] = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert the ``EvaluationResult`` returned by
    ``azure.ai.evaluation.evaluate()`` into a spec-valid EvalPort ResultSet.

    Args:
        result: The dict-like object ``evaluate()`` returns (has ``rows``,
            ``metrics``, ``studio_url`` keys).
        suite_id: Id of the EvalPort suite this run corresponds to.
        run_id: Id for this run. Defaults to a generated uuid4.
        started_at: ISO 8601 timestamp. Defaults to now (UTC).

    Returns:
        A dict matching ``spec/schemas/resultset.json``.
    """
    rows = result.get("rows", [])
    started = started_at or datetime.now(timezone.utc).isoformat()

    results: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        test_case_id = str(row.get("inputs.id", i))
        groups = _split_metric_groups(row)
        grader_results = [
            _grader_result_for_group(evaluator_key, fields) for evaluator_key, fields in groups.items()
        ]
        actual_output = row.get("inputs.response")

        test_result: Dict[str, Any] = {
            "test_case_id": test_case_id,
            "grader_results": grader_results,
            "passed": all(gr["passed"] for gr in grader_results) if grader_results else False,
        }
        if actual_output is not None:
            test_result["actual_output"] = str(actual_output)
        results.append(test_result)

    result_set: Dict[str, Any] = {
        "version": SPEC_VERSION,
        "suite_id": suite_id or "azure-ai-evaluation-run",
        "run_id": run_id or str(uuid.uuid4()),
        "started_at": started,
        "results": results,
        "runner": {"name": "azure-ai-evaluation-openeval-adapter", "version": "0.1.0"},
    }
    if results:
        passed_count = sum(1 for r in results if r["passed"])
        result_set["summary"] = {
            "total": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results),
        }
    metrics = result.get("metrics")
    studio_url = result.get("studio_url")
    if metrics or studio_url:
        result_set["metadata"] = {
            "azure_ai_evaluation": {"metrics": metrics, "studio_url": studio_url}
        }
    return result_set
