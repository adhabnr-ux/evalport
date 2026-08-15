"""Convert between Vertex AI's Gen AI Evaluation Service (`vertexai.evaluation`)
metrics, evaluation instances, and results and EvalPort
(https://github.com/adhabnr-ux/evalport) suites and result sets.

EvalPort is an open interchange format (Apache 2.0) for portable LLM
evaluation datasets: test cases, graders, suites, and results as plain JSON,
shared across evaluation tools (DeepEval, Promptfoo, Inspect AI, AutoGen,
CrewAI, Ragas, LangSmith, Braintrust, MLflow, Opik, Arize Phoenix, Weights &
Biases Weave, UpTrain, Langfuse, Giskard, LlamaIndex, Patronus AI, and now
Vertex AI).

This module has three entry points, matching the shape used by every other
EvalPort adapter in the ecosystem:

    to_openeval(instances, metrics, ...)
        Converts evaluation instances -- everything needed to *define* a
        run, deliberately excluding the "response" field, since the
        response being graded doesn't exist yet at suite-definition time --
        into an EvalPort suite.

    from_openeval(suite)
        Converts an EvalPort suite back into instances plus reconstructed
        ``vertexai.evaluation`` metric objects, ready to hand to
        ``EvalTask(dataset=..., metrics=...).evaluate(...)``.

    batch_eval_result_to_openeval(metrics_table, test_case_ids, metrics, ...)
        Converts a `metrics_table`-shaped ``pandas.DataFrame`` -- the exact
        shape ``EvalResult.metrics_table`` itself uses, one row per
        instance with ``f"{metric_name}/score"``/``f"{metric_name}/explanation"``
        columns (verified directly from ``vertexai/evaluation/_evaluation.py``,
        not guessed) -- into an EvalPort ResultSet.

Grader mapping
--------------

Vertex's evaluation metrics split into three kinds, mapped honestly rather
than force-fit into one grader type:

- ``PointwiseMetric`` -- an LLM-as-judge metric scoring a single response,
  configured with a real, literal prompt template
  (``metric.metric_prompt_template``, which ``str()``s to the full rendered
  instruction/criteria/rubric text Vertex actually sends the judge model --
  unlike several other adapters in this ecosystem, this is a genuine
  extracted prompt, not a synthesized description). Maps onto EvalPort's
  ``llm_judge`` grader type; the real template text is preserved verbatim
  in ``params.prompt``, with the ``{output}``/``{input}``/``{expected}``
  tokens EvalPort's real validator requires appended as an explicit
  addendum block (Vertex's own template uses its own `{response}`/`{prompt}`
  placeholder convention, not EvalPort's, so both are kept side by side
  rather than one silently overwriting the other). ``params.model`` is
  filled with the placeholder ``"vertex-hosted-judge"``, since the judge
  model itself is an internal Vertex service model, never named by the
  SDK -- the *model being judged* is instead supplied separately to
  ``EvalTask.evaluate(model=...)`` at run time, and has no suite-time
  representation to preserve.
- ``CustomMetric`` -- "computed on the client-side using the user-defined
  metric function in SDK only, not by the Vertex Gen AI Evaluation
  Service" (Vertex's own docstring, quoted verbatim). Maps onto
  ``custom``, export-only, since there is no safe generic way to
  reconstruct an arbitrary Python callable from a grader record on import
  -- the same reasoning ``code``/``human`` graders clean-skip across the
  whole EvalPort ecosystem.
- ``PairwiseMetric`` -- judges a *candidate* response against a *baseline*
  response, not one response against a query/context/reference. There is
  no EvalPort grader shape for a two-response comparison, so (matching the
  LlamaIndex adapter's handling of its own ``PairwiseComparisonEvaluator``)
  it maps onto ``custom`` with its full config preserved, exported only.

What round-trips losslessly, and what doesn't
-----------------------------------------------

A ``PointwiseMetric``'s identity (``metric_name``, the full rendered prompt
template text) round-trips exactly -- ``from_openeval()`` reconstructs the
exact same ``PointwiseMetric`` (as a string-template metric; the original
structured ``PointwiseMetricPromptTemplate`` fields -- separate
criteria/rubric/instruction dicts -- are not reconstructed as structured
data, since Vertex's own API accepts an already-rendered string template
just as validly, and the rendered text is everything the metric actually
needs to run) whenever the grader carries this adapter's own
``metadata.vertexai``. ``CustomMetric``/``PairwiseMetric`` cannot be
reconstructed on import (see above) -- their graders are exported so their
presence and config are never silently dropped, but running them again
requires the caller to keep their own reference to the original metric
object.

Instance fields beyond ``prompt``/``reference`` are honestly *not* mapped
onto EvalPort's ``context`` field: Vertex's per-metric instance schema
varies by metric (some need ``context``, others ``instruction``,
``baseline_response``, tool-call fields, and so on), and guessing which
extra key means what would misrepresent the data. Every extra instance
field is instead preserved verbatim under
``test_case.metadata.vertexai.extra_instance_fields``, so nothing is lost,
just not force-mapped onto a field it may not semantically match.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "to_openeval",
    "from_openeval",
    "batch_eval_result_to_openeval",
]

try:
    from vertexai.evaluation import CustomMetric, PairwiseMetric, PointwiseMetric
except ImportError as e:  # pragma: no cover - exercised by the packaging itself
    raise ImportError(
        "vertexai-openeval-adapter requires the 'google-cloud-aiplatform[evaluation]' "
        "package. Install it with: pip install google-cloud-aiplatform[evaluation]"
    ) from e

try:
    from openeval.version import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_EVALPORT_TOKEN_ADDENDUM = (
    "\n\n# EvalPort interchange fields\n"
    "(Added by vertexai-openeval-adapter so this prompt validates against "
    "EvalPort's llm_judge schema -- Vertex's own template above uses its "
    "own {{response}}/{{prompt}} placeholder convention.)\n"
    "Input: {input}\n"
    "Response: {output}\n"
    "Reference answer (if applicable): {expected}\n"
)


def _metric_name(metric: Any) -> str:
    return str(metric)


def _metric_to_grader(metric: Any) -> Dict[str, Any]:
    """Build an EvalPort grader dict describing one Vertex evaluation metric."""
    grader_id = _metric_name(metric)

    if isinstance(metric, PointwiseMetric):
        real_prompt_text = str(metric.metric_prompt_template)
        return {
            "id": grader_id,
            "type": "llm_judge",
            "params": {
                "model": "vertex-hosted-judge",
                "prompt": real_prompt_text + _EVALPORT_TOKEN_ADDENDUM,
            },
            "metadata": {
                "vertexai": {
                    "class": "PointwiseMetric",
                    "metric_name": grader_id,
                    "metric_prompt_template": real_prompt_text,
                }
            },
        }

    if isinstance(metric, PairwiseMetric):
        return {
            "id": grader_id,
            "type": "custom",
            "params": {"handler": grader_id},
            "metadata": {
                "vertexai": {
                    "class": "PairwiseMetric",
                    "metric_name": grader_id,
                    "metric_prompt_template": str(metric.metric_prompt_template),
                    "baseline_model": (
                        str(metric.baseline_model)
                        if getattr(metric, "baseline_model", None)
                        else None
                    ),
                    "note": (
                        "PairwiseMetric compares a candidate response against a "
                        "baseline response -- there is no single-response "
                        "EvalPort grader shape for that, so this is export-only, "
                        "never reconstructed on import."
                    ),
                }
            },
        }

    if isinstance(metric, CustomMetric):
        return {
            "id": grader_id,
            "type": "custom",
            "params": {"handler": grader_id},
            "metadata": {
                "vertexai": {
                    "class": "CustomMetric",
                    "metric_name": grader_id,
                }
            },
        }

    raise TypeError(
        f"to_openeval: unsupported metric type {type(metric).__name__!r} for "
        f"metric {grader_id!r} -- only PointwiseMetric, PairwiseMetric, and "
        "CustomMetric are supported (raw string metric names like "
        "\"rouge_1\"/\"bleu\"/\"exact_match\" are deliberately not accepted: "
        "whether they compute client-side or via a live Vertex API call "
        "could not be verified offline, so this adapter does not guess)."
    )


def _grader_to_metric(grader: Dict[str, Any]) -> Optional[Any]:
    """Reconstruct a Vertex evaluation metric from a grader dict, where possible.

    Only ``llm_judge`` graders carrying this adapter's own
    ``metadata.vertexai`` (i.e. ones this adapter itself exported as a
    PointwiseMetric) can be reconstructed -- a hand-authored ``llm_judge``
    grader with no such metadata, or any ``custom`` grader (CustomMetric or
    PairwiseMetric), has no safe generic reconstruction and is
    clean-skipped, returning ``None``.
    """
    metadata = grader.get("metadata") or {}
    vertex_meta = metadata.get("vertexai")

    if grader.get("type") == "llm_judge" and vertex_meta and vertex_meta.get(
        "class"
    ) == "PointwiseMetric" and vertex_meta.get("metric_prompt_template"):
        return PointwiseMetric(
            metric=vertex_meta.get("metric_name") or grader["id"],
            metric_prompt_template=vertex_meta["metric_prompt_template"],
        )

    return None


def to_openeval(
    instances: Sequence[Dict[str, Any]],
    metrics: Sequence[Any],
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    version: str = "1.0.0",
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an EvalPort suite from evaluation instances and Vertex metrics.

    Args:
        instances: One dict per test case, in the shape Vertex's own
            dataset rows take. Each must have a ``"prompt"`` key (string).
            An optional ``"reference"`` key becomes ``expected_output``.
            Any other keys (``"context"``, ``"instruction"``,
            ``"baseline_response"``, tool-call fields, etc. -- these vary
            per metric) are preserved verbatim under
            ``metadata.vertexai.extra_instance_fields`` rather than guessed
            at (see module docstring). Deliberately excludes
            ``"response"`` even if present -- that's the output being
            graded, which doesn't exist yet at suite-definition time.
        metrics: ``PointwiseMetric``/``PairwiseMetric``/``CustomMetric``
            instances -- every metric becomes one EvalPort grader, applied
            to every test case (the same "run every metric against every
            instance" shape every other batch-style adapter in this
            ecosystem uses). Raw string metric names (``"rouge_1"``,
            ``"bleu"``, ``"exact_match"``, etc.) are not accepted -- see
            ``_metric_to_grader``'s docstring for why.
        ids: Optional explicit test case ids; auto-generated
            (``vertex_tc_<n>``) if omitted.
        suite_id, version, description: EvalPort Suite-level fields.

    Returns:
        A dict matching EvalPort's Suite schema
        (validate with ``openeval.validate.validate_suite``).

    Raises:
        ValueError: if ``instances``/``metrics`` is empty, ``ids`` has a
            mismatched length, or an instance is missing ``"prompt"``.
        TypeError: if a metric is not one of the three supported types.
    """
    if not instances:
        raise ValueError("to_openeval: instances is empty -- nothing to convert.")
    if not metrics:
        raise ValueError("to_openeval: metrics is empty -- nothing to grade with.")
    if ids is not None and len(ids) != len(instances):
        raise ValueError(
            f"to_openeval: ids has length {len(ids)}, expected {len(instances)} "
            "(one entry per instance)."
        )

    graders = [_metric_to_grader(m) for m in metrics]
    grader_ids = [g["id"] for g in graders]

    test_cases = []
    for i, instance in enumerate(instances):
        if "prompt" not in instance:
            raise ValueError(
                f"to_openeval: instance {i} is missing the required 'prompt' key."
            )
        tc_id = ids[i] if ids else f"vertex_tc_{i}"
        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": instance["prompt"],
            "graders": list(grader_ids),
        }
        if instance.get("reference") is not None:
            test_case["expected_output"] = instance["reference"]

        extra_fields = {
            k: v for k, v in instance.items() if k not in ("prompt", "reference", "response")
        }
        if extra_fields:
            test_case["metadata"] = {"vertexai": {"extra_instance_fields": extra_fields}}

        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or "vertex_suite",
        "graders": graders,
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an EvalPort suite back into Vertex-runnable instances.

    Returns a dict with:
        - ``instances``: list of ``{"prompt": ..., "reference": ...}`` dicts
          (``reference`` omitted when absent), with any preserved
          ``extra_instance_fields`` merged back in.
        - ``ids``: list of test case ids, in the same order as ``instances``.
        - ``metrics``: list of every metric this adapter can reconstruct
          (see ``_grader_to_metric`` -- only ``PointwiseMetric`` graders
          this adapter itself exported; ``CustomMetric``/``PairwiseMetric``
          graders are omitted, not fabricated).

    Each test case's ``input`` must be a single string (Vertex's
    ``"prompt"`` instance field is a single string, not the multi-turn
    array shape EvalPort's ``TestCase.input`` also allows) -- a
    multi-turn/array input raises ``ValueError`` naming the offending test
    case, rather than silently collapsing it.

    Raises:
        ValueError: if the suite has no test cases, or any test case's
            ``input`` is not a plain string.
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")

    instances: List[Dict[str, Any]] = []
    ids: List[str] = []

    for tc in test_cases:
        tc_input = tc.get("input")
        if not isinstance(tc_input, str):
            raise ValueError(
                f"from_openeval: test case {tc.get('id')!r} has a non-string "
                "input (Vertex evaluation instances take a single 'prompt' "
                "string, not EvalPort's multi-turn array-of-strings form)."
            )
        instance: Dict[str, Any] = {"prompt": tc_input}
        if tc.get("expected_output") is not None:
            instance["reference"] = tc["expected_output"]

        extra_fields = (
            (tc.get("metadata") or {}).get("vertexai", {}).get("extra_instance_fields")
        )
        if extra_fields:
            instance.update(extra_fields)

        instances.append(instance)
        ids.append(tc.get("id"))

    graders_by_id = {g["id"]: g for g in suite.get("graders", []) if isinstance(g, dict)}
    reconstructed_metrics = []
    for grader in graders_by_id.values():
        metric = _grader_to_metric(grader)
        if metric is not None:
            reconstructed_metrics.append(metric)

    return {"instances": instances, "ids": ids, "metrics": reconstructed_metrics}


def batch_eval_result_to_openeval(
    metrics_table: Any,
    test_case_ids: Sequence[str],
    metrics: Sequence[Any],
    suite_id: str = "vertex_suite",
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    pass_threshold: float = 0.5,
    version: str = "1.0.0",
) -> Dict[str, Any]:
    """Convert a Vertex `metrics_table`-shaped DataFrame into an EvalPort ResultSet.

    Args:
        metrics_table: A ``pandas.DataFrame`` shaped exactly like
            ``EvalResult.metrics_table`` -- one row per instance (aligned
            by position with ``test_case_ids``), with ``f"{metric_name}/score"``
            and optionally ``f"{metric_name}/explanation"`` columns per
            metric (this exact naming convention is read directly out of
            ``vertexai/evaluation/_evaluation.py``, not guessed). A missing
            or ``NaN`` score for a given (row, metric) produces no grader
            result for that case/metric rather than a fabricated pass or
            fail.
        test_case_ids: The EvalPort test case id each row corresponds to.
        metrics: The same ``PointwiseMetric``/``PairwiseMetric``/``CustomMetric``
            instances used to build the suite -- used to decide each
            grader's ``type``, matching ``to_openeval()``'s own mapping so
            a suite and its results agree on grader type.
        suite_id, run_id, started_at, completed_at, version: ResultSet-level
            fields. ``started_at`` is required by the EvalPort schema and
            defaults to the current time if omitted, since
            ``EvalResult`` doesn't expose a run-level start timestamp
            itself -- pass it explicitly if you have a more accurate one.
        pass_threshold: A row/metric passes when ``score >= pass_threshold``.

    Returns:
        A dict matching EvalPort's ResultSet schema
        (validate with ``openeval.validate.validate_result_set``).

    Raises:
        ValueError: if ``metrics_table`` is empty, or its row count doesn't
            match ``len(test_case_ids)``.
    """
    if metrics_table is None or len(metrics_table) == 0:
        raise ValueError(
            "batch_eval_result_to_openeval: metrics_table is empty -- nothing to convert."
        )
    if len(metrics_table) != len(test_case_ids):
        raise ValueError(
            f"batch_eval_result_to_openeval: metrics_table has "
            f"{len(metrics_table)} rows, expected {len(test_case_ids)} "
            "(len(test_case_ids))."
        )

    metric_types = {_metric_name(m): _grader_type_for(m) for m in metrics}

    results_out = []
    rows = metrics_table.to_dict(orient="records")
    for i, (tc_id, row) in enumerate(zip(test_case_ids, rows)):
        grader_results = []
        for metric_name, grader_type in metric_types.items():
            score_col = f"{metric_name}/score"
            if score_col not in row:
                continue
            raw_score = row[score_col]
            if raw_score is None or (isinstance(raw_score, float) and raw_score != raw_score):
                continue  # NaN or missing -- no evidence, no fabricated result

            score = max(0.0, min(1.0, float(raw_score)))
            grader_result: Dict[str, Any] = {
                "grader_id": metric_name,
                "type": grader_type,
                "score": score,
                "passed": score >= pass_threshold,
            }
            explanation_col = f"{metric_name}/explanation"
            if explanation_col in row and row[explanation_col] not in (None, ""):
                grader_result["metadata"] = {"vertexai": {"explanation": row[explanation_col]}}
            grader_results.append(grader_result)

        result_entry: Dict[str, Any] = {
            "test_case_id": tc_id,
            "grader_results": grader_results,
            "passed": (
                all(g["passed"] for g in grader_results) if grader_results else False
            ),
        }
        if row.get("response") is not None:
            result_entry["actual_output"] = row["response"]

        results_out.append(result_entry)

    total = len(results_out)
    passed_count = sum(1 for r in results_out if r["passed"])

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"vertex_run_{uuid.uuid4().hex[:12]}",
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


def _grader_type_for(metric: Any) -> str:
    return "llm_judge" if isinstance(metric, PointwiseMetric) else "custom"
