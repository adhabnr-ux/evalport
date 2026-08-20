"""Convert Hugging Face `evaluate` (https://github.com/huggingface/evaluate)
metric definitions and computed scores to and from EvalPort
(https://github.com/adhabnr-ux/evalport), the open interchange format for
portable LLM evaluation test cases, graders, suites, and results.

`evaluate`'s core object is an `EvaluationModule` (returned by
`evaluate.load(name)`, covering the library's Metric/Comparison/Measurement
subtypes) whose `.compute(predictions=..., references=..., **kwargs)`
returns an aggregate dict for the whole batch -- there is no per-example
score in that return value, by design (this is the same interface `datasets`
and the Hub's evaluation Spaces use). EvalPort's `ResultSet`, in contrast,
requires one `Result` per test case with its own `GraderResult.score`.

This module does not paper over that gap by fabricating per-example numbers
out of a single aggregate. Instead `compute_per_example()` gets *real*
per-example scores the only honest way available: by calling the same
metric's real `.compute()` once per example (in addition to one real
whole-batch call for the true aggregate), so every number this module
produces came from an actual `evaluate` computation, never an approximation
or interpolation. See the "What round-trips losslessly, and what doesn't"
section of this package's README for which metrics that approach is and
isn't meaningful for.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "to_openeval",
    "from_openeval",
    "compute_per_example",
    "metric_result_to_openeval",
]

_SPEC_VERSION = "1.0.0"
_CUSTOM_HANDLER_PREFIX = "huggingface_evaluate:"


def _import_evaluate():
    try:
        import evaluate  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via importorskip in tests
        raise ImportError(
            "huggingface_evaluate_openeval_adapter requires the `evaluate` package. "
            'Install it with: pip install "huggingface-evaluate-openeval-adapter[evaluate]" '
            "(or `pip install evaluate` directly). See this package's README for details."
        ) from exc
    return evaluate


def _grader_type_for_metric(metric_name: str) -> str:
    """`exact_match` is the one `evaluate` metric with a direct, zero-required-
    -param EvalPort grader equivalent (`exact_match`, per spec/schemas/grader.json
    -- no `allOf` branch means no required params at all). Every other metric
    (accuracy, f1, bleu, rouge, bertscore, ...) takes metric-specific keyword
    arguments this adapter has no honest default for, so it maps to `custom`
    -- the same "don't fabricate required params" rule every adapter in this
    ecosystem follows (see e.g. the haystack and evidently adapters' READMEs)."""
    return "exact_match" if metric_name == "exact_match" else "custom"


def _grader_for_metric(metric_name: str, metric_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    grader_type = _grader_type_for_metric(metric_name)
    if grader_type == "exact_match":
        return {"id": metric_name, "type": "exact_match"}
    return {
        "id": metric_name,
        "type": "custom",
        "params": {
            "handler": f"{_CUSTOM_HANDLER_PREFIX}{metric_name}",
            **({"metric_kwargs": dict(metric_kwargs)} if metric_kwargs else {}),
        },
    }


def _clamp(value: float) -> Tuple[float, bool]:
    """Clamp into EvalPort's required [0, 1] score range. Returns
    (clamped_value, was_clamped) so callers can decide whether to preserve
    the raw value in metadata -- the same clamp-and-preserve pattern the
    evidently and azure-ai-evaluation adapters use."""
    clamped = max(0.0, min(1.0, float(value)))
    return clamped, clamped != float(value)


def to_openeval(
    inputs: Sequence[str],
    references: Sequence[Any],
    metric_name: str,
    *,
    metric_kwargs: Optional[Dict[str, Any]] = None,
    suite_id: Optional[str] = None,
    description: Optional[str] = None,
    test_case_ids: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build an EvalPort suite from the (inputs, references) pair `evaluate`
    itself has no first-class object for -- `EvaluationModule` only models
    "score these predictions against these references," not "here is a
    suite of cases to be scored later." `inputs` is required (not inferred
    from `references`) because EvalPort's `TestCase.input` must be a real,
    non-empty prompt: fabricating one from the reference would misrepresent
    what was actually asked of the system under test.

    `references` may be any type `evaluate`'s metric expects (e.g. `int`
    class labels for `accuracy`/`f1`, not just `str`) -- EvalPort's
    `expected_output` is string-typed by spec, so non-string references are
    stored via `str()` and the original Python type is preserved under
    `metadata.huggingface_evaluate.reference_type` so `from_openeval()` can
    cast back precisely rather than guessing from the string's shape.
    """
    if len(inputs) != len(references):
        raise ValueError(f"inputs and references must be the same length (got {len(inputs)} and {len(references)})")
    if test_case_ids is not None and len(test_case_ids) != len(inputs):
        raise ValueError("test_case_ids must be the same length as inputs")

    grader = _grader_for_metric(metric_name, metric_kwargs)

    test_cases = []
    for i, (inp, ref) in enumerate(zip(inputs, references)):
        case_id = test_case_ids[i] if test_case_ids is not None else f"case_{i}"
        reference_type = type(ref).__name__
        test_case: Dict[str, Any] = {
            "id": case_id,
            "input": inp,
            "expected_output": str(ref),
            "graders": [grader],
            "metadata": {"huggingface_evaluate": {"reference_type": reference_type}},
        }
        if tags:
            test_case["tags"] = list(tags)
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": _SPEC_VERSION,
        "id": suite_id or f"huggingface_evaluate_{metric_name}",
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reverse of `to_openeval()`, grouped by metric -- a suite can
    legitimately mix test cases scored by different metrics (or by graders
    this adapter didn't produce), so this returns one group per distinct
    `huggingface_evaluate:<metric>` handler found, each ready to hand
    straight to `compute_per_example()`.

    Only inline graders of type `exact_match`, or type `custom` whose
    `params.handler` starts with `"huggingface_evaluate:"`, are recognized.
    Everything else (a bare grader-id string, a grader this module didn't
    produce, an `exact_match` grader on a test case with more than one
    grader) is clean-skipped per test case rather than guessed at -- the
    same clean-skip convention every adapter in this ecosystem uses for a
    grader type it doesn't own.
    """
    groups: Dict[str, Dict[str, Any]] = {}

    for test_case in suite.get("test_cases", []):
        graders = test_case.get("graders") or []
        if not graders:
            continue
        grader = graders[0]
        if not isinstance(grader, dict):
            continue  # bare grader-id string -- nothing to reconstruct from

        metric_name: Optional[str] = None
        metric_kwargs: Dict[str, Any] = {}
        if grader.get("type") == "exact_match":
            metric_name = grader.get("id") if grader.get("id") else "exact_match"
            if metric_name != "exact_match":
                # An exact_match-typed grader with some other id -- still a
                # real exact_match check, id is just a label. Honor it.
                pass
            metric_name = "exact_match"
        elif grader.get("type") == "custom":
            handler = (grader.get("params") or {}).get("handler", "")
            if isinstance(handler, str) and handler.startswith(_CUSTOM_HANDLER_PREFIX):
                metric_name = handler[len(_CUSTOM_HANDLER_PREFIX):]
                metric_kwargs = dict((grader.get("params") or {}).get("metric_kwargs") or {})

        if metric_name is None:
            continue

        input_value = test_case.get("input", "")
        input_str = input_value if isinstance(input_value, str) else " ".join(input_value)

        reference_raw = test_case.get("expected_output", "")
        reference_type = (
            (test_case.get("metadata") or {}).get("huggingface_evaluate", {}).get("reference_type")
        )
        reference: Any = reference_raw
        if reference_type == "int":
            try:
                reference = int(reference_raw)
            except (TypeError, ValueError):
                reference = reference_raw
        elif reference_type == "float":
            try:
                reference = float(reference_raw)
            except (TypeError, ValueError):
                reference = reference_raw

        group = groups.setdefault(
            metric_name,
            {"metric_name": metric_name, "test_case_ids": [], "inputs": [], "references": [], "metric_kwargs": metric_kwargs},
        )
        group["test_case_ids"].append(test_case.get("id"))
        group["inputs"].append(input_str)
        group["references"].append(reference)

    return list(groups.values())


def compute_per_example(
    metric_name: str,
    predictions: Sequence[Any],
    references: Sequence[Any],
    **metric_kwargs: Any,
) -> Tuple[List[float], Dict[str, Any]]:
    """Get real, individually-computed per-example scores for a metric that
    only natively exposes a whole-batch aggregate.

    Loads `metric_name` once via `evaluate.load()`, then:
    1. Calls `.compute(predictions=predictions, references=references,
       **metric_kwargs)` exactly once for the real, unmodified aggregate
       `evaluate` itself would report for this batch.
    2. Calls `.compute()` again, once per example (`predictions=[p]`,
       `references=[r]`), to get that example's own real score.

    Every number returned came from the library's own scoring code -- none
    are interpolated, averaged backward, or otherwise fabricated. That said,
    this is only *meaningful* for metrics whose scoring function is honestly
    example-independent (`exact_match`, `accuracy`, `f1` on a per-item
    basis, and similar). For a metric that is inherently a corpus-level
    statistic -- classic corpus BLEU is the standard example, where
    smoothing and brevity-penalty terms are computed over the whole corpus,
    not per sentence -- calling it once per example still returns a real
    number from the real metric, but that number means something subtly
    different from what "BLEU" usually refers to. This function does not
    attempt to detect that distinction (there is no general, reliable way to
    tell from the `evaluate` API alone); the caller is expected to know
    whether their metric's per-example score is the right number to report.
    See the README's "What round-trips losslessly, and what doesn't"
    section.

    Multi-output metrics (e.g. `rouge`, which returns `rouge1`/`rouge2`/
    `rougeL`/`rougeLsum`) report their per-example *and* aggregate score
    under the key that matches `metric_name`, if present, else the first
    key in the result dict -- the rest of the aggregate's keys are still
    returned in full in the second tuple element, nothing is dropped.
    """
    evaluate = _import_evaluate()
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must be the same length (got {len(predictions)} and {len(references)})"
        )

    module = evaluate.load(metric_name)
    aggregate = module.compute(predictions=list(predictions), references=list(references), **metric_kwargs)
    if aggregate is None:
        raise ValueError(f"evaluate metric '{metric_name}' returned no result for this batch")

    primary_key = metric_name if metric_name in aggregate else next(iter(aggregate))

    item_scores: List[float] = []
    for pred, ref in zip(predictions, references):
        per_example = module.compute(predictions=[pred], references=[ref], **metric_kwargs)
        key = primary_key if primary_key in per_example else next(iter(per_example))
        item_scores.append(float(per_example[key]))

    return item_scores, dict(aggregate)


def metric_result_to_openeval(
    predictions: Sequence[Any],
    references: Sequence[Any],
    metric_name: str,
    item_scores: Sequence[float],
    *,
    suite_id: str,
    run_id: str,
    started_at: str,
    aggregate: Optional[Dict[str, Any]] = None,
    test_case_ids: Optional[Sequence[str]] = None,
    completed_at: Optional[str] = None,
    threshold: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an EvalPort ResultSet from real per-example scores (typically
    produced by `compute_per_example()`, but any caller-supplied real
    per-example scores work -- this function does not call `evaluate`
    itself).

    `aggregate`, if supplied, is preserved verbatim under
    `metadata.huggingface_evaluate.aggregate` on the returned ResultSet --
    the real whole-batch number `evaluate` itself would report, kept
    alongside the per-example breakdown rather than only exposing the
    derived view.
    """
    if not (len(predictions) == len(references) == len(item_scores)):
        raise ValueError("predictions, references, and item_scores must all be the same length")
    if test_case_ids is not None and len(test_case_ids) != len(predictions):
        raise ValueError("test_case_ids must be the same length as predictions")

    grader_type = _grader_type_for_metric(metric_name)

    results = []
    passed_count = 0
    score_sum = 0.0
    for i, (pred, score) in enumerate(zip(predictions, item_scores)):
        case_id = test_case_ids[i] if test_case_ids is not None else f"case_{i}"
        clamped, was_clamped = _clamp(score)
        passed = clamped >= threshold
        if passed:
            passed_count += 1
        score_sum += clamped

        grader_result: Dict[str, Any] = {
            "grader_id": metric_name,
            "type": grader_type,
            "score": clamped,
            "passed": passed,
        }
        if was_clamped:
            grader_result["metadata"] = {"huggingface_evaluate": {"raw_score": float(score)}}

        results.append(
            {
                "test_case_id": case_id,
                "actual_output": str(pred),
                "grader_results": [grader_result],
                "passed": passed,
            }
        )

    total = len(results)
    result_set: Dict[str, Any] = {
        "version": _SPEC_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else 0.0,
            "avg_score": (score_sum / total) if total else 0.0,
        },
    }
    if completed_at:
        result_set["completed_at"] = completed_at

    combined_metadata: Dict[str, Any] = dict(metadata or {})
    hf_metadata = dict(combined_metadata.get("huggingface_evaluate", {}))
    if aggregate is not None:
        hf_metadata["aggregate"] = dict(aggregate)
    if hf_metadata:
        combined_metadata["huggingface_evaluate"] = hf_metadata
    if combined_metadata:
        result_set["metadata"] = combined_metadata

    return result_set
