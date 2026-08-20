"""Convert EleutherAI ``lm-evaluation-harness`` (``lm-eval``) per-document
samples and aggregate results to/from the EvalPort open evaluation format.

lm-evaluation-harness is the de facto standard few-shot evaluation harness
for language models -- the numbers reported in most open model cards and
leaderboards come from it. Its evaluation surface is ``lm_eval.simple_evaluate()``
/ ``lm_eval.evaluator.evaluate()``, which return a single dict conforming to
the ``EvalResults`` TypedDict in ``lm_eval/result_schema.py``. That dict has
two parts relevant here:

- ``results[task_name]`` -- aggregate metrics for the whole task (e.g.
  ``{"acc,none": 0.42, "acc_stderr,none": 0.01}``).
- ``samples[task_name]`` (only present when ``log_samples=True``) -- a list
  of per-document ``SampleResult`` dicts, one entry **per (document, filter)
  pair**. Each carries the document, the gold target, the raw model-request
  arguments, the raw/filtered model responses, which filter produced this
  entry (``"none"``, ``"strict-match"``, ``"flexible-extract"``, ...), and
  the computed metric score(s) for that document under that filter.

This is the real per-document detail EvalPort's ``ResultSet`` schema
requires (one ``Result`` per test case, each with its own
``GraderResult.score`` -- ``spec/schemas/resultset.json``, ``results`` is
``minItems: 1``), so unlike the ``huggingface-evaluate`` adapter in this
same repo, no design workaround is needed here: ``log_samples=True`` output
already has exactly the granularity EvalPort wants. The catch is that
``log_samples`` is opt-in and off by default, so this adapter only works
against runs that passed it.

Everything here was verified directly against the actually-installed
``lm-eval`` package (0.4.12), not against the ``result_schema.py``
docstrings alone -- two real discrepancies were found and are handled
below rather than assumed away:

1. The docstring for ``SampleResult.arguments`` describes a dict shape
   (``{"gen_args_N": {"arg_0": ..., "arg_1": ...}}``); the actually-returned
   value in 0.4.12 is a **list of lists** instead
   (``[[context, continuation_or_gen_kwargs], ...]``, one entry per model
   request for that document). This module reads the real list shape.
2. Per-sample metric scores (e.g. ``sample["exact_match"]``,
   ``sample["acc"]``) come back as ``numpy.float64``, not plain Python
   ``float`` -- not JSON-serializable as-is. Every score this module reads
   is explicitly cast with ``float(...)`` before it goes anywhere near a
   ``Suite``/``ResultSet`` document.

Verified live against real tasks pulled from the Hugging Face Hub using
``lm_eval``'s own built-in ``dummy`` model (a stub that returns random
loglikelihoods / the literal string ``"lol"`` for generation, with no real
model weights involved) via
``simple_evaluate(model="dummy", tasks=[...], log_samples=True)``:
``copa``/``boolq`` (loglikelihood-based multiple choice) and ``gsm8k``
(generation, two filters: ``strict-match``/``flexible-extract``, metric
``exact_match``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "to_openeval",
    "from_openeval",
    "result_to_openeval",
]

_RESERVED_METADATA_KEY = "lm_eval"


def _sample_prompt(sample: Dict[str, Any]) -> str:
    """Extract the real, literal context/prompt string lm-eval sent the
    model for this document.

    ``arguments`` is a list of per-request argument lists. For a generation
    task there is one request: ``[[prompt, gen_kwargs]]``. For a
    loglikelihood-based multiple-choice task there is one request per
    choice: ``[[context, continuation_1], [context, continuation_2], ...]``
    -- the context (``arguments[0][0]``) is shared across all of them, and
    is the actual prompt text asked of the model either way.
    """
    arguments = sample.get("arguments")
    if not arguments or not isinstance(arguments, (list, tuple)) or not arguments[0]:
        raise ValueError(
            f"sample (doc_id={sample.get('doc_id')!r}) has no usable "
            "'arguments' -- was this produced with log_samples=True?"
        )
    first = arguments[0]
    if not isinstance(first, (list, tuple)) or not first:
        raise ValueError(
            f"sample (doc_id={sample.get('doc_id')!r})['arguments'][0] is "
            f"not a non-empty list/tuple: {first!r}"
        )
    prompt = first[0]
    if not isinstance(prompt, str) or prompt == "":
        raise ValueError(
            f"sample (doc_id={sample.get('doc_id')!r})['arguments'][0][0] "
            f"is not a non-empty string: {prompt!r}"
        )
    return prompt


def _sample_target(sample: Dict[str, Any]) -> str:
    target = sample.get("target", "")
    return target if isinstance(target, str) else str(target)


def _grader_id_for(metric_name: str, filter_name: str) -> str:
    """One grader per (metric, filter) pair -- a task like gsm8k computes
    'exact_match' under two different filters ('strict-match' vs
    'flexible-extract'), and those are two meaningfully different scores
    for the same document, not duplicates to collapse."""
    if filter_name and filter_name != "none":
        return f"{metric_name}__{filter_name}"
    return metric_name


def _grader_for(metric_name: str, filter_name: str) -> Dict[str, Any]:
    grader_id = _grader_id_for(metric_name, filter_name)
    # "exact_match" is the one lm-eval metric name that maps directly onto
    # EvalPort's own exact_match grader type -- per spec/schemas/grader.json
    # it is the only grader type with no required params, so nothing here
    # is fabricated. Every other metric (acc, acc_norm, f1, bleu, rouge,
    # perplexity variants, ...) has no honest EvalPort-native equivalent
    # (mapping to e.g. semantic_similarity would require a threshold this
    # module has no basis for), so it maps to custom with the real metric
    # name and filter preserved in params -- same convention every other
    # adapter in this ecosystem uses for a grader type it can't natively
    # represent.
    if metric_name == "exact_match":
        return {"id": grader_id, "type": "exact_match"}
    return {
        "id": grader_id,
        "type": "custom",
        "params": {
            "handler": f"lm-evaluation-harness:{metric_name}",
            "metric_name": metric_name,
            "filter": filter_name,
        },
    }


def to_openeval(
    task_name: str,
    samples: Sequence[Dict[str, Any]],
    *,
    suite_id: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a ``samples[task_name]`` list (from ``simple_evaluate(...,
    log_samples=True)``) into an EvalPort ``Suite``.

    A document can appear more than once in ``samples`` (once per filter,
    e.g. gsm8k's ``strict-match``/``flexible-extract``) -- this produces
    exactly one ``TestCase`` per unique ``doc_id``, deduplicated on the
    document's prompt/target (identical across filters by construction),
    with one grader per (metric, filter) pair seen for that document across
    *all* its sample entries.
    """
    if not samples:
        raise ValueError("samples is empty -- nothing to convert")

    by_doc: Dict[int, Dict[str, Any]] = {}
    graders_by_doc: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for sample in samples:
        doc_id = sample.get("doc_id")
        if doc_id is None:
            raise ValueError(f"sample missing 'doc_id': {sample!r}")
        prompt = _sample_prompt(sample)
        target = _sample_target(sample)
        filter_name = sample.get("filter", "none")
        metrics = sample.get("metrics") or []
        if not metrics:
            raise ValueError(
                f"sample (doc_id={doc_id!r}) has no 'metrics' -- nothing to grade on"
            )

        if doc_id not in by_doc:
            by_doc[doc_id] = {
                "prompt": prompt,
                "target": target,
                "doc": sample.get("doc"),
                "doc_hash": sample.get("doc_hash"),
                "prompt_hash": sample.get("prompt_hash"),
                "target_hash": sample.get("target_hash"),
                "arguments": sample.get("arguments"),
            }
            graders_by_doc[doc_id] = {}

        for metric_name in metrics:
            grader = _grader_for(metric_name, filter_name)
            graders_by_doc[doc_id][grader["id"]] = grader

    test_cases: List[Dict[str, Any]] = []
    for doc_id in sorted(by_doc.keys()):
        info = by_doc[doc_id]
        graders = list(graders_by_doc[doc_id].values())
        test_case: Dict[str, Any] = {
            "id": f"{task_name}_{doc_id}",
            "input": info["prompt"],
            "expected_output": info["target"],
            "graders": graders,
            "metadata": {
                _RESERVED_METADATA_KEY: {
                    "task_name": task_name,
                    "doc_id": doc_id,
                    "doc": info["doc"],
                    "doc_hash": info["doc_hash"],
                    "prompt_hash": info["prompt_hash"],
                    "target_hash": info["target_hash"],
                    "arguments": info["arguments"],
                }
            },
        }
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": "1.0.0",
        "id": suite_id or f"lm_eval_{task_name}",
        "test_cases": test_cases,
    }
    if description is not None:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reverse ``to_openeval()``: recover per-document prompt/target/doc
    info from an EvalPort ``Suite`` that this module produced.

    Returns one dict per test case:
    ``{"doc_id", "task_name", "prompt", "target", "doc", "arguments"}``,
    enough to re-drive an equivalent evaluation loop against the original
    documents. Test cases whose ``metadata`` doesn't carry the
    ``"lm_eval"`` namespace this module writes (i.e. not produced by
    ``to_openeval()`` above) are cleanly skipped, per the same convention
    every other adapter in this ecosystem uses for data it didn't
    originate -- there's no reliable way to reconstruct lm-eval's internal
    ``doc``/``arguments`` shape from an arbitrary third-party TestCase.
    """
    results: List[Dict[str, Any]] = []
    for test_case in suite.get("test_cases", []):
        metadata = test_case.get("metadata") or {}
        info = metadata.get(_RESERVED_METADATA_KEY)
        if not info:
            continue
        results.append(
            {
                "doc_id": info.get("doc_id"),
                "task_name": info.get("task_name"),
                "prompt": test_case.get("input"),
                "target": test_case.get("expected_output"),
                "doc": info.get("doc"),
                "arguments": info.get("arguments"),
            }
        )
    return results


def result_to_openeval(
    task_name: str,
    samples: Sequence[Dict[str, Any]],
    *,
    suite_id: str,
    run_id: str,
    started_at: str,
    aggregate: Optional[Dict[str, Any]] = None,
    completed_at: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Convert a ``samples[task_name]`` list into an EvalPort ``ResultSet``.

    One ``Result`` per unique ``doc_id``, with one ``GraderResult`` per
    (metric, filter) pair found across that document's sample entries.
    Scores are real per-document numbers straight from lm-eval's own
    scoring -- nothing here is derived or interpolated from the aggregate.
    The real ``results[task_name]`` aggregate dict, when supplied, is
    preserved verbatim under
    ``result_set["metadata"]["lm_eval"]["aggregate"]`` so a consumer isn't
    limited to only the per-document view.
    """
    if not samples:
        raise ValueError("samples is empty -- nothing to convert")

    by_doc_results: Dict[int, List[Dict[str, Any]]] = {}

    for sample in samples:
        doc_id = sample.get("doc_id")
        if doc_id is None:
            raise ValueError(f"sample missing 'doc_id': {sample!r}")
        filter_name = sample.get("filter", "none")
        metrics = sample.get("metrics") or []
        if not metrics:
            raise ValueError(
                f"sample (doc_id={doc_id!r}) has no 'metrics' -- nothing to grade on"
            )

        grader_results = by_doc_results.setdefault(doc_id, [])
        for metric_name in metrics:
            raw_value = sample.get(metric_name)
            if raw_value is None:
                # lm-eval sometimes records a metric name without a
                # same-keyed score for the sample (rare, but observed on
                # some rolling-loglikelihood task types) -- skip rather
                # than fabricate a score.
                continue
            score = float(raw_value)
            clamped = min(1.0, max(0.0, score))
            grader_id = _grader_id_for(metric_name, filter_name)
            grader_result: Dict[str, Any] = {
                "grader_id": grader_id,
                "type": "exact_match" if metric_name == "exact_match" else "custom",
                "score": clamped,
                "passed": clamped >= threshold,
            }
            if clamped != score:
                grader_result["metadata"] = {
                    _RESERVED_METADATA_KEY: {"raw_score": score}
                }
            grader_results.append(grader_result)

    results: List[Dict[str, Any]] = []
    for doc_id in sorted(by_doc_results.keys()):
        grader_results = by_doc_results[doc_id]
        results.append(
            {
                "test_case_id": f"{task_name}_{doc_id}",
                "grader_results": grader_results,
                "passed": all(gr["passed"] for gr in grader_results),
            }
        )

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    result_set: Dict[str, Any] = {
        "version": "1.0.0",
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else 0.0,
        },
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    if aggregate is not None:
        result_set["metadata"] = {
            _RESERVED_METADATA_KEY: {"task_name": task_name, "aggregate": aggregate}
        }
    return result_set
