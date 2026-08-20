"""Convert Hugging Face `lighteval` per-document evaluation details and
results to and from EvalPort (https://github.com/adhabnr-ux/evalport), the
open interchange format for portable LLM evaluation test cases, graders,
suites, and results.

`lighteval` is the evaluation library behind the Hugging Face Open LLM
Leaderboard and is built directly on top of `inspect_ai` (it's a declared
hard dependency, and several of its own task definitions -- e.g. `gsm8k` --
are written using `inspect_ai`'s `Sample`/`solver`/`scorer` primitives
directly). EvalPort already has a merged integration into `inspect_ai`
itself; this adapter closes the loop on the other side of that dependency.

Everything below was verified against the actually-installed `lighteval`
0.13.0 package by running a real `Pipeline.evaluate()` call with
`lighteval`'s own built-in `dummy` model (`lighteval.models.dummy`, no real
model weights, but a genuine evaluation loop against a real task/dataset
pulled live from the Hugging Face Hub) -- not against the docstrings alone,
and not mocked.

## A real bug found while building this adapter

A completely fresh `pip install lighteval` today resolves `xxhash` (an
unpinned transitive dependency -- `lighteval`'s own `pyproject.toml` does
not pin it) to whatever is newest on PyPI, currently 4.0.1. `lighteval`'s
`lighteval/logging/info_loggers.py` (`DetailsLogger.log()` and its
neighbours) calls `xxhash.xxh64(doc.query)` and three siblings with a raw
Python `str`, relying on `xxhash`'s old implicit str-to-bytes encoding.
`xxhash` 4.0 removed that: `xxhash.xxh64("hello")` now raises
`TypeError: Strings must be encoded before hashing` on its own, with no
`lighteval` code involved at all -- confirmed by reproducing the bare
`xxhash` call in isolation, then confirming the exact same exception and
traceback line surfaces from a real `Pipeline.evaluate()` call. That makes
`Pipeline.evaluate()` -- the core of `lighteval`'s Python API, and the
method the `accelerate`/`vllm`/`endpoint` CLI backends all call under the
hood -- crash on every run for anyone installing `lighteval` fresh today.
This adapter's own code never imports or calls `xxhash`; the `test` extra
pins `xxhash<4.0` purely so this package's tests can drive a real
`Pipeline.evaluate()` run without hitting that unrelated crash. Already
tracked upstream as huggingface/lighteval#1330, with a fix already up as
huggingface/lighteval#1332 -- independently reproduced and confirmed there
while building this adapter (see the adapter README's "A real bug found
while building this adapter" section for the comment).

## Two more real things this module gets right because they were verified, not assumed

1. **`pipeline.get_results()` does not return per-sample data.** It returns
   the aggregate summary dict (`config_general`/`results`/`versions`/
   `config_tasks`/`summary_tasks`/`summary_general`). The real per-document
   records -- one `DetailsLogger.Detail(doc, model_response, metric)` per
   evaluated sample -- come from the separate `pipeline.get_details()`
   method, which returns `{task_name: [Detail, ...]}`. This module reads
   from `get_details()`'s output, not `get_results()`'s.

2. **Even a classic multiple-choice task is scored generatively in this
   version.** `hellaswag` looks like a `Doc.choices` / `Doc.gold_index`
   loglikelihood task, but its real installed `LightevalTaskConfig` uses
   `Metrics.exact_match` (`metric_name="em"`, backed by
   `ExactMatches(strip_strings=True)` -- confirmed by reading
   `lighteval/metrics/metrics.py` directly), which scores
   `model_response.text` (the model's generated string) against the gold
   choice text, not `model_response.logprobs` /
   `model_response.argmax_logits_eq_gold`. This module reads whichever of
   `model_response.text` / `logprobs` is actually populated for a given
   sample rather than assuming one path from the task shape alone. Because
   `Metrics.exact_match`'s real `metric_name` genuinely is `"em"` backed by
   literal exact-match semantics (not just named similarly), mapping it to
   EvalPort's native `exact_match` grader type is an honest mapping, not a
   guess.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from openeval.types import OPENEVAL_VERSION

__all__ = ["to_openeval", "from_openeval", "result_to_openeval"]


def _gold_indices(doc: Any) -> List[int]:
    """Doc.gold_index is `int | list[int]`. `-1` is lighteval's own
    sentinel for "no gold label" (seen in hellaswag_prompt when the source
    row's label is an empty string) -- dropped rather than treated as a
    real (wraparound) index into `choices`.
    """
    raw = getattr(doc, "gold_index", None)
    if raw is None:
        return []
    indices = raw if isinstance(raw, list) else [raw]
    return [i for i in indices if isinstance(i, int) and i >= 0]


def _gold_text(doc: Any) -> str:
    choices = getattr(doc, "choices", None) or []
    for idx in _gold_indices(doc):
        if 0 <= idx < len(choices):
            return str(choices[idx])
    return ""


def _metric_value(raw: Any) -> float:
    """Every metric score this module has seen from a real run is either a
    plain Python number or a numpy scalar (e.g. `results['results']`'s
    aggregate values come back as `numpy.float64`) -- explicitly cast so
    nothing numpy-typed ends up in a JSON document."""
    return float(raw)


def _clamp_unit(score: float) -> tuple[float, bool]:
    """EvalPort's GraderResult.score is required to be within [0, 1]
    (spec/schemas/resultset.json). Most lighteval sample-level metrics
    ("em", "f1", accuracy-style scores) already are, but this module makes
    no assumption for the full metric catalogue (edit-distance-based and
    translation metrics in particular are not naturally bounded) -- clamp
    defensively and flag it rather than silently emit an out-of-range or
    silently-wrong value."""
    if score < 0.0:
        return 0.0, True
    if score > 1.0:
        return 1.0, True
    return score, False


def _grader_for(metric_name: str) -> Dict[str, Any]:
    if metric_name == "em":
        # lighteval's own metric_name for Metrics.exact_match, backed by
        # ExactMatches(strip_strings=True) -- genuinely exact-match
        # semantics, confirmed by reading lighteval/metrics/metrics.py, not
        # inferred from the name alone.
        return {"id": metric_name, "type": "exact_match"}
    return {
        "id": metric_name,
        "type": "custom",
        "params": {"handler": f"lighteval:{metric_name}"},
    }


def _test_case_ids(details: Sequence[Any]) -> List[str]:
    """One id per `Detail`, keyed on the real `doc.id` lighteval assigned.
    `doc.id` can repeat within one `details` list when a task is run with
    `num_fewshot_seeds > 1` (the same document evaluated under more than
    one few-shot sampling) -- later occurrences get a `_dupN` suffix so
    every id stays unique rather than silently colliding.

    `to_openeval()` and `result_to_openeval()` must be called with the same
    `details` list (in the same order) for a given task so the ids line up
    between the produced Suite and ResultSet -- both functions compute ids
    with this exact same function.
    """
    seen: Dict[Any, int] = {}
    ids: List[str] = []
    for detail in details:
        doc_id = detail.doc.id
        n = seen.get(doc_id, 0)
        seen[doc_id] = n + 1
        ids.append(f"doc_{doc_id}" if n == 0 else f"doc_{doc_id}_dup{n}")
    return ids


def to_openeval(
    task_name: str,
    details: Sequence[Any],
    *,
    suite_id: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict:
    """Convert a list of real `lighteval` `DetailsLogger.Detail` objects
    (from `pipeline.get_details()[task_name]`) into an EvalPort `Suite`.

    Args:
        task_name: The real lighteval task key, e.g. "hellaswag|0" (as it
            appears in `pipeline.get_details()`'s keys).
        details: `pipeline.get_details()[task_name]` -- a list of
            `DetailsLogger.Detail(doc, model_response, metric)`.
        suite_id: Optional explicit suite id; defaults to a slug of
            `task_name`.
        description: Optional human-readable suite description.
    """
    if not details:
        raise ValueError("to_openeval() requires at least one Detail; got an empty list")

    safe_task = task_name.replace("|", "_").replace(":", "_")
    ids = _test_case_ids(details)

    graders_by_id: Dict[str, Dict[str, Any]] = {}
    test_cases: List[Dict[str, Any]] = []

    for tc_id, detail in zip(ids, details):
        doc = detail.doc
        metric = detail.metric or {}

        tc_graders: List[str] = []
        for metric_name in metric.keys():
            grader = _grader_for(metric_name)
            graders_by_id.setdefault(grader["id"], grader)
            tc_graders.append(grader["id"])
        if not tc_graders:
            # A sample with no computed metric yet (e.g. a partially logged
            # run) still needs at least one grader per spec/schemas/testcase.json
            # (minItems: 1) -- fall back to a placeholder custom grader
            # rather than silently drop the sample.
            fallback = {"id": "lighteval_unscored", "type": "custom", "params": {"handler": "lighteval:unscored"}}
            graders_by_id.setdefault(fallback["id"], fallback)
            tc_graders.append(fallback["id"])

        query = getattr(doc, "query", None) or "(empty query)"

        lighteval_meta: Dict[str, Any] = {
            "task_name": getattr(doc, "task_name", task_name),
            "doc_id": doc.id,
            "choices": list(getattr(doc, "choices", None) or []),
            "gold_index": getattr(doc, "gold_index", None),
        }
        instruction = getattr(doc, "instruction", None)
        if instruction:
            lighteval_meta["instruction"] = instruction
        specific = getattr(doc, "specific", None)
        if specific:
            lighteval_meta["specific"] = specific

        tc: Dict[str, Any] = {
            "id": tc_id,
            "input": query,
            "graders": tc_graders,
            "metadata": {"lighteval": lighteval_meta},
        }
        gold_text = _gold_text(doc)
        if gold_text:
            tc["expected_output"] = gold_text

        test_cases.append(tc)

    suite = {
        "version": OPENEVAL_VERSION,
        "id": suite_id or f"suite_lighteval_{safe_task}",
        "name": description or f"lighteval task: {task_name}",
        "graders": list(graders_by_id.values()),
        "test_cases": test_cases,
        "metadata": {"lighteval": {"task_name": task_name}},
    }
    return suite


def from_openeval(suite: Dict) -> List[Dict[str, Any]]:
    """Recover per-document prompt/gold/doc-id information from a `Suite`
    produced by `to_openeval()`. Test cases missing the
    `metadata["lighteval"]` namespace (i.e. not produced by this module)
    are cleanly skipped, the same convention every adapter in this
    ecosystem uses for data it didn't originate.
    """
    recovered: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        meta = (tc.get("metadata") or {}).get("lighteval")
        if not meta:
            continue
        entry: Dict[str, Any] = {
            "test_case_id": tc["id"],
            "doc_id": meta.get("doc_id"),
            "task_name": meta.get("task_name"),
            "query": tc.get("input"),
            "choices": meta.get("choices", []),
            "gold_index": meta.get("gold_index"),
        }
        if "instruction" in meta:
            entry["instruction"] = meta["instruction"]
        if "specific" in meta:
            entry["specific"] = meta["specific"]
        recovered.append(entry)
    return recovered


def result_to_openeval(
    task_name: str,
    details: Sequence[Any],
    *,
    suite_id: str,
    run_id: str,
    started_at: str,
    aggregate: Optional[Dict[str, Any]] = None,
    completed_at: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict:
    """Convert the same `details` list `to_openeval()` was given into an
    EvalPort `ResultSet` -- one `Result` per real evaluated document, one
    `GraderResult` per metric lighteval actually computed for that document.
    Every score is the real per-sample number lighteval itself produced;
    nothing here is derived from, or interpolated out of, the aggregate.

    `aggregate` (the real corpus-level dict from
    `pipeline.get_results()["results"][task_name]`, e.g.
    `{"em": 0.42, "em_stderr": 0.03}`), when supplied, is preserved verbatim
    under `result_set["metadata"]["lighteval"]["aggregate"]`.
    """
    if not details:
        raise ValueError("result_to_openeval() requires at least one Detail; got an empty list")

    ids = _test_case_ids(details)
    results: List[Dict[str, Any]] = []

    for tc_id, detail in zip(ids, details):
        metric = detail.metric or {}
        grader_results: List[Dict[str, Any]] = []
        for metric_name, raw_score in metric.items():
            score = _metric_value(raw_score)
            clamped, was_clamped = _clamp_unit(score)
            gr: Dict[str, Any] = {
                "grader_id": metric_name,
                "type": _grader_for(metric_name)["type"],
                "score": clamped,
                "passed": clamped >= threshold,
            }
            if was_clamped:
                gr["metadata"] = {"lighteval": {"raw_score": score}}
            grader_results.append(gr)

        if not grader_results:
            grader_results.append(
                {
                    "grader_id": "lighteval_unscored",
                    "type": "custom",
                    "score": 0.0,
                    "passed": False,
                }
            )

        results.append(
            {
                "test_case_id": tc_id,
                "grader_results": grader_results,
                "passed": all(gr["passed"] for gr in grader_results),
            }
        )

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
    }
    if completed_at:
        result_set["completed_at"] = completed_at

    lighteval_meta: Dict[str, Any] = {"task_name": task_name}
    if aggregate is not None:
        lighteval_meta["aggregate"] = {k: _metric_value(v) for k, v in aggregate.items()}
    result_set["metadata"] = {"lighteval": lighteval_meta}

    return result_set
