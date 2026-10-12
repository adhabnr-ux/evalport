"""FlashRAG <-> EvalPort adapter.

Standalone converter between RUC-NLPIR/FlashRAG's dataset/evaluator shapes
and the EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside FlashRAG
itself: proposed in https://github.com/RUC-NLPIR/FlashRAG/issues/246, where
the FlashRAG maintainers (@ignorejjj) said they "don't currently see enough
demand for this integration to justify adding and maintaining it in
FlashRAG" and invited building it "in your own repository" instead -- option
1 from that issue, which is exactly what this package is.

This module is grounded in FlashRAG's actual current shapes, read directly
from the source rather than assumed:

- ``flashrag.dataset.dataset.Item`` -- ``id``, ``question``,
  ``golden_answers: list[str]``, ``choices: list[str]``,
  ``metadata: dict``, and ``output: dict`` (populated via
  ``update_output``/``update_evaluation_score``, read back through
  ``__getattr__`` so ``item.pred`` and ``item.retrieval_result`` both work
  once set).
- ``flashrag.dataset.dataset.Dataset`` -- a list of ``Item`` wrapped with
  list-returning properties (``.question``, ``.golden_answers``, ``.id``,
  ``.output``) and ``__getattr__`` fan-out for everything else (so
  ``dataset.pred`` returns ``[item.pred for item in dataset.data]``).
- ``flashrag.evaluator.evaluator.Evaluator.evaluate(data)`` -- runs each
  configured ``BaseMetric.calculate_metric(data)``, which returns
  ``(aggregate_dict, per_item_score_list)``; the aggregate dict is merged
  into the return value, and each item gets
  ``item.update_evaluation_score(metric_name, score)``, landing at
  ``item.output["metric_score"][metric_name]``. ``to_openeval_resultset()``
  below is meant to run *after* ``evaluate()`` has populated that.
- Multi-choice handling in ``flashrag.evaluator.metrics.BaseMetric.get_dataset_answer``:
  when ``item.choices`` is non-empty, ``item.golden_answers`` holds indices
  into ``choices`` rather than literal answer strings, and the real golden
  text is ``[item.choices[i] for i in item.golden_answers]``. This module
  reproduces that exact resolution (see ``_resolve_golden_answers``) so
  multi-choice items (e.g. GAOKAO-MM) get the right ``expected_output``
  instead of a bare choice index.

One real quirk flagged in the proposal and confirmed against FlashRAG's own
history: ``LLMJudge.calculate_metric`` (``flashrag/evaluator/metrics.py``)
rescaled its raw 0-10 rating as ``score / 10 + 1`` as of the commit the
issue was filed against (2026-09-22), which for a top rating of 10 gives
``2.0``, not ``1.0`` -- outside EvalPort's required ``[0, 1]`` score range.
FlashRAG's maintainers fixed this in commit ``d98e892`` ("fix: correct LLM
judge score parsing", 2026-09-29), a week after the issue was filed and
before their reply -- current ``main`` does plain ``score / 10``, confirmed
by reading ``flashrag/evaluator/metrics.py`` directly. This adapter clamps
every score into ``[0, 1]`` regardless (``_clamp_score``), both as a
defensive measure against that now-fixed bug recurring or against any other
metric producing an out-of-range value, and because EvalPort's
``GraderResult.score`` schema requires it -- silently clamping would mask a
real out-of-range value, so a clamp that actually fires also gets recorded
under the result's ``metadata`` (``flashrag_raw_score``) rather than
disappearing.

Design credit: the ``to_openeval_suite()``/``to_openeval_resultset()`` split
and the ``custom`` grader-per-metric convention mirror
``ares_openeval_adapter`` and ``ragrank_openeval_adapter`` in this same
``adapters/`` directory, per the "type openness" convention referenced in
the original proposal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval_suite", "to_openeval_resultset", "from_openeval", "__version__"]
__version__ = "0.1.0"

GRADER_HANDLER_PREFIX = "flashrag"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    FlashRAG's `Item`/`Dataset` are attribute-based (`__getattr__` fan-out
    into `.output`/`.data`), but a round trip through `from_openeval()` or a
    JSON-loaded FlashRAG `.jsonl` row is a plain dict, so every accessor in
    this module goes through here rather than assuming one shape -- matching
    the pattern `ares_openeval_adapter` and `autogen_openeval_adapter` use.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_golden_answers(item: Any) -> List[str]:
    """Return the real golden-answer text for one FlashRAG `Item`.

    Mirrors `flashrag.evaluator.metrics.BaseMetric.get_dataset_answer`
    exactly: when `item.choices` is non-empty this is a multi-choice sample
    and `item.golden_answers` holds indices into `choices`, not literal
    answer strings -- the real text is `[choices[idx] for idx in idx_list]`.
    Otherwise `golden_answers` is already the list of literal strings.
    """
    choices = _get(item, "choices") or []
    golden_answers = _get(item, "golden_answers") or []
    if not choices:
        return list(golden_answers)
    return [choices[idx] for idx in golden_answers]


def _item_payload(item: Any, index: int, grader_ids: Sequence[str]) -> Dict[str, Any]:
    """Normalize a single FlashRAG `Item` into an EvalPort TestCase dict."""
    item_id = _get(item, "id") or f"tc_{index}"
    question = _get(item, "question") or ""
    golden_answers = _resolve_golden_answers(item)

    tc: Dict[str, Any] = {
        "id": str(item_id),
        "input": question,
        "graders": list(grader_ids),
    }
    if golden_answers:
        # Only `expected_output` (singular) is standard per the EvalPort spec.
        # The full `golden_answers` list (FlashRAG datasets commonly carry more
        # than one acceptable answer) is preserved under metadata rather than
        # collapsed, since EvalPort's `expected_output` is a single string.
        tc["expected_output"] = str(golden_answers[0])

    metadata: Dict[str, Any] = dict(_get(item, "metadata") or {})
    if len(golden_answers) > 1:
        metadata["flashrag_golden_answers"] = golden_answers
    choices = _get(item, "choices") or []
    if choices:
        metadata["flashrag_choices"] = list(choices)
    if metadata:
        tc["metadata"] = metadata

    retrieval_result = _get(item, "retrieval_result")
    if retrieval_result:
        # FlashRAG's retrieval_result is a list of dicts with a "contents" key
        # (consumed this way by Retrieval_Recall/Retrieval_Precision in
        # flashrag/evaluator/metrics.py); EvalPort's retrieval_context is a
        # flat array of strings.
        tc["retrieval_context"] = [
            _get(doc, "contents", "") for doc in retrieval_result
        ]

    return tc


def to_openeval_suite(
    dataset: Any,
    metric_names: Sequence[str],
    suite_id: str = "flashrag_dataset",
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a FlashRAG `Dataset` to an EvalPort-shaped `EvalSuite` (dict).

    `dataset` is a `flashrag.dataset.dataset.Dataset` (or any object/list
    exposing the same shape: an iterable of `Item`-like objects, each with
    `id`/`question`/`golden_answers`/`choices`/`metadata` and an `output`
    dict). `metric_names` are the FlashRAG metric names you intend to run
    against it (e.g. `["em", "f1", "retrieval_recall"]`), matching
    `config["metrics"]` as passed to `flashrag.evaluator.evaluator.Evaluator`.

    Each metric becomes one `custom`-type `Grader`
    (`gr_flashrag_<metric_name>`, `params.handler = "flashrag:<metric_name>"`)
    attached to every test case -- FlashRAG metrics apply uniformly across
    a dataset rather than per-sample, so there is one grader per configured
    metric, not per item. This follows the same "type openness" convention
    `ares_openeval_adapter`/`ragrank_openeval_adapter` use for
    framework-specific scorers that don't map onto EvalPort's ~11 well-known
    grader types (`em`/`acc`/`retrieval_recall_top{k}`/... have no direct
    EvalPort equivalent).

    Returns a plain dict conforming to the EvalPort `EvalSuite` schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    grader_ids = [f"gr_flashrag_{m}" for m in metric_names]
    items = dataset.data if hasattr(dataset, "data") else list(dataset)
    test_cases = [_item_payload(item, i, grader_ids) for i, item in enumerate(items)]

    graders = [
        {
            "id": gid,
            "type": "custom",
            "description": f"FlashRAG {metric!r} metric (flashrag/evaluator/metrics.py)",
            "params": {"handler": f"{GRADER_HANDLER_PREFIX}:{metric}"},
        }
        for gid, metric in zip(grader_ids, metric_names)
    ]

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name or f"FlashRAG dataset ({suite_id})",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "flashrag"}},
    }


def _clamp_score(score: Any) -> Optional[float]:
    """Clamp a FlashRAG per-item metric score into EvalPort's required [0, 1].

    FlashRAG's standard metrics (em/acc/f1/precision/recall/rouge/bleu/
    retrieval_recall_top{k}/retrieval_precision_top{k}) already produce a
    float in [0, 1]. `llm_judge` did not prior to FlashRAG commit `d98e892`
    (see module docstring) -- this clamp is the defensive backstop EvalPort's
    schema requires either way. Returns `None` for a missing/non-numeric
    score rather than raising, since EvalPort allows `null`.
    """
    if score is None or isinstance(score, bool):
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value))


def to_openeval_resultset(
    dataset: Any,
    metric_names: Sequence[str],
    suite_id: str = "flashrag_dataset",
    run_id: str = "flashrag_run",
    aggregate: Optional[Dict[str, float]] = None,
    passed_threshold: float = 1.0,
    runner_name: str = "flashrag-openeval-adapter",
    runner_version: str = __version__,
) -> Dict[str, Any]:
    """Export FlashRAG evaluation output to an EvalPort-shaped `ResultSet` (dict).

    Call this *after* `flashrag.evaluator.evaluator.Evaluator.evaluate(dataset)`
    has run: that call mutates `dataset` in place, populating
    `item.output["metric_score"][metric_name]` for each configured metric
    (via `Item.update_evaluation_score`) and returning the aggregate dict
    (e.g. `{"em": 0.6, "f1": 0.73}`) that `evaluate()` itself returns -- pass
    that dict as `aggregate` to have it folded into `summary.metadata`; it is
    optional because the per-item scores alone are enough to build a valid
    `ResultSet`.

    `test_case_id`s are assigned the same way `to_openeval_suite()` assigns
    them (`item.id`, falling back to positional `tc_{i}`), so a suite built
    by `to_openeval_suite()` and a result set built by this function from the
    same dataset reference the same ids.

    Mapping choices, made explicit here because they are FlashRAG-specific
    and not spec-mandated:

    - `actual_output` is `item.output["pred"]` -- FlashRAG's generated
      answer, set by the pipeline before `Evaluator.evaluate()` runs.
      Omitted (not emitted as `null`) when absent, matching
      `resultset.json`'s `actual_output` being an optional string.
    - Each configured metric becomes one `GraderResult` with
      `grader_id = "gr_flashrag_<metric_name>"`, `type = "custom"`, and
      `score` = that metric's per-item value from
      `item.output["metric_score"][metric_name]`, clamped into `[0, 1]`
      (`_clamp_score` -- see its docstring and the module docstring's note
      on the `llm_judge` rescaling bug this guards against).
    - `grader_results[j].passed = score >= passed_threshold` (default
      `1.0`, i.e. only an exact-match-style score of `1.0` counts as
      passed) -- FlashRAG mixes genuinely binary metrics (`em`, `acc`,
      `retrieval_recall_top{k}`) with continuous ones (`f1`, `rouge-*`,
      `bleu`, `retrieval_precision_top{k}`, `llm_judge`), and EvalPort's
      schema requires a bool here regardless, so callers scoring continuous
      metrics should pass a lower `passed_threshold` (e.g. `0.5`) rather
      than rely on the strict default. A `None` score is never passed.
    - The top-level `result.passed` is `True` only when every configured
      metric passed for that item (`all(...)`) -- an "overall pass" rollup,
      the same convention `ares_openeval_adapter.to_openeval_resultset`
      uses for its own single grader.
    - A missing/un-evaluated metric for a given item (e.g. a metric raised
      inside `Evaluator.evaluate()`, which catches and logs per-metric
      exceptions rather than raising -- see `Evaluator.evaluate`) produces a
      `GraderResult` with `score: None, passed: False` rather than being
      silently dropped, so the gap is visible in the ResultSet itself.
    """
    grader_ids = [f"gr_flashrag_{m}" for m in metric_names]
    items = dataset.data if hasattr(dataset, "data") else list(dataset)

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        item_id = _get(item, "id") or f"tc_{i}"
        output = _get(item, "output") or {}
        metric_scores = output.get("metric_score", {}) if isinstance(output, dict) else {}

        grader_results = []
        for grader_id, metric in zip(grader_ids, metric_names):
            raw_score = metric_scores.get(metric)
            score = _clamp_score(raw_score)
            passed = score is not None and score >= passed_threshold
            gr: Dict[str, Any] = {
                "grader_id": grader_id,
                "type": "custom",
                "score": score,
                "passed": passed,
                "reason": f"flashrag:{metric}={raw_score!r}",
            }
            if raw_score is not None and score is not None and float(raw_score) != score:
                gr["metadata"] = {"flashrag_raw_score": raw_score}
            grader_results.append(gr)

        result: Dict[str, Any] = {
            "test_case_id": str(item_id),
            "passed": all(gr["passed"] for gr in grader_results) if grader_results else False,
            "grader_results": grader_results,
        }
        pred = output.get("pred") if isinstance(output, dict) else None
        if isinstance(pred, str):
            result["actual_output"] = pred
        results.append(result)

    now = datetime.now(timezone.utc).isoformat()
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    summary: Dict[str, Any] = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": passed_count / total if total else 0.0,
    }
    metadata: Dict[str, Any] = {"openeval": {"source": "flashrag"}}
    if aggregate:
        metadata["flashrag_aggregate"] = dict(aggregate)

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": now,
        "completed_at": now,
        "results": results,
        "runner": {"name": runner_name, "version": runner_version},
        "summary": summary,
        "metadata": metadata,
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort `EvalSuite` (dict) into FlashRAG-item-shaped dicts.

    Returns plain dicts matching the constructor shape
    `flashrag.dataset.dataset.Item.__init__` expects
    (`id`/`question`/`golden_answers`/`metadata`) -- pass a list of these as
    the `data=` argument to `flashrag.dataset.dataset.Dataset(...)` to load
    an EvalPort suite as a FlashRAG dataset, or wrap each in `Item(...)`
    directly.

    `expected_output` round-trips to a single-element `golden_answers` list
    unless `metadata.flashrag_golden_answers` is present (the full list
    `to_openeval_suite()`/`_item_payload` preserves there when a FlashRAG
    item originally had more than one golden answer), in which case that
    full list is restored and the `flashrag_golden_answers` /
    `flashrag_choices` bookkeeping keys are popped back out of `metadata`
    before it round-trips to `Item.metadata`. `retrieval_context`, when
    present, round-trips back to FlashRAG's `output["retrieval_result"]`
    shape (a list of `{"contents": ...}` dicts).
    """
    items: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        metadata = dict(tc.get("metadata") or {})

        golden_answers = metadata.pop("flashrag_golden_answers", None)
        if golden_answers is None:
            expected_output = tc.get("expected_output")
            golden_answers = [expected_output] if expected_output is not None else []

        choices = metadata.pop("flashrag_choices", None)

        item: Dict[str, Any] = {
            "id": tc.get("id"),
            "question": tc.get("input"),
            "golden_answers": golden_answers,
        }
        if choices:
            item["choices"] = choices
        if metadata:
            item["metadata"] = metadata

        retrieval_context = tc.get("retrieval_context")
        if retrieval_context:
            item["output"] = {
                "retrieval_result": [{"contents": doc} for doc in retrieval_context]
            }

        items.append(item)
    return items
