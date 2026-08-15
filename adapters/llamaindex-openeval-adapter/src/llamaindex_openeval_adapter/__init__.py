"""
llamaindex-openeval-adapter
============================

Convert LlamaIndex (``llama_index.core.evaluation``) evaluators, evaluation
inputs, and ``BatchEvalRunner`` results to and from EvalPort, the open
interchange format for portable LLM evaluation datasets
(https://github.com/adhabnr-ux/evalport).

Three public functions, matching the "definition vs. execution" split used
throughout this repository's adapters:

- :func:`to_openeval` -- converts evaluation *inputs* (queries, references,
  contexts, and the ``evaluators`` mapping) into an EvalPort suite
  definition. This is the same shape ``BatchEvalRunner`` itself takes, so it
  can be called before anything has actually been run.
- :func:`from_openeval` -- the inverse: rebuilds queries/references/contexts
  and, where possible, real LlamaIndex evaluator instances from an EvalPort
  suite, ready to hand straight to ``BatchEvalRunner``.
- :func:`batch_eval_result_to_openeval` -- converts the
  ``Dict[str, List[EvaluationResult]]`` a ``BatchEvalRunner`` run produces
  into an EvalPort ``ResultSet``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["to_openeval", "from_openeval", "batch_eval_result_to_openeval"]

_SPEC_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Evaluator <-> grader-type classification
# ---------------------------------------------------------------------------

# SemanticSimilarityEvaluator is the one deterministic (non-LLM) evaluator in
# llama_index.core.evaluation: it compares embeddings, not an LLM judgment.
_SEMANTIC_SIMILARITY_CLASSES = {"SemanticSimilarityEvaluator"}

# Everything else in the public API is an LLM-as-judge evaluator of one
# response against a query/context/reference. `ResponseEvaluator` and
# `QueryResponseEvaluator` are llama_index's own legacy aliases for
# `FaithfulnessEvaluator` and `RelevancyEvaluator` respectively -- included
# here for clarity even though `type(x).__name__` will only ever report the
# canonical name.
_LLM_JUDGE_CLASSES = {
    "FaithfulnessEvaluator",
    "ResponseEvaluator",
    "RelevancyEvaluator",
    "QueryResponseEvaluator",
    "AnswerRelevancyEvaluator",
    "ContextRelevancyEvaluator",
    "CorrectnessEvaluator",
    "GuidelineEvaluator",
}

# PairwiseComparisonEvaluator judges *two* candidate responses against each
# other rather than one response against a query/context/reference -- there
# is no EvalPort grader shape for a two-response comparison, so it (and any
# other evaluator class this adapter doesn't recognize) exports as a
# "custom" grader instead of being silently dropped.
_OPAQUE_CLASSES = {"PairwiseComparisonEvaluator"}

# Short, honest descriptions of what each built-in judge evaluates, used to
# synthesize a real (not placeholder) `llm_judge` prompt on export.
_LLM_JUDGE_RUBRICS = {
    "FaithfulnessEvaluator": (
        "Is the response faithful to (fully supported by) the provided context? "
        "Answer YES only if every claim in the response is backed by the context, "
        "NO otherwise."
    ),
    "ResponseEvaluator": (
        "Is the response faithful to (fully supported by) the provided context? "
        "Answer YES only if every claim in the response is backed by the context, "
        "NO otherwise."
    ),
    "RelevancyEvaluator": (
        "Is the response relevant to the query, and is it consistent with the "
        "provided context?"
    ),
    "QueryResponseEvaluator": (
        "Is the response relevant to the query, and is it consistent with the "
        "provided context?"
    ),
    "AnswerRelevancyEvaluator": (
        "How relevant is the response to the query? Score how directly and "
        "completely the response addresses what was asked."
    ),
    "ContextRelevancyEvaluator": (
        "How relevant is the retrieved context to the query? Score whether the "
        "context contains the information needed to answer it."
    ),
    "CorrectnessEvaluator": (
        "Score the relevance and correctness of the response against the "
        "reference answer, on a scale of 1 (worst) to 5 (best)."
    ),
}

_DEFAULT_GUIDELINES = (
    "The response should fully answer the query. "
    "The response should avoid being vague or ambiguous. "
    "The response should be specific and use statistics or numbers when possible."
)


def _json_safe(value: Any) -> Any:
    """Recursively convert a value into something ``json.dumps`` can handle,
    used for stashing evaluator config / raw EvaluationResult fields into
    grader/result metadata without losing information."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):  # pydantic (llama_index.core.bridge.pydantic)
        return _json_safe(value.model_dump())
    return str(value)


def _evaluator_class_name(evaluator: Any) -> str:
    return type(evaluator).__name__


def _describe_evaluator(evaluator: Any) -> Dict[str, Any]:
    """Best-effort snapshot of an evaluator's configuration -- enough to
    reconstruct an equivalent instance on import, and to build an honest
    llm_judge prompt on export."""
    cls_name = _evaluator_class_name(evaluator)
    info: Dict[str, Any] = {"class": cls_name}
    if hasattr(evaluator, "_score_threshold"):
        info["score_threshold"] = _json_safe(evaluator._score_threshold)
    if hasattr(evaluator, "_similarity_threshold"):
        info["similarity_threshold"] = _json_safe(evaluator._similarity_threshold)
    if cls_name == "GuidelineEvaluator" and hasattr(evaluator, "_guidelines"):
        info["guidelines"] = _json_safe(evaluator._guidelines)
    return info


def _grader_for_evaluator(name: str, evaluator: Any) -> Dict[str, Any]:
    cls_name = _evaluator_class_name(evaluator)
    info = _describe_evaluator(evaluator)
    grader: Dict[str, Any] = {
        "id": name,
        "description": f"LlamaIndex {cls_name} (evaluator name: {name!r})",
        "metadata": {"llama_index": info},
    }

    if cls_name in _SEMANTIC_SIMILARITY_CLASSES:
        threshold = info.get("similarity_threshold", 0.8)
        grader["type"] = "semantic_similarity"
        grader["params"] = {"threshold": threshold}
        return grader

    if cls_name in _LLM_JUDGE_CLASSES:
        if cls_name == "GuidelineEvaluator":
            rubric = info.get("guidelines") or _DEFAULT_GUIDELINES
        else:
            rubric = _LLM_JUDGE_RUBRICS.get(cls_name, "Judge the quality of the response.")
        prompt = f"{rubric}\n\nQuery: {{input}}\nResponse to evaluate: {{output}}\n"
        if cls_name == "CorrectnessEvaluator":
            prompt += "Reference answer: {expected}\n"
        grader["type"] = "llm_judge"
        # llama_index evaluators resolve their LLM from `Settings.llm` (or a
        # per-evaluator override) at call time rather than storing a model
        # name/id on the evaluator itself, so there is no real value to put
        # here -- same gap the giskard-openeval-adapter documents for
        # giskard's LLMJudge check.
        grader["params"] = {"model": "llamaindex-configured-llm", "prompt": prompt}
        return grader

    # Unrecognized evaluator class (including the opaque
    # PairwiseComparisonEvaluator): never silently drop it, export as
    # "custom" with the full config preserved in metadata instead.
    grader["type"] = "custom"
    grader["params"] = {"handler": f"llama_index.core.evaluation.{cls_name}"}
    return grader


# ---------------------------------------------------------------------------
# to_openeval / from_openeval  (suite definitions)
# ---------------------------------------------------------------------------


def to_openeval(
    queries: Sequence[str],
    evaluators: Mapping[str, Any],
    references: Optional[Sequence[Optional[str]]] = None,
    contexts_list: Optional[Sequence[Optional[Sequence[str]]]] = None,
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    version: str = _SPEC_VERSION,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a batch of LlamaIndex evaluation *inputs* into an EvalPort suite.

    ``queries``, ``references``, and ``contexts_list`` are exactly the shape
    ``BatchEvalRunner(evaluators).aevaluate_response_strs(queries=...,
    contexts_list=..., reference=...)`` takes -- everything needed to *run*
    an evaluation, except the responses themselves, since those don't exist
    yet at suite-definition time (they're produced by the system under test
    and captured later by :func:`batch_eval_result_to_openeval`).

    Each entry in ``evaluators`` (the same ``Dict[str, BaseEvaluator]``
    ``BatchEvalRunner`` takes) becomes one EvalPort grader, applied to every
    test case -- matching how ``BatchEvalRunner`` itself runs every
    evaluator against every query uniformly.
    """
    if not queries:
        raise ValueError("queries must be a non-empty sequence")
    if not evaluators:
        raise ValueError("evaluators must be a non-empty mapping of name -> BaseEvaluator")
    n = len(queries)
    if references is not None and len(references) != n:
        raise ValueError("references must be the same length as queries")
    if contexts_list is not None and len(contexts_list) != n:
        raise ValueError("contexts_list must be the same length as queries")
    if ids is not None and len(ids) != n:
        raise ValueError("ids must be the same length as queries")

    graders = [_grader_for_evaluator(name, evaluator) for name, evaluator in evaluators.items()]
    grader_ids = [g["id"] for g in graders]

    test_cases: List[Dict[str, Any]] = []
    for i, query in enumerate(queries):
        tc_id = ids[i] if ids is not None else f"tc_{i}"
        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": query,
            "graders": list(grader_ids),
        }
        if references is not None and references[i] is not None:
            test_case["expected_output"] = references[i]
        if contexts_list is not None and contexts_list[i] is not None:
            test_case["context"] = list(contexts_list[i])
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or "llamaindex_suite",
        "graders": graders,
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def _grader_to_evaluator(grader: Dict[str, Any], li_eval: Any) -> Optional[Any]:
    gtype = grader.get("type")
    meta = ((grader.get("metadata") or {}).get("llama_index")) or {}
    cls_name = meta.get("class")

    if cls_name in _OPAQUE_CLASSES:
        # Exported as "custom" on purpose (e.g. PairwiseComparisonEvaluator
        # needs a *second* response to compare against, which doesn't fit
        # this adapter's single-response grader shape) -- export-only,
        # never reconstructed on import even though the class name is
        # preserved in metadata for reference.
        return None

    if cls_name and hasattr(li_eval, cls_name):
        cls = getattr(li_eval, cls_name)
        kwargs: Dict[str, Any] = {}
        if cls_name == "CorrectnessEvaluator" and "score_threshold" in meta:
            kwargs["score_threshold"] = meta["score_threshold"]
        if cls_name == "SemanticSimilarityEvaluator" and "similarity_threshold" in meta:
            kwargs["similarity_threshold"] = meta["similarity_threshold"]
        if cls_name == "GuidelineEvaluator" and "guidelines" in meta:
            kwargs["guidelines"] = meta["guidelines"]
        try:
            return cls(**kwargs)
        except Exception:
            return None

    # No round-trip metadata available (a grader authored outside this
    # adapter, or by hand): fall back to a generic reconstruction from the
    # grader `type` alone, rather than refusing to import it.
    if gtype == "semantic_similarity":
        threshold = (grader.get("params") or {}).get("threshold", 0.8)
        return li_eval.SemanticSimilarityEvaluator(similarity_threshold=threshold)
    if gtype == "llm_judge":
        prompt = (grader.get("params") or {}).get("prompt") or _DEFAULT_GUIDELINES
        return li_eval.GuidelineEvaluator(guidelines=prompt)

    # code / human / "model graded" / unrecognized custom: clean-skip,
    # same convention `openeval run` itself uses for a grader type it
    # doesn't know how to execute.
    return None


def from_openeval(suite: Mapping[str, Any]) -> Dict[str, Any]:
    """Rebuild everything needed to run a LlamaIndex ``BatchEvalRunner`` from
    an EvalPort suite.

    Returns a dict::

        {
          "ids": [...],                    # test case ids, in suite order
          "queries": [...],                # test case inputs
          "references": [...] | None,      # expected_output per test case, or None if none set
          "contexts_list": [...] | None,   # context per test case, or None if none set
          "evaluators": {grader_id: BaseEvaluator instance},
        }

    ready to call as::

        result = from_openeval(suite)
        runner = BatchEvalRunner(evaluators=result["evaluators"])
        eval_results = await runner.aevaluate_response_strs(
            queries=result["queries"],
            response_strs=my_app_responses,           # from actually running your app
            contexts_list=result["contexts_list"],
            reference=result["references"],
        )

    Graders this adapter cannot turn into a real evaluator (an inline
    grader of a type with no LlamaIndex equivalent, or a bare grader-id
    string with no inline definition in ``suite["graders"]``) are
    clean-skipped from ``evaluators`` -- ``ids``/``queries``/``references``/
    ``contexts_list`` are always returned in full regardless.
    """
    from llama_index.core import evaluation as li_eval  # lazy import

    grader_defs: Dict[str, Dict[str, Any]] = {}
    for g in suite.get("graders", []) or []:
        if isinstance(g, dict) and "id" in g:
            grader_defs[g["id"]] = g

    ids: List[str] = []
    queries: List[str] = []
    references: List[Optional[str]] = []
    contexts_list: List[Optional[List[str]]] = []
    grader_ids_used: List[str] = []

    for tc in suite.get("test_cases", []) or []:
        tc_input = tc["input"]
        query = tc_input if isinstance(tc_input, str) else " ".join(tc_input)
        ids.append(tc["id"])
        queries.append(query)
        references.append(tc.get("expected_output"))
        ctx = tc.get("context")
        contexts_list.append(list(ctx) if ctx else None)
        for g in tc.get("graders", []) or []:
            gid = g if isinstance(g, str) else g.get("id")
            if not gid:
                continue
            if isinstance(g, dict) and gid not in grader_defs:
                grader_defs[gid] = g
            if gid not in grader_ids_used:
                grader_ids_used.append(gid)

    evaluators: Dict[str, Any] = {}
    for gid in grader_ids_used:
        grader = grader_defs.get(gid)
        if grader is None:
            continue  # bare grader-id string with no inline definition available
        evaluator = _grader_to_evaluator(grader, li_eval)
        if evaluator is not None:
            evaluators[gid] = evaluator

    return {
        "ids": ids,
        "queries": queries,
        "references": references if any(r is not None for r in references) else None,
        "contexts_list": contexts_list if any(c is not None for c in contexts_list) else None,
        "evaluators": evaluators,
    }


# ---------------------------------------------------------------------------
# batch_eval_result_to_openeval  (executed results)
# ---------------------------------------------------------------------------


def _normalize_score(cls_name: str, raw_score: Optional[float]) -> Optional[float]:
    """EvalPort's GraderResult.score is required to be in [0, 1]. Every
    built-in llama_index evaluator except CorrectnessEvaluator already
    emits a score in that range (a similarity, or a 0.0/1.0 pass/fail);
    CorrectnessEvaluator alone scores on a 1-5 scale (default
    score_threshold=4.0), so it needs an explicit rescale rather than being
    passed through (which would fail EvalPort's own validator)."""
    if raw_score is None:
        return None
    if cls_name == "CorrectnessEvaluator":
        normalized = (raw_score - 1.0) / 4.0
    else:
        normalized = raw_score
    return max(0.0, min(1.0, normalized))


def batch_eval_result_to_openeval(
    results: Mapping[str, Sequence[Any]],
    test_case_ids: Sequence[str],
    evaluators: Mapping[str, Any],
    response_strs: Optional[Sequence[str]] = None,
    suite_id: str = "llamaindex_suite",
    run_id: str = "run",
    started_at: str = "1970-01-01T00:00:00Z",
    completed_at: Optional[str] = None,
    version: str = _SPEC_VERSION,
) -> Dict[str, Any]:
    """Convert a ``BatchEvalRunner`` result into an EvalPort ``ResultSet``.

    ``results`` is the ``Dict[str, List[EvaluationResult]]`` returned by
    ``BatchEvalRunner.aevaluate_response_strs()`` (evaluator name -> one
    ``EvaluationResult`` per test case, aligned by index).

    ``test_case_ids`` must align by index with those lists -- normally the
    ``ids`` list :func:`from_openeval` returned, or the ``ids``/synthesized
    ``tc_{i}`` ids :func:`to_openeval` used, so results land on the right
    test cases.

    ``evaluators`` must be the *same* ``{name: BaseEvaluator}`` mapping the
    ``BatchEvalRunner`` was constructed with (or an equivalent one keyed the
    same way) -- an ``EvaluationResult`` carries no record of which
    evaluator class produced it, so this is the only way to know how to
    normalize each grader's score (see :func:`_normalize_score`) and which
    EvalPort grader ``type`` it corresponds to.
    """
    n = len(test_case_ids)
    for name in results:
        if name not in evaluators:
            raise ValueError(
                f"results contains evaluator name {name!r} with no matching entry "
                "in `evaluators` -- pass the same mapping used with BatchEvalRunner"
            )
    for name, per_case in results.items():
        if len(per_case) != n:
            raise ValueError(
                f"results[{name!r}] has {len(per_case)} items, expected {n} "
                "(one per test_case_id)"
            )
    if response_strs is not None and len(response_strs) != n:
        raise ValueError("response_strs must be the same length as test_case_ids")

    evaluator_class_by_name = {name: _evaluator_class_name(ev) for name, ev in evaluators.items()}

    result_rows: List[Dict[str, Any]] = []
    for i, tc_id in enumerate(test_case_ids):
        grader_results: List[Dict[str, Any]] = []
        for name, per_case in results.items():
            r = per_case[i]
            cls_name = evaluator_class_by_name[name]
            dump = r.model_dump() if hasattr(r, "model_dump") else dict(r)

            invalid = bool(dump.get("invalid_result"))
            raw_score = dump.get("score")
            passing = dump.get("passing")

            if invalid:
                # EvaluationResult.invalid_result means the judge's output
                # couldn't be parsed into a verdict -- no verdict reached,
                # same "score: null" semantics EvalPort reserves for a
                # skipped/errored grader rather than a falsely-confident 0.0.
                score: Optional[float] = None
                passed = False
            else:
                score = _normalize_score(cls_name, raw_score)
                if passing is not None:
                    passed = bool(passing)
                elif score is not None:
                    passed = score >= 0.5
                else:
                    passed = False

            grader_type = "llm_judge"
            if cls_name in _SEMANTIC_SIMILARITY_CLASSES:
                grader_type = "semantic_similarity"
            elif cls_name in _OPAQUE_CLASSES:
                grader_type = "custom"

            grader_result: Dict[str, Any] = {
                "grader_id": name,
                "type": grader_type,
                "score": score,
                "passed": passed,
            }

            reason_bits = []
            if dump.get("feedback"):
                reason_bits.append(str(dump["feedback"]))
            if invalid and dump.get("invalid_reason"):
                reason_bits.append(f"invalid_result: {dump['invalid_reason']}")
            if reason_bits:
                grader_result["reason"] = " | ".join(reason_bits)

            grader_result["metadata"] = {
                "llama_index": {
                    "class": cls_name,
                    "raw_score": _json_safe(raw_score),
                    "invalid_result": invalid,
                    "invalid_reason": dump.get("invalid_reason"),
                    "pairwise_source": dump.get("pairwise_source"),
                }
            }
            grader_results.append(grader_result)

        passed_overall = all(gr["passed"] for gr in grader_results) if grader_results else False
        row: Dict[str, Any] = {
            "test_case_id": tc_id,
            "grader_results": grader_results,
            "passed": passed_overall,
        }
        if response_strs is not None:
            row["actual_output"] = response_strs[i]
        result_rows.append(row)

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": result_rows,
    }
    if completed_at:
        result_set["completed_at"] = completed_at
    return result_set
