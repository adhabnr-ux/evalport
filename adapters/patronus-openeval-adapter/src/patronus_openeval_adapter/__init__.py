"""Convert between Patronus AI (`patronus-core` / the `patronus` PyPI package)
evaluators, evaluation inputs, and evaluation results and EvalPort
(https://github.com/adhabnr-ux/evalport) suites and result sets.

EvalPort is an open interchange format (Apache 2.0) for portable LLM
evaluation datasets: test cases, graders, suites, and results as plain JSON,
shared across evaluation tools (DeepEval, Promptfoo, Inspect AI, AutoGen,
CrewAI, Ragas, LangSmith, Braintrust, MLflow, Opik, Arize Phoenix, Weights &
Biases Weave, UpTrain, Langfuse, Giskard, LlamaIndex, and now Patronus).

This module has three entry points, matching the shape used by every other
EvalPort adapter in the ecosystem:

    to_openeval(inputs, evaluators, ...)
        Converts evaluation inputs -- everything needed to *define* a run,
        deliberately excluding the responses being graded, since those
        don't exist yet at suite-definition time -- into an EvalPort suite.

    from_openeval(suite)
        Converts an EvalPort suite back into inputs plus reconstructed
        ``patronus.evals`` evaluator objects, ready to call
        ``evaluator.evaluate(task_input=..., task_output=..., ...)``
        against your own system's real responses.

    batch_eval_result_to_openeval(eval_results, test_case_ids, evaluators, ...)
        Converts a ``{evaluator_name: [EvaluationResult, ...]}`` mapping --
        the natural shape of running several Patronus evaluators over a
        batch of test cases -- into an EvalPort ResultSet.

Grader mapping
--------------

Patronus's evaluator surface splits cleanly into two kinds, and this
adapter maps each honestly rather than force-fitting both into one grader
type:

- ``RemoteEvaluator`` / ``AsyncRemoteEvaluator`` -- Patronus's hosted,
  criteria-driven judges (``evaluator_id_or_alias`` like ``"judge"``,
  ``"lynx"``, ``"hallucination"``; an optional ``criteria`` name selecting
  a specific rubric). These are LLM-as-judge evaluators under the hood, so
  they map onto EvalPort's ``llm_judge`` grader type. Patronus does not
  expose a literal prompt string or model id for a remote evaluator (the
  judge model and prompt template are configured server-side, resolved by
  ``evaluator_id_or_alias``/``criteria`` at call time) -- the same kind of
  gap documented in the Giskard and LlamaIndex adapters for their own
  server/session-resolved LLM judges. ``to_openeval()`` synthesizes an
  honest, human-readable rubric description (always including the
  ``{output}``/``{input}``/``{expected}`` tokens EvalPort's real validator
  requires ``llm_judge`` prompts to contain) rather than fabricating a fake
  extracted prompt, and fills ``params.model`` with the placeholder
  ``"patronus-hosted-judge"``.
- Everything else -- ``Evaluator``/``StructuredEvaluator`` subclasses,
  and functions wrapped with ``@patronus.evaluator()`` -- runs locally
  (deterministic code, an embedding comparison, a regex, or any other
  programmatic check) with no fixed shape EvalPort can generically
  interpret. These map onto ``custom``, with ``params.handler`` naming the
  evaluator's ``canonical_name`` (``get_evaluator_id():get_criteria()``)
  and its full identity preserved under ``metadata.patronus`` for
  reference -- exported only, per the spec's "Custom grader handling" rule,
  since there is no safe, generic way to reconstruct arbitrary evaluator
  code from a grader record (the same reasoning ``code``/``human`` graders
  clean-skip on import across the whole EvalPort ecosystem).

What round-trips losslessly, and what doesn't
-----------------------------------------------

A ``RemoteEvaluator``'s identity (``evaluator_id_or_alias``, ``criteria``,
``explain_strategy``, ``tags``) round-trips exactly through
``metadata.patronus`` -- ``from_openeval()`` reconstructs the exact same
``RemoteEvaluator`` construction, not a generic stand-in. A local
``Evaluator``/``@evaluator()``-wrapped function cannot be reconstructed on
import (see above) -- its grader is exported so its presence and config are
never silently dropped, but running it again requires the caller to keep
their own reference to the original evaluator object and supply it
themselves (e.g. to ``batch_eval_result_to_openeval()``'s ``evaluators``
argument by the same name used in ``to_openeval()``), exactly the same
"you own execution, we own the data shape" boundary every other EvalPort
adapter draws.

``EvaluationResult.score``/``pass_`` map directly onto EvalPort's
``GraderResult.score``/``passed`` (a result with only one of the two gets
the other honestly derived -- see ``_normalize_score_and_passed()`` --
never fabricated as an exact duplicate). ``text_output``, ``explanation``,
``tags``, ``dataset_id``, and ``dataset_sample_id`` -- Patronus-specific
fields with no EvalPort equivalent -- are preserved verbatim under
``grader_result.metadata.patronus`` rather than dropped.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

__all__ = [
    "to_openeval",
    "from_openeval",
    "batch_eval_result_to_openeval",
]

try:
    from patronus.evals import Evaluator, RemoteEvaluator
except ImportError as e:  # pragma: no cover - exercised by the packaging itself
    raise ImportError(
        "patronus-openeval-adapter requires the 'patronus' package. "
        "Install it with: pip install patronus"
    ) from e

try:
    from openeval.version import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"


_LLM_JUDGE_RUBRIC_TEMPLATE = (
    "Judge whether the response satisfies the Patronus evaluator "
    '"{evaluator_id}"{criteria_clause}.\n\n'
    "Input: {{input}}\n"
    "Response: {{output}}\n"
    "Reference answer (if applicable): {{expected}}\n\n"
    "Score the response according to this evaluator's criteria."
)


def _is_remote_evaluator(ev: Any) -> bool:
    return isinstance(ev, RemoteEvaluator)


def _canonical_name(ev: "Evaluator") -> str:
    try:
        return ev.canonical_name
    except Exception:
        return getattr(ev, "evaluator_id", None) or ev.__class__.__qualname__


def _evaluator_to_grader(name: str, ev: "Evaluator") -> Dict[str, Any]:
    """Build an EvalPort grader dict describing one Patronus evaluator."""
    grader_id = name

    if _is_remote_evaluator(ev):
        # Deliberately read the raw `evaluator_id_or_alias`/`criteria`
        # attributes rather than get_evaluator_id()/get_criteria(): the
        # latter raise RuntimeError on an unloaded RemoteEvaluator (loading
        # resolves a criteria revision via a live Patronus API call, which
        # this adapter has no business making just to build a suite).
        evaluator_id = getattr(ev, "evaluator_id_or_alias", None) or ev.get_evaluator_id()
        criteria = getattr(ev, "criteria", None)
        criteria_clause = f' with criteria "{criteria}"' if criteria else ""
        prompt = _LLM_JUDGE_RUBRIC_TEMPLATE.format(
            evaluator_id=evaluator_id, criteria_clause=criteria_clause
        )
        return {
            "id": grader_id,
            "type": "llm_judge",
            "params": {
                "model": "patronus-hosted-judge",
                "prompt": prompt,
            },
            "metadata": {
                "patronus": {
                    "class": ev.__class__.__name__,
                    "evaluator_id_or_alias": evaluator_id,
                    "criteria": criteria,
                    "explain_strategy": getattr(ev, "explain_strategy", None),
                    "tags": getattr(ev, "tags", None),
                }
            },
        }

    return {
        "id": grader_id,
        "type": "custom",
        "params": {"handler": _canonical_name(ev)},
        "metadata": {
            "patronus": {
                "class": ev.__class__.__name__,
                "evaluator_id": ev.get_evaluator_id(),
                "criteria": ev.get_criteria(),
            }
        },
    }


def _grader_to_evaluator(grader: Dict[str, Any]) -> Optional["Evaluator"]:
    """Reconstruct a Patronus evaluator from a grader dict, where possible.

    Only ``llm_judge`` graders carrying this adapter's own
    ``metadata.patronus`` (i.e. ones this adapter itself exported as a
    RemoteEvaluator) can be reconstructed -- a hand-authored ``llm_judge``
    grader with no such metadata, or any ``custom`` grader, has no safe
    generic reconstruction and is clean-skipped, returning ``None``.
    """
    metadata = grader.get("metadata") or {}
    patronus_meta = metadata.get("patronus")

    if grader.get("type") == "llm_judge" and patronus_meta and patronus_meta.get(
        "evaluator_id_or_alias"
    ):
        return RemoteEvaluator(
            patronus_meta["evaluator_id_or_alias"],
            criteria=patronus_meta.get("criteria"),
            tags=patronus_meta.get("tags"),
            explain_strategy=patronus_meta.get("explain_strategy") or "always",
        )

    return None


def to_openeval(
    inputs: Sequence[str],
    evaluators: Mapping[str, "Evaluator"],
    expected_outputs: Optional[Sequence[Optional[str]]] = None,
    contexts_list: Optional[Sequence[Optional[List[str]]]] = None,
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    version: str = "1.0.0",
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an EvalPort suite from evaluation inputs and Patronus evaluators.

    Args:
        inputs: One prompt/task_input per test case.
        evaluators: ``{name: Evaluator}`` -- every evaluator in this mapping
            becomes one EvalPort grader, applied to every test case (the same
            "run every evaluator against every input" shape every other
            batch-style adapter in this ecosystem uses).
        expected_outputs: Optional per-test-case gold answers
            (``EvaluationResult``/``evaluate()``'s ``gold_answer``).
        contexts_list: Optional per-test-case context strings
            (``evaluate()``'s ``task_context``).
        ids: Optional explicit test case ids; auto-generated
            (``patronus_tc_<n>``) if omitted.
        suite_id, version, description: EvalPort Suite-level fields.

    Returns:
        A dict matching EvalPort's Suite schema
        (validate with ``openeval.validate.validate_suite``).

    Raises:
        ValueError: if ``inputs`` is empty, or if ``expected_outputs``/
            ``contexts_list``/``ids`` is provided with a mismatched length.
    """
    if not inputs:
        raise ValueError("to_openeval: inputs is empty -- nothing to convert.")

    n = len(inputs)
    for name, seq in (
        ("expected_outputs", expected_outputs),
        ("contexts_list", contexts_list),
        ("ids", ids),
    ):
        if seq is not None and len(seq) != n:
            raise ValueError(
                f"to_openeval: {name} has length {len(seq)}, expected {n} "
                "(one entry per input)."
            )

    if not evaluators:
        raise ValueError("to_openeval: evaluators is empty -- nothing to grade with.")

    graders = [_evaluator_to_grader(name, ev) for name, ev in evaluators.items()]
    grader_ids = [g["id"] for g in graders]

    test_cases = []
    for i, text_input in enumerate(inputs):
        tc_id = ids[i] if ids else f"patronus_tc_{i}"
        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": text_input,
            "graders": list(grader_ids),
        }
        if expected_outputs is not None and expected_outputs[i] is not None:
            test_case["expected_output"] = expected_outputs[i]
        if contexts_list is not None and contexts_list[i] is not None:
            test_case["context"] = list(contexts_list[i])
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or "patronus_suite",
        "graders": graders,
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an EvalPort suite back into Patronus-runnable inputs.

    Returns a dict with:
        - ``inputs``: list of ``task_input`` strings, one per test case.
        - ``expected_outputs``: list of ``gold_answer`` values (``None`` if absent).
        - ``contexts_list``: list of ``task_context`` lists (``None`` if absent).
        - ``ids``: list of test case ids, in the same order as the above.
        - ``evaluators``: ``{grader_id: Evaluator}`` for every grader this
          adapter can reconstruct (see ``_grader_to_evaluator`` -- only
          ``llm_judge`` graders this adapter itself exported as a
          ``RemoteEvaluator``; ``custom`` graders and hand-authored
          ``llm_judge`` graders are omitted, not fabricated).

    Each test case's ``input`` must be a single string (Patronus's
    ``task_input`` is a single string, not the multi-turn array shape
    EvalPort's ``TestCase.input`` also allows) -- a multi-turn/array input
    raises ``ValueError`` naming the offending test case, rather than
    silently collapsing it.

    Raises:
        ValueError: if the suite has no test cases, or any test case's
            ``input`` is not a plain string.
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")

    inputs: List[str] = []
    expected_outputs: List[Optional[str]] = []
    contexts_list: List[Optional[List[str]]] = []
    ids: List[str] = []

    for tc in test_cases:
        tc_input = tc.get("input")
        if not isinstance(tc_input, str):
            raise ValueError(
                f"from_openeval: test case {tc.get('id')!r} has a non-string "
                "input (Patronus evaluators take a single task_input string, "
                "not EvalPort's multi-turn array-of-strings form)."
            )
        inputs.append(tc_input)
        expected_outputs.append(tc.get("expected_output"))
        contexts_list.append(tc.get("context"))
        ids.append(tc.get("id"))

    graders_by_id = {
        g["id"]: g for g in suite.get("graders", []) if isinstance(g, dict)
    }
    evaluators: Dict[str, "Evaluator"] = {}
    for grader_id, grader in graders_by_id.items():
        reconstructed = _grader_to_evaluator(grader)
        if reconstructed is not None:
            evaluators[grader_id] = reconstructed

    return {
        "inputs": inputs,
        "expected_outputs": expected_outputs,
        "contexts_list": contexts_list,
        "ids": ids,
        "evaluators": evaluators,
    }


def _normalize_score_and_passed(result: Any, pass_threshold: float = 0.5):
    """Derive an honest (score, passed) pair from an EvaluationResult.

    Never fabricates a value that wasn't there: if only one of
    ``score``/``pass_`` is set, the other is derived from it; if neither is
    set, the grader result carries no score and is treated as not passed
    (never as passed-by-default), matching the "no evidence of a pass is a
    fail" convention every other adapter in this ecosystem follows.
    """
    score = getattr(result, "score", None)
    passed = getattr(result, "pass_", None)

    if score is not None:
        score = max(0.0, min(1.0, float(score)))
        if passed is None:
            passed = score >= pass_threshold
    elif passed is not None:
        score = 1.0 if passed else 0.0
    else:
        score = None
        passed = False

    return score, passed


def batch_eval_result_to_openeval(
    eval_results: Mapping[str, Sequence[Optional[Any]]],
    test_case_ids: Sequence[str],
    evaluators: Mapping[str, "Evaluator"],
    suite_id: str = "patronus_suite",
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    pass_threshold: float = 0.5,
    version: str = "1.0.0",
) -> Dict[str, Any]:
    """Convert a batch of Patronus ``EvaluationResult``s into an EvalPort ResultSet.

    Args:
        eval_results: ``{evaluator_name: [EvaluationResult | None, ...]}``,
            one list per evaluator, aligned by index with ``test_case_ids``
            -- the natural shape of running several evaluators over the same
            batch of test cases. ``None`` entries (an evaluator that was
            skipped for a given case -- Patronus's own
            ``@evaluator()``-decorated functions may return ``None`` to mean
            "skipped", per its docstring) produce no grader result for that
            case rather than a fabricated pass or fail.
        test_case_ids: The EvalPort test case id each index corresponds to.
        evaluators: ``{name: Evaluator}`` -- must contain the same keys as
            ``eval_results``; used to decide each grader's ``type``
            (``llm_judge`` vs ``custom``), matching ``to_openeval()``'s own
            mapping so a suite and its results agree on grader type.
        suite_id, run_id, started_at, completed_at, version: ResultSet-level
            fields. ``started_at`` is required by the EvalPort schema and
            defaults to the current time if omitted, since Patronus's
            evaluators don't expose a run-level start timestamp themselves
            -- pass it explicitly if you have a more accurate one from your
            own run harness. ``completed_at`` is left unset unless provided.
        pass_threshold: see ``_normalize_score_and_passed``.

    Returns:
        A dict matching EvalPort's ResultSet schema
        (validate with ``openeval.validate.validate_result_set``).

    Raises:
        ValueError: if ``eval_results`` is empty, or any evaluator's result
            list has a different length than ``test_case_ids``.
    """
    if not eval_results:
        raise ValueError(
            "batch_eval_result_to_openeval: eval_results is empty -- nothing to convert."
        )

    n = len(test_case_ids)
    for name, results in eval_results.items():
        if len(results) != n:
            raise ValueError(
                f"batch_eval_result_to_openeval: eval_results[{name!r}] has "
                f"length {len(results)}, expected {n} (len(test_case_ids))."
            )

    results_out = []
    for i, tc_id in enumerate(test_case_ids):
        grader_results = []
        for name, results in eval_results.items():
            result = results[i]
            if result is None:
                continue

            ev = evaluators.get(name)
            grader_type = "llm_judge" if ev is not None and _is_remote_evaluator(ev) else "custom"
            score, passed = _normalize_score_and_passed(result, pass_threshold)

            meta: Dict[str, Any] = {}
            for field in ("text_output", "explanation", "tags", "dataset_id", "dataset_sample_id"):
                value = getattr(result, field, None)
                if value is not None:
                    meta[field] = value

            grader_results.append(
                {
                    "grader_id": name,
                    "type": grader_type,
                    "score": score,
                    "passed": bool(passed),
                    **({"metadata": {"patronus": meta}} if meta else {}),
                }
            )

        result_entry: Dict[str, Any] = {
            "test_case_id": tc_id,
            "grader_results": grader_results,
            "passed": (
                all(g["passed"] for g in grader_results) if grader_results else False
            ),
        }
        text_outputs = [
            getattr(results[i], "text_output", None)
            for results in eval_results.values()
            if results[i] is not None
        ]
        first_text_output = next((t for t in text_outputs if t), None)
        if first_text_output is not None:
            result_entry["actual_output"] = first_text_output

        results_out.append(result_entry)

    total = len(results_out)
    passed_count = sum(1 for r in results_out if r["passed"])

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"patronus_run_{uuid.uuid4().hex[:12]}",
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
