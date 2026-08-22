"""
athina-openeval-adapter
========================

Converts between Athina's LLM-evaluator data model (`athina.evals.llm.*`:
``DoesResponseAnswerQuery``, ``ContextContainsEnoughInformation``, ``Faithfulness``,
``CustomGrader``) and EvalPort (https://github.com/adhabnr-ux/evalport).

This adapter never imports ``athina`` itself. It converts the plain dicts and
``TypedDict``s that Athina's public API already produces and consumes:

- ``DataPoint`` (``athina.loaders.loader.DataPoint``) — a ``TypedDict`` declaring only
  ``response: str``, but used in practice as a ``**kwargs``-style dict carrying whatever
  keys the target evaluator's ``required_args()`` needs. Verified against the real,
  installed ``athina==1.7.39`` package's evaluator source (not guessed from docs):

  - ``DoesResponseAnswerQuery.required_args()`` -> ``["query", "response"]``
  - ``ContextContainsEnoughInformation.required_args()`` -> ``["query", "context"]``
  - ``Faithfulness.required_args()`` -> ``["context", "response"]``
  - ``CustomGrader.required_args()`` -> ``["response"]``

- ``LlmEvalResult`` (``athina.interfaces.result.LlmEvalResult``) — the ``TypedDict`` every
  one of the above evaluators' ``.run()``/``.run_batch()`` returns:
  ``{name, data, failure, reason, runtime, model}``. There is no separate numeric
  confidence score in this shape — Athina's LLM evaluators are pass/fail plus a
  natural-language ``reason``, nothing else. See "Why score is a booleanized 0/1" below.

Why this adapter is scoped to `athina.evals.llm.*` only
---------------------------------------------------------
Athina also ships function evals (regex/contains/PII/...), Ragas-wrapping evals, safety
evals, and conversation evals, each under its own subpackage. Reading the installed
package's source showed those are *not* uniformly built on the same
``LlmEvaluator``/``LlmEvalResult`` contract that the four public LLM evaluators share
(different base classes, different result shapes for at least the function-eval family).
Rather than guess a mapping for surface this adapter hasn't actually verified against real
classes, it covers only the one data model it read and tested against directly: the four
LLM evaluators exported from ``athina.evals`` (``__all__``). If Athina's function/Ragas/
safety/conversation eval result shapes turn out to share enough structure, a follow-up
adapter module can extend this one — see the "What's not covered" section in the README.

Round-trip design
-------------------
``to_openeval()`` takes the same ``data: List[dict]`` you'd pass to
``evaluator.run_batch(data)`` and builds an EvalPort suite from it *before* running
anything — the suite describes the inputs, not yet the outputs.

``result_to_openeval()`` takes that same ``data`` plus the ``List[LlmEvalResult]``
``run_batch()`` returned and builds an EvalPort ``ResultSet`` — now with outputs and
grades.

``from_openeval()`` reconstructs the *input* side only (query/context/expected_response) —
by design it never has a ``response`` to hand back, because in EvalPort's model the
generated output belongs to a ``ResultSet.Result.actual_output``, not to the ``TestCase``
that was scored. That's the real, honest shape of the cross-tool boundary: you can port
someone else's EvalPort test cases *into* Athina's input format, run them through Athina's
LLM, and then convert the results back out with ``result_to_openeval()`` — but you cannot
reconstruct an Athina ``DataPoint`` in one step from a suite alone, because a suite alone
was never run. See "What round-trips losslessly, and what doesn't" in the README.
"""

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["to_openeval", "result_to_openeval", "from_openeval"]

_SUITE_VERSION = "1.0.0"
_RESULTSET_VERSION = "1.0.0"

# The four required_args() sets actually verified against athina==1.7.39's installed
# source. Used only to decide which DataPoint keys are "known" input fields versus
# evaluator-specific extras (e.g. CustomGrader's caller-supplied grading_criteria) that
# get preserved under metadata rather than silently dropped.
_KNOWN_INPUT_KEYS = {"query", "context", "response", "expected_response"}


def _test_case_id(index: int, ids: Optional[Sequence[str]]) -> str:
    if ids is not None:
        return ids[index]
    return f"tc_{index}"


def _build_input_and_metadata(entry: Dict[str, Any]) -> "tuple[Any, Dict[str, Any]]":
    """
    Decide the EvalPort TestCase `input` for one Athina DataPoint/kwargs dict.

    EvalPort requires a non-empty `input`. Athina DataPoints that only carry `response`
    (e.g. for CustomGrader, whose required_args is just ["response"]) have no query at
    all -- there is nothing that is semantically "the input" in that case. Rather than
    fabricate one, this adapter honestly falls back to using `response` itself as the
    input and flags that fallback in metadata, so a consumer of the suite can tell the
    difference between a real query and a borrowed one.
    """
    metadata: Dict[str, Any] = {}
    if "query" in entry and entry["query"] is not None:
        return entry["query"], metadata
    if "response" in entry and entry["response"] is not None:
        metadata["athina.input_synthesized_from_response"] = True
        return entry["response"], metadata
    raise ValueError(
        "Entry has neither 'query' nor 'response' to use as the EvalPort test case "
        "input; athina-openeval-adapter cannot synthesize one from nothing."
    )


def to_openeval(
    data: Sequence[Dict[str, Any]],
    eval_name: str,
    *,
    suite_id: Optional[str] = None,
    suite_name: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    grader_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a list of Athina DataPoint dicts (the same `data` you'd pass to
    `evaluator.run_batch(data)`) into an EvalPort suite, before running anything.

    Args:
        data: list of dicts, each carrying whatever keys the target evaluator's
            required_args() needs -- e.g. {"query": ..., "response": ...} for
            DoesResponseAnswerQuery, {"context": ..., "response": ...} for Faithfulness.
        eval_name: the evaluator's own `.name()` (e.g. "does_response_answer_query",
            "faithfulness", "context_contains_enough_information", or a CustomGrader's
            configured name). Becomes both the suite-level grader's id suffix and its
            `params.handler`.
        suite_id: EvalPort suite id. Defaults to f"athina-{eval_name}".
        suite_name: optional human-readable suite name.
        ids: optional explicit test case ids, same length/order as `data`.
        grader_description: optional human-readable description for the grader.

    Returns:
        A dict conforming to EvalPort's suite.json schema.
    """
    if not data:
        raise ValueError("data must contain at least one entry (EvalSuite.test_cases requires minItems: 1)")
    if ids is not None and len(ids) != len(data):
        raise ValueError(f"ids has {len(ids)} entries but data has {len(data)}")

    grader_id = f"gr_{eval_name}"
    test_cases = []
    for i, entry in enumerate(data):
        input_value, metadata = _build_input_and_metadata(entry)

        test_case: Dict[str, Any] = {
            "id": _test_case_id(i, ids),
            "input": input_value,
            "graders": [grader_id],
        }
        if entry.get("expected_response") is not None:
            test_case["expected_output"] = entry["expected_response"]
        if entry.get("context") is not None:
            test_case["context"] = [entry["context"]]

        extra_keys = {k: v for k, v in entry.items() if k not in _KNOWN_INPUT_KEYS}
        if extra_keys:
            metadata["athina.extra_args"] = extra_keys
        if metadata:
            test_case["metadata"] = metadata

        test_cases.append(test_case)

    grader: Dict[str, Any] = {
        "id": grader_id,
        "type": "custom",
        "params": {"handler": eval_name},
    }
    if grader_description:
        grader["description"] = grader_description

    suite: Dict[str, Any] = {
        "version": _SUITE_VERSION,
        "id": suite_id or f"athina-{eval_name}",
        "test_cases": test_cases,
        "graders": [grader],
    }
    if suite_name:
        suite["name"] = suite_name
    return suite


def result_to_openeval(
    data: Sequence[Dict[str, Any]],
    eval_results: Sequence[Optional[Dict[str, Any]]],
    eval_name: str,
    *,
    suite_id: str,
    run_id: str,
    started_at: str,
    completed_at: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Convert the (data, eval_results) pair from `evaluator.run_batch(data)` into an
    EvalPort ResultSet.

    Args:
        data: the same list passed to to_openeval() / run_batch().
        eval_results: the `List[LlmEvalResult]` (or None entries for errored rows --
            `_run_batch_generator` in athina's own source yields None on a per-row
            exception rather than raising) that run_batch() returned. Must be the same
            length and order as `data`.
        eval_name: same eval_name passed to to_openeval() -- used to rebuild the
            matching grader_id (f"gr_{eval_name}") and test_case ids.
        suite_id: id of the suite this result set is scoring.
        run_id: unique id for this run.
        started_at / completed_at: ISO-8601 timestamps.
        ids: optional explicit test case ids matching what was passed to to_openeval().

    Returns:
        A dict conforming to EvalPort's resultset.json schema.

    Why score is a booleanized 0/1, not a continuous confidence value
    ---------------------------------------------------------------------
    `LlmEvalResult` (verified against the installed athina==1.7.39 source) carries only
    `failure: bool` and a natural-language `reason` -- there is no separate numeric
    confidence/probability field the underlying LLM judge exposes. EvalPort's
    GraderResult.score is `[0, 1] | null`; this adapter reports `1.0` for a pass and
    `0.0` for a fail rather than inventing a confidence value Athina never computed.
    That's a real, documented information loss versus a grader that does expose a
    graded score (like Ragas's 0-1 metrics) -- not a bug in this adapter.
    """
    if len(data) != len(eval_results):
        raise ValueError(
            f"data has {len(data)} entries but eval_results has {len(eval_results)}"
        )
    if not data:
        raise ValueError("data must contain at least one entry (ResultSet.results requires minItems: 1)")
    if ids is not None and len(ids) != len(data):
        raise ValueError(f"ids has {len(ids)} entries but data has {len(data)}")

    grader_id = f"gr_{eval_name}"
    results = []
    passed_count = 0
    score_sum = 0.0
    scored_count = 0

    for i, (entry, eval_result) in enumerate(zip(data, eval_results)):
        test_case_id = _test_case_id(i, ids)
        actual_output = entry.get("response")

        if eval_result is None:
            # athina's own _run_batch_generator yields None for a row that raised during
            # evaluation (logged, not re-raised) -- preserve that as an explicit error
            # rather than silently treating it as either a pass or a fail.
            result: Dict[str, Any] = {
                "test_case_id": test_case_id,
                "grader_results": [
                    {
                        "grader_id": grader_id,
                        "type": "custom",
                        "score": None,
                        "passed": False,
                        "reason": "athina evaluator raised during run_batch for this entry",
                        "metadata": {"athina.errored": True},
                    }
                ],
                "passed": False,
                "error": {
                    "type": "runner_error",
                    "message": "athina evaluator raised during run_batch for this entry",
                    "retryable": True,
                },
            }
            if actual_output is not None:
                result["actual_output"] = actual_output
            results.append(result)
            continue

        failure = bool(eval_result["failure"])
        passed = not failure
        score = 0.0 if failure else 1.0
        passed_count += 1 if passed else 0
        score_sum += score
        scored_count += 1

        grader_result = {
            "grader_id": grader_id,
            "type": "custom",
            "score": score,
            "passed": passed,
            "reason": eval_result.get("reason", ""),
            "metadata": {
                "athina.model": eval_result.get("model"),
                "athina.runtime_ms": eval_result.get("runtime"),
            },
        }

        result = {
            "test_case_id": test_case_id,
            "grader_results": [grader_result],
            "passed": passed,
        }
        if actual_output is not None:
            result["actual_output"] = actual_output
        if eval_result.get("runtime") is not None:
            result["duration_ms"] = eval_result["runtime"]
        results.append(result)

    summary = {
        "total": len(results),
        "passed": passed_count,
        "failed": scored_count - passed_count,
        "skipped": len(results) - scored_count,
        "pass_rate": (passed_count / scored_count) if scored_count else 0.0,
        "avg_score": (score_sum / scored_count) if scored_count else 0.0,
    }

    result_set: Dict[str, Any] = {
        "version": _RESULTSET_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "summary": summary,
    }
    if completed_at:
        result_set["completed_at"] = completed_at
    return result_set


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reconstruct Athina DataPoint-shaped dicts (query/context/expected_response) from an
    EvalPort suite's test cases.

    This deliberately does NOT return a `response` key -- a suite alone was never run,
    so there is no generated output to hand back. The intended flow is:

        entries = from_openeval(suite)
        # ... caller runs their model on each entries[i] to get a response, or calls
        #     evaluator.run_batch() after adding "response" to each entry ...
        for entry, response in zip(entries, responses):
            entry["response"] = response
        eval_results = evaluator.run_batch(entries)
        result_set = result_to_openeval(entries, eval_results, eval_name, ...)

    Each returned dict also carries `metadata["athina.test_case_id"]` so the caller can
    map results back to the original EvalPort test case ids after running Athina.

    Raises:
        ValueError: if `suite` has no test_cases, or a test case's `input` is a list
            (multi-turn input) -- Athina's DataPoint model has no concept of multi-turn
            conversation input for these four LLM evaluators, so that case is rejected
            rather than silently flattened into something wrong.
    """
    test_cases = suite.get("test_cases")
    if not test_cases:
        raise ValueError("suite has no test_cases to convert")

    entries = []
    for tc in test_cases:
        if isinstance(tc["input"], list):
            raise ValueError(
                f"test case {tc.get('id')!r} has multi-turn `input` (a list); "
                "athina-openeval-adapter's four covered LLM evaluators take a single "
                "string query, so this cannot be converted without lossy flattening."
            )

        entry: Dict[str, Any] = {}
        tc_metadata = tc.get("metadata") or {}
        if not tc_metadata.get("athina.input_synthesized_from_response"):
            entry["query"] = tc["input"]
        if tc.get("expected_output") is not None:
            entry["expected_response"] = tc["expected_output"]
        context = tc.get("context")
        if context:
            # EvalPort context is a list of strings; Athina's context-taking evaluators
            # (Faithfulness, ContextContainsEnoughInformation) take a single string.
            # Joining preserves every character losslessly (double-newline separated) --
            # nothing is dropped, but the original list boundaries aren't recoverable
            # from the joined string alone, so this is documented as a partial loss.
            entry["context"] = "\n\n".join(context)

        extra_args = tc_metadata.get("athina.extra_args")
        if extra_args:
            entry.update(extra_args)

        entry.setdefault("metadata", {})["athina.test_case_id"] = tc["id"]
        entries.append(entry)

    return entries
