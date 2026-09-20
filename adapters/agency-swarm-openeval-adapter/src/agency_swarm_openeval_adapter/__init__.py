"""agency-swarm <-> EvalPort adapter.

Converts the results of `Agency.get_response()` calls
(https://github.com/VRSEN/agency-swarm) -- `RunResult` objects from the
OpenAI Agents SDK that agency-swarm returns unchanged -- into EvalPort
(https://github.com/adhabnr-ux/evalport) `Result`/`ResultSet` documents, so a
regression suite of `(message, recipient_agent) -> expected_output` test
cases can be tracked as portable, structured eval data instead of ad hoc
assertions scattered across a test file.

Why this exists as a standalone package rather than an agency-swarm core
change: maintainer @nicko-ai closed the proposal
(https://github.com/VRSEN/agency-swarm/issues/772) with "Third-party
integrations live outside this repository -- at most we ship a docs recipe.
This adapter is pure glue on top of `Agency.get_response` and needs no core
change, so it works best as an external package... we can link it from the
docs if you publish it."

## No runtime dependency on agency-swarm or the OpenAI Agents SDK

`Agency.get_response()` returns a `RunResult` dataclass from the `agents`
package (the OpenAI Agents SDK agency-swarm is built on), but this module
never imports `agents` or `agency_swarm` at runtime. Every attribute read
(`final_output`, `context_wrapper.usage`, `last_agent`, the four guardrail
result lists) goes through `getattr()` with a safe default. That means this
adapter works against a real `RunResult`, a hand-built test double, or a
future SDK version that renames or removes a field, without pinning to
either package's release cadence. (Tests DO depend on the real `agents`
package -- see the `test` extra in `pyproject.toml` -- to construct genuine
`RunResult`/`Agent`/`Usage`/guardrail-result instances rather than ad hoc
stand-ins, but that is a test-only dependency, never a runtime one.)

## What this adapter does NOT do

It does not call `Agency.get_response()` itself, and it does not touch
`communication_flows` or agent routing. The caller runs their own suite --
awaiting `get_response()` calls however they like: sequentially,
concurrently, from pytest, from a script -- and hands each
`(test_case, RunResult)` pair (or `(test_case, exception)` on failure) to
this adapter. This is deliberately "pure glue on top of
`Agency.get_response`", exactly as the maintainer scoped it: agency
construction, `communication_flows`, and how a suite is actually run stay
entirely the caller's responsibility.

## Grading

Unlike this repo's other adapters, there's no on-disk or in-SDK "evaluation
config" to translate -- agency-swarm doesn't score its own runs. This
adapter defaults to what the original proposal specified: an `exact_match`
grader comparing `final_output` (coerced to a string; see
`_coerce_output_to_str()` for how structured/Pydantic outputs are handled)
against a test case's `expected_output`, when one is supplied. A test case
with no `expected_output` gets a placeholder `custom` grader
(`agency_swarm:no_expected_output`) instead of a fabricated pass/fail --
callers who want real scoring (an LLM judge, a domain-specific check, a
handoff-path assertion) pass their own grader callables to
`result_to_openeval()` / `build_result_set()` via the `graders` parameter.
Nothing in this module calls an LLM or invents a score.

## Guardrails and the `passed` field

`Agency.get_response()` can trip an input, output, tool-input, or
tool-output guardrail. If any guardrail's `tripwire_triggered` is true, the
Result's `passed` is forced to `False` regardless of grader scores -- a
blocked run is a failed run. Otherwise `passed` follows the supplied
graders (`all()` of their individual `passed` flags); with no graders and no
tripped guardrail, `passed` defaults to `True`, meaning only "the call
completed without raising and without a guardrail block" -- documented
explicitly here so it is never mistaken for a real correctness signal.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0-rc.5"

__all__ = [
    "build_test_suite",
    "suite_to_test_cases",
    "default_exact_match_grader",
    "result_to_openeval",
    "build_result_set",
    "__version__",
]
__version__ = "0.1.0"

Grader = Callable[[Dict[str, Any], Any], Optional[Dict[str, Any]]]


def _coerce_output_to_str(value: Any) -> str:
    """Best-effort stringification of a RunResult.final_output-shaped value.

    `final_output` can be plain text, or a structured (often Pydantic) object
    when the agent uses a typed `output_type`. Every non-string path funnels
    through `json.dumps(..., sort_keys=True)` on a plain dict/list/scalar --
    including the Pydantic case, via `model_dump()` (or, if that's
    unavailable, `json.loads(model_dump_json())`) -- rather than ever
    returning a library's own JSON formatting verbatim. This matters because
    `default_exact_match_grader()` coerces both `result.final_output` (often
    a live model instance) and `test_case["expected_output"]` (often authored
    as a plain dict) through this same function: if a Pydantic instance's
    `model_dump_json()` were returned as-is, its compact formatting would
    differ from `json.dumps()`'s spaced-out formatting for the logically
    identical plain dict, and an exact_match comparison between them would
    spuriously fail despite the data being equal. Never raises -- this is
    used to build human-readable output/expected strings for comparison and
    display, not to round-trip an exact type.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if dumped is not None:
            try:
                return json.dumps(dumped, sort_keys=True, default=str)
            except TypeError:
                pass

    model_dump_json = getattr(value, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return json.dumps(json.loads(model_dump_json()), sort_keys=True, default=str)
        except Exception:
            pass

    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _message_to_openeval_input(message: Any) -> Union[str, List[str]]:
    """Convert a `get_response(message=...)` argument into EvalPort's input shape.

    agency-swarm accepts either a plain string or a list of
    `TResponseInputItem` dicts (the OpenAI Agents SDK's message-item type,
    typically `{"role": ..., "content": ...}` but not guaranteed to always be
    that shape). A string passes through; a list is rendered item-by-item as
    `"role: content"` when both keys are present (mirroring how the
    cannonade-openeval-adapter renders chat messages), and JSON-dumped
    defensively otherwise.
    """
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        items: List[str] = []
        for item in message:
            if isinstance(item, dict) and "role" in item and "content" in item:
                items.append(f"{item.get('role')}: {item.get('content')}")
            else:
                try:
                    items.append(json.dumps(item, sort_keys=True, default=str))
                except TypeError:
                    items.append(str(item))
        return items or [""]
    return _coerce_output_to_str(message)


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    return {
        "requests": getattr(usage, "requests", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _guardrail_result_to_dict(guardrail_result: Any) -> Dict[str, Any]:
    output = getattr(guardrail_result, "output", None)
    output_info = getattr(output, "output_info", None)
    try:
        json.dumps(output_info)
        safe_info = output_info
    except TypeError:
        safe_info = repr(output_info)

    guardrail_obj = getattr(guardrail_result, "guardrail", None)
    name = (
        getattr(guardrail_obj, "name", None)
        or getattr(guardrail_obj, "__name__", None)
        or (type(guardrail_obj).__name__ if guardrail_obj is not None else None)
    )
    return {
        "name": name,
        "tripwire_triggered": bool(getattr(output, "tripwire_triggered", False)),
        "output_info": safe_info,
    }


def _any_tripwire_triggered(*guardrail_lists: Any) -> bool:
    for guardrail_list in guardrail_lists:
        for guardrail_result in guardrail_list or []:
            if _guardrail_result_to_dict(guardrail_result)["tripwire_triggered"]:
                return True
    return False


def default_exact_match_grader(test_case: Dict[str, Any], result: Any) -> Optional[Dict[str, Any]]:
    """Compare `result.final_output` against `test_case["expected_output"]`.

    Returns `None` (no grader_result contributed) when the test case has no
    `expected_output` -- this deliberately never invents a comparison target.
    """
    expected = test_case.get("expected_output")
    if expected is None:
        return None

    actual_str = _coerce_output_to_str(getattr(result, "final_output", None))
    expected_str = expected if isinstance(expected, str) else _coerce_output_to_str(expected)
    matched = actual_str == expected_str
    return {
        "grader_id": f"{test_case.get('id', '')}__exact_match",
        "type": "exact_match",
        "score": 1.0 if matched else 0.0,
        "passed": matched,
        "reason": None if matched else "final_output did not match expected_output",
        "metadata": {"expected": expected_str, "actual": actual_str},
    }


def build_test_suite(
    test_cases: List[Dict[str, Any]],
    *,
    suite_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an EvalPort suite dict from a list of agency-swarm test cases.

    Each `test_cases` entry is `{"id", "input", "recipient_agent",
    "expected_output"? }` -- the same shape `result_to_openeval()` and
    `build_result_set()` expect when a suite is actually run. This function
    only describes the suite (what to run and, optionally, what to expect);
    it never calls `Agency.get_response()`.

    Validate the return value with `openeval.validate.validate_suite()`.
    """
    openeval_test_cases: List[Dict[str, Any]] = []
    graders: List[Dict[str, Any]] = []

    for tc in test_cases:
        tc_id = tc.get("id", "")
        expected = tc.get("expected_output")

        if expected is not None:
            grader: Dict[str, Any] = {
                "id": f"{tc_id}__exact_match",
                "type": "exact_match",
                "params": {"expected": expected if isinstance(expected, str) else _coerce_output_to_str(expected)},
                "description": "agency-swarm default exact_match grader on final_output",
            }
        else:
            # No expected_output for this test case -- it still needs a
            # resolvable grader reference, scoped to THIS test case. A
            # shared placeholder id across the whole suite would dangle
            # whenever any other test case in the same suite does have a
            # real (expected_output-backed) grader -- the same class of bug
            # the cannonade-openeval-adapter's own test suite caught and
            # fixed (see its README/module docstring) before it shipped.
            grader = {
                "id": f"{tc_id}__gr_default",
                "type": "custom",
                "params": {"handler": "agency_swarm:no_expected_output"},
                "description": (
                    "No expected_output supplied for this test case; grade it with a "
                    "custom grader function passed to result_to_openeval()/build_result_set()."
                ),
            }
        graders.append(grader)

        openeval_tc: Dict[str, Any] = {
            "id": tc_id,
            "input": _message_to_openeval_input(tc.get("input", "")),
            "graders": [grader["id"]],
            "metadata": {
                "agency_swarm": {
                    "recipient_agent": tc.get("recipient_agent"),
                    "raw_input": tc.get("input"),
                    "expected_output": expected,
                }
            },
        }
        if expected is not None:
            openeval_tc["expected_output"] = expected if isinstance(expected, str) else _coerce_output_to_str(expected)
        if tc.get("tags"):
            openeval_tc["tags"] = tc["tags"]
        openeval_test_cases.append(openeval_tc)

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name,
        "description": description,
        "test_cases": openeval_test_cases,
        "graders": graders,
        "config": {},
        "metadata": {"openeval": {"source": "agency-swarm"}},
    }


def suite_to_test_cases(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort, lossy reconstruction of the original test_cases list.

    Only recovers what `build_test_suite()` itself put into
    `metadata["agency_swarm"]` -- this is not a general importer for suites
    built by other EvalPort producers, which have no reason to carry that
    block.
    """
    out: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        meta = (tc.get("metadata") or {}).get("agency_swarm", {})
        out.append(
            {
                "id": tc.get("id"),
                "input": meta.get("raw_input", tc.get("input")),
                "recipient_agent": meta.get("recipient_agent"),
                "expected_output": meta.get("expected_output"),
            }
        )
    return out


def result_to_openeval(
    test_case: Dict[str, Any],
    result: Optional[Any] = None,
    *,
    graders: Optional[List[Grader]] = None,
    exception: Optional[BaseException] = None,
) -> Dict[str, Any]:
    """Convert one `Agency.get_response()` outcome into an EvalPort Result dict.

    Exactly one of `result` (the awaited `RunResult`) or `exception` (what
    the caller caught, if `get_response()` raised) should be provided.
    `test_case` is `{"id", "input", "recipient_agent", "expected_output"?}` --
    the same shape passed to `build_test_suite()`.
    """
    test_case_id = test_case.get("id", "")

    if exception is not None:
        # The call itself failed -- build the most honest Result possible
        # from just the exception, rather than fabricating grader output.
        return {
            "test_case_id": test_case_id,
            "passed": False,
            "grader_results": [],
            "error": {"type": type(exception).__name__, "detail": str(exception)},
            "metadata": {"agency_swarm": {"recipient_agent": test_case.get("recipient_agent")}},
        }

    if result is None:
        raise ValueError("result_to_openeval() requires either `result` or `exception`.")

    effective_graders: List[Grader] = graders if graders is not None else [default_exact_match_grader]
    grader_results = [gr for gr in (grader(test_case, result) for grader in effective_graders) if gr is not None]

    input_guardrails = getattr(result, "input_guardrail_results", None) or []
    output_guardrails = getattr(result, "output_guardrail_results", None) or []
    tool_input_guardrails = getattr(result, "tool_input_guardrail_results", None) or []
    tool_output_guardrails = getattr(result, "tool_output_guardrail_results", None) or []
    guardrail_tripped = _any_tripwire_triggered(
        input_guardrails, output_guardrails, tool_input_guardrails, tool_output_guardrails
    )

    if guardrail_tripped:
        # A blocked run is a failed run, regardless of what any grader said
        # about the (possibly partial, possibly absent) final_output.
        passed = False
    elif grader_results:
        passed = all(bool(gr.get("passed")) for gr in grader_results)
    else:
        # No grading criteria supplied and nothing tripped -- the only
        # available signal is that the call completed. This is NOT a
        # correctness signal and is documented as such in the module
        # docstring; it exists so a suite with unscored test cases still
        # produces a valid EvalPort Result rather than an invented failure.
        passed = True

    context_wrapper = getattr(result, "context_wrapper", None)
    usage = getattr(context_wrapper, "usage", None) if context_wrapper is not None else None
    last_agent = getattr(result, "last_agent", None)

    return {
        "test_case_id": test_case_id,
        "passed": passed,
        "grader_results": grader_results,
        "actual_output": _coerce_output_to_str(getattr(result, "final_output", None)),
        "metadata": {
            "agency_swarm": {
                "recipient_agent": test_case.get("recipient_agent"),
                "last_agent": getattr(last_agent, "name", None),
                "usage": _usage_to_dict(usage),
                "new_items_count": len(getattr(result, "new_items", None) or []),
                "raw_responses_count": len(getattr(result, "raw_responses", None) or []),
                "guardrails": {
                    "input": [_guardrail_result_to_dict(g) for g in input_guardrails],
                    "output": [_guardrail_result_to_dict(g) for g in output_guardrails],
                    "tool_input": [_guardrail_result_to_dict(g) for g in tool_input_guardrails],
                    "tool_output": [_guardrail_result_to_dict(g) for g in tool_output_guardrails],
                },
            }
        },
    }


def build_result_set(
    runs: List[Dict[str, Any]],
    *,
    suite_id: str,
    run_id: str,
    started_at: str,
    completed_at: Optional[str] = None,
    graders: Optional[List[Grader]] = None,
    provider: Optional[Dict[str, Any]] = None,
    runner: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one EvalPort ResultSet dict from a list of get_response() outcomes.

    Each `runs` entry is `{"test_case": {...}, "result": RunResult}` for a
    successful call or `{"test_case": {...}, "exception": exc}` for one that
    raised. Validate the return value with
    `openeval.validate.validate_result_set()`.
    """
    if not runs:
        raise ValueError("build_result_set() requires at least one run.")

    results = [
        result_to_openeval(
            run.get("test_case") or {},
            run.get("result"),
            graders=graders,
            exception=run.get("exception"),
        )
        for run in runs
    ]

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "metadata": {"openeval": {"source": "agency-swarm"}, **(metadata or {})},
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    if provider is not None:
        result_set["provider"] = provider
    if runner is not None:
        result_set["runner"] = runner
    return result_set
