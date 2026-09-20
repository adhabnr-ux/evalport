"""Cannonade <-> EvalPort adapter.

Converts Cannonade's on-disk `TestSuite` and `TestRun` JSON files
(https://github.com/cannonade-ai/cannonade) to and from the EvalPort
interchange format (https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than a Cannonade core change:
Cannonade's maintainer passed on adding an EvalPort dependency or a
`TestRun.version` field, but confirmed the on-disk JSON contract is
deliberate and documented ("Local first: prompts, test suites, and test runs
are all stored locally in JSON files") and said "You're welcome to build the
adapter, of course." See the full discussion at
https://github.com/cannonade-ai/cannonade/issues/69.

This adapter reads the plain JSON files Cannonade writes to:

- `~/.cannonade/suites/<id>.json`   -- one `TestSuite` per file
- `~/.cannonade/runs/<id>-<slug>.json` -- one `TestRun` per file

grounded directly against Cannonade's real TypeScript source (not guessed):
`src/shared/app/test-suite.ts` (`TestSuite`, `TestCase`, `TestInput`,
`EvaluationConfig` -- 14 evaluation types), `src/shared/app/test-run.ts`
(`TestRun`, `PerModelRun`, `TestCaseRun`, `ModelRef`), and
`src/shared/app/judge.ts` (`JudgeUsage`).

## An explicit, unresolved caveat (not glossed over)

The maintainer was direct about this: "the layout of `~/.cannonade/*` is an
internal implementation detail, not a published contract... anything reading
those files should expect to break and validate defensively." `TestRun` also
carries no `version` field (unlike `TestSuite.version`), so this adapter has
NO reliable signal for detecting an incompatible on-disk schema change in a
future Cannonade release. Every field access below uses `.get()` with safe
defaults specifically because of this, and a real Cannonade upgrade may still
require an update to this adapter. Pin your Cannonade version if you depend
on this pipeline for anything long-running.

## Why two conversion entry points instead of one `to_openeval()`

Unlike most adapters in this repo, Cannonade produces two structurally
different artifacts that map onto two different EvalPort documents: a
`TestSuite` (maps to an EvalPort Suite) and a `TestRun` (maps to N EvalPort
ResultSets, one per model). Forcing both through one `to_openeval()` would
require guessing the caller's intent from shape alone, so this adapter
exposes `suite_to_openeval()` and `run_to_openeval()` explicitly, plus a
`to_openeval()` dispatcher for callers who don't care which they're passing.

## Mapping notes

- `TestRun.modelRuns[]` (`PerModelRun`) each become one EvalPort `ResultSet`,
  all sharing `suite_id = TestRun.suiteId` -- exactly the "compare N models
  against one suite" shape EvalPort's grouped/sibling ResultSets feature
  (RFC Discussion #45, merged in PR #54) was built for. Each ResultSet's
  `group` field ties all of one TestRun's PerModelRuns together via
  `group_id = TestRun.id`, with `role`/`label` set to a human-readable model
  identifier and `sequence` set to the PerModelRun's position in
  `modelRuns[]`.
- `EvaluationConfig.type` maps directly to an EvalPort standard grader type
  for `exact_match`, `contains`, `regex`, and `cosine_similarity` (->
  `semantic_similarity`). The other nine types (`json_match`, `bleu`,
  `rouge`, `levenshtein`, `f1`, `custom`, `code_execution`,
  `html_validation`, `llm_rubric`, `g_eval`) have no first-class EvalPort
  grader type in v1 and are mapped to `type="custom"` with
  `params.handler="cannonade:<type>"`, the same pattern the
  ragas-openeval-adapter uses for Ragas metrics without a 1:1 EvalPort type,
  rather than forcing a fake direct mapping. `llm_rubric`/`g_eval` are
  included in this "custom" bucket rather than `llm_judge` because
  Cannonade's suite files don't record which judge model will score a given
  rubric (that's a run-time/app-level setting, not per-test-case), and
  `llm_judge`'s EvalPort schema requires a concrete `params.model` -- putting
  a fabricated model name there would misrepresent something this adapter
  doesn't actually know. The real judge model *used*, when there is one, is
  preserved verbatim per-result via `JudgeUsage` (see below).
- `TestCase.passingLogic` ('all' | 'any') has no EvalPort equivalent --
  EvalPort graders score independently. It rides along as
  `TestCase.metadata["cannonade"]["passing_logic"]`, informational only, and
  the already-computed `Result.passed` (from Cannonade's own
  `TestCaseResult.passed`) is carried through as-is rather than re-derived,
  so this adapter never second-guesses Cannonade's own pass/fail logic.
- `AggregateMetrics` (Cannonade's own per-model summary) is preserved
  verbatim as `ResultSet.summary` rather than recomputed.
- `JudgeUsage` (LLM-judge cost/token info, when a rubric/g_eval grader used
  one) is preserved verbatim in each `GraderResult.metadata["judge_usage"]`.
- Cannonade's `TestSuite.version` (the suite CONTENT's own version, not a
  spec version) is preserved in `metadata["cannonade"]["suite_version"]`;
  EvalPort's own top-level `version` field is always the EvalPort spec
  version this document targets (`OPENEVAL_VERSION`), matching every other
  adapter in this repo.

## Lossiness of `openeval_to_suite()`

The reverse direction can only round-trip what this adapter itself put into
`metadata["cannonade"]`. It is not a general EvalPort -> Cannonade importer
for suites built by other producers: EvalPort's schema has no equivalent for
`RunConfig` sampling parameters, `promptRef`, or `timeoutMs`, so those are
recovered only when this adapter's own `raw` metadata block is present.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0-rc.5"

__all__ = [
    "suite_to_openeval",
    "run_to_openeval",
    "to_openeval",
    "openeval_to_suite",
    "__version__",
]
__version__ = "0.1.0"

# EvaluationConfig.type values with a direct EvalPort standard grader type.
_DIRECT_GRADER_TYPES = {
    "exact_match": "exact_match",
    "contains": "contains",
    "regex": "regex",
    "cosine_similarity": "semantic_similarity",
}
# Everything else (json_match, bleu, rouge, levenshtein, f1, custom,
# code_execution, html_validation, llm_rubric, g_eval) maps to "custom" --
# see module docstring for why llm_rubric/g_eval are included here.


def _clip_unit(value: Optional[float]) -> Optional[float]:
    """Clip a score into EvalPort's required [0, 1] grader-score range.

    Cannonade's own EvaluationMethodResult.score is documented as a plain
    number, not guaranteed to already be in [0,1] for every one of its 14
    evaluation types (e.g. a `custom` JS validator can return anything).
    Rather than let an out-of-range value silently fail
    validate_result_set(), this clips defensively and preserves the raw
    value in metadata so nothing is lost.
    """
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _grader_id(test_case_id: str, index: int, eval_type: str) -> str:
    """Shared, collision-safe id scheme used on both the suite and result side.

    Scoped by test_case_id so it is unique across an entire suite (graders
    are not shared/reused across Cannonade test cases the way they can be in
    EvalPort's model), and by index+type so the same scheme independently
    reproduced from a TestRun's `evalResults` (which carries `type` per
    entry but not the suite's grader definition) still lines up with the
    suite side, letting the two be correlated without passing the suite
    into run_to_openeval().
    """
    return f"{test_case_id}__ec{index}_{eval_type}"


def _eval_config_to_grader(test_case_id: str, index: int, ec: Dict[str, Any]) -> Dict[str, Any]:
    ec_type = ec.get("type", "custom")
    grader_type = _DIRECT_GRADER_TYPES.get(ec_type, "custom")
    params: Dict[str, Any] = {}

    expected = ec.get("expected")
    if grader_type == "contains":
        params["substring"] = expected if isinstance(expected, str) else json.dumps(expected)
    elif grader_type == "regex":
        params["pattern"] = expected if isinstance(expected, str) else json.dumps(expected)
    elif grader_type == "semantic_similarity":
        params["threshold"] = ec.get("threshold", 0.8)
        if expected is not None:
            params["expected"] = expected
    elif grader_type == "custom":
        params["handler"] = f"cannonade:{ec_type}"
        if expected is not None:
            params["expected"] = expected
        if ec.get("llmRubric"):
            params["rubric"] = ec["llmRubric"].get("rubric")
        if ec.get("gEval"):
            params["criteria"] = ec["gEval"].get("criteria")
        if ec.get("codeExecution"):
            params["code_execution"] = ec["codeExecution"]
        if ec.get("htmlValidation"):
            params["html_validation"] = ec["htmlValidation"]
        if ec.get("customValidator"):
            params["custom_validator"] = ec["customValidator"]
    else:  # exact_match and anything else with no required params
        if expected is not None:
            params["expected"] = expected

    if ec.get("negate") is not None:
        params["negate"] = ec["negate"]
    if ec.get("caseSensitive") is not None:
        params["case_sensitive"] = ec["caseSensitive"]

    return {
        "id": _grader_id(test_case_id, index, ec_type),
        "type": grader_type,
        "params": params,
        "description": f"Cannonade evaluation type '{ec_type}'" + (" (negated)" if ec.get("negate") else ""),
    }


def _test_input_to_openeval(test_input: Dict[str, Any]) -> Union[str, List[str]]:
    input_type = test_input.get("type")
    if input_type == "completion":
        return test_input.get("prompt") or ""
    if input_type == "chat":
        messages = test_input.get("messages") or []
        return [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages] or [""]
    if input_type in ("json", "code"):
        data = test_input.get("data")
        return json.dumps(data, sort_keys=True) if data is not None else ""
    # Unknown/future input type -- stringify defensively rather than crash.
    return json.dumps(test_input, sort_keys=True)


def suite_to_openeval(suite: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one Cannonade `TestSuite` dict into an EvalPort suite dict.

    `suite` is the parsed JSON content of `~/.cannonade/suites/<id>.json`.
    Returns a plain dict conforming to EvalPort's EvalSuite schema; validate
    with `openeval.validate.validate_suite()` before use.
    """
    test_cases: List[Dict[str, Any]] = []
    all_graders: List[Dict[str, Any]] = []

    for tc in suite.get("testCases", []):
        tc_id = tc.get("id", "")
        evaluations = tc.get("evaluations", []) or []
        graders_for_case = [
            _eval_config_to_grader(tc_id, idx, ec) for idx, ec in enumerate(evaluations)
        ]
        if not graders_for_case:
            # A test case with zero configured evaluations still needs a
            # valid grader reference. This MUST be scoped to this test case
            # (not a shared "gr_default" id) -- a shared id would dangle
            # whenever any OTHER test case in the same suite has real
            # evaluations, since it would never get added to `all_graders`.
            graders_for_case = [
                {
                    "id": f"{tc_id}__gr_default",
                    "type": "custom",
                    "params": {"handler": "cannonade:no_evaluations"},
                    "description": "Cannonade test case defined zero evaluations",
                }
            ]
        all_graders.extend(graders_for_case)

        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": _test_input_to_openeval(tc.get("input", {})),
            "graders": [g["id"] for g in graders_for_case],
            "metadata": {
                "cannonade": {
                    "passing_logic": tc.get("passingLogic"),
                    "run_config": tc.get("runConfig"),
                    "timeout_ms": tc.get("timeoutMs"),
                    "prompt_ref": tc.get("promptRef"),
                    "raw_input": tc.get("input"),
                }
            },
        }
        if tc.get("name"):
            test_case["tags"] = [tc["name"]]
        test_cases.append(test_case)

    if not all_graders:
        # Only reachable when suite["testCases"] is itself empty --
        # validate_suite() separately rejects that (test_cases must be
        # non-empty), but this still avoids ever returning graders=[].
        # (Every non-empty test_cases list produces at least one grader per
        # case via the per-test-case placeholder above.)
        all_graders = [{"id": "gr_default", "type": "custom", "params": {"handler": "cannonade:no_evaluations"}}]

    return {
        "version": OPENEVAL_VERSION,
        "id": suite.get("id", ""),
        "name": suite.get("name"),
        "description": suite.get("description"),
        "test_cases": test_cases,
        "graders": all_graders,
        "config": {k: v for k, v in {"defaultRunConfig": suite.get("defaultRunConfig")}.items() if v is not None},
        "metadata": {
            "openeval": {"source": "cannonade"},
            "cannonade": {
                "suite_version": suite.get("version"),
                "created_at": suite.get("createdAt"),
                "updated_at": suite.get("updatedAt"),
            },
        },
    }


def _model_ref_label(model_ref: Dict[str, Any]) -> str:
    source = model_ref.get("source", "unknown")
    identifier = model_ref.get("modelKey") or model_ref.get("modelId") or "unknown"
    return f"{source}:{identifier}"


def _case_run_to_result(case_run: Dict[str, Any]) -> Dict[str, Any]:
    test_case_id = case_run.get("testCaseId", "")
    result = case_run.get("result")

    if result is None:
        # The run never produced a TestCaseResult (e.g. cancelled/failed
        # before completion) -- build the most honest Result we can from
        # just the TestCaseRun's own status, rather than fabricating one.
        return {
            "test_case_id": test_case_id,
            "passed": False,
            "grader_results": [],
            "completed_at": case_run.get("completedAt"),
            "error": {"type": f"cannonade_status_{case_run.get('status', 'unknown')}", "detail": case_run.get("error")},
            "metadata": {"cannonade": {"status": case_run.get("status")}},
        }

    grader_results: List[Dict[str, Any]] = []
    for idx, er in enumerate(result.get("evalResults", []) or []):
        er_type = er.get("type", "custom")
        grader_type = _DIRECT_GRADER_TYPES.get(er_type, "custom")
        raw_score = er.get("score")
        clipped = _clip_unit(raw_score)
        gr_metadata: Dict[str, Any] = {"raw_score": raw_score, "cannonade_eval_type": er_type}
        if er.get("judgeUsage"):
            gr_metadata["judge_usage"] = er["judgeUsage"]
        if er.get("error"):
            gr_metadata["error"] = er["error"]
        grader_results.append(
            {
                "grader_id": _grader_id(test_case_id, idx, er_type),
                "type": grader_type,
                "score": clipped,
                "passed": bool(er.get("passed", False)),
                "reason": er.get("details"),
                "metadata": gr_metadata,
            }
        )

    metrics = result.get("metrics", {}) or {}
    return {
        "test_case_id": test_case_id,
        "passed": bool(result.get("passed", False)),
        "grader_results": grader_results,
        "actual_output": result.get("output"),
        "duration_ms": metrics.get("durationMs"),
        "completed_at": case_run.get("completedAt"),
        "metadata": {
            "cannonade": {
                "status": case_run.get("status"),
                "reasoning": result.get("reasoning"),
                "metrics": metrics,
                "result_error": result.get("error"),
            }
        },
        **({"error": {"type": "cannonade_result_error", "detail": result["error"]}} if result.get("error") else {}),
    }


def run_to_openeval(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert one Cannonade `TestRun` dict into a list of EvalPort ResultSets.

    `run` is the parsed JSON content of one `~/.cannonade/runs/<id>-<slug>.json`
    file. Returns one EvalPort ResultSet dict per `PerModelRun` in
    `run["modelRuns"]`, all sharing `suite_id = run["suiteId"]` and joined
    via the `group` field (`group_id = run["id"]`) using EvalPort's
    grouped/sibling ResultSets extension -- exactly the "compare N models
    against one suite" shape this adds up to. Validate each with
    `openeval.validate.validate_result_set()` before use.
    """
    suite_id = run.get("suiteId", "")
    run_id_base = run.get("id", "")
    model_runs = run.get("modelRuns", []) or []

    result_sets: List[Dict[str, Any]] = []
    for sequence, pmr in enumerate(model_runs):
        model_ref = pmr.get("modelRef", {}) or {}
        label = _model_ref_label(model_ref)
        results = [_case_run_to_result(cr) for cr in pmr.get("caseRuns", []) or []]

        started_at = pmr.get("startedAt") or run.get("startedAt") or run.get("createdAt") or ""
        completed_at = pmr.get("completedAt") or run.get("completedAt")

        result_sets.append(
            {
                "version": OPENEVAL_VERSION,
                "suite_id": suite_id,
                "run_id": f"{run_id_base}:{pmr.get('id', sequence)}",
                "started_at": started_at,
                "completed_at": completed_at,
                "provider": dict(model_ref),
                "runner": {"name": "cannonade", "version": model_ref.get("source", "unknown")},
                "group": {
                    "group_id": run_id_base,
                    "role": label,
                    "label": label,
                    "sequence": sequence,
                },
                "results": results or [
                    # validate_result_set() requires a non-empty results list;
                    # a PerModelRun that produced zero case runs (e.g. it
                    # failed before running anything) still needs one
                    # placeholder Result to be a valid ResultSet on its own.
                    {
                        "test_case_id": "cannonade_no_case_runs",
                        "passed": False,
                        "grader_results": [],
                        "error": {"type": f"cannonade_status_{pmr.get('status', 'unknown')}", "detail": pmr.get("error")},
                    }
                ],
                "summary": pmr.get("aggregate"),
                "metadata": {
                    "openeval": {"source": "cannonade"},
                    "cannonade": {
                        "suite_name": run.get("suiteName"),
                        "auto_downloaded": pmr.get("autoDownloaded"),
                        "status": pmr.get("status"),
                        "note": (
                            "This ResultSet is one sibling in a Cannonade multi-model "
                            "comparison (see `group.group_id`); every PerModelRun under "
                            "the same TestRun shares that group_id."
                        ),
                    },
                },
            }
        )
    return result_sets


def to_openeval(obj: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Dispatch to `suite_to_openeval()` or `run_to_openeval()` by shape.

    Convenience wrapper for callers who don't want to check which Cannonade
    JSON file they loaded. A `TestSuite` (has `testCases`) returns a single
    suite dict; a `TestRun` (has `modelRuns`) returns a list of ResultSet
    dicts. Raises ValueError for anything else.
    """
    if "testCases" in obj:
        return suite_to_openeval(obj)
    if "modelRuns" in obj:
        return run_to_openeval(obj)
    raise ValueError(
        "to_openeval() couldn't tell whether this is a Cannonade TestSuite "
        "(expected a 'testCases' key) or TestRun (expected a 'modelRuns' "
        "key) -- call suite_to_openeval() or run_to_openeval() directly if "
        "you know which one this is."
    )


def openeval_to_suite(suite: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort, lossy conversion of an EvalPort suite back to a Cannonade TestSuite dict.

    Only reconstructs Cannonade-specific fields (`passingLogic`, `runConfig`,
    `timeoutMs`, `promptRef`, the original structured `input`) when they
    were preserved in `metadata["cannonade"]` by this adapter's own
    `suite_to_openeval()` -- this is NOT a general importer for suites built
    by other EvalPort producers, which have no reason to carry that block.
    """
    cannonade_meta = (suite.get("metadata") or {}).get("cannonade", {})
    test_cases: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        tc_meta = (tc.get("metadata") or {}).get("cannonade", {})
        raw_input = tc_meta.get("raw_input")
        test_cases.append(
            {
                "id": tc.get("id"),
                "name": (tc.get("tags") or [tc.get("id")])[0],
                "input": raw_input if raw_input is not None else {"type": "completion", "prompt": tc.get("input")},
                "evaluations": [],  # grader -> EvaluationConfig is not reversible without the original ec; see docstring.
                "passingLogic": tc_meta.get("passing_logic", "all"),
                "runConfig": tc_meta.get("run_config"),
                "timeoutMs": tc_meta.get("timeout_ms"),
                "promptRef": tc_meta.get("prompt_ref"),
            }
        )
    return {
        "id": suite.get("id"),
        "name": suite.get("name"),
        "description": suite.get("description"),
        "version": cannonade_meta.get("suite_version", "0.0.0"),
        "createdAt": cannonade_meta.get("created_at"),
        "updatedAt": cannonade_meta.get("updated_at"),
        "testCases": test_cases,
    }
