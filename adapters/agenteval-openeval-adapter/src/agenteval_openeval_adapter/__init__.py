"""AgentEval -> EvalPort adapter.

Standalone, one-directional exporter from AgentEval's (https://github.com/lokesh75-kank/agenteval)
public `SuiteReport` JSON shape to the EvalPort ResultSet interchange format
(https://github.com/adhabnr-ux/evalport).

Why one-directional, and why a ResultSet (not an EvalSuite): AgentEval's
defining mechanic is determinism sampling -- `runner.ts` runs every scenario
`runs` times and reports `passingRuns / totalRuns` as `ScenarioRunSummary.
determinism`, the flakiness signal the tool exists to produce. That is
*evidence from an execution*, which is what EvalPort's ResultSet document
represents (`Result` objects joined by `test_case_id` + `run_id`), not a
portable test-case definition (`EvalSuite`) -- AgentEval's assertion
vocabulary (`tool_called`, `every_claim_has_citation`, `citations_resolve`,
`quote_matches_source`, `refusal`, `recall_at_k`, ...) has no equivalent
among EvalPort's well-known grader types, so an AgentEval `Scenario` cannot
honestly become a portable `TestCase` a different runner could execute.
Going the other direction (EvalPort `TestCase` -> AgentEval `Scenario`)
would silently drop that same assertion vocabulary and produce a Scenario
with no meaningful `asserts`, so `from_openeval` is intentionally not
provided here -- see the discussion this adapter follows from at
https://github.com/lokesh75-kank/agenteval/issues/13.

Ground truth this module was written against (AgentEval main @ agenteval-core
0.3.2, both read directly from source, not guessed):
  - src/core/types.ts   (Scenario, ScenarioResult, ScenarioRunSummary, SuiteReport)
  - src/core/trace.ts   (AgentTrace, ToolCall, Citation)
  - src/report/json.ts  (renderJson: `JSON.stringify(report, ...)`, key-sorted --
    confirms the JSON a `SuiteReport` is written to disk as *is* the SuiteReport
    shape verbatim, with no extra wrapper, so `to_openeval()` here accepts
    exactly what `agenteval run --json` (or an equivalent report-file read)
    produces.)

Design note: `to_openeval()`'s output shape follows the same `dict`-in,
`dict`-out convention as this repo's other standalone adapters (see
https://github.com/adhabnr-ux/evalport/tree/main/adapters/crewai-openeval-adapter).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "__version__"]
__version__ = "0.1.0"


def _assertion_grader_result(assertion_result: Dict[str, Any]) -> Dict[str, Any]:
    """One `AssertionResult` (src/core/types.ts) -> one EvalPort `GraderResult`.

    AgentEval's assertion `kind`s (tool_called, every_claim_has_citation,
    citations_resolve, quote_matches_source, refusal, recall_at_k, ...) are
    not among EvalPort's well-known grader types, so each becomes an open,
    non-well-known `type` prefixed `agenteval_<kind>` -- readable in a
    generic EvalPort viewer/report, but only a receiving system that already
    knows AgentEval's grounding/assertion logic can re-execute it. That
    semantic gap is the one the original proposal (agenteval#13) called out
    explicitly and is not something this adapter can close.
    """
    assertion = assertion_result.get("assertion") or {}
    kind = assertion.get("kind", "unknown")
    passed = bool(assertion_result.get("pass"))
    return {
        "grader_id": kind,
        "type": f"agenteval_{kind}",
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "reason": assertion_result.get("detail"),
    }


def _judge_grader_result(judge: Dict[str, Any]) -> Dict[str, Any]:
    """AgentEval's `ScenarioResult.judge` (LLM-as-judge, self-consistency
    voting) -> a `GraderResult`. Dropped from the original issue#13 sketch;
    added per maintainer review (agenteval#13, lokesh75-kank): "perRun[].judge
    should map into grader_results too -- your sketch drops the LLM-judge
    outcomes."

    `score` is the self-consistency vote fraction (passingVotes / votes) when
    votes were actually cast, matching AgentEval's own `JudgeSpec.passThreshold`
    semantics (a fraction of votes); `None` when votes is 0 (e.g. the judge
    ran but AgentEval's own runner recorded a fail-closed 0/0 because no LLM
    client was provided -- see runner.ts's `runOnce`) rather than fabricating
    a 0.0 or 1.0 that would misrepresent "no votes were cast" as a graded 0%.
    """
    votes = judge.get("votes") or 0
    passing_votes = judge.get("passingVotes") or 0
    return {
        "grader_id": "judge",
        "type": "agenteval_llm_judge",
        "score": (passing_votes / votes) if votes else None,
        "passed": bool(judge.get("pass")),
        "reason": judge.get("detail"),
        "metadata": {"votes": votes, "passing_votes": passing_votes},
    }


def _result_for_run(scenario_id: str, attempt: int, run: Dict[str, Any]) -> Dict[str, Any]:
    """One `ScenarioResult` (one of `ScenarioRunSummary.perRun`) -> one EvalPort `Result`."""
    trace: Dict[str, Any] = run.get("trace") or {}

    grader_results: List[Dict[str, Any]] = [
        _assertion_grader_result(a) for a in (run.get("assertions") or [])
    ]
    judge = run.get("judge")
    if judge is not None:
        grader_results.append(_judge_grader_result(judge))

    error: Optional[Dict[str, Any]] = None
    trace_error = trace.get("error")
    if trace_error:
        error = {"type": "runner_error", "message": trace_error}

    tool_calls = trace.get("toolCalls") or []
    citations = trace.get("citations") or []

    result: Dict[str, Any] = {
        "test_case_id": scenario_id,
        "attempt": attempt,
        "passed": bool(run.get("pass")),
        "actual_output": trace.get("finalText", ""),
        "grader_results": grader_results,
        "metadata": {
            "agenteval": {
                "tool_calls_made": [tc.get("name") for tc in tool_calls if isinstance(tc, dict)],
                "citation_count": len(citations),
                "iterations": trace.get("iterations"),
                "tokens": trace.get("tokens"),
            }
        },
    }
    if trace.get("durationMs") is not None:
        result["duration_ms"] = trace["durationMs"]
    if error is not None:
        result["error"] = error
    return result


def to_openeval(
    report: Dict[str, Any],
    run_id: Optional[str] = None,
    suite_id: str = "agenteval_suite",
) -> Dict[str, Any]:
    """Export an AgentEval `SuiteReport` (dict, as written by AgentEval's JSON
    reporter -- src/report/json.ts) to an EvalPort ResultSet (dict).

    `run_id` defaults to a value derived from the report's own `generatedAt`
    timestamp so repeated exports of the same report are reproducible; pass
    an explicit value (e.g. a CI run id) to tie the ResultSet to your own
    run-tracking instead.

    Each AgentEval scenario's `perRun` entries become `attempt: 1..totalRuns`
    Results sharing one `test_case_id`, with `ResultSet.isolation: "fresh"` --
    AgentEval's `runOnce()` (src/core/runner.ts) invokes the adapter fresh on
    every run with no shared state carried between runs, which is exactly
    what `isolation: "fresh"` asserts.

    Per-scenario `determinism` (passingRuns / totalRuns -- the headline
    metric this tool exists to produce) is preserved explicitly in
    `summary["scenarios"][scenario_id]`, not left for a consumer to
    reconstruct by counting passing attempts. This was maintainer feedback
    on the original proposal (agenteval#13): "worth preserving in summary
    metadata rather than leaving it implicit in the attempts."

    Raises `ValueError` if `report` has no scenarios -- EvalPort's
    `ResultSet.results` is required and non-empty
    (`openeval.validate.validate_result_set`), and there is no honest empty
    ResultSet to emit for a report that ran nothing.
    """
    scenarios = report.get("scenarios") or []
    if not scenarios:
        raise ValueError(
            "report has no scenarios to export; EvalPort's ResultSet.results "
            "must be non-empty (see openeval.validate.validate_result_set)"
        )

    results: List[Dict[str, Any]] = []
    scenario_summaries: Dict[str, Any] = {}
    for scenario in scenarios:
        scenario_id = scenario.get("scenarioId")
        per_run = scenario.get("perRun") or []
        for i, run in enumerate(per_run):
            results.append(_result_for_run(scenario_id, i + 1, run))
        scenario_summaries[scenario_id] = {
            "total_runs": scenario.get("totalRuns"),
            "passing_runs": scenario.get("passingRuns"),
            "determinism": scenario.get("determinism"),
            "pass": scenario.get("pass"),
        }

    total = report.get("totalScenarios", len(scenarios))
    passing = report.get("passingScenarios", 0)
    failed = (total - passing) if isinstance(total, int) and isinstance(passing, int) else None
    pass_rate = (passing / total) if isinstance(total, int) and total else None

    generated_at = report.get("generatedAt")
    resolved_run_id = run_id or f"agenteval_{generated_at}" if generated_at else (run_id or "agenteval_run")

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": resolved_run_id,
        "started_at": generated_at,
        "isolation": "fresh",
        "results": results,
        "summary": {
            "total": total,
            "passed": passing,
            "failed": failed,
            "pass_rate": pass_rate,
            "scenarios": scenario_summaries,
        },
        "metadata": {
            "openeval": {"source": "agenteval"},
            "agenteval": {"config": report.get("config") or {}},
        },
    }
