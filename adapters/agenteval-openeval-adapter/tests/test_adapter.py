from openeval.validate import validate_result_set

from agenteval_openeval_adapter import to_openeval


def _assertion_result(kind, passed, detail=None, **extra):
    """Stand-in for AgentEval's `AssertionResult` (src/core/types.ts)."""
    return {
        "assertion": {"kind": kind, **extra},
        "pass": passed,
        "detail": detail,
    }


def _trace(final_text="hello", tool_calls=None, citations=None, iterations=1,
           tokens=None, duration_ms=120, error=None):
    """Stand-in for AgentEval's `AgentTrace` (src/core/trace.ts)."""
    trace = {
        "input": {"user_message": "hi"},
        "finalText": final_text,
        "toolCalls": tool_calls or [],
    }
    if citations is not None:
        trace["citations"] = citations
    if iterations is not None:
        trace["iterations"] = iterations
    if tokens is not None:
        trace["tokens"] = tokens
    if duration_ms is not None:
        trace["durationMs"] = duration_ms
    if error is not None:
        trace["error"] = error
    return trace


def _scenario_result(scenario_id, passed, trace=None, assertions=None, judge=None):
    """Stand-in for AgentEval's `ScenarioResult`."""
    sr = {
        "scenarioId": scenario_id,
        "pass": passed,
        "trace": trace or _trace(),
        "assertions": assertions or [],
    }
    if judge is not None:
        sr["judge"] = judge
    return sr


def _scenario_run_summary(scenario_id, per_run):
    """Stand-in for AgentEval's `ScenarioRunSummary` -- computed the same way
    runner.ts's `runScenario` computes it, so fixtures can't drift from the
    real determinism formula."""
    total_runs = len(per_run)
    passing_runs = sum(1 for r in per_run if r["pass"])
    return {
        "scenarioId": scenario_id,
        "totalRuns": total_runs,
        "passingRuns": passing_runs,
        "determinism": passing_runs / total_runs,
        "pass": (passing_runs / total_runs) >= (2 / 3),
        "perRun": per_run,
    }


def _suite_report(scenarios, generated_at="2026-09-01T12:00:00.000Z", config=None):
    """Stand-in for AgentEval's `SuiteReport` -- the exact shape
    src/report/json.ts JSON.stringify's to disk."""
    return {
        "generatedAt": generated_at,
        "totalScenarios": len(scenarios),
        "passingScenarios": sum(1 for s in scenarios if s["pass"]),
        "config": config or {"runs": 3, "passThreshold": 2 / 3},
        "scenarios": scenarios,
    }


def test_attempt_numbers_and_isolation_for_determinism_sampling():
    """The core mapping this adapter exists for: N runs of one scenario ->
    N Results sharing test_case_id, attempt 1..N, isolation "fresh"."""
    per_run = [
        _scenario_result("s1", True, assertions=[_assertion_result("tool_called", True)]),
        _scenario_result("s1", True, assertions=[_assertion_result("tool_called", True)]),
        _scenario_result("s1", False, assertions=[_assertion_result("tool_called", False, "tool not called")]),
    ]
    report = _suite_report([_scenario_run_summary("s1", per_run)])

    rs = to_openeval(report)

    assert rs["isolation"] == "fresh"
    assert [r["attempt"] for r in rs["results"]] == [1, 2, 3]
    assert all(r["test_case_id"] == "s1" for r in rs["results"])
    assert [r["passed"] for r in rs["results"]] == [True, True, False]


def test_judge_outcome_included_in_grader_results():
    """Maintainer feedback on agenteval#13: the original sketch dropped
    perRun[].judge from grader_results -- this must not regress."""
    judge = {"pass": True, "votes": 3, "passingVotes": 2, "detail": "meets rubric"}
    per_run = [_scenario_result("s1", True, judge=judge)]
    report = _suite_report([_scenario_run_summary("s1", per_run)])

    rs = to_openeval(report)

    grader_ids = {g["grader_id"] for g in rs["results"][0]["grader_results"]}
    assert "judge" in grader_ids
    judge_gr = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "judge")
    assert judge_gr["type"] == "agenteval_llm_judge"
    assert judge_gr["passed"] is True
    assert judge_gr["score"] == 2 / 3
    assert judge_gr["reason"] == "meets rubric"


def test_judge_with_zero_votes_scores_none_not_zero():
    """A judge that ran fail-closed with no LLM client (runner.ts's runOnce:
    votes=0, passingVotes=0) must not be reported as a graded 0.0 -- that
    would misrepresent 'no votes were cast' as 'failed every vote'."""
    judge = {"pass": False, "votes": 0, "passingVotes": 0, "detail": "no llm client provided"}
    per_run = [_scenario_result("s1", False, judge=judge)]
    report = _suite_report([_scenario_run_summary("s1", per_run)])

    rs = to_openeval(report)
    judge_gr = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "judge")
    assert judge_gr["score"] is None
    assert judge_gr["passed"] is False


def test_determinism_preserved_explicitly_in_summary():
    """Maintainer feedback on agenteval#13: determinism is 'the headline
    metric of this tool' and must be preserved in summary metadata, not
    left implicit in the attempts."""
    per_run = [
        _scenario_result("s1", True),
        _scenario_result("s1", True),
        _scenario_result("s1", False),
    ]
    report = _suite_report([_scenario_run_summary("s1", per_run)])

    rs = to_openeval(report)

    s1_summary = rs["summary"]["scenarios"]["s1"]
    assert s1_summary["determinism"] == 2 / 3
    assert s1_summary["total_runs"] == 3
    assert s1_summary["passing_runs"] == 2


def test_assertion_kind_becomes_non_well_known_grader_type():
    """AgentEval's assertion vocabulary has no EvalPort well-known
    equivalent (agenteval#13's own honest gap) -- each becomes an open,
    prefixed, non-well-known grader type rather than being force-fit into
    one of EvalPort's 11 standard types."""
    per_run = [
        _scenario_result(
            "s1",
            False,
            assertions=[
                _assertion_result("every_claim_has_citation", False, "claim 2 uncited"),
                _assertion_result("citations_resolve", True),
            ],
        )
    ]
    report = _suite_report([_scenario_run_summary("s1", per_run)])

    rs = to_openeval(report)
    types = {g["type"] for g in rs["results"][0]["grader_results"]}
    assert types == {"agenteval_every_claim_has_citation", "agenteval_citations_resolve"}


def test_run_error_maps_to_result_error():
    per_run = [
        _scenario_result(
            "s1", False,
            trace=_trace(final_text="", error="agent timed out"),
        )
    ]
    report = _suite_report([_scenario_run_summary("s1", per_run)])

    rs = to_openeval(report)
    result = rs["results"][0]
    assert result["passed"] is False
    assert result["error"] == {"type": "runner_error", "message": "agent timed out"}


def test_multi_scenario_attempts_stay_unique_and_document_validates():
    scenarios = [
        _scenario_run_summary("s1", [_scenario_result("s1", True), _scenario_result("s1", True)]),
        _scenario_run_summary(
            "s2",
            [_scenario_result("s2", True), _scenario_result("s2", False), _scenario_result("s2", True)],
        ),
    ]
    report = _suite_report(scenarios)

    rs = to_openeval(report)

    keys = [(r["test_case_id"], r["attempt"]) for r in rs["results"]]
    assert len(keys) == len(set(keys))

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_custom_run_id_and_suite_id():
    report = _suite_report([_scenario_run_summary("s1", [_scenario_result("s1", True)])])
    rs = to_openeval(report, run_id="ci-run-42", suite_id="my_agenteval_suite")
    assert rs["run_id"] == "ci-run-42"
    assert rs["suite_id"] == "my_agenteval_suite"


def test_run_id_defaults_from_generated_at_when_not_given():
    report = _suite_report(
        [_scenario_run_summary("s1", [_scenario_result("s1", True)])],
        generated_at="2026-09-01T12:00:00.000Z",
    )
    rs = to_openeval(report)
    assert rs["run_id"] == "agenteval_2026-09-01T12:00:00.000Z"


def test_empty_report_raises_rather_than_emitting_invalid_document():
    report = _suite_report([])
    try:
        to_openeval(report)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_full_report_end_to_end_validates_against_evalport_spec():
    """A realistic multi-scenario report, including a judge and an error
    run, still produces a spec-valid ResultSet end to end."""
    scenarios = [
        _scenario_run_summary(
            "grounded_answer",
            [
                _scenario_result(
                    "grounded_answer", True,
                    trace=_trace(final_text="Per [E1], the limit is 10mg."),
                    assertions=[_assertion_result("every_claim_has_citation", True)],
                    judge={"pass": True, "votes": 1, "passingVotes": 1, "detail": "on rubric"},
                ),
                _scenario_result(
                    "grounded_answer", False,
                    trace=_trace(final_text="The limit is 10mg.", error=None),
                    assertions=[_assertion_result("every_claim_has_citation", False, "no citation")],
                    judge={"pass": False, "votes": 1, "passingVotes": 0, "detail": "off rubric"},
                ),
            ],
        ),
        _scenario_run_summary(
            "tool_use",
            [
                _scenario_result(
                    "tool_use", False,
                    trace=_trace(final_text="", error="tool invocation failed"),
                    assertions=[_assertion_result("tool_called", False, "search never called")],
                ),
            ],
        ),
    ]
    report = _suite_report(scenarios)

    rs = to_openeval(report, run_id="nightly-2026-09-01")

    assert len(rs["results"]) == 3
    assert rs["summary"]["total"] == 2
    assert rs["summary"]["passed"] == 0  # neither ScenarioRunSummary.pass is True given the fixtures above
    validation = validate_result_set(rs)
    assert validation.valid, validation.errors
