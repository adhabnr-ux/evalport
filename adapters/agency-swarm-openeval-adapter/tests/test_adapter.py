import json

import pytest
from agents import Agent, RunResult
from agents.guardrail import (
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
)
from agents.run_context import RunContextWrapper
from agents.usage import Usage
from openeval.validate import validate_result_set, validate_suite
from pydantic import BaseModel

from agency_swarm_openeval_adapter import (
    build_result_set,
    build_test_suite,
    default_exact_match_grader,
    result_to_openeval,
    suite_to_test_cases,
)


def _make_agent(name: str) -> Agent:
    return Agent(name=name, instructions=f"You are {name}.")


def _make_run_result(
    *,
    final_output,
    last_agent: Agent,
    input_text: str = "hello",
    requests: int = 1,
    input_tokens: int = 10,
    output_tokens: int = 5,
    total_tokens: int = 15,
    new_items=None,
    raw_responses=None,
    input_guardrail_results=None,
    output_guardrail_results=None,
    tool_input_guardrail_results=None,
    tool_output_guardrail_results=None,
) -> RunResult:
    usage = Usage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)
    ctx = RunContextWrapper(context=None, usage=usage)
    return RunResult(
        input=input_text,
        new_items=new_items or [],
        raw_responses=raw_responses or [],
        final_output=final_output,
        input_guardrail_results=input_guardrail_results or [],
        output_guardrail_results=output_guardrail_results or [],
        tool_input_guardrail_results=tool_input_guardrail_results or [],
        tool_output_guardrail_results=tool_output_guardrail_results or [],
        context_wrapper=ctx,
        _last_agent=last_agent,
    )


def _tripped_output_guardrail_result(agent: Agent, name: str = "pii_check") -> OutputGuardrailResult:
    def _fn(ctx, agent, output):
        return GuardrailFunctionOutput(output_info={"flagged": True}, tripwire_triggered=True)

    guardrail = OutputGuardrail(guardrail_function=_fn, name=name)
    return OutputGuardrailResult(
        guardrail=guardrail,
        agent_output="blocked output",
        agent=agent,
        output=GuardrailFunctionOutput(output_info={"flagged": True}, tripwire_triggered=True),
    )


def _clean_input_guardrail_result(name: str = "profanity_check") -> InputGuardrailResult:
    def _fn(ctx, agent, input):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    guardrail = InputGuardrail(guardrail_function=_fn, name=name)
    return InputGuardrailResult(guardrail=guardrail, output=GuardrailFunctionOutput(output_info=None, tripwire_triggered=False))


class StructuredOutput(BaseModel):
    answer: str
    confidence: float


class TestBuildTestSuite:
    def test_validates_against_real_sdk(self):
        test_cases = [
            {"id": "tc1", "input": "What is 2+2?", "recipient_agent": "CEO", "expected_output": "4"},
            {"id": "tc2", "input": "Say hi", "recipient_agent": "Developer"},
        ]
        suite = build_test_suite(test_cases, suite_id="suite-1", name="Basic")
        validation = validate_suite(suite)
        assert validation.valid, validation.errors

    def test_test_case_with_expected_output_gets_exact_match_grader(self):
        suite = build_test_suite(
            [{"id": "tc1", "input": "hi", "recipient_agent": "CEO", "expected_output": "hello"}],
            suite_id="s",
        )
        tc1 = suite["test_cases"][0]
        assert tc1["graders"] == ["tc1__exact_match"]
        assert tc1["expected_output"] == "hello"
        grader = next(g for g in suite["graders"] if g["id"] == "tc1__exact_match")
        assert grader["type"] == "exact_match"
        assert grader["params"] == {"expected": "hello"}

    def test_test_case_without_expected_output_gets_scoped_placeholder_grader(self):
        # Regression guard mirroring the cannonade-openeval-adapter's real
        # dangling-grader-reference bug: mixing a test case with no
        # expected_output alongside one that HAS a real grader must not
        # produce a suite-wide shared placeholder id.
        suite = build_test_suite(
            [
                {"id": "tc1", "input": "hi", "recipient_agent": "CEO", "expected_output": "hello"},
                {"id": "tc2", "input": "do something", "recipient_agent": "Developer"},
            ],
            suite_id="s",
        )
        tc2 = next(tc for tc in suite["test_cases"] if tc["id"] == "tc2")
        assert tc2["graders"] == ["tc2__gr_default"]
        assert "expected_output" not in tc2
        grader_ids = {g["id"] for g in suite["graders"]}
        assert "tc2__gr_default" in grader_ids
        validation = validate_suite(suite)
        assert validation.valid, validation.errors

    def test_non_string_expected_output_is_coerced(self):
        suite = build_test_suite(
            [{"id": "tc1", "input": "hi", "recipient_agent": "CEO", "expected_output": {"a": 1}}],
            suite_id="s",
        )
        tc1 = suite["test_cases"][0]
        assert tc1["expected_output"] == json.dumps({"a": 1}, sort_keys=True)

    def test_list_message_input_rendered_as_role_content_strings(self):
        suite = build_test_suite(
            [
                {
                    "id": "tc1",
                    "input": [{"role": "user", "content": "hi"}, {"role": "system", "content": "be nice"}],
                    "recipient_agent": "CEO",
                    "expected_output": "hello",
                }
            ],
            suite_id="s",
        )
        assert suite["test_cases"][0]["input"] == ["user: hi", "system: be nice"]

    def test_raw_input_preserved_in_metadata_for_round_trip(self):
        suite = build_test_suite(
            [{"id": "tc1", "input": "hi", "recipient_agent": "CEO", "expected_output": "hello"}],
            suite_id="s",
        )
        meta = suite["test_cases"][0]["metadata"]["agency_swarm"]
        assert meta["raw_input"] == "hi"
        assert meta["recipient_agent"] == "CEO"
        assert meta["expected_output"] == "hello"


class TestSuiteToTestCases:
    def test_round_trips_adapter_authored_metadata(self):
        original = [
            {"id": "tc1", "input": "hi", "recipient_agent": "CEO", "expected_output": "hello"},
            {"id": "tc2", "input": "do X", "recipient_agent": "Developer"},
        ]
        suite = build_test_suite(original, suite_id="s")
        recovered = suite_to_test_cases(suite)
        assert recovered[0] == {"id": "tc1", "input": "hi", "recipient_agent": "CEO", "expected_output": "hello"}
        assert recovered[1] == {"id": "tc2", "input": "do X", "recipient_agent": "Developer", "expected_output": None}


class TestDefaultExactMatchGrader:
    def test_returns_none_without_expected_output(self):
        agent = _make_agent("CEO")
        result = _make_run_result(final_output="anything", last_agent=agent)
        assert default_exact_match_grader({"id": "tc1"}, result) is None

    def test_matches_string_output(self):
        agent = _make_agent("CEO")
        result = _make_run_result(final_output="4", last_agent=agent)
        gr = default_exact_match_grader({"id": "tc1", "expected_output": "4"}, result)
        assert gr["passed"] is True
        assert gr["score"] == 1.0

    def test_detects_mismatch(self):
        agent = _make_agent("CEO")
        result = _make_run_result(final_output="5", last_agent=agent)
        gr = default_exact_match_grader({"id": "tc1", "expected_output": "4"}, result)
        assert gr["passed"] is False
        assert gr["score"] == 0.0
        assert gr["reason"] is not None

    def test_structured_pydantic_output_is_coerced_for_comparison(self):
        agent = _make_agent("CEO")
        structured = StructuredOutput(answer="Paris", confidence=0.9)
        result = _make_run_result(final_output=structured, last_agent=agent)
        expected = json.loads(structured.model_dump_json())
        gr = default_exact_match_grader({"id": "tc1", "expected_output": expected}, result)
        assert gr["passed"] is True


class TestResultToOpenEval:
    def test_result_with_expected_output_validates_and_passes(self):
        agent = _make_agent("CEO")
        result = _make_run_result(final_output="4", last_agent=agent)
        r = result_to_openeval({"id": "tc1", "expected_output": "4", "recipient_agent": "CEO"}, result)
        assert r["passed"] is True
        assert r["actual_output"] == "4"
        assert r["metadata"]["agency_swarm"]["last_agent"] == "CEO"

    def test_result_without_expected_output_defaults_passed_true_and_empty_graders(self):
        agent = _make_agent("Developer")
        result = _make_run_result(final_output="anything at all", last_agent=agent)
        r = result_to_openeval({"id": "tc2", "recipient_agent": "Developer"}, result)
        assert r["grader_results"] == []
        assert r["passed"] is True  # documented: "call completed", not a correctness signal

    def test_tripped_output_guardrail_forces_passed_false_even_with_matching_grader(self):
        agent = _make_agent("CEO")
        tripped = _tripped_output_guardrail_result(agent)
        result = _make_run_result(
            final_output="4",  # would otherwise exactly match
            last_agent=agent,
            output_guardrail_results=[tripped],
        )
        r = result_to_openeval({"id": "tc1", "expected_output": "4"}, result)
        # The default exact_match grader still reports passed=True on its own...
        assert r["grader_results"][0]["passed"] is True
        # ...but the guardrail trip overrides the Result's overall `passed`.
        assert r["passed"] is False
        assert r["metadata"]["agency_swarm"]["guardrails"]["output"][0]["tripwire_triggered"] is True
        assert r["metadata"]["agency_swarm"]["guardrails"]["output"][0]["name"] == "pii_check"

    def test_clean_guardrails_do_not_affect_passed(self):
        agent = _make_agent("CEO")
        clean = _clean_input_guardrail_result()
        result = _make_run_result(final_output="4", last_agent=agent, input_guardrail_results=[clean])
        r = result_to_openeval({"id": "tc1", "expected_output": "4"}, result)
        assert r["passed"] is True
        assert r["metadata"]["agency_swarm"]["guardrails"]["input"][0]["tripwire_triggered"] is False

    def test_usage_and_counts_preserved(self):
        agent = _make_agent("CEO")
        result = _make_run_result(
            final_output="4",
            last_agent=agent,
            requests=3,
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            new_items=[object(), object()],
            raw_responses=[object()],
        )
        r = result_to_openeval({"id": "tc1"}, result)
        usage = r["metadata"]["agency_swarm"]["usage"]
        assert usage == {"requests": 3, "input_tokens": 100, "output_tokens": 40, "total_tokens": 140}
        assert r["metadata"]["agency_swarm"]["new_items_count"] == 2
        assert r["metadata"]["agency_swarm"]["raw_responses_count"] == 1

    def test_exception_produces_honest_failure_not_fabricated_output(self):
        r = result_to_openeval({"id": "tc1", "recipient_agent": "CEO"}, exception=ValueError("boom"))
        assert r["passed"] is False
        assert r["grader_results"] == []
        assert r["error"] == {"type": "ValueError", "detail": "boom"}
        assert "actual_output" not in r

    def test_requires_result_or_exception(self):
        with pytest.raises(ValueError, match="requires either"):
            result_to_openeval({"id": "tc1"})

    def test_custom_grader_is_used_instead_of_default(self):
        agent = _make_agent("Developer")
        result = _make_run_result(final_output="4", last_agent=agent)

        def handoff_grader(test_case, res):
            matched = res.last_agent.name == test_case.get("expected_last_agent")
            return {
                "grader_id": f"{test_case['id']}__handoff",
                "type": "custom",
                "score": 1.0 if matched else 0.0,
                "passed": matched,
                "metadata": {"handler": "agency_swarm:last_agent_check"},
            }

        r = result_to_openeval(
            {"id": "tc1", "expected_last_agent": "Developer"},
            result,
            graders=[handoff_grader],
        )
        assert r["grader_results"][0]["grader_id"] == "tc1__handoff"
        assert r["passed"] is True


class TestBuildResultSet:
    def test_produces_valid_result_set_from_mixed_success_and_failure(self):
        agent = _make_agent("CEO")
        ok_result = _make_run_result(final_output="4", last_agent=agent)
        runs = [
            {"test_case": {"id": "tc1", "expected_output": "4", "recipient_agent": "CEO"}, "result": ok_result},
            {"test_case": {"id": "tc2", "recipient_agent": "CEO"}, "exception": RuntimeError("timeout")},
        ]
        result_set = build_result_set(runs, suite_id="s", run_id="r1", started_at="2026-01-01T00:00:00Z")
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        assert len(result_set["results"]) == 2
        assert result_set["results"][0]["passed"] is True
        assert result_set["results"][1]["passed"] is False

    def test_empty_runs_raises(self):
        with pytest.raises(ValueError, match="at least one run"):
            build_result_set([], suite_id="s", run_id="r1", started_at="2026-01-01T00:00:00Z")

    def test_optional_fields_included_when_provided(self):
        agent = _make_agent("CEO")
        result = _make_run_result(final_output="4", last_agent=agent)
        runs = [{"test_case": {"id": "tc1", "expected_output": "4"}, "result": result}]
        result_set = build_result_set(
            runs,
            suite_id="s",
            run_id="r1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:01:00Z",
            provider={"source": "agency-swarm"},
            runner={"name": "agency-swarm", "version": "1.0"},
        )
        assert result_set["completed_at"] == "2026-01-01T00:01:00Z"
        assert result_set["provider"] == {"source": "agency-swarm"}
        assert result_set["runner"] == {"name": "agency-swarm", "version": "1.0"}
