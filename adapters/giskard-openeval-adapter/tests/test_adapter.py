"""Tests for giskard_openeval_adapter, run against the real giskard-checks
package (Scenario/Suite/Check/SuiteResult objects, not mocks) and the real
EvalPort validator (`openeval.validate.validate_suite`/`validate_result_set`).

Requires giskard-checks to be installed (a PyPI pre-release, currently
1.0.2rc1 -- see README.md "Running the tests" for the install command) and
Python >=3.12.
"""

from __future__ import annotations

import builtins
import sys

import pytest
from openeval.validate import validate_result_set, validate_suite

giskard_checks = pytest.importorskip(
    "giskard.checks",
    reason="giskard-checks is required for these tests; see README.md for the install command.",
)

from giskard.checks import (  # noqa: E402
    Equals,
    GreaterThan,
    JsonValid,
    LessThan,
    LLMJudge,
    NotEquals,
    RegexMatching,
    Scenario,
    SemanticSimilarity,
    StringMatching,
    Suite,
)
from giskard.checks.builtin.composition import Not  # noqa: E402
from giskard.checks.core import (  # noqa: E402
    CheckResult,
    CheckStatus,
    Interaction,
    ScenarioResult,
    SuiteResult,
    TestCaseError as GiskardTestCaseError,
    TestCaseResult as GiskardTestCaseResult,
    Trace,
)

from giskard_openeval_adapter import (  # noqa: E402
    from_openeval,
    suite_result_to_openeval,
    to_openeval,
)


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpenevalSingleStep:
    def test_single_scenario_produces_valid_suite(self):
        scenario = (
            Scenario("geo_fact")
            .interact("What is the capital of France?", outputs="Paris is the capital of France.")
            .check(Equals(target_key="trace.last.outputs", expected_value="Paris is the capital of France."))
        )
        suite = Suite(name="geo_quiz")
        suite.append(scenario)

        eval_suite = to_openeval(suite, suite_id="geo_quiz")

        assert eval_suite["id"] == "geo_quiz"
        assert eval_suite["version"] == "1.0.0"
        assert len(eval_suite["test_cases"]) == 1
        tc = eval_suite["test_cases"][0]
        assert tc["id"] == "geo_fact"
        assert tc["input"] == "What is the capital of France?"
        assert tc["expected_output"] == "Paris is the capital of France."
        assert len(tc["graders"]) == 1
        assert tc["graders"][0]["type"] == "exact_match"

        result = validate_suite(eval_suite)
        assert result.valid, result.errors

    def test_default_suite_id_is_suite_name(self):
        suite = Suite(name="unnamed_id_suite")
        suite.append(Scenario("s1").interact("hi", outputs="hey").check(Equals(target_key="trace.last.outputs", expected_value="hey")))
        eval_suite = to_openeval(suite)
        assert eval_suite["id"] == "unnamed_id_suite"

    def test_description_is_passed_through(self):
        suite = Suite(name="s")
        suite.append(Scenario("s1").interact("hi", outputs="hey").check(Equals(target_key="trace.last.outputs", expected_value="hey")))
        eval_suite = to_openeval(suite, description="A test suite.")
        assert eval_suite["description"] == "A test suite."

    def test_string_matching_maps_to_contains(self):
        scenario = (
            Scenario("keyword_check")
            .interact("Tell me about Paris", outputs="Paris is known for the Eiffel Tower.")
            .check(StringMatching(keyword="Eiffel Tower", case_sensitive=False))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["type"] == "contains"
        assert grader["params"] == {"substring": "Eiffel Tower", "ignore_case": True}
        assert eval_suite["test_cases"][0]["expected_output"] == "Eiffel Tower"
        assert validate_suite(eval_suite).valid

    def test_regex_matching_maps_to_regex(self):
        scenario = (
            Scenario("price_check")
            .interact("Give me a price", outputs="The price is $12.50")
            .check(RegexMatching(pattern=r"\$\d+\.\d{2}"))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["type"] == "regex"
        assert grader["params"] == {"pattern": r"\$\d+\.\d{2}"}
        assert validate_suite(eval_suite).valid

    def test_semantic_similarity_maps_through(self):
        scenario = (
            Scenario("sem_check")
            .interact("Capital of France?", outputs="Paris")
            .check(SemanticSimilarity(reference_text="Paris", threshold=0.9))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["type"] == "semantic_similarity"
        assert grader["params"] == {"threshold": 0.9}
        assert eval_suite["test_cases"][0]["expected_output"] == "Paris"
        assert validate_suite(eval_suite).valid

    def test_llm_judge_maps_through_with_placeholder_model(self):
        scenario = (
            Scenario("judge_check")
            .interact("Explain gravity", outputs="Gravity pulls masses together.")
            .check(LLMJudge(prompt="Is this a correct explanation of gravity?"))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["type"] == "llm_judge"
        # The original giskard prompt text survives verbatim; EvalPort's
        # validator requires an `{output}`/`{input}`/`{expected}` token, which
        # giskard's Jinja2-templated prompt doesn't naturally contain, so
        # this adapter appends one rather than rewriting the prompt in place.
        assert grader["params"]["prompt"].startswith("Is this a correct explanation of gravity?")
        assert "{output}" in grader["params"]["prompt"]
        assert grader["params"]["model"] == "giskard-default"
        assert validate_suite(eval_suite).valid

    def test_llm_judge_prompt_already_containing_a_token_is_not_modified(self):
        scenario = (
            Scenario("judge_check2")
            .interact("2+2?", outputs="4")
            .check(LLMJudge(prompt="Does {output} correctly answer the question?"))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["params"]["prompt"] == "Does {output} correctly answer the question?"
        assert validate_suite(eval_suite).valid

    def test_llm_judge_with_prompt_path_uses_placeholder_text(self):
        scenario = (
            Scenario("judge_check3")
            .interact("hi", outputs="hey")
            .check(LLMJudge(prompt_path="checks::custom_judge.j2"))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert "checks::custom_judge.j2" in grader["params"]["prompt"]
        assert "{output}" in grader["params"]["prompt"]
        assert validate_suite(eval_suite).valid

    def test_json_valid_with_schema_maps_to_json_schema(self):
        schema = {"type": "object", "required": ["status"]}
        scenario = (
            Scenario("json_check")
            .interact("Return status json", outputs='{"status": "ok"}')
            .check(JsonValid(schema=schema))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["type"] == "json_schema"
        assert grader["params"]["schema"] == schema
        assert validate_suite(eval_suite).valid

    def test_comparison_check_maps_to_json_path(self):
        scenario = (
            Scenario("count_check")
            .interact("How many?", outputs={"count": 5})
            .check(GreaterThan(target_key="trace.last.outputs.count", expected_value=3))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["type"] == "json_path"
        assert grader["params"] == {"path": "$.count", "expected": "3", "operator": "gt"}
        assert validate_suite(eval_suite).valid

    def test_comparison_on_whole_output_uses_dollar_root(self):
        scenario = (
            Scenario("len_check")
            .interact("Score?", outputs=7)
            .check(GreaterThan(target_key="trace.last.outputs", expected_value=5))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["params"]["path"] == "$"
        assert validate_suite(eval_suite).valid

    def test_opaque_check_maps_to_custom_grader_and_preserves_definition(self):
        scenario = (
            Scenario("negated_check")
            .interact("Say hi", outputs="hi")
            .check(Not(check=Equals(target_key="trace.last.outputs", expected_value="bye")))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"]["handler"] == "giskard.checks.not"
        assert grader["metadata"]["giskard"]["check"]["kind"] == "not"
        assert validate_suite(eval_suite).valid

    def test_comparison_check_on_unrelated_key_maps_to_custom(self):
        scenario = (
            Scenario("input_len_check")
            .interact("hello", outputs="hi")
            .check(GreaterThan(target_key="trace.last.inputs", expected_value="a"))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        grader = eval_suite["test_cases"][0]["graders"][0]
        # `trace.last.inputs` isn't under `trace.last.outputs` -- EvalPort's
        # json_path grader only ever inspects `actual_output`, so this has
        # no representation there and falls back to a "custom" grader.
        assert grader["type"] == "custom"
        assert validate_suite(eval_suite).valid


class TestToOpenevalMultiStepAndSkipping:
    def test_multi_step_scenario_produces_step_suffixed_ids(self):
        scenario = (
            Scenario("multi_turn")
            .interact("Hello", outputs="Hi there!")
            .check(Equals(target_key="trace.last.outputs", expected_value="Hi there!"))
            .interact("How are you?", outputs="I'm doing well!")
            .check(Equals(target_key="trace.last.outputs", expected_value="I'm doing well!"))
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        ids = [tc["id"] for tc in eval_suite["test_cases"]]
        assert ids == ["multi_turn::step_0", "multi_turn::step_1"]
        assert validate_suite(eval_suite).valid

    def test_multiple_interacts_in_one_step_export_as_array_input(self):
        scenario = Scenario("multi_interact")
        scenario.interact("Hello")
        scenario.interact("How are you?")
        scenario.check(Equals(target_key="trace.last.outputs", expected_value="fine"))
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        tc = eval_suite["test_cases"][0]
        assert tc["input"] == ["Hello", "How are you?"]
        assert validate_suite(eval_suite).valid

    def test_dynamic_input_step_is_skipped(self):
        scenario = Scenario("dynamic").interact(
            lambda trace: "generated input",
            outputs="some output",
        ).check(Equals(target_key="trace.last.outputs", expected_value="some output"))
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        assert eval_suite["test_cases"] == []

    def test_step_with_no_checks_is_skipped(self):
        scenario = Scenario("no_checks").interact("hello", outputs="hi")
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        assert eval_suite["test_cases"] == []

    def test_scenario_tags_are_preserved(self):
        scenario = (
            Scenario("tagged")
            .interact("hi", outputs="hey")
            .check(Equals(target_key="trace.last.outputs", expected_value="hey"))
            .with_tags(["Category:Greeting"])
        )
        suite = Suite(name="s")
        suite.append(scenario)
        eval_suite = to_openeval(suite)
        assert eval_suite["test_cases"][0]["tags"] == ["Category:Greeting"]
        assert validate_suite(eval_suite).valid


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


def _minimal_suite(**test_case_overrides):
    test_case = {
        "id": "tc1",
        "input": "hello",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "expected_output": "hey",
    }
    test_case.update(test_case_overrides)
    return {
        "version": "1.0.0",
        "id": "s",
        "test_cases": [test_case],
    }


class TestFromOpeneval:
    def test_exact_match_builds_equals_check(self):
        scenarios = from_openeval(_minimal_suite())
        assert len(scenarios) == 1
        scenario = scenarios[0]
        assert scenario.name == "tc1"
        assert len(scenario.steps) == 1
        assert len(scenario.steps[0].interacts) == 1
        assert scenario.steps[0].interacts[0].inputs == "hello"
        check = scenario.steps[0].checks[0]
        assert isinstance(check, Equals)
        assert check.expected_value == "hey"
        assert check.target_key == "trace.last.outputs"

    def test_contains_builds_string_matching(self):
        suite = _minimal_suite(
            graders=[{"id": "g1", "type": "contains", "params": {"substring": "Paris", "ignore_case": True}}]
        )
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, StringMatching)
        assert check.keyword == "Paris"
        assert check.case_sensitive is False

    def test_regex_builds_regex_matching(self):
        suite = _minimal_suite(graders=[{"id": "g1", "type": "regex", "params": {"pattern": r"\d+"}}])
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, RegexMatching)
        assert check.pattern == r"\d+"

    def test_semantic_similarity_builds_check(self):
        suite = _minimal_suite(graders=[{"id": "g1", "type": "semantic_similarity", "params": {"threshold": 0.8}}])
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, SemanticSimilarity)
        assert check.threshold == 0.8
        assert check.reference_text == "hey"

    def test_llm_judge_builds_check(self):
        suite = _minimal_suite(
            graders=[{"id": "g1", "type": "llm_judge", "params": {"model": "gpt-4o", "prompt": "Is this correct?"}}]
        )
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, LLMJudge)
        assert check.prompt == "Is this correct?"

    def test_json_schema_builds_json_valid(self):
        schema = {"type": "object"}
        suite = _minimal_suite(graders=[{"id": "g1", "type": "json_schema", "params": {"schema": schema}}])
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, JsonValid)
        assert check.expected_schema == schema

    def test_json_path_eq_builds_equals_on_translated_key(self):
        suite = _minimal_suite(
            graders=[{"id": "g1", "type": "json_path", "params": {"path": "$.count", "expected": "5", "operator": "eq"}}]
        )
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, Equals)
        assert check.target_key == "trace.last.outputs.count"
        assert check.expected_value == "5"

    def test_json_path_gt_builds_greater_than(self):
        suite = _minimal_suite(
            graders=[{"id": "g1", "type": "json_path", "params": {"path": "$.count", "expected": "3", "operator": "gt"}}]
        )
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, GreaterThan)

    def test_json_path_root_dollar_maps_to_outputs_root(self):
        suite = _minimal_suite(
            graders=[{"id": "g1", "type": "json_path", "params": {"path": "$", "expected": "5", "operator": "lt"}}]
        )
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, LessThan)
        assert check.target_key == "trace.last.outputs"

    def test_json_path_contains_builds_string_matching_with_text_key(self):
        suite = _minimal_suite(
            graders=[
                {
                    "id": "g1",
                    "type": "json_path",
                    "params": {"path": "$.message", "expected": "hello", "operator": "contains"},
                }
            ]
        )
        scenario = from_openeval(suite)[0]
        check = scenario.steps[0].checks[0]
        assert isinstance(check, StringMatching)
        assert check.target_key == "trace.last.outputs.message"
        assert check.keyword == "hello"

    def test_unsupported_grader_type_is_clean_skipped(self):
        suite = _minimal_suite(
            graders=[
                {"id": "g1", "type": "human"},
                {"id": "g2", "type": "exact_match"},
            ]
        )
        scenario = from_openeval(suite)[0]
        assert len(scenario.steps[0].checks) == 1
        assert isinstance(scenario.steps[0].checks[0], Equals)

    def test_bare_grader_id_string_is_skipped(self):
        suite = _minimal_suite(graders=["some_shared_grader_id"])
        scenario = from_openeval(suite)[0]
        assert scenario.steps[0].checks == []

    def test_array_input_becomes_multiple_interacts_in_one_step(self):
        suite = _minimal_suite(input=["Hello", "How are you?"])
        scenario = from_openeval(suite)[0]
        assert len(scenario.steps) == 1
        assert [i.inputs for i in scenario.steps[0].interacts] == ["Hello", "How are you?"]

    def test_tags_are_applied(self):
        suite = _minimal_suite(tags=["Category:Test"])
        scenario = from_openeval(suite)[0]
        assert scenario.tags == ["Category:Test"]

    def test_missing_giskard_checks_raises_helpful_import_error(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "giskard.checks" or name.startswith("giskard.checks."):
                raise ImportError("No module named 'giskard.checks'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for mod in list(sys.modules):
            if mod == "giskard.checks" or mod.startswith("giskard.checks."):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        with pytest.raises(ImportError, match="giskard-checks"):
            from_openeval(_minimal_suite())


# ---------------------------------------------------------------------------
# suite_result_to_openeval
# ---------------------------------------------------------------------------


class TestSuiteResultToOpenevalWithRealRun:
    @pytest.mark.asyncio
    async def test_running_a_real_suite_produces_valid_result_set(self):
        s1 = (
            Scenario("geo_fact")
            .interact("What is the capital of France?", outputs="Paris is the capital of France.")
            .check(Equals(target_key="trace.last.outputs", expected_value="Paris is the capital of France."))
        )
        s2 = (
            Scenario("keyword_check")
            .interact("Tell me about Paris", outputs="A city known for the Eiffel Tower.")
            .check(StringMatching(keyword="Eiffel Tower"))
        )
        s3 = (
            Scenario("failing_check")
            .interact("2+2?", outputs="5")
            .check(Equals(target_key="trace.last.outputs", expected_value="4"))
        )
        suite = Suite(name="mixed")
        for s in (s1, s2, s3):
            suite.append(s)

        result = await suite.run(verbose=False)
        assert result.passed_count == 2
        assert result.failed_count == 1

        result_set = suite_result_to_openeval(
            result, suite_id="mixed", run_id="run-1", started_at="2026-08-15T00:00:00Z",
            completed_at="2026-08-15T00:00:01Z",
        )

        assert result_set["suite_id"] == "mixed"
        assert result_set["completed_at"] == "2026-08-15T00:00:01Z"
        assert result_set["summary"]["passed"] == 2
        assert result_set["summary"]["failed"] == 1
        by_id = {r["test_case_id"]: r for r in result_set["results"]}
        assert by_id["geo_fact"]["passed"] is True
        assert by_id["geo_fact"]["grader_results"][0]["score"] == 1.0
        assert by_id["failing_check"]["passed"] is False
        assert by_id["failing_check"]["grader_results"][0]["score"] == 0.0
        assert by_id["failing_check"]["actual_output"] == "5"

        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors

    @pytest.mark.asyncio
    async def test_round_trip_from_openeval_suite_through_a_real_run(self):
        """from_openeval() output is runnable, and its results convert back cleanly."""
        eval_suite = _minimal_suite()
        scenarios = from_openeval(eval_suite)
        suite = Suite(name="roundtrip")
        for scenario in scenarios:
            scenario.with_target(lambda inputs, trace=None: "hey")
            suite.append(scenario)

        result = await suite.run(verbose=False)
        assert result.passed_count == 1

        result_set = suite_result_to_openeval(
            result, suite_id="roundtrip", run_id="run-2", started_at="2026-08-15T00:00:00Z"
        )
        assert result_set["results"][0]["test_case_id"] == "tc1"
        assert validate_result_set(result_set).valid


class TestSuiteResultToOpenevalWithConstructedResults:
    """Exercise status-mapping edge cases (ERROR, SKIP) that are awkward to
    trigger organically through a real check, by constructing the immutable
    result objects directly -- these are the same real pydantic model classes
    the runner produces, just built by hand for the specific status."""

    def _suite_result(self, check_results):
        trace = Trace(interactions=[Interaction(inputs="hi", outputs="hey")])
        test_case_result = GiskardTestCaseResult(results=check_results, duration_ms=5)
        scenario_result = ScenarioResult(
            scenario_name="s1", steps=[test_case_result], duration_ms=5, final_trace=trace
        )
        return SuiteResult(results=[scenario_result], duration_ms=5)

    def test_error_status_maps_to_null_score_and_not_passed(self):
        check_result = CheckResult(
            status=CheckStatus.ERROR,
            message="No value found",
            details={"check_kind": "equals", "check_name": None},
        )
        suite_result = self._suite_result([check_result])
        result_set = suite_result_to_openeval(
            suite_result, suite_id="s", run_id="r", started_at="2026-08-15T00:00:00Z"
        )
        grader_result = result_set["results"][0]["grader_results"][0]
        assert grader_result["score"] is None
        assert grader_result["passed"] is False
        assert validate_result_set(result_set).valid

    def test_skip_status_maps_to_null_score_and_not_passed(self):
        check_result = CheckResult(
            status=CheckStatus.SKIP,
            message="Precondition not met",
            details={"check_kind": "equals", "check_name": None},
        )
        suite_result = self._suite_result([check_result])
        result_set = suite_result_to_openeval(
            suite_result, suite_id="s", run_id="r", started_at="2026-08-15T00:00:00Z"
        )
        grader_result = result_set["results"][0]["grader_results"][0]
        assert grader_result["score"] is None
        assert grader_result["passed"] is False
        assert validate_result_set(result_set).valid

    def test_test_case_error_is_surfaced(self):
        trace = Trace(interactions=[])
        test_case_result = GiskardTestCaseResult(
            results=[],
            duration_ms=1,
            error=GiskardTestCaseError(message="boom", exception_type="ValueError"),
        )
        scenario_result = ScenarioResult(
            scenario_name="errored", steps=[test_case_result], duration_ms=1, final_trace=trace
        )
        suite_result = SuiteResult(results=[scenario_result], duration_ms=1)
        result_set = suite_result_to_openeval(
            suite_result, suite_id="s", run_id="r", started_at="2026-08-15T00:00:00Z"
        )
        result = result_set["results"][0]
        assert result["error"]["type"] == "runner_error"
        assert "boom" in result["error"]["message"]
        # A TestCase with zero graders is allowed on the *results* side
        # (unlike the suite/testcase.json side, which requires >=1 grader
        # per TestCase) -- an errored test case may never have reached the
        # point of producing any CheckResults at all.
        assert result["grader_results"] == []
        assert validate_result_set(result_set).valid

    def test_multi_step_scenario_result_produces_step_suffixed_ids(self):
        trace = Trace(interactions=[Interaction(inputs="a", outputs="1"), Interaction(inputs="b", outputs="2")])
        check_result = CheckResult(status=CheckStatus.PASS, details={"check_kind": "equals"})
        step1 = GiskardTestCaseResult(results=[check_result], duration_ms=1)
        step2 = GiskardTestCaseResult(results=[check_result], duration_ms=1)
        scenario_result = ScenarioResult(
            scenario_name="multi", steps=[step1, step2], duration_ms=2, final_trace=trace
        )
        suite_result = SuiteResult(results=[scenario_result], duration_ms=2)
        result_set = suite_result_to_openeval(
            suite_result, suite_id="s", run_id="r", started_at="2026-08-15T00:00:00Z"
        )
        ids = [r["test_case_id"] for r in result_set["results"]]
        assert ids == ["multi::step_0", "multi::step_1"]
        # Only the final step carries the (final) trace's last output.
        assert "actual_output" not in result_set["results"][0]
        assert result_set["results"][1]["actual_output"] == "2"
        assert validate_result_set(result_set).valid

    def test_grader_id_falls_back_to_check_kind_when_unnamed(self):
        check_result = CheckResult(
            status=CheckStatus.PASS,
            details={"check_kind": "regex_matching", "check_name": None},
        )
        suite_result = self._suite_result([check_result])
        result_set = suite_result_to_openeval(
            suite_result, suite_id="s", run_id="r", started_at="2026-08-15T00:00:00Z"
        )
        assert result_set["results"][0]["grader_results"][0]["grader_id"] == "regex_matching"
        assert result_set["results"][0]["grader_results"][0]["type"] == "regex"

    def test_summary_counts_and_pass_rate(self):
        pass_check = CheckResult(status=CheckStatus.PASS, details={"check_kind": "equals"})
        fail_check = CheckResult(status=CheckStatus.FAIL, details={"check_kind": "equals"})
        suite_result = self._suite_result([pass_check])
        result_set = suite_result_to_openeval(
            suite_result, suite_id="s", run_id="r", started_at="2026-08-15T00:00:00Z"
        )
        assert result_set["summary"]["total"] == 1
        assert result_set["summary"]["passed"] == 1
        assert result_set["summary"]["pass_rate"] == 1.0
