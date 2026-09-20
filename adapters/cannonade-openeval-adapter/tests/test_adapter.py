import json

import pytest
from openeval.validate import validate_result_set, validate_suite

from cannonade_openeval_adapter import (
    openeval_to_suite,
    run_to_openeval,
    suite_to_openeval,
    to_openeval,
)


def _suite_fixture():
    return {
        "id": "suite-1",
        "name": "Basic checks",
        "description": "A small mixed suite covering every EvaluationConfig bucket.",
        "version": "3",
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-02T00:00:00Z",
        "defaultRunConfig": {"temperature": 0.2},
        "testCases": [
            {
                "id": "tc1",
                "name": "exact match case",
                "input": {"type": "completion", "prompt": "What is 2+2?"},
                "evaluations": [{"type": "exact_match", "expected": "4"}],
                "passingLogic": "all",
                "runConfig": {"temperature": 0},
                "timeoutMs": 5000,
                "promptRef": None,
            },
            {
                "id": "tc2",
                "name": "chat contains/regex",
                "input": {
                    "type": "chat",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Say hi"},
                    ],
                },
                "evaluations": [
                    {"type": "contains", "expected": "hello", "caseSensitive": False},
                    {"type": "regex", "expected": "^h.*"},
                ],
                "passingLogic": "any",
            },
            {
                "id": "tc3",
                "name": "similarity",
                "input": {"type": "json", "data": {"a": 1}},
                "evaluations": [
                    {"type": "cosine_similarity", "expected": "foo", "threshold": 0.9},
                ],
            },
            {
                "id": "tc4",
                "name": "custom bucket",
                "input": {"type": "code", "data": {"lang": "python", "code": "print(1)"}},
                "evaluations": [
                    {"type": "llm_rubric", "llmRubric": {"rubric": "Is it correct?"}},
                    {"type": "g_eval", "gEval": {"criteria": "Coherence"}},
                    {"type": "code_execution", "codeExecution": {"language": "python"}},
                    {"type": "html_validation", "htmlValidation": {"strict": True}},
                    {"type": "custom", "customValidator": {"fn": "myValidator"}},
                    {"type": "json_match", "expected": {"a": 1}, "negate": True},
                ],
            },
            {
                # No "name" -- exercises the tags-omitted / id-fallback path.
                "id": "tc5",
                "input": {"type": "completion", "prompt": "..."},
                "evaluations": [],
            },
        ],
    }


def _run_fixture():
    return {
        "id": "run-1",
        "suiteId": "suite-1",
        "suiteName": "Basic checks",
        "createdAt": "2026-02-01T00:00:00Z",
        "startedAt": "2026-02-01T00:00:05Z",
        "completedAt": "2026-02-01T00:10:00Z",
        "modelRuns": [
            {
                "id": "pmr-a",
                "modelRef": {"source": "openai", "modelKey": "gpt-4o"},
                "status": "completed",
                "startedAt": "2026-02-01T00:00:05Z",
                "completedAt": "2026-02-01T00:05:00Z",
                "autoDownloaded": False,
                "aggregate": {"passRate": 0.67, "avgScore": 0.72, "totalCostUsd": 0.02},
                "caseRuns": [
                    {
                        "testCaseId": "tc1",
                        "status": "completed",
                        "completedAt": "2026-02-01T00:01:00Z",
                        "result": {
                            "passed": True,
                            "output": "4",
                            "reasoning": None,
                            "metrics": {"durationMs": 1234},
                            "error": None,
                            "evalResults": [
                                {"type": "exact_match", "score": 1.0, "passed": True, "details": "matched"}
                            ],
                        },
                    },
                    {
                        "testCaseId": "tc2",
                        "status": "completed",
                        "completedAt": "2026-02-01T00:02:00Z",
                        "result": {
                            # Cannonade's own passingLogic="any" verdict: regex passed
                            # so the case as a whole passed, even though contains did
                            # not. This adapter must carry that through as-is.
                            "passed": True,
                            "output": "howdy",
                            "metrics": {"durationMs": 987},
                            "evalResults": [
                                {"type": "contains", "score": 0, "passed": False, "details": "no match"},
                                {"type": "regex", "score": 1, "passed": True, "details": "matched pattern"},
                            ],
                        },
                    },
                    {
                        "testCaseId": "tc4",
                        "status": "failed",
                        "completedAt": "2026-02-01T00:03:00Z",
                        "result": None,
                        "error": "timeout while executing code",
                    },
                ],
            },
            {
                "id": "pmr-b",
                "modelRef": {"source": "anthropic", "modelId": "claude-sonnet-5"},
                "status": "completed",
                "startedAt": "2026-02-01T00:00:05Z",
                "completedAt": "2026-02-01T00:06:00Z",
                "aggregate": {"passRate": 1.0, "avgScore": 0.85},
                "caseRuns": [
                    {
                        "testCaseId": "tc1",
                        "status": "completed",
                        "completedAt": "2026-02-01T00:01:30Z",
                        "result": {
                            "passed": True,
                            "output": "4",
                            "metrics": {"durationMs": 2000},
                            "evalResults": [
                                {
                                    "type": "llm_rubric",
                                    "score": 0.85,
                                    "passed": True,
                                    "details": "good",
                                    "judgeUsage": {
                                        "model": "gpt-4o",
                                        "inputTokens": 500,
                                        "outputTokens": 120,
                                        "totalTokens": 620,
                                        "cost": 0.001,
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "id": "pmr-c",
                "modelRef": {"source": "local", "modelKey": "llama-3"},
                "status": "failed",
                "startedAt": "2026-02-01T00:00:05Z",
                "completedAt": "2026-02-01T00:00:10Z",
                "aggregate": None,
                "caseRuns": [],
            },
        ],
    }


class TestSuiteToOpenEval:
    def test_validates_against_real_sdk(self):
        result = suite_to_openeval(_suite_fixture())
        validation = validate_suite(result)
        assert validation.valid, validation.errors

    def test_direct_grader_types_mapped_correctly(self):
        result = suite_to_openeval(_suite_fixture())
        graders = {g["id"]: g for g in result["graders"]}

        exact = graders["tc1__ec0_exact_match"]
        assert exact["type"] == "exact_match"
        assert exact["params"] == {"expected": "4"}

        contains = graders["tc2__ec0_contains"]
        assert contains["type"] == "contains"
        assert contains["params"] == {"substring": "hello", "case_sensitive": False}

        regex = graders["tc2__ec1_regex"]
        assert regex["type"] == "regex"
        assert regex["params"] == {"pattern": "^h.*"}

        similarity = graders["tc3__ec0_cosine_similarity"]
        assert similarity["type"] == "semantic_similarity"
        assert similarity["params"] == {"threshold": 0.9, "expected": "foo"}

    def test_custom_bucket_covers_all_nine_indirect_types(self):
        result = suite_to_openeval(_suite_fixture())
        graders = {g["id"]: g for g in result["graders"]}

        rubric = graders["tc4__ec0_llm_rubric"]
        assert rubric["type"] == "custom"
        assert rubric["params"]["handler"] == "cannonade:llm_rubric"
        assert rubric["params"]["rubric"] == "Is it correct?"

        geval = graders["tc4__ec1_g_eval"]
        assert geval["type"] == "custom"
        assert geval["params"]["handler"] == "cannonade:g_eval"
        assert geval["params"]["criteria"] == "Coherence"

        code_exec = graders["tc4__ec2_code_execution"]
        assert code_exec["params"]["handler"] == "cannonade:code_execution"
        assert code_exec["params"]["code_execution"] == {"language": "python"}

        html = graders["tc4__ec3_html_validation"]
        assert html["params"]["handler"] == "cannonade:html_validation"
        assert html["params"]["html_validation"] == {"strict": True}

        custom = graders["tc4__ec4_custom"]
        assert custom["params"]["handler"] == "cannonade:custom"
        assert custom["params"]["custom_validator"] == {"fn": "myValidator"}

        json_match = graders["tc4__ec5_json_match"]
        assert json_match["type"] == "custom"
        assert json_match["params"]["handler"] == "cannonade:json_match"
        assert json_match["params"]["expected"] == {"a": 1}
        assert json_match["params"]["negate"] is True

    def test_llm_rubric_and_g_eval_never_become_llm_judge(self):
        # Explicit regression guard for the module's documented design
        # decision: Cannonade suite files don't record a judge model per
        # test case, and EvalPort's llm_judge grader requires a concrete
        # params.model -- so these must never be mapped to "llm_judge".
        result = suite_to_openeval(_suite_fixture())
        for g in result["graders"]:
            if g["params"].get("handler") in ("cannonade:llm_rubric", "cannonade:g_eval"):
                assert g["type"] == "custom"

    def test_test_case_with_zero_evaluations_gets_own_scoped_grader(self):
        # Regression test: a shared "gr_default" id across the whole suite
        # would dangle here, because tc1-tc4 already define real graders --
        # only a per-test-case id keeps every reference resolvable.
        result = suite_to_openeval(_suite_fixture())
        tc5 = next(tc for tc in result["test_cases"] if tc["id"] == "tc5")
        assert tc5["graders"] == ["tc5__gr_default"]
        grader_ids = {g["id"] for g in result["graders"]}
        assert "tc5__gr_default" in grader_ids

    def test_suite_with_all_test_cases_empty_still_validates(self):
        suite = {
            "id": "empty-suite",
            "testCases": [
                {"id": "only", "input": {"type": "completion", "prompt": "hi"}, "evaluations": []}
            ],
        }
        result = suite_to_openeval(suite)
        validation = validate_suite(result)
        assert validation.valid, validation.errors
        assert result["test_cases"][0]["graders"] == ["only__gr_default"]

    def test_passing_logic_and_run_config_ride_along_as_metadata(self):
        result = suite_to_openeval(_suite_fixture())
        tc1 = next(tc for tc in result["test_cases"] if tc["id"] == "tc1")
        meta = tc1["metadata"]["cannonade"]
        assert meta["passing_logic"] == "all"
        assert meta["run_config"] == {"temperature": 0}
        assert meta["timeout_ms"] == 5000
        assert meta["raw_input"] == {"type": "completion", "prompt": "What is 2+2?"}

    def test_test_input_conversion_by_type(self):
        result = suite_to_openeval(_suite_fixture())
        by_id = {tc["id"]: tc for tc in result["test_cases"]}
        assert by_id["tc1"]["input"] == "What is 2+2?"
        assert by_id["tc2"]["input"] == [
            "system: You are helpful.",
            "user: Say hi",
        ]
        assert by_id["tc3"]["input"] == json.dumps({"a": 1}, sort_keys=True)

    def test_unknown_input_type_stringifies_defensively(self):
        suite = {
            "id": "s",
            "testCases": [
                {
                    "id": "tc",
                    "input": {"type": "future_type", "weird": True},
                    "evaluations": [{"type": "exact_match", "expected": "x"}],
                }
            ],
        }
        result = suite_to_openeval(suite)
        tc = result["test_cases"][0]
        assert isinstance(tc["input"], str)
        assert json.loads(tc["input"]) == {"type": "future_type", "weird": True}

    def test_top_level_version_is_openeval_spec_version_not_suite_content_version(self):
        result = suite_to_openeval(_suite_fixture())
        assert result["version"] != "3"  # that's Cannonade's suite content version
        assert result["metadata"]["cannonade"]["suite_version"] == "3"

    def test_tags_omitted_when_test_case_has_no_name(self):
        result = suite_to_openeval(_suite_fixture())
        tc5 = next(tc for tc in result["test_cases"] if tc["id"] == "tc5")
        assert "tags" not in tc5


class TestRunToOpenEval:
    def test_produces_one_result_set_per_model_run_all_valid(self):
        result_sets = run_to_openeval(_run_fixture())
        assert len(result_sets) == 3
        for rs in result_sets:
            validation = validate_result_set(rs)
            assert validation.valid, validation.errors

    def test_all_result_sets_share_suite_id(self):
        result_sets = run_to_openeval(_run_fixture())
        assert {rs["suite_id"] for rs in result_sets} == {"suite-1"}

    def test_sibling_grouping_uses_group_field_from_merged_rfc(self):
        result_sets = run_to_openeval(_run_fixture())
        group_ids = {rs["group"]["group_id"] for rs in result_sets}
        assert group_ids == {"run-1"}

        by_sequence = {rs["group"]["sequence"]: rs for rs in result_sets}
        assert by_sequence[0]["group"]["label"] == "openai:gpt-4o"
        assert by_sequence[1]["group"]["label"] == "anthropic:claude-sonnet-5"
        assert by_sequence[2]["group"]["label"] == "local:llama-3"
        # role mirrors label for this adapter (Cannonade has no separate
        # "role" concept beyond which model produced the ResultSet).
        assert by_sequence[0]["group"]["role"] == by_sequence[0]["group"]["label"]

    def test_run_ids_are_unique_per_model_and_traceable_to_the_run(self):
        result_sets = run_to_openeval(_run_fixture())
        run_ids = [rs["run_id"] for rs in result_sets]
        assert len(set(run_ids)) == 3
        assert all(rid.startswith("run-1:") for rid in run_ids)

    def test_passed_is_carried_through_as_is_not_rederived(self):
        result_sets = run_to_openeval(_run_fixture())
        pmr_a = next(rs for rs in result_sets if rs["group"]["sequence"] == 0)
        tc2_result = next(r for r in pmr_a["results"] if r["test_case_id"] == "tc2")
        # Cannonade's own passingLogic="any" already decided this passed
        # (regex matched even though contains didn't) -- the adapter must
        # not recompute pass/fail from the individual grader_results.
        assert tc2_result["passed"] is True
        contains_gr = next(g for g in tc2_result["grader_results"] if g["grader_id"] == "tc2__ec0_contains")
        assert contains_gr["passed"] is False

    def test_grader_ids_correlate_with_suite_side_ids(self):
        # By design (see module docstring), a ResultSet's grader_id scheme
        # is independently reproducible from just the TestRun's evalResults
        # -- it should line up with what suite_to_openeval() generated for
        # the same test case, without passing the suite into run_to_openeval.
        suite_result = suite_to_openeval(_suite_fixture())
        suite_grader_ids = {g["id"] for g in suite_result["graders"]}

        result_sets = run_to_openeval(_run_fixture())
        pmr_a = next(rs for rs in result_sets if rs["group"]["sequence"] == 0)
        tc1_result = next(r for r in pmr_a["results"] if r["test_case_id"] == "tc1")
        for gr in tc1_result["grader_results"]:
            assert gr["grader_id"] in suite_grader_ids

    def test_null_result_produces_honest_failure_not_fabricated_scores(self):
        result_sets = run_to_openeval(_run_fixture())
        pmr_a = next(rs for rs in result_sets if rs["group"]["sequence"] == 0)
        tc4_result = next(r for r in pmr_a["results"] if r["test_case_id"] == "tc4")
        assert tc4_result["passed"] is False
        assert tc4_result["grader_results"] == []
        assert tc4_result["error"] == {
            "type": "cannonade_status_failed",
            "detail": "timeout while executing code",
        }

    def test_zero_case_runs_gets_placeholder_result_not_empty_list(self):
        result_sets = run_to_openeval(_run_fixture())
        pmr_c = next(rs for rs in result_sets if rs["group"]["sequence"] == 2)
        assert len(pmr_c["results"]) == 1
        assert pmr_c["results"][0]["test_case_id"] == "cannonade_no_case_runs"
        assert pmr_c["results"][0]["passed"] is False

    def test_aggregate_metrics_preserved_verbatim_as_summary(self):
        result_sets = run_to_openeval(_run_fixture())
        pmr_a = next(rs for rs in result_sets if rs["group"]["sequence"] == 0)
        assert pmr_a["summary"] == {"passRate": 0.67, "avgScore": 0.72, "totalCostUsd": 0.02}

    def test_judge_usage_preserved_verbatim_in_grader_result_metadata(self):
        result_sets = run_to_openeval(_run_fixture())
        pmr_b = next(rs for rs in result_sets if rs["group"]["sequence"] == 1)
        tc1_result = pmr_b["results"][0]
        gr = tc1_result["grader_results"][0]
        assert gr["metadata"]["judge_usage"] == {
            "model": "gpt-4o",
            "inputTokens": 500,
            "outputTokens": 120,
            "totalTokens": 620,
            "cost": 0.001,
        }

    def test_duration_ms_read_directly_no_unit_conversion(self):
        # Cannonade's own TestCaseMetrics.durationMs is already milliseconds
        # (unlike e.g. nasde's duration_sec) -- must NOT be multiplied here.
        result_sets = run_to_openeval(_run_fixture())
        pmr_a = next(rs for rs in result_sets if rs["group"]["sequence"] == 0)
        tc1_result = next(r for r in pmr_a["results"] if r["test_case_id"] == "tc1")
        assert tc1_result["duration_ms"] == 1234

    def test_score_out_of_range_or_missing_is_clipped_or_null(self):
        run = _run_fixture()
        run["modelRuns"][0]["caseRuns"][0]["result"]["evalResults"][0]["score"] = 55.0
        result_sets = run_to_openeval(run)
        pmr_a = result_sets[0]
        gr = pmr_a["results"][0]["grader_results"][0]
        assert gr["score"] == 1.0
        assert gr["metadata"]["raw_score"] == 55.0

    def test_missing_model_identifier_falls_back_to_unknown_not_a_crash(self):
        run = _run_fixture()
        run["modelRuns"][0]["modelRef"] = {"source": "custom"}
        result_sets = run_to_openeval(run)
        assert result_sets[0]["group"]["label"] == "custom:unknown"


class TestToOpenEval:
    def test_dispatches_suite_by_shape(self):
        result = to_openeval(_suite_fixture())
        assert isinstance(result, dict)
        assert result["id"] == "suite-1"

    def test_dispatches_run_by_shape(self):
        result = to_openeval(_run_fixture())
        assert isinstance(result, list)
        assert len(result) == 3

    def test_raises_on_unrecognized_shape(self):
        with pytest.raises(ValueError, match="couldn't tell"):
            to_openeval({"somethingElse": True})


class TestOpenEvalToSuite:
    def test_round_trips_adapter_authored_metadata(self):
        original = _suite_fixture()
        converted = suite_to_openeval(original)
        recovered = openeval_to_suite(converted)

        assert recovered["id"] == "suite-1"
        assert recovered["name"] == "Basic checks"
        assert recovered["description"] == original["description"]
        assert recovered["version"] == "3"
        assert recovered["createdAt"] == "2026-01-01T00:00:00Z"
        assert recovered["updatedAt"] == "2026-01-02T00:00:00Z"

        tc1 = next(tc for tc in recovered["testCases"] if tc["id"] == "tc1")
        assert tc1["input"] == {"type": "completion", "prompt": "What is 2+2?"}
        assert tc1["passingLogic"] == "all"
        assert tc1["runConfig"] == {"temperature": 0}
        assert tc1["timeoutMs"] == 5000
        assert tc1["evaluations"] == []  # documented as non-reversible

    def test_name_falls_back_to_id_when_tag_absent(self):
        original = _suite_fixture()
        converted = suite_to_openeval(original)
        recovered = openeval_to_suite(converted)
        tc5 = next(tc for tc in recovered["testCases"] if tc["id"] == "tc5")
        assert tc5["name"] == "tc5"

    def test_is_not_a_general_importer_for_foreign_suites(self):
        # A suite built by a different EvalPort producer has no
        # metadata["cannonade"] block at all -- this must degrade
        # gracefully (defaults), not crash.
        foreign_suite = {
            "version": "1.0.0-rc.5",
            "id": "foreign",
            "test_cases": [{"id": "x", "input": "hi", "graders": ["g1"]}],
            "graders": [{"id": "g1", "type": "exact_match", "params": {"expected": "hi"}}],
        }
        recovered = openeval_to_suite(foreign_suite)
        assert recovered["version"] == "0.0.0"
        assert recovered["testCases"][0]["input"] == {"type": "completion", "prompt": "hi"}
