"""Offline tests for daimax-openeval-adapter.

Every fixture is a plain dict shaped exactly like ``EvalRun.model_dump()`` /
``PromptResult.model_dump()`` from open-daimax/daimax-appbench's ``evalapp``
package (read from the real source). When ``evalapp`` is importable the same
fixtures are ALSO fed through the real pydantic models, so a drift in the
upstream field set shows up here. No network, no API keys.
"""
from __future__ import annotations

import copy
from datetime import timedelta, timezone

import pytest
from openeval.validate import validate_result_set, validate_suite

from daimax_openeval_adapter import (
    E2E_GRADER_HANDLER,
    E2E_GRADER_TYPE,
    GRADER_E2E,
    GRADER_EXPERIENCE,
    GRADER_QUALITY,
    GRADER_SUCCESS_RATE,
    PRIORITY_WEIGHTS,
    from_openeval,
    run_from_openeval,
    run_to_openeval,
    to_openeval,
)

try:  # real upstream models, optional
    from evalapp.benchset.samples.models import EvalSample
    from evalapp.benchset.testcases.models import TestCase as DaimaxTestCase
    from evalapp.evaluation.metrics.models import (
        ExperienceMetrics,
        QualityMetrics,
        SuccessRateMetrics,
    )
    from evalapp.evaluation.results.models.execution import (
        EvalRun,
        PromptResult,
        TestCaseResult as DaimaxTestCaseResult,
    )

    HAVE_EVALAPP = True
except ImportError:  # pragma: no cover
    HAVE_EVALAPP = False


# ---------------------------------------------------------------------------
# Fixtures (dict shape == evalapp model_dump())
# ---------------------------------------------------------------------------

SUCCESS_RATE = {
    "initial_generation_rate": 100.0,
    "initial_generation_weight": 0.6,
    "initial_generation_reason": "首页无崩溃/白屏",
    "issue_fix_rate": 50.0,
    "issue_fix_weight": 0.2,
    "issue_fix_count": 2,
    "issue_fix_success_count": 1,
    "issue_fix_reason": "1/2 fixes succeeded",
    "requirement_extension_rate": 0.0,
    "requirement_extension_weight": 0.2,
    "requirement_extension_count": 0,
    "requirement_extension_success_count": 0,
    "requirement_extension_reason": "",
    "composite_score": 70.0,
}

QUALITY = {
    "usecase_completeness": 75.0,
    "e2e_pass_count": 3,
    "e2e_total_count": 4,
    "usecase_reason": "3/4 E2E cases passed",
    "stability_score": 90.0,
    "crash_count": 0,
    "anr_count": 0,
    "white_screen_count": 1,
    "crash_free": True,
    "stability_reason": "one white screen",
    "backend_completeness": 50.0,
    "backend_completeness_reason": "mock backend",
    "compliance_score": 0.0,
    "compliance_issues": [],
    "compliance_reason": "",
    "stability_deduction": 1.5,
    "backend_deduction": 11.25,
    "composite_score": 62.25,
}

EXPERIENCE = {
    "end_to_end_duration_ms": 183000.0,
    "duration_score": 80.0,
    "duration_reason": "under 5 minutes",
    "package_size_bytes": 12_345_678,
    "package_size_score": 70.0,
    "package_size_reason": "",
    "token_input": 12000,
    "token_output": 3400,
    "token_total": 15400,
    "aesthetics_score": 7.5,
    "aesthetics_reason": "clean layout",
    "aesthetics_issues": ["low contrast footer"],
    "aesthetics_dimensions": {"layout": 8, "color": 7},
    "aesthetics_rule_version": "v1",
    "aesthetics_scored_frames": ["frames/home.png"],
    "composite_score": 77.0,
}


def _tr(test_case_id, passed, status, details="", duration=0.0, **extra):
    d = {
        "test_case_id": test_case_id,
        "passed": passed,
        "status": status,
        "details": details,
        "duration": duration,
        "report_path": "",
        "report_started_at": 0.0,
        "report_generated_at": 0.0,
        "verifications": None,
    }
    d.update(extra)
    return d


FULL_PROMPT_RESULT = {
    "prompt_id": "p_todo",
    "platform": "android",
    "generator_name": "gen-x",
    "item_type": "sample",
    "sample_id": "s_todo_001",
    "sample_title": "待办清单",
    "sample_complexity": "medium",
    "sample_top_category": "工具",
    "requirement": "Build a todo app with add/complete/delete.",
    "session_id": "sess-1",
    "project_id": "proj-1",
    "generation_success": True,
    "generation_duration": 183.456,
    "project_path": "/tmp/proj-1",
    "error_message": "",
    "process_data": {
        "collector_name": "c",
        "session_id": "sess-1",
        "project_id": "proj-1",
        "work_dir": "",
        "error_type": "",
        "error_message": "",
        "token_input": 12000,
        "token_output": 3400,
        "token_total": 15400,
        "durations": {
            "understanding_ms": None,
            "planning_ms": None,
            "codegen_ms": None,
            "build_ms": None,
            "total_ms": 183000,
        },
        "raw": {},
    },
    "result_data": {
        "task_id": "",
        "system_id": "",
        "requirement": "",
        "platform": "android",
        "generation_status": "success",
        "build_status": "success",
        "install_status": "success",
        "launch_status": "success",
        "duration_build_ms": None,
        "duration_total_ms": None,
        "artifact_path": "",
        "h5_url": "",
        "e2e_result": {
            "pass_count": 0,
            "fail_count": 0,
            "total_count": 0,
            "pass_rate": 0.0,
            "test_results": [],
        },
        "cr_result": {},
        "stability_metrics": None,
        "core_function_coverage": None,
        "state_handling": None,
        "code_quality": None,
    },
    "test_results": [
        _tr("TC_LAUNCH", True, "PASS", "launched", 12.5,
            report_started_at=1758900000.0, report_generated_at=1758900012.0,
            verifications={"white_screen": {"detected": False}}),
        _tr("TC_ADD", True, "PASS", "added item", 20.0),
        _tr("TC_DELETE", False, "FAIL", "delete button not found", 25.25),
    ],
    "success_rate": SUCCESS_RATE,
    "quality": QUALITY,
    "experience": EXPERIENCE,
    "e2e_report_path": "reports/s_todo_001",
    "requires_backend": True,
}

FAILED_PROMPT_RESULT = {
    "prompt_id": "p_fail",
    "platform": "ios",
    "generator_name": "gen-x",
    "generation_success": False,
    "generation_duration": 4.0,
    "error_message": "codegen crashed: syntax error in main.dart",
    "process_data": {"error_type": "code_generation", "error_message": ""},
    "result_data": {"build_status": "failed", "install_status": "skipped", "launch_status": "failed"},
    "test_results": [],
    "success_rate": None,
    "quality": None,
    "experience": None,
}

SKIPPED_PROMPT_RESULT = {
    "prompt_id": "p_skip",
    "platform": "android",
    "generator_name": "gen-x",
    "generation_success": True,
    "generation_duration": 100.0,
    "test_results": [
        _tr("TC_LAUNCH", False, "FAIL", "app crashed on launch", 30.0),
        _tr("TC_ADD", False, "SKIPPED", "Skipped due to TC_LAUNCH failure (app failed to launch)"),
        _tr("TC_DELETE", False, "SKIPPED", "Skipped due to TC_LAUNCH failure (app failed to launch)"),
    ],
    "success_rate": None,
    "quality": None,
    "experience": None,
}

# EvalRun.timestamp is datetime.now().isoformat(): naive, no offset.
NAIVE_TIMESTAMP = "2026-09-27T02:20:36.253783"

EVAL_RUN = {
    "run_id": "run_abc123",
    "generator_name": "gen-x",
    "run_type": "sample",
    "sample_source": "appbench_v2",
    "timestamp": NAIVE_TIMESTAMP,
    "prompt_results": [FULL_PROMPT_RESULT, FAILED_PROMPT_RESULT, SKIPPED_PROMPT_RESULT],
    "summary": {
        "total_prompts": 3,
        "total_test_cases": 6,
        "total_passed": 2,
        "total_failed": 4,
        "overall_pass_rate": 2 / 6,
    },
}

DAIMAX_TEST_CASES = [
    {"id": "TC_LAUNCH", "name": "Launch", "description": "App launches", "steps": ["open app"],
     "expected_result": "home visible", "priority": "P0", "category": "launch_check"},
    {"id": "TC_ADD", "name": "Add", "description": "Add a todo", "steps": ["tap +", "type", "save"],
     "expected_result": "item listed", "priority": "P1", "category": "core_function"},
    {"id": "TC_DELETE", "name": "Delete", "description": "Delete a todo", "steps": ["swipe", "confirm"],
     "expected_result": "item gone", "priority": "P2", "category": "core_function"},
]

SAMPLES = [
    {
        "sample_id": "s_todo_001",
        "title": "待办清单",
        "requirement": "Build a todo app with add/complete/delete.",
        "platforms": ["android"],
        "app_type": "tool",
        "complexity": "medium",
        "top_category": "工具",
        "core_functions": ["add", "complete", "delete"],
        "constraints": ["offline"],
        "notes": "",
        "requires_backend": True,
        "requires_auth": False,
    },
    {"sample_id": "p_fail", "requirement": "Build a weather app.", "app_type": "general"},
    {"sample_id": "p_skip", "requirement": "Build a notes app.", "app_type": "general"},
]


def _by_id(result_set, test_case_id):
    return next(r for r in result_set["results"] if r["test_case_id"] == test_case_id)


def _grader(result, grader_id, nth=0):
    return [g for g in result["grader_results"] if g["grader_id"] == grader_id][nth]


# ---------------------------------------------------------------------------
# ResultSet export
# ---------------------------------------------------------------------------


def test_run_to_openeval_validates_against_real_validator():
    rs = run_to_openeval(EVAL_RUN)
    v = validate_result_set(rs)
    print(v.errors)
    assert v.valid, v.errors
    assert rs["suite_id"] == "appbench_v2"  # defaults to sample_source
    assert rs["run_id"] == "run_abc123"
    assert rs["provider"] == {"model": "gen-x"}
    assert rs["runner"] == {"name": "daimax-appbench"}
    assert len(rs["results"]) == 3


def test_full_run_three_synthetic_graders_keep_their_own_breakdowns():
    rs = run_to_openeval(EVAL_RUN)
    r = _by_id(rs, "s_todo_001")  # item_id prefers sample_id over prompt_id
    assert r["duration_ms"] == 183456
    assert "error" not in r
    assert r["passed"] is False  # TC_DELETE failed natively

    sr = _grader(r, GRADER_SUCCESS_RATE)
    q = _grader(r, GRADER_QUALITY)
    ex = _grader(r, GRADER_EXPERIENCE)

    # scores are composite_score / 100, raw kept
    assert sr["score"] == pytest.approx(0.70)
    assert q["score"] == pytest.approx(0.6225)
    assert ex["score"] == pytest.approx(0.77)
    assert q["metadata"]["daimax"]["composite_score_raw"] == 62.25

    # Quality owns usecase_completeness / stability_deduction / backend_deduction
    qb = q["metadata"]["daimax"]["breakdown"]
    assert qb["usecase_completeness"] == 75.0
    assert qb["stability_deduction"] == 1.5
    assert qb["backend_deduction"] == 11.25
    assert qb["e2e_pass_count"] == 3 and qb["e2e_total_count"] == 4
    assert q["metadata"]["daimax"]["normalized"]["usecase_completeness"] == pytest.approx(0.75)
    for key in ("usecase_completeness", "stability_deduction", "backend_deduction"):
        assert key not in sr["metadata"]["daimax"]["breakdown"]
        assert key not in ex["metadata"]["daimax"]["breakdown"]

    # SuccessRate owns its three rates + weights + reasons
    sb = sr["metadata"]["daimax"]["breakdown"]
    assert sb["initial_generation_rate"] == 100.0
    assert sb["initial_generation_weight"] == 0.6
    assert sb["issue_fix_rate"] == 50.0 and sb["issue_fix_weight"] == 0.2
    assert sb["requirement_extension_rate"] == 0.0
    assert sb["initial_generation_reason"] == "首页无崩溃/白屏"
    assert sr["reason"] == "首页无崩溃/白屏"

    # Experience owns duration / package / token / aesthetics
    eb = ex["metadata"]["daimax"]["breakdown"]
    assert eb["duration_score"] == 80.0
    assert eb["package_size_score"] == 70.0
    assert eb["token_total"] == 15400
    assert eb["aesthetics_score"] == 7.5  # raw 0..10 kept
    assert ex["metadata"]["daimax"]["normalized"]["aesthetics_score"] == pytest.approx(0.75)
    assert ex["metadata"]["daimax"]["normalized"]["duration_score"] == pytest.approx(0.80)

    # Result-level daimax metadata
    m = r["metadata"]["daimax"]
    assert m["prompt_id"] == "p_todo" and m["sample_id"] == "s_todo_001"
    assert m["platform"] == "android"
    assert m["requires_backend"] is True
    assert m["tokens"] == {"token_input": 12000, "token_output": 3400, "token_total": 15400}
    assert m["result_data"]["build_status"] == "success"
    assert m["adapter_policy"]["name"] == "native_booleans"


def test_e2e_grader_results_use_namespaced_type_and_native_pass():
    rs = run_to_openeval(EVAL_RUN, test_cases=DAIMAX_TEST_CASES)
    r = _by_id(rs, "s_todo_001")
    e2e = [g for g in r["grader_results"] if g["grader_id"] == GRADER_E2E]
    assert len(e2e) == 3
    assert {g["type"] for g in e2e} == {E2E_GRADER_TYPE} == {"org.daimax.vision_e2e_step"}

    launch = next(g for g in e2e if g["metadata"]["daimax"]["test_case_id"] == "TC_LAUNCH")
    assert launch["score"] == 1.0 and launch["passed"] is True
    assert launch["metadata"]["daimax"]["status"] == "PASS"
    assert launch["metadata"]["daimax"]["duration_ms"] == 12500
    assert launch["metadata"]["daimax"]["verifications"] == {"white_screen": {"detected": False}}
    assert launch["metadata"]["daimax"]["report_started_at"] == 1758900000.0
    # P0/P1/P2 -> 3/2/1 convention
    assert launch["metadata"]["daimax"]["priority"] == "P0"
    assert launch["metadata"]["daimax"]["priority_weight"] == 3

    delete = next(g for g in e2e if g["metadata"]["daimax"]["test_case_id"] == "TC_DELETE")
    assert delete["score"] == 0.0 and delete["passed"] is False
    assert delete["reason"] == "delete button not found"
    assert delete["metadata"]["daimax"]["priority_weight"] == 1

    # the shared grader definition keeps the original handler
    e2e_def = next(g for g in rs["metadata"]["daimax"]["graders"] if g["id"] == GRADER_E2E)
    assert e2e_def["type"] == E2E_GRADER_TYPE
    assert e2e_def["params"]["handler"] == E2E_GRADER_HANDLER == "daimax:vision_e2e_step"
    assert PRIORITY_WEIGHTS == {"P0": 3, "P1": 2, "P2": 1}


def test_metrics_none_give_score_null():
    rs = run_to_openeval(EVAL_RUN)
    r = _by_id(rs, "p_skip")
    for gid in (GRADER_SUCCESS_RATE, GRADER_QUALITY, GRADER_EXPERIENCE):
        g = _grader(r, gid)
        assert g["score"] is None
        assert g["metadata"]["daimax"]["skip_reason"] == "metric_not_computed"
        assert "breakdown" not in g["metadata"]["daimax"]
        # threshold None: passed is the native generation_success (True here)
        assert g["passed"] is True
    assert validate_result_set(rs).valid


def test_failed_generation_maps_to_error_and_native_false():
    rs = run_to_openeval(EVAL_RUN)
    r = _by_id(rs, "p_fail")  # no sample_id -> item_id falls back to prompt_id
    assert r["passed"] is False
    assert r["error"] == {
        "type": "provider_error",
        "message": "codegen crashed: syntax error in main.dart",
        "code": "code_generation",
    }
    assert r["duration_ms"] == 4000
    for gid in (GRADER_SUCCESS_RATE, GRADER_QUALITY, GRADER_EXPERIENCE):
        g = _grader(r, gid)
        assert g["score"] is None and g["passed"] is False
    assert r["metadata"]["daimax"]["result_data"] == {
        "build_status": "failed", "install_status": "skipped", "launch_status": "failed"
    }
    assert r["metadata"]["daimax"]["process_error_type"] == "code_generation"


def test_failed_generation_timeout_error_type():
    pr = copy.deepcopy(FAILED_PROMPT_RESULT)
    pr["process_data"] = {"error_type": "generation_timeout"}
    pr["error_message"] = ""
    rs = run_to_openeval({"run_id": "r", "timestamp": NAIVE_TIMESTAMP, "prompt_results": [pr]})
    err = rs["results"][0]["error"]
    assert err["type"] == "timeout"
    assert "message" not in err  # no error_message -> no fabricated message
    assert err["code"] == "generation_timeout"
    assert validate_result_set(rs).valid


def test_successful_generation_never_has_error_even_with_stale_message():
    pr = copy.deepcopy(FULL_PROMPT_RESULT)
    pr["error_message"] = "leftover warning text"
    rs = run_to_openeval({"run_id": "r", "timestamp": NAIVE_TIMESTAMP, "prompt_results": [pr]})
    assert "error" not in rs["results"][0]
    assert rs["results"][0]["metadata"]["daimax"]["error_message"] == "leftover warning text"


def test_skipped_statuses_give_score_null_and_keep_native_passed():
    rs = run_to_openeval(EVAL_RUN)
    r = _by_id(rs, "p_skip")
    e2e = [g for g in r["grader_results"] if g["grader_id"] == GRADER_E2E]
    by_tc = {g["metadata"]["daimax"]["test_case_id"]: g for g in e2e}
    assert by_tc["TC_LAUNCH"]["score"] == 0.0 and by_tc["TC_LAUNCH"]["passed"] is False
    for tc_id in ("TC_ADD", "TC_DELETE"):
        g = by_tc[tc_id]
        assert g["score"] is None
        assert g["passed"] is False  # daimax's own passed=False kept intact
        assert g["metadata"]["daimax"]["status"] == "SKIPPED"
        assert g["metadata"]["daimax"]["skip_reason"] == "SKIPPED"
        assert g["reason"].startswith("Skipped due to TC_LAUNCH failure")
    assert r["passed"] is False
    assert "error" not in r  # generation itself succeeded
    assert validate_result_set(rs).valid


def test_threshold_none_vs_threshold_provided():
    # default: nothing derived from scores
    rs_native = run_to_openeval(EVAL_RUN)
    r = _by_id(rs_native, "s_todo_001")
    assert _grader(r, GRADER_QUALITY)["passed"] is True  # generation_success, not 62.25 >= anything
    assert rs_native["metadata"]["daimax"]["adapter_policy"] == {
        "name": "native_booleans",
        "pass_threshold": None,
        "description": rs_native["metadata"]["daimax"]["adapter_policy"]["description"],
    }

    # explicit 70-point cutoff on daimax's native scale
    rs_70 = run_to_openeval(EVAL_RUN, pass_threshold=70)
    r70 = _by_id(rs_70, "s_todo_001")
    assert _grader(r70, GRADER_SUCCESS_RATE)["passed"] is True   # 70.0 >= 70
    assert _grader(r70, GRADER_QUALITY)["passed"] is False       # 62.25 < 70
    assert _grader(r70, GRADER_EXPERIENCE)["passed"] is True     # 77.0 >= 70
    # scores are unchanged by the policy
    assert _grader(r70, GRADER_QUALITY)["score"] == _grader(r, GRADER_QUALITY)["score"]
    policy = rs_70["metadata"]["daimax"]["adapter_policy"]
    assert policy["name"] == "composite_threshold"
    assert policy["pass_threshold"] == 70.0
    assert policy["pass_threshold_normalized"] == pytest.approx(0.70)
    assert "not a daimax-wide pass criterion" in policy["description"]
    assert r70["metadata"]["daimax"]["adapter_policy"] == policy
    # missing metric cannot reach a threshold verdict
    skip = _by_id(rs_70, "p_skip")
    assert _grader(skip, GRADER_QUALITY)["score"] is None
    assert _grader(skip, GRADER_QUALITY)["passed"] is False
    assert validate_result_set(rs_70).valid

    with pytest.raises(ValueError):
        run_to_openeval(EVAL_RUN, pass_threshold=101)  # outside daimax's 0..100 scale
    with pytest.raises(ValueError):
        run_to_openeval(EVAL_RUN, pass_threshold=-1)


def test_naive_timestamp_handling():
    rs = run_to_openeval(EVAL_RUN)
    assert rs["started_at"] == "2026-09-27T02:20:36.253783+00:00"
    assert rs["metadata"]["daimax"]["timestamp_raw"] == NAIVE_TIMESTAMP
    assert rs["metadata"]["daimax"]["timestamp_assumed_timezone"] == "UTC"

    cst = timezone(timedelta(hours=8))
    rs_cst = run_to_openeval(EVAL_RUN, assume_timezone=cst)
    assert rs_cst["started_at"] == "2026-09-27T02:20:36.253783+08:00"
    assert rs_cst["metadata"]["daimax"]["timestamp_assumed_timezone"] == "UTC+08:00"

    rs_none = run_to_openeval(EVAL_RUN, assume_timezone=None)
    assert rs_none["started_at"] == NAIVE_TIMESTAMP
    assert rs_none["metadata"]["daimax"]["timestamp_assumed_timezone"] is None

    # an offset-bearing timestamp is never rewritten
    run = dict(EVAL_RUN, timestamp="2026-09-27T02:20:36+02:00")
    rs_aware = run_to_openeval(run)
    assert rs_aware["started_at"] == "2026-09-27T02:20:36+02:00"
    assert rs_aware["metadata"]["daimax"]["timestamp_assumed_timezone"] is None


def test_summary_is_schema_shaped_and_native_summary_kept_in_metadata():
    rs = run_to_openeval(EVAL_RUN)
    s = rs["summary"]
    assert s["total"] == 3 and s["passed"] == 0 and s["failed"] == 3
    assert s["pass_rate"] == 0.0
    assert s["duration_ms"] == 183456 + 4000 + 100000
    assert 0.0 <= s["avg_score"] <= 1.0
    assert set(s) <= {"total", "passed", "failed", "skipped", "pass_rate", "avg_score", "duration_ms", "by_grader"}
    assert rs["metadata"]["daimax"]["summary"]["total_test_cases"] == 6


def test_run_to_openeval_rejects_empty_run():
    with pytest.raises(ValueError):
        run_to_openeval({"run_id": "r", "timestamp": NAIVE_TIMESTAMP, "prompt_results": []})


def test_run_to_openeval_rejects_prompt_result_without_ids():
    bad = {"platform": "android", "generator_name": "g", "generation_success": True}
    with pytest.raises(ValueError):
        run_to_openeval({"run_id": "r", "timestamp": NAIVE_TIMESTAMP, "prompt_results": [bad]})


# ---------------------------------------------------------------------------
# Reverse direction
# ---------------------------------------------------------------------------


def test_run_from_openeval_round_trips_prompt_results():
    rs = run_to_openeval(EVAL_RUN)
    back = run_from_openeval(rs)
    assert [p["prompt_id"] for p in back] == ["p_todo", "p_fail", "p_skip"]

    full = back[0]
    assert full["sample_id"] == "s_todo_001"
    assert full["platform"] == "android"
    assert full["generation_success"] is True
    assert full["generation_duration"] == pytest.approx(183.456)
    assert full["requires_backend"] is True
    assert full["success_rate"] == SUCCESS_RATE
    assert full["quality"] == QUALITY
    assert full["experience"] == EXPERIENCE
    assert [(t["test_case_id"], t["passed"], t["status"]) for t in full["test_results"]] == [
        ("TC_LAUNCH", True, "PASS"), ("TC_ADD", True, "PASS"), ("TC_DELETE", False, "FAIL")
    ]
    assert full["test_results"][0]["duration"] == 12.5
    assert full["test_results"][0]["verifications"] == {"white_screen": {"detected": False}}

    failed = back[1]
    assert failed["generation_success"] is False
    assert failed["error_message"] == "codegen crashed: syntax error in main.dart"
    assert failed["quality"] is None
    assert failed["result_data"]["build_status"] == "failed"

    skipped = back[2]
    assert [t["status"] for t in skipped["test_results"]] == ["FAIL", "SKIPPED", "SKIPPED"]


def test_run_from_openeval_skips_foreign_results():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [{"test_case_id": "x", "passed": True,
                     "grader_results": [{"grader_id": "g", "type": "exact_match", "score": 1.0, "passed": True}]}],
    }
    assert run_from_openeval(rs) == []


# ---------------------------------------------------------------------------
# Suite (bonus) + id sharing
# ---------------------------------------------------------------------------


def test_to_openeval_suite_validates_and_carries_priority_convention():
    suite = to_openeval(SAMPLES, test_cases={"s_todo_001": DAIMAX_TEST_CASES}, suite_id="appbench_v2")
    v = validate_suite(suite)
    print(v.errors)
    assert v.valid, v.errors
    assert {g["id"] for g in suite["graders"]} == {
        GRADER_SUCCESS_RATE, GRADER_QUALITY, GRADER_EXPERIENCE, GRADER_E2E
    }
    e2e_def = next(g for g in suite["graders"] if g["id"] == GRADER_E2E)
    assert e2e_def["type"] == "org.daimax.vision_e2e_step"
    assert e2e_def["params"]["handler"] == "daimax:vision_e2e_step"

    tc = suite["test_cases"][0]
    assert tc["id"] == "s_todo_001"
    assert tc["input"] == "Build a todo app with add/complete/delete."
    assert tc["tags"] == ["tool", "工具", "medium"]
    m = tc["metadata"]["daimax"]
    assert m["requires_backend"] is True
    assert [(t["id"], t["priority"], t["priority_weight"]) for t in m["test_cases"]] == [
        ("TC_LAUNCH", "P0", 3), ("TC_ADD", "P1", 2), ("TC_DELETE", "P2", 1)
    ]
    assert m["priority_weight_convention"] == {"P0": 3, "P1": 2, "P2": 1}


def test_from_openeval_round_trips_samples():
    suite = to_openeval(SAMPLES, test_cases={"s_todo_001": DAIMAX_TEST_CASES})
    back = from_openeval(suite)
    assert len(back) == 3
    s = back[0]
    for key, value in SAMPLES[0].items():
        assert s[key] == value
    assert s["test_cases"] == DAIMAX_TEST_CASES  # priority_weight stripped again
    assert "test_cases" not in back[1]


def test_to_openeval_rejects_empty_or_missing_requirement():
    with pytest.raises(ValueError):
        to_openeval([])
    with pytest.raises(ValueError):
        to_openeval([{"sample_id": "x", "requirement": ""}])
    with pytest.raises(ValueError):
        to_openeval([{"requirement": "no id"}])


def test_suite_and_result_set_share_test_case_ids():
    suite = to_openeval(SAMPLES, suite_id="appbench_v2")
    rs = run_to_openeval(EVAL_RUN, suite_id="appbench_v2")
    assert {tc["id"] for tc in suite["test_cases"]} == {r["test_case_id"] for r in rs["results"]}
    assert rs["suite_id"] == suite["id"]
    suite_grader_ids = {g["id"] for g in suite["graders"]}
    for r in rs["results"]:
        assert {g["grader_id"] for g in r["grader_results"]} <= suite_grader_ids
    assert validate_suite(suite).valid
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# Real evalapp pydantic models (only when installed; same fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_EVALAPP, reason="evalapp (daimax-appbench) not installed")
def test_real_evalapp_models_produce_identical_result_set():
    run = EvalRun(
        run_id="run_abc123",
        generator_name="gen-x",
        run_type="sample",
        sample_source="appbench_v2",
        timestamp=NAIVE_TIMESTAMP,
        prompt_results=[
            PromptResult(**FULL_PROMPT_RESULT),
            PromptResult(**FAILED_PROMPT_RESULT),
            PromptResult(**SKIPPED_PROMPT_RESULT),
        ],
    )
    assert run.prompt_results[0].item_id == "s_todo_001"
    assert isinstance(run.prompt_results[0].quality, QualityMetrics)
    assert isinstance(run.prompt_results[0].success_rate, SuccessRateMetrics)
    assert isinstance(run.prompt_results[0].experience, ExperienceMetrics)
    assert isinstance(run.prompt_results[0].test_results[0], DaimaxTestCaseResult)

    from_models = run_to_openeval(run, test_cases=[DaimaxTestCase(**tc) for tc in DAIMAX_TEST_CASES])
    from_dump = run_to_openeval(run.model_dump(), test_cases=DAIMAX_TEST_CASES)
    v = validate_result_set(from_models)
    print(v.errors)
    assert v.valid, v.errors

    # The pydantic path only differs from the hand-written dict path by the
    # defaults pydantic fills in on the compact fixtures; compare the
    # model_dump-driven output (fully defaulted) against the model-driven one.
    assert from_models == from_dump

    # And the recovered PromptResult dicts re-validate as real models.
    for pr in run_from_openeval(from_models):
        PromptResult(**pr)


@pytest.mark.skipif(not HAVE_EVALAPP, reason="evalapp (daimax-appbench) not installed")
def test_real_evalapp_samples_and_test_cases_build_a_valid_suite():
    samples = [EvalSample(**s) for s in SAMPLES]
    tcs = {"s_todo_001": [DaimaxTestCase(**tc) for tc in DAIMAX_TEST_CASES]}
    suite = to_openeval(samples, test_cases=tcs, suite_id="appbench_v2")
    assert validate_suite(suite).valid, validate_suite(suite).errors
    assert suite["test_cases"][0]["metadata"]["daimax"]["test_cases"][0]["priority_weight"] == 3
    # legacy priority strings normalize through the real model, then map
    legacy = DaimaxTestCase(id="TC_X", name="x", description="d", priority="high")
    assert legacy.priority == "P0"
    s2 = to_openeval(samples[:1], test_cases={"s_todo_001": [legacy]})
    assert s2["test_cases"][0]["metadata"]["daimax"]["test_cases"][0]["priority_weight"] == 3
    for s in from_openeval(suite):
        EvalSample(**{k: v for k, v in s.items() if k != "test_cases"})
