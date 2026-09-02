"""Tests for benchflow-openeval-adapter.

Every `results.jsonl` row exercised here is produced by BenchFlow's own
`benchflow.trajectories.results.build_rollout_results_record` (not a
hand-typed fixture that could silently drift from the real row shape), and
every EvalPort document produced by the adapter is checked against the real,
installed `evalport-sdk`'s `openeval.validate.validate_result_set` -- both
conventions this repo's other adapters already use (see e.g.
`opik-openeval-adapter/tests/test_adapter.py`).
"""
from __future__ import annotations

import pytest
from openeval.validate import validate_result_set

from benchflow.trajectories.results import build_rollout_results_record

from benchflow_openeval_adapter import (
    job_results_file_to_openeval,
    job_results_to_openeval,
)


def _row(tmp_path, **overrides):
    rollout_dir = tmp_path / overrides.pop("rollout_name", "rollout_0")
    rollout_dir.mkdir(exist_ok=True)
    kwargs = dict(
        rollout_dir=rollout_dir,
        task_name="swe-bench/django__django-11848",
        rollout_name=rollout_dir.name,
        agent="openclaw",
        agent_name="openclaw-agent",
        model="anthropic/claude-sonnet-4-5",
        n_tool_calls=3,
        prompts=["Fix the failing test in django/db/models/query.py"],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
    )
    kwargs.update(overrides)
    return build_rollout_results_record(**kwargs)


# ---------------------------------------------------------------------------
# Real BenchFlow rows, real EvalPort validation
# ---------------------------------------------------------------------------


def test_passed_rollout_reward_exactly_one(tmp_path):
    row = _row(tmp_path, rewards={"reward": 1.0})
    rs = job_results_to_openeval([row], suite_id="swe-bench", run_id="run_1")

    v = validate_result_set(rs)
    assert v.valid, v.errors

    result = rs["results"][0]
    assert result["test_case_id"] == "swe-bench/django__django-11848"
    assert result["passed"] is True
    assert "attempt" not in result  # single row for this task_id

    reward_gr = next(g for g in result["grader_results"] if g["grader_id"] == "bf_reward")
    assert reward_gr["score"] == 1.0
    assert reward_gr["passed"] is True
    assert reward_gr["metadata"]["openeval"]["raw_score"] == 1.0


def test_partial_credit_reward_is_not_promoted_to_a_pass(tmp_path):
    """BenchFlow's own convention: reward == 1.0 is the only passing value.

    A 0.9 reward -- which a naive >= 0.5 threshold would call a pass -- must
    come back as passed: False, matching
    benchflow._utils.scoring.classify_result exactly.
    """
    row = _row(tmp_path, rewards={"reward": 0.9})
    rs = job_results_to_openeval([row], suite_id="swe-bench", run_id="run_1")
    assert validate_result_set(rs).valid

    result = rs["results"][0]
    assert result["passed"] is False
    reward_gr = next(g for g in result["grader_results"] if g["grader_id"] == "bf_reward")
    assert reward_gr["score"] == 0.9
    assert reward_gr["passed"] is False


def test_named_sub_scores_become_their_own_graders_without_duplicating_reward(tmp_path):
    row = _row(
        tmp_path,
        rewards={
            "reward": 1.0,
            "rubric": [
                {"name": "code_style", "score": 0.8},
                {"name": "test_coverage", "score": 1.0},
            ],
        },
    )
    rs = job_results_to_openeval([row], suite_id="swe-bench", run_id="run_1")
    assert validate_result_set(rs).valid

    grader_ids = [g["grader_id"] for g in rs["results"][0]["grader_results"]]
    assert grader_ids.count("bf_reward") == 1  # never duplicated
    assert "bf_code_style" in grader_ids
    assert "bf_test_coverage" in grader_ids

    code_style = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "bf_code_style")
    assert code_style["score"] == 0.8
    assert code_style["passed"] is False  # 0.8 != its own range max (1.0)

    coverage = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "bf_test_coverage")
    assert coverage["passed"] is True  # hit its own range max


def test_overall_passed_follows_reward_not_a_strict_and_of_every_grader(tmp_path):
    """A rollout with reward == 1.0 but an imperfect rubric sub-score must
    still be Result.passed == True -- BenchFlow's own pass/fail definition
    is reward-only (see classify_result), and demoting the whole result over
    a non-authoritative sub-metric would contradict what BenchFlow itself
    would report for this same rollout.
    """
    row = _row(
        tmp_path,
        rewards={"reward": 1.0, "rubric": [{"name": "style", "score": 0.2}]},
    )
    rs = job_results_to_openeval([row], suite_id="swe-bench", run_id="run_1")
    assert validate_result_set(rs).valid
    assert rs["results"][0]["passed"] is True


def test_verifier_error_is_unscored_not_a_fabricated_zero(tmp_path):
    row = _row(
        tmp_path,
        rewards=None,
        error=None,
        verifier_error="verifier container exited with code 137",
    )
    rs = job_results_to_openeval([row], suite_id="swe-bench", run_id="run_1")
    assert validate_result_set(rs).valid

    result = rs["results"][0]
    assert result["passed"] is False
    reward_gr = next(g for g in result["grader_results"] if g["grader_id"] == "bf_reward")
    assert reward_gr["score"] is None  # not a fabricated 0.0
    assert reward_gr["passed"] is False
    assert result["error"]["type"] == "runner_error"  # closed enum, see _row_error
    assert "verifier container exited" in result["error"]["message"]
    assert result["metadata"]["benchflow"]["error_category"] == "verifier_error"
    assert result["metadata"]["openeval"]["aggregation_status"] == "unscored"
    assert rs["metadata"]["openeval"]["unscored_count"] == 1
    assert rs["summary"]["passed"] == 0
    assert rs["summary"]["failed"] == 0  # unscored, not failed
    assert rs["summary"]["avg_score"] == 0  # no scored graders at all


def test_agent_error_before_verification_is_unscored(tmp_path):
    row = _row(tmp_path, rewards=None, error="agent process crashed", verifier_error=None)
    rs = job_results_to_openeval([row], suite_id="swe-bench", run_id="run_1")
    assert validate_result_set(rs).valid
    reward_gr = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "bf_reward")
    assert reward_gr["score"] is None
    assert rs["results"][0]["metadata"]["benchflow"]["error_category"] == "agent_error"


def test_export_error_after_successful_verification_keeps_the_real_reward(tmp_path):
    """The verifier already produced a real reward before export failed --
    that reward must NOT be nulled out just because export_error is set.
    """
    row = _row(tmp_path, rewards={"reward": 1.0}, export_error="failed to write skill export")
    rs = job_results_to_openeval([row], suite_id="swe-bench", run_id="run_1")
    assert validate_result_set(rs).valid

    result = rs["results"][0]
    reward_gr = next(g for g in result["grader_results"] if g["grader_id"] == "bf_reward")
    assert reward_gr["score"] == 1.0  # real reward preserved
    assert reward_gr["passed"] is True
    assert result["error"] is not None  # export_error still surfaced
    assert result["metadata"]["benchflow"]["error_category"] == "export_error"


def test_repeated_trials_of_the_same_task_get_ascending_attempt_numbers(tmp_path):
    rows = [
        _row(tmp_path, rollout_name="trial_1", rewards={"reward": 1.0}),
        _row(tmp_path, rollout_name="trial_2", rewards={"reward": 0.0}),
        _row(tmp_path, rollout_name="trial_3", rewards={"reward": 1.0}),
    ]
    rs = job_results_to_openeval(
        rows, suite_id="swe-bench", run_id="run_stability", isolation="fresh"
    )
    assert validate_result_set(rs).valid
    assert rs["isolation"] == "fresh"

    attempts = [r["attempt"] for r in rs["results"]]
    assert attempts == [1, 2, 3]
    assert all(r["test_case_id"] == "swe-bench/django__django-11848" for r in rs["results"])
    # (test_case_id, run_id, attempt) uniqueness -- validate_result_set's own check.
    assert len({(r["test_case_id"], a) for r, a in zip(rs["results"], attempts)}) == 3


def test_single_row_per_task_gets_no_attempt_field(tmp_path):
    row = _row(tmp_path)
    rs = job_results_to_openeval([row], suite_id="swe-bench", run_id="run_1")
    assert "attempt" not in rs["results"][0]


def test_two_different_tasks_are_not_grouped_as_attempts_of_each_other(tmp_path):
    rows = [
        _row(tmp_path, task_name="task_a", rollout_name="a_0"),
        _row(tmp_path, task_name="task_b", rollout_name="b_0"),
    ]
    rs = job_results_to_openeval(rows, suite_id="s", run_id="run_1")
    assert validate_result_set(rs).valid
    ids = {r["test_case_id"] for r in rs["results"]}
    assert ids == {"task_a", "task_b"}
    assert all("attempt" not in r for r in rs["results"])


def test_wider_reward_range_is_normalized_not_truncated(tmp_path):
    """A task whose verifier is documented to score in [-1, 1]: -0.5 and
    -0.9 must land at different normalized scores (0.25 vs 0.05), not both
    collapse to a clamped 0.0 the way a naive max(0, min(1, x)) would.
    """
    row_a = _row(tmp_path, rollout_name="a", rewards={"reward": -0.5})
    row_b = _row(tmp_path, rollout_name="b", task_name="other_task", rewards={"reward": -0.9})
    rs = job_results_to_openeval(
        [row_a, row_b], suite_id="s", run_id="run_1", reward_range=(-1.0, 1.0)
    )
    assert validate_result_set(rs).valid

    scores = {
        r["test_case_id"]: next(g for g in r["grader_results"] if g["grader_id"] == "bf_reward")["score"]
        for r in rs["results"]
    }
    assert scores["swe-bench/django__django-11848"] == pytest.approx(0.25)
    assert scores["other_task"] == pytest.approx(0.05)
    # Raw, un-normalized values are always preserved.
    raw = {
        r["test_case_id"]: next(g for g in r["grader_results"] if g["grader_id"] == "bf_reward")["metadata"]["openeval"]["raw_score"]
        for r in rs["results"]
    }
    assert raw["swe-bench/django__django-11848"] == -0.5
    assert raw["other_task"] == -0.9


def test_reward_outside_declared_range_is_defensively_clamped(tmp_path):
    row = _row(tmp_path, rewards={"reward": 5.0})  # outside default (0, 1)
    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    assert validate_result_set(rs).valid  # would fail schema validation without the clamp
    reward_gr = next(g for g in rs["results"][0]["grader_results"] if g["grader_id"] == "bf_reward")
    assert reward_gr["score"] == 1.0  # clamped for the schema-required field...
    assert reward_gr["metadata"]["openeval"]["raw_score"] == 5.0  # ...but the real value is not lost


def test_metadata_carries_agent_model_and_tool_call_info(tmp_path):
    row = _row(tmp_path, agent="openclaw", model="anthropic/claude-sonnet-4-5", n_tool_calls=7)
    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    meta = rs["results"][0]["metadata"]["benchflow"]
    assert meta["agent"] == "openclaw"
    assert meta["model"] == "anthropic/claude-sonnet-4-5"
    assert meta["total_tool_calls"] == 7.0
    assert meta["stop_condition"] == "agent_completed"


def test_duration_ms_from_timing_total_seconds(tmp_path):
    row = _row(tmp_path, timing={"total": 12.5, "agent_execution": 10.0})
    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    assert rs["results"][0]["duration_ms"] == 12500


def test_no_timing_omits_duration_ms(tmp_path):
    row = _row(tmp_path, timing=None)
    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    assert "duration_ms" not in rs["results"][0]


def test_summary_stats_across_a_mixed_job(tmp_path):
    rows = [
        _row(tmp_path, rollout_name="p1", task_name="t1", rewards={"reward": 1.0}),
        _row(tmp_path, rollout_name="f1", task_name="t2", rewards={"reward": 0.0}),
        _row(tmp_path, rollout_name="u1", task_name="t3", rewards=None, verifier_error="boom"),
    ]
    rs = job_results_to_openeval(rows, suite_id="s", run_id="run_1")
    assert validate_result_set(rs).valid
    assert rs["summary"]["total"] == 3
    assert rs["summary"]["passed"] == 1
    assert rs["summary"]["failed"] == 1
    assert rs["metadata"]["openeval"]["unscored_count"] == 1
    assert rs["summary"]["avg_score"] == pytest.approx(0.5)  # avg of the two *scored* rewards (1.0, 0.0)


def test_empty_rows_raises_instead_of_emitting_an_invalid_document():
    with pytest.raises(ValueError):
        job_results_to_openeval([], suite_id="s", run_id="run_1")


def test_started_at_and_completed_at_default_sanely(tmp_path):
    row = _row(tmp_path)
    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    assert validate_result_set(rs).valid
    assert rs["started_at"] == rs["completed_at"]

    rs2 = job_results_to_openeval(
        [row], suite_id="s", run_id="run_1", started_at="2026-08-30T09:00:00Z"
    )
    assert rs2["started_at"] == "2026-08-30T09:00:00Z"
    assert rs2["completed_at"] == "2026-08-30T09:00:00Z"


def test_runner_name_is_benchflow(tmp_path):
    row = _row(tmp_path)
    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    assert rs["runner"]["name"] == "benchflow"
    assert "version" not in rs["runner"]

    rs2 = job_results_to_openeval([row], suite_id="s", run_id="run_1", runner_version="0.7.6")
    assert rs2["runner"]["version"] == "0.7.6"


def test_actual_output_extracted_from_completion_when_present(tmp_path):
    row = _row(
        tmp_path,
        trajectory=[],
        prompts=["say hi"],
    )
    # build_rollout_results_record derives `completion` from a real
    # trajectory/llm_trajectory.jsonl file (see _llm_steps_from_trajectory);
    # without one, completion is None -- exercised by the "omitted" test
    # below. This test instead exercises the extraction helper directly
    # against a completion shape the row *would* carry if a trajectory file
    # were present, since constructing a fully valid llm_trajectory.jsonl
    # exchange is far more machinery than this adapter's own logic warrants
    # testing through.
    from benchflow_openeval_adapter import _stringify_completion

    assert _stringify_completion([{"role": "assistant", "content": "The fix is..."}]) == "The fix is..."
    assert _stringify_completion([{"role": "assistant", "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}]) == "part1\npart2"
    assert _stringify_completion(None) is None
    assert _stringify_completion([]) is None

    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    assert "actual_output" not in rs["results"][0]  # no trajectory file -> no completion


def test_isolation_is_not_set_unless_explicitly_passed(tmp_path):
    row = _row(tmp_path)
    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    assert "isolation" not in rs


def test_invalid_metric_range_raises():
    from benchflow_openeval_adapter import _normalize

    with pytest.raises(ValueError):
        _normalize(0.5, (1.0, 0.0))  # high <= low


# ---------------------------------------------------------------------------
# Internal helpers against malformed/minimal rows -- real
# build_rollout_results_record never produces these shapes (its own return
# always has a populated `info` dict, `timing` dict, etc.), but every
# fallback branch that guards against a row not shaped exactly like that
# should still behave correctly rather than crash, since a caller can feed
# job_results_to_openeval any dict shaped roughly like results.jsonl (e.g. a
# hand-edited or third-party-produced file), not only BenchFlow's own output.
# ---------------------------------------------------------------------------


def test_test_case_id_falls_back_through_the_full_chain():
    from benchflow_openeval_adapter import _test_case_id

    assert _test_case_id({"info": {"task_id": "tid"}}) == "tid"
    assert _test_case_id({"info": {"task_name": "tname"}}) == "tname"
    assert _test_case_id({"task_id": "top_level_tid"}) == "top_level_tid"
    assert _test_case_id({"task_name": "top_level_tname"}) == "top_level_tname"
    assert _test_case_id({"example_id": 3}) == "3"
    assert _test_case_id({}) == "unknown_task"


def test_row_metadata_carries_reward_details_and_token_usage():
    from benchflow_openeval_adapter import _row_metadata

    row = {
        "info": {"reward_details": {"exact_match": True}},
        "token_usage": {"input_tokens": 100, "output_tokens": 0, "final_input_tokens": 0, "final_output_tokens": 0, "total_tokens": 100},
    }
    meta = _row_metadata(row)["benchflow"]
    assert meta["reward_details"] == {"exact_match": True}
    assert meta["token_usage"]["input_tokens"] == 100


def test_row_metadata_omits_all_zero_token_usage():
    from benchflow_openeval_adapter import _row_metadata

    row = {"token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
    assert "token_usage" not in _row_metadata(row)["benchflow"]


def test_row_metadata_is_empty_dict_for_a_bare_row():
    from benchflow_openeval_adapter import _row_metadata

    assert _row_metadata({}) == {"benchflow": {}}


def test_duration_ms_handles_missing_or_malformed_timing():
    from benchflow_openeval_adapter import _duration_ms

    assert _duration_ms({}) is None
    assert _duration_ms({"timing": "not a dict"}) is None
    assert _duration_ms({"timing": {}}) is None
    assert _duration_ms({"timing": {"total": "not a number"}}) is None
    assert _duration_ms({"timing": {"total": True}}) is None  # bool excluded despite being an int subclass


def test_row_error_ignores_non_dict_error():
    from benchflow_openeval_adapter import _row_error

    assert _row_error({"error": None}) is None
    assert _row_error({"error": "a bare string, not the expected dict shape"}) is None


def test_stringify_completion_skips_non_dict_messages():
    from benchflow_openeval_adapter import _stringify_completion

    assert _stringify_completion([{"role": "assistant", "content": "hi"}, "not a dict", {"content": 42}]) == "hi"


def test_end_to_end_row_with_completion_and_a_non_numeric_metric_value():
    """A raw dict shaped like results.jsonl (not routed through
    build_rollout_results_record) -- exercises `actual_output` extraction
    end-to-end through `job_results_to_openeval`, and confirms a
    non-numeric `metrics` value (which real BenchFlow never emits, but a
    third-party-produced results.jsonl might) is skipped rather than
    crashing `_normalize`/`float()`.
    """
    row = {
        "info": {"task_id": "t1"},
        "completion": [{"role": "assistant", "content": "The answer is 42."}],
        "metrics": {
            "reward": 1.0,
            "n_tool_calls": 2,
            "n_prompts": 1,
            "judge_notes": "looks good",  # non-numeric -- must be skipped, not crash
        },
        "error": None,
        "is_truncated": False,
        "stop_condition": "agent_completed",
    }
    rs = job_results_to_openeval([row], suite_id="s", run_id="run_1")
    assert validate_result_set(rs).valid
    result = rs["results"][0]
    assert result["actual_output"] == "The answer is 42."
    grader_ids = [g["grader_id"] for g in result["grader_results"]]
    assert grader_ids == ["bf_reward"]  # judge_notes silently skipped, not emitted as a grader


# ---------------------------------------------------------------------------
# job_results_file_to_openeval: real file I/O
# ---------------------------------------------------------------------------


def test_job_results_file_to_openeval_reads_a_real_jsonl_file(tmp_path):
    import json

    rows = [
        _row(tmp_path, rollout_name="r1", task_name="t1", rewards={"reward": 1.0}),
        _row(tmp_path, rollout_name="r2", task_name="t2", rewards={"reward": 0.0}),
    ]
    results_path = tmp_path / "results.jsonl"
    with results_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write("\n")  # blank line, must be skipped

    rs = job_results_file_to_openeval(results_path, suite_id="s", run_id="run_1")
    assert validate_result_set(rs).valid
    assert len(rs["results"]) == 2


def test_job_results_file_to_openeval_accepts_str_path(tmp_path):
    import json

    row = _row(tmp_path)
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(json.dumps(row) + "\n")

    rs = job_results_file_to_openeval(str(results_path), suite_id="s", run_id="run_1")
    assert validate_result_set(rs).valid
