import json
from pathlib import Path

import pytest
from openeval.validate import validate_result_set

from nasde_openeval_adapter import from_openeval, to_openeval, trial_to_result


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _make_trial(
    tmp_path: Path,
    job: str,
    trial: str,
    *,
    task_name: str = "refactor_module",
    agent_name: str = "claude-agent",
    model_name: str = "claude-sonnet-5",
    source: str = "harbor-bench",
    harbor_reward: float = 1.0,
    started_at: str = "2026-09-01T00:00:00+00:00",
    finished_at: str = "2026-09-01T00:05:00+00:00",
    duration_sec: float = 300.0,
    dominant_score: float = 0.82,
    dominant_std: float = 0.05,
    eval_n: int = 3,
    include_assessment: bool = True,
    include_second_cluster: bool = False,
    exception_info=None,
    reasoning_effort: str = "high",
) -> Path:
    trial_dir = tmp_path / f"{job}__{trial}"
    trial_dir.mkdir(parents=True)

    metrics = {
        "trial_name": trial,
        "task_name": task_name,
        "agent_name": agent_name,
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
        "source": source,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "harbor_reward": harbor_reward,
        "score": dominant_score if include_assessment else None,
        "score_eval_std": dominant_std if include_assessment else None,
        "score_eval_n": eval_n if include_assessment else None,
        "single_eval": eval_n == 1 if include_assessment else None,
        "token_usage": {"input_tokens": 12000, "output_tokens": 3400},
        "cost_usd": 0.4521,
        "pricing_as_of": "2026-08-15",
        "exception_info": exception_info,
    }
    _write_json(trial_dir / "metrics.json", metrics)

    if include_assessment:
        dimensions = [
            {"name": "correctness", "max_score": 10, "mean": 8.2, "std": 0.4, "min": 8.0, "max": 9.0},
            {"name": "code_quality", "max_score": 5, "mean": 4.1, "std": 0.1, "min": 4.0, "max": 4.0},
        ]
        groups = [
            {
                "evaluator_model": "gpt-4o",
                "dimensions_fingerprint": "abc123def456",
                "n": eval_n,
                "dominant": True,
                "dimensions": dimensions,
                "normalized_score_mean": dominant_score,
                "normalized_score_std": dominant_std,
                "normalized_score_min": 0.78,
                "normalized_score_max": 0.87,
                "total_score_mean": 12.3,
                "harbor_reward": harbor_reward,
                "duration_sec_mean": duration_sec,
            }
        ]
        if include_second_cluster:
            groups.append(
                {
                    "evaluator_model": "claude-opus",
                    "dimensions_fingerprint": "zzz999",
                    "n": 1,
                    "dominant": False,
                    "dimensions": [
                        {"name": "correctness", "max_score": 10, "mean": 9.0, "std": 0.0, "min": 9.0, "max": 9.0},
                    ],
                    "normalized_score_mean": 0.9,
                    "normalized_score_std": 0.0,
                    "normalized_score_min": 0.9,
                    "normalized_score_max": 0.9,
                    "total_score_mean": 9.0,
                    "harbor_reward": harbor_reward,
                    "duration_sec_mean": duration_sec,
                }
            )
        summary = {
            "task_name": task_name,
            "trial_name": trial,
            "agent_name": agent_name,
            "groups": groups,
            "model_name": model_name,
            "reasoning_effort": reasoning_effort,
            "token_usage": metrics["token_usage"],
            "cost_usd": metrics["cost_usd"],
            "pricing_as_of": metrics["pricing_as_of"],
        }
        _write_json(trial_dir / "assessment_summary.json", summary)

        for i in range(1, eval_n + 1):
            eval_result = {
                "task_name": task_name,
                "trial_name": trial,
                "agent_name": agent_name,
                "evaluator_model": "gpt-4o",
                "timestamp": f"2026-09-01T00:0{i}:00+00:00",
                "dimensions": dimensions,
                "total_score": 12,
                "normalized_score": dominant_score,
                "summary": f"Repetition {i}: the agent correctly refactored the module.",
                "harbor_reward": harbor_reward,
                "duration_sec": duration_sec,
                "dimensions_fingerprint": "abc123def456",
            }
            _write_json(trial_dir / f"assessment_eval_{i}.json", eval_result)

    return trial_dir


class TestTrialToResult:
    def test_basic_mapping_produces_valid_grader_results(self, tmp_path):
        trial_dir = _make_trial(tmp_path, "job1", "trial1")
        result = trial_to_result(trial_dir)

        assert result["test_case_id"] == "refactor_module"
        grader_ids = {g["grader_id"] for g in result["grader_results"]}
        assert grader_ids == {"dim_correctness", "dim_code_quality", "harbor_reward"}

        correctness = next(g for g in result["grader_results"] if g["grader_id"] == "dim_correctness")
        assert correctness["score"] == pytest.approx(8.2 / 10)
        assert correctness["metadata"]["max_score"] == 10
        assert correctness["metadata"]["mean"] == 8.2
        assert correctness["metadata"]["evaluator_model"] == "gpt-4o"
        assert correctness["metadata"]["eval_n"] == 3

    def test_passed_uses_dominant_normalized_score_not_harbor_reward(self, tmp_path):
        # harbor_reward is 0.0 (verifier failed) but the judge score is well
        # above threshold -- passed must follow the judge score, exactly the
        # distinction the maintainer raised in issue #79.
        trial_dir = _make_trial(tmp_path, "job1", "trial1", harbor_reward=0.0, dominant_score=0.9)
        result = trial_to_result(trial_dir, pass_threshold=0.5)
        assert result["passed"] is True
        assert result["metadata"]["nasde"]["passed_basis"] == "dominant_cluster_normalized_score"

        harbor_gr = next(g for g in result["grader_results"] if g["grader_id"] == "harbor_reward")
        assert harbor_gr["score"] == 0.0
        assert harbor_gr["passed"] is False

    def test_pass_threshold_is_configurable(self, tmp_path):
        trial_dir = _make_trial(tmp_path, "job1", "trial1", dominant_score=0.6)
        assert trial_to_result(trial_dir, pass_threshold=0.5)["passed"] is True
        assert trial_to_result(trial_dir, pass_threshold=0.7)["passed"] is False

    def test_no_assessment_falls_back_to_harbor_reward_and_says_so(self, tmp_path):
        trial_dir = _make_trial(tmp_path, "job1", "trial1", include_assessment=False, harbor_reward=1.0)
        result = trial_to_result(trial_dir, reward_fallback_threshold=1.0)
        assert result["passed"] is True
        assert result["metadata"]["nasde"]["passed_basis"] == "harbor_reward_fallback_no_judge_assessment"
        # No dimension grader results without an assessment, but harbor_reward
        # is still recorded.
        assert [g["grader_id"] for g in result["grader_results"]] == ["harbor_reward"]

    def test_exception_info_produces_error_and_no_signal(self, tmp_path):
        trial_dir = _make_trial(
            tmp_path,
            "job1",
            "trial1",
            include_assessment=False,
            harbor_reward=None,
            exception_info={"type": "TimeoutError", "message": "agent exceeded time budget"},
        )
        result = trial_to_result(trial_dir)
        assert result["passed"] is False
        assert result["metadata"]["nasde"]["passed_basis"] == "no_signal_available"
        assert result["error"] == {
            "type": "agent_exception",
            "detail": {"type": "TimeoutError", "message": "agent exceeded time budget"},
        }
        assert result["grader_results"] == []

    def test_non_dominant_clusters_preserved_and_never_merged_into_grader_results(self, tmp_path):
        trial_dir = _make_trial(tmp_path, "job1", "trial1", include_second_cluster=True)
        result = trial_to_result(trial_dir)

        # grader_results reflects ONLY the dominant cluster (gpt-4o).
        for gr in result["grader_results"]:
            if gr["grader_id"].startswith("dim_"):
                assert gr["metadata"]["evaluator_model"] == "gpt-4o"

        non_dominant = result["metadata"]["nasde"]["non_dominant_clusters"]
        assert len(non_dominant) == 1
        assert non_dominant[0]["evaluator_model"] == "claude-opus"
        assert non_dominant[0]["normalized_score_mean"] == 0.9

    def test_economics_fields_preserved_verbatim(self, tmp_path):
        trial_dir = _make_trial(tmp_path, "job1", "trial1")
        result = trial_to_result(trial_dir)
        economics = result["metadata"]["nasde"]["economics"]
        assert economics == {
            "token_usage": {"input_tokens": 12000, "output_tokens": 3400},
            "cost_usd": 0.4521,
            "pricing_as_of": "2026-08-15",
            "model_name": "claude-sonnet-5",
            "reasoning_effort": "high",
        }

    def test_actual_output_pulled_from_most_recent_matching_assessment_eval(self, tmp_path):
        trial_dir = _make_trial(tmp_path, "job1", "trial1", eval_n=3)
        result = trial_to_result(trial_dir)
        assert result["actual_output"] == "Repetition 3: the agent correctly refactored the module."

    def test_duration_ms_conversion(self, tmp_path):
        trial_dir = _make_trial(tmp_path, "job1", "trial1", duration_sec=12.5)
        result = trial_to_result(trial_dir)
        assert result["duration_ms"] == 12500

    def test_missing_metrics_json_raises(self, tmp_path):
        empty_dir = tmp_path / "job1__trial1"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            trial_to_result(empty_dir)

    def test_dimension_score_out_of_range_is_clipped_defensively(self, tmp_path):
        # Defensive: nothing in nasde's schema guarantees scores stay in
        # [0, 1] forever (e.g. a future max_score=0 edge case elsewhere);
        # harbor_reward is the field this actually matters for today.
        trial_dir = _make_trial(tmp_path, "job1", "trial1", harbor_reward=1.4)
        result = trial_to_result(trial_dir)
        harbor_gr = next(g for g in result["grader_results"] if g["grader_id"] == "harbor_reward")
        assert harbor_gr["score"] == 1.0
        assert harbor_gr["metadata"]["raw_harbor_reward"] == 1.4


class TestToOpenEval:
    def test_builds_valid_result_set(self, tmp_path):
        t1 = _make_trial(tmp_path, "jobA", "trial1", task_name="task_one")
        t2 = _make_trial(tmp_path, "jobA", "trial2", task_name="task_two")

        result_set = to_openeval([t1, t2])
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors

        assert result_set["suite_id"] == "harbor-bench"
        assert result_set["run_id"] == "jobA"
        assert len(result_set["results"]) == 2
        assert result_set["started_at"] == "2026-09-01T00:00:00+00:00"
        assert result_set["completed_at"] == "2026-09-01T00:05:00+00:00"

    def test_rejects_mixed_sources(self, tmp_path):
        t1 = _make_trial(tmp_path, "jobA", "trial1", source="bench-one")
        t2 = _make_trial(tmp_path, "jobA", "trial2", source="bench-two")
        with pytest.raises(ValueError, match="multiple nasde 'source'"):
            to_openeval([t1, t2])

    def test_rejects_mixed_jobs_without_explicit_run_id(self, tmp_path):
        t1 = _make_trial(tmp_path, "jobA", "trial1")
        t2 = _make_trial(tmp_path, "jobB", "trial1")
        with pytest.raises(ValueError, match="multiple job names"):
            to_openeval([t1, t2])

    def test_explicit_run_id_overrides_inference(self, tmp_path):
        t1 = _make_trial(tmp_path, "jobA", "trial1")
        result_set = to_openeval([t1], run_id="custom-run")
        assert result_set["run_id"] == "custom-run"

    def test_empty_trial_dirs_raises(self):
        with pytest.raises(ValueError):
            to_openeval([])

    def test_result_set_with_crashed_and_healthy_trials_validates(self, tmp_path):
        healthy = _make_trial(tmp_path, "jobA", "trial1")
        crashed = _make_trial(
            tmp_path,
            "jobA",
            "trial2",
            include_assessment=False,
            harbor_reward=None,
            exception_info={"type": "RuntimeError", "message": "boom"},
        )
        result_set = to_openeval([healthy, crashed])
        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        crashed_result = next(r for r in result_set["results"] if r["test_case_id"] == "refactor_module" and r.get("error"))
        assert crashed_result["passed"] is False


class TestFromOpenEval:
    def test_round_trip_recovers_flattened_fields(self, tmp_path):
        t1 = _make_trial(tmp_path, "jobA", "trial1", model_name="claude-sonnet-5")
        result_set = to_openeval([t1])
        flattened = from_openeval(result_set)

        assert len(flattened) == 1
        row = flattened[0]
        assert row["task_name"] == "refactor_module"
        assert row["model_name"] == "claude-sonnet-5"
        assert row["cost_usd"] == pytest.approx(0.4521)
        assert row["harbor_reward"] == 1.0
        assert row["score"] == pytest.approx(0.82)
        assert row["duration_sec"] == pytest.approx(300.0)

    def test_lossy_round_trip_has_no_per_repetition_detail(self, tmp_path):
        t1 = _make_trial(tmp_path, "jobA", "trial1", eval_n=3)
        result_set = to_openeval([t1])
        flattened = from_openeval(result_set)
        # By design: no assessment_eval_N-level detail survives the round trip.
        assert "dimensions" not in flattened[0]
        assert "assessment_eval" not in json.dumps(flattened[0])
