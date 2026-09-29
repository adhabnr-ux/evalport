"""Tests for agent_skills_eval_openeval_adapter.

Runs against the real `openeval.validate` validators (no mocks) and against
fixture files shaped exactly like the real `grading.json` / `timing.json` /
`outputs/response.txt` layout documented in agent-skills-eval's
`docs/artifact-contract.md` (no reinvented stand-ins).
"""
import json

from openeval.validate import validate_result_set

from agent_skills_eval_openeval_adapter import (
    from_openeval,
    iter_eval_dirs,
    load_run_artifacts,
    to_openeval,
)


def _grading(*rows, pass_rate=None):
    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)
    summary = {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "pass_rate": pass_rate if pass_rate is not None else (passed / total if total else 1),
    }
    return {"assertion_results": list(rows), "summary": summary}


def _row(text, passed, evidence="quoted evidence"):
    return {"text": text, "passed": passed, "evidence": evidence}


# ---------------------------------------------------------------------------
# to_openeval()
# ---------------------------------------------------------------------------


class TestToOpeneval:
    def test_basic_conversion_validates(self):
        grading = _grading(_row("output mentions Paris", True))
        result_set = to_openeval(
            [{"test_case_id": "tc1", "grading": grading}],
            suite_id="agent-skills-eval-run",
            run_id="run1",
            mode="with_skill",
        )
        v = validate_result_set(result_set)
        assert v.valid, v.errors
        assert result_set["results"][0]["test_case_id"] == "tc1"
        assert result_set["results"][0]["passed"] is True

    def test_classification_with_eval_def_splits_rubric_from_tool_rows(self):
        # Mirrors gradeOutputs() in src/grade.ts: assertion_results is always
        # [...rubricResults, ...toolResults].
        grading = _grading(
            _row("mentions the capital", True),
            _row('tool "search" was called', True, evidence="search called 1 time(s)"),
            _row('search.query contains "capital"', False, evidence="expected substring..."),
        )
        eval_def = {
            "assertions": ["mentions the capital"],
            "tool_assertions": [
                {"type": "tool-called", "name": "search"},
                {"type": "tool-arg-contains", "name": "search", "path": "query", "value": "capital"},
            ],
        }
        result_set = to_openeval(
            [{"test_case_id": "tc1", "grading": grading, "eval_def": eval_def}],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
        )
        grs = result_set["results"][0]["grader_results"]
        assert [g["type"] for g in grs] == ["llm_judge", "custom", "custom"]
        assert all(g["metadata"]["kind_inferred"] for g in grs)
        assert grs[0]["metadata"]["kind"] == "llm_judge"
        assert grs[1]["metadata"]["kind"] == "tool_assertion"
        # Original text preserved verbatim regardless of classification.
        assert grs[2]["metadata"]["text"] == 'search.query contains "capital"'
        assert grs[2]["passed"] is False
        assert grs[2]["score"] == 0.0

    def test_no_eval_def_never_labels_llm_judge(self):
        # This is the specific behavior the upstream maintainer asked for:
        # https://github.com/darkrishabh/agent-skills-eval/issues/34#issuecomment-5888398803
        grading = _grading(_row("some assertion", True), _row("some tool check", False))
        result_set = to_openeval(
            [{"test_case_id": "tc1", "grading": grading}],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
        )
        grs = result_set["results"][0]["grader_results"]
        assert all(g["type"] == "custom" for g in grs)
        assert all(g["metadata"]["kind_inferred"] is False for g in grs)

    def test_mismatched_eval_def_falls_back_conservatively(self):
        # eval_def claims 1 rubric + 1 tool assertion, but grading.json has 3
        # rows -- e.g. a stale eval_def. Must not mis-split.
        grading = _grading(_row("a", True), _row("b", True), _row("c", False))
        eval_def = {"assertions": ["a"], "tool_assertions": [{"type": "tool-called", "name": "x"}]}
        result_set = to_openeval(
            [{"test_case_id": "tc1", "grading": grading, "eval_def": eval_def}],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
        )
        grs = result_set["results"][0]["grader_results"]
        assert all(g["type"] == "custom" for g in grs)
        assert all(g["metadata"]["kind_inferred"] is False for g in grs)

    def test_error_output_convention_sets_result_error(self):
        grading = _grading(_row("x", False, evidence="provider failed"))
        result_set = to_openeval(
            [{
                "test_case_id": "tc1",
                "grading": grading,
                "actual_output": "ERROR: rate limited by provider",
            }],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
        )
        result = result_set["results"][0]
        assert result["actual_output"] == "ERROR: rate limited by provider"
        assert result["error"] == {"type": "provider_error", "message": "rate limited by provider"}

    def test_zero_assertions_counts_as_passed(self):
        # Matches grade.ts summarize(): total === 0 -> pass_rate 1.
        grading = _grading()
        result_set = to_openeval(
            [{"test_case_id": "tc1", "grading": grading}],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
        )
        assert result_set["results"][0]["passed"] is True
        assert result_set["results"][0]["grader_results"] == []

    def test_mode_declared_once_on_resultset_and_each_result(self):
        grading = _grading(_row("x", True))
        result_set = to_openeval(
            [{"test_case_id": "tc1", "grading": grading}],
            suite_id="s1",
            run_id="run1",
            mode="without_skill",
        )
        assert result_set["metadata"]["agent_skills_eval"]["mode"] == "without_skill"
        assert result_set["results"][0]["metadata"]["agent_skills_eval"]["mode"] == "without_skill"

    def test_duration_ms_and_extra_metadata_preserved(self):
        grading = _grading(_row("x", True))
        result_set = to_openeval(
            [{
                "test_case_id": "tc1",
                "grading": grading,
                "duration_ms": 1234,
                "metadata": {"skill": "basic-skill", "eval_slug": "case-1"},
            }],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
        )
        result = result_set["results"][0]
        assert result["duration_ms"] == 1234
        assert result["metadata"]["agent_skills_eval"]["skill"] == "basic-skill"
        assert result["metadata"]["agent_skills_eval"]["eval_slug"] == "case-1"

    def test_resultset_summary_computed_from_results(self):
        grading_pass = _grading(_row("x", True))
        grading_fail = _grading(_row("y", False))
        result_set = to_openeval(
            [
                {"test_case_id": "tc1", "grading": grading_pass},
                {"test_case_id": "tc2", "grading": grading_fail},
            ],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
        )
        assert result_set["summary"] == {"total": 2, "passed": 1, "failed": 1, "pass_rate": 0.5}
        v = validate_result_set(result_set)
        assert v.valid, v.errors

    def test_empty_runs_raises_rather_than_producing_an_invalid_resultset(self):
        # EvalPort's spec requires $.results to be non-empty
        # (openeval.validate.validate_result_set rejects `[]`), so this is
        # raised proactively instead of handed back as an unvalidatable dict.
        import pytest

        with pytest.raises(ValueError, match="zero runs"):
            to_openeval([], suite_id="s1", run_id="run1", mode="with_skill")

    def test_runner_version_included_when_given(self):
        result_set = to_openeval(
            [{"test_case_id": "tc1", "grading": _grading(_row("x", True))}],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
            runner_version="0.1.1",
        )
        assert result_set["runner"] == {"name": "agent-skills-eval", "version": "0.1.1"}


# ---------------------------------------------------------------------------
# from_openeval()
# ---------------------------------------------------------------------------


class TestFromOpeneval:
    def test_round_trip_preserves_text_passed_evidence(self):
        grading = _grading(_row("mentions Paris", True, evidence="says 'Paris'"))
        result_set = to_openeval(
            [{"test_case_id": "tc1", "grading": grading}],
            suite_id="s1",
            run_id="run1",
            mode="with_skill",
        )
        back = from_openeval(result_set)
        assert back["tc1"]["mode"] == "with_skill"
        row = back["tc1"]["grading"]["assertion_results"][0]
        assert row == {"text": "mentions Paris", "passed": True, "evidence": "says 'Paris'"}
        assert back["tc1"]["grading"]["summary"] == grading["summary"]

    def test_recomputes_summary_when_absent(self):
        # A ResultSet not produced by this adapter's to_openeval() has no
        # agent_skills_eval.summary metadata to round-trip.
        result_set = {
            "version": "1.0.0",
            "suite_id": "s1",
            "run_id": "r1",
            "started_at": "2026-01-01T00:00:00Z",
            "results": [
                {
                    "test_case_id": "tc1",
                    "passed": True,
                    "grader_results": [
                        {"grader_id": "g1", "type": "custom", "score": 1.0, "passed": True,
                         "reason": "ok", "metadata": {"text": "check x"}},
                    ],
                }
            ],
        }
        back = from_openeval(result_set)
        assert back["tc1"]["grading"]["summary"] == {"passed": 1, "failed": 0, "total": 1, "pass_rate": 1}


# ---------------------------------------------------------------------------
# load_run_artifacts() / iter_eval_dirs() -- real files on disk, matching
# docs/artifact-contract.md's documented layout exactly.
# ---------------------------------------------------------------------------


class TestLoadRunArtifacts:
    def _write_run(self, tmp_path, rel, grading, response=None, timing=None):
        run_dir = tmp_path / rel
        run_dir.mkdir(parents=True)
        (run_dir / "grading.json").write_text(json.dumps(grading))
        if response is not None:
            (run_dir / "outputs").mkdir()
            (run_dir / "outputs" / "response.txt").write_text(response)
        if timing is not None:
            (run_dir / "timing.json").write_text(json.dumps(timing))
        return run_dir

    def test_reads_grading_response_and_timing(self, tmp_path):
        grading = _grading(_row("x", True))
        run_dir = self._write_run(
            tmp_path,
            "iteration-1/basic-skill-case-1/with_skill",
            grading,
            response="The capital of France is Paris.",
            timing={"duration_ms": 842, "total_tokens": 173},
        )
        run = load_run_artifacts(run_dir)
        assert run["test_case_id"] == "basic-skill-case-1"
        assert run["grading"] == grading
        assert run["actual_output"] == "The capital of France is Paris."
        assert run["duration_ms"] == 842
        assert run["metadata"]["total_tokens"] == 173
        assert run["metadata"]["source_path"] == str(run_dir)

    def test_missing_grading_json_raises(self, tmp_path):
        run_dir = tmp_path / "empty" / "with_skill"
        run_dir.mkdir(parents=True)
        import pytest

        with pytest.raises(FileNotFoundError, match="grading.json"):
            load_run_artifacts(run_dir)

    def test_explicit_test_case_id_overrides_default(self, tmp_path):
        grading = _grading(_row("x", True))
        run_dir = self._write_run(tmp_path, "iteration-1/case-1/with_skill", grading)
        run = load_run_artifacts(run_dir, test_case_id="custom-id")
        assert run["test_case_id"] == "custom-id"

    def test_missing_optional_files_are_skipped_not_errored(self, tmp_path):
        grading = _grading(_row("x", True))
        run_dir = self._write_run(tmp_path, "iteration-1/case-1/without_skill", grading)
        run = load_run_artifacts(run_dir)
        assert "actual_output" not in run
        assert "duration_ms" not in run

    def test_end_to_end_disk_to_validated_resultset(self, tmp_path):
        grading = _grading(_row("mentions the capital", True), _row("tool called", False))
        run_dir = self._write_run(
            tmp_path,
            "iteration-1/geo-skill/case-1/with_skill",
            grading,
            response="Paris.",
            timing={"duration_ms": 500},
        )
        run = load_run_artifacts(run_dir)
        result_set = to_openeval([run], suite_id="s1", run_id="run1", mode="with_skill")
        v = validate_result_set(result_set)
        assert v.valid, v.errors


class TestIterEvalDirs:
    def test_discovers_nested_and_flat_layouts(self, tmp_path):
        grading = _grading(_row("x", True))
        paths = [
            "iteration-1/case-a/with_skill",
            "iteration-1/skill-a/case-b/without_skill",
            "flat-workspace/skill-b/case-c/with_skill",
        ]
        for rel in paths:
            d = tmp_path / rel
            d.mkdir(parents=True)
            (d / "grading.json").write_text(json.dumps(grading))

        found = {str(p.relative_to(tmp_path)) for p in iter_eval_dirs(tmp_path)}
        assert found == set(paths)

    def test_ignores_directories_without_grading_json(self, tmp_path):
        (tmp_path / "not-a-run").mkdir()
        (tmp_path / "not-a-run" / "notes.txt").write_text("hello")
        assert list(iter_eval_dirs(tmp_path)) == []
