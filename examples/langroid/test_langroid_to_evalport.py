"""Tests for examples/langroid.

Everything here runs Langroid's real ``Task`` machinery (only the LLM is mocked), so the
first group of tests is a canary: if a Langroid release changes which status a scenario
ends in, those tests fail instead of the documents quietly changing.

The ``TestProposedVerdictRfc49Branch`` tests check the optional ``Result.verdict`` field
against the validator on the reference branch of EvalPort PR #83 (Discussion #49). They
need a checkout of that branch and are skipped unless ``EVALPORT_RFC49_CHECKOUT`` points
at one, because that field is a proposal, not part of ``main``.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import langroid as lr
import pytest
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.configuration import settings

from openeval.validate import validate_result_set, validate_suite

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import langroid_to_evalport as m  # noqa: E402

SAMPLE_DIR = HERE / "sample_output"
QUESTION = m.QUESTION
# Langroid's global settings as imported, before any scenario touches them.
DEFAULT_SETTINGS = {k: getattr(settings, k) for k in ("max_turns", "quiet", "cache")}

# What each scenario is expected to do: (Langroid status, did Task.run() return None?)
EXPECTED_RUNTIME = {
    "done_correct": ("DONE", False),
    "done_wrong": ("DONE", False),
    "fixed_turns": ("FIXED_TURNS", False),
    "stalled": ("STALLED", True),
    "max_turns": ("MAX_TURNS", True),
    "timeout": ("TIMEOUT", False),
    "killed": ("KILL", False),
    "agent_raises": ("EXCEPTION", False),
}
UNVERIFIED_IDS = {"stalled", "max_turns", "timeout", "killed", "agent_raises"}


@pytest.fixture(scope="module")
def observations():
    return m.run_all()


@pytest.fixture(scope="module")
def obs_by_id(observations):
    return {o.scenario.id: o for o in observations}


@pytest.fixture(scope="module")
def suite():
    return m.build_suite()


@pytest.fixture(scope="module")
def status_aware(observations):
    return m.build_result_set(observations, deterministic=True)


@pytest.fixture(scope="module")
def naive(observations):
    return m.build_result_set(observations, policy="naive", deterministic=True)


@pytest.fixture(scope="module")
def with_verdict(observations):
    return m.build_result_set(observations, proposed_verdict=True, deterministic=True)


def by_id(rs):
    return {r["test_case_id"]: r for r in rs["results"]}


def codes(validation):
    return {e["code"] for e in validation.errors}


# --------------------------------------------------------------------------------------
# The real Langroid runtime
# --------------------------------------------------------------------------------------
class TestRealLangroidRuntime:
    def test_scenarios_cover_expected_list(self):
        assert [s.id for s in m.SCENARIOS] == list(EXPECTED_RUNTIME)
        assert {s.id: s.expected_status for s in m.SCENARIOS} == {
            k: v[0] for k, v in EXPECTED_RUNTIME.items()
        }

    @pytest.mark.parametrize("sid", list(EXPECTED_RUNTIME))
    def test_each_scenario_ends_in_the_expected_status(self, obs_by_id, sid):
        status, returned_none = EXPECTED_RUNTIME[sid]
        o = obs_by_id[sid]
        assert o.status == status
        assert o.run_returned_none is returned_none

    def test_stalled_and_max_turns_return_none_so_the_status_is_otherwise_lost(self):
        """An unmodified lr.Task, not RecordingTask: run() hands back nothing at all."""
        for sid in ("stalled", "max_turns"):
            sc = m.SCENARIOS_BY_ID[sid]
            agent = lr.ChatAgent(
                lr.ChatAgentConfig(
                    name="A",
                    llm=MockLMConfig(default_response=sc.default_response, cache_config=None),
                    system_message="You answer.",
                    use_functions_api=False,
                    use_tools=False,
                )
            )
            task = lr.Task(
                agent,
                interactive=False,
                config=lr.TaskConfig(enable_loggers=False, enable_html_logging=False),
                **sc.task_kwargs,
            )
            with m.langroid_settings(quiet=True, cache=False, **sc.global_settings):
                assert task.run(QUESTION, **sc.run_kwargs) is None

    def test_partial_output_exists_for_timeout_and_kill(self, obs_by_id):
        assert obs_by_id["timeout"].content == "still thinking"
        assert obs_by_id["killed"].content == "partial work"

    def test_exception_is_recorded_not_swallowed(self, obs_by_id):
        assert obs_by_id["agent_raises"].exception == "RuntimeError: provider down"
        assert obs_by_id["agent_raises"].content is None

    def test_settings_are_restored_after_a_run(self, observations):
        # max_turns scenario sets the global limit; it must not leak into later runs.
        assert settings.max_turns == DEFAULT_SETTINGS["max_turns"]
        assert settings.quiet == DEFAULT_SETTINGS["quiet"]
        assert settings.cache == DEFAULT_SETTINGS["cache"]

    def test_judging_policy(self, obs_by_id):
        judged = {sid for sid, o in obs_by_id.items() if m.is_judged(o)}
        assert judged == {"done_correct", "done_wrong", "fixed_turns"}

    def test_unseen_statuses_fail_closed(self):
        sc = m.SCENARIOS_BY_ID["stalled"]
        base = dict(scenario=sc, run_returned_none=False, exception=None, duration_ms=0)
        # Statuses this example could not produce with MockLM, a judged status without
        # content, and an unrecorded status are all "not verified".
        for status, content in [
            ("MAX_COST", "x"),
            ("MAX_TOKENS", "x"),
            ("INF_LOOP", "x"),
            ("USER_QUIT", "x"),
            ("NO_ANSWER", "x"),
            ("ERROR", "x"),
            ("DONE", None),
            (None, "x"),
        ]:
            assert not m.is_judged(m.Observation(status=status, content=content, **base)), status


# --------------------------------------------------------------------------------------
# Status-aware documents
# --------------------------------------------------------------------------------------
class TestStatusAwareDocuments:
    def test_suite_is_valid(self, suite):
        v = validate_suite(suite)
        assert v.valid, v.errors

    def test_resultset_is_valid(self, status_aware):
        v = validate_result_set(status_aware)
        assert v.valid, v.errors

    def test_one_result_per_test_case(self, suite, status_aware):
        assert [t["id"] for t in suite["test_cases"]] == [r["test_case_id"] for r in status_aware["results"]]

    def test_outcomes(self, status_aware):
        rows = by_id(status_aware)
        got = {
            sid: (r["passed"], r["grader_results"][0]["score"]) for sid, r in rows.items()
        }
        assert got == {
            "done_correct": (True, 1),
            "done_wrong": (False, 0),
            "fixed_turns": (True, 1),
            "stalled": (False, None),
            "max_turns": (False, None),
            "timeout": (False, None),
            "killed": (False, None),
            "agent_raises": (False, None),
        }

    def test_rule_6_invariants_hold_on_every_row(self, status_aware):
        for r in status_aware["results"]:
            for g in r["grader_results"]:
                if g["score"] is None:
                    assert g["passed"] is False and g["reason"].startswith("not verified")
            if all(g["score"] is None for g in r["grader_results"]):
                assert r["passed"] is False
                assert r["metadata"]["openeval"] == {"aggregation_status": "unscored"}
            else:
                assert "openeval" not in r["metadata"]

    def test_verified_failure_is_distinguishable_from_unverified(self, status_aware):
        rows = by_id(status_aware)
        assert rows["done_wrong"]["grader_results"][0]["score"] == 0
        assert rows["stalled"]["grader_results"][0]["score"] is None

    def test_errors_only_where_the_run_errored(self, status_aware):
        errors = {sid: r["error"]["type"] for sid, r in by_id(status_aware).items() if "error" in r}
        assert errors == {"timeout": "timeout", "agent_raises": "runner_error"}

    def test_stalled_and_killed_rows_carry_no_error(self, status_aware):
        """No runtime error exists for these states; Rule 6 is the only carrier."""
        rows = by_id(status_aware)
        for sid in ("stalled", "max_turns", "killed"):
            assert "error" not in rows[sid]

    def test_recorded_langroid_status_is_in_metadata(self, status_aware):
        for sid, r in by_id(status_aware).items():
            assert r["metadata"]["langroid"]["status"] == EXPECTED_RUNTIME[sid][0]
            assert r["metadata"]["langroid"]["run_returned_none"] is EXPECTED_RUNTIME[sid][1]

    def test_partial_output_is_kept_but_not_judged(self, status_aware):
        rows = by_id(status_aware)
        assert rows["timeout"]["actual_output"] == "still thinking"
        assert rows["killed"]["actual_output"] == "partial work"
        assert "actual_output" not in rows["stalled"]
        assert "actual_output" not in rows["max_turns"]

    def test_summary(self, status_aware):
        s = status_aware["summary"]
        assert (s["total"], s["passed"], s["failed"], s["skipped"]) == (8, 2, 1, 5)
        assert s["total"] == s["passed"] + s["failed"] + s["skipped"]
        assert s["pass_rate"] == 0.25  # 2 / 8: unverified rows stay in the denominator
        assert s["avg_score"] == pytest.approx(2 / 3, abs=1e-4)  # mean of the 3 scored rows
        assert s["by_grader"][m.GRADER_ID] == {"passed": 2, "failed": 1, "avg_score": s["avg_score"]}

    def test_summary_is_recomputed_from_the_rows(self, status_aware):
        assert m.summarize(status_aware["results"]) == status_aware["summary"]

    def test_checked_in_sample_output_is_current(self, suite, status_aware):
        assert json.loads((SAMPLE_DIR / "suite.json").read_text()) == suite
        assert json.loads((SAMPLE_DIR / "results.json").read_text()) == status_aware

    def test_sample_output_validates(self):
        assert validate_suite(json.loads((SAMPLE_DIR / "suite.json").read_text())).valid
        assert validate_result_set(json.loads((SAMPLE_DIR / "results.json").read_text())).valid


# --------------------------------------------------------------------------------------
# What grading run()'s return value would have produced
# --------------------------------------------------------------------------------------
class TestNaiveComparison:
    def test_naive_document_is_valid_so_no_validator_can_catch_it(self, naive):
        v = validate_result_set(naive)
        assert v.valid, v.errors

    def test_naive_scores_every_non_answer_as_a_wrong_answer(self, naive):
        rows = by_id(naive)
        for sid in UNVERIFIED_IDS:
            assert rows[sid]["grader_results"][0]["score"] == 0
            assert rows[sid]["passed"] is False

    def test_naive_cannot_tell_a_stall_from_a_wrong_answer(self, naive):
        rows = by_id(naive)
        wrong = (rows["done_wrong"]["passed"], rows["done_wrong"]["grader_results"][0]["score"])
        for sid in UNVERIFIED_IDS:
            assert (rows[sid]["passed"], rows[sid]["grader_results"][0]["score"]) == wrong

    def test_the_two_policies_disagree_on_accuracy_not_on_pass_rate(self, status_aware, naive):
        a, n = status_aware["summary"], naive["summary"]
        assert a["pass_rate"] == n["pass_rate"] == 0.25
        assert (a["failed"], a["skipped"]) == (1, 5)
        assert (n["failed"], n["skipped"]) == (6, 0)
        assert n["avg_score"] == 0.25
        assert a["avg_score"] == pytest.approx(0.6667, abs=1e-4)

    def test_unknown_policy_is_rejected(self, observations):
        with pytest.raises(ValueError):
            m.build_result_set(observations, policy="lenient")


# --------------------------------------------------------------------------------------
# Negative controls: the validator must reject a document that breaks Rule 6
# --------------------------------------------------------------------------------------
class TestNegativeControls:
    def test_unverified_grader_marked_passed_is_rejected(self, status_aware):
        bad = copy.deepcopy(status_aware)
        by_id(bad)["stalled"]["grader_results"][0]["passed"] = True
        assert "NULL_SCORE_PASSED" in codes(validate_result_set(bad))

    def test_all_unscored_result_marked_passed_is_rejected(self, status_aware):
        bad = copy.deepcopy(status_aware)
        by_id(bad)["max_turns"]["passed"] = True
        assert "UNSCORED_RESULT_PASSED" in codes(validate_result_set(bad))

    def test_missing_score_key_is_not_the_same_as_null(self, status_aware):
        bad = copy.deepcopy(status_aware)
        del by_id(bad)["killed"]["grader_results"][0]["score"]
        assert "REQUIRED" in codes(validate_result_set(bad))

    def test_out_of_range_score_is_rejected(self, status_aware):
        bad = copy.deepcopy(status_aware)
        by_id(bad)["done_correct"]["grader_results"][0]["score"] = 4
        assert "OUT_OF_RANGE" in codes(validate_result_set(bad))

    def test_unknown_error_type_is_rejected(self, status_aware):
        bad = copy.deepcopy(status_aware)
        by_id(bad)["timeout"]["error"]["type"] = "unverifiable"
        assert validate_result_set(bad).valid is False


# --------------------------------------------------------------------------------------
# Result.verdict (Discussion #49, a proposal; not part of the spec)
# --------------------------------------------------------------------------------------
def rule6_verdict(result):
    """The verdict derivable from Rule 6 alone: all grader scores null => unverified."""
    scores = [g["score"] for g in result["grader_results"]]
    if scores and all(s is None for s in scores):
        return "unverified"
    return "passed" if result["passed"] else "failed"


class TestProposedVerdict:
    def test_flag_adds_verdict_to_every_row_and_nothing_else_changes(self, status_aware, with_verdict):
        stripped = copy.deepcopy(with_verdict)
        for r in stripped["results"]:
            del r["verdict"]
        assert stripped == status_aware

    def test_verdicts(self, with_verdict):
        assert {sid: r["verdict"] for sid, r in by_id(with_verdict).items()} == {
            "done_correct": "passed",
            "done_wrong": "failed",
            "fixed_turns": "passed",
            "stalled": "unverified",
            "max_turns": "unverified",
            "timeout": "unverified",
            "killed": "unverified",
            "agent_raises": "unverified",
        }

    def test_main_branch_validator_accepts_the_document(self, with_verdict):
        # On main, ``verdict`` is an unknown optional field and is ignored.
        assert validate_result_set(with_verdict).valid

    def test_in_this_data_rule_6_already_gives_the_same_verdict(self, with_verdict):
        """The honest counter-point: nothing in these eight rows needs verdict.

        Every unverified row has all grader scores null, so a consumer can derive
        "unverified" from Rule 6 alone. The field would only add something where a grader
        ran and scored a run whose outcome was still not established.
        """
        for r in with_verdict["results"]:
            assert rule6_verdict(r) == r["verdict"]

    def test_status_aware_document_without_verdict_still_separates_the_states(self, status_aware):
        derived = {sid: rule6_verdict(r) for sid, r in by_id(status_aware).items()}
        assert sorted(derived.values()).count("unverified") == 5
        assert derived["done_wrong"] == "failed"


RFC49 = os.environ.get("EVALPORT_RFC49_CHECKOUT")


@pytest.mark.skipif(
    not RFC49 or not (pathlib.Path(RFC49) / "sdk" / "python" / "openeval").is_dir(),
    reason="set EVALPORT_RFC49_CHECKOUT to a checkout of EvalPort PR #83 (Discussion #49)",
)
class TestProposedVerdictRfc49Branch:
    """Run the PR #83 validator in a fresh interpreter (its ``openeval`` replaces main's)."""

    SCRIPT = textwrap.dedent(
        """
        import json, sys
        import openeval, pathlib
        from openeval.validate import validate_result_set
        assert str(pathlib.Path(sys.argv[1]).resolve()) in str(pathlib.Path(openeval.__file__).resolve()), openeval.__file__
        doc = json.load(sys.stdin)
        v = validate_result_set(doc)
        print(json.dumps({"valid": v.valid, "codes": sorted({e["code"] for e in v.errors})}))
        """
    )

    def validate(self, doc):
        sdk = str(pathlib.Path(RFC49) / "sdk" / "python")
        env = {**os.environ, "PYTHONPATH": sdk}
        out = subprocess.run(
            [sys.executable, "-c", self.SCRIPT, sdk],
            input=json.dumps(doc),
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_branch_validator_accepts_the_proposed_verdict_document(self, with_verdict):
        assert self.validate(with_verdict) == {"valid": True, "codes": []}

    def test_timeout_row_has_error_plus_unverified_which_the_proposal_allows(self, with_verdict):
        row = by_id(with_verdict)["timeout"]
        assert row["error"]["type"] == "timeout" and row["verdict"] == "unverified"

    def test_unverified_but_passed_true_is_rejected(self, with_verdict):
        bad = copy.deepcopy(with_verdict)
        by_id(bad)["stalled"]["passed"] = True
        result = self.validate(bad)
        assert result["valid"] is False and "VERDICT_PASSED_MISMATCH" in result["codes"]

    def test_failed_verdict_on_an_error_row_is_rejected(self, with_verdict):
        bad = copy.deepcopy(with_verdict)
        by_id(bad)["timeout"]["verdict"] = "failed"
        result = self.validate(bad)
        assert result["valid"] is False and "VERDICT_ERROR_CONFLICT" in result["codes"]

    def test_invalid_verdict_value_is_rejected(self, with_verdict):
        bad = copy.deepcopy(with_verdict)
        by_id(bad)["stalled"]["verdict"] = "needs-input"
        result = self.validate(bad)
        assert result["valid"] is False and "INVALID_VALUE" in result["codes"]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
class TestCli:
    def test_cli_writes_valid_documents(self, tmp_path, capsys):
        assert m.main(["--out-dir", str(tmp_path), "--deterministic"]) == 0
        out = capsys.readouterr().out
        assert "STALLED" in out and "pass_rate=0.25" in out
        assert validate_suite(json.loads((tmp_path / "suite.json").read_text())).valid
        assert validate_result_set(json.loads((tmp_path / "results.json").read_text())).valid

    def test_cli_proposed_verdict_flag(self, tmp_path):
        assert m.main(["--out-dir", str(tmp_path), "--deterministic", "--proposed-verdict"]) == 0
        rs = json.loads((tmp_path / "results.json").read_text())
        assert all("verdict" in r for r in rs["results"])

    def test_cli_non_deterministic_mode_has_timestamps_and_valid_output(self, tmp_path):
        assert m.main(["--out-dir", str(tmp_path)]) == 0
        rs = json.loads((tmp_path / "results.json").read_text())
        assert "completed_at" in rs and all("duration_ms" in r for r in rs["results"])
        assert validate_result_set(rs).valid
