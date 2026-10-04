"""Tests for examples/errored-graders.

Three layers:

1. ``TestRealFrameworkBehavior`` runs the real DeepEval and Inspect AI code and asserts what
   each one did when a grader raised. These are canaries: if a framework release changes the
   behavior, they fail first, and the README's claims need re-checking. Versions are pinned
   in requirements.txt because the values below were observed with exactly those versions.
2. The conversion tests check the EvalPort documents: validity, Rule 6 invariants, the numbers
   the README quotes, and a few negative controls on the validator.
3. ``TestProposedVerdictRfc49Branch`` needs a checkout of PR #83 (Discussion #49, a draft that
   is NOT part of the spec) and is skipped unless ``EVALPORT_RFC49_CHECKOUT`` points at one.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import errored_graders_to_evalport as m  # noqa: E402
import observe  # noqa: E402
from openeval.validate import validate_result_set, validate_suite  # noqa: E402


@pytest.fixture(scope="module")
def deepeval_obs():
    return observe.observe_deepeval()


@pytest.fixture(scope="module")
def inspect_obs():
    return observe.observe_inspect()


@pytest.fixture(scope="module", params=["deepeval", "inspect_ai"])
def obs(request, deepeval_obs, inspect_obs):
    return {"deepeval": deepeval_obs, "inspect_ai": inspect_obs}[request.param]


def scores(row):
    return [g.score for g in row.graders]


def by_id(doc):
    return {r["test_case_id"]: r for r in doc["results"]}


# --------------------------------------------------------------------------------------
# Real framework behavior
# --------------------------------------------------------------------------------------
class TestRealFrameworkBehavior:
    def test_versions_are_the_pinned_ones(self, deepeval_obs, inspect_obs):
        assert deepeval_obs.version == "4.2.8"
        assert inspect_obs.version == "0.3.276"

    def test_deepeval_reports_each_errored_metric_with_no_score_and_its_error_text(self, deepeval_obs):
        rows = {r.case_id: r for r in deepeval_obs.rows}
        assert scores(rows["q1"]) == [1.0, 1.0]
        assert scores(rows["q2"]) == [1.0, None]
        assert scores(rows["q3"]) == [0.0, None]
        assert scores(rows["q4"]) == [None, None]
        flaky = rows["q2"].graders[1]
        assert flaky.success is False                      # DeepEval writes success=False for an errored metric
        assert "flaky grader unreachable" in flaky.error

    def test_deepeval_row_success_is_fail_closed(self, deepeval_obs):
        got = {r.case_id: r.native_row_success for r in deepeval_obs.rows}
        assert got == {"q1": True, "q2": False, "q3": False, "q4": False}

    def test_deepeval_native_row_success_equals_this_examples_fail_closed_policy(self, deepeval_obs):
        for row in deepeval_obs.rows:
            assert row.native_row_success == m.result_passed(row, "fail-closed")

    def test_inspect_keeps_the_score_of_the_scorer_that_ran_and_records_the_error_on_the_sample(self, inspect_obs):
        rows = {r.case_id: r for r in inspect_obs.rows}
        assert scores(rows["q1"]) == [1.0, 1.0] and rows["q1"].native_row_error is None
        assert scores(rows["q2"]) == [1.0, None]
        assert scores(rows["q3"]) == [0.0, None]
        assert scores(rows["q4"]) == [None, None]
        for cid in ("q2", "q3", "q4"):
            assert "grader unreachable" in rows[cid].native_row_error

    def test_inspect_has_no_row_level_pass_flag(self, inspect_obs):
        assert all(r.native_row_success is None for r in inspect_obs.rows)

    def test_inspect_run_still_says_success_and_excludes_errored_samples_per_scorer(self, inspect_obs):
        rl = inspect_obs.run_level
        assert rl["status"] == "success"
        assert (rl["total_samples"], rl["completed_samples"]) == (4, 1)
        # each scorer's accuracy is over the samples that scorer scored, not over all four
        assert rl["scores"]["exact"]["accuracy"] == pytest.approx(2 / 3)
        assert rl["scores"]["flaky"]["accuracy"] == pytest.approx(1.0)


class TestFrameworkDefaults:
    """What happens without the opt-in flags this example sets (observed, pinned versions)."""

    def test_deepeval_default_lets_the_scorer_exception_escape_evaluate(self):
        with pytest.raises(RuntimeError, match="flaky grader unreachable"):
            observe.observe_deepeval(ignore_errors=False)

    def test_inspect_default_fail_on_error_marks_the_whole_run_as_errored(self):
        obs = observe.observe_inspect(fail_on_error=True)
        assert obs.run_level["status"] == "error"
        assert obs.run_level["scores"] == {}          # no aggregate scores at all


class TestInspectScorerOrder:
    """Observed with inspect-ai 0.3.276: once a scorer raises, scorers listed AFTER it have no
    score in the log, so which graders have a score depends on the order they were listed in.
    The log cannot say whether the later scorers were never attempted or were attempted and
    their scores discarded; only the absence is observed."""

    def test_scorers_listed_after_the_one_that_raised_have_no_score(self):
        rows = {r.case_id: r for r in observe.observe_inspect(["flaky", "exact"]).rows}
        # rows list graders in GRADERS order: (exact, flaky)
        assert scores(rows["q1"]) == [1.0, 1.0]
        assert scores(rows["q2"]) == [None, None]     # exact would have passed, but flaky was listed first
        assert scores(rows["q3"]) == [None, None]     # exact would have failed: that failure is not in the log

    def test_the_same_data_with_the_other_order_keeps_the_exact_score(self, inspect_obs):
        rows = {r.case_id: r for r in inspect_obs.rows}
        assert scores(rows["q2"]) == [1.0, None]
        assert scores(rows["q3"]) == [0.0, None]


# --------------------------------------------------------------------------------------
# Classifications
# --------------------------------------------------------------------------------------
class TestClassification:
    def test_true_situation(self, obs):
        got = {r.case_id: m.true_situation(r) for r in obs.rows}
        assert got == {"q1": "passed", "q2": "unverified", "q3": "failed", "q4": "unverified"}

    def test_spec_default_passed(self, obs):
        got = {r.case_id: m.result_passed(r, "spec-default") for r in obs.rows}
        assert got == {"q1": True, "q2": True, "q3": False, "q4": False}

    def test_fail_closed_passed(self, obs):
        got = {r.case_id: m.result_passed(r, "fail-closed") for r in obs.rows}
        assert got == {"q1": True, "q2": False, "q3": False, "q4": False}

    def test_rule6_reading_cannot_say_unverified_for_a_mixed_row(self, obs):
        """The central observation. q2 is "unverified" in fact, but under Rule 6 it reads as
        passed (spec-default) or as verified failing (fail-closed); neither is true."""
        spec_doc = m.build_documents(obs, policy="spec-default")[1]
        closed_doc = m.build_documents(obs, policy="fail-closed")[1]
        assert m.rule6_class(by_id(spec_doc)["q2"]) == "passed"
        assert m.rule6_class(by_id(closed_doc)["q2"]) == "verified failing"
        assert m.true_situation(next(r for r in obs.rows if r.case_id == "q2")) == "unverified"

    def test_rule6_agrees_with_the_true_situation_everywhere_else(self, obs):
        for policy in m.POLICIES:
            doc = m.build_documents(obs, policy=policy)[1]
            r6 = {cid: m.rule6_class(r) for cid, r in by_id(doc).items()}
            assert r6["q1"] == "passed"
            assert r6["q3"] == "verified failing"
            assert r6["q4"] == "not verified"

    def test_verdict_for_matches_true_situation(self, obs):
        for row in obs.rows:
            assert m.verdict_for(row) == m.true_situation(row)


# --------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------
class TestDocuments:
    @pytest.mark.parametrize("policy", m.POLICIES)
    @pytest.mark.parametrize("proposed", [False])
    def test_documents_validate(self, obs, policy, proposed):
        suite, rs = m.build_documents(obs, policy=policy, proposed_verdict=proposed)
        v = validate_suite(suite)
        assert v.valid, v.errors
        v = validate_result_set(rs)
        assert v.valid, v.errors

    @pytest.mark.parametrize("policy", m.POLICIES)
    def test_rule6_invariants_hold_in_both_policies(self, obs, policy):
        rs = m.build_documents(obs, policy=policy)[1]
        for r in rs["results"]:
            for g in r["grader_results"]:
                if g["score"] is None:
                    assert g["passed"] is False                       # NULL_SCORE_PASSED
            if all(g["score"] is None for g in r["grader_results"]):
                assert r["passed"] is False                           # UNSCORED_RESULT_PASSED
                assert r["metadata"]["openeval"]["aggregation_status"] == "unscored"
            else:
                assert "openeval" not in r.get("metadata", {})

    def test_errored_grader_has_null_score_and_a_reason(self, obs):
        rs = m.build_documents(obs, policy="spec-default")[1]
        flaky = by_id(rs)["q2"]["grader_results"][1]
        assert flaky["score"] is None and flaky["passed"] is False
        assert flaky["metadata"][obs.framework] == {"produced_score": False}
        if obs.framework == "deepeval":
            # DeepEval attributes the exception to the metric that raised.
            assert "flaky grader unreachable" in flaky["reason"]
        else:
            # Inspect AI records the exception on the sample, not on a scorer, so the grader's own
            # reason can only say that no score exists; the error text is on Result.error.
            assert flaky["reason"] == "grader produced no score"

    def test_inspect_sample_error_becomes_result_error_and_deepeval_has_none(self, deepeval_obs, inspect_obs):
        de = m.build_documents(deepeval_obs)[1]
        ins = m.build_documents(inspect_obs)[1]
        assert all("error" not in r for r in de["results"])
        assert "error" not in by_id(ins)["q1"]
        for cid in ("q2", "q3", "q4"):
            assert by_id(ins)[cid]["error"]["type"] == "runner_error"

    def test_numbers_the_readme_quotes(self, obs):
        spec = m.build_documents(obs, policy="spec-default")[1]["summary"]
        closed = m.build_documents(obs, policy="fail-closed")[1]["summary"]
        assert (spec["passed"], spec["failed"], spec["skipped"], spec["pass_rate"]) == (2, 1, 1, 0.5)
        assert (closed["passed"], closed["failed"], closed["skipped"], closed["pass_rate"]) == (1, 2, 1, 0.25)
        # null scores are left out of the averages (Rule 6)
        assert spec["by_grader"]["gr_exact"] == {"passed": 2, "failed": 1, "avg_score": pytest.approx(0.6667, abs=1e-4)}
        assert spec["by_grader"]["gr_flaky"] == {"passed": 1, "failed": 0, "avg_score": 1.0}

    def test_summary_counts_are_consistent(self, obs):
        for policy in m.POLICIES:
            s = m.build_documents(obs, policy=policy)[1]["summary"]
            assert s["total"] == s["passed"] + s["failed"] + s["skipped"]

    def test_both_policies_differ_only_in_the_mixed_row(self, obs):
        a = by_id(m.build_documents(obs, policy="spec-default")[1])
        b = by_id(m.build_documents(obs, policy="fail-closed")[1])
        assert [cid for cid in a if a[cid]["passed"] != b[cid]["passed"]] == ["q2"]

    def test_python_validator_accepts_fail_closed_even_though_the_spec_prose_does_not_describe_it(self, obs):
        rs = m.build_documents(obs, policy="fail-closed")[1]
        assert by_id(rs)["q2"]["passed"] is False
        assert validate_result_set(rs).valid

    def test_negative_control_null_grader_passed_true_is_rejected(self, obs):
        rs = copy.deepcopy(m.build_documents(obs)[1])
        by_id(rs)["q2"]["grader_results"][1]["passed"] = True
        v = validate_result_set(rs)
        assert not v.valid and "NULL_SCORE_PASSED" in {e["code"] for e in v.errors}

    def test_negative_control_all_unscored_row_passed_true_is_rejected(self, obs):
        rs = copy.deepcopy(m.build_documents(obs)[1])
        by_id(rs)["q4"]["passed"] = True
        v = validate_result_set(rs)
        assert not v.valid and "UNSCORED_RESULT_PASSED" in {e["code"] for e in v.errors}

    def test_unknown_policy_is_rejected(self, obs):
        with pytest.raises(ValueError):
            m.build_documents(obs, policy="optimistic")

    def test_deterministic_documents_are_reproducible(self, obs):
        a = m.build_documents(obs, deterministic=True)
        b = m.build_documents(obs, deterministic=True)
        assert a == b and "completed_at" not in a[1]


# --------------------------------------------------------------------------------------
# The metadata-convention alternative to Result.verdict ("partial"; NOT in the spec)
# --------------------------------------------------------------------------------------
class TestPartialMarkerConvention:
    """Discussion #49 names one competing option to a ``verdict`` field: extend
    ``metadata.openeval.aggregation_status`` with ``"partial"``. These tests say what that
    convention can and cannot do on the same rows, with the validator on ``main``."""

    def test_marker_is_written_only_on_mixed_rows_and_only_when_asked(self, obs):
        plain = by_id(m.build_documents(obs)[1])
        marked = by_id(m.build_documents(obs, mark_partial=True)[1])
        status = lambda r: r.get("metadata", {}).get("openeval", {}).get("aggregation_status")  # noqa: E731
        assert {cid: status(r) for cid, r in plain.items()} == {"q1": None, "q2": None, "q3": None, "q4": "unscored"}
        assert {cid: status(r) for cid, r in marked.items()} == {"q1": None, "q2": "partial", "q3": "partial", "q4": "unscored"}

    @pytest.mark.parametrize("policy", m.POLICIES)
    def test_marked_documents_validate_on_main(self, obs, policy):
        """No schema or validator change is needed: the key is free-form metadata."""
        suite, rs = m.build_documents(obs, policy=policy, mark_partial=True)
        assert validate_suite(suite).valid
        v = validate_result_set(rs)
        assert v.valid, v.errors

    def test_marker_only_changes_metadata(self, obs):
        plain = m.build_documents(obs, deterministic=True)[1]
        marked = m.build_documents(obs, deterministic=True, mark_partial=True)[1]
        strip = lambda rs: [{k: v for k, v in r.items() if k != "metadata"} for r in rs["results"]]  # noqa: E731
        assert strip(plain) == strip(marked)
        assert plain["summary"] == marked["summary"]

    def test_with_the_spec_default_aggregation_the_convention_recovers_the_true_situation(self, obs):
        """The result that matters: on these rows, ``partial`` + Rule 6's ``unscored`` lets a
        consumer reach the same classification ``verdict`` would give, without a new field."""
        rs = m.build_documents(obs, policy="spec-default", mark_partial=True)[1]
        got = {cid: m.convention_class(r) for cid, r in by_id(rs).items()}
        want = {r.case_id: m.true_situation(r) for r in obs.rows}
        assert got == want == {"q1": "passed", "q2": "unverified", "q3": "failed", "q4": "unverified"}

    def test_but_the_reading_depends_on_an_aggregation_policy_the_document_does_not_declare(self, obs):
        """Under fail-closed, the mixed row is ``passed: false`` + ``partial``, which the convention
        must read as a verified failure; the row is in fact unverified. Nothing in the document
        says which policy produced ``passed`` (fail-closed is not an ``openeval.aggregation``
        strategy), so a consumer cannot know which reading applies. ``verdict`` is asserted by
        the producer and does not have this problem."""
        rs = m.build_documents(obs, policy="fail-closed", mark_partial=True)[1]
        q2 = by_id(rs)["q2"]
        assert q2["passed"] is False and q2["metadata"]["openeval"]["aggregation_status"] == "partial"
        assert m.convention_class(q2) == "failed"
        assert m.true_situation(next(r for r in obs.rows if r.case_id == "q2")) == "unverified"

    def test_without_the_marker_the_convention_reads_like_rule6(self, obs):
        rs = m.build_documents(obs, policy="spec-default")[1]
        assert m.convention_class(by_id(rs)["q2"]) == "passed"   # the mixed row is invisible again

    def test_the_marked_mixed_row_still_has_passed_true(self, obs):
        """What the convention cannot do: the unverified row keeps ``passed: true``, so a consumer
        that reads only ``passed`` (every consumer today) counts it as a pass. #49 would forbid
        exactly this combination (``unverified`` requires ``passed: false``)."""
        rs = m.build_documents(obs, policy="spec-default", mark_partial=True)[1]
        assert by_id(rs)["q2"]["passed"] is True and m.convention_class(by_id(rs)["q2"]) == "unverified"

    def test_cli_mark_partial_flag(self, tmp_path, capsys):
        assert m.main(["--out-dir", str(tmp_path), "--framework", "deepeval", "--mark-partial", "--deterministic"]) == 0
        out = capsys.readouterr().out
        assert "mark_partial=True" in out and "unverified / failed" in out
        rs = json.loads((tmp_path / "deepeval" / "results.json").read_text())
        assert rs["metadata"]["deepeval"]["partial_marker"].startswith("metadata.openeval.aggregation_status")
        assert by_id(rs)["q2"]["metadata"]["openeval"]["aggregation_status"] == "partial"
        assert validate_result_set(rs).valid


# --------------------------------------------------------------------------------------
# Sample output must not go stale
# --------------------------------------------------------------------------------------
class TestSampleOutput:
    @pytest.mark.parametrize("fw", ["deepeval", "inspect_ai"])
    @pytest.mark.parametrize("name", ["suite", "results"])
    def test_checked_in_sample_output_matches_a_fresh_run(self, deepeval_obs, inspect_obs, fw, name):
        obs = {"deepeval": deepeval_obs, "inspect_ai": inspect_obs}[fw]
        suite, rs = m.build_documents(obs, deterministic=True)
        fresh = {"suite": suite, "results": rs}[name]
        saved = json.loads((HERE / "sample_output" / fw / f"{name}.json").read_text())
        assert saved == fresh


# --------------------------------------------------------------------------------------
# Opt-in: the proposed Result.verdict, against the PR #83 validator
# --------------------------------------------------------------------------------------
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
        out = subprocess.run([sys.executable, "-c", self.SCRIPT, sdk], input=json.dumps(doc),
                             capture_output=True, text=True, env=env, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_fail_closed_plus_verdict(self, obs):
        """DeepEval rows carry no ``error``, so they are accepted. Inspect AI's q3 has a verified
        failing grader AND a sample ``error``; this example writes its verdict as ``failed`` and the
        proposal rejects that combination (VERDICT_ERROR_CONFLICT)."""
        rs = m.build_documents(obs, policy="fail-closed", proposed_verdict=True)[1]
        result = self.validate(rs)
        if obs.framework == "deepeval":
            assert result == {"valid": True, "codes": []}
        else:
            assert by_id(rs)["q3"]["verdict"] == "failed" and "error" in by_id(rs)["q3"]
            assert result == {"valid": False, "codes": ["VERDICT_ERROR_CONFLICT"]}

    def test_spec_default_plus_verdict_is_rejected_on_the_mixed_row(self, obs):
        """The interaction the RFC text does not address. The spec's default aggregation makes q2
        ``passed: true``; the proposal requires ``unverified`` to be ``passed: false``."""
        rs = m.build_documents(obs, policy="spec-default", proposed_verdict=True)[1]
        assert by_id(rs)["q2"]["passed"] is True and by_id(rs)["q2"]["verdict"] == "unverified"
        result = self.validate(rs)
        assert result["valid"] is False and "VERDICT_PASSED_MISMATCH" in result["codes"]
        expected = ["VERDICT_PASSED_MISMATCH"] if obs.framework == "deepeval" else ["VERDICT_ERROR_CONFLICT", "VERDICT_PASSED_MISMATCH"]
        assert result["codes"] == expected

    def test_inspect_q3_is_valid_only_if_the_producer_downgrades_the_verdict_to_unverified(self, inspect_obs):
        """The workaround the proposal leaves a producer: drop a failure that *was* established from
        the verdict. (Dropping the verdict instead leaves Rule 6's reading, "verified failing".)"""
        rs = m.build_documents(inspect_obs, policy="fail-closed", proposed_verdict=True)[1]
        by_id(rs)["q3"]["verdict"] = "unverified"
        assert self.validate(rs) == {"valid": True, "codes": []}
        del by_id(rs)["q3"]["verdict"]
        assert self.validate(rs) == {"valid": True, "codes": []}

    def test_inspect_error_row_with_verdict_unverified_is_allowed(self, inspect_obs):
        rs = m.build_documents(inspect_obs, policy="fail-closed", proposed_verdict=True)[1]
        for cid in ("q2", "q4"):
            assert by_id(rs)[cid]["error"]["type"] == "runner_error" and by_id(rs)[cid]["verdict"] == "unverified"

    def test_partial_marker_and_verdict_can_coexist_under_fail_closed(self, deepeval_obs):
        """The two mechanisms are not exclusive: a producer can write both. #83 ignores the
        metadata key, and the marker does not change ``passed``, so the fail-closed document is
        still accepted and the spec-default one is still rejected on the mixed row."""
        ok = m.build_documents(deepeval_obs, policy="fail-closed", proposed_verdict=True, mark_partial=True)[1]
        assert self.validate(ok) == {"valid": True, "codes": []}
        bad = m.build_documents(deepeval_obs, policy="spec-default", proposed_verdict=True, mark_partial=True)[1]
        assert self.validate(bad) == {"valid": False, "codes": ["VERDICT_PASSED_MISMATCH"]}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
class TestCli:
    def test_cli_writes_valid_documents_for_both_frameworks(self, tmp_path, capsys):
        assert m.main(["--out-dir", str(tmp_path), "--deterministic"]) == 0
        out = capsys.readouterr().out
        assert "pass_rate=0.5" in out and "native run-level" in out
        for fw in ("deepeval", "inspect_ai"):
            assert validate_suite(json.loads((tmp_path / fw / "suite.json").read_text())).valid
            assert validate_result_set(json.loads((tmp_path / fw / "results.json").read_text())).valid

    def test_cli_fail_closed_policy(self, tmp_path, capsys):
        assert m.main(["--out-dir", str(tmp_path), "--framework", "deepeval", "--policy", "fail-closed",
                       "--deterministic"]) == 0
        assert "pass_rate=0.25" in capsys.readouterr().out

    def test_cli_proposed_verdict_flag(self, tmp_path):
        assert m.main(["--out-dir", str(tmp_path), "--framework", "deepeval", "--policy", "fail-closed",
                       "--proposed-verdict", "--deterministic"]) == 0
        rs = json.loads((tmp_path / "deepeval" / "results.json").read_text())
        assert [r["verdict"] for r in rs["results"]] == ["passed", "unverified", "failed", "unverified"]

    def test_cli_non_deterministic_mode_has_timestamps(self, tmp_path):
        assert m.main(["--out-dir", str(tmp_path), "--framework", "inspect_ai"]) == 0
        rs = json.loads((tmp_path / "inspect_ai" / "results.json").read_text())
        assert "completed_at" in rs and validate_result_set(rs).valid
