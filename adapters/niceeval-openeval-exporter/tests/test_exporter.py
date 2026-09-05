"""
Tests for niceeval-openeval-exporter.

Every fixture here mirrors the *real* NiceEval Inspection query protocol
shapes, verified directly against NiceEval source (see the module docstring
in src/niceeval_openeval_exporter/__init__.py for exact file paths):

  - InspectionRunSummaryResultSchema / InspectionAttemptResultSchema /
    InspectionScoredValueSchema / AssertionIndexSchema
      (packages/niceeval/src/inspection/results.ts)
  - RunDocument / AttemptDocument field shapes
      (packages/niceeval/src/record/model/definition.ts)

No NiceEval runtime is imported or required -- these are hand-built dicts
matching the documented Effect Schema shapes, the same "duck-typed against
real, verified source" approach used by every other adapter in this repo
that targets an uninstallable/foreign-language source system.

Every produced ResultSet is validated against the REAL
openeval.validate.validate_result_set() from the installed evalport-sdk,
never a hand-rolled check.
"""

from __future__ import annotations

import json

import pytest
from openeval.validate import validate_result_set

from niceeval_openeval_exporter import (
    OPENEVAL_VERSION_FALLBACK,
    member_to_result,
    to_openeval,
    to_openeval_json,
)
from niceeval_openeval_exporter import _clamp01, _iso_from_utc_millis, _score_from_scored_value


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def make_run(run_id="run_1", experiment_id="exp_1", started_at_ms=1_700_000_000_000, completed_at_ms=1_700_000_060_000):
    return {
        "runId": run_id,
        "experimentId": experiment_id,
        "context": {},
        "startedAt": started_at_ms,
        "completedAt": completed_at_ms,
        "expectedSlots": [],
    }


def make_member(
    *,
    run_id="run_1",
    slot_id="slot_1",
    eval_id="evals/weather-tool.eval.ts::fetches-forecast",
    attempt_ordinal=0,
    digest="a" * 64,
    state="executed",
    locator="loc_1",
    outcome="completed",
    verdict="passed",
    score=None,
):
    member = {
        "runId": run_id,
        "slotId": slot_id,
        "evalId": eval_id,
        "attemptOrdinal": attempt_ordinal,
        "executionIdentityDigest": digest,
        "state": state,
        "locator": locator,
        "outcome": outcome,
        "verdict": verdict,
    }
    if score is not None:
        member["score"] = score
    return member


def make_run_summary(members, runs=None, expected=None, observed=None):
    runs = runs if runs is not None else [make_run()]
    return {
        "runs": runs,
        "denominator": {
            "expected": expected if expected is not None else len(members),
            "observed": observed if observed is not None else len(members),
        },
        "members": members,
    }


def make_attempt(locator="loc_1", entries=None):
    return {
        "core": {
            "attemptId": "att_1",
            "originRunId": "run_1",
            "slotId": "slot_1",
            "evalId": "evals/weather-tool.eval.ts::fetches-forecast",
            "executionIdentityDigest": "a" * 64,
            "outcome": "completed",
        },
        "locator": locator,
        "originRun": make_run(),
        "targets": [],
        "evidence": {"state": "available", "entryCount": len(entries or []), "sourceSiteCount": 1},
        "assertions": {
            "state": "available",
            "entries": [
                {
                    "entryId": e["entry_id"],
                    "display": {
                        "label": e.get("label"),
                        "key": e.get("key"),
                        "groupPath": e.get("group_path", []),
                    },
                }
                for e in (entries or [])
            ],
        },
        "sections": {},
        "verdict": "passed",
        "score": {"state": "not-scored"},
        "evidenceCoverage": [],
        "limitations": [],
    }


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------

class TestIsoFromUtcMillis:
    def test_epoch_zero(self):
        assert _iso_from_utc_millis(0) == "1970-01-01T00:00:00+00:00"

    def test_typical_value(self):
        result = _iso_from_utc_millis(1_700_000_000_000)
        assert result.startswith("2023-11-14")

    def test_rejects_negative(self):
        assert _iso_from_utc_millis(-1) is None

    def test_rejects_non_number(self):
        assert _iso_from_utc_millis("1700000000000") is None
        assert _iso_from_utc_millis(None) is None

    def test_rejects_bool(self):
        # bool is a subclass of int in Python; UtcMillis is never a boolean.
        assert _iso_from_utc_millis(True) is None

    def test_rejects_nan_and_inf(self):
        assert _iso_from_utc_millis(float("nan")) is None
        assert _iso_from_utc_millis(float("inf")) is None


class TestClamp01:
    def test_within_range(self):
        assert _clamp01(0.5) == 0.5

    def test_clamps_high(self):
        assert _clamp01(1.5) == 1.0

    def test_clamps_low(self):
        assert _clamp01(-0.5) == 0.0

    def test_rejects_non_number(self):
        assert _clamp01("0.5") is None

    def test_rejects_nan(self):
        assert _clamp01(float("nan")) is None


class TestScoreFromScoredValue:
    def test_none(self):
        score, raw = _score_from_scored_value(None)
        assert score is None
        assert raw == {"state": "absent"}

    def test_not_scored(self):
        score, raw = _score_from_scored_value({"state": "not-scored"})
        assert score is None
        assert raw == {"state": "not-scored"}

    def test_complete(self):
        score, raw = _score_from_scored_value({"state": "complete", "earned": 3, "possible": 4})
        assert score == 0.75
        assert raw == {"state": "complete", "earned": 3, "possible": 4}

    def test_unavailable(self):
        score, raw = _score_from_scored_value(
            {"state": "unavailable", "earned": 2, "possible": 4, "unavailable": 1}
        )
        assert score == 0.5
        assert raw["unavailable"] == 1

    def test_complete_zero_possible_never_divides_by_zero(self):
        score, raw = _score_from_scored_value({"state": "complete", "earned": 0, "possible": 0})
        assert score is None
        assert raw["possible"] == 0


# ---------------------------------------------------------------------------
# member_to_result branch coverage
# ---------------------------------------------------------------------------

class TestMemberToResult:
    def test_passed_with_complete_score(self):
        member = make_member(verdict="passed", score={"state": "complete", "earned": 5, "possible": 5})
        result = member_to_result(member, test_case_id="eval_1")
        assert result["passed"] is True
        assert "error" not in result
        assert len(result["grader_results"]) == 1
        gr = result["grader_results"][0]
        assert gr["passed"] is True
        assert gr["score"] == 1.0
        assert gr["type"] == "niceeval_verdict"
        assert gr["grader_id"] == "gr_niceeval_verdict"

    def test_failed_with_partial_score(self):
        member = make_member(
            verdict="failed", score={"state": "unavailable", "earned": 1, "possible": 3, "unavailable": 1}
        )
        result = member_to_result(member, test_case_id="eval_1")
        assert result["passed"] is False
        assert "error" not in result
        gr = result["grader_results"][0]
        assert gr["passed"] is False
        assert abs(gr["score"] - (1 / 3)) < 1e-9
        assert "1 assertion(s) unavailable" in gr["reason"]

    def test_failed_with_not_scored(self):
        member = make_member(verdict="failed", score={"state": "not-scored"})
        result = member_to_result(member, test_case_id="eval_1")
        gr = result["grader_results"][0]
        assert gr["score"] is None
        assert gr["passed"] is False

    def test_errored_execution_failure(self):
        member = make_member(verdict="errored", outcome="errored")
        result = member_to_result(member, test_case_id="eval_1")
        assert result["passed"] is False
        assert result["grader_results"] == []
        assert result["error"]["type"] == "runner_error"

    def test_errored_cancelled_outcome(self):
        member = make_member(verdict="errored", outcome="cancelled")
        result = member_to_result(member, test_case_id="eval_1")
        assert result["error"]["type"] == "runner_error"

    def test_errored_assertion_failure_with_completed_outcome(self):
        member = make_member(verdict="errored", outcome="completed")
        result = member_to_result(member, test_case_id="eval_1")
        assert result["passed"] is False
        assert result["grader_results"] == []
        assert result["error"]["type"] == "assertion_error"

    def test_skipped(self):
        member = make_member(verdict="skipped", outcome="completed")
        result = member_to_result(member, test_case_id="eval_1")
        assert result["passed"] is False
        assert result["grader_results"] == []
        assert result["error"]["type"] == "skipped"

    def test_not_dispatched_never_evaluated(self):
        member = make_member(verdict=None, outcome=None, state="not-dispatched", locator=None)
        result = member_to_result(member, test_case_id="eval_1")
        assert result["passed"] is False
        assert result["grader_results"] == []
        assert result["error"]["type"] == "not_evaluated"

    def test_missing_state_never_evaluated(self):
        member = make_member(verdict=None, outcome=None, state="missing", locator=None)
        result = member_to_result(member, test_case_id="eval_1")
        assert result["error"]["type"] == "not_evaluated"

    def test_interrupted_state_never_evaluated(self):
        member = make_member(verdict=None, outcome=None, state="interrupted", locator=None)
        result = member_to_result(member, test_case_id="eval_1")
        assert result["error"]["type"] == "not_evaluated"

    def test_attempt_ordinal_converted_to_one_based(self):
        # NiceEval's attemptOrdinal is zero-based; EvalPort's Result.attempt
        # must be >= 1 (verified against the real validator), so it is
        # converted, not passed through raw.
        member = make_member(attempt_ordinal=2)
        result = member_to_result(member, test_case_id="eval_1")
        assert result["attempt"] == 3

    def test_zero_attempt_ordinal_converted_to_one(self):
        member = make_member(attempt_ordinal=0)
        result = member_to_result(member, test_case_id="eval_1")
        assert result["attempt"] == 1

    def test_metadata_preserves_real_niceeval_fields(self):
        member = make_member()
        result = member_to_result(member, test_case_id="eval_1")
        meta = result["metadata"]["niceeval"]
        assert meta["run_id"] == "run_1"
        assert meta["slot_id"] == "slot_1"
        assert meta["execution_identity_digest"] == "a" * 64
        assert meta["locator"] == "loc_1"

    def test_assertion_entries_attached_when_attempt_supplied(self):
        member = make_member(verdict="passed", score={"state": "not-scored"})
        attempt = make_attempt(
            locator="loc_1",
            entries=[
                {"entry_id": "e1", "label": "checks forecast field", "key": "forecast-present", "group_path": ["output"]},
            ],
        )
        result = member_to_result(member, attempt=attempt, test_case_id="eval_1")
        gr = result["grader_results"][0]
        assertions = gr["metadata"]["niceeval"]["assertions"]
        assert assertions == [
            {"entry_id": "e1", "label": "checks forecast field", "key": "forecast-present", "group_path": ["output"]}
        ]

    def test_no_assertions_key_when_index_unavailable(self):
        member = make_member(verdict="passed", score={"state": "not-scored"})
        attempt = make_attempt(locator="loc_1", entries=[])
        attempt["assertions"] = {"state": "not-recorded", "entries": []}
        result = member_to_result(member, attempt=attempt, test_case_id="eval_1")
        gr = result["grader_results"][0]
        assert "assertions" not in gr["metadata"]["niceeval"]


# ---------------------------------------------------------------------------
# to_openeval end-to-end, validated against the real EvalPort validator
# ---------------------------------------------------------------------------

class TestToOpenEval:
    def test_single_passed_member_validates(self):
        run_summary = make_run_summary(
            [make_member(verdict="passed", score={"state": "complete", "earned": 1, "possible": 1})]
        )
        rs = to_openeval(run_summary)
        outcome = validate_result_set(rs)
        assert outcome.valid, outcome.errors
        assert rs["run_id"] == "run_1"
        assert rs["suite_id"] == "niceeval_exp_1"
        assert rs["summary"]["total"] == 1
        assert rs["summary"]["passed"] == 1

    def test_mixed_verdicts_validate_and_summarize(self):
        members = [
            make_member(locator="loc_1", eval_id="e1", verdict="passed", score={"state": "complete", "earned": 1, "possible": 1}),
            make_member(locator="loc_2", eval_id="e2", verdict="failed", score={"state": "complete", "earned": 0, "possible": 1}),
            make_member(locator="loc_3", eval_id="e3", verdict="errored", outcome="errored"),
            make_member(locator="loc_4", eval_id="e4", verdict="skipped"),
            make_member(locator=None, eval_id="e5", verdict=None, outcome=None, state="not-dispatched"),
        ]
        run_summary = make_run_summary(members)
        rs = to_openeval(run_summary)
        outcome = validate_result_set(rs)
        assert outcome.valid, outcome.errors
        assert rs["summary"]["total"] == 5
        assert rs["summary"]["passed"] == 1
        assert rs["summary"]["failed"] == 4

    def test_derives_timestamps_from_run_document(self):
        run_summary = make_run_summary(
            [make_member()], runs=[make_run(started_at_ms=1_700_000_000_000, completed_at_ms=1_700_000_100_000)]
        )
        rs = to_openeval(run_summary)
        assert rs["started_at"].startswith("2023-11-14")
        assert rs["completed_at"].startswith("2023-11-14")

    def test_multi_run_uses_min_start_max_complete(self):
        run_summary = make_run_summary(
            [make_member(run_id="run_1"), make_member(run_id="run_2", locator="loc_2", eval_id="e2")],
            runs=[
                make_run(run_id="run_1", started_at_ms=1_700_000_010_000, completed_at_ms=1_700_000_050_000),
                make_run(run_id="run_2", started_at_ms=1_700_000_000_000, completed_at_ms=1_700_000_090_000),
            ],
        )
        rs = to_openeval(run_summary)
        assert rs["started_at"] == _iso_from_utc_millis(1_700_000_000_000)
        assert rs["completed_at"] == _iso_from_utc_millis(1_700_000_090_000)

    def test_explicit_overrides_win(self):
        run_summary = make_run_summary([make_member()])
        rs = to_openeval(
            run_summary,
            run_id="custom_run",
            suite_id="custom_suite",
            started_at="2020-01-01T00:00:00+00:00",
            completed_at="2020-01-01T00:01:00+00:00",
            version="9.9.9",
        )
        assert rs["run_id"] == "custom_run"
        assert rs["suite_id"] == "custom_suite"
        assert rs["started_at"] == "2020-01-01T00:00:00+00:00"
        assert rs["completed_at"] == "2020-01-01T00:01:00+00:00"
        assert rs["version"] == "9.9.9"

    def test_multi_slot_prefixes_test_case_id(self):
        members = [
            make_member(slot_id="slot_a", eval_id="e1", locator="loc_a"),
            make_member(slot_id="slot_b", eval_id="e1", locator="loc_b"),
        ]
        run_summary = make_run_summary(members)
        rs = to_openeval(run_summary)
        ids = {r["test_case_id"] for r in rs["results"]}
        assert ids == {"slot_a::e1", "slot_b::e1"}

    def test_single_slot_uses_bare_eval_id(self):
        run_summary = make_run_summary([make_member(eval_id="e1")])
        rs = to_openeval(run_summary)
        assert rs["results"][0]["test_case_id"] == "e1"

    def test_duplicate_test_case_ids_are_disambiguated(self):
        members = [
            make_member(eval_id="e1", attempt_ordinal=0, locator="loc_1"),
            make_member(eval_id="e1", attempt_ordinal=0, locator="loc_2"),
        ]
        run_summary = make_run_summary(members)
        rs = to_openeval(run_summary)
        ids = [r["test_case_id"] for r in rs["results"]]
        assert ids[0] == "e1"
        assert ids[1] == "e1#1"
        outcome = validate_result_set(rs)
        assert outcome.valid, outcome.errors

    def test_attempts_enrich_matching_members_by_locator(self):
        run_summary = make_run_summary(
            [make_member(locator="loc_1", verdict="passed", score={"state": "not-scored"})]
        )
        attempt = make_attempt(locator="loc_1", entries=[{"entry_id": "e1", "label": "l1", "group_path": []}])
        rs = to_openeval(run_summary, attempts=[attempt])
        gr = rs["results"][0]["grader_results"][0]
        assert gr["metadata"]["niceeval"]["assertions"][0]["entry_id"] == "e1"

    def test_attempts_with_no_matching_locator_do_not_enrich(self):
        run_summary = make_run_summary(
            [make_member(locator="loc_1", verdict="passed", score={"state": "not-scored"})]
        )
        attempt = make_attempt(locator="loc_other", entries=[{"entry_id": "e1", "label": "l1", "group_path": []}])
        rs = to_openeval(run_summary, attempts=[attempt])
        gr = rs["results"][0]["grader_results"][0]
        assert "assertions" not in gr["metadata"]["niceeval"]

    def test_denominator_preserved_in_metadata(self):
        run_summary = make_run_summary([make_member()], expected=5, observed=1)
        rs = to_openeval(run_summary)
        assert rs["metadata"]["niceeval"]["denominator"] == {"expected": 5, "observed": 1}

    def test_accepts_full_cli_envelope(self):
        run_summary = make_run_summary([make_member()])
        envelope = {
            "protocol": "niceeval.query/v1",
            "outcome": "success",
            "operation": "run.summary",
            "source": {"kind": "project-record", "sealedCutoffIdentity": "x"},
            "sealedCutoff": {"kind": "inspection-sealed-cutoff", "identity": "x", "runCount": 1, "runs": []},
            "selection": {"requestedRunIds": ["run_1"], "selectedRunIds": ["run_1"], "missingRunIds": []},
            "issues": [],
            "evidence": {"refs": []},
            **run_summary,
        }
        rs = to_openeval(envelope)
        outcome = validate_result_set(rs)
        assert outcome.valid, outcome.errors

    def test_rejects_failure_document(self):
        failure_doc = {
            "protocol": "niceeval.query/v1",
            "outcome": "failure",
            "operation": "run.summary",
            "failure": {"code": "inspection-selection-missing", "reason": "no such run", "correction": "choose-existing-selection"},
        }
        with pytest.raises(ValueError, match="failure document"):
            to_openeval(failure_doc)

    def test_rejects_empty_members(self):
        with pytest.raises(ValueError, match="no members"):
            to_openeval(make_run_summary([]))

    def test_rejects_no_runs_without_run_id_override(self):
        run_summary = make_run_summary([make_member()], runs=[])
        with pytest.raises(ValueError, match="run_id"):
            to_openeval(run_summary)

    def test_no_runs_but_run_id_and_started_at_supplied_succeeds(self):
        run_summary = make_run_summary([make_member()], runs=[])
        rs = to_openeval(run_summary, run_id="manual_run", started_at="2024-01-01T00:00:00+00:00")
        outcome = validate_result_set(rs)
        assert outcome.valid, outcome.errors
        assert rs["run_id"] == "manual_run"

    def test_default_version_matches_installed_sdk(self):
        from openeval.types import OPENEVAL_VERSION

        run_summary = make_run_summary([make_member()])
        rs = to_openeval(run_summary)
        assert rs["version"] == OPENEVAL_VERSION

    def test_version_fallback_constant_is_a_string(self):
        assert isinstance(OPENEVAL_VERSION_FALLBACK, str)
        assert OPENEVAL_VERSION_FALLBACK


class TestToOpenEvalJson:
    def test_parses_json_text(self):
        run_summary = make_run_summary([make_member()])
        rs = to_openeval_json(json.dumps(run_summary))
        outcome = validate_result_set(rs)
        assert outcome.valid, outcome.errors

    def test_parses_attempts_json(self):
        run_summary = make_run_summary(
            [make_member(locator="loc_1", verdict="passed", score={"state": "not-scored"})]
        )
        attempt = make_attempt(locator="loc_1", entries=[{"entry_id": "e1", "label": "l1", "group_path": []}])
        rs = to_openeval_json(json.dumps(run_summary), attempts_json=[json.dumps(attempt)])
        gr = rs["results"][0]["grader_results"][0]
        assert gr["metadata"]["niceeval"]["assertions"][0]["entry_id"] == "e1"
