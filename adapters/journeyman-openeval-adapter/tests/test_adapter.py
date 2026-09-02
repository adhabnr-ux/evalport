"""Tests for journeyman_openeval_adapter.

Two tiers, both against real code -- no mocks, no reinvented stand-ins:

1. The primary suite drives the REAL, installed `journeyman-bench` package's
   own pipeline (`driver.run_grid`, `judge.judge_cell`, `record.RunDir`,
   `report.render`, `scene.REGISTRY`) -- the same functions
   `journeyman selftest` and `journeyman run` call -- to produce a real run
   directory (real `cells/<id>.json`, real `report.json`), then feeds that
   real output into this adapter and validates the result with the real
   `openeval.validate` module (installed from PyPI as `evalport-sdk`).
   It also performs the maintainer's own suggested "strictest round-trip
   test" (see codechu/journeyman#1): re-rendering via journeyman's real
   `journeyman report <run_dir>` reproduces the same axis scores, and this
   adapter's ResultSet reproduces them too.

2. A second, smaller tier hand-builds cell-record dicts for shapes the
   `echo-well` selftest scene doesn't exercise (a not-applicable verdict,
   an invalid cell) -- each one matching the documented shape in
   `record.py` / `docs/run-guide.md` and journeyman's own observed
   `report.py` handling of it, not a guess.
"""
import json
import subprocess
import sys

import pytest

from journeyman.driver import run_grid
from journeyman.judge import judge_cell
from journeyman.record import RunDir
from journeyman.report import render
from journeyman.scene import REGISTRY
from journeyman.selftest import FakeEndpoint  # imports echo-well as a side effect

from openeval.validate import validate_suite, validate_result_set

from journeyman_openeval_adapter import cells_to_testcases, cells_to_result_set, SUPPORTED_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Tier 1: a real journeyman run, produced by real journeyman code
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """A real run directory: real driver.run_grid + judge.judge_cell +
    report.render against the real, registered `echo-well` scene and the
    real `FakeEndpoint` journeyman's own selftest.py uses -- two seeds, so
    per-seed and per-axis behaviour is genuinely exercised, not a
    single-cell toy."""
    root = tmp_path_factory.mktemp("jman")
    rd = RunDir(root=str(root), stamp="run1")
    fake = FakeEndpoint()
    seal = run_grid(fake, ["echo-well"], [4242, 777], rd, log=lambda *_: None)
    cells = list(rd.read_cells())
    for c in cells:
        judge_fake = FakeEndpoint()
        c["verdicts"] = judge_cell(judge_fake, REGISTRY["echo-well"](), c, log=lambda *_: None)
        rd.write_cell(c["cell_id"], c)
    render(rd, seal, "SELF (default)", self_judged=True)
    report = json.load(open(f"{rd.path}/report.json"))
    cells = list(rd.read_cells())
    return {"run_dir": rd.path, "report": report, "cells": cells}


def test_real_report_schema_version_matches_adapter(real_run):
    # If journeyman ever bumps SCHEMA_VERSION, this is the canary that
    # should fail before any silent-guess bug does.
    assert real_run["report"]["schema_version"] == SUPPORTED_SCHEMA_VERSION


def test_suite_from_real_cells_validates(real_run):
    suite = cells_to_testcases(real_run["cells"], suite_id="jman_demo")
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_result_set_from_real_cells_validates(real_run):
    result_set = cells_to_result_set(
        real_run["cells"], real_run["report"], suite_id="jman_demo", run_id="run1",
        started_at="2026-09-01T00:00:00Z",
    )
    result = validate_result_set(result_set)
    assert result.valid, result.errors


def test_suite_test_case_ids_match_result_set_ids(real_run):
    suite = cells_to_testcases(real_run["cells"], suite_id="jman_demo")
    result_set = cells_to_result_set(
        real_run["cells"], real_run["report"], suite_id="jman_demo", run_id="run1",
    )
    suite_ids = {tc["id"] for tc in suite["test_cases"]}
    result_ids = {r["test_case_id"] for r in result_set["results"]}
    assert suite_ids == result_ids == {"echo-well_s4242", "echo-well_s777"}


def test_judged_axis_never_uses_llm_judge_type(real_run):
    suite = cells_to_testcases(real_run["cells"], suite_id="jman_demo")
    for g in suite["graders"]:
        assert g["type"] != "llm_judge"
    route = next(g for g in suite["graders"] if g["id"] == "route-discipline")
    assert route["params"]["kind"] == "judged"
    probe = next(g for g in suite["graders"] if g["id"] == "probe-economy")
    assert probe["params"]["kind"] == "counted"
    assert probe["params"]["deterministic"] is True


def test_counted_axis_grader_result_marked_deterministic(real_run):
    result_set = cells_to_result_set(
        real_run["cells"], real_run["report"], suite_id="jman_demo", run_id="run1",
    )
    for r in result_set["results"]:
        counted = [g for g in r["grader_results"] if g["grader_id"] == "probe-economy"]
        assert len(counted) == 1
        assert counted[0]["metadata"]["kind"] == "counted"
        assert counted[0]["metadata"]["deterministic"] is True
        judged = [g for g in r["grader_results"] if g["grader_id"] == "route-discipline"]
        assert judged[0]["metadata"]["kind"] == "judged"


def test_seal_judge_self_judged_travel_with_the_result_set(real_run):
    """The condition obarlik set on codechu/journeyman#1: a listing must
    carry the conditions a score was true under, or it does not travel."""
    result_set = cells_to_result_set(
        real_run["cells"], real_run["report"], suite_id="jman_demo", run_id="run1",
    )
    meta = result_set["metadata"]["journeyman"]
    assert meta["seal"] == real_run["report"]["seal"]
    assert meta["judge"] == real_run["report"]["judge"]
    assert meta["self_judged"] is True
    assert meta["comparability"] == "NOT_COMPARABLE"


def test_missing_report_fields_refused(real_run):
    incomplete = {k: v for k, v in real_run["report"].items() if k != "self_judged"}
    with pytest.raises(ValueError, match="self_judged"):
        cells_to_result_set(real_run["cells"], incomplete, suite_id="s", run_id="r1")


def test_unknown_schema_version_refused_by_default(real_run):
    bumped = dict(real_run["report"], schema_version=99)
    with pytest.raises(ValueError, match="schema_version"):
        cells_to_result_set(real_run["cells"], bumped, suite_id="s", run_id="r1")
    # explicit opt-out still works
    result_set = cells_to_result_set(
        real_run["cells"], bumped, suite_id="s", run_id="r1", strict_schema=False,
    )
    assert result_set["metadata"]["journeyman"]["schema_version"] == 99


def test_judge_field_passed_through_opaque_not_parsed(real_run):
    result_set = cells_to_result_set(
        real_run["cells"], real_run["report"], suite_id="jman_demo", run_id="run1",
    )
    # exact string equality -- proves nothing was split/parsed/host-extracted
    assert result_set["metadata"]["journeyman"]["judge"] == real_run["report"]["judge"] == "SELF (default)"


def test_strictest_round_trip_matches_journeymans_own_rerender(real_run):
    """The maintainer's own suggested test (codechu/journeyman#1): re-render
    via the REAL `journeyman report <run_dir>` CLI and confirm this
    adapter's ResultSet scores match both the original report.json and the
    re-rendered one exactly."""
    subprocess.run(
        [sys.executable, "-m", "journeyman", "report", real_run["run_dir"]],
        check=True, capture_output=True, text=True,
    )
    rerendered = json.load(open(f"{real_run['run_dir']}/report.json"))
    assert rerendered["axes"] == real_run["report"]["axes"]

    cells = list(RunDir.attach(real_run["run_dir"]).read_cells())
    result_set = cells_to_result_set(
        cells, rerendered, suite_id="jman_demo", run_id="run1",
    )
    # Recompute each axis's mean score from the adapter's own GraderResults
    # and compare against report.json's authoritative per-axis score --
    # the adapter must reproduce journeyman's own arithmetic, not just its
    # shape.
    for axis, body in rerendered["axes"].items():
        if body["score"] is None:
            continue
        scores = [
            g["score"] for r in result_set["results"] for g in r["grader_results"]
            if g["grader_id"] == axis and g["score"] is not None
        ]
        assert scores, f"no scores recovered for axis {axis}"
        assert round(sum(scores) / len(scores), 2) == body["score"], axis


# ---------------------------------------------------------------------------
# Tier 2: hand-built cells for shapes echo-well doesn't exercise
# ---------------------------------------------------------------------------

def _base_cell(**overrides):
    cell = {
        "cell_id": "closed-roads-noway_s4242", "scene": "closed-roads-noway", "seed": 4242,
        "invalid": False, "invalid_reason": None,
        "messages": [{"role": "user", "content": "Detour task.\n\nTool budget: 6 calls."}],
        "final_text": "Filed the report.", "budget": 6, "calls": 3,
        "tokens_in": 500, "tokens_out": 80, "seconds": 4.2,
        "events": {"total_calls": 3}, "event_axes": {}, "verdicts": {},
    }
    cell.update(overrides)
    return cell


_REPORT = {
    "schema_version": 1,
    "seal": {"bench": "0.4.0", "scene_set": "dev", "scene_md5": {}, "seeds": [4242], "model": "m"},
    "judge": "SELF (default)", "self_judged": True, "nonstandard": None,
}


def test_not_applicable_verdict_excluded_but_not_dropped():
    cell = _base_cell(verdicts={
        "wall-pricing": {"verdict": "na", "positive": "priced", "na_means": "not-applicable", "raw": "no wall hit"}
    })
    result_set = cells_to_result_set([cell], _REPORT, suite_id="s", run_id="r1")
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["score"] is None
    assert gr["passed"] is False
    assert gr["metadata"]["excluded_from_axis_score"] is True
    assert validate_result_set(result_set).valid


def test_na_as_failure_scores_zero_not_excluded():
    cell = _base_cell(verdicts={
        "wall-pricing": {"verdict": "na", "positive": "priced", "na_means": "failure", "raw": "empty relief page"}
    })
    result_set = cells_to_result_set([cell], _REPORT, suite_id="s", run_id="r1")
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["score"] == 0.0
    assert gr["passed"] is False
    assert gr["metadata"]["excluded_from_axis_score"] is False


def test_unparsed_verdict_scores_zero_like_journeymans_own_axis_scores():
    # judge.py: an unparsed verdict is stored verbatim as "__unparsed__",
    # which never equals `positive` -- report.py's axis_scores() therefore
    # already scores it 0.0 without any special-casing, and this adapter
    # matches that rather than adding a third state journeyman doesn't have.
    cell = _base_cell(verdicts={
        "wall-pricing": {"verdict": "__unparsed__", "positive": "priced", "na_means": "failure", "raw": "garbled"}
    })
    result_set = cells_to_result_set([cell], _REPORT, suite_id="s", run_id="r1")
    gr = result_set["results"][0]["grader_results"][0]
    assert gr["score"] == 0.0
    assert gr["passed"] is False


def test_invalid_cell_produces_no_grader_results_and_is_not_a_silent_pass():
    cell = _base_cell(invalid=True, invalid_reason="TimeoutError('endpoint')", events=None, event_axes=None, verdicts=None)
    result_set = cells_to_result_set([cell], _REPORT, suite_id="s", run_id="r1")
    r = result_set["results"][0]
    assert r["grader_results"] == []
    assert r["passed"] is False
    assert r["error"]["type"] == "invalid_cell"
    assert "TimeoutError" in r["error"]["message"]
    assert validate_result_set(result_set).valid


def test_invalid_cell_still_becomes_a_testcase_with_no_graders():
    cell = _base_cell(invalid=True, invalid_reason="boom", event_axes={}, verdicts={})
    suite = cells_to_testcases([cell], suite_id="s")
    tc = suite["test_cases"][0]
    assert tc["graders"] == []
    assert tc["metadata"]["journeyman"]["invalid"] is True


def test_final_text_missing_falls_back_to_closing_tool_call():
    cell = _base_cell(final_text=None, messages=[
        {"role": "user", "content": "Task.\n\nTool budget: 4 calls."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "report", "arguments": '{"text": "Closing via tool call."}'}},
        ]},
    ])
    result_set = cells_to_result_set([cell], _REPORT, suite_id="s", run_id="r1")
    r = result_set["results"][0]
    assert r["actual_output"] == "Closing via tool call."
    assert r["metadata"]["journeyman"]["output_source"] == "closing_tool_call"


def test_rubric_index_enriches_judged_grader_params_but_is_optional():
    cell = _base_cell(verdicts={
        "wall-pricing": {"verdict": "priced", "positive": "priced", "na_means": "failure", "raw": "..."}
    })
    from dataclasses import dataclass

    @dataclass
    class FakeRubricItem:
        axis: str
        question: str
        verdicts: tuple
        positive: str
        na_means: str = "failure"

    item = FakeRubricItem(axis="wall-pricing", question="Was the wall priced?",
                          verdicts=("priced", "unpriced", "none"), positive="priced")
    suite_with = cells_to_testcases([cell], suite_id="s", rubric_index={"wall-pricing": item})
    suite_without = cells_to_testcases([cell], suite_id="s")
    g_with = suite_with["graders"][0]
    g_without = suite_without["graders"][0]
    assert g_with["params"]["question"] == "Was the wall priced?"
    assert "question" not in g_without["params"]
    # discrimination (judged vs counted) is identical either way -- rubric_index
    # only adds readability, never changes what's counted vs judged
    assert g_with["params"]["kind"] == g_without["params"]["kind"] == "judged"
    assert validate_suite(suite_with).valid
    assert validate_suite(suite_without).valid


def test_missing_input_raises():
    cell = _base_cell(messages=[{"role": "assistant", "content": "no user turn"}])
    with pytest.raises(ValueError, match="no user-role message"):
        cells_to_testcases([cell], suite_id="s")
