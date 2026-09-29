"""Tests for frontieror-openeval-adapter.

Framework-free tests (everything except the "upstream parity" block) run with
only evalport-sdk installed, as in CI's min mode. The parity tests import
FrontierOR's own code from a checkout (it is not on PyPI): set
FRONTIEROR_ROOT=/path/to/FrontierOR to run them; they skip otherwise.
"""
import csv
import importlib.util
import json
import os
import sys

import pytest
from openeval.validate import validate_result_set, validate_suite

import frontieror_openeval_adapter as A
from frontieror_openeval_adapter import (
    GRADER_BINARY_QTE,
    GRADER_QUALITY_ONLY,
    GRADER_STAGED_QTE,
    NonPublicInstanceError,
    from_openeval,
    quality_only_score,
    results_to_openeval,
    signed_gap,
    staged_qte_builtin,
    to_openeval,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.dirname(HERE)
EXAMPLES = os.path.join(ADAPTER_DIR, "examples")
REPO_SCHEMA = os.path.join(ADAPTER_DIR, "..", "..", "schema", "resultset.json")

FRONTIEROR_ROOT = os.environ.get("FRONTIEROR_ROOT")
HAS_UPSTREAM = bool(
    FRONTIEROR_ROOT
    and os.path.isfile(os.path.join(FRONTIEROR_ROOT, "test_time_self_evolution", "scoring", "staged_qte.py"))
    and sys.version_info >= (3, 11)  # trusted_eval_infra.contracts uses enum.StrEnum
)
requires_upstream = pytest.mark.skipif(
    not HAS_UPSTREAM, reason="FRONTIEROR_ROOT not set to a FrontierOR checkout (or Python < 3.11)"
)
requires_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("yaml") is None,
    reason="compute_benchmark_main_metrics.py needs pandas and PyYAML",
)
requires_jsonschema = pytest.mark.skipif(
    importlib.util.find_spec("jsonschema") is None or not os.path.isfile(REPO_SCHEMA),
    reason="jsonschema not installed or repo schema not found",
)

# Synthetic reference rows in the shape of metadata/gurobi_references.csv.gz.
# (The real-data checks are the example-export tests further down.)
REFS = [
    {"task_id": "pmin", "instance": "tiny", "objective_value": "100.0", "runtime": "10.0", "feasible": "True", "status": "2", "time_limit": "300"},
    {"task_id": "pmin", "instance": "large_1", "objective_value": "100.0", "runtime": "1000.0", "feasible": "True", "status": "2", "time_limit": "3600"},
    {"task_id": "pmin", "instance": "large_2", "objective_value": "100.0", "runtime": "1000.0", "feasible": "True", "status": "2", "time_limit": "3600"},
    {"task_id": "pmin", "instance": "large_3", "objective_value": "100.0", "runtime": "3601.2", "feasible": "True", "status": "9", "time_limit": "3600"},
    {"task_id": "pmax", "instance": "tiny", "objective_value": "50.0", "runtime": "5.0", "feasible": "True", "status": "2", "time_limit": "300"},
    {"task_id": "pmax", "instance": "large_1", "objective_value": "50.0", "runtime": "400.0", "feasible": "True", "status": "2", "time_limit": "3600"},
    {"task_id": "pzero", "instance": "large_1", "objective_value": "0.0", "runtime": "20.0", "feasible": "True", "status": "2", "time_limit": "3600"},
    {"task_id": "earl2005", "instance": "large_1", "objective_value": "0.0", "runtime": "20.0", "feasible": "True", "status": "2", "time_limit": "3600"},
]
META = [
    {"paper_id": "pmin", "direction": "min"},
    {"paper_id": "pmax", "direction": "max"},
    {"paper_id": "pzero", "direction": "min"},
    {"paper_id": "earl2005", "direction": "min"},
]


def csv_row(paper_id, instance, *, obj, time, feasible=True, status="pass", fail_reason="", error="", gap=None, model="m1"):
    """A row as csv.DictReader returns it from a one-shot results CSV."""
    return {
        "paper_id": paper_id, "model": model, "instance": instance,
        "status": status, "fail_reason": fail_reason, "error": error,
        "gap": "" if gap is None else repr(gap), "delta_time": "",
        "feasible": "" if feasible is None else str(feasible),
        "debug_retries": "0", "correction_retries": "0",
        "obj": "" if obj is None else repr(obj), "time": "" if time is None else repr(time), "aocc": "",
        "first_status": status, "first_fail_reason": fail_reason, "first_gap": "",
        "first_feasible": "", "first_time": "", "first_obj": "",
    }


def convert(rows, **kw):
    kw.setdefault("references", REFS)
    kw.setdefault("paper_meta", META)
    kw.setdefault("run_id", "run-1")
    kw.setdefault("started_at", "2026-09-29T00:00:00Z")
    kw.setdefault("staged_qte_source", "builtin")
    rs = results_to_openeval(rows, **kw)
    v = validate_result_set(rs)
    assert v.valid, v.errors
    return rs


def by_id(rs):
    return {r["test_case_id"]: r for r in rs["results"]}


def grader(result, gid):
    return next(g for g in result["grader_results"] if g["grader_id"] == gid)


def staged(result):
    return grader(result, GRADER_STAGED_QTE)["metadata"]["frontieror"]


# ---------------------------------------------------------------- scoring

def test_signed_gap_directions_and_scale():
    assert signed_gap(105.0, 100.0, "min") == pytest.approx(0.05)
    assert signed_gap(95.0, 100.0, "min") == pytest.approx(-0.05)
    assert signed_gap(45.0, 50.0, "max") == pytest.approx(0.1)
    assert signed_gap(55.0, 50.0, "max") == pytest.approx(-0.1)
    # near-zero reference: D = max(|c|, 0.001)
    assert signed_gap(0.5, 0.0, "min") == pytest.approx(1.0)
    assert signed_gap(0.0005, 0.0, "min") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        signed_gap(1.0, 1.0, "minimize")


def test_quality_only_is_clamped_on_both_sides():
    assert quality_only_score(0.05) == 0.95
    assert quality_only_score(1.7) == 0.0      # lower clamp: g > 1
    assert quality_only_score(-0.3) == 1.0     # upper clamp: g < 0 (beats reference)
    assert quality_only_score(-1e9) == 1.0


def test_stage_1_gap_above_boundary():
    rs = convert([csv_row("pmin", "large_1", obj=105.0, time=50.0, gap=0.05)])
    r = by_id(rs)["pmin:large_1"]
    q = grader(r, GRADER_QUALITY_ONLY)
    assert q["score"] == 0.95 and q["passed"] is False
    s = staged(r)
    assert s["stage_id"] == 1 and s["staged_qte"] == 0.95 and s["speed_part"] == 0.0
    assert grader(r, GRADER_STAGED_QTE)["score"] is None
    assert r["passed"] is False  # binary QTE needs gap <= 1%


def test_stage_2_quality_plus_speed():
    rs = convert([csv_row("pmin", "large_1", obj=100.5, time=500.0, gap=0.005)])
    r = by_id(rs)["pmin:large_1"]
    s = staged(r)
    assert s["stage_id"] == 2
    assert s["quality_part"] == 0.995 and s["speed_part"] == 0.5
    assert s["staged_qte"] == 1.495           # > 1: kept as-is, never clamped
    assert grader(r, GRADER_STAGED_QTE)["score"] is None
    assert grader(r, GRADER_STAGED_QTE)["passed"] is True
    assert grader(r, GRADER_QUALITY_ONLY)["score"] == 0.995
    assert r["passed"] is True and grader(r, GRADER_BINARY_QTE)["score"] == 1.0


def test_negative_gap_clamps_quality_only_to_1_but_not_staged_qte():
    rs = convert([
        csv_row("pmin", "large_1", obj=90.0, time=100.0, gap=-0.1),
        csv_row("pmax", "large_1", obj=60.0, time=40.0, gap=-0.2),
    ])
    rmin, rmax = by_id(rs)["pmin:large_1"], by_id(rs)["pmax:large_1"]
    q = grader(rmin, GRADER_QUALITY_ONLY)
    assert q["score"] == 1.0
    assert q["metadata"]["frontieror"]["signed_gap"] == -0.1
    assert q["metadata"]["frontieror"]["clamped_from"] == 1.1
    s = staged(rmin)
    assert s["stage_id"] == 2 and s["staged_qte"] == pytest.approx(1.1 + 0.9)
    assert s["beat_gurobi"] is True and s["beat_amount"] == 0.1
    assert grader(rmax, GRADER_QUALITY_ONLY)["score"] == 1.0
    assert staged(rmax)["staged_qte"] == pytest.approx(1.2 + 0.9)


def test_infeasible_keeps_null_quality_and_reason():
    # Real pattern from the example run: an infeasible solution whose
    # (invalid) objective "beats" the reference by 92%.
    rs = convert([csv_row("pmin", "tiny", obj=7.26, time=0.1, feasible=False, status="fail",
                          fail_reason="infeasible", error="Solution is INFEASIBLE Violations: x", gap=-0.9274)])
    r = by_id(rs)["pmin:tiny"]
    q = grader(r, GRADER_QUALITY_ONLY)
    assert q["score"] is None and q["passed"] is False
    assert "not feasible" in q["reason"]
    s = staged(r)
    assert s["staged_qte"] == 0.0 and s["stage_id"] == 0 and s["reason"] == "infeasible"
    assert r["passed"] is False
    meta = r["metadata"]["frontieror"]
    assert meta["fail_reason"] == "infeasible" and meta["error_message"].startswith("Solution is INFEASIBLE")
    assert meta["candidate_objective"] == 7.26  # recorded, but never scored


def test_gate_fail_and_missing_instances():
    rows = [
        csv_row("pmin", "tiny", obj=150.0, time=1.0, feasible=False, status="fail", fail_reason="infeasible", gap=0.5),
        csv_row("pmin", "large_1", obj=None, time=None, feasible=None, status="gate_fail",
                fail_reason="infeasible", error="Skipped: tiny-instance gate failed on instance tiny"),
    ]
    rs = convert(rows, declared_instances={"pmin": ["tiny", "large_1", "large_2"]})
    res = by_id(rs)
    assert set(res) == {"pmin:tiny", "pmin:large_1", "pmin:large_2"}
    gate = res["pmin:large_1"]
    assert "gate failed" in grader(gate, GRADER_QUALITY_ONLY)["reason"]
    assert grader(gate, GRADER_BINARY_QTE)["metadata"]["frontieror"]["excluded_by_tiny_gate"] is True
    missing = res["pmin:large_2"]
    assert missing["metadata"]["frontieror"]["missing_result"] is True
    assert staged(missing)["reason"] == "missing_result" and staged(missing)["staged_qte"] == 0.0
    assert grader(missing, GRADER_QUALITY_ONLY)["score"] is None
    assert not any(r["passed"] for r in rs["results"])
    agg = rs["metadata"]["frontieror"]["aggregates"]
    assert agg["staged_qte_mean"] == 0.0 and agg["declared_instances_complete"] is True
    assert agg["quality_only_null_count"] == 3
    assert "avg_score" not in rs["summary"]


def test_summary_vs_contract_aggregate():
    rows = [
        csv_row("pmin", "tiny", obj=100.0, time=5.0, gap=0.0),                       # stage 2
        csv_row("pmin", "large_1", obj=110.0, time=100.0, gap=0.1),                  # stage 1
        csv_row("pmin", "large_2", obj=1.0, time=1.0, feasible=False, status="fail", fail_reason="infeasible", gap=-0.99),
    ]
    rs = convert(rows)
    scores = {r["test_case_id"]: staged(r)["staged_qte"] for r in rs["results"]}
    agg = rs["metadata"]["frontieror"]["aggregates"]
    assert agg["staged_qte_mean"] == pytest.approx(round(sum(scores.values()) / 3, 6))
    # EvalPort avg_score = mean of non-null quality-only scores; the
    # zero-filled mean is the contract-style number.
    assert rs["summary"]["avg_score"] == pytest.approx((1.0 + 0.9) / 2)
    assert agg["quality_only_mean_null_as_zero"] == pytest.approx((1.0 + 0.9) / 3, abs=1e-6)
    assert rs["summary"]["total"] == 3 and rs["summary"]["passed"] == 1
    assert agg["binary_qte_large_cells"] == 2 and agg["binary_qte_large_passed"] == 0


def test_binary_qte_time_rules():
    rows = [
        csv_row("pmin", "large_1", obj=100.0, time=1000.9, gap=0.0),   # within 0.001*1000 = 1.0s tolerance
        csv_row("pmin", "large_2", obj=100.0, time=1002.0, gap=0.0),   # too slow
        csv_row("pmin", "large_3", obj=100.0, time=3700.0, gap=0.0),   # both capped at 3600
    ]
    res = by_id(convert(rows))
    assert res["pmin:large_1"]["passed"] is True
    assert res["pmin:large_2"]["passed"] is False
    assert res["pmin:large_3"]["passed"] is True


def test_binary_qte_zero_time_never_fast_enough():
    ok, meta = A.binary_qte(paper_id="p", instance="large_1", feasible=True, candidate_objective=1.0,
                            candidate_time=0.0, gap=0.0, reference_runtime=10.0, reference_time_limit=3600.0)
    assert ok is False and meta["fast_enough"] is False


def test_proven_zero_reference_uses_absolute_tolerance():
    rows = [csv_row("earl2005", "large_1", obj=5e-7, time=1.0, gap=None)]
    r = by_id(convert(rows))["earl2005:large_1"]
    b = grader(r, GRADER_BINARY_QTE)["metadata"]["frontieror"]
    assert b["proven_zero_optimum_tolerance"] == 1e-6 and b["quality_ok"] is True
    assert r["passed"] is True


def test_near_zero_reference_quality_only():
    r = by_id(convert([csv_row("pzero", "large_1", obj=0.5, time=1.0, gap=None)]))["pzero:large_1"]
    assert grader(r, GRADER_QUALITY_ONLY)["score"] == 0.0  # g = 0.5/0.5 = 1
    assert staged(r)["stage_id"] == 1


def test_timeout_maps_to_result_error():
    rows = [csv_row("pmin", "large_1", obj=None, time=3600.0, feasible=False, status="fail",
                    fail_reason="timeout", error="Process timed out after 3600s")]
    r = by_id(convert(rows))["pmin:large_1"]
    assert r["error"]["type"] == "timeout"


# ---------------------------------------------------------------- visibility

def test_non_public_instance_is_refused_by_default():
    rows = [csv_row("pmin", "large_1", obj=100.0, time=1.0, gap=0.0),
            csv_row("pmin", "final_opaque_7", obj=42.0, time=1.0, gap=0.0)]
    with pytest.raises(NonPublicInstanceError):
        convert(rows)


def test_non_public_instance_omitted_without_leaking_ids():
    rows = [csv_row("pmin", "large_1", obj=100.0, time=1.0, gap=0.0),
            csv_row("hidden_paper", "final_opaque_7", obj=42.0, time=1.0, gap=0.0)]
    rs = convert(rows, on_non_public="omit")
    assert [r["test_case_id"] for r in rs["results"]] == ["pmin:large_1"]
    assert rs["metadata"]["frontieror"]["non_public_results_omitted"] == 1
    dumped = json.dumps(rs)
    assert "hidden_paper" not in dumped and "final_opaque_7" not in dumped


def test_references_required():
    with pytest.raises(ValueError, match="references is required"):
        results_to_openeval([csv_row("pmin", "tiny", obj=1.0, time=1.0)], references=None,
                            paper_meta=META, run_id="r", started_at="2026-09-29T00:00:00Z")


def test_suite_contains_no_reference_values():
    items = [{"paper_id": "pmin", "instance": "large_1", "problem_description": "Minimize cost."}]
    suite = to_openeval(items, paper_meta=META)
    dumped = json.dumps(suite)
    assert "objective_value" not in dumped and "reference_objective" not in dumped


# ---------------------------------------------------------------- input handling

def test_rejects_multiple_models_and_unknown_direction():
    with pytest.raises(ValueError, match="several models"):
        convert([csv_row("pmin", "tiny", obj=1.0, time=1.0, model="a"),
                 csv_row("pmin", "large_1", obj=1.0, time=1.0, model="b")])
    with pytest.raises(ValueError, match="direction"):
        convert([csv_row("pmin", "tiny", obj=1.0, time=1.0)], paper_meta=[])


def test_precomputed_augmented_results_are_carried_unchanged():
    # Shape written by eval_modes.augment_results_with_staged_qte().
    results = {
        "large_1": {
            "feasible": True, "llm_obj": 90.0, "gurobi_obj": 100.0, "solve_time": 100.0,
            "candidate_time_limit": 3600, "gurobi_time_limit": 3600.0,
            "score": 2.0, "stage_id": 2.0, "quality_part": 1.1, "speed_part": 0.9,
            "signed_gap": -0.1, "beat_amount": 0.1, "beat_gurobi_flag": 1.0, "matched_flag": 0.0,
        },
        "large_2": {
            "feasible": False, "llm_obj": None, "gurobi_obj": 100.0, "solve_time": None,
            "score": 0.0, "stage_id": 0.0, "quality_part": 0.0, "speed_part": 0.0,
            "signed_gap": 1.0, "beat_amount": 0.0, "beat_gurobi_flag": 0.0, "matched_flag": 0.0,
        },
        "large_3": None,  # skipped by the pipeline
    }
    rows = A.augmented_results_to_rows(results, paper_id="pmin", model="m1")
    rs = convert(rows, staged_qte_source="auto")
    res = by_id(rs)
    s1 = staged(res["pmin:large_1"])
    assert s1["staged_qte"] == 2.0 and s1["scorer_source"].startswith("precomputed")
    assert s1["beat_gurobi"] is True
    assert grader(res["pmin:large_1"], GRADER_QUALITY_ONLY)["score"] == 1.0
    s2 = staged(res["pmin:large_2"])
    assert s2["stage_id"] == 0 and "sentinel" in s2["signed_gap_note"]
    assert grader(res["pmin:large_2"], GRADER_QUALITY_ONLY)["score"] is None  # not 1 - 1.0 = 0.0
    assert "pmin:large_3" not in res


def test_row_reference_mismatch_is_an_error():
    rows = A.augmented_results_to_rows(
        {"large_1": {"feasible": True, "llm_obj": 100.0, "gurobi_obj": 123.0, "solve_time": 1.0}},
        paper_id="pmin")
    with pytest.raises(ValueError, match="differs from the public"):
        convert(rows)


@requires_jsonschema
def test_output_matches_json_schema():
    import jsonschema
    schema = json.load(open(REPO_SCHEMA))
    rs = convert([csv_row("pmin", "large_1", obj=100.5, time=500.0, gap=0.005),
                  csv_row("pmin", "tiny", obj=None, time=None, feasible=False, status="fail")])
    jsonschema.Draft202012Validator(schema).validate(rs)


# ---------------------------------------------------------------- suite

def test_to_openeval_and_round_trip():
    items = [
        {"paper_id": "pmin", "instance": "tiny", "problem_description": "Minimize cost.",
         "instance_path": "tasks/pmin/instance/tiny_instance.json"},
        {"paper_id": "pmin", "instance": "large_1", "problem_description": "Minimize cost.",
         "mathematical_formulation": "min c^T x", "instance_path": "tasks/pmin/instance/large_instance_1.json"},
    ]
    suite = to_openeval(items, paper_meta=META, suite_id="frontieror_demo")
    v = validate_suite(suite)
    assert v.valid, v.errors
    assert [tc["id"] for tc in suite["test_cases"]] == ["pmin:tiny", "pmin:large_1"]
    assert {g["id"] for g in suite["graders"]} == {GRADER_QUALITY_ONLY, GRADER_STAGED_QTE, GRADER_BINARY_QTE}
    assert all(g["params"]["handler"].startswith("frontieror:") for g in suite["graders"])
    assert suite["test_cases"][0]["metadata"]["frontieror"]["direction"] == "min"
    assert from_openeval(suite) == items
    with pytest.raises(ValueError):
        to_openeval([{"paper_id": "p", "instance": "tiny", "problem_description": "  "}])


def test_load_dataset_instances(tmp_path):
    task = tmp_path / "tasks" / "pmin"
    (task / "instance").mkdir(parents=True)
    (task / "problem_description.txt").write_text("Minimize cost.")
    (task / "mathematical_formulation.md").write_text("min c^T x")
    (task / "instance" / "tiny_instance.json").write_text("{}")
    items = A.load_dataset_instances(str(tmp_path), ["pmin"])
    assert [i["instance"] for i in items] == ["tiny"]
    assert items[0]["instance_path"] == "tasks/pmin/instance/tiny_instance.json"
    assert items[0]["mathematical_formulation_path"] == "tasks/pmin/mathematical_formulation.md"
    assert "mathematical_formulation" not in items[0]
    with_text = A.load_dataset_instances(str(tmp_path), ["pmin"], include_formulation=True)
    assert with_text[0]["mathematical_formulation"] == "min c^T x"
    suite = to_openeval(with_text)
    assert validate_suite(suite).valid
    assert from_openeval(suite) == with_text


# ---------------------------------------------------------------- real example export

EXAMPLE_CSVS = [
    os.path.join(EXAMPLES, "data", "eval_results_quickstart.csv"),
    os.path.join(EXAMPLES, "data", "eval_results_quickstart_gurobi_tiny.csv"),
]


def _example_rows():
    rows = []
    for path in EXAMPLE_CSVS:
        rows.extend(A.read_results_csv(path))
    return rows


def test_example_export_is_reproducible_and_valid():
    """The shipped example ResultSets regenerate byte-for-byte from the
    shipped real run CSVs with the built-in scorer, and validate."""
    spec = importlib.util.spec_from_file_location("export_example", os.path.join(EXAMPLES, "export_example.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    regenerated = mod.build_result_sets(staged_qte_source="builtin")
    out_dir = os.path.join(EXAMPLES, "output")
    shipped = sorted(f for f in os.listdir(out_dir) if f.startswith("resultset_"))
    assert shipped and sorted(regenerated) == shipped
    for fname, rs in regenerated.items():
        assert validate_result_set(rs).valid
        with open(os.path.join(out_dir, fname)) as fh:
            on_disk = json.load(fh)
        # scorer_source differs only when the shipped file was produced with
        # the upstream scorer; the numbers must not.
        for a, b in zip(rs["results"], on_disk["results"]):
            sa, sb = staged(a), staged(b)
            for k in ("staged_qte", "stage_id", "quality_part", "speed_part", "signed_gap", "beat_amount", "matched", "beat_gurobi"):
                assert sa[k] == sb[k], (fname, a["test_case_id"], k)
            assert a["passed"] == b["passed"]
            assert grader(a, GRADER_QUALITY_ONLY)["score"] == grader(b, GRADER_QUALITY_ONLY)["score"]
    with open(os.path.join(out_dir, "suite.json")) as fh:
        suite = json.load(fh)
    assert validate_suite(suite).valid
    ids = {tc["id"] for tc in suite["test_cases"]}
    for fname, rs in regenerated.items():
        assert {r["test_case_id"] for r in rs["results"]} <= ids


# ---------------------------------------------------------------- upstream parity

def _upstream():
    if FRONTIEROR_ROOT not in sys.path:
        sys.path.insert(0, FRONTIEROR_ROOT)
    from test_time_self_evolution.scoring.base import ScoreContext
    from test_time_self_evolution.scoring.staged_qte import StagedQteScorer, signed_quality_gap
    return StagedQteScorer, ScoreContext, signed_quality_gap


PARITY_CASES = [
    # (direction, c, r, t, tau, tau_limit, time_limit, feasible)
    ("min", 105.0, 100.0, 50.0, 1000.0, 3600.0, 3600, True),
    ("min", 100.5, 100.0, 500.0, 1000.0, 3600.0, 3600, True),
    ("min", 90.0, 100.0, 100.0, 1000.0, 3600.0, 3600, True),
    ("max", 60.0, 50.0, 40.0, 400.0, 3600.0, 3600, True),
    ("max", 45.0, 50.0, 40.0, 400.0, 3600.0, 3600, True),
    ("min", 0.5, 0.0, 1.0, 20.0, 3600.0, 3600, True),
    ("min", 0.0004, 0.0002, 1.0, 20.0, 3600.0, 3600, True),
    ("min", 100.0, 100.0, 3700.0, 3601.2, 3600.0, 3600, True),
    ("min", 100.0, 100.0, None, 1000.0, 3600.0, 300, True),
    ("min", 100.0, 100.0, 5.0, 0.0, 3600.0, 3600, True),
    ("min", 100.0, 100.0, 5.0, None, None, 3600, True),
    ("min", 350.0, 100.0, 5.0, 10.0, 300.0, 300, True),
    ("min", 100.0, 100.0, 5.0, 10.0, 300.0, 300, False),
    ("min", None, 100.0, 5.0, 10.0, 300.0, 300, True),
    ("min", 100.0, None, 5.0, 10.0, 300.0, 300, True),
]


@requires_upstream
@pytest.mark.parametrize("case", PARITY_CASES)
def test_builtin_staged_qte_matches_upstream(case):
    StagedQteScorer, ScoreContext, _ = _upstream()
    direction, c, r, t, tau, tau_limit, limit, feasible = case
    result = {"feasible": feasible, "llm_obj": c, "solve_time": t}
    ctx = ScoreContext(time_limit=limit, gurobi_time=tau, gurobi_time_limit=tau_limit, gurobi_obj=r, direction=direction)
    expected = StagedQteScorer().score_instance(result, ctx)
    got = staged_qte_builtin(result, time_limit=limit, gurobi_obj=r, gurobi_time=tau,
                             gurobi_time_limit=tau_limit, direction=direction)
    assert got == expected


@requires_upstream
def test_signed_gap_matches_upstream():
    _, _, signed_quality_gap = _upstream()
    for d, c, r, *_ in PARITY_CASES:
        if c is not None and r is not None:
            assert signed_gap(c, r, d) == signed_quality_gap(c, r, d)


@requires_upstream
def test_contract_constants_match_upstream():
    if FRONTIEROR_ROOT not in sys.path:
        sys.path.insert(0, FRONTIEROR_ROOT)
    from trusted_eval_infra import contracts
    assert contracts.SCORING_CONTRACT_VERSION == A.SCORING_CONTRACT_VERSION
    assert contracts.CONTRACT_SCHEMA_VERSION == A.CONTRACT_SCHEMA_VERSION
    assert contracts.NEAR_ZERO_REFERENCE == A.NEAR_ZERO_REFERENCE
    assert contracts.INSTANCE_SCORE_DECIMALS == A.INSTANCE_SCORE_DECIMALS
    pub = contracts.public_scoring_contract()
    assert pub["instance_score"]["stage_boundary"] == A.STAGE_BOUNDARY
    assert pub["aggregation_details"]["missing_or_failed_instance_score"] == 0.0
    vis = contracts.visibility_contract()["artifacts"]
    assert vis["reference_objective"] == "trusted_only" and vis["reference_runtime"] == "trusted_only"


@requires_upstream
def test_upstream_scorer_used_when_available():
    rs = convert([csv_row("pmin", "large_1", obj=100.5, time=500.0, gap=0.005)],
                 staged_qte_source="upstream", frontieror_root=FRONTIEROR_ROOT)
    s = staged(rs["results"][0])
    assert s["scorer_source"] == "upstream StagedQteScorer" and s["staged_qte"] == 1.495
    assert rs["metadata"]["frontieror"]["upstream_contract_version_checked"] == "staged-qte-v1"


def _metrics_module():
    path = os.path.join(FRONTIEROR_ROOT, "scripts", "compute_benchmark_main_metrics.py")
    spec = importlib.util.spec_from_file_location("frontieror_benchmark_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requires_upstream
@requires_pandas
def test_proven_zero_table_matches_upstream():
    assert _metrics_module().PROVEN_ZERO_OPTIMUM_TOLERANCE == A.PROVEN_ZERO_OPTIMUM_TOLERANCE


@requires_upstream
@requires_pandas
def test_binary_qte_matches_compute_metrics(tmp_path):
    """Per-cell beat_gurobi_1 parity: one paper per cell, so the paper's
    beat_gurobi_1 (mean over 5 large cells) is exactly our indicator / 5."""
    import pandas as pd
    metrics = _metrics_module()
    cells = [
        # (paper, obj, time, feasible, gurobi_obj, gurobi_time, gurobi_limit)
        ("c1", 100.0, 500.0, True, 100.0, 1000.0, 3600),
        ("c2", 100.5, 500.0, True, 100.0, 1000.0, 3600),
        ("c3", 102.0, 500.0, True, 100.0, 1000.0, 3600),
        ("c4", 100.0, 1000.9, True, 100.0, 1000.0, 3600),
        ("c5", 100.0, 1002.0, True, 100.0, 1000.0, 3600),
        ("c6", 100.0, 3700.0, True, 100.0, 3601.2, 3600),
        ("c7", 90.0, 1.0, False, 100.0, 1000.0, 3600),
        ("c8", 100.0, 0.0, True, 100.0, 1000.0, 3600),
        ("c9", 49.0, 10.0, True, 50.0, 400.0, 3600),
        ("earl2005", 5e-7, 1.0, True, 0.0, 20.0, 3600),
        ("earl2005x", 5e-7, 1.0, True, 0.0, 20.0, 3600),
    ]
    rows, refs, meta = [], [], []
    for paper, obj, t, feas, gobj, gt, gl in cells:
        direction = "max" if paper == "c9" else "min"
        gap = A._oneshot_gap(obj, gobj, direction)
        rows.append(csv_row(paper, "tiny", obj=gobj, time=1.0, gap=0.0))
        rows.append(csv_row(paper, "large_1", obj=obj, time=t, feasible=feas,
                            status="pass" if feas else "fail", gap=None if gap is None else round(gap, 6)))
        refs.append({"task_id": paper, "instance": "tiny", "objective_value": gobj, "runtime": 1.0, "time_limit": 300, "feasible": True})
        refs.append({"task_id": paper, "instance": "large_1", "objective_value": gobj, "runtime": gt, "time_limit": gl, "feasible": True})
        meta.append({"paper_id": paper, "direction": direction})
    csv_path = tmp_path / "model.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gurobi = pd.DataFrame([
        {"paper_id": r["task_id"], "instance": r["instance"], "gurobi_solution": r["objective_value"],
         "gurobi_time": r["runtime"], "gurobi_time_limit": r["time_limit"]}
        for r in refs if r["instance"] != "tiny"
    ])
    for paper, *_ in cells:
        expected = metrics.compute_metrics(csv_path, {paper}, gurobi)["beat_gurobi_1"] * 5
        paper_rows = [r for r in rows if r["paper_id"] == paper]
        rs = results_to_openeval(paper_rows, references=[r for r in refs if r["task_id"] == paper],
                                 paper_meta=meta, run_id="p", started_at="2026-09-29T00:00:00Z",
                                 staged_qte_source="builtin")
        got = by_id(rs)[f"{paper}:large_1"]["passed"]
        assert got == bool(round(expected)), paper
