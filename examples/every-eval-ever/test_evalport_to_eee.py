"""Tests for the EvalPort -> Every Eval Ever (EEE) converter.

The record is validated against ``vendor/eval.schema.json`` (EEE schema 0.3.0,
unmodified, see vendor/README.md) with ``jsonschema``. If the official
``every-eval-ever`` package is installed (it needs Python >= 3.12) the record is
also run through EEE's own validator, including its semantic merge-gate checks;
that part is skipped otherwise.
"""
import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import uuid

import pytest
from jsonschema import Draft7Validator, FormatChecker
from referencing import Registry

from openeval.validate import validate_result_set, validate_suite

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from evalport_to_eee import (  # noqa: E402
    DEPLOYMENT_TYPES,
    EVALUATOR_RELATIONSHIPS,
    MODEL_AVAILABILITIES,
    ConversionError,
    EEEContext,
    main,
    to_eee,
)

SCRIPT = HERE / "evalport_to_eee.py"
SUITE = json.loads((ROOT / "examples" / "basic-suite.json").read_text())
RESULTS = json.loads((ROOT / "examples" / "results.json").read_text())
EEE_SCHEMA = json.loads((HERE / "vendor" / "eval.schema.json").read_text())

HAS_EEE = importlib.util.find_spec("every_eval_ever") is not None

CTX_ARGS = dict(
    model_id="openai/gpt-4o",
    deployment_type="externally_managed",
    model_availability="closed_weights",
    source_organization_name="Example Org",
    evaluator_relationship="third_party",
    retrieved_timestamp="1790000000.0",
)


def ctx(**overrides):
    return EEEContext(**{**CTX_ARGS, **overrides})


# Draft 7 per the schema's own "$schema". Every $ref in it is internal ("#/$defs/..."),
# asserted below, so an empty Registry is all that is needed; FormatChecker is enabled
# although the schema declares no "format" keyword today.
_VALIDATOR = Draft7Validator(EEE_SCHEMA, format_checker=FormatChecker(), registry=Registry())


def schema_errors(record):
    return sorted(f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in _VALIDATOR.iter_errors(record))


def assert_inputs_valid(suite, result_set):
    for name, result in (("suite", validate_suite(suite)), ("resultset", validate_result_set(result_set))):
        assert result.valid, f"{name} is not a valid EvalPort document: {result.errors}"


def convert(suite=None, result_set=None, **ctx_overrides):
    suite = SUITE if suite is None else suite
    result_set = RESULTS if result_set is None else result_set
    assert_inputs_valid(suite, result_set)
    record = to_eee(suite, result_set, ctx(**ctx_overrides))
    assert schema_errors(record) == []
    return record


def by_id(record):
    return {r["evaluation_result_id"]: r for r in record["evaluation_results"]}


def make_run(cases, graders=("g",), suite_id="s", types=None):
    """Build a (suite, resultset) pair. ``cases`` maps test_case_id -> {grader: (score, passed)}
    or a list of such dicts (repeated attempts); ``passed`` per case is the AND of non-null graders."""
    types = types or {}
    suite = {
        "version": "1.0.0",
        "id": suite_id,
        "graders": [{"id": g, "type": types.get(g, "exact_match")} for g in graders],
        "test_cases": [{"id": c, "input": "q", "graders": list(graders)} for c in cases],
    }
    results = []
    for case_id, attempts in cases.items():
        attempts = attempts if isinstance(attempts, list) else [attempts]
        for i, per_grader in enumerate(attempts, 1):
            scored = [p for s, p in per_grader.values() if s is not None]
            row = {
                "test_case_id": case_id,
                "grader_results": [
                    {"grader_id": g, "type": types.get(g, "exact_match"), "score": s, "passed": p}
                    for g, (s, p) in per_grader.items()
                ],
                "passed": bool(scored) and all(scored),
            }
            if len(attempts) > 1:
                row["attempt"] = i
            results.append(row)
    result_set = {
        "version": "1.0.0",
        "suite_id": suite_id,
        "run_id": "run_x",
        "started_at": "2026-01-15T10:00:00Z",
        "results": results,
    }
    return suite, result_set


# ---------------------------------------------------------------- the vendored schema


def test_vendored_schema_is_a_valid_draft7_schema():
    assert EEE_SCHEMA["$schema"] == "http://json-schema.org/draft-07/schema#"
    Draft7Validator.check_schema(EEE_SCHEMA)
    assert EEE_SCHEMA["version"] == "0.3.0"


def _refs(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref":
                yield v
            else:
                yield from _refs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _refs(v)


def test_schema_refs_are_all_internal():
    refs = list(_refs(EEE_SCHEMA))
    assert refs and all(r.startswith("#/") for r in refs)


def test_context_enums_match_the_vendored_schema():
    source_meta = EEE_SCHEMA["properties"]["source_metadata"]["properties"]
    assert tuple(source_meta["evaluator_relationship"]["enum"]) == EVALUATOR_RELATIONSHIPS
    axes = EEE_SCHEMA["$defs"]["model_info"]["properties"]["additional_details"]["properties"]
    assert tuple(axes["deployment_type"]["enum"]) == DEPLOYMENT_TYPES
    assert tuple(axes["model_availability"]["enum"]) == MODEL_AVAILABILITIES


# ---------------------------------------------------------------------- conversion


def test_basic_example_is_schema_valid_and_source_type_is_evaluation_run():
    record = convert()
    assert record["schema_version"] == "0.3.0"
    assert record["source_metadata"]["source_type"] == "evaluation_run"
    assert record["evaluation_id"] == "suite_qa_basic/openai/gpt-4o/1768471200"
    assert record["evaluation_timestamp"] == "1768471200"  # 2026-01-15T10:00:00Z
    assert record["eval_library"] == {"name": "evalport-cli", "version": "1.0.0"}
    assert record["model_info"]["additional_details"] == {
        "deployment_type": "externally_managed",
        "model_availability": "closed_weights",
    }


def test_basic_example_arithmetic():
    # Two test cases: tc_1 scored 1 (passed), tc_2 scored 0 (failed).
    results = by_id(convert())
    grader = results["suite_qa_basic/gr_exact/mean_score"]["score_details"]
    assert grader["score"] == 0.5
    assert grader["details"] == {
        "scored_count": "2",
        "unscored_count": "0",
        "passed_count": "1",
        "pass_rate_over_scored": "0.5",
    }
    # sample SD of [1, 0] is sqrt(0.5); SE = sqrt(0.5) / sqrt(2) = 0.5
    assert grader["uncertainty"]["standard_deviation"] == pytest.approx(0.5 ** 0.5)
    assert grader["uncertainty"]["standard_error"]["value"] == pytest.approx(0.5)
    assert grader["uncertainty"]["standard_error"]["method"] == "analytic"
    assert grader["uncertainty"]["num_samples"] == 2
    cases = results["suite_qa_basic/test_case_pass_rate"]["score_details"]
    assert cases["score"] == 0.5
    assert cases["details"]["scored_cases"] == "2"
    assert cases["details"]["passed_cases"] == "1"


def test_mean_score_and_standard_error_by_hand():
    scores = [0.9, 0.6, 0.3, 0.8]
    suite, rs = make_run({f"c{i}": {"g": (s, s >= 0.5)} for i, s in enumerate(scores)})
    grader = by_id(convert(suite, rs))["s/g/mean_score"]["score_details"]
    assert grader["score"] == pytest.approx(0.65)
    # deviations .25 -.05 -.35 .15 -> squares sum .21 -> /(4-1) = .07
    sd = 0.07 ** 0.5
    assert grader["uncertainty"]["standard_deviation"] == pytest.approx(sd)
    assert grader["uncertainty"]["standard_error"]["value"] == pytest.approx(sd / 2)
    assert grader["details"]["passed_count"] == "3"
    assert grader["details"]["pass_rate_over_scored"] == "0.75"


def test_metric_config_is_continuous_zero_to_one_higher_is_better():
    for result in convert()["evaluation_results"]:
        mc = result["metric_config"]
        assert (mc["score_type"], mc["min_score"], mc["max_score"], mc["lower_is_better"]) == ("continuous", 0.0, 1.0, False)
        assert mc["additional_details"]["metric_id_status"] == "unregistered"
        assert 0.0 <= result["score_details"]["score"] <= 1.0


def test_evaluation_id_is_stable_across_conversions():
    a = to_eee(SUITE, RESULTS, ctx(retrieved_timestamp=None))
    b = to_eee(SUITE, RESULTS, ctx(retrieved_timestamp=None))
    assert a["evaluation_id"] == b["evaluation_id"]
    assert a["evaluation_id"].endswith("/1768471200")


def test_provider_settings_become_generation_config():
    rs = copy.deepcopy(RESULTS)
    rs["provider"] = {"model": "gpt-4o", "temperature": 0.2, "max_tokens": 256,
                      "api_base": "https://internal.example/v1", "extra": {"seed": 7, "stop": ["x"]}}
    record = convert(result_set=rs)
    assert record["model_info"]["name"] == "gpt-4o"
    assert record["model_info"]["additional_details"]["evalport_provider_model"] == "gpt-4o"
    for result in record["evaluation_results"]:
        assert result["generation_config"]["generation_args"] == {"temperature": 0.2, "max_tokens": 256}
        assert result["generation_config"]["additional_details"] == {
            "provider_extra.seed": "7",
            "provider_extra.stop": '["x"]',
        }
    assert "internal.example" not in json.dumps(record)  # api_base is deliberately not exported


def test_optional_context_is_passed_through_and_dataset_variants_validate():
    record = convert(developer="openai", inference_platform="openai", source_organization_url="https://example.org",
                     model_name="GPT-4o", eval_library_name="lib", eval_library_version="9")
    assert record["model_info"]["developer"] == "openai"
    assert record["model_info"]["name"] == "GPT-4o"
    assert record["eval_library"] == {"name": "lib", "version": "9"}
    url = convert(dataset_urls=["https://example.org/data"])["evaluation_results"][0]["source_data"]
    assert (url["source_type"], url["url"]) == ("url", ["https://example.org/data"])
    hf = convert(hf_repo="org/data", hf_split="test")["evaluation_results"][0]["source_data"]
    assert (hf["source_type"], hf["hf_repo"], hf["hf_split"]) == ("hf_dataset", "org/data", "test")
    private = convert()["evaluation_results"][0]["source_data"]
    assert private["source_type"] == "other" and private["dataset_name"] == "Basic Q&A"


def test_eval_library_falls_back_to_unknown_without_runner():
    rs = {k: v for k, v in RESULTS.items() if k != "runner"}
    record = convert(result_set=rs)
    assert record["eval_library"] == {"name": "unknown", "version": "unknown"}
    assert record["source_metadata"]["source_name"] == "EvalPort"


# ------------------------------------------------------------------- null handling


def test_null_scores_are_excluded_from_the_mean_not_zero_filled():
    suite, rs = make_run({
        "c1": {"g": (1.0, True)},
        "c2": {"g": (None, False)},
        "c3": {"g": (0.0, False)},
        "c4": {"g": (None, False)},
    })
    assert_inputs_valid(suite, rs)
    result = by_id(convert(suite, rs))["s/g/mean_score"]["score_details"]
    assert result["score"] == 0.5  # (1 + 0) / 2; zero-filling would give 0.25
    assert result["score"] != 0.25
    assert result["details"]["scored_count"] == "2"
    assert result["details"]["unscored_count"] == "2"
    # Nulls are not scored failures: pass rate is 1 of 2 scored, not 1 of 4.
    assert result["details"]["pass_rate_over_scored"] == "0.5"
    assert result["uncertainty"]["num_samples"] == 2


def test_all_null_grader_gets_no_numeric_score_but_others_remain():
    suite, rs = make_run(
        {"c1": {"g": (1.0, True), "h": (None, False)}, "c2": {"g": (0.5, True), "h": (None, False)}},
        graders=("g", "h"),
        types={"h": "human"},
    )
    assert_inputs_valid(suite, rs)
    record = convert(suite, rs)
    ids = set(by_id(record))
    assert "s/g/mean_score" in ids and "s/h/mean_score" not in ids
    assert not any("human" in json.dumps(r["metric_config"]) for r in record["evaluation_results"])
    details = record["source_metadata"]["additional_details"]
    assert json.loads(details["graders_without_scored_results"]) == ["h"]
    assert details["evalport_unscored_grader_results"] == "2"


def test_case_with_only_null_scores_is_not_a_failed_case():
    suite, rs = make_run({
        "c1": {"g": (1.0, True)},
        "c2": {"g": (0.0, False)},
        "c3": {"g": (None, False)},  # not verified
    })
    cases = by_id(convert(suite, rs))["s/test_case_pass_rate"]["score_details"]
    assert cases["score"] == 0.5  # 1 of 2 cases with a verdict, not 1 of 3
    assert cases["details"]["scored_cases"] == "2"
    assert cases["details"]["unscored_cases"] == "1"


def test_nothing_scored_refuses_to_write_an_empty_record():
    suite, rs = make_run({"c1": {"g": (None, False)}, "c2": {"g": (None, False)}})
    assert_inputs_valid(suite, rs)
    with pytest.raises(ConversionError, match="no grader produced a non-null score"):
        to_eee(suite, rs, ctx())


def test_eee_schema_itself_has_no_null_score():
    """Why nulls cannot be carried: score_details.score must be a number."""
    record = convert()
    record["evaluation_results"][0]["score_details"]["score"] = None
    assert schema_errors(record)


# ------------------------------------------------------------------ uncertainty


def test_uncertainty_omitted_for_a_single_observation():
    suite, rs = make_run({"c1": {"g": (1.0, True)}})
    result = by_id(convert(suite, rs))["s/g/mean_score"]["score_details"]
    assert "uncertainty" not in result
    assert result["details"]["uncertainty_omitted"] == "fewer than 2 scored results"


def test_uncertainty_omitted_when_attempts_repeat_a_test_case():
    suite, rs = make_run({
        "c1": [{"g": (1.0, True)}, {"g": (0.0, False)}],
        "c2": [{"g": (1.0, True)}, {"g": (1.0, True)}],
    })
    assert_inputs_valid(suite, rs)
    results = by_id(convert(suite, rs))
    for key in ("s/g/mean_score", "s/test_case_pass_rate"):
        sd = results[key]["score_details"]
        assert "uncertainty" not in sd
        assert "repeated attempts" in sd["details"]["uncertainty_omitted"]
    assert results["s/g/mean_score"]["score_details"]["score"] == 0.75


# ------------------------------------------------------------ rejected input / context


def test_suite_id_mismatch_is_rejected():
    rs = copy.deepcopy(RESULTS)
    rs["suite_id"] = "some_other_suite"
    with pytest.raises(ConversionError, match="does not match Suite.id"):
        to_eee(SUITE, rs, ctx())


def test_results_for_unknown_test_cases_are_rejected():
    rs = copy.deepcopy(RESULTS)
    rs["results"][0]["test_case_id"] = "tc_not_in_suite"
    with pytest.raises(ConversionError, match="not in the suite"):
        to_eee(SUITE, rs, ctx())


@pytest.mark.parametrize("missing", ["model_id", "source_organization_name", "deployment_type",
                                     "model_availability", "evaluator_relationship"])
def test_missing_required_context_fails_instead_of_guessing(missing):
    with pytest.raises(ConversionError, match=missing):
        to_eee(SUITE, RESULTS, ctx(**{missing: ""}))


@pytest.mark.parametrize("field,value", [("deployment_type", "api"), ("model_availability", "open"),
                                         ("evaluator_relationship", "self")])
def test_context_values_outside_the_eee_enums_are_rejected(field, value):
    with pytest.raises(ConversionError, match=field):
        to_eee(SUITE, RESULTS, ctx(**{field: value}))


def test_naive_started_at_is_rejected():
    rs = copy.deepcopy(RESULTS)
    rs["started_at"] = "2026-01-15T10:00:00"
    with pytest.raises(ConversionError, match="no UTC offset"):
        to_eee(SUITE, rs, ctx())


@pytest.mark.parametrize("bad", [1.5, -0.1, True, "1"])
def test_out_of_range_scores_are_rejected(bad):
    rs = copy.deepcopy(RESULTS)
    rs["results"][0]["grader_results"][0]["score"] = bad
    with pytest.raises(ConversionError, match="score"):
        to_eee(SUITE, rs, ctx())


# ------------------------------------------------------------ negative controls
#
# Each mutation breaks a valid record in one way the EEE schema should reject. If the
# jsonschema check were a no-op (wrong schema, validator not actually run) these would
# not raise, so they prove the positive assertions above mean something.


def _drop(path):
    def mutate(record):
        node = record
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return mutate


def _set(path, value):
    def mutate(record):
        node = record
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return mutate


MUTATIONS = {
    "drop schema_version": _drop(["schema_version"]),
    "drop eval_library": _drop(["eval_library"]),
    "drop eval_library.version": _drop(["eval_library", "version"]),
    "drop model_info.additional_details": _drop(["model_info", "additional_details"]),
    "drop source_metadata.evaluator_relationship": _drop(["source_metadata", "evaluator_relationship"]),
    "drop evaluation_results": _drop(["evaluation_results"]),
    "drop result.source_data": _drop(["evaluation_results", 0, "source_data"]),
    "drop metric_config.lower_is_better": _drop(["evaluation_results", 0, "metric_config", "lower_is_better"]),
    "continuous without min_score": _drop(["evaluation_results", 0, "metric_config", "min_score"]),
    "drop score_details.score": _drop(["evaluation_results", 0, "score_details", "score"]),
    "bad source_type enum": _set(["source_metadata", "source_type"], "scrape"),
    "bad evaluator_relationship enum": _set(["source_metadata", "evaluator_relationship"], "self"),
    "bad score_type enum": _set(["evaluation_results", 0, "metric_config", "score_type"], "ordinal"),
    "bad deployment_type enum": _set(["model_info", "additional_details", "deployment_type"], "api"),
    "non-string additional_details value": _set(["source_metadata", "additional_details", "evalport_run_id"], 7),
    "null score": _set(["evaluation_results", 0, "score_details", "score"], None),
    "string score": _set(["evaluation_results", 0, "score_details", "score"], "0.5"),
    "unknown top-level property": _set(["grader_results"], []),
}


def test_positive_control_unmodified_record_is_valid():
    assert schema_errors(convert()) == []


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_negative_control_broken_record_fails_the_schema(name):
    record = copy.deepcopy(convert())
    MUTATIONS[name](record)
    assert schema_errors(record), f"schema accepted a record with: {name}"


# -------------------------------------------------------------------------- CLI

REQUIRED_CLI = [
    "--model-id", "openai/gpt-4o",
    "--deployment-type", "externally_managed",
    "--model-availability", "closed_weights",
    "--source-organization-name", "Example Org",
    "--evaluator-relationship", "third_party",
]
BASIC = [str(ROOT / "examples" / "basic-suite.json"), str(ROOT / "examples" / "results.json")]


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_cli_writes_a_schema_valid_record_to_stdout():
    proc = run_cli(*BASIC, *REQUIRED_CLI)
    assert proc.returncode == 0, proc.stderr
    record = json.loads(proc.stdout, parse_constant=lambda c: pytest.fail(f"non-strict JSON token {c}"))
    assert schema_errors(record) == []
    assert record["model_info"]["id"] == "openai/gpt-4o"
    assert len(record["evaluation_results"]) == 2


def test_cli_output_option_writes_a_file(tmp_path):
    out = tmp_path / "record.json"
    proc = run_cli(*BASIC, *REQUIRED_CLI, "-o", str(out))
    assert proc.returncode == 0 and proc.stdout == ""
    assert schema_errors(json.loads(out.read_text())) == []


def test_cli_fails_clearly_when_required_context_is_missing():
    proc = run_cli(*BASIC, "--model-id", "openai/gpt-4o")
    assert proc.returncode == 2
    for flag in ("--deployment-type", "--model-availability", "--source-organization-name", "--evaluator-relationship"):
        assert flag in proc.stderr
    assert proc.stdout == ""


def test_cli_rejects_a_mismatched_suite(tmp_path):
    other = copy.deepcopy(SUITE)
    other["id"] = "a_different_suite"
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(other))
    proc = run_cli(str(suite_path), BASIC[1], *REQUIRED_CLI)
    assert proc.returncode == 1
    assert "does not match Suite.id" in proc.stderr
    assert proc.stdout == ""


def test_cli_reports_unreadable_input(tmp_path):
    proc = run_cli(str(tmp_path / "missing.json"), BASIC[1], *REQUIRED_CLI)
    assert proc.returncode == 1 and "cannot read" in proc.stderr


def test_main_is_callable_in_process(capsys):
    assert main([*BASIC, *REQUIRED_CLI]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "0.3.0"


# -------------------------------------------------- EEE's own validator (optional)


@pytest.mark.skipif(not HAS_EEE, reason="every-eval-ever is not installed (it needs Python >= 3.12)")
class TestOfficialValidator:
    """EEE's validator: pydantic models plus the semantic checks its merge gate runs
    (score bounds, deployment axes, datastore path / model identity)."""

    @staticmethod
    def official_report(record, tmp_path):
        from every_eval_ever.helpers.io import datastore_repo_file_path
        from every_eval_ever.validator.validation_core import validate_aggregate

        repo_path = datastore_repo_file_path(
            "evalport-demo", record["model_info"]["id"], record["model_info"].get("developer"), f"{uuid.uuid4()}.json"
        )
        path = tmp_path / "record.json"
        path.write_text(json.dumps(record))
        return validate_aggregate(path, repo_path=repo_path, run_semantic_checks=True)

    def test_vendored_schema_is_the_installed_packages_schema(self):
        from every_eval_ever.schema import get_schema_version, schema_json

        if get_schema_version() != EEE_SCHEMA["version"]:
            pytest.skip(f"installed every-eval-ever schema is {get_schema_version()}, vendored is {EEE_SCHEMA['version']}")
        assert schema_json("eval.schema.json") == EEE_SCHEMA

    def test_basic_example_passes_with_no_errors_or_warnings(self, tmp_path):
        report = self.official_report(convert(), tmp_path)
        assert report.valid, report.errors
        assert report.errors == [] and report.warnings == []

    def test_null_heavy_run_passes(self, tmp_path):
        suite, rs = make_run(
            {"c1": {"g": (1.0, True), "h": (None, False)}, "c2": {"g": (None, False), "h": (None, False)},
             "c3": {"g": (0.25, False), "h": (None, False)}},
            graders=("g", "h"), types={"h": "human"},
        )
        report = self.official_report(convert(suite, rs), tmp_path)
        assert report.valid and report.warnings == [], (report.errors, report.warnings)

    def test_negative_control_official_validator_rejects_broken_records(self, tmp_path):
        for name in ("drop eval_library", "bad source_type enum", "drop model_info.additional_details"):
            record = copy.deepcopy(convert())
            MUTATIONS[name](record)
            assert not self.official_report(record, tmp_path).valid, name

    def test_negative_control_score_outside_declared_bounds_is_a_semantic_error(self, tmp_path):
        record = copy.deepcopy(convert())
        record["evaluation_results"][0]["score_details"]["score"] = 1.5  # schema-valid, outside [0, 1]
        assert schema_errors(record) == []
        report = self.official_report(record, tmp_path)
        assert not report.valid and any("outside" in e["msg"] for e in report.errors)
