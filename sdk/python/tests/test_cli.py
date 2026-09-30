"""Tests for the evalport-validate CLI (openeval/cli.py).

Uses the repo's real documents: examples/*.json (valid suites + a result
set) and spec/conformance/fixtures/*.json (each fixture wraps a `document`
with its declared `type` and `expect.valid`, which these tests unwrap into
standalone files -- exactly what a user's *.evalport.json would contain).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openeval import cli

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "examples"
FIXTURES = REPO / "spec" / "conformance" / "fixtures"
FIXTURE_PATHS = sorted(FIXTURES.glob("*.json"))
EXAMPLE_PATHS = sorted(EXAMPLES.glob("*.json"))


def _unwrap(fixture_path, dest_dir):
    fixture = json.loads(fixture_path.read_text())
    out = dest_dir / fixture_path.name
    out.write_text(json.dumps(fixture["document"], indent=2))
    return out, fixture["type"], fixture["expect"]["valid"]


def _allow_unknown(fixture_path):
    # PROPOSED (Discussion #108): fixtures may declare mode "allow_unknown".
    return json.loads(fixture_path.read_text()).get("mode", "strict") == "allow_unknown"


def _run(argv, capsys):
    code = cli.main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_fixture_sets_are_present():
    assert len(EXAMPLE_PATHS) >= 5
    assert len(FIXTURE_PATHS) >= 10


def test_all_examples_valid_exit_0(capsys):
    code, out, _ = _run(EXAMPLE_PATHS, capsys)
    assert code == 0, out
    assert f"{len(EXAMPLE_PATHS)} files checked: {len(EXAMPLE_PATHS)} valid, 0 invalid" in out
    assert "examples" in out and ": valid (suite)" in out
    assert "results.json: valid (resultset)" in out


@pytest.mark.parametrize("fixture_path", FIXTURE_PATHS, ids=lambda p: p.name)
def test_conformance_fixture_auto_detect_and_verdict(fixture_path, tmp_path):
    doc_path, declared_type, expected_valid = _unwrap(fixture_path, tmp_path)
    report = cli.check_file(str(doc_path), allow_unknown=_allow_unknown(fixture_path))
    assert report["type"] == declared_type
    assert report["valid"] is expected_valid, report["errors"]


@pytest.mark.parametrize("fixture_path", FIXTURE_PATHS, ids=lambda p: p.name)
def test_conformance_fixture_exit_code(fixture_path, tmp_path, capsys):
    doc_path, declared_type, expected_valid = _unwrap(fixture_path, tmp_path)
    flags = ["--allow-unknown"] if _allow_unknown(fixture_path) else []
    code, _, _ = _run(["--type", declared_type, *flags, doc_path], capsys)
    assert code == (0 if expected_valid else 1)


def test_invalid_document_text_output(tmp_path, capsys):
    doc_path, _, _ = _unwrap(FIXTURES / "score_out_of_range_rejected.json", tmp_path)
    code, out, _ = _run([EXAMPLES / "basic-suite.json", doc_path], capsys)
    assert code == 1
    assert f"{doc_path}: invalid (resultset), 1 error" in out
    assert "  $.results[0].grader_results[0].score: must be in [0,1] or null [OUT_OF_RANGE]" in out
    assert out.rstrip().endswith("2 files checked: 1 valid, 1 invalid")


def test_github_format_annotations(tmp_path, capsys):
    doc_path, _, _ = _unwrap(FIXTURES / "score_out_of_range_rejected.json", tmp_path)
    code, out, _ = _run(["--format", "github", doc_path], capsys)
    assert code == 1
    assert (
        f"::error file={doc_path},title=EvalPort::"
        "$.results[0].grader_results[0].score: must be in [0,1] or null [OUT_OF_RANGE]"
    ) in out.splitlines()


def test_github_format_escapes_properties_and_data(tmp_path, capsys):
    bad = tmp_path / "a,b:c.json"
    bad.write_text("{not json")
    code, out, _ = _run(["--format", "github", bad], capsys)
    assert code == 1
    line = next(l for l in out.splitlines() if l.startswith("::error"))
    # ',' and ':' inside a property value would otherwise end the property.
    assert "file=" + str(bad).replace(":", "%3A").replace(",", "%2C") + ",line=1,title=EvalPort::" in line
    assert "[INVALID_JSON]" in line


def test_github_format_valid_emits_no_annotations(capsys):
    code, out, _ = _run(["--format", "github", EXAMPLES / "basic-suite.json"], capsys)
    assert code == 0
    assert "::error" not in out


def test_json_format(tmp_path, capsys):
    doc_path, _, _ = _unwrap(FIXTURES / "group_self_parent_rejected.json", tmp_path)
    code, out, _ = _run(["--format", "json", EXAMPLES / "results.json", doc_path], capsys)
    assert code == 1
    data = json.loads(out)
    assert data["valid"] is False
    assert [f["valid"] for f in data["files"]] == [True, False]
    assert data["files"][1]["type"] == "resultset"
    assert any(e["code"] == "SELF_PARENT" for e in data["files"][1]["errors"])


def test_not_json_is_clear_error(tmp_path, capsys):
    bad = tmp_path / "broken.json"
    bad.write_text('{"version": "1.0.0",\n  "id": }')
    code, out, _ = _run([bad], capsys)
    assert code == 1
    assert "not valid JSON" in out and "(line 2," in out and "[INVALID_JSON]" in out


def test_non_utf8_is_clear_error(tmp_path, capsys):
    bad = tmp_path / "latin1.json"
    bad.write_bytes(b'{"id": "\xff"}')
    code, out, _ = _run([bad], capsys)
    assert code == 1
    assert "could not read file" in out and "[READ_ERROR]" in out


def test_missing_file_is_error(tmp_path, capsys):
    code, out, _ = _run([tmp_path / "missing.json"], capsys)
    assert code == 1
    assert "file not found [NOT_FOUND]" in out


def test_unmatched_glob_is_usage_error(tmp_path, capsys):
    code, out, err = _run([str(tmp_path / "**" / "*.evalport.json")], capsys)
    assert code == 2
    assert "no files matched" in err
    assert out == ""


def test_unknown_type_needs_explicit_type(tmp_path, capsys):
    f = tmp_path / "other.json"
    f.write_text('{"name": "package"}')
    code, out, _ = _run([f], capsys)
    assert code == 1
    assert "[UNKNOWN_TYPE]" in out


def test_explicit_type_overrides_detection(capsys):
    # A valid suite forced to be read as a result set fails.
    code, out, _ = _run(["--type", "resultset", EXAMPLES / "basic-suite.json"], capsys)
    assert code == 1
    assert "invalid (resultset)" in out


def test_directory_recursion_with_include(tmp_path, capsys):
    nested = tmp_path / "evals" / "deep"
    nested.mkdir(parents=True)
    (nested / "a.evalport.json").write_text((EXAMPLES / "basic-suite.json").read_text())
    (tmp_path / "evals" / "b.evalport.json").write_text((EXAMPLES / "results.json").read_text())
    (tmp_path / "evals" / "package.json").write_text('{"name": "not-evalport"}')
    hidden = tmp_path / "evals" / ".cache"
    hidden.mkdir()
    (hidden / "c.evalport.json").write_text("{}")

    code, out, _ = _run(["--include", "*.evalport.json", tmp_path / "evals"], capsys)
    assert code == 0, out
    assert "2 files checked: 2 valid, 0 invalid" in out
    assert "package.json" not in out and ".cache" not in out

    # Default --include is *.json, which picks up package.json too.
    code, out, _ = _run([tmp_path / "evals"], capsys)
    assert code == 1
    assert "package.json: invalid (unknown type)" in out


def test_recursive_glob_and_dedup(tmp_path, capsys):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "s.evalport.json").write_text((EXAMPLES / "basic-suite.json").read_text())
    pattern = str(tmp_path / "**" / "*.evalport.json")
    code, out, _ = _run([pattern, tmp_path / "x" / "s.evalport.json"], capsys)
    assert code == 0
    assert "1 file checked: 1 valid, 0 invalid" in out


@pytest.mark.parametrize("doc,expected", [
    ({"results": [], "suite_id": "s"}, "resultset"),
    ({"test_cases": [], "id": "s"}, "suite"),
    ({"test_cases_file": "tc.jsonl", "id": "s"}, "suite"),
    ({"id": "tc", "input": "hi", "graders": ["g"]}, "testcase"),
    ({"id": "g", "type": "exact_match"}, "grader"),
    ({"id": "g"}, None),
    ([1, 2], None),
])
def test_detect_type(doc, expected):
    assert cli.detect_type(doc) == expected


def test_module_entry_point_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "openeval.cli", "--format", "github", str(EXAMPLES / "safety.json")],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 file checked: 1 valid, 0 invalid" in proc.stdout


def test_console_script_declared():
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        pytest.skip("importlib.metadata unavailable")
    eps = entry_points()
    scripts = eps.select(group="console_scripts") if hasattr(eps, "select") else eps.get("console_scripts", [])
    matches = [ep for ep in scripts if ep.name == "evalport-validate"]
    if not matches:
        pytest.skip("evalport-sdk not installed (run `pip install -e sdk/python`)")
    assert matches[0].value == "openeval.cli:main"


# ---------------------------------------------------------------------------
# --allow-unknown (issue #107 / Discussion #108, PROPOSED -- DO NOT MERGE).
# ---------------------------------------------------------------------------

def test_unknown_field_rejected_by_default_and_accepted_with_allow_unknown(tmp_path, capsys):
    doc_path, _, _ = _unwrap(FIXTURES / "unknown_field_resultset_rejected.json", tmp_path)
    code, out, _ = _run([doc_path], capsys)
    assert code == 1
    assert f"{doc_path}: invalid (resultset), 3 errors" in out
    assert "  $.verdict: unknown field 'verdict' is not defined by the schema; put producer-specific data under metadata [UNKNOWN_FIELD]" in out
    assert "  $.results[0].extra_result_key: " in out
    assert "  $.results[0].grader_results[0].extra_gr_key: " in out
    code, out, _ = _run(["--allow-unknown", doc_path], capsys)
    assert code == 0, out
    assert f"{doc_path}: valid (resultset)" in out


def test_allow_unknown_still_reports_other_errors(tmp_path, capsys):
    doc_path, _, _ = _unwrap(FIXTURES / "unknown_field_allow_unknown_still_type_checks_rejected.json", tmp_path)
    code, out, _ = _run(["--allow-unknown", "--format", "json", doc_path], capsys)
    assert code == 1
    report = json.loads(out)["files"][0]
    assert [(e["path"], e["code"]) for e in report["errors"]] == [("$.results[0].actual_output", "TYPE_ERROR")]


def test_allow_unknown_flag_in_help():
    assert "--allow-unknown" in cli.build_parser().format_help()
