"""Unknown fields: strict validation by default, allow_unknown to opt out.

Issue #107 / Discussion #108 (PROPOSED -- reference implementation, DO NOT
MERGE). Unit tests for UNKNOWN_FIELD in openeval.validate; the agreement with
the raw JSON Schemas is checked in test_schema_consistency_unknown_fields.py.
"""
import pytest

from openeval.validate import (
    validate_document,
    validate_grader,
    validate_result_set,
    validate_suite,
    validate_test_case,
)

# ---------------------------------------------------------------------------
# Unknown fields (issue #107 / Discussion #108, PROPOSED -- DO NOT MERGE).
# Strict by default: every object the schemas close with
# additionalProperties: false rejects an undefined key with UNKNOWN_FIELD at
# that key's path. metadata / provider.extra / params / summary.by_grader
# entries stay open. allow_unknown=True turns only this check off.
# Mirrors sdk/typescript/tests/unknown-fields.test.ts test-for-test.
# ---------------------------------------------------------------------------

def _uf_rs():
    return {"version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
            "results": [{"test_case_id": "tc1", "passed": True,
                         "grader_results": [{"grader_id": "g1", "type": "exact_match", "score": 1.0, "passed": True}]}]}

def _uf_suite():
    return {"version": "1.0.0", "id": "s", "graders": [{"id": "g1", "type": "exact_match"}],
            "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}]}

def _unknown(r):
    return [(e["path"], e["code"]) for e in r.errors if e["code"] == "UNKNOWN_FIELD"]

# (name, mutate, expected UNKNOWN_FIELD path) -- one per closed ResultSet object level.
RESULTSET_UNKNOWN_CASES = [
    ("resultset root", lambda d: d.update(verdict="pass"), "$.verdict"),
    ("provider", lambda d: d.update(provider={"model": "m", "seed": 1}), "$.provider.seed"),
    ("runner", lambda d: d.update(runner={"name": "x", "commit": "abc"}), "$.runner.commit"),
    ("summary", lambda d: d.update(summary={"total": 1, "threshold": 0.8}), "$.summary.threshold"),
    ("group", lambda d: d.update(group={"group_id": "g", "gruop_id": "typo"}), "$.group.gruop_id"),
    ("result", lambda d: d["results"][0].update(extra_key=1), "$.results[0].extra_key"),
    ("result.error", lambda d: d["results"][0].update(passed=False, error={"type": "timeout", "stack": "..."}), "$.results[0].error.stack"),
    ("grader_result", lambda d: d["results"][0]["grader_results"][0].update(threshold=0.5), "$.results[0].grader_results[0].threshold"),
]

@pytest.mark.parametrize("name,mutate,path", RESULTSET_UNKNOWN_CASES, ids=[c[0] for c in RESULTSET_UNKNOWN_CASES])
def test_resultset_unknown_field_rejected_at_each_level(name, mutate, path):
    doc = _uf_rs(); mutate(doc)
    r = validate_result_set(doc)
    assert not r.valid
    assert _unknown(r) == [(path, "UNKNOWN_FIELD")], r.errors
    assert [e["code"] for e in r.errors] == ["UNKNOWN_FIELD"], r.errors

@pytest.mark.parametrize("name,mutate,path", RESULTSET_UNKNOWN_CASES, ids=[c[0] for c in RESULTSET_UNKNOWN_CASES])
def test_resultset_unknown_field_accepted_with_allow_unknown(name, mutate, path):
    doc = _uf_rs(); mutate(doc)
    r = validate_result_set(doc, allow_unknown=True)
    assert r.valid, r.errors

SUITE_UNKNOWN_CASES = [
    ("suite root (suite-level threshold, gauntlet#76)", lambda d: d.update(threshold=0.8), "$.threshold"),
    ("config", lambda d: d.update(config={"parallel": 2, "concurrency": 2}), "$.config.concurrency"),
    ("config.provider", lambda d: d.update(config={"provider": {"model": "m", "seed": 1}}), "$.config.provider.seed"),
    ("config.defaults", lambda d: d.update(config={"defaults": {"threshold": 0.5}}), "$.config.defaults.threshold"),
    ("config.retry", lambda d: d.update(config={"retry": {"max_attempts": 2, "jitter": True}}), "$.config.retry.jitter"),
    # Nested paths keep validate_suite's existing "$.test_cases[i]." + "$.<field>" prefixing.
    ("test case", lambda d: d["test_cases"][0].update(extra_tc_key=1), "$.test_cases[0].$.extra_tc_key"),
    ("test case provider", lambda d: d["test_cases"][0].update(provider={"model": "m", "seed": 1}), "$.test_cases[0].$.provider.seed"),
    ("shared grader (config instead of params, llama-cookbook#1072)", lambda d: d["graders"][0].update(config={"x": 1}), "$.graders[0].$.config"),
    ("inline grader in a test case", lambda d: d["test_cases"][0].update(graders=["g1", {"id": "g2", "type": "exact_match", "threshold": 1}]), "$.test_cases[0].$.graders[1].$.threshold"),
]

@pytest.mark.parametrize("name,mutate,path", SUITE_UNKNOWN_CASES, ids=[c[0] for c in SUITE_UNKNOWN_CASES])
def test_suite_unknown_field_rejected_at_each_level(name, mutate, path):
    doc = _uf_suite(); mutate(doc)
    r = validate_suite(doc)
    assert not r.valid
    assert _unknown(r) == [(path, "UNKNOWN_FIELD")], r.errors
    assert [e["code"] for e in r.errors] == ["UNKNOWN_FIELD"], r.errors

@pytest.mark.parametrize("name,mutate,path", SUITE_UNKNOWN_CASES, ids=[c[0] for c in SUITE_UNKNOWN_CASES])
def test_suite_unknown_field_accepted_with_allow_unknown(name, mutate, path):
    doc = _uf_suite(); mutate(doc)
    assert validate_suite(doc, allow_unknown=True).valid

def test_testcase_and_grader_unknown_field_rejected_standalone():
    r = validate_test_case({"id": "tc1", "input": "hi", "graders": ["g1"], "extra_tc_key": 1, "provider": {"seed": 1}})
    assert _unknown(r) == [("$.extra_tc_key", "UNKNOWN_FIELD"), ("$.provider.seed", "UNKNOWN_FIELD")]
    r = validate_grader({"id": "judge", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}"}, "config": {"temperature": 0}})
    assert [(e["path"], e["code"]) for e in r.errors] == [("$.config", "UNKNOWN_FIELD")]
    assert validate_test_case({"id": "tc1", "input": "hi", "graders": ["g1"], "extra_tc_key": 1}, allow_unknown=True).valid
    assert validate_grader({"id": "g", "type": "exact_match", "config": {}}, allow_unknown=True).valid

def test_issue_107_reproduction_rejected_strict_accepted_lenient():
    # The exact document shape from issue #107's reproduction.
    rs = _uf_rs()
    rs["verdict"] = "pass"
    rs["results"][0]["extra_result_key"] = 1
    rs["results"][0]["grader_results"][0]["extra_gr_key"] = 1
    r = validate_result_set(rs)
    assert _unknown(r) == [("$.verdict", "UNKNOWN_FIELD"), ("$.results[0].extra_result_key", "UNKNOWN_FIELD"),
                           ("$.results[0].grader_results[0].extra_gr_key", "UNKNOWN_FIELD")]
    assert validate_result_set(rs, allow_unknown=True).valid
    suite = _uf_suite()
    suite["threshold"] = 0.8
    suite["test_cases"][0]["extra_tc_key"] = 1
    r = validate_suite(suite)
    assert _unknown(r) == [("$.threshold", "UNKNOWN_FIELD"), ("$.test_cases[0].$.extra_tc_key", "UNKNOWN_FIELD")]
    assert validate_suite(suite, allow_unknown=True).valid

def test_metadata_and_other_open_objects_stay_open_in_strict_mode():
    rs = _uf_rs()
    rs["metadata"] = {"anything": {"nested": [1, 2]}, "com.example.x": True}
    rs["provider"] = {"model": "m", "extra": {"seed": 1, "top_p": 0.9}}
    rs["summary"] = {"total": 1, "by_grader": {"g1": {"passed": 1, "failed": 0, "p95_ms": 12}}}
    rs["results"][0]["metadata"] = {"trace_id": "abc", "verdict": "pass"}
    rs["results"][0]["grader_results"][0]["metadata"] = {"threshold": 0.5, "openeval": {"raw_score": 4}}
    assert validate_result_set(rs).valid, validate_result_set(rs).errors
    suite = _uf_suite()
    suite["metadata"] = {"threshold": 0.8}
    suite["config"] = {"provider": {"model": "m", "extra": {"seed": 1}}}
    suite["graders"][0]["params"] = {"ignore_case": True, "custom_knob": 1}
    suite["test_cases"][0].update(metadata={"k": 1}, params={"top_p": 1}, provider={"extra": {"a": 1}})
    assert validate_suite(suite).valid, validate_suite(suite).errors

def test_allow_unknown_only_disables_the_unknown_field_check():
    rs = _uf_rs()
    rs["verdict"] = "pass"
    rs["results"][0]["actual_output"] = ["a", "b"]  # still a TYPE_ERROR
    r = validate_result_set(rs, allow_unknown=True)
    assert [(e["path"], e["code"]) for e in r.errors] == [("$.results[0].actual_output", "TYPE_ERROR")]
    r = validate_result_set(rs)
    assert sorted(e["code"] for e in r.errors) == ["TYPE_ERROR", "UNKNOWN_FIELD"]

def test_unknown_field_reported_once_per_key_and_does_not_mask_other_errors():
    rs = _uf_rs()
    rs["results"][0].update(a=1, b=2)
    del rs["results"][0]["passed"]
    r = validate_result_set(rs)
    assert ("$.results[0].passed", "REQUIRED") in [(e["path"], e["code"]) for e in r.errors]
    assert _unknown(r) == [("$.results[0].a", "UNKNOWN_FIELD"), ("$.results[0].b", "UNKNOWN_FIELD")]

def test_validate_document_passes_allow_unknown_through():
    rs = _uf_rs(); rs["verdict"] = "pass"
    suite = _uf_suite(); suite["threshold"] = 1
    tc = {"id": "tc1", "input": "hi", "graders": ["g1"], "x": 1}
    g = {"id": "g1", "type": "exact_match", "x": 1}
    for doc, t in ((rs, "resultset"), (suite, "suite"), (tc, "testcase"), (g, "grader")):
        assert not validate_document(doc, t).valid
        assert validate_document(doc, t, allow_unknown=True).valid

def test_allow_unknown_defaults_to_strict_and_is_keyword_usable():
    rs = _uf_rs(); rs["verdict"] = "pass"
    assert not validate_result_set(rs).valid
    assert not validate_result_set(rs, allow_unknown=False).valid
    assert validate_result_set(rs, True).valid
