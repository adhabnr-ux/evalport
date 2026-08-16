from openeval.validate import validate_suite, validate_grader, validate_test_case, validate_result_set

def test_valid_suite():
    assert validate_suite({"version":"1.0.0","id":"s","graders":[{"id":"g1","type":"exact_match"}],"test_cases":[{"id":"tc1","input":"hi","graders":["g1"]}]}).valid

def test_empty():
    assert not validate_suite({"version":"1.0.0","id":"s","test_cases":[]}).valid

def test_grader():
    assert validate_grader({"id":"g1","type":"exact_match"}).valid
    assert not validate_grader({"id":"g1","type":"bad"}).valid

def test_tc():
    assert validate_test_case({"id":"tc1","input":"hi","graders":["g1"]}).valid
    assert not validate_test_case({"id":"tc1","graders":["g1"]}).valid

def test_rs():
    assert validate_result_set({"version":"1.0.0","suite_id":"s","run_id":"r","started_at":"2026-01-01T00:00:00Z","results":[{"test_case_id":"tc1","passed":True,"grader_results":[{"grader_id":"g1","type":"exact_match","score":1.0,"passed":True}]}]}).valid

def test_semver_prerelease_versions_are_valid():
    # Previously only "X.Y.Z" or "X.Y.Z-draft" passed -- real prerelease/build
    # metadata per semver 2.0.0 (e.g. what this project's own README calls its
    # current spec version) was silently rejected as INVALID_VERSION.
    base = {"id":"s","graders":[{"id":"g1","type":"exact_match"}],"test_cases":[{"id":"tc1","input":"hi","graders":["g1"]}]}
    for v in ["1.0.0-rc.1", "1.1.0-beta.2", "2.0.0-alpha.1+build.5", "1.0.0", "1.0.0-draft"]:
        assert validate_suite({**base, "version": v}).valid, v

def test_garbage_version_still_rejected():
    base = {"id":"s","graders":[{"id":"g1","type":"exact_match"}],"test_cases":[{"id":"tc1","input":"hi","graders":["g1"]}]}
    r = validate_suite({**base, "version": "not-a-version"})
    assert not r.valid
    assert any(e["code"] == "INVALID_VERSION" for e in r.errors)

def test_non_standard_grader_type_requires_handler_like_custom():
    # A framework-specific type name (not one of the 11 well-known types) is
    # permitted per SPEC.md's "Custom Grader Types" section, but -- like
    # type: "custom" -- must carry params.handler so an unrecognizing runner
    # can skip it gracefully instead of guessing at its semantics.
    assert not validate_grader({"id": "g1", "type": "trulens_feedback"}).valid
    assert not validate_grader({"id": "g1", "type": "trulens_feedback", "params": {}}).valid
    assert validate_grader({"id": "g1", "type": "trulens_feedback", "params": {"handler": "trulens.feedback"}}).valid

def test_score_out_of_range_rejected():
    base_result = {"test_case_id":"tc1","passed":True,"grader_results":[{"grader_id":"g1","type":"exact_match","score":1.5,"passed":True}]}
    rs = {"version":"1.0.0","suite_id":"s","run_id":"r","started_at":"2026-01-01T00:00:00Z","results":[base_result]}
    r = validate_result_set(rs)
    assert not r.valid
    assert any(e["code"] == "OUT_OF_RANGE" for e in r.errors)

def test_null_score_still_valid_for_skipped_or_pending_graders():
    base_result = {"test_case_id":"tc1","passed":False,"grader_results":[{"grader_id":"g1","type":"human","score":None,"passed":False,"metadata":{"skip_reason":"pending_review"}}]}
    rs = {"version":"1.0.0","suite_id":"s","run_id":"r","started_at":"2026-01-01T00:00:00Z","results":[base_result]}
    assert validate_result_set(rs).valid

def test_bool_is_not_a_valid_score():
    # bool is a subclass of int in Python -- make sure True/False don't sneak
    # through the numeric-score check.
    base_result = {"test_case_id":"tc1","passed":True,"grader_results":[{"grader_id":"g1","type":"exact_match","score":True,"passed":True}]}
    rs = {"version":"1.0.0","suite_id":"s","run_id":"r","started_at":"2026-01-01T00:00:00Z","results":[base_result]}
    assert not validate_result_set(rs).valid
