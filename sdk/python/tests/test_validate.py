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

# --- Discussion #22 / issue #20: attempt + isolation ---

def _grader_result(score=1.0, passed=True):
    return {"grader_id":"g1","type":"exact_match","score":score,"passed":passed}

def test_multiple_attempts_per_test_case_id_valid():
    # Repeated trials of the same test_case_id, distinguished by ascending
    # attempt, are exactly what Discussion #22 added attempt to represent.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [
            {"test_case_id": "tc1", "attempt": 1, "passed": True, "grader_results": [_grader_result()]},
            {"test_case_id": "tc1", "attempt": 2, "passed": True, "grader_results": [_grader_result()]},
            {"test_case_id": "tc1", "attempt": 3, "passed": False, "grader_results": [_grader_result(0.0, False)]},
        ],
    }
    result = validate_result_set(rs)
    assert result.valid, result.errors

def test_duplicate_test_case_id_run_id_attempt_rejected():
    # The normative uniqueness rule: (test_case_id, run_id, attempt) must be
    # unique across results[] whenever attempt is present. Two Results for the
    # same test_case_id both stamped attempt: 1 is a collision, not a second
    # repetition (which would be attempt: 2).
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [
            {"test_case_id": "tc1", "attempt": 1, "passed": True, "grader_results": [_grader_result()]},
            {"test_case_id": "tc1", "attempt": 1, "passed": False, "grader_results": [_grader_result(0.0, False)]},
        ],
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["code"] == "DUPLICATE_ATTEMPT" for e in result.errors)

def test_same_attempt_number_different_test_case_id_is_not_a_collision():
    # attempt uniqueness is scoped to (test_case_id, run_id, attempt) -- two
    # different test cases can both have an attempt: 1 with no conflict.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [
            {"test_case_id": "tc1", "attempt": 1, "passed": True, "grader_results": [_grader_result()]},
            {"test_case_id": "tc2", "attempt": 1, "passed": True, "grader_results": [_grader_result()]},
        ],
    }
    assert validate_result_set(rs).valid

def test_attempt_must_be_positive_integer():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [{"test_case_id": "tc1", "attempt": 0, "passed": True, "grader_results": [_grader_result()]}],
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["code"] == "OUT_OF_RANGE" and e["path"] == "$.results[0].attempt" for e in result.errors)

def test_resultset_level_isolation_validates_fine():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "isolation": "fresh",
        "results": [{"test_case_id": "tc1", "attempt": 1, "passed": True, "grader_results": [_grader_result()]}],
    }
    assert validate_result_set(rs).valid

def test_isolation_is_an_open_string_not_an_enum():
    # mrwersa's explicit ask in Discussion #22: isolation values stay an open
    # string so a new isolation strategy never needs a spec change just to be
    # nameable. Any non-empty string, not just "fresh"/"shared", is valid.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "isolation": "sandboxed_container_per_attempt",
        "results": [{"test_case_id": "tc1", "passed": True, "grader_results": [_grader_result()]}],
    }
    assert validate_result_set(rs).valid

def test_non_string_isolation_rejected():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "isolation": 123,
        "results": [{"test_case_id": "tc1", "passed": True, "grader_results": [_grader_result()]}],
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.isolation" for e in result.errors)

def test_attempt_and_isolation_free_resultset_still_validates():
    # Backward compatibility: a ResultSet with neither field (every ResultSet
    # produced before this change) must remain fully valid, unchanged.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [{"test_case_id": "tc1", "passed": True, "grader_results": [_grader_result()]}],
    }
    result = validate_result_set(rs)
    assert result.valid
    assert "attempt" not in rs["results"][0]
    assert "isolation" not in rs

# --- PR #35 post-merge Copilot review: attempt-uniqueness key must not crash
# on unhashable test_case_id/run_id (github.com/adhabnr-ux/evalport/pull/35) ---

def test_non_string_test_case_id_with_attempt_reports_error_without_crashing():
    # Before the fix, building the (test_case_id, run_id, attempt) uniqueness
    # key with an unhashable test_case_id (e.g. a list, as a malformed
    # producer might emit) raised an uncaught TypeError instead of returning
    # a structured validation error.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [
            {"test_case_id": ["not", "a", "string"], "attempt": 1, "passed": True, "grader_results": [_grader_result()]},
        ],
    }
    result = validate_result_set(rs)  # must not raise
    assert not result.valid
    assert any(e["path"] == "$.results[0].test_case_id" and e["code"] == "REQUIRED" for e in result.errors)

def test_unhashable_run_id_with_attempt_reports_error_without_crashing():
    # Same failure mode as above, but for run_id: an unhashable run_id (e.g. a
    # dict) must not crash the uniqueness-key computation.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": {"not": "a string"}, "started_at": "2026-01-01T00:00:00Z",
        "results": [
            {"test_case_id": "tc1", "attempt": 1, "passed": True, "grader_results": [_grader_result()]},
        ],
    }
    result = validate_result_set(rs)  # must not raise
    assert not result.valid
    assert any(e["path"] == "$.run_id" and e["code"] == "REQUIRED" for e in result.errors)

def test_duplicate_attempt_still_caught_when_test_case_id_and_run_id_are_valid():
    # Guard against a regression where the crash fix above accidentally
    # disables the uniqueness check for the normal (all-strings) case.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [
            {"test_case_id": "tc1", "attempt": 1, "passed": True, "grader_results": [_grader_result()]},
            {"test_case_id": "tc1", "attempt": 1, "passed": False, "grader_results": [_grader_result(0.0, False)]},
        ],
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["code"] == "DUPLICATE_ATTEMPT" for e in result.errors)

# --- Hard Constraints: Result.constraint_violations (following on from a real
# maintainer discussion about AOBench's RBAC hard-fail case, raised while
# building the aobench-openeval-adapter interop sketch) ---

def _rs_with_constraint_violations(cvs, passed=False, run_id="cv-run-1"):
    return {
        "version": "1.0.0", "suite_id": "s", "run_id": run_id, "started_at": "2026-01-01T00:00:00Z",
        "results": [
            {
                "test_case_id": "tc1",
                "passed": passed,
                "grader_results": [_grader_result()],
                "constraint_violations": cvs,
            }
        ],
    }

def test_constraint_violations_absent_is_valid_and_unchanged():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": [{"test_case_id": "tc1", "passed": True, "grader_results": [_grader_result()]}],
    }
    assert validate_result_set(rs).valid

def test_constraint_violation_invalidating_requires_passed_false():
    rs = _rs_with_constraint_violations(
        [{"id": "rbac_scope", "type": "authorization", "invalidates_result": True}],
        passed=False,
    )
    assert validate_result_set(rs).valid

def test_constraint_violation_invalidating_but_passed_true_rejected():
    rs = _rs_with_constraint_violations(
        [{"id": "rbac_scope", "type": "authorization", "invalidates_result": True}],
        passed=True,
    )
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["code"] == "CONSTRAINT_INVALIDATES_PASS" and e["path"] == "$.results[0].passed" for e in result.errors)

def test_constraint_violation_informational_does_not_force_failure():
    # invalidates_result=False is purely a recorded/informational flag --
    # passed can legitimately stay True.
    rs = _rs_with_constraint_violations(
        [{"id": "rbac_scope", "type": "authorization", "invalidates_result": False}],
        passed=True,
    )
    assert validate_result_set(rs).valid

def test_constraint_violation_missing_id_rejected():
    rs = _rs_with_constraint_violations([{"type": "authorization", "invalidates_result": True}], passed=False)
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["code"] == "REQUIRED" and e["path"] == "$.results[0].constraint_violations[0].id" for e in result.errors)

def test_constraint_violation_missing_invalidates_result_rejected():
    rs = _rs_with_constraint_violations([{"id": "rbac_scope"}], passed=True)
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["code"] == "REQUIRED" and e["path"] == "$.results[0].constraint_violations[0].invalidates_result" for e in result.errors)

def test_constraint_violation_non_boolean_invalidates_result_rejected():
    rs = _rs_with_constraint_violations([{"id": "rbac_scope", "invalidates_result": "true"}], passed=True)
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["code"] == "REQUIRED" and "invalidates_result" in e["path"] for e in result.errors)

def test_constraint_violations_not_a_list_rejected():
    rs = _rs_with_constraint_violations({"id": "rbac_scope", "invalidates_result": True}, passed=False)
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["code"] == "TYPE_ERROR" and e["path"] == "$.results[0].constraint_violations" for e in result.errors)

def test_multiple_constraint_violations_only_one_invalidating_still_requires_passed_false():
    rs = _rs_with_constraint_violations(
        [
            {"id": "pii_leak", "type": "privacy", "invalidates_result": False},
            {"id": "rbac_scope", "type": "authorization", "invalidates_result": True},
        ],
        passed=False,
    )
    assert validate_result_set(rs).valid
    bad = _rs_with_constraint_violations(
        [
            {"id": "pii_leak", "type": "privacy", "invalidates_result": False},
            {"id": "rbac_scope", "type": "authorization", "invalidates_result": True},
        ],
        passed=True,
    )
    result = validate_result_set(bad)
    assert not result.valid
    assert any(e["code"] == "CONSTRAINT_INVALIDATES_PASS" for e in result.errors)

def test_constraint_violation_aobench_rbac_worked_example_raw_vs_effective():
    # Mirrors the real case from the AOBench maintainer discussion: a result
    # can have every grader_result pass (quality was genuinely good) and
    # still be correctly reported as failed overall because of a hard
    # constraint -- proving grader_results and constraint_violations really
    # are independent axes, not folded into one score.
    rs = {
        "version": "1.0.0", "suite_id": "hpc-ops-suite", "run_id": "aobench-run", "started_at": "2026-01-01T00:00:00Z",
        "results": [
            {
                "test_case_id": "tc_rbac",
                "actual_output": "cluster-07 billing overrun traced to a retry storm.",
                "passed": False,
                "grader_results": [
                    {"grader_id": "grounding", "type": "custom", "score": 0.97, "passed": True},
                    {"grader_id": "outcome", "type": "custom", "score": 1.0, "passed": True},
                ],
                "constraint_violations": [
                    {"id": "rbac_scope", "type": "authorization", "detail": "out of scope cluster", "invalidates_result": True},
                ],
            }
        ],
    }
    result = validate_result_set(rs)
    assert result.valid
    # Every individual grader genuinely passed...
    assert all(gr["passed"] for gr in rs["results"][0]["grader_results"])
    # ...yet the result as a whole is correctly reported as failed, because
    # a naive average of the grader scores (a real number close to 1.0)
    # would silently launder the RBAC violation into an apparent success.
    assert rs["results"][0]["passed"] is False
