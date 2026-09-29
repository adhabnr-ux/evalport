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

def test_empty_string_input_rejected():
    result = validate_test_case({"id":"tc1","input":"","graders":["g1"]})
    assert not result.valid
    assert any(
        e["path"] == "$.input" and e["code"] == "MIN_LENGTH"
        for e in result.errors
    )
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

# --- Discussion #45 (proposed): grouped/sibling ResultSets ---

def _minimal_result_list():
    return [{"test_case_id": "tc1", "passed": True, "grader_results": [_grader_result()]}]

def test_group_absent_still_validates_unchanged():
    # Backward compatibility: a ResultSet with no group field (every ResultSet
    # produced before this proposal) must remain fully valid, unchanged.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "results": _minimal_result_list(),
    }
    result = validate_result_set(rs)
    assert result.valid
    assert "group" not in rs

def test_group_with_only_group_id_is_valid():
    # group_id is the only required sub-field -- role/label/sequence are all optional.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "mutant-017-run", "started_at": "2026-01-01T00:00:00Z",
        "group": {"group_id": "mutation-sweep-2026-09-01"},
        "results": _minimal_result_list(),
    }
    assert validate_result_set(rs).valid

def test_group_with_all_fields_populated_is_valid():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "mutant-017-run", "started_at": "2026-01-01T00:00:00Z",
        "group": {
            "group_id": "mutation-sweep-2026-09-01",
            "role": "mutant",
            "label": "mutant_017 (relational-operator-swap in billing.py:42)",
            "sequence": 17,
        },
        "results": _minimal_result_list(),
    }
    assert validate_result_set(rs).valid

def test_group_missing_group_id_rejected():
    # group_id is REQUIRED whenever group is present -- an empty/absent group_id
    # is exactly the "which sweep is this?" ambiguity the field exists to remove.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": {"role": "mutant"},
        "results": _minimal_result_list(),
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.group_id" and e["code"] == "REQUIRED" for e in result.errors)

def test_group_empty_string_group_id_rejected():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": {"group_id": ""},
        "results": _minimal_result_list(),
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.group_id" and e["code"] == "REQUIRED" for e in result.errors)

def test_group_must_be_an_object():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": "mutation-sweep-2026-09-01",  # a bare string is not a valid group
        "results": _minimal_result_list(),
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group" and e["code"] == "TYPE_ERROR" for e in result.errors)

def test_group_role_is_an_open_string_not_an_enum():
    # Mirrors isolation's precedent (Discussion #22): role stays a free string so
    # a new grouping strategy never needs a spec change just to be nameable --
    # not just the mutation-testing-flavored values used in the RFC's examples.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": {"group_id": "grid-search-42", "role": "candidate_config_7"},
        "results": _minimal_result_list(),
    }
    assert validate_result_set(rs).valid

def test_group_non_string_role_rejected():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": {"group_id": "g1", "role": 42},
        "results": _minimal_result_list(),
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.role" and e["code"] == "TYPE_ERROR" for e in result.errors)

def test_group_non_string_label_rejected():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": {"group_id": "g1", "label": ["not", "a", "string"]},
        "results": _minimal_result_list(),
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.label" and e["code"] == "TYPE_ERROR" for e in result.errors)

def test_group_sequence_must_be_non_negative_integer():
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": {"group_id": "g1", "sequence": -1},
        "results": _minimal_result_list(),
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.sequence" and e["code"] == "OUT_OF_RANGE" for e in result.errors)

def test_group_sequence_zero_is_valid():
    # sequence is 0-indexed -- the first member of a group is sequence 0, not 1.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": {"group_id": "g1", "sequence": 0},
        "results": _minimal_result_list(),
    }
    assert validate_result_set(rs).valid

def test_group_sequence_bool_rejected():
    # bool is a subclass of int in Python -- same class of gotcha as the
    # boolean-score check above; True/False must not sneak through as 1/0.
    rs = {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "group": {"group_id": "g1", "sequence": True},
        "results": _minimal_result_list(),
    }
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.sequence" and e["code"] == "OUT_OF_RANGE" for e in result.errors)

def test_group_hyperparameter_sweep_second_domain_is_valid():
    # Discussion #45's own generalization claim (mutation testing, seed sweeps,
    # model comparisons) is only as credible as its weakest-checked domain --
    # this fixture mirrors spec/conformance/fixtures/group_hyperparameter_sweep_valid.json,
    # a hyperparameter grid search unrelated to mutation testing, with a
    # different role value ("candidate" rather than "mutant") and a sequence
    # populated the way optuna/optuna's real FrozenTrial.number is documented
    # to work ("unique and consecutive... zero-based") for a grid search whose
    # trial count is known upfront. Proves role isn't silently mutation-testing-shaped.
    rs = {
        "version": "1.1.0", "suite_id": "rag-retrieval-suite", "run_id": "trial-003-run",
        "started_at": "2026-09-05T14:00:00Z",
        "group": {
            "group_id": "lr-batchsize-grid-2026-09-05",
            "role": "candidate",
            "label": "lr=1e-4, batch_size=32",
            "sequence": 3,
        },
        "results": [
            {"test_case_id": "case_1", "passed": True, "grader_results": [
                {"grader_id": "gr1", "type": "semantic_similarity", "score": 0.88, "passed": True}
            ]},
            {"test_case_id": "case_2", "passed": True, "grader_results": [
                {"grader_id": "gr1", "type": "semantic_similarity", "score": 0.91, "passed": True}
            ]},
        ],
    }
    result = validate_result_set(rs)
    assert result.valid, result.errors

def test_group_multi_model_comparison_third_domain_is_valid():
    # Discussion #45's own generalization claim named TWO use cases beyond
    # mutation testing in issue #36: "a seed-sweep or a multi-model
    # comparison." Hyperparameter sweeps were grounded above via Optuna;
    # this fixture grounds the second, previously-ungrounded one. Verified
    # against promptfoo/promptfoo's actual current source (src/types/index.ts):
    # EvaluateResult.provider: {id, label} identifies which model produced a
    # row, and CompletedPrompt.provider + its .metrics (score, testPassCount,
    # ...) is exactly the consumer-computed per-provider rollup this RFC's
    # group design already assumes -- confirming the join-key-on-member /
    # rollup-computed-by-consumer shape a fifth time (after W&B, MLflow,
    # Stryker, Optuna), independently. providers are configured as an
    # ordered array (providers: z.array(ApiProviderSchema)) and evaluated in
    # that order, so sequence is well-defined here too -- not just for
    # adaptive producers that must omit it. Provider ids and premise are
    # taken directly from promptfoo's own real example built for exactly
    # this purpose: examples/compare-claude-vs-gpt-image/promptfooconfig.yaml.
    # Mirrors spec/conformance/fixtures/group_multi_model_comparison_valid.json.
    rs = {
        "version": "1.1.0", "suite_id": "image-description-suite", "run_id": "openai-gpt-4.1-run",
        "started_at": "2026-09-07T10:00:00Z",
        "group": {
            "group_id": "claude-vs-gpt-vs-gemini-image-2026-09-07",
            "role": "openai:gpt-4.1",
            "label": "GPT-4.1 (image description accuracy)",
            "sequence": 1,
        },
        "results": [
            {"test_case_id": "great_wave_off_kanagawa", "passed": True, "grader_results": [
                {"grader_id": "gr1", "type": "llm_judge", "score": 0.92, "passed": True}
            ]},
        ],
    }
    result = validate_result_set(rs)
    assert result.valid, result.errors

def test_group_role_survived_with_real_gap_metadata_is_valid():
    # AshwinUgale's Discussion #45 refinement: muteval's "survived" isn't one
    # thing -- a survivor is either a real coverage gap or an inert/equivalent
    # mutant muteval excludes from its effective mutation score. role stays
    # the coarse, conventional "survived" verdict (schema doesn't distinguish
    # further); the real-gap-vs-inert bit rides in free-form metadata instead,
    # reusing muteval's own MutantOutcome.output_changed field name rather than
    # inventing new spec vocabulary. This is the real-coverage-gap branch --
    # mirrors spec/conformance/fixtures/group_role_metadata_real_gap_valid.json.
    rs = {
        "version": "1.1.0", "suite_id": "billing-suite", "run_id": "mutant-021-run",
        "started_at": "2026-09-06T09:00:00Z",
        "group": {"group_id": "mutation-sweep-2026-09-06", "role": "survived", "sequence": 21},
        "metadata": {"output_changed": True},
        "results": [
            {"test_case_id": "case_1", "passed": True, "grader_results": [
                {"grader_id": "gr1", "type": "exact_match", "score": 1.0, "passed": True}
            ]},
        ],
    }
    result = validate_result_set(rs)
    assert result.valid, result.errors

def test_group_role_survived_with_inert_metadata_is_valid():
    # Paired case: same coarse role: "survived" verdict, but
    # metadata.output_changed: False marks this member as one an
    # effective-mutation-score rollup should exclude -- muteval's own
    # distinction, not folded into role. Mirrors
    # spec/conformance/fixtures/group_role_metadata_inert_survivor_valid.json.
    rs = {
        "version": "1.1.0", "suite_id": "billing-suite", "run_id": "mutant-022-run",
        "started_at": "2026-09-06T09:00:05Z",
        "group": {"group_id": "mutation-sweep-2026-09-06", "role": "survived", "sequence": 22},
        "metadata": {"output_changed": False},
        "results": [
            {"test_case_id": "case_1", "passed": True, "grader_results": [
                {"grader_id": "gr1", "type": "exact_match", "score": 1.0, "passed": True}
            ]},
        ],
    }
    result = validate_result_set(rs)
    assert result.valid, result.errors

def test_group_role_metadata_lets_consumer_reconstruct_raw_and_effective_score():
    # End-to-end proof of the RFC refinement's actual point: given a full
    # 3-mutant group (one killed, one survived-real-gap, one
    # survived-inert-equivalent), a consumer can derive BOTH muteval's raw
    # mutation score (killed / total) and its effective score
    # (killed / (total - inert-excluded)) purely from group.role +
    # metadata.output_changed -- no schema change, no growth in role's
    # vocabulary, exactly what AshwinUgale's comment asked whether this
    # design could support.
    group_id = "mutation-sweep-2026-09-06-full"
    members = [
        {
            "version": "1.1.0", "suite_id": "billing-suite", "run_id": "mutant-020-run",
            "started_at": "2026-09-06T08:59:55Z",
            "group": {"group_id": group_id, "role": "killed", "sequence": 20},
            "results": [
                {"test_case_id": "case_1", "passed": False, "grader_results": [
                    {"grader_id": "gr1", "type": "exact_match", "score": 0.0, "passed": False}
                ]},
            ],
        },
        {
            "version": "1.1.0", "suite_id": "billing-suite", "run_id": "mutant-021-run",
            "started_at": "2026-09-06T09:00:00Z",
            "group": {"group_id": group_id, "role": "survived", "sequence": 21},
            "metadata": {"output_changed": True},
            "results": [
                {"test_case_id": "case_1", "passed": True, "grader_results": [
                    {"grader_id": "gr1", "type": "exact_match", "score": 1.0, "passed": True}
                ]},
            ],
        },
        {
            "version": "1.1.0", "suite_id": "billing-suite", "run_id": "mutant-022-run",
            "started_at": "2026-09-06T09:00:05Z",
            "group": {"group_id": group_id, "role": "survived", "sequence": 22},
            "metadata": {"output_changed": False},
            "results": [
                {"test_case_id": "case_1", "passed": True, "grader_results": [
                    {"grader_id": "gr1", "type": "exact_match", "score": 1.0, "passed": True}
                ]},
            ],
        },
    ]
    for rs in members:
        result = validate_result_set(rs)
        assert result.valid, result.errors

    # Consumer-side rollup math (deliberately NOT part of the schema/validator
    # -- the RFC explicitly declines to standardize this, see "What this
    # deliberately does not do" in SPEC.md). Demonstrated here only to prove
    # both numbers are actually reconstructible from the documents above.
    total = len(members)
    killed = sum(1 for m in members if m["group"]["role"] == "killed")
    inert_excluded = sum(
        1 for m in members
        if m["group"]["role"] == "survived" and m.get("metadata", {}).get("output_changed") is False
    )
    raw_score = killed / total
    effective_score = killed / (total - inert_excluded)

    assert raw_score == 1 / 3
    assert effective_score == 1 / 2
    assert effective_score > raw_score  # excluding the inert survivor raises the score, as it should

# --- parent_group_id: nested/hierarchical groups (sweep-of-sweeps) ---
#
# Grew out of an explicit "is a flat group the right model?" audit against real
# systems: MLflow's nested runs (parent_run_id chains, arbitrarily deep via
# active_run_stack) form an arbitrarily deep tree in real usage -- confirmed
# against mlflow/mlflow#16685's actual GrandParent/Parent/150-Child test case,
# not just the API surface. W&B's Run.sweep_id and its separate
# wandb.init(group=...) primitive are both flat, single-level, with no parent
# construct anywhere in wandb/wandb's source -- so EvalPort's flat-only design
# matched W&B but not MLflow. parent_group_id closes that gap the same way
# group_id itself is modeled: a pointer on the member, not an embedded tree.

def _rs_with_group(group, run_id="mutant-017-run"):
    return {
        "version": "1.0.0", "suite_id": "s", "run_id": run_id, "started_at": "2026-01-01T00:00:00Z",
        "group": group,
        "results": _minimal_result_list(),
    }

def test_group_parent_group_id_absent_is_valid_and_unchanged():
    # Backward compatibility: every group-bearing ResultSet before this
    # addition had no parent_group_id and must remain valid unchanged.
    rs = _rs_with_group({"group_id": "mutation-sweep-2026-09-01"})
    result = validate_result_set(rs)
    assert result.valid, result.errors
    assert "parent_group_id" not in rs["group"]

def test_group_parent_group_id_valid_nested_sweep():
    # Mirrors the MLflow-grounded worked example: a child sweep nested under a
    # parent sweep, the same shape as MLflow's GrandParent/Parent/Child chain.
    rs = _rs_with_group({
        "group_id": "child-sweep-lr-1e-4",
        "parent_group_id": "parent-sweep-lr-batchsize-grid-2026-09-08",
        "role": "candidate",
        "sequence": 3,
    })
    result = validate_result_set(rs)
    assert result.valid, result.errors

def test_group_parent_group_id_empty_string_rejected():
    rs = _rs_with_group({"group_id": "g1", "parent_group_id": ""})
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.parent_group_id" and e["code"] == "REQUIRED" for e in result.errors)

def test_group_parent_group_id_non_string_rejected():
    rs = _rs_with_group({"group_id": "g1", "parent_group_id": 42})
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.parent_group_id" and e["code"] == "REQUIRED" for e in result.errors)

def test_group_parent_group_id_equal_to_group_id_rejected():
    # A group cannot be its own parent -- the one structural self-consistency
    # check a single document CAN make (a real cross-document cycle, e.g. A's
    # parent is B and B's parent is A, needs multi-document reasoning this
    # validator deliberately doesn't attempt -- see the schema description).
    rs = _rs_with_group({"group_id": "sweep-42", "parent_group_id": "sweep-42"})
    result = validate_result_set(rs)
    assert not result.valid
    assert any(e["path"] == "$.group.parent_group_id" and e["code"] == "SELF_PARENT" for e in result.errors)

def test_group_three_level_nesting_matches_mlflow_grandparent_parent_child_shape():
    # Proves the pointer-chain mechanism actually supports 3+ levels, the same
    # depth mlflow/mlflow#16685's real GrandParent/Parent/Child test exercises
    # -- each ResultSet only ever names its OWN immediate parent, never an
    # embedded ancestor list, exactly like MLflow's parent_run_id.
    grandparent = _rs_with_group({"group_id": "campaign-2026-09-08"}, run_id="grandparent-run")
    parent = _rs_with_group(
        {"group_id": "sweep-lr-grid", "parent_group_id": "campaign-2026-09-08"}, run_id="parent-run"
    )
    child = _rs_with_group(
        {"group_id": "trial-003", "parent_group_id": "sweep-lr-grid", "sequence": 3}, run_id="child-run"
    )

    for rs in (grandparent, parent, child):
        result = validate_result_set(rs)
        assert result.valid, result.errors

    # A consumer walks the chain purely from parent_group_id pointers, the
    # same way MLflow's get_parent_run() walks parent_run_id one hop at a time.
    by_group_id = {rs["group"]["group_id"]: rs for rs in (grandparent, parent, child)}
    chain = [child["group"]["group_id"]]
    cur = child
    while cur["group"].get("parent_group_id") in by_group_id:
        cur = by_group_id[cur["group"]["parent_group_id"]]
        chain.append(cur["group"]["group_id"])
    assert chain == ["trial-003", "sweep-lr-grid", "campaign-2026-09-08"]

# --- Validator fidelity: optional-field types, RFC 3339 timestamps, Rule 6 ---
# The hand-rolled validator used to accept documents spec/schemas/*.json rejects:
# an optional field of the wrong type (e.g. Result.actual_output as a list --
# found by ChelseaKR in ChelseaKR/gauntlet#76), a date-only or offset-less
# started_at/completed_at, and a null-scored GraderResult with passed: true.
# Mirrored case-for-case in sdk/typescript/tests/validate.test.ts.

import pytest
from openeval.validate import is_rfc3339_date_time

def _fid_rs():
    return {
        "version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-15T10:30:00Z",
        "results": [{
            "test_case_id": "tc1", "passed": True,
            "grader_results": [{"grader_id": "g1", "type": "exact_match", "score": 1.0, "passed": True}],
        }],
    }

def _set(doc, path, value):
    # path like ("results", 0, "grader_results", 0, "reason")
    cur = doc
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return doc

def _del(doc, path):
    cur = doc
    for k in path[:-1]:
        cur = cur[k]
    del cur[path[-1]]
    return doc

R0 = ("results", 0)
GR0 = ("results", 0, "grader_results", 0)

# (name, path to set, value, expected error path, expected code)
RESULTSET_REJECT_CASES = [
    ("suite_version not string", ("suite_version",), 1, "$.suite_version", "TYPE_ERROR"),
    ("completed_at date-only", ("completed_at",), "2026-01-15", "$.completed_at", "INVALID_DATE_TIME"),
    ("completed_at not string", ("completed_at",), 1700000000, "$.completed_at", "TYPE_ERROR"),
    ("provider not object", ("provider",), "gpt-4o", "$.provider", "TYPE_ERROR"),
    ("provider null", ("provider",), None, "$.provider", "TYPE_ERROR"),
    ("provider.model not string", ("provider",), {"model": 4}, "$.provider.model", "TYPE_ERROR"),
    ("provider.api_base not string", ("provider",), {"api_base": ["x"]}, "$.provider.api_base", "TYPE_ERROR"),
    ("provider.temperature string", ("provider",), {"temperature": "0.1"}, "$.provider.temperature", "TYPE_ERROR"),
    ("provider.temperature bool", ("provider",), {"temperature": True}, "$.provider.temperature", "TYPE_ERROR"),
    ("provider.max_tokens fractional", ("provider",), {"max_tokens": 1.5}, "$.provider.max_tokens", "TYPE_ERROR"),
    ("provider.max_tokens bool", ("provider",), {"max_tokens": True}, "$.provider.max_tokens", "TYPE_ERROR"),
    ("provider.extra not object", ("provider",), {"extra": []}, "$.provider.extra", "TYPE_ERROR"),
    ("runner not object", ("runner",), [], "$.runner", "TYPE_ERROR"),
    ("runner.name not string", ("runner",), {"name": 1}, "$.runner.name", "TYPE_ERROR"),
    ("runner.version not string", ("runner",), {"version": 1.0}, "$.runner.version", "TYPE_ERROR"),
    ("summary not object", ("summary",), [], "$.summary", "TYPE_ERROR"),
    ("summary.total negative", ("summary",), {"total": -1}, "$.summary.total", "OUT_OF_RANGE"),
    ("summary.total bool", ("summary",), {"total": True}, "$.summary.total", "TYPE_ERROR"),
    ("summary.passed fractional", ("summary",), {"passed": 0.5}, "$.summary.passed", "TYPE_ERROR"),
    ("summary.failed string", ("summary",), {"failed": "0"}, "$.summary.failed", "TYPE_ERROR"),
    ("summary.skipped negative", ("summary",), {"skipped": -2}, "$.summary.skipped", "OUT_OF_RANGE"),
    ("summary.pass_rate above 1", ("summary",), {"pass_rate": 1.5}, "$.summary.pass_rate", "OUT_OF_RANGE"),
    ("summary.avg_score string", ("summary",), {"avg_score": "0.5"}, "$.summary.avg_score", "TYPE_ERROR"),
    ("summary.duration_ms fractional", ("summary",), {"duration_ms": 1.5}, "$.summary.duration_ms", "TYPE_ERROR"),
    ("summary.by_grader not object", ("summary",), {"by_grader": []}, "$.summary.by_grader", "TYPE_ERROR"),
    ("summary.by_grader entry not object", ("summary",), {"by_grader": {"g1": 3}}, "$.summary.by_grader.g1", "TYPE_ERROR"),
    ("summary.by_grader.passed fractional", ("summary",), {"by_grader": {"g1": {"passed": 1.5}}}, "$.summary.by_grader.g1.passed", "TYPE_ERROR"),
    ("summary.by_grader.failed bool", ("summary",), {"by_grader": {"g1": {"failed": False}}}, "$.summary.by_grader.g1.failed", "TYPE_ERROR"),
    ("summary.by_grader.avg_score string", ("summary",), {"by_grader": {"g1": {"avg_score": "x"}}}, "$.summary.by_grader.g1.avg_score", "TYPE_ERROR"),
    ("metadata not object", ("metadata",), [], "$.metadata", "TYPE_ERROR"),
    ("metadata null", ("metadata",), None, "$.metadata", "TYPE_ERROR"),
    ("isolation null", ("isolation",), None, "$.isolation", "TYPE_ERROR"),
    ("group null", ("group",), None, "$.group", "TYPE_ERROR"),
    ("started_at date-only", ("started_at",), "2026-01-15", "$.started_at", "INVALID_DATE_TIME"),
    ("started_at no offset", ("started_at",), "2026-01-15T10:30:00", "$.started_at", "INVALID_DATE_TIME"),
    ("started_at no seconds", ("started_at",), "2026-01-15T10:30Z", "$.started_at", "INVALID_DATE_TIME"),
    ("started_at space separator", ("started_at",), "2026-01-15 10:30:00Z", "$.started_at", "INVALID_DATE_TIME"),
    ("started_at offset without colon", ("started_at",), "2026-01-15T10:30:00+0530", "$.started_at", "INVALID_DATE_TIME"),
    ("started_at impossible day", ("started_at",), "2026-02-30T10:30:00Z", "$.started_at", "INVALID_DATE_TIME"),
    ("started_at month 13", ("started_at",), "2026-13-01T10:30:00Z", "$.started_at", "INVALID_DATE_TIME"),
    ("started_at hour 24", ("started_at",), "2026-01-15T24:00:00Z", "$.started_at", "INVALID_DATE_TIME"),
    ("started_at epoch string", ("started_at",), "1700000000", "$.started_at", "INVALID_DATE_TIME"),
    # Result
    ("actual_output list (ChelseaKR/gauntlet#76)", R0 + ("actual_output",), ["a", "b"], "$.results[0].actual_output", "TYPE_ERROR"),
    ("actual_output object", R0 + ("actual_output",), {"text": "a"}, "$.results[0].actual_output", "TYPE_ERROR"),
    ("actual_output null", R0 + ("actual_output",), None, "$.results[0].actual_output", "TYPE_ERROR"),
    ("test_case_id empty", R0 + ("test_case_id",), "", "$.results[0].test_case_id", "REQUIRED"),
    ("duration_ms fractional", R0 + ("duration_ms",), 1.5, "$.results[0].duration_ms", "TYPE_ERROR"),
    ("duration_ms negative", R0 + ("duration_ms",), -1, "$.results[0].duration_ms", "OUT_OF_RANGE"),
    ("duration_ms bool", R0 + ("duration_ms",), True, "$.results[0].duration_ms", "TYPE_ERROR"),
    ("result completed_at offset-less", R0 + ("completed_at",), "2026-01-15T10:30:05", "$.results[0].completed_at", "INVALID_DATE_TIME"),
    ("error not object", R0 + ("error",), "boom", "$.results[0].error", "TYPE_ERROR"),
    ("error.type not in enum", R0 + ("error",), {"type": "crash"}, "$.results[0].error.type", "INVALID_VALUE"),
    ("error.message not string", R0 + ("error",), {"message": 1}, "$.results[0].error.message", "TYPE_ERROR"),
    ("error.code fractional", R0 + ("error",), {"code": 1.5}, "$.results[0].error.code", "TYPE_ERROR"),
    ("error.code bool", R0 + ("error",), {"code": True}, "$.results[0].error.code", "TYPE_ERROR"),
    ("error.retryable string", R0 + ("error",), {"retryable": "yes"}, "$.results[0].error.retryable", "TYPE_ERROR"),
    ("result metadata not object", R0 + ("metadata",), "trace", "$.results[0].metadata", "TYPE_ERROR"),
    ("attempt null", R0 + ("attempt",), None, "$.results[0].attempt", "OUT_OF_RANGE"),
    # GraderResult
    ("grader_id empty", GR0 + ("grader_id",), "", "$.results[0].grader_results[0].grader_id", "REQUIRED"),
    ("reason not string", GR0 + ("reason",), 1, "$.results[0].grader_results[0].reason", "TYPE_ERROR"),
    ("grader metadata not object", GR0 + ("metadata",), [], "$.results[0].grader_results[0].metadata", "TYPE_ERROR"),
]

@pytest.mark.parametrize("name,path,value,err_path,code", RESULTSET_REJECT_CASES, ids=[c[0] for c in RESULTSET_REJECT_CASES])
def test_resultset_optional_field_rejections(name, path, value, err_path, code):
    doc = _set(_fid_rs(), path, value)
    r = validate_result_set(doc)
    assert not r.valid, name
    assert any(e["path"] == err_path and e["code"] == code for e in r.errors), r.errors

def test_grader_result_missing_score_is_required_not_null():
    # score is REQUIRED; an absent key used to be read as null and accepted.
    doc = _del(_fid_rs(), GR0 + ("score",))
    _set(doc, GR0 + ("passed",), False)
    _set(doc, R0 + ("passed",), False)
    r = validate_result_set(doc)
    assert any(e["path"] == "$.results[0].grader_results[0].score" and e["code"] == "REQUIRED" for e in r.errors), r.errors

def test_rule6_null_score_with_passed_true_rejected():
    doc = _fid_rs()
    doc["results"][0]["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": True})
    r = validate_result_set(doc)
    assert not r.valid
    assert [e["code"] for e in r.errors] == ["NULL_SCORE_PASSED"]
    assert r.errors[0]["path"] == "$.results[0].grader_results[1].passed"

def test_rule6_all_null_scored_result_with_passed_true_rejected():
    doc = _fid_rs()
    doc["results"][0]["grader_results"] = [
        {"grader_id": "g1", "type": "human", "score": None, "passed": False},
        {"grader_id": "g2", "type": "llm_judge", "score": None, "passed": False},
    ]
    r = validate_result_set(doc)
    assert not r.valid
    assert [(e["path"], e["code"]) for e in r.errors] == [("$.results[0].passed", "UNSCORED_RESULT_PASSED")]

def test_rule6_all_null_scored_result_with_passed_false_valid():
    doc = _fid_rs()
    doc["results"][0]["passed"] = False
    doc["results"][0]["grader_results"] = [{"grader_id": "g1", "type": "human", "score": None, "passed": False}]
    assert validate_result_set(doc).valid

def test_rule6_mixed_null_and_scored_result_can_pass():
    # Null-scored graders are excluded from aggregation, so a Result can pass on
    # its scored graders alone -- only the ALL-null case must be passed: false.
    doc = _fid_rs()
    doc["results"][0]["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": False})
    assert validate_result_set(doc).valid

def test_rule6_empty_grader_results_is_not_all_null():
    # An empty grader_results list is not "all null-scored" (nothing to aggregate
    # either way) and was valid before; keep it that way.
    doc = _fid_rs()
    doc["results"][0]["grader_results"] = []
    assert validate_result_set(doc).valid

def test_resultset_with_every_optional_field_well_typed_is_valid():
    doc = _fid_rs()
    doc.update({
        "$schema": "https://evalport.org/schema/resultset.json",
        "suite_version": "1.0.0",
        "completed_at": "2026-01-15T10:31:45.123+05:30",
        "provider": {"model": "gpt-4o", "api_base": "https://api.example.com", "temperature": 0, "max_tokens": 256, "extra": {"seed": 1}},
        "runner": {"name": "evalport-cli", "version": "1.3.1"},
        "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "pass_rate": 1, "avg_score": 1.0, "duration_ms": 1200,
                    "by_grader": {"g1": {"passed": 1, "failed": 0, "avg_score": 1.0}}},
        "metadata": {"openeval.partial": False},
    })
    doc["results"][0].update({
        "actual_output": "Paris", "duration_ms": 1200, "completed_at": "2026-01-15t10:31:44z",
        "error": {"type": "provider_error", "message": "rate limited", "code": 429, "retryable": True},
        "metadata": {"trace_id": "abc"},
    })
    doc["results"][0]["grader_results"][0].update({"reason": "exact", "metadata": {"k": "v"}})
    r = validate_result_set(doc)
    assert r.valid, r.errors

def test_error_code_may_be_string_or_integer():
    for code in ("RATE_LIMIT", 429):
        doc = _set(_fid_rs(), R0 + ("error",), {"type": "timeout", "code": code})
        assert validate_result_set(doc).valid, code

def test_integral_float_counts_as_integer_like_json_schema():
    # JSON Schema 2020-12 "integer" matches any number with a zero fractional
    # part, so 1200.0 is a valid duration_ms (and JSON.parse makes it 1200 in TS).
    doc = _set(_fid_rs(), R0 + ("duration_ms",), 1200.0)
    assert validate_result_set(doc).valid

@pytest.mark.parametrize("value", [
    "2026-01-15T10:30:00Z", "2026-01-15T10:30:00+05:30", "2026-01-15T10:30:00-08:00",
    "2026-01-15t10:30:00z", "2026-01-15T10:30:00.123456Z", "2024-02-29T00:00:00Z",
    "2026-01-15T10:30:00.5+00:00", "2016-12-31T23:59:60Z",
])
def test_rfc3339_date_time_accepts(value):
    assert is_rfc3339_date_time(value)
    assert validate_result_set(_set(_fid_rs(), ("started_at",), value)).valid

@pytest.mark.parametrize("value", [
    "2026-01-15", "2026-01-15T10:30:00", "2026-01-15T10:30Z", "2026-01-15 10:30:00Z",
    "2026-01-15T10:30:00+0530", "2026-01-15T10:30:00+05", "2025-02-29T00:00:00Z", "2026-04-31T00:00:00Z",
    "2026-00-10T00:00:00Z", "2026-01-00T00:00:00Z", "2026-01-15T10:60:00Z", "2026-01-15T10:30:61Z",
    "2026-01-15T10:30:00.Z", "2026-01-15T10:30:00Z\n", "20260115T103000Z", "", "now",
])
def test_rfc3339_date_time_rejects(value):
    assert not is_rfc3339_date_time(value)
    r = validate_result_set(_set(_fid_rs(), ("started_at",), value))
    assert any(e["path"] == "$.started_at" and e["code"] == "INVALID_DATE_TIME" for e in r.errors)

def test_python_isoformat_utc_timestamp_is_valid():
    # What create_result_set() and most Python adapters emit.
    from datetime import datetime, timezone
    assert is_rfc3339_date_time(datetime.now(timezone.utc).isoformat())
    assert is_rfc3339_date_time(datetime(2026, 1, 15, tzinfo=timezone.utc).isoformat())
    # A naive datetime has no offset -- not RFC 3339.
    assert not is_rfc3339_date_time(datetime(2026, 1, 15).isoformat())

# TestCase / Grader / Suite optional fields

TC_BASE = {"id": "tc1", "input": "hi", "graders": ["g1"]}
TESTCASE_REJECT_CASES = [
    ("expected_output", 1, "$.expected_output", "TYPE_ERROR"),
    ("expected_output", None, "$.expected_output", "TYPE_ERROR"),
    ("context", "doc", "$.context", "TYPE_ERROR"),
    ("context", ["ok", 2], "$.context", "TYPE_ERROR"),
    ("retrieval_context", [None], "$.retrieval_context", "TYPE_ERROR"),
    ("tools_called", "search", "$.tools_called", "TYPE_ERROR"),
    ("expected_tools", [{"name": "search"}], "$.expected_tools", "TYPE_ERROR"),
    ("metadata", [], "$.metadata", "TYPE_ERROR"),
    ("tags", "smoke", "$.tags", "TYPE_ERROR"),
    ("provider", "gpt-4o", "$.provider", "TYPE_ERROR"),
    ("provider", {"max_tokens": 0}, "$.provider.max_tokens", "OUT_OF_RANGE"),
    ("provider", {"api_key_env": 1}, "$.provider.api_key_env", "TYPE_ERROR"),
    ("provider", {"temperature": "hot"}, "$.provider.temperature", "TYPE_ERROR"),
    ("params", [], "$.params", "TYPE_ERROR"),
    ("timeout_ms", 0, "$.timeout_ms", "OUT_OF_RANGE"),
    ("timeout_ms", 1.5, "$.timeout_ms", "TYPE_ERROR"),
    ("weight", -1, "$.weight", "OUT_OF_RANGE"),
    ("weight", True, "$.weight", "TYPE_ERROR"),
]

@pytest.mark.parametrize("key,value,err_path,code", TESTCASE_REJECT_CASES)
def test_testcase_optional_field_rejections(key, value, err_path, code):
    r = validate_test_case({**TC_BASE, key: value})
    assert not r.valid
    assert any(e["path"] == err_path and e["code"] == code for e in r.errors), r.errors

def test_testcase_every_optional_field_well_typed_is_valid():
    tc = {**TC_BASE, "expected_output": "Paris", "context": ["a"], "retrieval_context": ["b"],
          "tools_called": ["search"], "expected_tools": ["search"], "metadata": {"k": 1}, "tags": ["smoke"],
          "provider": {"model": "gpt-4o", "api_base": "x", "api_key_env": "OPENAI_API_KEY", "temperature": 0.2, "max_tokens": 1, "extra": {}},
          "params": {"top_p": 1}, "timeout_ms": 1, "weight": 0}
    r = validate_test_case(tc)
    assert r.valid, r.errors

GRADER_REJECT_CASES = [
    ({"id": "g", "type": "exact_match", "params": []}, "$.params", "TYPE_ERROR"),
    ({"id": "g", "type": "custom", "params": ["x"]}, "$.params", "TYPE_ERROR"),
    ({"id": "g", "type": "exact_match", "weight": -0.5}, "$.weight", "OUT_OF_RANGE"),
    ({"id": "g", "type": "exact_match", "weight": "1"}, "$.weight", "TYPE_ERROR"),
    ({"id": "g", "type": "exact_match", "description": 1}, "$.description", "TYPE_ERROR"),
    ({"id": "g", "type": "contains", "params": {"substring": "a", "ignore_case": "yes"}}, "$.params.ignore_case", "TYPE_ERROR"),
    ({"id": "g", "type": "regex", "params": {"pattern": "a", "flags": 1}}, "$.params.flags", "TYPE_ERROR"),
    ({"id": "g", "type": "semantic_similarity", "params": {"threshold": True}}, "$.params.threshold", "OUT_OF_RANGE"),
    ({"id": "g", "type": "semantic_similarity", "params": {"threshold": 0.8, "model": 1}}, "$.params.model", "TYPE_ERROR"),
    ({"id": "g", "type": "semantic_similarity", "params": {"threshold": 0.8, "provider": 1}}, "$.params.provider", "TYPE_ERROR"),
    ({"id": "g", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}", "temperature": 3}}, "$.params.temperature", "OUT_OF_RANGE"),
    ({"id": "g", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}", "schema": "x"}}, "$.params.schema", "TYPE_ERROR"),
    ({"id": "g", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}", "provider": 1}}, "$.params.provider", "TYPE_ERROR"),
    ({"id": "g", "type": "json_schema", "params": {"schema": {}, "strict": "true"}}, "$.params.strict", "TYPE_ERROR"),
    ({"id": "g", "type": "json_path", "params": {"path": "$.a", "expected": "1", "operator": "like"}}, "$.params.operator", "INVALID_VALUE"),
    ({"id": "g", "type": "code", "params": {"language": "python", "source": "x", "timeout_ms": 50}}, "$.params.timeout_ms", "OUT_OF_RANGE"),
]

@pytest.mark.parametrize("doc,err_path,code", GRADER_REJECT_CASES, ids=[f"{c[0]['type']}:{c[1]}" for c in GRADER_REJECT_CASES])
def test_grader_optional_field_rejections(doc, err_path, code):
    r = validate_grader(doc)
    assert not r.valid
    assert any(e["path"] == err_path and e["code"] == code for e in r.errors), r.errors

def test_grader_non_object_params_still_reports_missing_handler_without_crashing():
    r = validate_grader({"id": "g", "type": "custom", "params": ["x"]})
    assert {(e["path"], e["code"]) for e in r.errors} == {("$.params", "TYPE_ERROR"), ("$.params.handler", "REQUIRED")}

def test_grader_optional_params_well_typed_are_valid():
    for g in [
        {"id": "g", "type": "contains", "params": {"substring": "a", "ignore_case": True}, "weight": 2, "description": "d"},
        {"id": "g", "type": "regex", "params": {"pattern": "a", "flags": "i"}},
        {"id": "g", "type": "semantic_similarity", "params": {"threshold": 0.8, "model": "m", "provider": "p"}},
        {"id": "g", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}", "provider": "p", "temperature": 2, "schema": {}}},
        {"id": "g", "type": "json_schema", "params": {"schema": {}, "strict": False}},
        {"id": "g", "type": "json_path", "params": {"path": "$.a", "expected": "1", "operator": "gte"}},
        {"id": "g", "type": "code", "params": {"language": "python", "source": "x", "timeout_ms": 100}},
    ]:
        r = validate_grader(g)
        assert r.valid, (g, r.errors)

SUITE_BASE = {"version": "1.0.0", "id": "s", "graders": [{"id": "g1", "type": "exact_match"}], "test_cases": [TC_BASE]}
SUITE_REJECT_CASES = [
    ("name", 1, "$.name", "TYPE_ERROR"),
    ("description", ["x"], "$.description", "TYPE_ERROR"),
    ("graders", {"g1": {}}, "$.graders", "TYPE_ERROR"),
    ("test_cases_file", 1, "$.test_cases_file", "TYPE_ERROR"),
    ("metadata", "x", "$.metadata", "TYPE_ERROR"),
    ("tags", [1], "$.tags", "TYPE_ERROR"),
    ("config", [], "$.config", "TYPE_ERROR"),
    ("config", {"parallel": 0}, "$.config.parallel", "OUT_OF_RANGE"),
    ("config", {"provider": []}, "$.config.provider", "TYPE_ERROR"),
    ("config", {"provider": {"max_tokens": 0}}, "$.config.provider.max_tokens", "OUT_OF_RANGE"),
    ("config", {"defaults": {"timeout_ms": 0}}, "$.config.defaults.timeout_ms", "OUT_OF_RANGE"),
    ("config", {"defaults": {"weight": -1}}, "$.config.defaults.weight", "OUT_OF_RANGE"),
    ("config", {"retry": {"max_attempts": 0}}, "$.config.retry.max_attempts", "OUT_OF_RANGE"),
    ("config", {"retry": {"backoff_ms": 10}}, "$.config.retry.backoff_ms", "OUT_OF_RANGE"),
    ("$schema", 1, "$.$schema", "TYPE_ERROR"),
]

@pytest.mark.parametrize("key,value,err_path,code", SUITE_REJECT_CASES)
def test_suite_optional_field_rejections(key, value, err_path, code):
    r = validate_suite({**SUITE_BASE, key: value})
    assert not r.valid
    assert any(e["path"] == err_path and e["code"] == code for e in r.errors), r.errors

def test_suite_nested_testcase_optional_field_rejection_is_reported():
    # Nested errors keep validate_suite's existing "$.test_cases[i]." + "$.<field>" prefixing.
    r = validate_suite({**SUITE_BASE, "test_cases": [{**TC_BASE, "tags": "smoke"}]})
    assert not r.valid
    assert [(e["path"], e["code"]) for e in r.errors] == [("$.test_cases[0].$.tags", "TYPE_ERROR")]

def test_suite_test_cases_wrong_type_rejected_even_with_test_cases_file():
    r = validate_suite({"version": "1.0.0", "id": "s", "test_cases_file": "cases.jsonl", "test_cases": "cases.jsonl"})
    assert any(e["path"] == "$.test_cases" and e["code"] == "TYPE_ERROR" for e in r.errors), r.errors

def test_suite_graders_validated_when_test_cases_come_from_file():
    r = validate_suite({"version": "1.0.0", "id": "s", "test_cases_file": "cases.jsonl", "graders": [{"id": "g1", "type": "custom"}]})
    assert any(e["path"] == "$.graders[0].$.params.handler" for e in r.errors), r.errors

def test_suite_every_optional_field_well_typed_is_valid():
    s = {**SUITE_BASE, "$schema": "https://evalport.org/schema/suite.json", "name": "n", "description": "d",
         "metadata": {"k": 1}, "tags": ["a"],
         "config": {"provider": {"model": "m", "max_tokens": 10}, "defaults": {"timeout_ms": 1000, "weight": 1},
                    "parallel": 4, "retry": {"max_attempts": 3, "backoff_ms": 100}}}
    r = validate_suite(s)
    assert r.valid, r.errors
