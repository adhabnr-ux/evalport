"""
Cross-validates the raw JSON Schema files (spec/schemas/*.json -- the source of truth
that any JSON-Schema-based tool, not just this SDK, would validate against) against
the hand-rolled Python validator in openeval.validate.

These two validation paths are maintained independently (the hand-rolled validators
exist for zero-dependency, fast, structured-error validation; the JSON Schema files
exist as the portable, tool-agnostic spec artifact). History has already shown they
drift: this project's own SDK once accepted "1.0.0-rc.1" while its JSON Schema's
`version` pattern silently rejected it, and the JSON Schema's grader `allOf` blocks
declared per-type `params` requirements that were never actually enforced (a `then`
block that says "if params is present, it must have `substring`" says nothing about
whether `params` itself must be present -- so `{"type": "custom"}` with no `params`
at all passed the JSON Schema while the hand-rolled validator correctly rejected it
for missing `params.handler`).

This test suite is the regression guard against that class of drift: every case here
is checked against BOTH validation paths and must agree. If a future edit to either
side breaks that agreement, this test fails loudly instead of the drift being
discovered by a downstream tool disagreeing with this SDK in production.

Requires the `jsonschema` package (see pyproject.toml's `test` extra). Skipped
gracefully if it isn't installed, so environments that only care about the
hand-rolled validator's own unit tests aren't forced to add the dependency.
"""
import json
import os
import re

import pytest

jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402

from openeval.validate import (  # noqa: E402
    SEMVER_RE,
    is_rfc3339_date_time,
    validate_grader,
    validate_result_set,
    validate_suite,
    validate_test_case,
)

_SCHEMA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "spec", "schemas")
)


def _load_schema(name):
    with open(os.path.join(_SCHEMA_DIR, name)) as f:
        return json.load(f)


TESTCASE_SCHEMA = _load_schema("testcase.json")
GRADER_SCHEMA = _load_schema("grader.json")
SUITE_SCHEMA = _load_schema("suite.json")
RESULTSET_SCHEMA = _load_schema("resultset.json")

# suite.json and testcase.json both $ref grader.json/testcase.json by their $id
# URL (https://evalport.org/schema/*.json). Register all four schemas locally
# by that $id so $ref resolution works fully offline -- without this, any
# validator built from suite.json/testcase.json alone would try (and fail) to
# fetch grader.json over the network at validation time.
_REGISTRY = Registry().with_resources(
    [
        (schema["$id"], Resource.from_contents(schema))
        for schema in (TESTCASE_SCHEMA, GRADER_SCHEMA, SUITE_SCHEMA, RESULTSET_SCHEMA)
    ]
)

TESTCASE_VALIDATOR = Draft202012Validator(TESTCASE_SCHEMA, registry=_REGISTRY)
GRADER_VALIDATOR = Draft202012Validator(GRADER_SCHEMA, registry=_REGISTRY)
SUITE_VALIDATOR = Draft202012Validator(SUITE_SCHEMA, registry=_REGISTRY)
# resultset.json declares `format: "date-time"` on started_at/completed_at. Formats
# are annotation-only in Draft 2020-12 unless a format checker is passed, so the
# ResultSet validator asserts them explicitly -- that is what the spec means by
# date-time, and what the hand-rolled RFC 3339 check is compared against below.
RESULTSET_VALIDATOR = Draft202012Validator(
    RESULTSET_SCHEMA, registry=_REGISTRY, format_checker=Draft202012Validator.FORMAT_CHECKER
)


def _js_accepts(validator, doc):
    return len(list(validator.iter_errors(doc))) == 0


# ---------------------------------------------------------------------------
# Schema files are themselves well-formed Draft 2020-12 schemas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "schema",
    [TESTCASE_SCHEMA, GRADER_SCHEMA, SUITE_SCHEMA, RESULTSET_SCHEMA],
    ids=["testcase", "grader", "suite", "resultset"],
)
def test_schema_is_well_formed(schema):
    Draft202012Validator.check_schema(schema)

def test_testcase_empty_string_input_rejected_by_both_paths():
    doc = {"id": "tc1", "input": "", "graders": ["g1"]}

    assert not _js_accepts(TESTCASE_VALIDATOR, doc)

    result = validate_test_case(doc)
    assert not result.valid
    assert any(
        e["path"] == "$.input" and e["code"] == "MIN_LENGTH"
        for e in result.errors
    )
# ---------------------------------------------------------------------------
# Grader: type openness + params.handler requirement for non-standard types
# ---------------------------------------------------------------------------

GRADER_CASES = [
    ("well-known type, valid params", {"id": "g1", "type": "exact_match"}, True),
    (
        "well-known type (contains), missing required param",
        {"id": "g2", "type": "contains", "params": {}},
        False,
    ),
    (
        "well-known type (contains), no params object at all",
        {"id": "g3", "type": "contains"},
        False,
    ),
    ("custom, no params at all", {"id": "g4", "type": "custom"}, False),
    (
        "custom, with handler",
        {"id": "g5", "type": "custom", "params": {"handler": "my.module:fn"}},
        True,
    ),
    (
        "non-standard type, no params at all",
        {"id": "g6", "type": "trulens_feedback"},
        False,
    ),
    (
        "non-standard type, empty params (no handler)",
        {"id": "g7", "type": "trulens_feedback", "params": {}},
        False,
    ),
    (
        "non-standard type, with handler",
        {
            "id": "g8",
            "type": "trulens_feedback",
            "params": {"handler": "trulens.feedback:run"},
        },
        True,
    ),
    ("empty-string type", {"id": "g9", "type": ""}, False),
]


@pytest.mark.parametrize("name,doc,expected", GRADER_CASES, ids=[c[0] for c in GRADER_CASES])
def test_grader_json_schema_and_hand_rolled_validator_agree(name, doc, expected):
    js_ok = _js_accepts(GRADER_VALIDATOR, doc)
    hand_ok = validate_grader(doc).valid
    assert js_ok == expected, f"{name}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand_ok == expected, f"{name}: hand-rolled validator acceptance was {hand_ok}, expected {expected}"


# ---------------------------------------------------------------------------
# Suite / ResultSet: semver 2.0.0 `version` field
# ---------------------------------------------------------------------------

VERSION_CASES = [
    ("plain release", "1.0.0", True),
    ("legacy -draft suffix", "1.0.0-draft", True),
    ("numeric prerelease", "1.0.0-rc.1", True),
    ("alpha prerelease", "1.1.0-beta.2", True),
    ("build metadata", "1.0.0+build.5", True),
    ("prerelease + build metadata", "1.0.0-rc.1+build.5", True),
    ("garbage string", "garbage", False),
    ("missing patch component", "1.0", False),
    ("trailing dash, no prerelease identifier", "1.0.0-", False),
]


@pytest.mark.parametrize("name,version,expected", VERSION_CASES, ids=[c[0] for c in VERSION_CASES])
def test_suite_version_pattern_matches_sdk_semver_regex(name, version, expected):
    suite_pattern = SUITE_SCHEMA["properties"]["version"]["pattern"]
    assert bool(re.match(suite_pattern, version)) == expected, name
    assert bool(SEMVER_RE.match(version)) == expected, name


@pytest.mark.parametrize("name,version,expected", VERSION_CASES, ids=[c[0] for c in VERSION_CASES])
def test_resultset_version_pattern_matches_sdk_semver_regex(name, version, expected):
    resultset_pattern = RESULTSET_SCHEMA["properties"]["version"]["pattern"]
    assert bool(re.match(resultset_pattern, version)) == expected, name
    assert bool(SEMVER_RE.match(version)) == expected, name


def _minimal_suite(version):
    return {
        "version": version,
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}],
    }


def _minimal_result_set(version):
    return {
        "version": version,
        "suite_id": "s1",
        "run_id": "run1",
        "started_at": "2026-08-16T00:00:00Z",
        "results": [
            {
                "test_case_id": "tc1",
                "grader_results": [
                    {"grader_id": "g1", "type": "exact_match", "score": 0.9, "passed": True}
                ],
                "passed": True,
            }
        ],
    }


@pytest.mark.parametrize("name,version,expected", VERSION_CASES, ids=[c[0] for c in VERSION_CASES])
def test_suite_semver_end_to_end_agreement(name, version, expected):
    doc = _minimal_suite(version)
    js_ok = _js_accepts(SUITE_VALIDATOR, doc)
    hand_ok = validate_suite(doc).valid
    assert js_ok == expected, f"{name}: JSON Schema suite acceptance was {js_ok}, expected {expected}"
    assert hand_ok == expected, f"{name}: hand-rolled validate_suite acceptance was {hand_ok}, expected {expected}"


@pytest.mark.parametrize("name,version,expected", VERSION_CASES, ids=[c[0] for c in VERSION_CASES])
def test_resultset_semver_end_to_end_agreement(name, version, expected):
    doc = _minimal_result_set(version)
    js_ok = _js_accepts(RESULTSET_VALIDATOR, doc)
    hand_ok = validate_result_set(doc).valid
    assert js_ok == expected, f"{name}: JSON Schema resultset acceptance was {js_ok}, expected {expected}"
    assert hand_ok == expected, f"{name}: hand-rolled validate_result_set acceptance was {hand_ok}, expected {expected}"


# ---------------------------------------------------------------------------
# ResultSet: [0,1] score range enforcement
# ---------------------------------------------------------------------------

SCORE_CASES = [
    ("in-range score", 0.5, True),
    ("lower bound", 0.0, True),
    ("upper bound", 1.0, True),
    ("null (skipped/pending grader)", None, True),
    ("above range", 1.5, False),
    ("below range", -0.1, False),
]


@pytest.mark.parametrize("name,score,expected", SCORE_CASES, ids=[c[0] for c in SCORE_CASES])
def test_resultset_score_range_json_schema_and_hand_rolled_agree(name, score, expected):
    doc = {
        "version": "1.0.0",
        "suite_id": "s1",
        "run_id": "run1",
        "started_at": "2026-08-16T00:00:00Z",
        "results": [
            {
                "test_case_id": "tc1",
                "grader_results": [
                    {"grader_id": "g1", "type": "human", "score": score, "passed": False}
                ],
                "passed": False,
            }
        ],
    }
    js_ok = _js_accepts(RESULTSET_VALIDATOR, doc)
    hand_ok = validate_result_set(doc).valid
    assert js_ok == expected, f"{name}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand_ok == expected, f"{name}: hand-rolled validator acceptance was {hand_ok}, expected {expected}"


# ---------------------------------------------------------------------------
# ResultSet: per-result `completed_at` (added for resumable/partial runs,
# Discussion #10 -- https://github.com/adhabnr-ux/evalport/discussions/10).
# This is exactly the class of drift test_schema_consistency.py exists to catch:
# the hand-rolled validator never enforced additionalProperties, so it already
# silently accepted an unknown `completed_at` key on a result item -- but the
# JSON Schema's `additionalProperties: false` on that object would have REJECTED
# it until the schema was updated to declare the field. Before this schema change,
# this exact fixture would have failed js_ok while still passing hand_ok.
# ---------------------------------------------------------------------------

def test_result_completed_at_present_validates_in_both_paths():
    doc = {
        "version": "1.0.0",
        "suite_id": "s1",
        "run_id": "run1",
        "started_at": "2026-08-16T00:00:00Z",
        "results": [
            {
                "test_case_id": "tc1",
                "completed_at": "2026-08-16T00:00:05Z",
                "grader_results": [
                    {"grader_id": "g1", "type": "exact_match", "score": 1.0, "passed": True}
                ],
                "passed": True,
            }
        ],
    }
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid


def test_result_completed_at_absent_still_validates_in_both_paths():
    # Optional field -- a ResultSet from a runner that doesn't emit per-result
    # timestamps must remain fully valid.
    doc = _minimal_result_set("1.0.0")
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid
    assert "completed_at" not in doc["results"][0]


def test_resultset_partial_marker_via_metadata_validates_in_both_paths():
    # metadata.openeval.partial (Discussion #10) needs no schema change --
    # ResultSet.metadata already declares additionalProperties: true -- but this
    # fixture proves that end to end rather than just asserting it from reading
    # the schema.
    doc = _minimal_result_set("1.0.0")
    doc["metadata"] = {"openeval": {"partial": True}}
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid


def test_multi_attempt_resultset_valid_in_both_paths():
    # Discussion #22 / issue #20: multiple Results per test_case_id,
    # distinguished by ascending attempt, plus a single ResultSet-level
    # isolation. Mirrors spec/conformance/fixtures/multi_attempt_resultset_valid.json.
    doc = _minimal_result_set("1.0.0")
    doc["isolation"] = "fresh"
    doc["results"] = [
        {
            "test_case_id": "tc1",
            "attempt": 1,
            "grader_results": [{"grader_id": "g1", "type": "exact_match", "score": 1.0, "passed": True}],
            "passed": True,
        },
        {
            "test_case_id": "tc1",
            "attempt": 2,
            "grader_results": [{"grader_id": "g1", "type": "exact_match", "score": 0.0, "passed": False}],
            "passed": False,
        },
    ]
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid


def test_duplicate_test_case_id_run_id_attempt_rejected_by_hand_rolled_validator():
    # additionalProperties: false + the JSON Schema's own `minimum: 1` on
    # attempt does NOT (and structurally cannot) express a cross-item
    # uniqueness constraint like (test_case_id, run_id, attempt) -- that's a
    # hand-rolled-validator-only rule, by design, the same way DUPLICATE_ID for
    # suite test case ids is. So this fixture is intentionally checked against
    # only the hand-rolled path, not asserted to also fail the raw JSON Schema.
    doc = _minimal_result_set("1.0.0")
    doc["results"] = [
        {
            "test_case_id": "tc1",
            "attempt": 1,
            "grader_results": [{"grader_id": "g1", "type": "exact_match", "score": 1.0, "passed": True}],
            "passed": True,
        },
        {
            "test_case_id": "tc1",
            "attempt": 1,
            "grader_results": [{"grader_id": "g1", "type": "exact_match", "score": 0.0, "passed": False}],
            "passed": False,
        },
    ]
    result = validate_result_set(doc)
    assert not result.valid
    assert any(e["code"] == "DUPLICATE_ATTEMPT" for e in result.errors)


def test_resultset_isolation_absent_still_validates_in_both_paths():
    # Optional field -- backward compatibility for every ResultSet produced
    # before this change.
    doc = _minimal_result_set("1.0.0")
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid
    assert "isolation" not in doc
    assert "attempt" not in doc["results"][0]


# ---------------------------------------------------------------------------
# ResultSet: `group` (Discussion #45, proposed -- grouped/sibling ResultSets).
# additionalProperties: false on the ResultSet object means the raw JSON Schema
# would have rejected a `group` key before this schema change landed here,
# exactly the same class of drift test_result_completed_at_present_validates_
# in_both_paths documents above for `completed_at` -- both sides (schema +
# hand-rolled validator) must be updated together, which is what this section
# checks.
# ---------------------------------------------------------------------------

def test_group_with_group_id_only_valid_in_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"group_id": "mutation-sweep-2026-09-01"}
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid


def test_group_with_all_fields_valid_in_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {
        "group_id": "mutation-sweep-2026-09-01",
        "role": "mutant",
        "label": "mutant_017 (relational-operator-swap in billing.py:42)",
        "sequence": 17,
    }
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid


def test_group_absent_still_valid_in_both_paths():
    # Optional field -- backward compatibility for every ResultSet produced
    # before this proposal.
    doc = _minimal_result_set("1.0.0")
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid
    assert "group" not in doc


def test_group_missing_group_id_rejected_by_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"role": "mutant"}  # group_id is REQUIRED when group is present
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)
    assert not validate_result_set(doc).valid


def test_group_unknown_subfield_rejected_by_json_schema():
    # additionalProperties: false on the group object itself -- a typo'd
    # sub-field (e.g. "gruop_id") must be caught structurally by the JSON
    # Schema even though the hand-rolled validator (like every other optional
    # object in this file) doesn't police unknown keys.
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"group_id": "g1", "not_a_real_field": "oops"}
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)


def test_group_sequence_must_be_non_negative_integer_in_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"group_id": "g1", "sequence": -1}
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)
    assert not validate_result_set(doc).valid


def test_group_wrong_type_rejected_by_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = "mutation-sweep-2026-09-01"  # must be an object, not a string
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)
    assert not validate_result_set(doc).valid


# ---------------------------------------------------------------------------
# group.parent_group_id (nested/hierarchical groups -- sweep-of-sweeps).
# Same drift class as the rest of this file: additionalProperties: false on
# the group object means the raw JSON Schema would reject parent_group_id
# entirely until the schema itself was updated alongside the hand-rolled
# validator. Mirrors sdk/typescript/tests/schema-consistency.test.ts's
# parent_group_id section rule-for-rule.
# ---------------------------------------------------------------------------

def test_group_parent_group_id_valid_nested_sweep_agrees_in_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {
        "group_id": "child-sweep-lr-1e-4",
        "parent_group_id": "parent-sweep-lr-batchsize-grid-2026-09-08",
        "role": "candidate",
        "sequence": 3,
    }
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid


def test_group_parent_group_id_absent_still_valid_in_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"group_id": "mutation-sweep-2026-09-01"}
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert validate_result_set(doc).valid
    assert "parent_group_id" not in doc["group"]


def test_group_parent_group_id_empty_string_rejected_by_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"group_id": "g1", "parent_group_id": ""}
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)
    assert not validate_result_set(doc).valid


def test_group_parent_group_id_wrong_type_rejected_by_both_paths():
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"group_id": "g1", "parent_group_id": 42}
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)
    assert not validate_result_set(doc).valid


def test_group_parent_group_id_equal_to_group_id_json_schema_allows_hand_rolled_rejects():
    # The self-parent rule (a group cannot be its own parent) is a
    # cross-field constraint JSON Schema's properties/required vocabulary
    # cannot express without a $data reference (not part of this project's
    # supported draft usage elsewhere in the schema) -- so, same as
    # uniqueness rules like DUPLICATE_ATTEMPT elsewhere in this suite, this
    # is intentionally enforced only by the hand-rolled validator. Documented
    # here (rather than silently skipped) so a future schema change that
    # *does* add a $data-based check is a deliberate decision, not a
    # rediscovery.
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"group_id": "sweep-42", "parent_group_id": "sweep-42"}
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    assert not validate_result_set(doc).valid


def test_boolean_score_rejected_by_both_python_bool_is_int_subclass():
    # Python's bool is a subclass of int, so a naive `isinstance(x, (int, float))`
    # range check would silently accept True/False as scores 1/0. Guard against
    # regressing that fix on the hand-rolled side; the JSON Schema's own `type`
    # keyword already excludes booleans from `["number", "null"]` structurally.
    doc = {
        "version": "1.0.0",
        "suite_id": "s1",
        "run_id": "run1",
        "started_at": "2026-08-16T00:00:00Z",
        "results": [
            {
                "test_case_id": "tc1",
                "grader_results": [
                    {"grader_id": "g1", "type": "human", "score": True, "passed": True}
                ],
                "passed": True,
            }
        ],
    }
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)
    assert not validate_result_set(doc).valid


# ---------------------------------------------------------------------------
# Validator fidelity: optional-field types, RFC 3339 date-times, and the Rule 6
# null-score/passed rules. Before this section the hand-rolled validator accepted
# every "reject" document below that the raw JSON Schema rejects -- e.g.
# Result.actual_output as a list (found by ChelseaKR in ChelseaKR/gauntlet#76),
# a date-only started_at, or a string temperature. Unknown-field
# (additionalProperties) handling is deliberately NOT covered here -- that is a
# separate spec question. Mirrors sdk/typescript/tests/schema-consistency.test.ts.
# ---------------------------------------------------------------------------

def test_date_time_format_is_actually_asserted_by_json_schema_path():
    # Without rfc3339-validator installed, jsonschema silently skips
    # format: date-time and every date-time agreement case below would be
    # vacuous on the JSON Schema side. Fail loudly instead.
    assert "date-time" in Draft202012Validator.FORMAT_CHECKER.checkers
    doc = _minimal_result_set("1.0.0")
    doc["started_at"] = "2026-08-16"
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)


def _rs_with(mutate):
    doc = _minimal_result_set("1.0.0")
    mutate(doc)
    return doc


def _r0(doc):
    return doc["results"][0]


def _gr0(doc):
    return doc["results"][0]["grader_results"][0]


RESULTSET_FIDELITY_CASES = [
    # (name, mutate, expected_valid)
    ("suite_version number", lambda d: d.update(suite_version=1), False),
    ("completed_at date-only", lambda d: d.update(completed_at="2026-08-16"), False),
    ("completed_at number", lambda d: d.update(completed_at=1700000000), False),
    ("provider string", lambda d: d.update(provider="gpt-4o"), False),
    ("provider null", lambda d: d.update(provider=None), False),
    ("provider.model number", lambda d: d.update(provider={"model": 4}), False),
    ("provider.api_base list", lambda d: d.update(provider={"api_base": ["x"]}), False),
    ("provider.temperature string", lambda d: d.update(provider={"temperature": "0.1"}), False),
    ("provider.temperature bool", lambda d: d.update(provider={"temperature": True}), False),
    ("provider.max_tokens fractional", lambda d: d.update(provider={"max_tokens": 1.5}), False),
    ("provider.max_tokens bool", lambda d: d.update(provider={"max_tokens": True}), False),
    ("provider.extra list", lambda d: d.update(provider={"extra": []}), False),
    ("runner list", lambda d: d.update(runner=[]), False),
    ("runner.name number", lambda d: d.update(runner={"name": 1}), False),
    ("runner.version number", lambda d: d.update(runner={"version": 1.0}), False),
    ("summary list", lambda d: d.update(summary=[]), False),
    ("summary.total negative", lambda d: d.update(summary={"total": -1}), False),
    ("summary.total bool", lambda d: d.update(summary={"total": True}), False),
    ("summary.passed fractional", lambda d: d.update(summary={"passed": 0.5}), False),
    ("summary.pass_rate above 1", lambda d: d.update(summary={"pass_rate": 1.5}), False),
    ("summary.avg_score string", lambda d: d.update(summary={"avg_score": "0.5"}), False),
    ("summary.duration_ms negative", lambda d: d.update(summary={"duration_ms": -5}), False),
    ("summary.by_grader list", lambda d: d.update(summary={"by_grader": []}), False),
    ("summary.by_grader entry number", lambda d: d.update(summary={"by_grader": {"g1": 3}}), False),
    ("summary.by_grader.passed fractional", lambda d: d.update(summary={"by_grader": {"g1": {"passed": 1.5}}}), False),
    ("summary.by_grader.avg_score string", lambda d: d.update(summary={"by_grader": {"g1": {"avg_score": "x"}}}), False),
    ("metadata list", lambda d: d.update(metadata=[]), False),
    ("isolation null", lambda d: d.update(isolation=None), False),
    ("group null", lambda d: d.update(group=None), False),
    ("actual_output list (ChelseaKR/gauntlet#76)", lambda d: _r0(d).update(actual_output=["a", "b"]), False),
    ("actual_output null", lambda d: _r0(d).update(actual_output=None), False),
    ("test_case_id empty", lambda d: _r0(d).update(test_case_id=""), False),
    ("duration_ms fractional", lambda d: _r0(d).update(duration_ms=1.5), False),
    ("duration_ms negative", lambda d: _r0(d).update(duration_ms=-1), False),
    ("duration_ms bool", lambda d: _r0(d).update(duration_ms=True), False),
    ("result completed_at offset-less", lambda d: _r0(d).update(completed_at="2026-08-16T00:00:05"), False),
    ("error string", lambda d: _r0(d).update(error="boom"), False),
    ("error.type not in enum", lambda d: _r0(d).update(error={"type": "crash"}), False),
    ("error.message number", lambda d: _r0(d).update(error={"message": 1}), False),
    ("error.code fractional", lambda d: _r0(d).update(error={"code": 1.5}), False),
    ("error.code bool", lambda d: _r0(d).update(error={"code": True}), False),
    ("error.retryable string", lambda d: _r0(d).update(error={"retryable": "yes"}), False),
    ("result metadata string", lambda d: _r0(d).update(metadata="trace"), False),
    ("attempt null", lambda d: _r0(d).update(attempt=None), False),
    ("grader_id empty", lambda d: _gr0(d).update(grader_id=""), False),
    ("score missing", lambda d: (_gr0(d).pop("score"), _gr0(d).update(passed=False), _r0(d).update(passed=False)), False),
    ("reason number", lambda d: _gr0(d).update(reason=1), False),
    ("grader metadata list", lambda d: _gr0(d).update(metadata=[]), False),
    # Valid documents must stay valid in both paths.
    ("every optional field well-typed", lambda d: (
        d.update({
            "$schema": "https://evalport.org/schema/resultset.json",
            "suite_version": "1.0.0",
            "completed_at": "2026-08-16T00:01:45.123+05:30",
            "provider": {"model": "gpt-4o", "api_base": "https://api.example.com", "temperature": 0, "max_tokens": 256, "extra": {"seed": 1}},
            "runner": {"name": "evalport-cli", "version": "1.3.1"},
            "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "pass_rate": 1, "avg_score": 0.9, "duration_ms": 1200,
                        "by_grader": {"g1": {"passed": 1, "failed": 0, "avg_score": 0.9}}},
            "metadata": {"openeval.partial": False},
        }),
        _r0(d).update({
            "actual_output": "Paris", "duration_ms": 1200, "completed_at": "2026-08-16t00:01:44z",
            "error": {"type": "provider_error", "message": "rate limited", "code": 429, "retryable": True},
            "metadata": {"trace_id": "abc"},
        }),
        _gr0(d).update({"reason": "close enough", "metadata": {"k": "v"}}),
    ), True),
    ("error.code string", lambda d: _r0(d).update(error={"type": "timeout", "code": "ETIMEDOUT"}), True),
    ("integral float duration_ms", lambda d: _r0(d).update(duration_ms=1200.0), True),
    ("negative temperature allowed (no schema minimum)", lambda d: d.update(provider={"temperature": -1}), True),
    # PROPOSED (Discussion #49, alt B): a partly-scored row must declare its
    # aggregation (PARTIAL_RESULT_UNDECLARED); both validators enforce it.
    ("mixed null and scored graders, passed, undeclared", lambda d: _r0(d)["grader_results"].append(
        {"grader_id": "g2", "type": "human", "score": None, "passed": False}), False),
    ("mixed null and scored graders, passed, declared on the Result (dotted)", lambda d: (
        _r0(d)["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": False}),
        _r0(d).update(metadata={"openeval.aggregation": {"strategy": "all"}})), True),
    ("mixed null and scored graders, passed, declared on the ResultSet (nested)", lambda d: (
        _r0(d)["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": False}),
        d.update(metadata={"openeval": {"aggregation": {"strategy": "strict"}}})), True),
    ("mixed null and scored graders, passed false, undeclared", lambda d: (
        _r0(d).update(passed=False),
        _r0(d)["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": False})), False),
    ("declared aggregation with unknown strategy", lambda d: (
        _r0(d)["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": False}),
        d.update(metadata={"openeval": {"aggregation": {"strategy": "fail-closed"}}})), False),
    ("declared weighted aggregation without threshold", lambda d: (
        _r0(d)["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": False}),
        _r0(d).update(metadata={"openeval.aggregation": {"strategy": "weighted"}})), False),
    ("declared weighted aggregation with threshold", lambda d: (
        _r0(d)["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": False}),
        _r0(d).update(metadata={"openeval.aggregation": {"strategy": "weighted", "threshold": 0.5}})), True),
    ("declared aggregation threshold out of range", lambda d: d.update(
        metadata={"openeval": {"aggregation": {"strategy": "all", "threshold": 2}}}), False),
    ("declared aggregation not an object", lambda d: d.update(metadata={"openeval.aggregation": "all"}), False),
    ("aggregation_status partial on a mixed, declared row", lambda d: (
        _r0(d)["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": False}),
        d.update(metadata={"openeval": {"aggregation": {"strategy": "all"}}}),
        _r0(d).update(metadata={"openeval": {"aggregation_status": "partial"}})), True),
    ("aggregation_status partial on a fully scored row", lambda d: _r0(d).update(
        metadata={"openeval.aggregation_status": "partial"}), False),
    ("aggregation_status unscored on a scored row", lambda d: _r0(d).update(
        metadata={"openeval": {"aggregation_status": "unscored"}}), False),
    ("aggregation_status unknown value", lambda d: _r0(d).update(
        metadata={"openeval": {"aggregation_status": "pending"}}), False),
    ("aggregation_status unscored on an all-null row", lambda d: _r0(d).update(
        passed=False, grader_results=[{"grader_id": "g1", "type": "human", "score": None, "passed": False}],
        metadata={"openeval": {"aggregation_status": "unscored"}}), True),
    ("all null-scored, passed false", lambda d: (
        _r0(d).update(passed=False, grader_results=[{"grader_id": "g1", "type": "human", "score": None, "passed": False}])), True),
]


@pytest.mark.parametrize("name,mutate,expected", RESULTSET_FIDELITY_CASES, ids=[c[0] for c in RESULTSET_FIDELITY_CASES])
def test_resultset_fidelity_json_schema_and_hand_rolled_agree(name, mutate, expected):
    doc = _rs_with(mutate)
    js_ok = _js_accepts(RESULTSET_VALIDATOR, doc)
    hand = validate_result_set(doc)
    assert js_ok == expected, f"{name}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand.valid == expected, f"{name}: hand-rolled acceptance was {hand.valid}, expected {expected}: {hand.errors}"


# RFC 3339 date-time values on which the hand-rolled check, jsonschema's
# FORMAT_CHECKER (rfc3339-validator) and ajv-formats (the TypeScript twin of this
# test) all agree. Deliberately excluded because the two JSON Schema format
# implementations disagree with RFC 3339 / each other, not with this SDK:
#   "2016-12-31T23:59:60Z" (leap second; RFC 3339 grammar allows it, rfc3339-validator doesn't)
#   "2026-01-15T10:30:00Z\n" (rfc3339-validator's `$` matches before a trailing newline)
#   "2026-01-15 10:30:00Z", "...+0530", "...+05" (ajv-formats accepts; RFC 3339 does not)
DATE_TIME_CASES = [
    ("2026-01-15T10:30:00Z", True),
    ("2026-01-15T10:30:00+05:30", True),
    ("2026-01-15T10:30:00-08:00", True),
    ("2026-01-15t10:30:00z", True),
    ("2026-01-15T10:30:00.123456Z", True),
    ("2024-02-29T00:00:00Z", True),
    ("2026-01-15T10:30:00.5+00:00", True),
    ("2026-01-15", False),
    ("2026-01-15T10:30:00", False),
    ("2026-01-15T10:30:05", False),
    ("2026-01-15T10:30Z", False),
    ("2025-02-29T00:00:00Z", False),
    ("2026-02-30T10:30:00Z", False),
    ("2026-04-31T00:00:00Z", False),
    ("2026-00-10T00:00:00Z", False),
    ("2026-13-01T10:30:00Z", False),
    ("2026-01-00T00:00:00Z", False),
    ("2026-01-15T24:00:00Z", False),
    ("2026-01-15T10:60:00Z", False),
    ("2026-01-15T10:30:61Z", False),
    ("2026-01-15T10:30:00.Z", False),
    ("20260115T103000Z", False),
    ("1700000000", False),
    ("now", False),
    ("", False),
]


@pytest.mark.parametrize("value,expected", DATE_TIME_CASES, ids=[repr(c[0]) for c in DATE_TIME_CASES])
@pytest.mark.parametrize("where", ["started_at", "completed_at", "result.completed_at"])
def test_date_time_json_schema_and_hand_rolled_agree(where, value, expected):
    doc = _minimal_result_set("1.0.0")
    if where == "result.completed_at":
        doc["results"][0]["completed_at"] = value
    else:
        doc[where] = value
    assert is_rfc3339_date_time(value) == expected
    js_ok = _js_accepts(RESULTSET_VALIDATOR, doc)
    hand_ok = validate_result_set(doc).valid
    assert js_ok == expected, f"{where}={value!r}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand_ok == expected, f"{where}={value!r}: hand-rolled acceptance was {hand_ok}, expected {expected}"


def test_null_score_passed_true_json_schema_allows_hand_rolled_rejects():
    # SPEC.md Validation Rule 6: "Skipped or not-yet-executed graders [...] MUST
    # be represented with `score: null` and `passed: false`." resultset.json
    # types score and passed independently and does not encode this cross-field
    # rule, so -- like SELF_PARENT and DUPLICATE_ATTEMPT above -- it is enforced
    # only by the hand-rolled validators (NULL_SCORE_PASSED).
    # PROPOSED (Discussion #49, alt B): the row is also mixed, so without a
    # declaration both paths reject it; the run-level declaration isolates Rule 6.
    doc = _minimal_result_set("1.0.0")
    doc["results"][0]["grader_results"].append({"grader_id": "g2", "type": "human", "score": None, "passed": True})
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)
    assert [e["code"] for e in validate_result_set(doc).errors] == ["NULL_SCORE_PASSED", "PARTIAL_RESULT_UNDECLARED"]
    doc["metadata"] = {"openeval": {"aggregation": {"strategy": "all"}}}
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    result = validate_result_set(doc)
    assert not result.valid
    assert [e["code"] for e in result.errors] == ["NULL_SCORE_PASSED"]


# PROPOSED (Discussion #49, alt B): exhaustive agreement between the JSON Schema
# and the hand-rolled validator over grader shapes x passed x Result metadata x
# ResultSet metadata. Mirrors sdk/typescript/tests/schema-consistency.test.ts.
def _altb_truth_table():
    import copy
    G = lambda score, passed=True: {"grader_id": "g", "type": "custom", "score": score, "passed": passed}
    shapes = {"mixed": [G(1.0), G(None, False)], "single": [G(1.0)], "allnull": [G(None, False)], "empty": []}
    decls = {
        "none": None,
        "dot": {"openeval.aggregation": {"strategy": "all"}},
        "nested": {"openeval": {"aggregation": {"strategy": "strict"}}},
        "bad_strategy": {"openeval": {"aggregation": {"strategy": "fail-closed"}}},
        "weighted_no_thr": {"openeval.aggregation": {"strategy": "weighted"}},
        "weighted_ok": {"openeval.aggregation": {"strategy": "weighted", "threshold": 0.5}},
        "thr_oob": {"openeval": {"aggregation": {"strategy": "all", "threshold": 2}}},
        "status_partial": {"openeval": {"aggregation_status": "partial"}},
        "status_partial_dot": {"openeval.aggregation_status": "partial"},
        "status_unscored": {"openeval": {"aggregation_status": "unscored"}},
        "status_bad": {"openeval.aggregation_status": "pending"},
        "not_obj": {"openeval.aggregation": "all"},
        "both": {"openeval": {"aggregation": {"strategy": "producer"}, "aggregation_status": "partial"}},
    }
    for shape, graders in shapes.items():
        for passed in (True, False):
            if shape == "allnull" and passed:
                continue  # UNSCORED_RESULT_PASSED, hand-rolled only (tested above)
            for rk, rm in decls.items():
                for kk, km in decls.items():
                    d = {"version": "1.0.0", "suite_id": "s", "run_id": "r", "started_at": "2026-01-15T10:30:00Z",
                         "results": [{"test_case_id": "t", "passed": passed, "grader_results": copy.deepcopy(graders)}]}
                    if rm is not None: d["results"][0]["metadata"] = copy.deepcopy(rm)
                    if km is not None: d["metadata"] = copy.deepcopy(km)
                    yield f"{shape} passed={passed} result={rk} run={kk}", d


def test_altb_truth_table_json_schema_and_hand_rolled_agree():
    rows = list(_altb_truth_table())
    assert len(rows) == 7 * 13 * 13
    disagreements = [name for name, d in rows if _js_accepts(RESULTSET_VALIDATOR, d) != validate_result_set(d).valid]
    assert disagreements == []


def test_all_null_scored_result_passed_true_json_schema_allows_hand_rolled_rejects():
    # SPEC.md Aggregation Extension: "A test case whose graders are *all*
    # null-scored has no basis for a pass/fail verdict; runners MUST report such
    # a case's `passed` as `false`". Hand-rolled only (UNSCORED_RESULT_PASSED).
    doc = _minimal_result_set("1.0.0")
    doc["results"][0]["grader_results"] = [{"grader_id": "g1", "type": "human", "score": None, "passed": False}]
    doc["results"][0]["passed"] = True
    assert _js_accepts(RESULTSET_VALIDATOR, doc)
    result = validate_result_set(doc)
    assert not result.valid
    assert [e["code"] for e in result.errors] == ["UNSCORED_RESULT_PASSED"]


TESTCASE_FIDELITY_CASES = [
    ("expected_output number", {"expected_output": 1}, False),
    ("context string", {"context": "doc"}, False),
    ("context non-string item", {"context": ["ok", 2]}, False),
    ("retrieval_context null item", {"retrieval_context": [None]}, False),
    ("tools_called string", {"tools_called": "search"}, False),
    ("expected_tools object item", {"expected_tools": [{"name": "search"}]}, False),
    ("metadata list", {"metadata": []}, False),
    ("tags string", {"tags": "smoke"}, False),
    ("provider string", {"provider": "gpt-4o"}, False),
    ("provider.max_tokens 0", {"provider": {"max_tokens": 0}}, False),
    ("provider.api_key_env number", {"provider": {"api_key_env": 1}}, False),
    ("params list", {"params": []}, False),
    ("timeout_ms 0", {"timeout_ms": 0}, False),
    ("weight negative", {"weight": -1}, False),
    ("weight bool", {"weight": True}, False),
    ("every optional field well-typed", {
        "expected_output": "Paris", "context": ["a"], "retrieval_context": ["b"], "tools_called": ["search"],
        "expected_tools": ["search"], "metadata": {"k": 1}, "tags": ["smoke"],
        "provider": {"model": "gpt-4o", "api_base": "x", "api_key_env": "OPENAI_API_KEY", "temperature": 0.2, "max_tokens": 1, "extra": {}},
        "params": {"top_p": 1}, "timeout_ms": 1, "weight": 0,
    }, True),
]


@pytest.mark.parametrize("name,extra,expected", TESTCASE_FIDELITY_CASES, ids=[c[0] for c in TESTCASE_FIDELITY_CASES])
def test_testcase_fidelity_json_schema_and_hand_rolled_agree(name, extra, expected):
    doc = {"id": "tc1", "input": "hi", "graders": ["g1"], **extra}
    js_ok = _js_accepts(TESTCASE_VALIDATOR, doc)
    hand = validate_test_case(doc)
    assert js_ok == expected, f"{name}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand.valid == expected, f"{name}: hand-rolled acceptance was {hand.valid}, expected {expected}: {hand.errors}"


GRADER_FIDELITY_CASES = [
    ("params list", {"id": "g", "type": "exact_match", "params": []}, False),
    ("weight negative", {"id": "g", "type": "exact_match", "weight": -0.5}, False),
    ("weight string", {"id": "g", "type": "exact_match", "weight": "1"}, False),
    ("description number", {"id": "g", "type": "exact_match", "description": 1}, False),
    ("contains.ignore_case string", {"id": "g", "type": "contains", "params": {"substring": "a", "ignore_case": "yes"}}, False),
    ("regex.flags number", {"id": "g", "type": "regex", "params": {"pattern": "a", "flags": 1}}, False),
    ("semantic_similarity.threshold bool", {"id": "g", "type": "semantic_similarity", "params": {"threshold": True}}, False),
    ("semantic_similarity.model number", {"id": "g", "type": "semantic_similarity", "params": {"threshold": 0.8, "model": 1}}, False),
    ("llm_judge.temperature above 2", {"id": "g", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}", "temperature": 3}}, False),
    ("llm_judge.schema string", {"id": "g", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}", "schema": "x"}}, False),
    ("json_schema.strict string", {"id": "g", "type": "json_schema", "params": {"schema": {}, "strict": "true"}}, False),
    ("json_path.operator not in enum", {"id": "g", "type": "json_path", "params": {"path": "$.a", "expected": "1", "operator": "like"}}, False),
    ("code.timeout_ms below 100", {"id": "g", "type": "code", "params": {"language": "python", "source": "x", "timeout_ms": 50}}, False),
    ("contains, all optional params well-typed", {"id": "g", "type": "contains", "params": {"substring": "a", "ignore_case": True}, "weight": 2, "description": "d"}, True),
    ("llm_judge, all optional params well-typed", {"id": "g", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}", "provider": "p", "temperature": 2, "schema": {}}}, True),
    ("json_path, operator in enum", {"id": "g", "type": "json_path", "params": {"path": "$.a", "expected": "1", "operator": "gte"}}, True),
    ("code, timeout_ms at minimum", {"id": "g", "type": "code", "params": {"language": "python", "source": "x", "timeout_ms": 100}}, True),
]


@pytest.mark.parametrize("name,doc,expected", GRADER_FIDELITY_CASES, ids=[c[0] for c in GRADER_FIDELITY_CASES])
def test_grader_fidelity_json_schema_and_hand_rolled_agree(name, doc, expected):
    js_ok = _js_accepts(GRADER_VALIDATOR, doc)
    hand = validate_grader(doc)
    assert js_ok == expected, f"{name}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand.valid == expected, f"{name}: hand-rolled acceptance was {hand.valid}, expected {expected}: {hand.errors}"


SUITE_FIDELITY_CASES = [
    ("name number", {"name": 1}, False),
    ("description list", {"description": ["x"]}, False),
    ("graders object", {"graders": {"g1": {}}}, False),
    ("metadata string", {"metadata": "x"}, False),
    ("tags non-string item", {"tags": [1]}, False),
    ("config list", {"config": []}, False),
    ("config.parallel 0", {"config": {"parallel": 0}}, False),
    ("config.provider.max_tokens 0", {"config": {"provider": {"max_tokens": 0}}}, False),
    ("config.defaults.timeout_ms 0", {"config": {"defaults": {"timeout_ms": 0}}}, False),
    ("config.defaults.weight negative", {"config": {"defaults": {"weight": -1}}}, False),
    ("config.retry.max_attempts 0", {"config": {"retry": {"max_attempts": 0}}}, False),
    ("config.retry.backoff_ms below 100", {"config": {"retry": {"backoff_ms": 10}}}, False),
    ("nested test case tags string", {"test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"], "tags": "smoke"}]}, False),
    ("every optional field well-typed", {
        "$schema": "https://evalport.org/schema/suite.json", "name": "n", "description": "d", "metadata": {"k": 1}, "tags": ["a"],
        "config": {"provider": {"model": "m", "max_tokens": 10}, "defaults": {"timeout_ms": 1000, "weight": 1},
                   "parallel": 4, "retry": {"max_attempts": 3, "backoff_ms": 100}},
    }, True),
]


@pytest.mark.parametrize("name,extra,expected", SUITE_FIDELITY_CASES, ids=[c[0] for c in SUITE_FIDELITY_CASES])
def test_suite_fidelity_json_schema_and_hand_rolled_agree(name, extra, expected):
    doc = {**_minimal_suite("1.0.0"), **extra}
    js_ok = _js_accepts(SUITE_VALIDATOR, doc)
    hand = validate_suite(doc)
    assert js_ok == expected, f"{name}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand.valid == expected, f"{name}: hand-rolled acceptance was {hand.valid}, expected {expected}: {hand.errors}"
