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
RESULTSET_VALIDATOR = Draft202012Validator(RESULTSET_SCHEMA, registry=_REGISTRY)


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
