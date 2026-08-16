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
