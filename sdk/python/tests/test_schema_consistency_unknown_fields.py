"""Unknown fields: the strict default agrees with the raw JSON Schemas.

Issue #107 / Discussion #108 (PROPOSED -- reference implementation, DO NOT
MERGE). Extends test_schema_consistency.py (same machinery: spec/schemas/*.json
loaded with Draft202012Validator and a local referencing.Registry) to documents
with unknown keys, and pins openeval.validate.ALLOWED_KEYS to the schemas' own
property sets so the hardcoded sets cannot drift from the schemas.

Requires the `jsonschema` package (the `test` extra), like
test_schema_consistency.py.
"""
import json
import os

import pytest

jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402

from openeval.validate import (  # noqa: E402
    ALLOWED_KEYS,
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

_REGISTRY = Registry().with_resources(
    [
        (schema["$id"], Resource.from_contents(schema))
        for schema in (TESTCASE_SCHEMA, GRADER_SCHEMA, SUITE_SCHEMA, RESULTSET_SCHEMA)
    ]
)

TESTCASE_VALIDATOR = Draft202012Validator(TESTCASE_SCHEMA, registry=_REGISTRY)
GRADER_VALIDATOR = Draft202012Validator(GRADER_SCHEMA, registry=_REGISTRY)
SUITE_VALIDATOR = Draft202012Validator(SUITE_SCHEMA, registry=_REGISTRY)
RESULTSET_VALIDATOR = Draft202012Validator(
    RESULTSET_SCHEMA, registry=_REGISTRY, format_checker=Draft202012Validator.FORMAT_CHECKER
)


def _js_accepts(validator, doc):
    return len(list(validator.iter_errors(doc))) == 0


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


# ---------------------------------------------------------------------------
# Unknown fields (issue #107 / Discussion #108, PROPOSED -- DO NOT MERGE).
# Before this section, every document below with an undefined key was ACCEPTED
# by the hand-rolled validator and REJECTED by the raw JSON Schema
# (additionalProperties: false) -- the disagreement issue #107 reproduces. The
# strict default must now agree with the raw schema on every one of them, and
# the hardcoded ALLOWED_KEYS sets must equal the schemas' own property sets so
# the two cannot drift. Mirrors
# sdk/typescript/tests/schema-consistency-unknown-fields.test.ts.
# ---------------------------------------------------------------------------

# JSON Pointer (within each schema file) of every object ALLOWED_KEYS mirrors.
_ALLOWED_KEYS_SCHEMA_LOCATIONS = {
    "testcase": ("testcase", ""),
    "testcase.provider": ("testcase", "/properties/provider"),
    "grader": ("grader", ""),
    "suite": ("suite", ""),
    "suite.config": ("suite", "/properties/config"),
    "suite.config.provider": ("suite", "/properties/config/properties/provider"),
    "suite.config.defaults": ("suite", "/properties/config/properties/defaults"),
    "suite.config.retry": ("suite", "/properties/config/properties/retry"),
    "resultset": ("resultset", ""),
    "resultset.provider": ("resultset", "/properties/provider"),
    "resultset.runner": ("resultset", "/properties/runner"),
    "resultset.group": ("resultset", "/properties/group"),
    "resultset.summary": ("resultset", "/properties/summary"),
    "resultset.result": ("resultset", "/properties/results/items"),
    "resultset.result.error": ("resultset", "/properties/results/items/properties/error"),
    "resultset.grader_result": ("resultset", "/properties/results/items/properties/grader_results/items"),
}
_SCHEMAS_BY_NAME = {"testcase": TESTCASE_SCHEMA, "grader": GRADER_SCHEMA, "suite": SUITE_SCHEMA, "resultset": RESULTSET_SCHEMA}


def _resolve_pointer(schema, pointer):
    node = schema
    for part in [p for p in pointer.split("/") if p]:
        node = node[part]
    return node


def _closed_object_pointers(node, pointer=""):
    """Every sub-schema declaring additionalProperties: false, by JSON Pointer."""
    found = []
    if isinstance(node, dict):
        if node.get("additionalProperties") is False:
            found.append(pointer)
        for k, v in node.items():
            found += _closed_object_pointers(v, f"{pointer}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += _closed_object_pointers(v, f"{pointer}/{i}")
    return found


@pytest.mark.parametrize("key", sorted(_ALLOWED_KEYS_SCHEMA_LOCATIONS), ids=sorted(_ALLOWED_KEYS_SCHEMA_LOCATIONS))
def test_allowed_keys_equal_schema_properties(key):
    schema_name, pointer = _ALLOWED_KEYS_SCHEMA_LOCATIONS[key]
    node = _resolve_pointer(_SCHEMAS_BY_NAME[schema_name], pointer)
    assert node.get("additionalProperties") is False, f"{schema_name}.json{pointer or '/'} is not a closed object"
    assert set(ALLOWED_KEYS[key]) == set(node["properties"]), (
        f"ALLOWED_KEYS[{key!r}] drifted from {schema_name}.json{pointer or '/'}: "
        f"missing {sorted(set(node['properties']) - set(ALLOWED_KEYS[key]))}, "
        f"extra {sorted(set(ALLOWED_KEYS[key]) - set(node['properties']))}"
    )


def test_every_closed_schema_object_has_an_allowed_keys_entry():
    # A new object closed with additionalProperties: false (or a new key in
    # ALLOWED_KEYS with no schema counterpart) fails here, not in production.
    closed = {(name, ptr) for name, schema in _SCHEMAS_BY_NAME.items() for ptr in _closed_object_pointers(schema)}
    assert closed == set(_ALLOWED_KEYS_SCHEMA_LOCATIONS.values())
    assert set(ALLOWED_KEYS) == set(_ALLOWED_KEYS_SCHEMA_LOCATIONS)


def _rs_unknown(mutate):
    doc = _minimal_result_set("1.0.0")
    mutate(doc)
    return doc


def _suite_unknown(mutate):
    doc = _minimal_suite("1.0.0")
    mutate(doc)
    return doc


# (name, document, expected strict validity). Every unknown key sits directly on
# a closed object; every "open" key sits under metadata / provider.extra /
# params / a summary.by_grader entry.
UNKNOWN_FIELD_RESULTSET_CASES = [
    ("root verdict (issue #107)", _rs_unknown(lambda d: d.update(verdict="pass")), False),
    ("provider", _rs_unknown(lambda d: d.update(provider={"model": "m", "seed": 1})), False),
    ("runner", _rs_unknown(lambda d: d.update(runner={"name": "x", "commit": "abc"})), False),
    ("summary", _rs_unknown(lambda d: d.update(summary={"total": 1, "threshold": 0.8})), False),
    ("group", _rs_unknown(lambda d: d.update(group={"group_id": "g", "gruop_id": "typo"})), False),
    ("result (issue #107)", _rs_unknown(lambda d: d["results"][0].update(extra_result_key=1)), False),
    ("result.error", _rs_unknown(lambda d: d["results"][0].update(passed=False, error={"type": "timeout", "stack": "..."})), False),
    ("grader_result (issue #107)", _rs_unknown(lambda d: d["results"][0]["grader_results"][0].update(extra_gr_key=1)), False),
    ("every metadata open", _rs_unknown(lambda d: (
        d.update(metadata={"verdict": "pass", "nested": {"a": [1]}}),
        d["results"][0].update(metadata={"extra_result_key": 1}),
        d["results"][0]["grader_results"][0].update(metadata={"extra_gr_key": 1}),
    )), True),
    ("provider.extra open", _rs_unknown(lambda d: d.update(provider={"model": "m", "extra": {"seed": 1}})), True),
    ("summary.by_grader entry open", _rs_unknown(lambda d: d.update(summary={"by_grader": {"g1": {"passed": 1, "p95_ms": 3}}})), True),
]

UNKNOWN_FIELD_SUITE_CASES = [
    ("root threshold (issue #107)", _suite_unknown(lambda d: d.update(threshold=0.8)), False),
    ("config", _suite_unknown(lambda d: d.update(config={"concurrency": 2})), False),
    ("config.provider", _suite_unknown(lambda d: d.update(config={"provider": {"seed": 1}})), False),
    ("config.defaults", _suite_unknown(lambda d: d.update(config={"defaults": {"threshold": 0.5}})), False),
    ("config.retry", _suite_unknown(lambda d: d.update(config={"retry": {"jitter": True}})), False),
    ("test case (issue #107)", _suite_unknown(lambda d: d["test_cases"][0].update(extra_tc_key=1)), False),
    ("test case provider", _suite_unknown(lambda d: d["test_cases"][0].update(provider={"seed": 1})), False),
    ("shared grader config (llama-cookbook#1072)", _suite_unknown(lambda d: d["graders"][0].update(config={"x": 1})), False),
    ("inline grader", _suite_unknown(lambda d: d["test_cases"][0].update(graders=["g1", {"id": "g2", "type": "exact_match", "threshold": 1}])), False),
    ("metadata / params / provider.extra open", _suite_unknown(lambda d: (
        d.update(metadata={"threshold": 0.8}, config={"provider": {"extra": {"seed": 1}}}),
        d["graders"][0].update(params={"ignore_case": True, "custom_knob": 1}),
        d["test_cases"][0].update(metadata={"k": 1}, params={"top_p": 1}, provider={"extra": {"a": 1}}),
    )), True),
]


@pytest.mark.parametrize("name,doc,expected", UNKNOWN_FIELD_RESULTSET_CASES, ids=[c[0] for c in UNKNOWN_FIELD_RESULTSET_CASES])
def test_unknown_field_resultset_strict_default_agrees_with_json_schema(name, doc, expected):
    js_ok = _js_accepts(RESULTSET_VALIDATOR, doc)
    hand = validate_result_set(doc)
    assert js_ok == expected, f"{name}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand.valid == expected, f"{name}: strict hand-rolled acceptance was {hand.valid}, expected {expected}: {hand.errors}"
    if not expected:
        assert {e["code"] for e in hand.errors} == {"UNKNOWN_FIELD"}
    # Leniency (consumption mode) accepts every one of these documents.
    assert validate_result_set(doc, allow_unknown=True).valid


@pytest.mark.parametrize("name,doc,expected", UNKNOWN_FIELD_SUITE_CASES, ids=[c[0] for c in UNKNOWN_FIELD_SUITE_CASES])
def test_unknown_field_suite_strict_default_agrees_with_json_schema(name, doc, expected):
    js_ok = _js_accepts(SUITE_VALIDATOR, doc)
    hand = validate_suite(doc)
    assert js_ok == expected, f"{name}: JSON Schema acceptance was {js_ok}, expected {expected}"
    assert hand.valid == expected, f"{name}: strict hand-rolled acceptance was {hand.valid}, expected {expected}: {hand.errors}"
    if not expected:
        assert {e["code"] for e in hand.errors} == {"UNKNOWN_FIELD"}
    assert validate_suite(doc, allow_unknown=True).valid


@pytest.mark.parametrize("doc,expected", [
    ({"id": "tc1", "input": "hi", "graders": ["g1"], "extra_tc_key": 1}, False),
    ({"id": "tc1", "input": "hi", "graders": ["g1"], "provider": {"seed": 1}}, False),
    ({"id": "tc1", "input": "hi", "graders": ["g1"], "metadata": {"extra_tc_key": 1}, "params": {"x": 1}}, True),
], ids=["root", "provider", "metadata/params open"])
def test_unknown_field_testcase_strict_default_agrees_with_json_schema(doc, expected):
    assert _js_accepts(TESTCASE_VALIDATOR, doc) == expected
    assert validate_test_case(doc).valid == expected
    assert validate_test_case(doc, allow_unknown=True).valid


@pytest.mark.parametrize("doc,expected", [
    ({"id": "judge", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}"}, "config": {"temperature": 0}}, False),
    ({"id": "judge", "type": "llm_judge", "params": {"model": "m", "prompt": "{output}", "rubric_version": 2}}, True),
], ids=["config instead of params", "params open"])
def test_unknown_field_grader_strict_default_agrees_with_json_schema(doc, expected):
    assert _js_accepts(GRADER_VALIDATOR, doc) == expected
    assert validate_grader(doc).valid == expected
    assert validate_grader(doc, allow_unknown=True).valid


def test_group_unknown_subfield_rejected_by_both_paths_under_strict_default():
    # Supersedes the "the hand-rolled validator doesn't police unknown keys"
    # caveat in test_schema_consistency.py's
    # test_group_unknown_subfield_rejected_by_json_schema.
    doc = _minimal_result_set("1.0.0")
    doc["group"] = {"group_id": "g1", "not_a_real_field": "oops"}
    assert not _js_accepts(RESULTSET_VALIDATOR, doc)
    assert [(e["path"], e["code"]) for e in validate_result_set(doc).errors] == [("$.group.not_a_real_field", "UNKNOWN_FIELD")]
