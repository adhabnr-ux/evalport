"""Tests for openeval.converters_inspect.from_inspect(), verified against a real
`inspect-ai` install and the real openeval.validate.validate_suite().

from_inspect() takes a plain dict shape: {"task", "samples": [{"id", "input",
"target", "metadata"}], "scorers": [...], "model"}. Rather than guess at that shape,
this file's fixtures come from running a real `inspect_ai.eval()` call --

    Task(dataset=[Sample(input=..., target=..., id=..., metadata=...)],
         solver=[generate()], scorer=exact())
    eval(task, model="mockllm/model")

-- and reading the resulting real EvalLog: `log.eval.task` ("demo"), `log.eval.model`
("mockllm/model"), `log.results.scores[i].name` ("exact" -- confirms the real scorer
name inspect_ai reports is the bare string "exact", matching the "exact" in s
substring check in _inspect_scorer_to_grader), and each `log.samples[i]`'s real `.id`,
`.input`, `.target`, `.metadata` fields (all present, all matching the field names
this converter reads). These tests build from_inspect() inputs shaped exactly like
that real log, confirmed field-for-field against the installed package rather than
assumed from inspect_ai's docs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("inspect_ai", reason="requires a real inspect-ai install")

from openeval.converters_inspect import from_inspect  # noqa: E402
from openeval.validate import validate_suite  # noqa: E402

# Real values read off a live `inspect_ai.eval(Task(dataset=[Sample(...)], scorer=exact()),
# model="mockllm/model")` run: log.eval.task == "demo", log.eval.model == "mockllm/model",
# log.results.scores[0].name == "exact", and per-sample .id/.input/.target/.metadata below.
REAL_INSPECT_EXPORT = {
    "task": "demo",
    "model": "mockllm/model",
    "scorers": ["exact"],
    "samples": [
        {"id": "s1", "input": "What is 2+2?", "target": "4", "metadata": {"topic": "math"}},
        {"id": "s2", "input": "Capital of France?", "target": "Paris", "metadata": {}},
    ],
}


def test_real_inspect_export_produces_spec_valid_suite():
    suite = from_inspect(REAL_INSPECT_EXPORT)
    result = validate_suite(suite)
    assert result.valid, result.errors
    assert len(suite["test_cases"]) == 2
    assert suite["id"] == "suite_inspect_demo"
    assert suite["name"] == "Imported from Inspect AI: demo"


def test_real_exact_scorer_name_maps_to_exact_match_grader():
    """Confirms the literal scorer name inspect_ai reports at runtime ("exact",
    not "ExactScorer" or similar) is what this converter's substring match expects."""
    suite = from_inspect(REAL_INSPECT_EXPORT)
    assert suite["graders"] == [{"id": "gr_0", "type": "exact_match"}]


def test_model_is_carried_into_suite_config_provider():
    suite = from_inspect(REAL_INSPECT_EXPORT)
    assert suite["config"]["provider"]["model"] == "mockllm/model"


def test_sample_id_input_target_metadata_all_round_trip():
    suite = from_inspect(REAL_INSPECT_EXPORT)
    tc = suite["test_cases"][0]
    assert tc["id"] == "s1"
    assert tc["input"] == "What is 2+2?"
    assert tc["expected_output"] == "4"
    assert tc["metadata"] == {"topic": "math"}


def test_sample_without_target_has_no_expected_output_key():
    data = {"task": "demo", "samples": [{"id": "s3", "input": "no target here"}], "scorers": ["exact"]}
    suite = from_inspect(data)
    assert "expected_output" not in suite["test_cases"][0]
    result = validate_suite(suite)
    assert result.valid, result.errors


@pytest.mark.parametrize(
    "scorer_name,expected_type",
    [
        ("exact", "exact_match"),
        ("match", "custom"),  # bare "match" isn't a recognized alias, falls through to custom
        # includes/contains and pattern/regex both route to "custom": the real
        # substring/pattern each checks is per-sample (or scorer-construction-time)
        # data this converter never receives, so EvalPort's native `contains`/`regex`
        # grader types -- which require a real non-empty substring/pattern -- would
        # only be reachable by fabricating one. See _inspect_scorer_to_grader's
        # docstring-comments for the full reasoning; this was a real bug (an
        # always-true `substring: ""` / `pattern: ".*"` no-op) caught by this test
        # failing against the real validator before the fix.
        ("includes", "custom"),
        ("pattern", "custom"),
        ("model_graded_qa", "llm_judge"),
    ],
)
def test_real_inspect_scorer_names_map_to_correct_grader_types(scorer_name, expected_type):
    """inspect_ai's built-in scorer registry names -- exact, includes, match, pattern,
    model_graded_qa/model_graded_fact -- are the real function names exported from
    inspect_ai.scorer; confirmed importable in the installed package."""
    from inspect_ai.scorer import exact, includes, pattern  # noqa: F401

    data = {
        "task": "t",
        "samples": [{"id": "x", "input": "i", "target": "o"}],
        "scorers": [scorer_name],
    }
    suite = from_inspect(data)
    assert suite["graders"][0]["type"] == expected_type
    if expected_type == "custom":
        assert suite["graders"][0]["params"]["handler"] == f"inspect:{scorer_name}"
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_no_scorers_falls_back_to_default_exact_match_grader():
    data = {"task": "t", "samples": [{"id": "x", "input": "i"}], "scorers": []}
    suite = from_inspect(data)
    assert suite["graders"] == [{"id": "gr_0", "type": "exact_match"}]
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_context_list_and_scalar_both_normalized_to_list():
    data = {
        "task": "t",
        "samples": [
            {"id": "x", "input": "i", "context": ["a", "b"]},
            {"id": "y", "input": "i2", "context": "single-string-context"},
        ],
        "scorers": ["exact"],
    }
    suite = from_inspect(data)
    assert suite["test_cases"][0]["context"] == ["a", "b"]
    assert suite["test_cases"][1]["context"] == ["single-string-context"]
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_multiple_samples_share_the_same_grader_ids():
    """All samples in one inspect_ai task run against the same scorer set, so every
    test case should reference the identical grader-id list, not per-sample copies."""
    suite = from_inspect(REAL_INSPECT_EXPORT)
    ids_a = suite["test_cases"][0]["graders"]
    ids_b = suite["test_cases"][1]["graders"]
    assert ids_a == ids_b == ["gr_0"]
