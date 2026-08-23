"""Tests for literalai-openeval-adapter.

Organized around the 3 wrinkles from issue #23 (input flattening, score
clamping, grader-type mapping), then the public API, with round trips
validated against the REAL `openeval.validate.validate_suite()` /
`validate_result_set()` -- not internal self-consistency, per the review
on PR #25. Uses the real `literalai` dataclasses (`DatasetItem`,
`DatasetExperimentItem`, `ScoreDict`) wherever practical, matching the
convention set by `adapters/autogen-openeval-adapter` and
`adapters/langfuse-openeval-adapter`.

Tests marked ★ are the highest-value regression tests -- the ones a
reviewer would check first.
"""

import pytest

from literalai import DatasetItem
from literalai.observability.step import ScoreDict
from openeval.validate import validate_suite, validate_result_set

from literalai_openeval_adapter import (
    clamp_score,
    flatten_dict_field,
    from_openeval,
    map_grader_type,
    map_grader_type_reverse,
    results_to_openeval,
    to_openeval,
)


def make_dataset_item(item_id, input_dict, expected_output=None, metadata=None):
    """Build a real `literalai.DatasetItem` (dataclass), not a mock."""
    return DatasetItem(
        id=item_id,
        created_at="2026-01-01T00:00:00Z",
        dataset_id="ds_1",
        metadata=metadata or {},
        input=input_dict,
        expected_output=expected_output,
        intermediary_steps=[],
    )


class FakeDataset:
    """Plain stand-in for `literalai.Dataset` (attribute-based, like the
    real dataclass) -- avoids constructing the real `Dataset`, which
    requires a live `api` client.
    """

    def __init__(self, id, name, items):
        self.id = id
        self.name = name
        self.items = items


class FakeExperimentItem:
    """Plain stand-in for `literalai.DatasetExperimentItem`."""

    def __init__(self, dataset_item_id, output, scores):
        self.dataset_item_id = dataset_item_id
        self.output = output
        self.scores = scores


# ============================================================
# A. Input/expected_output flattening
# ============================================================

class TestFlattenDictField:
    def test_dict_with_question_key(self):
        """★ Core case from the issue: {"question": "..."} -> plain string."""
        assert flatten_dict_field({"question": "What is 2+2?"}) == "What is 2+2?"

    def test_already_a_string_passes_through_unchanged(self):
        assert flatten_dict_field("What is 2+2?") == "What is 2+2?"

    @pytest.mark.parametrize("key", ["input", "prompt", "text", "query", "output", "answer"])
    def test_other_common_keys_are_recognized(self, key):
        assert flatten_dict_field({key: "hello"}) == "hello"

    def test_preferred_key_wins_over_other_string_fields(self):
        data = {"note": "ignore me", "question": "real question"}
        assert flatten_dict_field(data) == "real question"

    def test_falls_back_to_first_string_value_if_no_known_key(self):
        data = {"custom_field": "the actual text"}
        assert flatten_dict_field(data) == "the actual text"

    def test_no_string_values_falls_back_to_json(self):
        """★ Honest degradation: a Literal AI row with no string field at
        all is still valid data -- it becomes documented JSON, not a
        crash, since flattening is a best-effort convenience, not a
        strict contract Literal AI's schema-free rows can guarantee."""
        result = flatten_dict_field({"count": 5, "ok": True})
        assert isinstance(result, str)
        import json
        assert json.loads(result) == {"count": 5, "ok": True}

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            flatten_dict_field({})

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            flatten_dict_field(12345)


# ============================================================
# B. Score clamping
# ============================================================

class TestClampScore:
    def test_score_above_one_is_clamped_to_one(self):
        """★ Core case: an 8.5-out-of-10 score must clamp to 1.0."""
        result = clamp_score(8.5)
        assert result["score"] == 1.0

    def test_raw_score_is_preserved(self):
        """★ Nothing gets silently lost -- required so results_to_openeval()
        can stash it under the spec's reserved `openeval.raw_score` key."""
        result = clamp_score(8.5)
        assert result["raw_score"] == 8.5

    def test_boundary_exactly_one_is_not_flagged_as_clamped(self):
        result = clamp_score(1.0)
        assert result["score"] == 1.0
        assert result["was_clamped"] is False

    def test_boundary_exactly_zero_is_not_flagged_as_clamped(self):
        result = clamp_score(0.0)
        assert result["score"] == 0.0
        assert result["was_clamped"] is False

    def test_negative_score_clamps_to_zero(self):
        result = clamp_score(-5)
        assert result["score"] == 0.0
        assert result["was_clamped"] is True

    def test_boolean_is_rejected_even_though_bool_is_an_int_subclass(self):
        """Python quirk guard: True/False are ints; a boolean flag must
        never be silently treated as a score of 1.0/0.0."""
        with pytest.raises(TypeError):
            clamp_score(True)


    def test_nan_score_raises(self):
        import math
        with pytest.raises(ValueError, match="NaN"):
            clamp_score(float('nan'))

    def test_non_numeric_score_raises(self):
        with pytest.raises(TypeError):
            clamp_score("8.5")


# ============================================================
# C. Grader type mapping
# ============================================================

class TestGraderTypeMapping:
    def test_ai_maps_to_llm_judge(self):
        """★ Core case from the issue."""
        assert map_grader_type("AI") == "llm_judge"

    def test_human_maps_to_human(self):
        assert map_grader_type("HUMAN") == "human"

    def test_code_maps_to_code(self):
        assert map_grader_type("CODE") == "code"

    def test_mapping_is_case_insensitive(self):
        assert map_grader_type("ai") == "llm_judge"

    def test_unknown_grader_type_raises(self):
        with pytest.raises(ValueError):
            map_grader_type("ROBOT")

    def test_reverse_mapping_round_trips(self):
        """★ Every OpenEval label maps back to the exact Literal AI label
        it came from -- required for `from_openeval()` fidelity."""
        for literalai_val, openeval_val in [("HUMAN", "human"), ("CODE", "code"), ("AI", "llm_judge")]:
            assert map_grader_type(literalai_val) == openeval_val
            assert map_grader_type_reverse(openeval_val) == literalai_val


# ============================================================
# Public API, validated against the REAL EvalPort validators
# ============================================================

class TestToOpenEval:
    def test_full_dataset_conversion_validates_against_real_schema(self):
        """★ The check the maintainer asked for: not just our own shape,
        but genuine EvalPort schema conformance via validate_suite()."""
        dataset = FakeDataset(
            id="quiz_1",
            name="geometry-quiz",
            items=[
                make_dataset_item(
                    "item_1",
                    {"question": "What is 2+2?"},
                    expected_output={"answer": "4"},
                    metadata={"difficulty": "easy"},
                )
            ],
        )
        suite = to_openeval(dataset)

        validation = validate_suite(suite)
        assert validation.valid, validation.errors

        assert suite["test_cases"][0]["input"] == "What is 2+2?"
        assert suite["test_cases"][0]["expected_output"] == "4"
        assert suite["test_cases"][0]["metadata"]["difficulty"] == "easy"

    def test_multiple_items_all_validate(self):
        dataset = FakeDataset(
            id="ds_2",
            name="multi",
            items=[
                make_dataset_item("a", {"prompt": "hi"}),
                make_dataset_item("b", {"question": "bye"}, expected_output={"answer": "later"}),
            ],
        )
        suite = to_openeval(dataset)
        validation = validate_suite(suite)
        assert validation.valid, validation.errors
        assert len(suite["test_cases"]) == 2

    def test_exact_match_grader_option_also_validates(self):
        dataset = FakeDataset(id="ds_3", name="exact", items=[make_dataset_item("a", {"question": "2+2"})])
        suite = to_openeval(dataset, grader_type="exact_match")
        assert suite["graders"][0]["type"] == "exact_match"
        assert validate_suite(suite).valid

    def test_empty_dataset_still_fails_validation_gracefully(self):
        """EvalPort requires at least one test case (schema MIN_ITEMS on
        test_cases) -- confirm our empty-dataset output is rejected by the
        real validator with a clear reason, not silently accepted."""
        dataset = FakeDataset(id="ds_empty", name="empty", items=[])
        suite = to_openeval(dataset)
        validation = validate_suite(suite)
        assert not validation.valid


class TestFromOpenEval:
    def test_round_trip_recovers_original_dict_shape(self):
        """★ to_openeval() -> from_openeval() must recover the *original*
        dict shape, not just the flattened string, so the result is
        directly usable with `Dataset.create_item(input=..., ...)`."""
        dataset = FakeDataset(
            id="ds_4",
            name="rt",
            items=[
                make_dataset_item(
                    "item_1", {"question": "2+2?"}, expected_output={"answer": "4"}, metadata={"tag": "math"}
                )
            ],
        )
        suite = to_openeval(dataset)
        recovered = from_openeval(suite)

        assert recovered[0]["input"] == {"question": "2+2?"}
        assert recovered[0]["expected_output"] == {"answer": "4"}
        assert recovered[0]["metadata"] == {"tag": "math"}
        assert "literalai" not in recovered[0]["metadata"]

    def test_from_openeval_on_a_suite_not_from_this_adapter(self):
        """A hand-authored EvalPort suite (no literalai metadata) must
        still convert into something usable, wrapping the flattened
        string back into a dict."""
        suite = {
            "version": "1.0.0",
            "id": "s1",
            "test_cases": [{"id": "tc1", "input": "hi", "expected_output": "hello", "graders": ["g1"]}],
        }
        items = from_openeval(suite)
        assert items[0]["input"] == {"question": "hi"}
        assert items[0]["expected_output"] == {"answer": "hello"}


class TestResultsToOpenEval:
    def test_full_results_conversion_validates_against_real_schema(self):
        """★ The results-side check the maintainer asked for: real
        validate_result_set(), combining score clamping and grader
        mapping the way they actually arrive together."""
        score = ScoreDict(
            id="score_1",
            name="correctness",
            type="AI",
            value=8.5,
            label=None,
            stepId=None,
            datasetExperimentItemId="exp_item_1",
            comment="close enough",
            tags=None,
        )
        item = FakeExperimentItem(dataset_item_id="tc_1", output={"answer": "4"}, scores=[score])

        result_set = results_to_openeval(
            [item], suite_id="literalai_quiz_1", run_id="run_1", started_at="2026-01-15T10:00:00Z"
        )

        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors

        gr = result_set["results"][0]["grader_results"][0]
        assert gr["score"] == 1.0
        assert gr["type"] == "llm_judge"
        assert gr["metadata"]["openeval"]["raw_score"] == 8.5

    def test_multiple_scores_each_clamped_and_mapped_independently(self):
        scores = [
            ScoreDict(id="s1", name="human_review", type="HUMAN", value=0.9, label=None, stepId=None,
                      datasetExperimentItemId=None, comment=None, tags=None),
            ScoreDict(id="s2", name="code_check", type="CODE", value=100, label=None, stepId=None,
                      datasetExperimentItemId=None, comment=None, tags=None),
            ScoreDict(id="s3", name="ai_judge", type="AI", value=-1, label=None, stepId=None,
                      datasetExperimentItemId=None, comment=None, tags=None),
        ]
        item = FakeExperimentItem(dataset_item_id="tc_1", output="4", scores=scores)
        result_set = results_to_openeval([item], suite_id="s", run_id="r", started_at="2026-01-15T10:00:00Z")

        assert validate_result_set(result_set).valid
        grs = result_set["results"][0]["grader_results"]
        assert [g["score"] for g in grs] == [0.9, 1.0, 0.0]
        assert [g["type"] for g in grs] == ["human", "code", "llm_judge"]

    def test_accepts_experiment_object_with_items_attribute(self):
        """`results_to_openeval` also accepts the DatasetExperiment-shaped
        container directly (an object with an `.items` list), not just a
        bare list of DatasetExperimentItem."""
        score = ScoreDict(id="s1", name="c", type="AI", value=0.7, label=None, stepId=None,
                           datasetExperimentItemId=None, comment=None, tags=None)
        item = FakeExperimentItem(dataset_item_id="tc_1", output="ok", scores=[score])

        class FakeExperiment:
            def __init__(self, items):
                self.items = items

        result_set = results_to_openeval(
            FakeExperiment([item]), suite_id="s", run_id="r", started_at="2026-01-15T10:00:00Z"
        )
        assert validate_result_set(result_set).valid

    def test_empty_scores_produces_unpassed_result(self):
        """An experiment item with no scores yet (pending grading) must
        still produce a spec-valid Result: `passed: false`,
        `grader_results: []`."""
        item = FakeExperimentItem(dataset_item_id="tc_1", output="pending", scores=[])
        result_set = results_to_openeval([item], suite_id="s", run_id="r", started_at="2026-01-15T10:00:00Z")

        assert validate_result_set(result_set).valid
        assert result_set["results"][0]["passed"] is False
        assert result_set["results"][0]["grader_results"] == []

    def test_invalid_grader_type_raises_before_producing_invalid_output(self):
        score = ScoreDict(id="s1", name="c", type="ROBOT", value=0.5, label=None, stepId=None,
                           datasetExperimentItemId=None, comment=None, tags=None)
        item = FakeExperimentItem(dataset_item_id="tc_1", output="x", scores=[score])
        with pytest.raises(ValueError):
            results_to_openeval([item], suite_id="s", run_id="r", started_at="2026-01-15T10:00:00Z")
