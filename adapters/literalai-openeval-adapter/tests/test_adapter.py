"""Test suite for literalai-openeval-adapter.

Organized around the 3 "wrinkles" from issue #23:
  A. Input flattening (dict -> string)
  B. Score clamping   (any scale -> [0.0, 1.0])
  C. Grader mapping    ("HUMAN"/"CODE"/"AI" -> "human"/"code"/"llm_judge")

Plus: public API round-trips, and malformed/edge-case input handling.

Tests marked with a star (★) in their docstring are the highest-value
regression tests -- the ones that most directly encode the behavior
described in the issue and that a reviewer would check first.
"""

import pytest

from literalai_openeval_adapter import (
    clamp_score,
    flatten_input,
    from_openeval,
    map_grader_type,
    map_grader_type_reverse,
    results_to_openeval,
    to_openeval,
)


# ============================================================
# A. Input flattening
# ============================================================

class TestFlattenInput:
    def test_dict_with_question_key(self):
        """★ Core case from the issue: {"question": "..."} -> plain string."""
        assert flatten_input({"question": "What is 2+2?"}) == "What is 2+2?"

    def test_already_a_string_passes_through_unchanged(self):
        """★ Idempotency: if LiteralAI already sends a string, don't touch it."""
        assert flatten_input("What is 2+2?") == "What is 2+2?"

    @pytest.mark.parametrize("key", ["input", "prompt", "text", "query"])
    def test_other_common_keys_are_recognized(self, key):
        assert flatten_input({key: "hello"}) == "hello"

    def test_preferred_key_wins_over_other_string_fields(self):
        """When multiple string fields exist, prefer the known semantic key
        over an arbitrary one, so results are deterministic."""
        data = {"metadata_note": "ignore me", "question": "real question"}
        assert flatten_input(data) == "real question"

    def test_falls_back_to_first_string_value_if_no_known_key(self):
        data = {"custom_field": "the actual text"}
        assert flatten_input(data) == "the actual text"

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            flatten_input({})

    def test_dict_with_no_string_values_raises(self):
        with pytest.raises(ValueError):
            flatten_input({"count": 5, "ok": True})

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            flatten_input(12345)


# ============================================================
# B. Score clamping
# ============================================================

class TestClampScore:
    def test_score_above_one_is_clamped_to_one(self):
        """★ Core case: 8.5 (out of 10) must clamp to 1.0, not overflow."""
        result = clamp_score(8.5)
        assert result["score"] == 1.0

    def test_raw_score_is_preserved_in_metadata(self):
        """★ We must never silently lose the original value -- it has to
        survive somewhere so users can recover their real scale."""
        result = clamp_score(8.5)
        assert result["raw_score"] == 8.5

    def test_score_of_100_out_of_100_clamps_to_one(self):
        result = clamp_score(100)
        assert result["score"] == 1.0
        assert result["raw_score"] == 100

    def test_score_already_in_range_is_unchanged(self):
        result = clamp_score(0.73)
        assert result["score"] == 0.73
        assert result["was_clamped"] is False

    def test_boundary_exactly_one_is_not_flagged_as_clamped(self):
        """★ Boundary test: 1.0 is valid as-is and shouldn't be treated as
        an overflow case (off-by-one bugs love this exact value)."""
        result = clamp_score(1.0)
        assert result["score"] == 1.0
        assert result["was_clamped"] is False

    def test_boundary_exactly_zero_is_not_flagged_as_clamped(self):
        result = clamp_score(0.0)
        assert result["score"] == 0.0
        assert result["was_clamped"] is False

    def test_negative_score_clamps_to_zero(self):
        """Scores shouldn't go negative even if some grader emits -5."""
        result = clamp_score(-5)
        assert result["score"] == 0.0
        assert result["was_clamped"] is True

    def test_integer_input_is_accepted_and_returned_as_float(self):
        result = clamp_score(1)
        assert result["score"] == 1.0
        assert isinstance(result["score"], float)

    def test_boolean_is_rejected_even_though_bool_is_an_int_subclass(self):
        """Python quirk: True/False are ints. Guard against silently
        treating a boolean flag as a score of 1.0 or 0.0."""
        with pytest.raises(TypeError):
            clamp_score(True)

    def test_non_numeric_score_raises(self):
        with pytest.raises(TypeError):
            clamp_score("8.5")


# ============================================================
# C. Grader type mapping
# ============================================================

class TestGraderTypeMapping:
    def test_ai_maps_to_llm_judge(self):
        """★ Core case from the issue: LiteralAI "AI" -> OpenEval "llm_judge"."""
        assert map_grader_type("AI") == "llm_judge"

    def test_human_maps_to_human(self):
        assert map_grader_type("HUMAN") == "human"

    def test_code_maps_to_code(self):
        assert map_grader_type("CODE") == "code"

    def test_mapping_is_case_insensitive(self):
        assert map_grader_type("ai") == "llm_judge"
        assert map_grader_type("Human") == "human"

    def test_unknown_grader_type_raises(self):
        with pytest.raises(ValueError):
            map_grader_type("ROBOT")

    def test_non_string_grader_type_raises(self):
        with pytest.raises(TypeError):
            map_grader_type(123)

    def test_reverse_mapping_round_trips(self):
        """★ Round trip: every OpenEval label must map back to exactly the
        LiteralAI label it came from -- required for from_openeval()."""
        for literalai_val, openeval_val in [("HUMAN", "human"), ("CODE", "code"), ("AI", "llm_judge")]:
            assert map_grader_type(literalai_val) == openeval_val
            assert map_grader_type_reverse(openeval_val) == literalai_val


# ============================================================
# Public API: to_openeval / from_openeval / results_to_openeval
# ============================================================

class TestToOpenEval:
    def test_full_dataset_conversion(self):
        """★ End-to-end conversion of a realistic LiteralAI dataset."""
        dataset = {
            "name": "geometry-quiz",
            "items": [
                {
                    "id": "item-1",
                    "input": {"question": "What is 2+2?"},
                    "expected_output": "4",
                    "metadata": {"difficulty": "easy"},
                }
            ],
        }
        suite = to_openeval(dataset)
        assert suite["suite_name"] == "geometry-quiz"
        assert len(suite["cases"]) == 1
        case = suite["cases"][0]
        assert case["input"] == "What is 2+2?"
        assert case["expected_output"] == "4"
        assert case["metadata"]["difficulty"] == "easy"

    def test_empty_dataset_produces_empty_suite(self):
        suite = to_openeval({"name": "empty", "items": []})
        assert suite["cases"] == []

    def test_missing_name_defaults_gracefully(self):
        suite = to_openeval({"items": []})
        assert suite["suite_name"] == "untitled"


class TestFromOpenEval:
    def test_reverses_to_openeval_for_dict_inputs(self):
        """★ Round trip: to_openeval() then from_openeval() must recover the
        *original* dict shape, not just the flattened string -- otherwise
        the translation is lossy in the direction LiteralAI needs."""
        original_dataset = {
            "name": "geometry-quiz",
            "items": [
                {
                    "id": "item-1",
                    "input": {"question": "What is 2+2?"},
                    "expected_output": "4",
                    "metadata": {"difficulty": "easy"},
                }
            ],
        }
        suite = to_openeval(original_dataset)
        recovered = from_openeval(suite)

        assert recovered["items"][0]["input"] == {"question": "What is 2+2?"}
        assert recovered["items"][0]["metadata"] == {"difficulty": "easy"}
        assert "_original_input" not in recovered["items"][0]["metadata"]

    def test_round_trip_is_stable_across_multiple_items(self):
        dataset = {
            "name": "multi",
            "items": [
                {"id": "1", "input": {"prompt": "a"}, "expected_output": "x", "metadata": {}},
                {"id": "2", "input": "already a string", "expected_output": "y", "metadata": {}},
            ],
        }
        recovered = from_openeval(to_openeval(dataset))
        assert recovered["items"][0]["input"] == {"prompt": "a"}
        assert recovered["items"][1]["input"] == "already a string"


class TestResultsToOpenEval:
    def test_full_results_conversion(self):
        """★ End-to-end: combines score clamping AND grader mapping in one
        call, matching how results actually arrive from LiteralAI."""
        literalai_results = [
            {"case_id": "item-1", "score": 8.5, "grader_type": "AI", "metadata": {"note": "auto-graded"}},
        ]
        openeval = results_to_openeval(literalai_results)
        result = openeval["results"][0]

        assert result["score"] == 1.0
        assert result["grader_type"] == "llm_judge"
        assert result["metadata"]["raw_score"] == 8.5
        assert result["metadata"]["note"] == "auto-graded"

    def test_multiple_results_each_clamped_and_mapped_independently(self):
        literalai_results = [
            {"case_id": "1", "score": 0.9, "grader_type": "HUMAN", "metadata": {}},
            {"case_id": "2", "score": 100, "grader_type": "CODE", "metadata": {}},
            {"case_id": "3", "score": -1, "grader_type": "AI", "metadata": {}},
        ]
        openeval = results_to_openeval(literalai_results)
        scores = [r["score"] for r in openeval["results"]]
        graders = [r["grader_type"] for r in openeval["results"]]

        assert scores == [0.9, 1.0, 0.0]
        assert graders == ["human", "code", "llm_judge"]

    def test_invalid_grader_type_in_results_raises(self):
        with pytest.raises(ValueError):
            results_to_openeval([{"case_id": "1", "score": 0.5, "grader_type": "ROBOT", "metadata": {}}])

    def test_empty_results_list_produces_empty_output(self):
        assert results_to_openeval([]) == {"results": []}
