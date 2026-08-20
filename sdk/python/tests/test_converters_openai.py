"""Tests for openeval.converters_openai.from_openai_evals().

No test file existed for this converter before this change, despite the
root README claiming "OpenAI Evals -> EvalPort: (check) Python SDK" --
these tests exist specifically to make that claim true, validated against
the real openeval.validate.validate_suite(), not just shape assertions.

Sample shapes below match the real openai/evals eval classes' own
assertions (read directly from evals/elsuite/basic/match.py's
Match.eval_sample: "input" must be present, "ideal" must be a str or
list[str]; chat-based evals pass "input" as a list of
{"role": ..., "content": ...} messages, confirmed by the same source).
"""

from openeval.converters_openai import from_openai_evals
from openeval.validate import validate_suite


def test_plain_string_input_still_works():
    data = {
        "id": "basic_eval",
        "test_data": [{"input": "What is 2+2?", "ideal": "4"}],
    }
    suite = from_openai_evals(data)
    result = validate_suite(suite)
    assert result.valid, result.errors
    assert suite["test_cases"][0]["input"] == "What is 2+2?"
    assert suite["test_cases"][0]["expected_output"] == "4"


def test_chat_message_list_input_produces_valid_suite():
    """The real bug this change fixes: openai/evals' Match/Includes/
    model-graded eval classes accept sample["input"] as a list of chat
    messages, not a string -- passing that straight through as
    TestCase.input used to fail validate_suite() with a REQUIRED error on
    'input' (confirmed before this fix), because spec/schemas/testcase.json
    requires input to be a string or array of strings, not an array of
    message objects.
    """
    data = {
        "id": "chat_eval",
        "test_data": [
            {
                "input": [
                    {"role": "system", "content": "You are a math tutor."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "ideal": "4",
            }
        ],
    }
    suite = from_openai_evals(data)
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_chat_message_list_input_is_flattened_readably():
    data = {
        "id": "chat_eval",
        "test_data": [
            {
                "input": [
                    {"role": "system", "content": "You are a math tutor."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "ideal": "4",
            }
        ],
    }
    suite = from_openai_evals(data)
    flat = suite["test_cases"][0]["input"]
    assert isinstance(flat, str)
    assert "system: You are a math tutor." in flat
    assert "user: What is 2+2?" in flat


def test_chat_message_list_input_preserved_losslessly_in_metadata():
    messages = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    data = {
        "id": "chat_eval",
        "test_data": [{"input": messages, "ideal": "4"}],
    }
    suite = from_openai_evals(data)
    assert suite["test_cases"][0]["metadata"]["openai_evals"]["messages"] == messages


def test_list_valued_ideal_uses_first_entry_as_expected_output():
    """Match.eval_sample allows sample["ideal"] to be a list of acceptable
    answers, not just a single string. Before this fix, str(["4", "four"])
    produced the literal Python repr "['4', 'four']" as expected_output --
    schema-valid (it's a string) but not a meaningful value."""
    data = {
        "id": "multi_ideal_eval",
        "test_data": [{"input": "What is 2+2?", "ideal": ["4", "four"]}],
    }
    suite = from_openai_evals(data)
    result = validate_suite(suite)
    assert result.valid, result.errors
    assert suite["test_cases"][0]["expected_output"] == "4"


def test_list_valued_ideal_preserves_all_variants_in_metadata():
    data = {
        "id": "multi_ideal_eval",
        "test_data": [{"input": "What is 2+2?", "ideal": ["4", "four"]}],
    }
    suite = from_openai_evals(data)
    assert suite["test_cases"][0]["metadata"]["openai_evals"]["ideal_variants"] == [
        "4",
        "four",
    ]


def test_string_list_input_kept_as_array_per_schema():
    """spec/schemas/testcase.json allows input to be an array of plain
    strings (not just a single string) -- that shape should pass through
    unchanged, not get flattened."""
    data = {
        "id": "multi_string_input_eval",
        "test_data": [{"input": ["line one", "line two"], "ideal": "ok"}],
    }
    suite = from_openai_evals(data)
    result = validate_suite(suite)
    assert result.valid, result.errors
    assert suite["test_cases"][0]["input"] == ["line one", "line two"]


def test_pre_existing_test_case_metadata_is_preserved_alongside_new_fields():
    data = {
        "id": "chat_eval",
        "test_data": [
            {
                "input": [{"role": "user", "content": "hi"}],
                "ideal": "hello",
                "metadata": {"difficulty": "easy"},
            }
        ],
    }
    suite = from_openai_evals(data)
    meta = suite["test_cases"][0]["metadata"]
    assert meta["difficulty"] == "easy"
    assert "messages" in meta["openai_evals"]


def test_grader_type_mapping_and_full_suite_validity():
    data = {
        "id": "graded_eval",
        "test_data": [{"input": "Explain gravity.", "ideal": "A force..."}],
        "config": {
            "sampling": {"model": "gpt-4o-mini", "temperature": 0.0},
            "grader": {
                "type": "model_graded",
                "model": "gpt-4o",
                "prompt": "Rate the explanation from {output} against {expected}.",
            },
        },
    }
    suite = from_openai_evals(data)
    result = validate_suite(suite)
    assert result.valid, result.errors
    grader = suite["graders"][0]
    assert grader["type"] == "model graded"
    assert grader["params"]["model"] == "gpt-4o"
    assert suite["config"]["provider"]["model"] == "gpt-4o-mini"
    assert suite["config"]["provider"]["temperature"] == 0.0


def test_empty_input_does_not_crash_or_produce_empty_string():
    """spec/schemas/testcase.json requires input's string form to have
    minLength: 1 -- an empty string would fail validation, so an empty/
    missing input falls back to a placeholder rather than silently
    producing an invalid suite."""
    data = {"id": "edge_case_eval", "test_data": [{"input": "", "ideal": "x"}]}
    suite = from_openai_evals(data)
    result = validate_suite(suite)
    assert result.valid, result.errors
    assert suite["test_cases"][0]["input"] != ""
