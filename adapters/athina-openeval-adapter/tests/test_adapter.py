"""
Tests for athina-openeval-adapter.

These construct real `athina.loaders.loader.DataPoint` / `athina.interfaces.result.EvalResult`
(called `LlmEvalResult` in older athina releases; renamed upstream, verified 2026-08-23 against
the currently installed athina==1.7.39, which exports the new name) instances rather than
inventing an ad hoc shape. TypedDict has no runtime behavior of its own -- constructing "a real
DataPoint" and "a plain dict with the same keys" are the same object at runtime -- so what makes
these real is that every field name and required_args() combination below was read directly out
of athina's installed source (see the module's own docstring for the grep trail), not guessed.

Every suite/result_set produced here is validated against the real openeval.validate functions,
not a hand-rolled schema check.
"""

import pytest
from athina.loaders.loader import DataPoint
from athina.interfaces.result import EvalResult

from openeval.validate import validate_suite, validate_result_set

from athina_openeval_adapter import to_openeval, result_to_openeval, from_openeval


# ---------------------------------------------------------------------------
# Real fixture data, built from athina's real required_args() combinations
# ---------------------------------------------------------------------------


def _does_response_answer_query_data():
    """DoesResponseAnswerQuery.required_args() == ["query", "response"] (verified)."""
    return [
        DataPoint(
            query="What is the capital of France?",
            response="Paris is the capital of France.",
        ),
        DataPoint(
            query="How do I reset my password?",
            response="The weather today is sunny with a high of 75F.",
        ),
    ]


def _faithfulness_data():
    """Faithfulness.required_args() == ["context", "response"] (verified)."""
    return [
        DataPoint(
            context="3M has increased its dividend for 65 consecutive years.",
            response="3M has raised its dividend every year for 65 years running.",
            expected_response="Yes, 3M has a 65-year streak of dividend increases.",
        ),
        DataPoint(
            context="Boeing faces multiple lawsuits from the 2018 Lion Air crash.",
            response="Boeing has never faced any product liability litigation.",
        ),
    ]


def _context_contains_enough_information_data():
    """ContextContainsEnoughInformation.required_args() == ["query", "context"] (verified)."""
    return [
        DataPoint(
            query="Does CVS Health pay a quarterly dividend?",
            context="CVS Health Corporation has paid cash dividends every quarter since becoming a public company.",
        ),
    ]


def _custom_grader_data():
    """CustomGrader.required_args() == ["response"] (verified) -- no query at all."""
    return [
        DataPoint(response="The answer is 42."),
        DataPoint(response="I cannot help with that request."),
    ]


def _llm_eval_result(*, failure, reason, model="gpt-4-1106-preview", runtime=842, data=None):
    return EvalResult(
        name="does_response_answer_query",
        data=data or {},
        failure=failure,
        reason=reason,
        runtime=runtime,
        model=model,
    )


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


def test_to_openeval_one_test_case_per_data_point():
    suite = to_openeval(_does_response_answer_query_data(), "does_response_answer_query")
    assert len(suite["test_cases"]) == 2


def test_to_openeval_query_becomes_input():
    suite = to_openeval(_does_response_answer_query_data(), "does_response_answer_query")
    assert suite["test_cases"][0]["input"] == "What is the capital of France?"
    assert suite["test_cases"][1]["input"] == "How do I reset my password?"


def test_to_openeval_default_suite_id_derived_from_eval_name():
    suite = to_openeval(_does_response_answer_query_data(), "does_response_answer_query")
    assert suite["id"] == "athina-does_response_answer_query"


def test_to_openeval_explicit_suite_id_and_name():
    suite = to_openeval(
        _does_response_answer_query_data(),
        "does_response_answer_query",
        suite_id="my-suite",
        suite_name="My Suite",
    )
    assert suite["id"] == "my-suite"
    assert suite["name"] == "My Suite"


def test_to_openeval_context_becomes_single_item_context_list():
    suite = to_openeval(_faithfulness_data(), "faithfulness")
    tc0 = suite["test_cases"][0]
    assert tc0["context"] == ["3M has increased its dividend for 65 consecutive years."]


def test_to_openeval_expected_response_becomes_expected_output():
    suite = to_openeval(_faithfulness_data(), "faithfulness")
    assert suite["test_cases"][0]["expected_output"] == "Yes, 3M has a 65-year streak of dividend increases."
    # second entry has no expected_response -- must not be present at all
    assert "expected_output" not in suite["test_cases"][1]


def test_to_openeval_faithfulness_uses_response_as_input_since_no_query_exists():
    # Faithfulness's required_args is ["context", "response"] -- there is no query key.
    suite = to_openeval(_faithfulness_data(), "faithfulness")
    tc0 = suite["test_cases"][0]
    assert tc0["input"] == "3M has raised its dividend every year for 65 years running."
    assert tc0["metadata"]["athina.input_synthesized_from_response"] is True


def test_to_openeval_custom_grader_response_only_flags_synthesized_input():
    data = _custom_grader_data()
    suite = to_openeval(data, "custom_grader")
    for tc, entry in zip(suite["test_cases"], data):
        assert tc["metadata"]["athina.input_synthesized_from_response"] is True
        assert tc["input"] == entry["response"]


def test_to_openeval_context_contains_enough_information_query_is_real_input_no_synthesis_flag():
    suite = to_openeval(
        _context_contains_enough_information_data(), "context_contains_enough_information"
    )
    tc0 = suite["test_cases"][0]
    assert tc0["input"] == "Does CVS Health pay a quarterly dividend?"
    assert "metadata" not in tc0 or not tc0["metadata"].get("athina.input_synthesized_from_response")


def test_to_openeval_extra_args_preserved_in_metadata():
    data = [DataPoint(response="42", grading_criteria="Must be numeric")]
    suite = to_openeval(data, "custom_grader")
    assert suite["test_cases"][0]["metadata"]["athina.extra_args"] == {
        "grading_criteria": "Must be numeric"
    }


def test_to_openeval_single_grader_with_custom_type_and_handler():
    suite = to_openeval(_does_response_answer_query_data(), "does_response_answer_query")
    assert len(suite["graders"]) == 1
    grader = suite["graders"][0]
    assert grader["id"] == "gr_does_response_answer_query"
    assert grader["type"] == "custom"
    assert grader["params"]["handler"] == "does_response_answer_query"


def test_to_openeval_all_test_cases_reference_the_single_grader():
    suite = to_openeval(_faithfulness_data(), "faithfulness")
    for tc in suite["test_cases"]:
        assert tc["graders"] == ["gr_faithfulness"]


def test_to_openeval_explicit_ids():
    suite = to_openeval(
        _does_response_answer_query_data(),
        "does_response_answer_query",
        ids=["case-a", "case-b"],
    )
    assert [tc["id"] for tc in suite["test_cases"]] == ["case-a", "case-b"]


def test_to_openeval_default_ids_are_positional():
    suite = to_openeval(_does_response_answer_query_data(), "does_response_answer_query")
    assert [tc["id"] for tc in suite["test_cases"]] == ["tc_0", "tc_1"]


def test_to_openeval_rejects_empty_data():
    with pytest.raises(ValueError, match="at least one entry"):
        to_openeval([], "does_response_answer_query")


def test_to_openeval_rejects_mismatched_ids_length():
    with pytest.raises(ValueError, match="ids has"):
        to_openeval(_does_response_answer_query_data(), "does_response_answer_query", ids=["only_one"])


def test_to_openeval_rejects_entry_with_neither_query_nor_response():
    with pytest.raises(ValueError, match="neither 'query' nor 'response'"):
        to_openeval([{"context": "some context, no response or query"}], "faithfulness")


def test_to_openeval_produces_spec_valid_suite_for_every_real_evaluator_shape():
    cases = [
        (_does_response_answer_query_data(), "does_response_answer_query"),
        (_faithfulness_data(), "faithfulness"),
        (_context_contains_enough_information_data(), "context_contains_enough_information"),
        (_custom_grader_data(), "custom_grader"),
    ]
    for data, eval_name in cases:
        suite = to_openeval(data, eval_name)
        result = validate_suite(suite)
        assert result.valid, f"{eval_name}: {result.errors}"


# ---------------------------------------------------------------------------
# result_to_openeval
# ---------------------------------------------------------------------------


def test_result_to_openeval_one_result_per_data_point():
    data = _does_response_answer_query_data()
    eval_results = [
        _llm_eval_result(failure=False, reason="Answers the query directly."),
        _llm_eval_result(failure=True, reason="Response is unrelated to the query."),
    ]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="athina-does_response_answer_query", run_id="run-1", started_at="2026-08-22T00:00:00Z",
    )
    assert len(rs["results"]) == 2


def test_result_to_openeval_pass_maps_to_score_1_and_passed_true():
    data = _does_response_answer_query_data()[:1]
    eval_results = [_llm_eval_result(failure=False, reason="Directly answers the query.")]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z",
    )
    gr = rs["results"][0]["grader_results"][0]
    assert gr["score"] == 1.0
    assert gr["passed"] is True
    assert rs["results"][0]["passed"] is True


def test_result_to_openeval_fail_maps_to_score_0_and_passed_false():
    data = _does_response_answer_query_data()[1:2]
    eval_results = [_llm_eval_result(failure=True, reason="Off-topic response.")]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z",
    )
    gr = rs["results"][0]["grader_results"][0]
    assert gr["score"] == 0.0
    assert gr["passed"] is False


def test_result_to_openeval_carries_real_reason_text():
    data = _does_response_answer_query_data()[:1]
    eval_results = [_llm_eval_result(failure=False, reason="The response directly and completely answers the user's question about France's capital.")]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z",
    )
    assert "directly and completely answers" in rs["results"][0]["grader_results"][0]["reason"]


def test_result_to_openeval_actual_output_is_the_real_response():
    data = _does_response_answer_query_data()[:1]
    eval_results = [_llm_eval_result(failure=False, reason="ok")]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z",
    )
    assert rs["results"][0]["actual_output"] == "Paris is the capital of France."


def test_result_to_openeval_none_entry_becomes_explicit_error_not_silent_pass_or_fail():
    data = _does_response_answer_query_data()
    eval_results = [_llm_eval_result(failure=False, reason="ok"), None]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z",
    )
    errored = rs["results"][1]
    assert errored["passed"] is False
    assert errored["error"]["type"] == "runner_error"
    assert errored["grader_results"][0]["score"] is None
    assert errored["grader_results"][0]["metadata"]["athina.errored"] is True


def test_result_to_openeval_summary_counts_are_accurate():
    data = _does_response_answer_query_data() * 2  # 4 entries
    eval_results = [
        _llm_eval_result(failure=False, reason="ok"),
        _llm_eval_result(failure=True, reason="no"),
        None,
        _llm_eval_result(failure=False, reason="ok"),
    ]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z",
    )
    summary = rs["summary"]
    assert summary["total"] == 4
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["skipped"] == 1  # the None (errored) entry
    assert summary["pass_rate"] == pytest.approx(2 / 3)  # only scored entries count
    assert summary["avg_score"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)


def test_result_to_openeval_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="data has"):
        result_to_openeval(
            _does_response_answer_query_data(), [_llm_eval_result(failure=False, reason="ok")],
            "does_response_answer_query", suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z",
        )


def test_result_to_openeval_rejects_empty_data():
    with pytest.raises(ValueError, match="at least one entry"):
        result_to_openeval([], [], "does_response_answer_query", suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z")


def test_result_to_openeval_produces_spec_valid_result_set():
    data = _does_response_answer_query_data()
    eval_results = [
        _llm_eval_result(failure=False, reason="Answers the query directly."),
        _llm_eval_result(failure=True, reason="Response is unrelated to the query."),
    ]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="athina-does_response_answer_query", run_id="run-1",
        started_at="2026-08-22T00:00:00Z", completed_at="2026-08-22T00:00:05Z",
    )
    result = validate_result_set(rs)
    assert result.valid, result.errors


def test_result_to_openeval_produces_spec_valid_result_set_with_errored_entry():
    data = _does_response_answer_query_data()
    eval_results = [_llm_eval_result(failure=False, reason="ok"), None]
    rs = result_to_openeval(
        data, eval_results, "does_response_answer_query",
        suite_id="s", run_id="r1", started_at="2026-08-22T00:00:00Z",
    )
    result = validate_result_set(rs)
    assert result.valid, result.errors


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


def test_from_openeval_reconstructs_query_and_expected_response():
    suite = to_openeval(_faithfulness_data(), "faithfulness")
    entries = from_openeval(suite)
    assert entries[0]["context"] == "3M has increased its dividend for 65 consecutive years."
    assert entries[0]["expected_response"] == "Yes, 3M has a 65-year streak of dividend increases."
    # this entry's input was synthesized from `response`, so no `query` key should reappear
    assert "query" not in entries[0]


def test_from_openeval_real_query_is_recovered_when_not_synthesized():
    suite = to_openeval(
        _context_contains_enough_information_data(), "context_contains_enough_information"
    )
    entries = from_openeval(suite)
    assert entries[0]["query"] == "Does CVS Health pay a quarterly dividend?"


def test_from_openeval_never_includes_a_response_key():
    suite = to_openeval(_does_response_answer_query_data(), "does_response_answer_query")
    entries = from_openeval(suite)
    for entry in entries:
        assert "response" not in entry


def test_from_openeval_context_list_joins_back_to_single_string():
    suite = {
        "version": "1.0.0",
        "id": "s",
        "test_cases": [
            {
                "id": "tc_0",
                "input": "query",
                "context": ["first paragraph", "second paragraph"],
                "graders": ["gr_x"],
            }
        ],
    }
    entries = from_openeval(suite)
    assert entries[0]["context"] == "first paragraph\n\nsecond paragraph"


def test_from_openeval_extra_args_round_trip_back_into_kwargs():
    data = [DataPoint(response="42", grading_criteria="Must be numeric")]
    suite = to_openeval(data, "custom_grader")
    entries = from_openeval(suite)
    assert entries[0]["grading_criteria"] == "Must be numeric"


def test_from_openeval_attaches_test_case_id_for_result_mapping():
    suite = to_openeval(_does_response_answer_query_data(), "does_response_answer_query", ids=["a", "b"])
    entries = from_openeval(suite)
    assert entries[0]["metadata"]["athina.test_case_id"] == "a"
    assert entries[1]["metadata"]["athina.test_case_id"] == "b"


def test_from_openeval_rejects_multi_turn_input():
    suite = {
        "version": "1.0.0",
        "id": "s",
        "test_cases": [{"id": "tc_0", "input": ["turn 1", "turn 2"], "graders": ["gr_x"]}],
    }
    with pytest.raises(ValueError, match="multi-turn"):
        from_openeval(suite)


def test_from_openeval_rejects_suite_with_no_test_cases():
    with pytest.raises(ValueError, match="no test_cases"):
        from_openeval({"version": "1.0.0", "id": "s", "test_cases": []})


def test_full_round_trip_to_openeval_then_result_to_openeval_stays_spec_valid():
    data = _faithfulness_data()
    suite = to_openeval(data, "faithfulness", suite_id="rt-suite")
    assert validate_suite(suite).valid

    eval_results = [
        _llm_eval_result(failure=False, reason="Grounded in the provided context."),
        _llm_eval_result(failure=True, reason="Contradicts the provided context."),
    ]
    rs = result_to_openeval(
        data, eval_results, "faithfulness",
        suite_id=suite["id"], run_id="rt-run", started_at="2026-08-22T00:00:00Z",
    )
    assert validate_result_set(rs).valid
    assert rs["suite_id"] == suite["id"]
